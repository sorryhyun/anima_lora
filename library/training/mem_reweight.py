"""Online per-sample memorization Δ-gap tracker + low-σ loss reweight.

Implements ``_archive/proposals/memorization_lowsigma_reweight.md`` (Arm A): track,
per dataset item, an EMA of the calibrated loss gap the weight-side MIA probe
scores offline (``bench/memorization/loss_gap.py``), estimated online during
training, and smoothly downweight flagged items' FM loss in the low-σ band.

THE STATISTIC. On measurement steps, one extra no-grad forward through the
same trainable DiT with the adapter zeroed (``network.set_multiplier(0)`` —
the VR control-variate trick, see ``library/training/forward/vr_forward.py``)
on the *identical* ``(x_t, ε, σ)`` the train step drew. Per item::

    Δ = log(mse_base + δ) − log(mse_adapted + δ)
      = R_adapted − R_base            with R = −log(mse + δ)

— exactly the per-probe ``delta`` of ``loss_gap.py``. Pairing on the same
``(x_t, ε, σ)`` cancels image easiness (the SOLACE flatness confound); a large
Δ means the adapter specifically learned *this* sample. Positive Δ ⇒ adapter
more confident than base on this member draw.

Both MSEs are raw full-latent fp32 means (no masked_loss, no weighting_scheme)
to stay commensurable with the offline probe's statistic.

MEASUREMENT is restricted to draws with σ ≤ ``sigma_max`` (the band the MIA
signal lives in, banked: σ ≤ 0.7) and to every K-th train batch
(``measure_every``). K must stay small — at bs=1, K=1 gives only ~2–4
measurements per item across a few-hundred-step run, which is why the EMA uses
Adam-style bias correction.

MULTI-DRAW mode (``mem_extra_sigmas`` non-empty): instead of the single train-draw pair, every measurement step scores
each batch item at a fixed σ grid × antithetic ±ε pairs (mirroring
``loss_gap.py::confidence``: MSE averaged across all draws *before* the log,
identical noise for base and adapted). One well-averaged Δ per item per
visit, and items are measured on EVERY visit — not only low-σ train draws —
so per-item counts rise ~3× and per-measurement variance drops ~4× at grid
{0.5, 0.7} × 2 noises. Extra noise comes from a DEDICATED generator so the
training RNG stream stays bit-identical to a run without measurement.
Caveat: the grid forwards reuse the train step's per-step adapter
conditioning — do not combine with σ-conditioned adapter state
(``use_timestep_mask=true`` or σ/FEI routers); plain-LoRA stacks only.
Cost: ``2 × len(grid) × n_noise`` no-grad forwards per measurement step.

INTERVENTION (mode="reweight") multiplies the per-sample ``loss_weights`` by

    w = floor + (1 − floor) · sigmoid((z_thr − z_i) / temp)

for items whose bias-corrected Δ-EMA z-scores (across all items with ≥ 1
measurement) exceed the threshold — only on σ ≤ sigma_max draws, only after
``warmup_updates`` total measurements. High-σ draws are untouched: the item
keeps teaching composition/style, it stops being pixel-copied. mode="measure"
runs the tracker without touching the loss.

The weight for a step is computed from tracker state *before* that step's
measurement is folded in (strictly causal — no same-step feedback loop).

STATE is a plain dict keyed by the dataset ``image_key`` (absolute path),
persisted as ``<output_dir>/<output_name>_memgap.json`` every few updates and
at process exit, for post-hoc inspection and correlation against the offline
probe's ``per_item.csv``.

HARD REQUIREMENT: ``blocks_to_swap == 0``. The measurement forward is a second
DiT forward per step, and the block-swap offloader desyncs on extra forwards;
train.py raises at setup.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)


def measure_base_logmse(
    *,
    anima_call: Any,
    network: Any,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    padding_mask: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
    noise: torch.Tensor,
    latents: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Base-model per-sample log(mse+δ) on the train step's own (x_t, ε, σ).

    One no-grad forward with the adapter zeroed (multiplier=0) — identical
    inputs to the primary forward (``noisy_model_input`` is the already-5D
    train input), so the pair (base, adapted) shares (x_t, ε, σ) exactly.
    Returns a (B,) fp32 tensor.
    """
    orig_mult = float(getattr(network, "multiplier", 1.0))
    network.set_multiplier(0.0)
    try:
        with torch.no_grad():
            base_pred = anima_call(
                noisy_model_input,
                timesteps,
                crossattn_emb,
                padding_mask=padding_mask,
                **forward_kwargs,
            )
    finally:
        network.set_multiplier(orig_mult)

    base_pred = base_pred.squeeze(2)  # 5D -> 4D, singleton frame axis is dim 2
    target = noise.float() - latents.float()  # rectified-flow velocity target
    mse = (base_pred.float() - target).pow(2).mean(dim=tuple(range(1, target.ndim)))
    return torch.log(mse + delta)


def measure_grid_delta(
    *,
    anima_call: Any,
    network: Any,
    latents: torch.Tensor,
    crossattn_emb: torch.Tensor,
    padding_mask: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
    model_dtype: torch.dtype,
    sigmas: Sequence[float],
    n_noise: int,
    delta: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Multi-draw paired Δ per batch item on a fixed σ grid (see docstring).

    Mirrors ``loss_gap.py::confidence``: rectified-flow re-noise
    ``z_t = (1−σ)·z0 + σ·ε`` over antithetic ±ε pairs, per-sample MSE against
    the velocity target ``ε − z0`` averaged across ALL grid draws before the
    log; identical noise for the adapted and base (multiplier=0) passes so the
    Δ is purely the model change. Returns a (B,) fp32 tensor
    ``log(mse_base+δ) − log(mse_adapted+δ)``.
    """
    z0 = latents.detach().float()
    bsz = z0.shape[0]
    n_base = (max(1, n_noise) + 1) // 2
    draws: list[torch.Tensor] = []
    for _ in range(n_base):
        e = torch.randn(
            z0.shape, generator=generator, device=z0.device, dtype=torch.float32
        )
        draws.append(e)
        if len(draws) < n_noise:
            draws.append(-e)

    mse = {
        "adapted": torch.zeros(bsz, device=z0.device),
        "base": torch.zeros(bsz, device=z0.device),
    }
    orig_mult = float(getattr(network, "multiplier", 1.0))
    try:
        with torch.no_grad():
            for arm, mult in (("adapted", orig_mult), ("base", 0.0)):
                network.set_multiplier(mult)
                for s in sigmas:
                    t = torch.full(
                        (bsz,), float(s), device=z0.device, dtype=torch.float32
                    )
                    for eps in draws:
                        z_t = ((1.0 - s) * z0 + s * eps).to(model_dtype)
                        pred = anima_call(
                            z_t.unsqueeze(2),
                            t,
                            crossattn_emb,
                            padding_mask=padding_mask,
                            **forward_kwargs,
                        )
                        pred = pred.squeeze(2).float()
                        tgt = eps - z0
                        mse[arm] += (
                            (pred - tgt).pow(2).mean(dim=tuple(range(1, tgt.ndim)))
                        )
    finally:
        network.set_multiplier(orig_mult)

    n = len(sigmas) * len(draws)
    return torch.log(mse["base"] / n + delta) - torch.log(mse["adapted"] / n + delta)


def adapted_logmse(
    model_pred: torch.Tensor, target: torch.Tensor, delta: float
) -> torch.Tensor:
    """Adapted-model per-sample log(mse+δ) from the primary forward's outputs.

    ``model_pred``/``target`` are the trainer's 4D tensors; detached — the
    measurement never carries gradient.
    """
    mse = (
        (model_pred.detach().float() - target.detach().float())
        .pow(2)
        .mean(dim=tuple(range(1, target.ndim)))
    )
    return torch.log(mse + delta)


class MemGapTracker:
    """Per-item Δ-gap EMA + reweight state (see module docstring)."""

    SAVE_EVERY_UPDATES = 25

    def __init__(self, args, save_path: str | os.PathLike):
        self.mode = str(getattr(args, "mem_reweight_mode", "") or "")
        assert self.mode in ("measure", "reweight"), self.mode
        self.sigma_max = float(getattr(args, "mem_sigma_max", 0.7))
        self.measure_every = max(1, int(getattr(args, "mem_measure_every", 1)))
        self.ema_beta = float(getattr(args, "mem_ema_beta", 0.3))
        self.z_threshold = float(getattr(args, "mem_z_threshold", 1.5))
        self.weight_temp = float(getattr(args, "mem_weight_temp", 0.5))
        self.weight_floor = float(getattr(args, "mem_weight_floor", 0.0))
        self.warmup_updates = int(getattr(args, "mem_warmup_updates", 100))
        self.delta = float(getattr(args, "mem_delta", 1e-3))
        self.snapshot_every = int(getattr(args, "mem_snapshot_every", 50))
        self.extra_sigmas = [
            float(s) for s in (getattr(args, "mem_extra_sigmas", None) or [])
        ]
        self.extra_noise = max(1, int(getattr(args, "mem_extra_noise", 2)))
        # Dedicated noise stream for multi-draw measurement — must NOT touch
        # the global RNG or the training noise sequence diverges from an
        # unmeasured run.
        self._noise_seed = int(getattr(args, "seed", None) or 0) + 0x6D656D
        self._noise_gen: torch.Generator | None = None

        self.save_path = Path(save_path)
        self._ema: dict[str, float] = {}  # raw (uncorrected) EMA per item key
        self._count: dict[str, int] = {}
        self._steps = 0  # train batches seen (measurement cadence)
        self._updates = 0  # total per-item measurements folded in
        # Flag-set snapshots [(updates, [stems…]), …] for churn analysis
        # (churn > ~50% ⇒ estimator too noisy to act on).
        self._history: list[tuple[int, list[str]]] = []
        self._dirty = False
        atexit.register(self.save)

    # ---------------------------------------------------------------- cadence

    def noise_generator(self, device) -> torch.Generator:
        if self._noise_gen is None:
            self._noise_gen = torch.Generator(device=device).manual_seed(
                self._noise_seed
            )
        return self._noise_gen

    def should_measure(self) -> bool:
        """Advance the train-batch counter; True on every K-th batch."""
        self._steps += 1
        return (self._steps - 1) % self.measure_every == 0

    # ------------------------------------------------------------ EMA / stats

    def update(self, keys: Sequence[str], deltas: Sequence[float]) -> None:
        beta = self.ema_beta
        for key, d in zip(keys, deltas):
            d = float(d)
            if not math.isfinite(d):
                continue
            prev = self._ema.get(key, 0.0)
            self._ema[key] = (1.0 - beta) * prev + beta * d
            self._count[key] = self._count.get(key, 0) + 1
            self._updates += 1
        if self.snapshot_every > 0 and self._history_due():
            self._history.append((self._updates, sorted(self.flagged_stems())))
        self._dirty = True
        if self._updates % self.SAVE_EVERY_UPDATES == 0:
            self.save()

    def _history_due(self) -> bool:
        last = self._history[-1][0] if self._history else -self.snapshot_every
        return self._updates - last >= self.snapshot_every

    def _corrected(self, key: str) -> float:
        """Adam-style bias-corrected EMA (counts here are tiny, 2–4)."""
        n = self._count.get(key, 0)
        if n == 0:
            return 0.0
        return self._ema[key] / (1.0 - (1.0 - self.ema_beta) ** n)

    def zscores(self) -> dict[str, float]:
        """Standardize corrected Δ-EMAs across items with ≥ 1 measurement."""
        keys = [k for k, n in self._count.items() if n > 0]
        if len(keys) < 8:  # too few items for a meaningful distribution
            return {}
        vals = [self._corrected(k) for k in keys]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        std = math.sqrt(var)
        if std < 1e-12:
            return dict.fromkeys(keys, 0.0)
        return {k: (v - mean) / std for k, v in zip(keys, vals)}

    def flagged_stems(self) -> list[str]:
        return [Path(k).stem for k, z in self.zscores().items() if z > self.z_threshold]

    # -------------------------------------------------------------- reweight

    def weights(
        self, keys: Sequence[str], low_sigma_mask: Sequence[bool]
    ) -> list[float]:
        """Per-sample loss multipliers for this batch (1.0 = untouched).

        Downweights only in reweight mode, only after warmup, only measured
        items, only on σ ≤ sigma_max draws. Uses state from *previous* steps
        (call before this step's ``update``).
        """
        if self.mode != "reweight" or self._updates < self.warmup_updates:
            return [1.0] * len(keys)
        z = self.zscores()
        out = []
        for key, low in zip(keys, low_sigma_mask):
            zi = z.get(key)
            if not low or zi is None:
                out.append(1.0)
                continue
            gate = 1.0 / (1.0 + math.exp(-(self.z_threshold - zi) / self.weight_temp))
            out.append(self.weight_floor + (1.0 - self.weight_floor) * gate)
        return out

    # ------------------------------------------------------------ persistence

    def save(self) -> None:
        if not self._dirty:
            return
        z = self.zscores()
        items = [
            {
                "key": k,
                "stem": Path(k).stem,
                "count": self._count[k],
                "ema": self._ema[k],
                "ema_corrected": self._corrected(k),
                "z": z.get(k),
            }
            for k in sorted(self._ema)
        ]
        payload = {
            "schema_version": 1,
            "mode": self.mode,
            "config": {
                "sigma_max": self.sigma_max,
                "measure_every": self.measure_every,
                "ema_beta": self.ema_beta,
                "z_threshold": self.z_threshold,
                "weight_temp": self.weight_temp,
                "weight_floor": self.weight_floor,
                "warmup_updates": self.warmup_updates,
                "delta": self.delta,
                "extra_sigmas": self.extra_sigmas,
                "extra_noise": self.extra_noise,
            },
            "steps_seen": self._steps,
            "updates": self._updates,
            "flagged_stems": sorted(self.flagged_stems()),
            "flag_history": [{"updates": u, "stems": s} for u, s in self._history],
            "items": items,
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.save_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.save_path)
        self._dirty = False
