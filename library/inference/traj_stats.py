"""Trajectory-resolved latent statistics — the passive per-step recorder.

Proposal: ``_archive/proposals/traj_latent_stats.md``. Observes the denoising
trajectory (per-step, per-token, per-channel statistics of the x̂₀ estimate)
without changing a single bit of the generation. Everything is computed on
``.float()`` copies of the loop tensors, out-of-place, and dumped to one
``.npz`` sidecar per generation at flush.

Hooked the same way the other sampler plug-ins are: built from ``args`` in
``generate_body`` (``None`` when off — zero allocations, hook sites are a
``if recorder is not None`` short-circuit), fed once per step *after* the CFG
combine and *before* the sampler step via :meth:`record`, flushed once per
generation.

"Token" = one DiT patch = a 2×2 latent-pixel cell (16 px at the VAE's 8×
downsample), so every map is directly comparable to attention/token-count
reasoning. x̂₀ is normalized per channel with the Qwen Image VAE's shipped
``latents_mean`` / ``latents_std`` so "a bit" means the same thing in every
channel.

Traces per step (see the proposal's table):

* ``tok``      — token-level normalized x̂₀ itself (f16) — the raw map for
  token-wise RMSE(σ) (codes are too coarse at k=4)
* ``codes``    — k-bit uniform quantization code of token-level x̂₀ (uint8)
* ``activity`` — ‖x̂₀(p, i) − x̂₀(p, i−1)‖ over channels (NaN at step 0)
* ``hf``       — Laplacian energy of x̂₀ around each token (foveation's x0var)
* ``guide``    — ‖v_final − v_uncond‖ per token (post-combine effective
  guidance; CFG runs only, all-NaN otherwise)
* ``cbits``    — per-channel entropy (bits) of the code histogram over tokens
* ``commit``   — derived at flush: last step whose code changed, per token
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Per-channel statistics of the Qwen Image VAE latent space (z_dim = 16).
# Mirror of the shipped defaults in
# library/models/qwen_vae.py::AutoencoderKLQwenImage.__init__ — pinned equal by
# tests/test_traj_stats.py so they cannot drift.
QWEN_LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
QWEN_LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]

# Quantization support: normalized x̂₀ is ~unit-scale per channel; codes are
# uniform bins over [-_QUANT_RANGE, +_QUANT_RANGE] (clamped).
_QUANT_RANGE = 4.0

_LAPLACIAN_3X3 = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]

# DiT patch = 2×2 latent pixels (16 px) — the token aggregation cell.
_TOKEN_CELL = 2


class TrajStatsRecorder:
    """Accumulates per-step trajectory traces on-GPU and dumps one npz at flush.

    Pure observation: :meth:`record` never mutates its inputs and never feeds
    anything back into the loop — recorder on/off latents are bit-identical
    (pinned by ``tests/test_traj_stats.py``).
    """

    # process-global: generations flushed so far (filename collision guard for
    # multi-prompt runs, mirroring CfgDeltaProbe._gen_counter)
    _gen_counter = 0

    def __init__(
        self,
        out_dir: str,
        *,
        k: int = 4,
        seed=None,
        meta: Optional[dict] = None,
    ) -> None:
        if not (1 <= int(k) <= 8):
            raise ValueError(f"traj_stats_k must be in [1, 8], got {k}")
        self.out_dir = out_dir
        self.k = int(k)
        self.levels = 1 << self.k
        self.seed = seed
        self.meta = dict(meta or {})
        self._steps: list[dict] = []
        self._sigmas: list[float] = []
        self._prev_tok: Optional[torch.Tensor] = None
        self._prev_codes: Optional[torch.Tensor] = None
        self._commit: Optional[torch.Tensor] = None
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._lap_k: Optional[torch.Tensor] = None

    @classmethod
    def from_args(cls, args, *, seed=None) -> Optional["TrajStatsRecorder"]:
        """Build from parsed CLI args; ``None`` unless ``--traj_stats`` is set."""
        if not getattr(args, "traj_stats", False):
            return None
        return cls(
            getattr(args, "traj_stats_dir", "output/traj_stats"),
            k=getattr(args, "traj_stats_k", 4),
            seed=seed,
            meta={
                "prompt_hash": format(
                    hash(getattr(args, "prompt", "") or "") & 0xFFFFFFFF, "08x"
                ),
                "sampler": getattr(args, "sampler", None),
                "infer_steps": getattr(args, "infer_steps", None),
                "guidance_scale": getattr(args, "guidance_scale", None),
                "image_size": list(getattr(args, "image_size", ()) or ()),
            },
        )

    def _norm_stats(self, device, channels: int):
        if self._mean is None or self._mean.shape[1] != channels:
            if channels == len(QWEN_LATENTS_MEAN):
                mean, std = QWEN_LATENTS_MEAN, QWEN_LATENTS_STD
            else:  # non-Qwen latent space: fall back to identity normalization
                mean, std = [0.0] * channels, [1.0] * channels
            self._mean = torch.tensor(mean, device=device).view(1, channels, 1, 1)
            self._std = torch.tensor(std, device=device).view(1, channels, 1, 1)
        return self._mean.to(device), self._std.to(device)

    @staticmethod
    def _to_spatial_4d(x: torch.Tensor) -> torch.Tensor:
        """Squeeze the 5D in-loop layout ``(B, C, 1, H, W)`` to 4D.

        Dim 2 explicitly — never a bare ``squeeze()`` (see the CLAUDE.md dim-2
        invariant; a bare squeeze corrupts B=1 batches).
        """
        if x.dim() == 5:
            assert x.shape[2] == 1, f"expected T=1 at dim 2, got {tuple(x.shape)}"
            x = x.squeeze(2)
        assert x.dim() == 4, f"expected 4D (B,C,H,W), got {tuple(x.shape)}"
        return x

    @torch.no_grad()
    def record(
        self,
        step: int,
        sigma,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        uncond_noise_pred: Optional[torch.Tensor] = None,
    ) -> None:
        """Observe one denoising step. Called post-CFG-combine, pre-sampler-step.

        ``sigma`` may be a python float or a 0-d tensor. Hook sites MUST pass
        the ``sigmas[i]`` tensor, not ``float(sigmas[i])`` — the float() read
        is a stream sync that breaks the loop's CPU run-ahead (measured ~28 ms
        per step at 1024²; the whole record() is ~0.4 ms). Conversion for the
        header is deferred to :meth:`flush`.
        """
        if torch.is_tensor(sigma):
            sigma_f = sigma.detach()  # 0-d tensor: arithmetic without a sync
        else:
            sigma_f = float(sigma)
        # x̂₀ = z − σ·v (flow matching) — on float copies, never in-place.
        x0 = self._to_spatial_4d(latents.float() - sigma_f * noise_pred.float())
        mean, std = self._norm_stats(x0.device, x0.shape[1])
        xn = (x0 - mean) / std  # (B, C, H, W), per-channel normalized

        b, c, h, w = xn.shape
        tok = F.avg_pool2d(xn, _TOKEN_CELL)  # (B, C, Ht, Wt) — token-level x̂₀

        # codes: k-bit uniform quantization per channel
        span = 2.0 * _QUANT_RANGE
        codes = (
            ((tok.clamp(-_QUANT_RANGE, _QUANT_RANGE) + _QUANT_RANGE) / span)
            .mul(self.levels)
            .floor()
            .clamp_(0, self.levels - 1)
            .to(torch.uint8)
        )

        # activity: channel-L2 of the token-level x̂₀ delta vs previous step
        if self._prev_tok is not None:
            activity = (tok - self._prev_tok).norm(dim=1)  # (B, Ht, Wt)
        else:
            activity = torch.full(
                tok.shape[:1] + tok.shape[2:], float("nan"), device=tok.device
            )

        # commit: last step whose code changed (any channel)
        if self._prev_codes is not None:
            changed = (codes != self._prev_codes).any(dim=1)  # (B, Ht, Wt)
            self._commit = torch.where(
                changed, torch.full_like(self._commit, step), self._commit
            )
        else:
            self._commit = torch.zeros(
                codes.shape[:1] + codes.shape[2:],
                dtype=torch.int16,
                device=codes.device,
            )

        # hf: 3×3 Laplacian energy of normalized x̂₀, channel-mean, token-pooled
        # (kernel cached per device — a per-step torch.tensor() would pay a
        # synchronous H2D copy every step)
        if self._lap_k is None or self._lap_k.device != xn.device:
            self._lap_k = torch.tensor(_LAPLACIAN_3X3, device=xn.device).view(
                1, 1, 3, 3
            )
        lap = F.conv2d(xn.reshape(b * c, 1, h, w), self._lap_k, padding=1)
        hf = F.avg_pool2d(
            lap.pow(2).reshape(b, c, h, w).mean(dim=1, keepdim=True), _TOKEN_CELL
        ).squeeze(1)  # (B, Ht, Wt)

        # guide: per-token norm of the effective (post-combine) guidance delta.
        # Under plain CFG this is g·‖v_cond − v_uncond‖ — same map, fixed scale.
        if uncond_noise_pred is not None:
            gdelta = self._to_spatial_4d(noise_pred.float() - uncond_noise_pred.float())
            guide = F.avg_pool2d(gdelta.norm(dim=1, keepdim=True), _TOKEN_CELL).squeeze(
                1
            )  # (B, Ht, Wt)
        else:
            guide = torch.full_like(hf, float("nan"))

        # cbits: per-channel entropy (bits) of the code histogram over all tokens
        flat = codes.permute(1, 0, 2, 3).reshape(c, -1).long()
        counts = torch.zeros(c, self.levels, device=codes.device)
        counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=counts.dtype))
        p = counts / counts.sum(dim=1, keepdim=True)
        cbits = -(p * p.clamp_min(1e-12).log2()).sum(dim=1)  # (C,)

        self._steps.append(
            {
                "tok": tok.to(torch.float16),
                "activity": activity.to(torch.float16),
                "hf": hf.to(torch.float16),
                "guide": guide.to(torch.float16),
                "cbits": cbits.float(),
                "codes": codes,
            }
        )
        self._sigmas.append(sigma_f)  # tensor stays a tensor; float()ed at flush
        self._prev_tok = tok
        self._prev_codes = codes

    def flush(self) -> Optional[str]:
        """Write the accumulated traces to one ``.npz``; idempotent (no-op when
        empty, clears state after writing). Returns the written path or None."""
        if not self._steps:
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        gen = TrajStatsRecorder._gen_counter
        TrajStatsRecorder._gen_counter += 1
        seed = self.seed
        if isinstance(seed, (list, tuple)):
            seed = seed[0] if seed else None
        name = f"traj_g{gen:03d}" + (f"_seed{seed}" if seed is not None else "")
        path = os.path.join(self.out_dir, name + ".npz")

        def stack(key: str) -> np.ndarray:
            return torch.stack([s[key] for s in self._steps]).cpu().numpy()

        header = dict(self.meta)
        header.update({"k": self.k, "seed": seed, "num_steps": len(self._steps)})
        # uncompressed: zlib on the ~4 MB payload costs >100 ms inside the
        # generation wall time.
        np.savez(
            path,
            tok=stack("tok"),  # (S, B, C, Ht, Wt) f16 — token-level x̂₀
            activity=stack("activity"),  # (S, B, Ht, Wt) f16, step 0 = NaN
            hf=stack("hf"),  # (S, B, Ht, Wt) f16
            guide=stack("guide"),  # (S, B, Ht, Wt) f16, NaN when no CFG
            cbits=stack("cbits"),  # (S, C) f32
            codes=stack("codes"),  # (S, B, C, Ht, Wt) u8
            commit=self._commit.cpu().numpy(),  # (B, Ht, Wt) i16
            # the one sync point: 0-d sigma tensors -> floats, at flush only
            sigmas=np.asarray([float(s) for s in self._sigmas], dtype=np.float64),
            header_json=np.array(json.dumps(header)),
        )
        self._steps.clear()
        self._sigmas.clear()
        self._prev_tok = self._prev_codes = self._commit = None
        return path


def derive_summary(npz_path: str, tau: Optional[float] = None) -> dict:
    """Derive the headline scalars from a recorded trajectory.

    * ``E`` — effective token usage per step: fraction of tokens with
      ``activity > τ``. Default τ is the σ→0 noise floor, provisionally the
      95th percentile of final-step activity.
    * ``commit_cdf`` — fraction of tokens committed (code last changed) at or
      before each step.
    """
    d = np.load(npz_path)
    act = d["activity"].astype(np.float32)  # (S, B, Ht, Wt)
    if tau is None:
        tau = float(np.nanquantile(act[-1], 0.95))
    with np.errstate(invalid="ignore"):
        e_curve = np.nanmean((act > tau).astype(np.float32), axis=(1, 2, 3))
    commit = d["commit"]  # (B, Ht, Wt)
    steps = act.shape[0]
    n_tok = commit.size
    commit_cdf = [float((commit <= i).sum() / n_tok) for i in range(steps)]
    return {
        "sigmas": [float(s) for s in d["sigmas"]],
        "tau": tau,
        "E": [float(x) for x in e_curve],
        "commit_cdf": commit_cdf,
        "cbits_first_step": [float(x) for x in d["cbits"][0]],
        "cbits_last_step": [float(x) for x in d["cbits"][-1]],
    }
