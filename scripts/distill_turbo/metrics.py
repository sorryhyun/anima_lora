"""GPU-side accumulators for the turbo loop, read in one sync per ``log_interval``.

Scalars (TB names in :func:`write_scalars`; how to read them:
``docs/anima/methods/turbo.md`` § Reading the metrics):

* ``grad`` rms of the DM grad signal into x_pred; ``dm`` rms(v_real − v_fake)
* ``xpred`` std of x_pred; ``v_student`` rms of the step-0 velocity
* ``rel_gap`` rms(τ·Δ_dm) / rms(τ·v_real_dm); ``mag_ratio`` rms(v_fake_dm) /
  rms(v_real_dm); ``cos`` cosine(v_fake_dm, v_real_dm)
* ``div`` step-0 anchor MSE (pre-weight)
* ``gn_*`` pre-clip gradient norm per optimizer (above ``grad_clip`` = clipped)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

import torch

from library.training.accumulator import ScalarAccumulator

# Uniform τ bins over [0, 1] for the per-τ fake-loss profile.
TAU_PROFILE_BINS = 8


@dataclass
class FlushedMetrics:
    fake: float
    grad: float
    dm: float
    xpred: float
    v_student: float
    rel_gap: float
    mag_ratio: float
    cos: float
    div: float
    gn_student: float
    gn_fake: float


_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FlushedMetrics))


class TurboMetrics:
    """:class:`ScalarAccumulator` behind a typed :class:`FlushedMetrics`. Every
    field is pre-touched, so an anchor-only run still flushes a complete record.
    """

    def __init__(self, device: torch.device):
        self._acc = ScalarAccumulator(device)
        for name in _FIELDS:
            self._acc.add(name, 0.0)

    @torch.no_grad()
    def accumulate_per_step(
        self,
        *,
        fake_loss_mean_t: torch.Tensor,
        grad_signal: torch.Tensor,
        delta_dm: torch.Tensor,
        x_pred: torch.Tensor,
        v_student: torch.Tensor,
        tau_dm_e: torch.Tensor,
        v_real_cond_dm: torch.Tensor,
        v_fake_cond_dm: torch.Tensor,
    ) -> None:
        eps_r = 1e-8
        self._acc.add("fake", fake_loss_mean_t.float())
        self._acc.add("grad", grad_signal.float().pow(2).mean().sqrt())
        self._acc.add("xpred", x_pred.detach().float().std())
        self._acc.add("v_student", v_student.detach().float().pow(2).mean().sqrt())
        self._acc.add("dm", delta_dm.float().pow(2).mean().sqrt())
        vr = v_real_cond_dm.float()
        vf = v_fake_cond_dm.float()
        dm_w = (tau_dm_e * delta_dm.float()).pow(2).mean().sqrt()
        self._acc.add("rel_gap", dm_w / ((tau_dm_e * vr).pow(2).mean().sqrt() + eps_r))
        self._acc.add(
            "mag_ratio", vf.pow(2).mean().sqrt() / (vr.pow(2).mean().sqrt() + eps_r)
        )
        self._acc.add("cos", (vf * vr).sum() / (vf.norm() * vr.norm() + eps_r))

    @torch.no_grad()
    def add_grad_norms(self, student: torch.Tensor, fake: torch.Tensor) -> None:
        """Accumulate the pre-clip parameter-gradient norms."""
        self._acc.add("gn_student", student.detach().float())
        self._acc.add("gn_fake", fake.detach().float())

    @torch.no_grad()
    def add_div(self, div_loss_t: torch.Tensor) -> None:
        """Accumulate the step-0 anchor loss (pre-weight)."""
        self._acc.add("div", div_loss_t.detach().float())

    def flush(self, log_interval: int) -> FlushedMetrics:
        """One CUDA sync per log boundary: read every accumulator, mean it."""
        m = self._acc.flush()
        return FlushedMetrics(**{k: m[k] / log_interval for k in _FIELDS})

    def reset(self) -> None:
        self._acc.reset()


class TauBinCriticLoss:
    """Per-τ-bin fake/critic loss profile.

    Buckets each fake-update loss by its drawn ``tau_fake`` into
    ``TAU_PROFILE_BINS`` uniform bins and logs the per-bin interval mean as
    ``{prefix}{i}``. GPU-resident; one small sync per log boundary.

    Not a gate signal: the FM target ``ε − x0`` is harder at some τ regardless of
    critic capacity, so the raw profile is nonuniform.

    The loop's fake loss is a batch-mean scalar, so at B>1 it is attributed to
    every sample's τ bin (exact at the shipped ``batch_size=1``).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        nbins: int = TAU_PROFILE_BINS,
        prefix: str = "train/fake_loss_tau",
    ):
        self.nbins = nbins
        self.prefix = prefix
        self._sum = torch.zeros(nbins, device=device)
        self._cnt = torch.zeros(nbins, device=device)

    @torch.no_grad()
    def add(self, fake_loss: torch.Tensor, tau: torch.Tensor) -> None:
        """Accumulate one fake update's loss under its ``(B,)`` τ draw."""
        idx = (tau.detach().float().clamp(0.0, 1.0) * self.nbins).long()
        idx = idx.clamp_(max=self.nbins - 1)  # τ = 1.0 → last bin
        val = fake_loss.detach().float().expand(idx.shape)
        self._sum.index_add_(0, idx, val)
        self._cnt.index_add_(0, idx, torch.ones_like(val))

    def flush(self) -> tuple[list[float], list[float]]:
        """One sync: return ``(per-bin loss sums, per-bin counts)`` and reset."""
        sums, cnts = torch.stack([self._sum, self._cnt]).tolist()
        self._sum.zero_()
        self._cnt.zero_()
        return sums, cnts

    def write(self, writer, step: int) -> None:
        """Flush and push per-bin means to TensorBoard (empty bins skipped). Resets
        even when ``writer`` is None, so call it at every log boundary."""
        sums, cnts = self.flush()
        if writer is None:
            return
        for i, (s, n) in enumerate(zip(sums, cnts)):
            if n > 0:
                writer.add_scalar(f"{self.prefix}{i}", s / n, step)


class StepGradLog:
    """One JSONL row per loop iteration: pre-clip grad norms beside the draws.

    The interval means of ``gn_*`` cannot be paired with the τ or g that produced
    them, so this keeps each iteration. Rows are GPU-resident until ``flush`` (one
    sync per log boundary) and appended to ``path``, so a resumed run continues the
    same file. Fields: ``step``, ``gn_student``, ``gn_fake``, ``sigma_a`` (DM query
    level τ, batch mean), ``s_g`` (σ at the grad step), ``g``.
    """

    def __init__(self, path, device: torch.device):
        self.path = path
        self.device = device
        self._rows: list[torch.Tensor] = []
        self._meta: list[tuple[int, float, int]] = []

    @torch.no_grad()
    def add(
        self,
        step: int,
        gn_student: torch.Tensor,
        gn_fake: torch.Tensor,
        sigma_a: torch.Tensor,
        s_g: float,
        g: int,
    ) -> None:
        self._rows.append(
            torch.stack(
                [
                    gn_student.detach().float().to(self.device),
                    gn_fake.detach().float().to(self.device),
                    sigma_a.detach().float().mean().to(self.device),
                ]
            )
        )
        self._meta.append((step, float(s_g), int(g)))

    def flush(self) -> None:
        if not self._rows:
            return
        vals = torch.stack(self._rows).tolist()
        with open(self.path, "a", encoding="utf-8") as f:
            for (step, s_g, g), (gs, gf, sa) in zip(self._meta, vals):
                row = {
                    "step": step,
                    "gn_student": round(gs, 5),
                    "gn_fake": round(gf, 5),
                    "sigma_a": round(sa, 5),
                    "s_g": round(s_g, 5),
                    "g": g,
                }
                f.write(json.dumps(row) + "\n")
        self._rows.clear()
        self._meta.clear()


def write_scalars(
    writer, m: FlushedMetrics, step: int, *, anchor_only: bool = False
) -> None:
    """Push every available scalar to TensorBoard at the canonical key names
    (``anchor_only``: the anchor loss and the student grad norm, nothing else runs)."""
    writer.add_scalar("train/div_loss", m.div, step)
    writer.add_scalar("train/grad_norm_student", m.gn_student, step)
    if anchor_only:
        return
    writer.add_scalar("train/fake_loss", m.fake, step)
    writer.add_scalar("train/grad_signal_rms", m.grad, step)
    writer.add_scalar("train/delta_dm_rms", m.dm, step)
    writer.add_scalar("train/x_pred_std", m.xpred, step)
    writer.add_scalar("train/v_student_rms", m.v_student, step)
    writer.add_scalar("train/dm_rel_gap", m.rel_gap, step)
    writer.add_scalar("train/dm_mag_ratio", m.mag_ratio, step)
    writer.add_scalar("train/dm_cos", m.cos, step)
    writer.add_scalar("train/grad_norm_fake", m.gn_fake, step)
