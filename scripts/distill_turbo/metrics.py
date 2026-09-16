"""GPU-side accumulators + single-sync flush for the turbo training loop.

All accumulators live on-device; they're flushed in one stacked ``.tolist()``
at every ``log_interval`` so per-step CUDA syncs go to zero.

Health-scalar semantics:

* ``grad``      — overall DMD gradient magnitude into x_pred
* ``dm``        — DM regularizer strength (v_real − v_fake)
* ``xpred``     — x_pred dispersion: → 0 means collapse to mean, drifting upward
  means student is exploding.
* ``v_student`` — direct student velocity magnitude; runaway student manifests
  here before x_pred_std catches up (x_pred = x_t − t·v_student).

Fake-tracking ratios (real DMD health signals — ``loss_student`` is a
sign-random gradient vehicle, not a real loss):

* ``rel_gap``  — rms(τ·Δ_dm) / rms(τ·v_real_dm): fraction of teacher score the DM
  gap still represents. ↑ = fake lagging → bump fake.
* ``mag_ratio`` — rms(v_fake_dm) / rms(v_real_dm): ≈1 healthy; collapse/blow-up bad.
* ``cos``       — cosine(v_fake_dm, v_real_dm): ↓ = fake pointing the wrong way.

DP-DMD diversity loss:

* ``div`` — the first-step diversity MSE ‖v_first − v_target‖² (pre-weight).
  Falling = the student's step-1 velocity is converging on the teacher's
  K-step anchor (the diverse landing point).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from tqdm import tqdm

from library.training.accumulator import ScalarAccumulator

# τ-binned critic-loss profile: bin count for the
# per-τ fake-loss telemetry. 8 uniform bins over τ ∈ [0, 1].
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
    gan_gen: float
    gan_disc: float
    gan_margin: float
    gan_spread: float
    softrank: float
    softrank_active: float
    cdm: float


# Single source of truth for the accumulator key set — adding a logged scalar is
# one field on FlushedMetrics (+ its accumulate call + a write_scalars line),
# never a fourth edit to a parallel stack/reset list.
_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FlushedMetrics))


class TurboMetrics:
    """GPU-resident accumulators with a single-sync flush.

    Thin wrapper over :class:`library.training.accumulator.ScalarAccumulator`
    that keeps the typed :class:`FlushedMetrics` public surface (consumed by
    :func:`write_scalars` / :func:`tqdm_postfix`). Every field is pre-touched so
    a disabled path (``div`` without DP-DMD, ``gan_*`` without the GAN) still
    flushes a complete record.
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
        self._acc.add("dm", delta_dm.float().pow(2).mean().sqrt())
        self._acc.add("xpred", x_pred.detach().float().std())
        self._acc.add("v_student", v_student.detach().float().pow(2).mean().sqrt())
        vr = v_real_cond_dm.float()
        vf = v_fake_cond_dm.float()
        dm_w = (tau_dm_e * delta_dm.float()).pow(2).mean().sqrt()
        self._acc.add("rel_gap", dm_w / ((tau_dm_e * vr).pow(2).mean().sqrt() + eps_r))
        self._acc.add(
            "mag_ratio", vf.pow(2).mean().sqrt() / (vr.pow(2).mean().sqrt() + eps_r)
        )
        self._acc.add("cos", (vf * vr).sum() / (vf.norm() * vr.norm() + eps_r))

    @torch.no_grad()
    def add_div(self, div_loss_t: torch.Tensor) -> None:
        """Accumulate the DP-DMD first-step diversity loss (pre-weight)."""
        self._acc.add("div", div_loss_t.detach().float())

    @torch.no_grad()
    def add_gan(
        self,
        gan_gen_loss: torch.Tensor,
        gan_disc_mean_t: torch.Tensor,
        gan_margin_t: torch.Tensor,
        gan_spread_t: torch.Tensor,
    ) -> None:
        """Accumulate the GAN generator/discriminator losses (pre-weight) plus
        the disc-update observables: margin = mean(real) − mean(fake) logit
        (disc discrimination strength — the hinge means sit at equilibrium even
        while per-logit structure collapses diversity, so read THIS, not the
        losses) and spread = std over the fake branch's logits (spatial
        structure of disc pressure under the token head; 0 for a single-logit
        pooled head)."""
        self._acc.add("gan_gen", gan_gen_loss.detach().float())
        self._acc.add("gan_disc", gan_disc_mean_t.detach().float())
        self._acc.add("gan_margin", gan_margin_t.detach().float())
        self._acc.add("gan_spread", gan_spread_t.detach().float())

    @torch.no_grad()
    def add_softrank(self, softrank_loss: torch.Tensor, *, active: bool) -> None:
        """Accumulate the soft-rank caption loss (pre-weight) + its run indicator.

        Amortized (``softrank.every_n``), so the flushed ``softrank`` mean is
        diluted by skipped steps; consumers normalize by ``softrank_active`` to
        recover the per-firing soft rank (0 = matched caption wins every pair).
        """
        self._acc.add("softrank", softrank_loss.detach().float())
        self._acc.add("softrank_active", 1.0 if active else 0.0)

    @torch.no_grad()
    def add_cdm(self, grad_cdm: torch.Tensor) -> None:
        """Accumulate the L_CDM gradient-signal rms (pre-weight).

        The surrogate loss value itself is a sign-random gradient vehicle (like
        ``loss_student``), so the health scalar is the rms of the detached
        off-trajectory delta — directly comparable to ``train/grad_signal_rms``.
        """
        self._acc.add("cdm", grad_cdm.detach().float().pow(2).mean().sqrt())

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
    ``{prefix}{i}``. GPU-resident like :class:`TurboMetrics` (``index_add_`` is
    async; one extra small sync per log boundary).

    NOT a gate signal: the raw per-τ profile is structurally nonuniform (the FM
    target ``ε − x0`` is intrinsically harder at some τ regardless of critic
    capacity) — it is a baseline for checking whether a τ-split flattens the
    *excess*.

    The loop's fake loss is a batch-mean scalar, so at B>1 it is attributed to
    every sample's τ bin (exact at the shipped ``batch_size=1``).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        nbins: int = TAU_PROFILE_BINS,
        prefix: str = "train/fake_loss_tau",
        aggregate_key: str | None = None,
    ):
        self.nbins = nbins
        self.prefix = prefix
        # Optional extra scalar: the count-weighted mean over ALL bins under
        # this key (per-bank overall fake loss when banks are split).
        self.aggregate_key = aggregate_key
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
        """Flush and push per-bin means to TensorBoard; empty bins are skipped.

        Always flushes (accumulators reset) even when ``writer`` is None, so
        call it unconditionally at every log boundary.
        """
        sums, cnts = self.flush()
        if writer is None:
            return
        for i, (s, n) in enumerate(zip(sums, cnts)):
            if n > 0:
                writer.add_scalar(f"{self.prefix}{i}", s / n, step)
        if self.aggregate_key is not None and sum(cnts) > 0:
            writer.add_scalar(self.aggregate_key, sum(sums) / sum(cnts), step)


def write_scalars(writer, m: FlushedMetrics, step: int) -> None:
    """Push every available scalar to TensorBoard at the canonical key names."""
    writer.add_scalar("train/fake_loss", m.fake, step)
    writer.add_scalar("train/grad_signal_rms", m.grad, step)
    writer.add_scalar("train/delta_dm_rms", m.dm, step)
    writer.add_scalar("train/x_pred_std", m.xpred, step)
    writer.add_scalar("train/v_student_rms", m.v_student, step)
    writer.add_scalar("train/dm_rel_gap", m.rel_gap, step)
    writer.add_scalar("train/dm_mag_ratio", m.mag_ratio, step)
    writer.add_scalar("train/dm_cos", m.cos, step)
    writer.add_scalar("train/div_loss", m.div, step)
    writer.add_scalar("train/gan_gen_loss", m.gan_gen, step)
    writer.add_scalar("train/gan_disc_loss", m.gan_disc, step)
    writer.add_scalar("train/gan_disc_margin", m.gan_margin, step)
    writer.add_scalar("train/gan_logit_spread", m.gan_spread, step)
    writer.add_scalar("train/softrank_active", m.softrank_active, step)
    writer.add_scalar("train/cdm_grad_rms", m.cdm, step)
    if m.softrank_active > 0:
        # Normalize the interval mean to active steps so the curve reads as the
        # per-firing matched-caption soft rank (→ 0 as discrimination recovers),
        # regardless of the every_n amortization.
        writer.add_scalar("train/softrank_loss", m.softrank / m.softrank_active, step)


def tqdm_rate(bar: tqdm) -> float | None:
    """Steps/sec off a tqdm bar: its EMA rate, or the run average as a fallback.

    ``format_dict['rate']`` is ``None`` until the bar has recorded an EMA
    interval, so fall back to (steps done / elapsed) rather than reporting no
    rate at all — the caller wants an ETA on the very first log line too.
    """
    fd = bar.format_dict
    rate = fd.get("rate")
    if rate:
        return rate
    elapsed = fd.get("elapsed") or 0.0
    done = (fd.get("n") or 0) - (fd.get("initial") or 0)
    return done / elapsed if elapsed > 0 and done > 0 else None


def console_step_line(
    m: FlushedMetrics, *, step: int, total: int, rate: float | None
) -> str:
    """Greppable one-line step summary for redirected / headless logs.

    Same numbers as :func:`tqdm_postfix`, but as a plain line: tqdm repaints a
    single ``\\r`` line on stderr, which leaves a redirected log with no
    periodic step record between the (500/1000-step) ckpt and val lines.
    """
    pct = f" ({100.0 * step / total:.1f}%)" if total else ""
    stats = " ".join(f"{k}={v}" for k, v in tqdm_postfix(m).items())
    if rate:
        speed = f"{rate:.2f} it/s | ETA {tqdm.format_interval((total - step) / rate)}"
    else:
        speed = "? it/s | ETA ?"
    return f"step {step}/{total}{pct} | {stats} | {speed}"


def tqdm_postfix(m: FlushedMetrics) -> dict:
    """tqdm postfix dict — short keys for the live progress line."""
    postfix = {
        "g": f"{m.grad:.2e}",
        "relg": f"{m.rel_gap:.3f}",
        "cos": f"{m.cos:.3f}",
        "xp": f"{m.xpred:.3f}",
        "fake": f"{m.fake:.2e}",
    }
    if m.div > 0:
        postfix["div"] = f"{m.div:.3f}"
    if m.gan_gen != 0 or m.gan_disc != 0:
        postfix["gen"] = f"{m.gan_gen:.3f}"
        postfix["dsc"] = f"{m.gan_disc:.3f}"
        postfix["mgn"] = f"{m.gan_margin:.3f}"
    if m.softrank_active > 0:
        postfix["srank"] = f"{m.softrank / m.softrank_active:.3f}"
    if m.cdm != 0:
        postfix["cdm"] = f"{m.cdm:.2e}"
    return postfix
