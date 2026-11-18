"""GPU-side accumulators for the turbo loop, read in one sync per ``log_interval``.

Scalars (TB names in :func:`write_scalars`; how to read them:
``docs/anima/methods/turbo.md`` § Reading the metrics):

* ``grad`` rms of the DM grad signal into x_pred; ``dm`` rms(v_real − v_fake)
* ``xpred`` std of x_pred; ``v_student`` rms of the step-0 velocity
* ``rel_gap`` rms(τ·Δ_dm) / rms(τ·v_real_dm); ``mag_ratio`` rms(v_fake_dm) /
  rms(v_real_dm); ``cos`` cosine(v_fake_dm, v_real_dm)
* ``div`` first-step diversity MSE (pre-weight)
* ``fm`` / ``fm_ratio`` anchor feature matching (pre-weight) and the rms of the
  weighted feature-path gradient (FM + paired GAN) into v over the anchor MSE's
* ``gan_share`` paired GAN: rms of its gradient over that of FM + GAN, on the
  teacher features
* ``gn_*`` pre-clip gradient norm per optimizer (above ``grad_clip`` = clipped)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

import torch
from tqdm import tqdm

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
    fm: float
    fm_ratio: float
    gan_gen: float
    gan_disc: float
    gan_margin: float
    gan_spread: float
    gan_share: float
    cdm: float
    gn_student: float
    gn_fake: float
    gn_disc: float


_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FlushedMetrics))


class TurboMetrics:
    """:class:`ScalarAccumulator` behind a typed :class:`FlushedMetrics`. Every
    field is pre-touched, so a disabled path (``gan_*`` without the GAN) still
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
    def add_grad_norms(
        self,
        student: torch.Tensor,
        fake: torch.Tensor,
        disc: torch.Tensor | None = None,
    ) -> None:
        """Accumulate the pre-clip parameter-gradient norms."""
        self._acc.add("gn_student", student.detach().float())
        self._acc.add("gn_fake", fake.detach().float())
        if disc is not None:
            self._acc.add("gn_disc", disc.detach().float())

    @torch.no_grad()
    def add_div(self, div_loss_t: torch.Tensor) -> None:
        """Accumulate the DP-DMD first-step diversity loss (pre-weight)."""
        self._acc.add("div", div_loss_t.detach().float())

    @torch.no_grad()
    def add_fm(self, fm: torch.Tensor, ratio: torch.Tensor) -> None:
        """Accumulate the anchor feature-matching loss and its gradient ratio."""
        self._acc.add("fm", fm.detach().float())
        self._acc.add("fm_ratio", ratio.detach().float())

    @torch.no_grad()
    def add_gan(
        self,
        gan_gen_loss: torch.Tensor,
        gan_disc_mean_t: torch.Tensor,
        gan_margin_t: torch.Tensor,
        gan_spread_t: torch.Tensor,
    ) -> None:
        """Accumulate the GAN losses (pre-weight) and the disc margin / spread
        (see :class:`steps.FakeUpdate`)."""
        self._acc.add("gan_gen", gan_gen_loss.detach().float())
        self._acc.add("gan_disc", gan_disc_mean_t.detach().float())
        self._acc.add("gan_margin", gan_margin_t.detach().float())
        self._acc.add("gan_spread", gan_spread_t.detach().float())

    @torch.no_grad()
    def add_gan_share(self, share: torch.Tensor) -> None:
        """Accumulate the paired GAN's share of the feature-path gradient."""
        self._acc.add("gan_share", share.detach().float())

    @torch.no_grad()
    def add_cdm(self, grad_cdm: torch.Tensor) -> None:
        """Accumulate the rms of the L_CDM grad signal (pre-weight), comparable to
        ``train/grad_signal_rms``; the surrogate loss value has a random sign."""
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

    The interval means of ``gn_*`` cannot be paired with the σ_a or g that
    produced them, so this keeps each iteration. Rows are GPU-resident until
    ``flush`` (one sync per log boundary) and appended to ``path``, so a resumed
    run continues the same file. Fields: ``step``, ``gn_student``, ``gn_fake``,
    ``gn_disc`` (null with the GAN off), ``sigma_a`` (DM query level of the first
    draw, batch mean), ``s_g`` (σ at the grad step), ``g``, ``n`` (rollout length).
    """

    def __init__(self, path, device: torch.device):
        self.path = path
        self.device = device
        self._rows: list[torch.Tensor] = []
        self._meta: list[tuple[int, float, int, int]] = []

    @torch.no_grad()
    def add(
        self,
        step: int,
        gn_student: torch.Tensor,
        gn_fake: torch.Tensor,
        gn_disc: torch.Tensor | None,
        sigma_a: torch.Tensor,
        s_g: float,
        g: int,
        n: int,
    ) -> None:
        nan = torch.full((), float("nan"), device=self.device)
        self._rows.append(
            torch.stack(
                [
                    gn_student.detach().float().to(self.device),
                    gn_fake.detach().float().to(self.device),
                    nan
                    if gn_disc is None
                    else gn_disc.detach().float().to(self.device),
                    sigma_a.detach().float().mean().to(self.device),
                ]
            )
        )
        self._meta.append((step, float(s_g), int(g), int(n)))

    def flush(self) -> None:
        if not self._rows:
            return
        vals = torch.stack(self._rows).tolist()
        with open(self.path, "a", encoding="utf-8") as f:
            for (step, s_g, g, n), (gs, gf, gd, sa) in zip(self._meta, vals):
                row = {
                    "step": step,
                    "gn_student": round(gs, 5),
                    "gn_fake": round(gf, 5),
                    "gn_disc": None if gd != gd else round(gd, 5),
                    "sigma_a": round(sa, 5),
                    "s_g": round(s_g, 5),
                    "g": g,
                    "n": n,
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
    if m.fm != 0:
        writer.add_scalar("train/fm_loss", m.fm, step)
    if m.fm_ratio != 0:
        writer.add_scalar("train/fm_grad_ratio", m.fm_ratio, step)
    if anchor_only:
        if m.gan_disc != 0:
            writer.add_scalar("train/gan_gen_loss", m.gan_gen, step)
            writer.add_scalar("train/gan_disc_loss", m.gan_disc, step)
            writer.add_scalar("train/gan_disc_margin", m.gan_margin, step)
            writer.add_scalar("train/gan_logit_spread", m.gan_spread, step)
            writer.add_scalar("train/gan_grad_share", m.gan_share, step)
            writer.add_scalar("train/grad_norm_disc", m.gn_disc, step)
        return
    writer.add_scalar("train/fake_loss", m.fake, step)
    writer.add_scalar("train/grad_signal_rms", m.grad, step)
    writer.add_scalar("train/delta_dm_rms", m.dm, step)
    writer.add_scalar("train/x_pred_std", m.xpred, step)
    writer.add_scalar("train/v_student_rms", m.v_student, step)
    writer.add_scalar("train/dm_rel_gap", m.rel_gap, step)
    writer.add_scalar("train/dm_mag_ratio", m.mag_ratio, step)
    writer.add_scalar("train/dm_cos", m.cos, step)
    writer.add_scalar("train/gan_gen_loss", m.gan_gen, step)
    writer.add_scalar("train/gan_disc_loss", m.gan_disc, step)
    writer.add_scalar("train/gan_disc_margin", m.gan_margin, step)
    writer.add_scalar("train/gan_logit_spread", m.gan_spread, step)
    writer.add_scalar("train/cdm_grad_rms", m.cdm, step)
    writer.add_scalar("train/grad_norm_fake", m.gn_fake, step)
    if m.gn_disc != 0:
        writer.add_scalar("train/grad_norm_disc", m.gn_disc, step)


def tqdm_rate(bar: tqdm) -> float | None:
    """Steps/sec off a tqdm bar: its EMA rate, else steps done / elapsed."""
    fd = bar.format_dict
    rate = fd.get("rate")
    if rate:
        return rate
    elapsed = fd.get("elapsed") or 0.0
    done = (fd.get("n") or 0) - (fd.get("initial") or 0)
    return done / elapsed if elapsed > 0 and done > 0 else None


def console_step_line(
    m: FlushedMetrics,
    *,
    step: int,
    total: int,
    rate: float | None,
    anchor_only: bool = False,
) -> str:
    """Greppable one-line step summary (the :func:`tqdm_postfix` numbers) for
    redirected / headless logs."""
    pct = f" ({100.0 * step / total:.1f}%)" if total else ""
    stats = " ".join(
        f"{k}={v}" for k, v in tqdm_postfix(m, anchor_only=anchor_only).items()
    )
    if rate:
        speed = f"{rate:.2f} it/s | ETA {tqdm.format_interval((total - step) / rate)}"
    else:
        speed = "? it/s | ETA ?"
    return f"step {step}/{total}{pct} | {stats} | {speed}"


def tqdm_postfix(m: FlushedMetrics, *, anchor_only: bool = False) -> dict:
    """tqdm postfix dict — short keys for the live progress line. ``anchor_only``
    shows the anchor MSE (``div``) and the student grad norm, the only live numbers."""
    if anchor_only:
        out = {"anchor": f"{m.div:.4f}", "gn": f"{m.gn_student:.2e}"}
        if m.fm != 0:
            out["fm"] = f"{m.fm:.4f}"
        if m.fm_ratio != 0:
            out["fmr"] = f"{m.fm_ratio:.2f}"
        if m.gan_disc != 0:
            out["gen"] = f"{m.gan_gen:.3f}"
            out["dsc"] = f"{m.gan_disc:.3f}"
            out["mgn"] = f"{m.gan_margin:.3f}"
            out["gsh"] = f"{m.gan_share:.2f}"
        return out
    postfix = {
        "g": f"{m.grad:.2e}",
        "relg": f"{m.rel_gap:.3f}",
        "cos": f"{m.cos:.3f}",
        "xp": f"{m.xpred:.3f}",
        "fake": f"{m.fake:.2e}",
        "gn": f"{m.gn_student:.2e}",
    }
    if m.div > 0:
        postfix["div"] = f"{m.div:.3f}"
    if m.gan_gen != 0 or m.gan_disc != 0:
        postfix["gen"] = f"{m.gan_gen:.3f}"
        postfix["dsc"] = f"{m.gan_disc:.3f}"
        postfix["mgn"] = f"{m.gan_margin:.3f}"
    if m.cdm != 0:
        postfix["cdm"] = f"{m.cdm:.2e}"
    return postfix
