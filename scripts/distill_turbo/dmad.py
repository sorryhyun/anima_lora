"""DMAD Phase 0 — a discriminator carries the student's distribution-matching signal.

docs/proposal/turbo_dmad.md § Phase 0. DMAD's Prop. 1: the balanced-BCE optimum
is ``h* = log p_target − log p_student``, so ``∂(−h)/∂x`` is the score
difference DMD estimates with a teacher + fake critic. Under ``[dmad]`` the DM
term and the critic are replaced by :class:`DmadDisc`:

* Backbone: the teacher DiT + a cold (zero-init) LoRA stack
  (``turbo.make_aux_stack``), tapped at one block (``return_features_early``).
* Head T scores teacher samples (the step-0 anchor rollout finished to σ = 0 on
  the teacher's CFG grid, :func:`finish_anchor_rollout`) against the student's
  ``x_pred``; head R scores real latents against ``x_pred``. Each head is a
  ``TeacherFeatureDiscriminator`` token head whose logits are averaged into one
  logit per sample. A head with λ = 0 is not built and adds no branch.
* Update: a replay window of the newest ``window`` pairs, one fresh (τ, ε) per
  pair from a dedicated generator; per pair the target branches and one student
  pass (both heads) are backwarded one at a time, losses scaled 1/n, then one
  AdamW (β = (0, 0.99)) step.
* Student signal: ``g = ∂(−λ_T h_T − λ_R h_R)/∂x_pred`` at the step's (τ, ε),
  RMS-normalized per sample to ``signal_rms`` (``signal_rms = 0``: raw, so λ
  alone sets its size) — the ``grad_signal`` the loop assembles exactly like
  the DM one.
* Gap routing (``gap_routing``, the official H3 trainer's rule): per critic-τ
  decile band, a bias-corrected EMA of ``h_R(real) − h_R(teacher)``; head T's
  weight in the student signal is ``sigmoid((median − gap_b)/gap_tau) / E_w``
  over the ready bands. The gap is tracked whenever both heads exist.

:func:`run_disc_warmup` trains the disc alone for ``disc_warmup_steps`` before
the student uses it. The Phase −1 probe (``dmad_probe.py``) shares the
module-level helpers here.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from tqdm import tqdm

from library.anima.uncond import uncond_for_batch
from networks.methods.turbo_dmd import TeacherFeatureDiscriminator

from .primitives import renoise, sample_dynamic_sigmas
from .steps import selective_block_grad_ckpt, teacher_anchor

logger = logging.getLogger(__name__)

# Offset on cfg.seed for the disc's own generator (the probe uses 7919).
_GEN_SEED_OFFSET = 7927

# Gap routing constants, as the official MiniMax-H3 DMAD trainer.
GAP_BANDS = 10
GAP_EMA_BETA = 0.99
GAP_READY_MIN = 5  # ready bands before the routing engages
GAP_BAND_MIN_COUNT = 10  # updates before a band counts as ready


def resolve_tap_block(model, idx: int, key: str) -> int:
    """``idx`` (−1 → middle block) checked against the DiT's depth."""
    bidx = model.num_blocks // 2 if idx < 0 else idx
    if not 0 <= bidx < model.num_blocks:
        raise ValueError(
            f"{key} resolved to {bidx}, out of range [0, {model.num_blocks})"
        )
    return bidx


@contextmanager
def disc_view(turbo, stack):
    """Teacher view + ``stack`` on: the base DiT as the disc backbone."""
    turbo.set_view("teacher")
    stack.set_enabled(True)
    try:
        yield
    finally:
        stack.set_enabled(False)


def gen_randn_like(gen: torch.Generator, x: torch.Tensor) -> torch.Tensor:
    return torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)


def gen_rand_tau(gen: torch.Generator, B: int, device, dtype) -> torch.Tensor:
    return torch.rand(B, generator=gen, device=device).to(dtype)


@torch.no_grad()
def finish_anchor_rollout(ctx, cfg, eps, v_target, c, c_null, B) -> torch.Tensor:
    """Teacher sample: the step-0 anchor rollout z_tk → σ = 0 on the CFG grid.

    ``z_tk`` is recovered from ``v_target`` (= (ε − z_tk)/(1 − t_k)), so the
    teacher and student samples share their noise ε.
    """
    z = (eps.float() - (1.0 - ctx.t_k_anchor) * v_target).to(ctx.dtype)
    sig = ctx.teacher_anchor_sigmas
    for i in range(cfg.k_anchor, cfg.teacher_anchor_steps):
        t_b = torch.full((B,), sig[i], device=ctx.device, dtype=ctx.dtype)
        v = ctx.teacher_cfg_velocity(z, t_b, c, c_null)
        z = (z.float() - (sig[i] - sig[i + 1]) * v).to(ctx.dtype)
    return z


@torch.no_grad()
def rollout_x_pred(ctx, cfg, eps, c, B) -> torch.Tensor:
    """No-grad replica of the loop's anchored student rollout → ``x_pred``.

    Same grid (static or a dynamic draw) and grad-step choice as
    ``distill.run_loop``: ``grad_step='random'`` returns the one-step x0
    prediction at g ~ U{1..N-1}; otherwise the rollout endpoint (the last
    step's x0 prediction, since σ_N = 0).
    """
    if cfg.dynamic_schedule:
        sig = sample_dynamic_sigmas(ctx.dyn_n_min, cfg.student_steps)
        n = len(sig) - 1
    else:
        sig, n = ctx.student_sigmas, cfg.student_steps
    g = (
        int(torch.randint(1, n, (1,)).item())
        if cfg.dmd_grad_step == "random"
        else n - 1
    )
    x = eps
    for i in range(g):
        t_b = torch.full((B,), sig[i], device=ctx.device, dtype=ctx.dtype)
        ctx.turbo.set_student_step(i)
        v = ctx.forward("student", x, t_b, c, no_grad=True).squeeze(2)
        x = x - (sig[i] - sig[i + 1]) * v
    t_b = torch.full((B,), sig[g], device=ctx.device, dtype=ctx.dtype)
    ctx.turbo.set_student_step(g)
    v = ctx.forward("student", x, t_b, c, no_grad=True).squeeze(2)
    return x - sig[g] * v


def normalize_signal_rms(g: torch.Tensor, target: float) -> torch.Tensor:
    """Scale each sample of ``g`` to RMS ``target``; an all-zero sample stays 0.

    Pre-divides by the per-sample max so a tiny gradient does not underflow
    in the square.
    """
    g = g.float()
    dims = tuple(range(1, g.ndim))
    peak = g.abs().amax(dim=dims, keepdim=True)
    unit = g / peak.clamp_min(torch.finfo(g.dtype).tiny)
    rms = unit.pow(2).mean(dim=dims, keepdim=True).sqrt()
    return unit * (target / rms.clamp_min(torch.finfo(g.dtype).tiny))


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean().sqrt()


class DmadDisc:
    """Disc backbone stack + heads T / R + optimizer + replay window."""

    def __init__(self, cfg, *, turbo, model, device, dtype) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.lambda_t = float(cfg.dmad_lambda_t)
        self.lambda_r = float(cfg.dmad_lambda_r)
        self.signal_rms = float(cfg.dmad_signal_rms)
        self.grad_clip = float(cfg.dmad_grad_clip)
        self.gap_routing = bool(cfg.dmad_gap_routing)
        self.gap_tau = float(cfg.dmad_gap_tau)
        self._gap_ema = torch.zeros(GAP_BANDS, device=self.device)
        self._gap_cnt = torch.zeros(GAP_BANDS, device=self.device)
        self.stack = turbo.make_aux_stack()
        self.stack.to(device=self.device, dtype=dtype)
        self.tap = resolve_tap_block(
            model, cfg.dmad_feature_block_idx, "dmad.feature_block_idx"
        )

        def _head():
            return TeacherFeatureDiscriminator(
                inner_dim=model.model_channels, num_taps=1, granularity="token"
            ).to(device=self.device)

        self.head_t = _head() if self.lambda_t > 0 else None
        self.head_r = _head() if self.lambda_r > 0 else None
        self.stack_params = [p for p in self.stack.parameters() if p.requires_grad]
        self.params = self.stack_params + [
            p for head in self._heads().values() for p in head.parameters()
        ]
        self.opt = torch.optim.AdamW(
            self.params,
            lr=cfg.dmad_lr,
            weight_decay=0.0,
            betas=(0.0, 0.99),
            fused=self.device.type == "cuda",
        )
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(cfg.seed) + _GEN_SEED_OFFSET)
        self.pairs: deque = deque(maxlen=cfg.dmad_window)
        logger.info(
            f"DMAD disc: cold stack + heads {'/'.join(self._heads())} on block "
            f"{self.tap}, {sum(p.numel() for p in self.params):,} params, "
            f"lr={cfg.dmad_lr}, window={cfg.dmad_window}, "
            f"gap_routing={self.gap_routing}"
        )

    def _heads(self) -> dict[str, torch.nn.Module]:
        heads = {}
        if self.head_t is not None:
            heads["t"] = self.head_t
        if self.head_r is not None:
            heads["r"] = self.head_r
        return heads

    def _features(self, ctx, x_t, tau, c, *, no_grad: bool) -> torch.Tensor:
        feats = ctx.forward(
            "teacher",
            x_t,
            tau,
            c,
            no_grad=no_grad,
            return_block_features={self.tap},
            return_features_early=True,
        )
        return feats[self.tap]

    @staticmethod
    def _logit(head, f: torch.Tensor) -> torch.Tensor:
        """One logit per sample: the token head's logits averaged, (B, 1)."""
        return head([f]).mean(dim=1, keepdim=True)

    @staticmethod
    def _band(tau: torch.Tensor) -> torch.Tensor:
        """CDF-decile band of each τ; τ ~ U(0, 1), so the CDF is τ itself."""
        return (tau.float() * GAP_BANDS).long().clamp(0, GAP_BANDS - 1)

    @torch.no_grad()
    def _gap_update(
        self, tau: torch.Tensor, h_real: torch.Tensor, h_teacher: torch.Tensor
    ) -> None:
        band = self._band(tau)
        gap = (h_real - h_teacher).float().view(-1)
        for i in range(gap.numel()):
            b = band[i]
            self._gap_ema[b] = (
                GAP_EMA_BETA * self._gap_ema[b] + (1.0 - GAP_EMA_BETA) * gap[i]
            )
            self._gap_cnt[b] += 1

    @torch.no_grad()
    def gap_weight(self, tau: torch.Tensor) -> torch.Tensor:
        """Head T's per-sample weight ``w_b / E_w``, (B,); 1 until
        ``GAP_READY_MIN`` bands are ready, and for a sample whose band is not."""
        ones = torch.ones(tau.shape[0], device=self.device)
        ready = self._gap_cnt >= GAP_BAND_MIN_COUNT
        if int(ready.sum()) < GAP_READY_MIN:
            return ones
        corr = self._gap_ema / (1.0 - GAP_EMA_BETA ** self._gap_cnt.clamp(min=1.0))
        w = torch.sigmoid((corr[ready].median() - corr) / self.gap_tau)
        e_w = w[ready].mean().clamp(min=1e-4)
        band = self._band(tau)
        return torch.where(ready[band], w[band] / e_w, ones)

    def _set_trainable(self, flag: bool) -> None:
        for p in self.params:
            p.requires_grad_(flag)

    def accumulate(
        self,
        ctx,
        *,
        x_teacher: torch.Tensor,
        x_student: torch.Tensor,
        x_real: torch.Tensor,
        crossattn_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Push the newest pair and backward the disc loss over the window.

        Per pair, at one (τ, ε): head T's teacher branch, head R's real
        branch, then one student pass scored by every head. Each branch
        backwards on its own, through the compiled blocks (no checkpointing).
        Losses are scaled 1/n, so the accumulated grad is that of the
        window-mean of BCE_T + BCE_R.

        Returns detached stats: the newest pair's per-head BCE / margin /
        rank accuracy (scored before the disc trains on it), the real − teacher
        gap of h_R on that pair (needs both heads), and the window loss.
        """
        self.pairs.appendleft(
            (
                x_teacher.detach().to(self.dtype) if self.head_t is not None else None,
                x_student.detach().to(self.dtype),
                x_real.detach().to(self.dtype) if self.head_r is not None else None,
                crossattn_emb.detach(),
            )
        )
        n = len(self.pairs)
        heads = self._heads()
        loss_sum = torch.zeros((), device=self.device)
        stats: dict[str, torch.Tensor] = {}
        for j, (xt, xs, xr, c) in enumerate(self.pairs):
            B = xs.shape[0]
            tau = gen_rand_tau(self.gen, B, self.device, self.dtype)
            eps = gen_randn_like(self.gen, xs)
            h_tgt: dict[str, torch.Tensor] = {}
            h_r_teacher = None
            with disc_view(ctx.turbo, self.stack):
                for key, x_tgt in (("t", xt), ("r", xr)):
                    if key not in heads:
                        continue
                    x_in = renoise(x_tgt, tau, eps)
                    f = self._features(ctx, x_in, tau, c, no_grad=False)
                    h = self._logit(heads[key], f)
                    loss = F.softplus(-h).mean()
                    (loss / n).backward()
                    loss_sum = loss_sum + loss.detach()
                    h_tgt[key] = h.detach()
                    if key == "t" and "r" in heads and j == 0:
                        with torch.no_grad():
                            h_r_teacher = self._logit(heads["r"], f.detach())
                x_in = renoise(xs, tau, eps)
                f = self._features(ctx, x_in, tau, c, no_grad=False)
                h_s = {key: self._logit(head, f) for key, head in heads.items()}
                loss = sum(F.softplus(h).mean() for h in h_s.values())
                (loss / n).backward()
                loss_sum = loss_sum + loss.detach()
            if j == 0:
                for key in heads:
                    ht, hs = h_tgt[key], h_s[key].detach()
                    stats[f"bce_{key}"] = F.softplus(-ht).mean() + F.softplus(hs).mean()
                    stats[f"margin_{key}"] = (ht - hs).mean()
                    stats[f"rank_acc_{key}"] = (ht > hs).float().mean()
                if h_r_teacher is not None:
                    stats["gap_r"] = (h_tgt["r"] - h_r_teacher).mean()
                    self._gap_update(tau, h_tgt["r"], h_r_teacher)
        stats["loss"] = loss_sum / n
        stats["n_pairs"] = torch.tensor(float(n), device=self.device)
        if "r" in heads and "t" in heads:
            stats["gap_ready"] = (self._gap_cnt >= GAP_BAND_MIN_COUNT).sum().float()
        return stats

    def step(self) -> torch.Tensor:
        """One optimizer step on the accumulated grads; returns the grad norm."""
        gn = torch.nn.utils.clip_grad_norm_(
            self.params,
            max_norm=self.grad_clip if self.grad_clip > 0 else float("inf"),
        )
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return gn.detach()

    def update(self, ctx, **pair) -> dict[str, torch.Tensor]:
        """:meth:`accumulate` the newest pair over the window, then :meth:`step`."""
        stats = self.accumulate(ctx, **pair)
        stats["grad_norm"] = self.step()
        return stats

    def student_signal(
        self,
        ctx,
        x_pred: torch.Tensor,
        tau: torch.Tensor,
        eps: torch.Tensor,
        crossattn_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``∂(−λ_T h_T − λ_R h_R)/∂x_pred`` at (τ, ε), RMS-normalized per sample
        unless ``signal_rms`` is 0.

        One grad-bearing disc forward on a detached leaf of ``x_pred``, under
        block checkpointing: the student's grad graph is still alive here, and
        the two together do not fit 16 GB. Disc params are frozen around it,
        since the unsloth checkpoint's recompute backward would otherwise
        accumulate into their ``.grad``. With both
        heads on, each head's input gradient is taken separately (two backwards
        through the tapped half of the DiT) for the raw-RMS / cosine stats.
        Returns ``(grad_signal fp32 detached, stats)``.
        """
        heads = self._heads()
        weights = {"t": self.lambda_t, "r": self.lambda_r}
        w_gap = None
        if self.gap_routing:
            w_gap = self.gap_weight(tau)
            weights["t"] = self.lambda_t * w_gap.view(-1, *([1] * (x_pred.ndim - 1)))
        grads: dict[str, torch.Tensor] = {}
        self._set_trainable(False)
        try:
            x_in = x_pred.detach().to(self.dtype).requires_grad_()
            with disc_view(ctx.turbo, self.stack), selective_block_grad_ckpt(ctx.model):
                f = self._features(
                    ctx, renoise(x_in, tau, eps), tau, crossattn_emb, no_grad=False
                )
                keys = list(heads)
                for i, key in enumerate(keys):
                    h = self._logit(heads[key], f)
                    (g,) = torch.autograd.grad(
                        h.sum(), x_in, retain_graph=i < len(keys) - 1
                    )
                    grads[key] = g.float()
        finally:
            self._set_trainable(True)
        g = -sum(weights[key] * grads[key] for key in grads)
        stats = {f"g_{key}_rms": _rms(grads[key]) for key in grads}
        if w_gap is not None:
            stats["gap_w"] = w_gap.mean()
        if len(grads) == 2:
            stats["cos_tr"] = F.cosine_similarity(
                grads["t"].flatten(), grads["r"].flatten(), dim=0
            )
        if self.signal_rms > 0:
            g = normalize_signal_rms(g, self.signal_rms)
        return g.detach(), stats


def run_disc_warmup(ctx, cfg):
    """Disc-only head start: ``dmad_disc_warmup_steps`` updates, student untouched.

    Per step, on a fresh batch: a no-grad student rollout
    (:func:`rollout_x_pred`), the teacher sample (K-step anchor + its finish;
    skipped without head T), then one :meth:`DmadDisc.update` — so the replay
    window is full when the main loop starts. Returns the advanced data
    iterator.
    """
    from .metrics import DmadMetrics

    steps = cfg.dmad_disc_warmup_steps
    data_iter = ctx.data_iter
    if steps <= 0:
        return data_iter
    logger.info(f"DMAD disc head-start: {steps} disc-only updates")
    metrics = DmadMetrics(ctx.device)
    for w in tqdm(range(steps), desc="dmad-disc-warmup"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(ctx.dataloader)
            batch = next(data_iter)
        latents = batch["latents"].to(ctx.device, dtype=ctx.dtype, non_blocking=True)
        c = batch["crossattn_emb"].to(ctx.device, dtype=ctx.dtype, non_blocking=True)
        B = latents.shape[0]
        torch.compiler.cudagraph_mark_step_begin()
        eps = torch.randn_like(latents)
        x_pred = rollout_x_pred(ctx, cfg, eps, c, B)
        x_teacher = None
        if ctx.dmad.head_t is not None:
            c_null = uncond_for_batch(ctx.uncond_base, c)
            v_target = teacher_anchor(ctx, cfg, eps, c, c_null, B)
            x_teacher = finish_anchor_rollout(ctx, cfg, eps, v_target, c, c_null, B)
        metrics.add(
            ctx.dmad.update(
                ctx,
                x_teacher=x_teacher,
                x_student=x_pred,
                x_real=latents,
                crossattn_emb=c,
            )
        )
        if (w + 1) % cfg.log_interval == 0 or w + 1 == steps:
            m = metrics.flush()
            metrics.write(ctx.writer, m, w + 1, prefix="warmup/dmad/")
            logger.info(f"[dmad warmup {w + 1}/{steps}] {DmadMetrics.line(m)}")
    return data_iter
