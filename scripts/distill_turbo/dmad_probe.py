"""DMAD Phase −1 — does a teacher-vs-student discriminator's gradient point along DM?

Measure-only (docs/proposal/turbo_dmad.md § Phase −1). DMAD's Prop. 1 says the
balanced-BCE optimum is ``h* = log p_teacher − log p_student``, so the generator
loss ``−h`` has gradient ``s_student − s_teacher`` — the same direction as the DMD
surrogate's ``grad_signal``. This module trains such a discriminator alongside an
unchanged DP-DMD run and reads how well its generator gradient agrees with the DM
signal the student is actually trained on.

Disc = its own fake-shaped LoRA stack on the frozen DiT (warm-started like the
fake) + head T over one block's tokens. The critic is not touched. Per step:

1. Teacher sample: finish the step-0 anchor rollout (``k_anchor`` →
   ``teacher_anchor_steps`` CFG Euler steps) from the same ε, so the teacher and
   student samples share their noise.
2. Disc update: renoise teacher and ``x_pred`` at one (τ, ε), balanced BCE —
   ``disc_steps`` times, a fresh (τ, ε) each. With ``window`` > 1 each update
   accumulates over the newest ``window`` pairs (a replay window: the student is
   batch 1, so this is the disc's batch). ``scalar_logit`` averages the token
   head's logits into one logit per sample before the BCE.
3. Probe: ``g_T = ∂(−h_T)/∂x_pred`` at the DMD's own (τ_dm, ε_dm), compared with
   ``grad_signal`` (pre f-distill reweight) — cosine and agree-energy, each with a
   permutation null on the same tensors. With ``ceiling``, a second DM estimate
   at an independent (τ', ε') gives the alignment two DM draws reach with each
   other.

Every random draw here comes from a dedicated generator, so the training RNG
stream — and with it the student/critic numerics — is the same as with the probe
off. ``stop_on_collapse`` ends the run once the disc's margin has gone flat.
One JSONL row per step lands in the run's log dir; read it with
``bench/turbo/dmad_probe_read.py``.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from networks.methods.turbo_dmd import (
    TeacherFeatureDiscriminator,
    gan_loss_discriminator,
    warm_start_plain_lora,
)

from .dmad import (
    disc_view,
    finish_anchor_rollout,
    gen_rand_tau,
    gen_randn_like,
    resolve_tap_block,
)
from .primitives import renoise
from .steps import selective_block_grad_ckpt

logger = logging.getLogger(__name__)


def _acc(h_t: torch.Tensor, h_s: torch.Tensor) -> float:
    return float(0.5 * ((h_t > 0).float().mean() + (h_s < 0).float().mean()))


def pair_stats(x_t: torch.Tensor, x_s: torch.Tensor) -> dict[str, float]:
    """How the teacher and student samples differ, before any renoise.

    ``dc_share`` = share of ``‖x_t − x_s‖²`` carried by the per-channel spatial
    means — a global colour / brightness offset survives heavy noise, so a disc
    can separate on it at high τ. ``ch_std_logratio`` = RMS over channels of
    ``log(std_t / std_s)`` (contrast / saturation).
    """
    x_t = x_t.detach().float()
    x_s = x_s.detach().float()
    sp = tuple(range(2, x_t.ndim))
    diff = x_t - x_s
    dc = diff.mean(dim=sp, keepdim=True)
    total = diff.pow(2).mean()
    log_ratio = torch.log(
        x_t.std(dim=sp).clamp_min(1e-6) / x_s.std(dim=sp).clamp_min(1e-6)
    )
    return {
        "pair_rms": float(total.sqrt()),
        "dc_share": float(dc.pow(2).mean() / total.clamp_min(1e-30)),
        "ch_std_logratio": float(log_ratio.pow(2).mean().sqrt()),
    }


def alignment_stats(
    g: torch.Tensor, d: torch.Tensor, generator: torch.Generator | None = None
) -> dict[str, float]:
    """Cosine and agree-energy of ``g`` against ``d``, plus a permutation null.

    agree-energy = the share of ``g``'s energy on elements whose sign matches
    ``d`` (0.5 for unrelated tensors, 1.0 for ``g = d``). The null re-scores
    ``g`` against an elementwise permutation of ``d`` drawn from ``generator``.
    """
    g = g.detach().float().flatten()
    d = d.detach().float().flatten()
    e = g * g
    e_sum = e.sum().clamp_min(1e-30)

    def _score(ref: torch.Tensor) -> tuple[float, float]:
        cos = torch.nn.functional.cosine_similarity(g, ref, dim=0)
        agree = (e * (torch.sign(g) == torch.sign(ref))).sum() / e_sum
        return float(cos), float(agree)

    cos, agree = _score(d)
    perm = torch.randperm(d.numel(), generator=generator, device=d.device)
    cos_null, agree_null = _score(d[perm])
    return {
        "cos": cos,
        "agree": agree,
        "cos_null": cos_null,
        "agree_null": agree_null,
    }


class DmadProbe:
    """Side discriminator + per-step alignment readout (see module docstring)."""

    def __init__(self, cfg, *, turbo, model, device, dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.stack = turbo.make_aux_stack()
        self.stack.to(device=device, dtype=dtype)
        if cfg.dmad_probe_warm_start and cfg.fake_init_weights:
            warm_start_plain_lora(self.stack, cfg.fake_init_weights, "dmad_disc")
        bidx = resolve_tap_block(
            model, cfg.dmad_probe_feature_block_idx, "dmad_probe.feature_block_idx"
        )
        self.tap = bidx
        self.head = TeacherFeatureDiscriminator(
            inner_dim=model.model_channels,
            num_taps=1,
            granularity=cfg.dmad_probe_head,
        ).to(device=device)
        self.stack_params = [p for p in self.stack.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            self.stack_params + list(self.head.parameters()),
            lr=cfg.dmad_probe_lr,
            weight_decay=0.0,
            betas=(0.0, 0.99),
            fused=torch.cuda.is_available(),
        )
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(cfg.seed) + 7919)
        self.pairs: deque = deque(maxlen=cfg.dmad_probe_window)
        self.out_path: Path | None = None
        self.collapsed = False
        self._flat_run = 0
        self._t = 0.0
        n = sum(p.numel() for p in self.stack_params) + sum(
            p.numel() for p in self.head.parameters()
        )
        logger.info(
            f"DMAD probe: disc stack + head T on block {bidx} "
            f"({cfg.dmad_probe_head}), {n:,} params, lr={cfg.dmad_probe_lr}, "
            f"window={cfg.dmad_probe_window}, scalar_logit={cfg.dmad_probe_scalar_logit}, "
            f"warm_start={cfg.dmad_probe_warm_start and bool(cfg.fake_init_weights)}"
        )

    def bind_log_dir(self, log_dir: str | Path) -> None:
        self.out_path = Path(log_dir) / "dmad_probe.jsonl"
        logger.info(f"DMAD probe rows → {self.out_path}")

    def _disc_view(self, turbo):
        return disc_view(turbo, self.stack)

    def _h(self, ctx, x_t, tau, c, *, no_grad: bool) -> torch.Tensor:
        feats = ctx.forward(
            "teacher",
            x_t,
            tau,
            c,
            no_grad=no_grad,
            return_block_features={self.tap},
            return_features_early=True,
        )
        h = self.head([feats[self.tap]])  # (B, 1) pooled | (B, N) token
        if self.cfg.dmad_probe_scalar_logit:
            h = h.mean(dim=1, keepdim=True)
        return h

    def _lap(self) -> float:
        """Seconds since the previous lap (synced, so GPU work is counted)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        dt, self._t = now - self._t, now
        return dt

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        return gen_randn_like(self.gen, x)

    def _rand_tau(self, B: int) -> torch.Tensor:
        return gen_rand_tau(self.gen, B, self.device, self.dtype)

    def step(
        self,
        ctx,
        cfg,
        *,
        step: int,
        eps: torch.Tensor,
        v_target: torch.Tensor,
        x_pred: torch.Tensor,
        dmd,
        crossattn_emb: torch.Tensor,
        c_null: torch.Tensor,
        grad_step_idx: int,
        grad_step_sigma: float,
    ) -> dict:
        turbo = ctx.turbo
        B = x_pred.shape[0]
        self._lap()
        x_s = x_pred.detach().to(self.dtype)
        x_t = finish_anchor_rollout(ctx, cfg, eps, v_target, crossattn_emb, c_null, B)

        t_teacher = self._lap()

        # --- disc update: teacher (real) vs student (fake), one shared (τ, ε) ---
        # The two branches backward one at a time under block checkpointing (a
        # batched pair OOM'd on the larger buckets at 16 GB). The input must
        # require grad: the unsloth checkpoint drops the LoRA param grads when
        # every input is detached (see cdm_off_trajectory_loss). Each update
        # accumulates over the replay window (newest pair first, each at its own
        # (τ, ε), losses scaled 1/n), then takes one optimizer step. The row's
        # bce / margin / acc are the newest pair's on the first update, scored
        # before the disc has trained on it; acc_window covers every pair in that
        # update (older ones already trained on).
        #
        # Approximate R1 (APT, as gan.r1_weight): w · MSE(h(x_t), h(x_t + αδ)) on
        # the teacher branch. The backbone trains here, so both logits carry grad;
        # the branches can't share a graph, so the MSE gradient is split exactly:
        # the clean branch takes w·MSE(h, h_a.detach()), the perturbed branch
        # w·MSE(h_a, h.detach()) — h_a's value comes from one no-grad forward.
        self.pairs.appendleft((x_t, x_s, crossattn_emb.detach()))
        n_pairs = len(self.pairs)
        clip = cfg.dmad_probe_grad_clip if cfg.dmad_probe_grad_clip > 0 else None
        r1_w = cfg.dmad_probe_r1_weight
        r1 = None
        for k in range(cfg.dmad_probe_disc_steps):
            accs = []
            for j, (xt_j, xs_j, c_j) in enumerate(self.pairs):
                tau_d = self._rand_tau(B)
                eps_d = self._randn_like(xs_j)
                with self._disc_view(turbo), selective_block_grad_ckpt(ctx.model):
                    x_rt = renoise(xt_j, tau_d, eps_d)
                    if r1_w > 0:
                        x_ra = x_rt + cfg.dmad_probe_r1_alpha * self._randn_like(x_rt)
                        h_a0 = self._h(ctx, x_ra, tau_d, c_j, no_grad=True)
                    h = self._h(ctx, x_rt.requires_grad_(), tau_d, c_j, no_grad=False)
                    loss_t = F.softplus(-h).mean()
                    if r1_w > 0:
                        loss_t = loss_t + r1_w * F.mse_loss(h, h_a0.detach())
                    (loss_t / n_pairs).backward()
                    h_tk = h.detach()
                    if r1_w > 0:
                        h_a = self._h(
                            ctx, x_ra.requires_grad_(), tau_d, c_j, no_grad=False
                        )
                        (r1_w * F.mse_loss(h_a, h_tk) / n_pairs).backward()
                        r1_k = float(F.mse_loss(h_a.detach(), h_tk))
                    x_rs = renoise(xs_j, tau_d, eps_d).requires_grad_()
                    h = self._h(ctx, x_rs, tau_d, c_j, no_grad=False)
                    (F.softplus(h).mean() / n_pairs).backward()
                    h_parts = [h_tk, h.detach()]
                accs.append(_acc(*h_parts))
                if k == 0 and j == 0:
                    h_t, h_s = h_parts
                    tau_d0 = tau_d
                    if r1_w > 0:
                        r1 = r1_k
                if j == 0:
                    h_t_last, h_s_last = h_parts
            gn = torch.nn.utils.clip_grad_norm_(
                self.stack_params + list(self.head.parameters()),
                max_norm=clip if clip is not None else float("inf"),
            )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if k == 0:
                disc_grad_norm = gn
                acc_window = sum(accs) / len(accs)
        loss = gan_loss_discriminator(h_t, h_s)
        t_disc = self._lap()

        # --- probe: generator gradient of −h_T at the DMD's (τ_dm, ε_dm) ---
        for p in self.stack_params:
            p.requires_grad_(False)
        self.head.requires_grad_(False)
        try:
            x_in = x_s.detach().requires_grad_()
            with self._disc_view(turbo), selective_block_grad_ckpt(ctx.model):
                h_probe = self._h(
                    ctx,
                    renoise(x_in, dmd.tau_dm, dmd.eps_dm),
                    dmd.tau_dm,
                    crossattn_emb,
                    no_grad=False,
                )
                (g_t,) = torch.autograd.grad(-h_probe.mean(), x_in)
        finally:
            for p in self.stack_params:
                p.requires_grad_(True)
            self.head.requires_grad_(True)

        stats = alignment_stats(g_t, dmd.grad_signal, self.gen)
        t_probe = self._lap()
        row: dict = {
            "step": step,
            "tau_dm": float(dmd.tau_dm[0]),
            "tau_d": float(tau_d0[0]),
            "grad_step": grad_step_idx,
            "grad_step_sigma": grad_step_sigma,
            "bce": float(loss),
            "disc_grad_norm": float(disc_grad_norm),
            "margin": float(h_t.detach().mean() - h_s.detach().mean()),
            "acc": _acc(h_t, h_s),
            "acc_last": _acc(h_t_last, h_s_last),
            "acc_window": acc_window,
            "n_pairs": n_pairs,
            "r1": r1,
            "h_probe": float(h_probe.detach().mean()),
            "g_t_rms": float(g_t.float().pow(2).mean().sqrt()),
            "dm_rms": float(dmd.grad_signal.float().pow(2).mean().sqrt()),
            "peak_mem_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
            **stats,
            **pair_stats(x_t, x_s),
        }

        if cfg.dmad_probe_ceiling:
            # Second DM estimate at an independent (τ', ε'): how far two DM draws
            # agree with each other is the scale g_T's agreement is read against.
            tau2 = self._rand_tau(B)
            eps2 = self._randn_like(x_s)
            x_r2 = renoise(x_s, tau2, eps2)
            v_real2 = ctx.teacher_cfg_velocity(x_r2, tau2, crossattn_emb, c_null)
            v_fake2 = ctx.forward(
                "fake", x_r2, tau2, crossattn_emb, no_grad=True
            ).squeeze(2)
            tau2_e = tau2.view(B, 1, 1, 1).float()
            dm2 = tau2_e * (v_real2 - v_fake2.float())
            if cfg.dm_x0_norm:
                dm2 = dm2 / (
                    (tau2_e * v_real2).abs().mean(dim=(1, 2, 3), keepdim=True)
                ).clamp_min(cfg.norm_floor)
            ceil = alignment_stats(dm2, dmd.grad_signal, self.gen)
            row["ceil_cos"] = ceil["cos"]
            row["ceil_agree"] = ceil["agree"]
            # g_T against the second draw too: the same-point advantage of
            # comparing at (τ_dm, ε_dm) is visible as cos − cos_dm2.
            row["cos_dm2"] = alignment_stats(g_t, dm2, self.gen)["cos"]
            row["tau_2"] = float(tau2[0])
            row["t_ceil"] = self._lap()
        row.update(t_teacher=t_teacher, t_disc=t_disc, t_probe=t_probe)

        n = cfg.dmad_probe_stop_on_collapse
        self._flat_run = self._flat_run + 1 if abs(row["margin"]) < 1e-2 else 0
        if n > 0 and self._flat_run >= n and not self.collapsed:
            self.collapsed = True
            logger.warning(
                f"DMAD probe: |margin| < 1e-2 for {n} consecutive steps at step "
                f"{step} — disc collapsed; stopping the run (stop_on_collapse)."
            )

        if self.out_path is not None:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return row
