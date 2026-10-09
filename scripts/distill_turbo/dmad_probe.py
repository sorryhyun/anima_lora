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
2. Disc update: renoise teacher and ``x_pred`` at one (τ, ε), balanced BCE.
3. Probe: ``g_T = ∂(−h_T)/∂x_pred`` at the DMD's own (τ_dm, ε_dm), compared with
   ``grad_signal`` (pre f-distill reweight) — cosine and agree-energy, each with a
   permutation null on the same tensors. With ``ceiling``, a second DM estimate
   at an independent (τ', ε') gives the alignment two DM draws reach with each
   other.

Every random draw here comes from a dedicated generator, so the training RNG
stream — and with it the student/critic numerics — is the same as with the probe
off. One JSONL row per step lands in the run's log dir; read it with
``bench/turbo/dmad_probe_read.py``.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from networks.methods.turbo_dmd import (
    TeacherFeatureDiscriminator,
    gan_loss_discriminator,
    warm_start_plain_lora,
)

from .primitives import renoise
from .steps import selective_block_grad_ckpt

logger = logging.getLogger(__name__)


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
        if cfg.fake_init_weights:
            warm_start_plain_lora(self.stack, cfg.fake_init_weights, "dmad_disc")
        bidx = cfg.dmad_probe_feature_block_idx
        if bidx < 0:
            bidx = model.num_blocks // 2
        if not 0 <= bidx < model.num_blocks:
            raise ValueError(
                f"dmad_probe.feature_block_idx resolved to {bidx}, out of range "
                f"[0, {model.num_blocks})"
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
        self.out_path: Path | None = None
        n = sum(p.numel() for p in self.stack_params) + sum(
            p.numel() for p in self.head.parameters()
        )
        logger.info(
            f"DMAD probe: disc stack + head T on block {bidx} "
            f"({cfg.dmad_probe_head}), {n:,} params, lr={cfg.dmad_probe_lr}"
        )

    def bind_log_dir(self, log_dir: str | Path) -> None:
        self.out_path = Path(log_dir) / "dmad_probe.jsonl"
        logger.info(f"DMAD probe rows → {self.out_path}")

    @contextmanager
    def _disc_view(self, turbo):
        """Teacher view + the disc stack on: the base DiT as the disc backbone."""
        turbo.set_view("teacher")
        self.stack.set_enabled(True)
        try:
            yield
        finally:
            self.stack.set_enabled(False)

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
        return self.head([feats[self.tap]])  # (B, 1) pooled | (B, N) token

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape, generator=self.gen, device=x.device, dtype=x.dtype)

    def _rand_tau(self, B: int) -> torch.Tensor:
        u = torch.rand(B, generator=self.gen, device=self.device)
        return u.to(self.dtype)

    @torch.no_grad()
    def _teacher_sample(self, ctx, cfg, eps, v_target, c, c_null, B):
        """Finish the anchor rollout: z_tk → σ=0 on the teacher's CFG grid."""
        z = (eps.float() - (1.0 - ctx.t_k_anchor) * v_target).to(self.dtype)
        sig = ctx.teacher_anchor_sigmas
        for i in range(cfg.k_anchor, cfg.teacher_anchor_steps):
            t_b = torch.full((B,), sig[i], device=self.device, dtype=self.dtype)
            v = ctx.teacher_cfg_velocity(z, t_b, c, c_null)
            z = (z.float() - (sig[i] - sig[i + 1]) * v).to(self.dtype)
        return z

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
        x_s = x_pred.detach().to(self.dtype)
        x_t = self._teacher_sample(ctx, cfg, eps, v_target, crossattn_emb, c_null, B)

        # --- disc update: teacher (real) vs student (fake), one shared (τ, ε) ---
        # The two branches backward one at a time under block checkpointing (a
        # batched pair OOM'd on the larger buckets at 16 GB). The input must
        # require grad: the unsloth checkpoint drops the LoRA param grads when
        # every input is detached (see cdm_off_trajectory_loss).
        tau_d = self._rand_tau(B)
        eps_d = self._randn_like(x_s)
        h_parts = []
        with self._disc_view(turbo), selective_block_grad_ckpt(ctx.model):
            for x_src, loss_fn in (
                (x_t, lambda h: F.softplus(-h).mean()),
                (x_s, lambda h: F.softplus(h).mean()),
            ):
                x_r = renoise(x_src, tau_d, eps_d).requires_grad_()
                h = self._h(ctx, x_r, tau_d, crossattn_emb, no_grad=False)
                loss_fn(h).backward()
                h_parts.append(h.detach())
        h_t, h_s = h_parts
        loss = gan_loss_discriminator(h_t, h_s)
        disc_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.stack_params + list(self.head.parameters()),
            max_norm=cfg.grad_clip if cfg.grad_clip > 0 else float("inf"),
        )
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

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
        row: dict = {
            "step": step,
            "tau_dm": float(dmd.tau_dm[0]),
            "tau_d": float(tau_d[0]),
            "grad_step": grad_step_idx,
            "grad_step_sigma": grad_step_sigma,
            "bce": float(loss),
            "disc_grad_norm": float(disc_grad_norm),
            "margin": float(h_t.detach().mean() - h_s.detach().mean()),
            "acc": float(
                0.5
                * (
                    (h_t.detach() > 0).float().mean()
                    + (h_s.detach() < 0).float().mean()
                )
            ),
            "h_probe": float(h_probe.detach().mean()),
            "g_t_rms": float(g_t.float().pow(2).mean().sqrt()),
            "dm_rms": float(dmd.grad_signal.float().pow(2).mean().sqrt()),
            "peak_mem_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
            **stats,
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

        if self.out_path is not None:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return row
