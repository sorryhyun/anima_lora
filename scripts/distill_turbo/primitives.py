"""Re-noising, τ samplers, and shared loop-side helpers.

These were the first copy of the distillation per-step primitives; they have
since been promoted to ``library/training`` and ``library/datasets`` so the
other distillation loops (``project/finished/mod_guidance``)
share one implementation. This module is now a thin compatibility shim — the
turbo loop imports the same names from here as before.
"""

from __future__ import annotations

import torch

from library.datasets.cache import make_cached_collate as make_collate
from library.training.forward import PadCache, renoise
from library.training.forward import sample_sigma as sample_t
from library.training.schedulers import make_warmup_cosine_scheduler

__all__ = [
    "renoise",
    "sample_t",
    "sample_t_routed",
    "gan_effective_weight",
    "make_scheduler",
    "PadCache",
    "make_collate",
    "sample_dynamic_sigmas",
    "cdm_extrapolate",
]


def sample_t_routed(
    B: int,
    *,
    turbo,
    fake_tau_banks: int,
    fake_tau_boundary: float,
    distribution: str,
    sigmoid_scale: float,
    device,
    dtype,
) -> torch.Tensor:
    """Draw τ for a fake update/query and arm the owner fake bank (τ-split critic).

    ``fake_tau_banks <= 1`` → a plain on-device :func:`sample_t`, byte-identical
    to the pre-bank loop (same RNG stream, no extra work). With 2 banks the draw
    happens on CPU (B=1, enforced at config resolve) so the routing comparison
    costs no GPU sync; the owner bank is selected via ``turbo.set_fake_bank``
    (bank 0 owns τ < boundary) and τ is then moved to ``device``. Callers must
    invoke this BEFORE the fake forward the τ conditions.
    """
    if fake_tau_banks <= 1:
        return sample_t(
            B,
            distribution=distribution,
            sigmoid_scale=sigmoid_scale,
            device=device,
            dtype=dtype,
        )
    tau_cpu = sample_t(
        B,
        distribution=distribution,
        sigmoid_scale=sigmoid_scale,
        device="cpu",
        dtype=torch.float32,
    )
    turbo.set_fake_bank(0 if float(tau_cpu) < fake_tau_boundary else 1)
    return tau_cpu.to(device=device, dtype=dtype)


def sample_dynamic_sigmas(n_min: int, n_max: int) -> list[float]:
    """One CDM-style dynamic rollout grid (arXiv:2605.06376 §3.2).

    Length N ~ U{n_min..n_max}; interior anchors drawn uniform in (0,1) and
    sorted descending, endpoints pinned to t₁=1 and 0 — the same (N+1)-point
    1 → 0 layout as the static grid, so ``sigmas[i] - sigmas[i+1]`` stays the
    Euler dt for step i. t₁=1 is what lets the DP-DMD diversity anchor compose
    unchanged (v_target only assumes the first forward sits at t=1). CPU RNG
    (seeded by ``torch.manual_seed``), no GPU sync.
    """
    n = int(torch.randint(n_min, n_max + 1, (1,)).item())
    if n == 1:
        return [1.0, 0.0]
    interior = torch.sort(torch.rand(n - 1), descending=True).values.tolist()
    return [1.0, *interior, 0.0]


def cdm_extrapolate(
    x: torch.Tensor, v: torch.Tensor, s_from: float, t_to: torch.Tensor
) -> torch.Tensor:
    """Velocity-driven Euler extrapolation to an off-trajectory point (CDM Eq. 7).

    ``x(t') = x + (t' − s)·v`` — the rollout's own Euler form, with an arbitrary
    (large, possibly noiseward) stride to ``t' ~ U(0,1)`` instead of the next
    grid point. Inputs are DETACHED and the result is a plain fp32 tensor with
    no graph: the extrapolation is a launch point for one fresh grad-bearing
    student forward, not a second BPTT chain (the deliberate deviation
    documented in docs/proposal/cdm.md Phase 1 — mirrors the DM renoise path).
    ``t_to`` is per-sample ``(B,)``, broadcast against ``x``'s trailing dims.
    """
    stride = t_to.detach().float().view(-1, *([1] * (x.dim() - 1))) - s_from
    return x.detach().float() + stride * v.detach().float()


def gan_effective_weight(cfg, step: int) -> float:
    """Generator-side GAN weight at ``step``: delay window, then linear ramp.

    0 for ``step < gan_delay_steps`` (the escape window — DM+div re-diversify a
    collapsed warm start before dense realism pressure lands; an unramped token
    head froze pose at the init's mode), then a linear 0 → ``gan_loss_weight_gen``
    ramp over ``gan_warmup_steps`` (instant-on when 0). The disc trains from
    step 0 regardless — only the student-side λ ramps. Both knobs 0 → constant
    ``gan_loss_weight_gen`` (byte-identical shipped loop).
    """
    if cfg.gan_loss_weight_gen == 0.0 or step < cfg.gan_delay_steps:
        return 0.0
    if cfg.gan_warmup_steps > 0:
        ramp = min(1.0, (step - cfg.gan_delay_steps + 1) / cfg.gan_warmup_steps)
        return cfg.gan_loss_weight_gen * ramp
    return cfg.gan_loss_weight_gen


def make_scheduler(opt, total_steps: int, lr: float):
    """Warmup (2% of ``total_steps``, ≥1 step) → cosine annealing to ``0.1·lr``.

    Cosine is the only shape: the ``lr_schedule="constant"`` variant was tried
    (superturbo_B2) and closed — it never settles (full-magnitude updates to the
    last step, ~5-10x the annealed per-step displacement) and rendered worse
    than the cosine twin. The tail is settling, not dead time.
    """
    warmup_steps = max(1, int(0.02 * total_steps))
    return make_warmup_cosine_scheduler(
        opt,
        total_steps,
        lr,
        warmup_steps=warmup_steps,
        eta_min_ratio=0.1,
    )
