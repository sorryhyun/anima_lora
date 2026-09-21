"""Per-step network conditioning hooks.

Drives the timestep / σ / FEI routers that live on the LoRA-family
networks. Most calls are no-ops unless the active network exposes the
corresponding ``set_*`` / ``step_*`` method.

Same hookpoint order is preserved so cudagraph capture sees a stable
sequence: timestep_mask → sigma → fei → balance.
"""

from __future__ import annotations

from typing import Any

import torch


def apply_router_conditioning(
    *,
    network: Any,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    is_train: bool,
    warmup_step: int,
    max_train_steps: int,
    gradient_accumulation_steps: int = 1,
) -> int:
    """Run all per-step router conditioning. Returns the next warmup_step value."""
    if hasattr(network, "set_timestep_mask"):
        network.set_timestep_mask(timesteps, max_timestep=1.0)
    # σ-conditional HydraLoRA router (timestep-hydra.md). No-op
    # unless use_sigma_router is on and the variant is hydra.
    if hasattr(network, "set_sigma"):
        network.set_sigma(timesteps)
    # FEI router input — set_fei() drives both the per-Linear FEI router
    # and the network-level GlobalRouter. FEI is a function of the actual input the model
    # sees this step (``noisy_model_input``), not a leak from the target.
    # No-op when the active network has no FEI router.
    if getattr(network, "use_fei_router", False):
        from library.runtime.fei import compute_fei_2band, fei_sigma_low

        z = noisy_model_input
        if z.dim() == 5:
            z = z.squeeze(2)
        h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
        div = float(getattr(network.cfg, "fei_sigma_low_div", 8.0))
        fei = compute_fei_2band(z, fei_sigma_low(h_lat, w_lat, div))
        network.set_fei(fei)

    incremented = False
    if is_train and hasattr(network, "step_balance_loss_warmup"):
        network.step_balance_loss_warmup(warmup_step, max_train_steps)
        incremented = True
    # Soft-tokens contrastive warmup: parallel branch (mutually exclusive with
    # the LoRA-family network above — different network class) but kept as a
    # separate ``if`` so the order of checks doesn't matter.
    if is_train and hasattr(network, "step_contrastive_warmup"):
        # ``warmup_step`` is the micro-batch counter; soft_tokens converts it to
        # the optimizer-step clock with ``accum`` for its cadence gate.
        network.step_contrastive_warmup(
            warmup_step, max_train_steps, gradient_accumulation_steps
        )
        incremented = True
    return warmup_step + 1 if incremented else warmup_step
