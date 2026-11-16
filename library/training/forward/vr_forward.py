"""Variance-reduced FM control-variate reference forward (AsymFlow §5.2).

For LoRA-family runs the base DiT is frozen and adapters are additive, so
``network.set_multiplier(0)`` produces the base-model prediction without
holding a second 2B DiT in VRAM. This runs one extra no-grad forward
through the same trainable DiT (adapter zeroed) on the FEI-low-passed
pair ``(x_t^L, t)`` sharing the same noise ``ε`` and returns the
per-element residual ``z = ref_pred_L - (ε − x_0^L)``.

The residual is consumed by the ``_flow_matching_vr_loss`` handler in
``library/training/losses.py``, which blends ``(y + λ·z)²`` in place of
``y²``. ``crossattn_emb`` is reused verbatim (postfix-appended or not).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


def run_vr_reference_forward(
    *,
    anima_call: Any,
    network: Any,
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    padding_mask: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
    weight_dtype: torch.dtype,
    fei_sigma_low_div: float,
) -> torch.Tensor:
    """Return the per-element control-variate residual `z` (4D, same shape as latents)."""
    from library.runtime.fei import fei_sigma_low, gaussian_blur_2d

    h_lat = int(latents.shape[-2])
    w_lat = int(latents.shape[-1])
    sigma_low = fei_sigma_low(h_lat, w_lat, fei_sigma_low_div)
    # gaussian_blur_2d promotes to fp32 internally; cast back to weight_dtype
    # so the DiT sees its native precision.
    x0_L = gaussian_blur_2d(latents.float(), sigma_low).to(latents.dtype)
    # Same noise `ε`; sigmas already broadcast to (B,1,1,1).
    x_t_L = (1.0 - sigmas) * x0_L + sigmas * noise
    x_t_L_5d = x_t_L.unsqueeze(2).to(weight_dtype)

    orig_mult = float(getattr(network, "multiplier", 1.0))
    network.set_multiplier(0.0)
    try:
        with torch.no_grad():
            ref_pred = anima_call(
                x_t_L_5d,
                timesteps,
                crossattn_emb,
                padding_mask=padding_mask,
                **forward_kwargs,
            )
    finally:
        network.set_multiplier(orig_mult)

    return ref_pred.squeeze(2) - (noise - x0_L)
