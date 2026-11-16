"""Second DiT forward against a sampled stochastic inversion run.

When inversion runs are present in the batch, this runs a no-grad forward
through the DiT using a randomly-sampled inversion run as ``crossattn_emb``
and computes an MSE between the main forward's and the inversion forward's
``cross_attn.output_proj`` captures at the configured blocks. The returned
scalar is consumed by the ``_functional_loss`` handler in
``library/training/losses.py``.

The trainer owns the capture state (forward hooks register on
``cross_attn.output_proj`` modules and populate ``captures``); this helper
only consumes the snapshot, runs the second forward, then reads the
inversion captures back out.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch


def compute_inversion_func_loss(
    *,
    anima_call: Any,
    captures: dict[int, torch.Tensor],
    block_indices: Iterable[int],
    batch: Mapping[str, Any],
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    padding_mask: torch.Tensor,
    has_postfix: bool,
    kw: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Return the inversion-MSE scalar, or None if the batch has nothing to compare against.

    ``captures`` is mutated by the trainer's forward hooks: it holds the
    primary forward's captures on entry, and will hold the inversion
    forward's captures after this call returns. The caller is responsible
    for clearing it before the next step's primary forward.
    """
    block_indices = list(block_indices)
    if not block_indices:
        return None

    inv_runs = batch.get("inversion_runs") if isinstance(batch, dict) else None
    inv_mask = batch.get("inversion_mask") if isinstance(batch, dict) else None
    if inv_runs is None or inv_mask is None:
        return None
    if not bool(inv_mask.any().item()):
        return None

    cap_main = dict(captures)
    missing = [bi for bi in block_indices if bi not in cap_main]
    if missing:
        raise RuntimeError(
            f"Functional loss: main forward did not populate captures for blocks {missing}"
        )

    inv_runs_dev = inv_runs.to(device, dtype=dtype)
    inv_mask_dev = inv_mask.to(device)
    B_inv, N_runs, _, _ = inv_runs_dev.shape
    run_idx = torch.randint(0, N_runs, (B_inv,), device=inv_runs_dev.device)
    sampled_inv = inv_runs_dev[
        torch.arange(B_inv, device=inv_runs_dev.device), run_idx
    ]  # [B, S, D]

    # Same pooled_text_override so AdaLN modulation is identical;
    # only cross-attn K/V differs between the two forwards.
    inv_kw: dict[str, Any] = {}
    if has_postfix and "pooled_text_override" in kw:
        inv_kw["pooled_text_override"] = kw["pooled_text_override"]

    with torch.no_grad():
        _ = anima_call(
            noisy_model_input,
            timesteps,
            sampled_inv,
            padding_mask=padding_mask,
            **inv_kw,
        )

    cap_inv = {bi: captures[bi].detach() for bi in block_indices}

    mask_f = inv_mask_dev.float()
    denom = mask_f.sum().clamp(min=1.0)
    block_losses = []
    for bi in block_indices:
        diff = cap_main[bi].float() - cap_inv[bi].float()
        per_sample = diff.pow(2).mean(dim=tuple(range(1, diff.ndim)))  # [B]
        block_losses.append((per_sample * mask_f).sum() / denom)
    return sum(block_losses) / len(block_losses)
