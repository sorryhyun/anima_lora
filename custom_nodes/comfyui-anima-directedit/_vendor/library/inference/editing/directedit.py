"""DirectEdit (Yang & Ye, arXiv:2605.02417v1) — flow-based image editing primitive.

Two-pass training-free editor for flow-matching DiTs:

1. **Inversion** (clean -> noise): step backward through the Euler ODE and
   record per-step residuals ``ΔZ_i = Z_inv[i+1] − Z_inv[i]``, the anchor
   that reconstructs the source instead of rectifying the inversion path.

2. **Editing** (noise -> clean): the generation loop with every model call
   queried at ``Z[i] + ΔZ[i]`` and the cross-attn prompt set to ψ_tar; ΔZ pins
   the trajectory to the source. With ``t_inj > 0`` the src stream's self-attn
   V is injected into the tar stream for the first ``t_inj`` steps (paper
   Eq. 13). Of Eq. 12 mask blending only the anchor-side half is implemented.

Conventions: sigmas[0] = 1 (noise), sigmas[T] = 0 (clean); latents are 5D
``[B, C, 1, H/8, W/8]``; ``embed`` is preprocessed crossattn (512-padded).
``invert`` / ``edit_forward`` take a loaded ``Anima`` and pre-encoded embeds
(driven by ``scripts/edit.py``).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Iterable, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

from library.anima import models as anima_models
from library.inference.corrections.smc_cfg import SMCCFGState

logger = logging.getLogger(__name__)


def _padding_mask_for(latents: torch.Tensor) -> torch.Tensor:
    """Anima expects a (B, 1, H_lat, W_lat) zero mask for non-padded inputs."""
    bs = latents.shape[0]
    h_lat = latents.shape[-2]
    w_lat = latents.shape[-1]
    return torch.zeros(bs, 1, h_lat, w_lat, dtype=torch.bfloat16, device=latents.device)


@torch.no_grad()
def _v_pred(
    anima: anima_models.Anima,
    latents: torch.Tensor,
    sigma: torch.Tensor,
    embed: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    guidance_scale: float,
    padding_mask: torch.Tensor,
    smc_cfg_state: Optional[SMCCFGState] = None,
) -> torch.Tensor:
    """One model forward (with optional CFG). Returns velocity prediction.

    ``smc_cfg_state`` routes the combine through SMC-CFG; reuse one state
    across steps for e_prev continuity. ``invert()`` does not pass it.
    """
    t_expand = sigma.expand(latents.shape[0]).to(latents.device, dtype=torch.bfloat16)
    noise_pred = anima(latents, t_expand, embed, padding_mask=padding_mask)
    if guidance_scale != 1.0 and embed_neg is not None:
        uncond = anima(latents, t_expand, embed_neg, padding_mask=padding_mask)
        if smc_cfg_state is not None:
            noise_pred = smc_cfg_state.combine(noise_pred, uncond, guidance_scale)
        else:
            noise_pred = uncond + guidance_scale * (noise_pred - uncond)
    return noise_pred


class _VInjectionState:
    """Row-indexed self-attn V swap inside a single batched forward.

    As in the author's reference, all branches share one transformer call. The
    caller stacks ``[neg_tar, cond_src, cond_tar]`` (CFG > 1) or
    ``[cond_src, cond_tar]`` (CFG = 1) and sets ``src_row`` / ``tar_row``; the
    hook does ``v[tar_row] = v[src_row]`` on the configured blocks. Both
    ``None`` = pass-through.
    """

    def __init__(self, block_indices: Set[int]) -> None:
        self.block_indices = block_indices
        self.src_row: Optional[int] = None
        self.tar_row: Optional[int] = None

    def hook(self, block_idx: int, v: torch.Tensor) -> torch.Tensor:
        if (
            self.src_row is None
            or self.tar_row is None
            or block_idx not in self.block_indices
        ):
            return v
        # out-of-place clone: in-place writes here can alias q/k tensors that
        # some backends share storage with via the qkv_proj split
        v = v.clone()
        v[self.tar_row] = v[self.src_row]
        return v

    def set_rows(self, src_row: Optional[int], tar_row: Optional[int]) -> None:
        self.src_row = src_row
        self.tar_row = tar_row


def _make_patched_self_attn_forward(attn, block_idx: int, state: _VInjectionState):
    """Build a replacement ``Attention.forward`` that routes V through ``state.hook``.

    ``library/anima/models.py::Attention`` and ComfyUI's
    ``comfy/ldm/cosmos/predict2.py::Attention`` have different forward
    signatures; ``compute_attention`` (comfy-only) picks the matching one, as
    the wrong one raises ``TypeError`` on the first edit step.
    """
    compute_qkv = attn.compute_qkv

    if hasattr(attn, "compute_attention"):
        compute_attention = attn.compute_attention

        def patched_comfy(
            x, context=None, rope_emb=None, transformer_options=None, **_kwargs
        ):
            q, k, v = compute_qkv(x, context, rope_emb=rope_emb)
            v = state.hook(block_idx, v)
            return compute_attention(
                q, k, v, transformer_options=transformer_options or {}
            )

        return patched_comfy

    output_proj = attn.output_proj
    output_dropout = attn.output_dropout
    dispatcher = anima_models.attention_dispatch.dispatch_attention

    def patched_library(x, attn_params, context, rope_cos_sin=None):
        q, k, v = compute_qkv(x, context, rope_cos_sin=rope_cos_sin)
        v = state.hook(block_idx, v)
        if q.dtype != v.dtype:
            if (
                not attn_params.supports_fp32 or attn_params.requires_same_dtype
            ) and torch.is_autocast_enabled():
                target_dtype = v.dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        qkv = [q, k, v]
        del q, k, v
        result = dispatcher(qkv, attn_params=attn_params)
        return output_dropout(output_proj(result))

    return patched_library


@contextmanager
def _v_injection_scope(anima: anima_models.Anima, block_indices: Set[int]):
    """Monkey-patch ``self_attn.forward`` on selected blocks for the scope.

    Yields the shared ``_VInjectionState``; the caller sets rows per forward.
    """
    state = _VInjectionState(block_indices)
    # restore `del`s the instance attribute unless one existed before
    patched: list[tuple[torch.nn.Module, bool, object]] = []
    for idx, block in enumerate(anima.blocks):
        if idx not in block_indices:
            continue
        attn = block.self_attn
        had_instance_forward = "forward" in attn.__dict__
        prior = attn.__dict__.get("forward")
        patched.append((attn, had_instance_forward, prior))
        attn.forward = _make_patched_self_attn_forward(attn, idx, state)
    try:
        yield state
    finally:
        for attn, had_instance_forward, prior in patched:
            if had_instance_forward:
                attn.forward = prior
            else:
                del attn.forward  # falls back to the class method
        state.set_rows(None, None)


def _resolve_t_inj_blocks(
    anima: anima_models.Anima,
    t_inj_blocks: Optional[Iterable[int]],
) -> Set[int]:
    """Default to "all blocks except the final one" (SD3.5-style) when None."""
    n = len(anima.blocks)
    if t_inj_blocks is None:
        return set(range(n - 1))
    out = {int(i) for i in t_inj_blocks}
    if not out:
        return out
    if min(out) < 0 or max(out) >= n:
        raise ValueError(
            f"t_inj_blocks {sorted(out)} out of range for model with {n} blocks "
            f"(valid: 0..{n - 1})."
        )
    return out


@torch.no_grad()
def invert(
    anima: anima_models.Anima,
    z_clean: torch.Tensor,
    embed_src: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    sigmas: torch.Tensor,
    guidance_scale: float = 1.0,
    step_callback: Optional[Callable[[int, int], None]] = None,
    traj_recorder=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Invert ``z_clean`` (= VAE-encoded source image) along the Anima ODE.

    Returns ``(z_inv, delta_z)``:
      * ``z_inv``: length T+1; ``z_inv[T] == z_clean``, ``z_inv[0]`` is the
        maximally-noised inversion.
      * ``delta_z``: length T, ``delta_z[i] = z_inv[i+1] − z_inv[i]`` — the
        anchor residuals ``edit_forward`` consumes.

    Step (paper §3.2 in our index), for ``i = T-1 .. 0``:
        ``z_inv[i] = z_inv[i+1] + (sigmas[i] − sigmas[i+1]) · v_θ(z_inv[i+1], σ=sigmas[i])``
    v_θ is queried at the destination σ (``sigmas[i]``): the first input is
    clean, and σ=0 is outside the trained range and the singular point of the
    sinusoidal ``t_embedder``. Matches the author's reference.

    Default ``guidance_scale=1.0`` skips CFG (no negative concept to push from).
    ``traj_recorder`` is a passive
    :class:`~library.inference.traj_stats.TrajStatsRecorder`, fed the 0-d σ
    tensor (``float()`` would sync the stream).
    """
    device = z_clean.device
    T = sigmas.shape[0] - 1
    padding_mask = _padding_mask_for(z_clean)

    z_inv: List[torch.Tensor] = [None] * (T + 1)  # type: ignore[list-item]
    z_inv[T] = z_clean.to(torch.bfloat16)

    delta_z: List[torch.Tensor] = [None] * T  # type: ignore[list-item]

    iterator = tqdm(range(T - 1, -1, -1), desc="DirectEdit inversion", total=T)
    for step_idx, i in enumerate(iterator, start=1):
        # destination σ (docstring)
        sigma_in = sigmas[i].to(device)
        v = _v_pred(
            anima,
            z_inv[i + 1],
            sigma_in,
            embed_src,
            embed_neg,
            guidance_scale,
            padding_mask,
        )
        # (sigmas[i] - sigmas[i+1]) > 0 in our index
        coeff = (sigmas[i] - sigmas[i + 1]).to(device, dtype=torch.float32)
        z_inv[i] = (z_inv[i + 1].float() + coeff * v.float()).to(torch.bfloat16)
        delta_z[i] = (z_inv[i + 1].float() - z_inv[i].float()).to(torch.bfloat16)
        if traj_recorder is not None:
            traj_recorder.record(step_idx - 1, sigma_in, z_inv[i], v)
        if step_callback is not None:
            step_callback(step_idx, T)

    return z_inv, delta_z


@torch.no_grad()
def edit_forward(
    anima: anima_models.Anima,
    z_init: torch.Tensor,
    delta_z: List[torch.Tensor],
    embed_tar: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    sigmas: torch.Tensor,
    guidance_scale: float = 4.0,
    embed_src: Optional[torch.Tensor] = None,
    t_inj: int = 0,
    t_inj_blocks: Optional[Iterable[int]] = None,
    z_inv: Optional[List[torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,  # Eq. 12 anchor mask (1 = edit region)
    anchor_scale: float = 1.0,
    step_callback: Optional[Callable[[int, int], None]] = None,
    smc_cfg_state: Optional[SMCCFGState] = None,
) -> torch.Tensor:
    """Forward (noise -> clean) edit pass anchored to the inversion residuals.

    Step rule (paper §3.2 in our index):
        ``ẑ_i = z[i] + delta_z[i]                        # anchor``
        ``v_i = v_θ(ẑ_i, σ=sigmas[i], ψ_tar)             # query at anchored pt``
        ``z[i+1] = z[i] − (sigmas[i] − sigmas[i+1]) · v_i # standard Euler step``

    For ``t_inj > 0`` (paper Eq. 13) the first ``t_inj`` steps stack src and
    tar into one batched forward and swap V by row index (see
    :class:`_VInjectionState`). Two separate forwards with a cached V would
    leak src V into the uncond branch.

    Args:
      z_init: should be ``z_inv[0]`` from ``invert(...)`` for the residual
        trick to fire correctly.
      embed_src: required when ``t_inj > 0`` (drives the cond_src row).
      t_inj: number of early steps to run the batched src+tar V-swap forward.
        ``0`` skips the src branch entirely (pure ΔZ-anchored edit).
      t_inj_blocks: block indices to swap V at; ``None`` = all but the last.
      z_inv: the inverted trajectory (length ``T+1``). Required when
        ``t_inj > 0``: there is no Euler-evolved src branch, so src is rebased
        to ``z_inv[i]`` at every injection step.
      mask: Eq. 12 anchor-side half, a latent mask (1 = edit region) that
        drops the Δz anchor there; the anchor pulls toward the source and
        would otherwise suppress the edit.
      anchor_scale: global λ on Δz (1.0 = full anchor, 0.0 = unanchored from
        the inverted init); the dial for whole-image edits, which a full
        anchor suppresses. Composes with ``mask``.

    The src row always runs at CFG=1 (as in paper Algorithm 1): a CFG-mixed V
    would conflate ψ_src with the negative concept.
    """
    device = z_init.device
    T = sigmas.shape[0] - 1
    if len(delta_z) != T:
        raise ValueError(
            f"delta_z has length {len(delta_z)} but sigmas implies T={T} steps "
            "- inversion and editing must use the same sigma schedule."
        )
    if t_inj < 0:
        raise ValueError(f"t_inj must be >= 0, got {t_inj}.")
    if t_inj > T:
        logger.warning(
            "t_inj=%d clamped to T=%d (full-trajectory injection).", t_inj, T
        )
        t_inj = T
    if t_inj > 0 and embed_src is None:
        raise ValueError("V-injection (t_inj > 0) requires embed_src (ψ_src).")
    if t_inj > 0 and z_inv is None:
        raise ValueError(
            "V-injection (t_inj > 0) requires z_inv (full inverted trajectory). "
            "The batched-forward shape doesn't carry a parallel Euler-evolved "
            "src branch — src must be GT-rebased to z_inv[i] each step."
        )
    if z_inv is not None and len(z_inv) != T + 1:
        raise ValueError(
            f"z_inv has length {len(z_inv)} but sigmas implies T+1={T + 1} states "
            "- inversion and editing must use the same sigma schedule."
        )
    if anchor_scale < 0.0:
        raise ValueError(f"anchor_scale must be >= 0, got {anchor_scale}.")
    if anchor_scale != 1.0:
        logger.info(
            "DirectEdit anchor scale: Δz × %.3f (global; 0 = unanchored "
            "generation from the inverted init).",
            anchor_scale,
        )
    anchor_keep: Optional[torch.Tensor] = None
    if mask is not None:
        anchor_keep = (1.0 - mask.float()).to(device)
        logger.info(
            "DirectEdit Eq. 12 anchor mask: Δz dropped on %.1f%% of latent "
            "cells (edit region), kept elsewhere.",
            100.0 * mask.float().mean().item(),
        )

    block_indices = _resolve_t_inj_blocks(anima, t_inj_blocks) if t_inj > 0 else set()
    has_cfg = guidance_scale != 1.0 and embed_neg is not None
    if smc_cfg_state is not None and not has_cfg:
        logger.warning(
            "smc_cfg_state passed but CFG is disabled (guidance_scale=%.2f, "
            "embed_neg=%s) — SMC operates on the cond/uncond residual and "
            "has nothing to clamp; ignored.",
            guidance_scale,
            "set" if embed_neg is not None else "None",
        )
    z_tar = z_init.to(torch.bfloat16)

    if t_inj > 0:
        logger.info(
            "DirectEdit V-injection: t_inj=%d / T=%d, injecting at %d / %d blocks "
            "(batched forward, %d rows)",
            t_inj,
            T,
            len(block_indices),
            len(anima.blocks),
            3 if has_cfg else 2,
        )

    with _v_injection_scope(anima, block_indices) as state:
        iterator = tqdm(range(T), desc="DirectEdit editing", total=T)
        for i in iterator:
            d = delta_z[i].to(device).float()
            if anchor_scale != 1.0:
                d = d * anchor_scale
            if anchor_keep is not None:
                d = d * anchor_keep
            z_hat_tar = (z_tar.float() + d).to(torch.bfloat16)
            sigma_in = sigmas[i].to(device)
            coeff = (sigmas[i] - sigmas[i + 1]).to(device, dtype=torch.float32)

            if i < t_inj:
                # rebase src to the inverted trajectory so drift doesn't
                # corrupt the captured V
                z_src_i = z_inv[i].to(device=device, dtype=torch.bfloat16)
                z_hat_src = (z_src_i.float() + d).to(torch.bfloat16)

                if has_cfg:
                    # [neg_tar, cond_src, cond_tar]; hook sets v[2] = v[1]
                    latents = torch.cat([z_hat_tar, z_hat_src, z_hat_tar], dim=0)
                    embeds = torch.cat([embed_neg, embed_src, embed_tar], dim=0)
                    state.set_rows(src_row=1, tar_row=2)
                else:
                    # [cond_src, cond_tar]
                    latents = torch.cat([z_hat_src, z_hat_tar], dim=0)
                    embeds = torch.cat([embed_src, embed_tar], dim=0)
                    state.set_rows(src_row=0, tar_row=1)

                pad = _padding_mask_for(latents)
                t_expand = sigma_in.expand(latents.shape[0]).to(
                    latents.device, dtype=torch.bfloat16
                )
                noise_pred = anima(latents, t_expand, embeds, padding_mask=pad)
                state.set_rows(None, None)

                if has_cfg:
                    v_neg = noise_pred[0:1]
                    v_cond_tar = noise_pred[2:3]
                    if smc_cfg_state is not None:
                        v_tar = smc_cfg_state.combine(v_cond_tar, v_neg, guidance_scale)
                    else:
                        v_tar = v_neg + guidance_scale * (v_cond_tar - v_neg)
                else:
                    v_tar = noise_pred[1:2]
            else:
                # past t_inj: tar only; rows are None so the hook passes through
                pad = _padding_mask_for(z_hat_tar)
                v_tar = _v_pred(
                    anima,
                    z_hat_tar,
                    sigma_in,
                    embed_tar,
                    embed_neg,
                    guidance_scale,
                    pad,
                    smc_cfg_state=smc_cfg_state,
                )

            z_tar = (z_tar.float() - coeff * v_tar.float()).to(torch.bfloat16)
            if step_callback is not None:
                step_callback(i + 1, T)

    return z_tar
