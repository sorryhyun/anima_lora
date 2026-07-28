"""DirectEdit (Yang & Ye, arXiv:2605.02417v1) — flow-based image editing primitive.

Two-pass training-free editor for flow-matching DiTs:

1. **Inversion** (clean -> noise): step backward through the same Euler ODE
   the generator runs forward, querying v_θ at each step's input. Record
   per-step residuals ``ΔZ_i = Z_inv[i+1] − Z_inv[i]`` — these are the
   "anchor" the paper uses to make reconstruction bit-exact instead of
   trying to rectify the inversion path itself.

2. **Editing** (noise -> clean): standard generation loop, but every model
   call is queried at ``Z[i] + ΔZ[i]`` instead of ``Z[i]``. The cross-attn
   prompt is the edit target ψ_tar; the residual ΔZ pins the trajectory to
   the source. For ``t_inj > 0`` we also run a parallel src stream with
   ψ_src and inject its self-attn V into the tar stream for the first
   ``t_inj`` steps (paper Eq. 13). Mask blending (paper Eq. 12) is still
   v3 — left as a stub here.

Anima conventions used:
  * sigmas[0] = 1 (pure noise), sigmas[T] = 0 (clean), per
    ``library/inference/sampling.py::get_timesteps_sigmas``.
  * Latents: 5D ``[B, C, 1, H/8, W/8]`` (frame dim of 1 — image, not video).
  * The model's call signature matches what ``generate_body`` uses:
    ``anima(latents, t_expand, embed, padding_mask=...)`` where ``embed`` is
    already-preprocessed crossattn (post-T5, 512-padded).

This module is self-contained: ``invert`` and ``edit_forward`` accept the
already-loaded ``Anima`` model and pre-encoded ``ψ_src`` / ``ψ_tar`` embeds,
so the calling script (``scripts/edit.py``) can reuse the existing TE/VAE/DiT
loaders from ``library.inference.{models,text}``.
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

    ``smc_cfg_state`` (optional) routes the cond/uncond combine through
    α-adaptive Sliding-Mode Control. No-op when CFG is disabled (single
    forward, no residual). Caller must reuse the same state across steps for
    e_prev continuity; ``invert()`` deliberately does not pass it.
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

    Author's reference (``DirectEdit/controller/attn_norm_ctrl_sd35.py:362``)
    runs all branches through one transformer call and swaps V in-place by
    row index — ``value[h//2:]`` operates on the cond half, with cond_tar's
    V replaced by cond_src's V before the dispatcher.

    Anima port: caller stacks ``[neg_tar, cond_src, cond_tar]`` (3 rows when
    CFG > 1) or ``[cond_src, cond_tar]`` (2 rows when CFG = 1) and sets
    ``src_row`` / ``tar_row`` before each forward. The hook then does
    ``v[tar_row] = v[src_row]`` on the configured blocks. Both ``None``
    disables the swap (no-op pass-through), used for steps ``i >= t_inj``
    when no injection is active.
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
        # v shape: [B, S, n_heads, head_dim] (post-norm, pre-dispatch).
        # Out-of-place clone — the dispatcher may consume v under autograd-off
        # but in-place writes alias against q/k tensors that some backends
        # share storage with via the qkv_proj split.
        v = v.clone()
        v[self.tar_row] = v[self.src_row]
        return v

    def set_rows(self, src_row: Optional[int], tar_row: Optional[int]) -> None:
        self.src_row = src_row
        self.tar_row = tar_row


def _make_patched_self_attn_forward(attn, block_idx: int, state: _VInjectionState):
    """Build a replacement ``Attention.forward`` that routes V through ``state.hook``.

    Two backends share this code path:
      * ``library/anima/models.py::Attention`` — used by the standalone CLI
        (``scripts/edit.py``). Forward signature is
        ``(x, attn_params, context, rope_cos_sin=None)`` and dispatches via
        ``attention_dispatch.dispatch_attention``.
      * ``comfy/comfy/ldm/cosmos/predict2.py::Attention`` — used inside
        ComfyUI (the bundled DiT impl). Forward signature is
        ``(x, context=None, rope_emb=None, transformer_options={})`` and
        dispatches via ``self.compute_attention``.

    We detect via ``compute_attention`` (comfy-only) and emit a patched
    function whose signature matches the actual call site. Patching with the
    wrong signature raises ``TypeError: ... unexpected keyword argument
    'rope_emb'`` on the first edit step.
    """
    compute_qkv = attn.compute_qkv

    # Comfy cosmos Attention path.
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

    # Library Anima Attention path.
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

    Yields the shared ``_VInjectionState``; outer code toggles
    ``state.mode`` per src/tar pass.
    """
    state = _VInjectionState(block_indices)
    # Track which attns had a pre-existing instance-level `forward` so we can
    # restore by either reassigning or `del`ing — a plain reassign would leave
    # an instance attribute behind (closing a refcycle through the bound
    # method), which is functionally fine but leaks state across scopes.
    patched: list[tuple[torch.nn.Module, bool, object]] = []
    for idx, block in enumerate(anima.blocks):
        if idx not in block_indices:
            continue
        attn = block.self_attn
        had_instance_forward = "forward" in attn.__dict__
        prior = attn.__dict__.get("forward")
        patched.append((attn, had_instance_forward, prior))
        # Assigning to the instance attribute shadows the class method; nn.Module
        # __call__ resolves `self.forward` via normal attribute lookup, so this
        # works without descriptor binding.
        attn.forward = _make_patched_self_attn_forward(attn, idx, state)
    try:
        yield state
    finally:
        for attn, had_instance_forward, prior in patched:
            if had_instance_forward:
                attn.forward = prior
            else:
                # Remove the instance attribute so attribute lookup falls back
                # to the class method — exactly the pre-patch state.
                del attn.forward
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
      * ``z_inv``: list of length T+1, where ``z_inv[T] == z_clean`` (cast/dtype
        match the input) and ``z_inv[0]`` is the maximally-noised inversion.
      * ``delta_z``: list of length T, with ``delta_z[i] = z_inv[i+1] − z_inv[i]``
        — the anchor residuals consumed by ``edit_forward``.

    Inversion convention (paper §3.2 in our index):
        ``z_inv[i] = z_inv[i+1] + (sigmas[i] − sigmas[i+1]) · v_θ(z_inv[i+1], σ=sigmas[i])``
    iterated for ``i = T-1 .. 0``. Note ``sigmas[i]`` (not ``sigmas[i+1]``):
    v_θ is queried at the **destination** σ, not the input σ. This matches
    the author's reference (``flow_direct_correction_inv_sd35.py:184, 200-208``)
    where ``t = timesteps[num_steps - 1 - cur_step]`` corresponds to
    ``sigma_next`` (the noisier endpoint of the current step) by way of
    diffusers' ``timesteps[k] = sigmas[k] · 1000`` pairing. Crucially, at
    the first iter (``i = T-1``) the input is clean (``z_inv[T]``, σ=0),
    and querying at the input σ would feed σ=0 exactly — outside the model's
    trained range and at the singular point of the sinusoidal ``t_embedder``.
    See ``docs/proposal/directedit_gaps.md#gap-3``.

    CFG during inversion is usually a wash (the source has no negative concept
    to push away from). Default ``guidance_scale=1.0`` skips it. Pass >1.0
    only if you want the inverted noise to land where re-generation with the
    same CFG would put it.

    ``traj_recorder`` (optional): a
    :class:`library.inference.traj_stats.TrajStatsRecorder` observing the
    inversion trajectory — the Phase 1 real-image arm of
    ``_archive/proposals/traj_latent_stats.md``. Passive: recorded on ``.float()``
    copies, ``z_inv``/``delta_z`` are bit-identical with it on or off (pinned
    by ``tests/test_traj_stats.py``). Per step it records
    ``(z_inv[i], sigmas[i], v)`` — since ``z_inv[i] = z_inv[i+1] +
    (σ_i − σ_{i+1})·v``, the recorder's ``x̂₀ = z − σ·v`` equals
    ``z_inv[i+1] − σ_{i+1}·v`` too, i.e. the Euler-consistent clean estimate
    regardless of which endpoint you view it from. Steps are recorded in
    *natural inversion order* (σ ascending, clean → noise); the sidecar's
    ``sigmas`` array carries the ordering, analysis reverses as needed. The
    caller flushes. Sigma is passed as the 0-d tensor (never ``float()`` —
    stream-sync trap, see the recorder's docstring).
    """
    device = z_clean.device
    T = sigmas.shape[0] - 1
    padding_mask = _padding_mask_for(z_clean)

    z_inv: List[torch.Tensor] = [None] * (T + 1)  # type: ignore[list-item]
    z_inv[T] = z_clean.to(torch.bfloat16)

    delta_z: List[torch.Tensor] = [None] * T  # type: ignore[list-item]

    iterator = tqdm(range(T - 1, -1, -1), desc="DirectEdit inversion", total=T)
    for step_idx, i in enumerate(iterator, start=1):
        # Query v_θ at the destination σ (sigmas[i], the noisier endpoint),
        # not the input's σ (sigmas[i+1]). Avoids feeding σ=0 at i=T-1.
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
        # z_inv[i] is at higher noise; (sigmas[i] - sigmas[i+1]) > 0 in our index.
        coeff = (sigmas[i] - sigmas[i + 1]).to(device, dtype=torch.float32)
        z_inv[i] = (z_inv[i + 1].float() + coeff * v.float()).to(torch.bfloat16)
        delta_z[i] = (z_inv[i + 1].float() - z_inv[i].float()).to(torch.bfloat16)
        if traj_recorder is not None:
            traj_recorder.record(step_idx - 1, sigma_in, z_inv[i], v)
        if step_callback is not None:
            step_callback(step_idx, T)

    dz_mean = torch.stack([d.float().abs().mean() for d in delta_z]).mean().item()
    logger.info(
        "DirectEdit invert: mean|dz|=%.6f over %d steps (anchor increment "
        "magnitude under the inversion-pass conditioning).",
        dz_mean,
        T,
    )
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
    tar into a single batched forward and swap V by row index inside the
    patched ``self_attn``:

      * CFG > 1 (3 rows): ``[neg_tar, cond_src, cond_tar]`` —
        the hook copies ``v[1] → v[2]`` on the configured blocks. Output is
        split per row; tar's CFG combine is
        ``v_tar = v[0] + scale · (v[2] − v[0])``.
      * CFG = 1 (2 rows): ``[cond_src, cond_tar]`` — hook copies
        ``v[0] → v[1]``; tar = ``v[1]``.

      Single forward + row-indexed swap matches the author's reference
      (``DirectEdit/controller/attn_norm_ctrl_sd35.py:362``,
      ``flow_direct_correction_inv_sd35.py:262-292``). Replaces the v1
      "two separate forwards with mode-toggle cache" path which leaked src V
      into the uncond branch (Gap 2 in ``docs/proposal/directedit_gaps.md``).

    Args:
      z_init: should be ``z_inv[0]`` from ``invert(...)`` for the residual
        trick to fire correctly.
      embed_src: required when ``t_inj > 0`` (drives the cond_src row).
      t_inj: number of early steps to run the batched src+tar forward with
        V-swap. ``0`` skips the src branch entirely (pure ΔZ-anchored edit).
      t_inj_blocks: which block indices to swap V at. Default ``None`` →
        all blocks except the final one (SD3.5-style default; Anima is
        single-stream cross-attn DiT, so this is the conservative analog).
      z_inv: full inverted trajectory from ``invert(...)`` (length ``T+1``,
        ``z_inv[i]`` at σ=``sigmas[i]``). **Required when ``t_inj > 0``** —
        the batched-forward shape no longer carries a parallel
        Euler-evolved src branch, so src is GT-rebased to ``z_inv[i]`` at
        every injection step. Matches author's
        ``prev_sample_src = gt_source_latent``.
      mask: paper Eq. 12, anchor-side half: a latent-space mask (broadcastable
        to ``delta_z[i]``, 1 = edit region) that DROPS the Δz anchor inside
        the edit region — the anchor is a per-step pull back to the source,
        so a global anchor suppresses the edit even when other preservation
        mechanisms (e.g. an EasyControl masked-cond prior) have released the
        region (project/directedit_ec/bench Phase 1a). Outside the region the anchor
        applies unchanged. The full background-lock latent BLEND (locking
        the outside to the inverted trajectory) remains future work.
      anchor_scale: global multiplier λ on the Δz residuals (1.0 = full
        anchor, 0.0 = unanchored generation from the inverted init). The
        continuous composition↔edit dial for instruction-style edits with no
        region mask — a global anchor at λ=1 suppresses whole-image edits
        (project/directedit_ec/bench in-place probe). Composes with ``mask``
        (regional release applies on top of the scaled residual).

    Notes on src CFG: the src row is always run at CFG=1 (no neg_src in the
    batch). Paper Algorithm 1 doesn't apply CFG to the src branch, and
    capturing V from a CFG-mixed branch would conflate ψ_src and the
    negative concept. If a separate ``inv_cfg != 1`` ever becomes necessary,
    extend the batch to 4 rows (``[neg_src, neg_tar, cond_src, cond_tar]``)
    and combine src per-branch — the row-indexed hook stays unchanged.
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
                # GT-rebase src to the inverted trajectory at this σ. The
                # batched forward's cond_src row is what feeds the V-swap,
                # so this drops any drift that would otherwise corrupt the
                # captured V. Matches author's ``prev_sample_src = gt_source_latent``
                # (DirectEdit/inversion/flow_direct_correction_inv_sd35.py:300).
                z_src_i = z_inv[i].to(device=device, dtype=torch.bfloat16)
                z_hat_src = (z_src_i.float() + d).to(torch.bfloat16)

                if has_cfg:
                    # 3 rows: [neg_tar, cond_src, cond_tar]. The hook swaps
                    # v[2] <- v[1] on configured blocks, so cond_tar's
                    # self-attn output is computed against cond_src's V.
                    # neg_tar (row 0) is untouched — uncond stays honest, no
                    # leak through the patched path.
                    latents = torch.cat([z_hat_tar, z_hat_src, z_hat_tar], dim=0)
                    embeds = torch.cat([embed_neg, embed_src, embed_tar], dim=0)
                    state.set_rows(src_row=1, tar_row=2)
                else:
                    # 2 rows: [cond_src, cond_tar]. No CFG, so neg row drops out.
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
                # Past t_inj: standard CFG (or single) forward on tar only.
                # _v_pred runs cond + uncond as two separate calls; that's
                # fine here because no patched-attn injection is active —
                # state.src_row/tar_row are None, so the hook is a pass-through.
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
