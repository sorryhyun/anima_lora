"""Block compile for OvisOCR3 training (``finetune_ovis3_lora.py --compile``).

Each vision block and LM decoder layer is ``torch.compile(dynamic=False)`` on its
``forward``, behind an eager prologue that soft-marks the varying axes on every call
(``maybe_mark_dynamic``), as in ``vl16_compile.py``. Grad-off calls (generation,
``predict``) go straight to the eager forward.

* Tower (``install_padded_tower``): the crops of a batch sit in a ``(B, L, D)`` layout,
  L = the largest crop's tower tokens rounded up to ``PAD`` (``pad_len``); pad rows carry
  zeros and an identity rotary, attention is one batched SDPA with a key mask, and the
  merger's output is gathered back to the real rows, so the LM receives the stock
  ``pooler_output``. B and L are marked. The stock SDPA path splits per image with
  ``lengths.tolist()``, one graph per crop size and count. ``flash=True`` runs the
  attention as ``flash_attn_varlen_func`` on the flattened ``(B · L, H, 64)`` instead,
  ``cu_seqlens`` alternating real and pad segments (``[0, n0, L, L + n1, 2L, …]``, a
  zero-length pad segment where n_i = L) so pad rows attend among themselves, and
  ``max_seqlen`` = L, symbolic under the mark.
* LM: B and S are marked on the hidden state, both masks, the rotary cos / sin and the
  text position ids. The trainer pads S to a multiple of 8 (Inductor's
  ``sdpa_constraint`` guards the full-attention backward on ``S % 8``). The GDN layers
  run graph-free of breaks: fla's ``chunk_gated_delta_rule`` (``@torch.compiler.disable``)
  is swapped for the custom op ``ocr_ovis3::gdr`` on fla's own kernels (q / k l2norm
  outside the op, in the graph; the backward recomputes the gate cumsum and the
  intra-chunk A, then runs fla's backward), and the causal-conv1d extension for the
  kernel-4 depthwise conv written as four shifted products (``F.conv1d`` in the graph
  had its S specialised by Inductor's conv lowering).

Graphs left: the tower's first block (its input has no grad) and B = 1 (dynamo
specialises sizes 0 / 1) each add one; a batch whose rows carry no padding gets
``None`` masks, one more graph per layer kind.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

PAD = 16
"""Tower-token padding: a multiple of the merger's 2 x 2 group (so groups never mix real
and pad tokens) and of the SDPA mask's alignment."""


def pad_len(tokens):
    """Tower tokens rounded up to ``PAD`` (int or array)."""
    return -(-tokens // PAD) * PAD


# --- GDN chunk rule as a custom op ------------------------------------------------


@torch.library.custom_op("ocr_ovis3::gdr", mutates_args=())
def gdr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """fla's chunk gated delta rule, no initial / final state, q and k already
    l2-normalised. ``[B, T, H, K]`` q / k, ``[B, T, H, V]`` v, ``[B, T, H]`` g / beta."""
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    _, o, *_ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
    )
    return o.to(q.dtype)


@gdr.register_fake
def _(q, k, v, g, beta, scale):
    return torch.empty_like(v, dtype=q.dtype)


@torch.library.custom_op("ocr_ovis3::gdr_bwd", mutates_args=())
def gdr_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``ChunkGatedDeltaRuleFunction.backward`` with the forward's cumsum g and A
    recomputed instead of saved."""
    from fla.ops.gated_delta_rule import chunk as fc

    gc = fc.chunk_local_cumsum(g, chunk_size=64, scale=fc.RCP_LN2)
    _, _, A = fc.chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=gc, beta=beta)
    dq, dk, dv, db, dg, *_ = fc.chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=gc,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=None,
        do=do,
        dht=None,
    )
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dg.to(g.dtype),
        db.to(beta.dtype),
    )


@gdr_bwd.register_fake
def _(do, q, k, v, g, beta, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v, g, beta))


def _gdr_setup(ctx, inputs, output):
    q, k, v, g, beta, scale = inputs
    ctx.save_for_backward(q, k, v, g, beta)
    ctx.scale = scale


def _gdr_backward(ctx, do):
    return (*gdr_bwd(do.contiguous(), *ctx.saved_tensors, ctx.scale), None)


gdr.register_autograd(_gdr_backward, setup_context=_gdr_setup)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """fla's ``l2norm_fwd``: fp32 ``x / sqrt(sum x² + 1e-6)``, back in x's dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


def _patch_gdn() -> None:
    """Route the GDN layers' training calls (grad on, no cache) to ``gdr`` and the
    torch conv; generation keeps fla and the causal-conv1d extension."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    if getattr(q35, "_ocr_ovis3_patched", False):
        return
    chunk_rule, conv_fn = q35.torch_chunk_gated_delta_rule, q35.causal_conv1d_fn

    def chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        if (
            not torch.is_grad_enabled()
            or initial_state is not None
            or output_final_state
            or kwargs.get("cu_seqlens") is not None
        ):
            return chunk_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs,
            )
        if use_qk_l2norm_in_kernel:
            query, key = _l2norm(query), _l2norm(key)
        o = gdr(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            key.shape[-1] ** -0.5,
        )
        return o, None

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if not torch.is_grad_enabled():
            return conv_fn(hidden_states, weight, bias, activation=activation, **kwargs)
        # depthwise causal conv as shifted products (pointwise in the graph, fp32 sum):
        # Inductor's conv lowering specialised S (smoke_ovis3_dyn60_1127)
        seq_len, width = hidden_states.shape[-1], weight.shape[-1]
        xp = F.pad(hidden_states.to(weight.dtype), (width - 1, 0)).float()
        w = weight.float()
        out = sum(w[:, j, None] * xp[..., j : j + seq_len] for j in range(width))
        if bias is not None:
            out = out + bias.float()[:, None]
        out = out.to(weight.dtype)
        if activation is not None:
            out = q35.ACT2FN[activation](out)
        return out.to(hidden_states.dtype)

    q35.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    q35.causal_conv1d_fn = causal_conv1d_fn
    q35._ocr_ovis3_patched = True


# --- padded tower -------------------------------------------------------------------


def install_padded_tower(model, flash: bool = False) -> None:
    """Run the vision tower on the ``(B, L, D)`` layout (module docstring), its attention
    on SDPA with a key mask or (``flash``) FlashAttention varlen. Grad-off calls take it
    too (eager), with L above the training range for the val crops."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    def attn_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kw):
        mask = kw.pop("pad_mask")
        B, L, _ = hidden_states.shape
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(B, L, 3, self.num_heads, -1)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )
        cos, sin = position_embeddings  # (B, L, d): unsqueezed over heads inside
        q, k = q35.apply_rotary_pos_emb_vision(q, k, cos, sin)
        if flash:
            from flash_attn import flash_attn_varlen_func

            cu = kw.pop("fa_cu")
            o = flash_attn_varlen_func(
                *(t.reshape(B * L, self.num_heads, -1) for t in (q, k, v)),
                cu,
                cu.clone(),  # dynamo breaks on one tensor fed to two Function inputs
                L,
                L,
                softmax_scale=self.scaling,
            )
            return self.proj(o.reshape(B, L, -1))
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
        return self.proj(o.transpose(1, 2).reshape(B, L, -1))

    def vision_forward(self, hidden_states, grid_thw, **kw):
        interp, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        pos_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        hs = self.patch_embed(hidden_states)
        hs = hs + (self.pos_embed(interp) * weights[:, :, None]).sum(1).to(hs.dtype)
        rot = self.rotary_pos_emb(pos_ids).reshape(hs.shape[0], -1)
        emb = torch.cat((rot, rot), dim=-1)

        n = grid_thw.prod(-1).cpu()
        B, L, D, d = len(n), int(pad_len(int(n.max()))), hs.shape[1], emb.shape[1]
        dev = hs.device
        idx = torch.cat([i * L + torch.arange(int(c)) for i, c in enumerate(n)]).to(dev)
        midx = torch.cat(
            [i * (L // 4) + torch.arange(int(c) // 4) for i, c in enumerate(n)]
        ).to(dev)
        pad = hs.new_zeros(B * L, D).index_copy(0, idx, hs).view(B, L, D)
        cos = emb.new_ones(B * L, d).index_copy(0, idx, emb.cos()).view(B, L, d)
        sin = emb.new_zeros(B * L, d).index_copy(0, idx, emb.sin()).view(B, L, d)
        mask = (torch.arange(L)[None] < n[:, None]).view(B, 1, 1, L).to(dev)
        cu = torch.zeros(1, dtype=torch.int32, device=dev)  # unused by attn_forward
        extra = {}
        if flash:  # segment ends n0, L, L + n1, 2L, …
            ends = torch.stack([torch.arange(B) * L + n, torch.arange(1, B + 1) * L], 1)
            extra["fa_cu"] = F.pad(ends.flatten(), (1, 0)).to(dev, torch.int32)
        for blk in self.blocks:
            pad = blk(
                pad,
                cu_seqlens=cu,
                position_embeddings=(cos, sin),
                pad_mask=mask,
                **extra,
            )
        merged = self.merger(pad)  # (B · L / 4, out): the merger views to rows itself
        return BaseModelOutputWithPooling(
            last_hidden_state=pad.reshape(B * L, D).index_select(0, idx),
            pooler_output=merged.index_select(0, midx),
        )

    vis = model.model.visual
    vis.forward = types.MethodType(vision_forward, vis)
    for blk in vis.blocks:
        blk.attn.forward = types.MethodType(attn_forward, blk.attn)


# --- LM full attention on FlashAttention --------------------------------------------


def install_lm_flash() -> None:
    """Run the LM's full-attention layers on causal ``flash_attn_varlen_func`` in
    training forwards (grad on); generation keeps SDPA. Needs a FlashAttention build
    with head dim 256 (the stock wheel, or the sm120 fork from its hdim64-96-128-256 release).

    Training pads on the left, so each row is a pad segment then a real one:
    ``cu_seqlens`` = ``[0, p0, S, S + p1, 2S, …]`` with p_i the row's pad count, read in
    the graph off the 4D mask's last query row (a left-padded row's last token is real
    and sees every real key); ``max_seqlen`` = S, symbolic under the mark. Overrides
    transformers' ``sdpa`` entry for ``Qwen3_5Attention`` modules only."""
    from flash_attn import flash_attn_varlen_func
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    def attention(module, query, key, value, attention_mask, scaling=None, **kw):
        if not (torch.is_grad_enabled() and isinstance(module, q35.Qwen3_5Attention)):
            return sdpa_attention_forward(
                module, query, key, value, attention_mask, scaling=scaling, **kw
            )
        B, _, S, _ = query.shape
        q, k, v = (
            t.transpose(1, 2).reshape(B * S, t.shape[1], -1)
            for t in (query, key, value)
        )
        if attention_mask is None:
            pads = torch.zeros(B, dtype=torch.int32, device=q.device)
        else:
            pads = S - attention_mask[:, 0, -1, :].sum(-1, dtype=torch.int32)
        base = torch.arange(B, device=q.device, dtype=torch.int32) * S
        cu = F.pad(torch.stack([base + pads, base + S], 1).flatten(), (1, 0))
        o = flash_attn_varlen_func(
            q, k, v, cu, cu.clone(), S, S, softmax_scale=scaling, causal=True
        )
        return o.view(B, S, query.shape[1], -1), None

    ALL_ATTENTION_FUNCTIONS.register("sdpa", attention)


# --- compile ------------------------------------------------------------------------


def _mark(t, *dims) -> None:
    if isinstance(t, torch.Tensor):
        for d in dims:
            torch._dynamo.maybe_mark_dynamic(t, d % t.ndim)


def compile_layers(model) -> tuple[int, int]:
    """Patch the GDN path and compile every vision block / LM decoder layer in place.
    Run after peft wraps the linears, ``install_padded_tower`` and the checkpoint flags.
    Returns the layer counts (tower, LM)."""
    from library.runtime.dynamo import pin_dynamo_limit

    _patch_gdn()
    pin_dynamo_limit("recompile_limit", 32)
    torch._dynamo.config.specialize_float = True
    torch._logging.set_logs(recompiles=True, graph_breaks=True)  # each with its reason
    n_threads = torch.get_num_threads()

    def tower(compiled, eager):
        def forward(hidden_states, cu_seqlens, position_embeddings=None, **kw):
            if not torch.is_grad_enabled():
                return eager(hidden_states, cu_seqlens, position_embeddings, **kw)
            _mark(hidden_states, 0, 1)
            for pe in position_embeddings:
                _mark(pe, 0, 1)
            _mark(kw.get("pad_mask"), 0, 3)
            _mark(kw.get("fa_cu"), 0)  # 2B + 1
            return compiled(hidden_states, cu_seqlens, position_embeddings, **kw)

        return forward

    def lm(compiled, eager):
        def forward(hidden_states, position_embeddings=None, attention_mask=None, **kw):
            if not torch.is_grad_enabled():
                return eager(hidden_states, position_embeddings, attention_mask, **kw)
            # a checkpoint recompute runs on the autograd thread, whose intra-op thread
            # count differs: dynamo's GLOBAL_STATE guard then doubles every graph
            if torch.get_num_threads() != n_threads:
                torch.set_num_threads(n_threads)
            _mark(hidden_states, 0, 1)
            for pe in position_embeddings or ():
                _mark(pe, -3, -2)  # (B, S, d)
            if isinstance(attention_mask, torch.Tensor):
                # full attention (B, 1, S, S) bool, linear attention (B, S) long
                _mark(
                    attention_mask, *((0, 2, 3) if attention_mask.ndim == 4 else (0, 1))
                )
            _mark(kw.get("position_ids"), 0, 1)
            return compiled(hidden_states, position_embeddings, attention_mask, **kw)

        return forward

    n_t = n_l = 0
    for m in model.modules():
        name = type(m).__name__
        if name == "Qwen3_5VisionBlock":
            m.forward = tower(torch.compile(m.forward, dynamic=False), m.forward)
            n_t += 1
        elif name == "Qwen3_5DecoderLayer":
            m.forward = lm(torch.compile(m.forward, dynamic=False), m.forward)
            n_l += 1
    return n_t, n_l
