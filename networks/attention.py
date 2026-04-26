# Unified attention function supporting various implementations

from dataclasses import dataclass
import torch
from typing import Optional, Union

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None
    _wrapped_flash_attn_forward = None
    _wrapped_flash_attn_backward = None

# Flash Attention 4 (flash-attention-sm120) is disabled; see docs/optimizations/fa4.md.
_flash_attn_4_func_raw = None
flash_attn_4_func = None
flash_attn_4_varlen_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask,
    )

    # Do NOT pre-compile flex_attention here. When blocks are individually
    # compiled (static_token_count mode) or the full model is compiled,
    # the outer torch.compile already traces into _flex_attention and fuses it.
    # Pre-compiling causes nested compilation which exhausts dynamo's
    # recompile limit (grad_mode guard × mask variants) and falls back to
    # the slow unfused path.
    compiled_flex_attention = _flex_attention

except ImportError:
    compiled_flex_attention = None
    create_block_mask = None


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None
    softmax_scale: Optional[float] = (
        None  # custom softmax scale (default: 1/sqrt(head_dim))
    )
    crossattn_block_mask: Optional[object] = (
        None  # pre-computed BlockMask for cross-attention (flex mode only)
    )
    selfattn_block_mask: Optional[object] = (
        None  # pre-computed BlockMask for self-attention padding (flex mode, static-shape training)
    )
    crossattn_full_len: Optional[int] = (
        None  # original KV length before bucketed trimming (for LSE sink correction)
    )
    uniform_seqlens: bool = (
        False  # caller guarantees all seqlens are equal (skips GPU sync check)
    )

    @property
    def supports_fp32(self) -> bool:
        # flash4 is not supported yet, but keep it in the exclusion list for parity.
        return self.attn_mode not in ["flash", "flash4"]

    @property
    def requires_same_dtype(self) -> bool:
        return self.attn_mode in ["xformers"]

    @staticmethod
    def create_attention_params(
        attn_mode: Optional[str],
        split_attn: bool,
        softmax_scale: Optional[float] = None,
    ) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn, softmax_scale=softmax_scale)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str],
        split_attn: bool,
        img_len: Optional[int],
        attention_mask: Optional[torch.Tensor],
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
            max_seqlen = attention_mask.shape[1] + img_len

            if split_attn:
                # cu_seqlens is not needed for split attention
                return AttentionParams(
                    attn_mode,
                    split_attn,
                    img_len,
                    attention_mask,
                    seqlens,
                    None,
                    max_seqlen,
                )

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens = torch.zeros(
                [2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device
            )
            offsets = (
                torch.arange(
                    batch_size, dtype=torch.int32, device=attention_mask.device
                )
                * max_seqlen
            )
            cu_seqlens[1::2] = offsets + seqlens  # end of valid tokens per batch
            cu_seqlens[2::2] = offsets + max_seqlen  # end of all tokens per batch

            # Expand attention mask to include image tokens
            attention_mask = torch.nn.functional.pad(
                attention_mask, (img_len, 0), value=1
            )  # [B, img_len + L]

            if attn_mode == "xformers":
                seqlens_list = seqlens.cpu().tolist()
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens_list, seqlens_list, device=attention_mask.device
                )
            elif attn_mode in ("torch", "flex"):
                attention_mask = attention_mask[:, None, None, :].to(
                    torch.bool
                )  # [B, 1, 1, img_len + L]

            return AttentionParams(
                attn_mode,
                split_attn,
                img_len,
                attention_mask,
                seqlens,
                cu_seqlens,
                max_seqlen,
            )


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_params: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, (
            "k and v must be provided if qkv_or_q is a tensor"
        )
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # Flex attention: early return using BlockMask for variable-length handling (compile-friendly)
    if attn_params.attn_mode == "flex":
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = attn_params.softmax_scale
        B, H, Q_LEN, D = q.shape
        KV_LEN = k.shape[2]

        block_mask = None
        if Q_LEN != KV_LEN and attn_params.crossattn_block_mask is not None:
            # Cross-attention with pre-computed BlockMask (skips padding in text tokens)
            block_mask = attn_params.crossattn_block_mask
        elif Q_LEN == KV_LEN and attn_params.selfattn_block_mask is not None:
            # Self-attention with pre-computed BlockMask (static-shape padding)
            block_mask = attn_params.selfattn_block_mask
        elif attn_params.seqlens is not None:
            # Variable-length: mask padding positions via BlockMask
            seqlens = attn_params.seqlens

            def mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < seqlens[b]

            block_mask = create_block_mask(
                mask_mod, B, H, Q_LEN, KV_LEN, device=q.device
            )
        elif attn_params.attention_mask is not None and isinstance(
            attn_params.attention_mask, torch.Tensor
        ):
            bool_mask = attn_params.attention_mask.squeeze(1).squeeze(1)  # [B, L]

            def mask_mod(b, h, q_idx, kv_idx):
                return bool_mask[b, kv_idx]

            block_mask = create_block_mask(
                mask_mod, B, H, Q_LEN, KV_LEN, device=q.device
            )

        x = compiled_flex_attention(q, k, v, block_mask=block_mask, scale=scale)
        del q, k, v
        x = x.transpose(1, 2)  # [B, L, H, D]
        x = x.flatten(2)  # [B, L, H*D]
        return x

    # If split attn is False, attention mask is provided and all sequence lengths are same, we can trim the sequence
    seqlen_trimmed = False
    if (
        not attn_params.split_attn
        and attn_params.attention_mask is not None
        and attn_params.seqlens is not None
    ):
        if attn_params.uniform_seqlens or torch.all(
            attn_params.seqlens == attn_params.seqlens[0]
        ):
            seqlen = attn_params.seqlens[0].item()
            q = q[:, :seqlen]
            k = k[:, :seqlen]
            v = v[:, :seqlen]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(
                attn_params.attn_mode, False, softmax_scale=attn_params.softmax_scale
            )  # do not in-place modify
            attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
            seqlen_trimmed = True

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn"
        and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):

        def transpose_fn(x):
            return x.transpose(
                1, 2
            )  # [B, H, L, D] for SDPA and sageattn with fixed length

        def pad_fn(x, pad_to):  # pad on sequence length dimension
            return torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:

        def transpose_fn(x):
            return x  # [B, L, H, D] for other implementations

        def pad_fn(x, pad_to):  # pad on sequence length dimension
            return torch.nn.functional.pad(
                x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0
            )

    # Process each batch element with its valid sequence lengths
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # If no seqlens provided, assume all tokens are valid
            attn_params = AttentionParams.create_attention_params(
                attn_params.attn_mode, True, softmax_scale=attn_params.softmax_scale
            )  # do not in-place modify
            attn_params.seqlens = torch.tensor(
                [q.shape[1]] * q.shape[0], device=q.device
            )
            attn_params.max_seqlen = q.shape[1]
        q = [
            transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))
        ]
        k = [
            transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(k))
        ]
        v = [
            transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(v))
        ]
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    scale = attn_params.softmax_scale  # None = default 1/sqrt(head_dim)

    if attn_params.attn_mode == "torch":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = torch.nn.functional.scaled_dot_product_attention(
                    q[i], k[i], v[i], dropout_p=drop_rate, scale=scale
                )
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            del q, k, v

        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_params.attention_mask,
                dropout_p=drop_rate,
                scale=scale,
            )
            del q, k, v

    elif attn_params.attn_mode == "xformers":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(
                    q[i], k[i], v[i], p=drop_rate, scale=scale
                )
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            del q, k, v

        else:
            x = xops.memory_efficient_attention(
                q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate, scale=scale
            )
            del q, k, v

    elif attn_params.attn_mode == "sageattn":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(
                    q[i], k[i], v[i], sm_scale=scale
                )  # B, H, L, D. No dropout support
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            del q, k, v
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v, sm_scale=scale)  # B, L, H, D. No dropout support
            del q, k, v
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv. No dropout support
            x = sageattn_varlen(
                q,
                k,
                v,
                attn_params.cu_seqlens,
                attn_params.cu_seqlens,
                attn_params.max_seqlen,
                attn_params.max_seqlen,
                sm_scale=scale,
            )
            del q, k, v

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = flash_attn_func(
                    q[i], k[i], v[i], drop_rate, softmax_scale=scale
                )  # B, L, H, D
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            del q, k, v
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = flash_attn_func(q, k, v, drop_rate, softmax_scale=scale)  # B, L, H, D
            del q, k, v
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
            x = flash_attn_varlen_func(
                q,
                k,
                v,
                attn_params.cu_seqlens,
                attn_params.cu_seqlens,
                attn_params.max_seqlen,
                attn_params.max_seqlen,
                drop_rate,
                softmax_scale=scale,
            )
            del q, k, v

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash4":
        raise NotImplementedError(
            "attn_mode='flash4' is disabled in this build "
            "(see docs/optimizations/fa4.md). "
            "Use 'flash', 'torch', 'flex', 'sageattn', or 'xformers'."
        )

    else:
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.flatten(2)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(
            x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0
        )  # pad back to max_seqlen

    return x


def attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mode: str,
    softmax_scale: Optional[float] = None,
    drop_rate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDPA that returns ``(out, softmax_lse)``.

    Used by EasyControl's LSE-decomposed extended self-attention to combine
    target+cond attention legs without materializing the
    ``[B, H, S_q, S_t+S_c]`` attention matrix that the masked-SDPA path falls
    into when a per-key bias forces dispatch to the math kernel.

    Args:
        q, k, v: BLHD tensors. K_q and K_kv may differ.
        attn_mode: only ``"flash"`` (FA2) is supported. Other backends raise
            ``NotImplementedError``; callers should fall back to the masked
            path.
        softmax_scale: 1/sqrt(d) by default.
        drop_rate: attention dropout (typically 0).

    Returns:
        out: ``[B, S_q, H, D]``, same dtype as ``q``.
        lse: ``[B, H, S_q]`` log-sum-exp of scaled QK^T over keys, fp32.

    Caveat: FA2's standard ``FlashAttnFunc.backward`` does NOT propagate
    gradient through the returned ``lse`` (the upstream gradient on ``lse``
    is silently dropped — see flash_attn 2.x source). Callers that need
    correct gradients through ``lse`` must use a custom
    ``torch.autograd.Function`` that calls FA's lower-level ops directly —
    see ``networks/easycontrol_anima.py:_ExtendedSelfAttnLSEFunc``.
    """
    if attn_mode != "flash" or flash_attn_func is None:
        raise NotImplementedError(
            f"attention_with_lse currently requires attn_mode='flash' with "
            f"flash-attn installed (got attn_mode={attn_mode!r}, "
            f"flash_attn={'installed' if flash_attn is not None else 'missing'})."
        )
    out, lse, _ = flash_attn_func(
        q, k, v, drop_rate, softmax_scale=softmax_scale, return_attn_probs=True
    )
    return out, lse
