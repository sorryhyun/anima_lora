"""LoRA injection for PE-Core's transformer.resblocks (IP-Adapter prototype).

Wraps every nn.Linear inside PE-Core's resblocks (attn.out_proj,
mlp.c_fc, mlp.c_proj) plus the qkv path (raw ``in_proj_weight`` Parameter on
``_SelfAttention``) with a low-rank delta. The base PE weights stay frozen;
only the LoRA params train. Cleaner-than-the-doc prototype: no cache split,
no implicit-alignment loss — just unfrozen-via-LoRA PE-Core through which the
FM gradient flows back from the DiT.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class PELoRALayer(nn.Module):
    """Standalone rank-r delta. ``y = up(down(x)) * alpha / rank``."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.lora_down(x)) * self.scale


def inject_pe_lora(
    pe_vit: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    target_qkv: bool = True,
    target_attn_out: bool = True,
    target_mlp: bool = True,
    layer_from: int = -1,
) -> nn.ModuleDict:
    """Wrap Linear modules across resblocks; return a ModuleDict the caller
    registers as a child so params show up in optimizer / state_dict.

    ``layer_from``: count of TRAILING resblocks to adapt. ``-1`` (default) or
    ``>= n_blocks`` adapts all blocks. Positive ``N`` < n_blocks adapts only
    the last ``N`` blocks (block indices ``[n_blocks - N, n_blocks)``); earlier
    blocks stay fully frozen with no LoRA params.

    qkv path: ``_SelfAttention.in_proj_weight`` is a raw Parameter (not nn.Linear),
    so we monkey-patch the whole ``_SelfAttention.forward`` to add a LoRA residual
    onto the concatenated ``[Q, K, V]`` projection before the head split.
    """
    if not (hasattr(pe_vit, "transformer") and hasattr(pe_vit.transformer, "resblocks")):
        raise ValueError(
            "inject_pe_lora expects a PEVisionTransformer (with .transformer.resblocks)"
        )

    layers: dict[str, PELoRALayer] = {}
    n_blocks = len(pe_vit.transformer.resblocks)
    if layer_from is None or layer_from < 0 or layer_from >= n_blocks:
        first_idx = 0
    elif layer_from == 0:
        logger.warning(
            "PE-LoRA: layer_from=0 ⇒ no resblocks will be adapted; encoder is "
            "effectively frozen. Use -1 for all layers."
        )
        first_idx = n_blocks  # adapt nothing
    else:
        first_idx = n_blocks - layer_from
    for i, block in enumerate(pe_vit.transformer.resblocks):
        if i < first_idx:
            continue
        attn = block.attn
        # qkv must be patched before attn.out_proj — _patch_pe_qkv calls
        # attn.out_proj(...) inside the patched forward, and we want THAT call
        # to pick up the wrapped (LoRA-augmented) forward when target_attn_out
        # is also on. Order: out_proj wrap first, then qkv patch.
        if target_attn_out and isinstance(getattr(attn, "out_proj", None), nn.Linear):
            _wrap_linear(attn.out_proj, layers, f"b{i}_attn_out_proj", rank, alpha)
        if target_qkv:
            if hasattr(attn, "in_proj_weight"):
                _patch_pe_qkv(attn, layers, f"b{i}_attn_qkv", rank, alpha)
            else:
                logger.warning(
                    f"PE-LoRA: block {i} attn lacks in_proj_weight (likely "
                    f"nn.MultiheadAttention with use_rope2d=False); skipping qkv"
                )
        if target_mlp:
            for name in ("c_fc", "c_proj"):
                lin = getattr(block.mlp, name, None)
                if isinstance(lin, nn.Linear):
                    _wrap_linear(lin, layers, f"b{i}_mlp_{name}", rank, alpha)

    n_adapted = n_blocks - first_idx if first_idx < n_blocks else 0
    logger.info(
        f"PE-LoRA: injected {len(layers)} LoRA layers across {n_adapted}/{n_blocks} resblocks "
        f"(rank={rank}, alpha={alpha}, qkv={target_qkv}, "
        f"attn_out={target_attn_out}, mlp={target_mlp}, "
        f"layer_from={layer_from} ⇒ first_idx={first_idx})"
    )
    return nn.ModuleDict(layers)


def _wrap_linear(
    lin: nn.Linear,
    layers: dict,
    key: str,
    rank: int,
    alpha: float,
) -> None:
    layer = PELoRALayer(lin.in_features, lin.out_features, rank, alpha)
    orig_forward = lin.forward

    def patched(x: torch.Tensor) -> torch.Tensor:
        out = orig_forward(x)
        delta = layer(x.to(layer.lora_down.weight.dtype)).to(out.dtype)
        return out + delta

    lin.forward = patched
    layers[key] = layer


def _patch_pe_qkv(
    attn: nn.Module,
    layers: dict,
    key: str,
    rank: int,
    alpha: float,
) -> None:
    """Replace ``_SelfAttention.forward`` with a copy that adds a LoRA residual
    onto the concatenated qkv projection. Mirrors the original forward in
    library/models/pe.py:182 — keep the two in sync if PE upstream changes.
    """
    embed_dim = attn.embed_dim
    layer = PELoRALayer(embed_dim, 3 * embed_dim, rank, alpha)

    def patched_forward(x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        proj = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        proj = proj + layer(x.to(layer.lora_down.weight.dtype)).to(proj.dtype)
        proj = (
            proj.unflatten(-1, (3, attn.embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]
        q = rearrange(q, "b s (h d) -> b h s d", h=attn.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=attn.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=attn.num_heads)
        if attn.rope is not None:
            q, k = attn.rope(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=attn.scale
        )
        out = rearrange(out, "b h s d -> b s (h d)")
        # attn.out_proj may itself be wrapped (LoRA on attn_out) — call it as a
        # module so its (possibly patched) forward runs.
        return attn.out_proj(out)

    attn.forward = patched_forward
    layers[key] = layer
