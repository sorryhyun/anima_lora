"""Vendored Perception Encoder (PE) vision tower — Meta AI, FAIR.

Source: facebookresearch/perception_models @ core/vision_encoder/{pe,rope}.py.
Trimmed to the vision-only path actually used by img2emb (CLIP / text tower
removed). Self-contained: depends on torch + einops + timm.layers.DropPath
only — no xformers, no perception_models package, no `core.*` imports.

License: see ``perception_models/LICENSE.PE`` (FAIR Noncommercial Research).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import DropPath
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint

logger = getLogger(__name__)


# =============================================================================
# RoPE-2D (vendored from core/vision_encoder/rope.py)
# =============================================================================


def _rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@torch.amp.autocast("cuda", enabled=False)
def _apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1]
    t_left, t_mid, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t_mid = (t_mid * freqs.cos() * scale) + (_rotate_half(t_mid) * freqs.sin() * scale)
    return torch.cat((t_left, t_mid, t_right), dim=-1).type(dtype)


class _RotaryEmbedding(nn.Module):
    """Trimmed RotaryEmbedding — only the lang-frequency, non-xpos branch."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("cached_freqs", torch.empty(0), persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", t.type(self.freqs.dtype), self.freqs)
        return repeat(freqs, "... n -> ... (n r)", r=2)


class Rope2D:
    """2D rotary position embedding for ViT patch grids."""

    def __init__(self, dim: int, use_cls_token: bool = False):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size: tuple[int, int] | None = None
        self.freq: torch.Tensor | None = None
        self.rope: _RotaryEmbedding | None = None

    def init_tensors(self):
        self.rope = _RotaryEmbedding(self.dim // 2)

    def update_grid(self, device, grid_h: int, grid_w: int):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)
            self.rope = self.rope.to(device)
            offset = 1 if self.use_cls_token else 0
            grid_y = torch.arange(grid_h, device=device) + offset
            grid_x = torch.arange(grid_w, device=device) + offset
            freqs_y = self.rope(grid_y)[:, None].expand(grid_h, grid_w, -1)
            freqs_x = self.rope(grid_x)[None, :].expand(grid_h, grid_w, -1)
            freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)
            if self.use_cls_token:
                freq = torch.cat(
                    [torch.zeros(1, freq.shape[-1], device=device), freq], dim=0
                )
            self.freq = freq[None, ...]
        self.freq = self.freq.to(device)

    def __call__(self, q, k):
        q = _apply_rotary_emb(self.freq[:, None, :, :], q)
        k = _apply_rotary_emb(self.freq[:, None, :, :], k)
        return q, k


# =============================================================================
# Vision tower (vendored from core/vision_encoder/pe.py)
# =============================================================================


class _LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.dim = dim
        self.init_values = init_values

    def forward(self, x):
        return x * self.gamma

    def init_tensors(self):
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))


class _AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.probe = nn.Parameter(torch.randn(1, num_probe, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layernorm = norm_layer(embed_dim)
        mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(embed_dim, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        return x + self.mlp(self.layernorm(x))


class _SelfAttention(nn.Module):
    """SDPA self-attention with RoPE — matches nn.MultiHeadAttention's param shapes."""

    def __init__(self, embed_dim: int, num_heads: int, rope: Optional[Rope2D] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.rope = rope
        self.scale = self.head_dim ** -0.5

    def init_tensors(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None):
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        proj = (
            proj.unflatten(-1, (3, self.embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        if self.rope is not None:
            q, k = self.rope(q, k)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class _ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[Rope2D] = None,
    ):
        super().__init__()
        if rope is not None:
            self.attn = _SelfAttention(d_model, n_head, rope=rope)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ls_1 = (
            _LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )
        self.ls_2 = (
            _LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def _call_attn(self, q_x, attn_mask=None):
        if isinstance(self.attn, _SelfAttention):
            return self.attn(q_x, attn_mask=attn_mask)
        return self.attn(q_x, q_x, q_x, attn_mask=attn_mask, need_weights=False)[0]

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class _Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[Rope2D] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.ModuleList(
            [
                _ResidualAttentionBlock(
                    width, heads, mlp_ratio,
                    ls_init_value=ls_init_value, act_layer=act_layer,
                    norm_layer=norm_layer, drop_path=drop_path, rope=rope,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, attn_mask=None, layer_idx: int = -1):
        stop_idx = (self.layers + layer_idx) % self.layers
        for i, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
            if i == stop_idx:
                break
        return x


class PEVisionTransformer(nn.Module):
    """PE vision tower — patch tokens out of ``forward_features``, pooled CLIP
    embedding out of ``forward``. Both available; pick what you need.
    """

    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float | None = None,
        drop_path: float = 0.0,
        image_size: int = 448,
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_cls_token: bool = False,
        output_dim: Optional[int] = 1280,
        attn_pooler_heads: int = 8,
        pool_type: Literal["attn", "tok", "avg", "none"] = "attn",
    ):
        super().__init__()
        assert pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = pool_type
        self.patch_size = patch_size
        self.output_dim = output_dim or width
        self.proj_dim = output_dim
        self.heads = heads
        self.width = width
        self.layers = layers
        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d
        self.image_size = image_size

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.rope = Rope2D(dim=width // heads, use_cls_token=use_cls_token) if use_rope2d else None
        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(width) if use_ln_post else nn.Identity()
        self.transformer = _Transformer(
            width, layers, heads, mlp_ratio,
            ls_init_value=ls_init_value, act_layer=act_layer,
            norm_layer=norm_layer, drop_path=drop_path, rope=self.rope,
        )
        self.attn_pool = (
            _AttentionPooling(
                embed_dim=width, num_heads=attn_pooler_heads,
                act_layer=act_layer, norm_layer=norm_layer,
            )
            if pool_type == "attn"
            else None
        )
        self._init_tensors()

    def _init_tensors(self):
        def init_submodule_tensors(module):
            for _, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)
        if self.rope is not None:
            self.rope.init_tensors()
        init_scale = self.width ** -0.5
        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))
        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                init_scale
                * torch.randn(int(self.use_cls_token) + self.posemb_grid_size ** 2, self.width)
            )
        if self.proj_dim is not None:
            self.proj = nn.Parameter(init_scale * torch.randn(self.width, self.proj_dim))

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]
        pos_embed = self.positional_embedding
        cls_token_embed = None
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width).contiguous()
        if self.use_cls_token and cls_token_embed is not None:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)
        return pos_embed[None, ...]

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "tok":
            return x[:, 0]
        if self.pool_type == "avg":
            return x.mean(dim=1)
        if self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        return x  # "none"

    def forward_features(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False,
    ) -> torch.Tensor:
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)
        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x], dim=1,
            )
        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)
        if self.use_rope2d:
            self.rope.update_grid(x.device, grid_h, grid_w)
        x = self.ln_pre(x)
        x = self.transformer(x, layer_idx=layer_idx)
        if norm:
            x = self.ln_post(x)
        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:, :]
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.forward_features(x, norm=True, **kwargs)
        x = self._pool(x)
        if self.proj_dim is not None:
            x = x @ self.proj
        return x

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward returning ``(last_hidden_state, pooled)``.

        ``last_hidden_state`` is ``(B, [1+]N, width)`` post-``ln_post``;
        ``pooled`` is the projected attn-pool / cls / avg output ``(B, output_dim)``.
        """
        feats = self.forward_features(x, norm=True)
        pooled = self._pool(feats)
        if self.proj_dim is not None:
            pooled = pooled @ self.proj
        return feats, pooled

    def load_pe_checkpoint(self, ckpt_path: str, verbose: bool = True) -> None:
        """Load Meta's official ``.pt`` (CLIP-format) weights into this vision tower.

        The released ``.pt`` is a full CLIP state_dict — text, vision, and a
        logit scale all in one. We strip ``module.`` and ``visual.`` so the
        vision tower keys match this module's namespace, and drop the rest.
        """
        sd = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "weights" in sd:
            sd = sd["weights"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        if any(k.startswith("visual.") for k in sd):
            sd = {k.replace("visual.", ""): v for k, v in sd.items() if k.startswith("visual.")}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if verbose and (missing or unexpected):
            logger.info(
                f"PE load: missing={len(missing)} (e.g. {missing[:3]})  "
                f"unexpected={len(unexpected)} (e.g. {unexpected[:3]})"
            )


# =============================================================================
# Configs (only the ones we actually use — extend as needed)
# =============================================================================


@dataclass(frozen=True)
class PEConfig:
    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: float
    output_dim: Optional[int]
    image_size: int = 448
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    use_rope2d: bool = True
    pool_type: Literal["attn", "tok", "avg", "none"] = "attn"
    attn_pooler_heads: int = 8
    use_ln_pre: bool = True
    use_ln_post: bool = True
    ls_init_value: float | None = None
    drop_path: float = 0.0


PE_CONFIGS: dict[str, PEConfig] = {
    "PE-Core-L14-336": PEConfig(
        image_size=336, patch_size=14, width=1024, layers=24, heads=16,
        mlp_ratio=4.0, output_dim=1024, use_cls_token=True, pool_type="attn",
    ),
}


def build_pe_vision(name: str = "PE-Core-L14-336") -> PEVisionTransformer:
    """Instantiate an uninitialized PE vision tower from a config name."""
    if name not in PE_CONFIGS:
        raise KeyError(f"Unknown PE config {name!r} (have: {list(PE_CONFIGS)})")
    cfg = PE_CONFIGS[name]
    return PEVisionTransformer(
        patch_size=cfg.patch_size, width=cfg.width, layers=cfg.layers,
        heads=cfg.heads, mlp_ratio=cfg.mlp_ratio, output_dim=cfg.output_dim,
        image_size=cfg.image_size, use_abs_posemb=cfg.use_abs_posemb,
        use_cls_token=cfg.use_cls_token, use_rope2d=cfg.use_rope2d,
        pool_type=cfg.pool_type, attn_pooler_heads=cfg.attn_pooler_heads,
        use_ln_pre=cfg.use_ln_pre, use_ln_post=cfg.use_ln_post,
        ls_init_value=cfg.ls_init_value, drop_path=cfg.drop_path,
    )
