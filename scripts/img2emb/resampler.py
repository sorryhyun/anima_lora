"""Perceiver resampler architecture — shared by phase1_5_anchored and the
phase-1 bench trainer.

Pure model code: no caching, loss, or I/O. Queries are init N(0, 0.15) — the
measured per-element std of the target crossattn_emb — so the prediction
starts at the right scale and the cosine loss isn't stuck on vanishing-norm
outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResamplerBlock(nn.Module):
    """Pre-LN block: (queries ← patches) cross-attn + self-attn over queries + FFN.

    Residual around each sub-layer; no dropout (phase-1 data is small and
    variants already act as implicit augmentation).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.q_norm_x = nn.LayerNorm(d_model)
        self.kv_norm_x = nn.LayerNorm(d_model)
        self.xattn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(d_model)
        self.sattn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv_n = self.kv_norm_x(kv)
        xa, _ = self.xattn(self.q_norm_x(q), kv_n, kv_n, need_weights=False)
        q = q + xa
        qn = self.norm_s(q)
        sa, _ = self.sattn(qn, qn, qn, need_weights=False)
        q = q + sa
        q = q + self.ffn(self.norm_f(q))
        return q


class PerceiverResampler(nn.Module):
    """N-layer resampler with ``n_slots`` learned queries → ``(B, n_slots, d_out)``.

    Queries are init N(0, 0.15) — the measured per-element std of the target
    crossattn_emb. This puts the prediction in the right scale from step 0
    so the cosine loss isn't stuck on vanishing-norm outputs.
    """

    def __init__(
        self,
        d_enc: int,
        d_model: int = 1024,
        n_heads: int = 8,
        n_slots: int = 512,
        n_layers: int = 4,
        d_out: int = 1024,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.15)
        self.kv_proj = nn.Linear(d_enc, d_model)
        self.blocks = nn.ModuleList(
            [ResamplerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        tokens = tokens.to(dtype=self.kv_proj.weight.dtype)
        kv = self.kv_proj(tokens)
        q = self.queries.expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, kv)
        return self.out(self.out_norm(q))
