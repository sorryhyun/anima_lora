"""Invariants for the Block._xattn_gain cross-attn residual gain.

The buffer backs the frontload_text_boost arm-(b) bench
(bench/frontload_text_boost/). Two things must hold:

1. gain = 1.0 (the default) is EXACT identity — the training path and every
   checkpoint rendered before the buffer existed are bit-unchanged.
2. gain = g scales exactly the cross-attn residual (not self-attn, not MLP),
   and set_xattn_gain reaches every block.
"""

import types

import torch
import torch.nn as nn

from library.anima.models import Block


class _StubAttn(nn.Module):
    """Deterministic stand-in for Attention — avoids the dispatcher on CPU."""

    def __init__(self, val: float):
        super().__init__()
        self.val = val

    def forward(self, x, attn_params, ctx, rope_cos_sin=None):
        return torch.full_like(x, self.val)


class _ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def _make_block(cross_val: float = 1.0) -> Block:
    torch.manual_seed(0)
    b = Block(x_dim=8, context_dim=8, num_heads=2)
    b.self_attn = _StubAttn(0.0)
    b.cross_attn = _StubAttn(cross_val)
    b.mlp = _ZeroMLP()
    return b.eval()


def _run(block: Block) -> torch.Tensor:
    torch.manual_seed(1)
    x = torch.randn(2, 1, 2, 2, 8)
    emb = torch.randn(2, 1, 8)
    ctx = torch.randn(2, 4, 8)
    with torch.no_grad():
        return block._forward(x, emb, ctx, attn_params=None)


def test_default_gain_is_exact_identity():
    b = _make_block()
    out_default = _run(b)
    b._xattn_gain.fill_(1.0)
    assert torch.equal(_run(b), out_default)


def test_gain_scales_only_cross_residual():
    b = _make_block(cross_val=1.0)
    out_g1 = _run(b)
    b._xattn_gain.fill_(0.0)
    out_g0 = _run(b)  # cross branch fully removed
    b._xattn_gain.fill_(2.0)
    out_g2 = _run(b)
    # residual doubles (×2 exact on the product; the surrounding residual
    # add/subtract reintroduces rounding, so allclose not bit-equal)
    assert torch.allclose(
        out_g2 - out_g0, 2.0 * (out_g1 - out_g0), atol=1e-6, rtol=1e-5
    )
    # with the cross stub emitting a nonzero residual, gain must change output
    assert not torch.equal(out_g1, out_g0)


def test_gain_buffer_not_in_state_dict():
    b = _make_block()
    assert "_xattn_gain" not in b.state_dict()


def test_renorm_image_mode_keeps_norm_distribution():
    """'img' matches the per-image mean norm but must preserve each token's
    norm RATIO to the boosted state (one shared scale per image) — the
    per-token variant flattens the distribution, which is the grey-tone bug
    this mode exists to fix. frac=0 must be exactly the raw boost."""
    from library.inference.adapters import set_xattn_renorm

    b = _make_block(cross_val=1.0)
    m = types.SimpleNamespace(blocks=[b])
    out_plain = _run(b)
    b._xattn_gain.fill_(2.0)
    out_boost = _run(b)

    set_xattn_renorm(m, True, per_token=False)
    out_img = _run(b)
    n_img = out_img.float().norm(dim=-1)
    n_boost = out_boost.float().norm(dim=-1)
    ratio = n_img / n_boost
    # one shared scale per image → per-token ratios all equal
    assert torch.allclose(ratio, ratio.mean(dim=(1, 2, 3), keepdim=True), atol=1e-4)
    # …and the mean norm lands on the plain shell
    assert torch.allclose(
        n_img.mean(dim=(1, 2, 3)),
        out_plain.float().norm(dim=-1).mean(dim=(1, 2, 3)),
        rtol=1e-3,
    )

    # frac=0 → scale**0 = 1 → raw boost
    set_xattn_renorm(m, True, per_token=True, frac=0.0)
    assert torch.allclose(_run(b), out_boost, atol=1e-6)

    # frac=0.5 lands strictly between raw boost and full shell (per token)
    set_xattn_renorm(m, True, per_token=True, frac=0.5)
    n_half = _run(b).float().norm(dim=-1)
    n_full = out_plain.float().norm(dim=-1)
    lo, hi = torch.minimum(n_full, n_boost), torch.maximum(n_full, n_boost)
    assert ((n_half >= lo - 1e-4) & (n_half <= hi + 1e-4)).all()

    set_xattn_renorm(m, False)


def test_boost_state_helper_resets_to_identity():
    """set_xattn_boost_state(model, 1.0) must fully restore identity —
    gain 1.0 AND renorm off — regardless of the renorm mode passed when
    the boost was set (runners reset with the bare 1.0 call)."""
    from library.inference.adapters import set_xattn_boost_state

    b = _make_block()
    m = types.SimpleNamespace(blocks=[b])
    out = _run(b)
    set_xattn_boost_state(m, 2.0, renorm_mode="img", frac=0.5)
    assert float(b._xattn_gain) == 2.0
    assert b._xattn_renorm is True
    assert b._xattn_renorm_pertoken is False
    assert b._xattn_renorm_frac == 0.5
    set_xattn_boost_state(m, 1.0)
    assert b._xattn_renorm is False
    assert torch.equal(_run(b), out)
