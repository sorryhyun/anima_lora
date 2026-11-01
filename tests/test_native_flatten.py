"""Native-shape flatten invariants (replaces the retired static-pad path).

`compile_blocks()` is the single switch that turns on native-shape flattening:
the forward reshapes each bucket's patch grid to a fake-5D `(B, 1, seq_len, 1, D)`
so dynamo keys the block graph on token count alone. The reshape must be
*bit-exact* to the eager 5D path (the gap=0 control the old pad-leak probe
verified), and eager (uncompiled) forwards must skip it entirely.

The token count exercised is the shipped 4032 (= 63·64) family.
"""

from __future__ import annotations

import pytest
import torch

from library.anima.models import Anima


def _tiny_anima() -> Anima:
    """A small but real Anima DiT runnable on CPU."""
    model = Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    return model.eval()


def _inputs(latent_h: int, latent_w: int):
    """Inputs whose patchified token count is (h/2)*(w/2)."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, latent_h, latent_w)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


def test_compile_blocks_sets_native_flatten_and_budget():
    import torch._dynamo as _dynamo

    model = _tiny_anima()
    assert model._native_flatten is False  # off until compile_blocks

    _dynamo.config.cache_size_limit = 1  # force the max() to raise it
    model.compile_blocks(backend="eager")

    assert model._native_flatten is True
    # 2 token-count families → 2*2 + 8 = 12, and never lowered below current.
    assert _dynamo.config.cache_size_limit >= 12


@torch.no_grad()
def _run(model: Anima, inp, *, native_flatten: bool) -> torch.Tensor:
    model._native_flatten = native_flatten
    x, timesteps, crossattn_emb = inp
    return model.forward_mini_train_dit(x, timesteps, crossattn_emb)


@pytest.mark.core
@torch.no_grad()
def test_flatten_is_bit_exact_4032_family():
    # latent 126x128 → (63)*(64) = 4032 tokens at patch_spatial=2
    model = _tiny_anima()
    inp = _inputs(126, 128)
    out_eager = _run(model, inp, native_flatten=False)
    out_flat = _run(model, inp, native_flatten=True)
    assert torch.equal(out_eager, out_flat)
    assert out_eager.shape == out_flat.shape
