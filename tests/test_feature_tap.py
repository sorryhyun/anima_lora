"""Feature-tap + early-exit invariants (idea 3.1 of docs/anima/proposal/turbo_gan).

``forward_mini_train_dit`` gained an opt-in ``return_block_features`` /
``return_features_early`` path so the Turbo GAN can read a frozen-teacher block
feature without an external forward hook — and, when only a feature is needed,
stop after the deepest tapped block so just ``blocks[0..k]`` run (the memory win
that fixes the GAN gen-forward OOM).

Invariants:
- default off (both args unset) → **bit-exact** to the plain velocity return,
  in both eager and native-flatten layouts;
- ``return_block_features`` (no early) returns ``(velocity, feats)`` with the
  velocity bit-identical to the plain forward;
- ``return_features_early`` returns the feature dict and the captured tensor
  matches the corresponding block's output from a full forward;
- early-exit really skips the tail blocks (a spy on the last block never fires).
"""

from __future__ import annotations

import torch

from library.anima.models import Anima


def _tiny_anima(num_blocks: int = 4) -> Anima:
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
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    return model.eval()


def _inputs(latent_h: int = 126, latent_w: int = 128):
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, latent_h, latent_w)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


@torch.no_grad()
def _fwd(model, inp, *, native_flatten=False, **kw):
    model._native_flatten = native_flatten
    x, t, c = inp
    return model.forward_mini_train_dit(x, t, c, **kw)


@torch.no_grad()
def test_default_off_is_bit_exact_eager_and_flatten():
    for native in (False, True):
        model = _tiny_anima()
        inp = _inputs()
        base = _fwd(model, inp, native_flatten=native)
        tapped = _fwd(model, inp, native_flatten=native)  # args still unset
        assert torch.equal(base, tapped)
        # request features but no early exit → (velocity, feats), velocity unchanged
        vel, feats = _fwd(model, inp, native_flatten=native, return_block_features={1})
        assert torch.equal(base, vel)
        assert set(feats.keys()) == {1}


@torch.no_grad()
def test_early_exit_returns_matching_feature():
    model = _tiny_anima(num_blocks=4)
    inp = _inputs()
    tap = 1
    # Full forward, capturing the tap (no early exit).
    _vel, full_feats = _fwd(model, inp, return_block_features={tap})
    # Early forward — should stop after block `tap` and return just the dict.
    early = _fwd(model, inp, return_block_features={tap}, return_features_early=True)
    assert isinstance(early, dict)
    assert set(early.keys()) == {tap}
    # The early-captured feature equals the full-forward capture (same block, same
    # input — the tail blocks don't feed back into block `tap`'s output).
    assert torch.equal(early[tap], full_feats[tap])


@torch.no_grad()
def test_early_exit_skips_tail_blocks():
    model = _tiny_anima(num_blocks=4)
    inp = _inputs()
    fired = []
    handle = model.blocks[3].register_forward_hook(lambda *_: fired.append(True))
    try:
        _fwd(model, inp, return_block_features={1}, return_features_early=True)
        assert fired == []  # last block never ran
        fired.clear()
        _fwd(model, inp, return_block_features={1})  # full forward → tail runs
        assert fired == [True]
    finally:
        handle.remove()
