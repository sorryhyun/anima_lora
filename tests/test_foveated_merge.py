"""Deferred-foveated merge invariants (networks/foveated.py).

Tier-1.5 obligations from the (now archived) ship proposal
`_archive/proposals/foveated_denoise.md`:

  * **All-fovea mask ≡ identity** — with every cell fovea, the merge is a pure
    permutation (identity order), so the merged forward must be bit-exact to
    the plain forward. This is the bench's 1b startup invariant, now pinned on
    the shipped ``token_merger`` path through ``forward_mini_train_dit``.
  * The merged forward's periphery is **group-shared**: every token patch in a
    merged cell carries the identical velocity (the P0 contract).
  * Merged rope is the **exact mean-position rope** for symmetric groups
    (renormalized elementwise mean of the members' (cos, sin) rows).

The bench probes (archived at `_archive/bench/foveated/`) imported the same
promoted code; with the line archived, this file is the sole live guard of the
plumbing.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from library.anima.models import Anima
from networks.foveated import (
    FoveatedTokenMerge,
    cells_to_mask4,
    rect_cells,
)


def _tiny_anima() -> Anima:
    """A small but real Anima DiT runnable on CPU (see test_native_flatten)."""
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
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, latent_h, latent_w)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


@torch.no_grad()
def test_all_fovea_merge_is_identity():
    # latent 32x32 → 16x16 tokens → 8x8 merge cells at merge_edge=2.
    model = _tiny_anima()
    x, timesteps, emb = _inputs(32, 32)
    ref = model.forward_mini_train_dit(x, timesteps, emb)

    mask4 = torch.ones(1, 1, 32, 32)
    merger = FoveatedTokenMerge(model, mask4, torch.device("cpu"))
    assert merger.L_red == merger.L_full  # pure permutation, nothing merged
    out = model.forward_mini_train_dit(x, timesteps, emb, token_merger=merger)
    assert torch.equal(ref, out)


@torch.no_grad()
def test_merged_periphery_velocity_is_group_shared():
    model = _tiny_anima()
    x, timesteps, emb = _inputs(32, 32)

    # Center-rect fovea on the 8x8 cell grid (cell = 4 latent px at patch 2).
    cells = rect_cells(8, 8, 0.25, 0.5, 0.5)
    assert bool(cells.any()) and not bool(cells.all())
    mask4 = cells_to_mask4(cells, 4, torch.device("cpu"))
    merger = FoveatedTokenMerge(model, mask4, torch.device("cpu"))
    assert merger.L_red < merger.L_full

    v = model.forward_mini_train_dit(x, timesteps, emb, token_merger=merger)
    assert v.shape == x.shape
    assert torch.isfinite(v).all()

    # Every 2x2-token periphery group (= 4x4 latent px cell) is one broadcast
    # row: its four 2x2-px token patches must be identical.
    v4 = v.squeeze(2)  # (B, C, H, W)
    for cy in range(8):
        for cx in range(8):
            block = v4[..., cy * 4 : cy * 4 + 4, cx * 4 : cx * 4 + 4]
            patches = (
                block.reshape(*block.shape[:-2], 2, 2, 2, 2)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(*block.shape[:-2], 4, 2, 2)
            )
            same = torch.equal(patches, patches[..., :1, :, :].expand_as(patches))
            if cells[cy, cx]:
                # Fovea cells keep per-token detail — generically NOT uniform.
                assert not same
            else:
                assert same


def test_merge_rope_is_exact_mean_position():
    # Synthetic 1D-frequency rope over an 8x8 token grid: angle = pos_y·ω. The
    # renormalized (cos, sin) mean of a symmetric 2x2 group must equal the rope
    # at the group's center position (mean angle).
    anima = SimpleNamespace(patch_spatial=2)
    H_t = W_t = 8
    omega = 0.37
    pos_y = torch.arange(H_t).repeat_interleave(W_t).float()  # row-major seq
    theta = (pos_y * omega).reshape(-1, 1, 1, 1)  # (L, 1, 1, 1)
    rope = (torch.cos(theta), torch.sin(theta))

    mask4 = torch.zeros(1, 1, 16, 16)  # all periphery
    merger = FoveatedTokenMerge(anima, mask4, torch.device("cpu"))
    cos_red, sin_red = merger.merge_rope(rope)

    # Cell (cy, cx) center row position = 2·cy + 0.5.
    centers = (
        (torch.arange(4).repeat_interleave(4).float() * 2 + 0.5) * omega
    ).reshape(-1, 1, 1, 1)
    assert torch.allclose(cos_red, torch.cos(centers), atol=1e-5)
    assert torch.allclose(sin_red, torch.sin(centers), atol=1e-5)
