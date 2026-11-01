"""Unit tests for the DirectEdit smart-edit primitives.

Two modules under test:
  * ``library.inference.editing.edit_dispatcher`` — REMOVE/REPLACE/APPEND routing.
  * ``library.inference.editing.directedit_splice`` — T5 diff-span + crossattn surgery.

The dispatcher's REPLACE branch normally calls Qwen3; we mock the encoder
shim with controlled cosine-similarity vectors so the test suite stays
CPU-only and fast. The live Qwen3 regression set is
``scripts/probes/edit_nearest_tag.py``.
"""

from __future__ import annotations

import torch

from library.inference.editing.directedit_splice import (
    find_t5_diff_span,
    splice_crossattn_emb,
)
from library.inference.editing.edit_dispatcher import derive_target_caption


# ---------------------------------------------------------------------------
# encoder mocks
# ---------------------------------------------------------------------------


def _make_encoder(tag_to_vec: dict[str, torch.Tensor]):
    """Return an ``encode_last_pooled`` shim that emits prepared vectors.

    Phrases not in ``tag_to_vec`` get a fresh near-orthogonal unit vector so
    they don't accidentally match anything we set up.
    """
    extras: dict[str, torch.Tensor] = {}

    def encode(phrases: list[str]) -> torch.Tensor:
        out = []
        for p in phrases:
            if p in tag_to_vec:
                out.append(tag_to_vec[p])
            else:
                if p not in extras:
                    # Pseudo-random orthogonal-ish unit vector — large enough
                    # dimension that cosine to anything else is ~0.
                    g = torch.Generator().manual_seed(hash(p) & 0xFFFFFFFF)
                    v = torch.randn(64, generator=g)
                    extras[p] = v / v.norm()
                out.append(extras[p])
        return torch.stack(out, dim=0)

    return encode


# ---------------------------------------------------------------------------
# dispatcher: REMOVE branch
# ---------------------------------------------------------------------------


def test_remove_dash_strips_matching_tag():
    plan = derive_target_caption(
        "1girl, blonde hair, hair ornament, smile",
        "-hair ornament",
        encode_last_pooled=_make_encoder({}),
    )
    assert plan.intent == "remove"
    assert plan.tar_caption == "1girl, blonde hair, smile"
    assert plan.detected_conflict_tag == "hair ornament"


def test_no_x_unmatched_falls_through_to_append():
    """`no shoes` isn't a removal directive when ψ_src has no "shoes" tag —
    it's a valid danbooru tag the user wants to add."""
    src = "1girl, blonde hair, smile"
    plan = derive_target_caption(
        src,
        "no shoes",
        encode_last_pooled=_make_encoder({}),
    )
    assert plan.intent == "append"
    # Literal phrase (with the leading "no") gets appended.
    assert plan.tar_caption == f"{src}, no shoes"


# ---------------------------------------------------------------------------
# dispatcher: REPLACE / APPEND branches
# ---------------------------------------------------------------------------


def _unit(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


def _lerp_unit(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Unit vector at fraction ``t`` along the slerp-ish line from a -> b."""
    v = (1 - t) * a + t * b
    return v / v.norm()


def test_replace_fires_on_confident_match():
    # Build a setup where "large breasts" is cosine-near "medium breasts" only.
    medium = _unit(64, 0)
    edit_vec = _lerp_unit(medium, _unit(64, 1), 0.05)  # very close to medium
    encoder = _make_encoder(
        {
            "large breasts": edit_vec,
            "medium breasts": medium,
        }
    )
    plan = derive_target_caption(
        "1girl, medium breasts, blonde hair, smile",
        "large breasts",
        encode_last_pooled=encoder,
    )
    assert plan.intent == "replace"
    assert plan.detected_conflict_tag == "medium breasts"
    # String substitution preserves position + neighbours.
    assert plan.tar_caption == "1girl, large breasts, blonde hair, smile"
    assert plan.detection_top1_sim is not None
    assert plan.detection_top1_sim >= 0.92
    assert plan.detection_gap is not None
    assert plan.detection_gap >= 0.04


def test_append_when_gap_too_small():
    """Two tags equally close to the edit vec → gap < replace_gap → abstain."""
    base = _unit(64, 42)
    almost = _lerp_unit(base, _unit(64, 43), 0.001)  # near-tie
    edit_vec = _lerp_unit(base, almost, 0.5)
    encoder = _make_encoder(
        {
            "huge breasts": base,
            "large breasts": almost,
            "small breasts": edit_vec,
        }
    )
    plan = derive_target_caption(
        "1girl, huge breasts, large breasts, pink hair",
        "small breasts",
        encode_last_pooled=encoder,
    )
    assert plan.intent == "append"
    # top1 still tracked for diagnostics.
    assert plan.detected_conflict_tag in {"huge breasts", "large breasts"}
    assert plan.detection_gap is not None
    assert plan.detection_gap < 0.04


# ---------------------------------------------------------------------------
# dispatcher: NOOP branch (edit phrase already literally in ψ_src)
# ---------------------------------------------------------------------------


def test_noop_when_edit_already_in_caption():
    """User retyping an existing tag → caption stays put; no encoder call."""
    src = "1girl, blonde hair, smile, school uniform"

    def boom(_phrases: list[str]) -> torch.Tensor:
        raise AssertionError("encoder must not be called for NOOP path")

    plan = derive_target_caption(src, "smile", encode_last_pooled=boom)
    assert plan.intent == "noop"
    assert plan.tar_caption == src
    assert plan.detected_conflict_tag == "smile"


# ---------------------------------------------------------------------------
# splice: diff-span finder
# ---------------------------------------------------------------------------


def test_find_diff_span_identical_returns_empty():
    a = [1, 10, 20, 30, 2, 0, 0]
    span = find_t5_diff_span(a, a, pad_id=0)
    assert span.start == span.src_end == span.tar_end
    assert span.src_span_len == 0
    assert span.tar_span_len == 0


# ---------------------------------------------------------------------------
# splice: crossattn surgery
# ---------------------------------------------------------------------------


def _enc(ids: list[int], length: int = 16) -> torch.Tensor:
    """Right-pad token IDs to ``length`` with 0 → shape (1, length)."""
    out = list(ids) + [0] * (length - len(ids))
    return torch.tensor([out], dtype=torch.long)


def _slot_tensor(ids: list[int], length: int = 16, d: int = 8) -> torch.Tensor:
    """Build a (1, length, d) tensor whose slot i is `id_i * ones(d)`. Lets
    tests assert "this slot came from src vs tar" by inspecting the constant.
    """
    pad = list(ids) + [0] * (length - len(ids))
    rows = torch.tensor(pad, dtype=torch.float32).unsqueeze(-1).expand(-1, d)
    return rows.unsqueeze(0).clone()


def test_splice_replace_transplants_only_diff_span():
    L, D = 16, 8
    src_ids = [1, 10, 20, 30, 2]
    tar_ids = [1, 10, 99, 30, 2]
    crossattn_src = _slot_tensor(src_ids, L, D)
    crossattn_tar = _slot_tensor(tar_ids, L, D)
    out, span = splice_crossattn_emb(
        crossattn_emb_src=crossattn_src,
        crossattn_emb_tar=crossattn_tar,
        t5_ids_src=_enc(src_ids, L),
        t5_ids_tar=_enc(tar_ids, L),
        pad_id=0,
    )
    # Output is the same length as src+tar (they have equal token count).
    assert out.shape == (1, L, D)
    # Slots 0,1 are from src (common prefix).
    assert out[0, 0, 0].item() == 1
    assert out[0, 1, 0].item() == 10
    # Slot 2 is from TAR (the diff span — should be 99, not 20).
    assert out[0, 2, 0].item() == 99
    # Slots 3,4 are from src (common suffix).
    assert out[0, 3, 0].item() == 30
    assert out[0, 4, 0].item() == 2
    # Tail is padded with zeros.
    assert (out[0, 5:, :] == 0).all().item()
    assert span.src_span_len == span.tar_span_len == 1


def test_splice_add_inserts_tar_tokens():
    L, D = 16, 4
    src_ids = [1, 10, 30, 2]
    tar_ids = [1, 10, 50, 60, 30, 2]
    crossattn_src = _slot_tensor(src_ids, L, D)
    crossattn_tar = _slot_tensor(tar_ids, L, D)
    out, span = splice_crossattn_emb(
        crossattn_emb_src=crossattn_src,
        crossattn_emb_tar=crossattn_tar,
        t5_ids_src=_enc(src_ids, L),
        t5_ids_tar=_enc(tar_ids, L),
        pad_id=0,
    )
    # Output runs prefix(2) + tar_span(2) + suffix(2) = 6 active slots.
    assert out[0, 0, 0].item() == 1
    assert out[0, 1, 0].item() == 10
    assert out[0, 2, 0].item() == 50  # from tar
    assert out[0, 3, 0].item() == 60  # from tar
    assert out[0, 4, 0].item() == 30  # from src (suffix)
    assert out[0, 5, 0].item() == 2  # from src (suffix)
    assert (out[0, 6:, :] == 0).all().item()
    assert span.src_span_len == 0
    assert span.tar_span_len == 2


def test_splice_remove_drops_src_tokens():
    L, D = 16, 4
    src_ids = [1, 10, 20, 30, 2]
    tar_ids = [1, 10, 30, 2]
    crossattn_src = _slot_tensor(src_ids, L, D)
    crossattn_tar = _slot_tensor(tar_ids, L, D)
    out, span = splice_crossattn_emb(
        crossattn_emb_src=crossattn_src,
        crossattn_emb_tar=crossattn_tar,
        t5_ids_src=_enc(src_ids, L),
        t5_ids_tar=_enc(tar_ids, L),
        pad_id=0,
    )
    # Output runs prefix(2) + tar_span(0) + suffix(2) = 4 active slots.
    assert out[0, 0, 0].item() == 1
    assert out[0, 1, 0].item() == 10
    assert out[0, 2, 0].item() == 30  # 20 is dropped; 30 is the suffix
    assert out[0, 3, 0].item() == 2
    assert (out[0, 4:, :] == 0).all().item()
    assert span.src_span_len == 1
    assert span.tar_span_len == 0
