import pytest

from library.datasets.buckets import choose_edge
from library.preprocess.resize_preview import (
    compute_resize_preview,
    normalize_target_res,
    normalize_crop_margins,
)


def test_resize_preview_uses_preprocess_bucket_tier():
    preview = compute_resize_preview(1440, 2560, [768, 1024])

    assert preview.target_edge == choose_edge(1440, 2560, [768, 1024])
    # Free-fit preserves the native aspect to sub-patch (≤16px residual on the
    # covering axis), so almost the whole frame is kept (no AR-snap crop).
    bw, bh = preview.bucket_size
    assert abs(bw / bh - 1440 / 2560) < (16 / min(bw, bh))
    assert preview.kept_rect.width == pytest.approx(1440, abs=16)
    assert preview.kept_rect.left == pytest.approx(0, abs=16)


def test_resize_preview_keeps_full_frame_when_aspect_is_exact_grid():
    # 1008x1024 is exactly 63x64 patches (4032 tok, inside the 1024 band), so
    # free-fit lands it with zero crop.
    preview = compute_resize_preview(1008, 1024, 1024)

    assert preview.bucket_size == (1008, 1024)
    assert preview.kept_rect.left == pytest.approx(0)
    assert preview.kept_rect.top == pytest.approx(0)
    assert preview.kept_rect.width == pytest.approx(1008)
    assert preview.kept_rect.height == pytest.approx(1024)


def test_normalize_target_res_accepts_config_shapes():
    assert normalize_target_res("768, 1024") == [768, 1024]
    assert normalize_target_res(1024) == [1024]
    assert normalize_target_res([896, 1024]) == [896, 1024]


def test_resize_preview_applies_crop_anchor_on_clamped_aspect():
    # Aspect 6.0 exceeds the default max_ratio (4.0), so free-fit clamps and the
    # image is cover-cropped horizontally — the only case anchor still matters.
    center = compute_resize_preview(3000, 500, 1024)
    right = compute_resize_preview(3000, 500, 1024, crop_anchor="right")

    assert center.kept_rect.left > 0
    assert right.kept_rect.left > center.kept_rect.left
    assert right.crop_anchor == "right"


def test_resize_preview_applies_crop_margins_before_bucket_crop():
    preview = compute_resize_preview(
        1000,
        1000,
        1024,
        crop_margins={"top": 10, "right": 0, "bottom": 0, "left": 20},
    )

    assert preview.margin_rect.left == pytest.approx(200)
    assert preview.margin_rect.top == pytest.approx(100)
    assert preview.margin_rect.width == pytest.approx(800)
    assert preview.margin_rect.height == pytest.approx(900)
    assert preview.kept_rect.left >= preview.margin_rect.left
    assert preview.kept_rect.top >= preview.margin_rect.top


def test_normalize_crop_margins_clamps_axis_totals():
    margins = normalize_crop_margins({"left": 80, "right": 80})

    assert margins["left"] + margins["right"] == pytest.approx(95)
