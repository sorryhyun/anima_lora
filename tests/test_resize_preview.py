import pytest

from library.preprocess.resize_preview import (
    compute_resize_preview,
)


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
