"""``pilot.deskew_crop``: the padded margin stops at the page edge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
HOME = Path(__file__).resolve().parents[1]


def _pilot():
    spec = importlib.util.spec_from_file_location("pilot", HOME / "ocr" / "pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _page(h=400, w=600):
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _box(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _edge_repeats(c):
    g = c[..., 0]
    return max(
        int((g[0] == g[1]).all()),
        int((g[-1] == g[-2]).all()),
        int((g[:, 0] == g[:, 1]).all()),
        int((g[:, -1] == g[:, -2]).all()),
    )


@pytest.mark.parametrize(
    "poly",
    [_box(200, 150, 320, 230), [210, 150, 330, 170, 320, 240, 200, 220]],
)
def test_interior_crop_is_unchanged(poly):
    p, img = _pilot(), _page()
    new, o1 = p.deskew_crop(img, poly, 0.12, 8)
    old, o2 = p.deskew_crop(img, poly, 0.12, 8, trim=False)
    assert o1 == o2 and np.array_equal(new, old)


def test_edge_crop_drops_the_replicated_lines():
    p, img = _pilot(), _page()
    poly = _box(2, 5, 150, 395)  # left, top and bottom margins all leave the page
    old, _ = p.deskew_crop(img, poly, 0.12, 8, trim=False)
    new, _ = p.deskew_crop(img, poly, 0.12, 8)
    assert _edge_repeats(old) == 1
    assert _edge_repeats(new) == 0
    assert new.shape[0] <= 400 and new.shape[1] < old.shape[1]


def test_tilted_edge_crop_stays_on_page():
    p, img = _pilot(), _page()
    poly = [0, 20, 140, 0, 160, 120, 20, 140]
    old, _ = p.deskew_crop(img, poly, 0.12, 8, trim=False)
    new, _ = p.deskew_crop(img, poly, 0.12, 8)
    assert new is not None and new.size < old.size
