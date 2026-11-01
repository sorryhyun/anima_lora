"""swap_text lettering regions: changed text kept in either polarity, line art
misregistration dropped."""

from __future__ import annotations

import cv2
import numpy as np

from i2i_tasks.swap_text.align import diff_regions, grow_regions

SIZE = (900, 640)


def _lettered(text: str, ink: int, paper: int) -> np.ndarray:
    page = np.full(SIZE, 225, np.uint8)
    page[0:50] = 20  # both percentiles of norm_gray stay on the page
    cv2.rectangle(page, (100, 300), (540, 420), paper, -1)
    for k, line in enumerate(text.split("|")):
        cv2.putText(
            page,
            line,
            (112, 335 + 32 * k),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            ink,
            2,
            cv2.LINE_AA,
        )
    return page


def _regions(a: np.ndarray, b: np.ndarray) -> list[dict]:
    return diff_regions(a, b, np.full(a.shape, 255, np.uint8))[0]


def _covers_lettering(regions: list[dict]) -> bool:
    return any(
        r["box"][0] < 300 < r["box"][2] and r["box"][1] < 350 < r["box"][3]
        for r in regions
    )


def test_dense_dark_lettering_is_kept():
    a = _lettered("WAIT FOR ME HERO|THE KING IS HERE|LOOK AT THE SKY", 20, 245)
    b = _lettered("HOLD ON BRAVE ONE|THE WORLD AT PEACE|DO NOT GO AWAY", 20, 245)
    assert _covers_lettering(_regions(a, b))


def test_white_on_black_lettering_is_kept():
    a = _lettered("WAIT FOR ME HERO|THE KING IS HERE|LOOK AT THE SKY", 245, 20)
    b = _lettered("HOLD ON BRAVE ONE|THE WORLD AT PEACE|DO NOT GO AWAY", 245, 20)
    assert _covers_lettering(_regions(a, b))


def test_misregistered_line_art_is_dropped():
    a = np.full(SIZE, 225, np.uint8)
    a[0:50] = 20
    for k in range(0, 200, 20):
        cv2.line(a, (120 + k, 300), (420 + k, 600), 20, 3, cv2.LINE_AA)
    b = np.roll(a, (2, 2), axis=(0, 1))
    assert _regions(a, b) == []


def test_grown_label_takes_a_stray_dot_and_fills_its_box():
    labels = np.zeros((200, 200), np.uint8)
    labels[50:60, 50:60] = 1
    labels[80:90, 80:90] = 1
    diff = np.zeros(labels.shape, np.float32)
    diff[92:95, 85:88] = 200  # dot beside the region, touching no component
    diff[150:154, 150:154] = 200  # out of reach
    out = grow_regions(labels, diff)
    assert (out[50:90, 50:90] == 1).all()  # box filled
    assert out[93, 86] == 1
    assert out[152, 152] == 0


def test_grown_label_stays_off_the_uncovered_strip():
    labels = np.zeros((200, 200), np.uint8)
    labels[50:60, 50:60] = 1
    labels[50:60, 70:90] = 255
    labels[80:90, 80:90] = 1
    out = grow_regions(labels, np.zeros(labels.shape, np.float32))
    assert (out[50:60, 70:90] == 255).all()
