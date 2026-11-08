"""swap_text lettering regions: changed text kept in either polarity, line art
misregistration dropped."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np

from i2i_tasks.swap_text.align import refine_ecc
from i2i_tasks.swap_text.build import colour_match
from i2i_tasks.swap_text.common import worker_pool
from i2i_tasks.swap_text.ocr import kept_lines, raw_row
from i2i_tasks.swap_text.regions import diff_regions, grow_regions

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


def _texture(shape: tuple[int, int]) -> np.ndarray:
    noise = np.random.default_rng(0).integers(0, 256, shape, np.uint8)
    return cv2.normalize(
        cv2.GaussianBlur(noise, (0, 0), 4), None, 0, 255, cv2.NORM_MINMAX
    )


def test_refine_ecc_lands_on_the_fit_of_a_larger_twin():
    src = _texture((1200, 900))
    true = np.float32([[2 / 3, 0, 1], [0, 2 / 3, -1]])
    twin = cv2.warpAffine(
        src, true, (1350, 1800), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )
    start = true + np.float32([[0, 0, 2], [0, 0, -1.5]])
    M = refine_ecc(src, twin, start, work_px=600)
    corners = np.array([[0, 0, 1], [1350, 0, 1], [0, 1800, 1], [1350, 1800, 1]], float)
    assert np.abs((M - true) @ corners.T).max() < 0.5


def test_colour_match_takes_a_rescan_tone_curve_back():
    src = cv2.cvtColor(_texture((600, 400)), cv2.COLOR_GRAY2BGR)
    rescan = (255 * (src / 255.0) ** 0.6).astype(np.uint8)
    name, out = colour_match(rescan, src, np.ones(src.shape[:2], bool))
    assert name == "lut"
    assert cv2.absdiff(out, src).mean() < 1.5
    name, out = colour_match(src, src, np.ones(src.shape[:2], bool))
    assert name == "id" and out is src


def test_grown_label_stays_off_the_uncovered_strip():
    labels = np.zeros((200, 200), np.uint8)
    labels[50:60, 50:60] = 1
    labels[50:60, 70:90] = 255
    labels[80:90, 80:90] = 1
    out = grow_regions(labels, np.zeros(labels.shape, np.float32))
    assert (out[50:60, 70:90] == 255).all()


def _shrink(_) -> int:
    page = np.zeros((2000, 2000, 3), np.uint8)
    return cv2.resize(page, (500, 500), interpolation=cv2.INTER_AREA).shape[0]


def test_worker_pool_forks_from_a_parent_that_ran_cv2():
    _shrink(0)
    with worker_pool(2) as pool:
        assert pool.map_async(_shrink, range(4)).get(timeout=30) == [500] * 4


def test_stored_read_keeps_the_filtered_lines_and_their_scores():
    def line(text, det, score, glyph):
        return SimpleNamespace(
            text=text, box=(1, 2, 3, 4), det=det, score=score, glyph_px=glyph
        )

    rows = [
        raw_row(line("만화에서 이런 걸", 0.90, 0.99, 14.7)),
        raw_row(line("만화에서 이런 걸", 0.90, 0.99, 11.9)),
    ]
    assert rows[1] == ["만화에서 이런 걸", [1, 2, 3, 4], 0.90, 0.99, 11.9]
    assert kept_lines(json.loads(json.dumps(rows))) == [
        ["만화에서 이런 걸", [1, 2, 3, 4]]
    ]


def test_line_changes_tells_kept_lettering_from_new():
    from i2i_tasks.swap_text.metrics import line_changes

    a = _lettered("WAIT FOR ME HERO|THE KING IS HERE", 20, 245)
    b = _lettered("WAIT FOR ME HERO|LOOK AT THE SKY", 20, 245)
    b = np.roll(b, (2, 1), axis=(0, 1))  # a registration slip, under the slack
    valid = np.ones(a.shape, np.uint8)
    kept, new = line_changes(a, b, valid, [[112, 310, 420, 345], [112, 345, 420, 380]])
    assert kept < 0.05 < new


def test_text_fallback_to_region_is_capped_by_free_pixels():
    from i2i_tasks.swap_text.pool import _Tally, _paste_mode

    a = SimpleNamespace(
        paste="text",
        max_seam=8.0,
        max_seam_text=6.0,
        max_leftover=0.01,
        max_free=3000,
        max_nontext=8.0,
        max_ink_shift=30.0,
        flip_contrast=40.0,
    )
    g = {"regions": [0], "seam": 5.0, "leftover": 0.0}
    base = {"text_px": 500, "seam_text": 12.0, "leftover_text": 0.0, "nontext": 2.0}
    tally = _Tally()
    assert (
        _paste_mode(0, g, {**base, "free": 2000}, {0}, set(), set(), a, tally)
        == "region"
    )
    assert (
        _paste_mode(0, g, {**base, "free": 40000}, {0}, set(), set(), a, tally) is None
    )
    assert tally.free_groups == 1
    a.paste = "region"
    assert (
        _paste_mode(0, g, {**base, "free": 40000}, {0}, set(), set(), a, tally)
        == "region"
    )


def test_text_seam_ignores_the_line_art_beside_the_box():
    from i2i_tasks.swap_text.metrics import page_metrics

    paper, ink = 240, 20
    src = np.full((300, 300, 3), paper, np.uint8)
    cv2.rectangle(src, (100, 100), (200, 200), ink, 2)  # a text box border
    cv2.putText(src, "JA", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ink, 2)
    tgt = np.full((300, 300, 3), paper, np.uint8)
    cv2.rectangle(tgt, (103, 103), (203, 203), ink, 2)  # the twin's border, 3 px off
    cv2.putText(tgt, "KO", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ink, 2)
    labels = np.zeros((300, 300), np.uint8)
    labels[110:170, 110:190] = 1
    d = __import__("pathlib").Path(__import__("tempfile").mkdtemp())
    for name, img in (("src", src), ("tgt", tgt)):
        cv2.imwrite(str(d / f"p.{name}.webp"), img, [cv2.IMWRITE_WEBP_QUALITY, 101])
    cv2.imwrite(str(d / "p.lab.png"), labels)
    boxes = [[112, 130, 185, 165]]
    _, m = page_metrics(
        ("k", d, "p", [{"regions": [0]}], boxes, 0.003, 4, [["KO", boxes[0]]])
    )
    g = m["groups"][0]
    assert g["text_px"] > 50 and g["seam_text"] < 1.0  # the border is not paper
    assert m["lines"][0] > 0.05
