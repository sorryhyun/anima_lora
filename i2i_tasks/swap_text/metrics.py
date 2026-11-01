"""What the pool's colour gates measure: how far the twin strays from the JA page
inside each group's paste mask, around the text."""

from __future__ import annotations

import cv2
import numpy as np

from i2i_tasks.swap_text.regions import INVALID, paste_mask


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def _mode_lab(px: np.ndarray) -> np.ndarray:
    """Mean of the most populated 16-level Lab bin: the background of a
    lettering region, robust to the ink on it."""
    q = np.clip(px // 16, 0, 15).astype(np.int32)
    code = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    return px[code == np.bincount(code, minlength=4096).argmax()].mean(0)


def _ink(lab: np.ndarray, bg: np.ndarray, sel: np.ndarray):
    """``(a, b, L − bg L)`` medians of the ink pixels ``sel``, None if too few."""
    if sel.sum() < 30:
        return None
    a, b, L = (float(np.median(lab[..., c][sel])) for c in (1, 2, 0))
    return a, b, L - float(bg[0])


def _text_mask(shape, boxes, det_pad: int) -> np.ndarray:
    text = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        text[
            max(0, y0 - det_pad) : y1 + det_pad, max(0, x0 - det_pad) : x1 + det_pad
        ] = True
    return text


def _group_row(labels, regions, text, diff, ls, lt, dilate_frac, pad) -> dict:
    # region boxes are the build's diff boxes; grow_regions may reach past
    # them, so take the crop from the labels themselves
    ys, xs = np.nonzero(np.isin(labels, [k + 1 for k in regions]))
    if not len(xs):
        return {}
    sl = (
        slice(max(0, ys.min() - pad), ys.max() + pad + 1),
        slice(max(0, xs.min() - pad), xs.max() + pad + 1),
    )
    m = paste_mask(labels[sl], regions, dilate_frac) > 0
    dm, s_, t_ = diff[sl], ls[sl], lt[sl]
    free = m & ~text[sl]
    row = {"free": int(free.sum())}
    if free.sum() >= 100:
        row["nontext"] = round(float(np.median(dm[free])), 2)
    if m.sum() >= 200:
        bs, bt = _mode_lab(s_[m]), _mode_lab(t_[m])
        changed = m & (dm > 30)
        ink_s = _ink(s_, bs, changed & (np.linalg.norm(s_ - bs, axis=-1) > 40))
        ink_t = _ink(t_, bt, changed & (np.linalg.norm(t_ - bt, axis=-1) > 40))
        if ink_s and ink_t:
            row["ink_shift"] = round(
                float(np.hypot(ink_s[0] - ink_t[0], ink_s[1] - ink_t[1])), 2
            )
            row["pol_src"], row["pol_tgt"] = round(ink_s[2], 1), round(ink_t[2], 1)
    return row


def page_metrics(job) -> tuple[str, dict]:
    """``(key, metrics)`` for one built page: per group, inside its paste mask,
    ``nontext`` (median |ΔLab| src vs twin outside the union of both sides'
    text blocks and the twin's OCR lines — the median, so lettering the
    detector missed does not count unless it fills the region), and the colour
    of the ink the swap replaces against the ink it brings
    (``ink_shift``: their a/b distance; ``pol_src`` / ``pol_tgt``: their
    lightness against the region's background). Page-wide: ``bg_resid``, the
    mean |ΔLab| outside every text block."""
    key, d, page, groups, boxes, dilate_frac, det_pad = job
    src = cv2.imread(str(d / f"{page}.src.webp"), cv2.IMREAD_COLOR)
    tgt = cv2.imread(str(d / f"{page}.tgt.webp"), cv2.IMREAD_COLOR)
    labels = cv2.imread(str(d / f"{page}.lab.png"), cv2.IMREAD_UNCHANGED)
    text = _text_mask(labels.shape, boxes, det_pad)
    ls, lt = _lab(src), _lab(tgt)
    diff = np.linalg.norm(ls - lt, axis=-1)
    bg = (labels != INVALID) & ~text
    pad = 2 * max(2, round(min(labels.shape) * dilate_frac)) + 2
    return key, {
        "bg_resid": round(float(diff[bg].mean()), 2) if bg.any() else 0.0,
        "groups": [
            _group_row(labels, g["regions"], text, diff, ls, lt, dilate_frac, pad)
            for g in groups
        ],
    }


def colour_gate(m: dict, a) -> str | None:
    """The gate a group's ``page_metrics`` row fails, else None."""
    if m.get("nontext", 0) > a.max_nontext:
        return "nontext"
    if m.get("ink_shift", 0) > a.max_ink_shift:
        return "ink_shift"
    ps, pt = m.get("pol_src", 0), m.get("pol_tgt", 0)
    if ps * pt < 0 and min(abs(ps), abs(pt)) >= a.flip_contrast:
        return "ink_flip"
    return None
