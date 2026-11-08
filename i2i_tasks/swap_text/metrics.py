"""What the pool's colour gates measure: how far the twin strays from the JA page
inside each group's paste mask, around the text."""

from __future__ import annotations

import cv2
import numpy as np

from i2i_tasks.swap_text.regions import (
    INVALID,
    _unshifted,
    box_mask,
    norm_gray,
    paste_mask,
    text_paste_mask,
)

# bump when a row's fields change meaning: ``pool`` re-measures cached rows of
# another version. 2: per-line ``lines``, ``seam_text`` on paper pixels only
METRICS_VERSION = 2
INK_L = 150  # Lab L below which a pixel is ink, not paper, on either side


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
            row["ink_l_src"] = round(ink_s[2] + float(bs[0]), 1)
            row["ink_l_tgt"] = round(ink_t[2] + float(bt[0]), 1)
    return row


def _text_row(
    labels, regions, text, absd, ink_flip, paper, dilate_frac, band=6, near_frac=0.03
) -> dict:
    """The group pasted as ``text_paste_mask``: ``text_px`` (its area),
    ``seam_text`` (mean |Δ| on the ``paper`` pixels of a ``band`` px ring
    outside it, other groups' regions and other text excluded — the step in
    tone the paste edge leaves; the ring's line art is left out, since a box
    border a few px off on a twin from another scan is not a tear) and
    ``leftover_text`` (ink a text-only paste leaves half-swapped within
    ``near_frac`` × the short edge: ``ink_flip`` pixels, ink on one side and
    paper on the other, outside the mask and other groups' regions, over the
    mask area). ``group_metrics``' measures on the build diff, for the
    cut-down mask."""
    m = text_paste_mask(labels, regions, dilate_frac, text) > 0
    if m.sum() < 50:
        return {"text_px": int(m.sum())}
    others = (
        (labels > 0) & (labels != INVALID) & ~np.isin(labels, [k + 1 for k in regions])
    )
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    r_near = max(4, round(near_frac * min(labels.shape)))
    near_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_near + 1,) * 2)
    mu = m.astype(np.uint8)
    ring = (cv2.dilate(mu, ker) > 0) & ~m & ~text & ~others & (labels != INVALID)
    ring &= paper
    near = (cv2.dilate(mu, near_ker) > 0) & ~m & ~others & ink_flip
    return {
        "text_px": int(m.sum()),
        "seam_text": round(float(absd[ring].mean()), 2) if ring.any() else 0.0,
        "leftover_text": round(float(near.sum()) / int(m.sum()), 4),
    }


def line_changes(
    src: np.ndarray, tgt: np.ndarray, valid: np.ndarray, boxes, slack_frac=0.004
) -> list[float]:
    """Per box, the share of its pixels still differing by ``PIX_DIFF`` under
    the box's best single shift within ±r px (``diff_regions``' seed test, on
    the whole box): ~0 for lettering both editions carry in place, 0.2–0.4
    where the twin's lettering is new."""
    na, nb = norm_gray(src, valid), norm_gray(tgt, valid)
    r = max(3, int(round(min(src.shape) * slack_frac)))
    nbp = cv2.copyMakeBorder(nb, r, r, r, r, cv2.BORDER_REPLICATE)
    out = []
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        h, w = int(round(y1)) - y0, int(round(x1)) - x0
        crop = na[y0 : y0 + h, x0 : x0 + w]
        if crop.size == 0:
            out.append(0.0)
            continue
        comp = np.ones(crop.shape, bool)
        out.append(round(_unshifted(na, nbp, comp, x0, y0, r) / crop.size, 3))
    return out


def page_metrics(job) -> tuple[str, dict]:
    """``(key, metrics)`` for one built page: per group, inside its paste mask,
    ``nontext`` (median |ΔLab| src vs twin outside the union of both sides'
    text blocks and the twin's OCR lines — the median, so lettering the
    detector missed does not count unless it fills the region), and the colour
    of the ink the swap replaces against the ink it brings
    (``ink_shift``: their a/b distance; ``pol_src`` / ``pol_tgt``: their
    lightness against the region's background), and the same group pasted
    text-only (``_text_row``). Per twin OCR line (``lines``, in ``read``
    order): ``line_changes`` of its box. Page-wide: ``bg_resid``, the mean
    |ΔLab| outside every text block."""
    key, d, page, groups, boxes, dilate_frac, det_pad, read = job
    src = cv2.imread(str(d / f"{page}.src.webp"), cv2.IMREAD_COLOR)
    tgt = cv2.imread(str(d / f"{page}.tgt.webp"), cv2.IMREAD_COLOR)
    labels = cv2.imread(str(d / f"{page}.lab.png"), cv2.IMREAD_UNCHANGED)
    text = box_mask(labels.shape, boxes, det_pad)
    ls, lt = _lab(src), _lab(tgt)
    diff = np.linalg.norm(ls - lt, axis=-1)
    absd = cv2.absdiff(src, tgt).mean(-1, dtype=np.float32)
    gs, gt = ls[..., 0], lt[..., 0]
    paper = (gs > INK_L) & (gt > INK_L)
    ink_flip = ((gs < 110) & (gt > 170)) | ((gt < 110) & (gs > 170))
    ink_flip = (
        cv2.morphologyEx(
            ink_flip.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        > 0
    )
    bg = (labels != INVALID) & ~text
    pad = 2 * max(2, round(min(labels.shape) * dilate_frac)) + 2
    return key, {
        "v": METRICS_VERSION,
        "bg_resid": round(float(diff[bg].mean()), 2) if bg.any() else 0.0,
        "lines": line_changes(
            cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY),
            (labels != INVALID).astype(np.uint8),
            [b for _, b in read],
        ),
        "groups": [
            {
                **_group_row(
                    labels, g["regions"], text, diff, ls, lt, dilate_frac, pad
                ),
                **_text_row(
                    labels, g["regions"], text, absd, ink_flip, paper, dilate_frac
                ),
            }
            for g in groups
        ],
    }


def colour_gate(m: dict, a, nontext: bool = True) -> str | None:
    """The gate a group's ``page_metrics`` row fails, else None. A text-only
    paste brings no art from the twin, so it skips ``nontext``."""
    if nontext and m.get("nontext", 0) > a.max_nontext:
        return "nontext"
    if m.get("ink_shift", 0) > a.max_ink_shift:
        return "ink_shift"
    ps, pt = m.get("pol_src", 0), m.get("pol_tgt", 0)
    # outlined lettering flips the sign against the region's modal background
    # (the outline on one side, the art behind on the other) with the same ink:
    # a flip also needs the inks themselves that far apart in lightness
    ink_gap = abs(m["ink_l_src"] - m["ink_l_tgt"]) if "ink_l_src" in m else float("inf")
    if (
        ps * pt < 0
        and min(abs(ps), abs(pt)) >= a.flip_contrast
        and ink_gap >= a.flip_contrast
    ):
        return "ink_flip"
    return None
