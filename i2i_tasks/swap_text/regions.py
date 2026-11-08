"""Lettering regions of a registered pair: the diff, its growth into paste labels,
and the groups a swap pastes.

Lifted from ``project/finished/ocr_reader/hito/build_crops.py`` (the recipe and
its thresholds are measured there); ``diff_regions`` keeps a component by its
shift residual and returns the kept components as a label map, which the
composite pastes by. Labels hold region ``k`` as ``k + 1``, 0 elsewhere and
``INVALID`` outside the warped twin.
"""

from __future__ import annotations

import cv2
import numpy as np

PIX_DIFF = 70  # normalised grey levels
INVALID = 255


def norm_gray(g: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = cv2.GaussianBlur(g, (3, 3), 0)
    lo, hi = np.percentile(b[valid > 0], [1, 99])
    return np.clip((b.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)


def _unshifted(a: np.ndarray, b: np.ndarray, comp: np.ndarray, x: int, y: int, r: int):
    """Fewest pixels of ``comp`` (a bool crop at ``(x, y)`` of ``a``) still
    differing under any one shift of ``b`` within ±``r`` px. ``b`` carries an
    ``r`` px border."""
    h, w = comp.shape
    crop = a[y : y + h, x : x + w]
    y, x = y + r, x + r
    return min(
        int(
            (
                (np.abs(crop - b[y + dy : y + dy + h, x + dx : x + dx + w]) > PIX_DIFF)
                & comp
            ).sum()
        )
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
    )


def diff_regions(
    src: np.ndarray,
    warped: np.ndarray,
    valid: np.ndarray,
    slack_frac=0.004,
    close_frac=0.012,
    seed_frac=0.1,
    min_area_frac=0.0004,
):
    """Lettering regions of a registered pair: ``(regions, labels, changed_frac)``.

    A region is a closed component of the pixel diff kept when enough of it is
    *seed* — pixels that still differ under the component's best single shift
    within ``r`` px. A few-px misregistration of line art leaves almost none.
    One shift for the whole component and an absolute difference, so the
    verdict holds for dense lettering, whose strokes sit within ``r`` px of the
    other edition's, and for either polarity. On 300 pages this keeps
    99.9 % of the components holding a translated OCR line (0.714 for the
    per-pixel "ink with no ink within r" it replaces) and 53 % of those away
    from any text block (0.757). ``labels`` (uint8) holds region ``k`` as
    ``k + 1``, 0 elsewhere."""
    na, nb = norm_gray(src, valid), norm_gray(warped, valid)
    pix = ((np.abs(na - nb) > PIX_DIFF) & (valid > 0)).astype(np.uint8)
    k = max(9, int(round(min(src.shape) * close_frac)))
    mask = cv2.morphologyEx(
        pix, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    r = max(3, int(round(min(src.shape) * slack_frac)))
    nb = cv2.copyMakeBorder(nb, r, r, r, r, cv2.BORDER_REPLICATE)
    page_area = src.shape[0] * src.shape[1]
    regions, labels = [], np.zeros(src.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_frac * page_area or area > 0.35 * page_area:
            continue
        comp = (lab[y : y + h, x : x + w] == i) & (valid[y : y + h, x : x + w] > 0)
        seeds = _unshifted(na, nb, comp, x, y, r)
        if seeds < seed_frac * area or len(regions) == INVALID - 1:
            continue
        labels[lab == i] = len(regions) + 1
        regions.append(
            {
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "seed": seeds,
            }
        )
    return regions, labels, float(pix[valid > 0].mean())


def grow_regions(
    labels: np.ndarray, diff: np.ndarray, thr=30.0, reach_frac=0.03, fill_frac=0.05
) -> np.ndarray:
    """``labels`` grown to everything the swap has to replace: each ``diff >
    thr`` pixel joins its nearest region up to ``reach_frac`` × the short edge
    away, then each region whose bounding box is lettering sized (at most
    ``fill_frac`` of the page) fills it. The components stop short of
    a glyph's faint strokes, and a dot or ``!`` that no component touches
    survives a paste by them as a ghost of the source lettering."""
    seeds = (labels > 0) & (labels != INVALID)
    if not seeds.any():
        return labels
    d = cv2.morphologyEx(
        (diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    d = cv2.dilate(d, np.ones((3, 3), np.uint8)) & (labels != INVALID)
    dist, idx = cv2.distanceTransformWithLabels(
        (~seeds).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lut = np.zeros(idx.max() + 1, np.uint8)
    lut[idx[seeds]] = labels[seeds]
    grow = (d > 0) & ~seeds & (dist <= reach_frac * min(labels.shape))
    out = labels.copy()
    out[grow] = lut[idx[grow]]
    for v in np.unique(out[(out > 0) & (out != INVALID)]):
        ys, xs = np.nonzero(out == v)
        box = out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        if box.size <= fill_frac * out.size:
            box[box == 0] = v
    return out


def paste_mask(labels: np.ndarray, chosen, dilate_frac: float) -> np.ndarray:
    """The composite's paste mask for region indices ``chosen`` (uint8 0/1)."""
    mask = np.isin(labels, [k + 1 for k in chosen]).astype(np.uint8)
    d = max(2, round(min(labels.shape) * dilate_frac))
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1,) * 2)
    )
    mask[labels == INVALID] = 0
    return mask


def box_mask(shape, boxes, pad: int) -> np.ndarray:
    """Bool mask of ``boxes`` (``x0, y0, x1, y1``) grown by ``pad`` px."""
    out = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        out[
            max(0, int(y0) - pad) : int(y1) + pad, max(0, int(x0) - pad) : int(x1) + pad
        ] = True
    return out


def text_paste_mask(
    labels: np.ndarray, chosen, dilate_frac: float, text: np.ndarray
) -> np.ndarray:
    """``paste_mask`` cut down to the text boxes ``text`` (both sides' blocks
    and the twin's OCR lines): the lettering without the art a twin from
    another scan brings along with it (uint8 0/1)."""
    return paste_mask(labels, chosen, dilate_frac) & text.astype(np.uint8)


def region_groups(labels: np.ndarray, regions, near_frac=0.025) -> list[list[int]]:
    """Region indices grouped by grown labels within ``near_frac`` × the short
    edge of each other: one balloon's column split by a glyph gap stays one
    group, so a partial swap never pastes half of it."""
    n = len(regions)
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    d = max(2, round(near_frac * min(labels.shape)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1,) * 2)
    for k, r in enumerate(regions):
        x0, y0, x1, y1 = r["box"]
        p = 2 * d + round(0.05 * min(labels.shape))
        sl = (slice(max(0, y0 - p), y1 + p), slice(max(0, x0 - p), x1 + p))
        lab = labels[sl]
        near = lab[cv2.dilate((lab == k + 1).astype(np.uint8), ker) > 0]
        for other in np.unique(near):
            if 0 < other < INVALID and other != k + 1:
                parent[find(int(other) - 1)] = find(k)
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def group_metrics(groups, regions, labels, diff, dilate_frac, band=6, near_frac=0.03):
    """Per group: ``seam`` (mean ``diff`` over the band just outside its paste
    mask, other lettering excluded) and ``leftover`` (strong diff outside every
    region within ``near_frac`` × the short edge of the paste mask, over its
    area) — lettering a paste of the group would leave half-swapped."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    r_near = max(4, round(near_frac * min(labels.shape)))
    near_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_near + 1,) * 2)
    strong = cv2.morphologyEx(
        (diff > 48).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    out = []
    for g in groups:
        x0 = min(regions[k]["box"][0] for k in g)
        y0 = min(regions[k]["box"][1] for k in g)
        x1 = max(regions[k]["box"][2] for k in g)
        y1 = max(regions[k]["box"][3] for k in g)
        p = 2 * r_near
        sl = (slice(max(0, y0 - p), y1 + p), slice(max(0, x0 - p), x1 + p))
        lab = labels[sl]
        m = paste_mask(lab, g, dilate_frac)
        free = (lab == 0) & (m == 0)
        ring = (cv2.dilate(m, ker) > 0) & free
        near = (cv2.dilate(m, near_ker) > 0) & free & (strong[sl] > 0)
        out.append(
            {
                "regions": g,
                "seam": round(float(diff[sl][ring].mean()), 2) if ring.any() else 0.0,
                "leftover": round(float(near.sum()) / max(int(m.sum()), 1), 4),
            }
        )
    return out
