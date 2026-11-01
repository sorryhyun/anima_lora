"""Page matching, registration and lettering diff for hitocrawl edition twins.

Lifted from ``project/finished/ocr_reader/hito/build_crops.py`` (the recipe and
its thresholds are measured there); ``diff_regions`` keeps a component by its
shift residual and returns the kept components as a label map, which the
composite pastes by.
"""

from __future__ import annotations

import cv2
import numpy as np

PIX_DIFF = 70  # normalised grey levels
ORB_FEATURES = (1500, 5000)
SURE_INLIERS = 200
MAX_MISFIT = 20.0  # grey levels: lettering alone stays well under it


def phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8]
    return (d > np.median(d)).flatten()


def align_pages(a: list[np.ndarray], b: list[np.ndarray], gap=16.0, thr=14):
    """[(i, j, hamming)] — monotonic Needleman-Wunsch over 64-bit phashes."""
    n, m = len(a), len(b)
    dist = np.array([[int((x != y).sum()) for y in b] for x in a]).reshape(n, m)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1) * gap
    d[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(
                d[i - 1, j - 1] + dist[i - 1, j - 1],
                d[i - 1, j] + gap,
                d[i, j - 1] + gap,
            )
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + dist[i - 1, j - 1]:
            if dist[i - 1, j - 1] <= thr:
                out.append((i - 1, j - 1, int(dist[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def _orb_fit(a: np.ndarray, b: np.ndarray, sa: float, sb: float, features: int):
    orb = cv2.ORB_create(features)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)
    if len(matches) < 20:
        return None, 0
    pb = np.float32([kb[mm.queryIdx].pt for mm in matches]) / sb
    pa = np.float32([ka[mm.trainIdx].pt for mm in matches]) / sa
    M, inl = cv2.estimateAffinePartial2D(
        pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M, int(inl.sum()) if inl is not None else 0


def _misfit(a: np.ndarray, b: np.ndarray, M: np.ndarray, sa: float, sb: float) -> float:
    """Mean |a - b| where the ``M``-warped ``b`` covers ``a``."""
    Mab = M * sa
    Mab[:, :2] /= sb
    size = a.shape[::-1]
    warped = cv2.warpAffine(b, Mab, size)
    covered = cv2.warpAffine(np.full(b.shape, 255, np.uint8), Mab, size) == 255
    return float(cv2.absdiff(a, warped)[covered].mean()) if covered.any() else 255.0


def register(src: np.ndarray, tw: np.ndarray, work_px=1200):
    """2×3 affine mapping twin → src coordinates, and its RANSAC inlier count.

    The brute-force match is quadratic in the features: 5000 of them took 308 ms
    of a 1.1 s page for ~2000 inliers. 1500 matches in 61 ms and lands within
    0.03 px after ECC, but loses pages with little shared art and can settle on a
    repeated pattern, so a fit under ``SURE_INLIERS`` or past ``MAX_MISFIT`` retries
    at 5000."""
    sa, sb = work_px / max(src.shape), work_px / max(tw.shape)
    a = cv2.resize(src, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    b = cv2.resize(tw, None, fx=sb, fy=sb, interpolation=cv2.INTER_AREA)
    for features in ORB_FEATURES:
        M, inl = _orb_fit(a, b, sa, sb, features)
        if inl >= SURE_INLIERS and _misfit(a, b, M, sa, sb) <= MAX_MISFIT:
            break
    return M, inl


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
        if seeds < seed_frac * area or len(regions) == 254:
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


def refine_ecc(src: np.ndarray, tw: np.ndarray, M: np.ndarray, work_px=1600):
    """``M`` refined by ECC: a full affine fit of the ``M``-warped twin to
    ``src`` at ``work_px``, composed onto ``M``. ORB's partial affine leaves a
    ~1 px misfit that lights every line in the diff; on 60 pages across the
    residual range ECC took the p90 residual from 10.0 to 6.5. Returns ``M``
    unchanged when ECC fails or wanders past 2 % of the page. A page already
    under ``work_px`` is fit at its own size.

    The fit is masked to where the warped twin covers the page: unmasked, a
    twin that leaves an uncovered strip drags the fit toward its black border
    (one such work: residual 6.6 before ECC, 8.5 after, 5.1 masked). ECC stops
    when the correlation moves < 1e-3; at 1e-6 some works oscillated to the
    iteration cap (~40 ms each) for < 0.03 px of change."""
    s = min(1.0, work_px / max(src.shape))
    k = min(1.0, s * float(np.hypot(M[0, 0], M[1, 0])))
    a, small = src, tw
    if s < 1:
        a = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if k < 1:
        small = cv2.resize(tw, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    # M between the two working rasters, cv2.resize's pixel-centre offsets included
    from_small = np.array([[1 / k, 0, 0.5 / k - 0.5], [0, 1 / k, 0.5 / k - 0.5]])
    to_work = np.array([[s, 0, 0.5 * s - 0.5], [0, s, 0.5 * s - 0.5]])
    Mw = to_work @ np.vstack([M, [0, 0, 1]]) @ np.vstack([from_small, [0, 0, 1]])
    size = a.shape[::-1]
    b = cv2.warpAffine(small, Mw, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.full(small.shape, 255, np.uint8), Mw, size)
    e = max(3, round(15 * s) | 1)
    mask = cv2.erode(mask, np.ones((e, e), np.uint8))
    R = np.eye(2, 3, dtype=np.float32)
    try:
        _, R = cv2.findTransformECC(
            a.astype(np.float32),
            b.astype(np.float32),
            R,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3),
            mask,
            5,
        )
    except cv2.error:
        return M
    if (
        np.abs(R[:, 2]).max() > 0.02 * max(a.shape)
        or np.abs(R[:, :2] - np.eye(2)).max() > 0.02
    ):
        return M
    R[:, 2] /= s
    Ri = np.vstack([cv2.invertAffineTransform(R), [0, 0, 1]])
    return (Ri @ np.vstack([M, [0, 0, 1]]))[:2]


def grow_regions(
    labels: np.ndarray, diff: np.ndarray, thr=30.0, reach_frac=0.03, fill_frac=0.05
) -> np.ndarray:
    """``labels`` grown to everything the swap has to replace: each ``diff >
    thr`` pixel joins its nearest region up to ``reach_frac`` × the short edge
    away, then each region whose bounding box is lettering sized (at most
    ``fill_frac`` of the page) fills it. The components stop short of
    a glyph's faint strokes, and a dot or ``!`` that no component touches
    survives a paste by them as a ghost of the source lettering."""
    seeds = (labels > 0) & (labels != 255)
    if not seeds.any():
        return labels
    d = cv2.morphologyEx(
        (diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    d = cv2.dilate(d, np.ones((3, 3), np.uint8)) & (labels != 255)
    dist, idx = cv2.distanceTransformWithLabels(
        (~seeds).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lut = np.zeros(idx.max() + 1, np.uint8)
    lut[idx[seeds]] = labels[seeds]
    grow = (d > 0) & ~seeds & (dist <= reach_frac * min(labels.shape))
    out = labels.copy()
    out[grow] = lut[idx[grow]]
    for v in np.unique(out[(out > 0) & (out != 255)]):
        ys, xs = np.nonzero(out == v)
        box = out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        if box.size <= fill_frac * out.size:
            box[box == 0] = v
    return out
