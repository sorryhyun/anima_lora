"""**build** (CPU, ``--workers``): per work × language, phash-align pages,
register the twin onto the JA page (ORB partial affine), drop unregistered
and off-scale pages; then, at the MOSS cache's ``target_size`` for
``--resolution`` (so the cache encodes the staged pages as-is), refine the
registration by ECC, diff out the lettering regions, warp
the twin into the JA frame, colour-match it there (``colour_match``) and
fill the strip it does not cover from the JA page, then grow each region
over the pixel diff around it and fill its box (``grow_regions``). Regions
whose grown labels touch form a *group*, the unit a swap pastes. Each page
records ``resid`` (mean |src − tgt| outside the lettering) and
``uncovered``; each group its ``seam`` (the same just outside its paste
mask) and ``leftover`` (strong diff connected to the paste mask but outside
it — strokes a paste would leave half-swapped). Writes
``<root>/work/<lang>/<anchor>/{page}.src.webp`` / ``.tgt.webp`` /
``.lab.png`` (region ``k`` as ``k + 1``, 255 outside the warped twin) and
``pages.json``, the work's completion marker.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from i2i_tasks.swap_text.align import align_pages, phash, refine_ecc, register
from i2i_tasks.swap_text.common import WEBP, imwrite, worker_pool, write_text
from i2i_tasks.swap_text.regions import (
    INVALID,
    diff_regions,
    group_metrics,
    grow_regions,
    region_groups,
)
from library.moss.cache import target_size


def _pages(side: Path) -> list[Path]:
    """The side's downloaded pages, ad banners / credit cards (``promo``) dropped."""
    files = side / "files.json"
    if files.exists():
        recs = json.loads(files.read_text(encoding="utf-8"))
        names = [r["file"] for r in recs if r.get("file") and not r.get("promo")]
        return [side / n for n in names if (side / n).exists()]
    return sorted(side.glob("*.webp"))


def _quantiles(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``q`` quantiles of the uint8 values ``v``, read off their histogram."""
    cdf = np.cumsum(np.bincount(v, minlength=256))
    return np.searchsorted(cdf, q * (len(v) - 1), side="right").astype(np.float64)


def colour_match(tgt: np.ndarray, src: np.ndarray, keep: np.ndarray, step=3):
    """``(name, tgt')``: whichever of the twin as-is, a per-channel gain / bias
    (``linear``) and a per-channel quantile match (``lut``, another scan's tone
    curve) leaves the lowest residual on ``keep``. Most pages need none — on 60
    probe pages the linear fit alone raised the median residual 2.8 → 3.4.
    Fit and scored on every ``step``-th pixel; both fits are 256-entry tables."""
    if keep.sum() < 1000:
        return "id", tgt
    t, s, k = tgt[::step, ::step], src[::step, ::step], keep[::step, ::step]
    tk, sk = t[k], s[k]
    x = np.arange(256, dtype=np.float32)
    q = np.linspace(0, 1, 257)
    linear, lut = [], []
    for c in range(3):
        tc, sc = tk[:, c], sk[:, c]
        g, b = np.polyfit(tc.astype(np.float32), sc.astype(np.float32), 1)
        if not 0.7 <= g <= 1.4:
            g, b = 1.0, float(sc.mean(dtype=np.float32) - tc.mean(dtype=np.float32))
        linear.append(g * x + b)
        qa = _quantiles(tc, q) + np.arange(257) * 1e-6
        lut.append(np.interp(x, np.maximum.accumulate(qa), _quantiles(sc, q)))
    cands = {
        "id": None,
        "linear": np.clip(np.stack(linear, -1)[:, None] + 0.5, 0, 255).astype(np.uint8),
        "lut": np.clip(np.stack(lut, -1)[:, None] + 0.5, 0, 255).astype(np.uint8),
    }
    scores = {
        name: cv2.absdiff(s, t if table is None else cv2.LUT(t, table))[k].mean()
        for name, table in cands.items()
    }
    name = min(scores, key=scores.get)
    return name, tgt if cands[name] is None else cv2.LUT(tgt, cands[name])


def _absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel mean |a - b| over the channels."""
    return cv2.absdiff(a, b).mean(-1, dtype=np.float32)


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == size:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _chroma(img: np.ndarray, keep: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1][keep].mean())


def _read_gray(paths: list[Path]) -> tuple[list[Path], list[np.ndarray]]:
    """The readable pages of ``paths`` and their greyscale arrays."""
    kept = [(p, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]
    kept = [(p, g) for p, g in kept if g is not None and g.size]
    return [p for p, _ in kept], [g for _, g in kept]


def _warp_pair(src_file, tw_file, src_g, tw_g, M, scale, resolution):
    """``(src, tgt, valid)`` at the staged size. ECC and everything after run at
    that size; the twin is warped at native size so both pages go through the
    same resize."""
    H, W = src_g.shape
    size = target_size(W, H, resolution)
    M = refine_ecc(src_g, tw_g, M, work_px=max(size))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    valid = cv2.warpAffine(np.full(tw_g.shape, 255, np.uint8), M, (W, H))
    valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
    valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    src = _resize(cv2.imread(str(src_file), cv2.IMREAD_COLOR), size)
    tw = cv2.imread(str(tw_file), cv2.IMREAD_COLOR)
    tgt = _resize(cv2.warpAffine(tw, M, (W, H), flags=interp), size)
    return src, tgt, valid


def _build_page(page, src_file, tw_file, src_g, tw_g, M, scale, out, a) -> dict:
    """The ``pages.json`` row of one registered pair; an ``ok`` row has written
    its ``.src.webp`` / ``.tgt.webp`` / ``.lab.png`` under ``out``."""
    src, tgt, valid = _warp_pair(src_file, tw_file, src_g, tw_g, M, scale, a.resolution)
    regions, labels, changed = diff_regions(_gray(src), _gray(tgt), valid)
    page["changed_frac"] = round(changed, 4)
    if not regions:
        return {**page, "status": "no_regions"}

    labels[valid == 0] = INVALID
    keep = labels == 0
    if abs(_chroma(src, keep) - _chroma(tgt, keep)) > a.max_chroma_gap:
        return {**page, "status": "chroma_mismatch"}
    page["colour"], tgt = colour_match(tgt, src, keep)
    uncovered = labels == INVALID
    tgt[uncovered] = src[uncovered]
    diff = _absdiff(src, tgt)
    labels = grow_regions(labels, diff)
    groups = group_metrics(
        region_groups(labels, regions), regions, labels, diff, a.dilate_frac
    )
    page["uncovered"] = round(float(uncovered.mean()), 4)
    H, W = src_g.shape
    tw_h, tw_w = tw_g.shape
    page["aspect"] = round((tw_w / tw_h) / (W / H) - 1, 4)
    # outside the components, before growth: grown labels take in most of
    # a rescan's diff
    page["resid"] = round(float(diff[keep].mean()), 2) if keep.any() else 255.0
    stem = src_file.stem
    imwrite(out / f"{stem}.src.webp", src, WEBP)
    imwrite(out / f"{stem}.tgt.webp", tgt, WEBP)
    imwrite(out / f"{stem}.lab.png", labels)
    return {
        **page,
        "size": [src.shape[1], src.shape[0]],
        "regions": regions,
        "groups": groups,
        "status": "ok",
    }


def _build_work(work: Path, lang: str, a) -> str:
    out = a.root / "work" / lang / work.name
    if (out / "pages.json").exists() and not a.force:
        return f"{lang}/{work.name}: done, skipped"
    src_files, tw_files = _pages(work / "ja"), _pages(work / lang)
    if not src_files or not tw_files:
        return f"{lang}/{work.name}: no ja/{lang} pages"
    src_files, src_g = _read_gray(src_files)
    tw_files, tw_g = _read_gray(tw_files)
    if not src_files or not tw_files:
        return f"{lang}/{work.name}: no readable ja/{lang} pages"
    pairs = align_pages([phash(g) for g in src_g], [phash(g) for g in tw_g])
    regs = [register(src_g[i], tw_g[j]) for i, j, _ in pairs]
    scales = [
        float(np.hypot(M[0, 0], M[1, 0]))
        for M, inl in regs
        if M is not None and inl >= a.min_inliers
    ]
    work_scale = float(np.median(scales)) if scales else 0.0

    pages = []
    for (i, j, hd), (M, inl) in zip(pairs, regs):
        page = {
            "page": src_files[i].stem,
            "twin": tw_files[j].name,
            "phash": hd,
            "inliers": inl,
        }
        if M is None or inl < a.min_inliers:
            pages.append({**page, "status": "noreg"})
            continue
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        page["scale"] = round(scale, 4)
        # a page off the work's scan scale is a redrawn cover / different art
        if abs(scale / work_scale - 1) > a.scale_tol:
            pages.append({**page, "status": "scale_outlier"})
            continue
        pages.append(
            _build_page(
                page, src_files[i], tw_files[j], src_g[i], tw_g[j], M, scale, out, a
            )
        )

    write_text(out / "pages.json", json.dumps(pages, indent=1, ensure_ascii=False))
    ok = [p for p in pages if p["status"] == "ok"]
    return (
        f"{lang}/{work.name}: ja {len(src_files)} {lang} {len(tw_files)} "
        f"matched {len(pairs)} ok {len(ok)} "
        f"regions {sum(len(p['regions']) for p in ok)}"
    )


def build_work(job) -> str:
    try:
        return _build_work(*job)
    except Exception as e:  # one bad work never takes the pool down
        return f"{job[1]}/{job[0].name}: FAILED {type(e).__name__}: {e}"


def stage_build(a) -> None:
    jobs = []
    for lang in a.langs:
        works = sorted(p.parent for p in a.hito.glob(f"*/*/{lang}") if p.is_dir())
        works = [w for w in works if (w / "ja").is_dir()]
        if a.works:
            works = [w for w in works if w.name in a.works]
        if a.limit:
            works = works[: a.limit]
        jobs += [(w, lang, a) for w in works]
    print(f"build: {len(jobs)} work × language jobs, {a.workers} workers", flush=True)
    if a.force:  # measured per group of the regions being rebuilt
        (a.root / "pool_metrics.json").unlink(missing_ok=True)
    with worker_pool(a.workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(build_work, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)
