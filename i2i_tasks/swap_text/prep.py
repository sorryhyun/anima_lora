#!/usr/bin/env python3
"""Stage the swap_text i2i task from hitocrawl edition twins.

Every hitocrawl work is the same pages lettered in JA and in a translated
edition (``en/``, ``ko/``). The condition is the JA page; the target is the
same page with its translated lettering pasted where the OCR can read it, and
the caption lists the text read off that target. Six stages, each idempotent:

1. **build** (CPU, ``--workers``) — per work × language: phash-align pages,
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
2. **ocr** (GPU) — reads every twin ``.tgt.webp`` within ``--max_resid`` /
   ``--max_uncovered`` through the anime_tools
   reader, with the MOSS caption helper's floors
   (``library.moss.colorize.readable``) and a Latin line at reader score
   ``LATIN_MIN_SCORE`` whatever its det, into ``<root>/ocr.json``.
3. **det** (GPU) — AnimeText text blocks on both sides of the same pages
   (``.src.webp`` and ``.tgt.webp``) into ``<root>/det.json``: the text mask the
   pool's colour gates measure around, independent of the build's diff regions
   (which hold whatever changed, text or not).
4. **pool** (CPU) — drops pages past ``--max_resid`` (another scan, not
   a relettering) or ``--max_uncovered``, and assigns each twin OCR line to the
   region its centre falls in. A line holding kana or CJK ideographs
   (``JA_SCRIPT``) inside a region is a misread of the translated lettering.
   A group is *clean* when a non-misread line landed in it, none misread, every
   twin text block (``det``) landing in it is covered by a kept line (pasted,
   an unread block is lettering the caption does not name), its
   ``seam`` ≤ ``--max_seam`` and ``leftover`` ≤ ``--max_leftover``, and the
   twin keeps the page's look inside its paste mask (``page_metrics``,
   cached in ``<root>/pool_metrics.json``): outside the union of both sides'
   text blocks and the twin's OCR lines the twin matches (``nontext`` ≤
   ``--max_nontext`` — a darker balloon fill or screentone, a censor bar or
   mosaic one edition lacks, a translator's end card), and the lettering keeps its
   colour (``ink_shift`` ≤ ``--max_ink_shift``, no light ↔ dark flip against
   the background of ≥ ``--flip_contrast`` on both sides). A page with no
   clean group drops. Writes ``<root>/pool/<lang>/{stem}`` (``.src.webp`` /
   ``.tgt.webp`` symlinks to the built pair, the twin uncomposited),
   ``pool.jsonl`` (clean groups, their lines, the metrics) and ``stats.json``;
   reruns rewrite the tree.
5. **assemble** (CPU) — samples ``--sample`` pool pages (0: all) and builds
   each target: the JA page with groups pasted from the twin — lettering the
   reader cannot read, that a paste would tear or recolour, stays JA:

   * **full** — every clean group pasted.
   * **partial** (``--partial_ratio`` of the pairs) — a random proper subset
     of the clean groups.

   Writes ``<root>/staging/<lang>/{stem}`` (``.tgt.webp``, ``.src.webp``
   symlink), ``plan.jsonl`` and ``stats.json``; reruns rewrite the tree.
6. **caption** — captions each target with the twin read's lines in its
   pasted groups (CPU), or with ``--caption_ocr`` (GPU) with a second read of
   the staged target itself (``<root>/ocr_staged.json``, keyed by page and
   pasted regions); either way only lines holding no kana or CJK ideograph, so
   JA lettering — left by the translator or by a partial swap — is never named
   and stays the source; a target with no such line drops. The caption is layout JSON:
   ``{"summary": {"caption": "comic, english text"}, "items": [one text item
   (text + bbox) per line]}``. Writes ``{stem}.txt``, ``pairs.jsonl`` and the
   final ``stats.json``.

Stems are ``{anchor}_{page}_{lang}``; ``pairs.jsonl`` rows carry ``work`` so a
holdout can split by work (the same work appears in both languages).

    make i2i-stage TASK=swap_text ARGS="--limit 4"
    make daemon-run ARGS="i2i_tasks/swap_text/prep.py --stages ocr det pool assemble caption"
    .venv/bin/python i2i_tasks/swap_text/prep.py --stages assemble caption --sample 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import (  # noqa: E402
    align_pages,
    diff_regions,
    grow_regions,
    phash,
    refine_ecc,
    register,
)

from library.env import resolve_under_home  # noqa: E402
from library.moss.cache import target_size  # noqa: E402

LANG_TAG = {"en": "english text", "ko": "korean text"}
# kana / CJK ideographs: in a translated edition's region these are the reader
# taking hand-lettered Hangul for katakana, or JA lettering left in place
JA_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]"
)
LATIN_MIN_SCORE = 0.8
INVALID = 255
WEBP = [cv2.IMWRITE_WEBP_QUALITY, 95]
# The staged target is lossless: outside its paste mask it is the decoded source
# exactly, so target − reference is zero wherever the edit is not.
WEBP_LOSSLESS = [cv2.IMWRITE_WEBP_QUALITY, 101]


def _single_thread() -> None:
    """Pool initializer: one OpenCV thread per worker. cv2 otherwise spawns a
    thread per core in every worker, and 12 workers ran at load ~100 on 12
    cores."""
    cv2.setNumThreads(1)


def _stable_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _write_atomic(out: Path, write) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp" + out.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _imwrite(out: Path, img: np.ndarray, params=()) -> None:
    def write(tmp):
        if not cv2.imwrite(tmp, img, list(params)):
            raise OSError(f"cv2.imwrite failed: {out}")

    _write_atomic(out, write)


def _write_text(out: Path, text: str) -> None:
    _write_atomic(out, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))


# ------------------------------------------------------------------ build


def _pages(side: Path) -> list[Path]:
    """The side's downloaded pages, ad banners / credit cards (``promo``) dropped."""
    files = side / "files.json"
    if files.exists():
        recs = json.loads(files.read_text(encoding="utf-8"))
        names = [r["file"] for r in recs if r.get("file") and not r.get("promo")]
        return [side / n for n in names if (side / n).exists()]
    return sorted(side.glob("*.webp"))


def _linear_match(tgt: np.ndarray, src: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``tgt`` with a per-channel gain / bias fit to ``src`` over ``keep``."""
    ys, xs = np.nonzero(keep)
    pick = np.random.default_rng(0).choice(len(xs), min(len(xs), 200_000), False)
    ys, xs = ys[pick], xs[pick]
    out = np.empty(tgt.shape, np.float32)
    for c in range(3):
        g, b = np.polyfit(tgt[ys, xs, c].astype(np.float32), src[ys, xs, c], 1)
        if not 0.7 <= g <= 1.4:
            g, b = 1.0, float(src[ys, xs, c].mean() - tgt[ys, xs, c].mean())
        out[..., c] = tgt[..., c] * g + b
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _lut_match(tgt: np.ndarray, src: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``tgt`` through a per-channel quantile-matching LUT onto ``src`` over
    ``keep`` — a tone curve, for editions scanned at other levels / gamma."""
    q = np.linspace(0, 100, 257)
    out = np.empty_like(tgt)
    for c in range(3):
        qa = np.percentile(tgt[..., c][keep], q) + np.arange(257) * 1e-6
        qb = np.percentile(src[..., c][keep], q)
        lut = np.interp(np.arange(256), np.maximum.accumulate(qa), qb)
        out[..., c] = np.clip(lut + 0.5, 0, 255).astype(np.uint8)[tgt[..., c]]
    return out


def colour_match(tgt: np.ndarray, src: np.ndarray, keep: np.ndarray):
    """``(name, tgt')``: whichever of the twin as-is, a linear fit and a LUT
    leaves the lowest residual on ``keep``. Most pages need none — on 60 probe
    pages the linear fit alone raised the median residual 2.8 → 3.4 — while
    another scan's tone curve needs the LUT."""
    if keep.sum() < 1000:
        return "id", tgt
    cands = [
        ("id", tgt),
        ("linear", _linear_match(tgt, src, keep)),
        ("lut", _lut_match(tgt, src, keep)),
    ]
    return min(cands, key=lambda c: _absdiff(src, c[1])[keep].mean())


def _absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel mean |a - b| over the channels."""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(-1)


def paste_mask(labels: np.ndarray, chosen, dilate_frac: float) -> np.ndarray:
    """The composite's paste mask for region indices ``chosen`` (uint8 0/1)."""
    mask = np.isin(labels, [k + 1 for k in chosen]).astype(np.uint8)
    d = max(2, round(min(labels.shape) * dilate_frac))
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1,) * 2)
    )
    mask[labels == INVALID] = 0
    return mask


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


def _group_metrics(groups, regions, labels, diff, dilate_frac, band=6, near_frac=0.03):
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


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == size:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _chroma(img: np.ndarray, keep: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1][keep].mean())


def build_work(job) -> str:
    try:
        return _build_work(*job)
    except Exception as e:  # one bad work never takes the pool down
        return f"{job[1]}/{job[0].name}: FAILED {type(e).__name__}: {e}"


def _read_gray(paths: list[Path]) -> tuple[list[Path], list[np.ndarray]]:
    """The readable pages of ``paths`` and their greyscale arrays."""
    kept = [(p, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]
    kept = [(p, g) for p, g in kept if g is not None and g.size]
    return [p for p, _ in kept], [g for _, g in kept]


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
        # ECC and everything after run at the staged size; the twin is warped
        # at native size so both pages go through the same resize
        H, W = src_g[i].shape
        size = target_size(W, H, a.resolution)
        M = refine_ecc(src_g[i], tw_g[j], M, work_px=max(size))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

        def warp(img):
            return _resize(cv2.warpAffine(img, M, (W, H), flags=interp), size)

        valid = cv2.warpAffine(np.full(tw_g[j].shape, 255, np.uint8), M, (W, H))
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
        valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
        regions, labels, changed = diff_regions(
            _resize(src_g[i], size), warp(tw_g[j]), valid
        )
        page["changed_frac"] = round(changed, 4)
        if not regions:
            pages.append({**page, "status": "no_regions"})
            continue

        src = _resize(cv2.imread(str(src_files[i]), cv2.IMREAD_COLOR), size)
        tgt = warp(cv2.imread(str(tw_files[j]), cv2.IMREAD_COLOR))
        labels[valid == 0] = INVALID
        keep = labels == 0
        if abs(_chroma(src, keep) - _chroma(tgt, keep)) > a.max_chroma_gap:
            pages.append({**page, "status": "chroma_mismatch"})
            continue
        page["colour"], tgt = colour_match(tgt, src, keep)
        uncovered = labels == INVALID
        tgt[uncovered] = src[uncovered]
        diff = _absdiff(src, tgt)
        labels = grow_regions(labels, diff)
        groups = _group_metrics(
            region_groups(labels, regions), regions, labels, diff, a.dilate_frac
        )
        page["uncovered"] = round(float(uncovered.mean()), 4)
        tw_h, tw_w = tw_g[j].shape
        page["aspect"] = round((tw_w / tw_h) / (W / H) - 1, 4)
        # outside the components, before growth: grown labels take in most of
        # a rescan's diff
        page["resid"] = round(float(diff[keep].mean()), 2) if keep.any() else 255.0
        stem = src_files[i].stem
        _imwrite(out / f"{stem}.src.webp", src, WEBP)
        _imwrite(out / f"{stem}.tgt.webp", tgt, WEBP)
        _imwrite(out / f"{stem}.lab.png", labels)
        pages.append(
            {
                **page,
                "size": [src.shape[1], src.shape[0]],
                "regions": regions,
                "groups": groups,
                "status": "ok",
            }
        )

    _write_text(out / "pages.json", json.dumps(pages, indent=1, ensure_ascii=False))
    ok = [p for p in pages if p["status"] == "ok"]
    return (
        f"{lang}/{work.name}: ja {len(src_files)} {lang} {len(tw_files)} "
        f"matched {len(pairs)} ok {len(ok)} "
        f"regions {sum(len(p['regions']) for p in ok)}"
    )


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
    with Pool(a.workers, initializer=_single_thread) as pool:
        for k, msg in enumerate(pool.imap_unordered(build_work, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)


# ------------------------------------------------------------------ pages


def ok_pages(a) -> list[dict]:
    """Every built page with regions: ``{key, lang, anchor, page, dir, size,
    regions}``, ``key = <lang>/<anchor>/<page>``."""
    out = []
    for lang in a.langs:
        for pj in sorted((a.root / "work" / lang).glob("*/pages.json")):
            anchor = pj.parent.name
            if a.works and anchor not in a.works:
                continue
            for p in json.loads(pj.read_text(encoding="utf-8")):
                if p["status"] == "ok":
                    out.append(
                        {
                            "key": f"{lang}/{anchor}/{p['page']}",
                            "lang": lang,
                            "anchor": anchor,
                            "page": p["page"],
                            "dir": pj.parent,
                            "size": p["size"],
                            "resid": p["resid"],
                            "uncovered": p["uncovered"],
                            "regions": p["regions"],
                            "groups": p["groups"],
                        }
                    )
    return out


# ------------------------------------------------------------------ ocr


def _ocr_reader(a):
    """``(engine, readable)``: the anime_tools reader with the MOSS caption
    helper's floors (``library.moss.colorize.readable``), plus a Latin line
    read at ``LATIN_MIN_SCORE`` whatever its det — the detector scores
    scanlation lettering 0.3–0.6 and the reader 0.8–0.9 on reads that are
    right."""
    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.animetext import DEFAULT_CONF
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.sfx import SfxReader, is_latin

    from library.moss.colorize import LATIN_MIN_GLYPH, readable

    def keep(ln) -> bool:
        return readable(ln) or (
            is_latin(ln.text)
            and ln.score >= LATIN_MIN_SCORE
            and ln.glyph_px >= LATIN_MIN_GLYPH
        )

    reader = SfxReader.load(device=a.device)
    engine = RereadEngine(
        engine=load_ocr(device=a.device),
        read_boxes=reader.read_boxes_scored,
        resized_dir=Path("/"),
        skip_en=False,
        min_det=DEFAULT_CONF,
    )
    return engine, keep


def _read_into(a, path: Path, todo: list, paths: list[Path], label: str) -> dict:
    """Read ``paths`` into the ``{key: [[text, box], ...]}`` cache at ``path``;
    ``todo`` holds each path's key. Returns the whole cache.

    ``--ocr_procs`` > 1 splits the unread pages over that many processes, each
    with its own reader and its own ``{stem}.{i}of{n}.json``: the VL re-read
    decodes one page's boxes at a time, so one process leaves the GPU idle
    between launches. Shard files left by an interrupted run fold in first."""
    reads = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    shards = sorted(path.parent.glob(f"{path.stem}.*of*.json"))
    for sp in shards:
        reads.update(json.loads(sp.read_text(encoding="utf-8")))
    if shards:
        _write_text(path, json.dumps(reads, ensure_ascii=False))
        for sp in shards:
            sp.unlink()
    keep = [(k, p) for k, p in zip(todo, paths) if k not in reads]
    print(f"{label}: {len(keep)} pages to read ({len(reads)} cached)", flush=True)
    if not keep:
        return reads
    n = max(1, min(a.ocr_procs, len(keep)))
    if n == 1:
        reads.update(_read_shard((a, path, keep, label)))
        _write_text(path, json.dumps(reads, ensure_ascii=False))
        return reads
    import multiprocessing as mp

    jobs = [
        (a, path.with_name(f"{path.stem}.{i}of{n}.json"), keep[i::n], f"{label}[{i}]")
        for i in range(n)
    ]
    with mp.get_context("spawn").Pool(n) as pool:
        for part in pool.imap_unordered(_read_shard, jobs):
            reads.update(part)
    _write_text(path, json.dumps(reads, ensure_ascii=False))
    for _, sp, _, _ in jobs:
        sp.unlink(missing_ok=True)
    return reads


def _read_shard(job) -> dict:
    """Read ``keep`` (``[(key, path)]``), checkpointing to ``out`` every 200
    pages and reporting every 50."""
    a, out, keep, label = job
    engine, readable = _ocr_reader(a)
    reads, t0 = {}, time.time()
    lines_iter = engine.read_iter([p for _, p in keep])
    for k, ((key, _), lines) in enumerate(zip(keep, lines_iter), 1):
        reads[key] = [[ln.text, list(ln.box)] for ln in lines if readable(ln)]
        if k % 200 == 0 or k == len(keep):
            _write_text(out, json.dumps(reads, ensure_ascii=False))
        # the daemon kills a job silent for 120 s; a shard does ~1 page/s
        if k % 50 == 0 or k == len(keep):
            rate = k / (time.time() - t0)
            print(
                f"  {label} {k}/{len(keep)}, {rate:.2f} page/s, "
                f"eta {(len(keep) - k) / rate / 60:.0f} min",
                flush=True,
            )
    return reads


def stage_ocr(a) -> None:
    pages = [
        p
        for p in ok_pages(a)
        if p["resid"] <= a.max_resid and p["uncovered"] <= a.max_uncovered
    ]
    _read_into(
        a,
        a.root / "ocr.json",
        [p["key"] for p in pages],
        [p["dir"] / f"{p['page']}.tgt.webp" for p in pages],
        "ocr",
    )


# ------------------------------------------------------------------ det


def stage_det(a, chunk=32) -> None:
    from concurrent.futures import ThreadPoolExecutor

    path = a.root / "det.json"
    dets = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [
        p
        for p in ok_pages(a)
        if p["resid"] <= a.max_resid
        and p["uncovered"] <= a.max_uncovered
        and p["key"] not in dets
    ]
    print(f"det: {len(todo)} pages to detect ({len(dets)} cached)", flush=True)
    if not todo:
        return
    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device=a.device)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as io:
        for s in range(0, len(todo), chunk):
            part = todo[s : s + chunk]
            files = [
                str(p["dir"] / f"{p['page']}.{side}.webp")
                for p in part
                for side in ("src", "tgt")
            ]
            boxes = det.detect_batch(list(io.map(cv2.imread, files)))
            for k, p in enumerate(part):
                dets[p["key"]] = {
                    side: [[int(v) for v in b] for b in boxes[2 * k + i]]
                    for i, side in enumerate(("src", "tgt"))
                }
            done = s + len(part)
            if done % (chunk * 20) == 0 or done == len(todo):
                _write_text(path, json.dumps(dets))
                rate = done / (time.time() - t0)
                print(
                    f"  det {done}/{len(todo)}, {rate:.1f} page/s, "
                    f"eta {(len(todo) - done) / rate / 60:.0f} min",
                    flush=True,
                )


# ------------------------------------------------------------------ pool


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


def page_metrics(job) -> tuple[str, dict]:
    """``(key, metrics)`` for one built page: per group, inside its paste mask,
    ``nontext`` (median |ΔLab| src vs twin outside the union of both sides'
    text blocks and the twin's OCR lines — the median, so lettering the
    detector missed does not count unless it fills the region), and the colour of the ink the swap replaces against the ink it brings
    (``ink_shift``: their a/b distance; ``pol_src`` / ``pol_tgt``: their
    lightness against the region's background). Page-wide: ``bg_resid``, the
    mean |ΔLab| outside every text block."""
    key, d, page, groups, boxes, dilate_frac, det_pad = job
    src = cv2.imread(str(d / f"{page}.src.webp"), cv2.IMREAD_COLOR)
    tgt = cv2.imread(str(d / f"{page}.tgt.webp"), cv2.IMREAD_COLOR)
    labels = cv2.imread(str(d / f"{page}.lab.png"), cv2.IMREAD_UNCHANGED)
    text = np.zeros(labels.shape, bool)
    for x0, y0, x1, y1 in boxes:
        text[
            max(0, y0 - det_pad) : y1 + det_pad, max(0, x0 - det_pad) : x1 + det_pad
        ] = True
    ls, lt = _lab(src), _lab(tgt)
    diff = np.linalg.norm(ls - lt, axis=-1)
    bg = (labels != INVALID) & ~text
    out = {"bg_resid": round(float(diff[bg].mean()), 2) if bg.any() else 0.0}
    rows = []
    pad = 2 * max(2, round(min(labels.shape) * dilate_frac)) + 2
    for g in groups:
        regions = g["regions"]
        # region boxes are the build's diff boxes; grow_regions may reach past
        # them, so take the crop from the labels themselves
        ys, xs = np.nonzero(np.isin(labels, [k + 1 for k in regions]))
        if not len(xs):
            rows.append({})
            continue
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
        rows.append(row)
    out["groups"] = rows
    return key, out


def _metrics_job(job):
    cv2.setNumThreads(1)
    return page_metrics(job)


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


def stage_pool(a) -> None:
    from anime_tools.ocr.reread import covered

    reads = json.loads((a.root / "ocr.json").read_text(encoding="utf-8"))
    dets = json.loads((a.root / "det.json").read_text(encoding="utf-8"))
    mpath = a.root / "pool_metrics.json"
    metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    n = dict.fromkeys(
        ["unread", "undetected", "resid", "uncovered", "uncaptioned", "bg_resid"], 0
    )
    pages = []
    for p in ok_pages(a):
        if p["resid"] > a.max_resid:
            n["resid"] += 1
        elif p["uncovered"] > a.max_uncovered:
            n["uncovered"] += 1
        elif p["key"] not in reads:
            n["unread"] += 1
        elif p["key"] not in dets:
            n["undetected"] += 1
        else:
            pages.append(p)

    todo = [p for p in pages if p["key"] not in metrics]
    print(f"pool: {len(todo)} pages to measure ({len(metrics)} cached)", flush=True)
    jobs = [
        (
            p["key"],
            p["dir"],
            p["page"],
            p["groups"],
            dets[p["key"]]["src"]
            + dets[p["key"]]["tgt"]
            + [[round(v) for v in b] for _, b in reads[p["key"]]],
            a.dilate_frac,
            a.det_pad,
        )
        for p in todo
    ]
    t0 = time.time()
    with Pool(a.workers, initializer=_single_thread) as pool:
        for k, (key, m) in enumerate(
            pool.imap_unordered(_metrics_job, jobs, chunksize=4), 1
        ):
            metrics[key] = m
            if k % 1000 == 0 or k == len(jobs):
                print(f"  measure {k}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    if todo:
        _write_text(mpath, json.dumps(metrics))

    gate_drops: dict[str, int] = {}
    misread_lines, unread_groups, rows = 0, 0, []
    for p in pages:
        pm = metrics[p["key"]]
        if pm["bg_resid"] > a.max_bg_resid:
            n["bg_resid"] += 1
            continue
        lines = [[t.strip(), b] for t, b in reads[p["key"]]]
        lines = [ln for ln in lines if ln[0]]
        owner = assign_lines(lines, p["regions"], a.pad_frac)
        ja = [bool(JA_SCRIPT.search(t)) for t, _ in lines]
        misread = [k is not None and j for k, j in zip(owner, ja)]
        misread_lines += sum(misread)
        # unpadded: the padded box also reaches JA lettering the twin left alone
        bad = {
            k
            for k, j in zip(assign_lines(lines, p["regions"], 0.0), ja)
            if j and k is not None
        }
        # A twin text block no kept line covers: pasted, it is lettering the
        # caption does not name.
        blocks = [["", b] for b in dets[p["key"]]["tgt"]]
        read_boxes = [tuple(b) for _, b in lines]
        unread = {
            k
            for (_, b), k in zip(blocks, assign_lines(blocks, p["regions"], a.pad_frac))
            if k is not None and not covered(tuple(b), read_boxes)
        }
        group_of = {k: gi for gi, g in enumerate(p["groups"]) for k in g["regions"]}
        has_line = {
            group_of[k] for k, m in zip(owner, misread) if k is not None and not m
        }
        clean = []
        for gi, g in enumerate(p["groups"]):
            if (
                gi not in has_line
                or bad & set(g["regions"])
                or g["seam"] > a.max_seam
                or g["leftover"] > a.max_leftover
            ):
                continue
            if unread & set(g["regions"]):
                unread_groups += 1
                continue
            why = colour_gate(pm["groups"][gi], a)
            if why:
                gate_drops[why] = gate_drops.get(why, 0) + 1
                continue
            clean.append(gi)
        if not clean:
            n["uncaptioned"] += 1
            continue
        rows.append(
            {
                "stem": f"{p['anchor']}_{p['page']}_{p['lang']}",
                "key": p["key"],
                "work": p["anchor"],
                "lang": p["lang"],
                "size": p["size"],
                "groups": [
                    {**g, **pm["groups"][gi], "clean": gi in clean}
                    for gi, g in enumerate(p["groups"])
                ],
                "bg_resid": pm["bg_resid"],
                "lines": [
                    [*ln, group_of[k]]
                    for ln, k, m in zip(lines, owner, misread)
                    if k is not None and not m and group_of[k] in clean
                ],
            }
        )

    pool_dir = a.root / "pool"
    for f in pool_dir.glob("*/*"):
        f.unlink()
    (pool_dir / "pool.jsonl").unlink(missing_ok=True)
    by_key = {p["key"]: p for p in pages}
    for r in rows:
        p = by_key[r["key"]]
        base = pool_dir / r["lang"]
        base.mkdir(parents=True, exist_ok=True)
        for side in ("src", "tgt"):
            (base / f"{r['stem']}.{side}.webp").symlink_to(
                os.path.relpath(p["dir"] / f"{p['page']}.{side}.webp", base)
            )
    _write_text(
        pool_dir / "pool.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    n_groups = sum(len(r["groups"]) for r in rows)
    stats = {
        "pages": len(rows),
        "by_lang": {lang: sum(r["lang"] == lang for r in rows) for lang in a.langs},
        "works": len({r["work"] for r in rows}),
        **{f"dropped_{k}": v for k, v in n.items()},
        "misread_lines": misread_lines,
        "groups_dropped_unread_block": unread_groups,
        "groups_dropped_by_colour_gate": gate_drops,
        "clean_group_frac": round(
            sum(g["clean"] for r in rows for g in r["groups"]) / max(n_groups, 1), 3
        ),
    }
    _write_text(pool_dir / "stats.json", json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


# ------------------------------------------------------------------ assemble


def assign_lines(lines: list, regions: list[dict], pad_frac: float) -> list[int | None]:
    """Per OCR line, the index of the smallest padded region box holding its
    centre, else ``None``."""
    out = []
    for _, (x0, y0, x1, y1) in lines:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best, best_area = None, None
        for k, r in enumerate(regions):
            bx0, by0, bx1, by1 = r["box"]
            p = pad_frac * max(bx1 - bx0, by1 - by0)
            if bx0 - p <= cx <= bx1 + p and by0 - p <= cy <= by1 + p:
                area = (bx1 - bx0) * (by1 - by0)
                if best_area is None or area < best_area:
                    best, best_area = k, area
        out.append(best)
    return out


def caption(lang: str, lines: list, size) -> str:
    from anime_tools.captions.layout_json import centered_box

    layout = {"summary": {"caption": f"comic, {LANG_TAG[lang]}"}}
    if lines:
        layout["items"] = [
            {
                "label": "text",
                "caption": "text",
                "text": text,
                "bbox": centered_box(box, size),
            }
            for text, box in lines
        ]
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def compose_job(job) -> None:
    """Write one pair's composite target and source symlink."""
    p, chosen, root, dilate, feather = job
    stem = f"{p['anchor']}_{p['page']}_{p['lang']}"
    base = root / "staging" / p["lang"]
    base.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(p["dir"] / f"{p['page']}.src.webp"), cv2.IMREAD_COLOR)
    tgt = cv2.imread(str(p["dir"] / f"{p['page']}.tgt.webp"), cv2.IMREAD_COLOR)
    labels = cv2.imread(str(p["dir"] / f"{p['page']}.lab.png"), cv2.IMREAD_UNCHANGED)
    mask = paste_mask(labels, chosen, dilate)
    alpha = mask.astype(np.float32)
    if feather:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather) * mask
    alpha = alpha[..., None]
    out = np.clip(src * (1 - alpha) + tgt * alpha + 0.5, 0, 255).astype(np.uint8)
    _imwrite(base / f"{stem}.tgt.webp", out, WEBP_LOSSLESS)
    (base / f"{stem}.src.webp").unlink(missing_ok=True)
    (base / f"{stem}.src.webp").symlink_to(
        os.path.relpath(p["dir"] / f"{p['page']}.src.webp", base)
    )


def stage_assemble(a) -> None:
    pool_rows = [
        json.loads(ln)
        for ln in (a.root / "pool" / "pool.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    pool_rows = [r for r in pool_rows if r["lang"] in a.langs]
    n_pool = len(pool_rows)
    if a.sample and a.sample < n_pool:
        pool_rows = sorted(
            pool_rows, key=lambda r: _stable_seed(f"{a.seed}:sample:{r['stem']}")
        )[: a.sample]
    plans = []
    clean_of = [
        [gi for gi, g in enumerate(r["groups"]) if g["clean"]] for r in pool_rows
    ]
    n_eligible = sum(len(c) >= 2 for c in clean_of)
    p_partial = min(1.0, a.partial_ratio * len(pool_rows) / max(n_eligible, 1))
    for r, clean in zip(pool_rows, clean_of):
        rng = random.Random(_stable_seed(f"{a.seed}:{r['stem']}"))
        groups = clean
        if len(clean) >= 2 and rng.random() < p_partial:
            groups = sorted(rng.sample(clean, rng.randint(1, len(clean) - 1)))
        chosen = sorted(k for gi in groups for k in r["groups"][gi]["regions"])
        keep = [[t, b] for t, b, gi in r["lines"] if gi in groups]
        plans.append((r, chosen, keep, len(groups), len(clean)))

    staging = a.root / "staging"
    for f in staging.glob("*/*"):
        f.unlink()
    (staging / "pairs.jsonl").unlink(missing_ok=True)
    jobs = [
        (
            {
                "anchor": r["work"],
                "page": r["key"].rsplit("/", 1)[1],
                "lang": r["lang"],
                "dir": a.root / "work" / r["lang"] / r["work"],
            },
            chosen,
            a.root,
            a.dilate_frac,
            a.feather,
        )
        for r, chosen, *_ in plans
    ]
    t0 = time.time()
    with Pool(a.workers, initializer=_single_thread) as pool:
        for k, _ in enumerate(pool.imap_unordered(compose_job, jobs, chunksize=4), 1):
            if k % 500 == 0 or k == len(jobs):
                print(
                    f"  assemble {k}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True
                )

    rows = [
        {
            "stem": r["stem"],
            "key": r["key"],
            "work": r["work"],
            "lang": r["lang"],
            "size": r["size"],
            "mode": "full" if n_swapped == n_clean else "partial",
            "groups": len(r["groups"]),
            "clean_groups": n_clean,
            "swapped_groups": n_swapped,
            "chosen": chosen,
            "lines": keep,
        }
        for r, chosen, keep, n_swapped, n_clean in plans
    ]
    _write_text(
        staging / "plan.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    stats = {
        "pool": n_pool,
        "planned": len(rows),
        "partial_eligible": n_eligible,
    }
    _write_text(staging / "stats.json", json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


# ------------------------------------------------------------------ caption


def stage_caption(a) -> None:
    staging = a.root / "staging"
    plan = [
        json.loads(ln)
        for ln in (staging / "plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    def key(r):
        return f"{r['key']}:{','.join(map(str, r['chosen']))}"

    if a.caption_ocr:
        reads = _read_into(
            a,
            a.root / "ocr_staged.json",
            [key(r) for r in plan],
            [staging / r["lang"] / f"{r['stem']}.tgt.webp" for r in plan],
            "caption ocr",
        )
    rows, n_ja, n_empty = [], 0, 0
    for r in plan:
        base = staging / r["lang"]
        read = reads[key(r)] if a.caption_ocr else r["lines"]
        lines = [[t.replace('"', "”").strip(), b] for t, b in read]
        lines = [ln for ln in lines if ln[0]]
        keep = [ln for ln in lines if not JA_SCRIPT.search(ln[0])]
        n_ja += len(lines) - len(keep)
        if not keep:
            n_empty += 1
            for ext in (".tgt.webp", ".src.webp", ".txt"):
                (base / f"{r['stem']}{ext}").unlink(missing_ok=True)
            continue
        _write_text(base / f"{r['stem']}.txt", caption(r["lang"], keep, r["size"]))
        rows.append(
            {
                "stem": r["stem"],
                "target": f"{r['lang']}/{r['stem']}.tgt.webp",
                "cond": f"{r['lang']}/{r['stem']}.src.webp",
                "caption": f"{r['lang']}/{r['stem']}.txt",
                "work": r["work"],
                "lang": r["lang"],
                "mode": r["mode"],
                "groups": r["groups"],
                "clean_groups": r["clean_groups"],
                "swapped_groups": r["swapped_groups"],
                "lines": len(keep),
            }
        )
    _write_text(
        staging / "pairs.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    stats = json.loads((staging / "stats.json").read_text(encoding="utf-8"))
    stats.update(
        {
            "pairs": len(rows),
            "by_lang": {lang: sum(r["lang"] == lang for r in rows) for lang in a.langs},
            "partial": sum(r["mode"] == "partial" for r in rows),
            "works": len({r["work"] for r in rows}),
            "dropped_unread_target": n_empty,
            "ja_lines_dropped": n_ja,
        }
    )
    _write_text(staging / "stats.json", json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["build", "ocr", "det", "pool", "assemble", "caption"],
        choices=["build", "ocr", "det", "pool", "assemble", "caption"],
    )
    ap.add_argument("--hito", type=Path, default=Path.home() / "hitocrawl/retrieved")
    ap.add_argument("--root", default="post_image_dataset/i2i/swap_text")
    ap.add_argument("--langs", nargs="+", default=["en", "ko"], choices=list(LANG_TAG))
    ap.add_argument("--works", nargs="*", help="anchor ids; default every work")
    ap.add_argument("--limit", type=int, default=0, help="works per language (build)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rebuild built works")
    ap.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="staged size: the MOSS cache's target_size at this resolution, so the "
        "cache encodes the pages as-is; the build diffs at this size too",
    )
    ap.add_argument("--min_inliers", type=int, default=50)
    ap.add_argument(
        "--scale_tol", type=float, default=0.02, help="max |scale / work median - 1|"
    )
    ap.add_argument(
        "--max_chroma_gap",
        type=float,
        default=20.0,
        help="mean HSV saturation gap above which a colour / B&W edition pair drops",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--caption_ocr",
        action="store_true",
        help="caption from a second read of each staged target (GPU) instead of "
        "the twin read's lines in the pasted groups",
    )
    ap.add_argument(
        "--ocr_procs",
        type=int,
        default=2,
        help="OCR processes sharing the GPU (each loads its own reader)",
    )
    ap.add_argument(
        "--partial_ratio",
        type=float,
        default=0.5,
        help="share of pairs that swap a proper subset of their captioned regions",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max_resid",
        type=float,
        default=8.0,
        help="page mean |src − tgt| outside the lettering above which it drops",
    )
    ap.add_argument(
        "--max_uncovered",
        type=float,
        default=0.005,
        help="page share the warped twin leaves uncovered above which it drops",
    )
    ap.add_argument(
        "--max_seam",
        type=float,
        default=8.0,
        help="group seam above which a partial pair never pastes it",
    )
    ap.add_argument(
        "--max_leftover",
        type=float,
        default=0.01,
        help="group leftover above which a partial pair never pastes it",
    )
    ap.add_argument(
        "--det_pad", type=int, default=4, help="text block pad (px) in the pool's mask"
    )
    ap.add_argument(
        "--max_bg_resid",
        type=float,
        default=1e9,
        help="page mean |ΔLab| outside every text block above which it leaves the pool",
    )
    ap.add_argument(
        "--max_nontext",
        type=float,
        default=8.0,
        help="group median |ΔLab| in its paste mask outside the text blocks above "
        "which it is never pasted",
    )
    ap.add_argument(
        "--max_ink_shift",
        type=float,
        default=15.0,
        help="group a/b distance between the replaced and the pasted ink above "
        "which it is never pasted",
    )
    ap.add_argument(
        "--flip_contrast",
        type=float,
        default=40.0,
        help="ink lightness vs background on both sides at which a sign flip "
        "(dark ↔ light lettering) keeps a group from being pasted",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="pool pages the assemble stage stages (0: all)",
    )
    ap.add_argument(
        "--pad_frac",
        type=float,
        default=0.15,
        help="region box pad (× its long side) an OCR line centre may fall in",
    )
    ap.add_argument(
        "--dilate_frac",
        type=float,
        default=0.003,
        help="paste mask dilation / short edge",
    )
    ap.add_argument("--feather", type=float, default=1.5, help="paste edge blur σ (px)")
    a = ap.parse_args()
    a.root = resolve_under_home(a.root)
    a.works = set(a.works or [])
    for stage in a.stages:
        {
            "build": stage_build,
            "ocr": stage_ocr,
            "det": stage_det,
            "pool": stage_pool,
            "assemble": stage_assemble,
            "caption": stage_caption,
        }[stage](a)


if __name__ == "__main__":
    main()
