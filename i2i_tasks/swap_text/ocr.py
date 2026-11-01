"""**ocr** and **det** (GPU): the two reads of the built pages within
``--max_resid`` / ``--max_uncovered``.

**ocr** reads every twin ``.tgt.webp`` through the anime_tools reader, with the
MOSS caption helper's floors (``library.moss.colorize.readable``) and a Latin
line at reader score ``LATIN_MIN_SCORE`` whatever its det, into
``<root>/ocr.json``.

**det** runs AnimeText text blocks on both sides of the same pages
(``.src.webp`` and ``.tgt.webp``) into ``<root>/det.json``: the text mask the
pool's colour gates measure around, independent of the build's diff regions
(which hold whatever changed, text or not).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from i2i_tasks.swap_text.common import ok_pages, read_cache, read_json, write_text

LATIN_MIN_SCORE = 0.8


def _gated_pages(a) -> list[dict]:
    return [
        p
        for p in ok_pages(a)
        if p["resid"] <= a.max_resid and p["uncovered"] <= a.max_uncovered
    ]


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


def read_into(a, path: Path, todo: list, paths: list[Path], label: str) -> dict:
    """Read ``paths`` into the ``{key: [[text, box], ...]}`` cache at ``path``;
    ``todo`` holds each path's key. Returns the whole cache.

    ``--ocr_procs`` > 1 splits the unread pages over that many processes, each
    with its own reader and its own ``{stem}.{i}of{n}.json``: the VL re-read
    decodes one page's boxes at a time, so one process leaves the GPU idle
    between launches. Shard files left by an interrupted run fold in first."""
    reads = read_cache(path)
    shards = sorted(path.parent.glob(f"{path.stem}.*of*.json"))
    for sp in shards:
        reads.update(read_json(sp))
    if shards:
        write_text(path, json.dumps(reads, ensure_ascii=False))
        for sp in shards:
            sp.unlink()
    keep = [(k, p) for k, p in zip(todo, paths) if k not in reads]
    print(f"{label}: {len(keep)} pages to read ({len(reads)} cached)", flush=True)
    if not keep:
        return reads
    n = max(1, min(a.ocr_procs, len(keep)))
    if n == 1:
        reads.update(_read_shard((a, path, keep, label)))
        write_text(path, json.dumps(reads, ensure_ascii=False))
        return reads
    import multiprocessing as mp

    jobs = [
        (a, path.with_name(f"{path.stem}.{i}of{n}.json"), keep[i::n], f"{label}[{i}]")
        for i in range(n)
    ]
    with mp.get_context("spawn").Pool(n) as pool:
        for part in pool.imap_unordered(_read_shard, jobs):
            reads.update(part)
    write_text(path, json.dumps(reads, ensure_ascii=False))
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
            write_text(out, json.dumps(reads, ensure_ascii=False))
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
    pages = _gated_pages(a)
    read_into(
        a,
        a.root / "ocr.json",
        [p["key"] for p in pages],
        [p["dir"] / f"{p['page']}.tgt.webp" for p in pages],
        "ocr",
    )


def stage_det(a, chunk=32) -> None:
    from concurrent.futures import ThreadPoolExecutor

    path = a.root / "det.json"
    dets = read_cache(path)
    todo = [p for p in _gated_pages(a) if p["key"] not in dets]
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
                write_text(path, json.dumps(dets))
                rate = done / (time.time() - t0)
                print(
                    f"  det {done}/{len(todo)}, {rate:.1f} page/s, "
                    f"eta {(len(todo) - done) / rate / 60:.0f} min",
                    flush=True,
                )
