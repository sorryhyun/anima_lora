"""**ocr** and **det** (GPU): the two reads of the built pages within
``--max_resid`` / ``--max_uncovered``.

**ocr** reads every twin ``.tgt.webp`` through the anime_tools reader into
``<root>/ocr.json``, every line the engine answers as ``[text, box, det, score,
glyph_px]``, unfiltered, so a line floor changes without a GPU. ``kept_lines``
applies the floors when the reads are used: the MOSS caption helper's
(``library.moss.colorize.readable``), a Latin line at reader score
``LATIN_MIN_SCORE`` whatever its det, and the Latin glyph floor on a Hangul
line. After a floor change remove ``<root>/pool_metrics.json``: its masks hold
the kept lines' boxes.

**det** runs AnimeText text blocks on both sides of the same pages
(``.src.webp`` and ``.tgt.webp``) into ``<root>/det.json``: the text mask the
pool's colour gates measure around, independent of the build's diff regions
(which hold whatever changed, text or not).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

from i2i_tasks.swap_text.common import (
    JA_SCRIPT,
    ok_pages,
    read_cache,
    read_json,
    write_text,
)

LATIN_MIN_SCORE = 0.8
HANGUL = re.compile(r"[\uac00-\ud7a3]")


def _gated_pages(a) -> list[dict]:
    return [
        p
        for p in ok_pages(a)
        if p["resid"] <= a.max_resid and p["uncovered"] <= a.max_uncovered
    ]


def keep_line(ln) -> bool:
    """A line ``readable`` keeps, a Latin line read at ``LATIN_MIN_SCORE``
    whatever its det — the detector scores scanlation lettering 0.3–0.6 and the
    reader 0.8–0.9 on reads that are right — or a Hangul line ``readable``
    lost to its glyph floor alone: small balloon lettering sits at 12–16 px and
    reads at 0.96–1.00."""
    from anime_tools.ocr.reread import LINE_MIN_DET
    from anime_tools.ocr.sfx import is_latin

    from library.moss.colorize import LATIN_MIN_GLYPH, LOW_DET_SCORE, readable

    if readable(ln):
        return True
    if ln.glyph_px < LATIN_MIN_GLYPH:
        return False
    if is_latin(ln.text):
        return ln.score >= LATIN_MIN_SCORE
    return (
        bool(HANGUL.search(ln.text))
        and not JA_SCRIPT.search(ln.text)
        and (ln.det >= LINE_MIN_DET or ln.score >= LOW_DET_SCORE)
    )


def raw_row(ln) -> list:
    """One engine line as stored in ``ocr.json``."""
    return [
        ln.text,
        list(ln.box),
        float(ln.det),
        float(ln.score),
        float(ln.glyph_px),
    ]


def kept_lines(rows: list) -> list:
    """The ``[text, box]`` of the stored rows ``keep_line`` keeps."""
    lines = (
        SimpleNamespace(text=text, box=box, det=det, score=score, glyph_px=glyph_px)
        for text, box, det, score, glyph_px in rows
    )
    return [[ln.text, ln.box] for ln in lines if keep_line(ln)]


def read_into(a, path: Path, todo: list, paths: list[Path], label: str) -> dict:
    """Read ``paths`` into the ``{key: [raw_row, ...]}`` cache at ``path``;
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
    from library.moss.colorize import load_reread_engine

    engine = load_reread_engine(a.device)
    reads, t0 = {}, time.time()
    lines_iter = engine.read_iter([p for _, p in keep])
    for k, ((key, _), lines) in enumerate(zip(keep, lines_iter), 1):
        reads[key] = [raw_row(ln) for ln in lines]
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
