"""**pool** (CPU): drops pages past ``--max_resid`` (another scan, not
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
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from i2i_tasks.swap_text.common import (
    JA_SCRIPT,
    ok_pages,
    read_cache,
    read_json,
    worker_pool,
    write_jsonl,
    write_stats,
    write_text,
)
from i2i_tasks.swap_text.metrics import colour_gate, page_metrics

PAGE_DROPS = ("unread", "undetected", "resid", "uncovered", "uncaptioned", "bg_resid")


@dataclass
class _Tally:
    pages: dict[str, int] = field(default_factory=lambda: dict.fromkeys(PAGE_DROPS, 0))
    misread_lines: int = 0
    unread_groups: int = 0
    colour_gates: dict[str, int] = field(default_factory=dict)


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


def _measurable(a, reads: dict, dets: dict, tally: _Tally) -> list[dict]:
    """The built pages within the page gates that both reads reached."""
    pages = []
    for p in ok_pages(a):
        if p["resid"] > a.max_resid:
            tally.pages["resid"] += 1
        elif p["uncovered"] > a.max_uncovered:
            tally.pages["uncovered"] += 1
        elif p["key"] not in reads:
            tally.pages["unread"] += 1
        elif p["key"] not in dets:
            tally.pages["undetected"] += 1
        else:
            pages.append(p)
    return pages


def _measure(a, pages: list[dict], reads: dict, dets: dict) -> dict:
    """``page_metrics`` of ``pages``, through ``<root>/pool_metrics.json``."""
    mpath = a.root / "pool_metrics.json"
    metrics = read_cache(mpath)
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
    with worker_pool(a.workers) as pool:
        for k, (key, m) in enumerate(
            pool.imap_unordered(page_metrics, jobs, chunksize=4), 1
        ):
            metrics[key] = m
            if k % 1000 == 0 or k == len(jobs):
                print(f"  measure {k}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    if todo:
        write_text(mpath, json.dumps(metrics))
    return metrics


def _unread_regions(lines: list, tgt_blocks: list, regions: list, pad_frac) -> set:
    """Regions holding a twin text block no kept line covers: pasted, it is
    lettering the caption does not name."""
    from anime_tools.ocr.reread import covered

    blocks = [["", b] for b in tgt_blocks]
    read_boxes = [tuple(b) for _, b in lines]
    return {
        k
        for (_, b), k in zip(blocks, assign_lines(blocks, regions, pad_frac))
        if k is not None and not covered(tuple(b), read_boxes)
    }


def _is_clean(gi, g, metrics, has_line, bad, unread, a, tally: _Tally) -> bool:
    regions = set(g["regions"])
    if (
        gi not in has_line
        or bad & regions
        or g["seam"] > a.max_seam
        or g["leftover"] > a.max_leftover
    ):
        return False
    if unread & regions:
        tally.unread_groups += 1
        return False
    why = colour_gate(metrics, a)
    if why:
        tally.colour_gates[why] = tally.colour_gates.get(why, 0) + 1
        return False
    return True


def _pool_row(p, read, tgt_blocks, pm, a, tally: _Tally) -> dict | None:
    """The ``pool.jsonl`` row of one measured page, None past the gates."""
    if pm["bg_resid"] > a.max_bg_resid:
        tally.pages["bg_resid"] += 1
        return None
    lines = [[t.strip(), b] for t, b in read]
    lines = [ln for ln in lines if ln[0]]
    owner = assign_lines(lines, p["regions"], a.pad_frac)
    ja = [bool(JA_SCRIPT.search(t)) for t, _ in lines]
    misread = [k is not None and j for k, j in zip(owner, ja)]
    tally.misread_lines += sum(misread)
    # unpadded: the padded box also reaches JA lettering the twin left alone
    bad = {
        k
        for k, j in zip(assign_lines(lines, p["regions"], 0.0), ja)
        if j and k is not None
    }
    unread = _unread_regions(lines, tgt_blocks, p["regions"], a.pad_frac)
    group_of = {k: gi for gi, g in enumerate(p["groups"]) for k in g["regions"]}
    has_line = {group_of[k] for k, m in zip(owner, misread) if k is not None and not m}
    clean = [
        gi
        for gi, g in enumerate(p["groups"])
        if _is_clean(gi, g, pm["groups"][gi], has_line, bad, unread, a, tally)
    ]
    if not clean:
        tally.pages["uncaptioned"] += 1
        return None
    return {
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


def _write_tree(pool_dir, rows: list[dict], pages: list[dict]) -> None:
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
    write_jsonl(pool_dir / "pool.jsonl", rows)


def stage_pool(a) -> None:
    reads = read_json(a.root / "ocr.json")
    dets = read_json(a.root / "det.json")
    tally = _Tally()
    pages = _measurable(a, reads, dets, tally)
    metrics = _measure(a, pages, reads, dets)
    rows = [
        r
        for p in pages
        if (
            r := _pool_row(
                p, reads[p["key"]], dets[p["key"]]["tgt"], metrics[p["key"]], a, tally
            )
        )
    ]
    pool_dir = a.root / "pool"
    _write_tree(pool_dir, rows, pages)
    n_groups = sum(len(r["groups"]) for r in rows)
    stats = {
        "pages": len(rows),
        "by_lang": {lang: sum(r["lang"] == lang for r in rows) for lang in a.langs},
        "works": len({r["work"] for r in rows}),
        **{f"dropped_{k}": v for k, v in tally.pages.items()},
        "misread_lines": tally.misread_lines,
        "groups_dropped_unread_block": tally.unread_groups,
        "groups_dropped_by_colour_gate": tally.colour_gates,
        "clean_group_frac": round(
            sum(g["clean"] for r in rows for g in r["groups"]) / max(n_groups, 1), 3
        ),
    }
    write_stats(pool_dir / "stats.json", stats)
