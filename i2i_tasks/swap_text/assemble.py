"""**assemble** (CPU): samples ``--sample`` pool pages (0: all) and builds
each target: the JA page with groups pasted from the twin — lettering the
reader cannot read, that a paste would tear or recolour, stays JA:

* **full** — every clean group pasted.
* **partial** (``--partial_ratio`` of the pairs) — a random proper subset
  of the clean groups.

Writes ``<root>/staging/<lang>/{stem}`` (``.tgt.webp``, ``.src.webp``
symlink), ``plan.jsonl`` and ``stats.json``; reruns rewrite the tree.
"""

from __future__ import annotations

import os
import random
import time

import cv2
import numpy as np

from i2i_tasks.swap_text.common import (
    WEBP_LOSSLESS,
    imwrite,
    read_jsonl,
    stable_seed,
    worker_pool,
    write_jsonl,
    write_stats,
)
from i2i_tasks.swap_text.regions import paste_mask


def compose_job(job) -> None:
    """Write one pair's composite target and source symlink."""
    d, page, base, stem, chosen, dilate, feather = job
    base.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(d / f"{page}.src.webp"), cv2.IMREAD_COLOR)
    tgt = cv2.imread(str(d / f"{page}.tgt.webp"), cv2.IMREAD_COLOR)
    labels = cv2.imread(str(d / f"{page}.lab.png"), cv2.IMREAD_UNCHANGED)
    mask = paste_mask(labels, chosen, dilate)
    alpha = mask.astype(np.float32)
    if feather:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather) * mask
    alpha = alpha[..., None]
    out = np.clip(src * (1 - alpha) + tgt * alpha + 0.5, 0, 255).astype(np.uint8)
    imwrite(base / f"{stem}.tgt.webp", out, WEBP_LOSSLESS)
    (base / f"{stem}.src.webp").unlink(missing_ok=True)
    (base / f"{stem}.src.webp").symlink_to(
        os.path.relpath(d / f"{page}.src.webp", base)
    )


def _plan(pool_rows: list[dict], a) -> tuple[list[tuple], int]:
    """``(plans, n_eligible)``: per pool row ``(row, chosen regions, kept lines,
    swapped groups, clean groups)``, and how many rows could go partial."""
    clean_of = [
        [gi for gi, g in enumerate(r["groups"]) if g["clean"]] for r in pool_rows
    ]
    n_eligible = sum(len(c) >= 2 for c in clean_of)
    p_partial = min(1.0, a.partial_ratio * len(pool_rows) / max(n_eligible, 1))
    plans = []
    for r, clean in zip(pool_rows, clean_of):
        rng = random.Random(stable_seed(f"{a.seed}:{r['stem']}"))
        groups = clean
        if len(clean) >= 2 and rng.random() < p_partial:
            groups = sorted(rng.sample(clean, rng.randint(1, len(clean) - 1)))
        chosen = sorted(k for gi in groups for k in r["groups"][gi]["regions"])
        keep = [[t, b] for t, b, gi in r["lines"] if gi in groups]
        plans.append((r, chosen, keep, len(groups), len(clean)))
    return plans, n_eligible


def _compose(plans: list[tuple], a) -> None:
    staging = a.root / "staging"
    for f in staging.glob("*/*"):
        f.unlink()
    (staging / "pairs.jsonl").unlink(missing_ok=True)
    jobs = [
        (
            a.root / "work" / r["lang"] / r["work"],
            r["key"].rsplit("/", 1)[1],
            staging / r["lang"],
            r["stem"],
            chosen,
            a.dilate_frac,
            a.feather,
        )
        for r, chosen, *_ in plans
    ]
    t0 = time.time()
    with worker_pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(compose_job, jobs, chunksize=4), 1):
            if k % 500 == 0 or k == len(jobs):
                print(
                    f"  assemble {k}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True
                )


def stage_assemble(a) -> None:
    pool_rows = read_jsonl(a.root / "pool" / "pool.jsonl")
    pool_rows = [r for r in pool_rows if r["lang"] in a.langs]
    n_pool = len(pool_rows)
    if a.sample and a.sample < n_pool:
        pool_rows = sorted(
            pool_rows, key=lambda r: stable_seed(f"{a.seed}:sample:{r['stem']}")
        )[: a.sample]
    plans, n_eligible = _plan(pool_rows, a)
    _compose(plans, a)
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
    staging = a.root / "staging"
    write_jsonl(staging / "plan.jsonl", rows)
    stats = {
        "pool": n_pool,
        "planned": len(rows),
        "partial_eligible": n_eligible,
    }
    write_stats(staging / "stats.json", stats)
