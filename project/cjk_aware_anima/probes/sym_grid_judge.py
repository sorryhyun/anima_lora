"""plan_zh2 U5 gate readout: does the symbol prompt render the described attribute?

Scores every rendered cell of one or more ``bench/cjk_adapter/run_bench.py``
run dirs (``<ts>_<seed>__<pid>_<arm>.png``) with the Anima Tagger's dbv4
backbone (``animetimm/caformer_b36.dbv4-full``, 12,477 Danbooru tags — it has
``^^^`` 55k / ``^_^`` 119k / ``:<`` 43k / ``>_<`` / ``\\||/`` / ``\\m/`` /
``heart``; ``star (symbol)`` is *not* in the card, so the ``☆`` block is
judged by eye) and reports, per symbol block and per arm, the mean probability
of the block's judge tag and the hit rate at the card's own best threshold.

Prompt ids carry their block as the prefix before the first ``_`` (``s1_…`` =
``^^^``); the id → judge-tag map lives in the prompt set's ``_judge`` entry.

Usage (GPU, small — a few hundred 1024² cells)::

    make daemon-run ARGS="project/cjk_aware_anima/probes/sym_grid_judge.py \
        --prompts project/cjk_aware_anima/assets/sym_eval_prompts.json \
        --runs bench/cjk_adapter/results/<base-run> bench/cjk_adapter/results/<u5-run> \
        --labels base u5"
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CELL_RE = re.compile(
    r"^(?P<ts>[\d-]+)_(?P<seed>\d+)__(?P<pid>.+?)_(?P<arm>en|[a-z]{2}_[a-z0-9_]+)\.png$"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None, help="one per --runs")
    ap.add_argument(
        "--out", type=Path, default=None, help="json dump (default: first run dir)"
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu to run beside a live GPU job")
    opts = ap.parse_args()
    labels = opts.labels or [r.name for r in opts.runs]
    if len(labels) != len(opts.runs):
        raise SystemExit("--labels must match --runs")

    prompts = json.loads(opts.prompts.read_text(encoding="utf-8"))
    judge: dict[str, str | None] = prompts[
        "_judge"
    ]  # block prefix → dbv4 tag (or null)

    from PIL import Image  # noqa: PLC0415
    from anime_tools.tagger.dbv4_backend import Dbv4Backend  # noqa: PLC0415

    backend = Dbv4Backend(device=opts.device)
    card = backend.card
    col = {r["name"]: j for j, r in enumerate(card.rows)}
    thr = card.best_thresholds()
    for blk, tag in judge.items():
        if tag is not None and tag not in col:
            raise SystemExit(f"judge tag {tag!r} for block {blk} not in dbv4 card")

    cells: list[
        tuple[str, str, str, str, int, Path]
    ] = []  # label, block, pid, arm, seed
    for label, run in zip(labels, opts.runs):
        for png in sorted(run.glob("*.png")):
            m = CELL_RE.match(png.name)
            if not m or m["pid"] not in prompts:
                continue
            blk = m["pid"].split("_", 1)[0]
            cells.append((label, blk, m["pid"], m["arm"], int(m["seed"]), png))
    if not cells:
        raise SystemExit("no cells matched")
    print(f"{len(cells)} cells over {len(opts.runs)} runs")

    probs: dict[Path, float | None] = {}
    todo = [c for c in cells if judge.get(c[1]) is not None]
    for i in range(0, len(todo), opts.batch):
        chunk = todo[i : i + opts.batch]
        out = backend.forward([Image.open(c[5]).convert("RGB") for c in chunk])
        for c, p in zip(chunk, out.probs):
            probs[c[5]] = float(p[col[judge[c[1]]]])

    # (label, block, arm) → [prob]; "other" = the judge tag's other blocks as a
    # specificity check (a symbol that fires every expression tag is not read).
    agg: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    hits: dict[tuple[str, str, str], list[int]] = collections.defaultdict(list)
    for label, blk, pid, arm, seed, png in todo:
        p = probs[png]
        agg[(label, blk, arm)].append(p)
        hits[(label, blk, arm)].append(int(p >= float(thr[col[judge[blk]]])))

    rows = []
    print(
        f"\n{'block':6s} {'judge':8s} {'arm':10s} "
        + " ".join(f"{lb:>14s}" for lb in labels)
    )
    for blk, tag in judge.items():
        if tag is None:
            print(f"{blk:6s} {'(eye)':8s}")
            continue
        arms = sorted({a for (_, b, a) in agg if b == blk})
        for arm in arms:
            line = f"{blk:6s} {tag:8s} {arm:10s} "
            for lb in labels:
                v, h = agg.get((lb, blk, arm), []), hits.get((lb, blk, arm), [])
                if v:
                    line += (
                        f" {sum(v) / len(v):6.3f}/{sum(h) / len(h):4.2f}@{len(v):<2d}"
                    )
                    rows.append(
                        {
                            "label": lb,
                            "block": blk,
                            "tag": tag,
                            "arm": arm,
                            "n": len(v),
                            "mean_prob": sum(v) / len(v),
                            "hit_rate": sum(h) / len(h),
                            "threshold": float(thr[col[tag]]),
                        }
                    )
                else:
                    line += f" {'-':>14s}"
            print(line)
    print("\n(mean prob / hit rate at card best-threshold @ n cells)")

    out = opts.out or (opts.runs[0] / "sym_grid_judge.json")
    out.write_text(
        json.dumps(
            {
                "runs": {lb: str(r) for lb, r in zip(labels, opts.runs)},
                "judge": judge,
                "rows": rows,
                "cells": [
                    {
                        "label": lb,
                        "block": b,
                        "pid": pid,
                        "arm": arm,
                        "seed": s,
                        "png": str(png),
                        "prob": probs.get(png),
                    }
                    for lb, b, pid, arm, s, png in cells
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"→ {out}")


if __name__ == "__main__":
    main()
