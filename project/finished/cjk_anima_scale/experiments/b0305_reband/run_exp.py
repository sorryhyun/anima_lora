#!/usr/bin/env python
"""b0305_reband — the seed's own b0305 kana items, trained warm at σ 0.75–0.93

findings.md § 8: the seed's b0305 strings die when the conditional is
dropped at 0.8, and § 4 / § 6–7 put glyph identity at 0.9–0.7 on the
trajectory. This arm keeps the b0305 data (``retrain_kana``'s
``scene_window`` items: 12–24 px kana dialogue windows, 5 800 items, routed)
and moves only the band: every item at ``--band`` (default 0.75–0.93),
warm from the seed rows (μ ``--mu``, default 0.02), the trainer's lr
(1e-3 cosine), ``--steps_per_row`` × the 174 kana rows (default 23 → 4 002
steps), every other row frozen at the seed.

Legs:
- ``data`` (CPU) → ``OUT/run1001_b0305_reband/data[_<tag>]``: the items'
  records with ``band`` replaced (images stay in ``retrain_kana/data/img``;
  latents and the TE cache are built here on first train);
- ``train`` (GPU) → ``OUT/experiments/b0305_reband_warm[_<tag>]``;
- ``read`` (GPU): the ``sent`` ruler vs the seed's routed floor —
  ``garble_replace``'s ``read`` leg on this arm.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/b0305_reband/run_exp.py \\
      --label r0 --legs data train read"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.builder import tier_of  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "b0305_reband"
SRC_RUN = "retrain_kana"
SRC = OUT / SRC_RUN / "data"
TIER = "bubbleN_18"  # the records of record: group b0305, recipe scene_window
BAND = (0.75, 0.93)
MU = 0.02
STEPS_PER_ROW = 23  # × 174 kana rows ≈ 4 000 steps


def data(dst: Path, band: tuple) -> dict:
    recs = [
        json.loads(ln) for ln in (SRC / "train.jsonl").read_text("utf-8").splitlines()
    ]
    recs = [r for r in recs if tier_of(r) == TIER]
    old = Counter(tuple(r["band"]) for r in recs)
    for r in recs:
        r["band"] = list(band)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
    )
    for f in ("vocabs.json", "eval.json"):
        shutil.copy2(SRC / f, dst / f)
    bj = json.loads((SRC / "build.json").read_text("utf-8"))
    (dst / "build.json").write_text(
        json.dumps(
            bj | {"band": list(band), "reband_from": str(SRC), "tier": TIER},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return {
        "from": str(SRC),
        "items": len(recs),
        "tiers": dict(Counter(tier_of(r) for r in recs)),
        "old_bands": {f"{a}-{b}": n for (a, b), n in old.items()},
        "band": list(band),
        "glyph_route": bj.get("glyph_route"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "read"]
    )
    p.add_argument("--band", type=float, nargs=2, default=list(BAND))
    p.add_argument("--mu", type=float, default=MU)
    p.add_argument("--steps_per_row", type=int, default=STEPS_PER_ROW)
    p.add_argument("--tag", default="", help="suffix for the data dir and the arm")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    dst = OUT / f"run1001_{NAME}" / f"data{sfx}"
    name = f"{NAME}_warm{sfx}"

    from cjk_scale.config import load_run

    rc = dataclasses.replace(load_run(SRC_RUN), name=name)
    vocabs = json.loads((SRC / "vocabs.json").read_text("utf-8"))
    n_items = sum(
        tier_of(json.loads(ln)) == TIER
        for ln in (SRC / "train.jsonl").read_text("utf-8").splitlines()
    )
    steps = args.steps_per_row * len(vocabs)
    print(
        f"{name}: {n_items} {TIER} items → band {args.band}, {len(vocabs)} rows × "
        f"{args.steps_per_row} = {steps} steps (≈ {steps * 4 / n_items:.1f} epochs at "
        f"batch 4), μ {args.mu}, warm from {SEED_ROWS}",
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {
        "src": str(SRC),
        "band": list(args.band),
        "mu": args.mu,
        "steps_per_row": args.steps_per_row,
        "steps": steps,
        "rows": len(vocabs),
    }
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = data(dst, tuple(args.band))
        print(json.dumps(metrics["data"], ensure_ascii=False), flush=True)
    if "train" in args.legs:
        from cjk_scale import train as T

        T.INIT_ANCHOR = args.mu
        T.train(
            rc,
            data=dst,
            out=OUT / "experiments" / name,
            cold=False,
            steps_per_row=args.steps_per_row,
            context=SEED_ROWS,
        )
    if "read" in args.legs:
        GR = load_experiment("garble_replace")
        GR.DATA = dst
        metrics["read"] = GR.read("warm", name)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(dst), str(OUT / "experiments" / name)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
