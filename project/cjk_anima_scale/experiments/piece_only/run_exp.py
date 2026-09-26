#!/usr/bin/env python
"""piece_only — run0925_300f's 300 pieces on scene_piece items only (reports/next_2026_09_25.md § 4a (2))

run0925_300f bought 2-glyph piece identity and the native trigger
(reports/piece_2026_09_25.md) and failed on doubling, sentence assembly and
3+-glyph pieces, on a mix where a piece row's draws were mostly a fragment
of a ``scene_sentence`` / ``scene_short`` line or a ``grid_string`` cell.
This arm keeps everything but the mix: the same 300 vocabs, the singles
frozen at the seed, the same volume (``ITEMS_PER_VOCAB``), the same two
piece band groups — each cut to its ``scene_piece`` tier (b0507 ≈ 35–48 px
at 0.5–0.7, b0305 12–24 px small bubbles at 0.3–0.5) — the fixed trainer
and the line's rulers.

Read on the piece ruler first (a trained piece alone in a native scene),
then ``word`` contained vs exact (does doubling fall — contained rising
without exact falling), then ``sent`` / ``target``; against run0925_300f on
the same rulers — the parity run's eval (``experiments/parity_300f``,
``--legs eval``) is run0925_300f's rows under this eval.

legs: data (CPU render, ``--workers``), train (GPU), eval (GPU) — submit
the GPU legs through ``make daemon-run``. ``--dry_run`` prints the table and
the group sizes without rendering.

The table stays an experiment's: ``builder.TABLE`` changes only if this
arm's read says so, with its report (plan.md § 2).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap  # noqa: E402

bootstrap()

from bench._common import make_run_dir, write_result  # noqa: E402

SOURCE_RUN = "run0925_300f"
NAME = "run0926_300f_sp"
KEEP = "scene_piece"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "train", "eval"],
    )
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument(
        "--comparator",
        default=str(OUT / "parity_300f_gpu1" / "reads.json"),
        help="run0925_300f's reads under this eval (the parity run's)",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def arm_rc():
    from cjk_scale.config import load_run

    rc = load_run(SOURCE_RUN)
    return dataclasses.replace(rc, name=NAME)


def piece_only_table() -> tuple:
    """``builder.TABLE`` with every piece group cut to its ``scene_piece``
    tier; the group bands, shares and the single group unchanged."""
    from cjk_scale.builder import TABLE

    out = []
    for g in TABLE:
        if g.kind == "piece":
            tiers = tuple(t for t in g.tiers if t.recipe == KEEP)
            assert tiers, f"{g.name} has no {KEEP} tier"
            g = dataclasses.replace(g, tiers=tiers)
        out.append(g)
    return tuple(out)


def describe(table: tuple, n_pieces: int) -> dict:
    from cjk_scale.builder import plan_groups

    groups = plan_groups({"single": [], "piece": ["p"] * n_pieces, "multi": []}, table)
    d = {
        g.name: {
            "band": list(g.band),
            "n_items": n,
            "tiers": [{"recipe": t.recipe, **t.params} for t in g.tiers],
        }
        for g, n in groups
    }
    for name, v in d.items():
        print(
            f"{name} σ {v['band']}: {v['n_items']} items — "
            + "; ".join(json.dumps(t, ensure_ascii=False) for t in v["tiers"]),
            flush=True,
        )
    return d


def totals(reads: dict) -> dict:
    """ruler → group → arm → {n, official, loose, contained} (reads.json)."""
    return {r: b["totals"] for r, b in reads.get("rulers", {}).items()}


def main():
    args = parse_args()
    rc = arm_rc()
    table = piece_only_table()
    from data.vocabs import parse_vocabs

    n_pieces = len({v for s in parse_vocabs(rc.vocab_specs()) for v in s.vocabs})
    print(f"{NAME}: {rc.vocabs} ({n_pieces} vocabs), read {list(rc.read)}", flush=True)
    metrics: dict = {"table": describe(table, n_pieces)}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "piece_only",
        label=args.label,
        root=LINE / "experiments" / "piece_only" / "results",
    )
    if "data" in args.legs:
        from cjk_scale.builder import build

        build(rc, workers=args.workers, table=table)
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc)
    if "eval" in args.legs:
        from cjk_scale.eval import run

        out = run(rc)
        metrics["arm"] = totals(json.loads((out / "reads.json").read_text("utf-8")))
        comp = Path(args.comparator)
        if comp.exists():
            metrics["comparator"] = totals(json.loads(comp.read_text("utf-8")))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / NAME)],
        extra={"source_run": SOURCE_RUN, "keep": KEEP},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
