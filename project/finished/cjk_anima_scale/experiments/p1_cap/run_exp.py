#!/usr/bin/env python
"""p1_cap — hypothesis.md § 4 P1: cold, in-word, capped vs uncapped (2026-09-27)

P0 (a trained row scaled down loses its glyph, gains no composition) and P0b
(at matched norm the seed rows sit 0.25–0.3 above EN's context cos; the raw
pack rows sit on EN's curve) put the context immunity in the direction
identity training gave the rows, not in their norm. Stage B's in-word donor
(F0: 0.95–0.97) started from the seed, so it cannot say whether in-word
training makes a context-free row or just keeps the seed's. Two arms on
Stage B's own items (its data dir, unchanged — same captions, same draws,
same 90 steps / row), both **cold** (the donors' rows start at the pack
rows, Δ 0):

    p1_cold   uncapped — Stage B minus the warm start
    p1_cap    every donor row's effective norm clamped after each step to
              the T5 table's mean row norm (≈ 212, EN's; the pack rows are
              155–208, the seed's 250–320)

The 2 × 2 against the record: Stage B (warm, uncapped) on the donor keys and
the floor (seed) are cached in ``native_spell/``, so no floor renders.

After P1 (cold composes 6 → 12 / 16 but singles official fall 91 → 29; the
cap is nearly redundant — hypothesis.md § 4 P1 read), ``p1_mix``: cold,
uncapped, on Stage B's items **plus the seed's lone singles group** (the
production ``b0709``: scene_single + grid_single, σ 0.7–0.9) at share
``MIX_SHARE`` of the kind's items (1 200 lone items beside Stage B's 2 400).
Its data dir is ``OUT/<MIX_DATA>/data``; the in-word groups restart from the
pools' rng state, so they are Stage B's items (the data leg asserts it).
Steps × (all items / in-word items) = ``MIX_STEPS`` / row, so the in-word
items get p1_cold's exposure and the lone tier is the one change.

After P1b, ``p1_lone`` (retrain_experiments.md § 4 C1): the same 36 donors, cold,
on the production table alone (``builder.TABLE``: the single kind is
``b0709`` only, 2 400 lone items, no in-word item), 90 steps / row. Is the
in-word tier load-bearing, or does a cold start alone compose? Its data dir
is ``OUT/<LONE_DATA>/data``.

Legs:
  data      (CPU) the mix / lone arms' data dirs (the mix checked against
            Stage B's)
  train     (GPU) the arms (``--arms``), cold, from their data dir
  read      (GPU) each arm on the donor keys (``HELD_IN`` spelled + the nine
            donor singles, en), vs the floor and vs Stage B's donor
  adapter   the context read is ``ctx_trigger --probe c4`` (separate job)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale import reads as scoring  # noqa: E402

SB = load_experiment("stage_b")

EXP = OUT / "experiments"
SB_DIR = OUT / SB.NAME  # the Stage B donor: its data dir, its native_spell/ reads
# name → (row_cap, data): "sb" = Stage B's items, "mix" = + the lone b0709
# group, "lone" = the production table alone
ARMS = {
    "p1_cold": (None, "sb"),
    "p1_cap": ("t5", "sb"),
    "p1_mix": (None, "mix"),
    "p1_lone": (None, "lone"),
}
MIX_DATA = "run0927_p1_mix"
LONE_DATA = "run0928_p1_lone"
MIX_SHARE = 0.5  # of the single kind's items: 1 200 beside Stage B's 2 400
MIX_STEPS = 135  # 90 × 3 600 / 2 400: the in-word items keep p1_cold's exposure


def mix_table() -> tuple:
    """Stage B's table + the production lone-singles group at ``MIX_SHARE``."""
    import dataclasses

    from cjk_scale.builder import TABLE

    lone = next(g for g in TABLE if (g.kind, g.band) == ("single", (0.7, 0.9)))
    return (*SB.table(), dataclasses.replace(lone, share=MIX_SHARE))


def check_mix(data: Path) -> dict:
    """The mix's in-word items are Stage B's: same (tier, text, caption,
    shape) per tier, as a multiset (worker order may differ)."""
    import json

    from cjk_scale.builder import tier_of

    def by_group(d: Path) -> dict:
        out: dict = {}
        for ln in (d / "train.jsonl").read_text(encoding="utf-8").splitlines():
            if ln:
                r = json.loads(ln)
                out.setdefault(tier_of(r), []).append(
                    (r["text"], r["caption"], tuple(r["shape"]))
                )
        return out

    mix, sb = by_group(data), by_group(SB_DIR / "data")
    for g, items in sb.items():
        assert sorted(mix.get(g, [])) == sorted(items), f"tier {g} differs"
    return {g: len(v) for g, v in mix.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["train"], choices=["data", "train", "read"]
    )
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    return p.parse_args()


def main():
    args = parse_args()
    from cjk_scale.config import RunConfig

    assert (SB_DIR / "data" / "train.jsonl").exists(), "no Stage B data"

    def rc_of(name: str) -> RunConfig:
        return RunConfig(
            name=name, path=Path(__file__), vocabs=(f"chars:{SB.DONOR}",), read=()
        )

    def data_of(name: str) -> Path:
        kind = ARMS[name][1]
        return {"sb": SB_DIR, "mix": OUT / MIX_DATA, "lone": OUT / LONE_DATA}[
            kind
        ] / "data"

    run_dir = make_run_dir(
        "p1_cap", label=args.label, root=LINE / "experiments" / "p1_cap" / "results"
    )
    metrics: dict = {
        "arms": {
            a: {"row_cap": ARMS[a][0], "items": ARMS[a][1], "data": str(data_of(a))}
            for a in args.arms
        }
    }
    if "data" in args.legs and "lone" in {ARMS[a][1] for a in args.arms}:
        from cjk_scale.builder import build

        build(rc_of(LONE_DATA), workers=args.workers)
        n = sum(1 for _ in (OUT / LONE_DATA / "data" / "train.jsonl").open())
        metrics["lone_items"] = n
        print(f"lone items {n}", flush=True)
    if "data" in args.legs and "mix" in {ARMS[a][1] for a in args.arms}:
        from cjk_scale import recipes
        from cjk_scale.builder import build

        recipes.RECIPES["scene_spelled"] = SB.scene_spelled
        recipes.RECIPES["scene_single_small"] = SB.scene_single_small
        SB.set_words(SB.donor_words())
        build(rc_of(MIX_DATA), workers=args.workers, table=mix_table())
        metrics["mix_items"] = check_mix(OUT / MIX_DATA / "data")
        print(f"mix items by tier {metrics['mix_items']} (in-word = Stage B's)")
    if "train" in args.legs:
        from cjk_scale.train import train

        for name in args.arms:
            row_cap, mix = ARMS[name]
            train(
                rc_of(name),
                data=data_of(name),
                out=EXP / name,
                cold=True,
                row_cap=row_cap,
                steps_per_row=MIX_STEPS if mix == "mix" else None,
            )
    if "read" in args.legs:
        chars = SB.donor_keys()
        fh = scoring.hits(SB.check_floor(chars, "en"), chars, "en")
        print("floor (seed), donor keys:", flush=True)
        metrics["floor"] = scoring.tally(fh)
        sb_reads = SB_DIR / f"native_{SB.TAG}" / "native_reads.json"
        bh = scoring.hits(sb_reads, chars, "en")
        print("stage_b (warm, uncapped), donor keys:", flush=True)
        metrics["stage_b"] = scoring.tally(bh)
        reads = metrics.setdefault("reads", {})
        for name in args.arms:
            print(f"{name}:", flush=True)
            h = scoring.hits(
                SB.native_read(EXP / name, EXP / name / "data", chars, "en"),
                chars,
                "en",
            )
            reads[name] = scoring.tally(h)
            reads[name]["paired_vs_floor"] = scoring.paired(h, fh)
            reads[name]["paired_vs_stage_b"] = scoring.paired(h, bh)
            print(f"  paired vs floor {reads[name]['paired_vs_floor']}", flush=True)
            print(f"  paired vs stage_b {reads[name]['paired_vs_stage_b']}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(EXP / n) for n in args.arms],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
