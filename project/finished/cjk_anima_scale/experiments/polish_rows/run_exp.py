#!/usr/bin/env python
"""polish_rows — plan_polish.md steps 2–4: ``retrain_kana``'s rows polished on the t4k pool (2026-09-28)

Data: ``retrain_kana``'s kana composited into ``OUT/scenes_t4k`` (the
1024-tier EN-anchor scenes of ``experiments/polish``), each at its scene's
own size. The builder runs as the rule, with four patches in-process (user,
2026-09-28):

- **Scene tiers only.** One group of ``retrain_kana``'s two scene recipes,
  weighted as its scene items were: lone ``scene_single`` (2 900 + 1 740 of
  14 500 → 0.32) and in-word ``scene_window`` (0.68, routed windows, the read
  words held out by trigram). ``grid_single`` is dropped (a flat grid is not
  the DiT's canvas); ``scene_single_small`` merges into ``scene_single``
  (the px rule below leaves them the same recipe).
- **Pool** ``t4k`` for every scene key (``scenes`` / ``single_scenes`` /
  ``horizontal_scenes``): the 30 % left-to-right items go to it too.
- **px = the erased EN anchor's px** × U(``JITTER``) per item. EN font px is
  ``√(box area / (EN_CELL × letters))`` over the anchor's ink box (``EN_CELL``
  = a Latin cell of 0.55 × 1.15 px², so a one-line anchor lands on its box
  height and a wrapped one on its line height); the fill follows it, capped
  at 1 (a vertical window in a wide bubble lands below it). The tier's
  ``glyph_px`` carries the jitter draw, ``_fill_for_px`` turns it into px.
- **σ band [0, 1]** for every item (``real_kana --full_sigma``): the band
  gate always passes and the trainer's affine map leaves the LoRA trainer's
  own σ draw (``configs/base.toml``: sigmoid, shift 1.0).

Items = steps: ``STEPS`` / row × 174 rows, batch 1 — one epoch.

Train: the line's trainer (μ 0, lr 1e-3 cosine, warmup 0.1, in-box share),
warm from ``retrain_kana``'s merged rows, routed, batch 1 at native size with
the blocks compiled dynamic-seq over the pool's token range
(``real_kana.patch_native``).

Read (routed, 1 k): ``real_kana``'s three reads — target, words, singles —
each paired against ``retrain_kana``'s cached renders.

Legs: ``data`` (CPU) → ``OUT/run0928_polish_t4k/data``; ``train`` (GPU) →
``OUT/experiments/polish_t4k``; ``read`` (GPU). ``--dry_run`` prints the
EN px per scene and the plan, writes nothing.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label polish-rows \\
      project/cjk_anima_scale/experiments/polish_rows/run_exp.py \\
      --label t4k24 --legs train read"
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # every leg is routed (docstring)
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale import reads as scoring  # noqa: E402


RK = load_experiment("real_kana")

EXP = OUT / "experiments"
NAME = "polish_t4k"
DATA = "run0928_polish_t4k"
BASE = "retrain_kana"  # the rows this pass warms from and is read against
CONTEXT = OUT / BASE / "trained.pt"
POOL = "t4k"
STEPS = 24  # steps / row (user, 2026-09-28); items = steps, batch 1
N_ROWS = 174
SINGLE_SHARE = 0.32  # retrain_kana's lone scene items: (2 900 + 1 740) / 14 500
EN_CELL = 0.55 * 1.15  # a Latin glyph's advance × line height, in font px²
JITTER = (0.85, 1.15)  # the kana px around the EN px, per item


def en_px(sc: dict) -> float:
    b = sc["box"]
    area = (b[2] - b[0]) * (b[3] - b[1])
    return (area / (EN_CELL * len(sc["anchor"]))) ** 0.5


def scenes() -> list[dict]:
    f = OUT / f"scenes_{POOL}" / "scenes.jsonl"
    return [json.loads(ln) for ln in f.read_text("utf-8").splitlines() if ln]


def table():
    from cjk_scale.builder import Group, Tier

    common = {"fill": 1.0, "min_glyph": 28, "glyph_px": list(JITTER)}
    return (
        # tier names: the file prefixes of the data of record (builder.tier_of)
        Group(
            "single",
            (0.0, 1.0),
            1.0,
            (
                Tier("b0010_scene_single", "bubble1", SINGLE_SHARE, dict(common)),
                Tier("b0010_scene_window", "bubbleN", 1 - SINGLE_SHARE, dict(common)),
            ),
        ),
    )


def patch_build(n_items: int) -> None:
    """The four data patches of the docstring."""
    from cjk_scale import builder, recipes
    from cjk_scale.config import DATA as D
    from cjk_scale.windows import Window

    D.update(
        scenes=POOL,
        scene_one_bubble=POOL,
        single_scenes=POOL,
        horizontal_scenes=POOL,
    )
    px_of = {tuple(sc["region"]): en_px(sc) for sc in scenes()}
    fit = recipes._fit_px

    def fill_for_en_px(region, n_glyphs, jitter, vertical, fill_max, max_lines=1):
        target = px_of[tuple(region)] * jitter
        return max(
            0.15,
            min(1.0, target / max(fit(region, n_glyphs, vertical, max_lines), 1e-6)),
        )

    recipes._fill_for_px = fill_for_en_px
    builder.window = lambda kind, px, layout: Window(0.0, 1.0, "polish: full σ")
    builder.plan_groups = lambda kinds, table, budget=1.0: [(g, n_items) for g in table]


def rc_of(name: str):
    from cjk_scale.config import RunConfig, load_run

    base = load_run(BASE)
    return RunConfig(
        name=name, path=Path(__file__), vocabs=base.vocabs, read=base.read, context=BASE
    )


def data_report(d: Path, sc_px: dict) -> dict:
    recs = [
        json.loads(ln)
        for ln in (d / "train.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    ratio: dict = {}
    for x in recs:
        ratio.setdefault(x["recipe"], []).append(x["px"] / sc_px[x["scene"]])
    out = {
        "n": len(recs),
        "recipes": dict(Counter(r["recipe"] for r in recs)),
        "shapes": dict(Counter("x".join(map(str, r["shape"])) for r in recs)),
        "horizontal": sum(bool(r.get("horizontal")) for r in recs),
        "scenes_used": len({r["scene"] for r in recs}),
        "px_median": {
            k: round(statistics.median(x["px"] for x in recs if x["recipe"] == k), 1)
            for k in ratio
        },
        "px_over_en_median": {
            k: round(statistics.median(v), 2) for k, v in ratio.items()
        },
        "px_over_en_below_0.7": {k: sum(x < 0.7 for x in v) for k, v in ratio.items()},
    }
    print(f"data report: {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "read"]
    )
    p.add_argument("--workers", type=int)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    n_items = STEPS * N_ROWS
    sc = scenes()
    sc_px = {s["i"]: en_px(s) for s in sc}
    pxs = sorted(sc_px.values())
    print(
        f"{POOL}: {len(sc)} scenes, EN px median {statistics.median(pxs):.0f} "
        f"(min {pxs[0]:.0f}, max {pxs[-1]:.0f}); {n_items} items = steps, batch 1",
        flush=True,
    )
    metrics: dict = {
        "pool": POOL,
        "n_scenes": len(sc),
        "en_px": {str(k): round(v, 1) for k, v in sc_px.items()},
        "steps_per_row": STEPS,
        "n_items": n_items,
        "single_share": SINGLE_SHARE,
        "jitter": list(JITTER),
        "context": str(CONTEXT),
    }
    if args.dry_run:
        for s in sc[:12]:
            print(
                f"  {s['i']:4d} {s['anchor']!r:22s} box {s['box']} EN px {sc_px[s['i']]:.0f}"
            )
        return
    run_dir = make_run_dir(
        "polish_rows",
        label=args.label,
        root=LINE / "experiments" / "polish_rows" / "results",
    )
    data = OUT / DATA / "data"
    if "data" in args.legs:
        from cjk_scale.builder import build

        patch_build(n_items)
        build(rc_of(DATA), workers=args.workers, table=table())
        metrics["data"] = data_report(data, sc_px)
    if "train" in args.legs:
        from cjk_scale.train import train

        assert CONTEXT.exists(), CONTEXT
        recs = [
            json.loads(ln)
            for ln in (data / "train.jsonl").read_text("utf-8").splitlines()
            if ln
        ]
        assert len(recs) == n_items, (len(recs), n_items)
        RK.patch_native(recs)
        train(
            rc_of(NAME),
            data=data,
            out=EXP / NAME,
            cold=False,
            steps_per_row=STEPS,
            context=CONTEXT,
        )
    if "read" in args.legs:
        rc = rc_of(NAME)
        arm, base = EXP / NAME, OUT / BASE
        words, singles = list(rc.read), list(RK.RR.HIRA + RK.RR.KATA)
        RR = RK.RR
        out = metrics.setdefault("reads", {})
        h = {
            "target": (
                RK.read_target(rc, arm, data),
                scoring.hits(
                    base / "target" / "native_reads.json",
                    ["はい", "こんにちは"],
                    "verbatim",
                ),
            ),
            "words": (
                RR.grid_hits(RR.ensure(rc, arm, words, "en"), words, "en"),
                RR.grid_hits(base / "native_r4_en" / "native_reads.json", words, "en"),
            ),
            "singles": (
                RR.grid_hits(RR.ensure(rc, arm, singles, "swap"), singles, "swap"),
                RR.grid_hits(
                    base / "native_r4_swap" / "native_reads.json", singles, "swap"
                ),
            ),
        }
        for grp, (mine, ref) in h.items():
            print(f"{NAME} · {grp}:", flush=True)
            out.setdefault(NAME, {})[grp] = scoring.tally(mine)
            print(f"{BASE} · {grp}:", flush=True)
            out.setdefault(BASE, {})[grp] = scoring.tally(ref)
            pr = scoring.paired(mine, ref)
            out[NAME][f"{grp}_paired_vs_{BASE}"] = pr
            print(f"  {grp} {NAME} vs {BASE} {pr}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / DATA), str(EXP / NAME)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
