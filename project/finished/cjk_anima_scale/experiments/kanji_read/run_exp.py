#!/usr/bin/env python
"""kanji_read — a kanji batch's singles against the floor caches it already has (2026-09-28)

A small read of a ``retrain_kanji_b*`` run: the batch's kanji that the seed
floor already rendered as lone singles (``en`` clause, the ``native`` stage's
8 prompts × 2 seeds) in ``native_single/``, ``native_densea0/`` and
``native_stagei/``. No floor render — a lone glyph encodes the same routed
or not, so the unrouted floor caches stand. The run renders the first
``N_PROMPTS`` prompts × ``SEEDS`` seeds per kanji (user: ≈ 90 renders, not
the full 16 per key) into ``<run>/native_r2_en/``, paired with the floor's
renders of the same prompt × seed (Stage B's scoring, McNemar).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label kanji-read-b1 \\
      project/cjk_anima_scale/experiments/kanji_read/run_exp.py \\
      --label b1 --run retrain_kanji_b1"

``--words`` reads C3's six words instead; ``--arm <name>`` reads an
experiment's rows (``OUT/experiments/<name>``, e.g. ``polish_b1``) on the
run's kanji, paired with the run itself as well.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = (
    "1"  # the run's own encoding (singles: same either way)
)
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, pin_old_seed, floor_dir, load_experiment  # noqa: E402

bootstrap()
pin_old_seed()  # the retrain reads against the old seed's floor of record
from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale import reads as scoring  # noqa: E402



N_PROMPTS = 2
SEEDS = 2
CLAUSE = "en"
FLOOR_CACHES = ("native_single", "native_densea0", "native_stagei")  # 8 × 2, en
SUB = f"r{N_PROMPTS}_{CLAUSE}"


def floor_reads(chars: set) -> tuple[list, dict]:
    """The floor's lone-glyph reads of ``chars`` on this grid, first cache
    wins; each key's source cache."""
    recs, src = [], {}
    for c in FLOOR_CACHES:
        for m in json.loads((floor_dir() / c / "native_reads.json").read_text("utf-8")):
            t = m["text"]
            if t not in chars or m["clause"] != CLAUSE or src.get(t, c) != c:
                continue
            if int(m["pi"]) < N_PROMPTS and int(m["seed"]) < SEEDS:
                src[t] = c
                recs.append(m)
    return recs, src


def grid(h: dict) -> dict:
    return {k: v for k, v in h.items() if int(k[2]) < N_PROMPTS and int(k[3]) < SEEDS}


def words_read(args, rc, name: str, arm: Path) -> None:
    """``--words``: C3's six kanji-bearing words (routed, en, C3's 8 × 2 grid)
    on the arm, paired with the floor's, ``c3_kanji_450``'s and (an
    experiment arm) the run's reads."""
    C3 = load_experiment("c3_kanji")
    arms = {"floor": None, "c3_kanji_450": C3.EXP / "c3_kanji_450"}
    if name != rc.name:
        arms[rc.name] = OUT / rc.name
    arms[name] = arm
    print(
        f"{name}: words {' '.join(C3.READ_WORDS)} → "
        f"{len(C3.READ_WORDS) * 16} renders on the run (floor / c3_kanji_450 cached)",
        flush=True,
    )
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "kanji_read",
        label=args.label,
        root=LINE / "experiments" / "kanji_read" / "results",
    )
    wh = {a: C3.read_words(p) for a, p in arms.items()}
    metrics: dict = {"words": list(C3.READ_WORDS), "reads": {}}
    for a, h in wh.items():
        print(f"{a} · words (routed):", flush=True)
        metrics["reads"][a] = scoring.tally(h)
    for ref in [a for a in arms if a != name]:
        pr = scoring.paired(wh[name], wh[ref])
        metrics["reads"][name][f"paired_vs_{ref}"] = pr
        print(f"  words {name} vs {ref} {pr}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(arm / f"native_{C3.TAG}")],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--words", action="store_true", help="C3's six words only")
    p.add_argument(
        "--arm", help="an experiment's rows under OUT/experiments/ (default: the run)"
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    from cjk_scale.config import load_run
    from cjk_scale.eval import TRAINED_ARM, _fold, _load_reads, probe_args
    from stages import run as run_stage

    rc = load_run(args.run)
    name = args.arm or rc.name
    arm = OUT / "experiments" / args.arm if args.arm else OUT / rc.name
    assert (arm / "trained.pt").exists(), arm
    if args.words:
        return words_read(args, rc, name, arm)
    vocabs = set(
        json.loads((OUT / rc.name / "data" / "vocabs.json").read_text("utf-8"))
    )
    frecs, src = floor_reads(vocabs)
    chars = sorted(src)
    print(
        f"{name}: {len(chars)} kanji with a floor ({''.join(chars)}), "
        f"{N_PROMPTS} prompts × {SEEDS} seeds → {len(chars) * N_PROMPTS * SEEDS} renders",
        flush=True,
    )
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "kanji_read",
        label=args.label,
        root=LINE / "experiments" / "kanji_read" / "results",
    )
    dst = arm / f"native_{SUB}" / "native_reads.json"
    held = {m["text"] for m in _load_reads(dst)}
    miss = [c for c in chars if c not in held]
    if miss:
        a = probe_args(
            rc,
            TRAINED_ARM,
            ["native"],
            [
                "--eval_tag",
                f"add_{SUB}",
                "--native_chars",
                ",".join(miss),
                "--native_clauses",
                CLAUSE,
                "--native_limit",
                str(N_PROMPTS),
                "--seeds",
                str(SEEDS),
            ],
        )
        a.arm_path = str(arm)
        run_stage("native", a)
        scratch = arm / f"native_add_{SUB}"
        _fold("native", _load_reads(scratch / "native_reads.json"), dst, move=True)
        shutil.rmtree(scratch)

    # the pairs must be the same prompt × seed
    fprompt = {(m["text"], int(m["pi"]), int(m["seed"])): m["prompt"] for m in frecs}
    for m in _load_reads(dst):
        k = (m["text"], int(m["pi"]), int(m["seed"]))
        if k in fprompt:
            assert fprompt[k] == m["prompt"], (k, fprompt[k], m["prompt"])

    ffile = run_dir / "floor_reads.json"
    ffile.write_text(json.dumps(frecs, ensure_ascii=False), encoding="utf-8")
    mine = grid(scoring.hits(dst, chars, CLAUSE))
    ref = grid(scoring.hits(ffile, chars, CLAUSE))
    print(f"{name} · singles ({CLAUSE}, {N_PROMPTS}×{SEEDS}):", flush=True)
    t_mine = scoring.tally(mine)
    print("floor:", flush=True)
    t_ref = scoring.tally(ref)
    by_cache = {
        c: scoring.paired(
            {k: v for k, v in mine.items() if src[k[0]] == c},
            {k: v for k, v in ref.items() if src[k[0]] == c},
        )
        for c in FLOOR_CACHES
    }
    pr = scoring.paired(mine, ref)
    print(f"  paired {name} vs floor {pr}", flush=True)
    for c, v in by_cache.items():
        print(f"    {c}: {v}", flush=True)
    vs_run = None
    if name != rc.name:  # the run's own renders of the same grid
        rf = OUT / rc.name / f"native_{SUB}" / "native_reads.json"
        vs_run = scoring.paired(mine, grid(scoring.hits(rf, chars, CLAUSE)))
        print(f"  paired {name} vs {rc.name} {vs_run}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={
            "chars": "".join(chars),
            "source": src,
            "grid": [N_PROMPTS, SEEDS],
            "run": t_mine,
            "floor": t_ref,
            "paired": pr,
            "paired_by_cache": by_cache,
            "arm": str(arm),
            f"paired_vs_{rc.name}": vs_run,
        },
        artifacts=[str(dst)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
