#!/usr/bin/env python
"""retrain_read — retrain_experiments § 5: a retrain run's routed read (2026-09-28)

The ``native`` stage on a smaller grid than the reads of record — the first
``N_PROMPTS`` (4) scene prompts × ``SEEDS`` (2) = 8 renders per key — routed
(``ANIMA_VOCAB_GLYPH_ROUTE``, set in-process), with the floor and ``p1_mix``
taken from their caches of record wherever they hold the key (the 8 × 2
grid restricted to prompts < 4): a new floor render only where no cache
holds it.

- **Words** (the run's ``read``, held out of its windows by trigram), ``en``,
  the run only. C2's eight words pair with the floor's and ``p1_mix``'s
  routed ``native_route/`` caches; the other five have no floor (routed
  words read ≈ 0 on the floor: C2 ≤ 1 edit 1 / 128).
- **Singles**, ``swap``: ``HIRA`` from the floor's ``native_spell/`` swap
  cache, ``KATA`` with the floor rendered once into ``native_r4_swap/``
  (no katakana single is cached).

The run's renders land in ``<run>/native_r4_<clause>/`` (this grid only).
Scoring is Stage B's per render, paired (McNemar) on shared prompt × seed.

    run_exp.py --label kana --run retrain_kana [--dry_run]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # every render is routed (docstring)

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, pin_old_seed, floor_dir, load_experiment  # noqa: E402

bootstrap()
pin_old_seed()  # the retrain reads against the old seed's floor of record
from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale import reads as scoring  # noqa: E402


P2 = load_experiment("p2_route")

N_PROMPTS = 4
SEEDS = 2
C2_WORDS = P2.C2_WORDS
HIRA = tuple("あうがくとひもり")  # in the floor's native_spell/ swap cache
KATA = tuple("アカシトノン")  # no floor cache: rendered once
FLOOR_CACHE = {"en": "native_route", "swap": "native_spell"}  # of record, 8 × 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--run", required=True, help="configs/runs/<run>.toml")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def ensure(rc, root: Path, keys: list[str], clause: str, floor: bool = False) -> Path:
    """``keys`` × ``clause`` on this grid in ``root``'s ``native_r4_<clause>/``:
    the missing ones render into a scratch dir and fold in."""
    import shutil

    from cjk_scale.eval import FLOOR_ARM, TRAINED_ARM, _fold, _load_reads, probe_args
    from stages import run as run_stage

    sub = f"r4_{clause}"
    dst = root / f"native_{sub}" / "native_reads.json"
    held = {m["text"] for m in _load_reads(dst) if m["clause"] == clause}
    miss = [k for k in keys if k not in held]
    if miss:
        a = probe_args(
            rc,
            FLOOR_ARM if floor else TRAINED_ARM,
            ["native"],
            [
                "--eval_tag",
                f"add_{sub}",
                "--native_chars",
                ",".join(miss),
                "--native_clauses",
                clause,
                "--native_limit",
                str(N_PROMPTS),
                "--seeds",
                str(SEEDS),
            ],
        )
        a.arm_path = str(root)
        run_stage("native", a)
        scratch = root / f"native_add_{sub}"
        n = _fold("native", _load_reads(scratch / "native_reads.json"), dst, move=True)
        shutil.rmtree(scratch)
        print(f"  {root.name}: {n} keys rendered into native_{sub}/", flush=True)
    return dst


def grid_hits(path: Path, keys: list[str], clause: str) -> dict:
    """Stage B's per-render hits restricted to this grid; every multi-glyph
    render must be routed (one ext row per glyph)."""
    import json

    for m in json.loads(path.read_text("utf-8")):
        if m["text"] in keys and len(m["text"]) > 1 and m["clause"] == clause:
            assert m["ext_rows"] == len(m["text"]), (m["file"], m["ext_rows"])
    h = scoring.hits(path, keys, clause)
    h = {k: v for k, v in h.items() if k[2] < N_PROMPTS and k[3] < SEEDS}
    for t in keys:
        n = sum(k[0] == t for k in h)
        assert n == N_PROMPTS * SEEDS, (path, t, n)
    return h


def main():
    args = parse_args()
    from cjk_scale.config import load_run

    rc = load_run(args.run)
    words = list(rc.read)
    c2 = [w for w in words if w in C2_WORDS]
    run_arm = OUT / rc.name
    fl = floor_dir()
    n_new = (len(words) + len(HIRA) + 2 * len(KATA)) * N_PROMPTS * SEEDS
    print(
        f"{rc.name}: {len(words)} words (en) + {len(HIRA) + len(KATA)} singles (swap), "
        f"{N_PROMPTS} prompts × {SEEDS} seeds; ≤ {n_new} new renders "
        f"(floor: {len(KATA)} katakana singles)",
        flush=True,
    )
    if args.dry_run:
        print("words", " ".join(words), "\nsingles", " ".join(HIRA + KATA), flush=True)
        for t in HIRA:
            grid_hits(fl / FLOOR_CACHE["swap"] / "native_reads.json", [t], "swap")
        for w in c2:
            for arm in (fl, P2.EXP / "p1_mix"):
                grid_hits(arm / FLOOR_CACHE["en"] / "native_reads.json", [w], "en")
        print("floor / p1_mix caches hold the grid", flush=True)
        return
    assert (run_arm / "trained.pt").exists(), f"{run_arm}: not trained yet"
    run_dir = make_run_dir(
        "retrain_read",
        label=args.label,
        root=LINE / "experiments" / "retrain_read" / "results",
    )
    singles = list(HIRA + KATA)
    run_w = grid_hits(ensure(rc, run_arm, words, "en"), words, "en")
    run_s = grid_hits(ensure(rc, run_arm, singles, "swap"), singles, "swap")
    fl_s = {
        **grid_hits(fl / FLOOR_CACHE["swap"] / "native_reads.json", list(HIRA), "swap"),
        **grid_hits(ensure(rc, fl, list(KATA), "swap", floor=True), list(KATA), "swap"),
    }
    cached = {
        a: grid_hits(p / FLOOR_CACHE["en"] / "native_reads.json", c2, "en")
        for a, p in (("floor", fl), ("p1_mix", P2.EXP / "p1_mix"))
    }
    metrics: dict = {
        "run": rc.name,
        "grid": {
            "prompts": N_PROMPTS,
            "seeds": SEEDS,
            "words": "en",
            "singles": "swap",
        },
        "words": words,
        "singles": singles,
        "reads": {},
    }
    R = metrics["reads"]

    def show(tag: str, h: dict) -> dict:
        print(f"{tag}:", flush=True)
        return scoring.tally(h)

    hira_w = [w for w in words if all("ぁ" <= c <= "ゖ" for c in w)]
    kata_w = [w for w in words if w not in hira_w]
    R[rc.name] = {
        "words_c2": show(
            f"{rc.name} · C2 words", {k: v for k, v in run_w.items() if k[0] in c2}
        ),
        "words_new_hira": show(
            f"{rc.name} · other hiragana words",
            {k: v for k, v in run_w.items() if k[0] in hira_w and k[0] not in c2},
        ),
        "words_kata": show(
            f"{rc.name} · katakana words",
            {k: v for k, v in run_w.items() if k[0] in kata_w},
        ),
        "singles_hira": show(
            f"{rc.name} · hiragana singles",
            {k: v for k, v in run_s.items() if k[0] in HIRA},
        ),
        "singles_kata": show(
            f"{rc.name} · katakana singles",
            {k: v for k, v in run_s.items() if k[0] in KATA},
        ),
    }
    for a, h in cached.items():
        R[a] = {"words_c2": show(f"{a} · C2 words (cache)", h)}
    R["floor"]["singles_hira"] = show(
        "floor · hiragana singles (cache)",
        {k: v for k, v in fl_s.items() if k[0] in HIRA},
    )
    R["floor"]["singles_kata"] = show(
        "floor · katakana singles", {k: v for k, v in fl_s.items() if k[0] in KATA}
    )
    c2_run = {k: v for k, v in run_w.items() if k[0] in c2}
    for a, h in cached.items():
        pr = scoring.paired(c2_run, h)
        R[rc.name][f"words_c2_paired_vs_{a}"] = pr
        print(f"  C2 words {rc.name} vs {a} {pr}", flush=True)
    for grp, chars in (("hira", HIRA), ("kata", KATA)):
        pr = scoring.paired(
            {k: v for k, v in run_s.items() if k[0] in chars},
            {k: v for k, v in fl_s.items() if k[0] in chars},
        )
        R[rc.name][f"singles_{grp}_paired_vs_floor"] = pr
        print(f"  {grp} singles {rc.name} vs floor {pr}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(run_arm / "native_r4_en"), str(run_arm / "native_r4_swap")],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
