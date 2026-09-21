#!/usr/bin/env python
"""row_dose — per-row exposure vs sentence content, off files a run already wrote.

A sentence arm and its seed read the same multi-glyph eval strings. For every
(item, ext piece) the piece is a *hit* when its string is in the item's read;
the paired gain is ``hit(arm) − hit(seed)`` on the same item and seed, binned
by how many multi-glyph training items carry that row (``train.jsonl``). CPU
only, no re-render.

    .venv/bin/python project/cjk_renderable_anima/src/probe/row_dose.py \
        --data output/wake_probe/data_step2_0919 \
        --seed output/wake_probe/rows_step2_0920_seedread \
        output/wake_probe/rows_step2_0920_plain_bs05c25_6k [<arm dir> …] \
        [--rows 1101,192,…]    # also report these ext rows as their own bin

The seed dir is the step-1 table copied into a ``rows_<data_tag>_seedread``
arm dir and read with ``--stage eval`` on the step-2 eval groups. Piece-in-read
counts a lone big glyph as a hit, which is why the gain is paired against the
seed instead of read as a level (2026-09-20: the seed already reads +0.20 on
the most frequent rows).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
GROUPS = ("short", "short_held", "phrase", "phrase_held", "gword", "gword_held")
EDGES = (0, 150, 400, 1000, 10**9)


def _pieces_fn():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO / "library/anima/configs/qwen3_06b")
    pack = json.loads(
        (REPO / "models/vocab_packs/anima_cjk_vocab_pack.json").read_text()
    )["qwen"]

    def pieces(text: str) -> set[tuple[int, str]]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out = {(pack[str(t)], tok.decode([t])) for t in ids if str(t) in pack}
        return {(e, s) for e, s in out if s.strip()}

    return pieces


def _reads(arm: Path) -> dict:
    out = {}
    for r in json.loads((arm / "eval_reads.json").read_text()):
        if r.get("cond") == "trained" and r["group"] in GROUPS:
            read = max((x.get("sfx") or "" for x in r["reads"]), key=len, default="")
            out[(r["group"], r["text"], r["seed"])] = read
    return out


def _gain(inst, arm, seed, boot, rng):
    """Mean paired gain and an item-bootstrap 95 % interval."""
    by_item = collections.defaultdict(list)
    for q, _, s in inst:
        by_item[q].append((s in arm[q]) - (s in seed[q]))
    items = sorted(by_item)
    mean = st.mean(d for q in items for d in by_item[q])
    ds = sorted(
        st.mean(d for q in rng.choices(items, k=len(items)) for d in by_item[q])
        for _ in range(boot)
    )
    return mean, ds[int(0.025 * boot)], ds[int(0.975 * boot) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs="+", type=Path)
    ap.add_argument("--seed", type=Path, required=True, help="the seed-read arm dir")
    ap.add_argument("--data", type=Path, required=True, help="the step-2 data dir")
    ap.add_argument("--rows", default="", help="ext ids to report as their own bin")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    pieces = _pieces_fn()
    occ = collections.Counter()
    for line in (a.data / "train.jsonl").read_text().splitlines():
        rec = json.loads(line)
        # a grid item is k texts (its cells), a composite one
        for text in rec.get("units") or [rec["text"]]:
            ps = pieces(text)
            if len(ps) > 1:
                occ.update(e for e, _ in ps)

    seed = _reads(a.seed)
    picked = {int(x) for x in a.rows.split(",") if x.strip()}
    rng = random.Random(0)
    for arm_dir in a.arms:
        arm = _reads(arm_dir)
        keys = sorted(set(arm) & set(seed))
        inst = [(q, e, s) for q in keys for e, s in pieces(q[1])]
        print(f"\n{arm_dir.name}  ({len(keys)} shared items, {len(inst)} pieces)")
        bins = [
            (f"occ [{lo}, {'inf' if hi > 10**8 else hi})", lambda e, lo=lo, hi=hi: lo <= occ[e] < hi)
            for lo, hi in zip(EDGES[:-1], EDGES[1:])
        ]
        if picked:
            bins = [(f"--rows ({len(picked)})", lambda e: e in picked)] + [
                (n + " other", lambda e, f=f: f(e) and e not in picked) for n, f in bins
            ]
        for scope, keep in (("all", lambda q: True), ("held", lambda q: q[0].endswith("_held"))):
            print(f"  {scope}:")
            for name, f in bins:
                b = [x for x in inst if f(x[1]) and keep(x[0])]
                if not b:
                    continue
                m, lo, hi = _gain(b, arm, seed, a.boot, rng)
                hit_s = st.mean(s in seed[q] for q, _, s in b)
                hit_a = st.mean(s in arm[q] for q, _, s in b)
                print(
                    f"    {name:24s} n={len(b):3d} rows={len({e for _, e, _ in b}):3d}  "
                    f"hit {hit_s:.3f} → {hit_a:.3f}  gain {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                )


if __name__ == "__main__":
    main()
