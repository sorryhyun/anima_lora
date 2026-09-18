#!/usr/bin/env python
"""cross_sheet — two arms' ``native`` renders on one sheet, row-interleaved.

Reading two arms from their own ``sheet_<kana>_<clause>.png`` means holding one
page in your head while looking at the other; at the hit rates this line runs
at, that is where a per-glyph reversal hides (2026-09-18). This writes the same
sheet the native stage writes, with **one row per arm per prompt**, the arms
stacked so the same prompt and seed sit directly above one another:

    p00  A   enref s0 | trained s0 | enref s1 | trained s1
    p00  B   enref s0 | trained s0 | enref s1 | trained s1
    p01  A   …

The EN reference cells come from the shared ``native_enref`` cache, so both
rows carry the *same* reference image and each arm's drift from it is the
vertical comparison. Free — it reads the ``native_reads.json`` both arms
already wrote, no GPU and no re-render.

    python src/probe/cross_sheet.py <arm A native dir> <arm B native dir> \
        [--tags prev,curr] [--out DIR] [--chars あ,か] [--clauses en,swap]

Defaults: every char and clause present in A, output into B's ``cross/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.readers import contact_sheet, hit  # noqa: E402


def _load(d: Path) -> dict:
    f = d / "native_reads.json"
    if not f.exists():
        raise SystemExit(f"no native_reads.json in {d} — run --stage native first")
    return {(m["pi"], m["text"], m["clause"], m["seed"]): m for m in json.loads(f.read_text())}


def _enref_dir(m: dict) -> Path | None:
    """The shared enref cache that produced this arm's EN-reference ruler."""
    for p in Path("/home/sorryhyun/anima/anima_lora/output/wake_probe/native_enref").glob("*"):
        if (p / f"enref_p{m['pi']:02d}_s{m['seed']}.png").exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs=2, type=Path, help="two <arm>/native dirs")
    ap.add_argument("--tags", default="prev,curr", help="row labels, comma-separated")
    ap.add_argument("--out", type=Path, default=None, help="default: <arm B>/cross")
    ap.add_argument("--chars", default="", help="default: every char in arm A")
    ap.add_argument("--clauses", default="", help="default: every clause in arm A")
    ap.add_argument("--thumb", type=int, default=192)
    a = ap.parse_args()

    from PIL import Image

    tags = a.tags.split(",")
    idx = [_load(d) for d in a.arms]
    if set(idx[0]) != set(idx[1]):
        miss = (set(idx[0]) ^ set(idx[1]))
        raise SystemExit(f"arms do not cover the same items ({len(miss)} differ, e.g. {sorted(miss)[:3]})")

    keys = sorted(idx[0])
    chars = a.chars.split(",") if a.chars else sorted({k[1] for k in keys})
    clauses = a.clauses.split(",") if a.clauses else sorted({k[2] for k in keys})
    pis = sorted({k[0] for k in keys})
    seeds = sorted({k[3] for k in keys})

    ed = _enref_dir(idx[0][keys[0]])
    enref_reads = {}
    if ed is not None and (ed / "enref_reads.json").exists():
        enref_reads = json.loads((ed / "enref_reads.json").read_text())

    out = a.out or (a.arms[1].parent / a.arms[1].name / "cross")
    out.mkdir(parents=True, exist_ok=True)

    def enref_cell(tag, pi, seed):
        f = ed / f"enref_p{pi:02d}_s{seed}.png"
        r = [x for x in (enref_reads.get(f.name) or []) if not x.get("whole")]
        r0 = r[-1] if r else {"sfx": "", "vl": ""}
        return Image.open(f).convert("RGB"), [
            f"p{pi:02d} {tag} enref s{seed}", f"sfx {r0['sfx'] or ''}", f"vl {r0['vl'] or ''}"]

    def trained_cell(tag, m):
        r0 = m["reads"][-1] if m["reads"] else {"sfx": "", "vl": ""}
        both = hit(m["reads"], m["text"], "sfx") and hit(m["reads"], m["text"], "vl")
        e = f" e{m['en_cos']:.2f}" if m.get("en_cos") is not None else ""
        return Image.open(m["file"]).convert("RGB"), [
            f"p{m['pi']:02d} {tag} s{m['seed']} {'HIT' if both else '-'}{e}",
            f"sfx {r0['sfx'] or ''}", f"vl {r0['vl'] or ''}"]

    for text in chars:
        for clause in clauses:
            rows, tally = [], []
            for pi in pis:
                for tag, ix in zip(tags, idx):
                    for seed in seeds:
                        if ed is not None:
                            rows.append(enref_cell(tag, pi, seed))
                        rows.append(trained_cell(tag, ix[(pi, text, clause, seed)]))
            for tag, ix in zip(tags, idx):
                n = sum(hit(ix[k]["reads"], text, "sfx") and hit(ix[k]["reads"], text, "vl")
                        for k in keys if k[1] == text and k[2] == clause)
                tally.append(f"{tag} {n}")
            cols = (2 if ed is not None else 1) * len(seeds)
            contact_sheet(rows, out / f"x_{text}_{clause}.png", thumb=a.thumb, cols=cols)
            print(f"x_{text}_{clause}.png  both: {' / '.join(tally)} of "
                  f"{len(pis) * len(seeds)}", flush=True)
    print(f"\nsheets: {out}")


if __name__ == "__main__":
    main()
