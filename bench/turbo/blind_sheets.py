#!/usr/bin/env python3
"""Blind two-arm sheets from ``probe_handoff`` PNGs, answered as a 0 / 1 string.

``make``: one sheet per prompt, a row per seed, two columns headed 0 and 1 with
the arms shuffled per row from OS entropy. Rows are numbered across the sheets.
The sheets carry no arm name; the key goes to ``key.json`` beside them.

``score``: the reader's picks, one digit per row in row order (0 = left,
1 = right, any other character = no pick), joined with the key.

    P="-m bench.turbo.blind_sheets"
    .venv/bin/python $P make --set 1122_vs_1028 \
        --arms bench/turbo/results/<run A>/png/<stem A> bench/turbo/results/<run B>/png/<stem B>
    .venv/bin/python $P score --set 1122_vs_1028 --picks 1010-1100101

An arm is ``<png dir>/<file stem>``; the files read are ``<stem>_p{P}_s{S}_{col}.png``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from library.env import resolve_under_home

HEAD, MARGIN = 40, 56


def cells(arm: str, col: str) -> dict[tuple[int, int], Path]:
    d, stem = Path(arm).parent, Path(arm).name
    pat = re.compile(rf"{re.escape(stem)}_p(\d+)_s(\d+)_{re.escape(col)}\.png$")
    out = {}
    for f in sorted(resolve_under_home(d).glob(f"{stem}_p*_s*_{col}.png")):
        m = pat.match(f.name)
        if m:
            out[(int(m[1]), int(m[2]))] = f
    return out


def cmd_make(o) -> None:
    arms = [cells(a, o.col) for a in o.arms]
    keys = sorted(set(arms[0]) & set(arms[1]))
    if not keys:
        raise SystemExit("no (prompt, seed) cell is in both arms")
    out = resolve_under_home(o.out) / o.set
    out.mkdir(parents=True, exist_ok=True)
    rng = random.SystemRandom()
    font = ImageFont.load_default(size=28)
    rows: list[dict[str, str]] = []
    for prompt in sorted({k[0] for k in keys}):
        seeds = [s for q, s in keys if q == prompt]
        w0, h0 = Image.open(arms[0][(prompt, seeds[0])]).size
        w, h = o.cell_width, round(h0 * o.cell_width / w0)
        sheet = Image.new("RGB", (MARGIN + 2 * w, HEAD + h * len(seeds)), "white")
        draw = ImageDraw.Draw(sheet)
        for c in (0, 1):
            draw.text((MARGIN + c * w + w // 2, 4), str(c), fill="black", font=font)
        for r, seed in enumerate(seeds):
            order = [0, 1]
            rng.shuffle(order)
            rows.append(
                {
                    "cell": f"p{prompt}_s{seed}",
                    "0": o.arms[order[0]],
                    "1": o.arms[order[1]],
                }
            )
            draw.text((8, HEAD + r * h + 8), f"#{len(rows)}", fill="black", font=font)
            for c, a in enumerate(order):
                im = Image.open(arms[a][(prompt, seed)]).convert("RGB")
                sheet.paste(
                    im.resize((w, h), Image.LANCZOS), (MARGIN + c * w, HEAD + r * h)
                )
        sheet.save(out / f"p{prompt}.jpg", quality=92)
    (out / "key.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows -> {out} (key.json beside the sheets, not printed)")


def cmd_score(o) -> None:
    rows = json.loads(
        (resolve_under_home(o.out) / o.set / "key.json").read_text(encoding="utf-8")
    )
    if len(o.picks) != len(rows):
        raise SystemExit(f"{len(o.picks)} picks for {len(rows)} rows")
    wins: Counter[str] = Counter()
    for n, (row, pick) in enumerate(zip(rows, o.picks), 1):
        arm = Path(row[pick]).name if pick in "01" else "no pick"
        wins[arm] += 1
        print(f"#{n:<3} {row['cell']:8s} {arm}")
    for arm, k in wins.most_common():
        print(f"{k:3d}  {arm}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--arms", nargs=2, required=True, help="<png dir>/<file stem>")
    m.add_argument("--col", default="S", help="probe column (S, TtoS_b1, ...)")
    m.add_argument("--cell_width", type=int, default=768, help="768 = the probe's PNGs")
    s = sub.add_parser("score")
    s.add_argument("--picks", required=True, help="one of 0 / 1 / other per row")
    for q in (m, s):
        q.add_argument("--set", required=True, help="output name")
        q.add_argument("--out", default="output/blind/turbo")
    o = p.parse_args()
    {"make": cmd_make, "score": cmd_score}[o.cmd](o)


if __name__ == "__main__":
    main()
