#!/usr/bin/env python3
"""Blind two-arm sheets from ``probe_handoff`` PNGs, answered as a 0 / 1 string.

``make``: a row per (prompt, seed), two columns headed 0 and 1 with the arms
shuffled per row from OS entropy; one file per row by default
(``--rows_per_sheet``), rows numbered across the files. The sheets carry no arm
name; the key goes to ``key.json`` beside them. ``--reuse_key`` recomposes a set
on the sides it already has.

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
    out = resolve_under_home(o.out) / o.set
    out.mkdir(parents=True, exist_ok=True)
    key_file = out / "key.json"
    if o.reuse_key:  # the same sides, recomposed (another layout or cell width)
        rows = json.loads(key_file.read_text(encoding="utf-8"))
        arm_names = sorted({r[c] for r in rows for c in "01"})
    else:
        arm_names = list(o.arms)
    arms = {a: cells(a, o.col) for a in arm_names}
    if not o.reuse_key:
        keys = sorted(set(arms[arm_names[0]]) & set(arms[arm_names[1]]))
        if not keys:
            raise SystemExit("no (prompt, seed) cell is in both arms")
        rng = random.SystemRandom()
        rows = []
        for prompt, seed in keys:
            order = list(arm_names)
            rng.shuffle(order)
            rows.append({"cell": f"p{prompt}_s{seed}", "0": order[0], "1": order[1]})
        key_file.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def cell_of(row: dict[str, str], c: str) -> Path:
        prompt, seed = (int(x[1:]) for x in row["cell"].split("_"))
        return arms[row[c]][(prompt, seed)]

    numbered = list(enumerate(rows, 1))
    if o.rows_per_sheet > 0:
        k = o.rows_per_sheet
        groups = [numbered[i : i + k] for i in range(0, len(numbered), k)]
        names = [f"r{g[0][0]:02d}" for g in groups]
    else:  # one sheet per prompt
        prompts = sorted({r["cell"].split("_")[0] for r in rows})
        groups = [
            [nr for nr in numbered if nr[1]["cell"].split("_")[0] == q] for q in prompts
        ]
        names = prompts
    font = ImageFont.load_default(size=28)
    for name, group in zip(names, groups):
        w0, h0 = Image.open(cell_of(group[0][1], "0")).size
        w, h = o.cell_width, round(h0 * o.cell_width / w0)
        sheet = Image.new("RGB", (MARGIN + 2 * w, HEAD + h * len(group)), "white")
        draw = ImageDraw.Draw(sheet)
        for c in (0, 1):
            draw.text((MARGIN + c * w + w // 2, 4), str(c), fill="black", font=font)
        for r, (n, row) in enumerate(group):
            draw.text((8, HEAD + r * h + 8), f"#{n}", fill="black", font=font)
            for c in (0, 1):
                im = Image.open(cell_of(row, str(c))).convert("RGB")
                if im.size != (w, h):
                    im = im.resize((w, h), Image.LANCZOS)
                sheet.paste(im, (MARGIN + c * w, HEAD + r * h))
        sheet.save(out / f"{name}.jpg", quality=95)
    print(f"{len(rows)} rows, {len(groups)} sheets -> {out} (key.json not printed)")


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
    m.add_argument("--arms", nargs=2, help="<png dir>/<file stem>")
    m.add_argument(
        "--reuse_key",
        action="store_true",
        help="recompose the set from its key.json (no new shuffle; --arms not read)",
    )
    m.add_argument(
        "--rows_per_sheet",
        type=int,
        default=1,
        help="rows in a file (r01.jpg, ...); 0 = one sheet per prompt (p0.jpg, ...)",
    )
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
