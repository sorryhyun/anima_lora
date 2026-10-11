#!/usr/bin/env python
"""jamo_vl_sheet — a ``jamo_vl.py score`` run laid out glyph by glyph (CPU, user 10-11).

One row per target (H, then the words), one column per arm: the render (seed 0)
with v4's free read of the whole image under it — green when a crop reads the
target exactly, amber when the whole-image read is one syllable sharing a jamo
with it. Beside the sheets, ``reads.md``: the same reads as a table, and per
arm the F1 of what was drawn against the target (user 10-11) — the whole-image
read's letters (L* / N*; punctuation and space dropped) as a multiset against
the target's, P = matched / drawn, R = matched / the target's, F1 per image
then the mean (an empty read: P = R = F1 = 0). Twice: on glyphs, and on
positioned jamo (cho / jung / jong of each syllable; a non-Hangul letter is
one token that never matches), so 식 for 시 scores P 2/3, R 1.
The chance line, ``jamo_f1_other``: the same reads scored against every other
target of the group (the jamo two syllables share by chance: ㅇ, ㅏ, no final).

    .venv/bin/python project/cjk_anima_reseed/probes/jamo_vl_sheet.py \\
        results/20261011-1409-jamo-vl-score --arms jamo_j64,jamo_j96,jamo_j128

→ ``results/<ts>-jamo-vl-sheet/``: ``sheet_<group>_<i>.png``, ``reads.md``,
``result.json`` (the F1 tables).
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
sys.path.insert(0, str(HOME / "probes"))

from jamo_vl import RENDER  # noqa: E402

ARMS = "jamo_j64,jamo_j96,jamo_j128,jamo_j64_lone,jamo_f64_reg"
CELL = 200
LABEL_H = 44
HEAD_W = 120
PER_SHEET = 8
FONT = HOME / "assets" / "fonts" / "kozh" / "LXGWWenKai-Regular.ttf"
OK, NEAR, MISS = (40, 150, 60), (200, 130, 0), (90, 90, 90)


def whole(rec: dict) -> str:
    """v4's free read of the whole image (the last crop)."""
    free = rec["v4"]["free"]
    return free[-1] if free else ""


def shares_jamo(text: str, read: str) -> bool:
    from reseed.jamo import decompose, is_syllable

    if len(read) != 1 or not (is_syllable(text) and is_syllable(read)):
        return False
    return any(a == b for a, b in zip(decompose(text), decompose(read)) if a)


def mark(rec: dict) -> str:
    if rec["v4"]["free_hit"]:
        return "ok"
    return "near" if shares_jamo(rec["text"], whole(rec)) else "miss"


def letters(s: str) -> list:
    s = unicodedata.normalize("NFKC", s)
    return [ch for ch in s if unicodedata.category(ch)[0] in "LN"]


def jamo_tokens(glyphs: list) -> list:
    from reseed.jamo import decompose, is_syllable

    out = []
    for ch in glyphs:
        if is_syllable(ch):
            c, v, f = decompose(ch)
            out += [("c", c), ("v", v)] + ([("f", f)] if f else [])
        else:
            out.append(("x", ch))
    return out


def prf(pred: list, gold: list) -> tuple[float, float, float]:
    m = sum((Counter(pred) & Counter(gold)).values())
    if not m:
        return 0.0, 0.0, 0.0
    p, r = m / len(pred), m / len(gold)
    return p, r, 2 * p * r / (p + r)


def f1_row(rs: list) -> dict:
    """Mean glyph / jamo P, R, F1 over the images, the mean letters drawn, and the
    jamo F1 against the other targets (chance)."""
    g, j, n, o = [], [], [], []
    targets = [r["text"] for r in rs]
    for r in rs:
        pred, gold = letters(whole(r)), letters(r["text"])
        g.append(prf(pred, gold))
        j.append(prf(jamo_tokens(pred), jamo_tokens(gold)))
        n.append(len(pred))
        o += [
            prf(jamo_tokens(pred), jamo_tokens(letters(t)))
            for t in targets
            if t != r["text"]
        ]
    mean = lambda xs, k: round(sum(x[k] for x in xs) / len(xs), 3)  # noqa: E731
    return {
        "n": len(rs),
        "drawn": round(sum(n) / len(n), 2),
        **{f"glyph_{m}": mean(g, k) for k, m in enumerate("prf")},
        **{f"jamo_{m}": mean(j, k) for k, m in enumerate("prf")},
        "jamo_f1_other": mean(o, 2),
    }


def sheet(rows: list, arms: list, recs: dict, fn: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    from reseed import OUT

    big = ImageFont.truetype(str(FONT), 72)
    small = ImageFont.truetype(str(FONT), 24)
    W = HEAD_W + CELL * len(arms)
    H = LABEL_H + (CELL + LABEL_H) * len(rows)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for j, arm in enumerate(arms):
        d.text((HEAD_W + j * CELL + 8, 8), arm.removeprefix("jamo_"), "black", small)
    for i, t in enumerate(rows):
        y = LABEL_H + i * (CELL + LABEL_H)
        f = big if len(t) == 1 else small
        d.text((10, y + CELL // 2 - 40), t, "black", f)
        for j, arm in enumerate(arms):
            x = HEAD_W + j * CELL
            r = recs[(arm, t)]
            src = OUT / RENDER[arm][0] / "render" / arm / f"bubble_{t}_s0.png"
            im.paste(Image.open(src).convert("RGB").resize((CELL, CELL)), (x, y))
            col = {"ok": OK, "near": NEAR, "miss": MISS}[mark(r)]
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=col, width=4)
            read = whole(r).replace("\n", " ")
            read = read if len(read) <= 9 else read[:8] + "…"
            d.text((x + 6, y + CELL + 6), read or "∅", col, small)
    im.save(fn)


def table(rows: list, arms: list, recs: dict) -> list:
    head = "| | " + " | ".join(a.removeprefix("jamo_") for a in arms) + " |"
    out = [head, "|---" * (len(arms) + 1) + "|"]
    for t in rows:
        cells = []
        for arm in arms:
            r = recs[(arm, t)]
            s = whole(r).replace("|", "\\|").replace("\n", " ") or "∅"
            s = s if len(s) <= 12 else s[:11] + "…"
            cells.append({"ok": f"**{s}** ✓", "near": f"{s} ~", "miss": s}[mark(r)])
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    return out


def main() -> None:
    from bench._common import make_run_dir, write_result

    p = argparse.ArgumentParser()
    p.add_argument("score_dir", help="a jamo_vl.py score run (holds reads.jsonl)")
    p.add_argument("--arms", default=ARMS)
    p.add_argument("--groups", default="H,words")
    a = p.parse_args()
    src = Path(a.score_dir)
    src = src if src.is_absolute() else HOME / src
    arms = a.arms.split(",")
    recs, order = {}, {}
    for line in open(src / "reads.jsonl"):
        r = json.loads(line)
        recs[(r["arm"], r["text"])] = r
        order.setdefault(r["group"], [])
        if r["text"] not in order[r["group"]]:
            order[r["group"]].append(r["text"])
    out = make_run_dir("cjk_anima_reseed", label="jamo-vl-sheet", root=HOME / "results")
    md = [
        f"# v4 free reads, by glyph — `{src.name}`",
        "",
        "The whole-image read; **✓** a crop reads the target exactly, "
        "~ one syllable sharing a jamo with it.",
    ]
    metrics = {}
    for g in a.groups.split(","):
        rows = order[g]
        f1 = {arm: f1_row([recs[(arm, t)] for t in rows]) for arm in arms}
        metrics[g] = f1
        for k in range(0, len(rows), PER_SHEET):
            sheet(
                rows[k : k + PER_SHEET],
                arms,
                recs,
                out / f"sheet_{g}_{k // PER_SHEET}.png",
            )
        ok = {arm: sum(mark(recs[(arm, t)]) == "ok" for t in rows) for arm in arms}
        near = {arm: sum(mark(recs[(arm, t)]) == "near" for t in rows) for arm in arms}
        md += ["", f"## {g} ({len(rows)})", ""]
        md += table(rows, arms, recs)
        md.append(
            "| **✓ / ~** | "
            + " | ".join(f"**{ok[x]} / {near[x]}**" for x in arms)
            + " |"
        )
        md += [
            "",
            f"F1 on {g} (mean per image; glyph / positioned jamo):",
            "",
            "| arm | drawn | glyph P | glyph R | glyph F1 | jamo P | jamo R | jamo F1 "
            "| jamo F1, other targets |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, v in f1.items():
            md.append(
                f"| {arm.removeprefix('jamo_')} | {v['drawn']} | "
                + " | ".join(
                    f"{v[f'{u}_{m}']:.2f}" for u in ("glyph", "jamo") for m in "prf"
                )
                + f" | {v['jamo_f1_other']:.2f} |"
            )
    write_result(
        out,
        script=__file__,
        args=vars(a),
        metrics=metrics,
        label="jamo-vl-sheet",
        artifacts=["reads.md"],
    )
    (out / "reads.md").write_text("\n".join(md) + "\n")
    print(out)


if __name__ == "__main__":
    main()
