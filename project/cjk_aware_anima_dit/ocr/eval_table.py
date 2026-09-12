#!/usr/bin/env python3
"""Regenerate ``project/cjk_aware_anima_dit/eval.md`` — every reader on one basis.

    python project/cjk_aware_anima_dit/ocr/eval_table.py           # print
    python project/cjk_aware_anima_dit/ocr/eval_table.py --write   # write eval.md

CPU only, no model: every number is re-derived from the stored ``pred_norm`` +
``text`` in ``output/ocr/eval/*.jsonl`` through the **current** ``exact_key``,
the same way :mod:`rescore_eval` does. That is the whole point of the file —
the per-run ``reports/ocr_eval_*.md`` were each written on whatever key and
label basis was live that day, so their headline numbers cannot be read down a
column (see § Comparability in ``eval.md``).

Headline is **♡-blind**: hearts are stripped from both sides before the
comparison. 80.6 % of sincos SFX labels carry a ``♡`` and the captions these
reads feed drop the symbol anyway, so strict exact mostly measures heart
agreement with a label set whose ``text_hand`` came from a heart-blind reader.
Strict exact is kept beside it because ``findings.md`` quotes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import textnorm  # noqa: E402
from rescore_eval import rescore  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OUT_MD = HERE / "eval.md"


def noheart(t: str) -> str:
    return ev.exact_key(t).replace("♡", "").replace("♥", "")


def space_key(t) -> str:
    """``exact_key``'s folds with whitespace *collapsed* rather than deleted, so
    a reader that cannot emit a space (every ``TARGET_NORM = 1`` run —
    ``whitespace_fixed.md``) fails the rows whose target has one."""
    return textnorm.normalize_target(t if isinstance(t, str) else "")


def score(path: Path):
    if not path.is_file():
        return None
    df = rescore(path)  # df.exact := strict, on the current key
    df["exact_nh"] = [noheart(p) == noheart(t) for p, t in zip(df.pred_norm, df.text)]
    df["spaced"] = [" " in space_key(t) for t in df.text]
    df["exact_sp"] = [
        space_key(p) == space_key(t) for p, t in zip(df.pred_norm, df.text)
    ]
    return df


# (run dir, sincos stem, COO stem, note) — run dir doubles as the display name
RUNS = [
    ("vl16_stock", "vl16_stock", None, "stock VL-1.6, no fine-tune"),
    ("manga_ocr_stock", "manga_ocr_stock", None, "stock manga-ocr"),
    ("mocr_lr5e-5", "mocr_lr5e-5", None, "manga-ocr fine-tuned"),
    ("vl16_lr1e-4", "vl16_lr1e-4", "vl16_lr1e-4", "arm B — LoRA, tower frozen"),
    (
        "vl16_tower_lr1e-5",
        "vl16_tower_lr1e-5",
        "vl16_tower_lr1e-5",
        "**arm B′** — LoRA + tower unfrozen; the published reader",
    ),
    (
        "vl16_tower_col100",
        "vl16_tower_col100",
        "vl16_tower_col100",
        "B′ + 1.6 % colorized append",
    ),
    (
        "vl16_tower_col1500sw",
        "vl16_tower_col1500sw",
        "vl16_tower_col1500sw",
        "B′ + 22.3 % colorized swap",
    ),
    (
        "vl16_pl_20k",
        "vl16_pl_20k",
        "vl16_pl_20k",
        "B′ + 20.6 % pseudo-label append (P1, cross-reader agreement)",
    ),
    (
        "vl16_pl_kozh",
        "vl16_pl_kozh",
        "vl16_pl_kozh",
        "B′ + 27.5 % pseudo append (P1 JA 20k + K2 KO 5 930 / ZH 3 300)",
    ),
    (
        "vl16_b2_norm2",
        "vl16_b2_norm2",
        "vl16_b2_norm2",
        "B′ recipe verbatim under TARGET_NORM 2 (plan_vl_respace R2)",
    ),
    (
        "vl16_b2_norm3",
        "vl16_b2_norm3",
        "vl16_b2_norm3",
        "norm2 + glyph fold, heart → `♡` (TARGET_NORM 3, R5) — lost every raw `♥`",
    ),
    (
        "vl16_b2_norm4",
        "vl16_b2_norm4",
        "vl16_b2_norm4",
        "norm2 + glyph fold, heart → `♥` (TARGET_NORM 4, R5b) — **Hub v3**; "
        "sincos on the 2026-09-12 corrected labels (rows above: pre-fix)",
    ),
    ("vl16_tower_ep3", "vl16_tower_ep3", "vl16_tower_ep3", "B′ × 3 epochs"),
    ("vl16_lpft", "vl16_lpft", "vl16_lpft", "LP-FT — arm B then B′"),
    (
        "vl16_tower_ssl",
        "vl16_tower_ssl",
        "vl16_tower_ssl",
        "B′ from an SSL tower (draw20k, 4.4k steps)",
    ),
    (
        "vl16_tower_ssl_all",
        "vl16_tower_ssl_all",
        "vl16_tower_ssl_all",
        "B′ from an SSL tower (manifest_all, 12k steps)",
    ),
    (
        "vl16_tower_ssl_all_lr5e5",
        "vl16_tower_ssl_all_lr5e5",
        "vl16_tower_ssl_all_lr5e5",
        "same tower, LoRA lr 5e-5",
    ),
    (
        "hayai_v2_1_5",
        "hayai_v2_1_5",
        None,
        "hayai v2.1.5 sidecar (~1/6 the parameters)",
    ),
    ("sfx_pkg", "sfx_pkg", None, "shipped `anime_tools.ocr.sfx` (B′ + decode guard)"),
]


def val_sfx(run: str):
    p = Path(f"output/ocr/{run}/history.jsonl")
    if not p.is_file():
        return None
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("tag") != "stock" and "sfx_exact" in r]
    return max(rows, key=lambda r: r["sfx_exact"])["sfx_exact"] if rows else None


def table() -> tuple[str, float, float]:
    head = (
        "| reader | sincos SFX ♡-blind | strict | COO SFX ♡-blind | COO speech ♡-blind | COO spaced | in-domain val | note |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for run, gname, cname, note in RUNS:
        g = score(ev.OUT / f"sfx_{gname}.jsonl")
        c = score(ev.OUT / f"{cname}_test.jsonl") if cname else None
        gcol = scol = cs = cp = sp = "—"
        if c is not None and c.spaced.any():
            cs_ = c[c.spaced]
            sp = f"{int(cs_.exact_sp.sum())} / {len(cs_)}"
        if g is not None and (g.kind == "sfx").any():
            gs = g[g.kind == "sfx"]
            nh, n = int(gs.exact_nh.sum()), len(gs)
            gcol, scol = (
                f"**{nh}** / {n} ({100 * nh / n:.1f} %)",
                str(int(gs.exact.sum())),
            )
        if c is not None:
            for k in ("sfx", "speech"):
                if (c.kind == k).any():
                    gk = c[c.kind == k]
                    nh, n = int(gk.exact_nh.sum()), len(gk)
                    v = f"{nh} / {n} ({100 * nh / n:.1f} %)"
                    cs, cp = (v, cp) if k == "sfx" else (cs, v)
        v = val_sfx(run)
        vcol = f"{100 * v:.1f} %" if v is not None else "—"
        lines.append(
            f"| `{run}` | {gcol} | {scol} | {cs} | {cp} | {sp} | {vcol} | {note} |"
        )
    b = score(ev.OUT / "sfx_vl16_tower_lr1e-5.jsonl")
    cb = score(ev.OUT / "vl16_tower_lr1e-5_test.jsonl")
    hs = 100 * b[b.kind == "sfx"].text.str.contains("♡|♥").mean()
    hc = 100 * cb[cb.kind == "sfx"].text.str.contains("♡|♥").mean()
    return "\n".join(lines), hs, hc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    tbl, hs, hc = table()
    body = OUT_MD.read_text(encoding="utf-8") if OUT_MD.is_file() else ""
    if a.write and "<!-- TABLE -->" in body:
        pre, _, rest = body.partition("<!-- TABLE -->")
        _, _, post = rest.partition("<!-- /TABLE -->")
        OUT_MD.write_text(
            f"{pre}<!-- TABLE -->\n{tbl}\n<!-- /TABLE -->{post}", encoding="utf-8"
        )
        print(f"wrote {OUT_MD}")
    else:
        print(tbl)
        print(f"\nheart share — sincos SFX {hs:.1f} % | COO SFX {hc:.1f} %")


if __name__ == "__main__":
    main()
