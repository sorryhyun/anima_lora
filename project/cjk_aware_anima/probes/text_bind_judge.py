#!/usr/bin/env python3
"""OCR judge + montage for the text-binding probe (``text_bind_probe.py``).

Reads ``run_bench.py`` run dirs (cells ``<ts>_<seed>__<cond>_<arm>.png``),
runs PP-OCRv6 (``anime_tools.ocr``) over every cell and scores each against the
prompt set's ground-truth line: character error rate (CER, Levenshtein over a
NFKC-normalized, punctuation-stripped string; best over OCR lines and their
concatenation), text presence (any CJK line), and a hit at ``CER <= --hit``.
Groups runs by label minus the ``-s<seed>`` suffix (one group per probe arm)
and writes a JSON + markdown table + one montage per group
(rows = condition, cols = source | seed x render-arm).

    .venv/bin/python project/cjk_aware_anima/probes/text_bind_judge.py \
        --prompts project/cjk_aware_anima/assets/text_bind_prompts_9095721.json \
        --glob 'bench/cjk_adapter/results/*textbind-*-9095721-s*' \
        --out project/cjk_aware_anima/reports/textbind_9095721.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CELL_RE = re.compile(
    r"^(?P<ts>[\d-]+)_(?P<seed>\d+)__(?P<pid>.+?)_(?P<arm>en|[a-z]{2}_[a-z0-9_]+)\.png$"
)
STRIP = set("。、！？!?…・「」『』()（）,.~～ー-—\"'“”")
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if not ch.isspace() and ch not in STRIP)


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    ref_n, hyp_n = norm(ref), norm(hyp)
    if not ref_n:
        return 1.0
    return levenshtein(hyp_n, ref_n) / len(ref_n)


def best_cer(lines: list[str], ref: str) -> float:
    cands = list(lines) + (["".join(lines)] if len(lines) > 1 else [])
    return min((cer(c, ref) for c in cands), default=1.0)


def group_label(label: str) -> str:
    return re.sub(r"-s\d+$", "", label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--glob", default=None, help="glob of run dirs (adds to --runs)")
    ap.add_argument(
        "--out", type=Path, required=True, help="JSON path; .md + montages beside it"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hit", type=float, default=0.5, help="CER threshold for a hit")
    ap.add_argument("--thumb", type=int, default=300, help="montage cell height")
    ap.add_argument(
        "--source", type=Path, default=None, help="source image (montage col 0)"
    )
    opts = ap.parse_args()

    prompts = json.loads(opts.prompts.read_text(encoding="utf-8"))
    gt, other = prompts["_gt"], prompts.get("_other", "")
    gt_lines = prompts.get("_gt_lines") or [gt]
    expect = prompts.get("_expect", {})
    conds = [k for k in prompts if not k.startswith("_")]
    stem = prompts.get("_stem")
    source = opts.source
    if source is None and stem:
        cand = REPO / "post_image_dataset" / "resized" / "sincos" / f"{stem}.png"
        source = cand if cand.exists() else None

    runs = [Path(r) for r in opts.runs]
    if opts.glob:
        runs += [Path(p) for p in sorted(glob.glob(opts.glob))]
    runs = [r if r.is_absolute() else REPO / r for r in runs]
    if not runs:
        raise SystemExit("no run dirs")

    from anime_tools.ocr import load_ocr

    engine = load_ocr(device=opts.device, min_score=0.5, min_chars=2, skip_en=True)

    cells = []  # dicts
    for run in runs:
        meta = json.loads((run / "result.json").read_text(encoding="utf-8"))
        label = (meta.get("args") or {}).get("label") or meta.get("label") or run.name
        grp = group_label(label)
        for png in sorted(run.glob("*.png")):
            m = CELL_RE.match(png.name)
            if not m:
                continue
            cond, arm, seed = m["pid"], m["arm"], int(m["seed"])
            if cond not in conds:
                continue
            lines = [ln.text for ln in engine.read(png) if CJK_RE.search(ln.text)]
            # multi-line GT: mean over GT lines of the best-matching OCR line
            c_gt = sum(best_cer(lines, gl) for gl in gt_lines) / len(gt_lines)
            c_other = best_cer(lines, other) if other else 1.0
            c_ref = c_other if expect.get(cond) == "other" else c_gt
            cells.append(
                {
                    "group": grp,
                    "run": str(run.relative_to(REPO)),
                    "cell": png.name,
                    "cond": cond,
                    "arm": arm,
                    "seed": seed,
                    "expect": expect.get(cond),
                    "ocr": lines,
                    "cer_gt": round(c_gt, 3),
                    "cer_other": round(c_other, 3),
                    "cer_ref": round(c_ref, 3),
                    "present": bool(lines),
                    "hit": bool(lines) and c_ref <= opts.hit,
                }
            )
            print(f"{grp:32s} {cond:11s} {arm:10s} s{seed:<5d} cer={c_ref:.2f} {lines}")

    # ---- aggregate: group x arm x cond ------------------------------------
    agg = defaultdict(list)
    for c in cells:
        agg[(c["group"], c["arm"], c["cond"])].append(c)
    table = []
    for (grp, arm, cond), rows in sorted(agg.items()):
        n = len(rows)
        table.append(
            {
                "group": grp,
                "arm": arm,
                "cond": cond,
                "expect": rows[0]["expect"],
                "n": n,
                "cer_ref_mean": round(sum(r["cer_ref"] for r in rows) / n, 3),
                "cer_gt_mean": round(sum(r["cer_gt"] for r in rows) / n, 3),
                "hit_rate": round(sum(r["hit"] for r in rows) / n, 2),
                "present_rate": round(sum(r["present"] for r in rows) / n, 2),
            }
        )

    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(
        json.dumps(
            {
                "gt": gt,
                "other": other,
                "hit_threshold": opts.hit,
                "table": table,
                "cells": cells,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        f"# text-binding judge — gt `{gt}` / other `{other}` (hit = CER ≤ {opts.hit})",
        "",
        "| group | arm | cond | expect | n | CER(ref) | CER(gt) | hit | present |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for t in table:
        md.append(
            f"| {t['group']} | {t['arm']} | {t['cond']} | {t['expect']} | {t['n']} | "
            f"{t['cer_ref_mean']:.2f} | {t['cer_gt_mean']:.2f} | {t['hit_rate']:.2f} | {t['present_rate']:.2f} |"
        )
    opts.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- montages ----------------------------------------------------------
    from PIL import Image, ImageDraw

    by_group = defaultdict(list)
    for c in cells:
        by_group[c["group"]].append(c)
    for grp, rows in by_group.items():
        seeds = sorted({c["seed"] for c in rows})
        arms = sorted({c["arm"] for c in rows}, key=lambda a: (a != "ja_ext", a))
        cols = [("source", None, None)] + [
            (f"s{s} {a}", s, a) for s in seeds for a in arms
        ]
        lookup = {(c["cond"], c["seed"], c["arm"]): c for c in rows}
        th = opts.thumb
        # cell width from the first image's aspect
        first = next(iter(rows))
        w0, h0 = Image.open(REPO / first["run"] / first["cell"]).size
        tw = int(round(th * w0 / h0))
        pad, head, left = 6, 22, 110
        W = left + len(cols) * (tw + pad)
        H = head + len(conds) * (th + head)
        sheet = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (name, _, _) in enumerate(cols):
            draw.text((left + j * (tw + pad), 4), name, fill="black")
        src_img = Image.open(source).convert("RGB") if source else None
        for i, cond in enumerate(conds):
            y = head + i * (th + head)
            draw.text((4, y + th // 2), cond, fill="black")
            for j, (_, seed, arm) in enumerate(cols):
                x = left + j * (tw + pad)
                if seed is None:
                    if src_img is not None:
                        sheet.paste(src_img.resize((tw, th)), (x, y))
                    continue
                c = lookup.get((cond, seed, arm))
                if c is None:
                    continue
                im = (
                    Image.open(REPO / c["run"] / c["cell"])
                    .convert("RGB")
                    .resize((tw, th))
                )
                sheet.paste(im, (x, y))
                tag = (
                    f"cer={c['cer_ref']:.2f}"
                    + (" HIT" if c["hit"] else "")
                    + ("" if c["present"] else " (no text)")
                )
                draw.text((x + 2, y + th + 2), tag, fill="red" if c["hit"] else "black")
        out_png = opts.out.with_name(f"{opts.out.stem}_montage_{grp}.png")
        sheet.save(out_png)
        print(f"montage -> {out_png}")


if __name__ == "__main__":
    main()
