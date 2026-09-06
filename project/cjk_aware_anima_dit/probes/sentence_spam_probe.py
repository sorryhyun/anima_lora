#!/usr/bin/env python3
"""Spam-direction probe for the sentence caption shape (plan_base1 B3 readout).

Does prompting the *address* of a page's text make an arm render more text?
Four CJK-free eval prompts × three conditions, rendered through the vocab pack
(``bench/cjk_adapter/run_bench.py --ext``, arm ``ja_ext``) for each LoRA arm
and seed, then PP-OCRv6 counts the text it can detect in every render:

    plain      the base prompt (no text address)
    tags       ``…, japanese text, 「<line>」``      — the C2–C9 caption shape
    sentence   ``…. Japanese text reads as "<line>".`` — the C10 shape (B2)

Not a quality metric: a spam-direction check (glyph area up or down vs the
arm's own ``plain``). Stages run as direct subprocesses (never nested daemon
jobs)::

    make daemon-run ARGS="--label sentprobe --stall-timeout 0 --queue \
        project/cjk_aware_anima_dit/probes/sentence_spam_probe.py --stage all"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE_DIR = HERE.parent
REPO = LINE_DIR.parents[1]
PY = sys.executable

ARMS = {"C10": "cjk_unmask_c10", "C9ISOQ": "cjk_unmask_c9_isoq"}
EXT_PREFIX = "output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256_isoq"
EVAL_PROMPTS = REPO / "project/cjk_aware_anima/assets/unmask_eval_prompts.txt"
BASE_ROWS = (0, 3, 5, 6)  # classroom, park bench, maid cafe, portrait
CONDS = ("plain", "tags", "sentence")


def build_prompts(line: str) -> dict[str, dict[str, str]]:
    from anime_tools.captions.position_clauses import compose_caption, text_clause

    rows = [ln.strip() for ln in EVAL_PROMPTS.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r]
    out: dict[str, dict[str, str]] = {}
    for i in BASE_ROWS:
        base = rows[i]
        tags = [t.strip() for t in base.split(",")]
        variants = {
            "plain": base,
            "tags": compose_caption([*tags, "japanese text", f"「{line}」"]),
            "sentence": compose_caption(tags, [text_clause([line])]),
        }
        for cond, text in variants.items():
            # ja_ext reads the ``ja`` field (T5 side through the pack); ``en``
            # is required by the bench's arm table and unused here.
            out[f"r{i + 1}_{cond}"] = {"en": text, "ja": text}
    return out


def run(stage: str, argv: list[str]) -> None:
    print(f"\n=== [{stage}] {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO, check=True)


def run_dirs(label: str) -> list[Path]:
    return sorted((REPO / "bench/cjk_adapter/results").glob(f"*-{label}"))


def render(opts, prompts_path: Path) -> None:
    for arm, method in ARMS.items():
        if arm not in opts.arms:
            continue
        for seed in opts.seeds:
            label = f"sentprobe-{arm}-s{seed}"
            if run_dirs(label) and not opts.overwrite:
                print(f"=== {label} already rendered, skip", flush=True)
                continue
            run(
                f"render {arm} s{seed}",
                [
                    PY,
                    "bench/cjk_adapter/run_bench.py",
                    "--ext",
                    "--ext_prefix",
                    opts.ext_prefix,
                    "--arms",
                    "ja_ext",
                    "--languages",
                    "ja",
                    "--prompts",
                    str(prompts_path),
                    "--lora",
                    f"output/ckpt/{method}.safetensors",
                    "--seed",
                    str(seed),
                    "--size",
                    "1024",
                    "1024",
                    "--steps",
                    "28",
                    "--cfg",
                    "4.0",
                    "--label",
                    label,
                ],
            )


def count(opts) -> None:
    from PIL import Image

    from anime_tools.ocr._onnx import load_ocr

    # Lenient: every detected box counts, ASCII too — spam is spam.
    engine = load_ocr(device="cpu", min_score=0.3, min_chars=1, skip_en=False)
    cells: list[dict] = []
    for arm in opts.arms:
        for seed in opts.seeds:
            dirs = run_dirs(f"sentprobe-{arm}-s{seed}")
            if not dirs:
                print(f"missing renders for {arm} s{seed}", flush=True)
                continue
            for png in sorted(dirs[-1].glob("*.png")):
                # <ts>_<seed>__<content>_ja_ext.png
                content = png.stem.split("__", 1)[1].removesuffix("_ja_ext")
                row, cond = content.rsplit("_", 1)
                w, h = Image.open(png).size
                lines = engine.read(png)
                area = sum(
                    max(0, x1 - x0) * max(0, y1 - y0)
                    for (x0, y0, x1, y1) in (ln.box for ln in lines)
                )
                cells.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "row": row,
                        "cond": cond,
                        "png": str(png.relative_to(REPO)),
                        "n_lines": len(lines),
                        "glyph_frac": area / (w * h),
                        "chars": sum(len(ln.text) for ln in lines),
                        "texts": [ln.text for ln in lines],
                    }
                )
    summary: dict[str, dict] = {}
    for arm in opts.arms:
        for cond in CONDS:
            sel = [c for c in cells if c["arm"] == arm and c["cond"] == cond]
            if not sel:
                continue
            summary[f"{arm}/{cond}"] = {
                "n": len(sel),
                "glyph_frac_mean": sum(c["glyph_frac"] for c in sel) / len(sel),
                "lines_mean": sum(c["n_lines"] for c in sel) / len(sel),
                "any_text": sum(1 for c in sel if c["n_lines"]) / len(sel),
                "chars_mean": sum(c["chars"] for c in sel) / len(sel),
            }
    out_json = REPO / opts.out
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {"line": opts.line, "summary": summary, "cells": cells},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    md = [
        f"# Sentence-shape spam-direction probe (line `{opts.line}`)",
        "",
        "PP-OCRv6 (lenient: score ≥ 0.3, 1+ char, ASCII kept) on every render;",
        "`glyph_frac` = detected box area / image area. 4 prompts × 3 seeds per cell.",
        "",
        "| arm / condition | n | glyph area % | lines / image | images with text | chars / image |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in summary.items():
        md.append(
            f"| {k} | {v['n']} | {100 * v['glyph_frac_mean']:.2f} | {v['lines_mean']:.2f} "
            f"| {100 * v['any_text']:.0f} % | {v['chars_mean']:.1f} |"
        )
    md += [
        "",
        "## Per cell",
        "",
        "| arm | seed | row | cond | lines | glyph % | OCR read |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in sorted(
        cells, key=lambda c: (c["arm"], c["row"], CONDS.index(c["cond"]), c["seed"])
    ):
        md.append(
            f"| {c['arm']} | {c['seed']} | {c['row']} | {c['cond']} | {c['n_lines']} "
            f"| {100 * c['glyph_frac']:.2f} | {' · '.join(c['texts'])[:60]} |"
        )
    out_json.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[: 6 + len(summary)]), flush=True)
    print(f"\n-> {out_json} / .md", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=("render", "count", "all"), default="all")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1234])
    ap.add_argument("--ext_prefix", default=EXT_PREFIX)
    ap.add_argument(
        "--line",
        default="温水くん、私と",
        help="the quoted line every text condition carries",
    )
    ap.add_argument("--prompts", default=str(HERE / "sentence_spam_prompts.json"))
    ap.add_argument(
        "--out", default="project/cjk_aware_anima_dit/reports/sentence_spam_probe.json"
    )
    ap.add_argument("--overwrite", action="store_true")
    opts = ap.parse_args()
    for arm in opts.arms:
        if arm not in ARMS:
            sys.exit(f"unknown arm {arm}; known: {list(ARMS)}")
    prompts_path = Path(opts.prompts)
    if opts.stage in ("render", "all"):
        prompts_path.write_text(
            json.dumps(build_prompts(opts.line), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(
            f"prompts -> {prompts_path} ({len(BASE_ROWS)} rows × {len(CONDS)} conditions)"
        )
        render(opts, prompts_path)
    if opts.stage in ("count", "all"):
        count(opts)


if __name__ == "__main__":
    main()
