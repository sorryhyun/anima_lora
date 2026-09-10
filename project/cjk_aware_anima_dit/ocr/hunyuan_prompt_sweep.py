#!/usr/bin/env python3
"""Sweep HunyuanOCR-1.5's instruction over both evals, one model load.

    make daemon-run ARGS="project/cjk_aware_anima_dit/ocr/hunyuan_prompt_sweep.py"
    … --arms zh_official,ja_verbatim --coo_limit 0       # subset / sincos only

`findings.md` § "Outside reader — HunyuanOCR-1.5" found the instruction to be
the largest lever on this model: under upstream's official Chinese crop prompt
**51 %** of its COO SFX reads carry no kana at all, and a bare Japanese
instruction drops that to 18 % (+3.5 SFX / +5.6 speech exact). That was two
points, not a curve — this sweeps the space around them.

Every arm is **off-recipe by construction**: upstream ships a fixed prompt per
task and no free-form one ("users pick a task, not a prompt"), so `zh_official`
is the only row that is HunyuanOCR as its authors ship it and the rest measure
how far a prompt can carry it. Arms are scored on the **sincos SFX gate** (the
doujin set this line is for) and on a COO test subset (in-domain, both kinds)
so a prompt tuned to one surface shows up on the other.

Writes `reports/0908_hunyuan_prompt_sweep.md` + one jsonl per arm under
`output/ocr/eval/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import eval_sfx as esfx  # noqa: E402
import manga109 as m109  # noqa: E402

ARMS = {
    # upstream's own crop task — the only shipped row (inference/utils/tasks.py)
    "zh_official": "提取图中的文字。",
    # the two-line probe that showed the language prior is promptable
    "ja_plain": "画像中の日本語のテキストを抽出してください。",
    # name the medium: does "manga" buy anything over "Japanese"?
    "ja_manga": "日本の漫画の一行です。描かれている文字をそのまま書き写してください。",
    # name the *kind* of line — the SFX half of the gate
    "ja_sfx": "日本の漫画の擬音語・描き文字です。読める通りに仮名で書き写してください。",
    # forbid the rewrite, and name the glyphs it drops (small kana, っ, ー)
    "ja_verbatim": (
        "画像の文字を、小さい仮名・促音「っ」・長音符「ー」も含めて"
        "そのままの表記で書き写してください。翻訳や言い換えはしないでください。"
    ),
    # the shortest possible Japanese instruction — is length itself a factor?
    "ja_short": "画像の文字をそのまま書き写してください。",
    # the language hint inside upstream's own Chinese wording
    "zh_ja_hint": "提取图中的日文文字，原样输出，不要翻译或改写。",
    # --- round 2: is the working token 「日本語」 itself? Single-variable pairs
    # against `ja_plain` (which has it) and `ja_manga` / `ja_short` (which
    # name the medium or nothing but never the language).
    "ja_plain_nolang": "画像中のテキストを抽出してください。",  # ja_plain minus 日本語
    "ja_manga_lang": "日本の漫画の一行です。画像中の日本語のテキストをそのまま書き写してください。",
    "ja_short_lang": "画像中の日本語のテキストをそのまま書き写してください。",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arms", help="comma-separated subset of ARMS (default: all)")
    ap.add_argument("--coo_limit", type=int, default=400, help="COO crops per kind")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    arms = {k: ARMS[k] for k in (a.arms.split(",") if a.arms else ARMS)}

    # sincos SFX — the gate
    lab = esfx.load_labels(["sfx"], False)
    sin_crops, sin_orients = esfx.crops_for(lab, 0.12)
    lab["orient"] = sin_orients

    # COO test subset — in-domain control, both kinds
    coo = pd.DataFrame()
    coo_crops: list = []
    if a.coo_limit:
        derived = m109.derived_root()
        df = pd.read_parquet(derived / "manifest.parquet")
        df = df[df.split == "test"]
        coo = (
            df.groupby("kind", group_keys=False)
            .head(a.coo_limit)
            .sort_values(["kind", "book", "page", "id"])
            .reset_index(drop=True)
        )
        import cv2

        coo_crops = [cv2.imread(str(derived / p)) for p in coo.path]
        assert all(c is not None for c in coo_crops)

    reader = ev.READERS["hunyuan"](None, a.device)
    rows = []
    for name, prompt in arms.items():
        reader.prompt = prompt
        t0 = time.time()
        sin = ev.score(lab, reader.read(sin_crops, sin_orients, a.bs))
        row = {
            "arm": name,
            "prompt": prompt,
            "sincos_n": len(sin),
            "sincos_exact": int(sin.exact.sum()),
            "sincos_sim": sin.sim.mean(),
            "sincos_nokana": nokana(sin),
        }
        sin.drop(columns=["box"]).to_json(
            ev.OUT / f"sfx_hunyuan_sweep_{name}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        if len(coo):
            c = ev.score(coo, reader.read(coo_crops, list(coo.orient), a.bs))
            for kind, g in c.groupby("kind"):
                row[f"coo_{kind}_exact"] = 100 * g.exact.mean()
                row[f"coo_{kind}_sim"] = g.sim.mean()
                row[f"coo_{kind}_nokana"] = nokana(g)
            row["coo_runaway"] = int(c.runaway.sum())
            c.to_json(
                ev.OUT / f"hunyuan_sweep_{name}_coo.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )
        row["wall"] = time.time() - t0
        rows.append(row)
        print(f"{name}: {row}", flush=True)

    out = pd.DataFrame(rows)
    md = ["# HunyuanOCR-1.5 — instruction sweep (2026-09-08)\n"]
    md.append(
        f"`ocr/hunyuan_prompt_sweep.py`, one model load, {len(arms)} arms. "
        f"sincos SFX n={len(lab)} (the gate); COO test subset "
        f"{a.coo_limit}/kind (in-domain control). `no-kana` = predictions "
        "containing no kana at all — the Chinese-prior failure.\n"
    )
    md.append(
        "| arm | sincos exact | sincos sim | no-kana | COO sfx | COO speech | COO no-kana (sfx) |"
    )
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| `{r['arm']}` | {r['sincos_exact']} / {r['sincos_n']} "
            f"({100 * r['sincos_exact'] / r['sincos_n']:.1f} %) | {r['sincos_sim']:.3f} | "
            f"{r['sincos_nokana']:.0f} % | "
            f"{r.get('coo_sfx_exact', float('nan')):.1f} % | "
            f"{r.get('coo_speech_exact', float('nan')):.1f} % | "
            f"{r.get('coo_sfx_nokana', float('nan')):.0f} % |"
        )
    md.append("\n## Prompts\n")
    for r in rows:
        md.append(f"- `{r['arm']}` — {r['prompt']}")
    (ev.REPORTS / "0908_hunyuan_prompt_sweep.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print("\n".join(md))
    out.to_json(ev.OUT / "hunyuan_prompt_sweep.jsonl", orient="records", lines=True)


def nokana(df) -> float:
    """Share of non-empty predictions with no kana — the Chinese-prior tell."""
    import re

    kana = re.compile(r"[ぁ-ゖァ-ー]")
    return 100 * sum(1 for p in df.pred if p and not kana.search(p)) / max(len(df), 1)


if __name__ == "__main__":
    main()
