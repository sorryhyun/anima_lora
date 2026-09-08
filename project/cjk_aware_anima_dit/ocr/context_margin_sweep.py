#!/usr/bin/env python3
"""Does *context* help the crop reader? Margin / marker / page-text sweep, one job.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/cjk_aware_anima_dit/ocr/context_margin_sweep.py"
    … --readers hunyuan --pads 0.12,0.7            # subset

The question behind it (2026-09-08): the 12 %-pad crop looks too tight to
*infer* a hard SFX from, so one proposal is to feed the reader a page-level
feature (PE-Spatial / PE-Core) as context. Before building a modality bridge,
measure whether context carries any signal at all, through the reader's own
tower — the cheapest carrier there is:

* **margin** — the same ``deskew_crop`` at a wider ``pad`` (fraction of the
  box's long edge, per side: 0.12 = today's crop; 0.7 ≈ 2.4× the box; 1.5 ≈ 4×,
  panel-ish). A reader with a fixed task prompt (VL-1.6 ``OCR:``) reads
  everything in the frame, so ``contains`` (label ⊂ prediction, on
  ``exact_key``) is the fair metric beside ``exact``.
* **marker** — the 12 %-pad box drawn in red on the wide crop, and (Hunyuan
  only — VL-1.6 has no free-form prompt) an instruction to read only inside it.
* **page text** (COO only, Hunyuan only) — the page's Manga109 ``<text>``
  lines quoted in the prompt: *oracle* textual context, an upper bound on
  "the surrounding dialogue disambiguates the SFX".

Surfaces: the sincos gate (617 hand-labelled SFX, out-of-domain) and a COO
test subset (in-domain, both kinds), as ``hunyuan_prompt_sweep.py``. Readers:
stock VL-1.6, B′ (``vl16_tower_lr1e-5`` — trained on 12 % crops, so a wider
frame is off-distribution for it; reported, not expected to win), Hunyuan-1.5
under the ``ja_plain`` prompt.

Writes ``reports/0908_context_margin_sweep.md`` + one jsonl per (reader, arm)
under ``output/ocr/eval/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import eval_sfx as esfx  # noqa: E402
import manga109 as m109  # noqa: E402

INNER_PAD = 0.12  # today's crop
LONG_EDGE_CAP = 1024  # a 1.5-pad crop of a big SFX would be > 1 MP; context is coarse
READERS = {
    "vl16": (None, "OCR:"),
    "vl16_bprime": ("output/ocr/vl16_tower_lr1e-5/best", "OCR:"),
    "hunyuan": (None, "画像中の日本語のテキストを抽出してください。"),
}
MARKED_PROMPT = "赤い枠の中の日本語のテキストだけを、そのまま書き写してください。"
PAGE_TEXT_PROMPT = (
    "これは日本の漫画の一部です。同じページのセリフ: {lines}\n"
    "赤い枠の中の日本語のテキストだけを、そのまま書き写してください。"
)


def context_crop(img, poly, pad: float, marker: bool):
    """``deskew_crop``'s rectangle rule (orientation preserved, angle swap) at
    ``pad`` per side; with ``marker`` the ``INNER_PAD`` box is drawn in red."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if angle > 45:
        angle -= 90
        w, h = h, w
    w, h = max(w, 1.0), max(h, 1.0)
    m = max(w, h)
    W, H = w + 2 * pad * m, h + 2 * pad * m
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    ih, iw = img.shape[:2]
    rotated = cv2.warpAffine(
        img, rot, (iw, ih), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    crop = cv2.getRectSubPix(rotated, (int(round(W)), int(round(H))), (cx, cy))
    ch, cw = crop.shape[:2]
    if marker:
        iw_, ih_ = w + 2 * INNER_PAD * m, h + 2 * INNER_PAD * m
        x0, y0 = int(round((cw - iw_) / 2)), int(round((ch - ih_) / 2))
        x1, y1 = int(round((cw + iw_) / 2)), int(round((ch + ih_) / 2))
        t = max(2, int(round(0.008 * max(cw, ch))))
        cv2.rectangle(crop, (x0 - t, y0 - t), (x1 + t, y1 + t), (0, 0, 255), t)
    if max(cw, ch) > LONG_EDGE_CAP:
        s = LONG_EDGE_CAP / max(cw, ch)
        crop = cv2.resize(
            crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_AREA
        )
    # orient off the *inner* box, as deskew_crop would report for the 12 % crop
    iw_, ih_ = w + 2 * INNER_PAD * m, h + 2 * INNER_PAD * m
    orient = (
        "vertical"
        if ih_ > iw_ * 1.3
        else ("horizontal" if iw_ > ih_ * 1.3 else "square")
    )
    return crop, orient


def crops_at(pages: list[Path], polys: list[list[float]], pad: float, marker: bool):
    cache: dict[Path, np.ndarray] = {}
    crops, orients = [], []
    for p, poly in zip(pages, polys):
        if p not in cache:
            cache[p] = cv2.imread(str(p))
            assert cache[p] is not None, p
        c, o = context_crop(cache[p], poly, pad, marker)
        crops.append(c)
        orients.append(o)
    return crops, orients


def nokana(df: pd.DataFrame) -> float:
    return 100 * df.pred.map(lambda t: not any(ev.is_kana(c) for c in t)).mean()


def contains(scored: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            ev.exact_key(t) != "" and ev.exact_key(t) in ev.exact_key(pn)
            for pn, t in zip(scored.pred_norm, scored.text)
        ],
        index=scored.index,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--readers", default=",".join(READERS))
    ap.add_argument("--pads", default="0.12,0.35,0.7,1.5")
    ap.add_argument("--coo_limit", type=int, default=400, help="COO crops per kind")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    pads = [float(p) for p in a.pads.split(",")]

    # sincos gate — box → 4-point poly on the resized page
    lab = esfx.load_labels(["sfx"], False)
    sin_pages = [esfx.PAGES / f"{s}.png" for s in lab.stem]
    sin_polys = [[x0, y0, x1, y0, x1, y1, x0, y1] for x0, y0, x1, y1 in lab.box]

    # COO test subset — re-cropped from the page at each pad (the manifest's
    # stored crops are the 0.12 row)
    coo = pd.DataFrame()
    coo_pages: list[Path] = []
    coo_polys: list[list[float]] = []
    page_text: list[str] = []
    if a.coo_limit:
        df = pd.read_parquet(m109.derived_root() / "manifest.parquet")
        df = df[df.split == "test"]
        coo = (
            df.groupby("kind", group_keys=False)
            .head(a.coo_limit)
            .sort_values(["kind", "book", "page", "id"])
            .reset_index(drop=True)
        )
        coo_pages = [m109.page_path(b, p) for b, p in zip(coo.book, coo.page)]
        coo_polys = [json.loads(p) for p in coo.poly]
        speech: dict[tuple[str, int], list[str]] = {}
        for book in sorted(set(coo.book)):
            for ln in m109.iter_text(book):
                if ln.text:
                    speech.setdefault((book, ln.page), []).append(ln.text)
        page_text = [
            "".join(f"「{t}」" for t in speech.get((b, p), [])[:12]) or "（なし）"
            for b, p in zip(coo.book, coo.page)
        ]

    rows = []
    for rname in a.readers.split(","):
        ckpt, base_prompt = READERS[rname]
        reader = ev.READERS["vl16" if rname.startswith("vl16") else "hunyuan"](
            ckpt, a.device
        )
        free_prompt = rname == "hunyuan"
        if free_prompt:
            reader.prompt = base_prompt

        arms: list[tuple[str, float, bool, str | None]] = []
        for pad in pads:
            arms.append((f"pad{pad:g}", pad, False, None))
            if pad > INNER_PAD:
                arms.append((f"pad{pad:g}_box", pad, True, MARKED_PROMPT))
        if free_prompt and len(coo):
            arms.append(("pad0.12_pagetext", INNER_PAD, True, PAGE_TEXT_PROMPT))

        for arm, pad, marker, prompt in arms:
            bs = a.bs if pad < 0.7 else max(1, a.bs // 4)
            t0 = time.time()
            row: dict = {"reader": rname, "arm": arm, "pad": pad, "marker": marker}
            if arm != "pad0.12_pagetext":
                if free_prompt:
                    reader.prompt = prompt or base_prompt
                crops, orients = crops_at(sin_pages, sin_polys, pad, marker)
                lab["orient"] = orients
                sin = ev.score(lab, reader.read(crops, orients, bs))
                sin["contains"] = contains(sin)
                row.update(
                    sincos_n=len(sin),
                    sincos_exact=int(sin.exact.sum()),
                    sincos_contains=int(sin.contains.sum()),
                    sincos_sim=sin.sim.mean(),
                    sincos_runaway=int(sin.runaway.sum()),
                    sincos_nokana=nokana(sin),
                )
                sin.drop(columns=["box"]).to_json(
                    ev.OUT / f"sfx_ctx_{rname}_{arm}.jsonl",
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )
            if len(coo):
                crops, orients = crops_at(coo_pages, coo_polys, pad, marker)
                if arm == "pad0.12_pagetext":
                    # per-crop prompt: read one at a time through the reader's
                    # prompt attribute (bs 1 keeps the template per image)
                    preds = []
                    for c, o, pt in zip(crops, orients, page_text):
                        reader.prompt = PAGE_TEXT_PROMPT.format(lines=pt)
                        preds += reader.read([c], [o], 1)
                    reader.prompt = base_prompt
                else:
                    preds = reader.read(crops, orients, bs)
                c = ev.score(coo, preds)
                c["contains"] = contains(c)
                for kind, g in c.groupby("kind"):
                    row[f"coo_{kind}_exact"] = 100 * g.exact.mean()
                    row[f"coo_{kind}_contains"] = 100 * g.contains.mean()
                    row[f"coo_{kind}_sim"] = g.sim.mean()
                row["coo_runaway"] = int(c.runaway.sum())
                row["coo_sfx_nokana"] = nokana(c[c.kind == "sfx"])
                c.to_json(
                    ev.OUT / f"ctx_{rname}_{arm}_coo.jsonl",
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )
            row["wall"] = time.time() - t0
            rows.append(row)
            print(f"{rname} {arm}: {row}", flush=True)
        del reader
        import torch

        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    md = ["# Context sweep — margin / marker / page text (2026-09-08)\n"]
    md.append(
        f"`ocr/context_margin_sweep.py`, readers {a.readers}, pads {a.pads}. "
        f"sincos SFX n={len(lab)} (the gate); COO test subset {a.coo_limit}/kind. "
        "`pad` = margin per side as a fraction of the box's long edge (0.12 = "
        "today's crop); `_box` = the 12 % box drawn in red (+ a read-inside-the-"
        "box instruction on Hunyuan); `pagetext` = the page's Manga109 `<text>` "
        "lines quoted in the prompt (oracle). `contains` = label ⊂ prediction "
        "on `exact_key` — the fair metric for a reader that reads the whole "
        "frame; `no-kana` = predictions with no kana at all.\n"
    )
    md.append(
        "| reader | arm | sincos exact | contains | sim | runaway | no-kana "
        "| COO sfx exact | contains | COO speech exact | contains | COO runaway | wall |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        g = lambda k, f="{:.1f}": f.format(r[k]) if k in r and pd.notna(r[k]) else "—"  # noqa: E731
        md.append(
            f"| `{r.reader}` | `{r.arm}` | {g('sincos_exact', '{:.0f}')} | "
            f"{g('sincos_contains', '{:.0f}')} | {g('sincos_sim', '{:.3f}')} | "
            f"{g('sincos_runaway', '{:.0f}')} | {g('sincos_nokana')} % | "
            f"{g('coo_sfx_exact')} % | {g('coo_sfx_contains')} % | "
            f"{g('coo_speech_exact')} % | {g('coo_speech_contains')} % | "
            f"{g('coo_runaway', '{:.0f}')} | {r.wall:.0f} s |"
        )
    md.append("\n## Prompts\n")
    md.append(f"- Hunyuan base — {READERS['hunyuan'][1]}")
    md.append(f"- `_box` — {MARKED_PROMPT}")
    md.append(f"- `pagetext` — {PAGE_TEXT_PROMPT!r}")
    ev.REPORTS.mkdir(exist_ok=True)
    path = ev.REPORTS / "0908_context_margin_sweep.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    out.to_json(ev.OUT / "context_margin_sweep.jsonl", orient="records", lines=True)
    print("\n".join(md))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
