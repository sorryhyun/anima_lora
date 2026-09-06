#!/usr/bin/env python3
"""The doujin gate (``plan_ocr.md`` O0/O2): any reader over the sincos hand labels.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/cjk_aware_anima_dit/ocr/eval_sfx.py --reader manga_ocr"
    … --reader manga_ocr --ckpt output/ocr/<run>/best
    … --reader vl16 [--ckpt output/ocr/<run>/best]       # a peft adapter dir merges onto the base
    … --reader record                                    # the pipeline's current read (no model)

Labels: ``assets/sfx_labels_sincos.tsv`` (one row per hybrid record; ``kind_hand``
/ ``text_hand`` are human — drafted off the contact sheet, corrected by the
user; ``status`` = draft | checked | unchecked). Crops are cut from the
resized sincos pages with the pilot's ``deskew_crop`` at 12 % pad (the box is
axis-aligned, so this is a padded rectangle crop; orientation preserved).

Rows: **SFX** = ``kind_hand == sfx`` (the 71-ish doujin lines the line is for),
**speech control** = ``kind_hand == speech``. Metrics as ``eval_manga109``:
exact (NFKC + whitespace-blind; the ``JOIN_SEP`` space between repeated blocks
is therefore free), sim, runaway. Writes ``reports/ocr_eval_sfx_<name>.md`` +
``output/ocr/eval/sfx_<name>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402

LABELS = m109.ASSETS / "sfx_labels_sincos.tsv"
PAGES = m109.REPO / "post_image_dataset/resized/sincos"


def load_labels(kinds: list[str] | None, include_unchecked: bool) -> pd.DataFrame:
    df = pd.read_csv(LABELS, sep="\t", dtype=str, keep_default_na=False)
    df["box"] = df.box.map(json.loads)
    if not include_unchecked:
        df = df[df.status != "unchecked"]
    df = df[df.text_hand.str.strip() != ""]  # unreadable / clipped rows are left blank
    if kinds:
        df = df[df.kind_hand.isin(kinds)]
    # eval_manga109.score reads .text / .orient / .kind
    df = df.rename(columns={"kind_hand": "kind", "text_hand": "text"})
    df["book"] = "sincos"
    df["page"] = 0
    df["id"] = df.row
    return df.reset_index(drop=True)


def crops_for(df: pd.DataFrame, pad: float):
    mt = m109.pilot_manga_text()
    crops, orients = [], []
    for _, r in df.iterrows():
        img = cv2.imread(str(PAGES / f"{r.stem}.png"))
        x0, y0, x1, y1 = r.box
        crop, orient = mt.deskew_crop(img, [x0, y0, x1, y0, x1, y1, x0, y1], pad, 8)
        crops.append(crop)
        orients.append(orient)
    return crops, orients


class RecordReader:
    """No model: the text the hybrid records carry today (the pipeline baseline)."""

    name = "record"

    def __init__(self, ckpt, device):
        pass

    def read(self, crops, orients, bs):
        raise NotImplementedError  # handled in main (needs the rows, not the crops)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=sorted(ev.READERS) + ["record"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--kind", action="append", help="default sfx + speech")
    ap.add_argument("--include_unchecked", action="store_true")
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    name = a.name or (
        f"{a.reader}-{Path(a.ckpt).parent.name}-{Path(a.ckpt).name}"
        if a.ckpt
        else a.reader
    )
    df = load_labels(a.kind or ["sfx", "speech", "chrome"], a.include_unchecked)
    crops, orients = crops_for(df, a.pad)
    df["orient"] = orients
    t0 = time.time()
    if a.reader == "record":
        preds = list(df.text_rec)
    else:
        reader = ev.READERS[a.reader](a.ckpt, a.device)
        preds = reader.read(crops, orients, a.bs)
    wall = time.time() - t0
    scored = ev.score(df, preds)
    # hearts are the one glyph the gate lets a rule patch (decision 6): report
    # a heart-blind exact beside the strict one so the ~12/71 pilot count reads
    strip = lambda t: ev.exact_key(t).replace("♡", "").replace("♥", "")  # noqa: E731
    scored["exact_noheart"] = [
        strip(pn) == strip(t) for pn, t in zip(scored.pred_norm, scored.text)
    ]

    ev.OUT.mkdir(parents=True, exist_ok=True)
    scored.drop(columns=["box"]).to_json(
        ev.OUT / f"sfx_{name}.jsonl", orient="records", lines=True, force_ascii=False
    )
    md = ev.summary(scored, name, "sincos", wall).replace(
        "Manga109-s `sincos` (official COO split ∩ Manga109-s)",
        "sincos hand labels (`assets/sfx_labels_sincos.tsv`)",
    )
    gate = scored[scored.kind_rec == "sfx"]  # the 71 records the O2 gate counts
    md += (
        f"\n## Gate set: the {len(gate)} `kind: sfx` records (scored against the hand text)\n\n"
        f"exact **{int(gate.exact.sum())} / {len(gate)}** (heart-blind "
        f"{int(gate.exact_noheart.sum())}), sim {gate.sim.mean():.3f}; "
        f"hand-relabelled: {(gate.kind != 'sfx').sum()} rows are not SFX by eye "
        f"({', '.join(sorted(set(gate[gate.kind != 'sfx'].kind)))}).\n"
    )
    sfx = scored[scored.kind == "sfx"]
    for k, g in scored.groupby("kind"):
        md += f"heart-blind exact, {k}: {int(g.exact_noheart.sum())} / {len(g)}\n"
    md += (
        "\n## Every SFX line\n\n| row | gt | pred | exact | ♡-blind | sim |\n|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r.row} | {r.text} | {r.pred.replace('|', '\\|')[:30]} | {'✓' if r.exact else ''} | {'✓' if r.exact_noheart else ''} | {r.sim:.2f} |"
            for _, r in sfx.iterrows()
        )
        + "\n"
    )
    ev.REPORTS.mkdir(exist_ok=True)
    (ev.REPORTS / f"ocr_eval_sfx_{name}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
