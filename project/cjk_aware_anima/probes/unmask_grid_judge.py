#!/usr/bin/env python3
"""plan_zh3 G0 readout for the unmask eval grids: prompt adherence + drift.

Per cell of `output/tests/cjk_unmask_eval2/arm<X>_s<seed>/` (8 rows of
`assets/unmask_eval_prompts.txt`, PNGs sorted = row order):

- **adherence** — dbv4 tagger (`anime_tools.tagger.dbv4_backend`): mean
  probability of the row's own tags (`assets/unmask_eval_prompts.json`) and
  recall at the card's best thresholds.
- **base_cos** — PE-Spatial pooled cosine to the *base model's* render of the
  same prompt + seed (`--base_dir`, default `output/tests/cjk_unmask_eval/`
  `base_s<seed>`), i.e. how far the LoRA moved the image.
- **sincos_cos** — cosine to the mean pooled PE-Spatial of the sincos
  training set (`post_image_dataset/lora/sincos/*_anima_pe_spatial.safetensors`):
  style adoption.

Writes `<out>.json` (cells + per-arm / per-arm-row means) and `<out>.md`.
CPU-safe (`--device cpu`) so it can run beside a live GPU job.

    .venv/bin/python project/cjk_aware_anima/probes/unmask_grid_judge.py \
        --arms A C2 C9 P R --out project/cjk_aware_anima/reports/unmask_grid_judge.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EVAL = REPO / "output" / "tests" / "cjk_unmask_eval2"
BASE = REPO / "output" / "tests" / "cjk_unmask_eval"
SINCOS_PE = REPO / "post_image_dataset" / "lora" / "sincos"


def pil_to_pm1(img) -> torch.Tensor:
    t = (
        torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1).float()
        / 127.5
        - 1.0
    )
    return t.unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1234])
    ap.add_argument("--eval_dir", type=Path, default=EVAL)
    ap.add_argument("--base_dir", type=Path, default=BASE)
    ap.add_argument(
        "--rows_json", type=Path, default=PROJ / "assets" / "unmask_eval_prompts.json"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no_pe", action="store_true")
    opts = ap.parse_args()

    from PIL import Image
    from anime_tools.tagger.dbv4_backend import Dbv4Backend

    rows = json.loads(opts.rows_json.read_text(encoding="utf-8"))["rows"]
    backend = Dbv4Backend(device=opts.device)
    col = {r["name"]: j for j, r in enumerate(backend.card.rows)}
    thr = backend.card.best_thresholds()
    row_tags: dict[int, list[str]] = {}
    for r in rows:
        keep = [t for t in r["tags"] if t in col]
        missing = [t for t in r["tags"] if t not in col]
        if missing:
            print(f"r{r['row']}: dropping tags not in dbv4 card: {missing}")
        row_tags[r["row"]] = keep

    # ---- gather cells --------------------------------------------------------
    cells = []
    for arm in opts.arms:
        for seed in opts.seeds:
            pngs = sorted((opts.eval_dir / f"arm{arm}_s{seed}").glob("*.png"))
            if len(pngs) != len(rows):
                print(f"missing/short grid: arm{arm}_s{seed} ({len(pngs)} cells)")
                continue
            for i, png in enumerate(pngs, 1):
                cells.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "row": i,
                        "cell": str(png.relative_to(REPO)),
                    }
                )
    base_cells = {}
    for seed in opts.seeds:
        pngs = sorted((opts.base_dir / f"base_s{seed}").glob("*.png"))
        for i, png in enumerate(pngs, 1):
            base_cells[(seed, i)] = png

    # ---- tagger --------------------------------------------------------------
    for i in range(0, len(cells), opts.batch):
        chunk = cells[i : i + opts.batch]
        out = backend.forward([Image.open(REPO / c["cell"]) for c in chunk])
        for c, p in zip(chunk, out.probs):
            tags = row_tags[c["row"]]
            probs = [float(p[col[t]]) for t in tags]
            hits = [int(float(p[col[t]]) >= float(thr[col[t]])) for t in tags]
            c["adherence_prob"] = sum(probs) / len(probs)
            c["adherence_recall"] = sum(hits) / len(hits)
            c["tag_probs"] = dict(zip(tags, [round(v, 3) for v in probs]))
        print(f"tagger {i + len(chunk)}/{len(cells)}", flush=True)
    del backend

    # ---- PE-Spatial drift ------------------------------------------------------
    if not opts.no_pe:
        from safetensors.torch import load_file

        from library.training.cmmd import pool_and_normalize
        from library.vision.encoder import (
            encode_pe_from_imageminus1to1,
            load_pe_encoder,
        )

        device = torch.device(opts.device)
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        bundle = load_pe_encoder(device, name="pe_spatial", dtype=dtype)

        @torch.no_grad()
        def pooled(png: Path) -> torch.Tensor:
            feats = encode_pe_from_imageminus1to1(
                bundle, pil_to_pm1(Image.open(png)).to(device)
            )[0]
            return pool_and_normalize(feats).cpu()

        sincos = []
        for f in sorted(SINCOS_PE.glob("*_anima_pe_spatial.safetensors")):
            sincos.append(pool_and_normalize(load_file(str(f))["image_features"]))
        sincos_mean = torch.nn.functional.normalize(torch.stack(sincos).mean(0), dim=0)
        print(f"sincos reference: {len(sincos)} cached PE-Spatial maps")
        base_feat = {k: pooled(v) for k, v in base_cells.items()}
        for n, c in enumerate(cells, 1):
            f = pooled(REPO / c["cell"])
            b = base_feat.get((c["seed"], c["row"]))
            c["base_cos"] = float((f * b).sum()) if b is not None else None
            c["sincos_cos"] = float((f * sincos_mean).sum())
            if n % 20 == 0:
                print(f"pe {n}/{len(cells)}", flush=True)

    # ---- aggregate -------------------------------------------------------------
    keys = ["adherence_prob", "adherence_recall"] + (
        [] if opts.no_pe else ["base_cos", "sincos_cos"]
    )

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    per_arm, per_arm_row = {}, {}
    by_arm = defaultdict(list)
    by_arm_row = defaultdict(list)
    for c in cells:
        by_arm[c["arm"]].append(c)
        by_arm_row[(c["arm"], c["row"])].append(c)
    for arm in opts.arms:
        per_arm[arm] = {k: mean(c.get(k) for c in by_arm[arm]) for k in keys}
        per_arm[arm]["n"] = len(by_arm[arm])
    for (arm, row), cs in by_arm_row.items():
        per_arm_row[f"{arm}/r{row}"] = {k: mean(c.get(k) for c in cs) for k in keys}

    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(
        json.dumps(
            {"per_arm": per_arm, "per_arm_row": per_arm_row, "cells": cells},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md = [
        "| arm | n | adherence prob | adherence recall |"
        + ("" if opts.no_pe else " cos→base | cos→sincos |"),
        "|---|---:|---:|---:|" + ("" if opts.no_pe else "---:|---:|"),
    ]
    for arm in opts.arms:
        a = per_arm[arm]
        line = f"| {arm} | {a['n']} | {a['adherence_prob']} | {a['adherence_recall']} |"
        if not opts.no_pe:
            line += f" {a['base_cos']} | {a['sincos_cos']} |"
        md.append(line)
    md += [
        "",
        "per row (adherence prob / recall"
        + ("" if opts.no_pe else " / cos→base")
        + "):",
        "",
        "| row | " + " | ".join(opts.arms) + " |",
        "|---|" + "---|" * len(opts.arms),
    ]
    for row in range(1, len(rows) + 1):
        vals = []
        for arm in opts.arms:
            a = per_arm_row.get(f"{arm}/r{row}", {})
            v = f"{a.get('adherence_prob')} / {a.get('adherence_recall')}"
            if not opts.no_pe:
                v += f" / {a.get('base_cos')}"
            vals.append(v)
        md.append(f"| r{row} | " + " | ".join(vals) + " |")
    opts.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
