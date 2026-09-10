#!/usr/bin/env python3
"""How far did each adapted vision tower move from stock — and from each other?

    ANIMA_MANGA109S_ROOT=… ANIMA_ANIMETEXT_ROOT=… make daemon-run ARGS="\\
        project/cjk_aware_anima_dit/ocr/tower_drift.py --out output/ocr/eval/tower_drift.md"

The 2026-09-09 SSL-tower post-mortem asked one question: was the SSL init
(``simmim_feat_*``) already inside the basin the COO SFT (B′,
``vl16_tower_lr1e-5``) reaches on its own, so that the SFT simply walked past it?
Every tower in ``output/ocr/*/ep*/tower.safetensors`` is put on one table:

* **weight space** — global relative Frobenius distance
  ``‖W_a − W_b‖ / ‖W_stock‖`` over all tower tensors, pairwise, plus the per-tensor
  relative ΔW median / max vs stock that ``tower_readthrough.py`` prints;
* **feature space** — per-token cosine of ``last_hidden_state`` between towers,
  pairwise, on COO val crops (grey Manga109; ``--coo N`` sfx + N speech) and on
  AnimeText test crops (colour doujin, the SSL domain; ``--animetext N``).

Reads the matrices, not a gate. The comparisons that matter:
``d(ssl, stock)`` vs ``d(B′, stock)`` (how much of B′'s move did SSL make),
``d(ssl→sft, B′)`` vs ``d(col100, B′)`` (did the SSL start land somewhere different
from a plain B′ re-run — col100 is a near-identical SFT and sets the floor), and
``d(ssl→sft, ssl)`` vs ``d(ssl, stock)`` (how far the SFT then moved it).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

BASE = m109.REPO / "models/paddleocr_vl_1.6"
OUT = m109.REPO / "output/ocr"
PREFIX = "model.visual."

DEFAULT_TOWERS = {
    "ssl_20k": "simmim_feat_e1/ep1",
    "ssl_all": "simmim_feat_all_e1/last",
    "B'": "vl16_tower_lr1e-5/ep1",
    "col100": "vl16_tower_col100/ep1",
    "ep3": "vl16_tower_ep3/ep3",
    "lpft": "vl16_lpft/ep1",
    "ssl_20k->sft": "vl16_tower_ssl/ep1",
    "ssl_all->sft": "vl16_tower_ssl_all/ep1",
    "ssl_all->sft_5e5": "vl16_tower_ssl_all_lr5e5/ep1",
}


def fmt_matrix(names, M, fmt="{:.3f}") -> str:
    w = max(len(n) for n in names)
    head = " " * (w + 2) + " ".join(f"{n:>{max(len(n), 7)}}" for n in names)
    rows = [head]
    for i, n in enumerate(names):
        cells = " ".join(
            f"{fmt.format(M[i, j]):>{max(len(names[j]), 7)}}" for j in range(len(names))
        )
        rows.append(f"{n:<{w}}  {cells}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tower",
        action="append",
        default=[],
        help="label=<run>/<ep> (adds to the default set)",
    )
    ap.add_argument("--only", help="comma-separated labels to keep")
    ap.add_argument(
        "--coo", type=int, default=32, help="COO val crops per kind (sfx + speech)"
    )
    ap.add_argument(
        "--animetext", type=int, default=64, help="AnimeText test crops (0 = skip)"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="write the tables here (markdown)")
    a = ap.parse_args()

    towers = dict(DEFAULT_TOWERS)
    for t in a.tower:
        k, v = t.split("=", 1)
        towers[k] = v
    if a.only:
        keep = set(a.only.split(","))
        towers = {k: v for k, v in towers.items() if k in keep}
    paths = {k: OUT / v / "tower.safetensors" for k, v in towers.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"skipping (no tower.safetensors): {missing}", flush=True)
        paths = {k: p for k, p in paths.items() if k not in missing}

    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(BASE))
    ip = proc.image_processor
    full = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16)
    visual = full.model.visual
    del full
    visual = visual.to(a.device).eval()

    # ---- weight space ---------------------------------------------------------
    sds = {
        "stock": {
            PREFIX + n: p.detach().to(torch.bfloat16).cpu()
            for n, p in visual.named_parameters()
        }
    }
    for k, p in paths.items():
        sd = load_file(str(p))
        extra = sorted(
            set(sd) - set(sds["stock"])
        )  # SFT towers also carry model.projector.*
        if extra:
            print(
                f"{k}: ignoring {len(extra)} non-tower tensors ({extra[0]} …)",
                flush=True,
            )
        sds[k] = {t: v for t, v in sd.items() if t in sds["stock"]}
        assert set(sds[k]) == set(sds["stock"]), f"{k}: missing tower keys"
    names = list(sds)
    keys = sorted(sds["stock"])
    stock_norm2 = sum(sds["stock"][t].float().norm() ** 2 for t in keys).item()
    n = len(names)
    Wd = np.zeros((n, n))
    per_tensor = {k: [] for k in names if k != "stock"}
    for t in keys:
        ws = [sds[k][t].float() for k in names]
        s_norm = ws[0].norm().item()
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (ws[i] - ws[j]).norm().item() ** 2
                Wd[i, j] += d2
                Wd[j, i] += d2
            if i > 0:
                per_tensor[names[i]].append(
                    (ws[i] - ws[0]).norm().item() / max(s_norm, 1e-12)
                )
    Wd = np.sqrt(Wd / stock_norm2)
    print(
        "\n## weight space — global relative Frobenius distance ‖W_a − W_b‖ / ‖W_stock‖\n"
    )
    wtab = fmt_matrix(names, Wd, "{:.2e}")
    print(wtab, flush=True)
    print("\nper-tensor relative ΔW vs stock (median / max):")
    ptab = []
    for k, r in per_tensor.items():
        r = sorted(r)
        ptab.append(f"  {k:<18} median {r[len(r) // 2]:.2e}  max {r[-1]:.2e}")
    print("\n".join(ptab), flush=True)

    # ---- crops ----------------------------------------------------------------
    sets: dict[str, list] = {}
    df = pd.read_parquet(m109.derived_root() / "manifest.parquet")
    coo = pd.concat(
        [
            df[(df.split == "val") & (df.kind == kd)].head(a.coo)
            for kd in ("sfx", "speech")
        ]
    )
    sets["COO val (grey)"] = [
        cv2.imread(str(m109.derived_root() / p)) for p in coo.path
    ]
    if a.animetext:
        root = Path(os.environ.get("ANIMA_ANIMETEXT_ROOT", "")).expanduser()
        man = (
            root / "animetext_crops" / "manifest_test_rest.parquet"
        )  # held out of both SSL runs
        if man.exists():
            at = pd.read_parquet(man).sample(a.animetext, random_state=0)
            sets["AnimeText test (colour)"] = [
                cv2.imread(str(root / p)) for p in at.path
            ]
        else:
            print(f"no AnimeText manifest at {man}; skipping", flush=True)
    for k, v in sets.items():
        assert all(c is not None for c in v), f"{k}: unreadable crop"
        print(f"{k}: {len(v)} crops", flush=True)

    @torch.inference_mode()
    def feats(crops):
        out = []
        for c in crops:
            enc = ip(images=[Image.fromarray(c[:, :, ::-1])], return_tensors="pt")
            pv = enc["pixel_values"].to(a.device, torch.bfloat16).unsqueeze(0)
            out.append(
                visual(pixel_values=pv, grid_thw=enc["image_grid_thw"].to(a.device))
                .last_hidden_state.float()
                .cpu()
            )
        return torch.cat(out)

    F = {}
    for k in names:
        res = visual.load_state_dict(
            {t.removeprefix(PREFIX): v.to(a.device) for t, v in sds[k].items()},
            strict=False,
        )
        assert not res.unexpected_keys, f"{k}: {res.unexpected_keys[:3]}"
        assert not res.missing_keys, f"{k}: {res.missing_keys[:3]}"
        F[k] = {s: feats(crops) for s, crops in sets.items()}
        print(f"features: {k}", flush=True)

    ftabs = []
    for s in sets:
        C = np.zeros((n, n))
        for i, ki in enumerate(names):
            for j, kj in enumerate(names):
                C[i, j] = (
                    torch.nn.functional.cosine_similarity(F[ki][s], F[kj][s], dim=-1)
                    .mean()
                    .item()
                )
        nr = " ".join(
            f"{k}×{(F[k][s].norm(dim=-1).mean() / F['stock'][s].norm(dim=-1).mean()).item():.3f}"
            for k in names[1:]
        )
        tab = fmt_matrix(names, C)
        print(f"\n## feature space — mean per-token cosine of last_hidden_state, {s}\n")
        print(tab)
        print(f"\nnorm ratio vs stock: {nr}", flush=True)
        ftabs.append((s, tab, nr))

    if a.out:
        md = [
            "# Tower drift\n",
            "Weight space — global relative Frobenius distance ‖W_a − W_b‖ / ‖W_stock‖\n",
            "```\n" + wtab + "\n```\n",
            "Per-tensor relative ΔW vs stock (median / max)\n",
            "```\n" + "\n".join(ptab) + "\n```\n",
        ]
        for s, tab, nr in ftabs:
            md += [
                f"Feature space — mean per-token cosine, {s}\n",
                "```\n" + tab + "\n```\n",
                f"norm ratio vs stock: {nr}\n",
            ]
        md += ["\nTowers:\n"] + [
            f"- `{k}` = `{towers[k]}`" for k in names if k != "stock"
        ]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(md))
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
