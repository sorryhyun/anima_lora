#!/usr/bin/env python3
"""Read-through gate for an adapted vision tower (``plan_ssl_tower.md`` decision 2).

    python project/cjk_aware_anima_dit/ocr/tower_readthrough.py output/ocr/<run>/ep1/tower.safetensors [--device cpu]

Swaps ``tower.safetensors`` (``model.visual.*`` keys) into the stock
PaddleOCR-VL-1.6 and asks two things of it, on 32 COO val crops:

* **feature drift** — cosine between the adapted and the stock tower's
  ``last_hidden_state`` per token (mean / min) and the feature-norm ratio;
* **can the untouched LM still read through it** — greedy ``OCR:`` reads of
  the first 6 crops, stock vs adapted, and the count of non-empty reads.

Gate: cosine mean ≥ 0.9 and ≥ 5 / 6 non-empty reads. The 2026-09-08
pixel-SimMIM smoke failed it (cosine 0.38, norm ×0.22, 0 / 6) after a median
relative ΔW of 4e-4 — the reason the feature target exists. Exit 1 on a fail so
a queued SFT behind it stops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

BASE = m109.REPO / "models/paddleocr_vl_1.6"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tower")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--reads", type=int, default=6)
    ap.add_argument("--min_cos", type=float, default=0.9)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(BASE))
    model = (
        AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16)
        .to(a.device)
        .eval()
    )
    stock = {
        k: v.detach().clone()
        for k, v in model.named_parameters()
        if k.startswith("model.visual.")
    }
    sd = load_file(a.tower)
    rel = sorted(
        (
            (sd[k].float() - stock[k].float().cpu()).norm()
            / stock[k].float().cpu().norm()
        ).item()
        for k in sd
    )
    print(
        f"{a.tower}: {len(sd)} tensors, relative ΔW median {rel[len(rel) // 2]:.2e} "
        f"max {rel[-1]:.2e}",
        flush=True,
    )

    df = pd.read_parquet(m109.derived_root() / "manifest.parquet")
    df = df[(df.split == "val") & (df.kind == "sfx")].head(a.n)
    crops = [cv2.imread(str(m109.derived_root() / p)) for p in df.path]
    ip = proc.image_processor

    @torch.inference_mode()
    def feats():
        out = []
        for c in crops:
            enc = ip(images=[Image.fromarray(c[:, :, ::-1])], return_tensors="pt")
            pv = enc["pixel_values"].to(a.device, torch.bfloat16).unsqueeze(0)
            out.append(
                model.model.visual(
                    pixel_values=pv, grid_thw=enc["image_grid_thw"].to(a.device)
                )
                .last_hidden_state.float()
                .cpu()
            )
        return out

    @torch.inference_mode()
    def read():
        out = []
        for c in crops[: a.reads]:
            msgs = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}],
                }
            ]
            text = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            inp = proc(
                text=[text],
                images=[Image.fromarray(c[:, :, ::-1])],
                return_tensors="pt",
            ).to(a.device)
            n = inp["input_ids"].shape[-1]
            o = model.generate(**inp, max_new_tokens=16, do_sample=False)
            out.append(
                proc.tokenizer.decode(o[0][n:], skip_special_tokens=True).strip()
            )
        return out

    f0, r0 = feats(), read()
    model.load_state_dict({k: v.to(a.device) for k, v in sd.items()}, strict=False)
    f1, r1 = feats(), read()
    cos = torch.cat(
        [torch.nn.functional.cosine_similarity(x, y, dim=-1) for x, y in zip(f1, f0)]
    )
    ratio = (
        torch.cat([x.norm(dim=-1) for x in f1]).mean()
        / torch.cat([x.norm(dim=-1) for x in f0]).mean()
    )
    nonempty = sum(1 for t in r1 if t)
    print(f"gt      {list(df.text[: a.reads])}")
    print(f"stock   {r0}")
    print(f"adapted {r1}")
    print(
        f"cosine mean {cos.mean():.4f} min {cos.min():.4f} | norm ratio {ratio:.3f} | "
        f"non-empty reads {nonempty} / {a.reads}",
        flush=True,
    )
    ok = cos.mean() >= a.min_cos and nonempty >= a.reads - 1
    print("READTHROUGH", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
