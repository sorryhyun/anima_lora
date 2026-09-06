#!/usr/bin/env python3
"""O0 scorer (``plan_ocr.md``): any reader over the Manga109-s crop manifest.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/cjk_aware_anima_dit/ocr/eval_manga109.py --reader manga_ocr"
    … --reader ppocr        # PP-OCRv6 rec ONNX (anime_tools), crop rotated per its own rule
    … --reader vl16         # PaddleOCR-VL-1.6 crop ``OCR:`` (batched, left-padded, use_cache)
    … --reader manga_ocr --ckpt output/ocr/<run>/best    # a tuned model, same report

Metrics per ``kind`` (``sfx`` = COO test crops, ``speech`` = the matched
``<text>`` control) on ``--split`` (default ``test``):

* **exact** — NFKC + whitespace-stripped string equality (hearts, ``ー``,
  small kana all count);
* **sim** — ``build_ocr_records.sim`` (NFKC, punctuation / symbols / spaces
  gone, ``SequenceMatcher`` ratio) after ``anime_tools.ocr._text.normalize_ja``
  on the prediction (vertical if the crop is) — the A/B's comparison key;
* **runaway** — ``is_runaway`` count (the VL failure class).

Writes ``reports/ocr_eval_<name>.md`` (summary table + worst 25 SFX lines)
and ``output/ocr/eval/<name>_<split>.jsonl`` (every prediction).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

REPORTS = m109.LINE / "reports"
OUT = m109.REPO / "output/ocr/eval"


HEART_FOLD = str.maketrans({"♥": "♡", "❤": "♡", "〜": "~"})


def exact_key(s: str) -> str:
    """NFKC + whitespace-blind; heart / wave variants folded (the hand labels
    write ``♡`` and ``〜``, readers emit ``♥`` / ``~`` for the same glyph)."""
    return "".join(unicodedata.normalize("NFKC", s).split()).translate(HEART_FOLD)


# --------------------------------------------------------------------------- readers


class MangaOcrReader:
    name = "manga_ocr"

    def __init__(self, ckpt: str | None, device: str):
        mt = m109.pilot_manga_text()
        self.m = mt.MangaOCR(ckpt or mt.OCR_MODEL, device=device)

    def read(self, crops, orients, bs):
        return [t for t, _ in self.m.read(crops, batch_size=bs)]


class PpocrReader:
    name = "ppocr"

    def __init__(self, ckpt: str | None, device: str):
        from anime_tools.ocr._onnx import TextRecognizer

        self.r = TextRecognizer.load(
            Path(ckpt) if ckpt else None, device=device, batch_size=32
        )

    def read(self, crops, orients, bs):
        # upstream's rule (crop_quad): taller than 1.5× wide → quarter turn
        rot = [
            np.rot90(c).copy() if c.shape[0] / max(c.shape[1], 1) >= 1.5 else c
            for c in crops
        ]
        return [t for t, _ in self.r.recognize(rot)]


class Vl16Reader:
    name = "vl16"

    def __init__(self, ckpt: str | None, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        base = str(m109.REPO / "models/paddleocr_vl_1.6")
        adapter = (
            ckpt if ckpt and (Path(ckpt) / "adapter_config.json").is_file() else None
        )
        path = base if adapter else (ckpt or base)
        self.torch = torch
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        if adapter:  # O2 arm B: a peft LoRA on the LM, merged for the read
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
            tower = Path(adapter) / "tower.safetensors"
            if tower.is_file():  # O2b: the full-finetuned vision tower + projector
                from safetensors.torch import load_file

                sd = load_file(str(tower))
                unexpected = model.load_state_dict(sd, strict=False).unexpected_keys
                assert not unexpected, unexpected[:5]
                print(f"loaded tower {tower} ({len(sd)} tensors)")
        self.model = model.to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.device = device
        self.min_edge = self.proc.image_processor.size["shortest_edge"]

    def read(self, crops, orients, bs):
        from PIL import Image

        order = sorted(
            range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1]
        )
        out = [""] * len(crops)
        tok = self.proc.tokenizer
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            images = [Image.fromarray(crops[i][:, :, ::-1]) for i in idx]
            msgs = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}],
                }
            ]
            text = self.proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            inputs = self.proc(
                text=[text] * len(images),
                images=images,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": self.min_edge,
                        "longest_edge": 1280 * 28 * 28,
                    }
                },
            ).to(self.device)
            n = inputs["input_ids"].shape[-1]
            with self.torch.inference_mode():
                o = self.model.generate(
                    **inputs, max_new_tokens=48, do_sample=False, use_cache=True
                )
            for i, row in zip(idx, o):
                ids = [
                    t
                    for t in row[n:].tolist()
                    if t not in (tok.eos_token_id, tok.pad_token_id)
                ]
                out[i] = tok.decode(ids).strip()
        return out


class SfxPkgReader:
    """O4: the shipped reader — ``anime_tools.ocr.sfx.SfxReader`` (B′ weights
    from the catalog rows, decode guard built in). ``--ckpt`` overrides the
    adapter dir; a guarded-out read scores as an empty string."""

    name = "sfx"

    def __init__(self, ckpt: str | None, device: str):
        try:
            from anime_tools.ocr import sfx
        except ImportError:  # dev loop before the pinned rev carries ocr/sfx.py
            sys.path.insert(0, str(m109.REPO.parent / "anime_tools"))
            for k in [k for k in sys.modules if k.startswith("anime_tools")]:
                del sys.modules[k]
            from anime_tools.ocr import sfx
        self.r = sfx.SfxReader.load(
            device=device,
            base_dir=m109.REPO / "models/paddleocr_vl_1.6",
            adapter_dir=Path(ckpt) if ckpt else None,
        )

    def read(self, crops, orients, bs):
        self.r.batch_size = bs
        return [t or "" for t in self.r.read(crops)]


READERS = {
    "manga_ocr": MangaOcrReader,
    "ppocr": PpocrReader,
    "vl16": Vl16Reader,
    "sfx": SfxPkgReader,
}


# --------------------------------------------------------------------------- scoring


def score(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    from anime_tools.ocr._text import normalize_ja

    rec = m109.pilot_records()
    rows = []
    for (_, r), p in zip(df.iterrows(), preds):
        pn = normalize_ja(p, vertical=(r.orient == "vertical"))
        rows.append(
            dict(
                pred=p,
                pred_norm=pn,
                exact=exact_key(pn) == exact_key(r.text),
                sim=rec.sim(pn, r.text),
                runaway=rec.is_runaway(p),
            )
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summary(scored: pd.DataFrame, name: str, split: str, wall: float) -> str:
    lines = [
        f"# OCR eval — `{name}` on Manga109-s `{split}` (official COO split ∩ Manga109-s)\n",
        f"Reader wall {wall:.0f} s for {len(scored)} crops ({len(scored) / max(wall, 1e-9):.1f} crops/s).\n",
        "| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, g in scored.groupby("kind"):
        lines.append(
            f"| {k} | {len(g)} | {int(g.exact.sum())} | {100 * g.exact.mean():.1f} | "
            f"{g.sim.mean():.3f} | {100 * (g.sim >= 0.8).mean():.1f} % | {int(g.runaway.sum())} |"
        )
    sfx = scored[scored.kind == "sfx"]
    if len(sfx):
        lines.append(
            "\n## SFX by orientation\n\n| orient | n | exact % | sim |\n|---|---|---|---|"
        )
        for o, g in sfx.groupby("orient"):
            lines.append(
                f"| {o} | {len(g)} | {100 * g.exact.mean():.1f} | {g.sim.mean():.3f} |"
            )
        lines.append(
            "\n## SFX by length\n\n| len | n | exact % | sim |\n|---|---|---|---|"
        )
        L = sfx.text.str.len().clip(upper=8)
        for n, g in sfx.groupby(L):
            lines.append(
                f"| {n}{'+' if n == 8 else ''} | {len(g)} | {100 * g.exact.mean():.1f} | {g.sim.mean():.3f} |"
            )
        lines.append(
            "\n## Worst 25 SFX (by sim)\n\n| book / page / id | gt | pred | sim |\n|---|---|---|---|"
        )
        for _, r in sfx.sort_values("sim").head(25).iterrows():
            pred = r.pred.replace("|", "\\|")[:40]
            lines.append(
                f"| {r.book} {r.page:03d} {r.id} | {r.text} | {pred} | {r.sim:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=sorted(READERS), required=True)
    ap.add_argument(
        "--ckpt", help="model dir overriding the stock weights (same reader class)"
    )
    ap.add_argument(
        "--name", help="report name (default: reader, or reader-<ckpt stem>)"
    )
    ap.add_argument("--split", default="test", choices=m109.SPLITS)
    ap.add_argument("--kind", choices=["sfx", "speech"], action="append")
    ap.add_argument("--limit", type=int, help="first N crops per kind (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    name = a.name or (f"{a.reader}-{Path(a.ckpt).stem}" if a.ckpt else a.reader)

    derived = m109.derived_root()
    df = pd.read_parquet(derived / "manifest.parquet")
    df = df[df.split == a.split]
    if a.kind:
        df = df[df.kind.isin(a.kind)]
    if a.limit:
        df = df.groupby("kind", group_keys=False).head(a.limit)
    df = df.sort_values(["kind", "book", "page", "id"]).reset_index(drop=True)
    crops = [cv2.imread(str(derived / p)) for p in df.path]
    assert all(c is not None for c in crops), (
        "missing crop png — rerun build_manga109_crops"
    )

    reader = READERS[a.reader](a.ckpt, a.device)
    t0 = time.time()
    preds = reader.read(crops, list(df.orient), a.bs)
    wall = time.time() - t0
    scored = score(df, preds)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_{a.split}.jsonl").open("w", encoding="utf-8") as f:
        for _, r in scored.iterrows():
            f.write(
                json.dumps(
                    {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    md = summary(scored, name, a.split, wall)
    if not a.limit:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / f"ocr_eval_{name}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
