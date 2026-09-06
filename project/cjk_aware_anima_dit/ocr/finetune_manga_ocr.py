#!/usr/bin/env python3
"""O2 arm A (``plan_ocr.md``): fine-tune manga-ocr-base on the O1 crops.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 600 \\
        project/cjk_aware_anima_dit/ocr/finetune_manga_ocr.py --lr 2e-5 --epochs 4 --run mocr_lr2e-5"
    … --smoke                       # 40 steps + a 64-crop val, throughput check

``VisionEncoderDecoderModel`` (ViT-B/16 224 encoder + 2-layer BERT decoder, 111M),
full fine-tune, bf16 autocast, AdamW + linear warmup/decay. Targets are the
original recipe's: ``BertJapaneseTokenizer`` (MeCab word split, then pure
*character* pieces — no ``##`` continuation in practice) → ``[CLS] chars [SEP]``;
the stock model emits ``[CLS]`` as its first predicted token, so labels keep
it. Mix: COO sfx : Manga109 speech 1 : 1 by count (``--speech_ratio``).

Per epoch: val-split scoring with ``eval_manga109``'s metrics (SFX exact / sim,
speech sim — the O2 gate's controls), ``epoch<N>/`` saved as a loadable
model dir (``--ckpt`` of ``eval_manga109.py``), ``best`` → the epoch with the
highest val SFX exact. Output ``output/ocr/<run>/``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_dataset as cd  # noqa: E402
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402

OUT_ROOT = m109.REPO / "output/ocr"


class Collate:
    def __init__(self, processor, tokenizer, max_tokens: int):
        self.p, self.t, self.max_tokens = processor, tokenizer, max_tokens

    def __call__(self, items):
        imgs, targets, idx = cd.collate_raw(items)
        pv = self.p(
            [Image.fromarray(i[:, :, ::-1]) for i in imgs], return_tensors="pt"
        ).pixel_values
        enc = self.t(
            targets,
            padding="longest",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100
        return pv, labels, idx


@torch.no_grad()
def predict(model, processor, decode, ds: cd.CropDataset, device, bs=64, workers=4):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=cd.collate_raw)
    preds = [""] * len(ds)
    for imgs, _, idx in dl:
        pv = processor(
            [Image.fromarray(i[:, :, ::-1]) for i in imgs], return_tensors="pt"
        ).pixel_values.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(pv, max_new_tokens=48, num_beams=1, do_sample=False)
        for i, row in zip(idx, out):
            preds[i] = decode(row.tolist())
    model.train()
    return preds


def val_metrics(scored: pd.DataFrame) -> dict:
    m = {}
    for k, g in scored.groupby("kind"):
        m[f"{k}_exact"] = float(g.exact.mean())
        m[f"{k}_sim"] = float(g.sim.mean())
        m[f"{k}_runaway"] = int(g.runaway.sum())
    return m


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, help="output/ocr/<run>")
    ap.add_argument("--base", default=None, help="model id / dir (default stock)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--speech_ratio", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=cd.MAX_TARGET_CHARS + 2)
    ap.add_argument("--max_train", type=int, help="rows per kind (subsample)")
    ap.add_argument("--val_limit", type=int, help="val rows per kind")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.max_train, a.val_limit, a.epochs = a.max_train or 1280, 64, 1

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    from transformers import (
        AutoImageProcessor,
        BertJapaneseTokenizer,
        VisionEncoderDecoderModel,
    )

    mt = m109.pilot_manga_text()
    base = a.base or mt.OCR_MODEL
    processor = AutoImageProcessor.from_pretrained(base)
    tokenizer = BertJapaneseTokenizer.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base).to(device)
    model.config.decoder_start_token_id = model.config.decoder_start_token_id or 2
    model.config.pad_token_id = tokenizer.pad_token_id
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    specials = {i for i, t in enumerate(vocab) if t.startswith("[") and t.endswith("]")}

    def decode(ids):
        return "".join(
            (vocab[i][2:] if vocab[i].startswith("##") else vocab[i])
            for i in ids
            if i not in specials and i < len(vocab)
        ).replace(" ", "")

    if a.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tr_df = cd.load_split(
        "train", speech_ratio=a.speech_ratio, limit=a.max_train, seed=a.seed
    )
    va_df = cd.load_split("val", limit=a.val_limit, seed=a.seed)
    tr = cd.CropDataset(tr_df, augment=not a.no_augment, seed=a.seed)
    va = cd.CropDataset(va_df, augment=False)
    print(
        f"train {len(tr)} ({tr_df.kind.value_counts().to_dict()}) val {len(va)} "
        f"trainable {n_train / 1e6:.1f}M lr {a.lr} bs {a.bs} epochs {a.epochs}",
        flush=True,
    )
    dl = DataLoader(
        tr,
        batch_size=a.bs,
        shuffle=True,
        num_workers=a.workers,
        collate_fn=Collate(processor, tokenizer, a.max_tokens),
        drop_last=True,
        persistent_workers=a.workers > 0,
    )
    steps_per_epoch = len(dl)
    total = steps_per_epoch * a.epochs
    if a.smoke:
        total = min(total, 40)
    warm = max(1, int(total * a.warmup))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.lr,
        weight_decay=a.weight_decay,
        betas=(0.9, 0.98),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            s / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
        ),
    )

    # stock row on this val cut = the reference the epochs are read against
    hist = out / "history.jsonl"
    hist.write_text("")

    def evaluate(tag: str, step: int):
        t0 = time.time()
        preds = predict(model, processor, decode, va, device, bs=64, workers=a.workers)
        scored = ev.score(va_df, preds)
        m = val_metrics(scored)
        m.update(tag=tag, step=step, wall=round(time.time() - t0, 1))
        with hist.open("a") as f:
            f.write(json.dumps(m) + "\n")
        scored.to_json(
            out / f"val_{tag}.jsonl", orient="records", lines=True, force_ascii=False
        )
        print(
            f"[val {tag}] "
            + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in m.items()
            ),
            flush=True,
        )
        return m

    evaluate("stock", 0)

    model.train()
    step, t0, seen = 0, time.time(), 0
    best = (-1.0, None)
    for ep in range(1, a.epochs + 1):
        losses = []
        for pv, labels, _ in dl:
            pv, labels = pv.to(device, non_blocking=True), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=pv, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += pv.shape[0]
            losses.append(loss.item())
            if step % 50 == 0 or step == total:
                el = time.time() - t0
                print(
                    f"ep {ep} step {step}/{total} loss {np.mean(losses[-50:]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {seen / el:.0f} crops/s "
                    f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                    flush=True,
                )
            if step >= total:
                break
        m = evaluate(f"ep{ep}", step)
        m["train_loss"] = float(np.mean(losses))
        ep_dir = out / f"ep{ep}"
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)
        tokenizer.save_pretrained(ep_dir)
        if m["sfx_exact"] > best[0]:
            best = (m["sfx_exact"], ep)
            b = out / "best"
            if b.is_symlink() or b.exists():
                b.unlink() if b.is_symlink() else shutil.rmtree(b)
            b.symlink_to(ep_dir.name)
        if step >= total:
            break

    rows = [json.loads(ln) for ln in hist.read_text().splitlines()]
    md = [
        f"# {a.run} — manga-ocr fine-tune (lr {a.lr}, bs {a.bs}, speech_ratio {a.speech_ratio}, "
        f"train {len(tr)}, val {len(va)})\n",
        "| tag | sfx exact | sfx sim | speech exact | speech sim | runaway |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['tag']} | {100 * r['sfx_exact']:.1f} % | {r['sfx_sim']:.3f} | "
            f"{100 * r['speech_exact']:.1f} % | {r['speech_sim']:.3f} | "
            f"{r['sfx_runaway'] + r['speech_runaway']} |"
        )
    md.append(f"\nbest = ep{best[1]} (val SFX exact {100 * best[0]:.1f} %)")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
