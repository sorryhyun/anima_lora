#!/usr/bin/env python3
"""O2 arm B (``plan_ocr.md``): a peft LoRA on PaddleOCR-VL-1.6's LM for the crop ``OCR:`` task.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 600 \\
        project/cjk_aware_anima_dit/ocr/finetune_vl16_lora.py --lr 1e-4 --epochs 2 --run vl16_lr1e-4"
    … --smoke                       # 30 steps + a 64-crop val, throughput check

Tower + projector frozen; LoRA (``--rank``, α = 2r) on the ERNIE LM's attention
(q/k/v/o) + MLP (gate/up/down) projections, selected by module path so the
vision tower's same-named projections are untouched. Prompt = the chat
template's ``User: <image>OCR:\\nAssistant:\\n`` (exactly the eval's), target =
normalised text + ``</s>``; loss on the target tokens only. Batches are grouped
by crop area (the A/B's batching rule) and left-padded (loss on the last K logits); the
per-epoch val generation is left-padded, greedy, ``use_cache=True``, no
repetition guard — the runaway count is reported like the stock row.

Per epoch: val scoring (``eval_manga109`` metrics), ``ep<N>/`` = adapter dir
(``eval_manga109.py --reader vl16 --ckpt`` merges it onto the base), ``best`` →
highest val SFX exact. Output ``output/ocr/<run>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
BASE = m109.REPO / "models/paddleocr_vl_1.6"
LORA_LEAVES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def lm_lora_targets(model) -> list[str]:
    """Full names of the LM's Linear projections (vision tower excluded by path)."""
    vision_prefixes = [
        n for n, m in model.named_modules() if "Vision" in type(m).__name__ and n
    ]
    vision_prefixes = [
        p
        for p in vision_prefixes
        if not any(p.startswith(q + ".") for q in vision_prefixes if q != p)
    ]
    names = []
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or n.split(".")[-1] not in LORA_LEAVES:
            continue
        if (
            any(n.startswith(p + ".") for p in vision_prefixes)
            or "projector" in n.lower()
        ):
            continue
        names.append(n)
    return names


class Collate:
    def __init__(self, proc, prompt: str, min_edge: int):
        self.proc, self.prompt, self.min_edge = proc, prompt, min_edge
        self.tok = proc.tokenizer
        self.eos = self.tok.eos_token

    def images_kwargs(self):
        return {
            "size": {"shortest_edge": self.min_edge, "longest_edge": 1280 * 28 * 28}
        }

    def __call__(self, items):
        imgs, targets, idx = cd.collate_raw(items)
        images = [Image.fromarray(i[:, :, ::-1]) for i in imgs]
        texts = [self.prompt + t + self.eos for t in targets]
        # LEFT padding for training too: every row's target then sits at the end,
        # so the loss can be taken on the last K logits only (``logits_to_keep``)
        # instead of materialising fp32 logits over the 103k vocab for every
        # image token — the OOM of the first launch (findings § O2).
        enc = self.proc(
            text=texts,
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            images_kwargs=self.images_kwargs(),
        )
        labels = torch.full_like(enc["input_ids"], -100)
        n_bad = 0
        for r, t in enumerate(targets):
            tid = self.tok(t + self.eos, add_special_tokens=False).input_ids
            L = len(tid)
            row = enc["input_ids"][r, -L:]
            if row.tolist() == tid:
                labels[r, -L:] = row
            else:  # boundary merge (rare): label from the prompt length instead
                p = self.proc(
                    text=[self.prompt],
                    images=[images[r]],
                    return_tensors="pt",
                    images_kwargs=self.images_kwargs(),
                )["input_ids"].shape[-1]
                n = int(enc["attention_mask"][r].sum())
                labels[r, -(n - p) :] = enc["input_ids"][r, -(n - p) :]
                n_bad += 1
        enc["labels"] = labels
        enc["n_boundary_fallback"] = n_bad
        return enc, idx


def target_loss(model, enc) -> torch.Tensor:
    """CE on the target suffix only: keep the last K logits (K = longest label
    run + 1), shift, ignore -100."""
    labels = enc.pop("labels")
    enc.pop("n_boundary_fallback", None)
    K = int((labels != -100).sum(1).max()) + 1
    out = model(**enc, logits_to_keep=K)
    logits = out.logits[
        :, :-1
    ].float()  # positions seq-K .. seq-2 predict seq-K+1 .. seq-1
    tgt = labels[:, -(K - 1) :]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100
    )


@torch.no_grad()
def predict(model, proc, prompt, min_edge, ds: cd.CropDataset, bs, workers):
    model.eval()
    tok = proc.tokenizer
    rng = random.Random(0)
    batches = cd.area_batches(ds.area, bs, rng)
    dl = DataLoader(
        ds, batch_sampler=batches, num_workers=workers, collate_fn=cd.collate_raw
    )
    preds = [""] * len(ds)
    for imgs, _, idx in dl:
        images = [Image.fromarray(i[:, :, ::-1]) for i in imgs]
        inputs = proc(
            text=[prompt] * len(images),
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            images_kwargs={
                "size": {"shortest_edge": min_edge, "longest_edge": 1280 * 28 * 28}
            },
        ).to(model.device)
        n = inputs["input_ids"].shape[-1]
        o = model.generate(**inputs, max_new_tokens=48, do_sample=False, use_cache=True)
        for i, row in zip(idx, o):
            ids = [
                t
                for t in row[n:].tolist()
                if t not in (tok.eos_token_id, tok.pad_token_id)
            ]
            preds[i] = tok.decode(ids).strip()
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--speech_ratio", type=float, default=1.0)
    ap.add_argument("--max_train", type=int, help="rows per kind (subsample)")
    ap.add_argument("--val_limit", type=int)
    ap.add_argument("--val_bs", type=int, default=16)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_stock_val", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.max_train, a.val_limit, a.epochs = a.max_train or 480, 64, 1

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(BASE))
    min_edge = proc.image_processor.size["shortest_edge"]
    prompt = proc.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}],
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        str(BASE), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    targets = lm_lora_targets(base)
    cfg = LoraConfig(
        r=a.rank,
        lora_alpha=2 * a.rank,
        lora_dropout=a.dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(base, cfg)
    if not a.no_grad_ckpt:
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            base.enable_input_require_grads()
        except Exception as e:  # custom modeling file without the hook
            print(f"(gradient checkpointing unavailable: {e})", flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"lora targets {len(targets)} modules (e.g. {targets[0]} … {targets[-1]}); trainable {n_train / 1e6:.1f}M",
        flush=True,
    )
    print(f"prompt {prompt!r}", flush=True)

    tr_df = cd.load_split(
        "train", speech_ratio=a.speech_ratio, limit=a.max_train, seed=a.seed
    )
    va_df = cd.load_split("val", limit=a.val_limit, seed=a.seed)
    tr = cd.CropDataset(tr_df, augment=not a.no_augment, seed=a.seed)
    va = cd.CropDataset(va_df, augment=False)
    print(
        f"train {len(tr)} ({tr_df.kind.value_counts().to_dict()}) val {len(va)} lr {a.lr} bs {a.bs}x{a.grad_accum} r {a.rank} epochs {a.epochs}",
        flush=True,
    )

    collate = Collate(proc, prompt, min_edge)
    rng = random.Random(a.seed)
    steps_per_epoch = len(tr) // a.bs // a.grad_accum
    total = steps_per_epoch * a.epochs
    if a.smoke:
        total = min(total, 30)
    warm = max(1, int(total * a.warmup))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            s / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
        ),
    )
    hist = out / "history.jsonl"
    hist.write_text("")

    def evaluate(tag, step):
        t0 = time.time()
        preds = predict(model, proc, prompt, min_edge, va, a.val_bs, a.workers)
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

    if not a.skip_stock_val:
        evaluate("stock", 0)

    model.train()
    step, micro, t0, seen = 0, 0, time.time(), 0
    best = (-1.0, None)
    for ep in range(1, a.epochs + 1):
        dl = DataLoader(
            tr,
            batch_sampler=cd.area_batches(tr.area, a.bs, rng),
            num_workers=a.workers,
            collate_fn=collate,
            persistent_workers=False,
        )
        losses = []
        for enc, _ in dl:
            enc = enc.to("cuda")
            loss = target_loss(model, enc) / a.grad_accum
            loss.backward()
            micro += 1
            seen += enc["input_ids"].shape[0]
            losses.append(loss.item() * a.grad_accum)
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == total:
                el = time.time() - t0
                print(
                    f"ep {ep} step {step}/{total} loss {np.mean(losses[-25 * a.grad_accum :]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {seen / el:.1f} crops/s "
                    f"seq {enc['input_ids'].shape[-1]} vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                    flush=True,
                )
            if step >= total:
                break
        m = evaluate(f"ep{ep}", step)
        m["train_loss"] = float(np.mean(losses))
        ep_dir = out / f"ep{ep}"
        model.save_pretrained(ep_dir)
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
        f"# {a.run} — PaddleOCR-VL-1.6 crop LoRA (r {a.rank}, lr {a.lr}, bs {a.bs}x{a.grad_accum}, "
        f"speech_ratio {a.speech_ratio}, train {len(tr)}, val {len(va)})\n",
        "| tag | sfx exact | sfx sim | sfx runaway | speech exact | speech sim | speech runaway |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['tag']} | {100 * r['sfx_exact']:.1f} % | {r['sfx_sim']:.3f} | {r['sfx_runaway']} | "
            f"{100 * r['speech_exact']:.1f} % | {r['speech_sim']:.3f} | {r['speech_runaway']} |"
        )
    md.append(f"\nbest = ep{best[1]} (val SFX exact {100 * best[0]:.1f} %)")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
