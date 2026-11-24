#!/usr/bin/env python3
"""A short LoRA run of OvisOCR3 on the v4 crop mix, to see whether the base is
worth a full run.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 900 \\
        project/ocr_reader/ocr/finetune_ovis3_lora.py --run ovis3_lora_micro --steps 1500"
    … --smoke                       # 30 steps, throughput / memory check

Data = P3's (``vl16_p2b_kozh``): Manga109-s COO SFX + speech at
``--speech_ratio`` 0.909525, plus the gelnote KO / ZH pseudo manifests, drawn
in area-grouped batches (``crop_dataset``). Prompt = the card's
``plain_ocr`` turn with thinking off (exactly ``Ovis3Reader``'s); target =
``crop_dataset`` target + ``<|im_end|>``, loss on the target tokens only
(``finetune_vl16_lora.target_loss``). Pixels 448² … 2880², as the card.

LoRA (α = 2r) on every Linear of the vision tower + merger at
``--tower_rank`` and on the LM decoder layers' projections (GDN
``in_proj_qkv`` / ``in_proj_z`` / ``out_proj``, attention q / k / v / o, MLP) at
``--rank``; embeddings, ``lm_head`` and the GDN ``in_proj_a`` / ``in_proj_b``
gates stay frozen. Linear warmup then linear decay to 0 over ``--steps``.

``--lm_full`` is P2b's split (``finetune_vl16_lora.py --lm_full``): no LM
LoRA, the 24 LM decoder layers train in full at ``--lm_lr`` on an fp32
master copy and land in ``final/lm.safetensors``, which ``Ovis3Reader`` loads
after merging the tower adapter.

Writes ``output/ocr/<run>/final/`` (peft adapter; ``eval_*.py --reader ovis3
--ckpt`` merges it) and ``history.jsonl``. The gates run as separate jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_dataset as cd  # noqa: E402
import finetune_vl16_lora as vl  # noqa: E402
import manga109 as m109  # noqa: E402

BASE = m109.REPO / "models/ovis_ocr3"
PIXELS = {"shortest_edge": 448 * 448, "longest_edge": 2880 * 2880}
LM_LEAVES = {
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class Collate(vl.Collate):
    def images_kwargs(self):
        return {"size": PIXELS}


def lora_targets(model) -> tuple[list[str], list[str]]:
    lm, tower = [], []
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if n.startswith("model.visual."):
            tower.append(n)
        elif (
            n.startswith("model.language_model.layers.")
            and n.split(".")[-1] in LM_LEAVES
        ):
            lm.append(n)
    return lm, tower


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tower_rank", type=int, default=64)
    ap.add_argument(
        "--lm_full",
        action="store_true",
        help="full-finetune the LM decoder layers (fp32 master) instead of the LM LoRA",
    )
    ap.add_argument("--lm_lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--speech_ratio", type=float, default=0.909525)
    ap.add_argument(
        "--extra_manifest",
        action="append",
        default=None,
        help="default: pseudo_gelnote_ko + pseudo_gelnote_zh (P3)",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.extra_manifest is None:
        a.extra_manifest = ["pseudo_gelnote_ko", "pseudo_gelnote_zh"]
    if a.smoke:
        a.steps = 30

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = vl.OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(BASE))
    task = (BASE / "prompts" / "plain_ocr.txt").read_text(encoding="utf-8").strip()
    prompt = proc.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": task}],
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        str(BASE), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    lm, tower = lora_targets(base)
    if a.lm_full:
        lm = []
    pattern = {re.escape(t): a.tower_rank for t in tower}
    cfg = LoraConfig(
        r=a.rank,
        lora_alpha=2 * a.rank,
        lora_dropout=a.dropout,
        target_modules=lm + tower,
        rank_pattern=pattern,
        alpha_pattern={t: 2 * r for t, r in pattern.items()},
        bias="none",
    )
    model = get_peft_model(base, cfg)
    full = (
        [(n, p) for n, p in model.named_parameters() if vl.is_lm_layer_param(n)]
        if a.lm_full
        else []
    )
    for _, p in full:
        p.requires_grad_(True)
    masters = [p.detach().float().clone().requires_grad_(True) for _, p in full]
    full_ids = {id(p) for _, p in full}
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in full_ids
    ]
    print(
        f"lora: {len(lm)} LM modules at r {a.rank}, {len(tower)} tower at r "
        f"{a.tower_rank}; trainable {sum(p.numel() for p in params) / 1e6:.1f}M"
        + (
            f" + LM layers full {sum(m.numel() for m in masters) / 1e6:.1f}M "
            f"in {len(masters)} tensors at lr {a.lm_lr}"
            if full
            else ""
        ),
        flush=True,
    )
    print(f"prompt {prompt!r}", flush=True)

    tr_df = cd.load_split(
        "train", speech_ratio=a.speech_ratio, seed=a.seed, extra=a.extra_manifest
    )
    tr = cd.CropDataset(tr_df, augment=True, seed=a.seed)
    print(
        f"train {len(tr)} ({tr_df.kind.value_counts().to_dict()}; "
        f"source {tr_df.source.value_counts().to_dict()}) — {a.steps} steps of "
        f"{a.bs}x{a.grad_accum} = {a.steps * a.bs * a.grad_accum / len(tr):.2f} epochs",
        flush=True,
    )

    collate = Collate(proc, prompt, PIXELS["shortest_edge"])
    warm = max(1, int(a.steps * a.warmup))
    groups = [{"params": params, "lr": a.lr}]
    if masters:
        groups.append({"params": masters, "lr": a.lm_lr})
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            s / warm if s < warm else max(0.0, (a.steps - s) / max(1, a.steps - warm))
        ),
    )
    hist = (out / "history.jsonl").open("w")
    rng = random.Random(a.seed)
    model.train()
    step, micro, seen, t0 = 0, 0, 0, time.time()
    losses: list[float] = []
    while step < a.steps:
        dl = DataLoader(
            tr,
            batch_sampler=cd.area_batches(tr.area, a.bs, rng),
            num_workers=a.workers,
            collate_fn=collate,
        )
        for enc, _ in dl:
            enc = enc.to("cuda")
            loss = vl.target_loss(model, enc) / a.grad_accum
            loss.backward()
            micro += 1
            seen += enc["input_ids"].shape[0]
            losses.append(loss.item() * a.grad_accum)
            if micro % a.grad_accum:
                continue
            for (_, p), m in zip(full, masters):
                m.grad = p.grad.float()
                p.grad = None
            torch.nn.utils.clip_grad_norm_(params + masters, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for (_, p), m in zip(full, masters):
                    p.copy_(m)
            step += 1
            if step % 25 == 0 or step == a.steps:
                el = time.time() - t0
                row = dict(
                    step=step,
                    loss=float(np.mean(losses[-25 * a.grad_accum :])),
                    lr=sched.get_last_lr()[0],
                    crops_s=seen / el,
                    wall=el,
                    vram=torch.cuda.max_memory_allocated() / 2**30,
                )
                hist.write(json.dumps(row) + "\n")
                hist.flush()
                print(
                    f"step {step}/{a.steps} loss {row['loss']:.4f} lr {row['lr']:.2e} "
                    f"{row['crops_s']:.1f} crops/s {el / 60:.1f} min "
                    f"seq {enc['input_ids'].shape[-1]} vram {row['vram']:.1f}G",
                    flush=True,
                )
            if step >= a.steps:
                break
    model.save_pretrained(out / "final")
    if full:
        vl.save_full(out / "final" / vl.LM_FILE, full)
    print(f"saved {out / 'final'} after {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
