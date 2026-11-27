#!/usr/bin/env python3
"""OvisOCR3 LoRA on the main / sub set, against the VL-1.6 LoRA arm at the same budget.

    make daemon-run ARGS="--queue --stall-timeout 900 \\
        project/ocr_reader/ocr/finetune_ovis3_lora.py --run ovis3ms_<mmdd_hhmm> --steps 1000"
    … --steps 30 --val_limit 64 --mem_log --largest_first   # throughput / peak check

Data = ``crop_dataset.load_mainsub`` (the same rows, neg share and val split as
``finetune_vl16_lora.py --mainsub``). Prompts: ``OCR:`` rows take the card's
``plain_ocr`` turn (the model's own crop-OCR task, as ``OCR:`` is VL's), ``OCR2:`` rows
the literal ``OCR2:``; thinking off. Target = the normalised text + ``<|im_end|>``, loss
on the target tokens only (``finetune_vl16_lora.target_loss``).

Pixels: the card's floor 448² (99.6 % of the train crops sit at or near it, 784–900
tower tokens of 16 px), training capped at ``--max_tokens`` × 256 px; val reads at the
card's 2880² cap. Batches are cut at ``--token_budget`` padded tower tokens
(``crop_dataset.token_batches``).

LoRA (α = 2r) on every Linear of the vision tower + merger at ``--tower_rank`` and on
the LM decoder layers' GDN ``in_proj_qkv`` / ``in_proj_z`` / ``out_proj``, attention
q / k / v / o and MLP at ``--rank``; embeddings, ``lm_head`` and the GDN
``in_proj_a`` / ``in_proj_b`` gates stay frozen. No gradient checkpointing unless
``--grad_ckpt``. ``--compile`` runs the tower on token bands (``BANDS``,
``install_banded_tower``), pads the LM's S to a multiple of 64 and wraps each vision
block and LM decoder layer in ``torch.compile(dynamic=False)`` for grad-on calls;
generation runs eager (on the banded tower).

Writes ``output/ocr/<run>/final/`` (peft adapter), ``history.jsonl``,
``val_final.jsonl`` (scored like the VL trainer's) and ``mem.jsonl`` under ``--mem_log``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from project.ocr_reader.core import crop_dataset as cd
from project.ocr_reader.core.paths import OUTPUT, OVIS3_BASE
from project.ocr_reader.core.readers import MAX_NEW_TOKENS
from project.ocr_reader.core.scoring import score
from project.ocr_reader.ocr import finetune_vl16_lora as vl

MIN_PX = 448 * 448
VAL_MAX_PX = 2880 * 2880
PATCH_PX = 16 * 16
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
COMPILE_CLASSES = ("Qwen3_5VisionBlock", "Qwen3_5DecoderLayer")


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


def ovis_tokens(w: np.ndarray, h: np.ndarray, max_px: int) -> np.ndarray:
    """Tower tokens per crop after the processor's resize (Qwen2-VL ``smart_resize``)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    return np.array(
        [
            int(np.prod(smart_resize(int(hh), int(ww), 32, MIN_PX, max_px))) // PATCH_PX
            for ww, hh in zip(w, h)
        ],
        dtype=np.int64,
    )


BANDS = (840, 896, 1024)
"""Tower-token bands: every crop of a batch is padded to the band of its largest crop
(train crops at ``--max_tokens`` 1024: 66.7 % ≤ 840, 98.3 % ≤ 896). Multiples of 4, so
the merger's 2 × 2 groups never mix real and pad tokens."""


def band_of(tokens: np.ndarray) -> np.ndarray:
    """The band of each count; above the top band (val reads at the card's 2880² cap,
    eager) the count rounded up to a multiple of 64."""
    tokens = np.asarray(tokens)
    b = np.array(BANDS)
    i = np.minimum(np.searchsorted(b, tokens), len(b) - 1)
    return np.where(tokens <= b[-1], b[i], -(-tokens // 64) * 64)


def install_banded_tower(model) -> None:
    """Run the vision tower on a padded ``(B · L, D)`` layout, L = the batch's band.

    The stock SDPA path splits the packed tokens per image (``lengths.tolist()``), one
    graph per crop size and count. Here each crop's tokens sit at ``[i · L, i · L + n_i)``,
    pad rows carry zeros and an identity rotary (cos 1, sin 0), attention is one batched
    SDPA with a key mask, and the merger's output is gathered back to the real rows, so
    the LM receives the stock ``pooler_output``. Each block then sees shapes that depend
    on (B, L) alone."""
    import types

    import torch.nn.functional as F
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    def attn_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kw):
        mask = kw.pop("pad_mask")
        T = hidden_states.shape[0]
        B, L = mask.shape[0], mask.shape[-1]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(T, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q, k = q35.apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = (
            t.reshape(B, L, self.num_heads, -1).transpose(1, 2) for t in (q, k, v)
        )
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
        return self.proj(o.transpose(1, 2).reshape(T, -1))

    def vision_forward(self, hidden_states, grid_thw, **kw):
        interp, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        pos_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        hs = self.patch_embed(hidden_states)
        hs = hs + (self.pos_embed(interp) * weights[:, :, None]).sum(1).to(hs.dtype)
        rot = self.rotary_pos_emb(pos_ids).reshape(hs.shape[0], -1)
        emb = torch.cat((rot, rot), dim=-1)

        n = grid_thw.prod(-1).cpu()
        B, L = len(n), int(band_of(n.max().numpy()[None])[0])
        dev = hs.device
        idx = torch.cat([i * L + torch.arange(int(c)) for i, c in enumerate(n)]).to(dev)
        midx = torch.cat(
            [i * (L // 4) + torch.arange(int(c) // 4) for i, c in enumerate(n)]
        ).to(dev)
        pad = hs.new_zeros(B * L, hs.shape[1]).index_copy(0, idx, hs)
        cos = emb.new_ones(B * L, emb.shape[1]).index_copy(0, idx, emb.cos())
        sin = emb.new_zeros(B * L, emb.shape[1]).index_copy(0, idx, emb.sin())
        mask = (torch.arange(L)[None] < n[:, None]).view(B, 1, 1, L).to(dev)
        cu = torch.zeros(1, dtype=torch.int32, device=dev)  # unused by attn_forward
        for blk in self.blocks:
            pad = blk(pad, cu_seqlens=cu, position_embeddings=(cos, sin), pad_mask=mask)
        merged = self.merger(pad)
        return BaseModelOutputWithPooling(
            last_hidden_state=pad.index_select(0, idx),
            pooler_output=merged.index_select(0, midx),
        )

    vis = model.model.visual
    vis.forward = types.MethodType(vision_forward, vis)
    for blk in vis.blocks:
        blk.attn.forward = types.MethodType(attn_forward, blk.attn)


def compile_layers(model) -> int:
    """``torch.compile(dynamic=False)`` per block for grad-on calls; grad-off calls
    (generation) take the eager forward. Static shapes come from the banded tower
    (``install_banded_tower``) and the LM's S padded to a multiple of 64.

    Every block of a class shares one code object: the GDN and full-attention LM layers
    and each (B, band) / (B, S) shape cost a graph, past the default ``recompile_limit``
    of 8 (``smoke_ovis3_compile_1127`` hit it and ran the rest eager)."""
    from library.runtime.dynamo import pin_dynamo_limit

    pin_dynamo_limit("recompile_limit", 64)
    n = 0
    for m in model.modules():
        if type(m).__name__ not in COMPILE_CLASSES:
            continue
        eager = m.forward
        comp = torch.compile(eager, dynamic=False)

        def fwd(*args, _e=eager, _c=comp, **kw):
            return _c(*args, **kw) if torch.is_grad_enabled() else _e(*args, **kw)

        m.forward = fwd
        n += 1
    return n


@torch.no_grad()
def predict(model, proc, prompts: list[str], ds: cd.CropDataset, bs, workers, eos):
    model.eval()
    tok = proc.tokenizer
    rng = random.Random(0)
    batches = []
    for p in sorted(set(prompts)):
        sel = np.array([i for i, q in enumerate(prompts) if q == p])
        batches += [sel[b].tolist() for b in cd.area_batches(ds.area[sel], bs, rng)]
    dl = DataLoader(
        ds, batch_sampler=batches, num_workers=workers, collate_fn=cd.collate_raw
    )
    preds = [""] * len(ds)
    for imgs, _, idx in dl:
        images = [Image.fromarray(i[:, :, ::-1]) for i in imgs]
        inputs = proc(
            text=[prompts[i] for i in idx],
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            images_kwargs={
                "size": {"shortest_edge": MIN_PX, "longest_edge": VAL_MAX_PX}
            },
        ).to(model.device)
        n = inputs["input_ids"].shape[-1]
        o = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos,
            pad_token_id=tok.pad_token_id,
        )
        for i, row in zip(idx, o):
            t = tok.decode(row[n:], skip_special_tokens=True)
            # the card's reader files a short crop as a Markdown heading
            preds[i] = "\n".join(
                re.sub(r"^\s*#+\s*", "", ln) for ln in t.split("\n")
            ).strip()
    model.train()
    return preds


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--tower_rank", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--neg_share", type=float, default=0.2)
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="training crops' tower-token cap (max_pixels = N x 256); 0.08 %% of the "
        "train crops sit above 1,024",
    )
    ap.add_argument(
        "--token_budget",
        type=int,
        default=4096,
        help="cap each batch's padded tower tokens (largest crop x crops); 4 x 1,024",
    )
    ap.add_argument("--val_limit", type=int, default=500)
    ap.add_argument("--val_bs", type=int, default=16)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mem_log", action="store_true")
    ap.add_argument("--largest_first", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = OUTPUT / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(OVIS3_BASE))
    tok = proc.tokenizer
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")})

    def chat(task: str) -> str:
        return proc.apply_chat_template(
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

    plain = (OVIS3_BASE / "prompts" / "plain_ocr.txt").read_text(encoding="utf-8")
    chats = {cd.OCR_TASK: chat(plain.strip()), cd.OCR2_TASK: chat(cd.OCR2_TASK)}
    print(f"prompts {chats}", flush=True)

    base = AutoModelForImageTextToText.from_pretrained(
        str(OVIS3_BASE), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    # the language model reads text_config: without it every training forward builds a
    # cache and writes the GDN conv states (a recompile per layer under --compile)
    base.config.use_cache = False
    base.config.text_config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    lm, tower = lora_targets(base)
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
    if a.grad_ckpt:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
    if a.compile:
        install_banded_tower(base)
        print(
            f"banded tower {BANDS}; compiled {compile_layers(base)} blocks (static)",
            flush=True,
        )
    params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"lora: {len(lm)} LM modules at r {a.rank}, {len(tower)} tower at r "
        f"{a.tower_rank}; trainable {sum(p.numel() for p in params) / 1e6:.1f}M; "
        f"grad_ckpt {a.grad_ckpt}, compile {a.compile}",
        flush=True,
    )

    tr_df = cd.load_mainsub("train", neg_share=a.neg_share, seed=a.seed)
    va_df = cd.load_mainsub(
        "val", neg_share=a.neg_share, limit=a.val_limit, seed=a.seed
    )
    tr = cd.CropDataset(tr_df, augment=True, seed=a.seed, tight=True)
    va = cd.CropDataset(va_df, augment=False)
    tr_prompt = [chats[t] for t in tr_df.task]
    va_prompt = [chats[t] for t in va_df.task]
    max_px = a.max_tokens * PATCH_PX
    # --compile: the LM's S padded to a multiple of 64 (a handful of static shapes)
    collate = vl.Collate(
        proc, tr_prompt, MIN_PX, max_px, pad_to_multiple_of=64 if a.compile else None
    )
    tr_tokens = ovis_tokens(tr_df.w.to_numpy(), tr_df.h.to_numpy(), max_px)
    if a.compile:  # a batch pads to its largest crop's band
        tr_tokens = band_of(tr_tokens)

    def padded_cost(b: list[int]) -> int:
        return int(tr_tokens[b].max()) * len(b)

    rng = random.Random(a.seed)
    probe = cd.token_batches(
        tr_tokens, a.token_budget, a.bs, random.Random(a.seed), padded=True
    )
    cost = np.array([padded_cost(b) for b in probe])
    print(
        f"train {len(tr)} val {len(va)}; batches {len(probe)} (budget {a.token_budget}, "
        f"max_tokens {a.max_tokens}): padded tower tokens max {cost.max()} median "
        f"{int(np.median(cost))}; {int((np.array([len(b) for b in probe]) < a.bs).sum())} short",
        flush=True,
    )

    warm = max(1, int(a.steps * a.warmup))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            s / warm if s < warm else max(0.0, (a.steps - s) / max(1, a.steps - warm))
        ),
    )
    hist = (out / "history.jsonl").open("w", buffering=1)
    mem_f = (out / "mem.jsonl").open("w", buffering=1) if a.mem_log else None
    peak_all = 0.0

    def take_peak() -> float:
        nonlocal peak_all
        gb = torch.cuda.max_memory_allocated() / 2**30
        peak_all = max(peak_all, gb)
        torch.cuda.reset_peak_memory_stats()
        return gb

    model.train()
    step, micro, seen, t0 = 0, 0, 0, time.time()
    losses: list[torch.Tensor] = []
    while step < a.steps:
        batches = cd.token_batches(tr_tokens, a.token_budget, a.bs, rng, padded=True)
        if a.largest_first:
            batches.sort(key=lambda b: -padded_cost(b))
        dl = DataLoader(
            tr,
            batch_sampler=batches,
            num_workers=a.workers,
            collate_fn=collate,
            pin_memory=True,
        )
        for enc, _ in dl:
            K = int((enc["labels"] != -100).sum(1).max()) + 1
            n = enc["input_ids"].shape[0]
            rec = (
                {
                    "micro": micro,
                    "vis": int(enc["image_grid_thw"].prod(-1).sum()),
                    "lm": int(enc["input_ids"].numel()),
                    "B": n,
                    "K": K,
                }
                if mem_f
                else None
            )
            enc = enc.to("cuda")
            raw = vl.target_loss(model, enc, K)
            # a short (token-budget) batch weighs by its crops, as if part of a --bs batch
            (raw * (min(n, a.bs) / a.bs) / a.grad_accum).backward()
            if mem_f:
                rec["peak"] = round(take_peak(), 3)
                mem_f.write(json.dumps(rec) + "\n")
            micro += 1
            seen += n
            losses.append(raw.detach())
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == a.steps:
                el = time.time() - t0
                peak = max(peak_all, torch.cuda.max_memory_allocated() / 2**30)
                row = dict(
                    step=step,
                    loss=torch.stack(losses[-25 * a.grad_accum :]).mean().item(),
                    lr=sched.get_last_lr()[0],
                    crops_s=seen / el,
                    wall=el,
                    vram=peak,
                )
                hist.write(json.dumps(row) + "\n")
                print(
                    f"step {step}/{a.steps} loss {row['loss']:.4f} lr {row['lr']:.2e} "
                    f"{row['crops_s']:.1f} crops/s t {el:.0f}s "
                    f"seq {enc['input_ids'].shape[-1]} vram {peak:.1f}G",
                    flush=True,
                )
            if step >= a.steps:
                break
    train_s = time.time() - t0
    model.save_pretrained(out / "final")
    print(f"saved {out / 'final'} after {train_s / 60:.1f} min", flush=True)

    t1 = time.time()
    preds = predict(model, proc, va_prompt, va, a.val_bs, a.workers, eos)
    scored = score(va_df, preds)
    m = vl.val_metrics(scored)
    m.update(
        tag="final",
        step=step,
        train_s=round(train_s, 1),
        wall=round(time.time() - t1, 1),
    )
    hist.write(json.dumps(m) + "\n")
    scored.to_json(
        out / "val_final.jsonl", orient="records", lines=True, force_ascii=False
    )
    print(
        "[val final] "
        + " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
