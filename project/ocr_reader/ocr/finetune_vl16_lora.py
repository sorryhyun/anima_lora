#!/usr/bin/env python3
"""O2 arm B (``plan_ocr.md``): a peft LoRA on PaddleOCR-VL-1.6's LM for the crop ``OCR:`` task.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 600 \\
        project/ocr_reader/ocr/finetune_vl16_lora.py --lr 1e-4 --epochs 2 --run vl16_lr1e-4"
    … --smoke                       # 30 steps + a 64-crop val, throughput check
    … --train_tower --tower_lr 1e-5 # O2b: unfreeze the NaViT tower + projector (full FT)
    … --extra_manifest colorized_1024_half_comic   # O3: + the colorized crops (manifest_<name>.parquet)
    … --extra_replace                # …swapped in for their grey originals instead of appended
    … --init_adapter output/ocr/vl16_lr1e-4/ep2 --train_tower   # LP-FT stage 2: start from arm B's LoRA
    … --train_tower --init_tower output/ocr/simmim_at/ep5/tower.safetensors   # SSL-adapted tower (ssl_tower_simmim.py)
    … --tower_rank 64 --lm_full --lm_lr 1e-5   # ocr_reader P2b: tower LoRA + LM decoder layers full FT
    … --sfx_prefix                   # two prompts: OCR: (speech) + SFX: (sound effects), build_sfx_prefix.py crops
    … --mainsub                      # two prompts: OCR: (main text) + OCR2: (sub), crop_dataset.load_mainsub

Tower + projector frozen; LoRA (``--rank``, α = 2r) on the ERNIE LM's attention
(q/k/v/o) + MLP (gate/up/down) projections, selected by module path so the
vision tower's same-named projections are untouched. Prompt = the chat
template's ``User: <image>OCR:\\nAssistant:\\n`` (exactly the eval's), target =
normalised text + ``</s>``; loss on the target tokens only. Batches hold up to
``--bs`` crops and are cut where the padded patch-token cost (largest crop × crops)
would pass ``--token_budget`` (3,072 by default), so a large crop trains alone;
``--token_budget 0`` groups fixed ``--bs`` batches by crop area (the A/B's rule). A
short batch's loss is scaled by crops / ``--bs``. Defaults: bs 4 × 4, no gradient
checkpointing (``--grad_ckpt`` turns it on). Batches are left-padded (loss on the last K logits); the
per-epoch val generation is left-padded, greedy, ``use_cache=True``, no
repetition guard — the runaway count is reported like the stock row.

``--train_tower`` (O2b) additionally trains the vision tower (466 M) + the
projector (26 M) in full: the bf16 weights stay in place for the
forward, an fp32 master copy takes the AdamW update at ``--tower_lr`` and is
copied back each step (a 1e-5 update is below bf16 resolution otherwise).
The trained tower is saved beside the adapter as ``tower.safetensors`` (bf16,
base-model key names) and ``Vl16Reader`` loads it after the merge.

``--tower_rank R`` (``project/ocr_reader`` P2b) puts a LoRA (α = 2R) on the
tower's attention / MLP Linears and the projector's two Linears instead.
``--lm_full`` drops the LM LoRA and trains the 18 LM decoder layers (255 M) in
full through the same fp32-master path at ``--lm_lr``; embed_tokens, the final
norm and lm_head stay frozen. Those weights land in ``lm.safetensors`` beside
the adapter, which ``Vl16Reader`` also loads after the merge.

``--init_adapter <dir>`` (LP-FT, Kumar et al. 2022) loads an already-trained
adapter dir as the starting point instead of a zero-init LoRA — its
``adapter_config.json`` wins over ``--rank`` / ``--dropout``. With
``--train_tower`` this is the linear-probe-then-fine-tune order: arm B (LoRA
only, tower frozen) is stage 1, this run is stage 2. A ``tower.safetensors``
beside the adapter is loaded too when present.

Per epoch: val scoring (``eval_manga109`` metrics), ``ep<N>/`` = adapter dir
(``eval_manga109.py --reader vl16 --ckpt`` merges it onto the base), ``best`` →
highest val SFX exact. ``--stock_val`` also scores the base model before training
(off by default). Output ``output/ocr/<run>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

from project.ocr_reader.core import crop_dataset as cd
from project.ocr_reader.core.paths import OUTPUT, VL16_BASE
from project.ocr_reader.core.readers import MAX_NEW_TOKENS
from project.ocr_reader.core.scoring import score

OUT_ROOT = OUTPUT
BASE = VL16_BASE
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


TOWER_LORA_LEAVES = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}


def tower_lora_targets(model) -> list[str]:
    """Full names of the tower's attention / MLP Linears + the projector's two."""
    return [
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and (
            (n.startswith("model.visual.") and n.split(".")[-1] in TOWER_LORA_LEAVES)
            or n.startswith("model.projector.linear_")
        )
    ]


PEFT_PREFIX = "base_model.model."
TOWER_FILE = "tower.safetensors"
LM_FILE = "lm.safetensors"


def is_tower_param(name: str) -> bool:
    """Vision tower + projector, by the transformers-side module path (the
    checkpoint's ``visual.`` / ``mlp_AR.`` land at ``model.visual.`` /
    ``model.projector.`` after loading)."""
    n = name.removeprefix(PEFT_PREFIX)
    return n.startswith("model.visual.") or n.startswith("model.projector.")


def is_lm_layer_param(name: str) -> bool:
    """The LM's decoder layers — embed_tokens, final norm and lm_head excluded."""
    return name.removeprefix(PEFT_PREFIX).startswith("model.language_model.layers.")


def save_full(path: Path, params) -> None:
    """bf16 state dict of fully-trained params under base-model key names."""
    from safetensors.torch import save_file

    sd = {
        n.removeprefix(PEFT_PREFIX): p.detach().to(torch.bfloat16).contiguous().cpu()
        for n, p in params
    }
    save_file(sd, str(path))


class Collate:
    """``prompt`` is one string, or one per dataset row (``--sfx_prefix``)."""

    def __init__(
        self,
        proc,
        prompt: str | list[str],
        min_edge: int,
        max_pixels: int = 1280 * 28 * 28,
        pad_to_multiple_of: int | None = None,
    ):
        self.proc, self.prompt, self.min_edge = proc, prompt, min_edge
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_pixels = max_pixels
        self.tok = proc.tokenizer
        self.eos = self.tok.eos_token

    def images_kwargs(self):
        return {
            "size": {"shortest_edge": self.min_edge, "longest_edge": self.max_pixels}
        }

    def __call__(self, items):
        imgs, targets, idx = cd.collate_raw(items)
        images = [Image.fromarray(i[:, :, ::-1]) for i in imgs]
        prompts = [
            self.prompt if isinstance(self.prompt, str) else self.prompt[i] for i in idx
        ]
        texts = [p + t + self.eos for p, t in zip(prompts, targets)]
        # LEFT padding for training too: every row's target then sits at the end,
        # so the loss can be taken on the last K logits only (``logits_to_keep``)
        # instead of materialising fp32 logits over the 103k vocab for every
        # image token — the OOM of the first launch (findings § O2).
        enc = self.proc(
            text=texts,
            images=images,
            padding=True,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
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
                    text=[prompts[r]],
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


def target_loss(model, enc, K: int | None = None) -> torch.Tensor:
    """CE on the target suffix only: keep the last K logits (K = longest label
    run + 1), shift, ignore -100. Pass ``K`` from the CPU labels to skip the
    device sync of reading it off the GPU."""
    labels = enc.pop("labels")
    enc.pop("n_boundary_fallback", None)
    if K is None:
        K = int((labels != -100).sum(1).max()) + 1
    out = model(**enc, logits_to_keep=K)
    logits = out.logits[:, :-1]  # positions seq-K .. seq-2 predict seq-K+1 .. seq-1
    tgt = labels[:, -(K - 1) :]
    # upcast the target positions only, not B x K x 103k (one long label among
    # short ones made that the largest fp32 tensor of the step)
    keep = tgt != -100
    return torch.nn.functional.cross_entropy(logits[keep].float(), tgt[keep])


def train_loss(model, enc, K: int, crop_mean: bool = False) -> torch.Tensor:
    """``target_loss`` for this reader with no host sync: every index, position and mask
    is worked out on the CPU batch and copied up non-blocking. The stock forward reads
    each off the card (``get_rope_index``, the placeholder check, the causal mask's
    ``all()``, boolean indexing and its backward), and each read drains the queue, so
    the GPU idles while the host dispatches what follows.

    ``crop_mean`` averages the CE inside each crop first and then over the crops, so a
    crop weighs the same whatever its label length (a token mean weighs it by length)."""
    from transformers.masking_utils import create_causal_mask
    from transformers.vision_utils import (
        get_vision_cu_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    m = model.get_base_model().model
    text_cfg, vis_cfg = m.language_model.config, m.visual.vision_model.config
    vis_emb = m.visual.vision_model.embeddings
    up = lambda t: t.pin_memory().to("cuda", non_blocking=True)  # noqa: E731

    ids, grid = enc["input_ids"], enc["image_grid_thw"]
    pos, _ = m.get_rope_index(
        ids,
        image_grid_thw=grid,
        attention_mask=enc["attention_mask"],
        mm_token_type_ids=enc["mm_token_type_ids"],
    )
    interp, weights = get_vision_interpolation_indices_and_weights(
        grid,
        num_grid_per_side=vis_emb.num_grid_per_side,
        mode=vis_emb.interpolation_mode,
        align_corners=vis_emb.interpolation_align_corners,
        spatial_merge_size=1,
    )
    cu = get_vision_cu_seqlens(grid)
    if vis_cfg._attn_implementation.startswith("flash"):
        cu = up(cu)  # flash_attn takes the bounds on the card
    at = (ids == m.config.image_token_id).flatten().nonzero().squeeze(1)
    n_img = int(grid.prod(-1).sum()) // vis_cfg.spatial_merge_size**2
    assert len(at) == n_img, f"{len(at)} image tokens, {n_img} image features"
    mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=torch.empty(*ids.shape, 0),  # shape and device only
        attention_mask=enc["attention_mask"],
        past_key_values=None,
    )
    tgt = enc["labels"][:, -(K - 1) :]
    row, col = (tgt != -100).nonzero(as_tuple=True)

    embeds = m.language_model.embed_tokens(up(ids))
    image = m.get_image_features(
        enc["pixel_values"].to("cuda", non_blocking=True),
        grid,
        return_dict=True,
        image_interp_indices=up(interp),
        image_interp_weights=up(weights),
        image_position_ids=up(get_vision_position_ids(grid, 1)),
        image_cu_seqlens=cu,
    ).pooler_output
    image = torch.cat(image, dim=0).to(embeds.dtype)
    embeds = embeds.flatten(0, 1).index_copy(0, up(at), image).view_as(embeds)
    out = model(
        inputs_embeds=embeds,
        position_ids=up(pos),
        attention_mask=None if mask is None else up(mask),
        logits_to_keep=K,
    )
    picked = out.logits[:, :-1][up(row), up(col)]
    if not crop_mean:
        return torch.nn.functional.cross_entropy(picked.float(), up(tgt[row, col]))
    # weights from the CPU labels: 1 / (the crop's target tokens × crops holding a target)
    per_crop = torch.bincount(row, minlength=tgt.shape[0])
    w = 1.0 / (per_crop[row].float() * int((per_crop > 0).sum()))
    ce = torch.nn.functional.cross_entropy(
        picked.float(), up(tgt[row, col]), reduction="none"
    )
    return (ce * up(w)).sum()


@torch.no_grad()
def predict(model, proc, prompt, min_edge, ds: cd.CropDataset, bs, workers):
    model.eval()
    tok = proc.tokenizer
    rng = random.Random(0)
    batches = cd.area_batches(ds.area, bs, rng)
    dl = DataLoader(
        ds, batch_sampler=batches, num_workers=workers, collate_fn=cd.collate_raw
    )
    if not isinstance(prompt, str):  # one prompt per row: batch within a prompt
        batches = []
        for p in sorted(set(prompt)):
            sel = np.array([i for i, q in enumerate(prompt) if q == p])
            batches += [sel[b].tolist() for b in cd.area_batches(ds.area[sel], bs, rng)]
        dl = DataLoader(
            ds, batch_sampler=batches, num_workers=workers, collate_fn=cd.collate_raw
        )
    preds = [""] * len(ds)
    for imgs, _, idx in dl:
        images = [Image.fromarray(i[:, :, ::-1]) for i in imgs]
        inputs = proc(
            text=[prompt if isinstance(prompt, str) else prompt[i] for i in idx],
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            images_kwargs={
                "size": {"shortest_edge": min_edge, "longest_edge": 1280 * 28 * 28}
            },
        ).to(model.device)
        n = inputs["input_ids"].shape[-1]
        o = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True
        )
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
    for k, g in scored.groupby("metric" if "metric" in scored else "kind"):
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
    ap.add_argument(
        "--steps",
        type=int,
        help="stop after this many optimizer steps; the LR decays over them "
        "(a budget-matched arm, e.g. finetune_ovis3_lora.py's --steps)",
    )
    # bs 4 x 4 without gradient checkpointing, batches cut at 3,072 padded tokens, crops
    # capped at 2,048 tokens: 16.7 crops/s at 11.8 GB peak on the costliest batches
    # (smoke_bs4_1117_2258, --largest_first) against 13.7 for bs 2 x 8 / 1,536 and 13.0
    # for bs 8 x 2 with checkpointing; bs 8 without checkpointing is out of memory on 16 GB
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument(
        "--train_tower",
        action="store_true",
        help="also full-finetune the vision tower + projector (fp32 master copy)",
    )
    ap.add_argument("--tower_lr", type=float, default=1e-5)
    ap.add_argument(
        "--tower_rank",
        type=int,
        default=0,
        help="LoRA rank on the tower + projector Linears (0 = no tower LoRA)",
    )
    ap.add_argument(
        "--lm_full",
        action="store_true",
        help="full-finetune the LM decoder layers (fp32 master) instead of the LM LoRA",
    )
    ap.add_argument("--lm_lr", type=float, default=1e-5)
    ap.add_argument(
        "--init_adapter",
        help="start from this trained adapter dir (LP-FT stage 2); "
        "its adapter_config.json overrides --rank/--dropout",
    )
    ap.add_argument(
        "--init_tower",
        help="tower.safetensors (model.visual.* keys) to load into the base before "
        "training — ssl_tower_simmim.py's output; pair with --train_tower",
    )
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--speech_ratio", type=float, default=1.0)
    ap.add_argument(
        "--extra_manifest",
        action="append",
        help="append derived/manifest_<name>.parquet rows to train (O3 colorized crops)",
    )
    ap.add_argument(
        "--extra_repeat", type=int, default=1, help="oversample the extra rows ×N"
    )
    ap.add_argument(
        "--extra_replace",
        action="store_true",
        help="drop each grey row the extra manifest re-cuts (swap, not append)",
    )
    ap.add_argument(
        "--sfx_prefix",
        action="store_true",
        help="two-prompt data (crop_dataset.load_sfxp): OCR: reads speech, SFX: sound "
        "effects, each answers '' on the other; ignores --speech_ratio / --extra_*",
    )
    ap.add_argument(
        "--mainsub",
        action="store_true",
        help="main / sub data (crop_dataset.load_mainsub, main-sub-approach.md): OCR: "
        "reads the main text, OCR2: the sub; best by val main exact",
    )
    ap.add_argument(
        "--neg_share", type=float, default=0.2, help="max negative share per prompt"
    )
    ap.add_argument(
        "--sfxp_det",
        default="sfxp_det_clean",
        help="det manifest for --sfx_prefix (sfxp_det = before the 11-17 clean)",
    )
    ap.add_argument(
        "--sfxp_missed",
        help="--sfx_prefix: also train the detector-missed COO polygons of this manifest "
        "(sfxp_missed, build_sfx_prefix.py --missed) as SFX: rows",
    )
    ap.add_argument("--max_train", type=int, help="rows per kind (subsample)")
    ap.add_argument("--val_limit", type=int)
    ap.add_argument("--val_bs", type=int, default=16)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument(
        "--grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="gradient checkpointing (off by default; on for bs 8 batches on 16 GB)",
    )
    ap.add_argument(
        "--ckpt_lm_only",
        action="store_true",
        help="checkpoint the 18 LM decoder layers only; the tower (~85 %% of the linear "
        "MACs) keeps its activations. Implies --grad_ckpt",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile each tower encoder layer and LM decoder layer, dynamic=False "
        "with the token axis marked per call (core/vl16_compile.py)",
    )
    ap.add_argument(
        "--mem_log",
        action="store_true",
        help="write mem.jsonl: per micro-step peak allocated GB, vision tokens, LM "
        "padded tokens (B x seq) and K",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="training crops' patch-token cap (processor max_pixels = N x 196); 60 sfxp "
        "crops sit above 2,048, a large-glyph note or SFX read the same lower. Val keeps "
        "the processor default (1280x28x28)",
    )
    ap.add_argument(
        "--attn",
        default="sdpa",
        choices=("sdpa", "flash_attention_2"),
        help="attn_implementation; flash_attention_2 needs a wheel with the tower's "
        "head dim 72 (stock flash-attn, not the fa2-sm120 hdim 64/128 build)",
    )
    ap.add_argument(
        "--fa_path",
        help="directory holding a stock flash-attn install (uv pip install --target), put "
        "ahead of the venv's on sys.path; the venv keeps its fa2-sm120 build",
    )
    ap.add_argument(
        "--token_budget",
        type=int,
        default=3072,
        help="cap each batch's padded patch tokens (largest crop × crops; "
        "crop_dataset.token_batches); 0 = fixed --bs area batches",
    )
    ap.add_argument(
        "--loss_mean",
        default="token",
        choices=["token", "crop"],
        help="token: CE averaged over the batch's target tokens (a crop weighs by its "
        "label length); crop: averaged inside each crop, then over the crops",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stock_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="score the base model on val before training (off: the val set and base "
        "rarely change, and the row costs ~3.5 min of generation)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--largest_first",
        action="store_true",
        help="run the largest-area batches first (a peak-VRAM probe with --steps)",
    )
    a = ap.parse_args()
    if a.mainsub and a.sfx_prefix:
        ap.error("--mainsub and --sfx_prefix are two data mixes; pick one")
    if a.tower_rank and a.train_tower:
        ap.error("--tower_rank and --train_tower both train the tower; pick one")
    if a.init_adapter and (a.tower_rank or a.lm_full):
        ap.error(
            "--init_adapter carries its own targets; not with --tower_rank/--lm_full"
        )
    if a.smoke:
        a.max_train, a.val_limit, a.epochs = a.max_train or 480, 64, 1
    if a.ckpt_lm_only:
        a.grad_ckpt = True
    if a.fa_path:
        sys.path.insert(0, str(Path(a.fa_path).resolve()))
        import flash_attn

        print(
            f"flash_attn {flash_attn.__version__} from {flash_attn.__file__}",
            flush=True,
        )

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(BASE))
    min_edge = proc.image_processor.size["shortest_edge"]

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
        )

    prompt = chat("OCR:")
    base = AutoModelForImageTextToText.from_pretrained(
        str(BASE), dtype=torch.bfloat16, attn_implementation=a.attn
    ).to("cuda")
    print(
        f"attn {base.config._attn_implementation} (vision "
        f"{base.config.vision_config._attn_implementation})",
        flush=True,
    )
    base.config.use_cache = False
    if a.init_tower:  # SSL-adapted tower (ssl_tower_simmim.py) as the starting point
        from safetensors.torch import load_file

        sd = load_file(a.init_tower)
        missing, unexpected = base.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        print(f"init tower {a.init_tower} ({len(sd)} tensors)", flush=True)
    for p in base.parameters():
        p.requires_grad_(False)
    targets = [] if a.lm_full else lm_lora_targets(base)
    tower_targets = tower_lora_targets(base) if a.tower_rank else []
    targets += tower_targets
    assert targets, "no LoRA targets — --lm_full needs --tower_rank"
    if a.init_adapter:
        init_dir = Path(a.init_adapter)
        model = PeftModel.from_pretrained(base, str(init_dir), is_trainable=True)
        pc = model.peft_config["default"]
        a.rank, a.dropout = pc.r, pc.lora_dropout
        print(
            f"init adapter {init_dir} (r {pc.r}, alpha {pc.lora_alpha}, "
            f"dropout {pc.lora_dropout}, {len(pc.target_modules)} targets)",
            flush=True,
        )
        if (init_dir / TOWER_FILE).exists():
            from safetensors.torch import load_file

            missing, unexpected = base.load_state_dict(
                load_file(str(init_dir / TOWER_FILE)), strict=False
            )
            print(
                f"init tower {init_dir / TOWER_FILE} (unexpected {len(unexpected)})",
                flush=True,
            )
    else:
        cfg = LoraConfig(
            r=a.rank,
            lora_alpha=2 * a.rank,
            lora_dropout=a.dropout,
            target_modules=targets,
            rank_pattern={re.escape(t): a.tower_rank for t in tower_targets},
            alpha_pattern={re.escape(t): 2 * a.tower_rank for t in tower_targets},
            bias="none",
        )
        model = get_peft_model(base, cfg)
    # (file, params, lr) per fully-trained block; each takes an fp32 master copy
    full_groups: list[tuple[str, list[tuple[str, torch.nn.Parameter]], float]] = []
    for on, file, pick, lr in (
        (a.train_tower, TOWER_FILE, is_tower_param, a.tower_lr),
        (a.lm_full, LM_FILE, is_lm_layer_param, a.lm_lr),
    ):
        if not on:
            continue
        ps = [(n, p) for n, p in model.named_parameters() if pick(n)]
        for _, p in ps:
            p.requires_grad_(True)
        full_groups.append((file, ps, lr))
    full_params = [np_ for _, ps, _ in full_groups for np_ in ps]
    full_masters = [
        p.detach().float().clone().requires_grad_(True) for _, p in full_params
    ]

    # Accumulate the full-FT grads in fp32 on the masters: each bf16 grad is folded in
    # and freed as soon as backward has produced it, so the --grad_accum micro-steps
    # never sum in bf16 (the LoRA adapters are fp32 already: peft autocast_adapter_dtype).
    def fold_into(master: torch.Tensor):
        def hook(p: torch.Tensor) -> None:
            if master.grad is None:
                master.grad = p.grad.float()
            else:
                master.grad.add_(p.grad)
            p.grad = None

        return hook

    for (_, p), m in zip(full_params, full_masters):
        p.register_post_accumulate_grad_hook(fold_into(m))
    if a.grad_ckpt:
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            base.enable_input_require_grads()
        except Exception as e:  # custom modeling file without the hook
            print(f"(gradient checkpointing unavailable: {e})", flush=True)
        if a.ckpt_lm_only:  # GradientCheckpointingLayer reads its own flag per call
            off = [
                m
                for m in base.modules()
                if type(m).__name__ == "PaddleOCRVisionEncoderLayer"
            ]
            for m in off:
                m.gradient_checkpointing = False
            on = sum(
                bool(getattr(m, "gradient_checkpointing", False))
                for m in base.modules()
                if type(m).__name__ == "PaddleOCRDecoderLayer"
            )
            print(
                f"checkpointing: LM decoder layers {on} on, tower layers {len(off)} off",
                flush=True,
            )
            assert on and off, "layer class names changed; --ckpt_lm_only did nothing"
    if a.compile:
        from project.ocr_reader.core.vl16_compile import compile_layers

        # packed tower tokens: one floor-size crop (576) up to the budget plus the
        # augmentation's overshoot (3,164 measured at --token_budget 3072); past 4095
        # compile_layers marks two ranges
        rng_t = (64, int(a.token_budget * 1.3) if a.token_budget else 4095)
        n_t, n_l = compile_layers(base, rng_t, a.bs)
        print(
            f"compiled {n_t} tower layers (T marked {rng_t[0]}-{rng_t[1]}, attention "
            f"core as ocr_vl16::tower_attn) + {n_l} LM layers (S soft-marked, S padded "
            f"to x8, B marked 2-{a.bs})",
            flush=True,
        )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    full_desc = ", ".join(
        f"{file} {sum(p.numel() for _, p in ps) / 1e6:.1f}M in {len(ps)} tensors"
        for file, ps, _ in full_groups
    )
    print(
        f"lora targets {len(targets)} modules (e.g. {targets[0]} … {targets[-1]}; "
        f"{len(tower_targets)} tower at r {a.tower_rank}); "
        f"trainable {n_train / 1e6:.1f}M (full: {full_desc or 'none'})",
        flush=True,
    )
    print(f"prompt {prompt!r}", flush=True)

    if a.mainsub:
        tr_df = cd.load_mainsub(
            "train", neg_share=a.neg_share, limit=a.max_train, seed=a.seed
        )
        va_df = cd.load_mainsub(
            "val", neg_share=a.neg_share, limit=a.val_limit, seed=a.seed
        )
    elif a.sfx_prefix:
        tr_df = cd.load_sfxp(
            "train",
            neg_share=a.neg_share,
            limit=a.max_train,
            seed=a.seed,
            det_manifest=a.sfxp_det,
            missed_manifest=a.sfxp_missed,
        )
        va_df = cd.load_sfxp(
            "val",
            neg_share=a.neg_share,
            limit=a.val_limit,
            seed=a.seed,
            det_manifest=a.sfxp_det,
            missed_manifest=a.sfxp_missed,
        )
    else:
        tr_df = cd.load_split(
            "train",
            speech_ratio=a.speech_ratio,
            limit=a.max_train,
            seed=a.seed,
            extra=a.extra_manifest,
            extra_repeat=a.extra_repeat,
            extra_replace=a.extra_replace,
        )
        va_df = cd.load_split("val", limit=a.val_limit, seed=a.seed)
    # --sfx_prefix crops carry the 4 px margin: no pad jitter, rotation on an expanded canvas
    tr = cd.CropDataset(
        tr_df,
        augment=not a.no_augment,
        seed=a.seed,
        tight=a.sfx_prefix or a.mainsub,
    )
    va = cd.CropDataset(va_df, augment=False)
    tr_prompt, va_prompt = prompt, prompt
    if a.sfx_prefix or a.mainsub:
        tasks = (cd.OCR_TASK, cd.OCR2_TASK) if a.mainsub else (cd.OCR_TASK, cd.SFX_TASK)
        chats = {t: chat(t) for t in tasks}
        tr_prompt = [chats[t] for t in tr_df.task]
        va_prompt = [chats[t] for t in va_df.task]
        print(f"prompts {chats}", flush=True)
        print(
            f"train by task/metric/source "
            f"{tr_df.groupby(['task', 'metric', 'source']).size().to_dict()}",
            flush=True,
        )
    print(
        f"train {len(tr)} ({tr_df.kind.value_counts().to_dict()}; "
        f"source {tr_df.source.value_counts().to_dict()}) val {len(va)} lr {a.lr} bs {a.bs}x{a.grad_accum} r {a.rank} epochs {a.epochs}",
        flush=True,
    )

    max_pixels = a.max_tokens * 196  # 14 px patches
    # --compile: S padded to a multiple of 8 so the LM's SDPA stride guard always holds
    collate = Collate(
        proc,
        tr_prompt,
        min_edge,
        max_pixels,
        pad_to_multiple_of=8 if a.compile else None,
    )
    rng = random.Random(a.seed)
    tr_tokens = cd.vl_tokens(
        tr_df.w.to_numpy(), tr_df.h.to_numpy(), min_edge, max_pixels
    )

    def make_batches(r: random.Random) -> list[list[int]]:
        if a.token_budget:
            return cd.token_batches(tr_tokens, a.token_budget, a.bs, r, padded=True)
        return cd.area_batches(tr.area, a.bs, r)

    def padded_cost(b: list[int]) -> int:
        return int(tr_tokens[b].max()) * len(b)

    probe = make_batches(random.Random(a.seed))
    cost = np.array([padded_cost(b) for b in probe])
    n_rows = np.array([len(b) for b in probe])
    print(
        f"batches {len(probe)} (token_budget {a.token_budget or 'off'}, min_pixels {min_edge}, "
        f"max_tokens {a.max_tokens}): "
        f"padded tokens max {cost.max()} p99 {int(np.percentile(cost, 99))} "
        f"median {int(np.median(cost))}; {int((n_rows < a.bs).sum())} short "
        f"(min {n_rows.min()} crops)",
        flush=True,
    )
    steps_per_epoch = len(probe) // a.grad_accum
    total = steps_per_epoch * a.epochs
    if a.steps:
        a.epochs = -(-a.steps // steps_per_epoch)
        total = a.steps
    if a.smoke:
        total = min(total, 30)
    warm = max(1, int(total * a.warmup))
    full_ids = {id(p) for _, p in full_params}
    lora_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in full_ids
    ]
    groups = [{"params": lora_params, "lr": a.lr}]
    i = 0
    for _, ps, lr in full_groups:
        groups.append({"params": full_masters[i : i + len(ps)], "lr": lr})
        i += len(ps)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0)
    clip_params = lora_params + full_masters
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
        # predict runs under no_grad: vl16_compile's layers take their eager forward
        preds = predict(model, proc, va_prompt, min_edge, va, a.val_bs, a.workers)
        scored = score(va_df, preds)
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

    if a.stock_val:
        evaluate("stock", 0)

    # --mem_log resets the allocator's peak per micro-step; the progress line keeps the
    # run's maximum through peak_all
    mem_f = (out / "mem.jsonl").open("w", buffering=1) if a.mem_log else None
    peak_all = 0.0

    def take_peak() -> float:
        nonlocal peak_all
        gb = torch.cuda.max_memory_allocated() / 2**30
        peak_all = max(peak_all, gb)
        torch.cuda.reset_peak_memory_stats()
        return gb

    def run_peak() -> float:
        return max(peak_all, torch.cuda.max_memory_allocated() / 2**30)

    model.train()
    step, micro, t0, seen = 0, 0, time.time(), 0
    best = (-1.0, None)
    best_key = "main_exact" if a.mainsub else "sfx_exact"
    for ep in range(1, a.epochs + 1):
        batches = make_batches(rng)
        if a.largest_first:
            batches.sort(key=lambda b: -padded_cost(b))
        dl = DataLoader(
            tr,
            batch_sampler=batches,
            num_workers=a.workers,
            collate_fn=collate,
            persistent_workers=False,
            pin_memory=True,
        )
        # no host sync inside the step: K is read off the CPU labels, the copy is
        # non-blocking from pinned memory, and losses stay on the card until logged
        losses = []
        for enc, _ in dl:
            K = int((enc["labels"] != -100).sum(1).max()) + 1
            if mem_f:
                rec = {
                    "micro": micro,
                    "vis": int(enc["image_grid_thw"].prod(-1).sum()),
                    "lm": int(enc["input_ids"].numel()),
                    "B": int(enc["input_ids"].shape[0]),
                    "K": K,
                }
            n = enc["input_ids"].shape[0]
            raw = train_loss(model, enc, K, crop_mean=a.loss_mean == "crop")
            # a short (token-budget) batch weighs by its crops, as if part of a --bs batch
            loss = raw * (min(n, a.bs) / a.bs) / a.grad_accum
            loss.backward()
            if mem_f:
                rec["peak"] = round(take_peak(), 3)
                mem_f.write(json.dumps(rec) + "\n")
            micro += 1
            seen += n
            losses.append(raw.detach())
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for (_, p), m in zip(full_params, full_masters):
                    p.copy_(m)
            if mem_f:
                mem_f.write(
                    json.dumps({"step": step, "peak": round(take_peak(), 3)}) + "\n"
                )
            step += 1
            if step % 25 == 0 or step == total:
                el = time.time() - t0
                print(
                    f"ep {ep} step {step}/{total} loss {torch.stack(losses[-25 * a.grad_accum :]).mean().item():.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {seen / el:.1f} crops/s t {el:.0f}s "
                    f"seq {enc['input_ids'].shape[-1]} vram {run_peak():.1f}G",
                    flush=True,
                )
            if step >= total:
                break
        # Save BEFORE scoring: a scorer crash (an import moved under the
        # anime_tools pin, 2026-09-08) must not cost the epoch's weights.
        ep_dir = out / f"ep{ep}"
        model.save_pretrained(ep_dir)
        for file, ps, _ in full_groups:
            save_full(ep_dir / file, ps)
        m = evaluate(f"ep{ep}", step)
        m["train_loss"] = torch.stack(losses).mean().item()
        if m[best_key] > best[0]:
            best = (m[best_key], ep)
            b = out / "best"
            if b.is_symlink() or b.exists():
                b.unlink() if b.is_symlink() else shutil.rmtree(b)
            b.symlink_to(ep_dir.name)
        if step >= total:
            break

    rows = [json.loads(ln) for ln in hist.read_text().splitlines()]
    md = [
        f"# {a.run} — PaddleOCR-VL-1.6 crop LoRA (r {a.rank}, lr {a.lr}, bs {a.bs}x{a.grad_accum}, "
        + (f"tower full FT lr {a.tower_lr}, " if a.train_tower else "")
        + (f"tower LoRA r {a.tower_rank}, " if a.tower_rank else "")
        + (f"LM layers full FT lr {a.lm_lr}, " if a.lm_full else "")
        + (f"init {a.init_adapter}, " if a.init_adapter else "")
        + f"speech_ratio {a.speech_ratio}, train {len(tr)}, val {len(va)})\n",
    ]
    # one column group per val metric (sfx / speech / *_neg, or main / sub / *_neg)
    groups = (
        [k[: -len("_exact")] for k in rows[-1] if k.endswith("_exact")] if rows else []
    )
    md.append("| tag | " + " | ".join(f"{g} exact | {g} sim" for g in groups) + " |")
    md.append("|---|" + "---|---|" * len(groups))
    for r in rows:
        md.append(
            f"| {r['tag']} | "
            + " | ".join(
                f"{100 * r.get(g + '_exact', float('nan')):.1f} % | "
                f"{r.get(g + '_sim', float('nan')):.3f}"
                for g in groups
            )
            + " |"
        )
    md.append(f"\nbest = ep{best[1]} (val {best_key} {100 * best[0]:.1f} %)")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
