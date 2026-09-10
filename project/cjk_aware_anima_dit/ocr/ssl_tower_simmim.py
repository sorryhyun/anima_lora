#!/usr/bin/env python3
"""SimMIM-style masked patch reconstruction on PaddleOCR-VL-1.6's vision tower —
self-supervised domain adaptation on unlabelled text crops (2026-09-08).

    ANIMA_ANIMETEXT_ROOT=… make daemon-run ARGS="--stall-timeout 600 \\
        project/cjk_aware_anima_dit/ocr/ssl_tower_simmim.py --run simmim_at --epochs 5"
    … --smoke                       # 30 steps, 64-crop val, throughput + a recon sheet
    … --manifest <path.parquet>     # any manifest with ``path`` (+ ``w``, ``h``) columns

Why (findings § tower): unfreezing the tower under COO supervision took the
sincos gate 106 → 304 and the garbage misses 185 → 43 — perception, not the
LM, was the bottleneck, and *grey Manga109* supervision transferred to colour
doujin. This is the vision-only, label-free version of the same lever: adapt
the tower to in-domain text pixels, then re-run the COO SFT
(``finetune_vl16_lora.py --train_tower --init_tower <ep>/tower.safetensors``)
to re-align projector + LM. ♡ / small-kana omissions are an LM-side label gap
and are **not** expected to move here.

**Target (``--target``, decision 1 of ``plan_ssl_tower.md``).** ``feat`` (default):
the masked tokens' features are regressed to the **frozen stock tower's**
``last_hidden_state`` on the unmasked crop (smooth-L1 on the post-LN features —
data2vec / BEiT-v2 with the base as its own tokenizer), plus ``--kd_unmasked`` ×
the same loss on the unmasked tokens as an anchor. The target space *is* the
projector's input, so the SFT re-aligns from a nearby point. ``pixel`` is the
2026-09-08 smoke's objective, kept for the record: 30 steps at lr 2e-5 moved
the weights by 4e-4 (median relative) and collapsed the output (cosine to stock
0.38, norm ×0.22, the stock LM read ``""``) — a pixel target on the final
features drags the representation out of the space the LM reads.

Objective (SimMIM, Xie et al. 2022, the linear-head variant): tokens are the
tower's own 14-px patches (``pixel_values`` = ``(N, 3, 14, 14)`` per crop set,
already normalised); a random ``--mask_ratio`` of the **2×2 merge blocks**
(28 px, the projector's unit) is replaced by a learned mask token *after* the
patch conv and *before* the position embedding (a forward hook on
``embeddings.patch_embedding``); the encoder runs unchanged (NaViT packing,
2-D RoPE, ``grid_thw``), and a linear head on ``last_hidden_state`` regresses
the 588 raw pixel values of each masked patch under L1. No decoder, no
teacher. Block ids come from ``get_vision_position_ids(grid_thw, 1)`` so the
mask follows the model's own token order.

Training: the whole tower (413 M; projector / LM are dropped from memory) in
the fp32-master AdamW pattern of ``finetune_vl16_lora.py`` (bf16 forward, fp32
update, copy back) at ``--lr`` cosine, ``--wd`` on matrices only, grad clip
1.0, gradient checkpointing. Per epoch: held-out masked L1, a recon contact
sheet (``ep<N>/recon.png``: masked input / reconstruction / original), and
``ep<N>/tower.safetensors`` under base-model key names (``model.visual.*``,
bf16) — what ``--init_tower`` loads. Output ``output/ocr/<run>/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_dataset as cd  # noqa: E402
import manga109 as m109  # noqa: E402
from animetext_crops import animetext_root  # noqa: E402

OUT_ROOT = m109.REPO / "output/ocr"
BASE = m109.REPO / "models/paddleocr_vl_1.6"
PATCH = 14
MERGE = 2
TOWER_PREFIX = "model.visual."


class UnlabelledCrops(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path):
        self.paths = [str(root / p) for p in df.path]
        self.area = (df.w * df.h).to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])
        if img is None:
            raise FileNotFoundError(self.paths[i])
        return img, "", i


def block_mask(
    grid_thw: torch.Tensor, ratio: float, gen: torch.Generator
) -> torch.Tensor:
    """Boolean mask over the packed token sequence, one draw per 2×2 block."""
    from transformers.vision_utils import get_vision_position_ids

    pos = get_vision_position_ids(grid_thw, 1)  # (N, 2): (row, col) in token order
    masks = []
    start = 0
    for t, h, w in grid_thw.tolist():
        n = t * h * w
        hb, wb = h // MERGE, w // MERGE
        nb = hb * wb
        k = max(1, int(round(nb * ratio)))
        perm = torch.randperm(nb, generator=gen)
        bm = torch.zeros(nb, dtype=torch.bool)
        bm[perm[:k]] = True
        r = pos[start : start + n, 0] // MERGE
        c = pos[start : start + n, 1] // MERGE
        masks.append(bm[(r * wb + c).clamp_(max=nb - 1)])
        start += n
    return torch.cat(masks)


def save_tower(path: Path, visual: torch.nn.Module) -> None:
    from safetensors.torch import save_file

    sd = {
        TOWER_PREFIX + n: p.detach().to(torch.bfloat16).contiguous().cpu()
        for n, p in visual.named_parameters()
    }
    save_file(sd, str(path))


def recon_sheet(pv, thw, mask, pred, path: Path, n_max: int = 8) -> None:
    """masked input | reconstruction | original, one row per crop (first n_max)."""
    pv = pv.float().cpu()
    pred = pred.float().cpu()
    mask = mask.cpu()
    full_pred = pv.clone().flatten(1)
    full_pred[mask] = pred
    full_pred = full_pred.view_as(pv)
    masked = pv.clone()
    masked[mask] = 0.0
    rows = []
    start = 0
    for t, h, w in thw.tolist()[:n_max]:
        n = t * h * w

        def tile(x):
            x = x[start : start + n].view(h, w, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2)
            x = x.reshape(h * PATCH, w * PATCH, 3)
            return ((x * 0.5 + 0.5).clamp(0, 1) * 255).byte().numpy()[:, :, ::-1]

        row = np.concatenate([tile(masked), tile(full_pred), tile(pv)], axis=1)
        rows.append(row)
        start += n
    W = max(r.shape[1] for r in rows)
    rows = [
        np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=128)
        for r in rows
    ]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--manifest",
        help="default: $ANIMA_ANIMETEXT_ROOT/animetext_crops/manifest_test.parquet",
    )
    ap.add_argument(
        "--manifest_name",
        help="manifest_test_<name>.parquet under $ANIMA_ANIMETEXT_ROOT/animetext_crops "
        "(the root may hold spaces the daemon's ARGS cannot carry)",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--target", choices=["feat", "pixel"], default="feat")
    ap.add_argument(
        "--kd_unmasked",
        type=float,
        default=0.1,
        help="--target feat: weight of the feature loss on the UNmasked tokens",
    )
    ap.add_argument("--val", type=int, default=512, help="held-out crops")
    ap.add_argument("--max_train", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.val, a.max_train = 64, a.max_train or 960

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    if a.manifest:
        mpath = Path(a.manifest)
        root = mpath.parent.parent
    else:
        root = animetext_root()
        name = a.manifest_name or ("smoke" if a.smoke else "")
        mpath = (
            root
            / "animetext_crops"
            / f"manifest_test{'_' + name if name else ''}.parquet"
        )
    df = pd.read_parquet(mpath)
    df = df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    va_df, tr_df = df.iloc[: a.val], df.iloc[a.val :]
    if a.max_train:
        tr_df = tr_df.iloc[: a.max_train]
    tr, va = UnlabelledCrops(tr_df, root), UnlabelledCrops(va_df, root)
    print(f"manifest {mpath}: train {len(tr)} val {len(va)} crops", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(BASE))
    ip = proc.image_processor
    ikw = {
        "size": {
            "shortest_edge": ip.size["shortest_edge"],
            "longest_edge": 1280 * 28 * 28,
        }
    }
    full = AutoModelForImageTextToText.from_pretrained(
        str(BASE), dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if not a.no_grad_ckpt:
        full.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    visual = full.model.visual
    del full.model.language_model, full.lm_head, full.model.projector
    del full
    gc.collect()
    teacher = None
    if a.target == "feat":  # the frozen stock tower — the target space
        import copy

        teacher = copy.deepcopy(visual).to("cuda").eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    visual = visual.to("cuda").train()
    for p in visual.parameters():
        p.requires_grad_(True)
    D = visual.config.hidden_size
    mask_token = torch.nn.Parameter(torch.zeros(D, device="cuda"))
    torch.nn.init.normal_(mask_token, std=0.02)
    head = torch.nn.Linear(D, 3 * PATCH * PATCH if a.target == "pixel" else D).to(
        "cuda"
    )
    if a.target == "feat":  # start as identity: the student's own features
        torch.nn.init.eye_(head.weight)
        torch.nn.init.zeros_(head.bias)
    state = {"mask": None}

    def patch_hook(_m, _i, out):  # (N, D, 1, 1) after the patch conv, before pos-embed
        m = state["mask"]
        if m is None:
            return out
        return torch.where(
            m.view(-1, 1, 1, 1), mask_token.to(out.dtype).view(1, -1, 1, 1), out
        )

    visual.vision_model.embeddings.patch_embedding.register_forward_hook(patch_hook)

    tower_params = list(visual.named_parameters())
    masters = [p.detach().float().clone().requires_grad_(True) for _, p in tower_params]
    print(
        f"tower {sum(p.numel() for _, p in tower_params) / 1e6:.1f}M in {len(tower_params)} tensors; "
        f"target {a.target} (kd_unmasked {a.kd_unmasked}), mask ratio {a.mask_ratio}, "
        f"lr {a.lr}, bs {a.bs}x{a.grad_accum}, epochs {a.epochs}",
        flush=True,
    )
    decay = [m for m in masters if m.dim() >= 2]
    no_decay = [m for m in masters if m.dim() < 2] + [mask_token]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": a.wd},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": head.parameters(), "weight_decay": a.wd, "lr": a.lr * 10},
        ],
        lr=a.lr,
        betas=(0.9, 0.999),
    )
    clip_params = masters + [mask_token] + list(head.parameters())
    rng = random.Random(a.seed)
    steps_per_epoch = max(1, len(tr) // a.bs // a.grad_accum)
    total = steps_per_epoch * a.epochs
    if a.smoke:
        total = min(total, 30)
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            s / warm
            if s < warm
            else 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, total - warm))))
        ),
    )

    def encode(imgs, gen):
        enc = ip(
            images=[Image.fromarray(i[:, :, ::-1]) for i in imgs],
            return_tensors="pt",
            **ikw,
        )
        pv, thw = enc["pixel_values"], enc["image_grid_thw"]
        mask = block_mask(thw, a.mask_ratio, gen)
        return pv.to("cuda"), thw.to("cuda"), mask.to("cuda")

    def step_loss(pv, thw, mask):
        x = pv.to(torch.bfloat16).unsqueeze(0)
        state["mask"] = mask
        feats = visual(pixel_values=x, grid_thw=thw).last_hidden_state
        state["mask"] = None
        if a.target == "pixel":
            pred = head(feats[mask].float())
            return torch.nn.functional.l1_loss(pred, pv[mask].flatten(1)), pred
        with torch.no_grad():
            tgt = teacher(pixel_values=x, grid_thw=thw).last_hidden_state.float()
        out = head(feats.float())
        sl1 = torch.nn.functional.smooth_l1_loss
        loss = sl1(out[mask], tgt[mask])
        if a.kd_unmasked > 0:
            loss = loss + a.kd_unmasked * sl1(out[~mask], tgt[~mask])
        # the recon sheet wants pixels: none under feat — hand back the input
        return loss, pv[mask].flatten(1)

    @torch.no_grad()
    def evaluate(tag):
        visual.eval()
        gen = torch.Generator().manual_seed(1234)
        dl = DataLoader(
            va, batch_size=16, num_workers=a.workers, collate_fn=cd.collate_raw
        )
        tot, n, sheet = 0.0, 0, None
        for imgs, _, _ in dl:
            pv, thw, mask = encode(imgs, gen)
            loss, pred = step_loss(pv, thw, mask)
            tot += loss.item() * len(imgs)
            n += len(imgs)
            if sheet is None:
                sheet = (pv, thw, mask, pred)
        visual.train()
        ep_dir = out / tag
        ep_dir.mkdir(exist_ok=True)
        if a.target == "pixel":  # feature targets have nothing to draw
            recon_sheet(*sheet, ep_dir / "recon.png")
        return tot / max(1, n)

    hist = out / "history.jsonl"
    hist.write_text("")
    v0 = evaluate("stock")
    print(f"[val stock] masked L1 {v0:.4f}", flush=True)
    hist.open("a").write(json.dumps({"tag": "stock", "step": 0, "val_l1": v0}) + "\n")

    step, micro, t0, seen = 0, 0, time.time(), 0
    gen = torch.Generator().manual_seed(a.seed)
    for ep in range(1, a.epochs + 1):
        dl = DataLoader(
            tr,
            batch_sampler=cd.area_batches(tr.area, a.bs, rng),
            num_workers=a.workers,
            collate_fn=cd.collate_raw,
        )
        losses = []
        for imgs, _, _ in dl:
            pv, thw, mask = encode(imgs, gen)
            loss, _ = step_loss(pv, thw, mask)
            (loss / a.grad_accum).backward()
            micro += 1
            seen += len(imgs)
            losses.append(loss.item())
            if micro % a.grad_accum:
                continue
            for (_, p), m in zip(tower_params, masters):
                m.grad = p.grad.float()
                p.grad = None
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for (_, p), m in zip(tower_params, masters):
                    p.copy_(m)
            step += 1
            if step % 25 == 0 or step == total or step == 1:
                el = time.time() - t0
                print(
                    f"ep {ep} step {step}/{total} loss {np.mean(losses[-25 * a.grad_accum :]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {seen / el:.1f} crops/s tokens {pv.shape[0]} "
                    f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                    flush=True,
                )
            if step >= total:
                break
        ep_dir = out / f"ep{ep}"
        ep_dir.mkdir(exist_ok=True)
        save_tower(ep_dir / "tower.safetensors", visual)
        torch.save(
            {"mask_token": mask_token.detach().cpu(), "head": head.state_dict()},
            ep_dir / "ssl_head.pt",
        )
        v = evaluate(f"ep{ep}")
        rec = {
            "tag": f"ep{ep}",
            "step": step,
            "val_l1": v,
            "train_l1": float(np.mean(losses)),
        }
        hist.open("a").write(json.dumps(rec) + "\n")
        print(
            f"[val ep{ep}] masked L1 {v:.4f} (train {rec['train_l1']:.4f}) → {ep_dir}",
            flush=True,
        )
        if step >= total:
            break


if __name__ == "__main__":
    main()
