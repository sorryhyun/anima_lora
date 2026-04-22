#!/usr/bin/env python
"""Phase 1.5 — Tag-anchored Perceiver Resampler.

Uses inversionv2's class prototypes to hard-anchor the "prototype-addressable"
slots (one per group in ``anchors.yaml``) from a classifier over the pooled
encoder feature. The resampler then only has to fit the content-heavy
mid/tail slots.

Why: phase-1 4-layer resampler at step 2500 matched the phase-0 1-layer probe
on mid/tail (cos 0.501 / 0.630 vs 0.501 / 0.631) — depth didn't help. inversionv2
showed rating / 1girl / solo / @artist sit in k@95 ≤ 5 per slot with within-class
cos > 0.9, so classifier + frozen-prototype lookup is the right shape for those
slots. Auxiliary CE (mutex) / BCE (multi-label) on the classifier heads
doubles as a diagnostic for encoder quality.

Usage:
    python scripts/img2emb/phase1_5_anchored.py
    python scripts/img2emb/phase1_5_anchored.py --steps 5000 --eval_every 1000
    python scripts/img2emb/phase1_5_anchored.py --anchors_yaml my_anchors.yaml
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# results/ still lives under bench/img2emb/ from before the move; keep writing there.
BENCH_DIR = REPO_ROOT / "bench" / "img2emb"
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.anchors import (  # noqa: E402
    AnchorSpec,
    AnchoredResampler,
    aux_cls_loss,
    build_anchor_labels,
    collate_anchor_batch,
    gather_sample_labels,
    index_flat_labels,
    inject_spec_anchors,
    labels_to_flat_tensors,
    load_anchor_spec,
)
from scripts.img2emb.data import (  # noqa: E402
    _infonce_loss,
    _pool,
    _resampler_loss,
    active_slice,
    load_cache,
)
from library.log import setup_logging  # noqa: E402
import torch.nn.functional as F  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_ANCHORS_YAML = Path(__file__).parent / "anchors.yaml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", default=str(BENCH_DIR / "results" / "phase0"))
    p.add_argument("--out_dir", default=str(BENCH_DIR / "results" / "phase1_5"))
    p.add_argument(
        "--tag_slot_dir",
        default=str(REPO_ROOT / "output" / "img2embs" / "anchors"),
        help="Directory with phase1_positions.json + phase2_class_prototypes.safetensors "
        "(produced by scripts/img2emb/rebuild_anchor_artifacts.py).",
    )
    p.add_argument(
        "--anchors_yaml",
        default=str(DEFAULT_ANCHORS_YAML),
        help="YAML spec listing anchor groups + classes.",
    )
    p.add_argument("--image_dir", default=None)
    p.add_argument("--encoder", default="siglip2")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument(
        "--n_slots",
        type=int,
        default=256,
        help="Resampler query count K. Default 256 (hard K-cap per proposal.md part 1). "
             "Target is sliced to [:K] and only slot < min(L, K) is supervised; "
             "predictions are zero-padded back to 512 at eval/inference boundary.",
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--cos_loss_weight", type=float, default=1.0)
    p.add_argument(
        "--cls_loss_weight",
        type=float,
        default=0.1,
        help="Weight on auxiliary classifier loss (CE for mutex, BCE for multi-label).",
    )
    p.add_argument(
        "--w_infonce",
        type=float,
        default=0.1,
        help="Weight on multi-positive InfoNCE over pooled per-variant targets "
             "(see proposal.md part 2). Set 0 to disable.",
    )
    p.add_argument(
        "--infonce_tau",
        type=float,
        default=0.07,
        help="InfoNCE temperature (CLIP default).",
    )
    p.add_argument(
        "--anchor_mode",
        choices=["replace", "residual"],
        default="replace",
        help="replace = hard-write prototype mix into anchor slot; "
             "residual = add to resampler output at anchor slot.",
    )
    p.add_argument("--seed", type=int, default=20260421)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_predictions", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- dataset


class AnchoredTrainDataset(Dataset):
    """Same as phase1's dataset but yields per-stem anchor labels too."""

    def __init__(
        self,
        spec: AnchorSpec,
        train_indices: list[int],
        te_paths: list[str],
        active_lengths: list[int],
        flat_labels: dict[str, torch.Tensor],
        num_variants: int,
    ):
        self.spec = spec
        self.train_indices = train_indices
        self.te_paths = te_paths
        self.active_lengths = active_lengths
        self.flat_labels = flat_labels
        self.V = num_variants

    def __len__(self) -> int:
        return len(self.train_indices)

    def __getitem__(self, pos: int):
        full_idx = self.train_indices[pos]
        sd = load_file(self.te_paths[full_idx])
        if self.V > 1:
            v = int(torch.randint(0, self.V, (1,)).item())
            target = sd[f"crossattn_emb_v{v}"].float()
        else:
            target = sd["crossattn_emb"].float()
        L = int(self.active_lengths[full_idx])
        if L < target.shape[0]:
            target = target.clone()
            target[L:] = 0
        anchors = gather_sample_labels(self.spec, self.flat_labels, full_idx)
        return full_idx, target.to(torch.bfloat16), L, anchors


def _collate(spec: AnchorSpec):
    def _fn(batch):
        full_idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
        target = torch.stack([b[1] for b in batch], dim=0)
        L = torch.tensor([b[2] for b in batch], dtype=torch.long)
        anchors = collate_anchor_batch(spec, [b[3] for b in batch])
        return full_idx, target, L, anchors
    return _fn


# --------------------------------------------------------------------------- schedule


def _warmup_cosine(step: int, total: int, warmup: int, eta_min_frac: float = 0.05):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return eta_min_frac + (1 - eta_min_frac) * 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- eval


def _slot_agg_cos(cos_mat: torch.Tensor, mask: torch.Tensor) -> list:
    N, S = cos_mat.shape
    out = []
    for s in range(S):
        m = mask[:, s]
        n_active = int(m.sum().item())
        if n_active < 2:
            out.append({"slot": s, "active": False, "n": n_active})
            continue
        c = cos_mat[m, s]
        out.append(
            {
                "slot": s,
                "active": True,
                "n": n_active,
                "cos_mean": float(c.mean().item()),
                "cos_median": float(c.median().item()),
                "cos_p10": float(torch.quantile(c, 0.10).item()),
            }
        )
    return out


def _strata(per_slot: list) -> dict:
    bands = {
        "prefix_0_8": lambda s: s < 8,
        "mid_8_64": lambda s: 8 <= s < 64,
        "tail_64_256": lambda s: 64 <= s < 256,
        "content_all": lambda s: 0 <= s < 256,
    }

    def _agg(slots, key):
        vals = [s[key] for s in slots if s.get("active") and key in s and not math.isnan(s[key])]
        if not vals:
            return None
        return {
            "n_slots": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
        }

    out = {}
    for name, pred in bands.items():
        slots = [s for s in per_slot if pred(s["slot"])]
        out[name] = {
            "cos_mean": _agg(slots, "cos_mean"),
            "cos_median": _agg(slots, "cos_median"),
            "cos_p10": _agg(slots, "cos_p10"),
        }
    return out


@torch.no_grad()
def _run_pred(
    model: AnchoredResampler,
    spec: AnchorSpec,
    tokens_all: torch.Tensor,
    pooled_all: torch.Tensor,
    flat_labels: dict[str, torch.Tensor],
    eval_idx: list[int],
    device: torch.device,
    anchor_mode: str,
    batch: int = 8,
    pad_to: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Run resampler + anchor injection over eval_idx; zero-pad the (B, K, D)
    output to (B, pad_to, D) so downstream eval can keep its 512-shaped contract.
    """
    out_pred = []
    cls_hits: dict[str, list[int]] = {g.name: [0, 0] for g in spec.groups}
    for i in range(0, len(eval_idx), batch):
        ids = eval_idx[i : i + batch]
        ids_t = torch.tensor(ids, dtype=torch.long)
        tok_b = tokens_all[ids_t].to(device=device, dtype=torch.bfloat16)
        pool_b = pooled_all[ids_t].to(device=device, dtype=torch.float32)
        fwd = model(tok_b, pool_b)
        pred = fwd["pred"].detach().float()

        batch_labels = index_flat_labels(spec, flat_labels, ids_t)
        # Move slot-related labels to device for injection.
        dev_labels = {k: v.to(device) for k, v in batch_labels.items()}
        detached_anchor = {k: v.detach().float() for k, v in fwd["anchor_emb"].items()}
        inject_spec_anchors(pred, spec, detached_anchor, dev_labels, mode=anchor_mode)

        K_pred = pred.shape[1]
        if K_pred < pad_to:
            pred = F.pad(pred, (0, 0, 0, pad_to - K_pred))
        out_pred.append(pred.cpu())

        for g in spec.groups:
            lg = fwd["logits"][g.name]
            if g.mutex:
                tg = batch_labels[f"{g.name}_class"]
                m = tg >= 0
                if not m.any():
                    continue
                pred_cls = lg.argmax(dim=-1).cpu()
                n_c = int((pred_cls[m] == tg[m]).sum().item())
                cls_hits[g.name][0] += n_c
                cls_hits[g.name][1] += int(m.sum().item())
            else:
                tg = batch_labels[f"{g.name}_labels"]
                lg_c = lg[..., : g.n_classes].cpu()
                pred_pos = (torch.sigmoid(lg_c) > 0.5).float()
                cls_hits[g.name][0] += int((pred_pos == tg).sum().item())
                cls_hits[g.name][1] += int(tg.numel())

    pred_eval = torch.cat(out_pred, dim=0)
    cls_acc = {
        name: (hit / tot) if tot else float("nan")
        for name, (hit, tot) in cls_hits.items()
    }
    return pred_eval, cls_acc


@torch.no_grad()
def _eval_per_variant(
    pred_eval: torch.Tensor,
    te_paths: list[str],
    eval_idx: list[int],
    active_lengths: list[int],
) -> dict:
    N, S, D = pred_eval.shape
    mask = active_slice([active_lengths[i] for i in eval_idx])
    pred_n = F.normalize(pred_eval.float(), dim=-1, eps=1e-8)

    cos_vs_mean = torch.zeros(N, S)
    cos_best = torch.zeros(N, S)
    cos_mean_v = torch.zeros(N, S)

    for n, full_idx in enumerate(tqdm(eval_idx, desc="eval")):
        sd = load_file(te_paths[full_idx])
        variant_keys = sorted(k for k in sd.keys() if k.startswith("crossattn_emb_v"))
        if variant_keys:
            variants = torch.stack([sd[k].float() for k in variant_keys], dim=0)
        else:
            variants = sd["crossattn_emb"].float().unsqueeze(0)
        L = int(active_lengths[full_idx])
        if L < variants.shape[1]:
            variants[:, L:] = 0

        t_mean = variants.mean(dim=0)
        tm_n = F.normalize(t_mean, dim=-1, eps=1e-8)
        var_n = F.normalize(variants, dim=-1, eps=1e-8)
        p_n = pred_n[n]

        cos_vs_mean[n] = (p_n * tm_n).sum(dim=-1)
        cos_all = (p_n.unsqueeze(0) * var_n).sum(dim=-1)
        cos_best[n] = cos_all.max(dim=0).values
        cos_mean_v[n] = cos_all.mean(dim=0)

    vs_mean_ps = _slot_agg_cos(cos_vs_mean, mask)
    best_ps = _slot_agg_cos(cos_best, mask)
    mean_v_ps = _slot_agg_cos(cos_mean_v, mask)
    return {
        "vs_mean": {"per_slot": vs_mean_ps, "stratified": _strata(vs_mean_ps)},
        "best_over_v": {"per_slot": best_ps, "stratified": _strata(best_ps)},
        "mean_over_v": {"per_slot": mean_v_ps, "stratified": _strata(mean_v_ps)},
    }


def _log_eval(result: dict, cls_acc: dict, tag: str):
    for metric in ("vs_mean", "best_over_v", "mean_over_v"):
        strat = result[metric]["stratified"]
        parts = []
        for band in ("prefix_0_8", "mid_8_64", "tail_64_256", "content_all"):
            g = strat[band]["cos_median"]
            v = g["mean"] if g else float("nan")
            parts.append(f"{band}={v:.3f}")
        logger.info(f"  [{tag}/{metric}] " + "  ".join(parts))
    acc_parts = [f"{name}={v:.3f}" for name, v in cls_acc.items()]
    logger.info(f"  [{tag}/cls_acc] " + "  ".join(acc_parts))


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_slot_dir = Path(args.tag_slot_dir)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    image_dir = args.image_dir
    if image_dir is None:
        act = json.loads((cache_dir / "active_lengths.json").read_text())
        image_dir = act.get("image_dir", "post_image_dataset")

    logger.info(
        f"encoder={args.encoder}  cache={cache_dir}  image_dir={image_dir}  "
        f"out={out_dir}  tag_slot={tag_slot_dir}  anchors={args.anchors_yaml}  "
        f"anchor_mode={args.anchor_mode}"
    )
    cache = load_cache(cache_dir, image_dir, args.encoder, args.num_workers)

    train_idx = cache["split"]["train_idx"]
    eval_idx = cache["split"]["eval_idx"]
    tokens_all = cache["tokens"]
    pooled_all = cache["pooled"]
    V = int(cache["num_variants"])
    S, D_y = cache["target_shape"]
    d_enc = int(tokens_all.shape[-1])
    d_pool = int(pooled_all.shape[-1])
    K = int(args.n_slots)
    if K > S:
        raise ValueError(f"--n_slots={K} exceeds cached target S={S}")
    logger.info(
        f"N_train={len(train_idx)}  N_eval={len(eval_idx)}  V={V}  "
        f"S={S}  K={K}  D_y={D_y}  d_enc={d_enc}  d_pool={d_pool}"
    )

    # InfoNCE target (optional). Loaded by data.load_cache if the artifact exists.
    tgt_pooled_all = cache.get("target_pooled")
    use_infonce = bool(args.w_infonce > 0.0 and tgt_pooled_all is not None)
    if args.w_infonce > 0.0 and tgt_pooled_all is None:
        logger.warning(
            "w_infonce > 0 but features/target_pooled.safetensors is missing — "
            "re-run extract_features.py or preprocess-img2emb to regenerate. "
            "InfoNCE disabled for this run."
        )

    # Anchor spec + labels
    spec = load_anchor_spec(Path(args.anchors_yaml), tag_slot_dir)
    anchor_labels = build_anchor_labels(
        spec, tag_slot_dir / "phase1_positions.json", cache["stems"]
    )
    flat_labels = labels_to_flat_tensors(spec, anchor_labels)

    model = AnchoredResampler(
        spec=spec,
        d_enc=d_enc,
        d_pool=d_pool,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_slots=K,
        n_layers=args.n_layers,
        d_out=D_y,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    logger.info(
        f"anchored resampler: {args.n_layers} layers × d={args.d_model} × "
        f"heads={args.n_heads}  trainable={n_params / 1e6:.1f}M  "
        f"frozen_protos={n_buf / 1e6:.1f}M"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_frac * args.steps))

    train_ds = AnchoredTrainDataset(
        spec=spec,
        train_indices=train_idx,
        te_paths=cache["te_paths"],
        active_lengths=cache["active_lengths"],
        flat_labels=flat_labels,
        num_variants=V,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=_collate(spec),
    )

    train_log = []
    intermediate = []
    t0 = time.time()
    pbar = tqdm(total=args.steps, desc=f"train/{args.encoder}", leave=True)
    step = 0
    done = False
    while not done:
        for full_idx, tgt_b, L_b, anchors_b in loader:
            if step >= args.steps:
                done = True
                break
            lr_scale = _warmup_cosine(step, args.steps, warmup)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_scale

            tok_b = tokens_all[full_idx].to(device=device, dtype=torch.bfloat16)
            pool_b = pooled_all[full_idx].to(device=device, dtype=torch.float32)
            tgt_b = tgt_b.to(device=device, dtype=torch.float32)[:, :K]
            L_clip = L_b.clamp_max(K).to(device)
            mask_b = torch.arange(K, device=device).unsqueeze(0) < L_clip.unsqueeze(1)

            dev_labels = {k: v.to(device) for k, v in anchors_b.items()}
            fwd = model(tok_b, pool_b)
            pred = fwd["pred"]

            inject_spec_anchors(pred, spec, fwd["anchor_emb"], dev_labels, mode=args.anchor_mode)

            loss_reg, comps = _resampler_loss(
                pred, tgt_b, mask_b,
                cos_w=args.cos_loss_weight, zero_w=0.0,
            )
            cls_loss, accs = aux_cls_loss(spec, fwd["logits"], anchors_b)
            loss = loss_reg + args.cls_loss_weight * cls_loss

            # InfoNCE auxiliary (optional) — pooled over anchored pred & per-variant targets.
            infonce_metrics: dict[str, float] = {}
            if use_infonce:
                pred_pool = _pool(pred, mask_b)                                # (B, D)
                tgt_pool_b = tgt_pooled_all[full_idx].to(device)               # (B, V, D)
                infonce, infonce_metrics = _infonce_loss(
                    pred_pool, tgt_pool_b, args.infonce_tau,
                )
                loss = loss + args.w_infonce * infonce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                row = {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "mse": comps["mse_active"],
                    "cos": comps["cos_loss"],
                    "cls": float(cls_loss.detach().item()),
                    "accs": accs,
                    "lr": float(opt.param_groups[0]["lr"]),
                }
                if infonce_metrics:
                    row["infonce"] = infonce_metrics["infonce_loss"]
                    row["infonce_acc"] = infonce_metrics["infonce_acc"]
                train_log.append(row)
                acc_str = " ".join(
                    f"{name[:1]}={v:.2f}" for name, v in accs.items()
                ) or "-"
                postfix = {
                    "loss": f"{float(loss.detach().item()):.3f}",
                    "cos": f"{comps['cos_loss']:.3f}",
                    "cls": f"{float(cls_loss.detach().item()):.3f}",
                    "acc": acc_str,
                    "lr": f"{opt.param_groups[0]['lr']:.1e}",
                }
                if infonce_metrics:
                    postfix["nce"] = f"{infonce_metrics['infonce_loss']:.3f}"
                    postfix["r@1"] = f"{infonce_metrics['infonce_acc']:.2f}"
                pbar.set_postfix(**postfix)

            if (
                args.eval_every > 0
                and step > 0
                and step % args.eval_every == 0
                and step < args.steps
            ):
                model.eval()
                pred_eval, cls_acc = _run_pred(
                    model, spec, tokens_all, pooled_all, flat_labels,
                    eval_idx, device, args.anchor_mode,
                )
                result = _eval_per_variant(
                    pred_eval, cache["te_paths"], eval_idx, cache["active_lengths"]
                )
                intermediate.append(
                    {
                        "step": step,
                        "stratified_vs_mean": result["vs_mean"]["stratified"],
                        "stratified_best_over_v": result["best_over_v"]["stratified"],
                        "stratified_mean_over_v": result["mean_over_v"]["stratified"],
                        "cls_acc": cls_acc,
                    }
                )
                _log_eval(result, cls_acc, f"eval@{step}")
                model.train()

            step += 1
            pbar.update(1)
    pbar.close()
    train_time = time.time() - t0
    logger.info(f"train time: {train_time:.1f}s")

    # Final eval
    model.eval()
    pred_eval, cls_acc = _run_pred(
        model, spec, tokens_all, pooled_all, flat_labels,
        eval_idx, device, args.anchor_mode,
    )
    result = _eval_per_variant(
        pred_eval, cache["te_paths"], eval_idx, cache["active_lengths"]
    )
    _log_eval(result, cls_acc, "final")

    payload = {
        "encoder": args.encoder,
        "probe": f"cross_attn_resampler_{args.n_layers}layer_anchored",
        "anchor_mode": args.anchor_mode,
        "anchor_spec": spec.to_metadata(),
        "d_enc": d_enc,
        "d_pool": d_pool,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_slots": K,
        "cache_slot_count": S,
        "n_steps": args.steps,
        "lr": args.lr,
        "warmup_frac": args.warmup_frac,
        "batch_size": args.batch_size,
        "num_variants": V,
        "variant_sampling": "per_sample_uniform",
        "cos_loss_weight": args.cos_loss_weight,
        "cls_loss_weight": args.cls_loss_weight,
        "w_infonce": args.w_infonce if use_infonce else 0.0,
        "infonce_tau": args.infonce_tau,
        "train_time_sec": train_time,
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "n_params_M": n_params / 1e6,
        "train_log": train_log,
        "intermediate_evals": intermediate,
        "final_eval": {
            "stratified_vs_mean": result["vs_mean"]["stratified"],
            "stratified_best_over_v": result["best_over_v"]["stratified"],
            "stratified_mean_over_v": result["mean_over_v"]["stratified"],
            "per_slot_vs_mean": result["vs_mean"]["per_slot"],
            "cls_acc": cls_acc,
        },
    }
    json_path = out_dir / f"{args.encoder}_resampler_{args.n_layers}layer_anchored.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  → {json_path}")

    ckpt_path = out_dir / f"{args.encoder}_resampler_{args.n_layers}layer_anchored.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(ckpt_path),
    )
    logger.info(f"  → {ckpt_path}")

    if args.save_predictions:
        pred_path = out_dir / f"{args.encoder}_eval_predictions.safetensors"
        save_file(
            {
                "pred": pred_eval.to(torch.bfloat16).contiguous(),
                "eval_idx": torch.tensor(eval_idx, dtype=torch.long),
            },
            str(pred_path),
        )
        logger.info(f"  → {pred_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
