#!/usr/bin/env python
"""Phase-1 training — 4-layer Perceiver Resampler over cached SigLIP2 features.

Reuses phase-0 scaffolding (cache loading, variant-sampling training dataset,
zero-pad loss). Adds:

- 4-layer resampler (cross-attn queries←patches, self-attn over queries, FFN).
- Per-variant eval: cos_med vs variant-mean, best-over-V, mean-over-V — each
  stratified by slot band. Variants are *labels*, not noise; training samples
  one per image per step, eval reports all three views.
- Padded-tail leakage diagnostic (mean |pred| in inactive slots; target < 1e-3).
- Saves final weights + optional held-out predictions for downstream DiT tests.

This is an exploratory trainer superseded by ``phase1_5_anchored`` on the
production pipeline (``train_img2emb pretrain``). Kept under ``bench/`` for
reproducing the pre-anchoring numbers; the resampler architecture itself lives
at ``scripts.img2emb.resampler``.

Usage:
    python bench/img2emb/phase1_resampler.py
    python bench/img2emb/phase1_resampler.py --encoder siglip2 --steps 10000
    python bench/img2emb/phase1_resampler.py --steps 200 --eval_every 0  # smoke
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
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.data import (  # noqa: E402
    _ResamplerTrainDataset,
    _resampler_loss,
    active_slice,
    load_cache,
)
from scripts.img2emb.resampler import PerceiverResampler  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", default=str(BENCH_DIR / "results" / "phase0"))
    p.add_argument("--out_dir", default=str(BENCH_DIR / "results" / "phase1"))
    p.add_argument(
        "--image_dir",
        default=None,
        help="Override; defaults to image_dir from active_lengths.json.",
    )
    p.add_argument("--encoder", default="siglip2")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument(
        "--eval_every",
        type=int,
        default=2500,
        help="Run held-out eval every N steps (0 = only final).",
    )
    p.add_argument("--cos_loss_weight", type=float, default=1.0)
    p.add_argument("--zero_pad_weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20260421)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--save_predictions",
        action="store_true",
        help="Also save held-out predictions to {encoder}_eval_predictions.safetensors",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- schedule


def _warmup_cosine(step: int, total: int, warmup: int, eta_min_frac: float = 0.05):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return eta_min_frac + (1 - eta_min_frac) * 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- eval


def _slot_agg_cos(cos_mat: torch.Tensor, mask: torch.Tensor) -> list:
    """Aggregate per-slot cosine stats over rows active in that slot.

    cos_mat: (N, S) float. mask: (N, S) bool (True = slot active in that image).
    """
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
        vals = [
            s[key]
            for s in slots
            if s.get("active") and key in s and not math.isnan(s[key])
        ]
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
def _run_pred(model, tokens_all, eval_idx, device, batch: int = 8) -> torch.Tensor:
    out = []
    for i in range(0, len(eval_idx), batch):
        ids = eval_idx[i : i + batch]
        tok_b = tokens_all[ids].to(device=device, dtype=torch.bfloat16)
        out.append(model(tok_b).detach().float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def _eval_per_variant(
    pred_eval: torch.Tensor,            # (N, S, D) fp32 cpu
    te_paths: list[str],
    eval_idx: list[int],
    active_lengths: list[int],
) -> dict:
    """Stream per-image variants; report three cosine views + pad leakage.

    Keeps peak memory tiny: at most one image's (V, S, D) in RAM at a time.
    """
    N, S, D = pred_eval.shape
    mask = active_slice([active_lengths[i] for i in eval_idx])  # (N, S) bool cpu

    pred_n = F.normalize(pred_eval.float(), dim=-1, eps=1e-8)

    cos_vs_mean = torch.zeros(N, S)
    cos_best = torch.zeros(N, S)
    cos_mean_v = torch.zeros(N, S)
    pad_abs_total = 0.0
    pad_count = 0

    for n, full_idx in enumerate(tqdm(eval_idx, desc="eval")):
        sd = load_file(te_paths[full_idx])
        variant_keys = sorted(
            k for k in sd.keys() if k.startswith("crossattn_emb_v")
        )
        if variant_keys:
            variants = torch.stack([sd[k].float() for k in variant_keys], dim=0)
        else:
            variants = sd["crossattn_emb"].float().unsqueeze(0)
        L = int(active_lengths[full_idx])
        if L < variants.shape[1]:
            variants[:, L:] = 0

        t_mean = variants.mean(dim=0)                                # (S, D)
        tm_n = F.normalize(t_mean, dim=-1, eps=1e-8)
        var_n = F.normalize(variants, dim=-1, eps=1e-8)              # (V, S, D)
        p_n = pred_n[n]                                              # (S, D)

        cos_vs_mean[n] = (p_n * tm_n).sum(dim=-1)
        cos_all = (p_n.unsqueeze(0) * var_n).sum(dim=-1)             # (V, S)
        cos_best[n] = cos_all.max(dim=0).values
        cos_mean_v[n] = cos_all.mean(dim=0)

        if L < S:
            # mean |pred| in inactive region for this image
            pad_abs_total += float(pred_eval[n, L:, :].abs().mean().item()) * (S - L)
            pad_count += S - L

    pad_abs_mean = pad_abs_total / max(1, pad_count)

    vs_mean_ps = _slot_agg_cos(cos_vs_mean, mask)
    best_ps = _slot_agg_cos(cos_best, mask)
    mean_v_ps = _slot_agg_cos(cos_mean_v, mask)

    return {
        "vs_mean": {"per_slot": vs_mean_ps, "stratified": _strata(vs_mean_ps)},
        "best_over_v": {"per_slot": best_ps, "stratified": _strata(best_ps)},
        "mean_over_v": {"per_slot": mean_v_ps, "stratified": _strata(mean_v_ps)},
        "pad_residual_mean_abs": pad_abs_mean,
    }


def _log_eval(result: dict, tag: str):
    for metric in ("vs_mean", "best_over_v", "mean_over_v"):
        strat = result[metric]["stratified"]
        parts = []
        for band in ("prefix_0_8", "mid_8_64", "tail_64_256", "content_all"):
            g = strat[band]["cos_median"]
            v = g["mean"] if g else float("nan")
            parts.append(f"{band}={v:.3f}")
        logger.info(f"  [{tag}/{metric}] " + "  ".join(parts))
    logger.info(
        f"  [{tag}/pad] mean |pred| in inactive = "
        f"{result['pad_residual_mean_abs']:.2e}"
    )


# --------------------------------------------------------------------------- summary


def _write_summary(out_dir: Path, args, payload: dict):
    s = payload["final_eval"]
    lines = []
    a = lines.append
    a("# Phase 1 — 4-layer Perceiver Resampler")
    a("")
    a(
        f"Encoder: `{args.encoder}`  Layers: {args.n_layers}  "
        f"d_model: {args.d_model}  heads: {args.n_heads}  "
        f"params: {payload['n_params_M']:.1f}M"
    )
    a(
        f"Steps: {args.steps}  LR: {args.lr} (warmup {args.warmup_frac} + cosine)  "
        f"Batch: {args.batch_size}  Train time: {payload['train_time_sec']:.0f}s"
    )
    a("")
    a("## Held-out cos_med (mean over slots, per band)")
    a("")
    a("| band | vs variant-mean | best over V | mean over V |")
    a("|---|---|---|---|")
    for band in ("prefix_0_8", "mid_8_64", "tail_64_256", "content_all"):

        def fmt(key):
            g = s[key][band]["cos_median"]
            return f"{g['mean']:.3f}" if g else "-"

        a(
            f"| {band} | {fmt('stratified_vs_mean')} | "
            f"{fmt('stratified_best_over_v')} | {fmt('stratified_mean_over_v')} |"
        )
    a("")
    a(
        f"Pad-tail leakage (mean |pred| in inactive slots): "
        f"{s['pad_residual_mean_abs']:.2e}"
    )
    a("")

    phase0 = BENCH_DIR / "results" / "phase0" / f"{args.encoder}_resampler_probe.json"
    if phase0.exists():
        a("## vs phase-0 1-layer probe (same encoder, same split)")
        a("")
        p0 = json.loads(phase0.read_text())
        a("| band | cos_med (1-layer) | cos_med (this run, vs mean) | Δ |")
        a("|---|---|---|---|")
        for band in ("prefix_0_8", "mid_8_64", "tail_64_256", "content_all"):
            g0 = p0["stratified"][band]["cos_median"]
            g1 = s["stratified_vs_mean"][band]["cos_median"]
            v0 = g0["mean"] if g0 else None
            v1 = g1["mean"] if g1 else None
            if v0 is None or v1 is None:
                a(f"| {band} | - | - | - |")
            else:
                a(f"| {band} | {v0:.3f} | {v1:.3f} | {v1 - v0:+.3f} |")
        a("")

    a("## Gates")
    a("")
    a("- **Proceed to DiT-inference test** if:")
    a("  - content_all `cos_med vs mean` clearly > 0.623 (phase-0 dataset-mean baseline)")
    a("  - AND prefix_0_8 `cos_med vs mean` > 0.93")
    a("  - AND pad leakage < 1e-3.")
    a("- **Stop-loss** (swap encoder, e.g. WD-Tagger / EVA-anime) if content_all")
    a("  `cos_med vs mean` ≤ 0.62 after 10k steps.")
    a("- Gap between `best over V` and `mean over V` should be small (<0.05). A")
    a("  large gap means the model collapsed onto one variant ordering rather")
    a("  than the shared direction.")
    a("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    logger.info(f"  → {out_dir / 'summary.md'}")


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        f"out={out_dir}"
    )
    cache = load_cache(cache_dir, image_dir, args.encoder, args.num_workers)

    train_idx = cache["split"]["train_idx"]
    eval_idx = cache["split"]["eval_idx"]
    tokens_all = cache["tokens"]
    V = int(cache["num_variants"])
    S, D_y = cache["target_shape"]
    d_enc = int(tokens_all.shape[-1])
    logger.info(
        f"N_train={len(train_idx)}  N_eval={len(eval_idx)}  V={V}  "
        f"S={S}  D_y={D_y}  d_enc={d_enc}"
    )

    model = PerceiverResampler(
        d_enc=d_enc,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_slots=S,
        n_layers=args.n_layers,
        d_out=D_y,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"resampler: {args.n_layers} layers × d={args.d_model} × heads={args.n_heads}"
        f"  params={n_params / 1e6:.1f}M"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = max(1, int(args.warmup_frac * args.steps))

    train_ds = _ResamplerTrainDataset(
        train_indices=train_idx,
        te_paths=cache["te_paths"],
        active_lengths=cache["active_lengths"],
        num_variants=V,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    train_log = []
    intermediate = []
    t0 = time.time()
    pbar = tqdm(total=args.steps, desc=f"train/{args.encoder}", leave=True)
    step = 0
    done = False
    while not done:
        for full_idx, tgt_b, L_b in loader:
            if step >= args.steps:
                done = True
                break
            lr_scale = _warmup_cosine(step, args.steps, warmup)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_scale

            tok_b = tokens_all[full_idx].to(device=device, dtype=torch.bfloat16)
            tgt_b = tgt_b.to(device=device, dtype=torch.float32)
            mask_b = (torch.arange(S).unsqueeze(0) < L_b.unsqueeze(1)).to(device)

            pred = model(tok_b)
            loss, comps = _resampler_loss(
                pred,
                tgt_b,
                mask_b,
                cos_w=args.cos_loss_weight,
                zero_w=args.zero_pad_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                train_log.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().item()),
                        "mse": comps["mse_active"],
                        "cos": comps["cos_loss"],
                        "pad": comps["pad_loss"],
                        "lr": float(opt.param_groups[0]["lr"]),
                    }
                )
                pbar.set_postfix(
                    loss=f"{float(loss.detach().item()):.3f}",
                    mse=f"{comps['mse_active']:.4f}",
                    cos=f"{comps['cos_loss']:.3f}",
                    lr=f"{opt.param_groups[0]['lr']:.1e}",
                )

            if (
                args.eval_every > 0
                and step > 0
                and step % args.eval_every == 0
                and step < args.steps  # final eval runs below
            ):
                model.eval()
                pred_eval = _run_pred(model, tokens_all, eval_idx, device)
                result = _eval_per_variant(
                    pred_eval,
                    cache["te_paths"],
                    eval_idx,
                    cache["active_lengths"],
                )
                intermediate.append(
                    {
                        "step": step,
                        "stratified_vs_mean": result["vs_mean"]["stratified"],
                        "stratified_best_over_v": result["best_over_v"]["stratified"],
                        "stratified_mean_over_v": result["mean_over_v"]["stratified"],
                        "pad_residual_mean_abs": result["pad_residual_mean_abs"],
                    }
                )
                _log_eval(result, f"eval@{step}")
                model.train()

            step += 1
            pbar.update(1)
    pbar.close()
    train_time = time.time() - t0
    logger.info(f"train time: {train_time:.1f}s")

    # Final eval
    model.eval()
    pred_eval = _run_pred(model, tokens_all, eval_idx, device)
    result = _eval_per_variant(
        pred_eval, cache["te_paths"], eval_idx, cache["active_lengths"]
    )
    _log_eval(result, "final")

    payload = {
        "encoder": args.encoder,
        "probe": f"cross_attn_resampler_{args.n_layers}layer",
        "d_enc": d_enc,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_slots": S,
        "n_steps": args.steps,
        "lr": args.lr,
        "warmup_frac": args.warmup_frac,
        "batch_size": args.batch_size,
        "num_variants": V,
        "variant_sampling": "per_sample_uniform",
        "cos_loss_weight": args.cos_loss_weight,
        "zero_pad_weight": args.zero_pad_weight,
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
            "pad_residual_mean_abs": result["pad_residual_mean_abs"],
            "per_slot_vs_mean": result["vs_mean"]["per_slot"],
        },
    }
    json_path = out_dir / f"{args.encoder}_resampler_{args.n_layers}layer.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  → {json_path}")

    ckpt_path = out_dir / f"{args.encoder}_resampler_{args.n_layers}layer.safetensors"
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

    _write_summary(out_dir, args, payload)
    logger.info("Done.")


if __name__ == "__main__":
    main()
