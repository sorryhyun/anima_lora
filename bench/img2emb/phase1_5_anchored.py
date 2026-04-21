#!/usr/bin/env python
"""Phase 1.5 — Tag-anchored Perceiver Resampler.

Uses inversionv2's phase2 class prototypes + phase3 artist prototypes to
hard-anchor the three "prototype-addressable" slots (rating / count-meta /
artist) from a classifier over the pooled encoder feature. The resampler then
only has to fit the content-heavy mid/tail slots.

Why: phase-1 4-layer resampler at step 2500 matched the phase-0 1-layer probe
on mid/tail (cos 0.501 / 0.630 vs 0.501 / 0.631) — depth didn't help. inversionv2
showed rating / 1girl / solo / @artist sit in k@95 ≤ 5 per slot with within-class
cos > 0.9, so classifier + frozen-prototype lookup is the right shape for those
slots. Auxiliary CE on the classifier heads doubles as a diagnostic: if tag
accuracy from SigLIP2 pooled is high the encoder is fine; if it collapses,
we need a tag-aware encoder (WD-Tagger / EVA-anime) instead.

Usage:
    python bench/img2emb/phase1_5_anchored.py
    python bench/img2emb/phase1_5_anchored.py --steps 5000 --eval_every 1000
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
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.img2emb.phase0_probes import (  # noqa: E402
    _resampler_loss,
    active_slice,
    load_cache,
)
from bench.img2emb.phase1_resampler import PerceiverResampler  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- class lists
# Mirror inversionv2/tag_slot_analysis.py so prototype keys line up.
RATING_CLASSES = ["explicit", "sensitive", "general", "absurdres"]
COUNT_CLASSES = [
    "1girl", "1boy", "2girls", "2boys", "3girls", "1other",
    "solo", "multiple_girls", "multiple_boys",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", default=str(BENCH_DIR / "results" / "phase0"))
    p.add_argument("--out_dir", default=str(BENCH_DIR / "results" / "phase1_5"))
    p.add_argument(
        "--tag_slot_dir",
        default=str(REPO_ROOT / "bench" / "inversionv2" / "results" / "tag_slot"),
        help="inversionv2 output with phase1_positions.json + phase2/3 prototypes.",
    )
    p.add_argument("--image_dir", default=None)
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
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--cos_loss_weight", type=float, default=1.0)
    p.add_argument("--zero_pad_weight", type=float, default=0.01)
    p.add_argument(
        "--cls_loss_weight",
        type=float,
        default=0.1,
        help="Weight on auxiliary CE over rating/count/artist classifier heads.",
    )
    p.add_argument(
        "--anchor_mode",
        choices=["replace", "residual"],
        default="replace",
        help="replace = hard-write prototype mix into anchor slot; "
             "residual = add to resampler output at anchor slot.",
    )
    p.add_argument("--seed", type=int, default=20260421)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_predictions", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- anchor labels


def _load_prototypes(
    tag_slot_dir: Path,
    class_names: list[str],
    file_name: str,
    key_prefix: str = "",
) -> tuple[torch.Tensor, list[str]]:
    """Stack class prototypes in class_names order; append a zero row for 'unknown'.

    Returns (``proto_table: (len+1, D)``, ``names_with_unknown``).
    Missing classes fill with zeros too (warns).
    """
    sd = load_file(str(tag_slot_dir / file_name))
    D = next(iter(sd.values())).shape[0]
    rows = []
    for name in class_names:
        key = f"{key_prefix}{name}"
        if key in sd:
            rows.append(sd[key].float())
        else:
            logger.warning(f"  missing prototype '{key}' in {file_name}; using zero vec")
            rows.append(torch.zeros(D))
    rows.append(torch.zeros(D))                                         # unknown
    return torch.stack(rows, dim=0), class_names + ["<unknown>"]


def _load_artist_prototypes(tag_slot_dir: Path) -> tuple[torch.Tensor, list[str]]:
    """All 55 artist prototypes in sorted order; append 'unknown' zero row."""
    sd = load_file(str(tag_slot_dir / "phase3_artist_prototypes.safetensors"))
    names = sorted(sd.keys())
    D = next(iter(sd.values())).shape[0]
    rows = [sd[n].float() for n in names]
    rows.append(torch.zeros(D))
    return torch.stack(rows, dim=0), names + ["<unknown>"]


def build_anchor_labels(
    positions_path: Path,
    stems: list[str],
    artist_names: list[str],
) -> dict[str, dict]:
    """Per-stem (rating/count/artist) class index + slot position.

    Uses v0 only — prefix slots are shuffle-fixed across v0..v7 (phase4
    confirmed cos > 0.99 for rating/count/@artist). Sentinel -1 for missing.
    When a caption has multiple count-meta tags (e.g. 1girl + solo), picks the
    one with the smallest slot index (canonical booru ordering).
    """
    positions = json.loads(Path(positions_path).read_text())
    occ = positions["per_class_occurrences"]

    artist_to_idx = {n: i for i, n in enumerate(artist_names[:-1])}  # minus <unknown>
    unknown_artist_idx = len(artist_names) - 1

    labels = {
        s: {
            "rating_class": -1, "rating_slot": -1,
            "count_class": -1, "count_slot": -1,
            "artist_class": -1, "artist_slot": -1,
        }
        for s in stems
    }

    # Rating — class name -> class idx. "rating=<name>" keys in phase1.
    for cls_idx, name in enumerate(RATING_CLASSES):
        key = f"rating={name}"
        for entry in occ.get(key, []):
            stem, vi, first, *_ = entry
            if vi != 0 or stem not in labels:
                continue
            labels[stem]["rating_class"] = cls_idx
            labels[stem]["rating_slot"] = int(first)

    # Count-meta — pick the earliest-slot class per image.
    for cls_idx, name in enumerate(COUNT_CLASSES):
        for entry in occ.get(name, []):
            stem, vi, first, *_ = entry
            if vi != 0 or stem not in labels:
                continue
            first = int(first)
            cur_slot = labels[stem]["count_slot"]
            if cur_slot == -1 or first < cur_slot:
                labels[stem]["count_class"] = cls_idx
                labels[stem]["count_slot"] = first

    # Artist — phase1 stores artist name as the 6th field of each entry.
    for entry in occ.get("@artist", []):
        if len(entry) < 6:
            continue
        stem, vi, first, _last, _n, artist = entry
        if vi != 0 or stem not in labels:
            continue
        labels[stem]["artist_class"] = int(artist_to_idx.get(artist, unknown_artist_idx))
        labels[stem]["artist_slot"] = int(first)

    # Coverage stats for the log
    n = len(stems)
    n_r = sum(1 for s in stems if labels[s]["rating_class"] >= 0)
    n_c = sum(1 for s in stems if labels[s]["count_class"] >= 0)
    n_a = sum(1 for s in stems if labels[s]["artist_class"] >= 0)
    n_a_known = sum(
        1
        for s in stems
        if 0 <= labels[s]["artist_class"] < unknown_artist_idx
    )
    logger.info(
        f"anchor coverage: rating {n_r}/{n}  count {n_c}/{n}  "
        f"artist {n_a}/{n} (known {n_a_known}, unknown {n_a - n_a_known})"
    )
    return labels


def labels_to_tensors(labels: dict[str, dict], stems: list[str]) -> dict[str, torch.Tensor]:
    """Pack the per-stem dict into aligned int tensors for fast batch gather."""
    def col(key: str) -> torch.Tensor:
        return torch.tensor([labels[s][key] for s in stems], dtype=torch.long)
    return {
        "rating_class": col("rating_class"),
        "rating_slot": col("rating_slot"),
        "count_class": col("count_class"),
        "count_slot": col("count_slot"),
        "artist_class": col("artist_class"),
        "artist_slot": col("artist_slot"),
    }


# --------------------------------------------------------------------------- model


class AnchoredResampler(nn.Module):
    """Resampler + 3 classifier heads + frozen prototype tables.

    Anchor slots (per-image) get their prediction replaced / augmented by a
    softmax-weighted mixture of the corresponding prototype table.
    """

    def __init__(
        self,
        d_enc: int,
        d_pool: int,
        rating_protos: torch.Tensor,
        count_protos: torch.Tensor,
        artist_protos: torch.Tensor,
        d_model: int = 1024,
        n_heads: int = 8,
        n_slots: int = 512,
        n_layers: int = 4,
        d_out: int = 1024,
    ):
        super().__init__()
        self.backbone = PerceiverResampler(
            d_enc=d_enc,
            d_model=d_model,
            n_heads=n_heads,
            n_slots=n_slots,
            n_layers=n_layers,
            d_out=d_out,
        )
        # Pooled feature → classifier stem.
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(d_pool),
            nn.Linear(d_pool, d_model),
            nn.GELU(),
        )
        self.rating_head = nn.Linear(d_model, rating_protos.shape[0])
        self.count_head = nn.Linear(d_model, count_protos.shape[0])
        self.artist_head = nn.Linear(d_model, artist_protos.shape[0])

        # Frozen prototype tables. Register as buffers so they move with .to()
        # but never receive gradients.
        self.register_buffer("rating_protos", rating_protos.float())
        self.register_buffer("count_protos", count_protos.float())
        self.register_buffer("artist_protos", artist_protos.float())

    def classify(self, pooled: torch.Tensor):
        p = self.pool_proj(pooled.float())
        return (
            self.rating_head(p),
            self.count_head(p),
            self.artist_head(p),
        )

    def prototype_mix(self, logits: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Softmax(logits) @ table — differentiable prototype lookup. Returns (B, D_out)."""
        return torch.softmax(logits, dim=-1) @ table

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor):
        out = self.backbone(tokens)                                     # (B, S, D_out)
        r_logits, c_logits, a_logits = self.classify(pooled)
        r_emb = self.prototype_mix(r_logits, self.rating_protos)        # (B, D_out)
        c_emb = self.prototype_mix(c_logits, self.count_protos)
        a_emb = self.prototype_mix(a_logits, self.artist_protos)
        return {
            "pred": out,
            "logits": (r_logits, c_logits, a_logits),
            "anchor_emb": (r_emb, c_emb, a_emb),
        }


def inject_anchors(
    pred: torch.Tensor,            # (B, S, D) — mutated in-place
    anchor_emb: torch.Tensor,      # (B, D)
    slots: torch.Tensor,           # (B,) long; -1 for missing
    mode: str = "replace",
) -> None:
    """Write/add anchor_emb[b] into pred[b, slots[b], :] for valid b."""
    valid = slots >= 0
    if not valid.any():
        return
    b_idx = torch.arange(pred.shape[0], device=pred.device)[valid]
    s_idx = slots[valid].to(device=pred.device)
    if mode == "replace":
        pred[b_idx, s_idx] = anchor_emb[valid]
    elif mode == "residual":
        pred[b_idx, s_idx] = pred[b_idx, s_idx] + anchor_emb[valid]
    else:
        raise ValueError(f"unknown anchor_mode '{mode}'")


# --------------------------------------------------------------------------- dataset


class AnchoredTrainDataset(Dataset):
    """Same as phase1's dataset but yields the per-stem anchor labels too."""

    def __init__(
        self,
        train_indices: list[int],
        te_paths: list[str],
        active_lengths: list[int],
        anchor_labels: dict[str, torch.Tensor],
        num_variants: int,
    ):
        self.train_indices = train_indices
        self.te_paths = te_paths
        self.active_lengths = active_lengths
        self.labels = anchor_labels
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
        anchors = {k: int(v[full_idx].item()) for k, v in self.labels.items()}
        return full_idx, target.to(torch.bfloat16), L, anchors


# --------------------------------------------------------------------------- schedule


def _warmup_cosine(step: int, total: int, warmup: int, eta_min_frac: float = 0.05):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return eta_min_frac + (1 - eta_min_frac) * 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- loss


def aux_cls_loss(
    logits: torch.Tensor,          # (B, N_classes)
    targets: torch.Tensor,          # (B,) long; -1 = ignore
) -> tuple[torch.Tensor, float]:
    """Cross-entropy ignoring -1 targets. Returns (loss, accuracy) over non-ignored rows."""
    mask = targets >= 0
    if not mask.any():
        zero = logits.sum() * 0.0
        return zero, float("nan")
    lg = logits[mask]
    tg = targets[mask].to(device=logits.device)
    loss = F.cross_entropy(lg, tg)
    with torch.no_grad():
        acc = (lg.argmax(dim=-1) == tg).float().mean().item()
    return loss, float(acc)


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
    tokens_all: torch.Tensor,
    pooled_all: torch.Tensor,
    anchor_labels: dict[str, torch.Tensor],
    eval_idx: list[int],
    device: torch.device,
    anchor_mode: str,
    batch: int = 8,
) -> tuple[torch.Tensor, dict]:
    out_pred = []
    r_correct = c_correct = a_correct = 0
    r_total = c_total = a_total = 0
    for i in range(0, len(eval_idx), batch):
        ids = eval_idx[i : i + batch]
        ids_t = torch.tensor(ids, dtype=torch.long)
        tok_b = tokens_all[ids_t].to(device=device, dtype=torch.bfloat16)
        pool_b = pooled_all[ids_t].to(device=device, dtype=torch.float32)
        fwd = model(tok_b, pool_b)
        pred = fwd["pred"].detach().float()
        r_emb, c_emb, a_emb = (e.detach().float() for e in fwd["anchor_emb"])

        r_slot = anchor_labels["rating_slot"][ids_t].to(device)
        c_slot = anchor_labels["count_slot"][ids_t].to(device)
        a_slot = anchor_labels["artist_slot"][ids_t].to(device)
        inject_anchors(pred, r_emb, r_slot, mode=anchor_mode)
        inject_anchors(pred, c_emb, c_slot, mode=anchor_mode)
        inject_anchors(pred, a_emb, a_slot, mode=anchor_mode)

        out_pred.append(pred.cpu())

        # Classifier accuracy
        r_logits, c_logits, a_logits = fwd["logits"]
        for logits, key in (
            (r_logits, "rating_class"),
            (c_logits, "count_class"),
            (a_logits, "artist_class"),
        ):
            tg = anchor_labels[key][ids_t]
            m = tg >= 0
            if not m.any():
                continue
            pred_cls = logits.argmax(dim=-1).cpu()
            n_c = int((pred_cls[m] == tg[m]).sum().item())
            n_t = int(m.sum().item())
            if key == "rating_class":
                r_correct += n_c; r_total += n_t
            elif key == "count_class":
                c_correct += n_c; c_total += n_t
            else:
                a_correct += n_c; a_total += n_t

    pred_eval = torch.cat(out_pred, dim=0)
    cls_acc = {
        "rating": (r_correct / r_total) if r_total else float("nan"),
        "count": (c_correct / c_total) if c_total else float("nan"),
        "artist": (a_correct / a_total) if a_total else float("nan"),
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
    pad_abs_total = 0.0
    pad_count = 0

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

        if L < S:
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


def _log_eval(result: dict, cls_acc: dict, tag: str):
    for metric in ("vs_mean", "best_over_v", "mean_over_v"):
        strat = result[metric]["stratified"]
        parts = []
        for band in ("prefix_0_8", "mid_8_64", "tail_64_256", "content_all"):
            g = strat[band]["cos_median"]
            v = g["mean"] if g else float("nan")
            parts.append(f"{band}={v:.3f}")
        logger.info(f"  [{tag}/{metric}] " + "  ".join(parts))
    logger.info(
        f"  [{tag}/cls_acc] rating={cls_acc['rating']:.3f}  "
        f"count={cls_acc['count']:.3f}  artist={cls_acc['artist']:.3f}"
    )
    logger.info(
        f"  [{tag}/pad] mean |pred| in inactive = "
        f"{result['pad_residual_mean_abs']:.2e}"
    )


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
        f"out={out_dir}  tag_slot={tag_slot_dir}  anchor_mode={args.anchor_mode}"
    )
    cache = load_cache(cache_dir, image_dir, args.encoder, args.num_workers)

    train_idx = cache["split"]["train_idx"]
    eval_idx = cache["split"]["eval_idx"]
    tokens_all = cache["tokens"]
    pooled_all = cache["pooled"]
    V = int(cache["num_variants"])
    S = int(cache["targets_mean"].shape[1])
    D_y = int(cache["targets_mean"].shape[2])
    d_enc = int(tokens_all.shape[-1])
    d_pool = int(pooled_all.shape[-1])
    logger.info(
        f"N_train={len(train_idx)}  N_eval={len(eval_idx)}  V={V}  "
        f"S={S}  D_y={D_y}  d_enc={d_enc}  d_pool={d_pool}"
    )
    del cache["targets_mean"]

    # Prototype tables
    rating_protos, rating_names = _load_prototypes(
        tag_slot_dir, RATING_CLASSES, "phase2_class_prototypes.safetensors",
        key_prefix="rating=",
    )
    count_protos, count_names = _load_prototypes(
        tag_slot_dir, COUNT_CLASSES, "phase2_class_prototypes.safetensors",
    )
    artist_protos, artist_names = _load_artist_prototypes(tag_slot_dir)
    logger.info(
        f"prototypes: rating={rating_protos.shape}  count={count_protos.shape}  "
        f"artist={artist_protos.shape}  (last row = <unknown>)"
    )

    # Anchor labels per stem (aligned to cache stems order)
    anchor_labels = build_anchor_labels(
        tag_slot_dir / "phase1_positions.json",
        cache["stems"],
        artist_names,
    )
    label_tensors = labels_to_tensors(anchor_labels, cache["stems"])

    model = AnchoredResampler(
        d_enc=d_enc,
        d_pool=d_pool,
        rating_protos=rating_protos,
        count_protos=count_protos,
        artist_protos=artist_protos,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_slots=S,
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
        train_indices=train_idx,
        te_paths=cache["te_paths"],
        active_lengths=cache["active_lengths"],
        anchor_labels=label_tensors,
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
        for full_idx, tgt_b, L_b, anchors_b in loader:
            if step >= args.steps:
                done = True
                break
            lr_scale = _warmup_cosine(step, args.steps, warmup)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_scale

            tok_b = tokens_all[full_idx].to(device=device, dtype=torch.bfloat16)
            pool_b = pooled_all[full_idx].to(device=device, dtype=torch.float32)
            tgt_b = tgt_b.to(device=device, dtype=torch.float32)
            mask_b = (torch.arange(S).unsqueeze(0) < L_b.unsqueeze(1)).to(device)

            fwd = model(tok_b, pool_b)
            pred = fwd["pred"]
            r_emb, c_emb, a_emb = fwd["anchor_emb"]

            r_slot = anchors_b["rating_slot"].to(device)
            c_slot = anchors_b["count_slot"].to(device)
            a_slot = anchors_b["artist_slot"].to(device)
            inject_anchors(pred, r_emb, r_slot, mode=args.anchor_mode)
            inject_anchors(pred, c_emb, c_slot, mode=args.anchor_mode)
            inject_anchors(pred, a_emb, a_slot, mode=args.anchor_mode)

            loss_reg, comps = _resampler_loss(
                pred, tgt_b, mask_b,
                cos_w=args.cos_loss_weight, zero_w=args.zero_pad_weight,
            )
            r_logits, c_logits, a_logits = fwd["logits"]
            r_loss, r_acc = aux_cls_loss(r_logits, anchors_b["rating_class"])
            c_loss, c_acc = aux_cls_loss(c_logits, anchors_b["count_class"])
            a_loss, a_acc = aux_cls_loss(a_logits, anchors_b["artist_class"])
            cls_loss = (r_loss + c_loss + a_loss) / 3.0

            loss = loss_reg + args.cls_loss_weight * cls_loss

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
                        "cls": float(cls_loss.detach().item()),
                        "acc_r": r_acc, "acc_c": c_acc, "acc_a": a_acc,
                        "lr": float(opt.param_groups[0]["lr"]),
                    }
                )
                pbar.set_postfix(
                    loss=f"{float(loss.detach().item()):.3f}",
                    cos=f"{comps['cos_loss']:.3f}",
                    cls=f"{float(cls_loss.detach().item()):.3f}",
                    r=f"{r_acc:.2f}" if not math.isnan(r_acc) else "-",
                    a=f"{a_acc:.2f}" if not math.isnan(a_acc) else "-",
                    lr=f"{opt.param_groups[0]['lr']:.1e}",
                )

            if (
                args.eval_every > 0
                and step > 0
                and step % args.eval_every == 0
                and step < args.steps
            ):
                model.eval()
                pred_eval, cls_acc = _run_pred(
                    model, tokens_all, pooled_all, label_tensors,
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
                        "pad_residual_mean_abs": result["pad_residual_mean_abs"],
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
        model, tokens_all, pooled_all, label_tensors,
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
        "d_enc": d_enc,
        "d_pool": d_pool,
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
        "cls_loss_weight": args.cls_loss_weight,
        "rating_classes": rating_names,
        "count_classes": count_names,
        "n_artist_classes": len(artist_names),
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
