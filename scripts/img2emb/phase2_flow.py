#!/usr/bin/env python
"""Phase 2a — flow-matching supervision through the frozen DiT.

Keeps the phase 1.5 architecture (AnchoredResampler + inject_anchors at
variant-specific slot positions) and swaps the loss. Resampler output feeds
the DiT's cross-attention KV; AdaLN's pooled-text path stays on the T5 manifold
via ``pooled_text_override``. See ``phase2_proposal.md``.

Phase 2a = 2k-step warm-start ablation from the phase 1.5 checkpoint. The full
20k run is a flag flip (``--steps 20000``) after the ablation passes.

Usage:
    python scripts/img2emb/phase2_flow.py \\
        --dit models/diffusion_models/anima-preview3-base.safetensors \\
        --warm_start bench/img2emb/results/phase1_5/siglip2_resampler_4layer_anchored.safetensors \\
        --steps 2000
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# results/ still lives under bench/img2emb/ from before the move; keep writing there.
BENCH_DIR = REPO_ROOT / "bench" / "img2emb"
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.data import load_cache  # noqa: E402
from scripts.img2emb.phase1_5_anchored import (  # noqa: E402
    COUNT_CLASSES,
    RATING_CLASSES,
    AnchoredResampler,
    _load_artist_prototypes,
    _load_prototypes,
    build_anchor_labels,
    inject_anchors,
    labels_to_tensors,
)
from library.anima import weights as anima_utils  # noqa: E402
from library.io.cache import get_latent_resolution  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- args


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    # paths
    p.add_argument("--dit", default="models/diffusion_models/anima-preview3-base.safetensors")
    p.add_argument("--cache_dir", default=str(BENCH_DIR / "results" / "phase0"))
    p.add_argument("--out_dir", default=str(BENCH_DIR / "results" / "phase2a"))
    p.add_argument(
        "--tag_slot_dir",
        default=str(REPO_ROOT / "bench" / "inversionv2" / "results" / "tag_slot"),
    )
    p.add_argument("--image_dir", default=None, help="Overrides active_lengths.json[image_dir].")
    p.add_argument("--warm_start", default=None, help="Phase 1.5 checkpoint (safetensors).")
    p.add_argument("--phase1_5_ckpt", default=None,
                   help="Separate phase 1.5 checkpoint used for the held-out FM baseline. "
                        "Defaults to --warm_start when unset.")

    # model
    p.add_argument("--encoder", default="siglip2")
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--attn_mode", default="flash")

    # DiT runtime
    p.add_argument("--blocks_to_swap", type=int, default=6,
                   help=">0 enables block-swap; <0 enables gradient checkpointing; 0 = neither.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile each DiT block's _forward (dynamic=False). "
                        "Bucket shapes vary, so expect one recompile per unique (H, W) bucket.")
    p.add_argument("--dynamo_backend", default="inductor")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # optim
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--batch_size", type=int, default=1)
    # LRs tuned for warm-start from phase 1.5: an order of magnitude lower
    # than phase 1.5's own training LR because the supervision signal
    # (FM-through-DiT) is fundamentally different and warm weights start
    # well-conditioned for the regression task. A 3-step smoke at lr=1e-4
    # diverged from fm=0.20 to fm=1.46 after 2 updates. Raise with care.
    p.add_argument("--lr_resampler", type=float, default=2e-5)
    p.add_argument("--lr_cls", type=float, default=6e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="Lowered from phase 1.5's 1.0 since FM gradient scale "
                        "through DiT can spike early.")

    # loss weights (calibrated at step 0; see --calibrate_only)
    p.add_argument("--w_pad", type=float, default=0.001,
                   help="Zero-pad regularizer on pred[L:]. Lower than phase 1.5's 0.01 "
                        "because FM-loss-scale differs; monitor mean(|pred[L:]|).")
    p.add_argument("--w_cls", type=float, default=0.1)
    p.add_argument("--w_retention", type=float, default=0.0,
                   help="Optional MSE(pred, variant_mean_crossattn_emb) safety term.")
    p.add_argument("--weighting_scheme", default="none",
                   choices=["none", "sigma_sqrt", "cosmap"])

    # sigma sampling
    p.add_argument("--sigma_sampling", default="sigmoid", choices=["sigmoid", "uniform"])
    p.add_argument("--sigmoid_scale", type=float, default=1.0)

    # pooled-text policy
    p.add_argument("--pooled_text", default="t5",
                   choices=["t5", "resampler", "skip"],
                   help="t5 = override with cached T5 variant-mean pooled (default); "
                        "resampler = use resampler ctx.max(dim=1); skip = no pooled text.")

    # logging / eval
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--eval_max_samples", type=int, default=128,
                   help="Cap on held-out samples for FM eval (full split is 397).")

    p.add_argument("--seed", type=int, default=20260421)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--calibrate_only", action="store_true",
                   help="Print step-0 loss calibration and exit.")
    return p.parse_args()


# --------------------------------------------------------------------------- dataset


def _locate_npz(image_dir: Path, stem: str) -> Path:
    """Return the single ``{stem}_<H>x<W>_anima.npz`` file (one bucket per stem)."""
    matches = list(image_dir.glob(f"{stem}_*_anima.npz"))
    if not matches:
        raise FileNotFoundError(f"No npz for stem '{stem}' in {image_dir}")
    return matches[0]


class Phase2Dataset(Dataset):
    """Yields per-sample tensors needed for FM-through-DiT training.

    Returns a dict (variable-shape ``vae_latent`` → batching handled by
    :class:`BucketBatchSampler` so every batch has a uniform (H, W)).
    """

    def __init__(
        self,
        stems: list[str],
        indices: list[int],
        te_paths: list[str],
        npz_paths: list[Path],
        active_lengths: list[int],
        anchor_labels: dict[str, torch.Tensor],
        num_variants: int,
    ):
        self.stems = stems
        self.indices = indices
        self.te_paths = te_paths
        self.npz_paths = npz_paths
        self.active_lengths = active_lengths
        self.labels = anchor_labels
        self.V = num_variants

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        full_idx = self.indices[pos]

        # VAE latent — already normalized (mean-subtracted, std-divided) at cache time.
        with np.load(self.npz_paths[full_idx]) as data:
            latent_key = next(k for k in data.keys() if k.startswith("latents_"))
            latent = torch.from_numpy(data[latent_key].copy())  # (C=16, H, W) float32

        # Cached T5 crossattn + T5 pooled.
        sd = load_file(self.te_paths[full_idx])
        variant = int(torch.randint(0, self.V, (1,)).item()) if self.V > 1 else 0
        t5_crossattn = sd[f"crossattn_emb_v{variant}"].float()  # (512, 1024)

        # Variant-mean T5 pooled (more stable than v-specific; matches DiT's
        # `crossattn_emb.max(dim=1).values` feed into pooled_text_proj).
        variant_keys = sorted(k for k in sd if k.startswith("crossattn_emb_v"))
        pooled_stack = torch.stack([sd[k].float().amax(dim=0) for k in variant_keys], dim=0)
        t5_pooled = pooled_stack.mean(dim=0)  # (1024,)

        L = int(self.active_lengths[full_idx])
        anchors = {k: int(lbl[full_idx].item()) for k, lbl in self.labels.items()}

        return {
            "full_idx": full_idx,
            "latent": latent,  # (16, H, W) f32 — normalized
            "t5_crossattn": t5_crossattn,  # (512, 1024) f32
            "t5_pooled": t5_pooled,  # (1024,) f32
            "L": L,
            "anchors": anchors,
            "variant": variant,
        }


class BucketBatchSampler(BatchSampler):
    """Sample batches where every item shares the same latent (H, W) bucket.

    Within-bucket sampling is uniform; buckets are visited in proportion to size.
    """

    def __init__(
        self,
        bucket_to_positions: dict[tuple[int, int], list[int]],
        batch_size: int,
        num_batches: int,
        seed: int,
    ):
        self.bucket_to_positions = bucket_to_positions
        self.buckets = list(bucket_to_positions.keys())
        self.weights = np.array(
            [len(bucket_to_positions[b]) for b in self.buckets], dtype=np.float64
        )
        self.weights = self.weights / self.weights.sum()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed)
        # Filter out buckets with fewer than batch_size members (rare; last bucket has 6).
        drop = [b for b in self.buckets if len(bucket_to_positions[b]) < batch_size]
        for b in drop:
            logger.warning(f"skipping bucket {b}: only {len(bucket_to_positions[b])} < bs={batch_size}")
            self.buckets.remove(b)
        if drop:
            self.weights = np.array(
                [len(bucket_to_positions[b]) for b in self.buckets], dtype=np.float64
            )
            self.weights = self.weights / self.weights.sum()

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            b_idx = self.rng.choice(len(self.buckets), p=self.weights)
            bucket = self.buckets[b_idx]
            positions = self.bucket_to_positions[bucket]
            chosen = self.rng.choice(len(positions), size=self.batch_size, replace=False)
            yield [positions[i] for i in chosen]


def make_bucket_map(
    indices: list[int], npz_paths: list[Path]
) -> dict[tuple[int, int], list[int]]:
    """Group dataset-position -> bucket (H_lat, W_lat). ``indices`` maps position -> full_idx."""
    buckets: dict[tuple[int, int], list[int]] = {}
    for pos, full_idx in enumerate(indices):
        res = get_latent_resolution(str(npz_paths[full_idx]))  # e.g. "144x112"
        h_lat, w_lat = map(int, res.split("x"))
        buckets.setdefault((h_lat, w_lat), []).append(pos)
    return buckets


def collate(batch: list[dict]) -> dict:
    """Stack batch items (all share latent shape by the BucketBatchSampler contract)."""
    out = {
        "full_idx": torch.tensor([b["full_idx"] for b in batch], dtype=torch.long),
        "latent": torch.stack([b["latent"] for b in batch], dim=0),
        "t5_crossattn": torch.stack([b["t5_crossattn"] for b in batch], dim=0),
        "t5_pooled": torch.stack([b["t5_pooled"] for b in batch], dim=0),
        "L": torch.tensor([b["L"] for b in batch], dtype=torch.long),
        "variant": torch.tensor([b["variant"] for b in batch], dtype=torch.long),
    }
    keys = list(batch[0]["anchors"].keys())
    out["anchors"] = {
        k: torch.tensor([b["anchors"][k] for b in batch], dtype=torch.long) for k in keys
    }
    return out


# --------------------------------------------------------------------------- flow-matching


def sample_sigmas(n: int, scheme: str, scale: float, device: torch.device) -> torch.Tensor:
    if scheme == "sigmoid":
        return torch.sigmoid(scale * torch.randn(n, device=device))
    return torch.rand(n, device=device)


def loss_weighting(scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    if scheme == "sigma_sqrt":
        return (sigmas.float() ** -2.0).clamp(max=1e4)
    if scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        return (2 / (math.pi * bot)).float()
    return torch.ones_like(sigmas, dtype=torch.float32)


def dit_fm_loss(
    anima,
    latent: torch.Tensor,             # (B, C, H, W) f32 — normalized
    ctx: torch.Tensor,                # (B, 512, 1024) bf16
    pooled_text: torch.Tensor | None, # (B, 1024) bf16 or None
    skip_pooled_text: bool,
    sigma_scheme: str,
    sigma_scale: float,
    weighting_scheme: str,
) -> tuple[torch.Tensor, dict]:
    """One flow-matching forward + MSE(v_pred, noise - latent). Returns scalar + diag."""
    B = latent.shape[0]
    device = latent.device
    sigmas = sample_sigmas(B, sigma_scheme, sigma_scale, device)  # (B,)
    noise = torch.randn_like(latent)

    sigmas_view = sigmas.view(-1, 1, 1, 1)
    x_sigma = (1.0 - sigmas_view) * latent + sigmas_view * noise  # (B, C, H, W)
    x_sigma_5d = x_sigma.to(torch.bfloat16).unsqueeze(2)          # (B, C, 1, H, W)

    h_lat, w_lat = latent.shape[-2], latent.shape[-1]
    padding_mask = torch.zeros(B, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    timesteps = sigmas.to(torch.bfloat16)

    kwargs = {"padding_mask": padding_mask}
    if skip_pooled_text:
        kwargs["skip_pooled_text_proj"] = True
    elif pooled_text is not None:
        kwargs["pooled_text_override"] = pooled_text

    v_pred = anima(x_sigma_5d, timesteps, ctx, **kwargs).squeeze(2)  # (B, C, H, W) bf16
    v_target = (noise - latent).float()

    weights = loss_weighting(weighting_scheme, sigmas).view(-1, 1, 1, 1)
    per_sample = F.mse_loss(v_pred.float(), v_target, reduction="none").mean(dim=(1, 2, 3))
    loss = (weights.squeeze() * per_sample).mean()

    with torch.no_grad():
        diag = {
            "fm_mse_unweighted": per_sample.mean().item(),
            "sigma_mean": sigmas.mean().item(),
            "sigma_min": sigmas.min().item(),
            "sigma_max": sigmas.max().item(),
        }
    return loss, diag


# --------------------------------------------------------------------------- warm-start

def load_warm_start(model: AnchoredResampler, ckpt_path: Path) -> None:
    sd = load_file(str(ckpt_path))
    # Strip any prefix mismatches; expect direct AnchoredResampler keys.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # ``*_protos`` buffers may be overwritten by the checkpoint's values
    # (should match exactly, but we prefer the on-disk prototype tables).
    if missing:
        logger.warning(f"warm-start missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"warm-start unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")


# --------------------------------------------------------------------------- training


def build_ctx(
    model: AnchoredResampler,
    tokens: torch.Tensor,      # (B, T, D_enc) bf16
    pooled: torch.Tensor,      # (B, D_enc) f32
    anchors: dict[str, torch.Tensor],
    device: torch.device,
    anchor_mode: str = "replace",
) -> tuple[torch.Tensor, tuple, tuple]:
    """Run resampler + classifier heads + anchor injection. Returns (ctx_bf16, logits, classes)."""
    fwd = model(tokens, pooled)
    pred = fwd["pred"]                                              # (B, 512, 1024) f32/bf16
    r_logits, c_logits, a_logits = fwd["logits"]
    r_emb, c_emb, a_emb = fwd["anchor_emb"]

    r_slot = anchors["rating_slot"].to(device)
    c_slot = anchors["count_slot"].to(device)
    a_slot = anchors["artist_slot"].to(device)
    inject_anchors(pred, r_emb, r_slot, mode=anchor_mode)
    inject_anchors(pred, c_emb, c_slot, mode=anchor_mode)
    inject_anchors(pred, a_emb, a_slot, mode=anchor_mode)

    return (
        pred.to(torch.bfloat16),
        (r_logits, c_logits, a_logits),
        (anchors["rating_class"], anchors["count_class"], anchors["artist_class"]),
    )


def aux_cls_loss(logits_trio, class_trio) -> tuple[torch.Tensor, dict]:
    total = 0.0
    accs = {}
    for name, lg, tg in zip(("rating", "count", "artist"), logits_trio, class_trio):
        m = tg >= 0
        if not m.any():
            continue
        tg_d = tg[m].to(lg.device)
        lg_d = lg[m]
        total = total + F.cross_entropy(lg_d, tg_d)
        with torch.no_grad():
            accs[name] = (lg_d.argmax(dim=-1) == tg_d).float().mean().item()
    if isinstance(total, float):
        return torch.tensor(0.0, device=logits_trio[0].device), accs
    return total, accs


def pad_tail_loss(pred: torch.Tensor, L: torch.Tensor) -> tuple[torch.Tensor, float]:
    """MSE(pred[b, L[b]:], 0). Also returns mean(|pred[L:]|) as a diagnostic."""
    B, S, D = pred.shape
    arange = torch.arange(S, device=pred.device).unsqueeze(0)       # (1, S)
    tail_mask = (arange >= L.to(pred.device).unsqueeze(1)).unsqueeze(-1)  # (B, S, 1)
    masked = pred * tail_mask
    denom = tail_mask.sum().clamp(min=1).float() * D
    mse = (masked.float() ** 2).sum() / denom
    with torch.no_grad():
        mean_abs = (masked.float().abs().sum() / denom).item()
    return mse, mean_abs


def _warmup_cosine(step: int, total: int, warmup: int, eta_min_frac: float = 0.05):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return eta_min_frac + (1 - eta_min_frac) * 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- eval


@torch.no_grad()
def eval_fm(
    anima,
    model: AnchoredResampler,
    phase15_model: AnchoredResampler | None,
    loader: DataLoader,
    args,
    device: torch.device,
    n_batches: int,
) -> dict:
    """Compute held-out FM loss under three contexts:
      - phase2  : current resampler output (+ anchors)
      - t5_real : variant-mean T5 crossattn_emb (real-caption baseline / ceiling)
      - phase15 : phase 1.5 checkpoint's prediction (+ anchors)
    Returns means over n_batches.
    """
    tots = {"phase2": 0.0, "t5_real": 0.0, "phase15": 0.0}
    counts = {k: 0 for k in tots}
    cls_totals = {"rating": [0, 0], "count": [0, 0], "artist": [0, 0]}

    model.eval()
    if phase15_model is not None:
        phase15_model.eval()

    it = iter(loader)
    for bi in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        latent = batch["latent"].to(device, dtype=torch.float32)
        t5_pooled = batch["t5_pooled"].to(device, dtype=torch.bfloat16)
        t5_crossattn = batch["t5_crossattn"].to(device, dtype=torch.bfloat16)
        anchors = {k: v.to(device) for k, v in batch["anchors"].items()}

        # Siglip2 features for the batch -- pre-loaded on args.tokens_all / pooled_all (closures).
        tok_b = args.tokens_all[batch["full_idx"]].to(device, dtype=torch.bfloat16)
        pool_b = args.pooled_all[batch["full_idx"]].to(device, dtype=torch.float32)

        # phase 2 context
        ctx2, logits_trio, class_trio = build_ctx(model, tok_b, pool_b, anchors, device)

        skip_pooled = args.pooled_text == "skip"
        pooled_for_dit = None
        if args.pooled_text == "t5":
            pooled_for_dit = t5_pooled
        elif args.pooled_text == "resampler":
            pooled_for_dit = ctx2.float().amax(dim=1).to(torch.bfloat16)

        loss2, _ = dit_fm_loss(
            anima, latent, ctx2, pooled_for_dit, skip_pooled,
            args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
        )
        tots["phase2"] += loss2.item()
        counts["phase2"] += 1

        # classifier accuracy
        for name, lg, tg in zip(("rating", "count", "artist"), logits_trio, class_trio):
            m = tg >= 0
            if not m.any():
                continue
            tg_d = tg[m].to(lg.device)
            n_correct = (lg[m].argmax(dim=-1) == tg_d).sum().item()
            cls_totals[name][0] += int(n_correct)
            cls_totals[name][1] += int(m.sum().item())

        # t5_real baseline -- pass cached crossattn straight through.
        pooled_t5 = t5_pooled if args.pooled_text != "skip" else None
        loss_t5, _ = dit_fm_loss(
            anima, latent, t5_crossattn, pooled_t5, skip_pooled,
            args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
        )
        tots["t5_real"] += loss_t5.item()
        counts["t5_real"] += 1

        # phase 1.5 baseline (optional)
        if phase15_model is not None:
            ctx15, _, _ = build_ctx(phase15_model, tok_b, pool_b, anchors, device)
            pooled_p15 = None
            if args.pooled_text == "t5":
                pooled_p15 = t5_pooled
            elif args.pooled_text == "resampler":
                pooled_p15 = ctx15.float().amax(dim=1).to(torch.bfloat16)
            loss15, _ = dit_fm_loss(
                anima, latent, ctx15, pooled_p15, skip_pooled,
                args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
            )
            tots["phase15"] += loss15.item()
            counts["phase15"] += 1

    out = {k: (tots[k] / max(1, counts[k])) for k in tots}
    out["cls_acc"] = {
        name: (n_c / n_t) if n_t else float("nan")
        for name, (n_c, n_t) in cls_totals.items()
    }
    model.train()
    if phase15_model is not None:
        phase15_model.eval()
    return out


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # -----------------------------------------------------------------  cache (siglip2 / split / te_paths)
    cache_dir = Path(args.cache_dir)
    image_dir = args.image_dir
    if image_dir is None:
        act = json.loads((cache_dir / "active_lengths.json").read_text())
        image_dir = act.get("image_dir", "post_image_dataset")
    image_dir_path = Path(image_dir)
    if not image_dir_path.is_absolute():
        image_dir_path = REPO_ROOT / image_dir_path

    logger.info(f"cache={cache_dir}  image_dir={image_dir_path}  out={out_dir}  pooled_text={args.pooled_text}")
    cache = load_cache(cache_dir, str(image_dir_path), args.encoder, num_workers=4)
    del cache["targets_mean"]  # not needed; we use per-image safetensors on the fly

    tokens_all = cache["tokens"]       # (N, T_enc, D_enc) bf16 on cpu
    pooled_all = cache["pooled"]       # (N, D_pool) f32 on cpu
    stems = cache["stems"]
    split = cache["split"]
    active_lengths = cache["active_lengths"]
    V = int(cache["num_variants"])
    te_paths = cache["te_paths"]
    d_enc = int(tokens_all.shape[-1])
    d_pool = int(pooled_all.shape[-1])
    logger.info(f"N_train={len(split['train_idx'])}  N_eval={len(split['eval_idx'])}  V={V}")

    # Stash for closures inside eval_fm (keeps signature tidy).
    args.tokens_all = tokens_all
    args.pooled_all = pooled_all

    # VAE NPZ paths — one per stem.
    npz_paths = [_locate_npz(image_dir_path, s) for s in stems]

    # -----------------------------------------------------------------  prototypes / labels
    tag_slot_dir = Path(args.tag_slot_dir)
    rating_protos, _ = _load_prototypes(
        tag_slot_dir, RATING_CLASSES, "phase2_class_prototypes.safetensors",
        key_prefix="rating=",
    )
    count_protos, _ = _load_prototypes(
        tag_slot_dir, COUNT_CLASSES, "phase2_class_prototypes.safetensors",
    )
    artist_protos, artist_names = _load_artist_prototypes(tag_slot_dir)
    anchor_labels = build_anchor_labels(tag_slot_dir / "phase1_positions.json", stems, artist_names)
    label_tensors = labels_to_tensors(anchor_labels, stems)

    # -----------------------------------------------------------------  model (trainable resampler)
    S = 512  # slot count (matches cached crossattn_emb shape)
    model = AnchoredResampler(
        d_enc=d_enc, d_pool=d_pool,
        rating_protos=rating_protos, count_protos=count_protos, artist_protos=artist_protos,
        d_model=args.d_model, n_heads=args.n_heads, n_slots=S, n_layers=args.n_layers,
    ).to(device)

    if args.warm_start:
        logger.info(f"warm-starting from {args.warm_start}")
        load_warm_start(model, Path(args.warm_start))
    n_params_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"trainable params: {n_params_M:.3f} M")

    # Phase 1.5 reference model (frozen, used only for held-out baseline).
    phase15_model = None
    p15_path = args.phase1_5_ckpt or args.warm_start
    if p15_path:
        phase15_model = AnchoredResampler(
            d_enc=d_enc, d_pool=d_pool,
            rating_protos=rating_protos, count_protos=count_protos, artist_protos=artist_protos,
            d_model=args.d_model, n_heads=args.n_heads, n_slots=S, n_layers=args.n_layers,
        ).to(device)
        load_warm_start(phase15_model, Path(p15_path))
        phase15_model.requires_grad_(False)
        phase15_model.eval()

    # -----------------------------------------------------------------  frozen DiT
    logger.info("loading DiT (frozen)")
    is_swapping = args.blocks_to_swap > 0
    grad_ckpt = args.blocks_to_swap < 0
    anima = anima_utils.load_anima_model(
        device="cpu" if is_swapping else device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device="cpu" if is_swapping else device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)
    anima.split_attn = False  # within-bucket batching -> uniform shape

    if is_swapping:
        logger.info(f"block swap: {args.blocks_to_swap} blocks to CPU")
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
    else:
        anima.to(device)
        if grad_ckpt:
            logger.info("gradient checkpointing ON")
            anima.enable_gradient_checkpointing()
            for block in anima.blocks:  # type: ignore[union-attr]
                block.train()

    if args.compile:
        logger.info(f"compiling DiT blocks (backend={args.dynamo_backend})")
        anima.compile_blocks(args.dynamo_backend)

    # -----------------------------------------------------------------  dataloaders
    train_idx = split["train_idx"]
    eval_idx = split["eval_idx"][: args.eval_max_samples]

    def build_loader(idx_list, batch_size, num_batches, seed):
        dset = Phase2Dataset(
            stems, idx_list, te_paths, npz_paths, active_lengths,
            label_tensors, V,
        )
        bucket_map = make_bucket_map(idx_list, npz_paths)
        sampler = BucketBatchSampler(bucket_map, batch_size, num_batches, seed)
        return DataLoader(dset, batch_sampler=sampler, num_workers=2, collate_fn=collate,
                          persistent_workers=True)

    train_loader = build_loader(train_idx, args.batch_size, args.steps, args.seed)
    # Eval loader: enough batches to cover eval_max_samples (approx).
    eval_n_batches = max(1, len(eval_idx) // args.eval_batch_size)
    eval_loader = build_loader(eval_idx, args.eval_batch_size, eval_n_batches, args.seed + 1)

    # -----------------------------------------------------------------  optimizer
    cls_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(("rating_head", "count_head", "artist_head", "pool_proj")):
            cls_params.append(p)
        else:
            other_params.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr_resampler, "weight_decay": args.weight_decay},
            {"params": cls_params, "lr": args.lr_cls, "weight_decay": args.weight_decay},
        ]
    )
    warmup = max(1, int(args.warmup_frac * args.steps))

    # -----------------------------------------------------------------  train
    model.train()
    log_rows = []
    t0 = time.time()

    step_iter = iter(train_loader)
    for step in tqdm(range(args.steps), desc="train", dynamic_ncols=True):
        batch = next(step_iter)
        latent = batch["latent"].to(device, dtype=torch.float32)
        t5_pooled = batch["t5_pooled"].to(device, dtype=torch.bfloat16)
        t5_crossattn = batch["t5_crossattn"].to(device, dtype=torch.bfloat16)  # noqa: F841
        L_b = batch["L"]
        anchors = {k: v.to(device) for k, v in batch["anchors"].items()}
        tok_b = tokens_all[batch["full_idx"]].to(device, dtype=torch.bfloat16)
        pool_b = pooled_all[batch["full_idx"]].to(device, dtype=torch.float32)

        # Resampler + anchor injection -> ctx
        ctx, logits_trio, class_trio = build_ctx(model, tok_b, pool_b, anchors, device)

        # Pooled-text policy
        skip_pooled = args.pooled_text == "skip"
        if args.pooled_text == "t5":
            pooled_for_dit = t5_pooled
        elif args.pooled_text == "resampler":
            pooled_for_dit = ctx.float().amax(dim=1).to(torch.bfloat16)
        else:  # "skip"
            pooled_for_dit = None

        fm_loss, fm_diag = dit_fm_loss(
            anima, latent, ctx, pooled_for_dit, skip_pooled,
            args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
        )
        ce_loss, accs = aux_cls_loss(logits_trio, class_trio)
        pad_loss, pad_mean_abs = pad_tail_loss(ctx, L_b)

        loss = fm_loss + args.w_cls * ce_loss + args.w_pad * pad_loss
        if args.w_retention > 0.0:
            # Variant-mean of cached crossattn_emb — can re-use t5_crossattn for v0 or compute mean.
            # Keep this optional; costs another per-step tensor if enabled.
            retention = F.mse_loss(ctx.float(), t5_crossattn.float())
            loss = loss + args.w_retention * retention

        # step-0 calibration: print magnitudes before the first backward.
        if step == 0:
            logger.info(
                f"[calibration] fm={fm_loss.item():.4f}  "
                f"ce={ce_loss.item():.4f}  "
                f"pad={pad_loss.item():.6f} (mean|tail|={pad_mean_abs:.2e})  "
                f"total={loss.item():.4f}"
            )
            if args.calibrate_only:
                return

        optim.zero_grad(set_to_none=True)
        loss.backward()
        # cosine w/ warmup, applied as a scalar multiplier on both groups.
        lr_scale = _warmup_cosine(step, args.steps, warmup)
        for g, base in zip(optim.param_groups, (args.lr_resampler, args.lr_cls)):
            g["lr"] = base * lr_scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optim.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "fm": float(fm_loss.item()),
                "fm_unweighted": float(fm_diag["fm_mse_unweighted"]),
                "ce": float(ce_loss.item()),
                "pad": float(pad_loss.item()),
                "pad_mean_abs": float(pad_mean_abs),
                "sigma_mean": float(fm_diag["sigma_mean"]),
                "lr_scale": float(lr_scale),
                "accs": accs,
            }
            log_rows.append(row)
            logger.info(
                f"step {step:>5}  loss={row['loss']:.4f}  fm={row['fm']:.4f}  "
                f"ce={row['ce']:.4f}  pad={row['pad']:.5f}  "
                f"|tail|={row['pad_mean_abs']:.2e}  "
                f"σ̄={row['sigma_mean']:.3f}  "
                f"accs={accs}"
            )

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            logger.info(f"--- eval @ step {step} ---")
            # Block-swap offloader relies on backward hooks to reset block
            # placement; eval has none, so toggle to forward-only (and back).
            if hasattr(anima, "switch_block_swap_for_inference"):
                anima.switch_block_swap_for_inference()
            try:
                ev = eval_fm(anima, model, phase15_model, eval_loader, args, device,
                             n_batches=eval_n_batches)
            finally:
                if hasattr(anima, "switch_block_swap_for_training"):
                    anima.switch_block_swap_for_training()
            logger.info(
                f"[eval] fm_phase2={ev['phase2']:.4f}  "
                f"fm_t5_real={ev['t5_real']:.4f}  "
                f"fm_phase15={ev['phase15']:.4f}  "
                f"cls_acc={ev['cls_acc']}"
            )
            log_rows.append({"step": step, "eval": ev})

    elapsed = time.time() - t0
    logger.info(f"done in {elapsed:.1f}s")

    # -----------------------------------------------------------------  save
    out_tag = f"{args.encoder}_phase2a_warm"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(out_dir / f"{out_tag}.safetensors"),
    )
    (out_dir / f"{out_tag}.json").write_text(json.dumps(
        {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                     if k not in ("tokens_all", "pooled_all")},
            "n_params_M": n_params_M,
            "train_time_sec": elapsed,
            "log": log_rows,
        },
        indent=2,
    ))
    logger.info(f"saved -> {out_dir / out_tag}.(safetensors|json)")


if __name__ == "__main__":
    main()
