#!/usr/bin/env python
"""Finetune — flow-matching supervision through the frozen DiT.

Keeps the pretrain architecture (AnchoredResampler + inject_spec_anchors at
variant-specific slot positions) and swaps the loss. Resampler output feeds
both the DiT's cross-attention KV and AdaLN's pooled-text path: pooled is
``ctx.amax(dim=1)`` from the resampler output, never from T5.

Warm-starts from the pretrain checkpoint (``--warm_start``).

Usage:
    python scripts/img2emb/finetune.py \\
        --dit models/diffusion_models/anima-preview3-base.safetensors \\
        --warm_start output/img2embs/pretrain/tipsv2_resampler_4layer_anchored.safetensors \\
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
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.anchors import (  # noqa: E402
    AnchorSpec,
    AnchoredResampler,
    aux_cls_loss,
    build_anchor_labels,
    collate_anchor_batch,
    gather_sample_labels,
    inject_spec_anchors,
    labels_to_flat_tensors,
    load_anchor_spec,
)
from scripts.img2emb.data import _infonce_loss, _pool, load_cache  # noqa: E402
from scripts.img2emb.preprocess import ENCODER_NAME  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.io.cache import get_latent_resolution  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_ANCHORS_YAML = Path(__file__).parent / "anchors.yaml"
DEFAULT_CACHE_DIR = REPO_ROOT / "output" / "img2embs" / "features"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "img2embs" / "finetune"
DEFAULT_TAG_SLOT_DIR = REPO_ROOT / "output" / "img2embs" / "anchors"


# --------------------------------------------------------------------------- args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # paths
    p.add_argument("--dit", default="models/diffusion_models/anima-preview3-base.safetensors")
    p.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--tag_slot_dir",
        default=str(DEFAULT_TAG_SLOT_DIR),
    )
    p.add_argument(
        "--anchors_yaml",
        default=str(DEFAULT_ANCHORS_YAML),
        help="YAML spec listing anchor groups + classes.",
    )
    p.add_argument("--image_dir", default=None, help="Overrides active_lengths.json[image_dir].")
    p.add_argument("--warm_start", default=None, help="Pretrain checkpoint (safetensors).")
    p.add_argument(
        "--pretrain_ckpt",
        default=None,
        help="Separate pretrain checkpoint used for the held-out FM baseline. "
             "Defaults to --warm_start when unset.",
    )

    # model
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument(
        "--n_slots",
        type=int,
        default=256,
        help="Resampler query count K (default 256). Output is zero-padded "
             "to 512 before feeding to DiT cross-attn (matches the cached T5 "
             "crossattn_emb shape).",
    )
    p.add_argument("--attn_mode", default="flash")

    # DiT runtime
    p.add_argument("--blocks_to_swap", type=int, default=26,
                   help=">0 enables block-swap; <0 enables gradient checkpointing; 0 = neither.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile each DiT block's _forward (dynamic=False). "
                        "Bucket shapes vary, so expect one recompile per unique (H, W) bucket.")
    p.add_argument("--dynamo_backend", default="inductor")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # optim
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--batch_size", type=int, default=1)
    # LRs tuned for warm-start from pretrain: an order of magnitude lower
    # than pretrain's own training LR because the supervision signal
    # (FM-through-DiT) is fundamentally different and warm weights start
    # well-conditioned for the regression task. A 3-step smoke at lr=1e-4
    # diverged from fm=0.20 to fm=1.46 after 2 updates. Raise with care.
    p.add_argument("--lr_resampler", type=float, default=2e-5)
    p.add_argument("--lr_cls", type=float, default=6e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="Lowered from pretrain's 1.0 since FM gradient scale "
                        "through DiT can spike early.")

    # loss weights (calibrated at step 0; see --calibrate_only)
    p.add_argument("--w_cls", type=float, default=0.1)
    p.add_argument("--w_retention", type=float, default=0.0,
                   help="Optional MSE(pred, variant_mean_crossattn_emb) safety term.")
    p.add_argument("--w_infonce", type=float, default=0,
                   help="Multi-positive InfoNCE over pooled per-variant "
                        "targets (SupCon-style; positives = all V variants "
                        "of the same image, negatives = variants of other "
                        "images in the batch). Set 0 to disable.")
    p.add_argument("--infonce_tau", type=float, default=0.07,
                   help="InfoNCE temperature (CLIP default).")
    p.add_argument("--weighting_scheme", default="none",
                   choices=["none", "sigma_sqrt", "cosmap"])

    # sigma sampling
    p.add_argument("--sigma_sampling", default="sigmoid", choices=["sigmoid", "uniform"])
    p.add_argument("--sigmoid_scale", type=float, default=1.0)

    # logging / eval
    p.add_argument("--eval_every", type=int, default=600)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--eval_max_samples", type=int, default=128,
                   help="Cap on held-out samples for FM eval (full split is 397).")

    p.add_argument("--seed", type=int, default=20260421)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--calibrate_only", action="store_true",
                   help="Print step-0 loss calibration and exit.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- dataset


def _locate_npz(image_dir: Path, stem: str) -> Path:
    """Return the single ``{stem}_<H>x<W>_anima.npz`` file (one bucket per stem)."""
    matches = list(image_dir.glob(f"{stem}_*_anima.npz"))
    if not matches:
        raise FileNotFoundError(f"No npz for stem '{stem}' in {image_dir}")
    return matches[0]


class FinetuneDataset(Dataset):
    """Yields per-sample tensors needed for FM-through-DiT training.

    Returns a dict (variable-shape ``vae_latent`` → batching handled by
    :class:`BucketBatchSampler` so every batch has a uniform (H, W)).
    """

    def __init__(
        self,
        spec: AnchorSpec,
        stems: list[str],
        indices: list[int],
        te_paths: list[str],
        npz_paths: list[Path],
        active_lengths: list[int],
        flat_labels: dict[str, torch.Tensor],
        num_variants: int,
    ):
        self.spec = spec
        self.stems = stems
        self.indices = indices
        self.te_paths = te_paths
        self.npz_paths = npz_paths
        self.active_lengths = active_lengths
        self.flat_labels = flat_labels
        self.V = num_variants

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        full_idx = self.indices[pos]

        with np.load(self.npz_paths[full_idx]) as data:
            latent_key = next(k for k in data.keys() if k.startswith("latents_"))
            latent = torch.from_numpy(data[latent_key].copy())  # (C=16, H, W) float32

        sd = load_file(self.te_paths[full_idx])
        variant = int(torch.randint(0, self.V, (1,)).item()) if self.V > 1 else 0
        t5_crossattn = sd[f"crossattn_emb_v{variant}"].float()  # (512, 1024)

        L = int(self.active_lengths[full_idx])
        anchors = gather_sample_labels(self.spec, self.flat_labels, full_idx)

        return {
            "full_idx": full_idx,
            "latent": latent,
            "t5_crossattn": t5_crossattn,
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
        res = get_latent_resolution(str(npz_paths[full_idx]))
        h_lat, w_lat = map(int, res.split("x"))
        buckets.setdefault((h_lat, w_lat), []).append(pos)
    return buckets


def _make_collate(spec: AnchorSpec):
    def collate(batch: list[dict]) -> dict:
        out = {
            "full_idx": torch.tensor([b["full_idx"] for b in batch], dtype=torch.long),
            "latent": torch.stack([b["latent"] for b in batch], dim=0),
            "t5_crossattn": torch.stack([b["t5_crossattn"] for b in batch], dim=0),
            "L": torch.tensor([b["L"] for b in batch], dtype=torch.long),
            "variant": torch.tensor([b["variant"] for b in batch], dtype=torch.long),
            "anchors": collate_anchor_batch(spec, [b["anchors"] for b in batch]),
        }
        return out
    return collate


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
    pooled_text: torch.Tensor,        # (B, 1024) bf16 — resampler ctx.amax(dim=1)
    sigma_scheme: str,
    sigma_scale: float,
    weighting_scheme: str,
) -> tuple[torch.Tensor, dict]:
    """One flow-matching forward + MSE(v_pred, noise - latent). Returns scalar + diag."""
    B = latent.shape[0]
    device = latent.device
    sigmas = sample_sigmas(B, sigma_scheme, sigma_scale, device)
    noise = torch.randn_like(latent)

    sigmas_view = sigmas.view(-1, 1, 1, 1)
    x_sigma = (1.0 - sigmas_view) * latent + sigmas_view * noise
    x_sigma_5d = x_sigma.to(torch.bfloat16).unsqueeze(2)

    h_lat, w_lat = latent.shape[-2], latent.shape[-1]
    padding_mask = torch.zeros(B, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    timesteps = sigmas.to(torch.bfloat16)

    kwargs = {"padding_mask": padding_mask, "pooled_text_override": pooled_text}

    v_pred = anima(x_sigma_5d, timesteps, ctx, **kwargs).squeeze(2)
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
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"warm-start missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"warm-start unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")


# --------------------------------------------------------------------------- training helpers


def build_ctx(
    model: AnchoredResampler,
    spec: AnchorSpec,
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    anchors: dict[str, torch.Tensor],
    device: torch.device,
    anchor_mode: str = "replace",
    pad_to: int = 512,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Run resampler + classifier heads + anchor injection, zero-pad to ``pad_to``.

    Returns ``(ctx_bf16, logits, batch_labels_on_device, pred_kcap)`` where
    ``pred_kcap`` is the (B, K, D) float tensor before the zero-pad (used by
    InfoNCE to pool over only the active K slots).
    """
    fwd = model(tokens, pooled)
    pred = fwd["pred"]

    dev_labels = {k: v.to(device) for k, v in anchors.items()}
    inject_spec_anchors(pred, spec, fwd["anchor_emb"], dev_labels, mode=anchor_mode)

    K_pred = pred.shape[1]
    if K_pred < pad_to:
        ctx = F.pad(pred, (0, 0, 0, pad_to - K_pred))
    else:
        ctx = pred
    return ctx.to(torch.bfloat16), fwd["logits"], dev_labels, pred


def anchor_keep_mask(
    spec: AnchorSpec,
    labels: dict[str, torch.Tensor],
    B: int,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """(B, K) bool, True where the slot was NOT overwritten by an anchor prototype.

    Used to exclude frozen anchor slots from ``pooled_for_dit``. Without this,
    the injected class prototypes (highest-magnitude, ~constant per classifier
    decision) dominate ``amax`` and AdaLN locks onto the anchor category
    regardless of the reference.
    """
    keep = torch.ones(B, K, dtype=torch.bool, device=device)
    for g in spec.groups:
        if g.mutex:
            slot = labels[f"{g.name}_slot"].to(device)
            valid = (slot >= 0) & (slot < K)
            if valid.any():
                b = torch.arange(B, device=device)[valid]
                keep[b, slot[valid]] = False
        else:
            slots = labels[f"{g.name}_slots"].to(device)
            lbls = labels[f"{g.name}_labels"].to(device) > 0.5
            valid = (slots >= 0) & (slots < K) & lbls
            if valid.any():
                b_rows, c_cols = torch.nonzero(valid, as_tuple=True)
                keep[b_rows, slots[b_rows, c_cols]] = False
    return keep


def masked_amax(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """amax over ``x`` (B, K, D) along K, restricted to ``keep`` (B, K) bool."""
    neg = torch.finfo(x.dtype).min
    return x.masked_fill(~keep.unsqueeze(-1), neg).amax(dim=1)


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
    pretrain_model: AnchoredResampler | None,
    spec: AnchorSpec,
    loader: DataLoader,
    args,
    device: torch.device,
    n_batches: int,
) -> dict:
    """Compute held-out FM loss under three contexts:
      - finetune : current resampler output (+ anchors)
      - t5_real  : variant-mean T5 crossattn_emb (real-caption baseline / ceiling)
      - pretrain : pretrain checkpoint's prediction (+ anchors)
    Returns means over n_batches.
    """
    tots = {"finetune": 0.0, "t5_real": 0.0, "pretrain": 0.0}
    counts = {k: 0 for k in tots}
    cls_totals: dict[str, list[int]] = {g.name: [0, 0] for g in spec.groups}

    model.eval()
    if pretrain_model is not None:
        pretrain_model.eval()

    it = iter(loader)
    for _ in tqdm(range(n_batches), desc="eval", dynamic_ncols=True, leave=False):
        try:
            batch = next(it)
        except StopIteration:
            break
        latent = batch["latent"].to(device, dtype=torch.float32)
        t5_crossattn = batch["t5_crossattn"].to(device, dtype=torch.bfloat16)
        anchors = batch["anchors"]

        tok_b = args.tokens_all[batch["full_idx"]].to(device, dtype=torch.bfloat16)
        pool_b = args.pooled_all[batch["full_idx"]].to(device, dtype=torch.float32)

        ctx2, logits, dev_labels, pred_k2 = build_ctx(model, spec, tok_b, pool_b, anchors, device)
        B_cur, K_cur = pred_k2.shape[:2]
        pool_keep2 = anchor_keep_mask(spec, dev_labels, B_cur, K_cur, device)
        pooled_for_dit = masked_amax(pred_k2, pool_keep2).to(torch.bfloat16)

        loss2, _ = dit_fm_loss(
            anima, latent, ctx2, pooled_for_dit,
            args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
        )
        tots["finetune"] += loss2.item()
        counts["finetune"] += 1

        for g in spec.groups:
            lg = logits[g.name]
            if g.mutex:
                tg = dev_labels[f"{g.name}_class"]
                m = tg >= 0
                if not m.any():
                    continue
                n_c = (lg[m].argmax(dim=-1) == tg[m]).sum().item()
                cls_totals[g.name][0] += int(n_c)
                cls_totals[g.name][1] += int(m.sum().item())
            else:
                tg = dev_labels[f"{g.name}_labels"]
                lg_c = lg[..., : g.n_classes]
                pred_pos = (torch.sigmoid(lg_c) > 0.5).float()
                cls_totals[g.name][0] += int((pred_pos == tg).sum().item())
                cls_totals[g.name][1] += int(tg.numel())

        pooled_t5 = t5_crossattn.amax(dim=1)
        loss_t5, _ = dit_fm_loss(
            anima, latent, t5_crossattn, pooled_t5,
            args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
        )
        tots["t5_real"] += loss_t5.item()
        counts["t5_real"] += 1

        if pretrain_model is not None:
            ctx_pre, _, dev_labels_pre, pred_k_pre = build_ctx(
                pretrain_model, spec, tok_b, pool_b, anchors, device,
            )
            B_pre, K_pre = pred_k_pre.shape[:2]
            pool_keep_pre = anchor_keep_mask(spec, dev_labels_pre, B_pre, K_pre, device)
            pooled_pre = masked_amax(pred_k_pre, pool_keep_pre).to(torch.bfloat16)
            loss_pre, _ = dit_fm_loss(
                anima, latent, ctx_pre, pooled_pre,
                args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
            )
            tots["pretrain"] += loss_pre.item()
            counts["pretrain"] += 1

    out = {k: (tots[k] / max(1, counts[k])) for k in tots}
    out["cls_acc"] = {
        name: (n_c / n_t) if n_t else float("nan")
        for name, (n_c, n_t) in cls_totals.items()
    }
    model.train()
    if pretrain_model is not None:
        pretrain_model.eval()
    return out


# --------------------------------------------------------------------------- stage entrypoint


def finetune(args: argparse.Namespace) -> None:
    """Run the finetune stage using ``args``. Importable from ``train.py``."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    cache_dir = Path(args.cache_dir)
    image_dir = args.image_dir
    if image_dir is None:
        act = json.loads((cache_dir / "active_lengths.json").read_text())
        image_dir = act.get("image_dir", "post_image_dataset")
    image_dir_path = Path(image_dir)
    if not image_dir_path.is_absolute():
        image_dir_path = REPO_ROOT / image_dir_path

    logger.info(f"cache={cache_dir}  image_dir={image_dir_path}  out={out_dir}")
    cache = load_cache(cache_dir, str(image_dir_path), ENCODER_NAME, num_workers=4)

    tokens_all = cache["tokens"]
    pooled_all = cache["pooled"]
    stems = cache["stems"]
    split = cache["split"]
    active_lengths = cache["active_lengths"]
    V = int(cache["num_variants"])
    te_paths = cache["te_paths"]
    d_enc = int(tokens_all.shape[-1])
    d_pool = int(pooled_all.shape[-1])
    logger.info(f"N_train={len(split['train_idx'])}  N_eval={len(split['eval_idx'])}  V={V}")

    tgt_pooled_all = cache.get("target_pooled")
    use_infonce = bool(args.w_infonce > 0.0 and tgt_pooled_all is not None)
    if args.w_infonce > 0.0 and tgt_pooled_all is None:
        logger.warning(
            "w_infonce > 0 but features/target_pooled.safetensors is missing — "
            "re-run preprocess.py. InfoNCE disabled for this run."
        )

    args.tokens_all = tokens_all
    args.pooled_all = pooled_all

    npz_paths = [_locate_npz(image_dir_path, s) for s in stems]

    tag_slot_dir = Path(args.tag_slot_dir)
    spec = load_anchor_spec(Path(args.anchors_yaml), tag_slot_dir)
    anchor_labels = build_anchor_labels(spec, tag_slot_dir / "phase1_positions.json", stems)
    flat_labels = labels_to_flat_tensors(spec, anchor_labels)

    S = 512  # DiT cross-attn slot count (matches cached crossattn_emb shape).
    K = int(args.n_slots)
    if K > S:
        raise ValueError(f"--n_slots={K} exceeds DiT slot count {S}")
    args.S = S
    args.K = K
    model = AnchoredResampler(
        spec=spec, d_enc=d_enc, d_pool=d_pool,
        d_model=args.d_model, n_heads=args.n_heads, n_slots=K, n_layers=args.n_layers,
    ).to(device)

    if args.warm_start:
        logger.info(f"warm-starting from {args.warm_start}")
        load_warm_start(model, Path(args.warm_start))
    n_params_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"trainable params: {n_params_M:.3f} M")

    pretrain_model = None
    pre_path = args.pretrain_ckpt or args.warm_start
    if pre_path:
        pretrain_model = AnchoredResampler(
            spec=spec, d_enc=d_enc, d_pool=d_pool,
            d_model=args.d_model, n_heads=args.n_heads, n_slots=K, n_layers=args.n_layers,
        ).to(device)
        load_warm_start(pretrain_model, Path(pre_path))
        pretrain_model.requires_grad_(False)
        pretrain_model.eval()

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
    anima.split_attn = False

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

    train_idx = split["train_idx"]
    eval_idx = split["eval_idx"][: args.eval_max_samples]
    collate = _make_collate(spec)

    def build_loader(idx_list, batch_size, num_batches, seed):
        dset = FinetuneDataset(
            spec, stems, idx_list, te_paths, npz_paths, active_lengths,
            flat_labels, V,
        )
        bucket_map = make_bucket_map(idx_list, npz_paths)
        sampler = BucketBatchSampler(bucket_map, batch_size, num_batches, seed)
        return DataLoader(dset, batch_sampler=sampler, num_workers=2, collate_fn=collate,
                          persistent_workers=True)

    train_loader = build_loader(train_idx, args.batch_size, args.steps, args.seed)
    eval_n_batches = max(1, len(eval_idx) // args.eval_batch_size)
    eval_loader = build_loader(eval_idx, args.eval_batch_size, eval_n_batches, args.seed + 1)

    cls_prefixes = model.cls_param_prefixes
    cls_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n.startswith(pref) for pref in cls_prefixes):
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

    model.train()
    log_rows = []
    t0 = time.time()

    step_iter = iter(train_loader)
    for step in tqdm(range(args.steps), desc="train", dynamic_ncols=True):
        batch = next(step_iter)
        latent = batch["latent"].to(device, dtype=torch.float32)
        t5_crossattn = batch["t5_crossattn"].to(device, dtype=torch.bfloat16)
        L_b = batch["L"]
        anchors = batch["anchors"]
        tok_b = tokens_all[batch["full_idx"]].to(device, dtype=torch.bfloat16)
        pool_b = pooled_all[batch["full_idx"]].to(device, dtype=torch.float32)

        ctx, logits, dev_labels, pred_k = build_ctx(
            model, spec, tok_b, pool_b, anchors, device, pad_to=args.S,
        )
        B_cur, K_cur = pred_k.shape[:2]
        pool_keep = anchor_keep_mask(spec, dev_labels, B_cur, K_cur, device)
        pooled_for_dit = masked_amax(pred_k, pool_keep).to(torch.bfloat16)

        fm_loss, fm_diag = dit_fm_loss(
            anima, latent, ctx, pooled_for_dit,
            args.sigma_sampling, args.sigmoid_scale, args.weighting_scheme,
        )
        ce_loss, accs = aux_cls_loss(spec, logits, dev_labels)

        loss = fm_loss + args.w_cls * ce_loss
        if args.w_retention > 0.0:
            retention = F.mse_loss(ctx.float(), t5_crossattn.float())
            loss = loss + args.w_retention * retention

        infonce_metrics: dict[str, float] = {}
        if use_infonce:
            K_cur = pred_k.shape[1]
            L_clip = L_b.clamp_max(K_cur).to(device)
            mask_k = torch.arange(K_cur, device=device).unsqueeze(0) < L_clip.unsqueeze(1)
            pred_pool = _pool(pred_k, mask_k)
            tgt_pool_b = tgt_pooled_all[batch["full_idx"]].to(device)
            infonce, infonce_metrics = _infonce_loss(
                pred_pool, tgt_pool_b, args.infonce_tau,
            )
            loss = loss + args.w_infonce * infonce

        if step == 0:
            nce_s = (
                f"  nce={infonce_metrics['infonce_loss']:.4f} "
                f"(r@1={infonce_metrics['infonce_acc']:.2f})"
                if infonce_metrics else ""
            )
            logger.info(
                f"[calibration] fm={fm_loss.item():.4f}  "
                f"ce={ce_loss.item():.4f}{nce_s}  "
                f"total={loss.item():.4f}"
            )
            if args.calibrate_only:
                return

        optim.zero_grad(set_to_none=True)
        loss.backward()
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
                "sigma_mean": float(fm_diag["sigma_mean"]),
                "lr_scale": float(lr_scale),
                "accs": accs,
            }
            if infonce_metrics:
                row["infonce"] = infonce_metrics["infonce_loss"]
                row["infonce_acc"] = infonce_metrics["infonce_acc"]
            log_rows.append(row)
            nce_s = (
                f"  nce={row['infonce']:.4f} r@1={row['infonce_acc']:.2f}"
                if infonce_metrics else ""
            )
            logger.info(
                f"step {step:>5}  loss={row['loss']:.4f}  fm={row['fm']:.4f}  "
                f"ce={row['ce']:.4f}{nce_s}  "
                f"σ̄={row['sigma_mean']:.3f}  "
                f"accs={accs}"
            )

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            logger.info(f"--- eval @ step {step} ---")
            if hasattr(anima, "switch_block_swap_for_inference"):
                anima.switch_block_swap_for_inference()
            try:
                ev = eval_fm(anima, model, pretrain_model, spec, eval_loader, args, device,
                             n_batches=eval_n_batches)
            finally:
                if hasattr(anima, "switch_block_swap_for_training"):
                    anima.switch_block_swap_for_training()
            logger.info(
                f"[eval] fm_finetune={ev['finetune']:.4f}  "
                f"fm_t5_real={ev['t5_real']:.4f}  "
                f"fm_pretrain={ev['pretrain']:.4f}  "
                f"cls_acc={ev['cls_acc']}"
            )
            log_rows.append({"step": step, "eval": ev})

    elapsed = time.time() - t0
    logger.info(f"done in {elapsed:.1f}s")

    out_tag = f"{ENCODER_NAME}_finetune"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(out_dir / f"{out_tag}.safetensors"),
    )
    (out_dir / f"{out_tag}.json").write_text(json.dumps(
        {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                     if k not in ("tokens_all", "pooled_all")},
            "anchor_spec": spec.to_metadata(),
            "n_params_M": n_params_M,
            "train_time_sec": elapsed,
            "log": log_rows,
        },
        indent=2,
    ))
    logger.info(f"saved -> {out_dir / out_tag}.(safetensors|json)")


def finetune_ckpt_path(out_dir: Path | str) -> Path:
    """Canonical final-ckpt path. Used by infer.py to load the trained resampler."""
    return Path(out_dir) / f"{ENCODER_NAME}_finetune.safetensors"


def main() -> None:
    finetune(parse_args())


if __name__ == "__main__":
    main()
