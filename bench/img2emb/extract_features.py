#!/usr/bin/env python
"""Extract and cache image encoder features for the phase-0 bench.

crossattn_emb targets are NOT re-materialized here — they already exist
per-image in ``post_image_dataset/*_anima_te.safetensors``. Probes load them
on demand (see ``phase0_probes.py``). This script only emits the encoder
features, the split, and a lightweight scan of active-token lengths.

Outputs (under ``--output_dir``, default ``bench/img2emb/results/phase0/``):

- ``features/{encoder}_tokens.safetensors``   (N, T, D_enc) bf16 — last_hidden_state
- ``features/{encoder}_pooled.safetensors``   (N, D_enc) fp32 — pooler_output / CLS / mean
- ``stems.json``                              ordered image stems for every tensor above
- ``active_lengths.json``                     per-image non-zero-prefix length
                                               (max over variants) + num_variants + image_dir
- ``split.json``                              80/20 train/eval stems + indices
- ``encoder_meta.json``                       resolved encoder config (model_id, image_size)

Run once, then iterate probes against the cache.

Usage:
    python bench/img2emb/extract_features.py
    python bench/img2emb/extract_features.py --max_images 100 --encoders dinov3
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.io.cache import discover_cached_images  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


ENCODERS = {
    "dinov3": {
        "model_id": str(REPO_ROOT / "models" / "dino"),
        "image_size": 224,
        # DINOv3: CLS + 4 register + 196 patches at 224×224, D=1280
    },
    "siglip2": {
        "model_id": "google/siglip2-large-patch16-384",
        "image_size": 384,
        # SigLIP2 large: 576 patches at 384×384, D=1024
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image_dir", default="post_image_dataset")
    p.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Cap at N random images from the dataset (seeded). Set to 0 for all.",
    )
    p.add_argument("--seed", type=int, default=42, help="Subsample + split seed")
    p.add_argument("--eval_frac", type=float, default=0.2)
    p.add_argument(
        "--encoders",
        default="dinov3,siglip2",
        help="Comma-separated encoder names from ENCODERS dict",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for image decode + preprocessing / TE load.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for encoder forward (features stay in RAM after)",
    )
    p.add_argument(
        "--output_dir",
        default=str(BENCH_DIR / "results" / "phase0"),
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip artifacts that already exist (resume / incremental).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- datasets


class _ImageDataset(Dataset):
    """Yields ``(idx, pixel_values[C,H,W])``. Pickled into DataLoader workers,
    so PIL decode + processor preprocessing overlap the GPU encoder forward."""

    def __init__(self, image_paths: list[str], processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return idx, pixel_values


class _ActiveLenDataset(Dataset):
    """Yields ``(idx, max_L, n_variants)`` — a lightweight scan that keeps the
    (V, 512, 1024) tensor in worker RAM just long enough to compute active
    length, then discards it. Nothing big crosses the worker/main boundary.
    """

    def __init__(self, te_paths: list[str], te_zero_eps: float = 1e-6):
        self.te_paths = te_paths
        self.te_zero_eps = te_zero_eps

    def __len__(self) -> int:
        return len(self.te_paths)

    def __getitem__(self, idx: int):
        sd = load_file(self.te_paths[idx])
        variant_keys = sorted(k for k in sd.keys() if k.startswith("crossattn_emb_v"))
        if variant_keys:
            variants = [sd[k] for k in variant_keys]
        elif "crossattn_emb" in sd:
            variants = [sd["crossattn_emb"]]
        else:
            raise RuntimeError(f"No crossattn_emb in {self.te_paths[idx]}")

        max_L = 0
        for v in variants:
            nz = v.abs().amax(dim=-1) > self.te_zero_eps
            L = int(nz.nonzero(as_tuple=False)[-1].item()) + 1 if nz.any() else 0
            max_L = max(max_L, L)
        return idx, max_L, len(variants)


# --------------------------------------------------------------------------- encoders


def load_encoder(name: str, device: torch.device):
    """Return ``(model, processor)`` for a named encoder; model in eval bf16."""
    from transformers import AutoImageProcessor, AutoModel

    cfg = ENCODERS[name]
    mid = cfg["model_id"]
    logger.info(f"Loading {name}: {mid}")

    processor = AutoImageProcessor.from_pretrained(mid, use_fast=True)

    if name == "siglip2":
        # We only need the vision tower; Siglip2Model also loads the text tower
        # which wastes ~500MB. Load the whole thing and keep only .vision_model.
        full = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16)
        model = full.vision_model
        del full
    else:
        model = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16)

    model.eval()
    model.to(device)
    model.requires_grad_(False)
    return model, processor


@torch.no_grad()
def extract_features(
    model,
    processor,
    name: str,
    image_paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    """Run encoder over every image, return ``(pooled, tokens)``.

    ``pooled`` is ``(N, D_enc)`` fp32, using ``out.pooler_output`` if the model
    provides one and a mean-over-tokens fallback otherwise.
    ``tokens`` is ``(N, T, D_enc)`` bf16, the full ``last_hidden_state`` (all
    CLS / register / patch tokens — the resampler doesn't care about the split).

    Outputs are pre-allocated and filled by index, so peak RAM is the final
    tensor size (no chunk-list doubling during ``torch.cat``).
    """
    dataset = _ImageDataset(image_paths, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        persistent_workers=num_workers > 0,
    )

    N = len(dataset)
    pooled_tensor: torch.Tensor | None = None
    tokens_tensor: torch.Tensor | None = None

    for idx, pixel_values in tqdm(loader, desc=f"encode/{name}"):
        pixel_values = pixel_values.to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        out = model(pixel_values=pixel_values)
        last_hidden = out.last_hidden_state  # (B, T, D)

        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            # Fallback: mean over all tokens. DINOv3 doesn't always expose
            # pooler_output; the CLS token is position 0 but mean-pool is a
            # safer apples-to-apples default across encoders.
            pooled = last_hidden.mean(dim=1)

        if tokens_tensor is None:
            T, D = last_hidden.shape[1], last_hidden.shape[2]
            D_pool = pooled.shape[-1]
            tokens_tensor = torch.empty((N, T, D), dtype=torch.bfloat16)
            pooled_tensor = torch.empty((N, D_pool), dtype=torch.float32)

        tokens_tensor[idx] = last_hidden.detach().to(torch.bfloat16).cpu()
        pooled_tensor[idx] = pooled.detach().float().cpu()

    assert pooled_tensor is not None and tokens_tensor is not None
    logger.info(
        f"  {name}: pooled={tuple(pooled_tensor.shape)}  tokens={tuple(tokens_tensor.shape)}"
    )
    return pooled_tensor, tokens_tensor


# --------------------------------------------------------------------------- targets


def scan_active_lengths(
    images,
    te_zero_eps: float = 1e-6,
    num_workers: int = 4,
    batch_size: int = 16,
):
    """Scan every TE file for non-zero-prefix length + verify V is consistent.

    No tensor stacking — targets stay on disk in per-image files and are loaded
    lazily by downstream probes. Fails loud if variant count differs across
    images (cache must be regenerated with a consistent shuffle count).

    Returns:
        active_lengths: ``list[int]`` — max L across variants per image.
        num_variants: ``int`` — V (usually 8).
    """
    te_paths = [img.te_path for img in images]
    dataset = _ActiveLenDataset(te_paths, te_zero_eps=te_zero_eps)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )

    N = len(dataset)
    L_list = [0] * N
    V_seen: int | None = None

    for idx, max_Ls, n_vs in tqdm(loader, desc="scan-targets"):
        # idx: (B,), max_Ls: (B,), n_vs: (B,)
        n_vs_unique = set(n_vs.tolist())
        if len(n_vs_unique) != 1:
            raise RuntimeError(
                f"Variant count differs within batch: {sorted(n_vs_unique)}. "
                "Cache must be regenerated with a consistent shuffle count."
            )
        batch_V = next(iter(n_vs_unique))
        if V_seen is None:
            V_seen = batch_V
        elif batch_V != V_seen:
            raise RuntimeError(
                f"Variant count differs across images: saw {V_seen} earlier, "
                f"now {batch_V}. Cache must be regenerated with a consistent "
                "shuffle count."
            )

        for i, L in zip(idx.tolist(), max_Ls.tolist()):
            L_list[i] = int(L)

    assert V_seen is not None
    return L_list, V_seen


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    images = discover_cached_images(args.image_dir)
    images = [
        i for i in images if i.te_path is not None and i.image_path is not None
    ]
    if not images:
        raise SystemExit(f"No images with TE cache + PNG in {args.image_dir}")

    rng = random.Random(args.seed)
    if args.max_images and len(images) > args.max_images:
        images = rng.sample(images, args.max_images)
    images = sorted(images, key=lambda i: i.stem)
    logger.info(f"Using {len(images)} images")

    stems = [i.stem for i in images]
    image_paths = [i.image_path for i in images]

    with open(out_dir / "stems.json", "w") as f:
        json.dump(stems, f, indent=2)

    # --- active-length scan (no target stacking — probes load per-image TEs directly)
    act_path = out_dir / "active_lengths.json"
    if args.skip_existing and act_path.exists():
        logger.info(f"active_lengths.json already present, skipping ({act_path})")
    else:
        logger.info("Scanning per-image TE files for active lengths...")
        L_list, num_variants = scan_active_lengths(
            images,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        with open(act_path, "w") as f:
            json.dump(
                {
                    "stems": stems,
                    "active_lengths": L_list,
                    "num_variants": num_variants,
                    "image_dir": str(args.image_dir),
                },
                f,
                indent=2,
            )
        logger.info(
            f"  → {act_path} (V={num_variants}, "
            f"L p50={sorted(L_list)[len(L_list) // 2]} "
            f"max={max(L_list)})"
        )

    # --- encoders
    device = torch.device(args.device)
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    for name in encoders:
        if name not in ENCODERS:
            logger.warning(f"Unknown encoder '{name}', skipping")
            continue
        p_pool = feat_dir / f"{name}_pooled.safetensors"
        p_tok = feat_dir / f"{name}_tokens.safetensors"
        if args.skip_existing and p_pool.exists() and p_tok.exists():
            logger.info(f"{name} features already cached, skipping")
            continue
        model, processor = load_encoder(name, device)
        pooled, tokens = extract_features(
            model,
            processor,
            name,
            image_paths,
            args.batch_size,
            args.num_workers,
            device,
        )
        save_file({"pooled": pooled}, str(p_pool))
        save_file({"tokens": tokens}, str(p_tok))
        logger.info(f"  → {p_pool}")
        logger.info(f"  → {p_tok}")
        del model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- split
    split_path = out_dir / "split.json"
    if args.skip_existing and split_path.exists():
        logger.info(f"Split already cached ({split_path})")
    else:
        split_rng = random.Random(args.seed + 1)
        idx = list(range(len(stems)))
        split_rng.shuffle(idx)
        n_eval = int(len(idx) * args.eval_frac)
        eval_idx = sorted(idx[:n_eval])
        train_idx = sorted(idx[n_eval:])
        with open(split_path, "w") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "eval_frac": args.eval_frac,
                    "train": [stems[i] for i in train_idx],
                    "eval": [stems[i] for i in eval_idx],
                    "train_idx": train_idx,
                    "eval_idx": eval_idx,
                },
                f,
                indent=2,
            )
        logger.info(
            f"  → {split_path} (train={len(train_idx)} eval={len(eval_idx)})"
        )

    # --- encoder meta (for probes)
    with open(out_dir / "encoder_meta.json", "w") as f:
        json.dump(
            {name: ENCODERS[name] for name in encoders if name in ENCODERS},
            f,
            indent=2,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
