#!/usr/bin/env python
"""Preprocess reference images for img2emb training.

Caches TIPSv2 patch tokens + pooled features, builds the train/eval split,
and scans cached T5 active-token lengths + per-variant pooled targets.

Outputs (under ``--output_dir``, default ``output/img2embs/features/``):

- ``features/tipsv2_tokens.safetensors``     (N, T_MAX_TOKENS, D) bf16 —
                                              zero-padded bucketed tokens
- ``features/tipsv2_pooled.safetensors``     (N, D) fp32 — CLS pooled
- ``features/tipsv2_buckets.json``           per-image bucket assignment
- ``features/target_pooled.safetensors``     (N, V, D) fp32 — per-variant
                                              mean over ``crossattn_emb_v*[:L]``
                                              (InfoNCE target)
- ``stems.json``                             ordered image stems
- ``active_lengths.json``                    per-image non-zero-prefix length
                                              (max over variants) + num_variants
                                              + image_dir
- ``split.json``                             80/20 train/eval stems + indices
- ``encoder_meta.json``                      resolved encoder config

Images are assigned to the patch-14 bucket whose aspect ratio matches theirs,
then all tokens zero-padded to ``T_MAX_TOKENS`` so the cache stays a single
``(N, T_MAX_TOKENS, D)`` tensor. See ``scripts/img2emb/buckets.py``.

Usage:
    python scripts/img2emb/preprocess.py
    python scripts/img2emb/preprocess.py --max_images 100
"""

import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.io.cache import discover_cached_images  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


# TIPSv2-L/14: 1 CLS + patch tokens (grid = bucket[0] × bucket[1]) at 14 px/patch,
# D=1024. Downloaded via `make download-tipsv2` (google/tipsv2-l14). Requires
# trust_remote_code=True. No HF image processor — reference preprocessing is
# plain torchvision Resize + ToTensor in [0, 1] (no ImageNet normalization).
ENCODER_NAME = "tipsv2"
ENCODER_MODEL_ID = str(REPO_ROOT / "models" / "tipsv2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        default=str(REPO_ROOT / "output" / "img2embs" / "features"),
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip artifacts that already exist (resume / incremental).",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- TIPSv2 wrapper


class TIPSv2Processor:
    """HF-AutoImageProcessor-shaped wrapper for TIPSv2.

    TIPSv2's reference preprocessing is plain Resize + ToTensor in ``[0, 1]`` —
    no ImageNet mean/std. Accepts ``image_size`` as int (square) or
    ``(H, W)`` tuple; bucket sizes are (H_pixels, W_pixels).
    """

    def __init__(self, image_size: int | tuple[int, int]):
        from torchvision import transforms

        if isinstance(image_size, int):
            size_hw = (image_size, image_size)
        else:
            size_hw = (int(image_size[0]), int(image_size[1]))
        self.image_size = size_hw
        self.transform = transforms.Compose(
            [
                transforms.Resize(size_hw),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, images, return_tensors: str = "pt"):
        assert return_tensors == "pt", "TIPSv2 processor only supports return_tensors='pt'"
        if not isinstance(images, (list, tuple)):
            images = [images]
        pixel_values = torch.stack([self.transform(img) for img in images], dim=0)
        return {"pixel_values": pixel_values}


class _TIPSv2Output:
    """Minimal HF ``BaseModelOutput``-shaped container."""

    __slots__ = ("last_hidden_state", "pooler_output")

    def __init__(self, last_hidden_state: torch.Tensor, pooler_output: torch.Tensor):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


class TIPSv2Encoder:
    """Adapt TIPSv2's ``encode_image`` API to HF's ``last_hidden_state`` /
    ``pooler_output`` convention.

    ``last_hidden_state`` = CLS prepended to patch tokens → ``(B, 1+N, D)``;
    ``pooler_output`` = the CLS token → ``(B, D)``.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, pixel_values: torch.Tensor) -> _TIPSv2Output:
        out = self.inner.encode_image(pixel_values)
        if isinstance(out, (tuple, list)):
            cls, patches = out[0], out[1]
        elif isinstance(out, dict):
            cls = out.get("cls_token", out.get("cls"))
            patches = out.get("patch_tokens", out.get("patches"))
        else:
            cls = getattr(out, "cls_token", None)
            patches = getattr(out, "patch_tokens", None)
        if cls is None or patches is None:
            raise RuntimeError(
                f"TIPSv2 encode_image returned unexpected structure: type={type(out)}. "
                "Expected (cls, patches) tuple, dict with 'cls_token'/'patch_tokens', "
                "or object with .cls_token/.patch_tokens."
            )
        if cls.dim() == 2:
            cls = cls.unsqueeze(1)  # (B, D) → (B, 1, D)
        last_hidden = torch.cat([cls, patches], dim=1)  # (B, 1+N, D)
        pooled = cls.squeeze(1)  # (B, D)
        return _TIPSv2Output(last_hidden_state=last_hidden, pooler_output=pooled)


def _ensure_tipsv2_siblings_cached(model_path: str) -> None:
    """TIPSv2's modeling_tips.py imports image_encoder.py / text_encoder.py as
    siblings at __init__ time. trust_remote_code only copies files referenced
    in auto_map into the transformers_modules cache, so these siblings go
    missing; the fallback then calls hf_hub_download(repo_id, ...) with the
    local path as repo_id and raises HFValidationError. Pre-copy them here."""
    src_dir = Path(model_path)
    if not src_dir.is_dir():
        return
    cache_dir = (
        Path.home()
        / ".cache/huggingface/modules/transformers_modules"
        / src_dir.name
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for sibling in ("image_encoder.py", "text_encoder.py"):
        src = src_dir / sibling
        if src.exists():
            shutil.copy2(src, cache_dir / sibling)


def load_encoder(device: torch.device) -> TIPSv2Encoder:
    """Return a TIPSv2 encoder wrapped in the HF-style output shim, in eval bf16."""
    from transformers import AutoModel

    logger.info(f"Loading tipsv2: {ENCODER_MODEL_ID}")
    _ensure_tipsv2_siblings_cached(ENCODER_MODEL_ID)
    inner = AutoModel.from_pretrained(
        ENCODER_MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    inner.eval().to(device).requires_grad_(False)
    return TIPSv2Encoder(inner)


# --------------------------------------------------------------------------- datasets


class _ImageDataset(Dataset):
    """Yields ``(idx, pixel_values[C,H,W])``. Pickled into DataLoader workers,
    so PIL decode + processor preprocessing overlap the GPU encoder forward."""

    def __init__(self, image_paths: list[str], processor: TIPSv2Processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return idx, pixel_values


class _ActiveLenDataset(Dataset):
    """Yields ``(idx, max_L, n_variants, pooled)`` — a lightweight scan that
    keeps the (V, 512, 1024) tensor in worker RAM just long enough to compute
    active length and per-variant pooled targets, then discards it.

    ``pooled`` is ``(V, D)`` fp32 — each row is the mean over the
    non-zero-prefix of one variant (``emb[:L].mean(dim=0)``). It is the
    multi-positive InfoNCE target consumed by phase 1.5 / phase 2. Pool uses
    per-variant L (not the max), independent of the K-cap — the target is
    what the resampler's pooled output should converge toward.
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

        D = int(variants[0].shape[-1])
        max_L = 0
        pooled = torch.zeros((len(variants), D), dtype=torch.float32)
        for vi, v in enumerate(variants):
            vf = v.float()
            nz = vf.abs().amax(dim=-1) > self.te_zero_eps
            L = int(nz.nonzero(as_tuple=False)[-1].item()) + 1 if nz.any() else 0
            max_L = max(max_L, L)
            if L > 0:
                pooled[vi] = vf[:L].mean(dim=0)
        return idx, max_L, len(variants), pooled


# --------------------------------------------------------------------------- encode


@torch.no_grad()
def encode_images(
    model: TIPSv2Encoder,
    image_paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Aspect-preserving bucketed encode over every image.

    Each image is assigned to the closest patch-14 bucket (see
    ``scripts/img2emb/buckets.py``), then bucket-groups are encoded one at a
    time so each DataLoader batch has uniform ``(H, W)``. Outputs are written
    into a single zero-padded ``(N, T_MAX_TOKENS, D)`` tokens tensor —
    downstream consumers keep the same flat cache schema.

    Returns ``(pooled, tokens, bucket_assignments)`` where
    ``bucket_assignments[i]`` is the ``[h_patches, w_patches]`` chosen for
    image ``i`` (diagnostic only; not read by training).
    """
    from scripts.img2emb.buckets import (
        PATCH,
        T_MAX_TOKENS,
        bucket_pixel_size,
        pick_bucket,
    )

    N = len(image_paths)
    bucket_by_idx: list[tuple[int, int]] = []
    for path in image_paths:
        with Image.open(path) as im:
            bucket_by_idx.append(pick_bucket(im.height, im.width))

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, bucket in enumerate(bucket_by_idx):
        groups[bucket].append(i)

    # Report the bucket distribution so preprocessing surprises are visible.
    dist = sorted(groups.items(), key=lambda kv: (-kv[0][0] / kv[0][1], kv[0]))
    logger.info(f"  tipsv2 bucket distribution (N={N}, T_MAX={T_MAX_TOKENS}):")
    for (h, w), idxs in dist:
        Hp, Wp = bucket_pixel_size((h, w))
        logger.info(
            f"    ({h:2d}x{w:2d}) {Hp}x{Wp} px  tokens={h * w + 1:4d}  count={len(idxs)}"
        )

    tokens_tensor: torch.Tensor | None = None
    pooled_tensor: torch.Tensor | None = None

    for bucket, indices in groups.items():
        Hp, Wp = bucket_pixel_size(bucket)
        T_bucket = bucket[0] * bucket[1] + 1  # +1 CLS
        processor = TIPSv2Processor(image_size=(Hp, Wp))
        subset_paths = [image_paths[i] for i in indices]
        ds = _ImageDataset(subset_paths, processor)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            shuffle=False,
            persistent_workers=num_workers > 0,
        )
        for local_idx, pixel_values in tqdm(
            loader, desc=f"encode/tipsv2/{bucket[0]}x{bucket[1]}", leave=False
        ):
            pixel_values = pixel_values.to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            out = model(pixel_values=pixel_values)
            last_hidden = out.last_hidden_state  # (B, T_bucket, D)
            pooled_b = out.pooler_output
            if pooled_b is None:
                pooled_b = last_hidden.mean(dim=1)
            assert last_hidden.shape[1] == T_bucket, (
                f"bucket {bucket} expected T={T_bucket}, got {last_hidden.shape[1]}"
            )

            if tokens_tensor is None:
                D = int(last_hidden.shape[-1])
                D_pool = int(pooled_b.shape[-1])
                tokens_tensor = torch.zeros(
                    (N, T_MAX_TOKENS, D), dtype=torch.bfloat16
                )
                pooled_tensor = torch.empty((N, D_pool), dtype=torch.float32)

            global_ids = torch.tensor(
                [indices[i] for i in local_idx.tolist()], dtype=torch.long
            )
            tokens_tensor[global_ids, :T_bucket] = (
                last_hidden.detach().to(torch.bfloat16).cpu()
            )
            pooled_tensor[global_ids] = pooled_b.detach().float().cpu()

    assert tokens_tensor is not None and pooled_tensor is not None
    assignments = [list(b) for b in bucket_by_idx]
    logger.info(
        f"  tipsv2: pooled={tuple(pooled_tensor.shape)}  "
        f"tokens={tuple(tokens_tensor.shape)}  patch={PATCH}"
    )
    return pooled_tensor, tokens_tensor, assignments


# --------------------------------------------------------------------------- targets


def scan_active_lengths(
    images,
    te_zero_eps: float = 1e-6,
    num_workers: int = 4,
    batch_size: int = 16,
):
    """Scan every TE file for non-zero-prefix length + per-variant pooled
    targets + verify V is consistent.

    No full-tensor stacking — targets stay on disk in per-image files and are
    loaded lazily by downstream probes. Fails loud if variant count differs
    across images (cache must be regenerated with a consistent shuffle count).

    Returns:
        active_lengths: ``list[int]`` — max L across variants per image.
        num_variants: ``int`` — V (usually 8).
        target_pooled: ``(N, V, D)`` fp32 — per-variant mean over ``emb[:L]``
            (multi-positive InfoNCE target for phase 1.5 / phase 2).
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
    pooled_tensor: torch.Tensor | None = None

    for idx, max_Ls, n_vs, pooled_b in tqdm(loader, desc="scan-targets"):
        n_vs_unique = set(n_vs.tolist())
        if len(n_vs_unique) != 1:
            raise RuntimeError(
                f"Variant count differs within batch: {sorted(n_vs_unique)}. "
                "Cache must be regenerated with a consistent shuffle count."
            )
        batch_V = next(iter(n_vs_unique))
        if V_seen is None:
            V_seen = batch_V
            D = int(pooled_b.shape[-1])
            pooled_tensor = torch.empty((N, V_seen, D), dtype=torch.float32)
        elif batch_V != V_seen:
            raise RuntimeError(
                f"Variant count differs across images: saw {V_seen} earlier, "
                f"now {batch_V}. Cache must be regenerated with a consistent "
                "shuffle count."
            )

        for i, L in zip(idx.tolist(), max_Ls.tolist()):
            L_list[i] = int(L)
        assert pooled_tensor is not None
        pooled_tensor[idx] = pooled_b.float()

    assert V_seen is not None and pooled_tensor is not None
    return L_list, V_seen, pooled_tensor


# --------------------------------------------------------------------------- stage entrypoint


def preprocess(args: argparse.Namespace) -> None:
    """Run the preprocess stage end-to-end using ``args``.

    Importable from ``train.py`` so the pipeline can run in-process instead of
    via ``subprocess``.
    """
    from scripts.img2emb.buckets import T_MAX_TOKENS

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

    # --- active-length scan + per-variant pooled targets (no full-tensor
    # stacking — targets stay on disk in per-image files, probes load them
    # lazily; only the per-variant pooled summary is cached in RAM).
    act_path = out_dir / "active_lengths.json"
    tgt_pool_path = feat_dir / "target_pooled.safetensors"
    if args.skip_existing and act_path.exists() and tgt_pool_path.exists():
        logger.info(
            "active_lengths.json + target_pooled.safetensors already present, skipping"
        )
    else:
        logger.info("Scanning per-image TE files for active lengths + pooled targets...")
        L_list, num_variants, target_pooled = scan_active_lengths(
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
        save_file({"pooled": target_pooled.contiguous()}, str(tgt_pool_path))
        logger.info(
            f"  → {tgt_pool_path} "
            f"(shape={tuple(target_pooled.shape)}, "
            f"≈{target_pooled.element_size() * target_pooled.numel() / 1e6:.1f} MB)"
        )

    # --- encoder features
    device = torch.device(args.device)
    p_pool = feat_dir / f"{ENCODER_NAME}_pooled.safetensors"
    p_tok = feat_dir / f"{ENCODER_NAME}_tokens.safetensors"
    p_buckets = feat_dir / f"{ENCODER_NAME}_buckets.json"
    existing = p_pool.exists() and p_tok.exists() and p_buckets.exists()
    if args.skip_existing and existing:
        logger.info(f"{ENCODER_NAME} features already cached, skipping")
    else:
        model = load_encoder(device)
        pooled, tokens, assignments = encode_images(
            model,
            image_paths,
            args.batch_size,
            args.num_workers,
            device,
        )
        save_file({"pooled": pooled}, str(p_pool))
        save_file({"tokens": tokens}, str(p_tok))
        with open(p_buckets, "w") as f:
            json.dump(
                {"stems": stems, "buckets": assignments, "t_max": int(tokens.shape[1])},
                f,
            )
        logger.info(f"  → {p_pool}")
        logger.info(f"  → {p_tok}")
        logger.info(f"  → {p_buckets}")
        del model
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

    # --- encoder meta (for probes / provenance)
    with open(out_dir / "encoder_meta.json", "w") as f:
        json.dump(
            {
                ENCODER_NAME: {
                    "model_id": ENCODER_MODEL_ID,
                    "t_max_tokens": T_MAX_TOKENS,
                    "bucketed": True,
                }
            },
            f,
            indent=2,
        )

    logger.info("Done.")


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
