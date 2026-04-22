#!/usr/bin/env python
"""Train the image→embedding (img2emb) resampler that maps a siglip2-encoded
image to a DiT-compatible cross-attention context, replacing text captions.

Three-stage pipeline (delegates to scripts/img2emb/*):

  1. features  — extract siglip2 patch tokens + pooled features for every
                 image, build train/eval split, scan active T5 lengths.
  2. pretrain  — train AnchoredResampler on cached crossattn_emb targets with
                 classifier heads + anchor injection (phase 1.5).
  3. finetune  — warm-start from pretrain ckpt and fine-tune via flow-matching
                 through the frozen DiT (phase 2a/2).

Outputs land in ``output/img2embs/{features,pretrain,finetune}/`` by default.

Usage:
    # Full pipeline with production defaults
    python scripts/img2emb/train_img2emb.py all

    # Just one stage
    python scripts/img2emb/train_img2emb.py pretrain
    python scripts/img2emb/train_img2emb.py finetune --steps 20000

    # Extra flags after the stage name pass through to the underlying script
    python scripts/img2emb/train_img2emb.py finetune --steps 20000 --batch_size 2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_DIR = Path(__file__).resolve().parent

STAGES = ("features", "pretrain", "finetune")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "stage",
        choices=(*STAGES, "all"),
        help="Which stage to run. 'all' runs features → pretrain → finetune.",
    )
    p.add_argument(
        "--out_dir",
        default="output/img2embs",
        help="Root output directory; creates features/, pretrain/, finetune/ under it.",
    )
    p.add_argument("--encoder", default="siglip2", help="Encoder name (siglip2 | dinov3).")
    p.add_argument("--image_dir", default="post_image_dataset",
                   help="Training image directory (must have cached VAE latents + T5 embeddings).")
    p.add_argument(
        "--dit",
        default="models/diffusion_models/anima-preview3-base.safetensors",
        help="Frozen DiT used for flow-matching supervision in the finetune stage.",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Trailing args forwarded verbatim to the underlying stage script.",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    env = os.environ.copy()
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def stage_features(out_root: Path, encoder: str, image_dir: str, extra: list[str]) -> None:
    features_dir = out_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        str(STAGE_DIR / "extract_features.py"),
        "--image_dir", image_dir,
        "--encoders", encoder,
        "--output_dir", str(features_dir),
        *extra,
    ])


def stage_pretrain(out_root: Path, encoder: str, image_dir: str, extra: list[str]) -> None:
    pretrain_dir = out_root / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        str(STAGE_DIR / "phase1_5_anchored.py"),
        "--cache_dir", str(out_root / "features"),
        "--out_dir", str(pretrain_dir),
        "--encoder", encoder,
        "--image_dir", image_dir,
        *extra,
    ])


def _pretrain_ckpt(out_root: Path, encoder: str, n_layers: int = 4) -> Path:
    # Matches phase1_5_anchored.py's ckpt naming convention.
    return out_root / "pretrain" / f"{encoder}_resampler_{n_layers}layer_anchored.safetensors"


def stage_finetune(
    out_root: Path, encoder: str, image_dir: str, dit: str, extra: list[str]
) -> None:
    finetune_dir = out_root / "finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)

    warm = _pretrain_ckpt(out_root, encoder)
    if not warm.exists():
        raise SystemExit(
            f"[train_img2emb] pretrain ckpt not found: {warm}\n"
            f"Run 'python scripts/img2emb/train_img2emb.py pretrain' first, or pass "
            f"--warm_start <path> as a trailing arg to override."
        )

    has_warm = any(a == "--warm_start" or a.startswith("--warm_start=") for a in extra)
    warm_args = [] if has_warm else ["--warm_start", str(warm)]

    _run([
        sys.executable,
        str(STAGE_DIR / "phase2_flow.py"),
        "--dit", dit,
        "--cache_dir", str(out_root / "features"),
        "--out_dir", str(finetune_dir),
        "--encoder", encoder,
        "--image_dir", image_dir,
        *warm_args,
        *extra,
    ])


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    extra = list(args.extra or [])
    # argparse.REMAINDER can yield a leading "--"; strip it.
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.stage == "all" and extra:
        raise SystemExit(
            "[train_img2emb] stage='all' does not accept pass-through args "
            "(ambiguous which stage they target). Run stages individually."
        )

    if args.stage in ("features", "all"):
        stage_features(out_root, args.encoder, args.image_dir, extra if args.stage == "features" else [])
    if args.stage in ("pretrain", "all"):
        stage_pretrain(out_root, args.encoder, args.image_dir, extra if args.stage == "pretrain" else [])
    if args.stage in ("finetune", "all"):
        stage_finetune(out_root, args.encoder, args.image_dir, args.dit, extra if args.stage == "finetune" else [])


if __name__ == "__main__":
    main()
