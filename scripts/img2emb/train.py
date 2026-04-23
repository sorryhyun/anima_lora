#!/usr/bin/env python
"""Train the image→embedding (img2emb) resampler end-to-end.

Three-stage pipeline, all run in-process (no subprocesses):

  1. preprocess — extract TIPSv2 patch tokens + pooled features for every
                  image, build train/eval split, scan active T5 lengths.
  2. pretrain   — train AnchoredResampler on cached crossattn_emb targets
                  with classifier heads + anchor injection.
  3. finetune   — warm-start from the pretrain ckpt and fine-tune via
                  flow-matching through the frozen DiT.

Outputs land in ``output/img2embs/{features,pretrain,finetune}/`` by default.

Usage:
    # Run pretrain → finetune (preprocess is separate; see below)
    python scripts/img2emb/train.py

    # Full pipeline including preprocessing
    python scripts/img2emb/train.py all

    # Just one stage. Extra args forward to that stage's parser.
    python scripts/img2emb/train.py preprocess
    python scripts/img2emb/train.py pretrain
    python scripts/img2emb/train.py finetune --steps 20000
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.finetune import (  # noqa: E402
    finetune as run_finetune,
    parse_args as parse_finetune_args,
)
from scripts.img2emb.preprocess import (  # noqa: E402
    preprocess as run_preprocess,
    parse_args as parse_preprocess_args,
)
from scripts.img2emb.pretrain import (  # noqa: E402
    parse_args as parse_pretrain_args,
    pretrain as run_pretrain,
    pretrain_ckpt_path,
)

STAGES = ("preprocess", "pretrain", "finetune")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "stage",
        nargs="?",
        default="train",
        choices=(*STAGES, "all", "train"),
        help="Which stage to run. 'train' (default) runs pretrain → finetune. "
             "'all' also runs preprocess first.",
    )
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "output" / "img2embs"),
        help="Root output directory; creates features/, pretrain/, finetune/ under it.",
    )
    p.add_argument(
        "--image_dir",
        default="post_image_dataset",
        help="Training image directory (must have cached VAE latents + T5 embeddings).",
    )
    p.add_argument(
        "--dit",
        default="models/diffusion_models/anima-preview3-base.safetensors",
        help="Frozen DiT used for flow-matching supervision in the finetune stage.",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Trailing args forwarded verbatim to the underlying stage parser.",
    )
    return p.parse_args()


def _extra_argv(extra: list[str] | None) -> list[str]:
    """Strip the leading '--' that argparse.REMAINDER may inject."""
    extra = list(extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def run_stage_preprocess(out_root: Path, image_dir: str, extra: list[str]) -> None:
    features_dir = out_root / "features"
    argv = [
        "--image_dir", image_dir,
        "--output_dir", str(features_dir),
        *extra,
    ]
    run_preprocess(parse_preprocess_args(argv))


def run_stage_pretrain(out_root: Path, image_dir: str, extra: list[str]) -> None:
    argv = [
        "--cache_dir", str(out_root / "features"),
        "--out_dir", str(out_root / "pretrain"),
        "--image_dir", image_dir,
        *extra,
    ]
    run_pretrain(parse_pretrain_args(argv))


def run_stage_finetune(
    out_root: Path, image_dir: str, dit: str, extra: list[str]
) -> None:
    warm = pretrain_ckpt_path(out_root / "pretrain")
    has_warm = any(a == "--warm_start" or a.startswith("--warm_start=") for a in extra)
    if not has_warm and not warm.exists():
        raise SystemExit(
            f"[train] pretrain ckpt not found: {warm}\n"
            f"Run 'python scripts/img2emb/train.py pretrain' first, or pass "
            f"--warm_start <path> as a trailing arg to override."
        )
    warm_args = [] if has_warm else ["--warm_start", str(warm)]

    argv = [
        "--dit", dit,
        "--cache_dir", str(out_root / "features"),
        "--out_dir", str(out_root / "finetune"),
        "--image_dir", image_dir,
        *warm_args,
        *extra,
    ]
    run_finetune(parse_finetune_args(argv))


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    extra = _extra_argv(args.extra)

    multi_stage = args.stage in ("all", "train")
    if multi_stage and extra:
        raise SystemExit(
            f"[train] stage='{args.stage}' does not accept pass-through args "
            "(ambiguous which stage they target). Run stages individually."
        )

    if args.stage in ("preprocess", "all"):
        run_stage_preprocess(
            out_root, args.image_dir, extra if args.stage == "preprocess" else []
        )
    if args.stage in ("pretrain", "all", "train"):
        run_stage_pretrain(
            out_root, args.image_dir, extra if args.stage == "pretrain" else []
        )
    if args.stage in ("finetune", "all", "train"):
        run_stage_finetune(
            out_root,
            args.image_dir,
            args.dit,
            extra if args.stage == "finetune" else [],
        )


if __name__ == "__main__":
    main()
