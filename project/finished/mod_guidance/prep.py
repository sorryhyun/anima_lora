"""Distillation prep — Phase 1 (uncond sidecar) + Phase 2 (synthetic latents).

Pre-generates auxiliary artifacts consumed by
``project/finished/mod_guidance/distill.py``.

Phase 1 — uncond TE sidecar:
    Emits ``<uncond_dir>/_anima_uncond_te.safetensors`` (default
    ``post_image_dataset/_anima_uncond_te.safetensors``) — the ``T5("")``
    cross-attention baseline used as the student's *unconditional* text input
    AND as CFG-negative during Phase 2 synthesis. ``make preprocess-te``
    normally produces this for free; this Phase 1 block is the explicit
    re-stager when you want a fresh encode. Replaces the
    ``torch.zeros_like(crossattn_emb)`` shortcut, which is neither paper-
    faithful (Starodubcev et al., ICLR 2026, arXiv:2602.09268v1 §5: "we
    propagate the textual prompt solely through the pooled text embedding,
    using an unconditional prompt for T5") nor what Anima's own CFG inference
    path uses (``library/inference/text.py:99-127``).

Phase 2 — teacher-driven synthetic clean latents:
    Walks each existing ``*_anima_te.safetensors`` in ``--cache_dir``, picks
    the sibling latent NPZ's resolution, runs the frozen teacher
    (base DiT, ``skip_pooled_text_proj=True``) from fresh noise through full
    CFG denoising (positive = cached crossattn_emb v0, negative = T5("") from
    the Phase 1 sidecar), saves the resulting clean latent under
    ``--synth_dir`` using the same NPZ layout as
    ``scripts/preprocess/cache_latents.py``. The trainer can then point at
    ``--synth_dir`` instead of (or alongside) the real-image cache to fit on
    the teacher's own manifold, removing the real-vs-teacher distribution gap
    that inflates the irreducible MSE floor.

Usage:
    # both phases (default — runs Phase 1 first if sidecar missing, then Phase 2)
    python -m project.finished.mod_guidance.prep

    # Phase 1 only (fast — staging only the uncond sidecar)
    python -m project.finished.mod_guidance.prep --skip_synth

    # Phase 2 only (assumes uncond sidecar exists)
    python -m project.finished.mod_guidance.prep --skip_uncond

    # cap synthesis for a smoke test
    python -m project.finished.mod_guidance.prep --max_samples 16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


from library.datasets.buckets import DCW_ASPECT_NAMES  # noqa: E402
from library.preprocess.uncond import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    DEFAULT_UNCOND_DIR,
    UNCOND_TE_FILENAME,
    stage_uncond_sidecar,
)
from .synth import generate_synthetic_latents  # noqa: E402

# Phase 2 synthesis allowlist: the measured top-5 buckets (DCW_ASPECT_NAMES) +
# the next 3 most-frequent in post_image_dataset/lora/, adding the
# near-square/landscape aspects the top-5 set lacks. Kept separate from
# DCW_ASPECT_BUCKETS so that frozen tuple's order stays stable for its consumers.
_DEFAULT_SYNTH_BUCKETS: tuple[str, ...] = DCW_ASPECT_NAMES + (
    "1120x960",  # near-square portrait
    "1024x1008",  # ~square
    "960x1120",  # near-square landscape
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="post_image_dataset/lora",
        help="LoRA cache dir (source TE + real-image latents).",
    )
    parser.add_argument(
        "--uncond_dir",
        type=str,
        default=str(DEFAULT_UNCOND_DIR),
        help=(
            'Where to stage the T5("") sidecar. Model-scoped and shipped as a '
            "bundled package asset (library/anima/assets/), so by default this "
            "reuses that file; pass --overwrite to regenerate it in place."
        ),
    )
    parser.add_argument(
        "--synth_dir",
        type=str,
        default="post_image_dataset/distill_mod_synth",
        help="Output dir for synthetic clean latents.",
    )
    parser.add_argument(
        "--qwen3",
        type=str,
        default="models/text_encoders/qwen_3_06b_base.safetensors",
    )
    parser.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-base-v1.0.safetensors",
    )
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Uncond TE seq length (default 512; matches CFG-uncond convention).",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="DiT attention mode for Phase 2 teacher forwards.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Denoising steps for synthesis (default 28 = Anima production).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="CFG scale for synthesis (default 4.0 = Anima production).",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help=(
            "Flow-matching sigma shift. Default 3.0 = Anima production env "
            "(configs/base.toml `discrete_flow_shift=3.0`). `inference.py`'s 5.0 "
            "default is upstream cruft that production callers override."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; per-sample seed = seed * 1_000_003 + sample_idx.",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=0,
        help="TE cache variant index to use as the conditioning prompt (default v0).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of synthetic latents (None = all discovered pairs).",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        # default="1200x896",
        default=",".join(_DEFAULT_SYNTH_BUCKETS),
        help=(
            "Comma-separated (H_pix x W_pix) resolution allowlist for synthesis. "
            "Default = DCW_ASPECT_NAMES (top-5 by frequency in "
            "post_image_dataset/lora/) plus 1120x960, 1024x1008, 960x1120 — the "
            "next three most-frequent buckets, each a distinct aspect not "
            "covered by the top-5 set. Pass empty string to disable the filter "
            "and synthesize every cached resolution."
        ),
    )
    parser.add_argument(
        "--n_per_bucket",
        type=int,
        default=1500,
        help=(
            "Cap synthesized stems per bucket (None = use every stem in the "
            "allowlist's buckets). With --shuffle_seed, picks deterministically "
            "across the bucket's full candidate pool."
        ),
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help=(
            "Deterministic shuffle seed for per-bucket selection when "
            "--n_per_bucket is set."
        ),
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help=(
            "Disable torch.compile of the DiT block stack. Compile is on by "
            "default (compile_blocks: native-shape flatten, one block graph per "
            "token-count family); auto-skipped when --blocks_to_swap > 0."
        ),
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Offload N transformer blocks to CPU during synthesis (low-VRAM).",
    )
    parser.add_argument(
        "--skip_uncond",
        action="store_true",
        help="Skip Phase 1 (assume the uncond sidecar already exists).",
    )
    parser.add_argument(
        "--skip_synth",
        action="store_true",
        help="Skip Phase 2 (stage only the uncond sidecar).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode the uncond sidecar AND re-synthesize already-present latents.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    synth_dir = Path(args.synth_dir)
    uncond_dir = Path(args.uncond_dir)

    # Phase 1 — uncond sidecar.
    uncond_path = uncond_dir / UNCOND_TE_FILENAME
    if not args.skip_uncond:
        uncond_path = stage_uncond_sidecar(
            uncond_dir,
            args.qwen3,
            args.dit,
            t5_tokenizer_path=args.t5_tokenizer_path,
            seq_len=args.seq_len,
            overwrite=args.overwrite,
        )
    elif not uncond_path.exists():
        raise FileNotFoundError(
            f"--skip_uncond was passed but {uncond_path} doesn't exist. "
            f"Run without --skip_uncond first."
        )

    # Phase 2 — synthetic latents.
    if args.skip_synth:
        logger.info("--skip_synth set; not generating synthetic latents.")
        return

    buckets: list[tuple[int, int]] | None = None
    if args.buckets.strip():
        try:
            buckets = [
                tuple(int(x) for x in tok.split("x"))
                for tok in (s.strip() for s in args.buckets.split(","))
                if tok
            ]
            if any(len(b) != 2 for b in buckets):
                raise ValueError
        except ValueError:
            raise SystemExit(
                f"--buckets must be comma-separated HxW (got {args.buckets!r})"
            )

    generate_synthetic_latents(
        cache_dir,
        synth_dir,
        dit_path=args.dit,
        uncond_path=uncond_path,
        attn_mode=args.attn_mode,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        variant=args.variant,
        max_samples=args.max_samples,
        blocks_to_swap=args.blocks_to_swap,
        overwrite=args.overwrite,
        buckets=buckets,
        n_per_bucket=args.n_per_bucket,
        shuffle_seed=args.shuffle_seed,
        do_compile=not args.no_compile,
    )


if __name__ == "__main__":
    main()
