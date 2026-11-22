"""Shared Anima-loading helpers for bench/ scripts.

`bench/_common.py` owns the result envelope. This module owns the
*bench-facing* CLI surface (`add_common_args`) and re-exports the run
harness from `library/`:

- ``build_anima`` / ``AnimaBundle`` → ``library.runtime.harness`` (the
  DiT + adapter loader with the compile-after-apply ordering).
- ``discover_bucketed_samples`` → ``library.io.cache`` (next to the other
  cache-discovery helpers).

It also owns the bench defaults: the DiT/VAE/text-encoder paths (resolved
through ``default_checkpoints``) and the standard test prompt/neg.

Usage::

    from bench._anima import add_common_args, add_model_args, build_anima

    p = argparse.ArgumentParser()
    add_model_args(p)             # injects --dit/--vae/--text_encoder at the
                                  # canonical default_checkpoints() paths
    p.add_argument("--adapter", default=None)
    add_common_args(p)            # injects --label/--seed/--device/--dtype/
                                  # --attn_mode/--gradient_checkpointing/
                                  # --compile/--compile_mode
    args = p.parse_args()

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    bucket, picks = discover_bucketed_samples(
        Path("post_image_dataset/lora"), args.bucket, args.num_samples, args.seed
    )

All helpers are opt-in. A bench that doesn't load the DiT (e.g. an
analytical simulator) simply doesn't import this module. A bench that
needs two DiTs calls ``build_anima`` twice with explicit ``dit_path=``
overrides.
"""

from __future__ import annotations

import argparse

import torch

from anima_lora import default_checkpoints

# Re-exports: bench scripts import `build_anima` and the library.io.cache
# discovery helpers from here (noqa F401 keeps ruff from stripping them).
from library.io.cache import (  # noqa: F401
    LatentCacheFile,
    discover_bucketed_samples,
    discover_cached_pairs,
    discover_latents_by_stem,
    parse_latent_cache_name,
)
from library.runtime.argparse_groups import add_device_args
from library.runtime.device import str_to_dtype
from library.runtime.harness import AnimaBundle, build_anima  # noqa: F401

__all__ = [
    "add_common_args",
    "add_model_args",
    "resolve_dtype",
    "build_anima",
    "AnimaBundle",
    "discover_bucketed_samples",
    "discover_cached_pairs",
    "discover_latents_by_stem",
    "parse_latent_cache_name",
    "LatentCacheFile",
    "DEFAULT_DIT",
    "DEFAULT_VAE",
    "DEFAULT_TEXT_ENCODER",
    "DEFAULT_PROMPT",
    "DEFAULT_NEG",
]

# ---------------------------------------------------------------------------
# Canonical defaults — one source of truth for bench scripts.
# ---------------------------------------------------------------------------

# Default DiT / VAE / text-encoder paths, resolved through the façade so a bench
# honors ANIMA_DIT / ANIMA_VAE / ANIMA_TEXT_ENCODER + configs/base.toml + .env
# instead of re-hardcoding the "models/…/anima-base-v1.0.safetensors" strings.
# Reach for these (or `add_model_args`) rather than copy-pasting path literals.
_CKPTS = default_checkpoints()
DEFAULT_DIT = _CKPTS.dit
DEFAULT_VAE = _CKPTS.vae
DEFAULT_TEXT_ENCODER = _CKPTS.text_encoder

# The canonical single-image test prompt/neg — mirrors
# `scripts/tasks/_common.INFERENCE_BASE` so a bench's default sample matches what
# `make test` renders. Scripts that need varied content define their own list.
DEFAULT_PROMPT = (
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds."
)
DEFAULT_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"


# ---------------------------------------------------------------------------
# Common argparse surface.
# ---------------------------------------------------------------------------


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_label: bool = True,
    include_seed: bool = True,
    include_device: bool = True,
    include_dtype: bool = True,
    include_model: bool = True,
    include_checkpointing: bool = True,
    include_compile: bool = True,
) -> argparse.ArgumentParser:
    """Inject the bench-common CLI surface.

    All groups are individually opt-out so a bench can skip what doesn't
    apply (e.g. a CPU-only analytical script can drop ``include_device``).
    ``--device`` / ``--dtype`` delegate to
    ``library.runtime.argparse_groups.add_device_args`` (the shared compute flag group).

    NOTE (issues.md DX5): the flags below are added *for* you — do NOT also
    ``parser.add_argument`` them in your bench or argparse raises a
    "conflicting option string" error. If your bench needs its own ``--label``
    or ``--seed`` semantics, opt out of ours with ``include_label=False`` /
    ``include_seed=False`` and add your own.

    Flags injected at defaults:
        --label             free-form run label, fed to make_run_dir
        --seed              int, default 0
        --device            "cuda" | "cpu" | "cuda:N", default "cuda" if available
        --dtype             bf16|fp16|fp32, default bf16
        --attn_mode         flash|torch|..., default "flash"
        --gradient_checkpointing  bool flag
        --compile           bool flag — torch.compile DiT blocks
        --compile_mode      str, default None (inductor default)
    """
    if include_label:
        parser.add_argument(
            "--label",
            type=str,
            default=None,
            help="Free-form label appended to the run directory name.",
        )
    if include_seed:
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="RNG seed for sample discovery and noise draws.",
        )
    add_device_args(parser, include_device=include_device, include_dtype=include_dtype)
    if include_model:
        parser.add_argument(
            "--attn_mode",
            type=str,
            default="flash",
            help="Attention backend (flash, torch, ...). Default: flash.",
        )
    if include_checkpointing:
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            help="Enable activation checkpointing on the DiT. Trades ~30%% "
            "compute for ~4-5x smaller activation footprint. Required for "
            "benches that backward through the full DiT at high resolutions.",
        )
    if include_compile:
        parser.add_argument(
            "--compile",
            action="store_true",
            help="torch.compile each DiT block (via DiT.compile_blocks). First "
            "batch pays the compile cost (~30-60s); subsequent batches run "
            "faster. compile_blocks runs AFTER adapter apply_to + load_weights "
            "so the LoRA monkey-patches are part of the compiled graph.",
        )
        parser.add_argument(
            "--compile_mode",
            type=str,
            default=None,
            help="Optional inductor mode for compile_blocks (e.g. "
            "'reduce-overhead'). Leave unset for the default.",
        )
    return parser


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    dit: bool = True,
    vae: bool = True,
    text_encoder: bool = True,
) -> argparse.ArgumentParser:
    """Inject ``--dit`` / ``--vae`` / ``--text_encoder``, defaulted to the
    repo's canonical checkpoints (:func:`default_checkpoints`, via the constants
    above).

    The one place to wire a bench's model paths — replaces the re-hardcoded
    ``default="models/…/anima-base-v1.0.safetensors"`` strings so all benches
    honor ``ANIMA_DIT`` / ``ANIMA_VAE`` / ``ANIMA_TEXT_ENCODER`` + ``.env``.
    Each flag is opt-out for benches that only need a subset.
    """
    if dit:
        parser.add_argument(
            "--dit", type=str, default=DEFAULT_DIT, help="DiT checkpoint path."
        )
    if vae:
        parser.add_argument(
            "--vae", type=str, default=DEFAULT_VAE, help="VAE checkpoint path."
        )
    if text_encoder:
        parser.add_argument(
            "--text_encoder",
            type=str,
            default=DEFAULT_TEXT_ENCODER,
            help="Text-encoder checkpoint path.",
        )
    return parser


def resolve_dtype(name: str) -> torch.dtype:
    """Map a --dtype string to a torch dtype. Raises ValueError on unknown.

    Delegates to ``library.runtime.device.str_to_dtype`` (the canonical home,
    also exported as ``anima_lora.str_to_dtype``) so the bench harness shares one
    source of truth for the mapping.
    """
    return str_to_dtype(name)
