"""Wrapper around ``train.AnimaTrainer`` for the ComfyUI trainer node.

Avoids ``library.config.io.read_config_from_file`` because that helper re-reads
``sys.argv`` — fine for CLI, broken inside ComfyUI where argv is the server's.
Instead we replicate its merge by calling ``load_method_preset`` directly, then
fill argparse defaults by parsing an empty argv against a pre-populated
namespace.

Relative paths in ``configs/base.toml`` (model / VAE / text encoder) are
rewritten to absolute so the trainer can run from any CWD.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional


def _resolve_relative_paths(cfg: dict, anima_lora_root: str) -> None:
    """Rewrite relative model-path keys to absolute, based at anima_lora_root.

    Only affects keys that (a) exist in the dict, (b) are strings, (c) are not
    already absolute. Output_dir/logging_dir are intentionally not rewritten —
    callers override those explicitly.
    """
    path_keys = (
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "tokenizer_cache_dir",
    )
    for key in path_keys:
        val = cfg.get(key)
        if isinstance(val, str) and val and not os.path.isabs(val):
            cfg[key] = os.path.abspath(os.path.join(anima_lora_root, val))


def build_training_namespace(
    *,
    anima_lora_root: str,
    method: str,
    preset: str,
    methods_subdir: str,
    overrides: dict,
) -> argparse.Namespace:
    """Produce a fully-populated argparse.Namespace ready for AnimaTrainer.train.

    Merge order: ``base.toml`` → ``presets.toml[<preset>]`` →
    ``<methods_subdir>/<method>.toml`` → ``overrides``. Then argparse defaults
    fill any remaining required fields.
    """
    # Imports are deferred so that the enclosing package can insert
    # anima_lora_root onto sys.path before anything touches `library.*`.
    from library.config import schema as _config_schema
    from library.config.io import load_method_preset
    from train import build_network_extras, setup_parser

    configs_dir = os.path.join(anima_lora_root, "configs")
    merged = load_method_preset(
        method,
        preset=preset,
        configs_dir=configs_dir,
        methods_subdir=methods_subdir,
    )
    merged.update(overrides)
    _resolve_relative_paths(merged, anima_lora_root)

    parser = setup_parser()
    _config_schema.populate_schema(parser, extras=build_network_extras())
    args = parser.parse_args([], namespace=argparse.Namespace(**merged))

    # Backward-compat shim — matches the top-of-file logic in train.py's __main__.
    if getattr(args, "attn_mode", None) == "sdpa":
        args.attn_mode = "torch"

    return args


def run_training(
    args: argparse.Namespace,
    *,
    anima_lora_root: str,
) -> str:
    """Call AnimaTrainer.train and return the absolute path of the saved LoRA.

    Temporarily switches CWD into ``anima_lora_root`` because some internal
    paths (e.g. the config-snapshot writer's ``_git_sha()`` call) are implicitly
    relative. The old CWD is restored unconditionally.
    """
    from library.training import verify_command_line_training_args
    from train import AnimaTrainer

    verify_command_line_training_args(args)

    old_cwd = os.getcwd()
    os.chdir(anima_lora_root)
    try:
        trainer = AnimaTrainer()
        trainer.train(args)
    finally:
        os.chdir(old_cwd)

    # AnimaTrainer writes `<output_dir>/<output_name>.safetensors` at the end of
    # training. We set both upstream, so this path is deterministic.
    saved_path = os.path.join(args.output_dir, f"{args.output_name}.safetensors")
    if not os.path.exists(saved_path):
        raise RuntimeError(
            f"Training finished but expected output not found: {saved_path}. "
            f"Check the ComfyUI console log for errors."
        )
    return saved_path


GPU_TIER_PRESET = {
    "8GB": "low_vram",
    "16GB": "default",
    "high": "32gb",
}


def gpu_tier_overrides(tier: str) -> dict:
    """Overrides to layer on top of the preset for each simple-node gpu tier.

    - 8GB → keep low_vram's grad_ckpt + unsloth offload, blocks_to_swap=0
    - 16GB → no grad_ckpt, blocks_to_swap=12
    - high → no grad_ckpt, blocks_to_swap=0
    """
    if tier == "16GB":
        return {
            "blocks_to_swap": 12,
            "gradient_checkpointing": False,
            "unsloth_offload_checkpointing": False,
        }
    if tier == "high":
        return {
            "blocks_to_swap": 0,
            "gradient_checkpointing": False,
            "unsloth_offload_checkpointing": False,
        }
    # 8GB → low_vram preset already sets the right flags; no overrides needed.
    return {}


def rank_overrides(rank: int) -> dict:
    """lr + dim + alpha mapping for the simple node."""
    if rank not in (16, 32):
        raise ValueError(f"Simple node only supports rank 16 or 32, got {rank}")
    return {
        "network_dim": rank,
        "network_alpha": float(rank),
        "learning_rate": 1e-4 if rank == 16 else 5e-5,
    }


def find_anima_lora_root(start: Optional[str] = None) -> str:
    """Walk up from ``start`` to find the directory holding ``train.py`` +
    ``configs/base.toml`` — i.e. the anima_lora workspace root."""
    here = os.path.abspath(start or os.path.dirname(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(here, "train.py")) and os.path.isdir(
            os.path.join(here, "configs")
        ):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise RuntimeError(
        "Could not locate anima_lora workspace root (train.py + configs/) "
        "above this custom_nodes/ directory."
    )
