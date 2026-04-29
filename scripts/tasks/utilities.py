"""Misc utility entry-points: merge, comfy-batch, distill-mod, test-unit, update,
export-logs, print-config."""

from __future__ import annotations

import os

from ._common import PY, _preset, run


def cmd_merge(extra):
    """Bake latest LoRA in ADAPTER_DIR (env, default 'output/ckpt') into the base DiT."""
    adapter_dir = os.environ.get("ADAPTER_DIR", "output/ckpt")
    multiplier = os.environ.get("MULTIPLIER", "1.0")
    run(
        [
            PY,
            "scripts/merge_to_dit.py",
            "--adapter_dir",
            adapter_dir,
            "--multiplier",
            multiplier,
            *extra,
        ]
    )


def cmd_comfy_batch(extra):
    workflow = extra[0] if extra else "workflows/modhydra.json"
    remaining = extra[1:] if extra else []
    run([PY, "scripts/comfy_batch.py", workflow, *remaining])


def cmd_distill_mod(extra):
    """Distill the pooled_text_proj MLP for modulation guidance.

    Saves to ``output/ckpt/pooled_text_proj.safetensors`` so ``test-mod`` picks it
    up automatically.
    """
    run(
        [
            PY,
            "scripts/distill_modulation.py",
            "--data_dir",
            "post_image_dataset/lora",
            "--dit_path",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--output_path",
            "output/ckpt/pooled_text_proj.safetensors",
            "--attn_mode",
            "flash",
            "--no_grad_ckpt",
            *extra,
        ]
    )


def cmd_test_unit(extra):
    run([PY, "-m", "pytest", "-q", "tests/", *extra])


def cmd_update(extra):
    """Update anima_lora from a GitHub release (preserves datasets/output/models;
    prompts on configs/methods/ + configs/gui-methods/ conflicts; runs uv sync)."""
    run([PY, "scripts/update.py", *extra])


def cmd_export_logs(extra):
    """Dump TB scalar logs to JSON. RUN=<dir> (default output/logs), ALL=1, JSONL=1."""
    run_path = os.environ.get("RUN", "output/logs")
    cmd = [PY, "scripts/export_logs_json.py", run_path]
    if os.environ.get("ALL"):
        cmd.append("--all")
    if os.environ.get("JSONL"):
        cmd.append("--jsonl")
    run([*cmd, *extra])


def cmd_print_config(extra):
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    run(
        [
            PY,
            "train.py",
            "--method",
            method,
            "--preset",
            preset,
            "--print-config",
            "--no-config-snapshot",
            *extra,
        ]
    )
