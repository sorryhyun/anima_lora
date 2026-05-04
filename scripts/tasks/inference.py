"""Inference entry-points for shipped methods (test / test-* commands).

All variants share ``INFERENCE_BASE`` from ``_common`` and add method-specific
flags. Experimental inference commands (exp-test-apex, exp-test-postfix*,
exp-test-prefix, exp-test-ref, exp-test-ip, exp-test-easycontrol) live in
``scripts/experimental_tasks/inference.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ._common import (
    INFERENCE_BASE,
    ROOT,
    latest_hydra,
    latest_lora,
    latest_output,
    run,
)


def cmd_test(extra):
    run([*INFERENCE_BASE, "--lora_weight", str(latest_lora()), *extra])


def cmd_test_mod(extra):
    """Inference with the latest distilled pooled_text_proj MLP for modulation guidance."""
    run(
        [
            *INFERENCE_BASE,
            "--pooled_text_proj",
            str(latest_output("pooled_text_proj")),
            *extra,
        ]
    )


def cmd_test_hydra(extra):
    # Uses the moe sibling (router-live); static-merge is auto-skipped in
    # library/inference_pipeline.py:_is_hydra_moe detection.
    run([*INFERENCE_BASE, "--lora_weight", str(latest_hydra()), *extra])


def cmd_test_merge(extra):
    """Inference with a baked (merged) DiT from MODEL_DIR (default 'output_temp').

    MODEL_DIR accepts either a directory (picks the latest
    ``*_merged.safetensors`` inside) or a direct ``.safetensors`` path. The
    merged file is a standalone DiT (LoRA folded in), so no ``--lora_weight``
    is passed. The trailing ``--dit`` overrides the base one in
    ``INFERENCE_BASE`` (argparse keeps the last value).
    """
    target = Path(os.environ.get("MODEL_DIR", "output_temp"))
    if not target.is_absolute():
        target = ROOT / target
    if target.is_file():
        chosen = target
    elif target.is_dir():
        candidates = sorted(
            target.glob("*_merged.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"No '*_merged.safetensors' files found in {target}", file=sys.stderr)
            sys.exit(1)
        chosen = candidates[0]
    else:
        print(f"MODEL_DIR path not found: {target}", file=sys.stderr)
        sys.exit(1)
    run([*INFERENCE_BASE, "--dit", str(chosen), *extra])


def cmd_test_spectrum(extra):
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            "--spectrum",
            "--spectrum_window_size",
            "2.0",
            "--spectrum_flex_window",
            "0.25",
            "--spectrum_warmup",
            "7",
            "--spectrum_w",
            "0.3",
            "--spectrum_m",
            "3",
            "--spectrum_lam",
            "0.1",
            "--spectrum_stop_caching_step",
            "29",
            "--spectrum_calibration",
            "0.0",
            *extra,
        ]
    )


def cmd_test_dcw(extra):
    """Inference with latest LoRA + DCW post-step correction.

    Defaults bake in λ=-0.010 + one_minus_sigma schedule (see
    bench/dcw/findings.md). Override via --dcw_lambda / --dcw_schedule in extra.
    """
    run([*INFERENCE_BASE, "--lora_weight", str(latest_lora()), "--dcw", *extra])


def _latest_fusion_head() -> str:
    """Resolve the most recent fusion_head.safetensors under any DCW root.

    Scans post_image_dataset/dcw/ (new `make dcw` output) first, then
    bench/dcw/results/ (legacy). Newest mtime wins across both.
    """
    from pathlib import Path

    roots = [Path("post_image_dataset/dcw"), Path("bench/dcw/results")]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("*/fusion_head.safetensors"))
    if not candidates:
        raise SystemExit(
            "no fusion_head.safetensors found under post_image_dataset/dcw/ "
            "or bench/dcw/results/ — run `make dcw-train` first"
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def cmd_test_dcw_v4(extra):
    """Inference with latest LoRA + DCW v4 learnable calibrator.

    Auto-resolves the most recent fusion_head.safetensors. Pass
    --dcw_v4 <path> in extra to override. Pass --dcw_v4_disable_shrinkage
    if the artifact's σ̂² channel didn't pass Gate B (the prototype's didn't).
    """
    extra_has_v4 = any(a.startswith("--dcw_v4") and not a.startswith("--dcw_v4_") for a in extra)
    v4_args = [] if extra_has_v4 else ["--dcw_v4", _latest_fusion_head()]
    run([
        *INFERENCE_BASE,
        "--lora_weight", str(latest_lora()),
        *v4_args,
        "--dcw_v4_disable_shrinkage",  # prototype σ̂² channel doesn't pass Gate B
        *extra,
    ])


def cmd_test_spectrum_dcw(extra):
    """Spectrum + DCW composed. Same Spectrum knobs as test-spectrum."""
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            "--spectrum",
            "--spectrum_window_size",
            "2.0",
            "--spectrum_flex_window",
            "0.25",
            "--spectrum_warmup",
            "7",
            "--spectrum_w",
            "0.3",
            "--spectrum_m",
            "3",
            "--spectrum_lam",
            "0.1",
            "--spectrum_stop_caching_step",
            "29",
            "--spectrum_calibration",
            "0.0",
            "--dcw",
            *extra,
        ]
    )
