#!/usr/bin/env python3
"""Template bench — copy `bench/_template/` to `bench/<your_topic>/` and edit.

Shows the full contract from `bench/README.md` in one runnable file:

  1. CLI surface via `bench._anima` (`add_model_args` + `add_common_args`)
  2. model build via `build_anima` (the compile-after-apply harness)
  3. result envelope via `bench._common` (`make_run_dir` + `write_result`)

The placeholder measurement — DiT load wall-clock, parameter count, and peak
VRAM after build — is real enough to run end-to-end with nothing but the DiT
downloaded, but it is a placeholder: replace `metrics` with the number(s) your
change claims to move, and report them for BOTH the before and the after tree
(a single-number claim with no reproducible script does not clear the Tier 1.5
bar — see CONTRIBUTING.md).

Usage::

    .venv/bin/python bench/_template/run_bench.py [--label smoke] [--compile]
    .venv/bin/python bench/_template/run_bench.py --adapter output/ckpt/my_lora.safetensors
"""

from __future__ import annotations

import argparse
import csv
import time

import torch

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --dit at the canonical default_checkpoints() path. This template never
    # decodes or encodes text, so skip --vae/--text_encoder; keep them if your
    # bench renders images or encodes prompts.
    add_model_args(parser, vae=False, text_encoder=False)
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA-family .safetensors to attach before compile.",
    )
    # --label/--seed/--device/--dtype/--attn_mode/--gradient_checkpointing/
    # --compile/--compile_mode. Do NOT re-add these yourself (conflicting
    # option strings) — opt out via include_* kwargs if you need your own.
    add_common_args(parser)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    load_seconds = time.perf_counter() - t0

    # --- placeholder measurement: replace from here down with your bench ----
    per_module = {
        name: sum(p.numel() for p in child.parameters())
        for name, child in bundle.anima.named_children()
    }
    metrics = {
        "load_seconds": round(load_seconds, 2),
        "dit_params": sum(p.numel() for p in bundle.anima.parameters()),
        "network_params": (
            sum(p.numel() for p in bundle.network.parameters())
            if bundle.network is not None
            else 0
        ),
        "peak_vram_gb": (
            round(torch.cuda.max_memory_allocated(bundle.device) / 2**30, 2)
            if bundle.device.type == "cuda"
            else None
        ),
    }
    # ------------------------------------------------------------------------

    # `make_run_dir("<your_topic>")` → bench/<your_topic>/results/<ts>[-label]/
    run_dir = make_run_dir("_template", label=args.label)

    # Artifacts are anything worth keeping next to result.json (CSVs, PNGs,
    # NPZs). List them by name so the envelope indexes them.
    with (run_dir / "param_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["module", "params"])
        writer.writerows(sorted(per_module.items()))

    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["param_counts.csv"],
        device=bundle.device,
    )
    print(f"metrics: {metrics}")
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
