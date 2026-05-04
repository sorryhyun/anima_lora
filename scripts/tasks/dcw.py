"""DCW v4 calibrator: collect calibration data + train fusion head.

`make dcw` runs `scripts/dcw_measure_bias.py --dump_per_sample_gaps` once per
aspect bucket (1024², 832×1248, 1248×832) at the production env (CFG=4,
mod_w=3.0), then chains `scripts/dcw_train_fusion_head.py`. End artifact
is `bench/dcw/results/<timestamp>-make-dcw/fusion_head.safetensors`, which
`make test-dcw-v4` auto-resolves.

`make dcw-train` skips the sampling phase and trains on the existing pool.

See `docs/proposal/dcw-learnable-calibrator-v4.md` §I1.
"""

from __future__ import annotations

import sys

from ._common import ROOT, run

# (H, W) — must match library/inference/dcw_v4.py:ASPECT_TABLE
DCW_BUCKETS = [(1024, 1024), (832, 1248), (1248, 832)]


def _pop_kv(extra: list[str], key: str, default: str) -> tuple[str, list[str]]:
    """Extract ``--key value`` from extra. Returns (value, remaining_extra)."""
    if key in extra:
        i = extra.index(key)
        if i + 1 >= len(extra):
            sys.exit(f"missing value after {key}")
        value = extra[i + 1]
        return value, extra[:i] + extra[i + 2:]
    return default, list(extra)


def cmd_dcw(extra):
    """Calibrate over 3 aspect buckets, then train fusion head.

    Default sampling spec: 80 prompts × 3 seeds × 3 buckets at CFG=4 +
    mod-on. Runtime ~3-5h on a 5060 Ti (sampling-dominated). Override
    via --n_images/--n_seeds in extra. Other extra args pass through to
    every measure_bias invocation (--dit, --lora_weight, --pooled_text_proj
    '', --guidance_scale, etc.).
    """
    n_images, extra = _pop_kv(extra, "--n_images", "80")
    n_seeds, extra = _pop_kv(extra, "--n_seeds", "3")
    label, extra = _pop_kv(extra, "--label", "make-dcw")

    out_root = "post_image_dataset/dcw"
    for H, W in DCW_BUCKETS:
        bucket_label = f"{label}-{H}x{W}"
        print(f"\n=== DCW: bucket {H}x{W} ({n_images} imgs × {n_seeds} seeds) ===")
        run([
            sys.executable, "scripts/dcw_measure_bias.py",
            "--image_h", str(H),
            "--image_w", str(W),
            "--n_images", n_images,
            "--n_seeds", n_seeds,
            "--dump_per_sample_gaps",
            "--label", bucket_label,
            "--out_root", out_root,
            *extra,
        ])

    print("\n=== DCW: training fusion head on pooled trajectories ===")
    run([
        sys.executable, "scripts/dcw_train_fusion_head.py",
        "--label", label,
    ])
    print(
        "\nDone. Run `make test-dcw-v4` to inference with the fresh artifact "
        "(auto-resolves the latest fusion_head.safetensors)."
    )


def cmd_dcw_train(extra):
    """Train-only on existing bench/dcw/results/ pool (no sampling, ~30s)."""
    run([sys.executable, "scripts/dcw_train_fusion_head.py", *extra])
