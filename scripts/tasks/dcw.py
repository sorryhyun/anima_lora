"""DCW v4 calibrator: collect calibration data + train fusion head.

`make dcw` runs `scripts/dcw/measure_bias.py --dump_per_sample_gaps` once
per aspect bucket (top-5 by sample count: 832×1248, 896×1152, 768×1344,
1152×896, 1248×832) at the production env (CFG=4, mod_w=3.0), then
chains `scripts/dcw/train_fusion_head.py`. End artifact is
`output/dcw/<timestamp>-v4-fusion-head-make-dcw/fusion_head.safetensors`,
which `make test-dcw-v4` auto-resolves.

The trainer is bucket-agnostic (single population μ_g, aspect_emb pinned
to zero) — see `project_dcw_bucket_prior_cosmetic` memory. Per-bucket
sampling is kept only to balance the prompt pool across aspect buckets.

`make dcw-train` skips the sampling phase and trains on the existing pool.

See `docs/proposal/dcw-learnable-calibrator-v4.md` §I1.
"""

from __future__ import annotations

import sys

from ._common import run

# (H, W) — must match library/inference/dcw_v4.py:ASPECT_TABLE.
# Top 5 buckets by post_image_dataset/lora/ count (aspect_id = list index).
DCW_BUCKETS = [
    (832, 1248),  # HD portrait — most common
    (896, 1152),  # 3:4 portrait
    (768, 1344),  # tall portrait
    (1152, 896),  # 3:4 landscape
    (1248, 832),  # HD landscape
]


def _pop_kv(extra: list[str], key: str, default: str) -> tuple[str, list[str]]:
    """Extract ``--key value`` from extra. Returns (value, remaining_extra)."""
    if key in extra:
        i = extra.index(key)
        if i + 1 >= len(extra):
            sys.exit(f"missing value after {key}")
        value = extra[i + 1]
        return value, extra[:i] + extra[i + 2 :]
    return default, list(extra)


def cmd_dcw(extra):
    """Sample baseline trajectories per bucket, then train fusion head.

    Generates ``--n_images`` × ``--n_seeds`` baseline trajectories per
    bucket (default 35×3) into ``output/dcw/``, then trains the head on
    the pooled rows. Buckets only stratify the prompt pool — the trainer
    aggregates them.

    Other extra args pass through to every measure_bias invocation
    (--dit, --lora_weight, --pooled_text_proj '', --guidance_scale, etc.).
    """
    n_images, extra = _pop_kv(extra, "--n_images", "35")
    n_seeds, extra = _pop_kv(extra, "--n_seeds", "3")
    label, extra = _pop_kv(extra, "--label", "make-dcw")

    out_root = "output/dcw"
    for H, W in DCW_BUCKETS:
        bucket_label = f"{label}-{H}x{W}"
        print(
            f"\n=== DCW sample: bucket {H}x{W} ({n_images} imgs × {n_seeds} seeds) ==="
        )
        run(
            [
                sys.executable,
                "scripts/dcw/measure_bias.py",
                "--image_h",
                str(H),
                "--image_w",
                str(W),
                "--n_images",
                n_images,
                "--n_seeds",
                n_seeds,
                "--dump_per_sample_gaps",
                "--label",
                bucket_label,
                "--out_root",
                out_root,
                *extra,
            ]
        )

    print("\n=== DCW: training fusion head on pooled trajectories ===")
    run(
        [
            sys.executable,
            "scripts/dcw/train_fusion_head.py",
            "--label",
            label,
        ]
    )
    print(
        "\nDone. Run `make test-dcw-v4` to inference with the fresh artifact "
        "(auto-resolves the latest fusion_head.safetensors)."
    )


def cmd_dcw_train(extra):
    """Train-only on existing pool (no sampling, ~30s)."""
    run([sys.executable, "scripts/dcw/train_fusion_head.py", *extra])
