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

# (H, W) — buckets the trainer's data loader stratifies over.
# Top 5 buckets by post_image_dataset/lora/ count (aspect_id = list index).
# Note: the inference calibrator no longer keys off aspect (post-cleanup), so
# this list is only relevant for `make dcw` data collection.
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
    bucket (default 130×1) into ``output/dcw/``, then trains the head on
    the pooled rows. Buckets only stratify the prompt pool — the trainer
    aggregates them.

    Defaults reflect the 2026-05-05 findings: prompt breadth dominates
    seed multiplicity for r_α (single-seed labels carry only ~13% noise
    floor at production target_window 7:; seed-mean averaging is
    net-harmful at this data scale — see project_dcw_seed_variance_dominates).
    n_images=130 is capped at the rarest top-5 bucket so all buckets stay
    aspect-balanced (132 stems available for 1248×832). ``--shuffle_seed=0``
    deterministically randomizes selection across the cache's 14×
    headroom (2477 stems vs 175 previously sampled).

    Other extra args pass through to every measure_bias invocation
    (--dit, --lora_weight, --pooled_text_proj '', --guidance_scale, etc.).
    """
    n_images, extra = _pop_kv(extra, "--n_images", "130")
    n_seeds, extra = _pop_kv(extra, "--n_seeds", "1")
    shuffle_seed, extra = _pop_kv(extra, "--shuffle_seed", "0")
    label, extra = _pop_kv(extra, "--label", "make-dcw")

    out_root = "output/dcw"
    for H, W in DCW_BUCKETS:
        bucket_label = f"{label}-{H}x{W}"
        print(
            f"\n=== DCW sample: bucket {H}x{W} ({n_images} imgs × {n_seeds} seeds, "
            f"shuffle_seed={shuffle_seed}) ==="
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
                "--shuffle_seed",
                shuffle_seed,
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
