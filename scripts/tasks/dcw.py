"""DCW v4 calibrator: collect calibration data + train fusion head.

`make dcw` runs `dcw_measure_bias.py --dump_per_sample_gaps` once per
aspect bucket (top-5 by sample count: 832×1248, 896×1152, 768×1344,
1152×896, 1248×832) at the production env (CFG=4, mod_w=3.0), then
chains `dcw_train_fusion_head.py`. End artifact is
`output/dcw/<timestamp>-v4-fusion-head-make-dcw/fusion_head.safetensors`,
which `make test-dcw-v4` auto-resolves.

By default the per-aspect bucket prior is zero (lam_scalar=0, S_pop=0
for every bucket). Pass ``--with_a2`` to also run the λ=0.01 sweep per
bucket and have the trainer auto-discover those sweep dirs to populate
the prior. Per docs/proposal …§A2 the bucket prior was empirically
cosmetic on Anima (~0% gap reduction), so the head-only path is the
preferred default; the flag exists for ablation runs.

`make dcw-train` skips the sampling phase and trains on the existing pool.

See `docs/proposal/dcw-learnable-calibrator-v4.md` §I1.
"""

from __future__ import annotations

import sys

from ._common import ROOT, run

# (H, W) — must match library/inference/dcw_v4.py:ASPECT_TABLE.
# Top 5 buckets by post_image_dataset/lora/ count (aspect_id = list index).
DCW_BUCKETS = [
    (832, 1248),   # HD portrait — most common
    (896, 1152),   # 3:4 portrait
    (768, 1344),   # tall portrait
    (1152, 896),   # 3:4 landscape
    (1248, 832),   # HD landscape
]


def _pop_kv(extra: list[str], key: str, default: str) -> tuple[str, list[str]]:
    """Extract ``--key value`` from extra. Returns (value, remaining_extra)."""
    if key in extra:
        i = extra.index(key)
        if i + 1 >= len(extra):
            sys.exit(f"missing value after {key}")
        value = extra[i + 1]
        return value, extra[:i] + extra[i + 2:]
    return default, list(extra)


def _pop_flag(extra: list[str], flag: str) -> tuple[bool, list[str]]:
    """Extract a boolean ``--flag`` from extra. Returns (present, remaining)."""
    if flag in extra:
        return True, [a for a in extra if a != flag]
    return False, list(extra)


def cmd_dcw(extra):
    """Sample A1 baseline per bucket, then train fusion head.

    Default path: A1 only (``--n_images`` × ``--n_seeds`` baseline
    trajectories per bucket; default 10×3). The trainer leaves the
    per-aspect bucket prior at zero — the head carries all correction.

    Pass ``--with_a2`` to also run a λ=0.01 sweep per bucket
    (``--a2_n_images`` × 1 seed; default 8) and propagate ``--with_a2``
    to the trainer so it auto-discovers those sweep dirs and populates
    S_pop / λ_scalar from them. Empirically cosmetic on Anima — present
    for ablations.

    Other extra args pass through to every measure_bias invocation
    (--dit, --lora_weight, --pooled_text_proj '', --guidance_scale, etc.).
    """
    with_a2, extra = _pop_flag(extra, "--with_a2")
    n_images, extra = _pop_kv(extra, "--n_images", "10")
    n_seeds, extra = _pop_kv(extra, "--n_seeds", "3")
    a2_n_images, extra = _pop_kv(extra, "--a2_n_images", "8")
    label, extra = _pop_kv(extra, "--label", "make-dcw")

    out_root = "output/dcw"
    for H, W in DCW_BUCKETS:
        bucket_label = f"{label}-{H}x{W}"
        print(f"\n=== DCW A1: bucket {H}x{W} ({n_images} imgs × {n_seeds} seeds) ===")
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

        if with_a2:
            a2_label = f"{label}-A2-{H}x{W}"
            print(f"\n=== DCW A2: bucket {H}x{W} ({a2_n_images} imgs × 1 seed, λ=0.01) ===")
            run([
                sys.executable, "scripts/dcw_measure_bias.py",
                "--image_h", str(H),
                "--image_w", str(W),
                "--n_images", a2_n_images,
                "--n_seeds", "1",
                "--dcw_sweep",
                "--dcw_scalers", "0.01",
                "--label", a2_label,
                "--out_root", out_root,
                *extra,
            ])

    print("\n=== DCW: training fusion head on pooled trajectories ===")
    run([
        sys.executable, "scripts/dcw_train_fusion_head.py",
        "--label", label,
        *(["--with_a2"] if with_a2 else []),
    ])
    print(
        "\nDone. Run `make test-dcw-v4` to inference with the fresh artifact "
        "(auto-resolves the latest fusion_head.safetensors)."
    )


def cmd_dcw_train(extra):
    """Train-only on existing pool (no sampling, ~30s).

    Default leaves the per-aspect bucket prior at zero. Pass ``--with_a2``
    to auto-discover A2 sweep dirs under ``--results_root`` (default scans
    output/dcw/, post_image_dataset/dcw/, bench/dcw/results/).
    """
    run([sys.executable, "scripts/dcw_train_fusion_head.py", *extra])
