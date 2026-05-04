#!/usr/bin/env python
"""Sweep ``measure_bias.py`` across constant-token buckets sequentially.

Each bucket runs as a fresh ``measure_bias.py`` subprocess so:

- ``torch.compile`` warms up at a single shape per process (no
  cross-bucket recompilation noise in one dynamo cache).
- DiT / text-encoder load happens per bucket but is small relative to
  200-prompt inference; in exchange there is no state leakage.
- Each bucket produces its own
  ``bench/dcw/results/<ts>-<label>-<W>x<H>/`` directory, indexable by
  the standard ``result.json`` envelope.

Why this exists
---------------
The CFG=4 sign-flip finding (``project_dcw_cfg_aspect_signflip``)
showed the LL gap depends on aspect ratio, so any DCW calibration that
generalizes past the square bucket needs data across multiple buckets.
This wrapper makes that data collection one command.

Examples
--------
    # 3-bucket diagnostic (square + 1.5-ratio mirror pair).
    uv run python bench/dcw/sweep_buckets.py \\
        --buckets minimal --label v2-bulk \\
        --n_images 200 --n_seeds 1 \\
        --guidance_scale 4.0 --dump_per_sample_gaps --compile

    # Aspect-ratio sweep for calibrator data (1.0 → 2.0, 7 buckets).
    uv run python bench/dcw/sweep_buckets.py \\
        --buckets aspect4 --label v2-aspect \\
        --n_images 200 --n_seeds 1 \\
        --guidance_scale 4.0 --dump_per_sample_gaps --compile

    # Custom bucket list — any constant-token resolutions.
    uv run python bench/dcw/sweep_buckets.py \\
        --buckets 1024x1024,896x1152,1152x896 --label aspect-pilot \\
        --n_images 100 --compile

    # Dry run: print planned commands without launching.
    uv run python bench/dcw/sweep_buckets.py \\
        --buckets aspect4 --label v2-aspect --dry_run \\
        --n_images 200 --compile

Notes
-----
- Args not consumed by this wrapper are forwarded verbatim to
  ``measure_bias.py`` (no ``--`` separator needed). ``--image_h``,
  ``--image_w``, and ``--label`` are managed per-bucket and rejected
  if forwarded.
- Default behavior on subprocess failure is to abort the sweep. Pass
  ``--continue_on_error`` to skip the failed bucket and keep going
  (useful if some buckets have no cached samples).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS

# Bucket presets. ``CONSTANT_TOKEN_BUCKETS`` is the canonical 17-entry
# list from ``library/datasets/buckets.py`` — every entry has
# (W/16)*(H/16) ≤ 4096 tokens and gets padded to exactly 4096 at runtime.
PRESETS: dict[str, list[tuple[int, int]]] = {
    # Square + 1.5 mirror pair. Cheapest sign-flip / sanity check.
    "minimal": [(1024, 1024), (832, 1248), (1248, 832)],
    # 1.0 → 2.0 with mirrored pairs. Right shape for calibrator
    # training data: 4 distinct aspect ratios.
    "aspect4": [
        (1024, 1024),                  # 1.00
        (1152, 896), (896, 1152),      # 1.29
        (1248, 832), (832, 1248),      # 1.50
        (768, 1344),      # 2.00
    ],
    # All 17 constant-token buckets. Overkill for most analyses.
    "all": list(CONSTANT_TOKEN_BUCKETS),
}

_FORBIDDEN_PASSTHROUGH = {"--image_h", "--image_w", "--label"}


def parse_buckets(spec: str) -> list[tuple[int, int]]:
    if spec in PRESETS:
        return PRESETS[spec]
    out: list[tuple[int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "x" not in tok:
            raise SystemExit(
                f"bucket spec '{tok}' must be 'WxH' (e.g. 1024x1024) or one "
                f"of the presets {list(PRESETS)}"
            )
        w_s, h_s = tok.split("x", 1)
        try:
            out.append((int(w_s), int(h_s)))
        except ValueError as e:
            raise SystemExit(f"invalid bucket '{tok}': {e}") from None
    if not out:
        raise SystemExit("no buckets parsed from --buckets")
    return out


def validate_passthrough(extra: list[str]) -> None:
    for a in extra:
        if a in _FORBIDDEN_PASSTHROUGH:
            raise SystemExit(
                f"{a} is set per-bucket by the sweep wrapper; remove it from "
                "the forwarded args. Use --buckets to pick resolutions and "
                "--label as a per-bucket prefix."
            )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--buckets",
        type=str,
        default="minimal",
        help=f"Preset name (one of {list(PRESETS)}) or comma-separated "
        "'WxH' list (e.g. '1024x1024,832x1248'). Default: minimal.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="sweep",
        help="Base label; per-bucket label is '<label>-<W>x<H>'. Default: sweep.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned subprocess commands without executing.",
    )
    p.add_argument(
        "--continue_on_error",
        action="store_true",
        help="If a bucket fails, log and continue. Default: abort the sweep.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=None,
        help="Override the python invocation. Default: 'uv run python'.",
    )
    return p.parse_known_args()


def main() -> int:
    args, extra = parse_args()
    buckets = parse_buckets(args.buckets)
    validate_passthrough(extra)

    script = ROOT / "bench" / "dcw" / "measure_bias.py"
    if not script.exists():
        raise SystemExit(f"measure_bias.py not found at {script}")

    py_cmd = shlex.split(args.python) if args.python else ["uv", "run", "python"]

    print(
        f"sweep: {len(buckets)} bucket(s) "
        f"{[f'{w}x{h}' for w, h in buckets]} label_base={args.label}"
    )
    if extra:
        print(f"forwarded args: {' '.join(shlex.quote(a) for a in extra)}")
    print()

    successes: list[tuple[tuple[int, int], float]] = []
    failures: list[tuple[tuple[int, int], int]] = []
    t0 = time.time()
    for i, (w, h) in enumerate(buckets, 1):
        label = f"{args.label}-{w}x{h}"
        cmd = py_cmd + [
            str(script),
            "--image_w", str(w),
            "--image_h", str(h),
            "--label", label,
        ] + extra

        print(f"[{i}/{len(buckets)}] {w}x{h} label={label}")
        print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
        if args.dry_run:
            print()
            continue

        t_bucket = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - t_bucket

        if proc.returncode != 0:
            print(f"  [err] exit {proc.returncode} ({elapsed:.0f}s)")
            failures.append(((w, h), proc.returncode))
            if not args.continue_on_error:
                print("\naborted — pass --continue_on_error to skip and continue.")
                return proc.returncode
        else:
            print(f"  [ok] {elapsed:.0f}s")
            successes.append(((w, h), elapsed))
        print()

    total = time.time() - t0
    print(
        f"sweep done in {total:.0f}s — "
        f"{len(successes)} ok, {len(failures)} failed."
    )
    if successes and not args.dry_run:
        per_bucket = ", ".join(f"{w}x{h}:{e:.0f}s" for (w, h), e in successes)
        print(f"per-bucket wall: {per_bucket}")
    if failures:
        print("failed buckets:")
        for (w, h), code in failures:
            print(f"  {w}x{h}: exit {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
