#!/usr/bin/env python3
"""Merge masks from multiple sources by taking the pixel-wise minimum (union of masked regions)."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_dirs", nargs="+", help="Input mask directories to merge")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged masks",
    )
    args = parser.parse_args()

    mask_dirs = [Path(d) for d in args.mask_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all mask filenames across all sources
    all_names: set[str] = set()
    for d in mask_dirs:
        if d.exists():
            all_names.update(p.name for p in d.glob("*_mask.png"))

    if not all_names:
        print("No masks found.")
        return

    merged = 0
    for name in tqdm(sorted(all_names), desc="Merging masks"):
        sources = [d / name for d in mask_dirs if (d / name).exists()]
        if len(sources) == 1:
            # Single source -- just copy
            arr = np.array(Image.open(sources[0]))
        else:
            # Pixel-wise minimum: lower alpha = more masking
            arr = np.array(Image.open(sources[0]))
            for src in sources[1:]:
                other = np.array(
                    Image.open(src).resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
                )
                arr = np.minimum(arr, other)

        Image.fromarray(arr, mode="L").save(output_dir / name)
        merged += 1

    print(f"Merged {merged} masks into {output_dir}/")


if __name__ == "__main__":
    main()
