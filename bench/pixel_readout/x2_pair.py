#!/usr/bin/env python3
"""Two ×2 runs side by side: one column of each run's sheets, per image.

``x2_head.py`` and ``p3_generated.py --x2`` write labelled multi-column sheets per run
(``chain | ×2 head``, ``VAE → ResShift ×2 | ×2 head | bicubic``). Two runs read on the
same held-out set or the same renders share file names and crop windows, so the same
column of each makes a two-column sheet ``A | B``. CPU only.

Usage::

    python -m bench.pixel_readout.x2_pair x2ctl_1120_0848 x2dmd_1120_0848 \
        --out ~/x2pair --glob '*_crop.png'
    # two head columns of one p3_generated.py --x2 run (teacher | A | B | bicubic)
    python -m bench.pixel_readout.x2_pair x2gen_mmdd_hhmm x2gen_mmdd_hhmm --cols 4 \
        --col 1 --col_b 2 --names control arm --out ~/x2pair_gen
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from bench.pixel_readout._shared import resolve_run

HEADER = 24  # the label strip of pair_sheet / p3_generated sheets


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--out", required=True)
    p.add_argument("--glob", default="*_crop.png")
    p.add_argument("--cols", type=int, default=2, help="columns in the source sheets")
    p.add_argument(
        "--col", type=int, default=1, help="column taken from each (0-based)"
    )
    p.add_argument("--col_b", type=int, default=None, help="run_b's column (--col)")
    p.add_argument("--names", nargs=2, default=None, help="column labels (the run ids)")
    args = p.parse_args()
    a, b = resolve_run(args.run_a), resolve_run(args.run_b)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fa in sorted(a.glob(args.glob)):
        fb = b / fa.name
        if not fb.exists():
            continue
        ims = [Image.open(f).convert("RGB") for f in (fa, fb)]
        assert ims[0].size == ims[1].size, (fa.name, ims[0].size, ims[1].size)
        w = ims[0].width // args.cols
        h = ims[0].height - HEADER
        canvas = Image.new("RGB", (2 * w, HEADER + h), "white")
        d = ImageDraw.Draw(canvas)
        names = args.names or (args.run_a, args.run_b)
        cols = (args.col, args.col if args.col_b is None else args.col_b)
        for i, (im, name, c) in enumerate(zip(ims, names, cols)):
            box = (c * w, HEADER, (c + 1) * w, HEADER + h)
            canvas.paste(im.crop(box), (i * w, HEADER))
            d.text((i * w + 6, 4), name, fill="black")
        canvas.save(out / fa.name)
        n += 1
    print(f"{n} sheets -> {out}")


if __name__ == "__main__":
    main()
