#!/usr/bin/env python3
"""Texture reads of ×1 or ×2 heads on a ``p3_generated.py`` run's sheets (CPU, read-only).

The sheets are ``sheet_<arm>_<k>_<prompt>_s<seed>.png`` in ``zoom_sheet``'s layout: a
24-px label row, the full images as equal columns (the VAE decode, then one column per
``--head_run``), and a zoom row. Per head column, on the mean of the three channels in
8-bit levels:

- **amp**: the std of image − Gaussian σ 1.5, as a share of the VAE column's.
- **stamp16** / **stamp8**: image − Gaussian σ 3, folded at 16 px (one DiT token at ×1)
  and at 8 px and averaged over all tiles; the std of that mean tile. A pattern repeated
  in every token raises it; the VAE column's value is the content floor.

``--x2`` reads a ``p3_generated.py --x2`` run: ``x2_<arm>_<k>_<prompt>_s<seed>_full.png``
in ``pair_sheet``'s layout (the label row, then ``VAE -> ResShift x2``, one column per
head and ``bicubic x2 of VAE``, no zoom row). The folds are at 32 px (one DiT token at
×2) and 16 px, and the reference column is the ResShift chain's.

Medians are given over the colour and the monochrome prompts (``_prompts.py``, a prompt
with ``monochrome``). ``--pairs A B`` also writes ``pairs_<A>_<B>/``: the two head
columns alone, with their label and zoom rows, one file per sheet (with ``--x2``, per
full sheet and per crop sheet).

Usage::

    .venv/bin/python -m bench.pixel_readout.sheet_texture \
        --run bench/pixel_readout/results/<p3 run> --pairs <head A> <head B> \
        --label tex_mmdd_hhmm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from bench._common import make_run_dir, write_result
from bench.pixel_readout._prompts import PROMPTS
from library.env import resolve_under_home

HEADER = 24  # zoom_sheet's label row, px
ZOOM = 768  # zoom_sheet's zoom row: a 384-px window doubled
MONO = {name for name, prompt, _ in PROMPTS if "monochrome" in prompt}


def stamp(x: np.ndarray, p: int) -> float:
    hp = x - gaussian_filter(x, 3)
    h, w = (x.shape[0] // p) * p, (x.shape[1] // p) * p
    return float(hp[:h, :w].reshape(h // p, p, w // p, p).mean((0, 2)).std())


def amp(x: np.ndarray) -> float:
    return float((x - gaussian_filter(x, 1.5)).std())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="a p3_generated.py result folder")
    ap.add_argument("--pairs", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--x2", action="store_true", help="a p3_generated.py --x2 run")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    src = Path(resolve_under_home(args.run))
    heads = [  # a run's head file is named by its run folder, as on the sheets
        Path(r).parent.name if Path(r).suffix else Path(r).name
        for r in json.loads((src / "result.json").read_text())["args"]["head_run"]
    ]
    if args.x2:
        cols = ["VAE -> ResShift x2", *heads, "bicubic x2 of VAE"]
        files, zoom, periods = sorted(src.glob("x2_*_full.png")), 0, (32, 16)
    else:
        cols = ["VAE decode", *heads]
        files, zoom, periods = sorted(src.glob("sheet_*.png")), ZOOM, (16, 8)
    keys = ("amp", *(f"stamp{q}" for q in periods))
    assert files, f"no sheet in {src}"
    run_dir = make_run_dir("pixel_readout", args.label)
    pair_dir = None
    if args.pairs:
        assert set(args.pairs) <= set(heads), (args.pairs, heads)
        pair_dir = run_dir / f"pairs_{args.pairs[0]}_{args.pairs[1]}"
        pair_dir.mkdir()

    def pair(f: Path) -> None:
        """The two ``--pairs`` columns of one sheet, side by side."""
        im = Image.open(f).convert("RGB")
        w, h = im.size
        cw = w // len(cols)
        out = Image.new("RGB", (2 * cw, h), "white")
        for i, name in enumerate(args.pairs):
            j = cols.index(name)
            out.paste(im.crop((j * cw, 0, (j + 1) * cw, h)), (i * cw, 0))
        out.save(pair_dir / f"pair_{f.name.split('_', 1)[1]}")

    rows = []
    for f in files:
        im = Image.open(f).convert("RGB")
        w, h = im.size
        cw, ih = w // len(cols), h - HEADER - zoom
        a = np.asarray(im, np.float32).mean(-1)
        full = [
            a[HEADER : HEADER + ih, i * cw : (i + 1) * cw] for i in range(len(cols))
        ]
        prompt = next(n for n, _, _ in PROMPTS if f"_{n}_s" in f.name)
        ref = amp(full[0])
        row = {"sheet": f.name, "mono": prompt in MONO}
        for name, x in zip(cols, full):
            row[name] = {"amp": amp(x) / ref} | {
                f"stamp{q}": stamp(x, q) for q in periods
            }
        rows.append(row)
        if pair_dir is not None:
            pair(f)
            if args.x2:
                pair(f.with_name(f.name.replace("_full.png", "_crop.png")))

    summary = {}
    for kind, mono in (("colour", False), ("mono", True)):
        sel = [r for r in rows if r["mono"] is mono]
        summary[kind] = {"n": len(sel)} | {
            name: {k: float(np.median([r[name][k] for r in sel])) for k in keys}
            for name in cols
        }
    for k in keys:
        print(
            f"{k}, median colour {summary['colour']['n']} / mono {summary['mono']['n']}"
        )
        for name in cols:
            c, m = summary["colour"][name][k], summary["mono"][name][k]
            print(f"  {name:28s} {c:.3f} / {m:.3f}")
    for i, a_ in enumerate(heads):  # paired counts, every two heads
        for b_ in heads[i + 1 :]:
            n = {k: sum(r[b_][k] > r[a_][k] for r in rows) for k in keys}
            print(f"{b_} above {a_} on (of {len(rows)}): {n}")
            summary[f"{b_}_above_{a_}"] = n | {"of": len(rows)}

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"summary": summary, "per_sheet": rows},
        label=args.label,
        artifacts=[pair_dir.name] if pair_dir is not None else [],
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
