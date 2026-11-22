#!/usr/bin/env python3
"""Fixed-pattern reads of a ×2 head on its training job's end sheets, against its parent.

The sheets are ``<run>/*<image>_full.png`` (left half the real 2× image, right half the
head, under a 24-px label row). The real image is read from the first ``--run``'s sheet
and every run's left half must equal it; the parent gives its right half only (a bench
sheet of ``x2_head.py --chain`` has the ResShift chain on the left). Per image, on the
mean of the three channels in 8-bit levels:

- **stamp**: the head's high-pass (minus a Gaussian blur at σ 3 px) folded at 32 px (one
  DiT token at ×2) and at 16 px, averaged over all tiles; the std of that mean tile. The
  same fold of the real image is the content floor.
- **seam**: in flat regions of the real image (smoothed gradient under 1 level), the
  residual head − real averaged per 8 × 8 cell; mean |difference| of horizontally adjacent
  cells across a 32-px token border / within a token.
- **phase** (GPU, ``--no_phase`` skips it): the frozen PE-Spatial-B16 tower's tokens after
  the discriminator's tapped blocks, on a 1024-px crop. d = mean token distance between
  the head's crop and the real one; the read is d(run) / d(parent) with the crop's origin
  on PE's 16-px patch grid (offsets (0, 0), (16, 16), (32, 32) from a token corner) and
  off it. A ratio higher off the grid than on it = the head matches the real image in the
  tower's features only at the patch phase it was trained on. The tower runs on a 64 × 64
  grid here (interpolated position embeddings), as in a ``pegan_crop = 1024`` run.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 300 bench/pixel_readout/experiments/x2_stamp_phase.py \
        --parent output/pixel_head/x2bll_1122_1629 --run x2llgf1024_1122_1919 --run x2llgf512_1122_2152 \
        --label x2sp_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home

HEADER = 24  # the sheets' label row, px
ON_GRID = [(0, 0), (16, 16), (32, 32)]
OFF_GRID = [(8, 8), (4, 12), (12, 4), (24, 8), (5, 27)]
LAYERS = (3, 7, 11)  # train_x2's pegan_layers default


def run_dir_of(name: str) -> Path:
    p = Path(resolve_under_home(name))
    return p if p.is_dir() else Path(resolve_under_home(f"output/pixel_head/{name}"))


def sheets(run: Path) -> dict[str, Path]:
    """image id → sheet, the id being the file name from ``test_`` / ``text_`` on."""
    out = {}
    for f in sorted(run.glob("*_full.png")):
        stem = f.name[: -len("_full.png")]
        i = max(stem.rfind("test_"), stem.rfind("text_"))
        out[stem[i:]] = f
    return out


def real_of(runs: dict[str, Path], im: str) -> np.ndarray:
    """The real 2× image of ``im``, checked equal across the runs' sheets."""
    reals = [halves(sheets(r)[im])[0] for r in runs.values()]
    for n, a in zip(runs, reals):
        assert np.abs(a - reals[0]).max() < 1, f"{n}: another left half on {im}"
    return reals[0]


def halves(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(real, head) as (H, W, 3) float32 in 8-bit levels."""
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)[HEADER:]
    w = a.shape[1] // 2
    return a[:, :w], a[:, w:]


def stamp(x: np.ndarray, p: int) -> float:
    hp = x - gaussian_filter(x, 3)
    h, w = (x.shape[0] // p) * p, (x.shape[1] // p) * p
    return float(hp[:h, :w].reshape(h // p, p, w // p, p).mean((0, 2)).std())


def seam(real: np.ndarray, head: np.ndarray, tok: int = 32, cell: int = 8):
    g = np.hypot(*np.gradient(gaussian_filter(real, 2)))
    h, w = (real.shape[0] // tok) * tok, (real.shape[1] // tok) * tok
    cells = lambda a: a[:h, :w].reshape(h // cell, cell, w // cell, cell)  # noqa: E731
    rc = cells(head - real).mean((1, 3))
    flat = cells(g).max((1, 3)) < 1
    d = np.abs(rc[:, 1:] - rc[:, :-1])
    ok = flat[:, 1:] & flat[:, :-1]
    border = np.arange(1, rc.shape[1]) % (tok // cell) == 0
    return float(d[:, border][ok[:, border]].mean()), float(
        d[:, ~border][ok[:, ~border]].mean()
    )


def phase_rows(runs: dict[str, Path], parent: Path, images: list[str], crop: int):
    import torch

    from library.models.pe import load_pe_spatial

    dev = "cuda"
    pe = load_pe_spatial(dev, dtype=torch.bfloat16)
    taps: dict[int, torch.Tensor] = {}
    for i in LAYERS:
        pe.transformer.resblocks[i].register_forward_hook(
            lambda _m, _a, o, i=i: taps.__setitem__(i, o)
        )

    def ten(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1).to(dev)

    @torch.no_grad()
    def feats(x: torch.Tensor) -> list[torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pe.forward_features(x[None], layer_idx=max(LAYERS) - pe.transformer.layers)
        return [taps[i][0, 1:].float() for i in LAYERS]

    dist = lambda a, b: [(x - y).norm(dim=-1).mean().item() for x, y in zip(a, b)]  # noqa: E731
    psheets = sheets(parent)
    rows = []
    for im in images:
        real, par = ten(real_of(runs, im)), ten(halves(psheets[im])[1])
        heads = {n: ten(halves(sheets(r)[im])[1]) for n, r in runs.items()}
        h, w = real.shape[-2:]
        if min(h, w) < crop + 64:
            continue
        # a token corner near the centre, with room for the offsets
        y0, x0 = ((h - crop - 64) // 64) * 32, ((w - crop - 64) // 64) * 32
        for dy, dx in ON_GRID + OFF_GRID:
            sl = (..., slice(y0 + dy, y0 + dy + crop), slice(x0 + dx, x0 + dx + crop))
            fr = feats(real[sl])
            dp = dist(feats(par[sl]), fr)
            for n, t in heads.items():
                dg = dist(feats(t[sl]), fr)
                rows.append(
                    {
                        "run": n,
                        "image": im,
                        "off": [dy, dx],
                        "on_grid": (dy, dx) in ON_GRID,
                        "ratio": [a / b for a, b in zip(dg, dp)],
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parent", required=True, help="the parent run's folder or name")
    ap.add_argument("--run", action="append", required=True, help="repeatable")
    ap.add_argument("--no_phase", action="store_true", help="sheet reads only (CPU)")
    ap.add_argument("--crop", type=int, default=1024, help="phase read's crop, px")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    parent = run_dir_of(args.parent)
    runs = {Path(r).name: run_dir_of(r) for r in args.run}
    images = sorted(
        set(sheets(parent)).intersection(*(sheets(r) for r in runs.values()))
    )
    assert images, "no sheet common to the parent and every run"

    table = []
    for im in images:
        real = real_of(runs, im).mean(-1)
        par = halves(sheets(parent)[im])[1].mean(-1)
        row = {"image": im, "real_stamp32": stamp(real, 32)}
        for n, hd in [("parent", par)] + [
            (n, halves(sheets(r)[im])[1].mean(-1)) for n, r in runs.items()
        ]:
            b, wi = seam(real, hd)
            row[n] = {
                "stamp32": stamp(hd, 32),
                "stamp16": stamp(hd, 16),
                "seam_border": b,
                "seam_within": wi,
            }
        table.append(row)
        print(
            f"{im:22s} real {row['real_stamp32']:.2f} | "
            + " | ".join(
                f"{n} stamp {v['stamp32']:.2f} / {v['stamp16']:.2f} "
                f"seam {v['seam_border']:.2f} / {v['seam_within']:.2f}"
                for n, v in row.items()
                if isinstance(v, dict)
            ),
            flush=True,
        )

    phase = []
    if not args.no_phase:
        phase = phase_rows(runs, parent, images, args.crop)
        print("phase: d(run) / d(parent) at the last tap, on the 16-px grid | off it")
        for n in runs:
            for im in images:
                on = [
                    r["ratio"][-1]
                    for r in phase
                    if (r["run"], r["image"], r["on_grid"]) == (n, im, True)
                ]
                off = [
                    r["ratio"][-1]
                    for r in phase
                    if (r["run"], r["image"], r["on_grid"]) == (n, im, False)
                ]
                if on:
                    print(
                        f"  {n:24s} {im:22s} {min(on):.3f} - {max(on):.3f} | "
                        f"{min(off):.3f} - {max(off):.3f}",
                        flush=True,
                    )

    run_dir = make_run_dir("pixel_readout", args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"sheets": table, "phase": phase},
        label=args.label,
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
