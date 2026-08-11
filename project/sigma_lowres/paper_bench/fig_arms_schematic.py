"""paper_v2 theory-section schematic: the three arms of Eq. (branches).

One concrete image walks the realizable lattice path. src (native content,
fine graph) and dem (demoted content, coarse graph) sit on the demotion
path — a straight dashed arrow, the thing training actually does; the
repromote arm rp hangs *below* that path as its hidden waypoint: the same
demoted content carried back to the native grid. src -> rp is the data
intervention at fixed graph; rp -> dem is the graph intervention at fixed
data; the two compose exactly to demotion.

Figure assembly only: the content is a crop of the model's own render
(the fig:kaaiyuki grid; prompt in app:armsprompt). No measured quantity
appears. For visibility the downscale is drawn at ratio 1/3 and the
repromote upscale nearest-neighbor (the shipped routes are 896/1024 and
768/1024, and the instrument's upscale is an ordinary resampler).

Usage:
    uv run python project/sigma_lowres/paper_bench/fig_arms_schematic.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

HERE = Path(__file__).resolve().parent  # project/sigma_lowres/paper_bench
SIGMA = HERE.parent  # project/sigma_lowres
RUNS = HERE / "runs"
FIGS = SIGMA / "paper_v2" / "figs"

GRID_JPG = FIGS / "kaai_yuki_arms_grid.jpg"
RATIO = 1 / 3  # drawn demotion ratio (for visibility; schematic only)

C_B, C_C, C_NET = "C0", "C3", "0.15"


# subplot frame-line coordinates of the fig:kaaiyuki grid (measured once:
# dark rows/cols of the jpg); crops exclude the grid's own axis labels
COL_EDGES = (82, 492, 906, 1320, 1734)
ROW_EDGES = (64, 474, 888, 1302)


def cell_crop(col: int = 0, row: int = 0, pad: int = 2) -> Image.Image:
    im = Image.open(GRID_JPG)
    cell = im.crop((COL_EDGES[col] + pad, ROW_EDGES[row] + pad,
                    COL_EDGES[col + 1] - pad, ROW_EDGES[row + 1] - pad))
    side = min(cell.size)
    return cell.crop((0, 0, side, side))


def token_grid(ax, n: int, color: str = "0.35") -> None:
    for i in range(n + 1):
        t = i / n
        ax.plot([t, t], [0, 1], color=color, lw=0.7)
        ax.plot([0, 1], [t, t], color=color, lw=0.7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_arrow(fig, p, q, color, lw=2.0, ls="-", mut=14, rad=0.0):
    fig.add_artist(FancyArrowPatch(
        p, q, transform=fig.transFigure, arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}", mutation_scale=mut,
        color=color, lw=lw, linestyle=ls, zorder=5))


def panel(fig, x, y, w, h, im, title):
    ax = fig.add_axes((x, y, w, h))
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.3")
    if title:
        ax.set_title(title, fontsize=8.6, pad=4)
    return ax


def main() -> None:
    src = cell_crop()
    side = src.size[0]
    small = src.resize((int(side * RATIO),) * 2, Image.LANCZOS)
    rp = small.resize((side,) * 2, Image.NEAREST)

    fig = plt.figure(figsize=(8.6, 4.8))

    img_w, img_h = 2.1 / 8.6, 2.1 / 4.8
    y_path = 0.72  # the demotion path's midline

    # src — left, on the path
    x_src, y_src = 0.035, y_path - img_h / 2
    panel(fig, x_src, y_src, img_w, img_h, src, None)
    xc = x_src + img_w / 2
    fig.text(xc, y_src - 0.030, r"$\bar g_{\mathrm{src}}$",
             ha="center", va="top", fontsize=15)

    # dem — right, on the path, at its own (small) grid
    dem_w, dem_h = img_w * RATIO, img_h * RATIO
    x_dem, y_dem = 0.825, y_path - dem_h / 2
    panel(fig, x_dem, y_dem, dem_w, dem_h, small, None)
    xc = x_dem + dem_w / 2
    fig.text(xc, y_dem - 0.030, r"$\bar g_{\mathrm{dem}}$",
             ha="center", va="top", fontsize=15)
    fig.text(xc, y_dem - 0.115, "demoted content,\ncoarse graph",
             ha="center", va="top", fontsize=9.5, color="0.45")

    # rp — the hidden waypoint, below the path
    x_rp, y_rp = 0.420, 0.020
    panel(fig, x_rp, y_rp, img_w, img_h, rp, None)
    xm_rp = x_rp + img_w / 2
    fig.text(xm_rp, y_rp + img_h + 0.075,
             "demoted content, fine graph",
             ha="center", va="bottom", fontsize=9.5, color="0.45")
    fig.text(xm_rp, y_rp + img_h + 0.012, r"$\bar g_{\mathrm{rp}}$",
             ha="center", va="bottom", fontsize=15)

    # the demotion path: src -> dem, straight across the top (solid)
    fig_arrow(fig, (x_src + img_w + 0.010, y_path),
              (x_dem - 0.010, y_path), C_NET, lw=2.2)
    fig.text((x_src + img_w + x_dem) / 2, y_path + 0.034,
             "demoted training", ha="center", va="bottom",
             fontsize=13.5, color=C_NET)

    # the hidden waypoint's detour: src -> rp -> dem (dashed)
    fig_arrow(fig, (x_src + img_w * 0.55, y_src - 0.010),
              (x_rp - 0.012, y_rp + img_h * 0.5), C_B, rad=0.25,
              ls=(0, (5, 3)))
    fig.text(0.290, 0.438, "data intervention", ha="center",
             fontsize=13, color=C_B)
    fig.text(0.290, 0.372,
             r"$\bar g_{\mathrm{rp}} - \bar g_{\mathrm{src}}$",
             ha="center", fontsize=12, color=C_B)

    fig_arrow(fig, (x_rp + img_w + 0.006, y_rp + img_h * 0.97),
              (x_dem - 0.010, y_path - dem_h * 0.35), C_C, rad=0.05,
              ls=(0, (5, 3)))
    fig.text(0.762, 0.408, "graph intervention", ha="center",
             fontsize=13, color=C_C)
    fig.text(0.762, 0.342,
             r"$\bar g_{\mathrm{dem}} - \bar g_{\mathrm{rp}}$",
             ha="center", fontsize=12, color=C_C)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-fig-arms-schematic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "arms_schematic.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    shutil.copy2(out, FIGS / "arms_schematic.png")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/fig_arms_schematic.py",
        label="fig-arms-schematic",
        timestamp_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(render_grid=str(GRID_JPG)),
        metrics={},
        caveats=[
            "schematic only: no measured quantity appears",
            "content is a crop of the model's own render (fig:kaaiyuki "
            "grid; prompt in app:armsprompt)",
            "downscale drawn at ratio 1/3 and the repromote upscale "
            "nearest-neighbor, both for visibility; the shipped routes "
            "are 896/1024 and 768/1024 and the instrument's upscale is "
            "an ordinary resampler",
        ],
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=2))
    print(f"wrote {out} (+ figs/ copy)")


if __name__ == "__main__":
    main()
