#!/usr/bin/env python3
"""Fixed-tile read of ×2 heads on a ``p3_generated.py --x2`` render folder (CPU).

The sheets are ``x2_base_*_full.png``: columns ``VAE → ResShift ×2 | head … | bicubic ×2
of VAE`` under a 24-px label row. Per column, the mean of the three channels is high-passed
(minus a Gaussian blur at σ 3 px) and cut into the 32-px tiles of the DiT token grid.

- **textured** tile: the chain's high-pass std ≥ 8 levels and the head's ≥ 6.
- **class** of a textured tile, by the cosine between the head's tile and the bicubic
  one: ``departed`` under 0.3, ``half`` 0.3 – 0.7, ``agreeing`` 0.7 and over.
- **dictionary share**: spherical k-means with 16 centroids fitted on the seed-0 renders'
  unit-norm departed tiles; the share of the seed-1 tiles' energy on the nearest
  centroid, and the share of tiles on the most used 1 / 2 / 4 centroids. The chain's and
  bicubic's tiles at the same places are the content floor. A head that writes one fixed
  tile reads near 1; halftone at its own pitch and phase reads like the floor.
- **mean tile**: the high-pass tile averaged over every tile of every render; its peak
  and where it sits (a speck at a fixed place inside the token).
- **edge ratio**: mean |first difference| across the token borders over within, per class.
- **flat borders**: on flat tokens (smoothed gradient of the bicubic column under 1 level
  over the whole token, token mean ≥ 40), the mean |horizontal first difference| inside
  the 8-px output patches, and its ratio across the patch borders (x = 7, 15, 23) and
  across the token border (x = 31). 1 = no box grid.

Zoom sheets ``zoom_<render>.png``: the 5 × 5-token window with the most departed tiles of
the last head, columns chain | heads, ×3 nearest.

Usage::

    .venv/bin/python bench/pixel_readout/experiments/tile_dict.py \
        bench/pixel_readout/results/<p3 --x2 folder> --label tile_mmdd_hhmm \
        --zoom 13_manga_page_s1 14_manga_action_s1 03_castle_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home

HEADER = 24  # the sheets' label row, px
TOK = 32  # one DiT token at ×2, px
K = 16


def grid(a: np.ndarray) -> np.ndarray:
    """(H, W) → (gh, gw, 32, 32) token tiles."""
    h, w = (a.shape[0] // TOK) * TOK, (a.shape[1] // TOK) * TOK
    return a[:h, :w].reshape(h // TOK, TOK, w // TOK, TOK).transpose(0, 2, 1, 3)


def unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(-1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def kmeans(x: torch.Tensor, k: int, it: int = 25) -> torch.Tensor:
    c = x[torch.randperm(len(x))[:k]].clone()
    for _ in range(it):
        a = (x @ c.T).argmax(1)
        c = torch.zeros_like(c).index_add_(0, a, x)
        dead = c.norm(dim=1) < 1e-6
        c[dead] = x[torch.randperm(len(x))[: int(dead.sum())]]
        c = c / c.norm(dim=1, keepdim=True)
    return c


def share(x0: torch.Tensor, x1: torch.Tensor) -> tuple[float, list[float]]:
    """Energy share of ``x1`` on the nearest of K centroids fitted on ``x0``, and the
    tile share of the most used 1 / 2 / 4 centroids."""
    if len(x0) < 4 * K or len(x1) < 4 * K:
        return float("nan"), [float("nan")] * 3
    a = x1 @ kmeans(x0, K).T
    use = torch.bincount(a.argmax(1), minlength=K).float() / len(x1)
    use = use.sort(descending=True).values
    return float((a.max(1).values.clamp_min(0) ** 2).mean()), [
        float(use[:n].sum()) for n in (1, 2, 4)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="a p3_generated.py --x2 result folder")
    ap.add_argument("--zoom", nargs="*", default=[], help="render names to zoom")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)

    d = Path(resolve_under_home(args.folder))
    heads = [
        Path(p).parent.name if Path(p).suffix else Path(p).name
        for p in json.loads((d / "result.json").read_text())["args"]["head_run"]
    ]
    names = ["chain", *heads, "bicubic"]
    recs = []
    for p in sorted(d.glob("x2_base_*_full.png")):
        rgb = np.asarray(Image.open(p).convert("RGB"), np.float32)[HEADER:]
        cw = rgb.shape[1] // len(names)
        cols = [rgb[:, i * cw : (i + 1) * cw].mean(2) for i in range(len(names))]
        hp = {n: grid(c - gaussian_filter(c, 3)) for n, c in zip(names, cols)}
        gh, gw = hp["chain"].shape[:2]
        flat = {n: v.reshape(gh, gw, -1) for n, v in hp.items()}
        tex = flat["chain"].std(-1) >= 8
        cls = {}
        for h in heads:
            cs = (unit(flat[h]) * unit(flat["bicubic"])).sum(-1)
            ok = tex & (flat[h].std(-1) >= 6)
            cls[h] = np.where(
                ok & (cs < 0.3), 0, np.where(ok & (cs >= 0.7), 2, np.where(ok, 1, -1))
            )
        g = np.hypot(*np.gradient(gaussian_filter(cols[-1], 2)))
        lit = (grid(g).max((2, 3)) < 1.0) & (grid(cols[-1]).mean((2, 3)) >= 40)
        dx = {
            n: grid(np.abs(np.diff(c, axis=1, append=c[:, -1:])))[lit].sum((0, 1))
            for n, c in zip(names, cols)
        }
        recs.append(
            {
                "flat": int(lit.sum()),
                "dx": dx,
                "name": p.name[len("x2_base_") : -len("_full.png")],
                "seed": int(p.name.split("_s")[-1][0]),
                "hp": flat,
                "raw": dict(zip(names, map(grid, cols))),
                "cls": cls,
                "rgb": rgb,
                "cw": cw,
            }
        )
    assert recs, f"no x2_base_*_full.png in {d}"

    def tiles(col: str, head: str, c: int, seed: int) -> torch.Tensor:
        return torch.from_numpy(
            np.concatenate(
                [
                    unit(r["hp"][col][r["cls"][head] == c])
                    for r in recs
                    if r["seed"] == seed
                ]
            )
        )

    def edge_ratio(col: str, head: str, c: int) -> float:
        bd, wi = [], []
        for r in recs:
            t, m = r["raw"][col], r["cls"][head] == c
            wi += [np.abs(np.diff(t[m], axis=-1)).ravel()]
            wi += [np.abs(np.diff(t[m], axis=-2)).ravel()]
            bd += [
                np.abs(t[:, 1:, :, 0] - t[:, :-1, :, -1])[m[:, :-1] & m[:, 1:]].ravel()
            ]
            bd += [np.abs(t[1:, :, 0, :] - t[:-1, :, -1, :])[m[:-1] & m[1:]].ravel()]
        return float(np.concatenate(bd).mean() / np.concatenate(wi).mean())

    out: dict = {"renders": len(recs), "heads": {}, "mean_tile": {}}
    for n in names:
        m = np.mean([r["hp"][n].mean((0, 1)) for r in recs], 0).reshape(TOK, TOK)
        y, x = np.unravel_index(np.abs(m).argmax(), m.shape)
        out["mean_tile"][n] = {"peak": float(m[y, x]), "at": [int(y), int(x)]}
    n_flat = sum(r["flat"] for r in recs)
    out["flat_tokens"] = n_flat
    out["flat_borders"] = {}
    print(f"flat tokens {n_flat}: inner |dx| (levels), patch borders, token border")
    for n in names:
        prof = np.sum([r["dx"][n] for r in recs], 0) / (n_flat * TOK)
        inner = float(np.delete(prof, [7, 15, 23, 31]).mean())
        row = {
            "inner": inner,
            "patch": float(prof[[7, 15, 23]].mean() / inner),
            "token": float(prof[31] / inner),
        }
        out["flat_borders"][n] = row
        print(f"  {n:22s} {inner:.2f}  x{row['patch']:.2f}  x{row['token']:.2f}")
    print("mean high-pass token tile, peak (levels) at (y, x):")
    for n, v in out["mean_tile"].items():
        print(f"  {n:22s} {v['peak']:+.2f} at {tuple(v['at'])}")
    for h in heads:
        n_tex = sum(int((r["cls"][h] >= 0).sum()) for r in recs)
        frac = [
            sum(int((r["cls"][h] == c).sum()) for r in recs) / n_tex for c in range(3)
        ]
        row = {
            "textured": n_tex,
            "departed": frac[0],
            "half": frac[1],
            "agreeing": frac[2],
        }
        for col in (h, "chain", "bicubic"):
            s, use = share(tiles(col, h, 0, 0), tiles(col, h, 0, 1))
            row[f"share_{'head' if col == h else col}"] = s
            if col == h:
                row["top_use"] = use
        row["share_head_agreeing"] = share(tiles(h, h, 2, 0), tiles(h, h, 2, 1))[0]
        row["edge_departed"] = edge_ratio(h, h, 0)
        row["edge_agreeing"] = edge_ratio(h, h, 2)
        row["per_render"] = {
            r["name"]: float(
                (r["cls"][h] == 0).sum() / max((r["cls"][h] >= 0).sum(), 1)
            )
            for r in recs
        }
        out["heads"][h] = row
        top = sorted(row["per_render"].items(), key=lambda kv: -kv[1])[:5]
        print(
            f"{h}: textured {n_tex}; departed {frac[0]:.3f}, half {frac[1]:.3f}\n"
            f"  dictionary share on departed: head {row['share_head']:.3f} "
            f"(top 1 / 2 / 4 use {' / '.join(f'{u:.2f}' for u in row['top_use'])}), "
            f"chain {row['share_chain']:.3f}, bicubic {row['share_bicubic']:.3f}; "
            f"head on agreeing {row['share_head_agreeing']:.3f}\n"
            f"  edge ratio departed {row['edge_departed']:.3f}, agreeing "
            f"{row['edge_agreeing']:.3f}\n"
            f"  departed per render: {', '.join(f'{n} {v:.2f}' for n, v in top)}",
            flush=True,
        )

    run_dir = make_run_dir("pixel_readout", args.label)
    arts = []
    k, win = 3, 5
    w = win * TOK
    show = names[:-1]
    for want in args.zoom:
        r = next(r for r in recs if r["name"] == want)
        m = (r["cls"][heads[-1]] == 0).astype(np.float32)
        best, by, bx = -1.0, 0, 0
        for y in range(m.shape[0] - win + 1):
            for x in range(m.shape[1] - win + 1):
                s = m[y : y + win, x : x + win].sum()
                if s > best:
                    best, by, bx = s, y, x
        sheet = Image.new("RGB", (len(show) * (w * k + 6) - 6, w * k + 22), "white")
        dr = ImageDraw.Draw(sheet)
        for j, n in enumerate(show):
            i = names.index(n)
            c = r["rgb"][
                by * TOK : by * TOK + w,
                i * r["cw"] + bx * TOK : i * r["cw"] + bx * TOK + w,
            ]
            sheet.paste(
                Image.fromarray(c.astype(np.uint8)).resize(
                    (w * k, w * k), Image.NEAREST
                ),
                (j * (w * k + 6), 22),
            )
            dr.text((j * (w * k + 6) + 4, 4), n, fill="black")
        sheet.save(run_dir / f"zoom_{want}.png")
        arts.append(f"zoom_{want}.png")
        out.setdefault("zoom", {})[want] = {"token": [by, bx], "departed": int(best)}
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=out,
        label=args.label,
        artifacts=arts,
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
