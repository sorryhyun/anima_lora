"""The alpha head's read: L1 per region on the cache's read sets and the GT | head sheets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from library.anima.pixel_alpha import estimate_bg, unmix
from scripts.pixel_head import _alpha_data as D

NEAR = 5 / 255  # "the same colour as the background"


def regions(cut: D.Cutout, comp: np.ndarray, bg: np.ndarray) -> dict[str, np.ndarray]:
    a = cut.alpha
    d = np.abs(comp - bg).max(2)
    fg = cut.fg.astype(np.float32) / 255
    return {
        **D.base_regions(a),
        "white_fg": (a >= 0.98) & (fg.min(2) >= 240 / 255),
        "bg_like_fg": (a >= 0.98) & (d <= 15 / 255),
        "hidden": (a > 0.02) & (a < 0.98) & (d <= NEAR),  # no pixel evidence
        "all": np.ones(a.shape, bool),
    }


def flood_alpha(c: np.ndarray, bg: np.ndarray) -> np.ndarray:
    return 1 - D.outer_bg(np.abs(c - bg).max(2) <= NEAR).astype(np.float32)


def on_checker(p: np.ndarray, a: np.ndarray, cell: int = 16) -> np.ndarray:
    h, w = a.shape
    y, x = np.mgrid[0:h, 0:w]
    board = np.where(((y // cell) + (x // cell)) % 2, 0.65, 0.25).astype(np.float32)
    return p + (1 - a[..., None]) * board[..., None]


def to_sample(row, cut: D.Cutout, comp, bg, sigma: float, f, lat) -> dict:
    """A read row and its features → what :func:`score` and :func:`write_sheets` take."""
    return {
        "set": row.set,
        "kind": row.kind,
        "cut": cut,
        "comp": comp,
        "bg": bg,
        "sigma": sigma,
        "regs": regions(cut, comp, bg),
        "f": f,
        "lat": lat,
    }


@torch.no_grad()
def predict(head, s: dict, device) -> np.ndarray:
    """→ α (H, W) in [0, 1], or the RGB head's (H, W, 3)."""
    f, lat = s["f"].to(device), s["lat"].to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        x = head(f, lat)
    x = ((x.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    return x[..., 0] if x.shape[-1] == 1 else x


def quick_l1(samples: list[dict], preds: list[np.ndarray]) -> dict[str, float]:
    """The eval line during training."""
    vals: dict[str, list[float]] = {}
    for s, a in zip(samples, preds):
        err = np.abs(a - s["cut"].alpha)
        for reg in ("all", "transition", "translucent", "thin"):
            m = s["regs"][reg]
            if m.sum() >= 64:
                vals.setdefault(f"{s['set']}_{reg}", []).append(float(err[m].mean()))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def score(s: dict, a: np.ndarray) -> list[dict]:
    """The head and the flood null (B handed, without grain) on one sample."""
    smooth = estimate_bg(s["bg"], np.zeros(a.shape, np.float32))
    out = []
    for name, alpha in (("head", a), ("flood", flood_alpha(s["rgb"], smooth))):
        err = np.abs(alpha - s["cut"].alpha)
        row = {
            "name": name,
            "set": s["set"],
            "bg": s["kind"],
            "key": s["cut"].key,
            "group": s["cut"].group,
        }
        for reg, m in s["regs"].items():
            if m.sum() >= 64:
                row[f"l1_{reg}"] = float(err[m].mean())
        est = estimate_bg(s["rgb"], alpha)
        p = unmix(s["rgb"], alpha, est)
        row["black_l1"] = float(np.abs(p - s["cut"].premul).mean() * 255)
        out.append(row)
    return out


def summarize(rows: list[dict], by: tuple[str, ...]) -> dict:
    groups: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        g = groups.setdefault("/".join(str(r[k]) for k in by), {})
        for k, v in r.items():
            if isinstance(v, float):
                g.setdefault(k, []).append(v)
    return {
        g: {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in d.items()}
        for g, d in groups.items()
    }


def _window(mask: np.ndarray, size: int = 448) -> tuple[slice, slice]:
    """The window that holds most of ``mask``."""
    h, w = mask.shape
    sh, sw = min(h, size), min(w, size)
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = ii[sh:, sw:] - ii[:-sh, sw:] - ii[sh:, :-sw] + ii[:-sh, :-sw]
    y, x = np.unravel_index(int(n.argmax()), n.shape)
    return slice(int(y), int(y) + sh), slice(int(x), int(x) + sw)


def _image(x: np.ndarray, size: int = 0) -> Image.Image:
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, 2)
    im = Image.fromarray(np.round(x.clip(0, 1) * 255).astype(np.uint8))
    if size:
        im.thumbnail((size, size))
    return im


def _sheet(rows: list[tuple[str, list[Image.Image]]], path: Path) -> None:
    pad, label = 8, 18
    width = max(im.width for _, ims in rows for im in ims) + pad
    height = label + sum(ims[0].height + label for _, ims in rows)
    out = Image.new("RGB", (2 * width, height), (90, 90, 90))
    d = ImageDraw.Draw(out)
    for j, name in enumerate(("GT", "alpha head")):
        d.text((j * width + 3, 3), name, fill="yellow")
    y = label
    for name, ims in rows:
        d.text((3, y + 3), name, fill="white")
        for j, im in enumerate(ims):
            out.paste(im, (j * width, y + label))
        y += ims[0].height + label
    out.save(path)


def write_sheets(
    out_dir: Path, samples: list[dict], preds: list[np.ndarray], n: int
) -> list[str]:
    """GT | head, one file per (set, stress kind), on a coloured background."""
    names = []
    for set_name in ("other", "new_bg"):
        for k, group in enumerate(("translucent", "thin", "white_fg", "holes")):
            want = (set_name, group, D.BG_KINDS[1 + k % 2])
            i = next(
                (
                    i
                    for i, s in enumerate(samples)
                    if (s["set"], s["cut"].group, s["kind"]) == want
                ),
                None,
            )
            if i is None or len(names) >= n:
                continue
            s, a = samples[i], preds[i]
            gt, p_gt = s["cut"].alpha, s["cut"].premul
            p = unmix(s["rgb"], a, estimate_bg(s["rgb"], a))
            win = _window(s["regs"]["transition" if group == "white_fg" else group])
            checker = (on_checker(p_gt, gt)[win], on_checker(p, a)[win])
            rows = [
                ("input | on black", [_image(s["comp"], 448), _image(p, 448)]),
                ("alpha, whole", [_image(gt, 448), _image(a, 448)]),
                ("alpha", [_image(gt[win]), _image(a[win])]),
                ("on black", [_image(p_gt[win]), _image(p[win])]),
                ("on checker", [_image(checker[0]), _image(checker[1])]),
            ]
            name = f"{set_name}_{group}_{s['cut'].key}_{s['kind']}.png"
            _sheet(rows, out_dir / name)
            names.append(name)
    return names
