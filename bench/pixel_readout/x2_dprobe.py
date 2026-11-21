"""Discriminator probe for the ×2 head (x2_plan.md, 2026-11-21).

What does the PE-Spatial discriminator of ``x2_pegan.py`` separate the ×2 output from
the real 2× image on? The generator is frozen: whole ×2 images are rendered once
(``x2_head.py --dprobe N`` builds the pool and calls :func:`run`), then one fresh
discriminator per arm is trained on paired crops, as in the B+GAN runs, and read on
held-out images.

Arms (the fake side; real = the real 2× image at the same crop):

- ``head``   the ×2 head's output;
- ``chain``  ResShift ×2 of the ×1 release head's output;
- ``null``   the real 2× image of another picture (the meter's floor);
- ``head_n<σ>``  ``head`` with Gaussian noise of that σ ([-1, 1] units) added to both
  crops, in training and in the read.

Arm set ``band`` splits the head's difference from the real image by frequency (a
Gaussian of σ px at the 2× scale; ``low`` = the blurred image, ``high`` = the rest):

- ``low<σ>real_high<σ>head``  the head's output with its low band replaced by the real
  image's (tone and colour corrected, the head's lines and texture kept);
- ``low<σ>head_high<σ>real``  the real image with its low band replaced by the head's
  (the real lines and texture under the head's tone);
- ``real_downup``  the real crop bicubic ×½ then ×2 (what a 1× image holds, no tone change).

Reads, per arm: the training margin ``D(real) − D(fake)`` over the run and by row kind
(``down`` / ``exact``); on the held-out images the margin per tap, over all PE tokens and
over the tokens whose 16-px cell is flat or holds an edge in the real image (range of the
σ 2-px blurred luma inside the cell: under ``FLAT_MAX`` levels, at least ``EDGE_MIN``).
Beside them a grain meter: the std of the luma's high-pass residual on the flat cells,
for the real image, the head and the chain.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

CELL = 16  # PE-Spatial-B16's patch
FLAT_MAX = 4.0  # 8-bit levels
EDGE_MIN = 24.0
BLUR_SIGMA = 2.0


def _luma8(img: torch.Tensor) -> torch.Tensor:
    """(..., 3, H, W) in [-1, 1] → (..., H, W) Rec.601 luma in 8-bit levels."""
    w = img.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return ((img.float() + 1) * 127.5 * w).sum(-3)


def _blur(x: torch.Tensor, sigma: float = BLUR_SIGMA) -> torch.Tensor:
    """(H, W) → (H, W), separable Gaussian, reflect padding."""
    r = int(3 * sigma)
    k = torch.exp(
        -(torch.arange(-r, r + 1, device=x.device).float() ** 2) / (2 * sigma**2)
    )
    k = k / k.sum()
    x = F.pad(x[None, None], (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))[0, 0]


def _blur3(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """(3, H, W) → (3, H, W), per channel."""
    return torch.stack([_blur(c, sigma) for c in img])


def _band_fake(it: dict, mode: str, sigma: float, y: int, x: int, c: int, device):
    """One band-arm fake crop (1, 3, c, c), computed on the crop plus a 3σ margin."""
    h, w = it["real"].shape[-2:]
    p = int(3 * sigma) + 1 if sigma else 8
    y0, y1, x0, x1 = max(0, y - p), min(h, y + c + p), max(0, x - p), min(w, x + c + p)
    real = it["real"][:, y0:y1, x0:x1].to(device).float()
    if mode == "downup":
        hh, ww = (real.shape[-2] // 2) * 2, (real.shape[-1] // 2) * 2
        r = real[None, :, :hh, :ww]
        out = F.interpolate(r, scale_factor=0.5, mode="bicubic", antialias=True)
        out = F.interpolate(out, size=(hh, ww), mode="bicubic").clamp(-1, 1)[0]
        out = F.pad(
            out[None], (0, real.shape[-1] - ww, 0, real.shape[-2] - hh), "replicate"
        )[0]
    else:
        head = it["head"][:, y0:y1, x0:x1].to(device).float()
        low = _blur3(head - real, sigma)  # the head's low band minus the real one's
        out = (head - low) if mode == "lowreal" else (real + low)
        out = out.clamp(-1, 1)
    return out[None, :, y - y0 : y - y0 + c, x - x0 : x - x0 + c]


def cells(real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(3, c, c) real crop → (flat mask, edge mask) over its CELL-px cells, row-major,
    and the blurred luma (c, c)."""
    lb = _blur(_luma8(real))
    hi = F.max_pool2d(lb[None, None], CELL)
    lo = -F.max_pool2d(-lb[None, None], CELL)
    rng_ = (hi - lo).flatten()
    return rng_ < FLAT_MAX, rng_ >= EDGE_MIN, lb


def _cell_std(img: torch.Tensor) -> torch.Tensor:
    """(3, c, c) → per-cell std of the luma's high-pass residual, row-major."""
    lum = _luma8(img)
    hp = (lum - _blur(lum))[None, None]
    m = F.avg_pool2d(hp, CELL)
    return (F.avg_pool2d(hp * hp, CELL) - m * m).clamp_min(0).sqrt().flatten()


def _grid(h: int, w: int, c: int) -> list[tuple[int, int]]:
    """Non-overlapping c-px crops, centred in the image."""
    ny, nx = h // c, w // c
    oy, ox = (h - ny * c) // 2, (w - nx * c) // 2
    return [(oy + i * c, ox + j * c) for i in range(ny) for j in range(nx)]


def _crop(t: torch.Tensor, y: int, x: int, c: int, device) -> torch.Tensor:
    return t[:, y : y + c, x : x + c].to(device).float()[None]


def _to_pil(img: torch.Tensor) -> Image.Image:
    a = ((img.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0)
    return Image.fromarray(a.cpu().numpy())


def _heat(m: torch.Tensor, g: int, lim: float) -> Image.Image:
    """(g·g,) margins → a (g·CELL)² map: red = real above fake, blue = below, ±lim."""
    v = (m.view(g, g) / lim).clamp(-1, 1).cpu().numpy()
    rgb = np.ones((g, g, 3), np.float32)
    rgb[..., 1] -= np.abs(v)
    rgb[..., 2] -= np.clip(v, 0, 1)
    rgb[..., 0] -= np.clip(-v, 0, 1)
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    return im.resize((g * CELL, g * CELL), Image.NEAREST)


def run(
    pool: list[dict],
    held: list[dict],
    device,
    run_dir: Path,
    *,
    steps: int,
    noise: tuple[float, ...],
    crop: int,
    layers: tuple[int, ...],
    lr: float,
    r_weight: float,
    r_sigma: float,
    seed: int,
    n_sheets: int = 4,
    arm_set: str = "source",
    bands: tuple[float, ...] = (8.0, 2.0),
) -> tuple[dict, list[str]]:
    """``pool`` / ``held``: dicts of CPU (3, 2H, 2W) images in [-1, 1] under ``real``,
    ``head``, ``chain``, plus ``kind`` and ``name``. → (metrics, artifacts)."""
    from bench.pixel_readout.x2_pegan import PEDisc

    if arm_set == "band":
        arms = [("real_downup", ("downup", 0.0), 0.0)]
        for b in bands:
            arms.append((f"low{b:g}real_high{b:g}head", ("lowreal", b), 0.0))
            arms.append((f"low{b:g}head_high{b:g}real", ("lowhead", b), 0.0))
    else:
        arms = [("head", "head", 0.0), ("chain", "chain", 0.0), ("null", "null", 0.0)]
        arms += [(f"head_n{s:g}", "head", s) for s in noise]
    metrics: dict = {
        "pool": len(pool),
        "pool_kinds": {
            k: sum(p["kind"] == k for p in pool)
            for k in sorted({p["kind"] for p in pool})
        },
        "held": len(held),
        "steps": steps,
        "flat_max": FLAT_MAX,
        "edge_min": EDGE_MIN,
    }
    artifacts: list[str] = []

    # grain meter and cell shares, held-out images
    acc = {k: [] for k in ("real", "head", "chain")}
    n_flat = n_edge = n_all = 0
    for it in held:
        for y, x in _grid(*it["real"].shape[-2:], crop):
            real = _crop(it["real"], y, x, crop, device)[0]
            flat, edge, _ = cells(real)
            n_flat, n_edge, n_all = (
                n_flat + int(flat.sum()),
                n_edge + int(edge.sum()),
                n_all + flat.numel(),
            )
            if flat.any():
                for k in acc:
                    src = real if k == "real" else _crop(it[k], y, x, crop, device)[0]
                    acc[k].append(_cell_std(src)[flat].cpu())
    metrics["cell_share"] = {"flat": n_flat / n_all, "edge": n_edge / n_all}
    metrics["grain_std_flat"] = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
    print(
        f"cells: flat {n_flat / n_all:.3f} edge {n_edge / n_all:.3f} of {n_all}; "
        "high-pass luma std on flat cells, 8-bit levels: "
        + "  ".join(f"{k} {v:.3f}" for k, v in metrics["grain_std_flat"].items()),
        flush=True,
    )

    def fake_of(items, i, src, y, x, rng):
        if isinstance(src, tuple):
            return _band_fake(items[i], src[0], src[1], y, x, crop, device)
        if src != "null":
            return _crop(items[i][src], y, x, crop, device)
        o = items[(i + 1 + rng.randrange(len(items) - 1)) % len(items)]["real"]
        h, w = o.shape[-2:]
        return _crop(
            o, rng.randint(0, h - crop), rng.randint(0, w - crop), crop, device
        )

    for name, src, sig in arms:
        torch.manual_seed(seed + 31)
        rng = random.Random(seed + 17)
        g = torch.Generator(device=device).manual_seed(seed + 23)
        disc = PEDisc(device, layers=layers, lr=lr, r_weight=r_weight, r_sigma=r_sigma)
        traj, win, by_kind = [], [], {}
        every = min(200, steps)  # the trajectory's bin
        for s in range(steps):
            i = rng.randrange(len(pool))
            it = pool[i]
            h, w = it["real"].shape[-2:]
            y, x = rng.randint(0, h - crop), rng.randint(0, w - crop)
            real = _crop(it["real"], y, x, crop, device)
            fake = fake_of(pool, i, src, y, x, rng)
            if sig > 0:
                real = real + sig * torch.randn(real.shape, generator=g, device=device)
                fake = fake + sig * torch.randn(fake.shape, generator=g, device=device)
            disc._stash = (fake, real)
            log = disc.step()
            win.append(log["d_margin"])
            if s >= steps - steps // 4:
                by_kind.setdefault(it["kind"], []).append(log["d_margin"])
            if (s + 1) % every == 0:
                traj.append(float(np.mean(win)))
                win = []
        # held-out read
        nt = len(layers)
        tot = {k: torch.zeros(nt, dtype=torch.float64) for k in ("all", "flat", "edge")}
        cnt = {k: 0 for k in tot}
        pos = torch.zeros(nt, dtype=torch.float64)
        per_img = []
        rng = random.Random(seed + 19)
        g = torch.Generator(device=device).manual_seed(seed + 29)
        sheets = 0
        for i, it in enumerate(held):
            ms = []
            grid = _grid(*it["real"].shape[-2:], crop)
            for j, (y, x) in enumerate(grid):
                real = _crop(it["real"], y, x, crop, device)
                fake = fake_of(held, i, src, y, x, rng)
                flat, edge, _ = cells(real[0])
                r_in, f_in = real, fake
                if sig > 0:
                    r_in = real + sig * torch.randn(
                        real.shape, generator=g, device=device
                    )
                    f_in = fake + sig * torch.randn(
                        fake.shape, generator=g, device=device
                    )
                with torch.no_grad():
                    m = (disc.logits(r_in) - disc.logits(f_in)).view(nt, -1).double()
                m = m.cpu()
                fl, ed = flat.cpu(), edge.cpu()
                tot["all"] += m.sum(1)
                tot["flat"] += m[:, fl].sum(1)
                tot["edge"] += m[:, ed].sum(1)
                cnt["all"] += m.shape[1]
                cnt["flat"] += int(fl.sum())
                cnt["edge"] += int(ed.sum())
                pos += (m > 0).double().sum(1)
                ms.append(float(m.mean()))
                if src != "null" and j == len(grid) // 2 and sheets < n_sheets:
                    gq = crop // CELL
                    cols = [_to_pil(r_in[0]), _to_pil(f_in[0])]
                    cols += [_heat(m[k].float(), gq, 10.0) for k in range(nt)]
                    labels = ["original 2x", name] + [
                        f"margin, tap after block {b + 1} (red: real above, ±10)"
                        for b in layers
                    ]
                    sheet = Image.new("RGB", (crop * len(cols), crop + 20), "white")
                    d = ImageDraw.Draw(sheet)
                    for k, (im, lab) in enumerate(zip(cols, labels)):
                        sheet.paste(im, (k * crop, 20))
                        d.text((k * crop + 4, 4), lab, fill="black")
                    fn = f"dprobe_{name}_{it['name']}.png"
                    sheet.save(run_dir / fn)
                    artifacts.append(fn)
                    sheets += 1
            per_img.append({"name": it["name"], "margin": float(np.mean(ms))})
        res = {
            "train_margin_per_bin": traj,
            "train_margin_bin": every,
            "train_margin_last_quarter_by_kind": {
                k: float(np.mean(v)) for k, v in sorted(by_kind.items())
            },
            "held_margin": float(tot["all"].sum() / (cnt["all"] * nt)),
            "held_real_above_share": float(pos.sum() / (cnt["all"] * nt)),
            "held_margin_by_tap": {
                f"block{b + 1}": {
                    k: float(tot[k][t] / max(cnt[k], 1))
                    for k in ("all", "flat", "edge")
                }
                for t, b in enumerate(layers)
            },
            "held_per_image": per_img,
        }
        metrics[name] = res
        print(
            f"[{name}] train margin {traj[0]:.2f} → {traj[-1]:.2f} "
            f"(last quarter by kind {res['train_margin_last_quarter_by_kind']}); held "
            f"{res['held_margin']:.2f}, real above {res['held_real_above_share']:.3f}; "
            "tap all / flat / edge: "
            + "  ".join(
                f"{k} {v['all']:.2f} / {v['flat']:.2f} / {v['edge']:.2f}"
                for k, v in res["held_margin_by_tap"].items()
            ),
            flush=True,
        )
        del disc
        torch.cuda.empty_cache()
    return metrics, artifacts
