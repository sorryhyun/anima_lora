#!/usr/bin/env python3
"""Pattern directions at the ×2 head's last hidden layer, and their fold into ``out2`` (CPU).

``norm2``'s output ``h`` (4 × 4 positions per DiT token, 256 values each) reaches the ×2
image through ``out2`` and the windowed overlap-add, both linear, so every read and edit
here is exact in ``h``; no Jacobian is fitted. The renders are the cached last steps of
``p3_generated.py --x2`` (``output/pixel_head/render_cache/<setting>/``); seed 0 fits,
seed 1 is held out.

Token sets, labelled once per render from ``--label_head``'s image by ``tile_dict.py``'s
rules and used for every head: ``all``, ``flat`` (flat tokens), ``depc`` / ``depm``
(departed tiles on the colour / the monochrome prompts), ``agr`` (agreeing tiles).

**Phase read.** Per head, set and period P (in positions; 4 = one token, 3 and 5 are the
null periods, which no pixel shuffle is locked to): the hidden values pooled over the
split's renders are split into the mean per phase (P² of them) and the rest. ``A`` is the
between-phase second moment and ``B`` the covariance of the rest; for content with no
phase lock ``A ≈ B · (P² − 1) / N``. Directions come from ``A u = λ B u`` on the fit split
and are read on the held-out one as ``λ · N / (P² − 1)``: 1 = that floor. Beside each
direction, ``out`` is the std in 8-bit levels of what it writes per unit of content std.

**Edits of ``--edit_head``**, each a new ``out2`` weight and bias:

- ``p32_<set>_k<k>``: the top ``k`` phase directions of the set at period 4 removed
  around the mean hidden, ``h − P (h − μ)`` with ``P = U Uᵀ B``.
- ``f8_k<k>``: on the top ``k`` principal directions of the flat tokens' hidden values
  (and at their mean), the part of the patch that survives the overlap-add of a constant
  field as a non-DC 8 × 8 tile is removed. ``k`` 0 corrects the bias alone; ``k`` 256
  removes that part from every patch.

Each edit is read like the records (``sheet_texture.py``, ``tile_dict.py``): amp and
stamp32 medians over the colour / the monochrome prompts, the grain on flat tokens, the
sky patch of anima_sign s0 (high-pass std, 8-px fold), the departed share on castle s0,
the mean-channel PSNR against the chain and the RMS change from the unedited head.
``--sheets`` writes ``pair_<edit>_<render>.png``: the unedited head | the edit on one
window at ×3, and for the sky window a second file at ×6 contrast. ``--write_heads``
saves an edit as ``<head_root>/<label>/<edit head>_<edit>/head_x2.safetensors``, a file
``p3_generated.py --x2`` reads; with ``--write_only`` nothing else is read or written.

Usage::

    .venv/bin/python -m bench.pixel_readout.experiments.lens_read \
        --heads x2ovll_1124_0139 x2r05gf_1124_1941 x2ovgf_1124_0920 --label x2lens_mmdd_hhmm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scipy.ndimage import gaussian_filter

from bench._common import make_run_dir, write_result
from bench.pixel_readout._prompts import PROMPTS
from bench.pixel_readout._shared import bicubic2
from bench.pixel_readout.experiments.tile_dict import TOK, grid, unit
from library.anima.pixel_head import PixelHeadX2, lat_patches, load_pixel_head
from library.env import resolve_under_home

MONO = {name for name, prompt, _ in PROMPTS if "monochrome" in prompt}
SETS = ("all", "flat", "depc", "depm", "agr")
PERIODS = (4, 3, 5)
SKY = ("anima_sign_s0", 576, 304, 736, 384)  # the records' sky patch, x0 y0 x1 y1
POS = 4  # hidden positions per token side


def lev(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [-1, 1] → the sheet's mean of channels in 8-bit levels."""
    u8 = ((img.clamp(-1, 1) + 1) * 127.5).round()
    return u8.mean(0).numpy().astype(np.float32)


def labels(head_lev: np.ndarray, chain: np.ndarray, bic: np.ndarray):
    """``tile_dict.py``'s tile classes of one head column: (gh, gw) of −1 / 0 departed /
    1 half / 2 agreeing."""
    hp = {
        k: grid(v - gaussian_filter(v, 3))
        for k, v in (("h", head_lev), ("c", chain), ("b", bic))
    }
    gh, gw = hp["c"].shape[:2]
    fl = {k: v.reshape(gh, gw, -1) for k, v in hp.items()}
    ok = (fl["c"].std(-1) >= 8) & (fl["h"].std(-1) >= 6)
    cs = (unit(fl["h"]) * unit(fl["b"])).sum(-1)
    return np.where(
        ok & (cs < 0.3), 0, np.where(ok & (cs >= 0.7), 2, np.where(ok, 1, -1))
    )


def flat_tokens(bic: np.ndarray) -> np.ndarray:
    g = np.hypot(*np.gradient(gaussian_filter(bic, 2)))
    return (grid(g).max((2, 3)) < 1.0) & (grid(bic).mean((2, 3)) >= 40)


def fold_matrix(head: PixelHeadX2) -> torch.Tensor:
    """F (8·8·3, 16·16·3): the 8 × 8 cell a spatially constant ``out2`` patch gives
    after the overlap-add. Checked against the head's own overlap-add."""
    k, p = 2 * head.patch2, head.patch2
    win = head.win2.double().cpu()
    F = torch.zeros(p, p, 3, k, k, 3, dtype=torch.float64)

    def src(y: int) -> tuple[int, int]:
        return y + p // 2, (y + p // 2 + p) if y < p // 2 else (y - p // 2)

    for y in range(p):
        for x in range(p):
            for iy in src(y):
                for ix in src(x):
                    for c in range(3):
                        F[y, x, c, iy, ix, c] = win[iy, ix]
    F = F.reshape(p * p * 3, k * k * 3)
    t = torch.randn(k * k * 3, dtype=torch.float64)
    img = head.double()._overlap_add(t.expand(6, 6, -1).contiguous())
    head.float()
    cell = img[:, 2 * p : 3 * p, 2 * p : 3 * p].permute(1, 2, 0).reshape(-1)
    assert torch.allclose(F @ t, cell, atol=1e-9), "fold matrix != overlap-add"
    return F


class Moments:
    """Per (set, period) phase sums and per set second moments of the hidden values."""

    def __init__(self, c: int):
        self.s2 = {s: torch.zeros(c, c, dtype=torch.float64) for s in SETS}
        self.s1 = {
            (s, p): torch.zeros(p * p, c, dtype=torch.float64)
            for s in SETS
            for p in PERIODS
        }
        self.n = {k: torch.zeros(k[1] ** 2, dtype=torch.float64) for k in self.s1}

    def add(self, h: torch.Tensor, masks: dict[str, torch.Tensor]) -> None:
        n, m, _ = h.shape
        yy, xx = torch.meshgrid(torch.arange(n), torch.arange(m), indexing="ij")
        for s, mask in masks.items():
            if not bool(mask.any()):
                continue
            hs = h[mask].double()
            self.s2[s] += hs.T @ hs
            for p in PERIODS:
                idx = ((yy % p) * p + xx % p)[mask]
                self.s1[s, p].index_add_(0, idx, hs)
                self.n[s, p] += torch.bincount(idx, minlength=p * p).double()

    def split(self, s: str, p: int):
        """(A, B, mean, N): between-phase second moment, covariance of the rest."""
        n = self.n[s, p]
        tot = n.sum()
        if tot == 0:
            return None
        mean = self.s1[s, p].sum(0) / tot
        d = self.s1[s, p] / n.clamp_min(1)[:, None] - mean
        a = (d * (n / tot)[:, None]).T @ d
        b = self.s2[s] / tot - torch.outer(mean, mean) - a
        return a, b, mean, float(tot)


def gen_eig(a: torch.Tensor, b: torch.Tensor):
    """Descending (λ, U) of ``A u = λ B u`` with ``Uᵀ B U = I`` (B ridged at 1e-4 of its
    mean eigenvalue)."""
    c = b.shape[0]
    br = b + 1e-4 * b.trace() / c * torch.eye(c, dtype=b.dtype)
    lam, u = scipy.linalg.eigh(a.numpy(), br.numpy())
    return torch.from_numpy(lam[::-1].copy()), torch.from_numpy(u[:, ::-1].copy()), br


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heads", nargs="+", required=True, help="runs under --head_root")
    ap.add_argument("--head_root", default="output/pixel_head")
    ap.add_argument("--edit_head", default=None, help="default: the last of --heads")
    ap.add_argument("--label_head", default=None, help="default: --edit_head")
    ap.add_argument(
        "--cache",
        default="output/pixel_head/render_cache/anima-base-v1.0__base__s28_cfg4_shift3_fp32",
    )
    ap.add_argument("--p32_sets", nargs="*", default=["all", "flat", "depc"])
    ap.add_argument("--p32_k", nargs="*", type=int, default=[4, 15])
    ap.add_argument("--f8_k", nargs="*", type=int, default=[0, 4, 16, 64, 256])
    ap.add_argument(
        "--sheets",
        nargs="*",
        default=["castle_s0", "anima_sign_s0", "neon_rain_s0"],
    )
    ap.add_argument(
        "--write_heads", nargs="*", default=[], help="edits to save as files"
    )
    ap.add_argument(
        "--write_only",
        action="store_true",
        help="write the --write_heads files and stop: no read of the edits, no result folder",
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    edit_head = args.edit_head or args.heads[-1]
    label_head = args.label_head or edit_head
    root = Path(resolve_under_home(args.head_root))

    heads = {}
    for r in args.heads:
        hd = load_pixel_head(root / r / "head_x2.safetensors", "cpu").float().eval()
        assert isinstance(hd, PixelHeadX2) and hd.overlap, (
            f"{r}: not an overlap ×2 head"
        )
        heads[r] = hd
    grab: dict[str, torch.Tensor] = {}
    for r, hd in heads.items():
        hd.norm2.register_forward_hook(lambda _m, _i, o, r=r: grab.__setitem__(r, o))
    c = heads[edit_head].out2.in_features

    # ---- pass 1: hidden values, labels, moments ----------------------------------
    files = sorted(Path(resolve_under_home(args.cache)).glob("base_*.safetensors"))
    assert files, f"no cached render in {args.cache}"
    mom = {(r, sp): Moments(c) for r in heads for sp in (0, 1)}
    recs, base_rows = [], {r: [] for r in heads}
    for path in files:
        name = path.stem[len("base_") :]
        prompt, seed = name.rsplit("_s", 1)
        t = load_file(str(path))
        chain, bic = lev(t["chain"].float()), lev(bicubic2(t["vae"].float()))
        f, lat = t["f"].float(), lat_patches(t["x_t"]).float()
        imgs = {}
        with torch.no_grad():
            for r, hd in heads.items():
                imgs[r] = hd(f, lat, scale=2).float().clamp(-1, 1)
        cls = labels(lev(imgs[label_head]), chain, bic)
        flat = flat_tokens(bic)
        mono = prompt in MONO
        tok_sets = {
            "all": np.ones_like(flat),
            "flat": flat,
            "depc": (cls == 0) & (not mono),
            "depm": (cls == 0) & mono,
            "agr": cls == 2,
        }
        masks = {
            s: torch.from_numpy(m).repeat_interleave(POS, 0).repeat_interleave(POS, 1)
            for s, m in tok_sets.items()
        }
        for r in heads:
            mom[r, int(seed)].add(grab[r], masks)
        rec = {
            "name": name,
            "seed": int(seed),
            "mono": mono,
            "chain": chain,
            "bic": bic,
            "flat": flat,
            "h": grab[edit_head].clone(),
            "img": imgs[edit_head],
            "cls": cls,
            "amp_ref": float((chain - gaussian_filter(chain, 1.5)).std()),
        }
        recs.append(rec)
        for r in heads:
            base_rows[r].append(read_image(rec, lev(imgs[r]), rec["img"]))
        print(
            f"{name}: tokens {flat.size}, flat {int(flat.sum())}, departed "
            f"{int((cls == 0).sum())}, agreeing {int((cls == 2).sum())}",
            flush=True,
        )
    grab.clear()

    out: dict = {"renders": len(recs), "edit_head": edit_head, "label_head": label_head}
    out["heads"] = {r: summarise(recs, base_rows[r]) for r in heads}
    print("\nunedited heads (the records' meters, recomputed in fp32):")
    for r in heads:
        print(f"  {r:22s} {fmt(out['heads'][r])}")

    # ---- the phase read ------------------------------------------------------------
    w = heads[edit_head].out2.weight.detach().double()  # (768, c)
    bias = heads[edit_head].out2.bias.detach().double()
    win = heads[edit_head].win2.double().reshape(-1, 1).expand(-1, 3).reshape(-1)
    out["phase"], fits = {}, {}
    print(
        "\nphase read, held-out λ·N/(P²−1) of the fit's top directions (1 = the floor)"
    )
    for r, hd in heads.items():
        wr = hd.out2.weight.detach().double()
        for s in SETS:
            for p in PERIODS:
                fit, held = mom[r, 0].split(s, p), mom[r, 1].split(s, p)
                if fit is None or held is None or min(fit[3], held[3]) < 8 * c:
                    continue
                lam, u, br = gen_eig(fit[0], fit[1])
                k = p * p - 1
                uk = u[:, :k]
                num = ((uk.T @ held[0]) * uk.T).sum(1)
                den = ((uk.T @ held[1]) * uk.T).sum(1)
                ho = num / den * held[3] / k
                # what the direction writes per unit of content std, in levels
                wout = ((wr @ uk) * win[:, None]).pow(2).mean(0).sqrt() * 127.5
                row = {
                    "n_fit": fit[3],
                    "n_held": held[3],
                    "fit": (lam[:k] * fit[3] / k).tolist(),
                    "held": ho.tolist(),
                    "out_levels": wout.tolist(),
                    "trace_held": float(
                        torch.linalg.solve(held[1], held[0]).trace() * held[3] / (k * c)
                    ),
                }
                out["phase"][f"{r}/{s}/P{p}"] = row
                top = ", ".join(f"{v:.0f}" for v in ho[:6].tolist())
                print(
                    f"  {r:22s} {s:5s} P{p}  n {held[3]:9.0f}  held {top}  "
                    f"(mean over all {row['trace_held']:.1f}; out of top 3 "
                    f"{', '.join(f'{v:.2f}' for v in wout[:3].tolist())} levels)"
                )
                if r == edit_head and p == 4:
                    fits[s] = (u, br, fit[2])

    # ---- the edits -----------------------------------------------------------------
    eye = torch.eye(c, dtype=torch.float64)
    edits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    mu = mom[edit_head, 0].split("all", 4)[2]
    for s in args.p32_sets:
        if s not in fits:
            continue
        u, br, _ = fits[s]
        for k in args.p32_k:
            pk = u[:, :k] @ u[:, :k].T @ br  # h' = h − P (h − μ), columns
            edits[f"p32_{s}_k{k}"] = (w @ (eye - pk), bias + w @ pk @ mu)
    fm = fold_matrix(heads[edit_head])
    dc = torch.eye(fm.shape[0], dtype=torch.float64)
    for ch in range(3):
        i = torch.arange(ch, fm.shape[0], 3)
        dc[i[:, None], i[None, :]] -= 1.0 / len(i)
    rproj = fm.T @ torch.linalg.solve(fm @ fm.T, dc @ fm)  # onto the fold's non-DC part
    fl = mom[edit_head, 0].split("flat", 4)
    hbar = fl[2]
    cov = fl[0] + fl[1]
    ev, evec = torch.linalg.eigh(cov)
    ev, evec = ev.flip(0), evec.flip(1)
    out["flat_pca_share"] = (ev.cumsum(0) / ev.sum())[[0, 1, 3, 7, 15, 31, 63]].tolist()
    tile = (dc @ fm @ (w @ hbar + bias)).reshape(8, 8, 3).mean(-1) * 127.5
    out["flat_mean_tile_std"] = float(tile.std())
    out["bias_tile_std"] = float(
        ((dc @ fm @ bias).reshape(8, 8, 3).mean(-1) * 127.5).std()
    )
    print(
        f"\nflat tokens' hidden PCA, energy in the top 1/2/4/8/16/32/64: "
        f"{', '.join(f'{v:.3f}' for v in out['flat_pca_share'])}\n"
        f"8 × 8 tile of a constant field, non-DC std in levels: bias alone "
        f"{out['bias_tile_std']:.2f}, at the flat tokens' mean hidden "
        f"{out['flat_mean_tile_std']:.2f}"
    )
    for k in args.f8_k:
        pi = evec[:, :k] @ evec[:, :k].T
        rw = rproj @ w
        edits[f"f8_k{k}"] = (
            w - rw @ pi,
            bias - rproj @ (w @ hbar + bias) + rw @ pi @ hbar,
        )

    if args.write_only:
        for ename in args.write_heads:
            write_head(root, edit_head, ename, *edits[ename], args.label)
        return

    # ---- pass 2: each edit read on every render -------------------------------------
    run_dir = make_run_dir("pixel_readout", args.label)
    arts: list[str] = []
    hd = heads[edit_head]
    out["edits"] = {}
    print("\nedits of", edit_head)
    print(f"  {'(unedited)':16s} {fmt(out['heads'][edit_head])}")
    for ename, (w2, b2) in edits.items():
        w32, b32 = w2.float(), b2.float()
        rows = []
        for rec in recs:
            with torch.no_grad():
                img = hd._overlap_add(rec["h"] @ w32.T + b32).clamp(-1, 1)
            rows.append(read_image(rec, lev(img), rec["img"]))
            short = rec["name"]
            if short in args.sheets:
                arts += sheet(run_dir, ename, rec, img)
        out["edits"][ename] = summarise(recs, rows)
        print(f"  {ename:16s} {fmt(out['edits'][ename])}", flush=True)
        if ename in args.write_heads:
            write_head(root, edit_head, ename, w2, b2, args.label)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=out,
        label=args.label,
        artifacts=arts,
    )
    print(f"wrote {run_dir}")


def write_head(root: Path, parent: str, ename: str, w, b, label: str | None) -> Path:
    """``<root>/<label>/<parent>_<edit>/head_x2.safetensors``: the parent's file with the
    edit's ``out2``. The folder name is the column name the render read gives the file."""
    src = root / parent / "head_x2.safetensors"
    with safe_open(str(src), "pt") as fh:
        meta = dict(fh.metadata() or {})
    sd = load_file(str(src))
    sd["out2.weight"], sd["out2.bias"] = w.float().contiguous(), b.float().contiguous()
    dst = root / (label or "x2lens") / f"{parent}_{ename}" / "head_x2.safetensors"
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(dst), metadata=meta | {"lens_edit": ename, "lens_parent": parent})
    print(f"wrote {dst}", flush=True)
    return dst


def read_image(rec: dict, x: np.ndarray, base: torch.Tensor) -> dict:
    """The records' meters of one head column ``x`` (levels) on one render."""
    hp3 = x - gaussian_filter(x, 3)
    h, w = (x.shape[0] // TOK) * TOK, (x.shape[1] // TOK) * TOK
    cls = labels(x, rec["chain"], rec["bic"])
    dx = grid(np.abs(np.diff(x, axis=1, append=x[:, -1:])))[rec["flat"]].sum((0, 1))
    row = {
        "amp": float((x - gaussian_filter(x, 1.5)).std()) / rec["amp_ref"],
        "stamp32": float(
            hp3[:h, :w].reshape(h // TOK, TOK, w // TOK, TOK).mean((0, 2)).std()
        ),
        "dep": int((cls == 0).sum()),
        "tex": int((cls >= 0).sum()),
        "dx": dx,
        "mse": float(((x - rec["chain"]) ** 2).mean()),
        "d2": float(((x - lev(base)) ** 2).mean()),
    }
    if rec["name"] == SKY[0]:
        _, x0, y0, x1, y1 = SKY
        box = hp3[y0:y1, x0:x1]
        row["sky_hp"] = float(box.std())
        row["sky_fold8"] = float(
            box.reshape((y1 - y0) // 8, 8, (x1 - x0) // 8, 8).mean((0, 2)).std()
        )
    return row


def summarise(recs: list[dict], rows: list[dict]) -> dict:
    s: dict = {}
    for kind, mono in (("colour", False), ("mono", True)):
        sel = [r for rec, r in zip(recs, rows) if rec["mono"] is mono]
        s[f"amp_{kind}"] = float(np.median([r["amp"] for r in sel]))
        s[f"stamp32_{kind}"] = float(np.median([r["stamp32"] for r in sel]))
    n_flat = sum(int(rec["flat"].sum()) for rec in recs)
    prof = np.sum([r["dx"] for r in rows], 0) / (n_flat * TOK)
    s["grain_flat"] = float(np.delete(prof, [7, 15, 23, 31]).mean())
    for rec, r in zip(recs, rows):
        if "sky_hp" in r:
            s["sky_hp"], s["sky_fold8"] = r["sky_hp"], r["sky_fold8"]
        if rec["name"] == "castle_s0":
            s["castle_s0_departed"] = r["dep"] / max(r["tex"], 1)
    for kind, mono in (("colour", False), ("mono", True)):
        sel = [r for rec, r in zip(recs, rows) if rec["mono"] is mono]
        s[f"departed_{kind}"] = sum(r["dep"] for r in sel) / max(
            sum(r["tex"] for r in sel), 1
        )
    s["psnr_chain"] = float(np.mean([10 * np.log10(255.0**2 / r["mse"]) for r in rows]))
    s["rms_change"] = float(np.sqrt(np.mean([r["d2"] for r in rows])))
    s["held_out"] = {
        "amp_colour": float(
            np.median(
                [
                    r["amp"]
                    for rec, r in zip(recs, rows)
                    if rec["seed"] == 1 and not rec["mono"]
                ]
            )
        ),
        "departed_colour": sum(
            r["dep"]
            for rec, r in zip(recs, rows)
            if rec["seed"] == 1 and not rec["mono"]
        )
        / max(
            sum(
                r["tex"]
                for rec, r in zip(recs, rows)
                if rec["seed"] == 1 and not rec["mono"]
            ),
            1,
        ),
    }
    return s


def fmt(s: dict) -> str:
    return (
        f"amp {s['amp_colour']:.3f} / {s['amp_mono']:.3f}  stamp32 "
        f"{s['stamp32_colour']:.3f} / {s['stamp32_mono']:.3f}  grain {s['grain_flat']:.2f}  "
        f"sky {s.get('sky_hp', float('nan')):.2f} / {s.get('sky_fold8', float('nan')):.2f}  "
        f"castle {s.get('castle_s0_departed', float('nan')):.3f}  dep c/m "
        f"{s['departed_colour']:.3f} / {s['departed_mono']:.3f}  dB {s['psnr_chain']:.2f}  "
        f"Δ {s['rms_change']:.2f}"
    )


def sheet(run_dir: Path, ename: str, rec: dict, img: torch.Tensor) -> list[str]:
    """``pair_<edit>_<render>.png``: the unedited head | the edit, one 256-px window at
    ×3 (the sky box's surroundings, else the window with the most departed tiles); the
    sky window also at ×6 contrast around its blur."""
    side, k = 8, 3
    if rec["name"] == SKY[0]:
        _, x0, y0, x1, y1 = SKY
        px = ((x0 + x1) // 2 - side * TOK // 2) // TOK * TOK
        py = ((y0 + y1) // 2 - side * TOK // 2) // TOK * TOK
    else:
        m = (rec["cls"] == 0).astype(np.float32)
        best, py, px = -1.0, 0, 0
        for y in range(m.shape[0] - side + 1):
            for x in range(m.shape[1] - side + 1):
                s = m[y : y + side, x : x + side].sum()
                if s > best:
                    best, py, px = s, y * TOK, x * TOK
    w = side * TOK

    def crop(t: torch.Tensor) -> np.ndarray:
        u8 = ((t.clamp(-1, 1) + 1) * 127.5).round().permute(1, 2, 0).numpy()
        return u8[py : py + w, px : px + w]

    def boost(a: np.ndarray) -> np.ndarray:
        low = np.stack([gaussian_filter(a[..., i], 3) for i in range(3)], -1)
        return np.clip(low + 6 * (a - low), 0, 255)

    names = []
    variants = [("", lambda a: a)]
    if rec["name"] == SKY[0]:
        variants.append(("_x6", boost))
    for tag, fn in variants:
        im = Image.new("RGB", (2 * w * k + 6, w * k + 22), "white")
        dr = ImageDraw.Draw(im)
        for j, (lab, t) in enumerate((("unedited", rec["img"]), (ename, img))):
            a = fn(crop(t)).astype(np.uint8)
            im.paste(
                Image.fromarray(a).resize((w * k, w * k), Image.NEAREST),
                (j * (w * k + 6), 22),
            )
            dr.text((j * (w * k + 6) + 4, 4), lab, fill="black")
        fn_ = f"pair_{ename}_{rec['name']}{tag}.png"
        im.save(run_dir / fn_)
        names.append(fn_)
    return names


if __name__ == "__main__":
    main()
