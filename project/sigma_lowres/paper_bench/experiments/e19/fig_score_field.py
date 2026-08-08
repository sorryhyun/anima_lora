"""E19 post-hoc illustration — the legs as score-style vector FIELDS.

The Yang-Song panel (image.png) draws the score as a spatial arrow field.
The E19 ledgers reduce each leg to pooled scalars (rho, amplitudes); the
bc_plane/bc_comb figures draw those pooled vectors in an abstract 2D
plane. This figure renders the MEASURED legs of the 19.2 residual store
as actual fields over the latent grid of one probe image:

    B   = repromote - reenc      (the data leg / score-shift analogue)
    C   = demote - repromote     (the graph leg)
    B+C = demote - reenc         (the net demotion error)

per sigma, 1024->768, reenc ref, area downsample, projected ⊥ to the
image's pooled native residual direction — the ledger's estimand,
field-resolved. Two complementary views per sigma row:

  quivers    per-location 16-ch vectors block-averaged and projected
             onto a shared 2-PC plane (fit on B⊕C blocks of that row;
             captured energy fraction annotated). Shared arrow scale
             across B / C / B+C, so the net's smallness is visible.
  cos map    FULL 16-dim per-location cos(B, C) (2x2 block-averaged) —
             the quantitative panel: is the anti-alignment pointwise-
             local or only a pooled fact? Annotated with the spatial
             median (exemplar + median across all 40 store images) and
             the pooled ledger rho at that bin.

Illustrative only — not verdict-bearing; every quantitative claim about
the legs lives in the ledgers (measured_192.json / dircos_195.json).
C_pi fields are NOT on disk (19.4 kept arm sums only) — a C_pi panel
would need the 19.4 arm rerun with saved fields.

Usage:
    uv run python project/sigma_lowres/paper_bench/experiments/e19/fig_score_field.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent  # experiments/e19
SIGMA = HERE.parents[2]  # project/sigma_lowres
STORE = SIGMA / "bench" / "results" / "20260806-2342-e192-rho-r-measured" / "residuals"

ROUTE = "768"
NATIVE = "1024"
SIGMA_IDX = {0.3: 5, 0.7: 8}  # verified against the stored sigma scalars
BLOCK_Q = 6  # quiver block size (122x70 -> ~20x11 arrows)
BLOCK_C = 2  # cos-map block size


def halves_mean(stem: str, grid: str, si: int) -> np.ndarray:
    f = np.load(STORE / f"{stem}__{grid}__s{si}.npz")
    assert abs(float(f["sigma"]) - {v: k for k, v in SIGMA_IDX.items()}[si]) < 1e-6
    return 0.5 * (f["m1"].astype(np.float64) + f["m2"].astype(np.float64))


def down_area(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Area (adaptive-average) downsample to hw, mirroring ledger's down()."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(x)[None]
    return F.adaptive_avg_pool2d(t, hw)[0].numpy()


def legs_for(stem: str, si: int):
    """(B, C, net) fields on the demoted grid, ⊥ pooled native direction."""
    dem = halves_mean(stem, ROUTE, si)
    hw = dem.shape[-2:]
    ref = down_area(halves_mean(stem, "reenc", si), hw)
    rp = down_area(halves_mean(stem, f"repromote{ROUTE}", si), hw)
    nat = down_area(halves_mean(stem, NATIVE, si), hw)
    nhat = nat.ravel() / np.linalg.norm(nat.ravel())

    def perp(v: np.ndarray) -> np.ndarray:
        flat = v.ravel()
        return (flat - (flat @ nhat) * nhat).reshape(v.shape)

    B = perp(rp - ref)
    C = perp(dem - rp)
    return B, C, B + C


def block_avg(x: np.ndarray, b: int) -> np.ndarray:
    c, h, w = x.shape
    h2, w2 = h // b * b, w // b * b
    return x[:, :h2, :w2].reshape(c, h2 // b, b, w2 // b, b).mean(axis=(2, 4))


def local_cos(B: np.ndarray, C: np.ndarray, b: int) -> np.ndarray:
    Bb, Cb = block_avg(B, b), block_avg(C, b)
    num = (Bb * Cb).sum(axis=0)
    den = np.linalg.norm(Bb, axis=0) * np.linalg.norm(Cb, axis=0)
    return num / np.maximum(den, 1e-12)


def main() -> None:
    stems = sorted({f.stem.rsplit("__", 2)[0] for f in STORE.glob("*.npz")})
    stem = stems[0]

    # pooled ledger rho at the chosen bins (context annotation)
    m192 = json.loads((HERE / "measured_192.json").read_text())
    tab = m192["routes"][ROUTE]["reenc_area"]["table"]
    sig_centers = m192["sigma_centers"]
    rho_ledger = {s: tab[SIGMA_IDX[s]]["rho"] for s in SIGMA_IDX}
    assert all(abs(sig_centers[SIGMA_IDX[s]] - s) < 1e-6 for s in SIGMA_IDX)

    # across-store spatial-median local cos (representativeness check)
    med_all = {}
    for s, si in SIGMA_IDX.items():
        meds = []
        for st in stems:
            B, C, _ = legs_for(st, si)
            meds.append(float(np.median(local_cos(B, C, BLOCK_C))))
        med_all[s] = (float(np.median(meds)), float(np.min(meds)), float(np.max(meds)))

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(13.6, 7.6),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.12]},
    )
    for row, (s, si) in enumerate(SIGMA_IDX.items()):
        B, C, net = legs_for(stem, si)
        fields = {
            "B  (repromote − reenc)": B,
            "C  (demote − repromote)": C,
            "B + C  (net demotion error)": net,
        }

        # shared 2-PC plane of the row, fit on B ⊕ C blocks
        Bb, Cb = block_avg(B, BLOCK_Q), block_avg(C, BLOCK_Q)
        stack = np.concatenate([Bb.reshape(16, -1).T, Cb.reshape(16, -1).T], axis=0)
        stack = stack - stack.mean(axis=0)
        _, sv, vt = np.linalg.svd(stack, full_matrices=False)
        basis = vt[:2]  # (2, 16)
        var2 = float((sv[:2] ** 2).sum() / (sv**2).sum())

        # shared arrow scale over the row
        blocks = {k: block_avg(v, BLOCK_Q) for k, v in fields.items()}
        uv = {k: np.einsum("pc,chw->phw", basis, v) for k, v in blocks.items()}
        vmax = max(np.hypot(uv[k][0], uv[k][1]).max() for k in uv)

        for col, (name, v) in enumerate(fields.items()):
            ax = axes[row, col]
            u, w = uv[name]
            h2, w2 = u.shape
            yy, xx = np.mgrid[0:h2, 0:w2]
            ax.quiver(
                xx,
                yy,
                w,
                -u,
                angles="xy",
                scale_units="xy",
                scale=vmax / 1.55,
                width=0.006,
                color="0.15",
            )
            rms = float(np.sqrt(np.mean(v**2)))
            ax.set_title(name if row == 0 else "", fontsize=10)
            ax.annotate(
                f"RMS {rms:.2f}",
                (0.03, 0.03),
                xycoords="axes fraction",
                fontsize=8.5,
                color="0.3",
            )
            ax.set_xlim(-0.8, w2 - 0.2)
            ax.set_ylim(h2 - 0.2, -0.8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(rf"$\sigma = {s}$", fontsize=12)
        axes[row, 0].annotate(
            f"2-PC plane: {var2 * 100:.0f}% of B⊕C energy",
            (0.03, 0.955),
            xycoords="axes fraction",
            fontsize=8,
            color="0.4",
            va="top",
        )

        # full 16-dim local cosine map
        ax = axes[row, 3]
        lc = local_cos(B, C, BLOCK_C)
        im = ax.imshow(lc, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(
            r"local $\cos(B, C)$ — full 16-dim" if row == 0 else "", fontsize=10
        )
        med, lo, hi = med_all[s]
        ax.annotate(
            f"median {np.median(lc):+.2f} (this img)\n"
            f"40-img med {med:+.2f}\n"
            f"   ({lo:+.2f}..{hi:+.2f})\n"
            rf"19.2 pooled $\rho$ {rho_ledger[s]:+.2f}"
            "\n"
            r"E14 $\bar\rho{\approx}-0.91$: g-est.,"
            "\nfields not stored",
            (0.04, 0.03),
            xycoords="axes fraction",
            fontsize=7,
            color="0.15",
            va="bottom",
            bbox=dict(fc="white", ec="0.8", alpha=0.9),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(
        rf"$1024\to{ROUTE}$ legs as score-style fields — probe {stem}, reenc "
        r"ref, $\perp$ pooled native dir — r-level (19.2) store; illustration, "
        "verdicts live in the ledgers",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / f"score_field_{ROUTE}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)
    for s in SIGMA_IDX:
        med, lo, hi = med_all[s]
        print(
            f"sigma {s}: 40-image spatial-median local cos "
            f"median {med:+.3f} range [{lo:+.3f}, {hi:+.3f}]  "
            f"(pooled ledger rho {rho_ledger[s]:+.3f})"
        )


if __name__ == "__main__":
    main()
