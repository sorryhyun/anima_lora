"""E25b vs E25e conceptual figure — the conditioning-delta ledger.

Companion to the E25e registration, in the fig_ledger_geometry idiom
(the paper's B/C triangle figure): panel 1 is the schematic
decomposition delta(s) = W.phi(0) + W.(phi(s) - phi(0)) (common-mode +
label, the tip-to-tail analogue of the lattice panel); panel 2 draws
the MEASURED E25b triangle from a carrier checkpoint's actual tensors
(the two long nearly-parallel legs W.phi(0) / W.phi(s) and the tiny
closing side that is the only resolution-discriminating part); panel 3
is E25e by construction — the common-mode leg deleted, s = 0 pinned at
the origin, only the closing side survives; panel 4 is the endpoint
consequence (E25b's measured in-batch dW angle to the native twin) and
the 25e-1 question mark.

Figure assembly only: panel 2/3 vectors are computed from
e25b2_hews_rescond_s1001.safetensors in the exact 2D plane spanned by
(d0, d896) (d768 is projected into that plane; out-of-plane fraction
in result.json); panel 4 reads e25b_ship_read.json. Nothing is refit.

Usage:
    uv run python project/sigma_lowres/paper_bench/experiments/e25e/e25e_concept_fig.py
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Arc, FancyArrowPatch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent  # experiments/e25e
BENCH = HERE.parents[1]  # paper_bench
REPO = HERE.parents[4]
RUNS = BENCH / "runs"

import sys  # noqa: E402

sys.path.insert(0, str(REPO))
from library.anima.models import sigma_lowres_res_cond_delta  # noqa: E402

CKPT = REPO / "output/ckpt/e25b2_hews_rescond_s1001.safetensors"
SHIP = HERE.parent / "e25b" / "e25b_ship_read.json"
OFFSET = HERE.parent / "e25b" / "e25b_native_offset.json"

S896 = math.log2(896 / 1024)
S768 = math.log2(768 / 1024)

# fig_ledger_geometry palette + the two new roles
C_NET = "0.15"
C_CM = "C1"  # common-mode W.phi(0) — the villain
C_DIFF = "C2"  # resolution label W.(phi(s) - phi(0))
C_COMBO, C_RESCOND = "C0", "C3"


def arrow(ax, p, q, color, lw=1.8, ls="-", zorder=3, mut=11):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=mut,
            color=color,
            lw=lw,
            linestyle=ls,
            zorder=zorder,
            shrinkA=0,
            shrinkB=0,
        )
    )


def plane_coords():
    """Exact 2D coordinates of d0 / d896 / d768 in the (d0, d896) plane."""
    with safe_open(str(CKPT), framework="pt") as f:
        proj = f.get_tensor("sigma_lowres_res_cond_proj").float()
    ts = torch.zeros(1, 1)
    d = {
        s: sigma_lowres_res_cond_delta(proj, s, ts).flatten() for s in (0.0, S896, S768)
    }
    e1 = d[0.0] / d[0.0].norm()
    r896 = d[S896] - (d[S896] @ e1) * e1
    e2 = r896 / r896.norm()

    def xy(v):
        return (float(v @ e1), float(v @ e2))

    inplane768 = math.hypot(*xy(d[S768])) / float(d[S768].norm())
    return {"d0": xy(d[0.0]), "d896": xy(d[S896]), "d768": xy(d[S768])}, inplane768


def draw_schematic(ax):
    o, cm = (0.02, 0.30), (0.72, 0.30)
    tip = (cm[0] + 0.10, cm[1] + 0.34)
    arrow(ax, o, cm, C_CM, lw=2.2)
    arrow(ax, cm, tip, C_DIFF, lw=2.2)
    arrow(ax, o, tip, C_NET, lw=1.3, ls=(0, (4, 3)), mut=9)
    ax.plot(*o, "o", ms=5, color=C_NET, zorder=4)
    ax.annotate(
        r"$W\phi(0)$ — common-mode",
        (0.37, 0.24),
        ha="center",
        va="top",
        fontsize=8.5,
        color=C_CM,
    )
    ax.annotate(
        "every step,\nnative included",
        (0.37, 0.10),
        ha="center",
        va="top",
        fontsize=6.8,
        color="0.55",
    )
    ax.annotate(
        r"$W(\phi(s){-}\phi(0))$",
        (0.88, 0.52),
        ha="left",
        va="center",
        fontsize=8.5,
        color=C_DIFF,
    )
    ax.annotate(
        "the resolution label",
        (0.88, 0.42),
        ha="left",
        va="center",
        fontsize=6.8,
        color="0.55",
    )
    ax.annotate(r"$\delta(s)$", (0.33, 0.52), ha="center", fontsize=8.5, color=C_NET)
    ax.annotate(
        "E25b trains both — E25e keeps only the label",
        (0.62, -0.13),
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.35",
    )
    ax.set_xlim(-0.05, 1.42)
    ax.set_ylim(-0.30, 0.90)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the conditioning delta (schematic)", fontsize=10)


def draw_e25b(ax, coords):
    d0, d896, d768 = coords["d0"], coords["d896"], coords["d768"]
    o = (0.0, 0.0)
    arrow(ax, o, d0, C_CM, lw=2.2)
    arrow(ax, o, d896, C_CM, lw=1.4)
    arrow(ax, o, d768, C_CM, lw=1.4)
    # the closing sides — the only resolution-discriminating part
    arrow(ax, d0, d896, C_DIFF, lw=2.0, mut=9)
    arrow(ax, d0, d768, C_DIFF, lw=2.0, mut=9)
    ax.plot(0, 0, "o", ms=4, color=C_NET, zorder=4)
    ax.annotate(
        r"$W\phi(0)$",
        (d0[0] * 0.55, d0[1] - 0.014),
        ha="center",
        va="top",
        fontsize=9,
        color=C_CM,
    )
    ax.annotate(
        r"$W\phi(s)$",
        (d768[0] * 0.52, d768[1] * 0.52 + 0.012),
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_CM,
        rotation=math.degrees(math.atan2(d768[1], d768[0])),
    )
    ax.annotate(
        "label",
        (d896[0] + 0.012, (d0[1] + d896[1]) / 2),
        ha="left",
        va="center",
        fontsize=8,
        color=C_DIFF,
    )
    ax.annotate(
        "$\\|W\\phi(0)\\| = 0.13$–$0.19$\n"
        "label $= 0.005$–$0.024$,  $\\cos \\geq 0.994$\n"
        "$\\Rightarrow$ ~90% common-mode",
        (0.085, -0.062),
        ha="center",
        va="top",
        fontsize=7.5,
        linespacing=1.6,
    )
    # scale bar (0.1 in delta-norm units)
    ax.plot([-0.035, 0.065], [0.135, 0.135], color="0.6", lw=2)
    ax.annotate(
        "0.1", (0.015, 0.128), ha="center", va="top", fontsize=7.5, color="0.45"
    )
    ax.set_xlim(-0.055, 0.215)
    ax.set_ylim(-0.155, 0.16)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("E25b (measured)", fontsize=10)


def draw_e25e(ax, coords):
    d0, d896, d768 = coords["d0"], coords["d896"], coords["d768"]
    o = (0.0, 0.0)
    # the deleted common-mode leg (ghost)
    arrow(ax, o, d0, "0.75", lw=1.4, ls=(0, (4, 3)), mut=9)
    mid = (d0[0] * 0.5, d0[1] * 0.5)
    ax.annotate(
        "×",
        mid,
        ha="center",
        va="center",
        fontsize=13,
        color="0.45",
        zorder=5,
    )
    ax.annotate(
        "removed by construction",
        (mid[0] + 0.005, mid[1] - 0.016),
        ha="center",
        va="top",
        fontsize=6.8,
        color="0.55",
    )
    # the surviving labels, re-anchored at the (pinned) origin
    c896 = (d896[0] - d0[0], d896[1] - d0[1])
    c768 = (d768[0] - d0[0], d768[1] - d0[1])
    arrow(ax, o, c896, C_DIFF, lw=2.0, mut=9)
    arrow(ax, o, c768, C_DIFF, lw=2.0, mut=9)
    ax.plot(0, 0, "o", ms=6, color=C_DIFF, zorder=4)
    ax.annotate(
        r"$W(\phi(s){-}\phi(0))$",
        (c768[0] + 0.006, c768[1] + 0.014),
        ha="left",
        va="bottom",
        fontsize=8,
        color=C_DIFF,
    )
    ax.annotate(
        r"$s{=}0$ pinned at $0$",
        (0.085, -0.062),
        ha="center",
        va="top",
        fontsize=8,
        color=C_DIFF,
    )
    ax.annotate(
        "bit-exact to control, zero grad on native steps",
        (0.085, -0.092),
        ha="center",
        va="top",
        fontsize=6.8,
        color="0.55",
    )
    ax.set_xlim(-0.055, 0.215)
    ax.set_ylim(-0.155, 0.16)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("E25e: only the label (same scale)", fontsize=10)


def draw_endpoint(ax, dw):
    def med2(arm):
        return (
            dw["per_corpus"]["hews"][arm]["median"]
            + dw["per_corpus"]["channel"][arm]["median"]
        ) / 2

    arms = {
        "native": (1.0, C_NET, "-", "native twin"),
        "combo": (med2("combo"), C_COMBO, "-", None),
        "rescond": (med2("rescond"), C_RESCOND, "-", None),
    }
    o = (0.0, 0.0)
    for name, (cos, color, ls, _label) in arms.items():
        th = math.acos(max(-1.0, min(1.0, cos)))
        tip = (math.cos(th), math.sin(th))
        arrow(ax, o, tip, color, lw=2.0, ls=ls)
    # the 25e question: does rescond_c land back at combo's angle?
    th_c = math.acos(med2("combo")) + 0.10
    tip_q = (0.9 * math.cos(th_c), 0.9 * math.sin(th_c))
    arrow(ax, o, tip_q, C_DIFF, lw=1.8, ls=(0, (3, 3)))
    ax.add_patch(
        Arc(
            o,
            1.1,
            1.1,
            theta1=0.0,
            theta2=math.degrees(math.acos(med2("rescond"))),
            color="0.8",
            lw=0.8,
        )
    )
    ax.plot(0, 0, "o", ms=4, color=C_NET, zorder=4)
    ax.annotate(
        "native twin", (1.02, 0.0), ha="left", va="center", fontsize=8, color=C_NET
    )
    ax.annotate(
        f"combo  {dw['per_corpus']['hews']['combo']['median']:.2f}/"
        f"{dw['per_corpus']['channel']['combo']['median']:.2f}",
        (
            math.cos(math.acos(med2("combo"))) + 0.03,
            math.sin(math.acos(med2("combo"))) + 0.02,
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        color=C_COMBO,
    )
    ax.annotate(
        f"rescond  {dw['per_corpus']['hews']['rescond']['median']:.2f}/"
        f"{dw['per_corpus']['channel']['rescond']['median']:.2f}",
        (
            math.cos(math.acos(med2("rescond"))) - 0.02,
            math.sin(math.acos(med2("rescond"))) + 0.03,
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_RESCOND,
    )
    ax.annotate(
        "rescond_c?",
        (tip_q[0] - 0.07, tip_q[1] + 0.06),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=C_DIFF,
    )
    ax.annotate(
        "in-batch $\\Delta W$ cos to native twin\n"
        "(25b measured; 25e-1 asks where\nthe green arrow lands)",
        (0.52, -0.10),
        ha="center",
        va="top",
        fontsize=7.5,
        linespacing=1.5,
    )
    ax.set_xlim(-0.10, 1.30)
    ax.set_ylim(-0.42, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the endpoint consequence", fontsize=10)


def main() -> None:
    coords, inplane768 = plane_coords()
    dw = json.loads(SHIP.read_text())["secondary_dw_toward_native"]

    fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.9))
    draw_schematic(axes[0])
    draw_e25b(axes[1], coords)
    draw_e25e(axes[2], coords)
    draw_endpoint(axes[3], dw)
    fig.tight_layout(w_pad=0.4)

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-fig-e25e-concept"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "e25e_concept.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    shutil.copy2(out, HERE / "e25e_concept.png")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/experiments/e25e/e25e_concept_fig.py",
        label="fig-e25e-concept",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(
            carrier_ckpt=str(CKPT), ship_read=str(SHIP), native_offset=str(OFFSET)
        ),
        metrics=dict(coords=coords, inplane_fraction_d768=inplane768),
        caveats=[
            "figure assembly only: panel 2/3 vectors computed from the "
            "carrier checkpoint's tensors in the exact (d0, d896) plane; "
            "d768 projected into that plane (inplane fraction recorded)",
            "panel 2 annotation ranges are the 12-carrier ranges from "
            "e25b_native_offset.json; the drawn vectors are the "
            "hews_rescond_s1001 carrier",
            "panel 4 arrow angles use the two-corpus mean of medians; "
            "labels give the per-corpus medians; rescond_c is a QUESTION "
            "(registered 25e-1), not a prediction of its answer",
            "panel 1 is schematic (identities, not magnitudes)",
        ],
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=2))
    print(f"wrote {out} (+ {HERE / 'e25e_concept.png'}); inplane768={inplane768:.4f}")


if __name__ == "__main__":
    main()
