"""E25b vs E25e conceptual figure — per-sigma gap correction + the delta ledger.

Panel 1 is an e24 bc_sphere_gap-style comb: at each verdict-bin sigma
the realized demoted-step gap h(B+C) radiates from the sigma axis as
a thick chord — ctrl (grey) and rescond (black) side by side, so the
per-sigma gap correction of 25b-1 is read as chord shrinkage along
the axis (shallow-sigma concentrated, h(C)-borne). Legs are
deliberately not drawn — h-units do not Euclidean-compose (see
draw_comb docstring). Panel 2 is the delta ledger from a carrier
checkpoint's actual tensors, annotated by WHERE each component acts:
the label W.(phi(s)-phi(0)) is the only part the demote-native pair
comparison sees (it is what closes panel 1's chords — the common-mode
W.phi(0) enters both arms and cancels there), while the common-mode
part shifts every step's operating point and acts on the endpoint.
Panel 3 is that endpoint consequence (measured in-batch dW angles to
the native twin) with the 25e-1 question arrow: E25e deletes the
common-mode leg by construction and keeps only the label.

Figure assembly only: panel 1 reads e25b_read.json (Stage-1 768-probe
bc_ledger rows, ctrl vs rescond stores; chord heights are the exact
h_B_plus_C values, tilt is a drawing convention); panel 2 vectors are
computed from e25b2_hews_rescond_s1001.safetensors in the exact
(d0, d896) plane; panel 3 reads e25b_ship_read.json. Nothing is refit.

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
READ1 = HERE.parent / "e25b" / "e25b_read.json"
SHIP = HERE.parent / "e25b" / "e25b_ship_read.json"
OFFSET = HERE.parent / "e25b" / "e25b_native_offset.json"

S896 = math.log2(896 / 1024)
S768 = math.log2(768 / 1024)

# fig_ledger_geometry palette + the two delta-component roles
C_B, C_C, C_NET = "C0", "C3", "0.15"
C_CM = "C1"  # common-mode W.phi(0)
C_DIFF = "C2"  # resolution label W.(phi(s) - phi(0))
C_SIGMA = "#20b2aa"  # the e24 comb's sigma-axis teal


def arrow(ax, p, q, color, lw=1.8, ls="-", zorder=3, mut=11, alpha=1.0):
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
            alpha=alpha,
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


def draw_comb(ax, ledger_ctrl, ledger_rescond):
    """e24 bc_sphere_gap convention: the gap RADIATES from the sigma axis,
    thick — one grey (ctrl) / black (rescond) chord pair per verdict bin.
    Legs are deliberately NOT drawn: h-units are reliability-weighted and
    do not Euclidean-compose (at sigma=0.5667 no triangle with sides
    h_B, h_C, h(B+C) even exists), so a drawn triangle would be
    fabricated geometry. Chord heights are exact h(B+C)."""
    xs = [i * 0.44 for i in range(5)]
    dx = 0.055
    tilt = math.radians(8)  # slight protractor tilt, the sphere-fig feel
    for x, rc, rr in zip(xs, ledger_ctrl, ledger_rescond):
        for xoff, row, color, lw in (
            (-dx, rc, "0.65", 3.0),
            (+dx, rr, C_NET, 3.4),
        ):
            g = row["h_B_plus_C"]
            tip = (x + xoff - g * math.sin(tilt), g * math.cos(tilt))
            arrow(ax, (x + xoff, 0.0), tip, color, lw=lw, mut=11)
        ax.plot(x, 0.0, "o", ms=3.5, color=C_NET, zorder=4)
        r_h = rr["h_B_plus_C"] / rc["h_B_plus_C"]
        ax.annotate(
            f"$\\sigma={rc['sigma']:g}$",
            (x, -0.055),
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.annotate(
            f"{rc['h_B_plus_C']:.2f}$\\to${rr['h_B_plus_C']:.2f}",
            (x, -0.105),
            ha="center",
            va="top",
            fontsize=7.5,
        )
        ax.annotate(
            f"$\\times${r_h:.2f}",
            (x, -0.152),
            ha="center",
            va="top",
            fontsize=7.5,
            color=C_DIFF if r_h < 0.9 else "0.5",
        )
    # sigma axis (the e24 comb convention)
    arrow(ax, (xs[0] - 0.16, 0.0), (xs[-1] + 0.22, 0.0), C_SIGMA, lw=2.2, mut=13)
    ax.annotate(
        r"$\sigma$",
        (xs[-1] + 0.24, 0.0),
        ha="left",
        va="center",
        fontsize=10,
        color=C_SIGMA,
    )
    ax.annotate(
        "grey: ctrl    black: rescond — same model's demote$\\,\\leftrightarrow\\,$native step gap\n"
        "(the shrink is $h(C)$-borne: the conditioning absorbs the graph leg; 25b-1)",
        ((xs[0] + xs[-1]) / 2, 0.44),
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.35",
        linespacing=1.5,
    )
    ax.set_xlim(xs[0] - 0.22, xs[-1] + 0.34)
    ax.set_ylim(-0.22, 0.50)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "per-$\\sigma$ gap $h(B{+}C)$ — 25b-1 measured (768 probe)", fontsize=10
    )


def draw_delta_ledger(ax, coords):
    d0, d896, d768 = coords["d0"], coords["d896"], coords["d768"]
    o = (0.0, 0.0)
    arrow(ax, o, d0, C_CM, lw=2.2)
    arrow(ax, o, d896, C_CM, lw=1.4)
    arrow(ax, o, d768, C_CM, lw=1.4)
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
        "the label",
        (d896[0] + 0.012, (d0[1] + d896[1]) / 2),
        ha="left",
        va="center",
        fontsize=8,
        color=C_DIFF,
    )
    ax.annotate(
        "label (0.005–0.024): the ONLY part the\n"
        "demote–native pair sees — closes the chords",
        (0.085, -0.055),
        ha="center",
        va="top",
        fontsize=7.2,
        color=C_DIFF,
        linespacing=1.5,
    )
    ax.annotate(
        "common-mode (0.13–0.19, ~90%): enters both\n"
        "arms $\\Rightarrow$ cancels in the gap — acts on the endpoint",
        (0.085, -0.118),
        ha="center",
        va="top",
        fontsize=7.2,
        color=C_CM,
        linespacing=1.5,
    )
    # scale bar (0.1 in delta-norm units)
    ax.plot([-0.035, 0.065], [0.135, 0.135], color="0.6", lw=2)
    ax.annotate(
        "0.1", (0.015, 0.128), ha="center", va="top", fontsize=7.5, color="0.45"
    )
    ax.set_xlim(-0.055, 0.225)
    ax.set_ylim(-0.20, 0.165)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the delta ledger — what acts where", fontsize=10)


def draw_endpoint(ax, dw):
    def med2(arm):
        return (
            dw["per_corpus"]["hews"][arm]["median"]
            + dw["per_corpus"]["channel"][arm]["median"]
        ) / 2

    arms = {
        "native": (1.0, C_NET),
        "combo": (med2("combo"), C_B),
        "rescond": (med2("rescond"), C_C),
    }
    o = (0.0, 0.0)
    for _name, (cos, color) in arms.items():
        th = math.acos(max(-1.0, min(1.0, cos)))
        arrow(ax, o, (math.cos(th), math.sin(th)), color, lw=2.0)
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
    th = math.acos(med2("combo"))
    ax.annotate(
        f"combo  {dw['per_corpus']['hews']['combo']['median']:.2f}/"
        f"{dw['per_corpus']['channel']['combo']['median']:.2f}",
        (math.cos(th) + 0.03, math.sin(th) + 0.02),
        ha="left",
        va="bottom",
        fontsize=8,
        color=C_B,
    )
    th = math.acos(med2("rescond"))
    ax.annotate(
        f"rescond  {dw['per_corpus']['hews']['rescond']['median']:.2f}/"
        f"{dw['per_corpus']['channel']['rescond']['median']:.2f}",
        (math.cos(th) - 0.02, math.sin(th) + 0.03),
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_C,
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
        "in-batch $\\Delta W$ cos to native twin (25b)\n"
        "E25e deletes $W\\phi(0)$, keeps the label —\n"
        "25e-1 asks where the green arrow lands",
        (0.52, -0.10),
        ha="center",
        va="top",
        fontsize=7.5,
        linespacing=1.5,
    )
    ax.set_xlim(-0.10, 1.30)
    ax.set_ylim(-0.45, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the endpoint consequence", fontsize=10)


def main() -> None:
    coords, inplane768 = plane_coords()
    read1 = json.loads(READ1.read_text())
    dw = json.loads(SHIP.read_text())["secondary_dw_toward_native"]

    fig = plt.figure(figsize=(11.5, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.05, 1.0], wspace=0.12)
    draw_comb(fig.add_subplot(gs[0]), read1["ledger_ctrl"], read1["ledger_rescond"])
    draw_delta_ledger(fig.add_subplot(gs[1]), coords)
    draw_endpoint(fig.add_subplot(gs[2]), dw)

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
            stage1_read=str(READ1),
            carrier_ckpt=str(CKPT),
            ship_read=str(SHIP),
            native_offset=str(OFFSET),
        ),
        metrics=dict(coords=coords, inplane_fraction_d768=inplane768),
        caveats=[
            "figure assembly only: comb panel draws the Stage-1 768-probe "
            "bc_ledger rows (hews tenth-scale twins, ctrl vs rescond stores); "
            "drawn interior angle = acos(rho) clamped to [-1,1] "
            "(fig_ledger_geometry convention); leg orientation beta is a "
            "drawing convention",
            "common-mode 'cancels in the gap' is a first-order statement: "
            "it enters both arms of the demote-native pair identically and "
            "acts on the pair only through the shared operating point",
            "delta-ledger vectors computed from the hews_rescond_s1001 "
            "carrier tensors in the exact (d0, d896) plane; annotation "
            "ranges are the 12-carrier ranges from e25b_native_offset.json",
            "endpoint arrow angles use the two-corpus mean of medians; "
            "labels give per-corpus medians; rescond_c is a QUESTION "
            "(registered 25e-1), not a prediction of its answer",
        ],
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=2))
    print(f"wrote {out} (+ {HERE / 'e25e_concept.png'}); inplane768={inplane768:.4f}")


if __name__ == "__main__":
    main()
