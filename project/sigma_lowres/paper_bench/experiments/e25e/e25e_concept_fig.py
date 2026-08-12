"""E25b vs E25e conceptual figure — two 3D scenes, same gap-closing, different rail.

One scene per lever, e24_bc_comb_3d style (things planted on the sigma
axis, read in 3D). In each scene the demoted-step gap h(B+C) stands as
a chord pair per verdict bin — grey ctrl vs black conditioned — so
"the lever closes the gap" is the chord shrink along sigma. The
difference between the levers is WHERE the comb stands:

- E25b (left, measured): the trained common-mode offset W.phi(0)
  (orange, ~90% of the delta) displaces the RAIL the whole comb stands
  on — every step, native included, runs off the native anchor; the
  endpoint diverges (dW cos to native twin 0.74 -> 0.41). The tiny
  green label arrows are what actually close the chords (the pair
  comparison cancels the common-mode).
- E25e (right, registered): the rail is PINNED to the native anchor —
  W.(phi(0)-phi(0)) = 0 exactly, zero grad on native steps — and only
  the green label survives. Chords drawn dashed with "?": the grid is
  running; 25e-1/25e-2 ask whether the closing survives and the
  endpoint stays.

Figure assembly only: chord heights are the exact Stage-1 h(B+C)
values (e25b_read.json, 768 probe, ctrl vs rescond stores); the E25e
side's dashed chords REUSE the rescond values as the registered
expectation, marked as such. Offset magnitude is the 12-carrier mean
of ||W.phi(0)|| (e25b_native_offset.json); green label arrows are
exaggerated ~3x for visibility (true 0.005-0.024, annotated). Chord
units (h) and offset units (delta norm) are heterogeneous — this is a
conceptual composition, recorded in the caveats.

Usage:
    uv run python project/sigma_lowres/paper_bench/experiments/e25e/e25e_concept_fig.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # experiments/e25e
BENCH = HERE.parents[1]  # paper_bench
RUNS = BENCH / "runs"

READ1 = HERE.parent / "e25b" / "e25b_read.json"
OFFSET = HERE.parent / "e25b" / "e25b_native_offset.json"

C_NET = "0.15"
C_CM = "C1"  # common-mode W.phi(0)
C_DIFF = "C2"  # resolution label W.(phi(s) - phi(0))
C_SIGMA = "#20b2aa"

GREEN_EXAG = 6.0  # label arrows exaggerated for visibility (annotated)
OFFSET_EXAG = 2.0  # rail offset exaggerated for visibility (annotated)


def draw_scene(ax, ledger_ctrl, ledger_cond, *, offset, dashed_cond, title, story):
    """One 3D comb: x = sigma, y = the offset direction, z = gap chords."""
    sigmas = [r["sigma"] for r in ledger_ctrl]
    xs = [(s - 0.3) * 2.2 for s in sigmas]
    x_lo, x_hi = xs[0] - 0.42, xs[-1] + 0.30
    y_rail = offset

    # native anchor rail (y = 0)
    ax.plot3D([x_lo, x_hi], [0, 0], [0, 0], color="0.55", lw=1.4, ls=(0, (4, 3)))
    ax.text(
        x_hi + 0.04, 0.0, -0.015, "native anchor", color="0.45", fontsize=7.5, va="top"
    )

    if offset > 0:
        # the common-mode offset displaces the rail the comb stands on
        ax.quiver(x_lo, 0, 0, 0, offset, 0, color=C_CM, lw=2.6, arrow_length_ratio=0.22)
        ax.text(
            x_lo - 0.06,
            offset * 0.45,
            0.035,
            "$W\\phi(0)$\n(every step,\nnative included)",
            color=C_CM,
            fontsize=7.2,
            ha="right",
            linespacing=1.4,
        )
    # the rail the comb actually stands on
    ax.plot3D(
        [x_lo, x_hi],
        [y_rail, y_rail],
        [0, 0],
        color=C_CM if offset > 0 else C_SIGMA,
        lw=2.0,
    )
    ax.text(x_hi + 0.04, y_rail, 0.02, "$\\sigma$", color=C_SIGMA, fontsize=10)

    for x, rc, rr in zip(xs, ledger_ctrl, ledger_cond):
        # ctrl chord (grey) and conditioned chord (black), exact h(B+C)
        ax.quiver(
            x - 0.045,
            y_rail,
            0,
            0,
            0,
            rc["h_B_plus_C"],
            color="0.62",
            lw=2.6,
            arrow_length_ratio=0.16,
        )
        ax.quiver(
            x + 0.045,
            y_rail,
            0,
            0,
            0,
            rr["h_B_plus_C"],
            color=C_NET,
            lw=3.0,
            arrow_length_ratio=0.16,
            alpha=0.5 if dashed_cond else 1.0,
            linestyle=(0, (3, 3)) if dashed_cond else "-",
        )
        # the label: the tiny per-step input difference that does the closing
        ax.quiver(
            x,
            y_rail,
            0,
            0,
            0.015 * GREEN_EXAG,
            0,
            color=C_DIFF,
            lw=2.2,
            arrow_length_ratio=0.3,
        )
        ax.text(
            x,
            y_rail - 0.01,
            -0.09,
            f"{rc['sigma']:g}",
            color="0.35",
            fontsize=6.5,
            ha="center",
        )
        ax.text(
            x + 0.045,
            y_rail,
            rr["h_B_plus_C"] + 0.02,
            "?" if dashed_cond else "",
            color=C_NET,
            fontsize=9,
            ha="center",
        )

    # one-time keys
    ax.text(
        xs[0] - 0.045,
        y_rail,
        ledger_ctrl[0]["h_B_plus_C"] + 0.045,
        "$h(B{+}C)$ ctrl",
        color="0.45",
        fontsize=7.2,
        ha="center",
    )
    ax.text(
        xs[-1] + 0.16,
        y_rail + 0.015 * GREEN_EXAG,
        0.10,
        "the label\n$W(\\phi(s){-}\\phi(0))$",
        color=C_DIFF,
        fontsize=7.2,
        ha="left",
        linespacing=1.4,
    )

    ax.set_box_aspect((2.6, 1.0, 1.0))
    ax.set_xlim(x_lo, x_hi + 0.2)
    ax.set_ylim(-0.06, 0.42)
    ax.set_zlim(-0.02, 0.40)
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, y=0.98)
    ax.text2D(
        0.5,
        -0.04,
        story,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="0.30",
        linespacing=1.6,
    )


def main() -> None:
    read1 = json.loads(READ1.read_text())
    off = json.loads(OFFSET.read_text())
    phi0 = sum(r["norm_phi0"] for r in off["per_ckpt"].values()) / len(off["per_ckpt"])

    fig = plt.figure(figsize=(10.5, 4.2))
    ax_b = fig.add_subplot(1, 2, 1, projection="3d")
    ax_e = fig.add_subplot(1, 2, 2, projection="3d")

    draw_scene(
        ax_b,
        read1["ledger_ctrl"],
        read1["ledger_rescond"],
        offset=phi0 * OFFSET_EXAG,
        dashed_cond=False,
        title="E25b — how rescond closes the gap (measured)",
        story=(
            "chords shrink $\\times$0.31–0.94, closed by the green label\n"
            "(the demote–native pair cancels the common-mode);\n"
            "but the rail $W\\phi(0)$ is pushed off the native anchor —\n"
            "every step trains there $\\Rightarrow$ endpoint diverges"
            " ($\\Delta W$ 0.74$\\to$0.41)"
        ),
    )
    draw_scene(
        ax_e,
        read1["ledger_ctrl"],
        read1["ledger_rescond"],
        offset=0.0,
        dashed_cond=True,
        title="E25e — same closing, rail pinned (registered)",
        story=(
            "$W(\\phi(0){-}\\phi(0)) = 0$ exactly, zero grad on native"
            " steps:\nthe rail IS the native anchor — only the label"
            " survives.\nFaded ? chords = registered expectation (grid"
            " running):\n25e-1 does the endpoint collapse disappear;"
            " 25e-2 do renders pass"
        ),
    )
    fig.text(
        0.5,
        0.015,
        "chord heights: exact Stage-1 h(B+C) (768 probe); rail offset: 12-carrier "
        "mean $\\|W\\phi(0)\\|$ $\\times$2 and green label arrows $\\times$6 for visibility "
        "(true 0.005–0.024); h-units and delta-units are heterogeneous — "
        "conceptual composition",
        ha="center",
        fontsize=6.8,
        color="0.5",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))

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
        sources=dict(stage1_read=str(READ1), native_offset=str(OFFSET)),
        metrics=dict(mean_norm_phi0=phi0),
        caveats=[
            "conceptual composition: chord heights (h-units, Stage-1 768 "
            "probe) and the rail offset (delta-norm units, 12-carrier mean "
            "||W.phi(0)||) live in heterogeneous spaces drawn into one scene",
            "green label arrows exaggerated 3x for visibility (true "
            "0.005-0.024); 'common-mode cancels in the pair' is first-order",
            "the E25e side's dashed chords reuse the measured rescond "
            "values as the REGISTERED EXPECTATION — the E25e grid is "
            "running and has not been read; the ? marks are the point",
        ],
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=2))
    print(f"wrote {out} (+ {HERE / 'e25e_concept.png'})")


if __name__ == "__main__":
    main()
