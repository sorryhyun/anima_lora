#!/usr/bin/env python
"""DAVE Phase 1 — derive the per-block DC-attenuation mask from the probe.

Desk exercise, no GPU. Reads the Phase-0 ``per_block.npz`` (the (S, L) matrices
of DC cross-seed similarity, AC cross-seed similarity and DC power ratio) and
turns it into a per-block attenuation *weight* ``w(l) ∈ {0, 1}`` — a **flat
statistical pool**, the paper's A.1.1 block-selection rule (arXiv 2606.06813)
with one Anima-specific term:

    member(l) = (DC_sim(l) ≥ dc_thresh)        # paper: cross-seed DC lock ≥ 0.99
                AND (DC_sim(l) − AC_sim(l) > gap_eps)   # Anima fix: drop block 0
                AND (l ≤ block_cap)            # paper: exclude final-stage blocks
    w(l)      = 1.0 if member(l) else 0.0

Each factor maps to a documented selection criterion:

  * **DC_sim ≥ 0.99** is the paper's statistical pool (App. A.1.1: pairwise
    cross-seed cosine of the DC, BH-FDR-tested, threshold 0.99). Flat membership: a ``power·gap`` weight peaks on blocks
    19–27, the artifact-source content blocks.
  * **gap > gap_eps** is the Anima-specific correction the paper's rule lacks:
    block 0 has DC_sim ≈ 0.998 (would pass the paper's threshold) but its **AC is
    also seed-locked** (gap ≈ 0.017), so attenuating it unlocks nothing. The gap
    term drops it; on SD3 the paper keeps block 0, but its AC is not locked there.
  * **l ≤ block_cap** is the paper's "exclude final-stage blocks" (A.1.1: those
    blocks have *limited impact on diversity* and are dropped from the pool). On
    Anima, attenuating 19–27 imprints a patch-grid **dot** artifact straight onto
    ``final_layer`` (Phase 2c).

All similarities are averaged over the early σ window (first ⌊S/3⌋ steps, the
probe's verdict window). The shipped mask pools **blocks 8–18**.

At inference ``--dave`` reads this ``weight`` vector and forms the per-block DC
attenuation factor ``(1 − α_l) = strength · w(l)`` (``--dave_strength`` is the
live knob), so every pooled block attenuates uniformly by ``strength`` and
out-of-pool blocks stay exact no-ops.

    .venv/bin/python bench/dave/derive_alpha_mask.py            # latest run → shipped npz
    .venv/bin/python bench/dave/derive_alpha_mask.py --probe bench/dave/results/<ts>/per_block.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from library.env import resolve_under_home

DEFAULT_OUT = "networks/calibration/dave_alpha.npz"


def _latest_probe() -> Path:
    """Newest ``per_block.npz`` under bench/dave/results/."""
    results = Path(__file__).resolve().parent / "results"
    cands = sorted(results.glob("*/per_block.npz"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(
            "No per_block.npz found under bench/dave/results/. Run "
            "probe_dc_convergence.py first."
        )
    return cands[-1]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--probe",
        default=None,
        help="Path to per_block.npz (default: latest under bench/dave/results/).",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"Output npz path (default: shipped {DEFAULT_OUT}).",
    )
    p.add_argument(
        "--early_frac",
        type=float,
        default=1.0 / 3.0,
        help="Fraction of leading steps that count as the early window.",
    )
    p.add_argument(
        "--dc_thresh",
        type=float,
        default=0.99,
        help="Cross-seed DC cosine-similarity threshold for pool membership "
        "(paper A.1.1 statistical criterion). Default 0.99.",
    )
    p.add_argument(
        "--gap_eps",
        type=float,
        default=0.03,
        help="Min DC−AC gap for membership: drops blocks whose AC is *also* "
        "seed-locked (block 0), where attenuation unlocks nothing. Default 0.03.",
    )
    p.add_argument(
        "--block_cap",
        type=int,
        default=18,
        help="Highest block index allowed in the pool (paper: exclude final-stage "
        "blocks; on Anima these cause the patch-grid dot artifact). Default 18.",
    )
    opts = p.parse_args()

    probe_path = Path(opts.probe) if opts.probe else _latest_probe()
    d = np.load(probe_path)
    dc, ac, pr = d["dc_sim"], d["ac_sim"], d["power_ratio"]
    heavy = set(int(b) for b in d["heavy_blocks"])
    S, L = dc.shape
    early = max(1, int(round(S * opts.early_frac)))

    pow_e = np.nanmean(pr[:early], axis=0)  # (L,)
    dc_e = np.nanmean(dc[:early], axis=0)  # (L,)
    gap_e = np.nanmean(dc[:early] - ac[:early], axis=0)  # (L,)

    blocks = np.arange(L)
    locked = dc_e >= opts.dc_thresh  # paper's flat DC-lock pool
    unlocked_ac = gap_e > opts.gap_eps  # Anima fix: AC not also locked
    not_final = blocks <= opts.block_cap  # exclude final-stage (dot) blocks
    member = locked & unlocked_ac & not_final
    weight = member.astype(np.float64)  # flat {0, 1}

    print(f"probe : {probe_path}")
    print(
        f"rule  : DC_sim≥{opts.dc_thresh}  AND  gap>{opts.gap_eps}  AND  block≤{opts.block_cap}"
    )
    print(f"shape : S={S} steps, L={L} blocks, early window = first {early} steps\n")
    print("block | dc_sim | gap_e  | pow_e | in pool | drop reason")
    print("-" * 60)
    for b in range(L):
        reason = ""
        if not member[b]:
            if not locked[b]:
                reason = f"DC_sim<{opts.dc_thresh}"
            elif not unlocked_ac[b]:
                reason = "AC also locked"
            elif not not_final[b]:
                reason = "final-stage (dots)"
        print(
            f"{b:5d} | {dc_e[b]:.4f} | {gap_e[b]:+.3f} | {pow_e[b]:.3f} "
            f"|   {'YES' if member[b] else ' · '}   | {reason}"
        )

    pool = blocks[member]
    if pool.size:
        print(
            f"\npool  : blocks {pool.min()}–{pool.max()} "
            f"({pool.size} blocks: {list(map(int, pool))})"
        )
    else:
        print("\n⚠️  empty pool — loosen --dc_thresh / --gap_eps / --block_cap")

    # Sanity gates the README calls out: block 0 must be excluded (AC also locked),
    # the final-stage dot blocks (>cap) must be out, and the pool must be non-empty.
    if member[0]:
        print(f"\n⚠️  block 0 in pool (gap={gap_e[0]:+.3f}) — gap_eps should drop it")
    elif pool.size and pool.max() <= opts.block_cap and 0 not in pool:
        print(
            f"\nblock 0 excluded ✓ (gap {gap_e[0]:+.3f}); "
            f"final-stage 19–27 excluded ✓ (cap {opts.block_cap})"
        )

    out = (
        Path(resolve_under_home(opts.out))
        if opts.out
        else Path(resolve_under_home(DEFAULT_OUT))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        weight=weight.astype(np.float32),
        dc_sim_early=dc_e.astype(np.float32),
        power_ratio_early=pow_e.astype(np.float32),
        gap_early=gap_e.astype(np.float32),
        pool_blocks=pool.astype(np.int64),
        dc_thresh=np.float32(opts.dc_thresh),
        gap_eps=np.float32(opts.gap_eps),
        block_cap=np.int64(opts.block_cap),
        heavy_blocks=np.array(sorted(heavy), dtype=np.int64),
        early_steps=np.int64(early),
        num_blocks=np.int64(L),
        source_probe=str(probe_path),
    )
    print(f"\n→ wrote {out}")


if __name__ == "__main__":
    main()
