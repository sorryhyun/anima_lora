#!/usr/bin/env python
"""sigma_lowres E21.1 — cell-level g-ledger + lattice figure (CPU).

Pre-registered in README.md (committed before this file existed). Single
pass over the two surviving arm_sums/ stores (19.3 depth-ledger, 19.4
pi-causal), computing the E14/E19 interventional ledger per
(depth-block x module-type) CELL:

    B = g_rp - g_reenc      (data leg, reenc ref primary)
    C = g_dem - g_rp        (graph leg)
    C_pi = g_dem,pi - g_rp  (graph leg minus phase geometry; 19.4 bins only)

Frozen conventions (no tuning on outputs):
 * cells = groups.json spans exactly — block:XX intersected with type:YY
   (verified: every core-type range nests inside one block range). The 10
   non-row-split types are a disjoint cover of all 77.7M dims ("core",
   the partition); the 5 fused-qkv row splits are supplementary rows.
 * legs/debias verbatim from ledger_pi.py / ledger_depth.slice_ledger:
   second moments from CROSS-SET products only, ref-noise subtraction on
   S, same-set values as bias checks, per-cell perp against the cell's own
   native direction; global-perp *_part quantities (additive to global).
 * gate rel_cos_B/C >= 0.5 (pi reads additionally rel_cos_Cpi >= 0.5);
   failing cells drawn hollow, excluded from verdict statistics.
 * verdict bins: e193 {0.3, 0.4333, 0.5667, 0.7}, e194 {0.7, 0.8333};
   0.9625 / 1.0 endpoint modes plotted detached, non-verdict. C_pi only
   on the 19.4 grid (G11 — no mid-sigma C_pi read, no GPU top-up).
 * per-cell glyphs are exact in span(B_cell, C_cell): B along +x, C at
   arccos(rho_cell). In the pi overlay C_pi is drawn at its exact angle
   to B (arccos(rho_pi)); the C–C_pi mutual angle is 3D and not planar —
   annotated, not drawn (mirrors fig_bc_plane honesty).

Outputs (this dir): e21_cells.json (the record), e21_lattice_<route>.png,
e21_rho_dist.png. Wording note: all cell values are POOLED PER-CELL
(stores are cross-image sums); per-image spread is not obtainable.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e21/e21_cells.py
    ... --figs_only   # redraw figures from the committed digest (21.2 re-reads)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from project.sigma_lowres.paper_bench.vector_ledger import Sums  # noqa: E402

RUNS = {
    "e193": "project/sigma_lowres/bench/results/20260807-0745-e193-depth-ledger",
    "e194": "project/sigma_lowres/bench/results/20260807-1400-e194-pi-causal",
}
GLOBAL_LEDGER = {"e193": "ledger_depth.json", "e194": "ledger_pi.json"}
VERDICT_SIGMA = {"e193": (0.3, 0.4333, 0.5667, 0.7), "e194": (0.7, 0.8333)}
REL_GATE = 0.5
POOLED_RHO_REF = -0.91  # E14/E19 pooled deep constant, drawn as reference line

# disjoint cover of the flat vector (lattice y order, bottom -> top)
CORE_TYPES = [
    "adaln_up_self_attn",
    "adaln_up_cross_attn",
    "adaln_up_mlp",
    "self_attn_qkv_proj",
    "self_attn_output_proj",
    "cross_attn_q_proj",
    "cross_attn_kv_proj",
    "cross_attn_output_proj",
    "mlp_layer1",
    "mlp_layer2",
]
TYPE_SHORT = {
    "adaln_up_self_attn": "ada.self",
    "adaln_up_cross_attn": "ada.cross",
    "adaln_up_mlp": "ada.mlp",
    "self_attn_qkv_proj": "sa.qkv",
    "self_attn_output_proj": "sa.out",
    "cross_attn_q_proj": "ca.q",
    "cross_attn_kv_proj": "ca.kv",
    "cross_attn_output_proj": "ca.out",
    "mlp_layer1": "mlp.1",
    "mlp_layer2": "mlp.2",
}
# fused-qkv row splits — subsets of the *_proj rows above (NOT in the partition)
SPLIT_TYPES = [
    "self_attn_up_q",
    "self_attn_up_k",
    "self_attn_up_v",
    "cross_attn_up_k",
    "cross_attn_up_v",
]
BANDS = {
    "adaln": ["adaln_up_self_attn", "adaln_up_cross_attn", "adaln_up_mlp"],
    "cross_attn": ["cross_attn_q_proj", "cross_attn_kv_proj", "cross_attn_output_proj"],
    "self_attn": ["self_attn_qkv_proj", "self_attn_output_proj"],
    "mlp": ["mlp_layer1", "mlp_layer2"],
}
CLAIM_BANDS = ("adaln", "cross_attn")  # pre-named localization set (README 21.2)

C_B, C_C, C_CPI = "#1f77b4", "#d62728", "#f08c8c"  # e19 figure palette


# ---------------------------------------------------------------- cells


def _intersect(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    a, b = sorted(a), sorted(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def build_cells(groups: dict) -> dict[str, dict]:
    """cell key 'bXX|<type>' -> {ranges, block, type, core, dims}."""
    blocks = sorted(k for k in groups if k.startswith("block:"))
    cells: dict[str, dict] = {}
    for bk in blocks:
        bi = int(bk.split(":")[1])
        for t in CORE_TYPES + SPLIT_TYPES:
            r = _intersect(groups[bk], groups[f"type:{t}"])
            if not r:
                continue
            cells[f"b{bi:02d}|{t}"] = {
                "ranges": r,
                "block": bi,
                "type": t,
                "core": t in CORE_TYPES,
                "dims": sum(e - s for s, e in r),
            }
    return cells


def _gather(v: np.ndarray, ranges: list[list[int]]) -> np.ndarray:
    if len(ranges) == 1:
        s, e = ranges[0]
        return np.asarray(v[s:e], dtype=np.float64)
    return np.concatenate([v[s:e] for s, e in ranges]).astype(np.float64, copy=False)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b) / (na * nb)


# ---------------------------------------------------------------- per-cell ledger


def cell_row(
    g0l: np.ndarray,
    B1l: np.ndarray,
    B2l: np.ndarray,
    C1l: np.ndarray,
    C2l: np.ndarray,
    rdl: np.ndarray,
    scal: dict[str, float],
    P1l: np.ndarray | None = None,
    P2l: np.ndarray | None = None,
) -> dict:
    """slice_ledger math + the ledger_pi extension, one cell."""
    # --- global-perp partition (additive over the core cover)
    gh_g = g0l / scal["G"]
    G2g = 2.0 * scal["G"] ** 2
    B1pg, B2pg = B1l - scal["cB1"] * gh_g, B2l - scal["cB2"] * gh_g
    C1pg, C2pg = C1l - scal["cC1"] * gh_g, C2l - scal["cC2"] * gh_g
    rdpg = rdl - scal["cRd"] * gh_g
    S_part = (float(B1pg @ B2pg) - float(rdpg @ rdpg) / 4.0) / G2g
    F_part = float(C1pg @ C2pg) / G2g
    I_part = (float(B1pg @ C2pg) + float(B2pg @ C1pg)) / G2g

    # --- slice-local ledger (the cell as its own space)
    Gl = float(np.linalg.norm(g0l))
    ghl = g0l / Gl
    B1p, B2p = B1l - float(ghl @ B1l) * ghl, B2l - float(ghl @ B2l) * ghl
    C1p, C2p = C1l - float(ghl @ C1l) * ghl, C2l - float(ghl @ C2l) * ghl
    rdp = rdl - float(ghl @ rdl) * ghl
    G2 = 2.0 * Gl * Gl
    S = (float(B1p @ B2p) - float(rdp @ rdp) / 4.0) / G2
    F = float(C1p @ C2p) / G2
    i1, i2 = float(B1p @ C2p) / G2 * 2.0, float(B2p @ C1p) / G2 * 2.0
    I_x = 0.5 * (i1 + i2)
    I_same = (float(B1p @ C1p) + float(B2p @ C2p)) / G2
    denom = 2.0 * math.sqrt(max(S * F, 0.0))
    rho = I_x / denom if denom > 0.0 else float("nan")
    sig_rho = (abs(i1 - i2) / 2.0) / denom if denom > 0.0 else float("nan")
    row = {
        "G_l": round(Gl, 5),
        "S": round(S, 6),
        "F": round(F, 6),
        "I": round(I_x, 6),
        "I_same": round(I_same, 6),
        "rho": round(rho, 5),
        "sig_rho": round(sig_rho, 5),
        "relB": round(_cos(B1p, B2p), 5),
        "relC": round(_cos(C1p, C2p), 5),
        "amp": round(math.sqrt(S / F), 5) if S > 0 and F > 0 else float("nan"),
        "Sp": round(S_part, 6),
        "Fp": round(F_part, 6),
        "Ip": round(I_part, 6),
    }
    if P1l is None:
        return row

    # --- pi extension (mirrors ledger_pi.pi_ledger; C1/P1 share rp set 1,
    # so C/C_pi second moments use set-crossed pairs only)
    P1pg, P2pg = P1l - scal["cP1"] * gh_g, P2l - scal["cP2"] * gh_g
    Ip_pi = (float(B1pg @ P2pg) + float(B2pg @ P1pg)) / G2g
    Fp_pi = float(P1pg @ P2pg) / G2g
    P1p, P2p = P1l - float(ghl @ P1l) * ghl, P2l - float(ghl @ P2l) * ghl
    F_pi = float(P1p @ P2p) / G2
    I_pi = (float(B1p @ P2p) + float(B2p @ P1p)) / G2
    R = (float(C1p @ P2p) + float(C2p @ P1p)) / G2
    d_pi = 2.0 * math.sqrt(max(S * F_pi, 0.0))
    d_cc = 2.0 * math.sqrt(max(F * F_pi, 0.0))
    Cm, Pm = 0.5 * (C1l + C2l), 0.5 * (P1l + P2l)
    h_C = 1.0 - _cos(g0l, g0l + Cm)
    h_Cpi = 1.0 - _cos(g0l, g0l + Pm)
    row.update(
        {
            "F_pi": round(F_pi, 6),
            "I_pi": round(I_pi, 6),
            "rho_pi": round(I_pi / d_pi, 5) if d_pi > 0.0 else float("nan"),
            "cos_C_Cpi": round(R / d_cc, 5) if d_cc > 0.0 else float("nan"),
            "relCpi": round(_cos(P1p, P2p), 5),
            "amp_pi": round(math.sqrt(S / F_pi), 5)
            if S > 0 and F_pi > 0
            else float("nan"),
            "h_C": round(h_C, 6),
            "h_Cpi": round(h_Cpi, 6),
            "Fp_pi": round(Fp_pi, 6),
            "Ip_pi": round(Ip_pi, 6),
        }
    )
    return row


def gated(row: dict, pi: bool = False) -> bool:
    ok = row["relB"] >= REL_GATE and row["relC"] >= REL_GATE
    if pi:
        ok = ok and row.get("relCpi", float("nan")) >= REL_GATE
    return ok


# ---------------------------------------------------------------- ledger pass


def run_store(store: str, digest: dict) -> None:
    run = REPO_ROOT / RUNS[store]
    s = Sums(run / "arm_sums", device="cpu")
    groups = json.loads((run / "arm_sums" / "groups.json").read_text())
    cells = build_cells(groups)
    centers = s.man["sigma_centers"]
    edges = [
        e
        for e in s.man["demote_edges"].split(",")
        if e and s.has(e) and s.has(f"{e}rp")
    ]
    stored = json.loads((run / GLOBAL_LEDGER[store]).read_text())
    assert stored["data_ref"] == "reenc", "global ledger is not the reenc-ref one"

    digest["cells"][store] = {e: [None] * s.n_bins for e in edges}
    digest["global"][store] = {e: [None] * s.n_bins for e in edges}
    digest["partition_check"][store] = {e: [None] * s.n_bins for e in edges}
    if store == "e193":
        digest["consistency_193"] = {"max_abs_diff": {}, "n_compared": 0}

    for bi in range(s.n_bins):
        t0 = time.time()
        a, b = s.mean("a", bi), s.mean("b", bi)
        g0 = 0.5 * (a + b)
        G = float(np.linalg.norm(g0))
        ghat = g0 / G
        re1, re2 = s.mean("reenc", bi), s.mean("reenc__2", bi)
        ref, rd = 0.5 * (re1 + re2), re1 - re2
        cRd = float(ghat @ rd)
        for e in edges:
            rp1, rp2 = s.mean(f"{e}rp", bi), s.mean(f"{e}rp__2", bi)
            dem1, dem2 = s.mean(e, bi), s.mean(f"{e}__2", bi)
            B1, B2 = rp1 - ref, rp2 - ref
            C1, C2 = dem1 - rp1, dem2 - rp2
            has_pi = s.has(f"{e}pi") and s.has(f"{e}pi__2")
            if has_pi:
                pi1, pi2 = s.mean(f"{e}pi", bi), s.mean(f"{e}pi__2", bi)
                P1, P2 = pi1 - rp1, pi2 - rp2
            scal = {
                "G": G,
                "cB1": float(ghat @ B1),
                "cB2": float(ghat @ B2),
                "cC1": float(ghat @ C1),
                "cC2": float(ghat @ C2),
                "cRd": cRd,
            }
            if has_pi:
                scal["cP1"], scal["cP2"] = float(ghat @ P1), float(ghat @ P2)

            rows: dict[str, dict] = {}
            for key, c in cells.items():
                r = c["ranges"]
                rows[key] = cell_row(
                    _gather(g0, r),
                    _gather(B1, r),
                    _gather(B2, r),
                    _gather(C1, r),
                    _gather(C2, r),
                    _gather(rd, r),
                    scal,
                    _gather(P1, r) if has_pi else None,
                    _gather(P2, r) if has_pi else None,
                )
            digest["cells"][store][e][bi] = rows

            # global reference row (from the committed run ledgers)
            if store == "e193":
                glob = dict(stored["global"][e][bi])
            else:
                glob = dict(stored["bc_ledger"][e][bi])
                glob.update(
                    {
                        k: stored["pi_ledger"][e][bi][k]
                        for k in ("F_pi", "I_pi", "rho_pi", "cos_C_Cpi")
                    }
                )
            digest["global"][store][e][bi] = glob

            # partition exactness: core cells resum to the global ledger
            core = [rows[k] for k in rows if cells[k]["core"]]
            chk = {
                "S": round(sum(r["Sp"] for r in core), 5),
                "F": round(sum(r["Fp"] for r in core), 5),
                "I": round(sum(r["Ip"] for r in core), 5),
                "global": {k: glob[k] for k in ("S", "F", "I")},
            }
            if has_pi:
                chk["I_pi"] = round(sum(r["Ip_pi"] for r in core), 5)
                chk["global"]["I_pi"] = glob["I_pi"]
            digest["partition_check"][store][e][bi] = chk

            # 19.3 anti-relitigation guard: recompute the run's own slices
            # through THIS code path and diff against the committed ledger
            if store == "e193":
                cons = digest["consistency_193"]["max_abs_diff"]
                for gkey, granges in groups.items():
                    mine = cell_row(
                        _gather(g0, granges),
                        _gather(B1, granges),
                        _gather(B2, granges),
                        _gather(C1, granges),
                        _gather(C2, granges),
                        _gather(rd, granges),
                        scal,
                    )
                    theirs = stored["groups"][e][gkey][bi]
                    for mk, tk in (
                        ("rho", "rho"),
                        ("S", "S"),
                        ("F", "F"),
                        ("I", "I"),
                        ("relB", "rel_cos_B"),
                        ("relC", "rel_cos_C"),
                    ):
                        d = abs(mine[mk] - theirs[tk])
                        if not math.isnan(d):
                            cons[tk] = max(cons.get(tk, 0.0), round(d, 6))
                    digest["consistency_193"]["n_compared"] += 1
            if has_pi:
                del pi1, pi2, P1, P2
            del rp1, rp2, dem1, dem2, B1, B2, C1, C2
        del a, b, g0, ghat, re1, re2, ref, rd
        s.drop_bin(bi)
        print(
            f"[{store}] bin {bi} (sigma={centers[bi]}) done in {time.time() - t0:.0f}s",
            flush=True,
        )
    digest["sigma_centers"][store] = centers
    digest["edges"][store] = edges
    digest["cell_dims"][store] = {k: c["dims"] for k, c in cells.items()}


# ---------------------------------------------------------------- 21.2 readings


def verdict_stats(digest: dict) -> None:
    """Gated-core-cell median/IQR per (store, route, verdict sigma) + the
    pre-registered LOCAL / MODAL / MIXED classification per route."""
    out: dict = {"rows": [], "per_route": {}}
    for store in RUNS:
        centers = digest["sigma_centers"][store]
        for e in digest["edges"][store]:
            for bi, sig in enumerate(centers):
                if sig not in VERDICT_SIGMA[store]:
                    continue
                rows = digest["cells"][store][e][bi]
                dims = digest["cell_dims"][store]
                vals = [
                    r["rho"]
                    for k, r in rows.items()
                    if dims.get(k)
                    and k.split("|")[1] in CORE_TYPES
                    and gated(r)
                    and not math.isnan(r["rho"])
                ]
                n_tot = sum(1 for k in rows if k.split("|")[1] in CORE_TYPES)
                med = float(np.median(vals)) if vals else float("nan")
                q25, q75 = (
                    (float(np.percentile(vals, 25)), float(np.percentile(vals, 75)))
                    if vals
                    else (float("nan"),) * 2
                )
                pooled = digest["global"][store][e][bi]["rho"]
                iqr = q75 - q25
                if med <= -0.7 and iqr <= 0.2:
                    cls = "LOCAL"
                elif pooled <= -0.7 and med >= -0.5:
                    cls = "MODAL"
                else:
                    cls = "MIXED"
                out["rows"].append(
                    {
                        "store": store,
                        "route": e,
                        "sigma": sig,
                        "pooled_rho": pooled,
                        "median_rho": round(med, 4),
                        "iqr": round(iqr, 4),
                        "q25": round(q25, 4),
                        "q75": round(q75, 4),
                        "n_gated": len(vals),
                        "n_cells": n_tot,
                        "class": cls,
                    }
                )
    for e in digest["edges"]["e193"]:
        cls = [r["class"] for r in out["rows"] if r["route"] == e]
        out["per_route"][e] = (
            "LOCAL"
            if all(c == "LOCAL" for c in cls)
            else "MODAL"
            if all(c == "MODAL" for c in cls)
            else "MIXED"
        )
    digest["verdict"] = out


def pi_swing(digest: dict) -> None:
    """Localization of the phase-borne component across type bands.

    Swing share is the ADDITIVE global-perp partition of the global
    I_pi - I (Ip_pi - Ip summed over a band's core cells; sums exactly to
    the global swing over the full cover, gate-independent). The uniform
    null share is the band's dimension fraction. Concentration is claimed
    only if the pre-named set (adaln + cross_attn) carries >= 2x uniform.
    """
    out: dict = {"rows": [], "claim_bands": list(CLAIM_BANDS)}
    store = "e194"
    centers = digest["sigma_centers"][store]
    dims = digest["cell_dims"][store]
    tot_dims = sum(v for k, v in dims.items() if k.split("|")[1] in CORE_TYPES)
    for e in digest["edges"][store]:
        for bi, sig in enumerate(centers):
            if sig not in VERDICT_SIGMA[store]:
                continue
            rows = digest["cells"][store][e][bi]
            core = {k: r for k, r in rows.items() if k.split("|")[1] in CORE_TYPES}
            total = sum(r["Ip_pi"] - r["Ip"] for r in core.values())
            glob = digest["global"][store][e][bi]
            entry = {
                "route": e,
                "sigma": sig,
                "total_swing_part": round(total, 5),
                "global_I_pi_minus_I": round(glob["I_pi"] - glob["I"], 5),
                "bands": {},
            }
            claim_share = claim_dim = 0.0
            for band, types in BANDS.items():
                keys = [k for k in core if k.split("|")[1] in types]
                share = sum(core[k]["Ip_pi"] - core[k]["Ip"] for k in keys) / total
                dshare = sum(dims[k] for k in keys) / tot_dims
                drho = [
                    core[k]["rho_pi"] - core[k]["rho"]
                    for k in keys
                    if gated(core[k], pi=True)
                    and not math.isnan(core[k]["rho_pi"])
                    and not math.isnan(core[k]["rho"])
                ]
                entry["bands"][band] = {
                    "swing_share": round(share, 4),
                    "dim_share": round(dshare, 4),
                    "ratio": round(share / dshare, 3) if dshare else float("nan"),
                    "median_drho_gated": round(float(np.median(drho)), 4)
                    if drho
                    else float("nan"),
                    "n_gated": len(drho),
                }
                if band in CLAIM_BANDS:
                    claim_share += share
                    claim_dim += dshare
            entry["claim_set_ratio"] = round(claim_share / claim_dim, 3)
            entry["concentrated"] = bool(claim_share / claim_dim >= 2.0)
            out["rows"].append(entry)
    digest["pi_swing"] = out


def cross_store(digest: dict) -> None:
    """Sigma=0.7 appears in both stores: per-cell rho agreement within the
    twin-based noise (sig_rho from the two cross-pairing halves)."""
    out: dict = {"sigma": 0.7, "rows": []}
    b193 = digest["sigma_centers"]["e193"].index(0.7)
    b194 = digest["sigma_centers"]["e194"].index(0.7)
    for e in digest["edges"]["e193"]:
        r3 = digest["cells"]["e193"][e][b193]
        r4 = digest["cells"]["e194"][e][b194]
        z, dr = [], []
        for k in r3:
            if k.split("|")[1] not in CORE_TYPES or k not in r4:
                continue
            a, b = r3[k], r4[k]
            if not (gated(a) and gated(b)):
                continue
            if math.isnan(a["rho"]) or math.isnan(b["rho"]):
                continue
            d = a["rho"] - b["rho"]
            s = math.sqrt(a["sig_rho"] ** 2 + b["sig_rho"] ** 2)
            dr.append(abs(d))
            if s > 0:
                z.append(abs(d) / s)
        row = {
            "route": e,
            "n_cogated": len(dr),
            "median_abs_drho": round(float(np.median(dr)), 4) if dr else float("nan"),
            "median_abs_z": round(float(np.median(z)), 3) if z else float("nan"),
            "frac_z_le_2": round(float(np.mean([v <= 2.0 for v in z])), 3)
            if z
            else float("nan"),
            "agree": bool(z and float(np.median(z)) <= 2.0),
        }
        out["rows"].append(row)
    out["instrument_discrepancy_flag"] = not all(r["agree"] for r in out["rows"])
    digest["cross_store"] = out


# ---------------------------------------------------------------- figures


def _panel_arrays(rows: dict, pi: bool):
    n_t = len(CORE_TYPES)
    shape = (n_t, 28)
    LB, LC, TH, RHO = (np.full(shape, np.nan) for _ in range(4))
    LP, THP = (np.full(shape, np.nan) for _ in range(2))
    GATE = np.zeros(shape, dtype=bool)
    for k, r in rows.items():
        blk, t = k.split("|")
        if t not in CORE_TYPES:
            continue
        y, x = CORE_TYPES.index(t), int(blk[1:])
        LB[y, x] = math.sqrt(max(2.0 * r["S"], 0.0))
        LC[y, x] = math.sqrt(max(2.0 * r["F"], 0.0))
        RHO[y, x] = r["rho"]
        TH[y, x] = (
            math.acos(min(1.0, max(-1.0, r["rho"])))
            if not math.isnan(r["rho"])
            else np.nan
        )
        GATE[y, x] = gated(r, pi=pi)
        if pi and "rho_pi" in r:
            LP[y, x] = math.sqrt(max(2.0 * r["F_pi"], 0.0))
            THP[y, x] = (
                math.acos(min(1.0, max(-1.0, r["rho_pi"])))
                if not math.isnan(r["rho_pi"])
                else np.nan
            )
    return LB, LC, TH, RHO, LP, THP, GATE


def draw_lattice(digest: dict, route: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    panels = []  # (label, store, bi, kind)
    for store in ("e193", "e194"):
        for bi, sig in enumerate(digest["sigma_centers"][store]):
            kind = "verdict" if sig in VERDICT_SIGMA[store] else "endpoint"
            panels.append((f"{store} sigma={sig}", store, bi, kind))
    pi_panels = [
        (f"e194 sigma={sig} + C_pi overlay", "e194", bi, "pi")
        for bi, sig in enumerate(digest["sigma_centers"]["e194"])
        if sig in VERDICT_SIGMA["e194"]
    ]
    order = (
        [p for p in panels if p[3] == "verdict"]
        + pi_panels
        + [p for p in panels if p[3] == "endpoint"]
    )

    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 3.35 * n))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    for ax, (label, store, bi, kind) in zip(np.atleast_1d(axes), order):
        rows = digest["cells"][store][route][bi]
        glob = digest["global"][store][route][bi]
        LB, LC, TH, RHO, LP, THP, GATE = _panel_arrays(rows, pi=(kind == "pi"))
        bg = np.ma.masked_invalid(np.where(GATE, RHO, np.nan))
        ax.pcolormesh(
            np.arange(29),
            np.arange(len(CORE_TYPES) + 1),
            bg,
            cmap="RdBu",
            norm=norm,
            alpha=0.30,
            edgecolors="none",
        )
        lens = np.concatenate([LB[GATE], LC[GATE]])
        if kind == "pi":
            lens = np.concatenate([lens, LP[GATE & ~np.isnan(LP)]])
        k = 0.44 / np.nanmax(lens) if lens.size else 1.0
        X, Y = np.meshgrid(np.arange(28) + 0.5, np.arange(len(CORE_TYPES)) + 0.22)

        def quiv(U, V, color, mask, alpha=1.0, lw=None):
            m = mask & ~np.isnan(U) & ~np.isnan(V)
            if not m.any():
                return
            ax.quiver(
                X[m],
                Y[m],
                U[m],
                V[m],
                color=color,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0016,
                headwidth=4.5,
                headlength=4.5,
                alpha=alpha,
            )

        UB, VB = np.clip(k * LB, 0, 0.48), np.zeros_like(LB)
        UC, VC = (
            np.clip(k * LC, 0, 0.75) * np.cos(TH),
            np.clip(k * LC, 0, 0.75) * np.sin(TH),
        )
        quiv(UB, VB, C_B, GATE)
        quiv(UC, VC, C_C, GATE)
        quiv(UB, VB, C_B, ~GATE, alpha=0.18)
        quiv(UC, VC, C_C, ~GATE, alpha=0.18)
        if kind == "pi":
            UP = np.clip(k * LP, 0, 0.75) * np.cos(THP)
            VP = np.clip(k * LP, 0, 0.75) * np.sin(THP)
            quiv(UP, VP, C_CPI, GATE)
            quiv(UP, VP, C_CPI, ~GATE, alpha=0.18)
        ax.set_xlim(0, 28)
        ax.set_ylim(0, len(CORE_TYPES))
        ax.set_xticks(np.arange(0, 28, 4) + 0.5)
        ax.set_xticklabels([str(i) for i in range(0, 28, 4)], fontsize=8)
        ax.set_yticks(np.arange(len(CORE_TYPES)) + 0.5)
        ax.set_yticklabels([TYPE_SHORT[t] for t in CORE_TYPES], fontsize=8)
        ax.set_xticks(np.arange(28), minor=True)
        ax.set_yticks(np.arange(len(CORE_TYPES)), minor=True)
        ax.grid(which="minor", color="0.88", lw=0.4)
        ax.tick_params(which="minor", length=0)
        pool = f"pooled rho={glob['rho']:+.2f}"
        if kind == "pi":
            pool += f", rho_pi={glob['rho_pi']:+.2f}"
        n_g = int(GATE.sum())
        title = f"{label}  ({pool}; {n_g}/280 cells pass rel gate)"
        col = "0.45" if kind == "endpoint" else "0.1"
        if kind == "endpoint":
            title += "  [endpoint mode — detached, non-verdict]"
        ax.set_title(title, fontsize=10, color=col, loc="left")
        ax.set_xlabel("depth block", fontsize=8)
    fig.suptitle(
        f"E21 cell-level g-ledger, route 1024→{route} (reenc ref)\n"
        "B (blue) along +x, C (red) at arccos(rho_cell) — exact per-cell plane;\n"
        "C_pi (pink) at its exact B-angle (C–C_pi angle is 3D, not drawn).\n"
        "Hollow = rel<0.5. Background: rho_cell (red = anti-aligned).\n"
        "Pooled per-cell over 40 images x 12 draws x 2 sets — NOT per-image.",
        fontsize=9,
        y=1.0 - 0.02 / n,
    )
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.5 / n))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"lattice -> {out}")


def draw_dists(digest: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    routes = digest["edges"]["e193"]
    fig, axes = plt.subplots(2, len(routes), figsize=(7.0 * len(routes), 9.0))
    for ci, e in enumerate(routes):
        ax = axes[0][ci]
        data, labels, pooled = [], [], []
        for store in ("e193", "e194"):
            for bi, sig in enumerate(digest["sigma_centers"][store]):
                if sig not in VERDICT_SIGMA[store]:
                    continue
                rows = digest["cells"][store][e][bi]
                vals = [
                    r["rho"]
                    for k, r in rows.items()
                    if k.split("|")[1] in CORE_TYPES
                    and gated(r)
                    and not math.isnan(r["rho"])
                ]
                data.append(vals)
                labels.append(f"{sig}\n{store[-3:]}")
                pooled.append(digest["global"][store][e][bi]["rho"])
        vp = ax.violinplot(data, showmedians=True, widths=0.8)
        for body in vp["bodies"]:
            body.set_facecolor("#9ecae1")
            body.set_alpha(0.7)
        vp["cmedians"].set_color("0.1")
        ax.plot(
            range(1, len(pooled) + 1),
            pooled,
            "D",
            color=C_C,
            ms=6,
            label="pooled rho (global ledger)",
        )
        ax.axhline(POOLED_RHO_REF, color="0.4", ls="--", lw=1, label="E14 pooled -0.91")
        ax.axhline(-0.7, color="0.75", ls=":", lw=1, label="LOCAL floor -0.7")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-1.05, 0.5)
        ax.set_ylabel("rho_cell (gated core cells)")
        ax.set_title(f"route 1024→{e}: cell-rho distribution per verdict sigma")
        ax.legend(fontsize=7, loc="upper right")

        ax = axes[1][ci]
        centers = digest["sigma_centers"]["e194"]
        bands = list(BANDS)
        width = 0.35
        for pi_i, (bi, sig) in enumerate(
            (bi, s) for bi, s in enumerate(centers) if s in VERDICT_SIGMA["e194"]
        ):
            rows = digest["cells"]["e194"][e][bi]
            med = []
            for bd in bands:
                vals = [
                    r["rho_pi"] - r["rho"]
                    for k, r in rows.items()
                    if k.split("|")[1] in BANDS[bd]
                    and gated(r, pi=True)
                    and not math.isnan(r["rho_pi"])
                    and not math.isnan(r["rho"])
                ]
                med.append(vals)
            pos = np.arange(len(bands)) + (pi_i - 0.5) * width
            bp = ax.boxplot(
                med,
                positions=pos,
                widths=width * 0.85,
                patch_artist=True,
                showfliers=False,
            )
            col = "#f4b6b6" if pi_i == 0 else "#f08c8c"
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
            for x0, vals in zip(pos, med):
                ax.plot(
                    np.full(len(vals), x0)
                    + np.random.default_rng(0).uniform(-0.05, 0.05, len(vals)),
                    vals,
                    ".",
                    color="0.3",
                    ms=2.5,
                    alpha=0.6,
                )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, fontsize=9)
        ax.set_ylabel("Delta rho_cell = rho(B,C_pi) - rho(B,C)")
        ax.set_title(
            f"route 1024→{e}: pi-swing by type band "
            "(e194 verdict bins 0.7 light / 0.8333 dark)"
        )
    fig.suptitle(
        "E21 — gated core cells only; pooled per-cell (cross-image sums), "
        "not per-image",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"dists -> {out}")


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--figs_only",
        action="store_true",
        help="redraw figures from the committed e21_cells.json",
    )
    ap.add_argument("--out", default=str(HERE / "e21_cells.json"))
    args = ap.parse_args()
    out_json = Path(args.out)

    if args.figs_only:
        digest = json.loads(out_json.read_text())
    else:
        digest: dict = {
            "meta": {
                "runs": {k: str(v) for k, v in RUNS.items()},
                "data_ref": "reenc",
                "rel_gate": REL_GATE,
                "verdict_sigma": {k: list(v) for k, v in VERDICT_SIGMA.items()},
                "core_types": CORE_TYPES,
                "split_types": SPLIT_TYPES,
                "bands": BANDS,
                "claim_bands": list(CLAIM_BANDS),
                "note": (
                    "pooled per-cell values (stores are cross-image sums); "
                    "per-image spread is not obtainable and not claimed"
                ),
            },
            "sigma_centers": {},
            "edges": {},
            "cell_dims": {},
            "cells": {},
            "global": {},
            "partition_check": {},
        }
        for store in RUNS:
            run_store(store, digest)
        verdict_stats(digest)
        pi_swing(digest)
        cross_store(digest)
        out_json.write_text(json.dumps(digest, indent=1))
        print(f"digest -> {out_json}")

    # ---- console: the 21.2 pre-registered readings
    print("\n== verdict (gated core cells) ==")
    for r in digest["verdict"]["rows"]:
        print(
            f"{r['store']} {r['route']} sigma={r['sigma']:<7} pooled={r['pooled_rho']:+.2f} "
            f"median={r['median_rho']:+.3f} IQR={r['iqr']:.3f} "
            f"[{r['q25']:+.2f},{r['q75']:+.2f}] gated {r['n_gated']}/{r['n_cells']} "
            f"-> {r['class']}"
        )
    print("per-route:", digest["verdict"]["per_route"])
    print("\n== pi swing localization (additive partition shares) ==")
    for r in digest["pi_swing"]["rows"]:
        bands = "  ".join(
            f"{b}:{v['swing_share']:+.2f}/{v['dim_share']:.2f}(x{v['ratio']:.1f})"
            for b, v in r["bands"].items()
        )
        print(
            f"{r['route']} sigma={r['sigma']:<7} total={r['total_swing_part']:+.3f} "
            f"(global {r['global_I_pi_minus_I']:+.3f})  {bands}  "
            f"claim x{r['claim_set_ratio']:.2f} -> "
            f"{'CONCENTRATED' if r['concentrated'] else 'delocalized (null stands)'}"
        )
    print("\n== cross-store sigma=0.7 ==")
    for r in digest["cross_store"]["rows"]:
        print(
            f"{r['route']}: n={r['n_cogated']} median|drho|={r['median_abs_drho']} "
            f"median|z|={r['median_abs_z']} frac(z<=2)={r['frac_z_le_2']} "
            f"agree={r['agree']}"
        )
    print(
        "instrument_discrepancy_flag:",
        digest["cross_store"]["instrument_discrepancy_flag"],
    )
    if "consistency_193" in digest:
        print(
            "\n== 19.3 guard: this code path vs committed ledger_depth.json ==\n",
            digest["consistency_193"]["max_abs_diff"],
            f"({digest['consistency_193']['n_compared']} slice rows compared)",
        )
    print("\n== partition exactness (worst |sum(core) - global|) ==")
    for store in digest["partition_check"]:
        worst = 0.0
        for e in digest["partition_check"][store]:
            for chk in digest["partition_check"][store][e]:
                for k in ("S", "F", "I"):
                    worst = max(worst, abs(chk[k] - chk["global"][k]))
        print(f"{store}: {worst:.5f}")

    for e in digest["edges"]["e193"]:
        draw_lattice(digest, e, HERE / f"e21_lattice_{e}.png")
    draw_dists(digest, HERE / "e21_rho_dist.png")


if __name__ == "__main__":
    main()
