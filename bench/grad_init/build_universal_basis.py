#!/usr/bin/env python3
"""E0 — does a *universal* gradient basis generalize to unseen artists?

No training. Reuses ``probe_subspace.py``'s sketch machinery: for every
LoRA-target Linear we accumulate ``S = Ωᵀ G`` (q × in, fp32) over an artist's
cached dataset with a frozen DiT. **Ω is seeded identically for every artist**,
so per-artist sketches are additive — the pooled sketch of the union of N
artists' data is just ``Σ_a S_a``, and its top-r right singular vectors are the
universal basis a shipped ``grad_basis_r*.safetensors`` would carry.

We then measure, on held-out artists never in the pool, how much of their
first-step full-FT gradient energy that basis passes:

    capture = ‖S_heldout · V‖² / ‖S_heldout‖²

against four references — the artist's own held-out image half (the reliability
ceiling), ``down_init="weight_svd"``, a random rank-r subspace, and the
universal basis built from 1, 2, 4, 8, … pool artists (the breadth curve that
answers "all 83 at 1 pass or 16 at 2?" from the proposal's open questions).

Kill (docs/proposal/grad_basis_init.md §E0): held-out capture < 0.45, or
< 2× weight_svd → ship only the per-run ``grad_svd`` mode.

Usage::

    make daemon-run ARGS="bench/grad_init/build_universal_basis.py \
        --train_artists a,b,c,... --heldout_artists w,x,y,z \
        --passes 2 --max_samples 32 --gradient_checkpointing"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from bench.grad_init.probe_subspace import (  # noqa: E402
    _BLOCK_RE,
    SketchAccumulator,
    capture,
    enumerate_targets,
    module_kind,
    random_basis,
    run_artist,
    top_right_basis,
    weight_svd_basis,
)
from library.env import resolve_under_home  # noqa: E402

# Depth bands from the noise-scale probe (README §gradient noise scale): early
# blocks are B_simple-dominated (the "gradient basis" there is really the
# input-activation covariance basis), 12-17 is the artist-specific depth,
# 18-27 carries ~90 % of the consistent gradient energy.
DEPTH_BANDS = (
    ("blocks_0_11", 0, 11),
    ("blocks_12_17", 12, 17),
    ("blocks_18_27", 18, 99),
)


def pool_sizes(n_train: int) -> list[int]:
    """1, 2, 4, 8, … up to n_train (always including n_train itself)."""
    ns, k = [], 1
    while k < n_train:
        ns.append(k)
        k *= 2
    ns.append(n_train)
    return ns


def sketch_artist(anima, targets, cache_dir: Path, args, device) -> tuple[dict, dict]:
    """→ ({name: S (4,q,in) cpu fp32}, run metadata). Ω is seed-fixed."""
    acc = SketchAccumulator(
        targets, args.rank + args.oversample, device, seed=args.seed
    )
    acc.attach(targets)
    meta = run_artist(anima, acc, cache_dir, args, device, seed=args.seed)
    acc.detach()
    out = {name: acc.sketch[name].cpu() for name, _o, _m in targets}
    acc.sketch.clear()
    acc.omega.clear()
    del acc
    torch.cuda.empty_cache()
    return out, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser, vae=False, text_encoder=False)
    add_common_args(parser, include_compile=False)
    parser.add_argument(
        "--train_artists",
        type=str,
        required=True,
        help="comma-separated pool artists (the universal basis).",
    )
    parser.add_argument(
        "--heldout_artists",
        type=str,
        required=True,
        help="comma-separated artists never in the pool.",
    )
    parser.add_argument("--cache_root", type=str, default="post_image_dataset/lora")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--oversample", type=int, default=32, help="sketch width q = r + this."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=32,
        help="per artist; caps pool dominance by image-rich artists.",
    )
    parser.add_argument("--max_tokens", type=int, default=4608)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument(
        "--save_basis",
        action="store_true",
        help="write the full-pool universal V_r safetensors.",
    )
    args = parser.parse_args()

    train = [a.strip() for a in args.train_artists.split(",") if a.strip()]
    heldout = [a.strip() for a in args.heldout_artists.split(",") if a.strip()]
    if not train or not heldout:
        raise SystemExit("need both --train_artists and --heldout_artists")
    if set(train) & set(heldout):
        raise SystemExit(f"leak: {sorted(set(train) & set(heldout))} in both pools")
    cache_root = Path(resolve_under_home(args.cache_root))
    for a in train + heldout:
        if not (cache_root / a).is_dir():
            raise SystemExit(f"missing cache dir: {cache_root / a}")

    start_heartbeat(label="grad_init_e0")
    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    targets = enumerate_targets(anima)
    q = args.rank + args.oversample
    print(
        f"targets: {len(targets)} Linear modules | pool={len(train)} heldout={len(heldout)}",
        flush=True,
    )

    ns = pool_sizes(len(train))
    # Cumulative pooled sketches, snapshotted at each N (keeps peak RAM at
    # len(ns) copies instead of one per pool artist).
    cum = {
        name: torch.zeros(q, m.in_features, dtype=torch.float32)
        for name, _o, m in targets
    }
    cum_pass0 = {k: torch.zeros_like(v) for k, v in cum.items()}  # 1-pass ablation
    cum_norm = {k: torch.zeros_like(v) for k, v in cum.items()}  # equal-weight artists
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    meta_train: dict[str, dict] = {}

    for i, a in enumerate(train, start=1):
        S4, meta = sketch_artist(anima, targets, cache_root / a, args, device)
        meta_train[a] = meta
        print(f"[pool {i}/{len(train)}] {a} {meta}", flush=True)
        for name, _o, _m in targets:
            S = S4[name]
            full = S.sum(0)
            cum[name] += full
            cum_pass0[name] += S[0] + S[2]
            cum_norm[name] += full / full.norm().clamp_min(1e-20)
        del S4
        if i in ns:
            snapshots[i] = {k: v.clone() for k, v in cum.items()}

    # ---- bases -------------------------------------------------------------
    V_univ = {n: {} for n in ns}
    for n in ns:
        for name, _o, _m in targets:
            V_univ[n][name], _ = top_right_basis(snapshots[n][name], args.rank)
        snapshots[n] = None  # free
    V_pass0, V_norm, wsvd = {}, {}, {}
    for name, _o, mod in targets:
        V_pass0[name], _ = top_right_basis(cum_pass0[name], args.rank)
        V_norm[name], _ = top_right_basis(cum_norm[name], args.rank)
        wsvd[name] = weight_svd_basis(mod.weight.data, args.rank).cpu()
    del cum_pass0, cum_norm

    # ---- held-out captures --------------------------------------------------
    meta_heldout: dict[str, dict] = {}
    rand_gen = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    rand_basis = {
        name: random_basis(m.in_features, min(args.rank, m.in_features), rand_gen)
        for name, _o, m in targets
    }
    rows: list[dict] = []
    for a in heldout:
        S4, meta = sketch_artist(anima, targets, cache_root / a, args, device)
        meta_heldout[a] = meta
        print(f"[heldout] {a} {meta}", flush=True)
        for name, orig, mod in targets:
            S = S4[name]
            full = S.sum(0)
            h0, h1 = S[0] + S[1], S[2] + S[3]  # image halves
            V_own, _ = top_right_basis(h1, args.rank)
            kind = module_kind(orig)
            row = {
                "artist": a,
                "lora_name": name,
                "kind": kind,
                "token_layer": int(not kind.startswith("adaln")),
                "block": int(m.group(1)) if (m := _BLOCK_RE.match(orig)) else -1,
                "in": mod.in_features,
                "null": min(args.rank, mod.in_features) / mod.in_features,
                # ceiling: own held-out image half (same n as a half-pool basis)
                "cap_by_own_half": capture(h0, V_own),
                "cap_by_wsvd": capture(full, wsvd[name]),
                "cap_by_random": capture(full, rand_basis[name]),
                "cap_by_univ_pass0": capture(full, V_pass0[name]),
                "cap_by_univ_norm": capture(full, V_norm[name]),
            }
            for n in ns:
                row[f"cap_by_univ_n{n}"] = capture(full, V_univ[n][name])
            # apples-to-apples with the ceiling: same measurement half
            row["cap_half_by_univ"] = capture(h0, V_univ[ns[-1]][name])
            rows.append(row)
        del S4

    # ---- aggregation --------------------------------------------------------
    keys = [
        "null",
        "cap_by_own_half",
        "cap_by_wsvd",
        "cap_by_random",
        "cap_by_univ_pass0",
        "cap_by_univ_norm",
        "cap_half_by_univ",
    ] + [f"cap_by_univ_n{n}" for n in ns]

    def agg(sel, key):
        vals = [r[key] for r in rows if sel(r)]
        return sum(vals) / len(vals) if vals else float("nan")

    def block_layer(r):
        return r["block"] >= 0 and r["token_layer"]

    blocks_only = {k: agg(block_layer, k) for k in keys}
    per_artist = {
        a: {k: agg(lambda r, a=a: block_layer(r) and r["artist"] == a, k) for k in keys}
        for a in heldout
    }
    by_band = {
        band: {
            k: agg(lambda r, lo=lo, hi=hi: block_layer(r) and lo <= r["block"] <= hi, k)
            for k in keys
        }
        for band, lo, hi in DEPTH_BANDS
    }
    by_kind = {}
    for k in sorted({r["kind"] for r in rows}):
        by_kind[k] = {key: agg(lambda r, k=k: r["kind"] == k, key) for key in keys}
        by_kind[k]["n"] = sum(1 for r in rows if r["kind"] == k) // len(heldout)

    n_full = ns[-1]
    cap_univ = blocks_only[f"cap_by_univ_n{n_full}"]
    cap_wsvd = blocks_only["cap_by_wsvd"]
    ratio = cap_univ / cap_wsvd if cap_wsvd > 0 else float("inf")
    verdict = "PASS" if (cap_univ >= 0.45 and ratio >= 2.0) else "KILL"

    metrics = {
        "train_artists": train,
        "heldout_artists": heldout,
        "pool_sizes": ns,
        "rank": args.rank,
        "sketch_q": q,
        "n_targets": len(targets),
        "per_artist_train": meta_train,
        "per_artist_heldout": meta_heldout,
        "block_token_layers_mean": blocks_only,
        "per_heldout_artist": per_artist,
        "by_depth_band": by_band,
        "by_kind": by_kind,
        "e0_gate": {
            "capture": cap_univ,
            "weight_svd": cap_wsvd,
            "ratio": ratio,
            "verdict": verdict,
            "rule": "capture >= 0.45 AND ratio >= 2.0",
        },
    }

    run_dir = make_run_dir("grad_init", label=args.label)
    with (run_dir / "per_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    artifacts = ["per_layer.csv"]
    if args.save_basis:
        fn = f"grad_basis_universal_r{args.rank}.safetensors"
        save_file(
            {k: v.to(torch.float16).contiguous() for k, v in V_univ[n_full].items()},
            str(run_dir / fn),
            metadata={
                "pool": ",".join(train),
                "n_artists": str(len(train)),
                "rank": str(args.rank),
                "passes": str(args.passes),
                "layout": "in x r",
            },
        )
        artifacts.append(fn)
    with (run_dir / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    artifacts.append("summary.json")

    print("\n=== held-out capture, block token layers (adaln excluded) ===")
    for k in keys:
        print(f"  {k:24s} {blocks_only[k]:.4f}")
    print("=== breadth curve (pool artists → held-out capture) ===")
    for n in ns:
        print(f"  N={n:3d}  {blocks_only[f'cap_by_univ_n{n}']:.4f}")
    print(f"  N={n_full} @1 pass  {blocks_only['cap_by_univ_pass0']:.4f}")
    print(f"  N={n_full} equal-wt {blocks_only['cap_by_univ_norm']:.4f}")
    print("=== per held-out artist (univ / own-half / weight_svd) ===")
    for a in heldout:
        d = per_artist[a]
        print(
            f"  {a:28s} {d[f'cap_by_univ_n{n_full}']:.3f} / "
            f"{d['cap_by_own_half']:.3f} / {d['cap_by_wsvd']:.3f}"
        )
    print("=== by depth band (univ / own-half / weight_svd) ===")
    for band, _lo, _hi in DEPTH_BANDS:
        d = by_band[band]
        print(
            f"  {band:14s} {d[f'cap_by_univ_n{n_full}']:.3f} / "
            f"{d['cap_by_own_half']:.3f} / {d['cap_by_wsvd']:.3f}"
        )
    print("=== by kind (univ / own-half / weight_svd) ===")
    for k, d in by_kind.items():
        print(
            f"  {k:36s} n={d['n']:3d}  {d[f'cap_by_univ_n{n_full}']:.3f} / "
            f"{d['cap_by_own_half']:.3f} / {d['cap_by_wsvd']:.3f}"
        )
    print(
        f"\n=== E0 gate: {verdict} — capture {cap_univ:.3f} "
        f"({ratio:.1f}× weight_svd {cap_wsvd:.3f}); "
        f"rule: >= 0.45 and >= 2.0× ==="
    )

    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
