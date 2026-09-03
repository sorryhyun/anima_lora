#!/usr/bin/env python3
"""plan_zh2 U0 — what the shared map does to rows the corpus never shows.

Materializes one or more trained packs and compares every row against its
zero-shot init, split by visit band (0 / 1-4 / 5-49 / 50-499 / 500+):

* ``cos``        cos(init, trained) — how far the row moved in direction
* ``norm``       ‖trained‖ / ‖init‖ — the gain the map applied
* ``pr``         participation ratio of the band's rows, before → after
* ``knn``        mean overlap of each row's k nearest neighbours *within the
                 band sample*, init vs trained — does the map preserve the
                 neighbourhood, or fold the band onto something else?
* ``nn_visited`` (band 0 only) fraction of unvisited rows whose nearest
                 neighbour in a mixed unvisited∪visited sample is a *visited*
                 row, before → after — the "folded onto the visited manifold"
                 read the plan asks for.

Gate (plan_zh2 U0): unvisited rows moving no more than visited ones (cos to
init higher, knn ≥ 0.5) means the map is already gentle and U2 shrinks to a
regularizer check; unvisited rows scaled to ~0.3 norm and losing their
neighbourhood (knn < 0.3) means U2 is load-bearing.

CPU only. Visits are read straight off the staged shards (``.sids`` only);
row indices below the pack's row count are stable across the symbol-block
append, so the 58,968-row packs read fine against the restaged caches::

    make daemon-run ARGS="project/cjk_aware_anima/probes/map_bands_probe.py \\
        --init bench/cjk_adapter/assets/ext_embed_v2 \\
        --pack output/ckpt/cjk_vocab_pack_synthjakozh1_r256 \\
        --pack output/ckpt/cjk_vocab_pack_synthjakozh1_fdiag \\
        --cache_dir cache_tags,cache_ko,cache_desc_ko,cache_zh \\
        --train_registers tags,tags_alt,names,tags_synth_ja,tags_ko,tags_alt_ko,\\
names_ko,names_synth_ko,desc_ko,tags_zh,tags_alt_zh,names_zh,tags_zh_hant,tags_synth_zh \\
        --out project/cjk_aware_anima/reports/u0_map_bands.json"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.distill_cjk.rows import (  # noqa: E402
    BANDS,
    band,
    row_scripts,
    visits_from_caches,
)


def participation_ratio(x: torch.Tensor) -> float:
    c = torch.cov((x - x.mean(0)).T)
    ev = torch.linalg.eigvalsh(c).clamp(min=0)
    return float(ev.sum() ** 2 / (ev**2).sum().clamp_min(1e-12))


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    g = F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).T
    g.fill_diagonal_(-2.0)
    return g.topk(k, dim=1).indices


def knn_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    na, nb = knn(a, k), knn(b, k)
    both = (na.unsqueeze(2) == nb.unsqueeze(1)).any(2).float().sum(1)
    return float((both / k).mean())


def nn_is_visited(unv: torch.Tensor, vis: torch.Tensor) -> tuple[float, float]:
    """(fraction of unvisited rows whose NN is visited, mean cos to that NN)."""
    u = F.normalize(unv, dim=-1)
    v = F.normalize(vis, dim=-1)
    g_uu = u @ u.T
    g_uu.fill_diagonal_(-2.0)
    best_u = g_uu.max(1).values
    g_uv = u @ v.T
    best_v = g_uv.max(1).values
    return float((best_v > best_u).float().mean()), float(best_v.mean())


def band_report(
    init: torch.Tensor, trained: torch.Tensor, rows: list[int], k: int
) -> dict:
    a, b = init[rows], trained[rows]
    cos = F.cosine_similarity(a, b, dim=-1)
    norm = b.norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)
    out = {
        "n": len(rows),
        "cos_mean": float(cos.mean()),
        "cos_p10": float(cos.quantile(0.1)),
        "norm_ratio_mean": float(norm.mean()),
        "norm_ratio_p10": float(norm.quantile(0.1)),
        "norm_ratio_p90": float(norm.quantile(0.9)),
        "pr_init": participation_ratio(a),
        "pr_trained": participation_ratio(b),
    }
    if len(rows) > k + 1:
        out["knn_overlap"] = knn_overlap(a, b, k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--init", type=Path, required=True, help="zero-shot asset prefix")
    ap.add_argument("--pack", type=Path, action="append", required=True)
    ap.add_argument("--cache_dir", required=True, help="comma list under cjk_distill/")
    ap.add_argument("--train_registers", default="")
    ap.add_argument("--sample", type=int, default=4000, help="rows per band")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from anima_lora import default_checkpoints
    from bench.cjk_adapter import ext_vocab
    from library.anima import strategy as strategy_anima

    init, mapping = ext_vocab.load_ext_assets(args.init)
    init = init.float()
    n0 = init.shape[0]
    base = REPO / "post_image_dataset" / "cjk_distill"
    dirs = [base / c.strip() for c in args.cache_dir.split(",") if c.strip()]
    regs = tuple(r.strip() for r in args.train_registers.split(",") if r.strip())
    print(f"visits over {len(dirs)} caches …", flush=True)
    visits = visits_from_caches(dirs, n0, regs)
    print(f"rows visited {int((visits > 0).sum())} / {n0}", flush=True)

    ckpt = default_checkpoints()
    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    scripts = row_scripts(mapping, tok.qwen3_tokenizer, n0)

    rng = random.Random(args.seed)
    by_band: dict[str, list[int]] = {b: [] for b in BANDS}
    for r in range(n0):
        by_band[band(int(visits[r]))].append(r)
    samples = {
        b: sorted(rng.sample(rows, min(args.sample, len(rows))))
        for b, rows in by_band.items()
        if rows
    }
    # Script split of the unvisited band — the plan's render prompt wants
    # unvisited-but-common characters of one script.
    unv_by_script: dict[str, list[int]] = {}
    for r in by_band["0"]:
        unv_by_script.setdefault(scripts[r], []).append(r)
    unv_script_samples = {
        s: sorted(rng.sample(rows, min(args.sample, len(rows))))
        for s, rows in unv_by_script.items()
        if s in ("han", "kana", "hangul") and len(rows) > args.k + 1
    }
    visited_sample = sorted(
        rng.sample(
            [r for r in range(n0) if visits[r] >= 5],
            min(args.sample, int((visits >= 5).sum())),
        )
    )

    report: dict = {
        "init": str(args.init),
        "n_rows": n0,
        "band_sizes": {b: len(rows) for b, rows in by_band.items()},
        "packs": {},
    }
    hdr = (
        f"{'band':8s} {'n':>6s} {'cos':>6s} {'cos_p10':>8s} {'norm':>6s} "
        f"{'n_p10':>6s} {'n_p90':>6s} {'PR init':>8s} {'PR trn':>7s} {'knn':>6s}"
    )
    for pack in args.pack:
        trained, _ = ext_vocab.load_ext_assets(pack)
        trained = trained.float()
        if trained.shape[0] < n0:
            raise SystemExit(f"{pack}: {trained.shape[0]} rows < init {n0}")
        trained = trained[:n0]
        print(f"\n== {pack.name}")
        print(hdr)
        rep: dict = {"bands": {}, "unvisited_by_script": {}}
        for b in BANDS:
            if b not in samples:
                continue
            d = band_report(init, trained, samples[b], args.k)
            rep["bands"][b] = d
            print(
                f"{b:8s} {len(by_band[b]):6d} {d['cos_mean']:6.3f} {d['cos_p10']:8.3f} "
                f"{d['norm_ratio_mean']:6.3f} {d['norm_ratio_p10']:6.3f} "
                f"{d['norm_ratio_p90']:6.3f} {d['pr_init']:8.1f} {d['pr_trained']:7.1f} "
                f"{d.get('knn_overlap', float('nan')):6.3f}"
            )
        for s, rows in sorted(unv_script_samples.items()):
            d = band_report(init, trained, rows, args.k)
            rep["unvisited_by_script"][s] = d
            print(
                f"0/{s:6s} {len(unv_by_script[s]):6d} {d['cos_mean']:6.3f} "
                f"{d['cos_p10']:8.3f} {d['norm_ratio_mean']:6.3f} {d['norm_ratio_p10']:6.3f} "
                f"{d['norm_ratio_p90']:6.3f} {d['pr_init']:8.1f} {d['pr_trained']:7.1f} "
                f"{d.get('knn_overlap', float('nan')):6.3f}"
            )
        if "0" in samples and visited_sample:
            u, v = samples["0"], visited_sample
            f0, c0 = nn_is_visited(init[u], init[v])
            f1, c1 = nn_is_visited(trained[u], trained[v])
            rep["nn_visited"] = {
                "frac_init": f0,
                "frac_trained": f1,
                "cos_to_visited_init": c0,
                "cos_to_visited_trained": c1,
            }
            print(
                f"unvisited rows whose nearest neighbour is a visited row: "
                f"{f0:.3f} -> {f1:.3f}  (cos to nearest visited {c0:.3f} -> {c1:.3f})"
            )
        report["packs"][pack.name] = rep

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
