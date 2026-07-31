#!/usr/bin/env python
"""sigma_lowres Measurement B — per-σ-bin gradient-equivalence probe.

The tier_routing Phase 3a instrument (redraw-floor null, re-encode confound
control, demote arms) with one change: gradients are accumulated into
**per-σ-bin buckets** instead of one pooled estimate, so the demotion gap
becomes a curve gap_e(σ) instead of a scalar. Phase 3a marginalized σ out;
SwD's spectral analysis (arXiv:2503.16397 §3) pre-registers the hypothesis
that the gap concentrates below a tier-specific σ* and collapses above it
(``project/sigma_lowres/record/initial_proposal.md`` H2/H3).

σ bins are uniform on (0, 1) — the mechanism axis. Per-bin means across
images are the verdict quantity (the estimator class that was reliable in
3a); per-image per-bin rows land in ``per_image.jsonl`` for split-half
reliability analysis.

This file is the driver: probe-set selection, the per-image arm loop, and the
result envelope. The parts it orchestrates live in ``sigma_probe/`` — ``cli``
(flags + validation), ``kernel`` (σ grids, RoPE patches, the gradient
estimator), ``stats`` (accumulators + every reduction).

Usage::

    uv run python project/sigma_lowres/bench/run_sigma_probe.py \
        --adapter output/ckpt/anima_soup_sincos.safetensors --label phase0
    uv run python project/sigma_lowres/bench/run_sigma_probe.py \
        --adapter <ckpt> --smoke --label smoke
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from project.sigma_lowres.bench.sigma_probe.cli import (  # noqa: E402
    build_arm_keys,
    parse_args,
    resolve_run_config,
)
from project.sigma_lowres.bench.sigma_probe.kernel import (  # noqa: E402
    build_groups,
    build_probe_bundle,
    build_sigmas,
    cosine,
    enable_deterministic,
    encode_probe_latents,
    grad_estimate_binned,
    grouped_cosine,
    pi_rope,
    yarn_rope,
)
from project.sigma_lowres.bench.sigma_probe.stats import (  # noqa: E402
    ArmStatter,
    ArmSumAccumulator,
    PoolAccumulator,
    build_headline,
    kappa_row,
    pool_stats,
)
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    select_probe_set,
)
from library.io.cache import (  # noqa: E402
    load_cached_latents,
    load_cached_text_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def select_probe(args, cfg) -> list:
    """Score the corpus and pick the probe images — the frozen ``--probe_list``
    in file order when given, else the 3a-compatible stratified selection."""
    log.info("scoring corpus + selecting probe set (3a-compatible)…")
    artists = args.artists.split(",") if args.artists else None
    if cfg.probe_order is not None:
        artists = sorted({a for a, _ in cfg.probe_order})
    records = score_corpus(
        artists=artists,
        k=args.quant_k,
        limit=args.score_limit,
        data_root=Path(args.data_root).resolve() if args.data_root else None,
    )
    if cfg.probe_order is not None:
        by_key = {(r.artist, r.stem): r for r in records}
        missing = [k for k in cfg.probe_order if k not in by_key]
        bad = [
            k
            for k in cfg.probe_order
            if k in by_key
            and not (by_key[k].tier == args.tier and by_key[k].complete())
        ]
        if missing or bad:
            raise SystemExit(
                f"--probe_list stems not usable as frozen: "
                f"missing={missing[:5]} incomplete-or-off-tier={bad[:5]}"
            )
        probe = [by_key[k] for k in cfg.probe_order]
    else:
        probe = select_probe_set(
            records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
        )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records in the scored pool")
    if args.pool:
        probe = sorted(probe, key=lambda r: r.redundancy)
        log.info(
            f"pool mode: {args.pool} images/stratum, sorted by redundancy "
            f"({probe[0].redundancy:.3f}..{probe[-1].redundancy:.3f})"
        )
    return probe


def build_demote_arms(args, cfg, bundle, extra_latents, stem: str, native) -> list:
    """(key, latent, rope-patch factory | None) per demote arm of one image.

    The RoPE-aligned arms reuse the demoted latent unchanged — only the
    position grid moves: per-axis stretch in patch units, demoted patch ``i``
    sitting at ``i * (native_patches / demoted_patches)`` so relative phases
    span the native coordinate range.
    """
    arms: list = []
    for e in cfg.edges:
        lat = extra_latents[(stem, f"demote{e}")]
        arms.append((str(e), lat, None))
        if args.repromote:
            arms.append((f"{e}rp", extra_latents[(stem, f"repromote{e}")], None))
        hs = (native.shape[-2] // 2) / (lat.shape[-2] // 2)
        ws = (native.shape[-1] // 2) / (lat.shape[-1] // 2)
        if args.pi_align:
            arms.append((f"{e}pi", lat, partial(pi_rope, bundle.anima, hs, ws)))
        if cfg.yarn_bands:
            a, b_ = cfg.yarn_bands
            arms.append(
                (f"{e}yarn", lat, partial(yarn_rope, bundle.anima, hs, ws, a, b_))
            )
            if cfg.yarn_gate:
                arms.append(
                    (
                        f"{e}yarnsig",
                        lat,
                        partial(yarn_rope, bundle.anima, hs, ws, a, b_, cfg.yarn_gate),
                    )
                )
    return arms


def main() -> None:
    args = parse_args(__doc__)
    if args.deterministic:  # must precede any CUDA/cublas init
        enable_deterministic()
    start_heartbeat()
    cfg = resolve_run_config(args)
    device = torch.device("cuda")
    sigmas = build_sigmas(
        args.bins, args.draws_per_bin, args.endpoint_bin, cfg.sigma_lo, cfg.sigma_hi
    )
    total_draws = int(sigmas.numel())

    probe = select_probe(args, cfg)
    log.info(
        f"probe set: {len(probe)} images, {len({r.artist for r in probe})} artists; "
        f"{args.bins} σ-bins × {args.draws_per_bin} draws × "
        f"{2 + int(cfg.reenc_control) + len(cfg.edges)} arms"
    )

    log.info(
        f"encoding demoted arms ({cfg.edges}) + reenc={cfg.reenc_control}"
        f" + repromote={args.repromote}…"
    )
    extra_latents = encode_probe_latents(
        probe, cfg.edges, args.vae, device, cfg.reenc_control, repromote=args.repromote
    )
    if args.x_zero:
        # keep the exact demoted grid shapes, drop all content
        extra_latents = {k: torch.zeros_like(v) for k, v in extra_latents.items()}

    bundle = build_probe_bundle(args, probe, extra_latents)

    groups: dict[str, list[tuple[int, int]]] | None = None
    if args.per_group:
        groups = build_groups(bundle.network)
        n_type = sum(1 for g in groups if g.startswith("type:"))
        n_block = sum(1 for g in groups if g.startswith("block:"))
        log.info(f"per-group: {n_type} type groups + {n_block} block groups")

    centers = [round(float(s), 4) for s in sigmas.mean(dim=1)]
    run_dir = make_run_dir(
        "sigma_lowres",
        args.label,
        root=Path(args.results_root).resolve()
        if args.results_root
        else Path(__file__).resolve().parent / "results",
    )
    rows_path = run_dir / "per_image.jsonl"
    rows: list[dict] = []
    arm_keys = build_arm_keys(args, cfg, total_draws)
    arm_sums = ArmSumAccumulator(run_dir / "arm_sums") if args.keep_arm_sums else None
    pool_strata: list[dict] = []
    pool_spill = run_dir / "pool_agg_spill"
    pool_agg = PoolAccumulator(backing_dir=pool_spill)
    cur_pool = PoolAccumulator()

    def finalize_stratum() -> None:
        if not args.pool or cur_pool.n == 0:
            return
        stats = pool_stats(cur_pool, arm_keys)
        pool_strata.append(stats)
        pool_agg.merge(cur_pool)
        pool_agg.release()  # drop memmap handles — RSS-free between merges
        cur_pool.__init__()
        gaps = " ".join(f"gap_{k}@last={stats[f'gap_{k}'][-1]:+.4f}" for k in arm_keys)
        log.info(
            f"[pool s{len(pool_strata) - 1}] n={stats['n_images']} "
            f"redundancy {stats['redundancy_range'][0]:.2f}"
            f"-{stats['redundancy_range'][1]:.2f} "
            f"floor@last={stats['cos_floor'][-1]:.4f} {gaps}"
        )

    t0 = time.time()
    # single worker: per-arm stat jobs are serial among themselves (cheap
    # relative to an arm's forwards) — the point is overlap with the GPU
    stats_pool = ThreadPoolExecutor(max_workers=1)

    for i, r in enumerate(probe):
        crossattn, _ = load_cached_text_features(r.te_path, variant=0)
        if crossattn is None:
            log.info(f"  [{i}] {r.artist}/{r.stem}: no crossattn_emb — skipped")
            continue
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        native = load_cached_latents(r.npz_path)[0]
        if args.x_zero:
            native = torch.zeros_like(native)

        def seeds(arm_idx: int, alt: bool = False) -> list[int]:
            # alt (self-floor second draw set): +500_000 keeps the base
            # disjoint from every primary base (i*10_000 < 500_000 for
            # i < 50) without shifting any primary seed vs. earlier runs
            base = (
                args.seed * 1_000_000
                + (500_000 if alt else 0)
                + i * 10_000
                + arm_idx * 1_000
            )
            return [base + d for d in range(total_draws)]

        def estimate(lat_, seed_list, rope_patch=None, alpha=1.0):
            bd = 1
            if args.draw_batch_tokens:
                grid_tokens = (lat_.shape[-2] // 2) * (lat_.shape[-1] // 2)
                bd = max(1, args.draw_batch_tokens // grid_tokens)
                bd = 1 << (bd.bit_length() - 1)  # pow2 → bounded graph count
            return grad_estimate_binned(
                bundle,
                lat_,
                crossattn,
                sigmas,
                seed_list,
                rope_patch=rope_patch,
                prefix_draws=cfg.sweep,
                batch_draws=bd,
                target_alpha=alpha,
            )

        row = {
            "artist": r.artist,
            "stem": r.stem,
            "redundancy": round(r.redundancy, 4),
            "tokens_native": r.tokens,
            "sigma_centers": centers,
        }
        if cfg.probe_tags is not None:
            row.update(cfg.probe_tags[(r.artist, r.stem)])
        if cfg.sweep:
            row["draw_prefixes"] = cfg.sweep

        # --target_alpha: one full pass per alpha. arm_idx (and therefore the
        # seeds) resets per alpha, so every alpha sees the SAME noise draws —
        # the alpha-slope is free of draw noise, and the seed-space bounds
        # (arm_idx * 1_000 < i * 10_000 spacing) are preserved.
        kap0: dict[str, list[torch.Tensor]] = {}
        for alpha_ in cfg.alphas:
            sfx = "" if alpha_ == 1.0 else f"@a{alpha_:g}"
            g_a, n_a = estimate(native, seeds(0), alpha=alpha_)
            g_b, n_b = estimate(native, seeds(1), alpha=alpha_)
            floor = [cosine(a, b) for a, b in zip(g_a, g_b)]
            row[f"cos_floor{sfx}"] = [round(c, 5) for c in floor]
            row[f"gnorm_native{sfx}"] = [
                round(0.5 * (x + y), 3) for x, y in zip(n_a, n_b)
            ]

            floor_g: list[dict[str, float]] | None = None
            if groups:
                floor_g = [grouped_cosine(a, b, groups) for a, b in zip(g_a, g_b)]
                row["cosg_floor"] = {
                    g: [round(fb[g], 5) for fb in floor_g] for g in groups
                }
            statter = ArmStatter(g_a, g_b, floor, groups=groups, floor_g=floor_g)

            stat_futs = []
            arms: dict[str, list[torch.Tensor]] = {"a": g_a, "b": g_b}
            arm_idx = 2
            if cfg.reenc_control:
                re_lat = extra_latents[(r.stem, "reenc")]
                g_re, _ = estimate(re_lat, seeds(arm_idx), alpha=alpha_)
                g_re2 = None
                if args.self_floor:
                    g_re2, _ = estimate(re_lat, seeds(arm_idx, alt=True), alpha=alpha_)
                    arms["reenc__2"] = g_re2
                stat_futs.append(
                    stats_pool.submit(statter.stats, "reenc" + sfx, g_re, None, g_re2)
                )
                arm_idx += 1
                arms["reenc"] = g_re
            for key, lat, patch in build_demote_arms(
                args, cfg, bundle, extra_latents, r.stem, native
            ):
                g_d, n_d = estimate(lat, seeds(arm_idx), rope_patch=patch, alpha=alpha_)
                g_d2 = None
                if args.self_floor:
                    g_d2, _ = estimate(
                        lat, seeds(arm_idx, alt=True), rope_patch=patch, alpha=alpha_
                    )
                    arms[f"{key}__2"] = g_d2
                stat_futs.append(
                    stats_pool.submit(statter.stats, key + sfx, g_d, n_d, g_d2)
                )
                arm_idx += 1
                arms[key] = g_d
            for fut in stat_futs:  # join stats before pooling/freeing this image
                row.update(fut.result())
            if args.pool:
                cur_pool.add_image(arms, r.redundancy)
                if cur_pool.n == args.pool:
                    finalize_stratum()
            if arm_sums is not None:
                for k, v in arms.items():
                    arm_sums.add(k + sfx, v)
            if args.target_kappa and alpha_ == 0.0:
                # tensors survive the free below (only the LISTS are cleared)
                kap0 = {k: list(v) for k, v in arms.items()}
            if args.target_kappa and alpha_ == 1.0 and kap0:
                row.update(kappa_row(arms, kap0, args.draws_per_bin))
                kap0 = {}
            # free this image's ~8 GB of flat gradient vectors now — otherwise
            # the locals keep them resident through the next image's compute
            for vecs in arms.values():
                vecs.clear()
            g_a = g_b = arms = statter = None  # noqa: F841
        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        gap_str = " ".join(
            f"{k}={[f'{v:+.3f}' for v in row[k]]}" for k in row if k.startswith("gap_")
        )
        log.info(f"  [{i + 1}/{len(probe)}] {r.artist}/{r.stem} {gap_str}")

    if not rows:
        raise SystemExit("no per-image rows produced")
    finalize_stratum()  # remainder stratum (may be smaller than --pool)

    headline = build_headline(
        rows,
        args,
        arm_keys=arm_keys,
        edges=cfg.edges,
        alphas=cfg.alphas,
        sweep=cfg.sweep,
        centers=centers,
        wall_time_s=time.time() - t0,
        pool_strata=pool_strata,
        pool_agg=pool_agg,
    )
    if args.pool and pool_strata:
        del pool_agg  # release memmap handles before removing the spill
        shutil.rmtree(pool_spill, ignore_errors=True)

    if arm_sums is not None:
        arm_sums.finalize(
            {
                "adapter": args.adapter,
                "sigma_centers": centers,
                "bins": args.bins,
                "endpoint_bin": args.endpoint_bin,
                "draws_per_bin": args.draws_per_bin,
                "target_alphas": cfg.alphas,
                "arm_keys": arm_keys,
                "self_floor": args.self_floor,
                "repromote": args.repromote,
                "demote_edges": args.demote_edges,
                "n_images": len(rows),
                "seed": args.seed,
            }
        )
        log.info(f"arm sums → {run_dir / 'arm_sums'} ({len(arm_sums.maps)} vectors)")

    log.info(json.dumps(headline, indent=2))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=[rows_path],
        extra={"probe_set": [f"{r['artist']}/{r['stem']}" for r in rows]},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
