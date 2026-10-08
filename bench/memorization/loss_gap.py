#!/usr/bin/env python3
"""Calibrated loss-gap memorization probe — has the LoRA overfit specific frames?

The WEIGHT-SIDE companion to ``bench/memorization/probe.py`` (which is
generation-side: does sampling actually reproduce training frames). This probe
asks the earlier, cheaper question: do the adapter's *weights* carry
member-specific information — before/without any xeroxing showing up in
samples? It needs only forward passes over already-cached latents + TE
embeddings (no sampling, no VAE, no PE encoder), so it's cheap enough to run
per-checkpoint and trace the ``tau_mem`` crossing curve the README asks for.

THE STATISTIC. Per image: re-noise its cached clean latent ``x0`` with known
noise to ``x_t=(1-s)x0+s*eps``, query the velocity field conditioned on the
image's own caption, recover ``eps_hat=v+x0``, and score the confidence
``c_s = -log(MSE(eps_hat,eps)+delta)`` over a sigma grid (K antithetic probes
per sigma, identical probe noise across models). This is the standard
loss-based membership-inference statistic for diffusion models (white-box MIA
arXiv:2308.06405, score-based MIA 2509.25003); fine-tuned/LoRA'd diffusion
models are its most vulnerable case (2601.21628, 2402.11989).

THE CALIBRATION (load-bearing).
Raw confidence tracks image *easiness* (flat/simple images score high — the
exact artifact that sank SOLACE-as-quality-reward), the known failure of naive
loss MIA (difficulty calibration: Watson et al. 2111.08440). Two paired
controls cancel it:

  1. ``delta_i = R_lora(x_i) - R_base(x_i)`` — same image under adapter vs
     base. Easiness affects both terms; the difference is the adapter-induced
     confidence gain.
  2. member vs HELD-OUT SAME-ARTIST images — a style LoRA legitimately raises
     ``delta`` on *everything* by that artist; only *members* should get an
     extra instance-specific bump. The gap
     ``mean(delta[member]) - mean(delta[same_artist_holdout])`` (and the
     member-vs-holdout AUC on ``delta``) is the memorization signal.

HOLDOUT MODES (resolved automatically from how the run was trained):
  * ``sample_ratio < 1`` runs: the deterministic shard complement (same
    ``validation_seed`` shuffle the trainer used) is a free same-artist
    holdout → full instance-memorization gate.
  * validation-split runs: val items are same-artist non-members → gate.
  * whole-shard runs (e.g. ``artists_shard`` with sample_ratio=1): every
    same-artist image is a member. Only the CROSS-ARTIST holdout exists →
    the probe reports style-fit (member vs other-artist delta) and refuses to
    call an instance-level verdict. Retrain with ``PRESET=half`` (or any
    sample_ratio<1) to run the real gate.

Membership is reconstructed by replaying the trainer's own blueprint machinery
(same deterministic ``validation_seed``-seeded subsample as
``library/datasets/dreambooth.py``), with the run's ``sample_ratio`` /
``artists_shard`` read from the adapter's ``.snapshot.toml`` when present.

READING IT
  * ``auc_member_vs_same`` ~0.5 ⇒ weights carry no instance-specific signal
    (generalizing); >0.65 with small permutation p ⇒ member-specific overfit —
    expect probe.py's contrast to climb at/after this point.
  * ``delta`` mean over holdout >> 0 with AUC ~0.5 is HEALTHY: the adapter
    fits the artist distribution, not the frames.
  * per-sigma breakdown shows where in the schedule the gap lives (literature:
    mid/low-noise carries most membership signal).

Usage (run from anima_lora/; cheap — forward passes only)::

    uv run python bench/memorization/loss_gap.py \
        --adapter output/ckpt/my_artist_lora.safetensors \
        --method lora --preset default --label my_artist

Run it across step-count checkpoints of one recipe to get the weight-side
``tau_mem`` curve; confirm any rise with ``probe.py``'s contact sheet (a loss
gap can exist without extractable copying — this is the early warning, not
the conviction).
"""

from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._anima import build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.memorization.probe import _Acc, build_training_args  # noqa: E402
from library.anima import strategy as strategy_anima  # noqa: E402
from library.anima.strategy import AnimaLatentsCachingStrategy  # noqa: E402
from library.anima.uncond import (  # noqa: E402
    default_uncond_path,
    load_uncond_crossattn,
)
from library.runtime.device import clean_memory_on_device, str_to_dtype  # noqa: E402
from library.training.validation import (  # noqa: E402
    _build_val_crossattn_emb,
    _load_safetensors,
)

DEFAULT_SIGMAS = [0.3, 0.5, 0.7, 0.9]
_SNAPSHOT_KNOBS = (
    "sample_ratio",
    "artists_shard",
    "validation_seed",
    "validation_split",
    "validation_split_num",
    "path_pattern",
)


def _read_snapshot_knobs(adapter: str) -> dict:
    """Membership-relevant knobs from the run's ``.snapshot.toml`` (the run's
    ground truth — method TOMLs mutate over time). Missing file → {}."""
    import tomllib

    snap = Path(adapter).with_suffix(".snapshot.toml")
    if not snap.exists():
        return {}
    with snap.open("rb") as f:
        raw = tomllib.load(f)
    return {k: raw[k] for k in _SNAPSHOT_KNOBS if k in raw}


def setup_strategies(targs):
    """Install the trainer's strategy singletons ONCE (they refuse re-set)
    and return the TE caching strategy build_items needs."""
    strategy_anima.setup_training_strategies(targs)
    te_strategy = strategy_anima.setup_text_encoder_outputs_caching_strategy(targs)
    if te_strategy is None:
        raise RuntimeError(
            "cache_text_encoder_outputs is off — the probe reads cached TE "
            "outputs and cannot run in live-encoding mode."
        )
    return te_strategy


def build_items(
    targs,
    te_strategy,
    *,
    sample_ratio,
    artists_shard,
    path_pattern=None,
    include_val: bool = False,
) -> list[tuple]:
    """(info, te_npz, latents_npz) triples from the trainer's own blueprint,
    with train.py's user_config mutations (global sample_ratio injection +
    artists_shard expansion + path_pattern subset filter) replayed so the item
    set is exactly what a run with those knobs trains on. ``include_val`` folds
    the validation split back in (for the full non-member pool)."""
    from library.config import loader as config_util  # noqa: PLC0415
    from library.config.io import load_dataset_config_from_base  # noqa: PLC0415

    lat_strategy = AnimaLatentsCachingStrategy(True, 1, True)

    # BlueprintGenerator falls back to the argparse namespace for subset
    # defaults, so a preset-level `sample_ratio` in targs (e.g. [tenth])
    # would silently override the value this call asked for — pin BOTH the
    # namespace and the subset-level key to the requested ratio. Same for
    # path_pattern: a method-TOML value on the namespace would leak into the
    # subsets, so pin it to exactly what this call asked for.
    targs.sample_ratio = 1.0 if sample_ratio is None else float(sample_ratio)
    targs.path_pattern = path_pattern

    user_config = load_dataset_config_from_base(
        overrides=vars(targs),
        method=getattr(targs, "method", None),
        methods_subdir=getattr(targs, "methods_subdir", None) or "methods",
    )
    if user_config is None:
        raise RuntimeError("No dataset blueprint found in configs/base.toml.")

    # Mirror train.py's pre-blueprint mutations (train.py:~1420).
    if sample_ratio is not None and sample_ratio != 1.0:
        for ds in user_config.get("datasets", []):
            for sub in ds.get("subsets", []):
                sub["sample_ratio"] = sample_ratio
    if path_pattern:
        for ds in user_config.get("datasets", []):
            for sub in ds.get("subsets", []):
                sub["path_pattern"] = path_pattern
    if artists_shard:
        from library.datasets.artist_shard import apply_artist_shard  # noqa: PLC0415
        from library.env import resolve_under_home  # noqa: PLC0415

        apply_artist_shard(user_config, artists_shard, resolve=resolve_under_home)

    blueprint_generator = config_util.BlueprintGenerator(
        config_util.ConfigSanitizer(support_dropout=True)
    )
    blueprint = blueprint_generator.generate(user_config, targs)
    train_group, val_group = config_util.generate_dataset_group_by_blueprint(
        blueprint.dataset_group, target_res=getattr(targs, "target_res", None)
    )
    groups = [g for g in (train_group, val_group if include_val else None) if g]

    items: list[tuple] = []
    for grp in groups:
        for ds in grp.datasets:
            for info in ds.image_data.values():
                subset = ds.image_to_subset.get(info.image_key)
                te_npz = te_strategy.get_outputs_npz_path(
                    info.absolute_path,
                    cache_dir=(
                        getattr(subset, "text_cache_dir", None)
                        or getattr(subset, "cache_dir", None)
                    ),
                    image_dir=getattr(subset, "image_dir", None),
                )
                lat_npz = lat_strategy.get_latents_npz_path(
                    info.absolute_path,
                    info.image_size,
                    cache_dir=getattr(subset, "cache_dir", None),
                    image_dir=getattr(subset, "image_dir", None),
                )
                if Path(te_npz).exists() and Path(lat_npz).exists():
                    items.append((info, te_npz, lat_npz))
    return items


def _load_z0(lat_strategy, lat_npz: str, bucket_reso, device, dtype) -> torch.Tensor:
    """Cached latent → (1,C,1,h,w) in the DiT's 5D layout (T singleton at dim 2)."""
    lat, _os, _crop, _flip, _alpha = lat_strategy.load_latents_from_disk(
        lat_npz, tuple(bucket_reso)
    )
    t = torch.from_numpy(np.asarray(lat))
    if t.dim() == 4:  # (C,1,h,w) — cached with the temporal singleton
        t = t.squeeze(1)
    assert t.dim() == 3, f"unexpected cached latent shape {tuple(t.shape)}"
    return t[None, :, None].to(device, dtype)  # (1,C,1,h,w)


@torch.no_grad()
def confidence(
    anima,
    z0: torch.Tensor,  # (1,C,1,h,w)
    emb: torch.Tensor,  # (1,512,D)
    sigmas: list[float],
    *,
    n_probes: int,
    delta: float,
    seed: int,
    device,
    dtype,
) -> dict[float, float]:
    """{sigma: -log(MSE+delta)} — the SOLACE/MIA re-noise statistic. Probe
    noise is seeded per-image so base and adapter passes see IDENTICAL eps
    (paired comparison; the delta is purely the model change)."""
    h, w = z0.shape[-2], z0.shape[-1]
    pad = torch.zeros(1, 1, h, w, dtype=dtype, device=device)
    z0_f = z0.float()
    g = torch.Generator(device=device).manual_seed(seed)

    n_base = (n_probes + 1) // 2
    base_draws = [
        torch.randn(z0.shape, generator=g, device=device, dtype=dtype)
        for _ in range(n_base)
    ]
    probes: list[torch.Tensor] = []
    for e in base_draws:
        probes.append(e)
        if len(probes) < n_probes:
            probes.append(-e)

    out: dict[float, float] = {}
    for s in sigmas:
        t_b = torch.full((1,), float(s), device=device, dtype=dtype)
        mses = []
        for eps in probes:
            z_t = ((1.0 - s) * z0_f + s * eps.float()).to(dtype)
            v = anima(z_t, t_b, emb, padding_mask=pad)
            eps_hat = v.float() + z0_f
            mses.append(float(((eps_hat - eps.float()) ** 2).mean().item()))
        out[s] = -float(np.log(float(np.mean(mses)) + delta))
    return out


def score_model(
    adapter: str | None,
    targs,
    shim,
    scored: list[dict],
    sigmas,
    *,
    n_probes,
    delta,
    dtype,
    null_captions: bool = False,
) -> list[dict[float, float]]:
    """Per-item {sigma: confidence} under one model (adapter=None → base).
    ``null_captions`` swaps every item's caption embedding for the T5("")
    uncond sidecar — the caption-independent uncond-Δ statistic
    (docs/proposal/memorization_uncond_gap.md)."""
    bundle = build_anima(targs, adapter=adapter, train_mode=False)
    dit = bundle.anima
    device = bundle.device
    null_emb = (
        load_uncond_crossattn(str(default_uncond_path()), device=device, dtype=dtype)
        if null_captions
        else None
    )
    lat_strategy = AnimaLatentsCachingStrategy(True, 1, True)
    results: list[dict[float, float]] = []
    try:
        if hasattr(dit, "prepare_block_swap_before_forward"):
            dit.prepare_block_swap_before_forward()
        for n, it in enumerate(scored):
            z0 = _load_z0(lat_strategy, it["lat_npz"], it["bucket_reso"], device, dtype)
            if null_emb is not None:
                sd = None
                emb = null_emb
            else:
                sd = _load_safetensors(it["te_npz"])
                emb = _build_val_crossattn_emb(dit, sd, shim)
            results.append(
                confidence(
                    dit,
                    z0,
                    emb,
                    sigmas,
                    n_probes=n_probes,
                    delta=delta,
                    seed=zlib.crc32(it["stem"].encode()) & 0x7FFFFFFF,
                    device=device,
                    dtype=dtype,
                )
            )
            del z0, emb, sd
            if (n + 1) % 16 == 0:
                print(
                    f"    {n + 1}/{len(scored)} scored "
                    f"({'adapter' if adapter else 'base'})",
                    flush=True,
                )
    finally:
        del bundle, dit
        clean_memory_on_device(shim.device)
    return results


# ---------------------------------------------------------------------------
# stats (no scipy dep)
# ---------------------------------------------------------------------------


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5*P(tie)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # midranks for ties
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _perm_p(pos: np.ndarray, neg: np.ndarray, rng, n_perm: int = 10000) -> float:
    """One-sided permutation p for mean(pos) - mean(neg) > 0."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    obs = pos.mean() - neg.mean()
    pool = np.concatenate([pos, neg])
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if pool[: pos.size].mean() - pool[pos.size :].mean() >= obs:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--adapter", required=True, help="LoRA checkpoint to probe.")
    ap.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="The RUN's sample_ratio (default: adapter .snapshot.toml, else "
        "merged config). Membership reconstruction depends on it.",
    )
    ap.add_argument(
        "--artists_shard",
        default=None,
        help="The RUN's artists_shard ('none' to force off; default: snapshot, "
        "else merged config).",
    )
    ap.add_argument(
        "--path_pattern",
        default=None,
        help="The RUN's path_pattern ('none' to force off; default: snapshot, "
        "else merged config). Filters the member pool only — holdout groups "
        "always come from the full unfiltered pool.",
    )
    ap.add_argument(
        "--null_captions",
        action="store_true",
        help="Score every item with the T5('') uncond sidecar instead of its "
        "own caption (uncond-Δ: caption-independent memorization statistic — "
        "docs/proposal/memorization_uncond_gap.md). Same paired forwards, "
        "same probe noise; only the crossattn embedding changes.",
    )
    ap.add_argument("--num_members", type=int, default=48)
    ap.add_argument("--num_holdout", type=int, default=48, help="per holdout group.")
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument("--n_probes", type=int, default=4, help="K antithetic probes.")
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    dtype = str_to_dtype(args.dtype)
    rng = np.random.default_rng(args.seed)

    snap = _read_snapshot_knobs(args.adapter)
    if snap:
        print(f"snapshot knobs: {snap}", flush=True)
    targs = build_training_args(args.method, args.preset, {})
    # Pin the run's split determinism onto the probe's namespace.
    for k in ("validation_seed", "validation_split", "validation_split_num"):
        if k in snap:
            setattr(targs, k, snap[k])
    # When a snapshot exists it is authoritative INCLUDING absences — a knob
    # missing from the snapshot means the run didn't use it, so we must NOT
    # fall back to today's method TOML (which may have gained it since).
    if args.sample_ratio is not None:
        run_sr = args.sample_ratio
    elif snap:
        run_sr = snap.get("sample_ratio")
    else:
        run_sr = getattr(targs, "sample_ratio", None)
    if args.artists_shard is not None:
        run_shard = None if args.artists_shard.lower() == "none" else args.artists_shard
    elif snap:
        run_shard = snap.get("artists_shard")
    else:
        run_shard = getattr(targs, "artists_shard", None)
    if args.path_pattern is not None:
        run_pp = None if args.path_pattern.lower() == "none" else args.path_pattern
    elif snap:
        run_pp = snap.get("path_pattern")
    else:
        run_pp = getattr(targs, "path_pattern", None)
    if run_pp in ("*", ""):
        run_pp = None  # matches everything — same as unset

    from library.runtime.accelerator import prepare_accelerator  # noqa: PLC0415

    acc = prepare_accelerator(targs)
    shim = _Acc(acc.device, dtype)

    # ── membership reconstruction ──────────────────────────────────────────
    print(
        f"members: replaying blueprint with sample_ratio={run_sr} "
        f"artists_shard={run_shard} path_pattern={run_pp}",
        flush=True,
    )
    te_strategy = setup_strategies(targs)
    member_items = build_items(
        targs,
        te_strategy,
        sample_ratio=run_sr,
        artists_shard=run_shard,
        path_pattern=run_pp,
    )
    full_items = build_items(
        targs, te_strategy, sample_ratio=1.0, artists_shard=None, include_val=True
    )
    if not member_items:
        raise RuntimeError("member set is empty — check --method/--preset/knobs.")

    member_paths = {it[0].absolute_path for it in member_items}
    member_dirs = {Path(it[0].absolute_path).parent for it in member_items}
    same_holdout = [
        it
        for it in full_items
        if it[0].absolute_path not in member_paths
        and Path(it[0].absolute_path).parent in member_dirs
    ]
    cross_holdout = [
        it
        for it in full_items
        if it[0].absolute_path not in member_paths
        and Path(it[0].absolute_path).parent not in member_dirs
    ]
    gate_possible = len(same_holdout) >= 8
    print(
        f"members={len(member_items)} same-artist holdout={len(same_holdout)} "
        f"cross-artist holdout={len(cross_holdout)} → "
        f"{'INSTANCE GATE' if gate_possible else 'STYLE-FIT ONLY'}",
        flush=True,
    )

    def pick(items: list, n: int) -> list:
        if len(items) <= n:
            return list(items)
        idx = sorted(rng.choice(len(items), n, replace=False))
        return [items[int(i)] for i in idx]

    scored: list[dict] = []
    for group, pool, n in (
        ("member", member_items, args.num_members),
        ("same_artist", same_holdout, args.num_holdout),
        ("cross_artist", cross_holdout, args.num_holdout),
    ):
        for info, te_npz, lat_npz in pick(pool, n):
            scored.append(
                {
                    "group": group,
                    "stem": Path(info.absolute_path).stem,
                    "artist": Path(info.absolute_path).parent.name,
                    "bucket_reso": tuple(info.bucket_reso),
                    "te_npz": te_npz,
                    "lat_npz": lat_npz,
                }
            )
    print(
        f"scoring {len(scored)} items x 2 models x {len(args.sigmas)} sigmas "
        f"x K={args.n_probes}",
        flush=True,
    )

    # ── the two paired passes ─────────────────────────────────────────────
    if args.null_captions:
        print("captions: T5('') uncond sidecar (uncond-Δ mode)", flush=True)
    print("base pass …", flush=True)
    base_scores = score_model(
        None,
        targs,
        shim,
        scored,
        args.sigmas,
        n_probes=args.n_probes,
        delta=args.delta,
        dtype=dtype,
        null_captions=args.null_captions,
    )
    print("adapter pass …", flush=True)
    lora_scores = score_model(
        args.adapter,
        targs,
        shim,
        scored,
        args.sigmas,
        n_probes=args.n_probes,
        delta=args.delta,
        dtype=dtype,
        null_captions=args.null_captions,
    )

    # ── stats ─────────────────────────────────────────────────────────────
    for it, cb, cl in zip(scored, base_scores, lora_scores):
        it["R_base"] = float(np.mean(list(cb.values())))
        it["R_lora"] = float(np.mean(list(cl.values())))
        it["delta"] = it["R_lora"] - it["R_base"]
        for s in args.sigmas:
            it[f"delta_s{s:g}"] = cl[s] - cb[s]

    def grp(name: str, key: str = "delta") -> np.ndarray:
        return np.array(
            [it[key] for it in scored if it["group"] == name], dtype=np.float64
        )

    d_mem, d_same, d_cross = grp("member"), grp("same_artist"), grp("cross_artist")
    metrics: dict = {
        "n_member": int(d_mem.size),
        "n_same_artist": int(d_same.size),
        "n_cross_artist": int(d_cross.size),
        "sigmas": list(args.sigmas),
        "n_probes": args.n_probes,
        "null_captions": bool(args.null_captions),
        "run_sample_ratio": run_sr,
        "run_artists_shard": run_shard,
        "delta_member_mean": round(float(d_mem.mean()), 5),
        "delta_cross_mean": (round(float(d_cross.mean()), 5) if d_cross.size else None),
        "auc_member_vs_cross": round(_auc(d_mem, d_cross), 4),
        # raw confidences for the flatness-confound sanity check
        "R_base_member_mean": round(
            float(np.mean([it["R_base"] for it in scored if it["group"] == "member"])),
            4,
        ),
    }
    per_sigma: dict = {}
    if d_same.size:
        gap = float(d_mem.mean() - d_same.mean())
        metrics.update(
            {
                "delta_same_mean": round(float(d_same.mean()), 5),
                "member_gap": round(gap, 5),
                "auc_member_vs_same": round(_auc(d_mem, d_same), 4),
                "perm_p_member_vs_same": round(
                    _perm_p(d_mem, d_same, np.random.default_rng(args.seed)), 5
                ),
            }
        )
        for s in args.sigmas:
            per_sigma[f"{s:g}"] = {
                "gap": round(
                    float(
                        grp("member", f"delta_s{s:g}").mean()
                        - grp("same_artist", f"delta_s{s:g}").mean()
                    ),
                    5,
                ),
                "auc": round(
                    _auc(
                        grp("member", f"delta_s{s:g}"),
                        grp("same_artist", f"delta_s{s:g}"),
                    ),
                    4,
                ),
            }
        metrics["per_sigma"] = per_sigma
        auc = metrics["auc_member_vs_same"]
        p = metrics["perm_p_member_vs_same"]
        if auc >= 0.65 and p < 0.05:
            verdict = (
                f"MEMBER-SPECIFIC OVERFIT — AUC {auc:.2f} (p={p:.4f}): the "
                "adapter's confidence gain is systematically larger on its own "
                "training frames than on held-out same-artist images. Weight-"
                "level memorization; confirm visible copying with probe.py."
            )
        elif auc >= 0.55:
            verdict = (
                f"WEAK member signal — AUC {auc:.2f} (p={p:.4f}). Watch the "
                "curve across checkpoints; not yet a conviction."
            )
        else:
            verdict = (
                f"NO weight-level memorization — AUC {auc:.2f} ≈ chance "
                f"(p={p:.4f}). The adapter fits the artist (delta on holdout "
                f"{d_same.mean():+.3f}), not the frames."
            )
    else:
        verdict = (
            "STYLE-FIT ONLY — no same-artist holdout exists for this run "
            f"(sample_ratio={run_sr}); every same-artist image is a member. "
            f"delta member {d_mem.mean():+.3f} vs cross-artist "
            f"{d_cross.mean():+.3f} measures style specificity, NOT instance "
            "memorization. Train with sample_ratio<1 (PRESET=half) to run the "
            "instance gate."
        )
    metrics["verdict"] = verdict
    print("\n" + verdict, flush=True)

    # ── artifacts ─────────────────────────────────────────────────────────
    run_dir = make_run_dir("memorization", label=args.label or "loss-gap")
    fields = [
        "group",
        "stem",
        "artist",
        "R_base",
        "R_lora",
        "delta",
        *[f"delta_s{s:g}" for s in args.sigmas],
    ]
    with (run_dir / "per_item.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for it in scored:
            w.writerow({k: it[k] for k in fields})

    _write_hist(run_dir / "delta_hist.png", d_mem, d_same, d_cross)

    M = ["# Loss-gap memorization probe (calibrated MIA statistic)\n"]
    M.append(f"- adapter: `{args.adapter}`")
    if args.null_captions:
        M.append(
            "- captions: **T5('') uncond sidecar** — this is the uncond-Δ "
            "(caption-independent) statistic, not the caption-Δ."
        )
    M.append(
        f"- members {d_mem.size} / same-artist {d_same.size} / "
        f"cross-artist {d_cross.size} · sigmas {args.sigmas} · K={args.n_probes}"
    )
    M.append(f"\n## Verdict\n\n**{verdict}**\n")
    M.append("| group | mean delta (R_lora − R_base) |")
    M.append("|---|---|")
    M.append(f"| member | {d_mem.mean():+.4f} |")
    if d_same.size:
        M.append(f"| same-artist holdout | {d_same.mean():+.4f} |")
    if d_cross.size:
        M.append(f"| cross-artist holdout | {d_cross.mean():+.4f} |")
    if d_same.size:
        M.append(
            f"\nmember gap {metrics['member_gap']:+.4f} · "
            f"AUC {metrics['auc_member_vs_same']:.3f} · "
            f"perm p {metrics['perm_p_member_vs_same']:.4f}\n"
        )
        M.append("| sigma | gap | AUC |")
        M.append("|---|---|---|")
        for s in args.sigmas:
            ps = per_sigma[f"{s:g}"]
            M.append(f"| {s:g} | {ps['gap']:+.4f} | {ps['auc']:.3f} |")
    M.append(
        "\nRaw R is easiness-confounded by design — only the paired delta and "
        "the member-vs-holdout comparison are interpretable "
        "(see `project_solace_confidence_is_flatness`).\n"
    )
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["summary.md", "per_item.csv", "delta_hist.png"],
        device=acc.device,
    )
    print(f"\n  -> {run_dir}", flush=True)


def _write_hist(path: Path, d_mem, d_same, d_cross) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (histogram skipped: {e})", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for arr, name in (
        (d_mem, "member"),
        (d_same, "same-artist"),
        (d_cross, "cross-artist"),
    ):
        if arr.size:
            ax.hist(arr, bins=24, alpha=0.55, label=f"{name} (n={arr.size})")
    ax.set_xlabel("delta = R_lora − R_base (adapter confidence gain)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
