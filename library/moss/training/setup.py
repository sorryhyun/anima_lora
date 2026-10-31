"""What ``train`` builds once before its loop, from the request and the samples."""

from __future__ import annotations

import math
from typing import Callable

import torch

from library.env import resolve_under_home
from library.moss.requests import TrainRequest
from library.moss.training.data import (
    NEG_SUFFIX,
    TEMPLATE_PREFIX,
    TEMPLATE_SUFFIX,
    Sample,
    delta_row_ids,
    load_neg,
    sample_captions,
)
from library.moss.training.ema import ema_decay
from library.moss.training.losses import edit_region, tone_projection

PRELOAD_LIMIT_GB = 16  # above this the samples are read from disk each step


def preload(samples: list[Sample], use_plain: bool) -> Callable:
    """``get(sample) → (latent, embeds, plain or None, ref or None)`` on CPU, resident
    up to ``PRELOAD_LIMIT_GB`` and read from disk per call above it."""

    def fetch(s):
        latent, embeds = s.load()
        alt = s.load_plain() if use_plain and s.load_plain is not None else None
        ref = s.load_ref() if s.load_ref is not None else None
        return latent, embeds, alt, ref

    # bf16 TE rows (S×4096×2 bytes) + fp32 latents (128×4 bytes per token).
    preload_bytes = sum(
        s.n_text * 8192 * (2 if use_plain and s.load_plain else 1)
        + (s.n_img + s.n_ref) * 512
        for s in samples
    )
    if preload_bytes <= PRELOAD_LIMIT_GB * 1024**3:
        resident = {s.stem: fetch(s) for s in samples}
        return lambda s: resident[s.stem]
    print(
        f"data ~{preload_bytes / 1024**3:.0f} GB > {PRELOAD_LIMIT_GB} GB: "
        "read from disk each step",
        flush=True,
    )
    return fetch


def _full_size_ref(s: Sample, latent, ref, what: str) -> None:
    if ref is None or ref.shape != latent.shape:
        raise SystemExit(
            f"{what} need a full-size reference (ref_scale 1); "
            f"{s.stem} has {None if ref is None else tuple(ref.shape)} "
            f"for latent {tuple(latent.shape)}"
        )


def edit_regions(
    req: TrainRequest, samples: list[Sample], get: Callable
) -> tuple[dict, float]:
    """``({stem: (h, w) bool}, region_weight at the last step)``; no regions when
    every loss multiplier is 1."""
    region_end = (
        req.region_weight if req.region_weight_end is None else req.region_weight_end
    )
    regions = {}
    if req.region_weight == 1.0 and region_end == 1.0 and req.rest_weight == 1.0:
        return regions, region_end
    for s in samples:
        latent, _, _, ref = get(s)
        _full_size_ref(s, latent, ref, "region_weight / rest_weight")
        regions[s.stem] = edit_region(
            latent,
            ref,
            req.region_threshold,
            req.region_min_tokens,
            req.region_dilate,
        )
    area = sorted(r.float().mean().item() for r in regions.values())
    print(
        f"edit region: x{req.region_weight}"
        f"{f' -> x{region_end}' if region_end != req.region_weight else ''}"
        f", rest x{req.rest_weight}, threshold "
        f"{req.region_threshold}, min {req.region_min_tokens} tokens, dilate "
        f"{req.region_dilate}; area median {100 * area[len(area) // 2]:.2f}% "
        f"({100 * area[0]:.2f}-{100 * area[-1]:.2f}%), "
        f"{sum(a == 0 for a in area)} empty",
        flush=True,
    )
    return regions, region_end


def tone_setup(
    req: TrainRequest, samples: list[Sample], get: Callable, device
) -> tuple[dict, torch.Tensor | None, torch.Tensor | None]:
    """``({stem: unedited tokens (h, w)}, Lab projection, its tone_b_weight form)``;
    empty when neither ``tone_weight`` nor ``tone_log`` is set."""
    rests, proj, proj_b = {}, None, None
    if not (req.tone_weight or req.tone_log):
        return rests, proj, proj_b
    for s in samples:
        latent, _, _, ref = get(s)
        _full_size_ref(s, latent, ref, "tone_weight / tone_log")
        rest = ~edit_region(
            latent,
            ref,
            req.region_threshold,
            req.region_min_tokens,
            req.region_dilate + req.tone_dilate,
        )
        if rest.any():
            rests[s.stem] = rest
    if req.tone_dirs:
        dirs = resolve_under_home(req.tone_dirs)
        proj = tone_projection(dirs).to(device)
        proj_b = tone_projection(dirs, req.tone_b_weight).to(device)
    share = sorted(r.float().mean().item() for r in rests.values())
    print(
        f"tone: x{req.tone_weight}{' (log only)' if not req.tone_weight else ''}, "
        f"pool {req.tone_pool}, dilate +{req.tone_dilate}, "
        f"{'Lab via ' + req.tone_dirs if req.tone_dirs else 'raw channels'}"
        f"{f' (b x{req.tone_b_weight} from {req.tone_b_start:g})' if req.tone_dirs and req.tone_b_weight != 1 else ''}"
        f"{f', added from {req.tone_start:g}' if req.tone_start else ''}; "
        f"{len(rests)}/{len(samples)} samples with unedited tokens, share median "
        f"{100 * share[len(share) // 2] if share else 0:.0f}%",
        flush=True,
    )
    return rests, proj, proj_b


def delta_rows(
    req: TrainRequest, samples: list[Sample], get: Callable
) -> dict[tuple[str, str], list[int]]:
    """``delta_rows``: the text rows, per sample and caption kind, the adapter may
    touch; empty at ``all``."""
    rows_of: dict[tuple[str, str], list[int]] = {}
    if req.delta_rows == "all":
        return rows_of
    if req.delta_rows == "trigger" and not req.trigger:
        raise SystemExit("delta_rows=trigger needs --trigger")
    if req.mod_rank:
        raise SystemExit("delta_rows applies to token rows; drop mod_rank")
    tok = None
    if req.delta_rows == "trigger":
        from library.moss.loader import load_tokenizer, trigger_rows

        tok = load_tokenizer(req.model_dir)
    found = {"full": 0, "plain": 0}
    for s in samples:
        if s.te_path is None:
            raise SystemExit(f"{s.stem}: delta_rows needs the cache's TE files")
        _, full, alt, _ = get(s)
        for kind, caption in sample_captions(s.te_path).items():
            if kind == "full":
                n = full.shape[0]
            elif kind == "plain":
                n = alt.shape[0] if alt is not None else s.load_plain().shape[0]
            elif kind == "empty":
                n = TEMPLATE_PREFIX + TEMPLATE_SUFFIX
            else:
                n = s.load_variant(int(kind[1:])).shape[0]
            hits = trigger_rows(tok, caption, req.trigger) if tok else []
            rows_of[s.stem, kind] = delta_row_ids(req.delta_rows, n, hits)
            if kind in found and hits:
                found[kind] += 1
    if req.delta_rows == "trigger":
        print(
            f"delta rows: trigger {req.trigger!r} found in {found['full']}/{len(samples)} "
            f"full and {found['plain']}/{sum(s.load_plain is not None for s in samples)} "
            "plain captions",
            flush=True,
        )
        if not found["full"]:
            raise SystemExit("the trigger is in none of the full captions")
    else:
        print(f"delta rows: {req.delta_rows}", flush=True)
    return rows_of


def load_negs(
    req: TrainRequest, samples: list[Sample]
) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """``preserve_weight``: each sample's trigger-stripped pair (:func:`load_neg`)."""
    negs: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    if req.preserve_weight <= 0:
        return negs
    for s in samples:
        if s.te_path is None:
            raise SystemExit(f"{s.stem}: preserve_weight needs the cache's TE files")
        negs[s.stem] = load_neg(s.te_path)
    print(
        f"preserve: x{req.preserve_weight:g} on the trigger-stripped caption "
        f"({NEG_SUFFIX} beside every TE file; full and plain steps)",
        flush=True,
    )
    return negs


def rank_pattern(req: TrainRequest, alpha: float) -> dict | None:
    """``mod_rank``'s adapter on ``shared_block_modulation``."""
    if not req.mod_rank:
        return None
    if req.algo != "lora":
        raise SystemExit(f"mod_rank is lora only, not {req.algo}")
    mod_alpha = req.mod_alpha
    if mod_alpha is None:
        mod_alpha = alpha * math.sqrt(req.mod_rank / req.rank)
    return {"shared_block_modulation": (req.mod_rank, mod_alpha)}


def adapter_detail(req: TrainRequest, network) -> str:
    """The adapter's init in the start-of-run log line."""
    detail = {
        "lora": f"{req.down_init} down"
        + (
            f" slice {req.svd_slice}"
            + (f"/{req.svd_slice_count} interleaved" if req.svd_slice_count else "")
            if req.svd_slice or req.svd_slice_count
            else ""
        )
        + (
            f", {req.svd_drop_loud} loud channel(s) dropped"
            if req.svd_drop_loud
            else ""
        ),
        "lokr": f"factor {req.factor}",
        "loha": "hadamard",
    }[req.algo]
    for pat, (r, a) in network.rank_pattern.items():
        detail += f"; {pat} rank {r} alpha {a:g}"
    return detail


def repa_setup(
    req: TrainRequest, samples: list[Sample], transformer
) -> tuple[object | None, dict]:
    """``(REPATerm, {stem: target})``; ``(None, {})`` at ``repa_weight`` 0."""
    if req.repa_weight <= 0:
        return None, {}
    from library.moss.repa import REPATerm, load_target, pe_sidecar

    targets = {}
    pe_root = resolve_under_home(req.repa_pe_root)
    for s in samples:
        if s.latent_path is None or s.latent_hw is None:
            continue
        h, w = s.latent_hw
        path = pe_sidecar(s.latent_path, s.stem, w * 16, h * 16, pe_root)
        if path is not None:
            targets[s.stem] = load_target(path, h, w)
    if not targets:
        raise SystemExit(
            f"repa_weight {req.repa_weight} but no sample has a PE-Spatial sidecar "
            f"under {pe_root} (make preprocess-pe-spatial)"
        )
    repa = REPATerm(transformer, req.repa_layer)
    print(
        f"REPA: block {req.repa_layer}, weight {req.repa_weight:g}, DoG-Gram target "
        f"on {len(targets)}/{len(samples)} samples",
        flush=True,
    )
    return repa, targets


def make_optimizer(
    req: TrainRequest, params: list
) -> tuple[torch.optim.Optimizer, str]:
    """``(optimizer, its name)``; muon only on ``lora``, whose matrices it needs."""
    name = req.optimizer
    if name == "muon" and req.algo != "lora":
        print(
            f"muon is measured on lora only; {req.algo} trains with adamw", flush=True
        )
        name = "adamw"
    if name != "muon":
        return torch.optim.AdamW(params, lr=req.lr), name
    # weight_decay = torch's AdamW default, so only the update rule changes.
    optimizer = torch.optim.Muon(
        params, lr=req.lr, weight_decay=0.01, adjust_lr_fn="match_rms_adamw"
    )
    print(f"muon on {len(params)} matrices", flush=True)
    return optimizer, name


def ema_plan(req: TrainRequest, updates: int) -> tuple[list[float], list[float]]:
    """``(spans, decays)``, ``ema_window``'s first."""
    try:
        extra = [float(x) for x in req.ema_extra.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(
            f"ema_extra must be comma-separated floats: {req.ema_extra!r}"
        ) from None
    spans = ([req.ema_window] if req.ema_window else []) + extra
    if not all(0.0 < s <= 1.0 for s in spans):
        raise SystemExit(f"EMA spans must be in (0, 1], got {spans}")
    return spans, [ema_decay(s, updates) for s in spans]


def run_metadata(
    req: TrainRequest,
    *,
    n_samples: int,
    total: int,
    image_tokens: str,
    n_plain: int,
    n_variant_samples: int,
    n_refs: int,
    optimizer: str,
    has_repa: bool,
    has_regions: bool,
    region_end: float,
) -> dict:
    """The run's LoRA metadata."""
    meta = {
        "base_variant": req.model,
        "lr": req.lr,
        "samples": n_samples,
        "steps": total,
        "image_tokens": image_tokens,
        "sigma": f"mu={req.sigma_mu} s={req.sigma_s} cutoff={req.sigma_cutoff} "
        f"uniform={req.sigma_uniform}",
        "caption_dropout": req.caption_dropout,
        "plain_caption_rate": req.plain_caption_rate if n_plain else 0.0,
        "caption_variant_rate": req.caption_variant_rate if n_variant_samples else 0.0,
        "reference_latents": n_refs,
    }
    if req.grad_accum > 1:
        meta["grad_accum"] = req.grad_accum
    meta["optimizer"] = optimizer
    if req.stream != "bf16":
        meta["stream"] = req.stream
    if req.delta_rows != "all":
        meta["delta_rows"] = req.delta_rows
    if req.trigger:
        meta["trigger"] = req.trigger
    if req.preserve_weight:
        meta["preserve_weight"] = req.preserve_weight
    if n_refs:
        meta["ref_timestep"] = req.ref_timestep
        if req.ref_isolate:
            meta["ref_isolate"] = "1"
    if has_repa:
        meta.update(repa_weight=req.repa_weight, repa_layer=req.repa_layer)
    if has_regions:
        meta.update(
            region_weight=req.region_weight,
            rest_weight=req.rest_weight,
            region_threshold=req.region_threshold,
            region_min_tokens=req.region_min_tokens,
            region_dilate=req.region_dilate,
        )
        if region_end != req.region_weight:
            meta["region_weight_end"] = region_end
    if req.tone_weight:
        meta.update(
            tone_weight=req.tone_weight,
            tone_pool=req.tone_pool,
            tone_dilate=req.tone_dilate,
            tone_dirs=req.tone_dirs or "raw",
        )
        if req.tone_dirs and req.tone_b_weight != 1:
            meta.update(tone_b_weight=req.tone_b_weight, tone_b_start=req.tone_b_start)
        if req.tone_start:
            meta.update(tone_start=req.tone_start)
    return meta
