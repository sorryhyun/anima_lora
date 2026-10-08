#!/usr/bin/env python3
"""Holdout-CMMD generalization probe — does adapter X generalize better than Y?

The quantitative companion to the eyeball sheets: loss_gap.py measures
member-specific *memorization* in the weights; this measures *generalization*
— how close each adapter's renders of NEVER-TRAINED same-artist captions land
to the artist's real distribution.

Design (paired, like the trainer's CMMD validation):
  * Prompts: N captions from the sample_ratio COMPLEMENT (images no candidate
    adapter ever trained on — cleaner than a validation_split_num replay,
    which would straddle the member half). Same prompts, same per-item seeds,
    same sizes for every model → per-model deltas are model-driven.
  * Render at 20 steps / CFG 1.0 (the trainer's CMMD convention).
  * Score: PE-Core pooled embeddings → Gaussian MMD² (library.training.cmmd)
    against TWO reference pools of real images:
      cmmd_holdout  vs real holdout images  — the generalization headline
                     (lower = renders sit closer to the artist's unseen work)
      cmmd_member   vs real member images   — the memorization pull
                     (an adapter that xeroxes scores oddly LOW here relative
                      to cmmd_holdout: its "new" renders are member look-alikes)
  * `--with_base` anchors the scale: base model with no adapter = how far
    plain Anima sits from this artist (the gap every adapter is closing).
  * Noise floor: CMMD between two disjoint halves of the real holdout pool —
    model deltas smaller than this are not interpretable.

Membership is reconstructed from `--membership_adapter`'s snapshot (default:
the first adapter) with the same trainer-blueprint replay as loss_gap.py; all
compared adapters must share the membership recipe (same sample_ratio /
validation_seed / target artist), which holds for the bench_sincos_half_*
family by construction.

Usage (from anima_lora/)::

    uv run python bench/memorization/generalize.py \
        --adapters output/ckpt/bench_sincos_half_plain.safetensors \
                   output/ckpt/bench_sincos_half_uncondpoolft.safetensors \
        [--with_base] [--num_prompts 24] [--num_refs 96] [--label ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
)
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.memorization.eyeball import _caption_for, _pick  # noqa: E402
from bench.memorization.loss_gap import (  # noqa: E402
    _read_snapshot_knobs,
    build_items,
    setup_strategies,
)
from bench.memorization.probe import build_training_args  # noqa: E402
from library.inference import load_shared_models  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.training.cmmd import cmmd_from_pools, pool_and_normalize  # noqa: E402
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

GEN_SEED_BASE = 20_000  # held-out noise; disjoint from eyeball's 0..2 draws


@dataclasses.dataclass
class Prompt:
    stem: str
    caption: str
    ref_path: Path
    size_hw: tuple[int, int]


# ---------------------------------------------------------------------------
# Membership replay + prompt/reference selection
# ---------------------------------------------------------------------------


def split_items(args) -> tuple[list, list, str]:
    """(member_items, holdout_items, target_artist) via the trainer replay."""
    snap = _read_snapshot_knobs(args.membership_adapter)
    if snap:
        print(f"membership snapshot knobs: {snap}", flush=True)
    pattern = snap.get("path_pattern") or ""
    target = pattern.split("/")[0] if "/" in pattern else None
    if not target:
        raise RuntimeError("membership adapter snapshot has no path_pattern")

    targs = build_training_args(args.method, args.preset, {})
    for k in ("validation_seed", "validation_split", "validation_split_num"):
        if k in snap:
            setattr(targs, k, snap[k])
    targs.path_pattern = snap.get("path_pattern")
    te_strategy = setup_strategies(targs)
    member_items = build_items(
        targs,
        te_strategy,
        sample_ratio=snap.get("sample_ratio"),
        artists_shard=snap.get("artists_shard"),
    )
    full_items = build_items(
        targs, te_strategy, sample_ratio=1.0, artists_shard=None, include_val=True
    )
    member_paths = {it[0].absolute_path for it in member_items}

    def target_only(items):
        return [it for it in items if Path(it[0].absolute_path).parent.name == target]

    members = target_only(member_items)
    holdouts = [
        it for it in target_only(full_items) if it[0].absolute_path not in member_paths
    ]
    print(
        f"target {target!r}: members={len(members)} holdouts={len(holdouts)}",
        flush=True,
    )
    return members, holdouts, target


def to_prompts(items: list, n: int, rng: Random) -> list[Prompt]:
    out: list[Prompt] = []
    for it in _pick(items, n, rng):
        path = Path(it[0].absolute_path)
        caption = _caption_for(path)
        if not caption:
            continue
        with Image.open(path) as im:
            w, h = im.size
        out.append(Prompt(path.stem, caption, path, (h, w)))
    return out


# ---------------------------------------------------------------------------
# Generation: same prompts + seeds under every model
# ---------------------------------------------------------------------------


def render_model(
    args, label: str, adapter: str | None, prompts: list[Prompt], device
) -> list[torch.Tensor]:
    """Latents (CPU) for every prompt under one model. Fresh DiT/TE load."""
    ckpt = default_checkpoints()
    base_req = GenerationRequest(
        dit=ckpt.dit,
        vae=ckpt.vae,
        text_encoder=ckpt.text_encoder,
        prompt="placeholder",
        save_path="__unused__",
        infer_steps=args.steps,
        guidance_scale=args.cfg,
        seed=0,
        lora_weight=[adapter] if adapter else None,
        lora_multiplier=[args.multiplier] if adapter else None,
        extra_argv=("--compile_blocks",) if args.compile_blocks else (),
    )
    base_args = base_req.to_args()
    base_args.device = device
    gen_settings = get_generation_settings(base_args)
    shared_models = load_shared_models(base_args)
    shared_models["conds_cache"] = {}

    latents: list[torch.Tensor] = []
    for i, p in enumerate(prompts):
        pa = dataclasses.replace(
            base_req,
            prompt=p.caption,
            seed=GEN_SEED_BASE + i,  # identical across models — paired design
            image_size=p.size_hw,
        ).to_args()
        pa.device = device
        print(f"[{label}] {i + 1}/{len(prompts)} {p.stem}", flush=True)
        latents.append(generate(pa, gen_settings, shared_models).detach().to("cpu"))

    shared_models.clear()
    del gen_settings
    clean_memory_on_device(device)
    return latents


# ---------------------------------------------------------------------------
# Feature pools
# ---------------------------------------------------------------------------


def _pil_to_minus1to1(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img.convert("RGB"))).float()
    return (t / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def encode_pool(bundle, images: list[torch.Tensor]) -> torch.Tensor:
    """[-1,1] (1,3,H,W) tensors → stacked pooled+normalized PE features [N,D]."""
    pooled = []
    for img in images:
        feats = encode_pe_from_imageminus1to1(bundle, img.to(bundle.device))
        pooled.append(pool_and_normalize(feats[0]).to("cpu"))
    return torch.stack(pooled)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--with_base", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--membership_adapter",
        default=None,
        help="Snapshot to replay membership from (default: first adapter).",
    )
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--num_prompts", type=int, default=24)
    ap.add_argument("--num_refs", type=int, default=96, help="Per reference pool.")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--pick_seed", type=int, default=0)
    ap.add_argument(
        "--compile_blocks", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--label", default="generalize")
    args = ap.parse_args()
    args.membership_adapter = args.membership_adapter or args.adapters[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = Random(args.pick_seed)

    members, holdouts, target = split_items(args)
    if len(holdouts) < args.num_prompts:
        raise RuntimeError("not enough holdout items for the prompt set")

    prompts = to_prompts(holdouts, args.num_prompts, rng)
    # Reference pools DISJOINT from the prompt items (their captions drove the
    # gens; excluding them keeps cmmd_holdout an unseen-data statistic).
    prompt_paths = {p.ref_path for p in prompts}
    ref_holdout_items = [
        it for it in holdouts if Path(it[0].absolute_path) not in prompt_paths
    ]
    ref_hold = _pick(ref_holdout_items, args.num_refs, rng)
    ref_memb = _pick(members, args.num_refs, rng)
    print(
        f"prompts={len(prompts)} ref_holdout={len(ref_hold)} ref_member={len(ref_memb)}",
        flush=True,
    )

    models: list[tuple[str, str | None]] = []
    if args.with_base:
        models.append(("base", None))
    models += [(Path(a).stem, a) for a in args.adapters]

    # Phase 1 — DiT sampling per model (paired prompts/seeds).
    latents_by_model = {
        label: render_model(args, label, adapter, prompts, device)
        for label, adapter in models
    }

    # Phase 2 — one VAE pass for every latent.
    ckpt = default_checkpoints()
    vae = load_vae(
        ckpt.vae, device="cpu", disable_mmap=True, dtype=torch.bfloat16, eval=True
    )
    pixels_by_model = {
        label: [decode_latent(vae, lat, device).to("cpu") for lat in lats]
        for label, lats in latents_by_model.items()
    }
    del vae, latents_by_model
    clean_memory_on_device(device)

    # Phase 3 — one PE-Core pass for gens + both reference pools.
    bundle = load_pe_encoder(device)
    pools = {
        label: encode_pool(
            bundle, [px.unsqueeze(0) if px.dim() == 3 else px for px in pxs]
        )
        for label, pxs in pixels_by_model.items()
    }
    ref_hold_pool = encode_pool(
        bundle,
        [_pil_to_minus1to1(Image.open(it[0].absolute_path)) for it in ref_hold],
    )
    ref_memb_pool = encode_pool(
        bundle,
        [_pil_to_minus1to1(Image.open(it[0].absolute_path)) for it in ref_memb],
    )
    del bundle
    clean_memory_on_device(device)

    # Noise floor: real-vs-real across two disjoint holdout halves.
    half = len(ref_hold_pool) // 2
    floor = cmmd_from_pools(ref_hold_pool[:half], ref_hold_pool[half:])

    metrics: dict[str, dict[str, float]] = {}
    for label, pool in pools.items():
        metrics[label] = {
            "cmmd_holdout": cmmd_from_pools(ref_hold_pool, pool),
            "cmmd_member": cmmd_from_pools(ref_memb_pool, pool),
        }

    run_dir = make_run_dir("memorization", args.label)
    lines = [
        "# Holdout-CMMD generalization probe",
        "",
        f"- target artist: `{target}` · prompts {len(prompts)} (holdout captions, "
        f"paired seeds) · refs {len(ref_hold)}/{len(ref_memb)} (holdout/member)",
        f"- steps {args.steps} · cfg {args.cfg} · real-vs-real noise floor "
        f"**{floor:.4f}**",
        "",
        "| model | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |",
        "|---|---|---|---|",
    ]
    for label, m in metrics.items():
        lines.append(
            f"| {label} | {m['cmmd_holdout']:.4f} | {m['cmmd_member']:.4f} | "
            f"{m['cmmd_member'] - m['cmmd_holdout']:+.4f} |"
        )
    lines += [
        "",
        "Reading: lower `cmmd_holdout` = renders of unseen captions sit closer "
        "to the artist's real unseen work (generalization). A strongly negative "
        "Δ (gens closer to the MEMBER pool than the holdout pool) = the "
        "renders are member look-alikes — memorization pull. Deltas below the "
        "noise floor are not interpretable.",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={"noise_floor": floor, **{k: v for k, v in metrics.items()}},
        artifacts=[run_dir / "summary.md"],
        extra={"prompts": [p.stem for p in prompts]},
    )
    print(f"done → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
