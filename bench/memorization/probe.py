#!/usr/bin/env python3
"""Phase 0 memorization probe — does a LoRA reproduce its training images?

Motivated by Bonnaire/Urfin/Biroli/Mézard, *"Why Diffusion Models Don't
Memorize: The Role of Implicit Dynamical Regularization in Training"*
(NeurIPS 2025, arXiv:2505.17638). Their picture: training a score model has two
well-separated timescales — ``tau_gen`` (novel high-quality samples appear,
~capacity-bound, n-independent) and ``tau_mem`` (the model starts reproducing
training images, grows ~linearly with n). Early-stopping inside ``[tau_gen,
tau_mem]`` generalizes; train past ``tau_mem`` and you memorize. They quantify
memorization with a nearest-neighbour ratio (their Eq. 6): a generated sample is
"memorized" if ``||x - a_nn1|| / ||x - a_nn2|| < k`` (k=1/3) over the training
set in pixel L2.

**What it measures (and why not naive pixel-L2).** Pixel L2 over the whole set
(the paper's literal metric) is a verbatim-copy detector for unconditional
DMs. For a *conditional* style LoRA it conflates two things: (a) global style
similarity — a *non*-memorizing LoRA still emits same-style images, so every
generation is pixel/feature-close to all training images — and (b) genuine
instance reproduction. We separate them with a **self-vs-other contrast** in
PE feature space (PE-Spatial pooled by default — its 1024 patch tokens, mean-
pooled, encode composition, so a verbatim copy snaps to a near-identical vector
while same-style/different-layout images separate), the cheap primary signal:

  Condition generation on training caption ``i`` (its cached TE), draw a
  *held-out* noise seed, render ``gen_i``. Then in pooled-PE cosine space:
    s_self[i]  = cos(gen_i, train_i)              # closeness to its own source
    s_other[i] = max_{j != i} cos(gen_i, train_j) # nearest *other* training img
  A *generalizing* LoRA re-renders the caption as a plausible new image in the
  style: s_self ~= s_other (style affects all train images alike), so the
  contrast ``mean(s_self - s_other)`` sits near 0. A *memorizing* LoRA snaps
  back to the exact source: contrast >> 0 and the global nearest neighbour of
  ``gen_i`` is its own source ``i`` (``self_lock``). The floor is ~0 by
  construction — no separate baseline run needed (though ``--with_base`` adds
  the no-adapter contrast for an absolute anchor).

Reported metrics
----------------
  contrast_mean      mean(s_self - s_other)            — headline; ~0 generalize, >>0 memorize
  self_lock_frac     frac where argmax_j cos(gen_i, train_j) == i (+ margin)
  mem_frac_paper     paper Eq.6 NN-ratio over PE distances < k (in feature space)
  s_self / s_other   pooled distributions (mean, p50, p90)
  real_real_nn_mean  per-train nearest *other* real image cos — same-style floor
  pixel_mem_frac     (optional --pixel_nn) paper-faithful low-res gray pixel-L2 ratio

Artifacts
---------
  per_item.csv       one row per generated item (stem, s_self, s_other, nn1, flags)
  memorized.png      contact sheet: most-locked items, gen | nearest-train side by side
                     — the human "your LoRA copied this frame" check

This is Phase 0: a diagnostic, no training-loop wiring. It reuses the trainer's
own config chain / dataset split / PE encoder (same machinery as
``library/training/validation.py``), so the items and caches are exactly the ones the run saw — no re-preprocess needed.

Usage
-----
    python bench/memorization/probe.py \
        --method lora --preset default \
        --adapter output/ckpt/my_artist_lora.safetensors \
        --num_items 64 --label my_artist

Run it on several checkpoints of the same recipe (different step counts) to
watch the contrast climb as the run crosses ``tau_mem`` — that curve is the
early-stopping signal the paper argues for.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import train as train_mod  # noqa: E402
from library.config import schema as config_schema  # noqa: E402
from anima_lora import default_checkpoints, load_vae  # noqa: E402
from bench._anima import build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import strategy as strategy_anima  # noqa: E402
from library.anima import training as anima_train_utils  # noqa: E402
from library.runtime.accelerator import prepare_accelerator  # noqa: E402
from library.runtime.device import clean_memory_on_device, str_to_dtype  # noqa: E402
from library.training.cmmd import (  # noqa: E402
    load_reference_features,
    pool_and_normalize,
    resolve_pe_sidecar,
)
from library.training.validation import (  # noqa: E402
    _build_val_crossattn_emb,
    _load_safetensors,
)
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

# Held-out noise seeds: large base so the per-item draw (GEN_SEED_BASE + idx)
# never collides with a training step's noise. Memorization must show up under
# *fresh* noise, not the noise the image was cached against.
GEN_SEED_BASE = 7_000_000


# ---------------------------------------------------------------------------
# Trainer-faithful args + train-split items.
# ---------------------------------------------------------------------------


_SNAPSHOT_KNOBS = ("sample_ratio", "artists_shard", "path_pattern")


def _read_snapshot_knobs(adapter: str) -> dict:
    """Membership-relevant knobs from the run's ``.snapshot.toml`` (the run's
    ground truth — method TOMLs drift over time). Missing file → {}. Mirrors
    ``loss_gap._read_snapshot_knobs`` so both probes see the same pool."""
    import tomllib  # noqa: PLC0415

    snap = Path(adapter).with_suffix(".snapshot.toml")
    if not snap.exists():
        return {}
    with snap.open("rb") as f:
        raw = tomllib.load(f)
    return {k: raw[k] for k in _SNAPSHOT_KNOBS if k in raw}


def build_training_args(
    method: str, preset: str, overrides: dict
) -> argparse.Namespace:
    """The exact args namespace a real run would build, then probe overrides."""
    parser = train_mod.setup_parser()
    config_schema.populate_schema(parser, extras=train_mod.build_network_extras())
    # --no-config-snapshot so the merge doesn't write a stray .snapshot.toml.
    args = parser.parse_args(
        ["--method", method, "--preset", preset, "--no-config-snapshot"]
    )
    # argv=[] so the config-merge override re-parse does NOT read the process
    # sys.argv (which carries the probe's own --adapter/--num_items flags the
    # trainer parser would reject). Probe overrides are applied below instead.
    args = train_mod.read_config_from_file(args, parser, argv=[])
    args.method = method
    # build_anima reads args.dit; the trainer config chain stores the base model
    # under pretrained_model_name_or_path. Bridge it (falling back to the
    # canonical default_checkpoints path).
    if not getattr(args, "dit", None):
        args.dit = getattr(args, "pretrained_model_name_or_path", None) or (
            default_checkpoints().dit
        )
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    from library.config.cli_args import verify_training_args  # noqa: PLC0415

    verify_training_args(args)
    for k, v in overrides.items():
        if v is not None:
            setattr(args, k, v)
    args.sample_prompts = None
    return args


def build_train_items(
    args,
    *,
    pe_encoder: str = "pe_spatial",
    sample_ratio=None,
    artists_shard=None,
    path_pattern=None,
) -> list:
    """The *training* split items (info, te_npz, pe_sidecar) — memorization is
    about the images the model actually trained on. Same blueprint path as the
    trainer; we take the train group. TE + PE caches must already exist (no live encoding).

    ``pe_encoder`` selects which cached PE sidecar to use as the reference
    feature: ``pe_spatial`` (``{stem}_anima_pe_spatial.safetensors`` — what
    `make preprocess` writes for REPA, present on every artist set) or ``pe``
    (the pooled PE-Core global, only cached where CMMD ran).

    ``sample_ratio`` / ``artists_shard`` / ``path_pattern`` replay train.py's
    pre-blueprint subset mutations so the reference pool is exactly the images
    the adapter trained on (mirrors ``loss_gap.py``). This matters for
    single-artist adapters: without a ``path_pattern`` filter the pool is the
    whole multi-artist blueprint and the memorization signal is diluted — most
    generated captions are images the adapter never saw (see README)."""
    from library.config import loader as config_util  # noqa: PLC0415
    from library.config.io import load_dataset_config_from_base  # noqa: PLC0415

    strategy_anima.setup_training_strategies(args)
    te_strategy = strategy_anima.setup_text_encoder_outputs_caching_strategy(args)
    if te_strategy is None:
        raise RuntimeError(
            "cache_text_encoder_outputs is off — the probe reads cached TE/PE "
            "outputs and cannot run in live-encoding mode."
        )

    # Pin BOTH the namespace and (below) the subset-level keys, so a preset- or
    # method-TOML value can't silently override what the snapshot asked for.
    if sample_ratio is not None:
        args.sample_ratio = float(sample_ratio)
    args.path_pattern = path_pattern

    user_config = load_dataset_config_from_base(
        overrides=vars(args),
        method=getattr(args, "method", None),
        methods_subdir=getattr(args, "methods_subdir", None) or "methods",
    )
    if user_config is None:
        raise RuntimeError("No dataset blueprint found in configs/base.toml.")

    # Mirror train.py's pre-blueprint mutations (as loss_gap.build_items).
    if sample_ratio is not None and float(sample_ratio) != 1.0:
        for ds in user_config.get("datasets", []):
            for sub in ds.get("subsets", []):
                sub["sample_ratio"] = float(sample_ratio)
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
    blueprint = blueprint_generator.generate(user_config, args)
    train_group, _val_group = config_util.generate_dataset_group_by_blueprint(
        blueprint.dataset_group, target_res=getattr(args, "target_res", None)
    )
    if train_group is None:
        raise RuntimeError("Training split is empty — check the blueprint.")

    items: list = []
    for ds in train_group.datasets:
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
            info.text_encoder_outputs_npz = te_npz
            if not Path(te_npz).exists():
                continue
            pe_sidecar = resolve_pe_sidecar(
                info.absolute_path,
                encoder=pe_encoder,
                cache_dir=str(Path(te_npz).parent),
            )
            if not Path(pe_sidecar).exists():
                continue
            items.append((info, te_npz, pe_sidecar))
    return items


# ---------------------------------------------------------------------------
# Minimal Accelerator shim for the sampler (device + autocast).
# ---------------------------------------------------------------------------


class _Acc:
    def __init__(self, device, dtype):
        self.device = device
        self._dtype = dtype

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self._dtype)


def _token_budget(items: list):
    from library.datasets.buckets import token_counts_for_resos  # noqa: PLC0415

    resos = {tuple(info.bucket_reso) for (info, _t, _p) in items}
    counts = token_counts_for_resos(resos) if resos else None
    if not counts:
        return None, None
    return len(counts), (min(counts), max(counts))


# ---------------------------------------------------------------------------
# Generation: one held-out-seed sample per chosen training caption.
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_gen_pool(
    *,
    adapter: str | None,
    args,
    acc,
    vae,
    pe_bundle,
    items: list,
    gen_idx: list[int],
    sample_steps: int,
    cfg_scale: float,
    flow_shift: float,
    compile_budget=None,
):
    """Render gen_i for each chosen item, return (gen_pool[M,D], pixels[list])."""
    bundle = build_anima(args, adapter=adapter, train_mode=False)
    dit = bundle.anima
    device = bundle.device

    if compile_budget is not None:
        from library.runtime.harness import compile_blocks_for_training  # noqa: PLC0415

        n_families, seq_range = compile_budget
        compile_blocks_for_training(
            dit,
            bundle.network,
            backend=getattr(args, "dynamo_backend", "inductor"),
            mode=getattr(args, "compile_inductor_mode", None),
            n_token_families=n_families,
            seq_range=seq_range,
            dynamic_seq=bool(getattr(args, "compile_dynamic_seq", True)),
            grad_ckpt=False,
        )

    pixel_images: list[torch.Tensor] = []
    try:
        with acc.autocast():
            if hasattr(dit, "prepare_block_swap_before_forward"):
                dit.prepare_block_swap_before_forward()
            for gi in gen_idx:
                info, te_npz, _pe = items[gi]
                sd = _load_safetensors(te_npz)
                crossattn_emb = _build_val_crossattn_emb(dit, sd, acc)
                bucket_w, bucket_h = info.bucket_reso
                image = anima_train_utils.sample_image_to_tensor(
                    accelerator=acc,
                    dit=dit,
                    vae=vae,
                    height=int(bucket_h),
                    width=int(bucket_w),
                    crossattn_emb=crossattn_emb,
                    sample_steps=sample_steps,
                    guidance_scale=cfg_scale,
                    flow_shift=flow_shift,
                    seed=GEN_SEED_BASE + gi,  # held-out noise
                    show_progress=False,
                )
                pixel_images.append(image.detach().cpu())
                del image, crossattn_emb
                clean_memory_on_device(device)

        # Hand GPU to PE: DiT -> CPU, PE -> GPU (same dance as validation).
        dit.to("cpu")
        clean_memory_on_device(device)
        pe_bundle.encoder.inner.to(device)
        try:
            bucket_groups: dict[tuple[int, int], list[int]] = {}
            for idx, img in enumerate(pixel_images):
                key = (int(img.shape[-2]), int(img.shape[-1]))
                bucket_groups.setdefault(key, []).append(idx)
            pooled: list[torch.Tensor | None] = [None] * len(pixel_images)
            for indices in bucket_groups.values():
                batch = torch.stack([pixel_images[i] for i in indices], dim=0).to(
                    device
                )
                feats_list = encode_pe_from_imageminus1to1(
                    pe_bundle, batch, same_bucket=True
                )
                for idx, feats in zip(indices, feats_list):
                    pooled[idx] = pool_and_normalize(feats).cpu()
                del batch, feats_list
            gen_pool = torch.stack([t for t in pooled if t is not None], dim=0)
        finally:
            pe_bundle.encoder.inner.to("cpu")
            clean_memory_on_device(device)
    finally:
        del bundle, dit
        clean_memory_on_device(acc.device)

    return gen_pool, pixel_images


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def compute_metrics(
    gen_pool: torch.Tensor,  # [M, D]
    ref_pool: torch.Tensor,  # [N, D]
    gen_idx: list[int],  # length M, indices into ref_pool of each gen's source
    *,
    k: float,
    self_margin: float,
):
    """Self-vs-other contrast + paper NN-ratio in pooled-PE cosine space."""
    sim = gen_pool @ ref_pool.t()  # [M, N], cosine (both unit-norm)
    M, N = sim.shape
    rows = []
    s_self = torch.empty(M)
    s_other = torch.empty(M)
    nn1 = torch.empty(M)
    mem_flag = torch.zeros(M, dtype=torch.bool)
    self_lock = torch.zeros(M, dtype=torch.bool)

    for m in range(M):
        src = gen_idx[m]
        row = sim[m]
        s_self[m] = row[src]
        masked = row.clone()
        masked[src] = -2.0
        s_other_val, _ = masked.max(dim=0)
        s_other[m] = s_other_val
        # global top-2 (incl. self) for the paper ratio + self_lock argmax
        top2 = torch.topk(row, k=min(2, N))
        nn1_idx = int(top2.indices[0])
        nn1[m] = top2.values[0]
        nn2_val = top2.values[1] if N > 1 else top2.values[0]
        # paper Eq.6: ||x-a1||/||x-a2|| < k; for unit vecs ||x-a||^2 = 2(1-cos)
        d1 = (2.0 * (1.0 - top2.values[0]).clamp_min(0)).sqrt()
        d2 = (2.0 * (1.0 - nn2_val).clamp_min(0)).sqrt()
        ratio = float(d1 / d2.clamp_min(1e-9))
        mem_flag[m] = ratio < k
        self_lock[m] = (nn1_idx == src) and ((s_self[m] - s_other[m]) > self_margin)
        rows.append(
            {
                "src_idx": src,
                "s_self": round(float(s_self[m]), 4),
                "s_other": round(float(s_other[m]), 4),
                "nn1": round(float(nn1[m]), 4),
                "nn_ratio": round(ratio, 4),
                "mem_flag": int(mem_flag[m]),
                "self_lock": int(self_lock[m]),
            }
        )

    contrast = s_self - s_other
    metrics = {
        "n_gen": M,
        "n_ref": N,
        "contrast_mean": round(float(contrast.mean()), 4),
        "contrast_p50": round(float(contrast.median()), 4),
        "contrast_p90": round(float(contrast.quantile(0.9)), 4),
        "self_lock_frac": round(float(self_lock.float().mean()), 4),
        "mem_frac_paper": round(float(mem_flag.float().mean()), 4),
        "s_self_mean": round(float(s_self.mean()), 4),
        "s_self_p90": round(float(s_self.quantile(0.9)), 4),
        "s_other_mean": round(float(s_other.mean()), 4),
    }
    return metrics, rows, contrast


def real_real_floor(ref_pool: torch.Tensor) -> dict:
    """Per-train nearest *other* real-image cosine — the same-style floor that
    tells you what 'close but not a copy' looks like in this dataset."""
    sim = ref_pool @ ref_pool.t()
    sim.fill_diagonal_(-2.0)
    nn, _ = sim.max(dim=1)
    return {
        "real_real_nn_mean": round(float(nn.mean()), 4),
        "real_real_nn_p90": round(float(nn.quantile(0.9)), 4),
    }


# ---------------------------------------------------------------------------
# Contact sheet: gen | nearest-train, most-locked items first.
# ---------------------------------------------------------------------------


def _load_image_tensor(path: str, tile: int) -> torch.Tensor:
    from PIL import Image  # noqa: PLC0415

    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(_to_array(img)).permute(2, 0, 1).float() / 255.0
    return _fit_tile(t, tile)  # [3,tile,tile] in [0,1]


def _fit_tile(t: torch.Tensor, tile: int) -> torch.Tensor:
    """Fit [3,H,W] into a tile×tile box preserving aspect ratio (gray letterbox
    pad) — a square squash distorts exactly the composition detail the sheet
    exists to compare."""
    _, h, w = t.shape
    scale = tile / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
    ).squeeze(0)
    out = torch.full((3, tile, tile), 0.15)
    y, x = (tile - nh) // 2, (tile - nw) // 2
    out[:, y : y + nh, x : x + nw] = t
    return out


def _to_array(img):
    import numpy as np  # noqa: PLC0415

    return np.asarray(img).copy()  # writable — silences torch.from_numpy warning


def save_contact_sheet(
    *,
    path: Path,
    gen_pixels: list[torch.Tensor],
    items: list,
    gen_idx: list[int],
    rows: list[dict],
    sim_argmax_other,  # callable m -> ref index of nearest other
    top: int = 8,
    tile: int = 512,
) -> bool:
    """Rows = most self-locked / highest-contrast gen items; each row is
    [generated, its source train image, its nearest *other* train image]."""
    try:
        import torchvision.utils as tvu  # noqa: PLC0415

        order = sorted(
            range(len(rows)),
            key=lambda m: (
                rows[m]["self_lock"],
                rows[m]["s_self"] - rows[m]["s_other"],
            ),
            reverse=True,
        )[:top]
        cells: list[torch.Tensor] = []
        for m in order:
            gi = gen_idx[m]
            gen = _fit_tile((gen_pixels[m].float().clamp(-1, 1) + 1) / 2, tile)
            src_img = _load_image_tensor(items[gi][0].absolute_path, tile)
            other_idx = sim_argmax_other(m)
            other_img = _load_image_tensor(items[other_idx][0].absolute_path, tile)
            cells.extend([gen, src_img, other_img])
        if not cells:
            return False
        grid = tvu.make_grid(torch.stack(cells), nrow=3, padding=4)
        tvu.save_image(grid, str(path))
        return True
    except Exception as exc:
        print(f"  (contact sheet skipped: {exc})", flush=True)
        return False


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--adapter", required=True, help="LoRA checkpoint to probe.")
    ap.add_argument(
        "--with_base",
        action="store_true",
        help="Also render with NO adapter for an absolute contrast anchor "
        "(doubles generation cost).",
    )
    ap.add_argument(
        "--num_items",
        type=int,
        default=64,
        help="How many training captions to generate from (NN search is always "
        "over the full training set). Default 64.",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG for the item subset.")
    ap.add_argument(
        "--pe_encoder",
        choices=["pe_spatial", "pe"],
        default="pe_spatial",
        help="Reference feature: 'pe_spatial' (cached by `make preprocess` on "
        "every artist set; composition-sensitive — better for instance copies) "
        "or 'pe' (pooled PE-Core global; only where CMMD cached it).",
    )
    ap.add_argument("--sample_steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument(
        "--k",
        type=float,
        default=1.0 / 3.0,
        help="Paper Eq.6 NN-distance-ratio threshold (default 1/3).",
    )
    ap.add_argument(
        "--self_margin",
        type=float,
        default=0.05,
        help="Min (s_self - s_other) cosine gap for a self_lock (instance copy).",
    )
    ap.add_argument(
        "--path_pattern",
        default=None,
        help="Restrict the reference pool to this fnmatch pattern ('none' to "
        "force off; default: adapter .snapshot.toml). Without it a "
        "single-artist adapter is scored against the whole blueprint and the "
        "signal is diluted — see README.",
    )
    ap.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="The RUN's sample_ratio (default: adapter .snapshot.toml, else 1.0).",
    )
    ap.add_argument(
        "--artists_shard",
        default=None,
        help="The RUN's artists_shard ('none' to force off; default: snapshot).",
    )
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=True,
        help="torch.compile DiT blocks with dynamic-seq (default on).",
    )
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument(
        "--sheet_top",
        type=int,
        default=8,
        help="rows on the memorized.png contact sheet (most-locked first).",
    )
    ap.add_argument(
        "--sheet_tile",
        type=int,
        default=512,
        help="contact-sheet tile edge in px (aspect-preserving letterbox).",
    )
    ap.add_argument("--label", default=None)
    args_cli = ap.parse_args()

    dtype = str_to_dtype(args_cli.dtype)
    targs = build_training_args(
        args_cli.method,
        args_cli.preset,
        {
            "validation_sample_steps": args_cli.sample_steps,
            "validation_cfg_scale": args_cli.cfg,
        },
    )
    acc = prepare_accelerator(targs)
    shim = _Acc(acc.device, dtype)

    # Snapshot is authoritative for the pool (matches loss_gap/generalize);
    # explicit CLI flags override it, 'none' forces a knob off.
    snap = _read_snapshot_knobs(args_cli.adapter)
    if snap:
        print(f"snapshot knobs: {snap}", flush=True)

    def _resolve(cli_val, key):
        if cli_val is not None:
            return None if str(cli_val).lower() == "none" else cli_val
        return snap.get(key)

    sample_ratio = _resolve(args_cli.sample_ratio, "sample_ratio")
    artists_shard = _resolve(args_cli.artists_shard, "artists_shard")
    path_pattern = _resolve(args_cli.path_pattern, "path_pattern")
    print(
        f"pool filter: path_pattern={path_pattern!r} "
        f"sample_ratio={sample_ratio!r} artists_shard={artists_shard!r}",
        flush=True,
    )

    items = build_train_items(
        targs,
        pe_encoder=args_cli.pe_encoder,
        sample_ratio=sample_ratio,
        artists_shard=artists_shard,
        path_pattern=path_pattern,
    )
    if not items:
        sidecar = (
            "_anima_pe_spatial.safetensors"
            if args_cli.pe_encoder == "pe_spatial"
            else "_anima_pe.safetensors"
        )
        raise RuntimeError(
            "No training items with cached TE + PE sidecars. The probe reads "
            f"'{{stem}}_anima_te.safetensors' and '{{stem}}{sidecar}' — run "
            "`make preprocess` on this dataset"
            + ("" if args_cli.pe_encoder == "pe_spatial" else " (with preprocess-pe)")
            + "."
        )
    N = len(items)
    print(f"Training items (reference pool): {N}", flush=True)

    # Full-training reference pool (real images), aligned with `items` order.
    ref_pool = load_reference_features([pe for (_i, _t, pe) in items]).to(acc.device)
    if ref_pool.shape[0] != N:
        raise RuntimeError(
            f"PE pool ({ref_pool.shape[0]}) != items ({N}); some sidecars failed "
            "to load — the index alignment self/other relies on would be wrong. "
            "Fill the PE cache (`make preprocess-pe`) and retry."
        )

    # Choose the generated subset (NN search still spans all N).
    g = torch.Generator().manual_seed(args_cli.seed)
    perm = torch.randperm(N, generator=g).tolist()
    gen_idx = sorted(perm[: min(args_cli.num_items, N)])
    print(
        f"Generating from {len(gen_idx)} training captions (held-out seeds).",
        flush=True,
    )

    pe_bundle = load_pe_encoder(acc.device, name=args_cli.pe_encoder)
    pe_bundle.encoder.inner.to("cpu")
    flow_shift = float(getattr(targs, "discrete_flow_shift", 3.0))
    vae = load_vae(targs.vae, dtype=dtype, eval=True, vae_2d=True, device="cpu")

    gen_items = [items[i] for i in gen_idx]
    compile_budget = _token_budget(gen_items) if args_cli.compile else None
    if compile_budget is not None:
        print(
            f"compile ON — token families={compile_budget[0]} seq_range={compile_budget[1]}",
            flush=True,
        )

    gen_pool, gen_pixels = generate_gen_pool(
        adapter=args_cli.adapter,
        args=targs,
        acc=shim,
        vae=vae,
        pe_bundle=pe_bundle,
        items=items,
        gen_idx=gen_idx,
        sample_steps=args_cli.sample_steps,
        cfg_scale=args_cli.cfg,
        flow_shift=flow_shift,
        compile_budget=compile_budget,
    )
    gen_pool = gen_pool.to(acc.device)

    metrics, rows, _contrast = compute_metrics(
        gen_pool, ref_pool, gen_idx, k=args_cli.k, self_margin=args_cli.self_margin
    )
    metrics.update(real_real_floor(ref_pool))

    # nearest-other resolver for the contact sheet (maps gen row m -> ref index).
    sim = gen_pool @ ref_pool.t()
    for m in range(sim.shape[0]):
        sim[m, gen_idx[m]] = -2.0
    nearest_other = sim.argmax(dim=1).tolist()

    # Optional absolute anchor: same contrast with the base model (no adapter).
    if args_cli.with_base:
        base_pool, _ = generate_gen_pool(
            adapter=None,
            args=targs,
            acc=shim,
            vae=vae,
            pe_bundle=pe_bundle,
            items=items,
            gen_idx=gen_idx,
            sample_steps=args_cli.sample_steps,
            cfg_scale=args_cli.cfg,
            flow_shift=flow_shift,
            compile_budget=compile_budget,
        )
        base_metrics, _, _ = compute_metrics(
            base_pool.to(acc.device),
            ref_pool,
            gen_idx,
            k=args_cli.k,
            self_margin=args_cli.self_margin,
        )
        metrics["base_contrast_mean"] = base_metrics["contrast_mean"]
        metrics["base_self_lock_frac"] = base_metrics["self_lock_frac"]
        metrics["contrast_over_base"] = round(
            metrics["contrast_mean"] - base_metrics["contrast_mean"], 4
        )

    # Verdict heuristic (the floor for `contrast` is ~0 by construction).
    verdict = "GENERALIZING"
    if metrics["self_lock_frac"] >= 0.5 or metrics["contrast_mean"] >= 0.15:
        verdict = "MEMORIZING"
    elif metrics["self_lock_frac"] >= 0.2 or metrics["contrast_mean"] >= 0.07:
        verdict = "PARTIAL"
    metrics["verdict"] = verdict

    run_dir = make_run_dir("memorization", label=args_cli.label)
    csv_path = run_dir / "per_item.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["stem", *rows[0].keys()] if rows else ["stem"]
        )
        w.writeheader()
        for m, r in enumerate(rows):
            stem = Path(items[gen_idx[m]][0].absolute_path).stem
            w.writerow({"stem": stem, **r})

    artifacts = ["per_item.csv"]
    sheet = run_dir / "memorized.png"
    if save_contact_sheet(
        path=sheet,
        gen_pixels=gen_pixels,
        items=items,
        gen_idx=gen_idx,
        rows=rows,
        sim_argmax_other=lambda m: nearest_other[m],
        top=args_cli.sheet_top,
        tile=args_cli.sheet_tile,
    ):
        artifacts.append("memorized.png")

    write_result(
        run_dir,
        script=__file__,
        args=args_cli,
        metrics=metrics,
        label=args_cli.label,
        artifacts=artifacts,
        device=acc.device,
    )

    print("\n=== Memorization probe ===", flush=True)
    for key in (
        "verdict",
        "contrast_mean",
        "self_lock_frac",
        "mem_frac_paper",
        "s_self_mean",
        "s_other_mean",
        "real_real_nn_mean",
    ):
        print(f"  {key:18s} {metrics[key]}", flush=True)
    if args_cli.with_base:
        print(f"  contrast_over_base {metrics['contrast_over_base']}", flush=True)
    print(f"\n  -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
