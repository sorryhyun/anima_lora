"""Mod-guidance distillation config: argparser + resolved dataclass.

Mirrors the ``distill_turbo/config.py`` precedent (frozen dataclass consumed by
the loop) but stays **CLI-first** — there is no TOML layer here, since
every documented invocation drives the script purely
through flags. ``resolve_config`` is a pure CLI→dataclass map plus the GAD
sanity checks; the door is left open for an optional ``[gad]`` TOML block later
without forcing one now.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Argparser — same flag names/defaults as the old distill.py inline parser, so
# every documented invocation is unchanged.
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modulation guidance distillation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="post_image_dataset/lora",
        help="Directory with cached latents and text encoder outputs",
    )
    parser.add_argument(
        "--uncond_te_path",
        type=str,
        default="post_image_dataset/_anima_uncond_te.safetensors",
        help=(
            'Path to the T5("") sidecar used as the student\'s unconditional '
            "cross-attention input. Defaults to "
            "``post_image_dataset/_anima_uncond_te.safetensors`` (staged by "
            "``prep.py`` Phase 1, or ``make preprocess-te``)."
        ),
    )
    parser.add_argument(
        "--synth_data_dir",
        type=str,
        default="post_image_dataset/distill_mod_synth",
        help=(
            "Optional dir of teacher-generated synthetic clean latents from "
            "``prep.py`` Phase 2. When set, training reads latents "
            "from here (matched by stem + resolution) and TE caches from "
            "``--data_dir`` — paper-faithful setup that removes the real-vs-"
            "teacher distribution gap. Default: real-image latents only."
        ),
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="models/diffusion_models/anima-base-v1.0.safetensors",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/ckpt/pooled_text_proj.safetensors",
        help="Where to save the trained projection weights",
    )
    parser.add_argument("--iterations", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Number of transformer blocks to offload to CPU",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="Attention mode (torch, flash). flash4 not supported yet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale for sigmoid timestep sampling",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a saved pooled_text_proj checkpoint",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=True,
        help="Compile block._forward with torch.compile",
    )
    parser.add_argument(
        "--no_compile",
        dest="torch_compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--compile_inductor_mode",
        type=str,
        default="",
        help="Inductor preset, e.g. 'reduce-overhead' for CUDAGraphs",
    )
    parser.add_argument(
        "--compile_dynamic_seq",
        action="store_true",
        help="Collapse the per-token-count block graphs into ONE symbolic-seq "
        "graph (mark_dynamic on the seq axis), bounded by the token counts present "
        "in the cached pool (data_dir + synth_data_dir). Mirrors the LoRA-training "
        "compile_dynamic_seq path. Only matters with --torch_compile (default on).",
    )
    parser.add_argument(
        "--no_compile_dynamic_seq",
        dest="compile_dynamic_seq",
        action="store_false",
        help="Disable dynamic-seq: trace one static graph per distinct token count.",
    )
    parser.add_argument(
        "--activation_memory_budget",
        type=float,
        default=1.0,
        help="torch.compile partitioner saved-activation fraction (<1.0 recomputes "
        "cheap intermediates in backward to cut peak VRAM; mirrors train.py). "
        "1.0 = off. Ignored under gradient checkpointing (CheckpointError otherwise; "
        "redundant there since ckpt already minimizes saved activations).",
    )
    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="Enable gradient checkpointing with CPU offload (default on)",
    )
    parser.add_argument(
        "--no_grad_ckpt",
        dest="grad_ckpt",
        action="store_false",
        help="Disable gradient checkpointing (faster, more VRAM)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.02,
        help="Warmup steps: int >= 1 for absolute steps, float < 1 for ratio of iterations",
    )
    parser.add_argument(
        "--no_shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable per-epoch shuffling of the (bucket-grouped) batch order. "
        "Default shuffles batch order each epoch while keeping every batch "
        "single-resolution and pinning the largest-token bucket to step 0.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Iterate entire DataLoader without training to test collation",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="output/logs/distill_mod",
        help="TensorBoard log directory. A timestamped subdir is created per run.",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log scalars to TensorBoard every N optimizer steps",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of (post-split) samples to keep per bucket. Mirrors the "
        "LoRA per-subset sample_ratio; useful with PRESET=debug/half/quarter/tenth "
        "for fast iteration on a small slice of the dataset.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.01,
        help="Fraction of dataset held out for validation (e.g. 0.05 for 5 percent)",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Seed for deterministic train/val split + validation noise",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=1500,
        help="Run validation every N optimizer steps (only if validation_split>0)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1500,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--validation_sigmas",
        type=float,
        nargs="+",
        default=[0.1, 0.4, 0.7],
        help="Fixed sigma values for validation loss (mirrors train.py default)",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Cap on validation batches per pass. None = use the entire val set.",
    )
    parser.add_argument(
        "--teacher_cache_K",
        type=int,
        default=6,
        help="Number of pre-sampled sigma bins for the teacher prediction cache. "
        "Each sample sees K distinct (sigma, noise) pairs over the run. "
        "Higher K = more diversity but slower cache fill / larger RAM.",
    )
    parser.add_argument(
        "--teacher_cache_seed",
        type=int,
        default=1234,
        help="Seed for the K-sigma grid and per-(sample, sigma) deterministic noise. "
        "Independent of --seed so cache contents are reproducible across training runs.",
    )
    parser.add_argument(
        "--no_teacher_cache",
        action="store_true",
        help="Disable teacher prediction caching (re-runs the teacher forward every step). "
        "Use to A/B against the cached path or to recover the original continuous-sigma sampler.",
    )
    parser.add_argument(
        "--prefill_teacher_cache",
        action="store_true",
        help="Eagerly run teacher predictions for every (sample, sigma_idx) before training. "
        "Adds ~K * N * t_teacher up front but eliminates teacher forwards during training.",
    )
    parser.add_argument(
        "--no_val_teacher_cache",
        action="store_true",
        help="Disable validation-time teacher prediction caching (re-runs the teacher "
        "forward on every val pass). Default is enabled — val is deterministic across "
        "calls, so the first pass fills a (batch_idx, sigma_idx) cache and every "
        "subsequent pass skips teacher forwards entirely.",
    )

    # GAD (geometry-aware distillation; arXiv 2606.01651) — also match the
    # teacher's finite-difference text response. gad_weight=0 (default) skips the
    # two extra forwards → bit-for-bit the MSE-only path. Motivation:
    # docs/findings/mod_guidance_quality_tag_axis.md §3 (text-derivative orthogonal).
    parser.add_argument(
        "--gad_weight",
        type=float,
        default=0.0,
        help="λ on the GAD text-response-matching term: L = L_mse + λ·MSE(ΔS, ΔT). "
        "0 (default) disables → exact reproduction of the MSE-only head. ΔT is the "
        "teacher's velocity response to a text swap (via cross-attn); ΔS is the "
        "student's (via the modulation MLP). UNTUNED — sweep (e.g. 0.25/1/4) and "
        "watch train/loss_gad vs train/loss_mse so GAD doesn't swamp the base fit.",
    )
    parser.add_argument(
        "--gad_h",
        type=float,
        default=1.0,
        help="Text-perturbation scale: text_pert = text_A + h·(text_B − text_A). "
        "1.0 (default) = full A→B prompt swap (best SNR; the text signal is ~2%%). "
        "h < 1 = a more local Jacobian. Only active with --gad_weight > 0.",
    )
    parser.add_argument(
        "--gad_loss",
        type=str,
        default="l2",
        choices=("l2", "cosine"),
        help="GAD loss form. 'l2' (default, paper-faithful): MSE(ΔS, ΔT) — also "
        "penalizes the magnitude gap (desirable given the high-σ ratio collapse). "
        "'cosine': 1 − cos(ΔS, ΔT) — direction-only A/B lever; do NOT default to it.",
    )
    parser.add_argument(
        "--gad_pair_source",
        type=str,
        default="auto",
        choices=("auto", "batch", "dataset"),
        help="Where the perturbation text (sample B) comes from. 'auto' (default): "
        "batch-roll when batch_size>1, else dataset-random. 'batch' needs "
        "batch_size>1 (a roll of a size-1 batch is a no-op → falls back to "
        "dataset). 'dataset': draw a random other sample's cached (crossattn, "
        "pooled) each step (reproducible via --seed).",
    )
    parser.add_argument(
        "--mod_sigma_film",
        action="store_true",
        help="σ-FiLM: timestep-condition the mod head's hidden activation (FiLM "
        "from the normed time embedding) so the text push can scale/re-aim per σ. "
        "Targets the σ-flat magnitude collapse (ratio ‖ΔS‖/‖ΔT‖ → 0.05 at σ=0.9) "
        "and disentangles the spatial-ceiling vs σ-averaging cause of GAD cos≈0. "
        "Adds a Linear(D,2D) generator; off ⇒ bit-exact to the plain head. "
        "Saved under a 'sigma_film.' prefix; probe/inference auto-arm on load.",
    )
    return parser


@dataclass(frozen=True)
class ModConfig:
    # Paths / IO
    data_dir: str
    uncond_te_path: str | None
    synth_data_dir: str | None
    dit_path: str
    output_path: str
    log_dir: str
    no_log: bool
    log_interval: int
    save_every: int

    # Run shape
    iterations: int
    lr: float
    batch_size: int
    grad_accum: int
    warmup: float
    seed: int
    sample_ratio: float
    shuffle: bool
    dry_run: bool
    resume: str | None

    # Runtime
    blocks_to_swap: int
    attn_mode: str
    sigmoid_scale: float
    torch_compile: bool
    compile_inductor_mode: str
    compile_dynamic_seq: bool
    activation_memory_budget: float
    grad_ckpt: bool

    # Validation
    validation_split: float
    validation_seed: int
    validate_every_n_steps: int
    validation_sigmas: list[float]
    max_validation_steps: int | None

    # Teacher cache
    teacher_cache_K: int
    teacher_cache_seed: int
    no_teacher_cache: bool
    prefill_teacher_cache: bool
    no_val_teacher_cache: bool

    # GAD (geometry-aware distillation) — Phase 2; off when gad_weight == 0
    gad_weight: float
    gad_h: float
    gad_loss: str
    gad_pair_source: str

    # σ-FiLM: timestep-condition the mod head (off ⇒ plain σ-flat head)
    mod_sigma_film: bool


def resolve_config(args: argparse.Namespace) -> ModConfig:
    """Pure CLI→dataclass map (no TOML) + GAD sanity checks."""

    if args.gad_weight < 0.0:
        raise ValueError(f"--gad_weight={args.gad_weight}: must be >= 0")
    if args.gad_weight > 0.0 and args.gad_h <= 0.0:
        raise ValueError(f"--gad_h={args.gad_h}: must be > 0 when --gad_weight > 0")
    if args.gad_weight > 0.0:
        logger.info(
            "GAD (geometry-aware distillation, arXiv 2606.01651) ON: "
            f"weight={args.gad_weight}, h={args.gad_h}, loss={args.gad_loss}, "
            f"pair_source={args.gad_pair_source} — matching the teacher's text-"
            "derivative ΔS → ΔT on top of the pointwise MSE."
        )

    return ModConfig(
        data_dir=args.data_dir,
        uncond_te_path=args.uncond_te_path,
        synth_data_dir=args.synth_data_dir,
        dit_path=args.dit_path,
        output_path=args.output_path,
        log_dir=args.log_dir,
        no_log=bool(args.no_log),
        log_interval=int(args.log_interval),
        save_every=int(args.save_every),
        iterations=int(args.iterations),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        warmup=float(args.warmup),
        seed=int(args.seed),
        sample_ratio=float(args.sample_ratio),
        shuffle=bool(args.shuffle),
        dry_run=bool(args.dry_run),
        resume=args.resume,
        blocks_to_swap=int(args.blocks_to_swap),
        attn_mode=args.attn_mode,
        sigmoid_scale=float(args.sigmoid_scale),
        torch_compile=bool(args.torch_compile),
        compile_inductor_mode=args.compile_inductor_mode,
        compile_dynamic_seq=bool(args.compile_dynamic_seq),
        activation_memory_budget=float(args.activation_memory_budget),
        grad_ckpt=bool(args.grad_ckpt),
        validation_split=float(args.validation_split),
        validation_seed=int(args.validation_seed),
        validate_every_n_steps=int(args.validate_every_n_steps),
        validation_sigmas=list(args.validation_sigmas),
        max_validation_steps=args.max_validation_steps,
        teacher_cache_K=int(args.teacher_cache_K),
        teacher_cache_seed=int(args.teacher_cache_seed),
        no_teacher_cache=bool(args.no_teacher_cache),
        prefill_teacher_cache=bool(args.prefill_teacher_cache),
        no_val_teacher_cache=bool(args.no_val_teacher_cache),
        gad_weight=float(args.gad_weight),
        gad_h=float(args.gad_h),
        gad_loss=args.gad_loss,
        gad_pair_source=args.gad_pair_source,
        mod_sigma_film=bool(args.mod_sigma_film),
    )
