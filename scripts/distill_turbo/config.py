"""Turbo distillation config: TOML loader + argparser + CLI/TOML resolver.

The resolved knobs are returned as a ``TurboConfig`` frozen dataclass so the
training loop never reaches back into ``args``/``cfg`` mid-step.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

from library.config.io import toml_get as _flatten

# Python 3.11+; fall back to ``tomli`` if needed.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------


def load_turbo_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _pick(cli_val: Any, cfg: dict, toml_key: str, default: Any) -> Any:
    """CLI override (non-sentinel) wins, else TOML, else default.

    Sentinels are: ``None`` (explicitly unset), ``-1`` (int default), ``-1.0``
    (float default). They mean "use the TOML/default value".
    """
    if cli_val is not None and cli_val != -1 and cli_val != -1.0:
        return cli_val
    return _flatten(cfg, toml_key, default)


# ---------------------------------------------------------------------------
# Argparser
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turbo Anima — Decoupled DMD2 distillation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/methods/turbo.toml",
        help="Path to the turbo TOML config (CLI flags override TOML values).",
    )
    # CLI overrides — every TOML key has a matching flag. Default sentinels
    # (None / -1.0) mean "use the TOML value".
    parser.add_argument("--dit_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=-1,
        help="Run the DAVE same-prompt diversity probe every N optimizer steps "
        "(0 disables; see scripts/distill_turbo/diversity.py). Logs "
        "val/div_ac_sim (lower = more diverse), val/div_dc_sim, val/div_gap.",
    )
    parser.add_argument(
        "--val_diversity_seeds",
        type=int,
        default=-1,
        help="Number of seeds the diversity probe rolls per validation (>=2).",
    )
    parser.add_argument(
        "--val_prompt_idx",
        type=int,
        default=-1,
        help="Held-out dataset index whose cached conditioning the diversity "
        "probe fixes (-1 = auto: last sample, distinct from --single_prompt_idx).",
    )
    parser.add_argument("--student_rank", type=int, default=-1)
    parser.add_argument("--fake_rank", type=int, default=-1)
    parser.add_argument(
        "--use_custom_down_autograd",
        action="store_true",
        default=None,
        help="DEPRECATED no-op (fp32-bottleneck path removed 2026-06-10; "
        "training GEMMs run in the activation dtype). Accepted so old "
        "snapshots/commands replay.",
    )
    parser.add_argument(
        "--no_use_custom_down_autograd",
        dest="use_custom_down_autograd",
        action="store_false",
    )
    parser.add_argument(
        "--channel_scaling_alpha",
        type=float,
        default=-1.0,
        help="Per-input-channel rebalance absorbed into lora_down (student + "
        "fake). 0.0 = off, 0.5 = sqrt-balance. Default: read from TOML "
        "(top-level scalar), else 0.0 (off).",
    )
    parser.add_argument(
        "--use_masked_loss",
        action="store_true",
        default=None,
        help="Apply the per-image foreground mask to the student DMD2 gradient "
        "(masked-out latents get zero student push). Fake/critic loss is "
        "unaffected. Default: read from TOML (top-level scalar), else off.",
    )
    parser.add_argument(
        "--no_use_masked_loss",
        dest="use_masked_loss",
        action="store_false",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Mask root for --use_masked_loss (default: TOML mask_dir, else "
        "post_image_dataset/masks). Mirrors data_dir's subdir layout.",
    )
    parser.add_argument("--student_lr", type=float, default=-1.0)
    parser.add_argument("--fake_lr", type=float, default=-1.0)
    parser.add_argument(
        "--fake_steps_per_student_step",
        type=int,
        default=-1,
        help="Number of fake (DM regularizer) updates per student step. "
        "Standard DMD2 practice keeps the fake ahead of the moving x_pred "
        "distribution; >1 gives the fake extra SGD iterations on resampled "
        "(τ, ε) noise against the same x_pred.detach(). Default: TOML "
        "(optim.fake_steps_per_student_step, default 1).",
    )
    parser.add_argument(
        "--fake_warmup_steps",
        type=int,
        default=-1,
        help="Fake-only (critic head-start) updates run BEFORE the main loop. "
        "The student LR warmup finishes at ~0.02·iterations, so the student "
        "starts full-strength steps while the zero-init fake/critic LoRA is "
        "still ≈ the teacher → a large, misaligned delta_dm and an early "
        "grad_signal_rms spike (~step 50). Pre-training the fake net against the "
        "student's (init ≈ teacher) x_pred distribution calibrates it first. "
        "The fake scheduler IS stepped during warmup (the main-loop scheduler "
        "is sized over iterations + fake_warmup_steps so the 2%% LR warmup "
        "overlaps the head-start and the fake enters the main loop at full LR). "
        "Default: TOML (optim.fake_warmup_steps, default 0 = off).",
    )
    parser.add_argument(
        "--student_steps",
        type=int,
        default=-1,
        help="Sampler step count baked into the student",
    )
    parser.add_argument(
        "--per_step_expert",
        dest="per_step_expert",
        action="store_const",
        const=True,
        default=None,
        help="Split the student into per-step up-heads (head k serves denoise "
        "step k) off a shared down-proj, so the diversity (step 0) and DMD "
        "(steps 1..N) gradients stop fighting over one set of up-weights. "
        "K = student_steps. Output is NOT a plain LoRA (kept-live only; merge "
        "refuses it). Default: TOML (network.per_step_expert, default false).",
    )
    parser.add_argument(
        "--no_per_step_expert",
        dest="per_step_expert",
        action="store_false",
    )
    parser.add_argument(
        "--dm_x0_norm",
        dest="dm_x0_norm",
        action="store_const",
        const=True,
        default=None,
        help="DMD per-sample x0-space magnitude normalization (policy 'b'): "
        "grad_dm = τ·Δ_dm / clamp(τ·mean|v_real|, norm_floor). Because the denom "
        "≈ τ·mean|v_real|, the τ CANCELS across the bulk → ≈ no-τ, magnitude-"
        "normalized. This REPLACES the default τ-damping (policy 'a'); it does NOT "
        "stack with it (that would be policy 'c'). A/B lever — see "
        "docs/proposal/dmd2_decoupled_improvements.md §2B.",
    )
    parser.add_argument(
        "--norm_floor",
        type=float,
        default=-1.0,
        help="clamp_min for the x0-norm denominator (latent scale); only active "
        "with --dm_x0_norm.",
    )
    parser.add_argument(
        "--dmd_grad_step",
        type=str,
        default=None,
        choices=("all", "last", "random"),
        help="Which rollout step(s) carry gradient in plain DMD2 (base_loss='dmd'); "
        "the rest are backward-simulated under no_grad (DMD2's train/inference "
        "input-match, Yin et al. 2024). 'all' = full-rollout BPTT (holds N forward "
        "graphs). 'last' = only the final, cleanest-σ step (memory-flat, but the "
        "noisy steps are never directly supervised). 'random' = canonical DMD2 "
        "multistep: sample g~U{0..N-1}, grad ONLY step g, supervise its one-step "
        "x0-prediction — memory-flat AND spreads supervision over every grid point. "
        "Default: TOML (dmd.grad_step, default 'all').",
    )
    # Mean-variance reg (lever B / paper Eq. 7; proposal §3.B / S2). Pulls each
    # generated image's (μ_i, σ²_i) toward the real-latent target — clamps the
    # variance inflation that is the over-bake's oversaturation.
    parser.add_argument(
        "--mean_var_weight",
        type=float,
        default=-1.0,
        help="Weight on the Eq.7 mean-variance KL added to the student loss. "
        "0 disables; S2 uses ~0.01–0.05. The target stats are read from TOML "
        "([mean_var].mu_t/sigma2_t), or measured exactly in a one-pass scan over "
        "the real latents when sigma2_t <= 0. Default: TOML (mean_var.weight, "
        "default 0).",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--attn_mode", type=str, default="flash")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=True,
        help="Compile block._forward. Off by default — multiple forwards per step "
        "are not yet validated under cudagraphs; turn on once Phase 0 is green.",
    )
    parser.add_argument(
        "--compile_dynamic_seq",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror the LoRA-training compile_dynamic_seq path: collapse the "
        "per-token-count block graphs to a single graph by marking only the "
        "seq-length axis dynamic (mark_dynamic). Sentinel None → TOML "
        "(compile_dynamic_seq, default true). Only matters when --torch_compile.",
    )
    parser.add_argument(
        "--target_res",
        type=int,
        nargs="+",
        default=None,
        help="Override the active multi-scale tier edges (e.g. 1024 768 1280) "
        "used to size the compile_dynamic_seq seq bound + dynamo cache budget. "
        "Unset (the default) → derived automatically from the token-count "
        "families present in the cached pool (data_dir).",
    )
    parser.add_argument(
        "--activation_memory_budget",
        type=float,
        default=None,
        help="torch.compile partitioner saved-activation fraction (<1.0 → "
        "recompute cheap intermediates in backward, mirrors the LoRA-training "
        "knob in base.toml). Only applies when --torch_compile is on and "
        "grad_ckpt is off (the two repartition the same graph and conflict). "
        "Sentinel None → TOML (activation_memory_budget, default 1.0 = off).",
    )
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument(
        "--single_prompt_idx",
        type=int,
        default=None,
        help="Phase 0 overfit mode — pin the dataloader to a single (latent, text) pair.",
    )
    parser.add_argument("--sample_ratio", type=float, default=1.0)

    # ---- DP-DMD (diversity-preserved DMD; arXiv 2602.03139) ----
    # The student is a genuine N-step rollout: step 1 supervised toward a teacher
    # K-step anchor (diversity), detached, then DMD on x_θ over steps 2..N
    # (quality). See docs/experimental/dpdmd.md.
    parser.add_argument(
        "--k_anchor",
        type=int,
        default=-1,
        help="DP-DMD: teacher steps rolled to the diversity anchor (their K). "
        "Default: TOML (dpdmd.k_anchor, default 5).",
    )
    parser.add_argument(
        "--teacher_anchor_steps",
        type=int,
        default=-1,
        help="DP-DMD: teacher σ-grid the K anchor is counted against. Default: "
        "TOML (dpdmd.teacher_anchor_steps, default 28).",
    )
    parser.add_argument(
        "--div_weight",
        type=float,
        default=-1.0,
        help="DP-DMD: λ on the first-step diversity loss. Default: TOML "
        "(dpdmd.div_weight, default 0.05).",
    )
    parser.add_argument(
        "--detach_after_first",
        dest="detach_after_first",
        action="store_const",
        const=True,
        default=None,
        help="DP-DMD: stop-grad after the diversity-supervised first step (the "
        "load-bearing detach; keep True except for A/B). Default: TOML "
        "(dpdmd.detach_after_first, default true).",
    )
    parser.add_argument(
        "--no_detach_after_first",
        dest="detach_after_first",
        action="store_false",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=-1.0,
        help="DP-DMD: σ-schedule shift for the student/teacher Euler grids "
        "(matches inference). Default: TOML (sampling.flow_shift, default 3.0).",
    )

    # ---- Base objective ----
    parser.add_argument(
        "--base_loss",
        type=str,
        default=None,
        choices=("dpdmd", "dmd"),
        help="Diversity mechanism: 'dpdmd' (first-step teacher anchor, default) "
        "or 'dmd' (plain DMD2 — no anchor, allows student_steps=1). Default: TOML "
        "(base_loss, default 'dpdmd').",
    )

    # ---- DMD2 teacher-feature GAN (FastGen idea 1; off by default) ----
    parser.add_argument(
        "--gan_loss_weight_gen",
        type=float,
        default=-1.0,
        help="λ on the GAN generator term (softplus hinge on teacher-feature "
        "disc logits), added to the student loss. 0 disables the whole GAN path "
        "(byte-identical to DP-DMD). FastGen QwenImage uses 0.03. Default: TOML "
        "(gan.weight_gen, default 0).",
    )
    parser.add_argument(
        "--gan_feature_block_idx",
        type=int,
        default=-2,
        help="Which DiT block's token output the discriminator taps. -1 = middle "
        "block (num_blocks//2). Default sentinel -2 → TOML (gan.feature_block_idx, "
        "default -1).",
    )
    parser.add_argument(
        "--gan_disc_lr",
        type=float,
        default=-1.0,
        help="Discriminator AdamW LR. Default: TOML (gan.disc_lr, default 1e-5).",
    )
    parser.add_argument(
        "--gan_r1_weight",
        type=float,
        default=-1.0,
        help="Weight on the approximate-R1 (APT) disc regularizer: MSE between "
        "real logits and logits of a slightly-perturbed real input. 0 disables. "
        "Default: TOML (gan.r1_weight, default 0).",
    )

    # ---- Turbo × REPA relational alignment (off by default) ----
    parser.add_argument(
        "--repa_weight",
        type=float,
        default=-1.0,
        help="λ on the relational (Gram) alignment of student block features to "
        "PE-Spatial on renoised REAL latents. 0 disables the whole path "
        "(byte-identical DP-DMD). LoRA-validated scale: 0.05. Default: TOML "
        "(repa.weight, default 0).",
    )
    parser.add_argument(
        "--repa_layer",
        type=int,
        default=-1,
        help="DiT block whose output the REPA term taps (matches LoRA REPA). "
        "Default: TOML (repa.layer, default 8).",
    )
    parser.add_argument(
        "--repa_every_n",
        type=int,
        default=-1,
        help="Run the REPA term every N student steps (amortizes the extra "
        "partial forward). Default: TOML (repa.every_n, default 4).",
    )
    parser.add_argument(
        "--repa_encoder",
        type=str,
        default=None,
        help="Vision-encoder sidecar the term aligns to "
        "({stem}_anima_{encoder}.safetensors next to the TE cache). Default: "
        "TOML (repa.encoder, default 'pe_spatial').",
    )
    parser.add_argument(
        "--repa_spatial_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="iREPA spatial standardization of the encoder target. Default: "
        "TOML (repa.spatial_norm, default true).",
    )
    parser.add_argument(
        "--repa_target_dog",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="REPA-DoG target band-pass (replaces spatial_norm's DC removal "
        "with a broader low-band strip; the two are mutually exclusive and dog "
        "wins). Default: TOML (repa.target_dog, default false).",
    )
    parser.add_argument(
        "--repa_dog_sigma1_div",
        type=float,
        default=-1.0,
        help="REPA-DoG low-pass σ₁ as min(gh,gw)/div (larger div = narrower "
        "strip). Default: TOML (repa.dog_sigma1_div, default 16).",
    )
    parser.add_argument(
        "--repa_dog_sigma2_div",
        type=float,
        default=-1.0,
        help="REPA-DoG second σ₂ as min(gh,gw)/div (0 = single low-pass, σ₂ "
        "off). Default: TOML (repa.dog_sigma2_div, default 0).",
    )
    parser.add_argument(
        "--repa_dog_norm_std",
        type=float,
        default=-1.0,
        help="REPA-DoG post band-pass std normalization (0 = empirical per-batch "
        "std). Default: TOML (repa.dog_norm_std, default 0).",
    )

    # ---- f-distill reweighting (FastGen idea 2; needs the GAN disc) ----
    parser.add_argument(
        "--f_div",
        type=str,
        default=None,
        choices=("rkl", "kl", "js", "sf", "neyman", "sh", "jf"),
        help="f-divergence whose weight h=f'(r) reweights the DMD signal "
        "(r=exp(disc_logits) from idea 1). 'rkl' ≡ uniform h ≡ plain DMD2 (no-op). "
        "Any other value REQUIRES gan_loss_weight_gen > 0. Default: TOML "
        "(f_distill.f_div, default 'rkl').",
    )
    return parser


# ---------------------------------------------------------------------------
# Resolved config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurboConfig:
    # Paths / IO
    dit_path: str
    data_dir: str
    output_dir: str
    output_name: str
    log_dir: str
    save_every: int
    log_interval: int
    no_log: bool

    # Diversity validation (DAVE same-prompt probe; 0 = off)
    validate_every_n_steps: int
    val_diversity_seeds: int
    val_prompt_idx: int

    # Run shape
    iterations: int
    batch_size: int
    seed: int
    sample_ratio: float
    single_prompt_idx: int | None

    # LoRA stacks
    student_rank: int
    fake_rank: int
    student_alpha: float
    fake_alpha: float
    attn_mode: str
    use_custom_down_autograd: bool
    # SmoothQuant-style per-input-channel rebalance on each lora_down (student +
    # fake). 0.0 = off (default; preserves pre-2026-06-09 turbo runs), 0.5 =
    # sqrt-balance. Bit-equivalent at init, merges out — a free gradient-
    # conditioning lever, not an inference correction.
    channel_scaling_alpha: float
    # Per-step expert (dual-B-head student). When on, step_expert_K is derived
    # = student_steps; head k serves denoise step k. Off → single-head student.
    per_step_expert: bool
    step_expert_K: int

    # Masked loss
    use_masked_loss: bool
    mask_dir: str

    # DP-DMD knobs
    k_anchor: int
    teacher_anchor_steps: int
    div_weight: float
    detach_after_first: bool
    flow_shift: float

    # DMD core
    student_steps: int
    teacher_cfg: float
    dm_x0_norm: bool
    norm_floor: float
    dmd_grad_step: str  # "all" | "last" | "random"

    # Base objective selector
    base_loss: str

    # DMD2 teacher-feature GAN (idea 1) + f-distill reweighting (idea 2)
    gan_loss_weight_gen: float
    gan_feature_block_idx: int  # -1 → middle block (resolved in distill.py)
    gan_disc_lr: float
    gan_disc_hidden: int  # <= 0 → inner_dim // 2
    gan_r1_weight: float
    gan_r1_alpha: float
    gan_use_same_t_noise: bool
    gan_grad_ckpt: bool  # checkpoint ONLY the grad-bearing GAN gen forward
    f_div: str
    f_ratio_lower: float
    f_ratio_upper: float
    f_ratio_ema_rate: float
    f_bin_num: int
    f_ratio_normalization: bool

    # Turbo × REPA relational alignment on real data (turbo_repa.md Phase 1)
    repa_weight: float
    repa_layer: int
    repa_encoder: str
    repa_every_n: int
    repa_spatial_norm: bool
    repa_target_dog: bool
    repa_dog_sigma1_div: float
    repa_dog_sigma2_div: float
    repa_dog_norm_std: float

    # Mean-variance reg (lever B / Eq. 7)
    mean_var_weight: float
    mv_mu_t: float
    mv_sigma2_t: float
    mv_calib_batches: int

    # Optimizer + scheduler
    student_lr: float
    fake_lr: float
    fake_steps_per_student_step: int
    fake_warmup_steps: int
    weight_decay: float
    grad_clip: float

    # Sampling distribution
    t_distribution: str
    sigmoid_scale: float

    # Runtime
    blocks_to_swap: int
    grad_ckpt: bool
    torch_compile: bool
    compile_dynamic_seq: bool  # single symbolic-seq block graph (mark_dynamic)
    target_res: list[int] | None  # override tier edges; None → derived from cached pool
    dynamo_recompile_limit: int  # per-_forward dynamo graph budget
    activation_memory_budget: (
        float  # compile partitioner saved-act fraction (<1 → recompute)
    )


def resolve_config(args: argparse.Namespace, cfg: dict) -> TurboConfig:
    """Apply CLI/TOML/default precedence and run sanity checks."""

    # Paths
    dit_path = _pick(
        args.dit_path,
        cfg,
        "dit_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    data_dir = _pick(args.data_dir, cfg, "data_dir", "post_image_dataset/lora")
    output_dir = _pick(args.output_dir, cfg, "output_dir", "output/ckpt")
    output_name = _pick(args.output_name, cfg, "output_name", "anima_turbo")
    log_dir = _pick(args.log_dir, cfg, "io.log_dir", "output/logs/turbo")
    save_every = int(_pick(args.save_every, cfg, "io.save_every", 1000))
    log_interval = int(_pick(args.log_interval, cfg, "io.log_interval", 2))
    validate_every_n_steps = int(
        _pick(args.validate_every_n_steps, cfg, "io.validate_every_n_steps", 0)
    )
    val_diversity_seeds = int(
        _pick(args.val_diversity_seeds, cfg, "io.val_diversity_seeds", 8)
    )
    val_prompt_idx = int(_pick(args.val_prompt_idx, cfg, "io.val_prompt_idx", -1))

    # Run shape
    iterations = int(_pick(args.iterations, cfg, "iterations", 20000))
    batch_size = int(_pick(args.batch_size, cfg, "batch_size", 1))
    seed = int(_pick(args.seed, cfg, "seed", 42))

    # LoRA stacks
    student_rank = int(_pick(args.student_rank, cfg, "network.student_rank", 48))
    fake_rank = int(_pick(args.fake_rank, cfg, "network.fake_rank", 48))
    student_alpha = float(_flatten(cfg, "network.student_alpha", student_rank))
    fake_alpha = float(_flatten(cfg, "network.fake_alpha", fake_rank))
    attn_mode = _pick(args.attn_mode, cfg, "network.attn_mode", "flash")
    # use_custom_down_autograd lives at TOML top level (matches the LoRA family's
    # config layout in methods/lora.toml). CLI flag wins when set explicitly.
    if args.use_custom_down_autograd is None:
        use_custom_down_autograd = bool(
            _flatten(cfg, "use_custom_down_autograd", False)
        )
    else:
        use_custom_down_autograd = bool(args.use_custom_down_autograd)
    # channel_scaling_alpha — top-level TOML scalar (mirrors base.toml's LoRA
    # family layout); CLI override wins. Defaults off so existing turbo
    # snapshots reproduce bit-for-bit.
    channel_scaling_alpha = float(
        _pick(args.channel_scaling_alpha, cfg, "channel_scaling_alpha", 0.0)
    )

    # Masked loss
    if args.use_masked_loss is None:
        use_masked_loss = bool(_flatten(cfg, "use_masked_loss", False))
    else:
        use_masked_loss = bool(args.use_masked_loss)
    mask_dir = _pick(args.mask_dir, cfg, "mask_dir", "post_image_dataset/masks")

    # DMD core
    student_steps = int(_pick(args.student_steps, cfg, "dmd.student_steps", 4))
    teacher_cfg = float(_flatten(cfg, "dmd.teacher_cfg", 4.0))
    # DM-branch gradient policy: (a) τ-damping [default] vs (b) DMD per-sample
    # x0-space magnitude normalization. Alternative policies, not additive;
    # (b) ≈ "drop the τ-weight, magnitude-normalize."
    dm_x0_norm = bool(_pick(args.dm_x0_norm, cfg, "dmd.dm_x0_norm", False))
    norm_floor = float(_pick(args.norm_floor, cfg, "dmd.norm_floor", 0.05))
    # Grad-step policy (all|last|random): which rollout step(s) carry gradient in
    # plain DMD2. CLI --dmd_grad_step wins, else TOML dmd.grad_step, else 'all'
    # (full-rollout BPTT).
    dmd_grad_step = str(_pick(args.dmd_grad_step, cfg, "dmd.grad_step", "all"))

    # Base objective selector. 'dpdmd' keeps the first-step teacher anchor;
    # 'dmd' is plain DMD2 (no anchor → student_steps >= 1 allowed).
    base_loss = _pick(args.base_loss, cfg, "base_loss", "dpdmd")

    # DMD2 teacher-feature GAN (idea 1) + f-distill (idea 2). weight_gen=0 keeps
    # the whole GAN/disc path off → byte-identical DP-DMD. feature_block_idx uses
    # sentinel -2 (not -1) because -1 is a meaningful value (middle block).
    gan_loss_weight_gen = float(
        _pick(args.gan_loss_weight_gen, cfg, "gan.weight_gen", 0.0)
    )
    if args.gan_feature_block_idx != -2:
        gan_feature_block_idx = int(args.gan_feature_block_idx)
    else:
        gan_feature_block_idx = int(_flatten(cfg, "gan.feature_block_idx", -1))
    gan_disc_lr = float(_pick(args.gan_disc_lr, cfg, "gan.disc_lr", 1e-5))
    gan_disc_hidden = int(_flatten(cfg, "gan.disc_hidden", 0))
    gan_r1_weight = float(_pick(args.gan_r1_weight, cfg, "gan.r1_weight", 0.0))
    gan_r1_alpha = float(_flatten(cfg, "gan.r1_alpha", 0.1))
    gan_use_same_t_noise = bool(_flatten(cfg, "gan.use_same_t_noise", True))
    # Selectively checkpoint the single grad-bearing GAN gen teacher forward.
    # Independent of the global ``--grad_ckpt`` (which arms unsloth offload for
    # ALL grad-bearing forwards): the GAN gen forward retains ~half the DiT's
    # block activations only to backprop into x_pred → student, so recomputing
    # them in backward reclaims that peak VRAM (~one half-depth forward of extra
    # compute) without paying the global recompute. Default on — numerically
    # equivalent (frozen teacher, no dropout).
    gan_grad_ckpt = bool(_flatten(cfg, "gan.grad_ckpt", True))
    f_div = _pick(args.f_div, cfg, "f_distill.f_div", "rkl")
    f_ratio_lower = float(_flatten(cfg, "f_distill.ratio_lower", 0.1))
    f_ratio_upper = float(_flatten(cfg, "f_distill.ratio_upper", 20.0))
    f_ratio_ema_rate = float(_flatten(cfg, "f_distill.ratio_ema_rate", 0.0))
    f_bin_num = int(_flatten(cfg, "f_distill.bin_num", 10))
    f_ratio_normalization = bool(_flatten(cfg, "f_distill.ratio_normalization", True))

    # Turbo × REPA relational alignment (turbo_repa.md Phase 1). weight=0 keeps
    # the whole path off → byte-identical DP-DMD (no dataset PE loading, no
    # extra RNG draws, no extra forward).
    repa_weight = float(_pick(args.repa_weight, cfg, "repa.weight", 0.0))
    repa_layer = int(_pick(args.repa_layer, cfg, "repa.layer", 8))
    repa_encoder = _pick(args.repa_encoder, cfg, "repa.encoder", "pe_spatial")
    repa_every_n = int(_pick(args.repa_every_n, cfg, "repa.every_n", 4))
    if args.repa_spatial_norm is None:
        repa_spatial_norm = bool(_flatten(cfg, "repa.spatial_norm", True))
    else:
        repa_spatial_norm = bool(args.repa_spatial_norm)
    # REPA-DoG target band-pass (docs/proposal/repa_dog_target.md). When on it
    # replaces the spatial_norm DC-removal block in the relational loss (dog
    # wins; they're the same family — DoG at σ₁→0 is DC removal).
    if args.repa_target_dog is None:
        repa_target_dog = bool(_flatten(cfg, "repa.target_dog", False))
    else:
        repa_target_dog = bool(args.repa_target_dog)
    repa_dog_sigma1_div = float(
        _pick(args.repa_dog_sigma1_div, cfg, "repa.dog_sigma1_div", 16.0)
    )
    repa_dog_sigma2_div = float(
        _pick(args.repa_dog_sigma2_div, cfg, "repa.dog_sigma2_div", 0.0)
    )
    repa_dog_norm_std = float(
        _pick(args.repa_dog_norm_std, cfg, "repa.dog_norm_std", 0.0)
    )

    # Per-step expert (dual-B-head student). step_expert_K is derived from
    # student_steps so head k ↔ denoise step k by construction (the plan's
    # K = student_steps invariant). K==1 (single step) would collapse to a
    # plain LoRA, so the network factory ignores it there.
    if args.per_step_expert is None:
        per_step_expert = bool(_flatten(cfg, "network.per_step_expert", False))
    else:
        per_step_expert = bool(args.per_step_expert)
    step_expert_K = student_steps if per_step_expert else 0

    # DP-DMD — diversity-anchor knobs.
    k_anchor = int(_pick(args.k_anchor, cfg, "dpdmd.k_anchor", 5))
    teacher_anchor_steps = int(
        _pick(args.teacher_anchor_steps, cfg, "dpdmd.teacher_anchor_steps", 28)
    )
    div_weight = float(_pick(args.div_weight, cfg, "dpdmd.div_weight", 5e-2))
    if args.detach_after_first is None:
        detach_after_first = bool(_flatten(cfg, "dpdmd.detach_after_first", True))
    else:
        detach_after_first = bool(args.detach_after_first)
    flow_shift = float(_pick(args.flow_shift, cfg, "sampling.flow_shift", 3.0))

    # Mean-variance reg (lever B / Eq. 7). weight=0 disables. Target stats are
    # pinned (sigma2_t > 0) or measured exactly in a one-pass scan over the real
    # latents (sigma2_t <= 0). The target is a static dataset statistic — a global
    # scalar (μ, σ²) over the cached training latents — so an exact pre-pass beats
    # the old running EMA (no decay lag, no batch wobble, deterministic).
    # `calib_batches` caps that scan (0 = full pass; the global scalar converges
    # in a few hundred images, so a cap is a cheap-but-near-exact knob).
    mean_var_weight = float(_pick(args.mean_var_weight, cfg, "mean_var.weight", 0.0))
    mv_mu_t = float(_flatten(cfg, "mean_var.mu_t", 0.0))
    mv_sigma2_t = float(_flatten(cfg, "mean_var.sigma2_t", -1.0))
    mv_calib_batches = int(_flatten(cfg, "mean_var.calib_batches", 0))

    # Optimizer
    student_lr = float(_pick(args.student_lr, cfg, "optim.student_lr", 1e-5))
    fake_lr = float(_pick(args.fake_lr, cfg, "optim.fake_lr", 1e-5))
    fake_steps_per_student_step = int(
        _pick(
            args.fake_steps_per_student_step,
            cfg,
            "optim.fake_steps_per_student_step",
            1,
        )
    )
    fake_warmup_steps = int(
        _pick(args.fake_warmup_steps, cfg, "optim.fake_warmup_steps", 0)
    )
    weight_decay = float(_flatten(cfg, "optim.weight_decay", 0.0))
    grad_clip = float(_flatten(cfg, "optim.grad_clip", 1.0))

    # Sampling
    t_distribution = _flatten(cfg, "sampling.t_distribution", "uniform")
    sigmoid_scale = float(_flatten(cfg, "sampling.sigmoid_scale", 1.0))

    # ----- Validation -----
    if base_loss not in ("dpdmd", "dmd"):
        raise ValueError(f"base_loss={base_loss!r}: expected 'dpdmd' or 'dmd'")
    use_anchor = base_loss == "dpdmd"
    if use_anchor and student_steps < 2:
        raise ValueError(
            f"DP-DMD requires dmd.student_steps >= 2 (got {student_steps}): step 1 "
            "is diversity-supervised + detached, so at least one further step must "
            "carry the DMD loss. (Use base_loss='dmd' for a 1-step student.)"
        )
    if not use_anchor and student_steps < 1:
        raise ValueError(
            f"base_loss='dmd' requires dmd.student_steps >= 1 (got {student_steps})."
        )
    if use_anchor and not (1 <= k_anchor < teacher_anchor_steps):
        raise ValueError(
            f"dpdmd.k_anchor={k_anchor} must satisfy 1 <= k_anchor < "
            f"teacher_anchor_steps={teacher_anchor_steps}."
        )
    if div_weight < 0.0:
        raise ValueError(f"dpdmd.div_weight={div_weight}: must be >= 0")
    if dmd_grad_step not in ("all", "last", "random"):
        raise ValueError(
            f"dmd.grad_step={dmd_grad_step!r}: expected 'all', 'last', or 'random'"
        )
    if (
        not use_anchor
        and student_steps > 1
        and not bool(args.grad_ckpt)
        and dmd_grad_step == "all"
    ):
        logger.warning(
            "base_loss='dmd' with student_steps=%d, grad_ckpt OFF, "
            "dmd.grad_step='all': plain DMD2 has no first-step anchor to detach, "
            "so the student backward holds the FULL %d-step rollout graph (≈%dx the "
            "activation memory of dpdmd@%d). Use student_steps=1 (the replacement "
            "arm), dmd.grad_step='random'/'last' (memory-flat), or --grad_ckpt.",
            student_steps,
            student_steps,
            student_steps,
            student_steps,
        )
    if dmd_grad_step == "last" and per_step_expert:
        logger.warning(
            "dmd.grad_step='last' with per_step_expert=True: only the final step's "
            "head receives gradient, so heads 0..N-2 never train. Use "
            "dmd.grad_step='random' (each iteration trains the sampled step's head) "
            "or 'all'."
        )
    if dmd_grad_step == "last":
        logger.info(
            "dmd.grad_step='last': rollout steps 0..N-2 run no_grad; only the final "
            "step backprops to x_pred (memory-flat at any student_steps)."
        )
    elif dmd_grad_step == "random" and use_anchor:
        logger.info(
            "dmd.grad_step='random' under base_loss='dpdmd': step 0 keeps the "
            "diversity anchor (detached); each iteration then samples a refinement "
            "step g~U{1..N-1}, backward-simulates the 1..g-1 prefix under no_grad, "
            "and grads only step g's one-step x0-prediction (memory-flat; supervises "
            "every refinement grid point + trains every head under per_step_expert, "
            "vs 'last' which only ever grads the clean tail)."
        )
    elif dmd_grad_step == "random":
        logger.info(
            "dmd.grad_step='random': canonical DMD2 multistep — each iteration "
            "samples g~U{0..N-1}, backward-simulates to g under no_grad, and grads "
            "only step g's one-step x0-prediction (memory-flat; supervises every "
            "grid point, not just the clean tail)."
        )
    if gan_loss_weight_gen < 0.0:
        raise ValueError(f"gan.weight_gen={gan_loss_weight_gen}: must be >= 0")
    if repa_weight < 0.0:
        raise ValueError(f"repa.weight={repa_weight}: must be >= 0")
    if repa_weight > 0.0:
        if repa_every_n < 1:
            raise ValueError(f"repa.every_n={repa_every_n}: must be >= 1")
        if int(args.blocks_to_swap) > 0:
            # The feature tap's early exit leaves the tail blocks' offloader
            # moves un-submitted (forward_mini_train_dit raises the same way);
            # fail at config time with the actionable message.
            raise ValueError(
                "repa.weight > 0 requires blocks_to_swap=0 — the block-feature "
                "tap is unsupported under block swap (turbo keeps the DiT "
                "resident by default)."
            )
        if per_step_expert and bool(args.grad_ckpt):
            # Same global-state × deferred-ckpt-recompute class as the view bug:
            # the REPA forward re-routes the student head (nearest-τ) AFTER the
            # rollout's checkpointed forwards but BEFORE their backward, so the
            # step-g recompute would run the wrong head — silent corruption.
            raise ValueError(
                "repa.weight > 0 with per_step_expert AND --grad_ckpt: the REPA "
                "head switch corrupts the rollout's checkpoint recompute. "
                "Disable one of the three."
            )
        if repa_target_dog and repa_dog_sigma1_div <= 0.0:
            raise ValueError(
                f"repa.dog_sigma1_div={repa_dog_sigma1_div}: must be > 0 "
                "(σ₁ = min(gh,gw)/div)."
            )
        # dog replaces spatial_norm's DC removal — they're mutually exclusive
        # (same family; dog wins). Surface the effective target preprocessing.
        target_desc = (
            f"dog(σ1=min/{repa_dog_sigma1_div:g}, "
            f"σ2={'off' if repa_dog_sigma2_div <= 0 else f'min/{repa_dog_sigma2_div:g}'}, "
            f"norm_std={'empirical' if repa_dog_norm_std <= 0 else repa_dog_norm_std})"
            if repa_target_dog
            else f"spatial_norm={repa_spatial_norm}"
        )
        logger.info(
            f"REPA (turbo×REPA relational alignment) ON: weight={repa_weight}, "
            f"layer={repa_layer}, encoder={repa_encoder!r}, "
            f"every_n={repa_every_n}, {target_desc}."
        )
    if bool(args.grad_ckpt) and gan_loss_weight_gen > 0.0:
        # Pre-existing member of the same hazard class (predates REPA): under
        # global --grad_ckpt the rollout's student forwards are checkpointed,
        # but the GAN gen forward switches the network view to 'teacher' before
        # loss_student.backward() — the rollout recompute then runs WITHOUT the
        # student LoRA enabled and the DMD gradient is silently corrupted.
        logger.warning(
            "--grad_ckpt with gan.weight_gen > 0: the rollout's checkpointed "
            "student forwards recompute after the GAN gen forward flipped the "
            "view to 'teacher' — the recomputed blocks drop the student LoRA "
            "and the DMD gradient is silently corrupted. Known-broken "
            "combination; turn one of the two off."
        )
    if gan_r1_weight < 0.0:
        raise ValueError(f"gan.r1_weight={gan_r1_weight}: must be >= 0")
    _F_DIVS = ("rkl", "kl", "js", "sf", "neyman", "sh", "jf")
    if f_div not in _F_DIVS:
        raise ValueError(f"f_distill.f_div={f_div!r}: expected one of {_F_DIVS}")
    if f_div != "rkl" and gan_loss_weight_gen <= 0.0:
        # r = exp(disc_logits) only exists once the GAN disc is built (idea 1).
        raise ValueError(
            f"f_distill.f_div={f_div!r} requires gan.weight_gen > 0 — the "
            "f-divergence weight reads the GAN discriminator's logits."
        )
    if not (0.0 < f_ratio_lower < f_ratio_upper):
        raise ValueError(
            f"f_distill: require 0 < ratio_lower ({f_ratio_lower}) < "
            f"ratio_upper ({f_ratio_upper})"
        )
    if not (0.0 <= f_ratio_ema_rate < 1.0):
        raise ValueError(
            f"f_distill.ratio_ema_rate={f_ratio_ema_rate}: must be in [0, 1)"
        )
    if f_bin_num < 1:
        raise ValueError(f"f_distill.bin_num={f_bin_num}: must be >= 1")
    if gan_loss_weight_gen > 0.0:
        logger.info(
            f"GAN (DMD2 teacher-feature disc, FastGen idea 1) ON: "
            f"weight_gen={gan_loss_weight_gen}, feature_block_idx="
            f"{gan_feature_block_idx} (-1 = middle), disc_lr={gan_disc_lr}, "
            f"r1_weight={gan_r1_weight}, use_same_t_noise={gan_use_same_t_noise}."
        )
        if f_div != "rkl":
            logger.info(
                f"f-distill (FastGen idea 2) ON: f_div={f_div!r}, ratio∈"
                f"[{f_ratio_lower}, {f_ratio_upper}], ema_rate={f_ratio_ema_rate}, "
                f"bin_num={f_bin_num}, normalization={f_ratio_normalization}."
            )
    if flow_shift <= 0.0:
        raise ValueError(f"sampling.flow_shift={flow_shift}: must be > 0")
    if use_anchor and not detach_after_first:
        logger.warning(
            "detach_after_first=False: the mode-seeking DMD gradient can override "
            "the diversity mapping (their Fig 5). A/B only — keep True for "
            "production."
        )
    if t_distribution not in ("uniform", "sigmoid"):
        raise ValueError(
            f"sampling.t_distribution={t_distribution!r}: expected 'uniform' or 'sigmoid'"
        )
    if fake_rank < student_rank:
        logger.warning(
            f"fake_rank={fake_rank} < student_rank={student_rank}: DM regularizer "
            "has less capacity than the student — proposal R1 risk amplified. "
            "Consider bumping fake_rank to 2 x student_rank."
        )
    if norm_floor <= 0.0:
        raise ValueError(f"dmd.norm_floor={norm_floor}: must be > 0 (latent scale)")
    if fake_steps_per_student_step < 1:
        raise ValueError(
            f"optim.fake_steps_per_student_step={fake_steps_per_student_step}: must be ≥ 1"
        )
    if args.single_prompt_idx is not None and batch_size != 1:
        # single-prompt mode slices the dataset to one sample. With drop_last=True
        # and batch_size > 1 the dataloader yields zero batches and the loop
        # silently no-ops.
        raise ValueError(
            f"--single_prompt_idx requires batch_size=1 (got {batch_size}). "
            "Single-prompt overfit mode pins the dataset to one sample; a "
            "batch_size > 1 dataloader with drop_last=True would yield zero batches."
        )
    if mean_var_weight < 0.0:
        raise ValueError(f"mean_var.weight={mean_var_weight}: must be ≥ 0")
    if mean_var_weight > 0.0:
        mv_auto = mv_sigma2_t <= 0.0
        logger.info(
            f"mean-variance reg ENABLED (Eq.7): weight={mean_var_weight}, target="
            + (
                "exact one-pass over real latents"
                + (
                    " (full pass)"
                    if mv_calib_batches <= 0
                    else f" (≤{mv_calib_batches} batches)"
                )
                if mv_auto
                else f"fixed μ_t={mv_mu_t}, σ²_t={mv_sigma2_t}"
            )
        )
    logger.info(
        "DM gradient policy: "
        + (
            f"(b) x0-norm, norm_floor={norm_floor} — τ cancels, ≈ magnitude-normalized"
            if dm_x0_norm
            else "(a) τ-damping [default]"
        )
    )
    if use_anchor:
        logger.info(
            "DP-DMD: first-step diversity anchor "
            f"k_anchor={k_anchor}/{teacher_anchor_steps} teacher steps, "
            f"div_weight={div_weight}, detach_after_first={detach_after_first}, "
            f"student N={student_steps} @ flow_shift={flow_shift}, "
            f"teacher_cfg={teacher_cfg}."
        )
    else:
        logger.info(
            f"plain DMD2 (no diversity anchor): student N={student_steps} @ "
            f"flow_shift={flow_shift}, teacher_cfg={teacher_cfg}."
        )
    if per_step_expert:
        if not detach_after_first:
            logger.warning(
                "per_step_expert=True with detach_after_first=False: the step-0 "
                "and DMD graphs stay entangled, so the diversity gradient reaches "
                "the DMD heads (and vice versa) through the shared rollout — the "
                "head split no longer cleanly separates the two objectives. Keep "
                "detach_after_first=True with per_step_expert."
            )
        logger.info(
            f"per-step-expert student ON: K={step_expert_K} up-heads / Linear "
            f"(head k ↔ denoise step k) off a shared down-proj. Output is "
            "kept-live only (not a plain LoRA; merge refuses it)."
        )

    return TurboConfig(
        dit_path=dit_path,
        data_dir=data_dir,
        output_dir=output_dir,
        output_name=output_name,
        log_dir=log_dir,
        save_every=save_every,
        log_interval=log_interval,
        no_log=bool(args.no_log),
        validate_every_n_steps=validate_every_n_steps,
        val_diversity_seeds=val_diversity_seeds,
        val_prompt_idx=val_prompt_idx,
        iterations=iterations,
        batch_size=batch_size,
        seed=seed,
        sample_ratio=float(args.sample_ratio),
        single_prompt_idx=args.single_prompt_idx,
        student_rank=student_rank,
        fake_rank=fake_rank,
        student_alpha=student_alpha,
        fake_alpha=fake_alpha,
        attn_mode=attn_mode,
        use_custom_down_autograd=use_custom_down_autograd,
        channel_scaling_alpha=channel_scaling_alpha,
        per_step_expert=per_step_expert,
        step_expert_K=step_expert_K,
        use_masked_loss=use_masked_loss,
        mask_dir=mask_dir,
        k_anchor=k_anchor,
        teacher_anchor_steps=teacher_anchor_steps,
        div_weight=div_weight,
        detach_after_first=detach_after_first,
        flow_shift=flow_shift,
        student_steps=student_steps,
        teacher_cfg=teacher_cfg,
        dm_x0_norm=dm_x0_norm,
        norm_floor=norm_floor,
        dmd_grad_step=dmd_grad_step,
        base_loss=base_loss,
        gan_loss_weight_gen=gan_loss_weight_gen,
        gan_feature_block_idx=gan_feature_block_idx,
        gan_disc_lr=gan_disc_lr,
        gan_disc_hidden=gan_disc_hidden,
        gan_r1_weight=gan_r1_weight,
        gan_r1_alpha=gan_r1_alpha,
        gan_use_same_t_noise=gan_use_same_t_noise,
        gan_grad_ckpt=gan_grad_ckpt,
        f_div=f_div,
        f_ratio_lower=f_ratio_lower,
        f_ratio_upper=f_ratio_upper,
        f_ratio_ema_rate=f_ratio_ema_rate,
        f_bin_num=f_bin_num,
        f_ratio_normalization=f_ratio_normalization,
        repa_weight=repa_weight,
        repa_layer=repa_layer,
        repa_encoder=repa_encoder,
        repa_every_n=repa_every_n,
        repa_spatial_norm=repa_spatial_norm,
        repa_target_dog=repa_target_dog,
        repa_dog_sigma1_div=repa_dog_sigma1_div,
        repa_dog_sigma2_div=repa_dog_sigma2_div,
        repa_dog_norm_std=repa_dog_norm_std,
        mean_var_weight=mean_var_weight,
        mv_mu_t=mv_mu_t,
        mv_sigma2_t=mv_sigma2_t,
        mv_calib_batches=mv_calib_batches,
        student_lr=student_lr,
        fake_lr=fake_lr,
        fake_steps_per_student_step=fake_steps_per_student_step,
        fake_warmup_steps=fake_warmup_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        t_distribution=t_distribution,
        sigmoid_scale=sigmoid_scale,
        blocks_to_swap=int(args.blocks_to_swap),
        grad_ckpt=bool(args.grad_ckpt),
        torch_compile=bool(args.torch_compile),
        compile_dynamic_seq=bool(
            _pick(args.compile_dynamic_seq, cfg, "compile_dynamic_seq", True)
        ),
        target_res=(
            [int(e) for e in args.target_res]
            if args.target_res is not None
            else (_flatten(cfg, "target_res", None))
        ),
        dynamo_recompile_limit=int(_flatten(cfg, "dynamo_recompile_limit", 64)),
        activation_memory_budget=float(
            _pick(args.activation_memory_budget, cfg, "activation_memory_budget", 1.0)
        ),
    )


def snapshot_toml_text(c: TurboConfig, *, source_config: str | None = None) -> str:
    """Render the fully-resolved turbo config as a provenance TOML snapshot.

    Unlike :func:`tb_config_text` (a TB summary of a hand-picked subset), this
    dumps *every* resolved field — CLI overrides folded in — so the run log dir
    becomes a self-contained record of "this run + the config that produced it".
    It's the turbo analogue of the ``<output_name>.snapshot.toml`` that
    ``train.py`` writes for the LoRA family (the bespoke turbo config never went
    through that path).
    """
    import dataclasses

    import tomlkit

    doc = tomlkit.document()
    doc.add(tomlkit.comment("Anima turbo distillation — resolved config snapshot"))
    if source_config:
        doc.add(tomlkit.comment(f"source config: {source_config}"))
    doc.add(tomlkit.nl())
    for k, v in dataclasses.asdict(c).items():
        if v is None:
            # tomlkit has no null; record the field as unset rather than drop it.
            doc.add(tomlkit.comment(f"{k} = (unset)"))
        else:
            doc[k] = v
    return tomlkit.dumps(doc)


def tb_config_text(c: TurboConfig) -> str:
    """Formatted TensorBoard config summary (same key set as v1)."""
    pairs = {
        "base_loss": c.base_loss,
        "gan_loss_weight_gen": c.gan_loss_weight_gen,
        "repa_weight": c.repa_weight,
        "repa_target_dog": c.repa_target_dog,
        "repa_dog_sigma1_div": c.repa_dog_sigma1_div,
        "repa_dog_sigma2_div": c.repa_dog_sigma2_div,
        "repa_dog_norm_std": c.repa_dog_norm_std,
        "f_div": c.f_div,
        "k_anchor": c.k_anchor,
        "teacher_anchor_steps": c.teacher_anchor_steps,
        "div_weight": c.div_weight,
        "detach_after_first": c.detach_after_first,
        "flow_shift": c.flow_shift,
        "student_rank": c.student_rank,
        "fake_rank": c.fake_rank,
        "channel_scaling_alpha": c.channel_scaling_alpha,
        "student_steps": c.student_steps,
        "teacher_cfg": c.teacher_cfg,
        "fake_warmup_steps": c.fake_warmup_steps,
        "student_lr": c.student_lr,
        "fake_lr": c.fake_lr,
        "fake_steps_per_student_step": c.fake_steps_per_student_step,
        "iterations": c.iterations,
        "batch_size": c.batch_size,
        "t_distribution": c.t_distribution,
        "mean_var_weight": c.mean_var_weight,
        "use_masked_loss": c.use_masked_loss,
        "data_dir": c.data_dir,
        "dit_path": c.dit_path,
    }
    return "  \n".join(f"{k}: {v}" for k, v in pairs.items())
