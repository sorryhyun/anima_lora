"""Turbo distillation config: TOML loader + argparser + CLI/TOML resolver.

The resolved knobs are returned as a ``TurboConfig`` frozen dataclass so the
training loop never reaches back into ``args``/``cfg`` mid-step.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

from library.config.io import toml_get as _flatten
from library.config.resolved import (
    dataclass_snapshot_toml,
    dataclass_tb_text,
)
from library.config.resolved import pick as _pick


logger = logging.getLogger(__name__)


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
    # Sentinel defaults (None / -1) mean "use the TOML value"; the defaults
    # behind the TOML live in resolve_config, the knob docs in
    # docs/anima/methods/turbo.md.
    parser.add_argument("--dit_path", type=str, default=None)
    parser.add_argument(
        "--dit_nonquant_path",
        type=str,
        default=None,
        help="The bf16 DiT an int8 --dit_path was quantized from (the float "
        "weights wq_t is built from); unset, they are rebuilt from the int8 copies.",
    )
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
        help="DAVE same-prompt diversity probe every N steps (0 off; diversity.py).",
    )
    parser.add_argument(
        "--val_diversity_seeds",
        type=int,
        default=-1,
        help="Seeds the diversity probe rolls (>=2).",
    )
    parser.add_argument(
        "--val_prompt_idx",
        type=int,
        default=-1,
        help="Held-out sample the probe conditions on (-1: the last one, not "
        "--single_prompt_idx).",
    )
    parser.add_argument("--student_rank", type=int, default=-1)
    parser.add_argument("--fake_rank", type=int, default=-1)
    parser.add_argument(
        "--channel_scaling_alpha",
        type=float,
        default=-1.0,
        help="Per-input-channel rebalance absorbed into lora_down (student and "
        "fake); 0 off, 0.5 sqrt-balance.",
    )
    parser.add_argument("--student_lr", type=float, default=-1.0)
    parser.add_argument(
        "--lr_schedule",
        choices=["cosine", "constant"],
        default=None,
        help="Shape after the 2%% linear warmup, for student, fake and disc: "
        "cosine to 0.1·lr (default) or constant.",
    )
    parser.add_argument(
        "--ema_window",
        type=float,
        default=-1.0,
        help="Student EMA span as a fraction of the run (0 = off); saves a "
        "*_ema.safetensors beside each checkpoint.",
    )
    parser.add_argument("--fake_lr", type=float, default=-1.0)
    parser.add_argument(
        "--fake_steps_per_student_step",
        type=int,
        default=-1,
        help="Fake updates per student step, each on fresh (τ, ε).",
    )
    parser.add_argument(
        "--fake_warmup_steps",
        type=int,
        default=-1,
        help="Fake-only head-start updates before the main loop. Without it the "
        "warm-start fake is miscalibrated when the student reaches full LR "
        "(grad_signal_rms spike near step 50).",
    )
    parser.add_argument(
        "--student_steps",
        type=int,
        default=-1,
        help="Sampler step count baked into the student.",
    )
    parser.add_argument(
        "--dm_x0_norm",
        dest="dm_x0_norm",
        action="store_const",
        const=True,
        default=None,
        help="Replace τ-damping with per-sample x0-space normalisation: "
        "τ·Δ / clamp(τ·mean|v_real|, norm_floor), so τ cancels.",
    )
    parser.add_argument(
        "--norm_floor",
        type=float,
        default=-1.0,
        help="Floor of the --dm_x0_norm denominator (latent scale).",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--attn_mode", type=str, default="flash")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    parser.add_argument(
        "--torch_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the DiT blocks (on by default).",
    )
    parser.add_argument(
        "--compile_dynamic_seq",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="One block graph with a dynamic sequence axis instead of one per "
        "token count (as in LoRA training).",
    )
    parser.add_argument(
        "--target_res",
        type=int,
        nargs="+",
        default=None,
        help="Tier edges that size the dynamic-seq bound and the dynamo budget; "
        "unset, they come from the token counts in the cached pool.",
    )
    parser.add_argument(
        "--activation_memory_budget",
        type=float,
        default=None,
        help="Compile partitioner saved-activation fraction (<1 recomputes in "
        "backward). Ignored under --grad_ckpt.",
    )
    parser.add_argument(
        "--int8",
        choices=("off", "fwd", "full", "stream"),
        default=None,
        help="Frozen DiT linears as int8 GEMMs, teacher included "
        "(docs/anima/optimizations/int8.md).",
    )
    parser.add_argument(
        "--int8_file",
        type=str,
        default=None,
        help="GPTQ forward copies; unset: <dit>_int8_gptq.safetensors beside the "
        "DiT, else round to nearest; 'rtn' forces round to nearest.",
    )
    parser.add_argument(
        "--stream",
        choices=("bf16", "fp32"),
        default=None,
        help="Residual stream precision, teacher included "
        "(docs/anima/optimizations/fp32_stream.md).",
    )
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Crash-resume bundle (nets, optimizers, schedulers, RNG). 'auto' "
        "uses output/ckpt/<name>/<name>_resume.pt if present, else starts fresh; "
        "a named path must exist. Written every --save_every.",
    )
    parser.add_argument("--log_interval", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument(
        "--single_prompt_idx",
        type=int,
        default=None,
        help="Overfit mode: pin the dataloader to one (latent, text) pair.",
    )
    parser.add_argument("--sample_ratio", type=float, default=1.0)

    # DP-DMD (arXiv:2602.03139).
    parser.add_argument(
        "--k_anchor",
        type=int,
        default=-1,
        help="Teacher steps rolled to the diversity anchor (their K).",
    )
    parser.add_argument(
        "--teacher_anchor_steps",
        type=int,
        default=-1,
        help="Teacher σ grid the K anchor is counted on.",
    )
    parser.add_argument(
        "--div_weight",
        type=float,
        default=-1.0,
        help="λ on the first-step diversity loss; 0 skips the teacher anchor.",
    )
    parser.add_argument(
        "--dm_first_step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw the grad step from every rollout step, so step 0 also gets the "
        "DM / MFD signal (off: refinement steps only).",
    )
    parser.add_argument(
        "--rollout_grad",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the graph from step 0 into the grad step, so the DM / MFD "
        "signal on the final x_θ reaches step 0 through step 1 (2-step static "
        "grid only; superturbo.md).",
    )
    parser.add_argument(
        "--detach_after_first",
        dest="detach_after_first",
        action="store_const",
        const=True,
        default=None,
        help="Stop-grad after the diversity-supervised first step (keep on but "
        "for A/B).",
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
        help="σ shift of the student and teacher Euler grids (as at inference).",
    )

    # DMD2 teacher-feature GAN.
    parser.add_argument(
        "--gan_loss_weight_gen",
        type=float,
        default=-1.0,
        help="λ on the GAN generator term; 0 turns the whole GAN path off "
        "(FastGen QwenImage: 0.03).",
    )
    parser.add_argument(
        "--gan_feature_block_idx",
        type=int,
        default=-2,
        help="DiT block the disc taps; -1 = middle. (-2 is the 'use TOML' sentinel.)",
    )
    parser.add_argument("--gan_disc_lr", type=float, default=-1.0)
    parser.add_argument(
        "--gan_r1_weight",
        type=float,
        default=-1.0,
        help="Approximate R1 (APT): MSE between real logits and those of a "
        "perturbed real input; 0 off.",
    )
    parser.add_argument(
        "--gan_disc_head",
        type=str,
        default=None,
        choices=["pooled", "token"],
        help="pooled: one logit per tap; token: the same MLP per token (LADD).",
    )
    parser.add_argument(
        "--gan_delay_steps",
        type=int,
        default=-1,
        help="Generator-side GAN weight held at 0 for N student steps (the disc "
        "still trains).",
    )
    parser.add_argument(
        "--gan_warmup_steps",
        type=int,
        default=-1,
        help="Then ramp it 0 → weight_gen over N steps (0 = instant).",
    )

    # CDM off-trajectory loss (arXiv:2605.06376 §3.3).
    parser.add_argument(
        "--cdm_grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint the grad-bearing CDM student forward.",
    )
    parser.add_argument(
        "--cdm_weight",
        type=float,
        default=-1.0,
        help="λ on L_CDM; 0 turns the path off. Costs +1 student grad forward, "
        "+2 teacher and +1 fake no-grad forwards per step.",
    )

    # Mean Flow Distillation in the DM slot (arXiv:2606.11155).
    parser.add_argument(
        "--mfd_dtau",
        type=float,
        default=None,
        help="Euler step of the MFD mean-flow integral; 0 = plain DMD. Costs K = "
        "ceil((σ_a − σ_b)/dtau) ≤ mfd_max_steps teacher-CFG and fake forwards "
        "instead of 1.",
    )
    parser.add_argument(
        "--mfd_max_steps",
        type=int,
        default=None,
        help="Cap on K per mean-flow integral.",
    )
    parser.add_argument(
        "--mfd_interval",
        type=str,
        default=None,
        choices=("to_zero", "uniform"),
        help="to_zero: σ_b = 0 (K=1 is exactly DMD); uniform: σ_b ~ U(0, σ_a).",
    )

    return parser


@dataclass(frozen=True)
class TurboConfig:
    # Paths / IO
    dit_path: str
    data_dir: str
    output_dir: str
    output_name: str
    log_dir: str
    save_every: int
    resume: str
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
    # Per-input-channel rebalance on each lora_down; merges out at save.
    channel_scaling_alpha: float
    student_down_init: str  # kaiming | weight_svd
    fake_down_init: str
    # Plain LoRA file seeding the stack's ΔW (SVD-truncated to its rank); "" = off.
    student_init_weights: str
    fake_init_weights: str
    # adaln_up_{branch} targets; the student ships them in the ComfyUI layout.
    train_adaln: bool
    # false drops the fake's 84 adaln modules (~0.8 GB at rank 96; unbenched).
    fake_adaln: bool
    # Own rank / alpha for the adaln modules (0 = network's). Scale-preserving
    # alpha: student_alpha × adaln_rank / student_rank.
    adaln_rank: int
    adaln_alpha: float
    # Transition-aware fake (vatdm.md idea 1): σ of the student step its sample
    # came from, as a zero-init delta on the fake's t-embedding.
    fake_transition: bool
    # Partitioner recompute tuning (as train.py's partitioner_*); ignored under
    # grad_ckpt.
    partitioner_recompute_views: bool
    partitioner_aggressive_recomputation: bool

    # DP-DMD
    k_anchor: int
    teacher_anchor_steps: int
    div_weight: float  # 0 = no teacher anchor
    detach_after_first: bool
    # Grad step g ~ U{0..N−1}: step 0 also takes the DM / MFD signal.
    dm_first_step: bool
    # No detach between step 0 and the grad step (N = 2, static grid).
    rollout_grad: bool
    flow_shift: float

    # DMD core
    student_steps: int
    teacher_cfg: float
    dm_x0_norm: bool
    norm_floor: float
    # CDM dynamic schedule: a fresh rollout grid per step (t₁ = 1 pinned);
    # validation and inference stay on the static grid.
    dynamic_schedule: bool

    # L_CDM; weight 0 = off
    cdm_weight: float
    cdm_grad_ckpt: bool

    # MFD in the DM slot; dtau 0 = plain DMD
    mfd_dtau: float
    mfd_max_steps: int
    mfd_interval: str  # to_zero | uniform

    # DMD2 teacher-feature GAN; weight_gen 0 = off
    gan_loss_weight_gen: float
    gan_feature_block_idx: int  # -1 → middle block
    gan_disc_lr: float
    gan_disc_hidden: int  # <= 0 → inner_dim // 2
    gan_disc_head: str  # pooled | token
    gan_delay_steps: int
    gan_warmup_steps: int
    gan_r1_weight: float
    gan_r1_alpha: float
    gan_use_same_t_noise: bool
    gan_grad_ckpt: bool

    # Optimizer + scheduler
    student_lr: float
    fake_lr: float
    fake_steps_per_student_step: int
    fake_warmup_steps: int
    weight_decay: float
    grad_clip: float
    lr_schedule: str  # cosine | constant
    ema_window: float  # student EMA span; 0 = off

    # Sampling distribution
    t_distribution: str
    sigmoid_scale: float

    # Runtime
    blocks_to_swap: int
    grad_ckpt: bool
    torch_compile: bool
    compile_dynamic_seq: bool
    target_res: list[int] | None  # None → from the cached pool
    dynamo_recompile_limit: int  # per block _forward
    activation_memory_budget: float
    int8: str = "off"  # off | fwd | full | stream
    int8_file: str | None = None
    dit_nonquant_path: str | None = None
    stream: str = "bf16"  # bf16 | fp32


# Keys of removed features (2026-11-05) → the only value the loop still runs. A
# TOML that asks for anything else is refused instead of silently running the
# DP-DMD / MFD recipe without it.
_REMOVED_KEYS = {
    "base_loss": "dpdmd",
    "dmd.grad_step": "random",
    "network.per_step_expert": False,
    "network.dual_pool": False,
    "network.fake_tau_banks": 1,
    "use_masked_loss": False,
    "softrank.weight": 0.0,
    "f_distill.f_div": "rkl",
    "dmad.enabled": False,
    "dmad_probe.enabled": False,
    "use_custom_down_autograd": False,
}


def _refuse_removed_keys(cfg: dict) -> None:
    for key, allowed in _REMOVED_KEYS.items():
        value = _flatten(cfg, key, allowed)
        if value != allowed:
            raise ValueError(
                f"{key} = {value!r}: this feature was removed from the turbo loop "
                f"(only {allowed!r} runs). Drop the key."
            )


def resolve_config(args: argparse.Namespace, cfg: dict) -> TurboConfig:
    """Apply CLI/TOML/default precedence and run sanity checks."""
    _refuse_removed_keys(cfg)

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
    # CLI only: a resume belongs to the launch, not the method config.
    resume = str(getattr(args, "resume", "") or "")
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
    student_down_init = str(_flatten(cfg, "network.student_down_init", "kaiming"))
    fake_down_init = str(_flatten(cfg, "network.fake_down_init", "kaiming"))
    for name, down_init in (("student", student_down_init), ("fake", fake_down_init)):
        if down_init not in ("kaiming", "weight_svd"):
            raise ValueError(
                f"network.{name}_down_init={down_init!r}: "
                "expected 'kaiming' or 'weight_svd'."
            )
    attn_mode = _pick(args.attn_mode, cfg, "network.attn_mode", "flash")
    channel_scaling_alpha = float(
        _pick(args.channel_scaling_alpha, cfg, "channel_scaling_alpha", 0.0)
    )

    # DMD core
    student_steps = int(_pick(args.student_steps, cfg, "dmd.student_steps", 4))
    teacher_cfg = float(_flatten(cfg, "dmd.teacher_cfg", 4.0))
    dm_x0_norm = bool(_pick(args.dm_x0_norm, cfg, "dmd.dm_x0_norm", False))
    norm_floor = float(_pick(args.norm_floor, cfg, "dmd.norm_floor", 0.05))
    dynamic_schedule = bool(_flatten(cfg, "dmd.dynamic_schedule", False))

    cdm_weight = float(_pick(args.cdm_weight, cfg, "cdm.weight", 0.0))
    cdm_grad_ckpt = bool(
        _flatten(cfg, "cdm.grad_ckpt", True)
        if args.cdm_grad_ckpt is None
        else args.cdm_grad_ckpt
    )

    mfd_dtau = float(_pick(args.mfd_dtau, cfg, "mfd.dtau", 0.0))
    mfd_max_steps = int(_pick(args.mfd_max_steps, cfg, "mfd.max_steps", 4))
    mfd_interval = str(_pick(args.mfd_interval, cfg, "mfd.interval", "to_zero"))

    # feature_block_idx's sentinel is -2 because -1 means the middle block.
    gan_loss_weight_gen = float(
        _pick(args.gan_loss_weight_gen, cfg, "gan.weight_gen", 0.0)
    )
    if args.gan_feature_block_idx != -2:
        gan_feature_block_idx = int(args.gan_feature_block_idx)
    else:
        gan_feature_block_idx = int(_flatten(cfg, "gan.feature_block_idx", -1))
    gan_disc_lr = float(_pick(args.gan_disc_lr, cfg, "gan.disc_lr", 1e-5))
    gan_disc_hidden = int(_flatten(cfg, "gan.disc_hidden", 0))
    gan_disc_head = str(_pick(args.gan_disc_head, cfg, "gan.disc_head", "pooled"))
    gan_delay_steps = int(_pick(args.gan_delay_steps, cfg, "gan.delay_steps", 0))
    gan_warmup_steps = int(_pick(args.gan_warmup_steps, cfg, "gan.warmup_steps", 0))
    gan_r1_weight = float(_pick(args.gan_r1_weight, cfg, "gan.r1_weight", 0.0))
    gan_r1_alpha = float(_flatten(cfg, "gan.r1_alpha", 0.1))
    gan_use_same_t_noise = bool(_flatten(cfg, "gan.use_same_t_noise", True))
    gan_grad_ckpt = bool(_flatten(cfg, "gan.grad_ckpt", True))
    student_init_weights = str(_flatten(cfg, "network.student_init_weights", ""))
    fake_init_weights = str(_flatten(cfg, "network.fake_init_weights", ""))
    train_adaln = bool(_flatten(cfg, "network.train_adaln", False))
    fake_adaln = bool(_flatten(cfg, "network.fake_adaln", train_adaln))
    adaln_rank = int(_flatten(cfg, "network.adaln_rank", 0))
    if adaln_rank > 0 and not train_adaln:
        raise SystemExit("network.adaln_rank > 0 requires network.train_adaln = true")
    adaln_alpha = float(_flatten(cfg, "network.adaln_alpha", 0.0))
    if adaln_alpha > 0 and not train_adaln:
        raise SystemExit("network.adaln_alpha > 0 requires network.train_adaln = true")
    fake_transition = bool(_flatten(cfg, "network.fake_transition", False))
    for name, path, down_init in (
        ("student", student_init_weights, student_down_init),
        ("fake", fake_init_weights, fake_down_init),
    ):
        if path and down_init != "kaiming":
            raise ValueError(
                f"network.{name}_init_weights is incompatible with "
                f"{name}_down_init='weight_svd' — the warm start overwrites what "
                "it seeds."
            )
        if path and not os.path.exists(path):
            raise ValueError(f"network.{name}_init_weights: {path!r} not found.")

    k_anchor = int(_pick(args.k_anchor, cfg, "dpdmd.k_anchor", 5))
    teacher_anchor_steps = int(
        _pick(args.teacher_anchor_steps, cfg, "dpdmd.teacher_anchor_steps", 28)
    )
    div_weight = float(_pick(args.div_weight, cfg, "dpdmd.div_weight", 5e-2))
    if args.detach_after_first is None:
        detach_after_first = bool(_flatten(cfg, "dpdmd.detach_after_first", True))
    else:
        detach_after_first = bool(args.detach_after_first)
    dm_first_step = bool(_pick(args.dm_first_step, cfg, "dpdmd.dm_first_step", False))
    rollout_grad = bool(_pick(args.rollout_grad, cfg, "dpdmd.rollout_grad", False))
    flow_shift = float(_pick(args.flow_shift, cfg, "sampling.flow_shift", 3.0))

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
    lr_schedule = str(_pick(args.lr_schedule, cfg, "optim.lr_schedule", "cosine"))
    if lr_schedule not in ("cosine", "constant"):
        raise ValueError(
            f"optim.lr_schedule={lr_schedule!r}: expected cosine or constant"
        )
    ema_window = float(_pick(args.ema_window, cfg, "ema.window", 0.0))
    if not 0.0 <= ema_window <= 1.0:
        raise ValueError(f"ema.window={ema_window}: must be in [0, 1]")

    # Sampling
    t_distribution = _flatten(cfg, "sampling.t_distribution", "uniform")
    sigmoid_scale = float(_flatten(cfg, "sampling.sigmoid_scale", 1.0))

    if student_steps < 2:
        raise ValueError(
            f"dmd.student_steps must be >= 2 (got {student_steps}): the rollout "
            "grids and the grad-step draw assume a refinement step after step 0."
        )
    if not (1 <= k_anchor < teacher_anchor_steps):
        raise ValueError(
            f"dpdmd.k_anchor={k_anchor} must satisfy 1 <= k_anchor < "
            f"teacher_anchor_steps={teacher_anchor_steps}."
        )
    if div_weight < 0.0:
        raise ValueError(f"dpdmd.div_weight={div_weight}: must be >= 0")
    if rollout_grad and (student_steps != 2 or dynamic_schedule or dm_first_step):
        raise ValueError(
            "dpdmd.rollout_grad needs dmd.student_steps = 2, dynamic_schedule off "
            f"and dm_first_step off (got N={student_steps}, dynamic_schedule="
            f"{dynamic_schedule}, dm_first_step={dm_first_step})."
        )
    if gan_loss_weight_gen < 0.0:
        raise ValueError(f"gan.weight_gen={gan_loss_weight_gen}: must be >= 0")
    int8 = str(_pick(args.int8, cfg, "int8", "off"))
    stream = str(_pick(args.stream, cfg, "stream", "bf16"))
    if mfd_dtau < 0.0:
        raise ValueError(f"mfd.dtau={mfd_dtau}: must be >= 0")
    if mfd_interval not in ("to_zero", "uniform"):
        raise ValueError(f"mfd.interval={mfd_interval!r}: to_zero | uniform")
    if mfd_dtau > 0.0:
        logger.info(
            f"MFD ON in the DM slot: dtau={mfd_dtau}, max_steps={mfd_max_steps}, "
            f"interval={mfd_interval}; L_CDM keeps the point delta."
        )
    if cdm_weight < 0.0:
        raise ValueError(f"cdm.weight={cdm_weight}: must be >= 0")
    # --grad_ckpt checkpoints the rollout's student forwards, which recompute at
    # the final backward under whatever view is current then. CDM and the GAN
    # flip the view to teacher / fake before that backward, so the recomputed
    # blocks drop the student LoRA and the gradient is silently wrong.
    if bool(args.grad_ckpt):
        for on, name in (
            (cdm_weight > 0.0, "cdm.weight > 0"),
            (gan_loss_weight_gen > 0.0, "gan.weight_gen > 0"),
        ):
            if on:
                raise ValueError(
                    f"--grad_ckpt with {name}: the rollout's checkpointed student "
                    "forwards recompute under the teacher/fake view this branch "
                    "leaves set, dropping the student LoRA from the gradient. "
                    "Turn one off."
                )
    if cdm_weight > 0.0:
        logger.info(
            f"L_CDM ON: weight={cdm_weight}; +1 student grad forward, "
            "+2 teacher and +1 fake no-grad forwards per step."
        )
    if gan_r1_weight < 0.0:
        raise ValueError(f"gan.r1_weight={gan_r1_weight}: must be >= 0")
    if gan_disc_head not in ("pooled", "token"):
        raise ValueError(
            f"gan.disc_head={gan_disc_head!r}: expected 'pooled' or 'token'"
        )
    if gan_delay_steps < 0 or gan_warmup_steps < 0:
        raise ValueError(
            f"gan.delay_steps={gan_delay_steps} / gan.warmup_steps="
            f"{gan_warmup_steps}: must be >= 0"
        )
    if gan_loss_weight_gen > 0.0:
        logger.info(
            f"GAN ON: "
            f"weight_gen={gan_loss_weight_gen}, feature_block_idx="
            f"{gan_feature_block_idx} (-1 = middle), disc_head={gan_disc_head}, "
            f"disc_lr={gan_disc_lr}, r1_weight={gan_r1_weight}, "
            f"use_same_t_noise={gan_use_same_t_noise}, "
            f"delay_steps={gan_delay_steps}, warmup_steps={gan_warmup_steps}."
        )
    if flow_shift <= 0.0:
        raise ValueError(f"sampling.flow_shift={flow_shift}: must be > 0")
    if not detach_after_first:
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
            f"fake_rank={fake_rank} < student_rank={student_rank}: the fake has "
            "less capacity than the student it tracks."
        )
    if norm_floor <= 0.0:
        raise ValueError(f"dmd.norm_floor={norm_floor}: must be > 0 (latent scale)")
    if fake_steps_per_student_step < 1:
        raise ValueError(
            f"optim.fake_steps_per_student_step={fake_steps_per_student_step}: must be ≥ 1"
        )
    if args.single_prompt_idx is not None and batch_size != 1:
        raise ValueError(
            f"--single_prompt_idx requires batch_size=1 (got {batch_size}): a "
            "one-sample dataset with drop_last yields zero batches."
        )
    logger.info(
        "DM gradient policy: "
        + (
            f"(b) x0-norm, norm_floor={norm_floor} — τ cancels, ≈ magnitude-normalized"
            if dm_x0_norm
            else "(a) τ-damping [default]"
        )
    )
    if div_weight == 0.0 and not dm_first_step:
        logger.warning(
            "dpdmd.div_weight=0 without dm_first_step: step 0 gets no loss and "
            "stays at its init."
        )
    logger.info(
        (
            "DP-DMD: first-step diversity anchor "
            f"k_anchor={k_anchor}/{teacher_anchor_steps} teacher steps, "
            f"div_weight={div_weight}, detach_after_first={detach_after_first}, "
            if div_weight > 0.0
            else "teacher anchor OFF (div_weight=0), "
        )
        + f"grad step g ~ U{{{0 if dm_first_step else 1}..N-1}}, "
        + ("rollout_grad (step 0 in the DM graph), " if rollout_grad else "")
        + f"student N={student_steps} @ flow_shift={flow_shift}, "
        f"teacher_cfg={teacher_cfg}."
    )
    return TurboConfig(
        dit_path=dit_path,
        data_dir=data_dir,
        output_dir=output_dir,
        output_name=output_name,
        log_dir=log_dir,
        resume=resume,
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
        channel_scaling_alpha=channel_scaling_alpha,
        student_down_init=student_down_init,
        fake_down_init=fake_down_init,
        student_init_weights=student_init_weights,
        fake_init_weights=fake_init_weights,
        train_adaln=train_adaln,
        fake_adaln=fake_adaln,
        adaln_rank=adaln_rank,
        adaln_alpha=adaln_alpha,
        fake_transition=fake_transition,
        k_anchor=k_anchor,
        teacher_anchor_steps=teacher_anchor_steps,
        div_weight=div_weight,
        detach_after_first=detach_after_first,
        dm_first_step=dm_first_step,
        rollout_grad=rollout_grad,
        flow_shift=flow_shift,
        student_steps=student_steps,
        teacher_cfg=teacher_cfg,
        dm_x0_norm=dm_x0_norm,
        norm_floor=norm_floor,
        dynamic_schedule=dynamic_schedule,
        cdm_weight=cdm_weight,
        cdm_grad_ckpt=cdm_grad_ckpt,
        mfd_dtau=mfd_dtau,
        mfd_max_steps=mfd_max_steps,
        mfd_interval=mfd_interval,
        gan_loss_weight_gen=gan_loss_weight_gen,
        gan_feature_block_idx=gan_feature_block_idx,
        gan_disc_lr=gan_disc_lr,
        gan_disc_hidden=gan_disc_hidden,
        gan_disc_head=gan_disc_head,
        gan_delay_steps=gan_delay_steps,
        gan_warmup_steps=gan_warmup_steps,
        gan_r1_weight=gan_r1_weight,
        gan_r1_alpha=gan_r1_alpha,
        gan_use_same_t_noise=gan_use_same_t_noise,
        gan_grad_ckpt=gan_grad_ckpt,
        student_lr=student_lr,
        fake_lr=fake_lr,
        fake_steps_per_student_step=fake_steps_per_student_step,
        fake_warmup_steps=fake_warmup_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        lr_schedule=lr_schedule,
        ema_window=ema_window,
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
        int8=int8,
        int8_file=_pick(args.int8_file, cfg, "int8_file", None),
        dit_nonquant_path=_pick(args.dit_nonquant_path, cfg, "dit_nonquant_path", None),
        stream=stream,
        partitioner_recompute_views=bool(
            _flatten(cfg, "partitioner_recompute_views", False)
        ),
        partitioner_aggressive_recomputation=bool(
            _flatten(cfg, "partitioner_aggressive_recomputation", False)
        ),
    )


def snapshot_toml_text(c: TurboConfig, *, source_config: str | None = None) -> str:
    """Render every resolved field (CLI overrides folded in) as a TOML snapshot.

    The turbo analogue of the ``<output_name>.snapshot.toml`` that ``train.py``
    writes for the LoRA family.
    """
    return dataclass_snapshot_toml(
        c,
        title="Anima turbo distillation — resolved config snapshot",
        source_config=source_config,
    )


# TensorBoard config summary: a hand-picked subset of the resolved fields.
_TB_KEYS = (
    "gan_loss_weight_gen",
    "cdm_weight",
    "mfd_dtau",
    "mfd_interval",
    "k_anchor",
    "teacher_anchor_steps",
    "div_weight",
    "detach_after_first",
    "dm_first_step",
    "rollout_grad",
    "flow_shift",
    "student_rank",
    "fake_rank",
    "channel_scaling_alpha",
    "student_steps",
    "teacher_cfg",
    "fake_warmup_steps",
    "student_lr",
    "fake_lr",
    "lr_schedule",
    "ema_window",
    "fake_steps_per_student_step",
    "iterations",
    "batch_size",
    "t_distribution",
    "data_dir",
    "dit_path",
)


def tb_config_text(c: TurboConfig) -> str:
    """Formatted TensorBoard config summary (``_TB_KEYS``)."""
    return dataclass_tb_text(c, include=_TB_KEYS)
