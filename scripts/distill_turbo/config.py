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
    load_toml,
)
from library.config.resolved import pick as _pick


logger = logging.getLogger(__name__)


def load_turbo_config(path: str) -> dict:
    return load_toml(path)


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
    parser.add_argument(
        "--dit_nonquant_path",
        type=str,
        default=None,
        help="The bf16 DiT an int8 --dit_path was quantized from: the float weights "
        "wq_t is built from. Sentinel None → TOML (dit_nonquant_path), else they are "
        "rebuilt from the int8 copies.",
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
        "_archive/proposals/dmd2_decoupled_improvements.md §2B.",
    )
    parser.add_argument(
        "--norm_floor",
        type=float,
        default=-1.0,
        help="clamp_min for the x0-norm denominator (latent scale); only active "
        "with --dm_x0_norm.",
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
    parser.add_argument(
        "--int8",
        choices=("off", "fwd", "full", "stream"),
        default=None,
        help="Run the frozen DiT's large linears as int8 GEMMs, teacher included "
        "(docs/anima/optimizations/int8.md). Sentinel None → TOML (int8, default off).",
    )
    parser.add_argument(
        "--int8_file",
        type=str,
        default=None,
        help="Forward copies from dit_gptq.py. Sentinel None → TOML (int8_file), "
        "else <dit>_int8_gptq.safetensors beside the DiT, else round to nearest; "
        "'rtn' forces round to nearest.",
    )
    parser.add_argument(
        "--stream",
        choices=("bf16", "fp32"),
        default=None,
        help="Residual stream precision, teacher included "
        "(docs/anima/optimizations/fp32_stream.md). Sentinel None → TOML (stream, "
        "default bf16).",
    )
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from a crash-resume bundle (student + fake + disc + all three "
        "optimizers/schedulers + RNG), restoring the LR schedule mid-flight. "
        "'auto' → output/ckpt/<output_name>/<output_name>_resume.pt if it exists, "
        "else start fresh — safe to bake into a restart wrapper. A named path must "
        "exist. Bundles are written every --save_every.",
    )
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

    # DP-DMD (arXiv 2602.03139): step 1 supervised toward a teacher K-step anchor
    # (diversity), detached, then DMD on x_θ over steps 2..N. See docs/anima/methods/turbo.md.
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

    # DMD2 teacher-feature GAN (off by default).
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
    parser.add_argument(
        "--gan_disc_head",
        type=str,
        default=None,
        choices=["pooled", "token"],
        help="Disc head granularity: 'pooled' = one logit per tap (mean-pooled "
        "tokens, v0); 'token' = LADD-style dense per-token logits (same MLP "
        "applied per token — identical params, denser real/fake signal). "
        "Default: TOML (gan.disc_head, default 'pooled').",
    )
    parser.add_argument(
        "--gan_delay_steps",
        type=int,
        default=-1,
        help="Hold the generator-side GAN weight at 0 for the first N student "
        "steps (the disc still trains from step 0). Gives DM+div an escape "
        "window from a collapsed warm start before dense realism pressure "
        "lands. Default: TOML (gan.delay_steps, default 0).",
    )
    parser.add_argument(
        "--gan_warmup_steps",
        type=int,
        default=-1,
        help="After the delay window, ramp the generator-side GAN weight "
        "linearly 0 → weight_gen over N student steps (0 = instant-on). "
        "Default: TOML (gan.warmup_steps, default 0).",
    )

    # CDM off-trajectory loss (L_CDM; off by default).
    parser.add_argument(
        "--cdm_grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint the grad-bearing CDM student forward. Default: TOML "
        "(cdm.grad_ckpt, default on).",
    )
    parser.add_argument(
        "--cdm_weight",
        type=float,
        default=-1.0,
        help="λ on the L_CDM off-trajectory loss (CDM §3.3, arXiv:2605.06376): "
        "Euler-extrapolate the DMD grad step's (x_g, v_g) by a random stride to "
        "t' ~ U(0,1), run one grad-bearing student forward there, and apply the "
        "same real-vs-fake DMD surrogate to its local x0 estimate (variant A: "
        "CFG'd real score, consistent with the fused DM). Supervises the "
        "truncation-drift region few-step Euler traverses off-manifold. 0 "
        "disables the whole path (byte-identical loop, no extra forwards/RNG). "
        "Cost when on: +1 student grad forward, +2 teacher & +1 fake no-grad "
        "per iteration. Default: TOML (cdm.weight, default 0).",
    )

    # Mean Flow Distillation in the DM slot (arXiv:2606.11155; off by default).
    parser.add_argument(
        "--mfd_dtau",
        type=float,
        default=None,
        help="Euler step size of the MFD mean-flow integral that replaces the DM "
        "branch's point delta: teacher and fake are integrated from σ_a to σ_b "
        "and x_θ follows the endpoint gap. 0 = plain DMD. Cost: K = "
        "ceil((σ_a − σ_b)/dtau) ≤ mfd_max_steps teacher-CFG + fake no-grad "
        "forwards per student step instead of 1. Default: TOML (mfd.dtau, 0).",
    )
    parser.add_argument(
        "--mfd_max_steps",
        type=int,
        default=None,
        help="Cap on K per mean-flow integral. Default: TOML (mfd.max_steps, 4).",
    )
    parser.add_argument(
        "--mfd_interval",
        type=str,
        default=None,
        choices=("to_zero", "uniform"),
        help="MFD interval end: to_zero = σ_b 0 (integrated x0 match; K=1 is "
        "exactly DMD), uniform = σ_b ~ U(0, σ_a). Default: TOML (mfd.interval, "
        "to_zero).",
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
    use_custom_down_autograd: bool
    # Per-input-channel rebalance on each lora_down. A no-op at init that merges
    # out at save, so it only conditions gradients.
    channel_scaling_alpha: float
    # SVD-Down init for the plain-LoRA student (down_init="weight_svd"): seed
    # lora_down from W0's top-r right singular vectors, scale-matched.
    student_down_init: str
    # Same lever on the fake/critic.
    fake_down_init: str
    # Warm start: initialize the stack's ΔW from a plain LoRA checkpoint
    # (e.g. an official-release delta extracted by
    # scripts/toolkits/extract_delta_lora.py), SVD-truncated to the stack's rank.
    # Empty = default init. Incompatible with down_init="weight_svd" (it seeds
    # what this overwrites).
    student_init_weights: str
    fake_init_weights: str
    # Target the AdaLN modulation up-projections (adaln_up_{branch}) on both the
    # student and fake. The student's shipped LoRA is saved in the ComfyUI adaln
    # layout so it loads natively there and in-repo (adaln.md).
    train_adaln: bool
    # Mirror adaln targets on the fake/critic (default: train_adaln). false =
    # VRAM lever — drops the critic's 84 adaln modules (~0.8 GB of fp32
    # params+grads+AdamW states at rank 96) at the cost of a score estimate
    # blind to the student's adaln subspace (unbenched trade).
    fake_adaln: bool
    # Build the adaln modules at their own rank (0 = share network rank). The
    # official turbo adaln ΔW is near-lossless well below attn rank (in-dim
    # 256: r64 keeps 99.5% of the r96 energy) — a lower adaln_rank cuts the
    # adaln VRAM add roughly in proportion. Requires train_adaln.
    adaln_rank: int
    # Their own alpha too (0 = network alpha, which runs rank/adaln_rank hotter
    # when adaln_rank is set). Scale-preserving value:
    # student_alpha × adaln_rank / student_rank. Requires train_adaln.
    adaln_alpha: float
    # Transition-aware fake (VA-TDM, vatdm.md idea 1): the fake also gets σ of the
    # student step its sample came from (σ_g; t' for the CDM query; the warmup's t),
    # as a zero-init delta on its t-embedding.
    fake_transition: bool
    # AOT min-cut partitioner tuning (mirrors train.py's partitioner_* args):
    # change what the default partition is willing to recompute, on top of
    # activation_memory_budget. Ignored under --grad_ckpt (same gate as the
    # budget). aggressive_recomputation is the settled memory lever.
    partitioner_recompute_views: bool
    partitioner_aggressive_recomputation: bool

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
    # CDM dynamic continuous schedule (arXiv:2605.06376): re-sample the student
    # rollout grid every iteration (N ~ U{..student_steps}, continuous anchors)
    # instead of training only on the fixed inference grid. Validation/inference
    # stay on the static grid; the DP anchor composes unchanged (t₁=1 pinned).
    dynamic_schedule: bool

    # CDM off-trajectory loss (L_CDM, arXiv:2605.06376): supervise
    # the student's local x0 estimate at a velocity-extrapolated off-trajectory
    # point (t' ~ U(0,1)) with the same real-vs-fake delta as the DM branch.
    # 0 = the whole path off.
    cdm_weight: float
    cdm_grad_ckpt: bool  # checkpoint ONLY the grad-bearing CDM student forward

    # Mean Flow Distillation (arXiv:2606.11155) in the DM slot; dtau 0 = plain DMD.
    mfd_dtau: float
    mfd_max_steps: int
    mfd_interval: str  # "to_zero" | "uniform"

    # DMD2 teacher-feature GAN
    gan_loss_weight_gen: float
    gan_feature_block_idx: int  # -1 → middle block (resolved in distill.py)
    gan_disc_lr: float
    gan_disc_hidden: int  # <= 0 → inner_dim // 2
    gan_disc_head: str  # "pooled" (per-tap logit) | "token" (LADD-style per-token)
    gan_delay_steps: int  # generator λ = 0 for the first N steps (disc still trains)
    gan_warmup_steps: int  # then λ ramps 0 → weight_gen over N steps (0 = instant-on)
    gan_r1_weight: float
    gan_r1_alpha: float
    gan_use_same_t_noise: bool
    gan_grad_ckpt: bool  # checkpoint ONLY the grad-bearing GAN gen forward
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
    # CLI-only: a resume is a property of *this launch*, not of the method config.
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
    if student_down_init not in ("kaiming", "weight_svd"):
        raise ValueError(
            f"network.student_down_init={student_down_init!r}: "
            "expected 'kaiming' or 'weight_svd'."
        )
    fake_down_init = str(_flatten(cfg, "network.fake_down_init", "kaiming"))
    if fake_down_init not in ("kaiming", "weight_svd"):
        raise ValueError(
            f"network.fake_down_init={fake_down_init!r}: "
            "expected 'kaiming' or 'weight_svd'."
        )
    attn_mode = _pick(args.attn_mode, cfg, "network.attn_mode", "flash")
    # use_custom_down_autograd is a top-level TOML scalar; CLI flag wins when set.
    if args.use_custom_down_autograd is None:
        use_custom_down_autograd = bool(
            _flatten(cfg, "use_custom_down_autograd", False)
        )
    else:
        use_custom_down_autograd = bool(args.use_custom_down_autograd)
    # Defaults off so existing turbo snapshots reproduce.
    channel_scaling_alpha = float(
        _pick(args.channel_scaling_alpha, cfg, "channel_scaling_alpha", 0.0)
    )

    # DMD core
    student_steps = int(_pick(args.student_steps, cfg, "dmd.student_steps", 4))
    teacher_cfg = float(_flatten(cfg, "dmd.teacher_cfg", 4.0))
    # DM-branch gradient policy: (a) τ-damping [default] vs (b) x0-norm. Alternative
    # policies, not additive; (b) ≈ "drop the τ-weight, magnitude-normalize."
    dm_x0_norm = bool(_pick(args.dm_x0_norm, cfg, "dmd.dm_x0_norm", False))
    norm_floor = float(_pick(args.norm_floor, cfg, "dmd.norm_floor", 0.05))
    # Default off so existing turbo snapshots reproduce.
    dynamic_schedule = bool(_flatten(cfg, "dmd.dynamic_schedule", False))

    # weight=0 keeps the whole L_CDM path off (no extra forwards or RNG draws).
    cdm_weight = float(_pick(args.cdm_weight, cfg, "cdm.weight", 0.0))
    cdm_grad_ckpt = bool(
        _flatten(cfg, "cdm.grad_ckpt", True)
        if args.cdm_grad_ckpt is None
        else args.cdm_grad_ckpt
    )

    mfd_dtau = float(_pick(args.mfd_dtau, cfg, "mfd.dtau", 0.0))
    mfd_max_steps = int(_pick(args.mfd_max_steps, cfg, "mfd.max_steps", 4))
    mfd_interval = str(_pick(args.mfd_interval, cfg, "mfd.interval", "to_zero"))

    # weight_gen=0 keeps the whole GAN/disc path off.
    # feature_block_idx sentinel is -2 (not -1) because -1 means middle block.
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
    # Both default 0 → the generator term engages at full weight from step 0.
    gan_delay_steps = int(_pick(args.gan_delay_steps, cfg, "gan.delay_steps", 0))
    gan_warmup_steps = int(_pick(args.gan_warmup_steps, cfg, "gan.warmup_steps", 0))
    gan_r1_weight = float(_pick(args.gan_r1_weight, cfg, "gan.r1_weight", 0.0))
    gan_r1_alpha = float(_flatten(cfg, "gan.r1_alpha", 0.1))
    gan_use_same_t_noise = bool(_flatten(cfg, "gan.use_same_t_noise", True))
    # Checkpoint only the grad-bearing GAN gen teacher forward (independent of the
    # global --grad_ckpt): it retains ~half the DiT's activations purely to
    # backprop into x_pred → student, so recompute reclaims that peak VRAM.
    # Default on — numerically equivalent (frozen teacher, no dropout).
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
    flow_shift = float(_pick(args.flow_shift, cfg, "sampling.flow_shift", 3.0))

    # Optimizer
    student_lr = float(_pick(args.student_lr, cfg, "optim.student_lr", 1e-5))
    fake_lr = float(_pick(args.fake_lr, cfg, "optim.fake_lr", 1e-5))
    # LR schedule is always warmup → cosine (see primitives.make_scheduler).
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

    if student_steps < 2:
        raise ValueError(
            f"DP-DMD requires dmd.student_steps >= 2 (got {student_steps}): step 1 "
            "is diversity-supervised + detached, so at least one further step must "
            "carry the DMD loss."
        )
    if not (1 <= k_anchor < teacher_anchor_steps):
        raise ValueError(
            f"dpdmd.k_anchor={k_anchor} must satisfy 1 <= k_anchor < "
            f"teacher_anchor_steps={teacher_anchor_steps}."
        )
    if div_weight < 0.0:
        raise ValueError(f"dpdmd.div_weight={div_weight}: must be >= 0")
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
    if cdm_weight > 0.0:
        if bool(args.grad_ckpt):
            # Same view × deferred-ckpt-recompute class as --grad_ckpt + GAN:
            # the CDM branch flips the view (student → teacher → fake) after the
            # rollout's checkpointed student forwards, corrupting their recompute.
            raise ValueError(
                "--grad_ckpt with cdm.weight > 0: the rollout's checkpointed "
                "student forwards recompute after the CDM branch flipped the "
                "view to teacher/fake — the recomputed blocks drop the student "
                "LoRA and the gradient is silently corrupted "
                "(project_turbo_view_ckpt_recompute_hazard). Turn one off."
            )
        logger.info(
            f"L_CDM off-trajectory loss ON: "
            f"weight={cdm_weight}, variant A (CFG'd real score), t' ~ U(0,1) "
            "launched from the DMD grad step; +1 student grad forward, "
            "+2 teacher & +1 fake no-grad per iteration."
        )
    if bool(args.grad_ckpt) and gan_loss_weight_gen > 0.0:
        # View × deferred-ckpt-recompute hazard: a forward that flips the global
        # view after the rollout's checkpointed forwards corrupts their recompute.
        raise ValueError(
            "--grad_ckpt with gan.weight_gen > 0: the rollout's checkpointed "
            "student forwards recompute after the GAN gen forward flipped the "
            "view to 'teacher' — the recomputed blocks drop the student LoRA "
            "and the DMD gradient is silently corrupted. Known-broken "
            "combination; turn one of the two off."
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
            f"GAN (DMD2 teacher-feature disc, FastGen idea 1) ON: "
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
    logger.info(
        "DM gradient policy: "
        + (
            f"(b) x0-norm, norm_floor={norm_floor} — τ cancels, ≈ magnitude-normalized"
            if dm_x0_norm
            else "(a) τ-damping [default]"
        )
    )
    logger.info(
        "DP-DMD: first-step diversity anchor "
        f"k_anchor={k_anchor}/{teacher_anchor_steps} teacher steps, "
        f"div_weight={div_weight}, detach_after_first={detach_after_first}, "
        f"student N={student_steps} @ flow_shift={flow_shift}, "
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
        use_custom_down_autograd=use_custom_down_autograd,
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
    "flow_shift",
    "student_rank",
    "fake_rank",
    "channel_scaling_alpha",
    "student_steps",
    "teacher_cfg",
    "fake_warmup_steps",
    "student_lr",
    "fake_lr",
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
