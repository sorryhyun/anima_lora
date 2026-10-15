"""Turbo distillation config: TOML loader + argparser + CLI/TOML resolver.

Every knob is one ``TurboConfig`` field declared with :func:`knob` (TOML key,
default, help). The argparser and the resolver are generated from those fields:
each knob gets a ``--<field>`` flag (bools also get ``--no_<field>``), and
resolution is CLI (when given) → TOML key → default. Cross-field validation and
the run-summary logging live in :func:`_validate`.

The resolved knobs are returned as a frozen ``TurboConfig`` so the training loop
never reaches back into ``args``/``cfg`` mid-step.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import types
import typing
from dataclasses import dataclass, field
from typing import Any

from library.config.io import toml_get
from library.config.resolved import (
    dataclass_snapshot_toml,
    dataclass_tb_text,
    load_toml,
)

logger = logging.getLogger(__name__)


def _default_mask_dir() -> str:
    """Shared ``mask_dir`` default (configs/preprocess.toml → base → preset).

    The bespoke DP-DMD loop bypasses ``train.py``'s merge chain, so read the
    same key here rather than re-hardcoding the path — otherwise moving the
    mask root silently leaves turbo pointed at the old one.
    """
    try:
        from library.config.io import load_path_overrides

        value = load_path_overrides().get("mask_dir")
    except Exception:  # noqa: BLE001 — config is optional; fall back to the default
        value = None
    return str(value) if value else "post_image_dataset/masks"


@dataclass(frozen=True)
class _Same:
    """Default = the value another knob (declared earlier) resolved to."""

    field: str


def knob(
    toml: str | None,
    default: Any,
    help: str | None = None,
    *,
    choices: tuple | None = None,
):
    """Declare a ``TurboConfig`` field as a config knob.

    ``toml`` is the dotted TOML key (``None`` = CLI-only). ``default`` may be a
    :class:`_Same` or a zero-arg callable, resolved at resolve time. ``choices``
    is enforced on the resolved value, whichever source it came from.
    """
    return field(
        metadata={"toml": toml, "default": default, "help": help, "choices": choices}
    )


@dataclass(frozen=True)
class TurboConfig:
    # Paths / IO
    dit_path: str = knob(
        "dit_path", "models/diffusion_models/anima-base-v1.0.safetensors"
    )
    data_dir: str = knob("data_dir", "post_image_dataset/lora")
    output_dir: str = knob("output_dir", "output/ckpt")
    output_name: str = knob("output_name", "anima_turbo")
    log_dir: str = knob("io.log_dir", "output/logs/turbo")
    save_every: int = knob("io.save_every", 1000)
    # CLI-only: a resume is a property of *this launch*, not of the method config.
    resume: str = knob(
        None,
        "",
        "Resume from a crash-resume bundle (student + fake + disc + all three "
        "optimizers/schedulers + RNG), restoring the LR schedule mid-flight. "
        "'auto' → output/ckpt/<output_name>/<output_name>_resume.pt if it exists, "
        "else start fresh — safe to bake into a restart wrapper. A named path must "
        "exist. Bundles are written every --save_every.",
    )
    log_interval: int = knob("io.log_interval", 2)
    no_log: bool = knob(None, False)

    # Diversity validation (DAVE same-prompt probe; 0 = off)
    validate_every_n_steps: int = knob(
        "io.validate_every_n_steps",
        0,
        "Run the DAVE same-prompt diversity probe every N optimizer steps "
        "(0 disables; see scripts/distill_turbo/diversity.py). Logs "
        "val/div_ac_sim (lower = more diverse), val/div_dc_sim, val/div_gap.",
    )
    val_diversity_seeds: int = knob(
        "io.val_diversity_seeds",
        8,
        "Number of seeds the diversity probe rolls per validation (>=2).",
    )
    val_prompt_idx: int = knob(
        "io.val_prompt_idx",
        -1,
        "Held-out dataset index whose cached conditioning the diversity probe "
        "fixes (-1 = auto: last sample, distinct from --single_prompt_idx).",
    )

    # Run shape
    iterations: int = knob("iterations", 20000)
    batch_size: int = knob("batch_size", 1)
    seed: int = knob("seed", 42)
    sample_ratio: float = knob(None, 1.0)
    single_prompt_idx: int | None = knob(
        None,
        None,
        "Phase 0 overfit mode — pin the dataloader to a single (latent, text) pair.",
    )

    # LoRA stacks
    student_rank: int = knob("network.student_rank", 48)
    fake_rank: int = knob("network.fake_rank", 48)
    student_alpha: float = knob("network.student_alpha", _Same("student_rank"))
    fake_alpha: float = knob("network.fake_alpha", _Same("fake_rank"))
    fake_tau_banks: int = knob(
        "network.fake_tau_banks",
        1,
        "τ-split critic: 2 = two fake LoRAs, one owning each τ band; both updates "
        "and DMD queries route by the drawn τ vs --fake_tau_boundary. Total critic "
        "compute is unchanged (the existing updates are partitioned, not added). "
        "Requires batch_size=1. 1 = byte-identical single-critic loop.",
    )
    fake_tau_boundary: float = knob(
        "network.fake_tau_boundary",
        0.5,
        "Raw-τ split point for --fake_tau_banks 2: bank 0 owns [0, b), bank 1 "
        "owns [b, 1]. Uniform t_distribution → even update split at 0.5.",
    )
    attn_mode: str = knob("network.attn_mode", "flash")
    channel_scaling_alpha: float = knob(
        "channel_scaling_alpha",
        0.0,
        "Per-input-channel rebalance absorbed into lora_down (student + fake). "
        "Bit-equivalent at init, merges out — a gradient-conditioning lever. "
        "0.0 = off, 0.5 = sqrt-balance.",
    )
    student_down_init: str = knob(
        "network.student_down_init",
        "kaiming",
        "'weight_svd' seeds the student's lora_down from W0's top-r right "
        "singular vectors, scale-matched.",
        choices=("kaiming", "weight_svd"),
    )
    fake_down_init: str = knob(
        "network.fake_down_init",
        "kaiming",
        "Same lever on the fake/critic.",
        choices=("kaiming", "weight_svd"),
    )
    student_init_weights: str = knob(
        "network.student_init_weights",
        "",
        "Warm start: initialize the stack's ΔW from a plain LoRA checkpoint (e.g. "
        "an official-release delta from scripts/toolkits/extract_delta_lora.py), "
        "SVD-truncated to the stack's rank. Empty = default init. Incompatible "
        "with down_init='weight_svd' (it seeds what this overwrites).",
    )
    fake_init_weights: str = knob(
        "network.fake_init_weights", "", "Warm start for the fake/critic stack."
    )
    train_adaln: bool = knob(
        "network.train_adaln",
        False,
        "Target the AdaLN modulation up-projections (adaln_up_{branch}) on both "
        "the student and fake. The student's shipped LoRA is saved in the ComfyUI "
        "adaln layout so it loads natively there and in-repo (adaln.md).",
    )
    fake_adaln: bool = knob(
        "network.fake_adaln",
        _Same("train_adaln"),
        "Mirror adaln targets on the fake/critic. false = VRAM lever — drops the "
        "critic's 84 adaln modules (~0.8 GB of fp32 params+grads+AdamW states at "
        "rank 96) at the cost of a score estimate blind to the student's adaln "
        "subspace (unbenched trade).",
    )
    adaln_rank: int = knob(
        "network.adaln_rank",
        0,
        "Build the adaln modules at their own rank (0 = share network rank). The "
        "official turbo adaln ΔW is near-lossless well below attn rank (in-dim "
        "256: r64 keeps 99.5%% of the r96 energy). Requires train_adaln.",
    )
    adaln_alpha: float = knob(
        "network.adaln_alpha",
        0.0,
        "Their own alpha too (0 = network alpha, which runs rank/adaln_rank "
        "hotter when adaln_rank is set). Scale-preserving value: student_alpha × "
        "adaln_rank / student_rank. Requires train_adaln.",
    )
    # AOT min-cut partitioner tuning (mirrors train.py's partitioner_* args):
    # change what the default partition is willing to recompute, on top of
    # activation_memory_budget. Ignored under --grad_ckpt (same gate as the
    # budget). aggressive_recomputation is the settled memory lever.
    partitioner_recompute_views: bool = knob("partitioner_recompute_views", False)
    partitioner_aggressive_recomputation: bool = knob(
        "partitioner_aggressive_recomputation", False
    )

    # Masked loss
    use_masked_loss: bool = knob(
        "use_masked_loss",
        False,
        "Apply the per-image foreground mask to the student DMD2 gradient "
        "(masked-out latents get zero student push). Fake/critic loss is "
        "unaffected.",
    )
    mask_dir: str = knob(
        "mask_dir",
        _default_mask_dir,
        "Mask root for --use_masked_loss (default: `mask_dir` from "
        "configs/preprocess.toml). Mirrors data_dir's subdir layout.",
    )

    # DP-DMD (arXiv 2602.03139): step 1 supervised toward a teacher K-step anchor
    # (diversity), detached, then DMD on x_θ over steps 2..N. See docs/methods/turbo.md.
    k_anchor: int = knob(
        "dpdmd.k_anchor", 5, "Teacher steps rolled to the diversity anchor (their K)."
    )
    teacher_anchor_steps: int = knob(
        "dpdmd.teacher_anchor_steps",
        28,
        "Teacher σ-grid the K anchor is counted against.",
    )
    div_weight: float = knob(
        "dpdmd.div_weight", 5e-2, "λ on the first-step diversity loss."
    )
    detach_after_first: bool = knob(
        "dpdmd.detach_after_first",
        True,
        "Stop-grad after the diversity-supervised first step (the load-bearing "
        "detach; keep True except for A/B).",
    )
    flow_shift: float = knob(
        "sampling.flow_shift",
        3.0,
        "σ-schedule shift for the student/teacher Euler grids (matches inference).",
    )

    # DMD core
    student_steps: int = knob(
        "dmd.student_steps", 4, "Sampler step count baked into the student."
    )
    teacher_cfg: float = knob("dmd.teacher_cfg", 4.0)
    dm_x0_norm: bool = knob(
        "dmd.dm_x0_norm",
        False,
        "DMD per-sample x0-space magnitude normalization (policy 'b'): grad_dm = "
        "τ·Δ_dm / clamp(τ·mean|v_real|, norm_floor). Because the denom ≈ "
        "τ·mean|v_real|, the τ CANCELS across the bulk → ≈ no-τ, magnitude-"
        "normalized. This REPLACES the default τ-damping (policy 'a'); it does "
        "NOT stack with it.",
    )
    norm_floor: float = knob(
        "dmd.norm_floor",
        0.05,
        "clamp_min for the x0-norm denominator (latent scale); only active with "
        "--dm_x0_norm.",
    )
    dmd_grad_step: str = knob(
        "dmd.grad_step",
        "all",
        "Which rollout step(s) carry gradient; the rest are backward-simulated "
        "under no_grad (DMD2's train/inference input-match, Yin et al. 2024). "
        "'all' = full-rollout BPTT (holds N forward graphs). 'last' = only the "
        "final, cleanest-σ step (memory-flat, but the noisy steps are never "
        "directly supervised). 'random' = canonical DMD2 multistep: sample g, "
        "grad ONLY step g, supervise its one-step x0-prediction — memory-flat AND "
        "spreads supervision over every grid point.",
        choices=("all", "last", "random"),
    )
    dynamic_schedule: bool = knob(
        "dmd.dynamic_schedule",
        False,
        "CDM dynamic continuous schedule (arXiv:2605.06376): re-sample the "
        "student rollout grid every iteration (N ~ U{..student_steps}, continuous "
        "anchors) instead of training only on the fixed inference grid. "
        "Validation/inference stay on the static grid.",
    )
    cdm_weight: float = knob(
        "cdm.weight",
        0.0,
        "λ on the L_CDM off-trajectory loss (CDM §3.3, arXiv:2605.06376): "
        "Euler-extrapolate the DMD grad step's (x_g, v_g) by a random stride to "
        "t' ~ U(0,1), run one grad-bearing student forward there, and apply the "
        "same real-vs-fake DMD surrogate to its local x0 estimate (variant A: "
        "CFG'd real score). 0 disables the whole path (byte-identical loop, no "
        "extra forwards/RNG). Cost when on: +1 student grad forward, +2 teacher "
        "& +1 fake no-grad per iteration.",
    )
    base_loss: str = knob(
        "base_loss",
        "dpdmd",
        "Objective: 'dpdmd' (first-step teacher anchor) or 'dmd' (plain DMD2 — "
        "no anchor, allows student_steps=1).",
        choices=("dpdmd", "dmd"),
    )

    # DMD2 teacher-feature GAN (off by default) + f-distill reweighting
    gan_loss_weight_gen: float = knob(
        "gan.weight_gen",
        0.0,
        "λ on the GAN generator term (softplus hinge on teacher-feature disc "
        "logits), added to the student loss. 0 disables the whole GAN path "
        "(byte-identical to DP-DMD). FastGen QwenImage uses 0.03.",
    )
    gan_feature_block_idx: int = knob(
        "gan.feature_block_idx",
        -1,
        "Which DiT block's token output the discriminator taps. -1 = middle "
        "block (num_blocks//2).",
    )
    gan_disc_lr: float = knob("gan.disc_lr", 1e-5, "Discriminator AdamW LR.")
    gan_disc_hidden: int = knob(
        "gan.disc_hidden", 0, "Disc MLP hidden width (<= 0 → inner_dim // 2)."
    )
    gan_disc_head: str = knob(
        "gan.disc_head",
        "pooled",
        "Disc head granularity: 'pooled' = one logit per tap (mean-pooled "
        "tokens); 'token' = LADD-style dense per-token logits (same MLP applied "
        "per token — identical params, denser real/fake signal).",
        choices=("pooled", "token"),
    )
    gan_delay_steps: int = knob(
        "gan.delay_steps",
        0,
        "Hold the generator-side GAN weight at 0 for the first N student steps "
        "(the disc still trains from step 0). Gives DM+div an escape window from "
        "a collapsed warm start before dense realism pressure lands.",
    )
    gan_warmup_steps: int = knob(
        "gan.warmup_steps",
        0,
        "After the delay window, ramp the generator-side GAN weight linearly "
        "0 → weight_gen over N student steps (0 = instant-on).",
    )
    gan_r1_weight: float = knob(
        "gan.r1_weight",
        0.0,
        "Weight on the approximate-R1 (APT) disc regularizer: MSE between real "
        "logits and logits of a slightly-perturbed real input. 0 disables.",
    )
    gan_r1_alpha: float = knob("gan.r1_alpha", 0.1)
    gan_use_same_t_noise: bool = knob("gan.use_same_t_noise", True)
    gan_grad_ckpt: bool = knob(
        "gan.grad_ckpt",
        True,
        "Checkpoint only the grad-bearing GAN gen teacher forward (independent "
        "of --grad_ckpt): it retains ~half the DiT's activations purely to "
        "backprop into x_pred → student. Numerically equivalent (frozen teacher, "
        "no dropout).",
    )
    f_div: str = knob(
        "f_distill.f_div",
        "rkl",
        "f-divergence whose weight h=f'(r) reweights the DMD signal "
        "(r=exp(disc_logits)). 'rkl' ≡ uniform h ≡ plain DMD2 (no-op). Any other "
        "value REQUIRES gan.weight_gen > 0.",
        choices=("rkl", "kl", "js", "sf", "neyman", "sh", "jf"),
    )
    f_ratio_lower: float = knob("f_distill.ratio_lower", 0.1)
    f_ratio_upper: float = knob("f_distill.ratio_upper", 20.0)
    f_ratio_ema_rate: float = knob("f_distill.ratio_ema_rate", 0.0)
    f_bin_num: int = knob("f_distill.bin_num", 10)
    f_ratio_normalization: bool = knob("f_distill.ratio_normalization", True)

    # Soft-rank caption-discrimination auxiliary (off by default)
    softrank_weight: float = knob(
        "softrank.weight",
        0.0,
        "λ on the step-0 soft-rank caption loss (pushes the matched caption to "
        "explain the diversity anchor better than k mismatched ones). 0 disables "
        "the whole path (byte-identical DP-DMD). First live value: 0.05.",
    )
    softrank_k: int = knob(
        "softrank.k",
        2,
        "Number of shuffled-caption negatives per firing (k extra no_grad student "
        "forwards). Must be >= 2.",
    )
    softrank_every_n: int = knob(
        "softrank.every_n",
        4,
        "Fire the soft-rank term every N student steps (amortizes the k extra "
        "forwards).",
    )
    softrank_softness: float = knob(
        "softrank.softness",
        0.1,
        "Temperature τ of the soft-rank relaxation (smaller = closer to the hard "
        "integer rank).",
    )
    softrank_pool_size: int = knob(
        "softrank.pool_size",
        64,
        "Capacity of the cross-step caption pool the negatives are drawn from "
        "(lets the term fire at batch_size=1). Must be >= k. Each caption is ~1 "
        "MiB (bf16 [512,1024]), so the pool costs pool_size MiB of VRAM.",
    )
    softrank_warmup_ratio: float = knob(
        "softrank.warmup_ratio",
        1.0,
        "Fraction of the caption pool that must fill before the term fires (so "
        "negatives are drawn from a representative shuffle, not the last few "
        "captions). 1.0 = wait for a full pool; 0 = fire as soon as k are cached.",
    )

    # Optimizer + scheduler (LR schedule is always warmup → cosine, see
    # primitives.make_scheduler)
    student_lr: float = knob("optim.student_lr", 1e-5)
    fake_lr: float = knob("optim.fake_lr", 1e-5)
    fake_steps_per_student_step: int = knob(
        "optim.fake_steps_per_student_step",
        1,
        "Number of fake (DM regularizer) updates per student step. Standard DMD2 "
        "practice keeps the fake ahead of the moving x_pred distribution; >1 "
        "gives the fake extra SGD iterations on resampled (τ, ε) noise against "
        "the same x_pred.detach().",
    )
    fake_warmup_steps: int = knob(
        "optim.fake_warmup_steps",
        0,
        "Fake-only (critic head-start) updates run BEFORE the main loop, so the "
        "zero-init fake/critic is calibrated against the student's x_pred "
        "distribution before the student takes full-strength steps (else a "
        "large, misaligned delta_dm and an early grad_signal_rms spike). The fake "
        "scheduler is sized over iterations + fake_warmup_steps, so its 2%% LR "
        "warmup overlaps the head-start. 0 = off.",
    )
    weight_decay: float = knob("optim.weight_decay", 0.0)
    grad_clip: float = knob("optim.grad_clip", 1.0)

    # Sampling distribution
    t_distribution: str = knob(
        "sampling.t_distribution", "uniform", choices=("uniform", "sigmoid")
    )
    sigmoid_scale: float = knob("sampling.sigmoid_scale", 1.0)

    # Runtime
    blocks_to_swap: int = knob("blocks_to_swap", 0)
    grad_ckpt: bool = knob("grad_ckpt", False)
    torch_compile: bool = knob("torch_compile", True, "Compile block._forward.")
    compile_dynamic_seq: bool = knob(
        "compile_dynamic_seq",
        True,
        "Mirror the LoRA-training compile_dynamic_seq path: collapse the "
        "per-token-count block graphs to a single graph by marking only the "
        "seq-length axis dynamic (mark_dynamic). Only matters with --torch_compile.",
    )
    target_res: list[int] | None = knob(
        "target_res",
        None,
        "Override the active multi-scale tier edges (e.g. 1024 768 1280) used to "
        "size the compile_dynamic_seq seq bound + dynamo cache budget. Unset → "
        "derived from the token-count families present in the cached pool.",
    )
    dynamo_recompile_limit: int = knob(
        "dynamo_recompile_limit", 64, "Per-_forward dynamo graph budget."
    )
    activation_memory_budget: float = knob(
        "activation_memory_budget",
        1.0,
        "torch.compile partitioner saved-activation fraction (<1.0 → recompute "
        "cheap intermediates in backward, mirrors the LoRA-training knob in "
        "base.toml). Only applies with --torch_compile and grad_ckpt off (the two "
        "repartition the same graph and conflict). 1.0 = off.",
    )

    # DMAD Phase 0 (scripts/distill_turbo/dmad.py; docs/proposal/turbo_dmad.md):
    # a discriminator replaces the DM term and the critic.
    dmad: bool = knob(
        "dmad.enabled",
        False,
        "Replace the DM term and the fake critic with DMAD: a balanced-BCE disc "
        "(cold LoRA stack on the teacher + head T: teacher samples vs student, "
        "head R: real latents vs student), whose input gradient, RMS-normalized "
        "per sample, is the student's grad_signal (and L_CDM's when cdm.weight > "
        "0). Needs gan.weight_gen = 0.",
    )
    dmad_lambda_t: float = knob(
        "dmad.lambda_t",
        150.0,
        "Head T weight in the student signal; 0 drops head T and its disc branch.",
    )
    dmad_lambda_r: float = knob(
        "dmad.lambda_r",
        150.0,
        "Head R weight in the student signal; 0 drops head R and its real-data branch.",
    )
    dmad_signal_rms: float = knob(
        "dmad.signal_rms",
        0.18,
        "Per-sample RMS the disc gradient is normalized to before it is used as "
        "grad_signal; 0 = off (raw lambda-weighted gradient, as the paper).",
    )
    dmad_lr: float = knob("dmad.lr", 4e-5, "Disc LR (LoRA stack + heads, constant).")
    dmad_grad_clip: float = knob(
        "dmad.grad_clip", 0.0, "Disc grad-norm clip; 0 = unclipped."
    )
    dmad_window: int = knob(
        "dmad.window",
        4,
        "Replay pairs per disc update (newest first, a fresh (τ, ε) each).",
    )
    dmad_disc_warmup_steps: int = knob(
        "dmad.disc_warmup_steps",
        0,
        "Disc-only updates before the main loop (student untouched).",
    )
    dmad_gap_routing: bool = knob(
        "dmad.gap_routing",
        False,
        "Gap routing (official DMAD): head T's weight per critic-σ decile band is "
        "sigmoid((median − gap_b)/gap_tau) / E_w, with gap_b the bias-corrected "
        "EMA (0.99) of h_R(real) − h_R(teacher). Engages once 5 bands have >= 10 "
        "updates; weight 1 before that.",
    )
    dmad_gap_tau: float = knob("dmad.gap_tau", 2.0, "Gap-routing temperature.")
    dmad_feature_block_idx: int = knob(
        "dmad.feature_block_idx",
        -1,
        "Block whose tokens the heads read; -1 = middle block.",
    )


_HINTS = typing.get_type_hints(TurboConfig)


def _knob_type(name: str) -> tuple[type, bool]:
    """``(scalar type, is_list)`` of a field: ``int | None`` → ``(int, False)``,
    ``list[int] | None`` → ``(int, True)``."""
    hint = _HINTS[name]
    if isinstance(hint, types.UnionType):
        hint = next(a for a in typing.get_args(hint) if a is not type(None))
    if typing.get_origin(hint) is list:
        return typing.get_args(hint)[0], True
    return hint, False


def _default_text(default: Any) -> str:
    if isinstance(default, _Same):
        return f"= {default.field}"
    if callable(default):
        return "see above"
    return repr(default)


def load_turbo_config(path: str) -> dict:
    return load_toml(path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Turbo Anima — DP-DMD distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/methods/turbo.toml",
        help="Path to the turbo TOML config (CLI flags override TOML values).",
    )
    # Every flag defaults to None = "not given" (the TOML key / default applies).
    for f in dataclasses.fields(TurboConfig):
        meta = f.metadata
        src = f"TOML {meta['toml']}, " if meta["toml"] else ""
        tail = f"[{src}default {_default_text(meta['default'])}]"
        help_text = f"{meta['help']} {tail}" if meta["help"] else tail
        kind, is_list = _knob_type(f.name)
        if kind is bool:
            parser.add_argument(
                f"--{f.name}",
                action="store_const",
                const=True,
                default=None,
                help=help_text,
            )
            if not f.name.startswith("no_"):
                parser.add_argument(f"--no_{f.name}", dest=f.name, action="store_false")
        else:
            parser.add_argument(
                f"--{f.name}",
                type=kind,
                nargs="+" if is_list else None,
                choices=meta["choices"],
                default=None,
                help=help_text,
            )
    return parser


def resolve_config(args: argparse.Namespace, cfg: dict) -> TurboConfig:
    """Apply CLI → TOML → default precedence, then run the cross-field checks."""
    resolved: dict[str, Any] = {}
    for f in dataclasses.fields(TurboConfig):
        meta = f.metadata
        value = getattr(args, f.name, None)
        if value is None and meta["toml"] is not None:
            value = toml_get(cfg, meta["toml"], None)
        if value is None:
            default = meta["default"]
            if isinstance(default, _Same):
                value = resolved[default.field]
            elif callable(default):
                value = default()
            else:
                value = default
        if value is not None:
            kind, is_list = _knob_type(f.name)
            value = [kind(v) for v in value] if is_list else kind(value)
        if meta["choices"] is not None and value not in meta["choices"]:
            raise ValueError(
                f"{meta['toml'] or f.name}={value!r}: expected one of {meta['choices']}"
            )
        resolved[f.name] = value
    c = TurboConfig(**resolved)
    _validate(c)
    return c


def _validate(c: TurboConfig) -> None:
    """Cross-field sanity checks + the run-summary log lines, in config order."""
    use_anchor = c.base_loss == "dpdmd"

    # τ-split critic
    if c.fake_tau_banks not in (1, 2):
        raise ValueError(
            f"network.fake_tau_banks={c.fake_tau_banks}: expected 1 (single critic) "
            "or 2 (τ-split)."
        )
    if not (0.0 < c.fake_tau_boundary <= 1.0):
        raise ValueError(
            f"network.fake_tau_boundary={c.fake_tau_boundary}: must be in (0, 1]."
        )
    if c.fake_tau_banks == 2 and c.batch_size != 1:
        # τ is per-sample; B=1 makes the routing decision a scalar. B>1 would
        # need a split-batch double forward (not implemented).
        raise ValueError(
            f"network.fake_tau_banks=2 requires batch_size=1 (got {c.batch_size}): "
            "the bank routing is by the batch's scalar τ."
        )
    if c.fake_tau_banks == 2:
        logger.info(
            f"τ-split critic ON: 2 fake banks, boundary={c.fake_tau_boundary} "
            "(bank 0 = low τ). Updates AND DMD queries route by drawn τ; total "
            "critic compute unchanged."
        )

    # LoRA stacks
    if c.adaln_rank > 0 and not c.train_adaln:
        raise SystemExit("network.adaln_rank > 0 requires network.train_adaln = true")
    if c.adaln_alpha > 0 and not c.train_adaln:
        raise SystemExit("network.adaln_alpha > 0 requires network.train_adaln = true")
    for name, path, down_init in (
        ("student", c.student_init_weights, c.student_down_init),
        ("fake", c.fake_init_weights, c.fake_down_init),
    ):
        if path and down_init != "kaiming":
            raise ValueError(
                f"network.{name}_init_weights is incompatible with "
                f"{name}_down_init='weight_svd' — the warm start overwrites "
                "what it seeds."
            )
        if path and not os.path.exists(path):
            raise ValueError(f"network.{name}_init_weights: {path!r} not found.")

    # Base objective + rollout gradient routing
    if use_anchor and c.student_steps < 2:
        raise ValueError(
            f"DP-DMD requires dmd.student_steps >= 2 (got {c.student_steps}): step 1 "
            "is diversity-supervised + detached, so at least one further step must "
            "carry the DMD loss. (Use base_loss='dmd' for a 1-step student.)"
        )
    if not use_anchor and c.student_steps < 1:
        raise ValueError(
            f"base_loss='dmd' requires dmd.student_steps >= 1 (got {c.student_steps})."
        )
    if use_anchor and not (1 <= c.k_anchor < c.teacher_anchor_steps):
        raise ValueError(
            f"dpdmd.k_anchor={c.k_anchor} must satisfy 1 <= k_anchor < "
            f"teacher_anchor_steps={c.teacher_anchor_steps}."
        )
    if c.div_weight < 0.0:
        raise ValueError(f"dpdmd.div_weight={c.div_weight}: must be >= 0")
    if (
        not use_anchor
        and c.student_steps > 1
        and not c.grad_ckpt
        and c.dmd_grad_step == "all"
    ):
        logger.warning(
            "base_loss='dmd' with student_steps=%d, grad_ckpt OFF, "
            "dmd.grad_step='all': plain DMD2 has no first-step anchor to detach, "
            "so the student backward holds the FULL %d-step rollout graph (≈%dx the "
            "activation memory of dpdmd@%d). Use student_steps=1 (the replacement "
            "arm), dmd.grad_step='random'/'last' (memory-flat), or --grad_ckpt.",
            c.student_steps,
            c.student_steps,
            c.student_steps,
            c.student_steps,
        )
    if c.dmd_grad_step == "last":
        logger.info(
            "dmd.grad_step='last': rollout steps 0..N-2 run no_grad; only the final "
            "step backprops to x_pred (memory-flat at any student_steps)."
        )
    elif c.dmd_grad_step == "random" and use_anchor:
        logger.info(
            "dmd.grad_step='random' under base_loss='dpdmd': step 0 keeps the "
            "diversity anchor (detached); each iteration then samples a refinement "
            "step g~U{1..N-1}, backward-simulates the 1..g-1 prefix under no_grad, "
            "and grads only step g's one-step x0-prediction (memory-flat; supervises "
            "every refinement grid point, vs 'last' which only ever grads the clean "
            "tail)."
        )
    elif c.dmd_grad_step == "random":
        logger.info(
            "dmd.grad_step='random': canonical DMD2 multistep — each iteration "
            "samples g~U{0..N-1}, backward-simulates to g under no_grad, and grads "
            "only step g's one-step x0-prediction (memory-flat; supervises every "
            "grid point, not just the clean tail)."
        )
    if c.gan_loss_weight_gen < 0.0:
        raise ValueError(f"gan.weight_gen={c.gan_loss_weight_gen}: must be >= 0")

    # Soft-rank caption auxiliary
    if c.softrank_weight < 0.0:
        raise ValueError(f"softrank.weight={c.softrank_weight}: must be >= 0")
    if c.softrank_weight > 0.0:
        if c.base_loss != "dpdmd":
            # The term sites on the DP-DMD step-0 diversity anchor (v_target); plain
            # DMD2 has no anchor, so there's nothing to rank against.
            raise ValueError(
                f"softrank.weight > 0 requires base_loss='dpdmd' (got "
                f"{c.base_loss!r}) — the soft-rank term rides the step-0 diversity "
                "anchor."
            )
        if c.softrank_k < 2:
            # softrank needs >= 2 candidates for a non-degenerate rank.
            raise ValueError(f"softrank.k={c.softrank_k}: must be >= 2")
        if c.softrank_every_n < 1:
            raise ValueError(f"softrank.every_n={c.softrank_every_n}: must be >= 1")
        if c.softrank_pool_size < c.softrank_k:
            # The pool must hold at least k captions or it never reaches `ready`.
            raise ValueError(
                f"softrank.pool_size={c.softrank_pool_size}: must be >= "
                f"softrank.k={c.softrank_k}."
            )
        if not (0.0 <= c.softrank_warmup_ratio <= 1.0):
            raise ValueError(
                f"softrank.warmup_ratio={c.softrank_warmup_ratio}: must be in [0, 1]."
            )
        if c.blocks_to_swap > 0:
            # Block swap desyncs on the k extra student forwards; turbo keeps the
            # DiT resident by default. Fail at config time.
            raise ValueError(
                "softrank.weight > 0 requires blocks_to_swap=0 — the extra "
                "caption-negative forwards are unaudited under block swap."
            )
        logger.info(
            "soft-rank caption auxiliary ON (turbo_caption_ranking.md Phase 1): "
            f"weight={c.softrank_weight}, k={c.softrank_k}, "
            f"every_n={c.softrank_every_n}, softness={c.softrank_softness}, "
            f"pool_size={c.softrank_pool_size} (~{c.softrank_pool_size} MiB), "
            f"warmup_ratio={c.softrank_warmup_ratio}."
        )

    # DMAD
    if c.dmad:
        if c.base_loss != "dpdmd":
            raise ValueError(
                "dmad needs base_loss='dpdmd': its teacher samples finish the "
                "step-0 anchor rollout."
            )
        if c.f_div != "rkl":
            raise ValueError(
                f"dmad refuses f_distill (f_div={c.f_div!r}): it reweights the DM "
                "score difference, which dmad removes."
            )
        if c.gan_loss_weight_gen > 0.0:
            raise ValueError(
                "dmad requires gan.weight_gen = 0 (head R replaces the [gan] "
                "real-vs-student disc in this phase)."
            )
        if c.fake_tau_banks > 1:
            raise ValueError(
                f"dmad requires network.fake_tau_banks = 1 (got {c.fake_tau_banks}): "
                "no fake critic is built."
            )
        if c.blocks_to_swap > 0:
            raise ValueError(
                "dmad requires blocks_to_swap=0 — the disc forwards are unaudited "
                "under block swap."
            )
        if c.grad_ckpt:
            # Same view × deferred-ckpt-recompute class as --grad_ckpt + GAN:
            # the student-signal disc forward flips the view to teacher before
            # the student backward recomputes the checkpointed rollout.
            raise ValueError(
                "--grad_ckpt with dmad: the rollout's checkpointed student "
                "forwards would recompute under the disc's teacher view. Turn "
                "one off."
            )
        if c.resume:
            raise ValueError(
                "dmad refuses --resume: the disc is not in the resume bundle in "
                "this phase."
            )
        if c.dmad_lambda_t < 0.0 or c.dmad_lambda_r < 0.0:
            raise ValueError(
                f"dmad.lambda_t={c.dmad_lambda_t} / dmad.lambda_r={c.dmad_lambda_r}: "
                "must be >= 0"
            )
        if c.dmad_lambda_t == 0.0 and c.dmad_lambda_r == 0.0:
            raise ValueError("dmad: lambda_t and lambda_r are both 0 — no head.")
        if c.dmad_signal_rms < 0.0:
            raise ValueError(f"dmad.signal_rms={c.dmad_signal_rms}: must be >= 0")
        if c.dmad_grad_clip < 0.0:
            raise ValueError(f"dmad.grad_clip={c.dmad_grad_clip}: must be >= 0")
        if c.dmad_window < 1:
            raise ValueError(f"dmad.window={c.dmad_window}: must be >= 1")
        if c.dmad_disc_warmup_steps < 0:
            raise ValueError(
                f"dmad.disc_warmup_steps={c.dmad_disc_warmup_steps}: must be >= 0"
            )
        if c.dmad_gap_routing and (c.dmad_lambda_t == 0.0 or c.dmad_lambda_r == 0.0):
            raise ValueError(
                "dmad.gap_routing needs both heads: the gap is h_R(real) − "
                "h_R(teacher) and it weights head T."
            )
        if c.dmad_gap_tau <= 0.0:
            raise ValueError(f"dmad.gap_tau={c.dmad_gap_tau}: must be > 0")
        logger.info(
            "DMAD ON (Phase 0): disc replaces the DM term and the fake critic — "
            f"lambda_t={c.dmad_lambda_t}, lambda_r={c.dmad_lambda_r}, "
            f"signal_rms={c.dmad_signal_rms}, lr={c.dmad_lr}, "
            f"grad_clip={c.dmad_grad_clip}, window={c.dmad_window}, "
            f"disc_warmup_steps={c.dmad_disc_warmup_steps}, "
            f"gap_routing={c.dmad_gap_routing} (tau {c.dmad_gap_tau}), "
            f"feature_block_idx={c.dmad_feature_block_idx} (-1 = middle). "
            "fake_lr / fake_steps_per_student_step / fake_warmup_steps are unused."
        )

    # L_CDM off-trajectory loss
    if c.cdm_weight < 0.0:
        raise ValueError(f"cdm.weight={c.cdm_weight}: must be >= 0")
    if c.cdm_weight > 0.0:
        if c.grad_ckpt:
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
        if c.blocks_to_swap > 0:
            # Block swap desyncs on extra per-step forwards; turbo keeps the
            # DiT resident by default. Fail at config time.
            raise ValueError(
                "cdm.weight > 0 requires blocks_to_swap=0 — the off-trajectory "
                "forwards are unaudited under block swap."
            )
        logger.info(
            f"L_CDM off-trajectory loss ON: "
            f"weight={c.cdm_weight}, variant A (CFG'd real score), t' ~ U(0,1) "
            "launched from the DMD grad step; +1 student grad forward, "
            "+2 teacher & +1 fake no-grad per iteration."
        )

    # GAN + f-distill
    if c.grad_ckpt and c.gan_loss_weight_gen > 0.0:
        # View × deferred-ckpt-recompute hazard: a forward that flips the global
        # view after the rollout's checkpointed forwards corrupts their recompute.
        raise ValueError(
            "--grad_ckpt with gan.weight_gen > 0: the rollout's checkpointed "
            "student forwards recompute after the GAN gen forward flipped the "
            "view to 'teacher' — the recomputed blocks drop the student LoRA "
            "and the DMD gradient is silently corrupted. Known-broken "
            "combination; turn one of the two off."
        )
    if c.gan_r1_weight < 0.0:
        raise ValueError(f"gan.r1_weight={c.gan_r1_weight}: must be >= 0")
    if c.gan_delay_steps < 0 or c.gan_warmup_steps < 0:
        raise ValueError(
            f"gan.delay_steps={c.gan_delay_steps} / gan.warmup_steps="
            f"{c.gan_warmup_steps}: must be >= 0"
        )
    if c.f_div != "rkl" and c.gan_loss_weight_gen <= 0.0:
        # r = exp(disc_logits) only exists once the GAN disc is built.
        raise ValueError(
            f"f_distill.f_div={c.f_div!r} requires gan.weight_gen > 0 — the "
            "f-divergence weight reads the GAN discriminator's logits."
        )
    if not (0.0 < c.f_ratio_lower < c.f_ratio_upper):
        raise ValueError(
            f"f_distill: require 0 < ratio_lower ({c.f_ratio_lower}) < "
            f"ratio_upper ({c.f_ratio_upper})"
        )
    if not (0.0 <= c.f_ratio_ema_rate < 1.0):
        raise ValueError(
            f"f_distill.ratio_ema_rate={c.f_ratio_ema_rate}: must be in [0, 1)"
        )
    if c.f_bin_num < 1:
        raise ValueError(f"f_distill.bin_num={c.f_bin_num}: must be >= 1")
    if c.gan_loss_weight_gen > 0.0:
        logger.info(
            f"GAN (DMD2 teacher-feature disc, FastGen idea 1) ON: "
            f"weight_gen={c.gan_loss_weight_gen}, feature_block_idx="
            f"{c.gan_feature_block_idx} (-1 = middle), disc_head={c.gan_disc_head}, "
            f"disc_lr={c.gan_disc_lr}, r1_weight={c.gan_r1_weight}, "
            f"use_same_t_noise={c.gan_use_same_t_noise}, "
            f"delay_steps={c.gan_delay_steps}, warmup_steps={c.gan_warmup_steps}."
        )
        if c.f_div != "rkl":
            logger.info(
                f"f-distill (FastGen idea 2) ON: f_div={c.f_div!r}, ratio∈"
                f"[{c.f_ratio_lower}, {c.f_ratio_upper}], "
                f"ema_rate={c.f_ratio_ema_rate}, bin_num={c.f_bin_num}, "
                f"normalization={c.f_ratio_normalization}."
            )

    # Sampling / schedule / optimizer
    if c.flow_shift <= 0.0:
        raise ValueError(f"sampling.flow_shift={c.flow_shift}: must be > 0")
    if use_anchor and not c.detach_after_first:
        logger.warning(
            "detach_after_first=False: the mode-seeking DMD gradient can override "
            "the diversity mapping (their Fig 5). A/B only — keep True for "
            "production."
        )
    if c.dynamic_schedule and use_anchor and c.student_steps < 2:
        raise ValueError(
            "dmd.dynamic_schedule=true with base_loss='dpdmd' needs "
            f"student_steps >= 2 (got {c.student_steps}): step 0 is the anchor "
            "and DMD needs at least one refinement step."
        )
    if c.fake_rank < c.student_rank:
        logger.warning(
            f"fake_rank={c.fake_rank} < student_rank={c.student_rank}: DM "
            "regularizer has less capacity than the student — proposal R1 risk "
            "amplified. Consider bumping fake_rank to 2 x student_rank."
        )
    if c.norm_floor <= 0.0:
        raise ValueError(f"dmd.norm_floor={c.norm_floor}: must be > 0 (latent scale)")
    if c.fake_steps_per_student_step < 1:
        raise ValueError(
            f"optim.fake_steps_per_student_step={c.fake_steps_per_student_step}: "
            "must be ≥ 1"
        )
    if c.single_prompt_idx is not None and c.batch_size != 1:
        # single-prompt mode slices the dataset to one sample. With drop_last=True
        # and batch_size > 1 the dataloader yields zero batches and the loop
        # silently no-ops.
        raise ValueError(
            f"--single_prompt_idx requires batch_size=1 (got {c.batch_size}). "
            "Single-prompt overfit mode pins the dataset to one sample; a "
            "batch_size > 1 dataloader with drop_last=True would yield zero batches."
        )

    # Run summary
    logger.info(
        "DM gradient policy: "
        + (
            f"(b) x0-norm, norm_floor={c.norm_floor} — τ cancels, ≈ "
            "magnitude-normalized"
            if c.dm_x0_norm
            else "(a) τ-damping [default]"
        )
    )
    if use_anchor:
        logger.info(
            "DP-DMD: first-step diversity anchor "
            f"k_anchor={c.k_anchor}/{c.teacher_anchor_steps} teacher steps, "
            f"div_weight={c.div_weight}, detach_after_first={c.detach_after_first}, "
            f"student N={c.student_steps} @ flow_shift={c.flow_shift}, "
            f"teacher_cfg={c.teacher_cfg}."
        )
    else:
        logger.info(
            f"plain DMD2 (no diversity anchor): student N={c.student_steps} @ "
            f"flow_shift={c.flow_shift}, teacher_cfg={c.teacher_cfg}."
        )


def snapshot_toml_text(c: TurboConfig, *, source_config: str | None = None) -> str:
    """Render the fully-resolved turbo config as a provenance TOML snapshot.

    Unlike :func:`tb_config_text` (a TB summary of a hand-picked subset), this
    dumps *every* resolved field — CLI overrides folded in — so the run log dir
    becomes a self-contained record of "this run + the config that produced it".
    It's the turbo analogue of the ``<output_name>.snapshot.toml`` that
    ``train.py`` writes for the LoRA family.
    """
    return dataclass_snapshot_toml(
        c,
        title="Anima turbo distillation — resolved config snapshot",
        source_config=source_config,
    )


# TensorBoard config summary — the hand-picked subset (v1 key set/order).
_TB_KEYS = (
    "base_loss",
    "gan_loss_weight_gen",
    "softrank_weight",
    "cdm_weight",
    "f_div",
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
    "use_masked_loss",
    "data_dir",
    "dit_path",
)


def tb_config_text(c: TurboConfig) -> str:
    """Formatted TensorBoard config summary (same key set as v1)."""
    return dataclass_tb_text(c, include=_TB_KEYS)
