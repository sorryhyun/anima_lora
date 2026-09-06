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
        "--fake_tau_banks",
        type=int,
        default=-1,
        help="τ-split critic (turbo_tau_split_critic Phase 1): 2 = two fake "
        "LoRAs, one owning each τ band; both updates and DMD queries route by "
        "the drawn τ vs --fake_tau_boundary. Total critic compute is unchanged "
        "(the existing updates are partitioned, not added). Requires "
        "batch_size=1. Default: TOML (network.fake_tau_banks, default 1 = "
        "byte-identical single-critic loop).",
    )
    parser.add_argument(
        "--fake_tau_boundary",
        type=float,
        default=-1.0,
        help="Raw-τ split point for --fake_tau_banks 2: bank 0 owns [0, b), "
        "bank 1 owns [b, 1]. Uniform t_distribution → even update split at "
        "0.5. Default: TOML (network.fake_tau_boundary, default 0.5).",
    )
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
        "`mask_dir` from configs/preprocess.toml). Mirrors data_dir's subdir "
        "layout.",
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
        "--dual_pool",
        dest="dual_pool",
        action="store_const",
        const=True,
        default=None,
        help="Split the student into TWO always-on plain-LoRA pools on the same "
        "frozen DiT: pool A (div_pool_rank, zero-init) receives ONLY the step-0 "
        "diversity gradient, pool B (student_rank, warm-started) ONLY the "
        "DMD/GAN/CDM refinement gradients. Both active every forward, so the "
        "merged ΔW_A+ΔW_B saves as a plain stock LoRA (exact concat, no SVD). "
        "Finishes the parameter-level div/DMD separation detach_after_first only "
        "does at the graph level. Requires dpdmd + detach_after_first; mutually "
        "exclusive with per_step_expert. Default: TOML "
        "(network.dual_pool, default false).",
    )
    parser.add_argument(
        "--no_dual_pool",
        dest="dual_pool",
        action="store_false",
    )
    parser.add_argument(
        "--div_pool_rank",
        type=int,
        default=-1,
        help="Rank of the diversity pool A under dual_pool (zero-init up, kaiming "
        "down; plain LoRA only — no adaln, no channel scaling). The merged "
        "checkpoint has rank student_rank + div_pool_rank. Default: TOML "
        "(network.div_pool_rank, default 16).",
    )
    parser.add_argument(
        "--div_pool_lr",
        type=float,
        default=-1.0,
        help="AdamW LR for pool A (dual_pool). 0 = inherit student_lr. The "
        "diversity signal is one MSE at div_weight, so A may want a hotter LR "
        "than B — start matched (0), tune later. Default: TOML "
        "(network.div_pool_lr, default 0.0).",
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
    # (diversity), detached, then DMD on x_θ over steps 2..N. See docs/methods/turbo.md.
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

    parser.add_argument(
        "--base_loss",
        type=str,
        default=None,
        choices=("dpdmd", "dmd"),
        help="Objective: 'dpdmd' (first-step teacher anchor, default) or 'dmd' "
        "(plain DMD2 — no anchor, allows student_steps=1). Default: TOML "
        "(base_loss, default 'dpdmd').",
    )

    # DMD2 teacher-feature GAN (FastGen idea 1; off by default).
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

    # Soft-rank caption-discrimination auxiliary (off by default).
    parser.add_argument(
        "--softrank_weight",
        type=float,
        default=-1.0,
        help="λ on the step-0 soft-rank caption loss (pushes the matched caption "
        "to explain the diversity anchor better than k mismatched ones). 0 "
        "disables the whole path (byte-identical DP-DMD). First live value: 0.05. "
        "Default: TOML (softrank.weight, default 0).",
    )
    parser.add_argument(
        "--softrank_k",
        type=int,
        default=-1,
        help="Number of shuffled-caption negatives per firing (k extra no_grad "
        "student forwards). Must be >= 2. Default: TOML (softrank.k, default 2).",
    )
    parser.add_argument(
        "--softrank_every_n",
        type=int,
        default=-1,
        help="Fire the soft-rank term every N student steps (amortizes the k "
        "extra forwards). Default: TOML (softrank.every_n, default 4).",
    )
    parser.add_argument(
        "--softrank_softness",
        type=float,
        default=-1.0,
        help="Temperature τ of the soft-rank relaxation (smaller = closer to the "
        "hard integer rank). Default: TOML (softrank.softness, default 0.1).",
    )
    parser.add_argument(
        "--softrank_pool_size",
        type=int,
        default=-1,
        help="Capacity of the cross-step caption pool the negatives are drawn from "
        "(lets the term fire at batch_size=1). Must be >= k. Each caption is ~1 MiB "
        "(bf16 [512,1024]), so the pool costs pool_size MiB of VRAM. Default: TOML "
        "(softrank.pool_size, default 64).",
    )
    parser.add_argument(
        "--softrank_warmup_ratio",
        type=float,
        default=-1.0,
        help="Fraction of the caption pool that must fill before the term fires "
        "(so negatives are drawn from a representative shuffle, not the last few "
        "captions). 1.0 = wait for a full pool; 0 = fire as soon as k are cached. "
        "Default: TOML (softrank.warmup_ratio, default 1.0).",
    )

    # CDM off-trajectory loss (L_CDM; docs/proposal/cdm.md Phase 1, off by default).
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

    # f-distill reweighting (FastGen idea 2; needs the GAN disc).
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
    # τ-split critic (turbo_tau_split_critic Phase 1): 1 = single fake (the
    # shipped, byte-identical loop); 2 = dual banks split at fake_tau_boundary
    # (bank 0 owns [0, b)). Requires batch_size=1.
    fake_tau_banks: int
    fake_tau_boundary: float
    attn_mode: str
    use_custom_down_autograd: bool
    # Per-input-channel rebalance on each lora_down. Bit-equivalent at init,
    # merges out — a gradient-conditioning lever, not an inference correction.
    channel_scaling_alpha: float
    # Per-step expert (dual-B-head student): step_expert_K = student_steps, head
    # k serves denoise step k. Off → single-head student.
    per_step_expert: bool
    step_expert_K: int
    # Dual-pool gradient routing (_archive/proposals/turbo_dual_pool_grad_routing.md):
    # two always-on plain-LoRA pools — pool A (div_pool_rank) sees only the
    # step-0 diversity gradient, pool B (student_rank) only the DMD/GAN/CDM
    # refinement gradients. Merged ΔW_A+ΔW_B saves as one plain LoRA. Requires
    # dpdmd + detach_after_first; mutually exclusive with per_step_expert. Off →
    # single-pool student, byte-identical to the shipped loop.
    dual_pool: bool
    div_pool_rank: int
    div_pool_lr: float  # 0 = inherit student_lr
    # SVD-Down init for the plain-LoRA student (down_init="weight_svd"): seed
    # lora_down from W0's top-r right singular vectors, scale-matched. Mutually
    # exclusive with per_step_expert (guarded in TurboDMDNetwork).
    student_down_init: str
    # Same lever on the fake/critic (always a plain single-head LoRA).
    fake_down_init: str
    # Warm start: initialize the stack's ΔW from a plain LoRA checkpoint
    # (e.g. an official-release delta extracted by
    # scripts/toolkits/extract_delta_lora.py), SVD-truncated to the stack's rank.
    # Empty = default init. Incompatible with per_step_expert /
    # down_init="weight_svd" (those parameterize or seed what this overwrites).
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
    # AOT min-cut partitioner tuning (mirrors train.py's partitioner_* args):
    # change what the default partition is willing to recompute, on top of
    # activation_memory_budget. Ignored under --grad_ckpt (same gate as the
    # budget). aggressive_recomputation is THE settled memory lever
    # ([[project_partitioner_flags_phase0]]).
    partitioner_recompute_views: bool
    partitioner_aggressive_recomputation: bool

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
    # CDM dynamic continuous schedule (arXiv:2605.06376): re-sample the student
    # rollout grid every iteration (N ~ U{..student_steps}, continuous anchors)
    # instead of training only on the fixed inference grid. Validation/inference
    # stay on the static grid; the DP anchor composes unchanged (t₁=1 pinned).
    dynamic_schedule: bool

    # CDM off-trajectory loss (L_CDM, docs/proposal/cdm.md Phase 1): supervise
    # the student's local x0 estimate at a velocity-extrapolated off-trajectory
    # point (t' ~ U(0,1)) with the same real-vs-fake delta as the DM branch.
    # 0 = the whole path off (byte-identical loop).
    cdm_weight: float

    # Base objective selector
    base_loss: str

    # DMD2 teacher-feature GAN (idea 1) + f-distill reweighting (idea 2)
    gan_loss_weight_gen: float
    gan_feature_block_idx: int  # -1 → middle block (resolved in distill.py)
    gan_disc_lr: float
    gan_disc_hidden: int  # <= 0 → inner_dim // 2
    gan_disc_head: str  # "pooled" (per-tap logit) | "token" (LADD-style per-token)
    gan_delay_steps: int  # generator-side λ held at 0 for the first N steps (disc still trains)
    gan_warmup_steps: int  # then λ ramps 0 → weight_gen over N steps (0 = instant-on)
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

    # Soft-rank caption-discrimination auxiliary (turbo_caption_ranking.md Phase 1)
    softrank_weight: float
    softrank_k: int
    softrank_every_n: int
    softrank_softness: float
    softrank_pool_size: int
    softrank_warmup_ratio: float

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
    fake_tau_banks = int(_pick(args.fake_tau_banks, cfg, "network.fake_tau_banks", 1))
    fake_tau_boundary = float(
        _pick(args.fake_tau_boundary, cfg, "network.fake_tau_boundary", 0.5)
    )
    if fake_tau_banks not in (1, 2):
        raise ValueError(
            f"network.fake_tau_banks={fake_tau_banks}: expected 1 (single critic) "
            "or 2 (τ-split)."
        )
    if not (0.0 < fake_tau_boundary <= 1.0):
        raise ValueError(
            f"network.fake_tau_boundary={fake_tau_boundary}: must be in (0, 1]."
        )
    if fake_tau_banks == 2 and batch_size != 1:
        # τ is per-sample; B=1 makes the routing decision a scalar. B>1 would
        # need a split-batch double forward — out of scope for v0.
        raise ValueError(
            f"network.fake_tau_banks=2 requires batch_size=1 (got {batch_size}): "
            "the bank routing is by the batch's scalar τ."
        )
    if fake_tau_banks == 2:
        logger.info(
            f"τ-split critic ON: 2 fake banks, boundary={fake_tau_boundary} "
            "(bank 0 = low τ). Updates AND DMD queries route by drawn τ; total "
            "critic compute unchanged."
        )
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
    # Defaults off so existing turbo snapshots reproduce bit-for-bit.
    channel_scaling_alpha = float(
        _pick(args.channel_scaling_alpha, cfg, "channel_scaling_alpha", 0.0)
    )

    # Masked loss
    if args.use_masked_loss is None:
        use_masked_loss = bool(_flatten(cfg, "use_masked_loss", False))
    else:
        use_masked_loss = bool(args.use_masked_loss)
    mask_dir = _pick(args.mask_dir, cfg, "mask_dir", _default_mask_dir())

    # DMD core
    student_steps = int(_pick(args.student_steps, cfg, "dmd.student_steps", 4))
    teacher_cfg = float(_flatten(cfg, "dmd.teacher_cfg", 4.0))
    # DM-branch gradient policy: (a) τ-damping [default] vs (b) x0-norm. Alternative
    # policies, not additive; (b) ≈ "drop the τ-weight, magnitude-normalize."
    dm_x0_norm = bool(_pick(args.dm_x0_norm, cfg, "dmd.dm_x0_norm", False))
    norm_floor = float(_pick(args.norm_floor, cfg, "dmd.norm_floor", 0.05))
    dmd_grad_step = str(_pick(args.dmd_grad_step, cfg, "dmd.grad_step", "all"))
    # Default off so existing turbo snapshots reproduce bit-for-bit.
    dynamic_schedule = bool(_flatten(cfg, "dmd.dynamic_schedule", False))

    base_loss = _pick(args.base_loss, cfg, "base_loss", "dpdmd")

    # weight=0 keeps the whole L_CDM path off → byte-identical loop (no extra
    # forwards, no extra RNG draws).
    cdm_weight = float(_pick(args.cdm_weight, cfg, "cdm.weight", 0.0))

    # weight_gen=0 keeps the whole GAN/disc path off → byte-identical DP-DMD.
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
    # Both default 0 → the generator term engages at full weight from step 0
    # (byte-identical shipped loop).
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
    f_div = _pick(args.f_div, cfg, "f_distill.f_div", "rkl")
    f_ratio_lower = float(_flatten(cfg, "f_distill.ratio_lower", 0.1))
    f_ratio_upper = float(_flatten(cfg, "f_distill.ratio_upper", 20.0))
    f_ratio_ema_rate = float(_flatten(cfg, "f_distill.ratio_ema_rate", 0.0))
    f_bin_num = int(_flatten(cfg, "f_distill.bin_num", 10))
    f_ratio_normalization = bool(_flatten(cfg, "f_distill.ratio_normalization", True))

    # weight=0 keeps the whole soft-rank path off → byte-identical DP-DMD (no
    # extra forwards, no extra RNG draws, no negatives loaded).
    softrank_weight = float(_pick(args.softrank_weight, cfg, "softrank.weight", 0.0))
    softrank_k = int(_pick(args.softrank_k, cfg, "softrank.k", 2))
    softrank_every_n = int(_pick(args.softrank_every_n, cfg, "softrank.every_n", 4))
    softrank_softness = float(
        _pick(args.softrank_softness, cfg, "softrank.softness", 0.1)
    )
    softrank_pool_size = int(
        _pick(args.softrank_pool_size, cfg, "softrank.pool_size", 64)
    )
    softrank_warmup_ratio = float(
        _pick(args.softrank_warmup_ratio, cfg, "softrank.warmup_ratio", 1.0)
    )

    # step_expert_K = student_steps so head k ↔ denoise step k by construction.
    # K==1 collapses to a plain LoRA, so the network factory ignores it there.
    if args.per_step_expert is None:
        per_step_expert = bool(_flatten(cfg, "network.per_step_expert", False))
    else:
        per_step_expert = bool(args.per_step_expert)
    step_expert_K = student_steps if per_step_expert else 0

    if args.dual_pool is None:
        dual_pool = bool(_flatten(cfg, "network.dual_pool", False))
    else:
        dual_pool = bool(args.dual_pool)
    div_pool_rank = int(_pick(args.div_pool_rank, cfg, "network.div_pool_rank", 16))
    div_pool_lr = float(_pick(args.div_pool_lr, cfg, "network.div_pool_lr", 0.0))
    if dual_pool and div_pool_rank < 1:
        raise ValueError(
            f"network.div_pool_rank={div_pool_rank}: must be >= 1 under dual_pool."
        )
    if dual_pool and div_pool_lr < 0.0:
        raise ValueError(
            f"network.div_pool_lr={div_pool_lr}: must be >= 0 (0 = inherit student_lr)."
        )

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
    for name, path, conflicts in (
        (
            "student",
            student_init_weights,
            per_step_expert or student_down_init != "kaiming",
        ),
        ("fake", fake_init_weights, fake_down_init != "kaiming"),
    ):
        if path and conflicts:
            raise ValueError(
                f"network.{name}_init_weights is incompatible with "
                f"per_step_expert / {name}_down_init='weight_svd' — the warm "
                "start overwrites what those parameterize or seed."
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
    # lr_schedule was removed 2026-07-14: the "constant" arm (superturbo_B2)
    # never settles and rendered worse than its cosine twin — cosine is the
    # only shape again (see primitives.make_scheduler).
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
    if softrank_weight < 0.0:
        raise ValueError(f"softrank.weight={softrank_weight}: must be >= 0")
    if softrank_weight > 0.0:
        if base_loss != "dpdmd":
            # The term sites on the DP-DMD step-0 diversity anchor (v_target); plain
            # DMD2 has no anchor, so there's nothing to rank against.
            raise ValueError(
                f"softrank.weight > 0 requires base_loss='dpdmd' (got {base_loss!r}) "
                "— the soft-rank term rides the step-0 diversity anchor."
            )
        if softrank_k < 2:
            # softrank needs >= 2 candidates for a non-degenerate rank (chance 1/3
            # at k=2, matching the Phase-0 probe).
            raise ValueError(f"softrank.k={softrank_k}: must be >= 2")
        if softrank_every_n < 1:
            raise ValueError(f"softrank.every_n={softrank_every_n}: must be >= 1")
        if softrank_pool_size < softrank_k:
            # The pool must hold at least k captions or it never reaches `ready`.
            raise ValueError(
                f"softrank.pool_size={softrank_pool_size}: must be >= "
                f"softrank.k={softrank_k}."
            )
        if not (0.0 <= softrank_warmup_ratio <= 1.0):
            raise ValueError(
                f"softrank.warmup_ratio={softrank_warmup_ratio}: must be in [0, 1]."
            )
        if int(args.blocks_to_swap) > 0:
            # The k extra student forwards are the offloader's audited-risk area
            # ([[project_blockswap_extra_forwards_gradcache]]); turbo keeps the DiT
            # resident by default. Fail at config time rather than desync the swap.
            raise ValueError(
                "softrank.weight > 0 requires blocks_to_swap=0 — the extra "
                "caption-negative forwards are unaudited under block swap."
            )
        logger.info(
            "soft-rank caption auxiliary ON (turbo_caption_ranking.md Phase 1): "
            f"weight={softrank_weight}, k={softrank_k}, every_n={softrank_every_n}, "
            f"softness={softrank_softness}, pool_size={softrank_pool_size} "
            f"(~{softrank_pool_size} MiB), warmup_ratio={softrank_warmup_ratio}."
        )
    if cdm_weight < 0.0:
        raise ValueError(f"cdm.weight={cdm_weight}: must be >= 0")
    if cdm_weight > 0.0:
        if per_step_expert:
            # Heads are keyed to fixed grid steps; the off-trajectory forward
            # runs at an arbitrary continuous t' no head owns.
            raise ValueError(
                "cdm.weight > 0 is incompatible with per_step_expert (the "
                "off-trajectory forward runs at t' ~ U(0,1), which no per-step "
                "head owns)."
            )
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
        if int(args.blocks_to_swap) > 0:
            # Extra per-step forwards are the offloader's audited-risk area
            # ([[project_blockswap_extra_forwards_gradcache]]); turbo keeps the
            # DiT resident by default. Fail at config time, don't desync.
            raise ValueError(
                "cdm.weight > 0 requires blocks_to_swap=0 — the off-trajectory "
                "forwards are unaudited under block swap."
            )
        logger.info(
            f"L_CDM off-trajectory loss ON (docs/proposal/cdm.md Phase 1): "
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
            f"{gan_feature_block_idx} (-1 = middle), disc_head={gan_disc_head}, "
            f"disc_lr={gan_disc_lr}, r1_weight={gan_r1_weight}, "
            f"use_same_t_noise={gan_use_same_t_noise}, "
            f"delay_steps={gan_delay_steps}, warmup_steps={gan_warmup_steps}."
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
    if dual_pool:
        # The routing window IS the split backward: pool A takes the step-0
        # diversity backward alone, pool B the combined refinement backward.
        # Without the split there is nothing to route.
        if not use_anchor:
            raise ValueError(
                "network.dual_pool=true requires base_loss='dpdmd' (use_anchor): "
                "plain DMD has no step-0 diversity gradient to route to pool A."
            )
        if not detach_after_first:
            raise ValueError(
                "network.dual_pool=true requires dpdmd.detach_after_first=true: "
                "the routing window is the split backward — with the two losses "
                "fused into one backward there is no seam to gate the pools on."
            )
        if per_step_expert:
            # Both restructure the student; the per_step_expert heads also break
            # the plain-LoRA bake this design exists to preserve.
            raise ValueError(
                "network.dual_pool=true is mutually exclusive with per_step_expert "
                "(both restructure the student; dual_pool is the plain-LoRA-mergeable "
                "formulation of the same div/DMD split)."
            )
        logger.info(
            f"dual-pool gradient routing ON: pool A rank={div_pool_rank} "
            f"(zero-init, step-0 diversity grad only), pool B rank={student_rank} "
            f"(warm-started, DMD/GAN/CDM grad only); merged rank="
            f"{student_rank + div_pool_rank}, saves as one plain LoRA. "
            f"div_pool_lr={div_pool_lr or student_lr}."
        )
    if t_distribution not in ("uniform", "sigmoid"):
        raise ValueError(
            f"sampling.t_distribution={t_distribution!r}: expected 'uniform' or 'sigmoid'"
        )
    if dynamic_schedule:
        if per_step_expert:
            # Step-expert heads are keyed to fixed grid positions; a per-iteration
            # random grid makes head identity meaningless.
            raise ValueError(
                "dmd.dynamic_schedule=true is incompatible with per_step_expert "
                "(heads are keyed to fixed grid steps)."
            )
        if use_anchor and student_steps < 2:
            raise ValueError(
                "dmd.dynamic_schedule=true with base_loss='dpdmd' needs "
                f"student_steps >= 2 (got {student_steps}): step 0 is the anchor "
                "and DMD needs at least one refinement step."
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
        fake_tau_banks=fake_tau_banks,
        fake_tau_boundary=fake_tau_boundary,
        attn_mode=attn_mode,
        use_custom_down_autograd=use_custom_down_autograd,
        channel_scaling_alpha=channel_scaling_alpha,
        per_step_expert=per_step_expert,
        step_expert_K=step_expert_K,
        dual_pool=dual_pool,
        div_pool_rank=div_pool_rank,
        div_pool_lr=div_pool_lr,
        student_down_init=student_down_init,
        fake_down_init=fake_down_init,
        student_init_weights=student_init_weights,
        fake_init_weights=fake_init_weights,
        train_adaln=train_adaln,
        fake_adaln=fake_adaln,
        adaln_rank=adaln_rank,
        adaln_alpha=adaln_alpha,
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
        dynamic_schedule=dynamic_schedule,
        cdm_weight=cdm_weight,
        base_loss=base_loss,
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
        f_div=f_div,
        f_ratio_lower=f_ratio_lower,
        f_ratio_upper=f_ratio_upper,
        f_ratio_ema_rate=f_ratio_ema_rate,
        f_bin_num=f_bin_num,
        f_ratio_normalization=f_ratio_normalization,
        softrank_weight=softrank_weight,
        softrank_k=softrank_k,
        softrank_every_n=softrank_every_n,
        softrank_softness=softrank_softness,
        softrank_pool_size=softrank_pool_size,
        softrank_warmup_ratio=softrank_warmup_ratio,
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
        partitioner_recompute_views=bool(
            _flatten(cfg, "partitioner_recompute_views", False)
        ),
        partitioner_aggressive_recomputation=bool(
            _flatten(cfg, "partitioner_aggressive_recomputation", False)
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
