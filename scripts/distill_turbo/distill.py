"""Turbo distillation main loop — DP-DMD (diversity-preserved DMD).

Usage:
    python -m scripts.distill_turbo.distill [--config configs/methods/turbo.toml] ...

The math walkthrough lives in :mod:`scripts.distill_turbo`; this file is the
per-step orchestrator (teacher K-step anchor → diversity-supervised first step →
DMD-refined N-step student rollout → fake/critic update → save).
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from library.anima import weights as anima_utils
from library.anima.models import Anima
from library.datasets.cache import CachedDataset
from library.inference.sampling import get_timesteps_sigmas
from library.inference.uncond import (
    default_uncond_path,
    load_uncond_crossattn,
    uncond_for_batch,
)
from library.runtime.dynamo import pin_dynamo_limit as _pin_dynamo_limit
from library.runtime.harness import (
    compile_dit_blocks,
    compile_signature,
    enable_training_grad_ckpt,
    isolate_compile_cache,
    place_dit_for_training,
)
from library.training.repa import relational_align_loss
from library.vision.buckets import get_bucket_spec
from networks.methods.turbo_dmd import (
    TurboDMDNetwork,
    gan_loss_discriminator,
    gan_loss_generator,
)

from .config import (
    build_argparser,
    load_turbo_config,
    resolve_config,
    snapshot_toml_text,
    tb_config_text,
)
from .diversity import run_diversity_validation
from .metrics import TurboMetrics, tqdm_postfix, write_scalars
from .primitives import (
    PadCache,
    make_collate,
    make_scheduler,
    renoise,
    sample_t,
)
from .warmup import run_fake_warmup

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# `{stem}_{W:04d}x{H:04d}_anima.npz` → pixel (W, H); token count = (W//16)*(H//16).
_LATENT_RES_RE = re.compile(r"_(\d{3,4})x(\d{3,4})_anima\.npz$")


def _cached_token_counts(data_dir: str) -> set:
    """Distinct token counts present in the cached latents under ``data_dir``.

    Self-describing compile budget (mirrors ``train.py::_derive_token_budget``):
    the on-disk caches are the source of truth for which tiers the distillation
    pool spans, so the dynamo cache is sized from what's really there rather than
    from ``target_res``. Parses the pixel (W, H) out of each paired latent's
    filename (no per-file I/O), so it tracks the actual training sample set.
    """
    from library.io.cache import discover_cached_pairs

    counts: set = set()
    for img in discover_cached_pairs(data_dir):
        m = _LATENT_RES_RE.search(os.path.basename(img.npz_path))
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            counts.add((w // 16) * (h // 16))
    return counts


def _step_tag(step: int) -> str:
    """Human checkpoint suffix: 1000 -> ``1k``, 8000 -> ``8k``, else raw count.

    Matches the hand-rolled ``_1k`` / ``_500`` naming the runs already use.
    """
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def nearest_student_step(student_sigmas: list[float], tau: float, n_steps: int) -> int:
    """Per-step-expert head selection for the REPA term (turbo_repa.md §σ choice).

    The student grid has ``n_steps + 1`` σ points; head ``k`` operates at
    ``student_sigmas[k]`` (k < n_steps). The REPA forward runs at a sampled
    renoise level τ that sits between grid points, so route it to the head
    whose operating σ is nearest. No-op for the single-head student.
    """
    return min(range(n_steps), key=lambda i: abs(student_sigmas[i] - tau))


def mean_var_kl(
    x: torch.Tensor, mu_t: torch.Tensor | float, sigma2_t: torch.Tensor | float
) -> torch.Tensor:
    """Mean-variance regularizer (lever B / paper Eq. 7, arXiv:2511.22677).

    KL of each generated image's per-image Gaussian ``N(μ_i, σ²_i)`` toward the
    real-latent target ``N(μ_t, σ²_t)``, averaged over the batch:

        L_mv = (1/B) Σ_i ½[ (σ_i² + (μ_i − μ_t)²)/σ_t² − 1 − log(σ_i²/σ_t²) ]

    Differentiable in ``x`` (= ``x_pred``), so it backprops into the student and
    directly clamps the **variance inflation** that *is* the over-bake's
    oversaturation (§3.2). Stats are per-image over (C, H, W); full-frame even
    under masked loss (paper-faithful — the reg is a global distribution clamp).
    """
    B = x.shape[0]
    flat = x.reshape(B, -1)
    mu_i = flat.mean(dim=1)
    var_i = flat.var(dim=1, unbiased=False)
    kl = 0.5 * (
        (var_i + (mu_i - mu_t) ** 2) / sigma2_t
        - 1.0
        - torch.log((var_i / sigma2_t).clamp_min(1e-8))
    )
    return kl.mean()


def calibrate_mean_var(
    dataloader: torch.utils.data.DataLoader,
    *,
    max_batches: int = 0,
    norm_floor: float = 0.05,
) -> tuple[float, float]:
    """Exact one-pass global mean/variance of the real cached latents.

    The Eq.7 reg target is a static dataset statistic — a single scalar
    ``(μ, σ²)`` over every latent element — so we measure it directly rather
    than EMA-tracking it during training. Accumulates count / sum / sum-of-
    squares in fp64 (population variance, ``unbiased=False`` — matching the
    per-image stats in :func:`mean_var_kl`) for numerical stability across the
    whole pool. ``max_batches <= 0`` scans the full dataset; a positive cap
    trades a little exactness for I/O (the global scalar converges fast).
    Returns ``(μ_t, σ²_t)`` with σ²_t floored at ``norm_floor²``.
    """
    n = 0
    s = 0.0
    s2 = 0.0
    seen = 0
    for batch in dataloader:
        # Batch layout mirrors the training loop: masked adds a trailing mask.
        latents = batch[1].double()
        flat = latents.reshape(-1)
        n += flat.numel()
        s += float(flat.sum())
        s2 += float((flat * flat).sum())
        seen += 1
        if max_batches > 0 and seen >= max_batches:
            break
    if n == 0:
        raise RuntimeError(
            "mean-variance calibration scanned 0 latents — empty dataloader "
            "(check data_dir / curation keep_list / drop_last vs batch_size)."
        )
    mu = s / n
    var = max(s2 / n - mu * mu, norm_floor**2)
    logger.info(
        f"mean-variance calibration: scanned {seen} batches / {n:,} latent "
        f"elements → μ_t={mu:.6g}, σ²_t={var:.6g}"
    )
    return mu, var


# --- f-distill reweighting (FastGen idea 2; f_distill.py:20 + _get_f_div_weighting_h)
# h = f'(r) where the density ratio r = exp(disc_logits) comes free from the GAN
# head (idea 1). "rkl" ≡ uniform h ≡ plain DMD2 (the off-by-default no-op).
_F_DIV_WEIGHTING = {
    "rkl": lambda r: torch.ones_like(r),
    "kl": lambda r: r,
    "js": lambda r: 1.0 - 1.0 / (1.0 + r),
    "sf": lambda r: 1.0 / (1.0 + r),
    "neyman": lambda r: 1.0 / torch.clamp(r, min=1e-8),
    "sh": lambda r: r**0.5,  # squared Hellinger
    "jf": lambda r: 1.0 + r,  # Jeffreys
}


def f_div_weighting_h(
    fake_logits: torch.Tensor,
    t: torch.Tensor,
    *,
    f_div: str,
    ratio_lower: float,
    ratio_upper: float,
    ema_rate: float,
    bins: torch.Tensor | None,
    bin_num: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-sample f-divergence reweight ``h(t, r)`` for the DMD signal.

    Port of ``FdistillModel._get_f_div_weighting_h`` (f_distill.py:59). ``r =
    exp(mean disc logits)`` clamped to ``[ratio_lower, ratio_upper]`` after a ±10
    logit clamp; an optional per-τ EMA histogram (``bins``) normalizes ``r`` so
    ``h`` isn't dominated by the batch's τ-distribution; ``h`` is renormalized to
    unit batch-mean. Everything is fp32 and detached — ``h`` only *scales* the
    already-detached DMD signal. Returns ``(h [B], updated_bins)``; ``bins`` is
    ``None`` when normalization is off.
    """
    logits = fake_logits.float()
    clamped = torch.clamp(logits.mean(dim=1), min=-10.0, max=10.0)
    ratio = torch.exp(clamped).detach()
    ratio = torch.clamp(ratio, ratio_lower, ratio_upper)
    if bins is not None:
        # τ is on [0, 1] (renoise level); bin directly over that range.
        tt = t.float().clamp(0.0, 1.0)
        bin_width = 1.0 / bin_num
        idx = (tt / bin_width).floor().long().clamp(0, bin_num - 1)
        cnt = torch.bincount(idx, minlength=bin_num).float()
        ratio_sum = torch.bincount(idx, weights=ratio, minlength=bin_num).float()
        valid = cnt > 0
        new_vals = ratio_sum / (cnt + 1e-6)
        bins = bins.clone()
        bins[valid] = bins[valid] * ema_rate + (1.0 - ema_rate) * new_vals[valid]
        ratio = ratio / (bins[idx] + 1e-6)
    h = _F_DIV_WEIGHTING[f_div](ratio)
    h = h / (h.mean() + 1e-6)
    return h.detach(), bins


@contextmanager
def selective_block_grad_ckpt(model: Anima):
    """Arm per-block gradient checkpointing for one forward, then restore.

    ``Block.forward`` self-checkpoints when ``gradient_checkpointing`` is set
    (gated on ``self.training`` + grad enabled). The decision is read eagerly per
    block, so flipping it per call costs no recompile. We snapshot each block's
    three checkpoint flags and restore them on exit, so this composes cleanly with
    a global ``--grad_ckpt`` run without clobbering it.

    We arm the **unsloth-offload** variant, NOT the standard ``torch_checkpoint``
    path. ``block._forward`` (the actual compute) is ``torch.compile``'d, and
    ``checkpoint(compiled_fn, use_reentrant=False)`` is unsupported: the recompute
    diverges from the inductor forward graph (dynamo recompile-storms on the
    GLOBAL_STATE ``num_threads`` flip, falls back to a non-autocast eager path →
    fp32 recompute, mismatched saved-tensor set, ``CheckpointError``). The unsloth
    path carries ``@torch._disable_dynamo`` (``models.py``), so the compiled
    ``_forward`` runs eager in BOTH forward and recompute → consistent, and it
    offloads saved tensors to CPU (extra VRAM win). The reentrant grad-drop bug
    ([[project_unsloth_reentrant_drops_grad]]) does not apply here: the frozen
    teacher view has no grad-requiring params inside the region, so grad flows
    purely through the grad-requiring input (x_renoised_gan → student).

    Used to wrap ONLY the grad-bearing GAN gen teacher forward: the frozen teacher
    retains ~half the DiT's block activations there purely to backprop into
    x_pred → student, so recomputing them in backward reclaims that peak VRAM
    (~one half-depth forward of compute) — numerically exact (no dropout).
    """
    saved = [
        (
            b.gradient_checkpointing,
            b.cpu_offload_checkpointing,
            b.unsloth_offload_checkpointing,
        )
        for b in model.blocks
    ]
    for b in model.blocks:
        b.gradient_checkpointing = True
        b.cpu_offload_checkpointing = False
        b.unsloth_offload_checkpointing = True
    try:
        yield
    finally:
        for b, (g, c, u) in zip(model.blocks, saved):
            b.gradient_checkpointing = g
            b.cpu_offload_checkpointing = c
            b.unsloth_offload_checkpointing = u


def main():
    args = build_argparser().parse_args()
    cfg = resolve_config(args, load_turbo_config(args.config))

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ---------------- Dynamo / threading hardening (compile-storm guard) ----
    # Pin BOTH the recompile budget and the intra-op thread count BEFORE any
    # block._forward is traced — the turbo loop drives the one compiled block
    # graph under many global states per step (grad_mode × requires_grad ×
    # student/fake/teacher view × {4032, 4200} token families), so the stock
    # per-frame limit of 8 spills to eager mid-run.
    #
    # The budget raise here is what guarantees the limit is up by construction:
    # warmup (the first trace) and the main loop run after this point regardless
    # of where compile first traces. The later, model-aware raise (after
    # compile_dit_blocks) refines ``accumulated_recompile_limit`` with the exact
    # block count and logs the final values; both are idempotent via ``max()``.
    #
    # The raise is context-pinned (``_pin_dynamo_limit`` sets the ContextVar's
    # global default, not just the main-thread override) so it survives into the
    # backward/AOTAutograd compile context where the grad-bearing forward first
    # traces — a plain ``config.recompile_limit = 64`` does NOT, and the budget
    # reverts to 8 at the first grad forward.
    #
    # ``set_num_threads`` force-initializes torch's intra-op pool NOW so
    # ``torch.get_num_threads()`` is constant for the rest of the run. Left to
    # itself the pool inits lazily on the first parallel op and the count flips
    # mid-run — a GLOBAL_STATE guard that recompiles every {grad_mode, view}
    # graph a second time AND is the documented CheckpointError trigger for the
    # GAN grad-ckpt path (see ``selective_block_grad_ckpt``).
    if cfg.torch_compile:
        _pin_dynamo_limit("recompile_limit", cfg.dynamo_recompile_limit)
    torch.set_num_threads(torch.get_num_threads())

    # ---------------- Model ----------------
    logger.info(f"loading DiT: {cfg.dit_path}")
    model: Anima = anima_utils.load_anima_model(
        device,
        cfg.dit_path,
        attn_mode=cfg.attn_mode,
        loading_device="cpu" if cfg.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )

    # Block swap (per-forward prepare hook done at each forward call below).
    # compile_dit_blocks is deferred until AFTER the student/fake apply_to below
    # (see the COMPILE LAST note further down) — order: block-swap → grad-ckpt →
    # apply_to → compile, matching library/runtime/harness.py.
    place_dit_for_training(model, device, blocks_to_swap=cfg.blocks_to_swap)
    enable_training_grad_ckpt(model, enabled=cfg.grad_ckpt)

    # ---------------- LoRA stacks ----------------
    # GAN feature tap (idea 1): resolve the tapped block (−1 → middle) and hand
    # the index set to TurboDMDNetwork so it builds the disc + block hooks. Off
    # when weight_gen == 0 (gan_indices=None → byte-identical DP-DMD).
    gan_on = cfg.gan_loss_weight_gen > 0.0
    gan_indices = None
    if gan_on:
        bidx = cfg.gan_feature_block_idx
        if bidx < 0:
            bidx = model.num_blocks // 2
        if not (0 <= bidx < model.num_blocks):
            raise ValueError(
                f"gan.feature_block_idx resolved to {bidx}, out of range "
                f"[0, {model.num_blocks})"
            )
        gan_indices = {bidx}

    turbo = TurboDMDNetwork(
        unet=model,
        student_rank=cfg.student_rank,
        fake_rank=cfg.fake_rank,
        student_alpha=cfg.student_alpha,
        fake_alpha=cfg.fake_alpha,
        use_custom_down_autograd=cfg.use_custom_down_autograd,
        channel_scaling_alpha=cfg.channel_scaling_alpha,
        student_step_expert_K=cfg.step_expert_K,
        gan_feature_indices=gan_indices,
        gan_disc_hidden=cfg.gan_disc_hidden if cfg.gan_disc_hidden > 0 else None,
    )
    turbo.freeze_dit()
    turbo.student.to(device=device, dtype=dtype)
    turbo.fake.to(device=device, dtype=dtype)
    # Disc stays fp32 (LayerNorm/Linear) for GAN-loss stability — its forward
    # casts the bf16 teacher features to float.
    if turbo.disc is not None:
        turbo.disc.to(device=device)

    # COMPILE LAST — both student.apply_to and fake.apply_to (inside
    # TurboDMDNetwork above) have now monkey-patched the targeted Linears, so
    # torch.compile traces the adapter forward chain rather than the bare DiT.
    # compile is lazy (traces on first _forward call), but compiling here makes
    # the ordering invariant hold by construction, not by the absence of a
    # warmup forward. native-shape flatten, one graph per token count; the pool
    # spans more than the 2 CONSTANT_TOKEN_BUCKETS families.
    # compile_dynamic_seq (mirrors the LoRA-training path): when on, collapse the
    # per-token-count block graphs to one symbolic-seq graph (mark_dynamic on the
    # seq axis only). Size the seq bound + dynamo cache from the token-count
    # families actually present in the cached pool (self-describing, like
    # train.py) — target_res is preprocess-only and inert here. An explicit
    # cfg.target_res (CLI/TOML) still overrides as an escape hatch.
    n_token_families = None
    seq_range = None
    if cfg.target_res:
        from library.datasets.buckets import (
            token_count_families,
            token_count_range,
        )

        n_token_families = token_count_families(cfg.target_res)
        seq_range = token_count_range(cfg.target_res)
    else:
        counts = _cached_token_counts(cfg.data_dir)
        if counts:
            n_token_families = len(counts)
            seq_range = (min(counts), max(counts))
    # Partitioner saved-activation cap (mirrors train.py). Removing
    # custom_down_autograd grows saved-for-backward intermediates; budget<1.0
    # makes the partitioner recompute the cheap ones in backward instead. Must be
    # set BEFORE compile_dit_blocks (partitioning happens at first-forward compile,
    # and this is a plain module attr — no ContextVar revert). Skipped under
    # grad_ckpt: the budget repartitions the joint graph, so checkpoint's recompute
    # pass can pick a different graph than forward → CheckpointError (torch #166926);
    # ckpt already minimizes saved activations, so the cap buys nothing there.
    if cfg.torch_compile and cfg.activation_memory_budget < 1.0 and not cfg.grad_ckpt:
        import torch._functorch.config as _functorch_config

        _functorch_config.activation_memory_budget = cfg.activation_memory_budget
        logger.info(
            f"torch.compile activation_memory_budget = {cfg.activation_memory_budget} "
            "(partitioner recomputes cheap intermediates in backward)"
        )
    elif cfg.activation_memory_budget < 1.0 and cfg.grad_ckpt:
        logger.info(
            "activation_memory_budget ignored: incompatible with grad_ckpt "
            "(and redundant under it)"
        )
    # Isolate the persistent compile caches per compile signature — entries
    # deposited by runs compiled under different seq-range bounds (e.g. an
    # inference run's canonical 4032-floored default) otherwise poison this
    # run's wider dynamic-seq marks: AOTAutogradCache replays the stale narrow
    # guard into the fresh ShapeEnv and the first ≥4032-token trace dies with a
    # ConstraintViolationError (see isolate_compile_cache). Same signature →
    # warm cache reuse, shared with train.py runs of the same tier set.
    if cfg.torch_compile:
        isolate_compile_cache(
            compile_signature(
                n_token_families=n_token_families,
                seq_range=seq_range,
                dynamic_seq=cfg.compile_dynamic_seq,
                mode="",
            )
        )
    compile_dit_blocks(
        model,
        enabled=cfg.torch_compile,
        mode="",
        dynamic_seq=cfg.compile_dynamic_seq,
        n_token_families=n_token_families,
        seq_range=seq_range,
    )
    # Refine the recompile budget now that the model exists: re-assert the
    # per-frame limit (already raised at the top of main(), kept here so the
    # runtime value is logged next to the compile) and size
    # ``accumulated_recompile_limit`` over the exact block code-object count.
    # Both writes are idempotent via ``max()`` — this never lowers the early raise.
    if cfg.torch_compile:
        rl = _pin_dynamo_limit("recompile_limit", cfg.dynamo_recompile_limit)
        arl = _pin_dynamo_limit(
            "accumulated_recompile_limit",
            len(model.blocks) * cfg.dynamo_recompile_limit,
        )
        logger.info(f"dynamo recompile_limit={rl}, accumulated_recompile_limit={arl}")
    # `model.training` gates grad-ckpt inside block.forward; toggled per
    # forward in `_forward` below so no_grad teacher/fake forwards don't
    # incur grad-ckpt setup cost. Initial state set by the first call.

    n_student = sum(p.numel() for p in turbo.student_params())
    n_fake = sum(p.numel() for p in turbo.fake_params())
    logger.info(f"trainable: student={n_student:,}  fake={n_fake:,}")

    # ---------------- Optimizers ----------------
    student_opt = torch.optim.AdamW(
        turbo.student_params(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )
    fake_opt = torch.optim.AdamW(
        turbo.fake_params(),
        lr=cfg.fake_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )

    student_sched = make_scheduler(student_opt, cfg.iterations, cfg.student_lr)
    # The fake optimizer takes ``iterations · fake_steps_per_student_step``
    # updates in the main loop plus ``fake_warmup_steps`` head-start updates
    # BEFORE it (the head-start is now counted in fake updates directly, NOT
    # scaled by the cadence — see warmup.py). The fake scheduler is stepped
    # through both phases, so its total update count — and hence the ``0.02·total``
    # LR warmup span — is sized over the same total. The fake LR warmup therefore
    # overlaps the head-start (the fake enters the main loop already calibrated
    # AND at full LR), and the cosine still lands at the end of the main loop.
    # The student schedule is independent: ``0.02·iterations``, no head-start offset.
    fake_sched = make_scheduler(
        fake_opt,
        cfg.iterations * cfg.fake_steps_per_student_step + cfg.fake_warmup_steps,
        cfg.fake_lr,
    )

    # ---------------- Discriminator optimizer (idea 1) ----------------
    # Stepped once per fake inner step (co-located with the fake/critic update,
    # the same cadence FastGen DMD2 ties the disc to the fake_score). No
    # head-start (disc trains only in the main loop), so its scheduler is sized
    # over iterations · fake_steps_per_student_step.
    disc_opt = disc_sched = None
    if turbo.disc is not None:
        disc_opt = torch.optim.AdamW(
            turbo.disc_params(),
            lr=cfg.gan_disc_lr,
            weight_decay=cfg.weight_decay,
            betas=(0.0, 0.99),  # standard GAN-disc betas
            fused=torch.cuda.is_available(),
        )
        disc_sched = make_scheduler(
            disc_opt, cfg.iterations * cfg.fake_steps_per_student_step, cfg.gan_disc_lr
        )
        n_disc = sum(p.numel() for p in turbo.disc_params())
        logger.info(f"trainable: disc={n_disc:,}")

    # f-distill (idea 2): per-τ EMA histogram buffer for ratio normalization.
    # Training-only scaffolding (never saved — save_student filters to LoRA keys).
    fdistill_on = gan_on and cfg.f_div != "rkl"
    fdistill_bins = None
    if fdistill_on and cfg.f_ratio_normalization:
        fdistill_bins = torch.ones(cfg.f_bin_num, device=device)

    # ---------------- Turbo × REPA (turbo_repa.md Phase 1) ----------------
    # Relational (Gram) alignment of student block-`repa_layer` features to the
    # cached PE-Spatial sidecars of the REAL latents, every `repa_every_n`-th
    # student step. Phase 0 (bench/turbo_repa/) confirmed the student's mid-block
    # representation drifts off the base manifold (worst at mid-σ). weight=0
    # keeps the whole path off → byte-identical DP-DMD.
    repa_on = cfg.repa_weight > 0.0
    repa_spec = None
    repa_feature_set: set | None = None
    if repa_on:
        if not (0 <= cfg.repa_layer < len(model.blocks)):
            raise ValueError(
                f"repa.layer={cfg.repa_layer} out of range "
                f"(DiT has {len(model.blocks)} blocks)"
            )
        repa_spec = get_bucket_spec(cfg.repa_encoder)
        repa_feature_set = {cfg.repa_layer}
        # dog replaces spatial_norm's DC removal when on (mutually exclusive).
        repa_target_desc = (
            f"dog(σ1=min/{cfg.repa_dog_sigma1_div:g}, "
            f"σ2={'off' if cfg.repa_dog_sigma2_div <= 0 else f'min/{cfg.repa_dog_sigma2_div:g}'})"
            if cfg.repa_target_dog
            else f"spatial_norm={cfg.repa_spatial_norm}"
        )
        logger.info(
            f"REPA: aligning student block {cfg.repa_layer}/{len(model.blocks)} "
            f"to {cfg.repa_encoder} (grid≤{repa_spec.t_max_patches}tok) on real "
            f"data, weight={cfg.repa_weight}, every_n={cfg.repa_every_n}, "
            f"{repa_target_desc}"
        )

    # ---------------- Dataset ----------------
    dataset = CachedDataset(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        sample_ratio=cfg.sample_ratio,
        mask_dir=cfg.mask_dir if cfg.use_masked_loss else None,
    )
    if repa_on:
        # Post-construction PE gate (the bespoke-loop mirror of train.py's
        # dataset.load_repa_pe): appends each sample's PE sidecar as the LAST
        # tuple element; collate must be built with the matching flag below.
        dataset.load_repa_pe = True
        dataset.repa_pe_encoder = cfg.repa_encoder
    # Held-out conditioning for the DAVE diversity probe — captured from the
    # FULL sample list before any single-prompt slice mutates it, and chosen
    # distinct from the overfit sample so a collapsed run is visible. We only
    # need the cached crossattn_emb + a latent shape; loaded once, reused every
    # validation pass.
    val_cond = None
    val_latent_shape = None
    val_clean = None
    if cfg.validate_every_n_steps > 0 and len(dataset.samples) > 0:
        n = len(dataset.samples)
        if cfg.val_prompt_idx >= 0:
            v_idx = cfg.val_prompt_idx % n
        else:
            v_idx = n - 1  # auto: last sample
            if cfg.single_prompt_idx is not None and v_idx == cfg.single_prompt_idx % n:
                v_idx = (v_idx - 1) % n  # avoid the overfit sample when possible
        _, v_lat, v_ca, _v_pool = dataset[v_idx][:4]
        val_cond = v_ca.unsqueeze(0).to(device, dtype=dtype)  # (1, seq, D)
        val_latent_shape = (1, *tuple(v_lat.shape))  # (1, C, H, W)
        val_clean = v_lat.unsqueeze(0).to(
            device, dtype=dtype
        )  # (1, C, H, W) for FM MSE
        logger.info(
            f"diversity validation: every {cfg.validate_every_n_steps} steps, "
            f"{cfg.val_diversity_seeds} seeds, held-out idx={v_idx} "
            f"(latent {tuple(v_lat.shape)})"
        )

    if cfg.single_prompt_idx is not None:
        # Phase 0 overfit — wrap as a 1-sample list so the dataloader cycles it.
        # The "N samples from ..." line above is CachedDataset.__init__'s own
        # log, fired BEFORE this slice; we re-log post-slice so the live
        # dataset state is unambiguous in the run log.
        pinned_idx = cfg.single_prompt_idx % len(dataset.samples)
        only = dataset.samples[pinned_idx]
        dataset.samples = [only]
        latent_stem = os.path.basename(only[0])
        logger.info(
            f"single-prompt overfit mode: pinned to idx={cfg.single_prompt_idx} "
            f"(post-slice len(dataset)={len(dataset)}, latent={latent_stem})"
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # bucket-grouped
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_collate(cfg.use_masked_loss, load_repa_pe=repa_on),
    )

    # ---------------- Logging ----------------
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Canonical config snapshot beside the checkpoint, matching train.py's
    # convention (library/config/io.py writes the canonical copy to output_dir
    # and *mirrors* it into the run log dir). This is the provenance record
    # `inference` / `merge` / tooling looks for next to `{output_name}.safetensors`,
    # so write it unconditionally — independent of TB logging (--no_log).
    canonical_snapshot = Path(cfg.output_dir) / f"{cfg.output_name}.snapshot.toml"
    try:
        canonical_snapshot.write_text(
            snapshot_toml_text(cfg, source_config=args.config),
            encoding="utf-8",
        )
        logger.info(f"Config snapshot written: {canonical_snapshot}")
    except OSError as e:
        logger.warning(f"Could not write config snapshot to {canonical_snapshot}: {e}")

    writer = None
    if not cfg.no_log:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log = Path(cfg.log_dir) / run_name
        run_log.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_log))
        writer.add_text("config", tb_config_text(cfg))
        logger.info(f"TB logs -> {run_log}")

        # Mirror the canonical snapshot into the run log dir too, so the
        # timestamped run is a self-contained record of "this run + the config
        # that produced it" (the canonical copy lives next to the checkpoint).
        snapshot_path = run_log / f"{cfg.output_name}.snapshot.toml"
        try:
            snapshot_path.write_text(
                snapshot_toml_text(cfg, source_config=args.config),
                encoding="utf-8",
            )
            logger.info(f"Config snapshot written: {snapshot_path}")
        except OSError as e:
            logger.warning(f"Could not write config snapshot to {snapshot_path}: {e}")

    # ---------------- Forward helper ----------------
    pad_cache = PadCache(dtype)
    # CFG-uncond cross-attention input. Anima's inference path uses the T5("")
    # embedding (real BOS/EOS/sentinel tokens nonzero; only padding zeroed) —
    # passing a fully-zero tensor here is fed-out-of-distribution and the
    # resulting `v_real_uncond_ca` is a meaningless direction that, amplified
    # at (α-1)=3×, drives the student off-manifold (saturated white output).
    # Staged by `make preprocess-te` (or `make distill-prep`); shared with
    # the mod-guidance distill (`library/inference/uncond.py`).
    uncond_path = str(default_uncond_path())
    uncond_base = load_uncond_crossattn(uncond_path, device=device, dtype=dtype)
    logger.info(
        f"loaded T5('') uncond sidecar: {uncond_path}  shape={tuple(uncond_base.shape)}"
    )

    def _forward(
        view: str,
        x: torch.Tensor,
        t_b: torch.Tensor,
        c: torch.Tensor,
        *,
        no_grad: bool,
        return_block_features: set | None = None,
        return_features_early: bool = False,
    ):
        """Switch view, prepare block swap, run forward.

        ``x`` is (B, 16, H, W); we unsqueeze to (B, 16, 1, H, W) inside.

        With ``return_features_early`` (GAN feature tap, idea 3.1) the forward
        stops after the deepest tapped block and returns the feature dict
        ``{block_idx: feat}`` instead of a velocity — the caller pools it through
        the disc and must NOT ``.squeeze(2)`` the result.

        Per-forward CPU prep is the GPU-idle window between launches —
        ``set_view`` short-circuits when already in ``view`` (see
        ``TurboDMDNetwork.set_view``), and the cudagraph step-begin marker
        is hoisted to once per outer step in the loop below.

        The DiT is frozen (``freeze_dit`` in ``__init__``), so ``model.training``
        is left at its post-construction value (``True``) for the whole run —
        grad-ckpt is gated on ``self.training`` inside ``Block.forward``, so it
        stays armed without a per-forward toggle. We deliberately do NOT flip
        train/eval per forward: the no_grad teacher/fake forwards build no
        backward graph regardless, and the recursive submodule walk a per-forward
        toggle triggered was the dominant per-forward CPU stall.

        Checkpointing has two independent levers, both numerically exact (frozen
        teacher, no dropout) and both no-ops on no_grad forwards: the global
        ``--grad_ckpt`` (default OFF) arms unsloth-offload ckpt on EVERY
        grad-bearing forward, while ``gan.grad_ckpt`` (default on) wraps only the
        GAN gen forward (same unsloth-offload path — compile needs the
        ``@torch._disable_dynamo`` it carries; see ``selective_block_grad_ckpt``)
        to reclaim its ~3 GB without the global recompute. Recompute only bites on
        the grad-bearing student/fake-update/GAN forwards.
        """
        turbo.set_view(view)
        if model.blocks_to_swap:
            # free_cache=False: base DiT is frozen, LoRA shapes are constant,
            # block swap moves params at identical shape, and static 4096 tokens
            # pins activation sizes — the allocator reaches a steady state within
            # a few steps and per-forward empty_cache() is pure sync +
            # refragmentation overhead.
            model.prepare_block_swap_before_forward(free_cache=False)
        pad = pad_cache.get(x)
        x_in = x.unsqueeze(2)  # add temporal dim
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=dtype):
            return model.forward_mini_train_dit(
                x_in,
                t_b,
                c,
                padding_mask=pad,
                skip_pooled_text_proj=True,
                return_block_features=return_block_features,
                return_features_early=return_features_early,
            )

    # ---------------- DP-DMD setup ----------------
    # The student/teacher Euler grids are static (token-count-invariant flow-
    # matching σ schedule), so build them once. Both span σ: 1 → 0; the student
    # has `student_steps + 1` points, the teacher anchor grid `teacher_anchor_steps
    # + 1`. `sigmas[i] - sigmas[i+1]` is the Euler dt for step i.
    student_sigmas = get_timesteps_sigmas(cfg.student_steps, cfg.flow_shift, "cpu")[
        1
    ].tolist()
    teacher_anchor_sigmas = get_timesteps_sigmas(
        cfg.teacher_anchor_steps, cfg.flow_shift, "cpu"
    )[1].tolist()
    # Continuous time at the anchor (incoming σ after k_anchor teacher steps).
    # `v_target = (ε − z_tk)/(1 − t_k)` — a σ mismatch here silently mis-scales
    # the diversity target (proposal §6.3), so it's read from the teacher grid,
    # not the student grid.
    t_k_anchor = float(teacher_anchor_sigmas[cfg.k_anchor])
    logger.info(
        f"DP-DMD grids: student σ={['%.3f' % s for s in student_sigmas]}, "
        f"anchor t_k={t_k_anchor:.4f} (teacher step {cfg.k_anchor}/"
        f"{cfg.teacher_anchor_steps})"
    )

    def _teacher_cfg_velocity(x, t_b, c_cond, c_null):
        """CFG-guided teacher velocity ``v_u + α·(v_c − v_u)`` (no grad, fp32).

        Used by the DP-DMD K-step anchor rollout. At ``teacher_cfg == 1`` the
        uncond forward is skipped (single forward).
        """
        v_c = _forward("teacher", x, t_b, c_cond, no_grad=True).squeeze(2)
        if cfg.teacher_cfg == 1.0:
            return v_c.float()
        v_u = _forward("teacher", x, t_b, c_null, no_grad=True).squeeze(2)
        return v_u.float() + cfg.teacher_cfg * (v_c.float() - v_u.float())

    # ---------------- Mean-variance reg target (lever B / S2) ----------------
    # Real-data stats the Eq.7 KL pulls each generated image toward. Either
    # pinned (cfg.mv_sigma2_t > 0) or measured EXACTLY in a one-pass scan over
    # the real latents (cfg.mv_sigma2_t <= 0). The target is a static dataset
    # statistic — a single global (μ, σ²) over all latent elements — so an exact
    # pre-pass strictly beats a running EMA: no decay lag, no batch-to-batch
    # wobble, deterministic, and free in the hot loop. Computed in fp64 via the
    # numerically-stable count/sum/sumsq route over the same `latents` the
    # dataloader yields (the REAL training latents — NOT teacher-synthetic, since
    # the reg is a shield against the teacher's variance inflation). Runs BEFORE
    # the fake head-start: it's model-independent, so doing it first fails fast on
    # an empty/misconfigured pool instead of after burning warmup compute.
    mv_auto = cfg.mean_var_weight > 0.0 and cfg.mv_sigma2_t <= 0.0
    mv_tgt_mu = cfg.mv_mu_t
    mv_tgt_var = cfg.mv_sigma2_t
    if cfg.mean_var_weight > 0.0:
        if mv_auto:
            mv_tgt_mu, mv_tgt_var = calibrate_mean_var(
                dataloader,
                max_batches=cfg.mv_calib_batches,
                norm_floor=cfg.norm_floor,
            )
        logger.info(
            "mean-variance reg ON (lever B / Eq.7): weight="
            f"{cfg.mean_var_weight}, target="
            + (
                f"measured μ_t={mv_tgt_mu:.6g}, σ²_t={mv_tgt_var:.6g} "
                f"(exact, over real latents)"
                if mv_auto
                else f"fixed μ_t={mv_tgt_mu}, σ²_t={mv_tgt_var}"
            )
        )

    # ---------------- Fake (critic) head-start ----------------
    data_iter = iter(dataloader)
    data_iter = run_fake_warmup(
        warmup_steps=cfg.fake_warmup_steps,
        turbo=turbo,
        forward_fn=_forward,
        data_iter=data_iter,
        dataloader=dataloader,
        fake_opt=fake_opt,
        fake_sched=fake_sched,
        grad_clip=cfg.grad_clip,
        t_distribution=cfg.t_distribution,
        sigmoid_scale=cfg.sigmoid_scale,
        device=device,
        dtype=dtype,
        log_interval=cfg.log_interval,
        writer=writer,
    )

    # ---------------- Training loop ----------------
    # base_loss='dpdmd' runs the first-step teacher anchor (diversity); 'dmd' is
    # plain DMD2 with no anchor (student_steps may be 1).
    use_anchor = cfg.base_loss == "dpdmd"
    logger.info(
        f"starting turbo training ({cfg.base_loss}): {cfg.iterations} iterations"
    )
    progress = tqdm(range(cfg.iterations), desc="turbo")
    metrics = TurboMetrics(device)
    repa_warned_missing = False

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = list(batch)
        # PE features ride LAST when the REPA gate is on (None when any sample
        # in the batch lacked its sidecar → the term is skipped this step).
        repa_pe = batch.pop() if repa_on else None
        if cfg.use_masked_loss:
            _idx, latents, crossattn_emb, _pooled, mask = batch
            # float (not bf16): the student loss is assembled in fp32. [B,1,H,W]
            # broadcasts over the [B,16,H,W] grad signal.
            mask = mask.to(device, dtype=torch.float32, non_blocking=True)
        else:
            _idx, latents, crossattn_emb, _pooled = batch
            mask = None

        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]

        # One step-begin marker per training step (not per forward).
        # ``compile_blocks(mode="default")`` doesn't enable cudagraphs, so this
        # is semantically a no-op today, but it's the right cadence if/when
        # the script switches to ``mode="reduce-overhead"``.
        torch.compiler.cudagraph_mark_step_begin()

        # ============ student update ============
        # No single t / x_t: the student rolls an N-step Euler grid from pure noise
        # ε (dpdmd anchors step 1 to a teacher K-step target then refines; dmd is
        # plain DMD2). See dpdmd.md §3.2.
        eps = torch.randn_like(latents)  # shared start for anchor + student
        c_null = uncond_for_batch(uncond_base, crossattn_emb)  # anchor + DMD eval

        # --- teacher K-step CFG anchor (no grad) → v_target (DP-DMD only) ---
        v_target = None
        if use_anchor:
            z = eps
            for i in range(cfg.k_anchor):
                s_i = teacher_anchor_sigmas[i]
                s_next = teacher_anchor_sigmas[i + 1]
                t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                v = _teacher_cfg_velocity(z, t_b, crossattn_emb, c_null)
                z = (z.float() - (s_i - s_next) * v).to(dtype)
            # Average velocity ε→z_tk over [t_k, 1]; this is exactly the target
            # for the student's t=1 first step (Euler x_next = x − dt·v_first).
            v_target = ((eps.float() - z.float()) / (1.0 - t_k_anchor)).detach()

        # --- student rollout → x_pred (= x_θ, B,16,H,W) + v_student (metric) ---
        # dpdmd: step-0 diversity anchor + DMD-refined steps 1..N-1.
        # dmd:   plain DMD2; cfg.dmd_grad_step picks which step(s) grad.
        split_bwd = use_anchor and cfg.detach_after_first
        last_step = cfg.student_steps - 1

        if use_anchor:
            # Step 0 is the diversity anchor (supervised toward v_target, then
            # detached under split_bwd); steps 1..N-1 carry the DMD-refine grad,
            # routed by grad_step ('all' BPTT | 'last' tail-only | 'random' grid).
            x = eps
            x.requires_grad_()  # grad-ckpt needs a grad-requiring forward input
            s0, s0_next = student_sigmas[0], student_sigmas[1]
            t_b = torch.full((B,), s0, device=device, dtype=dtype)
            turbo.set_student_step(0)  # head 0 (no-op unless per-step-expert)
            v_first = _forward("student", x, t_b, crossattn_emb, no_grad=False).squeeze(
                2
            )
            x = x - (s0 - s0_next) * v_first
            div_loss_t = nn.functional.mse_loss(v_first.float(), v_target)
            if split_bwd:
                # Load-bearing stop-grad: the DMD reverse-KL from steps 1..N-1 must
                # NOT flow into the diversity mapping (their Fig 5). Backward the
                # diversity term now, then re-leaf for a fresh DMD-chain root.
                (cfg.div_weight * div_loss_t).backward()
                x = x.detach().requires_grad_()
            if cfg.dmd_grad_step == "random":
                # Memory-flat anchored DMD: sample ONE refinement step g~U{1..N-1},
                # backward-simulate the 1..g-1 prefix under no_grad from the
                # post-anchor latent, then grad only step g on its one-step
                # x0-prediction (x_g − σ_g·v_g; the true endpoint would need BPTT
                # through g+1..N-1). Unlike 'last' this supervises every refinement
                # grid point over training and, under per_step_expert, trains head g
                # (every head over time) instead of only head N-1. Step 0's diversity
                # graph rides v_first and is untouched by the detach below, so
                # div_loss_t still backprops correctly under either split_bwd.
                g = int(torch.randint(1, cfg.student_steps, (1,)).item())
                for i in range(1, g):  # backward simulation (no graph kept)
                    s_i = student_sigmas[i]
                    s_next = student_sigmas[i + 1]
                    t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                    turbo.set_student_step(i)
                    v = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=True
                    ).squeeze(2)
                    x = x - (s_i - s_next) * v
                x = x.detach().requires_grad_()  # fresh leaf; head g trains
                s_g = student_sigmas[g]
                t_b = torch.full((B,), s_g, device=device, dtype=dtype)
                turbo.set_student_step(g)
                v_g = _forward("student", x, t_b, crossattn_emb, no_grad=False).squeeze(
                    2
                )
                x_pred = x - s_g * v_g  # one-step x0-prediction at step g
            else:
                # 'all' → full BPTT over 1..N-1; else ('last') → only the final step
                # grads (1..N-2 backward-simulated under no_grad). Both memory-flat
                # except 'all', and land the DMD grad on the true rollout endpoint.
                grad_dmd_last_only = cfg.dmd_grad_step != "all"
                for i in range(1, cfg.student_steps):
                    s_i = student_sigmas[i]
                    s_next = student_sigmas[i + 1]
                    t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                    turbo.set_student_step(i)
                    step_no_grad = grad_dmd_last_only and i != last_step
                    if grad_dmd_last_only and i == last_step:
                        x = (
                            x.detach().requires_grad_()
                        )  # fresh leaf after no_grad prefix
                    v = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=step_no_grad
                    ).squeeze(2)
                    x = x - (s_i - s_next) * v
                    if step_no_grad:
                        x = x.detach()
                x_pred = x
            v_student = v_first  # step-0 velocity for the runaway-student metric
        else:
            # Plain DMD2. Non-grad steps are backward-SIMULATED under no_grad (the
            # generator trains on its OWN trajectory — DMD2's train/inference input
            # match, Yin et al. 2024 — not forward-noised real latents).
            div_loss_t = torch.zeros((), device=device)  # uniform metrics path
            if cfg.dmd_grad_step == "all":
                # Full-rollout BPTT: every step grads into the endpoint x_pred.
                x = eps
                x.requires_grad_()
                v_student = None
                for i in range(cfg.student_steps):
                    s_i = student_sigmas[i]
                    s_next = student_sigmas[i + 1]
                    t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                    turbo.set_student_step(i)
                    v = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=False
                    ).squeeze(2)
                    if v_student is None:
                        v_student = v
                    x = x - (s_i - s_next) * v
                x_pred = x
            else:
                # Single grad-step: 'last' pins g=N-1; 'random' samples g~U{0..N-1}
                # (canonical DMD2 — supervises every grid point, not just the clean
                # tail). Roll to g under no_grad, grad ONLY step g, supervise its
                # one-step x0-prediction x_g − σ_g·v_g. Memory-flat (1 forward graph).
                if cfg.dmd_grad_step == "random":
                    # CPU RNG → no per-step GPU sync (seeded by torch.manual_seed).
                    g = int(torch.randint(0, cfg.student_steps, (1,)).item())
                else:
                    g = last_step
                x = eps
                for i in range(g):  # backward simulation (no_grad → no graph kept)
                    s_i = student_sigmas[i]
                    s_next = student_sigmas[i + 1]
                    t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                    turbo.set_student_step(i)
                    v = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=True
                    ).squeeze(2)
                    x = x - (s_i - s_next) * v
                x = x.detach().requires_grad_()  # fresh leaf; head g trains
                s_g = student_sigmas[g]
                t_b = torch.full((B,), s_g, device=device, dtype=dtype)
                turbo.set_student_step(g)
                v_g = _forward("student", x, t_b, crossattn_emb, no_grad=False).squeeze(
                    2
                )
                x_pred = x - s_g * v_g  # one-step x0-prediction at step g
                v_student = v_g

        # --- DMD on x_θ (steps 2..N), against teacher + fake ---
        # The real score is CFG-GUIDED (v_u + α·(v_c − v_u)), NOT cond-only.
        # Guidance rides the single DMD real score — exactly like the reference
        # `compute_dmd_loss` (dpdmd/train_sd35_dpdmd.py:118-129, teacher cat
        # cond+uncond → v_u + scale·(v_c−v_u)). Without it v_real≈v_fake (both
        # unguided cond preds collapse, dm_cos≈0.9999) and the quality gradient
        # is noise. The fake stays cond-only, matching the reference.
        tau_dm = torch.rand(B, device=device, dtype=dtype)
        eps_dm = torch.randn_like(x_pred)
        x_renoised_dm = renoise(x_pred.detach(), tau_dm, eps_dm)
        v_real_cond_dm = _teacher_cfg_velocity(
            x_renoised_dm, tau_dm, crossattn_emb, c_null
        )
        v_fake_cond_dm = _forward(
            "fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
        ).squeeze(2)
        delta_dm = v_real_cond_dm - v_fake_cond_dm

        tau_dm_e = tau_dm.view(B, 1, 1, 1).float()
        grad_dm = tau_dm_e * delta_dm.float()
        if cfg.dm_x0_norm:
            denom = (
                (tau_dm_e * v_real_cond_dm.float())
                .abs()
                .mean(dim=(1, 2, 3), keepdim=True)
                .clamp_min(cfg.norm_floor)
            )
            grad_dm = grad_dm / denom
        grad_signal = grad_dm.detach()

        # --- REPA relational alignment on REAL data (turbo_repa.md Phase 1) ---
        # One extra grad-bearing PARTIAL student forward (the feature tap exits
        # after repa_layer ≈ 9/28 of the stack) on renoised real latents, every
        # repa_every_n-th step, backwarded IMMEDIATELY (the div_loss split-bwd
        # pattern; grads accumulate into the same optimizer step, grad_clip
        # below still sees the full student gradient).
        #
        # PLACEMENT + IMMEDIATE BACKWARD ARE LOAD-BEARING. set_view flips
        # global per-module enabled flags read at forward time, and checkpoint
        # recompute is deferred to .backward() — so every checkpointed forward
        # must be backwarded while its own view is still the live one. Running
        # this block after the GAN gen forward and riding loss_student.backward()
        # (the first wiring) left the view at "student" when the GAN's teacher
        # checkpoint recomputed → student-contaminated teacher features →
        # silently corrupted GAN generator grads for a whole run. Backward here
        # (student view live ✓), then the GAN gen forward below is again the
        # last view-switch before loss_student.backward() (teacher recompute ✓).
        repa_loss = torch.zeros((), device=device)
        repa_ran = False
        if repa_on and step % cfg.repa_every_n == 0:
            if repa_pe is None:
                if not repa_warned_missing:
                    logger.warning(
                        "REPA: batch lacks a %s sidecar — term skipped for such "
                        "batches (run `make preprocess-pe` for the spatial "
                        "encoder to cover the pool).",
                        cfg.repa_encoder,
                    )
                    repa_warned_missing = True
            else:
                # τ from the DMD path's renoise distribution (U[0,1)), but ONE
                # level shared across the batch: per-step-expert head selection
                # is a per-forward switch, not per-sample. CPU RNG → no GPU
                # sync (mirrors the grad_step='random' g draw).
                tau_val = float(torch.rand(()))
                turbo.set_student_step(
                    nearest_student_step(student_sigmas, tau_val, cfg.student_steps)
                )
                tau_repa = torch.full((B,), tau_val, device=device, dtype=dtype)
                x_tau = renoise(latents, tau_repa, torch.randn_like(latents))
                # The grad-requiring input is load-bearing twice over: the
                # unsloth checkpoint silently drops grads that reach only
                # closed-over params ([[project_unsloth_reentrant_drops_grad]]),
                # and a grad-requiring tensor input resurrects them (the
                # EasyControl-under-unsloth precedent) — the student-LoRA grads
                # here have no other input path.
                x_tau.requires_grad_()
                # Checkpoint this forward like the GAN gen forward below: it is
                # otherwise the only grad-bearing forward in the loop that
                # retains its activations (the rollout is memory-flat under
                # grad_step='random', the GAN tap is ckpt'd) — ~repa_layer/28 of
                # a full-depth graph held until backward, which alone tipped a
                # 16 GB card over. Recompute cost: repa_layer eager blocks every
                # every_n-th step. Numerically exact (no dropout).
                with selective_block_grad_ckpt(model):
                    feats_repa = _forward(
                        "student",
                        x_tau,
                        tau_repa,
                        crossattn_emb,
                        no_grad=False,
                        return_block_features=repa_feature_set,
                        return_features_early=True,
                    )
                repa_loss = relational_align_loss(
                    feats_repa[cfg.repa_layer],
                    repa_pe.to(device, dtype=torch.float32, non_blocking=True),
                    (int(latents.shape[-2]), int(latents.shape[-1])),
                    int(model.patch_spatial),
                    repa_spec,
                    spatial_norm=cfg.repa_spatial_norm,
                    dog=cfg.repa_target_dog,
                    dog_sigma1_div=cfg.repa_dog_sigma1_div,
                    dog_sigma2_div=cfg.repa_dog_sigma2_div,
                    dog_norm_std=cfg.repa_dog_norm_std,
                )
                (cfg.repa_weight * repa_loss).backward()
                repa_loss = repa_loss.detach()  # metrics-only from here
                repa_ran = True

        # --- GAN generator term + f-distill reweighting (ideas 1 & 2) ---
        # The discriminator scores the frozen TEACHER's block features of the
        # student's (renoised) x_pred. The generator GAN loss must flow grad back
        # into x_pred → student, so this renoise keeps x_pred attached (unlike the
        # DMD path, which detaches) and the teacher forward runs grad-enabled. The
        # disc is frozen here (set_disc_requires_grad(False)) — grad reaches x_pred
        # through the differentiable teacher forward, not the disc weights. This is
        # +1 teacher forward/step vs DP-DMD, but `return_features_early` (idea 3.1)
        # stops it after the deepest tapped block, so only blocks[0..k] run and
        # retain activations for backward — the half-depth grad forward that keeps
        # this term inside the memory budget (was the OOM with the full-stack tap).
        gan_gen_loss = torch.zeros((), device=device)
        if turbo.disc is not None:
            turbo.set_disc_requires_grad(False)
            x_renoised_gan = renoise(x_pred, tau_dm, eps_dm)  # grad-bearing
            # Selectively checkpoint just this forward (the only GAN extra that
            # retains a backward graph); recompute trades ~half-depth compute for
            # the ~3 GB of retained teacher activations. nullcontext when off.
            gan_ckpt = (
                selective_block_grad_ckpt(model) if cfg.gan_grad_ckpt else nullcontext()
            )
            with gan_ckpt:
                feats_gen = _forward(
                    "teacher",
                    x_renoised_gan,
                    tau_dm,
                    crossattn_emb,
                    no_grad=False,
                    return_block_features=turbo.gan_feature_set,
                    return_features_early=True,
                )
            fake_logits_gen = turbo.disc(
                turbo.features_in_order(feats_gen)
            )  # (B, taps), grad→x_pred
            gan_gen_loss = gan_loss_generator(fake_logits_gen)

            # f-distill: scale the (detached) DMD signal by h(τ, r), r=exp(logits).
            if fdistill_on:
                h, fdistill_bins = f_div_weighting_h(
                    fake_logits_gen,
                    tau_dm,
                    f_div=cfg.f_div,
                    ratio_lower=cfg.f_ratio_lower,
                    ratio_upper=cfg.f_ratio_upper,
                    ema_rate=cfg.f_ratio_ema_rate,
                    bins=fdistill_bins,
                    bin_num=cfg.f_bin_num,
                )
                grad_signal = grad_signal * h.view(B, 1, 1, 1)

        # --- assemble: DMD surrogate on x_θ (+ optional mean-var) ---
        # The diversity term was already backwarded above when split_bwd; otherwise
        # it rides this combined backward (graphs still entangled). grad_clip below
        # runs once on the ACCUMULATED .grad (div + DMD), so the clipped norm is the
        # full student gradient either way.
        if mask is not None:
            loss_dmd = (grad_signal * x_pred.float() * mask).mean()
        else:
            loss_dmd = (grad_signal * x_pred.float()).mean()
        loss_student = loss_dmd

        mv_loss = torch.zeros((), device=device)
        if cfg.mean_var_weight > 0.0:
            mv_loss = mean_var_kl(x_pred.float(), mv_tgt_mu, mv_tgt_var)
            loss_student = loss_student + cfg.mean_var_weight * mv_loss

        if use_anchor and not split_bwd:
            loss_student = loss_student + cfg.div_weight * div_loss_t

        if turbo.disc is not None:
            loss_student = loss_student + cfg.gan_loss_weight_gen * gan_gen_loss

        loss_student.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                turbo.student_params(), max_norm=cfg.grad_clip
            )
        student_opt.step()
        student_opt.zero_grad(set_to_none=True)
        student_sched.step()

        # --- 5. FAKE UPDATE ---
        # Run ``fake_steps_per_student_step`` inner updates against the same
        # x_pred.detach(), resampling (τ_fake, ε_fake) each iteration so each
        # inner step sees a different rung of the flow-matching forward path.
        # Standard DMD2 practice: keep the fake's regression target ahead of
        # the student's moving x_pred distribution.
        x_pred_d = x_pred.detach()
        fake_loss_sum = torch.zeros((), device=device)
        gan_disc_sum = torch.zeros((), device=device)
        for _ in range(cfg.fake_steps_per_student_step):
            tau_fake = sample_t(
                B,
                distribution=cfg.t_distribution,
                sigmoid_scale=cfg.sigmoid_scale,
                device=device,
                dtype=dtype,
            )
            eps_fake = torch.randn_like(x_pred_d)
            x_t_fake = renoise(x_pred_d, tau_fake, eps_fake).requires_grad_()
            v_fake = _forward(
                "fake", x_t_fake, tau_fake, crossattn_emb, no_grad=False
            ).squeeze(2)
            target_v_fake = eps_fake - x_pred_d  # flow-matching target
            fake_loss = nn.functional.mse_loss(v_fake.float(), target_v_fake.float())
            fake_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    turbo.fake_params(), max_norm=cfg.grad_clip
                )
            fake_opt.step()
            fake_opt.zero_grad(set_to_none=True)
            fake_sched.step()
            fake_loss_sum = fake_loss_sum + fake_loss.detach()

            # ============ discriminator update (idea 1) ============
            # Co-located with the fake/critic update (FastGen ties the disc to the
            # fake_score cadence). The disc scores frozen-TEACHER block features of
            # renoised fake (x_pred) vs renoised real latents — no grad to the
            # teacher (no_grad), grad only to the disc head. gan_use_same_t_noise
            # reuses (τ_fake, ε_fake) for the real branch (FastGen default).
            if turbo.disc is not None:
                turbo.set_disc_requires_grad(True)
                if cfg.gan_use_same_t_noise:
                    tau_d, eps_d = tau_fake, eps_fake
                else:
                    tau_d = sample_t(
                        B,
                        distribution=cfg.t_distribution,
                        sigmoid_scale=cfg.sigmoid_scale,
                        device=device,
                        dtype=dtype,
                    )
                    eps_d = torch.randn_like(x_pred_d)

                # Feature-only teacher forwards (no_grad → grad only to the disc
                # head). Early-exit at the deepest tap; each call returns its own
                # feature dict, so the fake/real captures never alias.
                def _disc_feats(latent_in):
                    return turbo.features_in_order(
                        _forward(
                            "teacher",
                            renoise(latent_in, tau_d, eps_d),
                            tau_d,
                            crossattn_emb,
                            no_grad=True,
                            return_block_features=turbo.gan_feature_set,
                            return_features_early=True,
                        )
                    )

                fake_logits_d = turbo.disc(_disc_feats(x_pred_d))
                real_logits_d = turbo.disc(_disc_feats(latents))
                loss_disc = gan_loss_discriminator(real_logits_d, fake_logits_d)

                # Approximate-R1 (APT): penalize disc logit change under a small
                # perturbation of the real disc input. Perturb the renoised real
                # latent directly (the tensor whose features feed the disc).
                if cfg.gan_r1_weight > 0.0:
                    x_t_real_a = renoise(
                        latents, tau_d, eps_d
                    ) + cfg.gan_r1_alpha * torch.randn_like(latents)
                    feats_a = _forward(
                        "teacher",
                        x_t_real_a,
                        tau_d,
                        crossattn_emb,
                        no_grad=True,
                        return_block_features=turbo.gan_feature_set,
                        return_features_early=True,
                    )
                    real_logits_a = turbo.disc(turbo.features_in_order(feats_a))
                    loss_disc = loss_disc + cfg.gan_r1_weight * nn.functional.mse_loss(
                        real_logits_d, real_logits_a
                    )

                loss_disc.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        turbo.disc_params(), max_norm=cfg.grad_clip
                    )
                disc_opt.step()
                disc_opt.zero_grad(set_to_none=True)
                disc_sched.step()
                turbo.set_disc_requires_grad(False)
                gan_disc_sum = gan_disc_sum + loss_disc.detach()
        fake_loss_mean_t = fake_loss_sum / cfg.fake_steps_per_student_step
        gan_disc_mean_t = gan_disc_sum / cfg.fake_steps_per_student_step

        # --- logging accumulators (all GPU-side; flushed below every log_interval
        # in one stacked .tolist() so per-step CUDA syncs go to zero) ---
        metrics.accumulate_per_step(
            fake_loss_mean_t=fake_loss_mean_t,
            grad_signal=grad_signal,
            delta_dm=delta_dm,
            x_pred=x_pred,
            v_student=v_student,
            tau_dm_e=tau_dm_e,
            v_real_cond_dm=v_real_cond_dm,
            v_fake_cond_dm=v_fake_cond_dm,
            mv_loss=mv_loss,
        )
        metrics.add_div(div_loss_t)
        if turbo.disc is not None:
            metrics.add_gan(gan_gen_loss, gan_disc_mean_t)
        if repa_on:
            metrics.add_repa(repa_loss, active=repa_ran)

        if (step + 1) % cfg.log_interval == 0:
            m = metrics.flush(cfg.log_interval)
            if writer is not None:
                write_scalars(writer, m, step + 1)
                writer.add_scalar(
                    "train/student_lr", student_sched.get_last_lr()[0], step + 1
                )
                writer.add_scalar(
                    "train/fake_lr", fake_sched.get_last_lr()[0], step + 1
                )
                if disc_sched is not None:
                    writer.add_scalar(
                        "train/disc_lr", disc_sched.get_last_lr()[0], step + 1
                    )
            # tqdm postfix at log_interval cadence (per-step would re-introduce
            # the syncs we just eliminated). First log_interval steps show no
            # postfix — harmless.
            progress.set_postfix(**tqdm_postfix(m))
            metrics.reset()

        # --- diversity validation (DAVE same-prompt probe) ---
        if val_cond is not None and (step + 1) % cfg.validate_every_n_steps == 0:
            dm = run_diversity_validation(
                model=model,
                forward_fn=_forward,
                set_student_step=turbo.set_student_step,
                student_sigmas=student_sigmas,
                crossattn_emb=val_cond,
                latent_shape=val_latent_shape,
                num_seeds=cfg.val_diversity_seeds,
                seed0=cfg.seed,
                device=device,
                dtype=dtype,
                clean_latent=val_clean,
            )
            if writer is not None:
                writer.add_scalar("val/div_ac_sim", dm.ac_sim, step + 1)
                writer.add_scalar("val/div_dc_sim", dm.dc_sim, step + 1)
                writer.add_scalar("val/div_gap", dm.gap, step + 1)
                writer.add_scalar("val/div_xpred_ac_sim", dm.xpred_ac_sim, step + 1)
                writer.add_scalar("val/fm_mse", dm.fm_mse, step + 1)
            logger.info(
                f"[val@{step + 1}] diversity: AC sim={dm.ac_sim:.4f} "
                f"(lower=more diverse) | DC sim={dm.dc_sim:.4f} | gap={dm.gap:+.4f} "
                f"| x_pred AC sim={dm.xpred_ac_sim:.4f} | FM MSE={dm.fm_mse:.4f} "
                f"(fidelity; not a quality score)"
            )

        # --- save ---
        # Every save_every checkpoint is kept under a step-tagged name (no
        # overwrite, so the whole training trajectory survives for eyeballing);
        # the final step also writes the canonical bare `{output_name}` that
        # inference / merge / `make test` look for.
        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
            n = step + 1
            is_final = n == cfg.iterations
            metadata = {
                "ss_turbo_objective": cfg.base_loss,
                "ss_turbo_student_rank": str(cfg.student_rank),
                "ss_turbo_student_alpha": str(cfg.student_alpha),
                "ss_turbo_student_steps": str(cfg.student_steps),
                "ss_turbo_teacher_cfg": str(cfg.teacher_cfg),
                "ss_turbo_step": str(n),
                "ss_turbo_k_anchor": str(cfg.k_anchor),
                "ss_turbo_div_weight": str(cfg.div_weight),
                "ss_turbo_gan_weight_gen": str(cfg.gan_loss_weight_gen),
                "ss_turbo_f_div": cfg.f_div,
            }
            if cfg.per_step_expert:
                # Drives loader detection (CLI + ComfyUI build StepExpertLoRAModule
                # and keep it live instead of merging). step_expert_K == the head
                # count == student_steps.
                metadata["ss_turbo_per_step_expert"] = "1"
                metadata["ss_turbo_step_expert_K"] = str(cfg.step_expert_K)
            save_names = [f"{cfg.output_name}_{_step_tag(n)}"]
            if is_final:
                save_names.append(cfg.output_name)  # canonical bare name
            for name in save_names:
                save_path = str(Path(cfg.output_dir) / f"{name}.safetensors")
                turbo.save_student(save_path, dtype=torch.bfloat16, metadata=metadata)
                logger.info(f"saved checkpoint: {save_path}")

    if writer is not None:
        writer.close()
    logger.info("turbo distillation complete.")


if __name__ == "__main__":
    main()
