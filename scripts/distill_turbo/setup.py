"""Turbo distillation run construction — everything built before the step loop.

``build_run`` builds the DiT + adapter stack, warm-start/crash-resume, the
three optimizer/scheduler pairs, the dataloader, and the per-forward closures,
returning a :class:`RunContext` that ``distill.run_loop`` reads/mutates.

Construction order is load-bearing (block-swap → grad-ckpt → apply_to →
compile; compile-limit pinned before the first trace; resume applied only once
every mutable object exists) — see inline comments for the individual gotchas.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from library.anima import weights as anima_utils
from library.anima.models import Anima
from library.anima.stream import latent_dtype, stream_fp32_
from library.anima.uncond import (
    default_uncond_path,
    load_uncond_crossattn,
)
from library.datasets.cache import CachedDataset
from library.inference.sampling import get_timesteps_sigmas
from library.int8.linear import int8_input_tag
from library.runtime.dynamo import (
    compile_signature,
    isolate_compile_cache,
    set_activation_memory_budget,
)
from library.runtime.dynamo import pin_dynamo_limit as _pin_dynamo_limit
from library.runtime.harness import (
    _apply_partitioner_tuning,
    compile_dit_blocks,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from library.training.distill_runtime import (
    apply_single_prompt_slice,
    create_tb_writer,
    ensure_dynamic_seq_for_freefit,
    resolve_device_dtype,
    write_config_snapshot,
)
from library.runtime.progress import ProgressSink
from networks.methods.turbo_dmd import TurboDMDNetwork, warm_start_plain_lora

from .config import TurboConfig, snapshot_toml_text, tb_config_text
from .metrics import TauBinCriticLoss, TurboMetrics
from .primitives import PadCache, make_collate, make_scheduler
from .resume import (
    apply_resume_state,
    load_resume_state,
    resolve_resume_arg,
    resume_path_for,
)
from .warmup import run_fake_warmup

logger = logging.getLogger(__name__)

# `{stem}_{W:04d}x{H:04d}_anima.npz` → pixel (W, H); token count = (W//16)*(H//16).
_LATENT_RES_RE = re.compile(r"_(\d{3,4})x(\d{3,4})_anima\.npz$")


def _cached_token_counts(data_dir: str) -> set:
    """Distinct token counts present in the cached latents under ``data_dir``.

    Mirrors ``library.anima.trainer.data.derive_token_budget``: caches are the source of truth
    for which tiers are present, not ``target_res``. Parses (W, H) out of each
    latent's filename (no per-file I/O).
    """
    from library.io.cache import discover_cached_pairs

    counts: set = set()
    for img in discover_cached_pairs(data_dir):
        m = _LATENT_RES_RE.search(os.path.basename(img.npz_path))
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            counts.add((w // 16) * (h // 16))
    return counts


@dataclass
class RunContext:
    """Everything ``distill.run_loop`` reads or mutates, built by ``build_run``.

    ``data_iter`` is mutated across the loop (re-``iter``'d at epoch boundaries).
    """

    model: Anima
    turbo: TurboDMDNetwork
    device: torch.device
    dtype: torch.dtype  # sampler state: latents, noise, σ
    cond_dtype: torch.dtype  # text embeddings

    # Optimizers / schedulers (disc pair is None when the GAN is off).
    student_opt: torch.optim.Optimizer
    fake_opt: torch.optim.Optimizer
    disc_opt: torch.optim.Optimizer | None
    student_sched: object
    fake_sched: object
    disc_sched: object | None

    # Data.
    dataloader: torch.utils.data.DataLoader
    data_iter: object  # mutated: re-iter at epoch boundaries

    # Per-forward closures (view switch + block-swap + autocast; CFG teacher).
    forward: Callable[..., torch.Tensor]
    teacher_cfg_velocity: Callable[..., torch.Tensor]

    # Static schedule constants.
    student_sigmas: list[float]
    teacher_anchor_sigmas: list[float]
    t_k_anchor: float

    # Conditioning / aux-loss state.
    uncond_base: torch.Tensor
    cdm_on: bool

    # Logging / progress.
    writer: object | None
    progress_sink: ProgressSink | None
    console_steps: bool
    metrics: TurboMetrics
    tau_profile: TauBinCriticLoss

    # Diversity validation (None when validate_every_n_steps <= 0 / empty set).
    val_cond: torch.Tensor | None
    val_latent_shape: tuple | None
    val_clean: torch.Tensor | None

    start_step: int
    int8_base_id: str = ""


def build_run(args, cfg: TurboConfig) -> RunContext:
    """Construct the full DP-DMD run state and return it as a ``RunContext``.

    Step order is load-bearing (see module docstring).
    """
    torch.manual_seed(cfg.seed)
    device, dtype = resolve_device_dtype()

    # Compile-storm guard: pin recompile budget + intra-op thread count BEFORE any
    # block._forward traces (many global states/step spill the default limit=8 to
    # eager). _pin_dynamo_limit is CONTEXT-PINNED so it survives into backward —
    # a plain config.recompile_limit assignment reverts at the first grad forward.
    # set_num_threads must be forced now: a later flip is a GLOBAL_STATE guard
    # (recompiles everything) and a CheckpointError trigger under GAN grad-ckpt
    # (see selective_block_grad_ckpt). The later model-aware raise only refines
    # accumulated_recompile_limit; both writes are idempotent via max().
    if cfg.torch_compile:
        _pin_dynamo_limit("recompile_limit", cfg.dynamo_recompile_limit)
    torch.set_num_threads(torch.get_num_threads())

    logger.info(f"loading DiT: {cfg.dit_path}")
    model: Anima = anima_utils.load_anima_model(
        device,
        cfg.dit_path,
        attn_mode=cfg.attn_mode,
        loading_device="cpu" if cfg.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
        dit_nonquant_path=cfg.dit_nonquant_path,
    )

    # Block swap; compile_dit_blocks deferred until after apply_to (COMPILE LAST
    # note below) — order: block-swap → grad-ckpt → apply_to → compile.
    place_dit_for_training(model, device, blocks_to_swap=cfg.blocks_to_swap)
    enable_training_grad_ckpt(model, enabled=cfg.grad_ckpt)

    # GAN feature tap: resolve the tapped block (−1 → middle) and hand the index
    # set to TurboDMDNetwork to build the disc + block hooks. gan_indices=None
    # when weight_gen==0.
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

    int8_base_id = ""

    def convert_base() -> None:
        nonlocal int8_base_id
        if cfg.int8 != "off":
            from library.anima.int8 import convert_for_turbo

            int8_base_id = convert_for_turbo(
                model, cfg.int8, cfg.int8_file, cfg.dit_path
            )
        if cfg.stream == "fp32":
            stream_fp32_(model)
        # the conversion placed the tail's new tensors on the device of each weight
        model.prepare_block_swap_before_forward(free_cache=False)

    turbo = TurboDMDNetwork(
        unet=model,
        student_rank=cfg.student_rank,
        fake_rank=cfg.fake_rank,
        student_alpha=cfg.student_alpha,
        fake_alpha=cfg.fake_alpha,
        use_custom_down_autograd=cfg.use_custom_down_autograd,
        channel_scaling_alpha=cfg.channel_scaling_alpha,
        student_down_init=cfg.student_down_init,
        fake_down_init=cfg.fake_down_init,
        train_adaln=cfg.train_adaln,
        fake_adaln=cfg.fake_adaln,
        adaln_rank=cfg.adaln_rank,
        adaln_alpha=cfg.adaln_alpha,
        gan_feature_indices=gan_indices,
        gan_disc_hidden=cfg.gan_disc_hidden if cfg.gan_disc_hidden > 0 else None,
        gan_disc_head=cfg.gan_disc_head,
        fake_transition=cfg.fake_transition,
        before_apply=convert_base,
    )
    turbo.freeze_dit()
    state_dtype = latent_dtype(model, dtype)
    turbo.student.to(device=device, dtype=dtype)
    turbo.fake.to(device=device, dtype=dtype)
    if turbo.fake_transition is not None:
        turbo.fake_transition.to(device=device)  # fp32

    # Crash-resume supersedes warm start entirely (seeding then overwriting would
    # waste an SVD). Resolved here so a bad --resume fails fast; applied later
    # once the optimizers/schedulers exist.
    resume_state = None
    if cfg.resume:
        arg = resolve_resume_arg(cfg.resume, cfg.output_dir, cfg.output_name)
        if arg.path is None:
            logger.info(
                "--resume auto: no bundle at "
                f"{resume_path_for(cfg.output_dir, cfg.output_name)} — starting fresh."
            )
        else:
            resume_state = load_resume_state(arg.path)
            logger.info(
                f"resuming from {arg.path} @ step {resume_state['step']} "
                f"(student + fake + disc + optimizers + LR schedule restored)"
            )

    # Warm start (network.student_init_weights / fake_init_weights): seed the
    # stack's ΔW from a plain LoRA file (e.g. an official-release delta from
    # scripts/toolkits/extract_delta_lora.py). After .to() so the SVD runs on-device,
    # before compile so traced forwards see the final tensors.
    if resume_state is not None:
        if cfg.student_init_weights or cfg.fake_init_weights:
            logger.info(
                "resume: skipping warm start — the bundle's weights supersede it."
            )
    else:
        if cfg.student_init_weights:
            warm_start_plain_lora(turbo.student, cfg.student_init_weights, "student")
        if cfg.fake_init_weights:
            warm_start_plain_lora(turbo.fake, cfg.fake_init_weights, "fake")
    # Disc stays fp32 (LayerNorm/Linear) for GAN-loss stability — its forward
    # casts the bf16 teacher features to float.
    if turbo.disc is not None:
        turbo.disc.to(device=device)
    # COMPILE LAST: apply_to above monkey-patches Linears, so compile must trace
    # the adapter forward, not the bare DiT (harness ordering invariant).
    # dynamic_seq collapses per-token-count graphs to one symbolic-seq graph,
    # sized from the token families actually in the cached pool (self-describing,
    # like train.py); explicit cfg.target_res still overrides.
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
    # Free-fit fail-safe (train.py's auto-enable never reaches this bespoke loop):
    # a free-fit pool spans many token counts, which would explode the static
    # compile cascade — detect off the cache and force dynamic_seq (cfg is
    # frozen → local override).
    dynamic_seq = cfg.compile_dynamic_seq
    if cfg.torch_compile and not dynamic_seq:
        dynamic_seq = ensure_dynamic_seq_for_freefit(
            _cached_token_counts(cfg.data_dir), dynamic_seq, logger=logger
        )
    # Partitioner saved-activation cap (mirrors train.py). Must be set BEFORE
    # compile_dit_blocks (partitioning happens at first-forward compile). Skipped
    # under grad_ckpt: repartitioning can pick a different backward graph than
    # forward → CheckpointError (torch #166926).
    if cfg.torch_compile and cfg.activation_memory_budget < 1.0 and not cfg.grad_ckpt:
        set_activation_memory_budget(cfg.activation_memory_budget)
        logger.info(
            f"torch.compile activation_memory_budget = {cfg.activation_memory_budget} "
            "(partitioner recomputes cheap intermediates in backward)"
        )
    elif cfg.activation_memory_budget < 1.0 and cfg.grad_ckpt:
        logger.info(
            "activation_memory_budget ignored: incompatible with grad_ckpt "
            "(and redundant under it)"
        )
    # Partitioner default-partition tuning (mirrors train.py's partitioner_*
    # args; the helper owns the grad_ckpt gate + logging). Applied before
    # compile_dit_blocks for the same reason as the budget.
    if cfg.torch_compile:
        _apply_partitioner_tuning(
            recompute_views=cfg.partitioner_recompute_views,
            aggressive_recomputation=cfg.partitioner_aggressive_recomputation,
            grad_ckpt=cfg.grad_ckpt,
            logger=logger,
        )
    # Isolate the compile cache per signature: a stale narrow seq-range guard
    # from a different run poisons this one's wider dynamic-seq mark, dying with
    # a ConstraintViolationError on the first ≥4032-token trace. Same signature
    # still warm-reuses, shared with train.py.
    if cfg.torch_compile:
        isolate_compile_cache(
            compile_signature(
                n_token_families=n_token_families,
                seq_range=seq_range,
                dynamic_seq=dynamic_seq,
                mode="",
            )
            + (
                ""
                if cfg.int8 == "off"
                else f";int8={cfg.int8};int8_in={int8_input_tag()};flops=fa2,int8"
            )
            + ("" if cfg.stream == "bf16" else f";stream={cfg.stream}")
        )
    compile_dit_blocks(
        model,
        enabled=cfg.torch_compile,
        mode="",
        dynamic_seq=dynamic_seq,
        n_token_families=n_token_families,
        seq_range=seq_range,
    )
    # Refine the budget now the model exists: size accumulated_recompile_limit
    # over the exact block count. Idempotent via max() (never lowers the early
    # raise at the top of main()).
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

    student_opt = torch.optim.AdamW(
        turbo.student_params(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )
    student_sched = make_scheduler(student_opt, cfg.iterations, cfg.student_lr)
    # Fake scheduler spans main-loop updates (iterations · fake_steps_per_student_step)
    # PLUS fake_warmup_steps head-start updates, stepped through both phases — so
    # the ``0.02·total`` LR warmup overlaps the head-start (fake enters the main
    # loop already at full LR) and the cosine still lands at loop end. Student
    # schedule is independent: ``0.02·iterations``, no head-start offset.
    fake_opt = torch.optim.AdamW(
        turbo.fake_params(),
        lr=cfg.fake_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )
    fake_sched = make_scheduler(
        fake_opt,
        cfg.iterations * cfg.fake_steps_per_student_step + cfg.fake_warmup_steps,
        cfg.fake_lr,
    )

    # Disc steps once per fake inner step (FastGen ties it to the fake_score
    # cadence). No head-start, so its scheduler is sized over the main loop only.
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
            disc_opt,
            cfg.iterations * cfg.fake_steps_per_student_step,
            cfg.gan_disc_lr,
        )
        n_disc = sum(p.numel() for p in turbo.disc_params())
        logger.info(f"trainable: disc={n_disc:,}")

    # Apply resume now that every mutable object exists (nets on-device + compiled,
    # optimizers/schedulers built). start_step = student
    # steps already completed; 0 on a fresh run.
    start_step = 0
    if resume_state is not None:
        start_step = apply_resume_state(
            resume_state,
            cfg=cfg,
            turbo=turbo,
            student_opt=student_opt,
            fake_opt=fake_opt,
            disc_opt=disc_opt,
            student_sched=student_sched,
            fake_sched=fake_sched,
            disc_sched=disc_sched,
        )
        if start_step >= cfg.iterations:
            raise SystemExit(
                f"resume: bundle is at step {start_step} but iterations={cfg.iterations} "
                "— nothing left to run. Raise --iterations to extend the run."
            )
        logger.info(
            f"resume: continuing at step {start_step}/{cfg.iterations} "
            f"(student LR {student_sched.get_last_lr()[0]:.3g})"
        )
        # Free the CPU-side copy before training allocates: the bundle holds two
        # full AdamW moment sets and is the largest transient in the process.
        resume_state = None

    # L_CDM off-trajectory loss. weight=0 → fully off (no extra forwards/RNG draws).
    cdm_on = cfg.cdm_weight > 0.0

    dataset = CachedDataset(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        sample_ratio=cfg.sample_ratio,
        need_pooled=False,  # DP-DMD conditions on crossattn only
    )
    # Held-out conditioning for the DAVE diversity probe — captured from the FULL
    # sample list before any single-prompt slice mutates it, chosen distinct from
    # the overfit sample so a collapsed run is visible. Loaded once, reused.
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
        v_sample = dataset[v_idx]
        v_lat, v_ca = v_sample["latents"], v_sample["crossattn_emb"]
        val_cond = v_ca.unsqueeze(0).to(device, dtype=dtype)  # (1, seq, D)
        val_latent_shape = (1, *tuple(v_lat.shape))  # (1, C, H, W)
        val_clean = v_lat.unsqueeze(0).to(
            device, dtype=state_dtype
        )  # (1, C, H, W) for FM MSE
        logger.info(
            f"diversity validation: every {cfg.validate_every_n_steps} steps, "
            f"{cfg.val_diversity_seeds} seeds, held-out idx={v_idx} "
            f"(latent {tuple(v_lat.shape)})"
        )

    if cfg.single_prompt_idx is not None:
        # Single-prompt overfit — wrap as a 1-sample list so the dataloader cycles it.
        apply_single_prompt_slice(dataset, cfg.single_prompt_idx, logger=logger)

    # Bucket-grouped batch sampler: every batch is one
    # resolution — free-fit gives each image its own token count, so a
    # cross-resolution batch can't stack. Order shuffled per epoch, seeded by
    # cfg.seed (data-order axis of the training lottery); largest-token bucket
    # pinned first for compile warmup.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=dataset.make_batch_sampler(shuffle=True, seed=cfg.seed),
        num_workers=2,
        pin_memory=True,
        collate_fn=make_collate(),
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    snapshot_text = snapshot_toml_text(cfg, source_config=args.config)
    # Config snapshot beside the checkpoint (train.py convention), written even
    # under --no_log.
    write_config_snapshot(
        Path(cfg.output_dir) / f"{cfg.output_name}.snapshot.toml",
        snapshot_text,
        logger=logger,
    )

    writer, run_log = create_tb_writer(
        cfg.log_dir, tb_config_text(cfg), enabled=not cfg.no_log, logger=logger
    )
    if run_log is not None:
        write_config_snapshot(
            run_log / f"{cfg.output_name}.snapshot.toml",
            snapshot_text,
            logger=logger,
        )

    # train.py's ProgressSink, same path convention. Disabled under --no_log.
    progress_sink = None
    if not cfg.no_log:
        sink_path = ProgressSink.resolve_path(cfg)
        if sink_path is not None:
            progress_sink = ProgressSink(
                sink_path, run=cfg.output_name, method="turbo", preset=None
            )
            progress_sink.run_start(
                total_steps=cfg.iterations,
                total_epochs=1,
                pid=os.getpid(),
                log_dir=str(run_log) if run_log is not None else None,
            )
            progress_sink.attach_log_mirror()
            logger.info(f"progress sink: {sink_path}")

    pad_cache = PadCache(dtype)
    # CFG-uncond input must be the T5("") embedding (real BOS/EOS/sentinel tokens
    # nonzero, only padding zeroed). A fully-zero tensor is off-distribution: the
    # resulting uncond direction, amplified at (α-1)=3×, drives the student
    # off-manifold (saturated white). Staged by `make preprocess-te`.
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

        ``x`` is (B, 16, H, W); unsqueezed to (B, 16, 1, H, W) inside.

        With ``return_features_early`` (GAN feature tap) the forward stops after
        the deepest tapped block and returns ``{block_idx: feat}`` instead of a
        velocity — the caller must NOT ``.squeeze(2)`` the result.

        ``model.training`` stays True for the whole run (DiT frozen) so
        grad-ckpt stays armed via ``Block.forward``'s ``self.training`` gate;
        toggling it per forward was the dominant per-forward CPU stall.

        Two independent, numerically-exact checkpoint levers, both no-ops on
        no_grad forwards: global ``--grad_ckpt`` (default OFF) arms
        unsloth-offload ckpt on every grad-bearing forward; ``gan.grad_ckpt``
        (default on) wraps only the GAN gen forward via the same path (see
        ``selective_block_grad_ckpt``).
        """
        turbo.set_view(view)
        model.rewind_block_swap()
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

    # Static σ grids (token-count-invariant), built once. Both span σ: 1 → 0;
    # student has student_steps+1 points, teacher anchor teacher_anchor_steps+1.
    # sigmas[i] - sigmas[i+1] is the Euler dt for step i.
    student_sigmas = get_timesteps_sigmas(cfg.student_steps, cfg.flow_shift, "cpu")[
        1
    ].tolist()
    teacher_anchor_sigmas = get_timesteps_sigmas(
        cfg.teacher_anchor_steps, cfg.flow_shift, "cpu"
    )[1].tolist()
    # Continuous time at the anchor (incoming σ after k_anchor teacher steps).
    # `v_target = (ε − z_tk)/(1 − t_k)` — a σ mismatch here silently mis-scales
    # the diversity target, so it's read from the teacher grid, not the student.
    t_k_anchor = float(teacher_anchor_sigmas[cfg.k_anchor])
    logger.info(
        f"DP-DMD grids: student σ={['%.3f' % s for s in student_sigmas]}, "
        f"anchor t_k={t_k_anchor:.4f} (teacher step {cfg.k_anchor}/"
        f"{cfg.teacher_anchor_steps})"
    )

    def _teacher_cfg_velocity(x, t_b, c_cond, c_null):
        """CFG-guided teacher velocity ``v_u + α·(v_c − v_u)`` (no grad, fp32).

        Used by the DP-DMD K-step anchor rollout; skips the uncond forward when
        ``teacher_cfg == 1``.
        """
        v_c = _forward("teacher", x, t_b, c_cond, no_grad=True).squeeze(2)
        if cfg.teacher_cfg == 1.0:
            return v_c.float()
        v_u = _forward("teacher", x, t_b, c_null, no_grad=True).squeeze(2)
        return v_u.float() + cfg.teacher_cfg * (v_c.float() - v_u.float())

    # Fake head-start skipped on resume: the restored critic is already
    # calibrated against the restored student.
    data_iter = iter(dataloader)
    if start_step > 0:
        logger.info(
            "resume: skipping the fake head-start (critic restored, already warm)."
        )
    else:
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
            dtype=state_dtype,
            cond_dtype=dtype,
            log_interval=cfg.log_interval,
            writer=writer,
        )

    # CDM dynamic schedule (arXiv:2605.06376): the student rollout grid is
    # re-sampled per iteration instead of reusing the fixed inference grid.
    # N >= 2 (step 0 is the anchor, DMD wants >= 1 refinement step). The
    # diversity validation + inference stay on the static student_sigmas.
    if cfg.dynamic_schedule:
        logger.info(
            f"dynamic schedule ON (CDM): per-iteration rollout grid, "
            f"N ~ U{{2..{cfg.student_steps}}}, continuous anchors "
            f"(t_1=1 pinned); validation/inference keep the static grid."
        )
    logger.info(
        f"{'resuming' if start_step else 'starting'} turbo training: "
        f"steps {start_step} → {cfg.iterations}"
    )

    # Off a TTY, mirror the tqdm metrics as a throttled logger.info at log_interval
    # cadence (``metrics.console_step_line``).
    try:
        console_steps = not sys.stderr.isatty()
    except Exception:  # stderr detached (pythonw) → no live line either way
        console_steps = True
    metrics = TurboMetrics(device)

    ctx = RunContext(
        model=model,
        turbo=turbo,
        device=device,
        dtype=state_dtype,
        cond_dtype=dtype,
        student_opt=student_opt,
        fake_opt=fake_opt,
        disc_opt=disc_opt,
        student_sched=student_sched,
        fake_sched=fake_sched,
        disc_sched=disc_sched,
        dataloader=dataloader,
        data_iter=data_iter,
        forward=_forward,
        teacher_cfg_velocity=_teacher_cfg_velocity,
        student_sigmas=student_sigmas,
        teacher_anchor_sigmas=teacher_anchor_sigmas,
        t_k_anchor=t_k_anchor,
        uncond_base=uncond_base,
        cdm_on=cdm_on,
        writer=writer,
        progress_sink=progress_sink,
        int8_base_id=int8_base_id,
        console_steps=console_steps,
        metrics=metrics,
        tau_profile=TauBinCriticLoss(device),
        val_cond=val_cond,
        val_latent_shape=val_latent_shape,
        val_clean=val_clean,
        start_step=start_step,
    )
    return ctx
