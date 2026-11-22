"""Run construction: everything ``build_run`` sets up before the step loop.

Order is load-bearing: block swap → grad ckpt → adapter stacks (the int8 / fp32-stream
conversion runs inside, before ``apply_to``) → warm start → compile; the dynamo limit
is pinned before the first trace; resume applies once every optimizer and scheduler
exists; the fake head-start runs last.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path

import torch

from library.anima import weights as anima_utils
from library.anima.models import Anima
from library.anima.stream import latent_dtype, stream_fp32_
from library.anima.uncond import default_uncond_path, load_uncond_crossattn
from library.datasets.cache import CachedDataset, make_cached_collate
from library.inference.sampling import get_timesteps_sigmas
from library.int8.linear import int8_input_tag
from library.io.cache import discover_cached_pairs
from library.runtime.dynamo import (
    compile_signature,
    isolate_compile_cache,
    pin_dynamo_limit,
    set_activation_memory_budget,
)
from library.runtime.harness import (
    _apply_partitioner_tuning,
    compile_dit_blocks,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from library.runtime.progress import ProgressSink, command_job_path
from library.training.distill_runtime import (
    apply_single_prompt_slice,
    create_tb_writer,
    ensure_dynamic_seq_for_freefit,
    resolve_device_dtype,
    write_config_snapshot,
)
from library.training.ema import AdapterEMA
from library.training.forward import PadCache
from networks.methods.turbo_dmd import TurboDMDNetwork, warm_start_plain_lora

from .anchor_cache import AnchorDataset, anchor_collate, keep_stems
from .config import TurboConfig, snapshot_toml_text, tb_config_text
from .context import RunContext
from .metrics import StepGradLog, TauBinCriticLoss, TurboMetrics
from .primitives import make_scheduler, schedule_lrs, scheduler_warmup_steps
from .resume import (
    apply_resume_state,
    load_resume_state,
    resolve_resume_arg,
    resume_path_for,
)
from .warmup import run_fake_warmup

logger = logging.getLogger(__name__)

# `{stem}_{W:04d}x{H:04d}_anima.npz` → pixel (W, H)
_LATENT_RES_RE = re.compile(r"_(\d{3,4})x(\d{3,4})_anima\.npz$")


def latent_token_counts(npz_paths) -> set[int]:
    """Distinct DiT token counts ``(W//16)·(H//16)`` from cached-latent file names."""
    counts = set()
    for path in npz_paths:
        m = _LATENT_RES_RE.search(os.path.basename(path))
        if m:
            counts.add((int(m.group(1)) // 16) * (int(m.group(2)) // 16))
    return counts


def compile_turbo_dit(
    model: Anima,
    *,
    enabled: bool,
    dynamic_seq: bool,
    n_token_families: int | None,
    seq_range: tuple[int, int] | None,
    int8: str,
    stream: str,
) -> None:
    """Compile the blocks under a compile cache isolated per signature.

    A stale narrow seq-range guard from another run poisons a wider dynamic-seq
    mark (ConstraintViolationError on the first ≥4032-token trace); the same
    signature still warm-reuses, shared with train.py.
    """
    if enabled:
        isolate_compile_cache(
            compile_signature(
                n_token_families=n_token_families,
                seq_range=seq_range,
                dynamic_seq=dynamic_seq,
                mode="",
            )
            + (
                ""
                if int8 == "off"
                else f";int8={int8};int8_in={int8_input_tag()};flops=fa2,int8"
            )
            + ("" if stream == "bf16" else f";stream={stream}")
        )
    compile_dit_blocks(
        model,
        enabled=enabled,
        mode="",
        dynamic_seq=dynamic_seq,
        n_token_families=n_token_families,
        seq_range=seq_range,
    )


def _build_model(cfg: TurboConfig, device, dtype) -> tuple[Anima, TurboDMDNetwork, str]:
    logger.info(f"loading DiT: {cfg.dit_path}")
    model: Anima = anima_utils.load_anima_model(
        device,
        cfg.dit_path,
        attn_mode=cfg.attn_mode,
        loading_device="cpu" if cfg.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
        dit_nonquant_path=cfg.dit_nonquant_path,
    )
    place_dit_for_training(model, device, blocks_to_swap=cfg.blocks_to_swap)
    enable_training_grad_ckpt(model, enabled=cfg.grad_ckpt)

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
        channel_scaling_alpha=cfg.channel_scaling_alpha,
        student_down_init=cfg.student_down_init,
        fake_down_init=cfg.fake_down_init,
        train_adaln=cfg.train_adaln,
        fake_adaln=cfg.fake_adaln,
        adaln_rank=cfg.adaln_rank,
        adaln_alpha=cfg.adaln_alpha,
        before_apply=convert_base,
    )
    turbo.freeze_dit()
    turbo.student.to(device=device, dtype=dtype)
    turbo.fake.to(device=device, dtype=dtype)
    return model, turbo, int8_base_id


def _load_resume(cfg: TurboConfig) -> dict | None:
    """The resume bundle, read now so a bad ``--resume`` fails before compile."""
    if not cfg.resume:
        return None
    arg = resolve_resume_arg(cfg.resume, cfg.output_dir, cfg.output_name)
    if arg.path is None:
        logger.info(
            "--resume auto: no bundle at "
            f"{resume_path_for(cfg.output_dir, cfg.output_name)}; starting fresh."
        )
        return None
    state = load_resume_state(arg.path)
    logger.info(f"resuming from {arg.path} @ step {state['step']}")
    return state


def _compile(cfg: TurboConfig, model: Anima) -> None:
    """Compile after ``apply_to`` so the traced forward includes the adapters. The
    dynamic-seq bound comes from the cached pool's token counts unless target_res
    is given."""
    counts = latent_token_counts(
        img.npz_path for img in discover_cached_pairs(cfg.data_dir)
    )
    n_token_families = seq_range = None
    if cfg.target_res:
        from library.datasets.buckets import token_count_families, token_count_range

        n_token_families = token_count_families(cfg.target_res)
        seq_range = token_count_range(cfg.target_res)
    elif counts:
        n_token_families = len(counts)
        seq_range = (min(counts), max(counts))
    dynamic_seq = cfg.compile_dynamic_seq
    if cfg.torch_compile and not dynamic_seq:
        dynamic_seq = ensure_dynamic_seq_for_freefit(counts, dynamic_seq, logger=logger)

    if cfg.torch_compile:
        # Before the first compiled forward. The budget is skipped under grad_ckpt:
        # repartitioning can pick a different backward graph than forward →
        # CheckpointError (torch #166926).
        if cfg.activation_memory_budget < 1.0 and cfg.grad_ckpt:
            logger.info("activation_memory_budget ignored under grad_ckpt")
        elif cfg.activation_memory_budget < 1.0:
            set_activation_memory_budget(cfg.activation_memory_budget)
            logger.info(
                f"torch.compile activation_memory_budget = {cfg.activation_memory_budget}"
            )
        _apply_partitioner_tuning(
            recompute_views=cfg.partitioner_recompute_views,
            aggressive_recomputation=cfg.partitioner_aggressive_recomputation,
            grad_ckpt=cfg.grad_ckpt,
            logger=logger,
        )

    compile_turbo_dit(
        model,
        enabled=cfg.torch_compile,
        dynamic_seq=dynamic_seq,
        n_token_families=n_token_families,
        seq_range=seq_range,
        int8=cfg.int8,
        stream=cfg.stream,
    )
    if cfg.torch_compile:
        # accumulated_recompile_limit over the real block count (max(), never lowers).
        rl = pin_dynamo_limit("recompile_limit", cfg.dynamo_recompile_limit)
        arl = pin_dynamo_limit(
            "accumulated_recompile_limit",
            len(model.blocks) * cfg.dynamo_recompile_limit,
        )
        logger.info(f"dynamo recompile_limit={rl}, accumulated_recompile_limit={arl}")


def _adamw(params, lr: float, cfg: TurboConfig, **kw) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
        **kw,
    )


def _build_data(cfg: TurboConfig, device, dtype, state_dtype, start_step: int):
    """Dataset, held-out validation sample and DataLoader.

    Returns ``(dataloader, anchor_cache, val_cond, val_latent_shape, val_clean)``.
    """
    dataset = CachedDataset(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        sample_ratio=cfg.sample_ratio,
        need_pooled=False,
        keep_list=keep_stems(cfg.anchor_cache) if cfg.anchor_cache else None,
    )
    # The diversity probe's sample comes from the full list, before a single-prompt
    # slice, and differs from the overfit sample so a collapsed run shows.
    val_cond = val_latent_shape = val_clean = None
    if cfg.validate_every_n_steps > 0 and len(dataset.samples) > 0:
        n = len(dataset.samples)
        if cfg.val_prompt_idx >= 0:
            v_idx = cfg.val_prompt_idx % n
        else:
            v_idx = n - 1
            if cfg.single_prompt_idx is not None and v_idx == cfg.single_prompt_idx % n:
                v_idx = (v_idx - 1) % n
        v_sample = dataset[v_idx]
        v_lat, v_ca = v_sample["latents"], v_sample["crossattn_emb"]
        val_cond = v_ca.unsqueeze(0).to(device, dtype=dtype)
        val_latent_shape = (1, *tuple(v_lat.shape))
        val_clean = v_lat.unsqueeze(0).to(device, dtype=state_dtype)
        logger.info(
            f"diversity validation: every {cfg.validate_every_n_steps} steps, "
            f"{cfg.val_diversity_seeds} seeds, held-out idx={v_idx} "
            f"(latent {tuple(v_lat.shape)})"
        )

    if cfg.single_prompt_idx is not None:
        apply_single_prompt_slice(dataset, cfg.single_prompt_idx, logger=logger)

    # One resolution per batch; the largest bucket goes first for compile warmup.
    batch_sampler = dataset.make_batch_sampler(shuffle=True, seed=cfg.seed)
    anchor_cache = None
    if cfg.anchor_cache:
        # A resumed run starts on its epoch's draw.
        anchor_cache = dataset = AnchorDataset(
            dataset,
            cfg.anchor_cache,
            cfg,
            epoch=start_step // max(1, len(batch_sampler)),
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=anchor_collate
        if anchor_cache is not None
        else make_cached_collate(),
    )
    return dataloader, anchor_cache, val_cond, val_latent_shape, val_clean


def _build_logging(args, cfg: TurboConfig):
    """Config snapshots, TensorBoard writer and the progress sink.

    Returns ``(writer, run_log, progress_sink)``.
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    snapshot_text = snapshot_toml_text(cfg, source_config=args.config)
    # Beside the checkpoint, written even under --no_log.
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
            run_log / f"{cfg.output_name}.snapshot.toml", snapshot_text, logger=logger
        )

    # Under the daemon the stream goes to the job dir (`make run-status RUN=<job id>`);
    # inline, logs/<output_name>.progress.jsonl.
    # --no_log keeps the console lines and drops the stream.
    sink_path = None if cfg.no_log else command_job_path(ProgressSink.resolve_path(cfg))
    progress_sink = ProgressSink(
        sink_path,
        run=cfg.output_name,
        method="turbo",
        preset=None,
        console_keys=(
            ("div", "gn_student")
            if cfg.anchor_only
            else ("grad", "rel_gap", "cos", "xpred", "fake", "gn_student", "div")
        ),
    )
    progress_sink.run_start(
        total_steps=cfg.iterations,
        total_epochs=1,
        pid=os.getpid(),
        log_dir=str(run_log) if run_log is not None else None,
        config=dataclasses.asdict(cfg),
    )
    progress_sink.attach_log_mirror()
    return writer, run_log, progress_sink


def _build_ema(cfg: TurboConfig, turbo, student_updates: int, start_step: int):
    """Student EMA from the end of the LR warmup; returns ``(ema, ema_lr_mean)``.
    Shadows are not in the resume bundle, so a resumed run re-seeds."""
    if cfg.ema_window <= 0.0:
        return None, 0.0
    ema = AdapterEMA(
        turbo.student_params(),
        cfg.ema_window,
        "",
        updates=student_updates,
        start=scheduler_warmup_steps(student_updates),
    )
    ema.n_updates = start_step // cfg.grad_accum
    logger.info(f"student {ema.describe()}")
    if cfg.lr_schedule == "constant":
        return ema, 0.0
    lrs = schedule_lrs(student_updates, cfg.student_lr, cfg.lr_schedule)
    tail = lrs[ema.start :]
    lr_mean = sum(tail) / max(1, len(tail))
    logger.info(
        f"student EMA span in summed LR ({cfg.lr_schedule}): mean lr "
        f"{lr_mean:.2e}, last update at rate {lrs[-1] / lr_mean:.2f}"
    )
    return ema, lr_mean


def build_run(args, cfg: TurboConfig) -> RunContext:
    torch.manual_seed(cfg.seed)
    device, dtype = resolve_device_dtype()

    # Before any block traces: the views give each block more graphs than the
    # default limit of 8, and pin_dynamo_limit survives into backward.
    # set_num_threads is forced now because a later change is a GLOBAL_STATE
    # guard: it recompiles everything.
    if cfg.torch_compile:
        pin_dynamo_limit("recompile_limit", cfg.dynamo_recompile_limit)
    torch.set_num_threads(torch.get_num_threads())

    model, turbo, int8_base_id = _build_model(cfg, device, dtype)
    state_dtype = latent_dtype(model, dtype)

    # A resume bundle supersedes the warm start. The warm start runs after .to()
    # (its SVD runs on the card) and before compile.
    resume_state = _load_resume(cfg)
    if resume_state is None:
        if cfg.student_init_weights:
            warm_start_plain_lora(turbo.student, cfg.student_init_weights, "student")
        if cfg.fake_init_weights:
            warm_start_plain_lora(turbo.fake, cfg.fake_init_weights, "fake")

    _compile(cfg, model)

    n_student = sum(p.numel() for p in turbo.student_params())
    n_fake = sum(p.numel() for p in turbo.fake_params())
    logger.info(f"trainable: student={n_student:,}  fake={n_fake:,}")
    student_updates = cfg.iterations // cfg.grad_accum
    student_opt = _adamw(turbo.student_params(), cfg.student_lr, cfg)
    student_sched = make_scheduler(
        student_opt, student_updates, cfg.student_lr, cfg.lr_schedule
    )
    # The fake schedule spans the head-start too (warmup.py).
    fake_opt = _adamw(turbo.fake_params(), cfg.fake_lr, cfg)
    fake_sched = make_scheduler(
        fake_opt,
        cfg.iterations * cfg.fake_steps_per_student_step + cfg.fake_warmup_steps,
        cfg.fake_lr,
        cfg.lr_schedule,
    )
    start_step = 0
    if resume_state is not None:
        start_step = apply_resume_state(
            resume_state,
            cfg=cfg,
            turbo=turbo,
            student_opt=student_opt,
            fake_opt=fake_opt,
            student_sched=student_sched,
            fake_sched=fake_sched,
        )
        if start_step >= cfg.iterations:
            raise SystemExit(
                f"resume: bundle is at step {start_step} but iterations="
                f"{cfg.iterations}; raise --iterations to extend the run."
            )
        logger.info(
            f"resume: continuing at step {start_step}/{cfg.iterations} "
            f"(student LR {student_sched.get_last_lr()[0]:.3g})"
        )
        # The bundle (two AdamW moment sets) is the largest transient in the process.
        resume_state = None

    dataloader, anchor_cache, val_cond, val_latent_shape, val_clean = _build_data(
        cfg, device, dtype, state_dtype, start_step
    )
    writer, run_log, progress_sink = _build_logging(args, cfg)

    # The CFG uncond input is the T5("") embedding (only padding zeroed). An all-zero
    # tensor is off-distribution: its uncond direction, amplified at (α-1)=3×, drives
    # the student off-manifold (saturated white). Staged by `make preprocess-te`.
    uncond_path = str(default_uncond_path())
    uncond_base = load_uncond_crossattn(uncond_path, device=device, dtype=dtype)
    logger.info(
        f"loaded T5('') uncond sidecar: {uncond_path}  shape={tuple(uncond_base.shape)}"
    )

    pad_cache = PadCache(dtype)

    def forward(
        view: str, x: torch.Tensor, t_b: torch.Tensor, c: torch.Tensor, *, no_grad: bool
    ):
        """The DiT as ``view`` on a 4D ``x`` → the 4D velocity.

        ``model.training`` stays True for the whole run (the DiT is frozen), which
        keeps ``Block.forward``'s grad-ckpt gate armed; toggling it per forward
        stalls the CPU on every forward.
        """
        turbo.set_view(view)
        model.rewind_block_swap()
        grad = torch.no_grad() if no_grad else torch.enable_grad()
        with grad, torch.autocast("cuda", dtype=dtype):
            out = model.forward_mini_train_dit(
                x.unsqueeze(2),
                t_b,
                c,
                padding_mask=pad_cache.get(x),
                skip_pooled_text_proj=True,
            )
        return out.squeeze(2)

    def teacher_cfg_velocity(x, t_b, c_cond, c_null):
        """CFG'd teacher velocity ``v_u + α·(v_c − v_u)`` (no grad, fp32)."""
        v_c = forward("teacher", x, t_b, c_cond, no_grad=True)
        if cfg.teacher_cfg == 1.0:
            return v_c.float()
        v_u = forward("teacher", x, t_b, c_null, no_grad=True)
        return v_u.float() + cfg.teacher_cfg * (v_c.float() - v_u.float())

    # σ grids 1 → 0; sigmas[i] - sigmas[i+1] is the Euler dt of step i.
    student_sigmas = get_timesteps_sigmas(cfg.student_steps, cfg.flow_shift, "cpu")[
        1
    ].tolist()
    teacher_anchor_sigmas = get_timesteps_sigmas(
        cfg.teacher_anchor_steps, cfg.flow_shift, "cpu"
    )[1].tolist()
    sigma_k = teacher_anchor_sigmas[cfg.k_anchor]
    logger.info(
        f"DP-DMD grids: student σ={['%.3f' % s for s in student_sigmas]}, "
        f"anchor σ_k={sigma_k:.4f} (teacher step "
        f"{cfg.k_anchor}/{cfg.teacher_anchor_steps})"
    )
    if cfg.div_weight > 0.0 and abs(sigma_k - student_sigmas[1]) > 1e-4:
        logger.warning(
            f"anchor interval mismatch: the target is the teacher's mean velocity "
            f"over σ [{sigma_k:.3f}, 1], and step 0 applies it over "
            f"σ [{student_sigmas[1]:.3f}, 1]. A step 0 at zero anchor loss then lands "
            f"on the chord's point at σ {student_sigmas[1]:.3f}, not on a teacher "
            "state. Set dpdmd.k_anchor so that σ_k is the student's σ_1."
        )

    ema, ema_lr_mean = _build_ema(cfg, turbo, student_updates, start_step)

    ctx = RunContext(
        model=model,
        turbo=turbo,
        device=device,
        dtype=state_dtype,
        cond_dtype=dtype,
        student_opt=student_opt,
        fake_opt=fake_opt,
        student_sched=student_sched,
        fake_sched=fake_sched,
        dataloader=dataloader,
        data_iter=iter(dataloader),
        anchor_cache=anchor_cache,
        forward=forward,
        teacher_cfg_velocity=teacher_cfg_velocity,
        student_sigmas=student_sigmas,
        teacher_anchor_sigmas=teacher_anchor_sigmas,
        uncond_base=uncond_base,
        writer=writer,
        progress_sink=progress_sink,
        grad_log=(
            StepGradLog(run_log / "grad_steps.jsonl", device)
            if run_log is not None
            else None
        ),
        metrics=TurboMetrics(device),
        tau_profile=TauBinCriticLoss(device),
        val_cond=val_cond,
        val_latent_shape=val_latent_shape,
        val_clean=val_clean,
        start_step=start_step,
        int8_base_id=int8_base_id,
        ema=ema,
        ema_lr_mean=ema_lr_mean,
    )

    # The restored critic of a resumed run is already calibrated.
    if start_step > 0:
        logger.info("resume: skipping the fake head-start.")
    else:
        run_fake_warmup(ctx, cfg)

    logger.info(
        f"{'resuming' if start_step else 'starting'} turbo training: "
        f"steps {start_step} → {cfg.iterations}"
    )
    return ctx
