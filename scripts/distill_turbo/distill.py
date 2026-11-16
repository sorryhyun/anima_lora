"""Turbo distillation main loop: DP-DMD (diversity-preserved DMD).

Usage:
    python -m scripts.distill_turbo.distill [--config configs/methods/turbo.toml] ...

Per iteration: teacher K-step anchor → diversity-supervised first step →
DMD-refined N-step student rollout → fake/critic update. ``run_loop`` consumes the
``RunContext`` that :mod:`scripts.distill_turbo.setup` builds.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

# train.py's allocator default, before the first CUDA allocation: without it long
# runs die of fragmentation (an OOM on a small alloc with free memory reported).
# print: no logging handler exists yet.
if default_expandable_segments():
    print(
        "PYTORCH_CUDA_ALLOC_CONF defaulted to expandable_segments:True "
        "(opt out: ANIMA_EXPANDABLE_SEGMENTS=0)",
        flush=True,
    )

import torch
from tqdm import tqdm

from library.anima.uncond import uncond_for_batch
from library.config.resolved import load_toml
from library.runtime.progress import run_scope

from .config import TurboConfig, build_argparser, resolve_config
from .context import Batch, RunContext
from .diversity import run_diversity_validation
from .metrics import console_step_line, tqdm_postfix, tqdm_rate, write_scalars
from .primitives import gan_effective_weight, sample_dynamic_sigmas
from .resume import resume_path_for, save_resume_state
from .setup import build_run
from .steps import (
    ClippedGradMean,
    anchor_only_loss,
    cdm_off_trajectory_loss,
    clip_or_norm,
    dm_slot,
    fake_update,
    gan_generator_term,
    student_rollout,
    teacher_anchor,
)

logger = logging.getLogger(__name__)

# Off a TTY: at most one console step line per this many seconds (TensorBoard
# and the progress stream still get every log_interval).
CONSOLE_EVERY_S = 30.0
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    args = build_argparser().parse_args()
    cfg = resolve_config(args, load_toml(args.config))
    ctx = build_run(args, cfg)
    run_loop(ctx, cfg)


def run_loop(ctx: RunContext, cfg: TurboConfig) -> None:
    grad_mean = (
        ClippedGradMean(ctx.turbo.student_params(), cfg.grad_accum)
        if cfg.grad_accum > 1
        else None
    )
    # Off a TTY the bar's carriage returns run into the logger lines; it keeps
    # counting (tqdm_rate) but writes nowhere, and the console line stands in.
    progress = tqdm(
        range(ctx.start_step, cfg.iterations),
        desc="turbo",
        initial=ctx.start_step,
        total=cfg.iterations,
        file=open(os.devnull, "w") if ctx.console_steps else None,
    )
    step = ctx.start_step - 1  # final_step() stays valid if the loop is empty
    with run_scope(ctx.progress_sink, final_step=lambda: step + 1):
        for step in progress:
            batch = ctx.next_batch()
            torch.compiler.cudagraph_mark_step_begin()
            if cfg.anchor_only:
                anchor_only_iteration(ctx, cfg, batch, grad_mean)
            else:
                dpdmd_iteration(ctx, cfg, batch, step, grad_mean)

            n = step + 1
            if n % cfg.log_interval == 0:
                _flush_logs(ctx, cfg, progress, n)
            if ctx.val_cond is not None and n % cfg.validate_every_n_steps == 0:
                _validate(ctx, cfg, n)
            if n % cfg.save_every == 0 or n == cfg.iterations:
                _save(ctx, cfg, n)

    if ctx.grad_log is not None:
        ctx.grad_log.flush()  # the tail past the last log boundary
    if cfg.int8 != "off":
        from library.anima.int8 import finish_training

        finish_training()
    if ctx.writer is not None:
        ctx.writer.close()
    logger.info("turbo distillation complete.")


def _student_update(ctx: RunContext, cfg, grad_mean) -> torch.Tensor:
    """Clip, then step the student (every ``grad_accum``-th call) and its EMA.
    Returns the pre-clip gradient norm."""
    gn = clip_or_norm(ctx.turbo.student_params(), cfg.grad_clip)
    if grad_mean is None or grad_mean.add():
        lr_now = ctx.student_opt.param_groups[0]["lr"]
        ctx.student_opt.step()
        ctx.student_sched.step()
        if ctx.ema is not None:
            ctx.ema.step(lr_now / ctx.ema_lr_mean if ctx.ema_lr_mean else 1.0)
    ctx.student_opt.zero_grad(set_to_none=True)
    return gn


def _anchor(ctx: RunContext, cfg, batch: Batch, c_null: torch.Tensor):
    """(ε, v_target): from the anchor cache, or ε fresh and the teacher rolled from
    it (v_target None with div_weight 0)."""
    if batch.anchor_eps is not None:
        return batch.anchor_eps, batch.anchor_v
    eps = torch.randn_like(batch.latents)
    if cfg.div_weight == 0.0:
        return eps, None
    B = eps.shape[0]
    return eps, teacher_anchor(ctx, cfg, eps, batch.crossattn_emb, c_null, B)


def anchor_only_iteration(ctx: RunContext, cfg, batch: Batch, grad_mean) -> None:
    """``dpdmd.anchor_only`` (project/anima_turbo/plan.md, S-b): step 0 on the anchor
    regression alone; ``anchor_draws`` k (S-c.2) adds k − 1 fresh-ε anchors for the
    same prompt and averages the gradient."""
    c, B = batch.crossattn_emb, batch.latents.shape[0]
    c_null = uncond_for_batch(ctx.uncond_base, c)
    eps, v_target = _anchor(ctx, cfg, batch, c_null)
    k = cfg.anchor_draws
    div_loss = anchor_only_loss(ctx, eps, v_target, c, B, scale=1.0 / k)
    for _ in range(k - 1):
        eps_k = torch.randn_like(batch.latents)
        v_target_k = teacher_anchor(ctx, cfg, eps_k, c, c_null, B)
        div_loss = div_loss + anchor_only_loss(ctx, eps_k, v_target_k, c, B, 1.0 / k)
    gn_student = _student_update(ctx, cfg, grad_mean)
    ctx.metrics.add_div(div_loss / k)
    ctx.metrics.add_grad_norms(gn_student, torch.zeros((), device=ctx.device))


def dpdmd_iteration(ctx: RunContext, cfg, batch: Batch, step: int, grad_mean) -> None:
    """One DP-DMD iteration: rollout to a grad step g, the DM-slot surrogate on its
    x_θ (+ L_CDM, + GAN generator term), the student update, then the fake/disc."""
    turbo = ctx.turbo
    latents, c = batch.latents, batch.crossattn_emb
    B = latents.shape[0]
    c_null = uncond_for_batch(ctx.uncond_base, c)
    eps, v_target = _anchor(ctx, cfg, batch, c_null)

    if cfg.dynamic_schedule:
        sigmas = sample_dynamic_sigmas(2, cfg.student_steps)
    else:
        sigmas = ctx.student_sigmas
    n_steps = len(sigmas) - 1
    # The grad step: a refinement step g ~ U{1..N−1}, or under dm_first_step any
    # step including the anchored step 0.
    g = int(torch.randint(0 if cfg.dm_first_step else 1, n_steps, (1,)).item())
    roll = student_rollout(ctx, cfg, eps, v_target, sigmas, g, c, B)
    x_pred = roll.x_pred
    # Transition-aware fake: every fake query on this x_pred is told it came from
    # step g; CDM sets its own t'.
    turbo.set_fake_source(roll.s_g)

    dmd = dm_slot(ctx, cfg, x_pred, c, c_null, B)
    if cfg.cdm_on:  # before the GAN gen forward (cdm_off_trajectory_loss)
        cdm_off_trajectory_loss(ctx, cfg, roll, c, c_null, B)
    gan_w = gan_effective_weight(cfg, step)
    gan_gen_loss = gan_generator_term(
        ctx, cfg, x_pred, dmd.tau_dm, dmd.eps_dm, c, gan_w
    )

    # The DMD surrogate on x_θ; grad_clip runs once on the accumulated .grad
    # (div, CDM, DMD, GAN).
    loss = (dmd.grad_signal * x_pred.float()).mean()
    if roll.div_pending:
        loss = loss + cfg.div_weight * roll.div_loss
    if turbo.disc is not None and gan_w > 0.0:
        loss = loss + gan_w * gan_gen_loss
    # Under --grad_ckpt the rollout's student forwards recompute at this backward,
    # and the DM term left the view on 'fake'.
    if cfg.grad_ckpt:
        turbo.set_view("student")
    loss.backward()
    gn_student = _student_update(ctx, cfg, grad_mean)

    fake = fake_update(ctx, cfg, x_pred, latents, c, B)

    gn_disc = fake.gn_disc if turbo.disc is not None else None
    ctx.metrics.accumulate_per_step(
        fake_loss_mean_t=fake.fake_loss,
        grad_signal=dmd.grad_signal,
        delta_dm=dmd.delta_dm,
        x_pred=x_pred,
        v_student=roll.v_first,
        tau_dm_e=dmd.tau_dm_e,
        v_real_cond_dm=dmd.v_real_cond_dm,
        v_fake_cond_dm=dmd.v_fake_cond_dm,
    )
    ctx.metrics.add_div(roll.div_loss)
    ctx.metrics.add_grad_norms(gn_student, fake.gn_fake, gn_disc)
    if turbo.disc is not None:
        ctx.metrics.add_gan(
            gan_gen_loss, fake.gan_disc, fake.gan_margin, fake.gan_spread
        )
    if ctx.grad_log is not None:
        ctx.grad_log.add(
            step + 1,
            gn_student,
            fake.gn_fake,
            gn_disc,
            dmd.tau_dm,
            roll.s_g,
            g,
            n_steps,
        )


def _flush_logs(ctx: RunContext, cfg, progress, n: int) -> None:
    """One sync per log boundary: TensorBoard, tqdm, console line, progress stream."""
    m = ctx.metrics.flush(cfg.log_interval)
    writer = ctx.writer
    if writer is not None:
        write_scalars(writer, m, n, anchor_only=cfg.anchor_only)
        writer.add_scalar("train/student_lr", ctx.student_sched.get_last_lr()[0], n)
        if not cfg.anchor_only:
            writer.add_scalar("train/fake_lr", ctx.fake_sched.get_last_lr()[0], n)
        if ctx.disc_sched is not None:
            writer.add_scalar("train/disc_lr", ctx.disc_sched.get_last_lr()[0], n)
        if ctx.turbo.disc is not None and not cfg.anchor_only:
            # The ramped generator-side λ of this step.
            writer.add_scalar(
                "train/gan_weight_gen_eff", gan_effective_weight(cfg, n - 1), n
            )
    progress.set_postfix(**tqdm_postfix(m, anchor_only=cfg.anchor_only), refresh=False)
    now = time.monotonic()
    if ctx.console_steps and (
        now - ctx.last_console >= CONSOLE_EVERY_S or n == cfg.iterations
    ):
        ctx.last_console = now
        logger.info(
            console_step_line(
                m,
                step=n,
                total=cfg.iterations,
                rate=tqdm_rate(progress),
                anchor_only=cfg.anchor_only,
            )
        )
    if ctx.progress_sink is not None:
        ctx.progress_sink.log(dataclasses.asdict(m), global_step=n, epoch=0)
    ctx.metrics.reset()
    ctx.tau_profile.write(writer, n)
    if ctx.grad_log is not None:
        ctx.grad_log.flush()


def _validate(ctx: RunContext, cfg, n: int) -> None:
    """DAVE same-prompt diversity probe on the held-out sample."""
    dm = run_diversity_validation(
        model=ctx.model,
        forward_fn=ctx.forward,
        student_sigmas=ctx.student_sigmas,
        crossattn_emb=ctx.val_cond,
        latent_shape=ctx.val_latent_shape,
        num_seeds=cfg.val_diversity_seeds,
        seed0=cfg.seed,
        device=ctx.device,
        dtype=ctx.dtype,
        clean_latent=ctx.val_clean,
    )
    if ctx.writer is not None:
        ctx.writer.add_scalar("val/div_ac_sim", dm.ac_sim, n)
        ctx.writer.add_scalar("val/div_dc_sim", dm.dc_sim, n)
        ctx.writer.add_scalar("val/div_gap", dm.gap, n)
        ctx.writer.add_scalar("val/div_xpred_ac_sim", dm.xpred_ac_sim, n)
        ctx.writer.add_scalar("val/fm_mse", dm.fm_mse, n)
    logger.info(
        f"[val@{n}] diversity: AC sim={dm.ac_sim:.4f} (lower=more diverse) | "
        f"DC sim={dm.dc_sim:.4f} | gap={dm.gap:+.4f} | x_pred AC sim="
        f"{dm.xpred_ac_sim:.4f} | FM MSE={dm.fm_mse:.4f}"
    )


def _step_tag(step: int) -> str:
    """Checkpoint suffix: 1000 → ``1k``, 8000 → ``8k``, else the raw count."""
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def _student_metadata(cfg: TurboConfig, ctx: RunContext, step: int) -> dict[str, str]:
    md: dict[str, str] = {}
    if cfg.int8 != "off":
        md["ss_int8_base"] = cfg.int8
        md["ss_int8_base_id"] = ctx.int8_base_id
    if cfg.stream != "bf16":
        md["ss_stream"] = cfg.stream
    md.update(
        ss_turbo_objective="dpdmd",
        ss_turbo_student_rank=str(cfg.student_rank),
        ss_turbo_student_alpha=str(cfg.student_alpha),
        ss_turbo_student_steps=str(cfg.student_steps),
        ss_turbo_dynamic_schedule="1" if cfg.dynamic_schedule else "0",
        ss_turbo_teacher_cfg=str(cfg.teacher_cfg),
        ss_turbo_step=str(step),
        ss_turbo_k_anchor=str(cfg.k_anchor),
        ss_turbo_div_weight=str(cfg.div_weight),
        ss_turbo_gan_weight_gen=str(cfg.gan_loss_weight_gen),
        ss_turbo_gan_disc_head=cfg.gan_disc_head,
        ss_turbo_gan_delay_steps=str(cfg.gan_delay_steps),
        ss_turbo_gan_warmup_steps=str(cfg.gan_warmup_steps),
        ss_turbo_cdm_weight=str(cfg.cdm_weight),
        ss_turbo_cdm_teacher=cfg.cdm_teacher,
        ss_turbo_lr_schedule=cfg.lr_schedule,
    )
    optional = {
        "ss_turbo_anchor_only": "1" if cfg.anchor_only else None,
        "ss_turbo_anchor_draws": str(cfg.anchor_draws)
        if cfg.anchor_draws > 1
        else None,
        "ss_turbo_anchor_cache": (
            os.path.basename(os.path.normpath(cfg.anchor_cache))
            if cfg.anchor_cache
            else None
        ),
        "ss_turbo_dm_first_step": "1" if cfg.dm_first_step else None,
        "ss_turbo_rollout_grad": "1" if cfg.rollout_grad else None,
        "ss_turbo_dm_draws": str(cfg.dm_draws) if cfg.dm_draws > 1 else None,
        "ss_turbo_grad_accum": str(cfg.grad_accum) if cfg.grad_accum > 1 else None,
        "ss_turbo_dm_tau": (
            f"{cfg.dm_tau_distribution}:{cfg.dm_tau_scale}"
            if cfg.dm_tau_distribution != "uniform"
            else None
        ),
    }
    if cfg.mfd_dtau > 0.0:
        optional.update(
            ss_turbo_mfd_dtau=str(cfg.mfd_dtau),
            ss_turbo_mfd_max_steps=str(cfg.mfd_max_steps),
            ss_turbo_mfd_interval=cfg.mfd_interval,
        )
    if cfg.train_adaln:
        optional["ss_turbo_train_adaln"] = "1"
        if cfg.adaln_rank > 0:  # provenance; per-module rank lives in the file
            optional["ss_turbo_adaln_rank"] = str(cfg.adaln_rank)
    if cfg.student_init_weights:
        optional["ss_turbo_student_init_weights"] = os.path.basename(
            cfg.student_init_weights
        )
    md.update({k: v for k, v in optional.items() if v is not None})
    return md


def _save(ctx: RunContext, cfg: TurboConfig, n: int) -> None:
    """Step-tagged student (+ EMA) under ``<output_dir>/<output_name>/``; the final
    step also writes the bare ``<output_name>`` at the root, where inference and
    merge look. Every save but the last also rewrites the rolling resume bundle."""
    is_final = n == cfg.iterations
    metadata = _student_metadata(cfg, ctx, n)
    run_dir = Path(cfg.output_dir) / cfg.output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(run_dir / f"{cfg.output_name}_{_step_tag(n)}.safetensors")]
    if is_final:
        paths.append(str(Path(cfg.output_dir) / f"{cfg.output_name}.safetensors"))
    for path in paths:
        ctx.turbo.save_student(path, dtype=torch.bfloat16, metadata=metadata)
        logger.info(f"saved checkpoint: {path}")
        if ctx.progress_sink is not None:
            ctx.progress_sink.ckpt(global_step=n, path=path)
    ema = ctx.ema
    if ema is not None and ema.started:
        ema_md = {**metadata, **ema.metadata()}
        with ema.swapped_in():
            for path in paths:
                ema_path = path.removesuffix(".safetensors") + "_ema.safetensors"
                ctx.turbo.save_student(ema_path, dtype=torch.bfloat16, metadata=ema_md)
                logger.info(f"saved EMA checkpoint: {ema_path}")

    if not is_final:
        rp = resume_path_for(cfg.output_dir, cfg.output_name)
        save_resume_state(
            rp,
            step=n,
            cfg=cfg,
            turbo=ctx.turbo,
            student_opt=ctx.student_opt,
            fake_opt=ctx.fake_opt,
            disc_opt=ctx.disc_opt,
            student_sched=ctx.student_sched,
            fake_sched=ctx.fake_sched,
            disc_sched=ctx.disc_sched,
        )
        logger.info(f"saved resume bundle: {rp} (step {n})")


if __name__ == "__main__":
    main()
