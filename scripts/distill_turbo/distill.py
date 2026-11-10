"""Turbo distillation main loop — DP-DMD (diversity-preserved DMD).

Usage:
    python -m scripts.distill_turbo.distill [--config configs/methods/turbo.toml] ...

The per-step orchestrator: teacher K-step anchor → diversity-supervised first step
→ DMD-refined N-step student rollout → fake/critic update → save. ``run_loop``
consumes the ``RunContext`` that :mod:`scripts.distill_turbo.setup` builds.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

# Mirror of train.py's allocator default (bespoke loops don't inherit train.py
# infra): must run before the first CUDA allocation. Without it long runs die
# of fragmentation (an OOM on a small alloc with free memory reported).
# print, not logging — no handler exists this early (basicConfig runs below).
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

from .config import build_argparser, resolve_config
from .diversity import run_diversity_validation
from .metrics import (
    console_step_line,
    tqdm_postfix,
    tqdm_rate,
    write_scalars,
)
from .primitives import gan_effective_weight, sample_dynamic_sigmas
from .resume import resume_path_for, save_resume_state
from .setup import RunContext, build_run
from .steps import (
    cdm_off_trajectory_loss,
    clip_or_norm,
    dmd_surrogate,
    mfd_surrogate,
    fake_update,
    gan_generator_term,
    student_rollout,
    teacher_anchor,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _step_tag(step: int) -> str:
    """Human checkpoint suffix: 1000 -> ``1k``, 8000 -> ``8k``, else raw count."""
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def _student_metadata(cfg, ctx: RunContext, step: int) -> dict[str, str]:
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
    )
    if cfg.dm_first_step:
        md["ss_turbo_dm_first_step"] = "1"
    if cfg.mfd_dtau > 0.0:
        md["ss_turbo_mfd_dtau"] = str(cfg.mfd_dtau)
        md["ss_turbo_mfd_max_steps"] = str(cfg.mfd_max_steps)
        md["ss_turbo_mfd_interval"] = cfg.mfd_interval
    if cfg.train_adaln:
        md["ss_turbo_train_adaln"] = "1"
        if cfg.adaln_rank > 0:  # provenance; per-module rank lives in the file
            md["ss_turbo_adaln_rank"] = str(cfg.adaln_rank)
    if cfg.student_init_weights:
        md["ss_turbo_student_init_weights"] = os.path.basename(cfg.student_init_weights)
    return md


def main():
    args = build_argparser().parse_args()
    cfg = resolve_config(args, load_toml(args.config))
    ctx = build_run(args, cfg)
    run_loop(ctx, cfg)


def run_loop(ctx: RunContext, cfg):
    """Per-step DP-DMD training loop over the objects built by ``build_run``.

    ``ctx`` fields are bound to locals up front so the loop body reads as the
    plain algorithm; only ``data_iter`` is mutated (epoch re-iter).
    """
    turbo = ctx.turbo
    device = ctx.device
    dtype = ctx.dtype
    student_opt, fake_opt, disc_opt = ctx.student_opt, ctx.fake_opt, ctx.disc_opt
    student_sched = ctx.student_sched
    fake_sched = ctx.fake_sched
    disc_sched = ctx.disc_sched
    data_iter = ctx.data_iter
    _forward = ctx.forward
    student_sigmas = ctx.student_sigmas
    cdm_on = ctx.cdm_on
    writer = ctx.writer
    progress_sink = ctx.progress_sink
    metrics = ctx.metrics

    progress = tqdm(
        range(ctx.start_step, cfg.iterations),
        desc="turbo",
        initial=ctx.start_step,
        total=cfg.iterations,
    )

    # Full-run lifecycle for the progress.jsonl sink: run_scope maps a clean
    # return / KeyboardInterrupt / crash onto the matching run_end status, so a
    # reader can tell 'done' from 'died'.
    step = ctx.start_step - 1  # sentinel: valid final_step() if the loop is empty
    with run_scope(progress_sink, final_step=lambda: step + 1):
        for step in progress:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(ctx.dataloader)
                batch = next(data_iter)
            latents = batch["latents"]
            crossattn_emb = batch["crossattn_emb"]

            latents = latents.to(device, dtype=dtype, non_blocking=True)
            crossattn_emb = crossattn_emb.to(
                device, dtype=ctx.cond_dtype, non_blocking=True
            )
            B = latents.shape[0]

            # No-op under mode="default"; "reduce-overhead" needs it once per step.
            torch.compiler.cudagraph_mark_step_begin()

            # Student update: roll an N-step Euler grid from pure noise ε, step 0
            # anchored to a teacher K-step target, steps 1..N-1 DMD-refined.
            eps = torch.randn_like(latents)  # shared start for anchor + student
            c_null = uncond_for_batch(
                ctx.uncond_base, crossattn_emb
            )  # anchor + DMD eval

            # --- teacher K-step CFG anchor (no grad) → v_target; div_weight 0 skips it ---
            v_target = (
                teacher_anchor(ctx, cfg, eps, crossattn_emb, c_null, B)
                if cfg.div_weight > 0.0
                else None
            )

            # --- student rollout → x_pred (= x_θ) ---
            if cfg.dynamic_schedule:
                sigmas_it = sample_dynamic_sigmas(2, cfg.student_steps)
                n_steps_it = len(sigmas_it) - 1
            else:
                sigmas_it, n_steps_it = student_sigmas, cfg.student_steps

            # One step g carries the DM grad: a refinement step, g ~ U{1..N−1},
            # or any step including the anchored step 0 under dm_first_step.
            g_min = 0 if cfg.dm_first_step else 1
            g = int(torch.randint(g_min, n_steps_it, (1,)).item())
            roll = student_rollout(
                ctx, cfg, eps, v_target, sigmas_it, g, crossattn_emb, B
            )
            x_pred = roll.x_pred  # one-step x0-prediction at step g
            s_g = roll.s_g
            div_loss_t = roll.div_loss
            cdm_src = (roll.x_g, roll.v_g, s_g) if cdm_on else None
            # Transition-aware fake: every fake query on this x_pred (DMD/MFD, the
            # fake update) is told it came from step g. CDM sets its own t'.
            turbo.set_fake_source(s_g)
            v_student = roll.v_first  # step-0 velocity for the runaway-student metric

            # --- DMD (or MFD) on x_θ, against teacher + fake ---
            if cfg.mfd_dtau > 0.0:
                dmd = mfd_surrogate(ctx, cfg, x_pred, crossattn_emb, c_null, B)
            else:
                dmd = dmd_surrogate(ctx, cfg, x_pred, crossattn_emb, c_null, B)
            grad_signal = dmd.grad_signal
            delta_dm = dmd.delta_dm
            tau_dm = dmd.tau_dm
            tau_dm_e = dmd.tau_dm_e
            v_real_cond_dm = dmd.v_real_cond_dm
            v_fake_cond_dm = dmd.v_fake_cond_dm
            eps_dm = dmd.eps_dm

            # --- L_CDM (before the GAN gen forward; see cdm_off_trajectory_loss) ---
            if cdm_src is not None:
                cdm_off_trajectory_loss(
                    ctx, cfg, cdm_src, crossattn_emb, c_null, latents, B
                )

            # --- GAN generator term ---
            gan_w = gan_effective_weight(cfg, step)
            gan_gen_loss = gan_generator_term(
                ctx, cfg, x_pred, tau_dm, eps_dm, crossattn_emb, gan_w
            )

            # --- assemble: DMD surrogate on x_θ ---
            # grad_clip runs once on the accumulated .grad (div, CDM, DMD, GAN).
            loss_student = (grad_signal * x_pred.float()).mean()
            if roll.div_pending:
                loss_student = loss_student + cfg.div_weight * div_loss_t

            if turbo.disc is not None and gan_w > 0.0:
                loss_student = loss_student + gan_w * gan_gen_loss

            # Under --grad_ckpt the rollout's student forwards recompute at this
            # backward, and the DM term left the view on 'fake'.
            if cfg.grad_ckpt:
                turbo.set_view("student")
            loss_student.backward()
            gn_student = clip_or_norm(turbo.student_params(), cfg.grad_clip)
            student_opt.step()
            student_opt.zero_grad(set_to_none=True)
            student_sched.step()

            # --- fake (critic) + discriminator update against x_pred.detach() ---
            (
                fake_loss_mean_t,
                gan_disc_mean_t,
                gan_margin_t,
                gan_spread_t,
                gn_fake,
                gn_disc,
            ) = fake_update(ctx, cfg, x_pred, latents, crossattn_emb, B)

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
            )
            metrics.add_div(div_loss_t)
            metrics.add_grad_norms(
                gn_student, gn_fake, gn_disc if turbo.disc is not None else None
            )
            if turbo.disc is not None:
                metrics.add_gan(
                    gan_gen_loss, gan_disc_mean_t, gan_margin_t, gan_spread_t
                )

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
                    if turbo.disc is not None:
                        # The ramped generator-side λ actually applied this step
                        # (deterministic from step; makes the delay/warmup window
                        # legible next to the margin/spread curves).
                        writer.add_scalar("train/gan_weight_gen_eff", gan_w, step + 1)
                # log_interval cadence (per-step would add CUDA syncs).
                progress.set_postfix(**tqdm_postfix(m))
                if ctx.console_steps:
                    logger.info(
                        console_step_line(
                            m,
                            step=step + 1,
                            total=cfg.iterations,
                            rate=tqdm_rate(progress),
                        )
                    )
                if progress_sink is not None:
                    # FlushedMetrics → dict of scalar floats; sink emits a `step`
                    # event (no _cmmd key, so it's not misread as a val pass).
                    progress_sink.log(
                        dataclasses.asdict(m), global_step=step + 1, epoch=0
                    )
                metrics.reset()
                ctx.tau_profile.write(writer, step + 1)

            # --- diversity validation (DAVE same-prompt probe) ---
            if (
                ctx.val_cond is not None
                and (step + 1) % cfg.validate_every_n_steps == 0
            ):
                dm = run_diversity_validation(
                    model=ctx.model,
                    forward_fn=_forward,
                    student_sigmas=student_sigmas,
                    crossattn_emb=ctx.val_cond,
                    latent_shape=ctx.val_latent_shape,
                    num_seeds=cfg.val_diversity_seeds,
                    seed0=cfg.seed,
                    device=device,
                    dtype=dtype,
                    clean_latent=ctx.val_clean,
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

            if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
                n = step + 1
                is_final = n == cfg.iterations
                metadata = _student_metadata(cfg, ctx, n)
                # Step-tagged files go to a per-run subdir; the final step also
                # writes the bare {output_name} at the root, where inference and
                # merge look for it.
                ckpt_subdir = Path(cfg.output_dir) / cfg.output_name
                ckpt_subdir.mkdir(parents=True, exist_ok=True)
                save_paths = [
                    str(ckpt_subdir / f"{cfg.output_name}_{_step_tag(n)}.safetensors")
                ]
                if is_final:
                    save_paths.append(
                        str(Path(cfg.output_dir) / f"{cfg.output_name}.safetensors")
                    )
                for save_path in save_paths:
                    turbo.save_student(
                        save_path, dtype=torch.bfloat16, metadata=metadata
                    )
                    logger.info(f"saved checkpoint: {save_path}")
                    if progress_sink is not None:
                        progress_sink.ckpt(global_step=n, path=save_path)

                # Rolling crash-resume bundle (~10× a student file); none at the end.
                if not is_final:
                    rp = resume_path_for(cfg.output_dir, cfg.output_name)
                    save_resume_state(
                        rp,
                        step=n,
                        cfg=cfg,
                        turbo=turbo,
                        student_opt=student_opt,
                        fake_opt=fake_opt,
                        disc_opt=disc_opt,
                        student_sched=student_sched,
                        fake_sched=fake_sched,
                        disc_sched=disc_sched,
                    )
                    logger.info(f"saved resume bundle: {rp} (step {n})")

    if cfg.int8 != "off":
        from library.anima.int8 import finish_training

        finish_training()
    if writer is not None:
        writer.close()
    logger.info("turbo distillation complete.")


if __name__ == "__main__":
    main()
