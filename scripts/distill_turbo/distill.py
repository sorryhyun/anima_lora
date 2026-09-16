"""Turbo distillation main loop — DP-DMD (diversity-preserved DMD).

Usage:
    python -m scripts.distill_turbo.distill [--config configs/methods/turbo.toml] ...

The math walkthrough lives in :mod:`scripts.distill_turbo`; this file is the
per-step orchestrator (teacher K-step anchor → diversity-supervised first step →
DMD-refined N-step student rollout → fake/critic update → save). Run construction
(model, adapters, optimizers, dataloader, resume, warmup) lives in
:mod:`scripts.distill_turbo.setup`; ``run_loop`` below consumes the ``RunContext``
it returns.
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
import torch.nn as nn
from tqdm import tqdm

from library.anima.uncond import uncond_for_batch
from library.training.progress import run_scope

from .config import build_argparser, load_turbo_config, resolve_config
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
from .softrank import caption_rank_loss
from .steps import (
    cdm_off_trajectory_loss,
    dmd_surrogate,
    fake_update,
    gan_generator_term,
    teacher_anchor,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _step_tag(step: int) -> str:
    """Human checkpoint suffix: 1000 -> ``1k``, 8000 -> ``8k``, else raw count."""
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def main():
    args = build_argparser().parse_args()
    cfg = resolve_config(args, load_turbo_config(args.config))
    ctx = build_run(args, cfg)
    run_loop(ctx, cfg)


def run_loop(ctx: RunContext, cfg):
    """Per-step DP-DMD training loop over the objects built by ``build_run``.

    ``ctx`` fields are bound to locals up front so the loop body reads as the
    plain algorithm; only ``data_iter`` and ``fdistill_bins`` are mutated
    (epoch re-iter / f-distill EMA), and neither is read after the loop.
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
    use_anchor = ctx.use_anchor
    softrank_on = ctx.softrank_on
    softrank_pool = ctx.softrank_pool
    cdm_on = ctx.cdm_on
    fdistill_bins = ctx.fdistill_bins
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
            if cfg.use_masked_loss:
                # float (not bf16): the student loss is assembled in fp32. [B,1,H,W]
                # broadcasts over the [B,16,H,W] grad signal.
                mask = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)
            else:
                mask = None

            latents = latents.to(device, dtype=dtype, non_blocking=True)
            crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
            B = latents.shape[0]

            # No-op today (mode="default" doesn't enable cudagraphs), but the right
            # cadence if/when the script switches to "reduce-overhead".
            torch.compiler.cudagraph_mark_step_begin()

            # Student update: roll an N-step Euler grid from pure noise ε (dpdmd
            # anchors step 1 to a teacher K-step target then refines; dmd is plain).
            eps = torch.randn_like(latents)  # shared start for anchor + student
            c_null = uncond_for_batch(
                ctx.uncond_base, crossattn_emb
            )  # anchor + DMD eval

            # --- teacher K-step CFG anchor (no grad) → v_target (DP-DMD only) ---
            v_target = None
            if use_anchor:
                v_target = teacher_anchor(ctx, cfg, eps, crossattn_emb, c_null, B)

            # --- student rollout → x_pred (= x_θ, B,16,H,W) + v_student (metric) ---
            # dpdmd: step-0 diversity anchor + DMD-refined steps 1..N-1.
            # dmd:   plain DMD2; cfg.dmd_grad_step picks which step(s) grad.
            split_bwd = use_anchor and cfg.detach_after_first
            # This iteration's rollout grid: the static inference grid, or a fresh
            # CDM dynamic draw. Everything below indexes sigmas_it/n_steps_it so
            # the two modes share one code path.
            if cfg.dynamic_schedule:
                sigmas_it = sample_dynamic_sigmas(ctx.dyn_n_min, cfg.student_steps)
                n_steps_it = len(sigmas_it) - 1
            else:
                sigmas_it, n_steps_it = student_sigmas, cfg.student_steps
            last_step = n_steps_it - 1

            # Soft-rank caption auxiliary state (metrics + non-split backward read it
            # even on steps/branches where the term doesn't fire). Zero leaf ⇒ adds
            # nothing to any backward when off/skipped.
            softrank_loss = torch.zeros((), device=device)
            softrank_ran = False

            # L_CDM launch point: the DMD grad step's on-trajectory (x_in, v, σ),
            # captured raw here and detached at use (cdm_extrapolate). Under
            # grad_step='random' the launch point sweeps the whole grid over
            # training; under 'all'/'last' it is the final (cleanest-σ) step.
            cdm_src = None

            if use_anchor:
                # Step 0 is the diversity anchor (supervised toward v_target, then
                # detached under split_bwd); steps 1..N-1 carry the DMD-refine grad,
                # routed by grad_step ('all' BPTT | 'last' tail-only | 'random' grid).
                x = eps
                x.requires_grad_()  # grad-ckpt needs a grad-requiring forward input
                s0, s0_next = sigmas_it[0], sigmas_it[1]
                t_b = torch.full((B,), s0, device=device, dtype=dtype)
                turbo.set_student_step(0)  # head 0 (no-op unless per-step-expert)
                # Dual-pool routing: gate grads to pool A so the step-0 diversity
                # (+ soft-rank) backward below lands on A alone (no-op single-pool).
                turbo.route_div()
                v_first = _forward(
                    "student", x, t_b, crossattn_emb, no_grad=False
                ).squeeze(2)
                x = x - (s0 - s0_next) * v_first
                div_loss_t = nn.functional.mse_loss(v_first.float(), v_target)

                # --- soft-rank caption-discrimination auxiliary -------------------
                # k extra no_grad student forwards at the SAME step-0 (ε, t=1, head 0),
                # only crossattn_emb swapped for a pooled sample's caption. Soft-rank
                # the matched caption's position against these negatives and push it
                # to 0 — grad flows through v_first ONLY (negatives detached), so the
                # term stays bounded and rides the step-0 backward below.
                if softrank_on:
                    if step % cfg.softrank_every_n == 0 and softrank_pool.ready(
                        ctx.softrank_min_pool
                    ):
                        # Pool negatives → works at any batch size (B=1 included). Head
                        # 0 stays selected → no per-step-expert recompute hazard.
                        v_negs = [
                            _forward("student", eps, t_b, c_neg, no_grad=True).squeeze(
                                2
                            )
                            for c_neg in softrank_pool.draw(cfg.softrank_k, B)
                        ]
                        softrank_loss = caption_rank_loss(
                            v_first, v_negs, v_target, tau=cfg.softrank_softness
                        )
                        softrank_ran = True
                    # Fill AFTER drawing so an anchor never draws its own caption.
                    softrank_pool.add(crossattn_emb)

                if split_bwd:
                    # Load-bearing stop-grad: the DMD reverse-KL (steps 1..N-1) must NOT
                    # flow into the diversity mapping (Fig 5). Backward the diversity
                    # term now, then re-leaf for a fresh DMD-chain root. Soft-rank joins
                    # this backward (rides v_first's step-0 graph), so the DMD graph
                    # separation stays intact.
                    (
                        cfg.div_weight * div_loss_t
                        + cfg.softrank_weight * softrank_loss
                    ).backward()
                    softrank_loss = softrank_loss.detach()  # metrics-only from here
                    x = x.detach().requires_grad_()
                    # Route grads to pool B for everything downstream: the DMD
                    # rollout, the L_CDM branch, and the GAN generator all build
                    # graphs that must reach B alone (no-op single-pool).
                    turbo.route_quality()
                if cfg.dmd_grad_step == "random":
                    # Memory-flat anchored DMD: sample ONE refinement step g~U{1..N-1},
                    # backward-simulate the prefix under no_grad, grad only step g's
                    # one-step x0-prediction (x_g − σ_g·v_g). Supervises every grid
                    # point over training (vs 'last') and trains head g under
                    # per_step_expert. Step 0's diversity graph rides v_first untouched.
                    g = int(torch.randint(1, n_steps_it, (1,)).item())
                    for i in range(1, g):  # backward simulation (no graph kept)
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
                        t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                        turbo.set_student_step(i)
                        v = _forward(
                            "student", x, t_b, crossattn_emb, no_grad=True
                        ).squeeze(2)
                        x = x - (s_i - s_next) * v
                    x = x.detach().requires_grad_()  # fresh leaf; head g trains
                    s_g = sigmas_it[g]
                    t_b = torch.full((B,), s_g, device=device, dtype=dtype)
                    turbo.set_student_step(g)
                    v_g = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=False
                    ).squeeze(2)
                    if cdm_on:
                        cdm_src = (x, v_g, s_g)
                    x_pred = x - s_g * v_g  # one-step x0-prediction at step g
                else:
                    # 'all' → full BPTT over 1..N-1; else ('last') → only the final step
                    # grads (1..N-2 backward-simulated under no_grad). Both memory-flat
                    # except 'all', and land the DMD grad on the true rollout endpoint.
                    grad_dmd_last_only = cfg.dmd_grad_step != "all"
                    for i in range(1, n_steps_it):
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
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
                        if cdm_on and i == last_step:
                            cdm_src = (x, v, s_i)
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
                    for i in range(n_steps_it):
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
                        t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                        turbo.set_student_step(i)
                        v = _forward(
                            "student", x, t_b, crossattn_emb, no_grad=False
                        ).squeeze(2)
                        if v_student is None:
                            v_student = v
                        if cdm_on and i == last_step:
                            cdm_src = (x, v, s_i)
                        x = x - (s_i - s_next) * v
                    x_pred = x
                else:
                    # Single grad-step: 'last' pins g=N-1; 'random' samples g~U{0..N-1}
                    # (canonical DMD2 — supervises every grid point, not just the clean
                    # tail). Roll to g under no_grad, grad ONLY step g, supervise its
                    # one-step x0-prediction x_g − σ_g·v_g. Memory-flat (1 forward graph).
                    if cfg.dmd_grad_step == "random":
                        # CPU RNG → no per-step GPU sync (seeded by torch.manual_seed).
                        g = int(torch.randint(0, n_steps_it, (1,)).item())
                    else:
                        g = last_step
                    x = eps
                    for i in range(g):  # backward simulation (no_grad → no graph kept)
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
                        t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                        turbo.set_student_step(i)
                        v = _forward(
                            "student", x, t_b, crossattn_emb, no_grad=True
                        ).squeeze(2)
                        x = x - (s_i - s_next) * v
                    x = x.detach().requires_grad_()  # fresh leaf; head g trains
                    s_g = sigmas_it[g]
                    t_b = torch.full((B,), s_g, device=device, dtype=dtype)
                    turbo.set_student_step(g)
                    v_g = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=False
                    ).squeeze(2)
                    if cdm_on:
                        cdm_src = (x, v_g, s_g)
                    x_pred = x - s_g * v_g  # one-step x0-prediction at step g
                    v_student = v_g

            # --- DMD on x_θ (steps 2..N), against teacher + fake ---
            dmd = dmd_surrogate(ctx, cfg, x_pred, crossattn_emb, c_null, B)
            grad_signal = dmd.grad_signal
            delta_dm = dmd.delta_dm
            tau_dm = dmd.tau_dm
            tau_dm_e = dmd.tau_dm_e
            v_real_cond_dm = dmd.v_real_cond_dm
            v_fake_cond_dm = dmd.v_fake_cond_dm
            eps_dm = dmd.eps_dm

            # --- L_CDM off-trajectory loss ---
            # Encapsulated in steps.cdm_off_trajectory_loss (extrapolate → one grad
            # forward → real-vs-fake surrogate → in-branch backward, view restored to
            # student, + metrics.add_cdm). ORDER MATTERS: it MUST run BEFORE the GAN
            # gen forward — that forward's checkpointed recompute happens at backward
            # under the then-current view, so CDM must be the last view flip before it.
            if cdm_on and cdm_src is not None:
                cdm_off_trajectory_loss(
                    ctx, cfg, cdm_src, crossattn_emb, c_null, latents, mask, B
                )

            # --- GAN generator term + f-distill reweighting ---
            # No-op when the GAN is off or the delay/warmup ramp still holds the
            # generator-side λ at 0 (disc keeps training below); otherwise returns
            # the gen loss, the (possibly f-distill-reweighted) grad_signal, and
            # the updated EMA bins.
            gan_w = gan_effective_weight(cfg, step)
            gan_gen_loss, grad_signal, fdistill_bins = gan_generator_term(
                ctx,
                cfg,
                x_pred,
                tau_dm,
                eps_dm,
                crossattn_emb,
                grad_signal,
                fdistill_bins,
                B,
                gan_w,
            )

            # --- assemble: DMD surrogate on x_θ ---
            # The diversity term was already backwarded above when split_bwd; otherwise
            # it rides this combined backward (graphs still entangled). grad_clip below
            # runs once on the ACCUMULATED .grad (div + DMD), so the clipped norm is the
            # full student gradient either way.
            if mask is not None:
                loss_dmd = (grad_signal * x_pred.float() * mask).mean()
            else:
                loss_dmd = (grad_signal * x_pred.float()).mean()
            loss_student = loss_dmd

            if use_anchor and not split_bwd:
                # div + soft-rank both ride v_first's retained step-0 graph here (no
                # split backward), so they join the combined student backward below.
                loss_student = (
                    loss_student
                    + cfg.div_weight * div_loss_t
                    + cfg.softrank_weight * softrank_loss
                )

            if turbo.disc is not None and gan_w > 0.0:
                loss_student = loss_student + gan_w * gan_gen_loss

            loss_student.backward()
            # Restore grad-on for both dual pools BEFORE clip/step: student_params()
            # filters requires_grad, so a still-gated pool's grads (pool A's, from the
            # step-0 backward) would otherwise escape clipping (no-op single-pool).
            turbo.route_all_on()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    turbo.student_params(), max_norm=cfg.grad_clip
                )
            student_opt.step()
            student_opt.zero_grad(set_to_none=True)
            student_sched.step()

            # --- fake (critic) + discriminator update against x_pred.detach() ---
            # Runs the fake + disc optimizer/scheduler steps in-place; returns the
            # mean fake / disc loss over the inner steps for logging.
            fake_loss_mean_t, gan_disc_mean_t, gan_margin_t, gan_spread_t = fake_update(
                ctx, cfg, x_pred, latents, crossattn_emb, B
            )

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
            if turbo.disc is not None:
                metrics.add_gan(
                    gan_gen_loss, gan_disc_mean_t, gan_margin_t, gan_spread_t
                )
            if softrank_on:
                metrics.add_softrank(softrank_loss, active=softrank_ran)

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
                for tp in ctx.tau_profiles:
                    tp.write(writer, step + 1)

            # --- diversity validation (DAVE same-prompt probe) ---
            if (
                ctx.val_cond is not None
                and (step + 1) % cfg.validate_every_n_steps == 0
            ):
                dm = run_diversity_validation(
                    model=ctx.model,
                    forward_fn=_forward,
                    set_student_step=turbo.set_student_step,
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

            # Each checkpoint is kept under a step-tagged name (no overwrite, so the
            # whole trajectory survives); the final step also writes the canonical
            # bare `{output_name}` that inference / merge / `make test` look for.
            if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
                n = step + 1
                is_final = n == cfg.iterations
                metadata = {
                    "ss_turbo_objective": cfg.base_loss,
                    "ss_turbo_student_rank": str(cfg.student_rank),
                    "ss_turbo_student_alpha": str(cfg.student_alpha),
                    "ss_turbo_student_steps": str(cfg.student_steps),
                    "ss_turbo_dynamic_schedule": "1" if cfg.dynamic_schedule else "0",
                    "ss_turbo_teacher_cfg": str(cfg.teacher_cfg),
                    "ss_turbo_step": str(n),
                    "ss_turbo_k_anchor": str(cfg.k_anchor),
                    "ss_turbo_div_weight": str(cfg.div_weight),
                    "ss_turbo_gan_weight_gen": str(cfg.gan_loss_weight_gen),
                    "ss_turbo_gan_disc_head": cfg.gan_disc_head,
                    "ss_turbo_gan_delay_steps": str(cfg.gan_delay_steps),
                    "ss_turbo_gan_warmup_steps": str(cfg.gan_warmup_steps),
                    "ss_turbo_cdm_weight": str(cfg.cdm_weight),
                    "ss_turbo_f_div": cfg.f_div,
                }
                if cfg.train_adaln:
                    # The student targets adaln_up_{branch}; save_student ships the
                    # adaln keys in the ComfyUI layout (adaln.md).
                    metadata["ss_turbo_train_adaln"] = "1"
                    if cfg.adaln_rank > 0:
                        # Provenance only — per-module rank/alpha live in the file.
                        metadata["ss_turbo_adaln_rank"] = str(cfg.adaln_rank)
                if cfg.student_init_weights:
                    # Provenance only — the warm start distills to a normal LoRA.
                    metadata["ss_turbo_student_init_weights"] = os.path.basename(
                        cfg.student_init_weights
                    )
                if cfg.per_step_expert:
                    # Drives loader detection (CLI + ComfyUI build StepExpertLoRAModule
                    # and keep it live instead of merging). step_expert_K == the head
                    # count == student_steps.
                    metadata["ss_turbo_per_step_expert"] = "1"
                    metadata["ss_turbo_step_expert_K"] = str(cfg.step_expert_K)
                # Step-tagged intermediates live in a per-run subdir so they don't
                # clutter output/ckpt/; the canonical bare {output_name} stays at the
                # root where inference / merge / `make test` look for it.
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

                # Crash-resume bundle: everything save_student drops on the floor (fake,
                # disc, three optimizers, three schedulers, f-distill EMA, RNG). Rolling
                # single file, written atomically — see resume.py. Skipped on the final
                # step: the run is complete, and the bundle is ~10× a student ckpt.
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
                        fdistill_bins=fdistill_bins,
                    )
                    logger.info(f"saved resume bundle: {rp} (step {n})")

    if writer is not None:
        writer.close()
    logger.info("turbo distillation complete.")


if __name__ == "__main__":
    main()
