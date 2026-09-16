"""Training-loop orchestration.

Owns the per-epoch / per-step body of ``AnimaTrainer.train()``. The entrypoint
is :func:`run_training_loop`, which takes a built :class:`LoopState` plus the
trainer instance so override hooks (``process_batch``, ``on_step_start``,
``sample_images``, ``generate_step_logs``, ``step_logging``,
``epoch_logging``) stay overridable. The validation pass lives in
:mod:`library.training.validation`.

Cross-call signaling state — ``_last_router_H_postfix``,
``_cudagraph_mark_step``, ``_hydra_warmup_step``, ``_adapters`` — lives on the
trainer; this module reads it through the ``trainer`` handle.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from library import train_util
from library.datasets import LossRecorder
from library.runtime.device import clean_memory_on_device
from library.training.checkpoints import CheckpointSaver
from library.training.contexts import TrainCtx, ValCtx
from library.training.method_adapter import StepCtx
from library.training.metrics import MetricContext, collect_metrics
from library.training.validation import run_validation

logger = logging.getLogger(__name__)

# Liveness early check: late enough that warmups / partial
# sidecar coverage have had a chance to fire at least once, early enough that
# a silently-dead feature aborts a strict run in minutes instead of hours.
LIVENESS_EARLY_CHECK_STEP = 25


@dataclass
class LoopState:
    """Bundles the locals of ``train()``'s for-epoch scope.

    Most fields are constants for the run; ``global_step``, ``profile_started``,
    ``profile_range``, ``initial_step``, and ``text_encoder(s)`` are mutated
    during the loop. ``current_epoch`` / ``current_step`` are mp.Value handles
    shared with :class:`CheckpointSaver` for state persistence.
    """

    args: Any
    accelerator: Accelerator
    train_ctx: TrainCtx
    val_ctx: ValCtx
    saver: CheckpointSaver

    network: Any
    unet: Any
    text_encoder: Any
    text_encoders: list
    vae: Any
    tokenizers: Any
    training_model: Any
    train_dataloader: Any
    optimizer: Any
    lr_scheduler: Any
    lr_descriptions: Optional[list]
    optimizer_train_fn: Callable
    optimizer_eval_fn: Callable
    weight_dtype: Any
    unet_weight_dtype: Any

    current_epoch: Any  # mp.Value
    current_step: Any  # mp.Value
    num_train_epochs: int
    epoch_to_start: int
    initial_step: int

    metadata: dict
    is_tracking: bool
    progress_bar: Any
    loss_recorder: LossRecorder
    val_step_loss_recorder: LossRecorder
    val_epoch_loss_recorder: LossRecorder

    validation_steps: int

    profile_range: Optional[tuple]
    on_step_start_for_network: Callable

    global_step: int = 0
    profile_started: bool = False


def build_loop_state(
    trainer,
    *,
    args,
    accelerator: Accelerator,
    saver: CheckpointSaver,
    network,
    unet,
    text_encoder,
    text_encoders,
    vae,
    tokenizers,
    training_model,
    train_dataloader,
    val_dataloader,
    val_dataset_group,
    optimizer,
    lr_scheduler,
    lr_descriptions,
    optimizer_train_fn,
    optimizer_eval_fn,
    weight_dtype,
    unet_weight_dtype,
    vae_dtype,
    text_encoding_strategy,
    tokenize_strategy,
    train_text_encoder,
    train_unet,
    current_epoch,
    current_step,
    num_train_epochs,
    epoch_to_start,
    initial_step,
    metadata,
) -> LoopState:
    """Build :class:`LoopState`: the pre-loop setup between
    ``_prepare_with_accelerator()`` and the for-epoch loop — noise scheduler, trackers, loss recorders, optional text
    encoder eviction, ``--sample_at_first``, train/val ctx construction,
    progress bar, profiler parsing.
    """
    noise_scheduler = trainer.get_noise_scheduler(args, accelerator.device)

    train_util.init_trackers(accelerator, args, "network_train")

    loss_recorder = LossRecorder()
    val_step_loss_recorder = LossRecorder()
    val_epoch_loss_recorder = LossRecorder()

    if hasattr(accelerator.unwrap_model(network), "on_step_start"):
        on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
    else:

        def on_step_start_for_network(*args, **kwargs):
            return None

    if trainer.is_text_encoder_not_needed_for_training(args):
        logger.info("text_encoder is not needed for training. deleting to save memory.")
        for t_enc in text_encoders:
            del t_enc
        text_encoders = []
        text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    optimizer_eval_fn()
    trainer.sample_images(
        accelerator,
        args,
        0,
        0,
        accelerator.device,
        vae,
        tokenizers,
        text_encoder,
        unet,
        network=network,
    )
    optimizer_train_fn()
    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        accelerator.log({}, step=0)

    train_ctx = TrainCtx(
        args=args,
        accelerator=accelerator,
        network=network,
        unet=unet,
        vae=vae,
        text_encoders=text_encoders,
        noise_scheduler=noise_scheduler,
        text_encoding_strategy=text_encoding_strategy,
        tokenize_strategy=tokenize_strategy,
        vae_dtype=vae_dtype,
        weight_dtype=weight_dtype,
        train_text_encoder=train_text_encoder,
        train_unet=train_unet,
        optimizer_eval_fn=optimizer_eval_fn,
        optimizer_train_fn=optimizer_train_fn,
        is_tracking=is_tracking,
    )

    # Resume skip prelude: fast-forward global_step before tqdm so the bar
    # total is sized right, and consume per-epoch skip credit so
    # skip_first_batches has the right first-epoch offset.
    global_step = 0
    if initial_step > 0:
        global_step = initial_step // args.gradient_accumulation_steps
        for skip_epoch in range(epoch_to_start):
            logger.info(
                f"skipping epoch {skip_epoch + 1} because initial_step "
                f"(multiplied) is {initial_step}"
            )
            initial_step -= len(train_dataloader)

    logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
    _ts_parts = [f"timestep_sampling={args.timestep_sampling}"]
    if args.timestep_sampling in ("sigmoid", "shift", "flux_shift"):
        _ts_parts.append(f"sigmoid_scale={args.sigmoid_scale}")
        _ts_parts.append(f"sigmoid_bias={getattr(args, 'sigmoid_bias', 0.0)}")
    if args.timestep_sampling in ("shift", "flux_shift"):
        _ts_parts.append(f"discrete_flow_shift={args.discrete_flow_shift}")
    if (
        getattr(args, "t_min", None) is not None
        or getattr(args, "t_max", None) is not None
    ):
        _ts_parts.append(
            f"σ∈[{getattr(args, 't_min', None)}, {getattr(args, 't_max', None)}]"
        )
    logger.info("sigma sampling: " + ", ".join(_ts_parts))
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()
        params_itr.__next__()  # CLIP first two params are embeddings
        param_3rd = params_itr.__next__()
        logger.info(
            f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}"
        )

    clean_memory_on_device(accelerator.device)

    progress_bar = tqdm(
        range(args.max_train_steps - global_step),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    validation_steps = (
        min(args.max_validation_steps, len(val_dataloader))
        if args.max_validation_steps is not None
        else len(val_dataloader)
    )
    # Fixed sigma values across the schedule: 0.1 near-clean / fine detail,
    # 0.4 mid / bulk structure, 0.7 high noise / coarse denoising.
    validation_sigmas = (
        args.validation_sigmas
        if args.validation_sigmas is not None
        else [0.1, 0.4, 0.7]
    )
    val_ctx = ValCtx(
        dataloader=val_dataloader,
        sigmas=validation_sigmas,
        steps=validation_steps,
        total_steps=validation_steps * len(validation_sigmas),
        train_loss_recorder=loss_recorder,
        original_t_min=args.t_min,
        original_t_max=args.t_max,
        dataset_group=val_dataset_group,
    )

    # nsys workflow: --profile_steps START-END toggles the cuda profiler API
    # around the requested step window. Wrap the launch with
    #   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
    # so nsys only records that window.
    profile_range = trainer._parse_profile_steps(args)

    return LoopState(
        args=args,
        accelerator=accelerator,
        train_ctx=train_ctx,
        val_ctx=val_ctx,
        saver=saver,
        network=network,
        unet=unet,
        text_encoder=text_encoder,
        text_encoders=text_encoders,
        vae=vae,
        tokenizers=tokenizers,
        training_model=training_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_descriptions=lr_descriptions,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        weight_dtype=weight_dtype,
        unet_weight_dtype=unet_weight_dtype,
        current_epoch=current_epoch,
        current_step=current_step,
        num_train_epochs=num_train_epochs,
        epoch_to_start=epoch_to_start,
        initial_step=initial_step,
        metadata=metadata,
        is_tracking=is_tracking,
        progress_bar=progress_bar,
        loss_recorder=loss_recorder,
        val_step_loss_recorder=val_step_loss_recorder,
        val_epoch_loss_recorder=val_epoch_loss_recorder,
        validation_steps=validation_steps,
        profile_range=profile_range,
        on_step_start_for_network=on_step_start_for_network,
        global_step=global_step,
    )


def run_training_loop(trainer, state: LoopState) -> None:
    """Run the full for-epoch training loop and the post-loop end-of-training
    metadata write. Mutates ``state.global_step``, profiler bookkeeping, and
    the metadata dict; the per-checkpoint saves go through ``state.saver``.
    """
    args = state.args
    accelerator = state.accelerator

    for epoch in range(state.epoch_to_start, state.num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{state.num_train_epochs}\n")
        state.current_epoch.value = epoch + 1
        state.metadata["ss_epoch"] = str(epoch + 1)

        accelerator.unwrap_model(state.network).on_epoch_start(
            state.text_encoder, state.unet
        )

        _run_epoch_steps(trainer, state, epoch)
        _run_epoch_validation(trainer, state, epoch)
        _log_epoch_average(trainer, state, epoch)
        _run_adapter_epoch_hooks(trainer, state)

        accelerator.wait_for_everyone()

        state.optimizer_eval_fn()
        state.saver.maybe_save_epoch(
            state.network, state.global_step, epoch, state.num_train_epochs
        )
        state.saver.maybe_save_resumable(
            state.network, state.global_step, epoch, state.num_train_epochs
        )

        trainer.sample_images(
            accelerator,
            args,
            epoch + 1,
            state.global_step,
            accelerator.device,
            state.vae,
            state.tokenizers,
            state.text_encoder,
            state.unet,
            network=state.network,
        )
        state.optimizer_train_fn()

    _audit_liveness(trainer, state, where="run end")

    state.metadata["ss_training_finished_at"] = str(time.time())


def _run_epoch_steps(trainer, state: LoopState, epoch: int) -> None:
    """Inner per-step loop: walk the dataloader, execute the accumulate
    scope, run sample / save / log / step-validation ticks."""
    args = state.args
    accelerator = state.accelerator

    skipped_dataloader = None
    if state.initial_step > 0:
        skipped_dataloader = accelerator.skip_first_batches(
            state.train_dataloader, state.initial_step - 1
        )
        state.initial_step = 1

    for step, batch in enumerate(skipped_dataloader or state.train_dataloader):
        state.current_step.value = state.global_step
        if state.initial_step > 0:
            state.initial_step -= 1
            continue

        _profiler_step_begin(state)

        loss = _run_step(trainer, state, batch)

        _profiler_step_end(state)

        keys_scaled, mean_norm, maximum_norm, max_mean_logs = _maybe_scale_norm(state)

        if accelerator.sync_gradients:
            state.progress_bar.update(1)
            state.global_step += 1
            if state.global_step == LIVENESS_EARLY_CHECK_STEP:
                _audit_liveness(
                    trainer, state, where=f"step {state.global_step} early check"
                )
            _sample_at_step(trainer, state)
            state.saver.maybe_save_step(state.network, state.global_step, epoch)
            state.optimizer_train_fn()

        _log_step(
            trainer,
            state,
            loss=loss,
            step=step,
            epoch=epoch,
            keys_scaled=keys_scaled,
            mean_norm=mean_norm,
            maximum_norm=maximum_norm,
            max_mean_logs=max_mean_logs,
        )
        _maybe_run_step_validation(trainer, state, epoch)

        if state.global_step >= args.max_train_steps:
            break


def _run_step(trainer, state: LoopState, batch) -> torch.Tensor:
    """The accumulate-scope body: on_step_start hooks, cudagraph mark, forward,
    backward gating, sync_gradients hooks (hydra warmup, grad capture, clip),
    optimizer step + zero_grad. Returns the loss (detached or live)."""
    args = state.args
    accelerator = state.accelerator
    network = state.network

    with accelerator.accumulate(state.training_model):
        state.on_step_start_for_network(state.text_encoder, state.unet)

        trainer.on_step_start(state.train_ctx, batch, is_train=True)

        # Clear last-step gate/σ refs + memoized router-stats caches before the
        # next forward. Unconditional: cudagraph needs it (lingering refs into
        # the cudagraph pool block reclamation → demotes to eager), and eager
        # needs it so per-step memoized stats invalidate instead of freezing at
        # their first values.
        net_unwrapped = accelerator.unwrap_model(network)
        if hasattr(net_unwrapped, "clear_step_caches"):
            net_unwrapped.clear_step_caches()

        # CUDAGraphs need an explicit iteration boundary before the forward
        # every step; without it the "pending, uninvoked backwards" fast-path
        # check fails and cudagraphs silently fall back to eager.
        if trainer._cudagraph_mark_step:
            torch.compiler.cudagraph_mark_step_begin()

        if state.profile_started:
            torch.cuda.nvtx.range_push("forward")
        loss = trainer.process_batch(state.train_ctx, batch, is_train=True)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        if state.profile_started:
            torch.cuda.nvtx.range_push("backward")
        accelerator.backward(loss)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        # Post-backward adapter hook (before clip/step) — injects extra grad
        # contributions that can't share the primary backward, e.g. soft-tokens
        # gradient-cached contrastive negatives under active block swapping.
        trainer.run_after_backward(state.train_ctx)

        if accelerator.sync_gradients:
            net_unwrapped = accelerator.unwrap_model(network)
            # Snapshot Hydra up-weight grad norms before zero_grad wipes them
            # (metric ``hydra_up_grad`` reads it later). Pre-clip so magnitudes
            # aren't distorted by the global rescale. Log-cadence only;
            # global_step increments below so predict the post-increment value.
            _log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
            _will_log_after = state.is_tracking and (
                ((state.global_step + 1) % _log_every == 0)
                or ((state.global_step + 1) >= args.max_train_steps)
            )
            if _will_log_after and hasattr(net_unwrapped, "capture_up_grad_stats"):
                net_unwrapped.capture_up_grad_stats()
            if args.max_grad_norm != 0.0:
                params_to_clip = accelerator.unwrap_model(
                    network
                ).get_trainable_params()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

        if state.profile_started:
            torch.cuda.nvtx.range_push("optimizer")
        state.optimizer.step()
        state.lr_scheduler.step()
        state.optimizer.zero_grad(set_to_none=True)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

    return loss


def _profiler_step_begin(state: LoopState) -> None:
    if (
        state.profile_range
        and state.global_step == state.profile_range[0]
        and not state.profile_started
    ):
        state.accelerator.print(f"\n[profiler] starting at step {state.global_step}")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        state.profile_started = True

    if state.profile_started:
        torch.cuda.nvtx.range_push(f"step={state.global_step}")


def _profiler_step_end(state: LoopState) -> None:
    if state.profile_started:
        torch.cuda.nvtx.range_pop()
    if state.profile_started and state.global_step >= state.profile_range[1]:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        state.accelerator.print(f"\n[profiler] stopped at step {state.global_step}")
        state.accelerator.print(
            "[profiler] open the .nsys-rep with the Nsight Systems GUI\n"
        )
        state.profile_started = False
        state.profile_range = None  # don't re-trigger
        # Hard-exit so the launcher exits and nsys finalizes the report.
        # sys.exit(0) hangs in interpreter shutdown (DataLoader workers +
        # NCCL/CUDA atexit handlers wait on futexes); the profile buffer is
        # already flushed by the preceding synchronize() + cuProfilerStop.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _maybe_scale_norm(state: LoopState):
    args = state.args
    if args.scale_weight_norms:
        keys_scaled, mean_norm, maximum_norm = state.accelerator.unwrap_model(
            state.network
        ).apply_max_norm_regularization(
            args.scale_weight_norms, state.accelerator.device
        )
        max_mean_logs = {
            "Keys Scaled": keys_scaled,
            "Average key norm": mean_norm,
        }
        return keys_scaled, mean_norm, maximum_norm, max_mean_logs
    return None, None, None, {}


def _sample_at_step(trainer, state: LoopState) -> None:
    state.optimizer_eval_fn()
    trainer.sample_images(
        state.accelerator,
        state.args,
        None,
        state.global_step,
        state.accelerator.device,
        state.vae,
        state.tokenizers,
        state.text_encoder,
        state.unet,
        network=state.network,
    )


def _log_step(
    trainer,
    state: LoopState,
    *,
    loss,
    step: int,
    epoch: int,
    keys_scaled,
    mean_norm,
    maximum_norm,
    max_mean_logs,
) -> None:
    args = state.args
    log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
    # Gate on sync_gradients: with gradient_accumulation_steps > 1 this hook
    # fires per micro-batch but global_step advances only on sync; without the
    # gate log_every_n_steps gives the same answer for every micro-batch,
    # bursting N tracker writes then N silent ones.
    should_log_step = state.accelerator.sync_gradients and (
        (state.global_step % log_every == 0)
        or (state.global_step >= args.max_train_steps)
    )

    current_loss = loss.detach().item()
    state.loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
    avr_loss: float = state.loss_recorder.moving_average
    logs = {"avr_loss": avr_loss}
    _unwrapped_net = state.accelerator.unwrap_model(state.network)
    # Refresh router_H only on log cadence — get_router_entropy does a full
    # get_router_stats compute (D2H syncs) wasted on the progress-bar postfix;
    # tqdm shows a harmlessly-stale cached value between log steps.
    if getattr(_unwrapped_net, "_use_hydra", False) and should_log_step:
        _router_H = _unwrapped_net.get_router_entropy()
        if _router_H is not None:
            trainer._last_router_H_postfix = _router_H
    _router_H_cached = getattr(trainer, "_last_router_H_postfix", None)
    if _router_H_cached is not None:
        logs["router_H"] = f"{_router_H_cached:.3f}"
    state.progress_bar.set_postfix(refresh=False, **{**max_mean_logs, **logs})

    # The progress sink (GUI / daemon tails progress.jsonl) needs `step` events
    # even with no tracker. When tracking, step_logging below already feeds it;
    # emit a lightweight event directly only when untracked so the bar advances
    # without the full generate_step_logs + collect_metrics path.
    progress_sink = getattr(trainer, "progress_sink", None)
    if should_log_step and not state.is_tracking and progress_sink is not None:
        progress_sink.log(logs, global_step=state.global_step, epoch=epoch + 1)

    if state.is_tracking and should_log_step:
        logs = trainer.generate_step_logs(
            args,
            current_loss,
            avr_loss,
            state.lr_scheduler,
            state.lr_descriptions,
            state.optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            None,  # mean_grad_norm — not tracked here
            None,  # mean_combined_norm — not tracked here
        )
        producers = [_unwrapped_net, *trainer._adapters]
        # Ledger is a MetricProducer — liveness/<name> coverage is the live
        # view of the run-end LIVENESS audit.
        _ledger = getattr(trainer, "_liveness", None)
        if _ledger is not None:
            producers.append(_ledger)
        logs.update(
            collect_metrics(
                producers,
                MetricContext(args=args, network=_unwrapped_net),
            )
        )
        trainer.step_logging(state.accelerator, logs, state.global_step, epoch + 1)


def _maybe_run_step_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_step = (
        args.validate_every_n_steps is not None
        and state.global_step % args.validate_every_n_steps == 0
    )
    if (
        state.accelerator.sync_gradients
        and state.validation_steps > 0
        and should_validate_step
    ):
        run_validation(
            trainer,
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_step_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="validation steps",
            postfix_label="val_avg_loss",
            log_avg_key="loss/validation/step_average",
            log_div_key="loss/validation/step_divergence",
            logging_fn=trainer.step_logging,
        )


def _run_epoch_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_epoch = (
        (epoch + 1) % args.validate_every_n_epochs == 0
        if args.validate_every_n_epochs is not None
        else True
    )
    if should_validate_epoch and len(state.val_ctx.dataloader) > 0:
        run_validation(
            trainer,
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_epoch_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="epoch validation steps",
            postfix_label="val_epoch_avg_loss",
            log_avg_key="loss/validation/epoch_average",
            log_div_key="loss/validation/epoch_divergence",
            logging_fn=trainer.epoch_logging,
        )


def _log_epoch_average(trainer, state: LoopState, epoch: int) -> None:
    if not state.is_tracking:
        return
    logs = {"loss/epoch_average": state.loss_recorder.moving_average}
    trainer.epoch_logging(state.accelerator, logs, state.global_step, epoch + 1)


def _audit_liveness(trainer, state: LoopState, *, where: str) -> None:
    """Liveness audit: a configured-ON aux loss that never
    consumed its aux input is a silent baseline — flag it loudly.

    Reads the trainer-owned ``LivenessLedger`` that the per-step composer
    feeds (``train.py`` threads it through ``build_loss_composer``). Dead
    features ERROR-log with the greppable ``LIVENESS:`` prefix (main process
    only — counts are per-rank but a dead dispatch is dead on every rank);
    ``--liveness_strict`` escalates to a hard abort, evaluated on each rank's
    own ledger so distributed runs fail together instead of hanging.
    """
    ledger = getattr(trainer, "_liveness", None)
    if ledger is None:
        return
    if state.accelerator.is_main_process:
        dead = ledger.audit(where=where)
    else:
        dead = ledger.dead_features()
    if dead and bool(getattr(state.args, "liveness_strict", False)):
        raise RuntimeError(
            f"LIVENESS: configured-but-dead feature(s) at {where}: "
            f"{', '.join(dead)} — aborting (--liveness_strict)"
        )


def _run_adapter_epoch_hooks(trainer, state: LoopState) -> None:
    """Per-method end-of-epoch hooks (IP-Adapter diagnostic dump, …).
    Main process only — adapters that need cross-rank reduction should do
    that internally."""
    if not (trainer._adapters and state.accelerator.is_main_process):
        return
    epoch_end_ctx = StepCtx(
        args=state.args,
        accelerator=state.accelerator,
        network=state.network,
        weight_dtype=state.weight_dtype,
    )
    for adapter in trainer._adapters:
        adapter.on_epoch_end(epoch_end_ctx)
