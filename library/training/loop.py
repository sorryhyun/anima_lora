"""Training-loop orchestration.

Owns the per-epoch / per-step body that used to live inline in
``AnimaTrainer.train()``. The entrypoint is :func:`run_training_loop`, which
takes a built :class:`LoopState` plus the trainer instance so override hooks
(``process_batch``, ``on_step_start``, ``sample_images``, ``_run_validation``,
``generate_step_logs``, ``step_logging``, ``epoch_logging``) keep working
unchanged.

State that used to be on ``self`` for cross-call signaling — ``_split_backward_consumed``,
``_last_router_H_postfix``, ``_cudagraph_mark_step``, ``_hydra_warmup_step``,
``_adapters`` — stays on the trainer; this module reads them through the
``trainer`` handle.
"""

from __future__ import annotations

import gc
import logging
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
from library.training.method_adapter import StepCtx
from library.training.metrics import MetricContext, collect_metrics

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Bundles every local that used to live in ``train()``'s for-epoch scope.

    Most fields are constants for the run; ``global_step``, ``profile_started``,
    ``profile_range``, ``initial_step``, and ``text_encoder(s)`` are mutated
    during the loop. ``current_epoch`` / ``current_step`` are mp.Value handles
    shared with :class:`CheckpointSaver` for state persistence.
    """

    args: Any
    accelerator: Accelerator
    train_ctx: Any  # TrainCtx
    val_ctx: Any  # ValCtx
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
    train_ctx_cls,
    val_ctx_cls,
) -> LoopState:
    """Build :class:`LoopState`. Mirrors the pre-loop setup that used to sit
    between ``_prepare_with_accelerator()`` and the for-epoch loop in
    ``train()``: noise scheduler, trackers, loss recorders, optional text
    encoder eviction, ``--sample_at_first``, train/val ctx construction,
    progress bar, profiler parsing.

    ``train_ctx_cls`` / ``val_ctx_cls`` are the trainer's ``TrainCtx`` /
    ``ValCtx`` dataclasses (defined in train.py); passing them as parameters
    avoids a circular import.
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

    # --sample_at_first
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
    )
    optimizer_train_fn()
    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        accelerator.log({}, step=0)

    train_ctx = train_ctx_cls(
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

    # Skip prelude: when resuming with skip_until_initial_step, fast-forward
    # the global_step counter before tqdm so the bar total is sized correctly,
    # and consume per-epoch skip credit so dataloader.skip_first_batches has
    # the right offset on the first epoch only. Runs before the dtype log so
    # the log order matches the original train() body.
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
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()  # skip the first parameter
        params_itr.__next__()  # skip the second parameter (CLIP first two are embeddings)
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
    # Validate at fixed sigma values across the schedule:
    # 0.1 = near-clean / fine detail, 0.4 = mid / bulk structure,
    # 0.7 = high noise / coarse denoising (early inference steps).
    validation_sigmas = (
        args.validation_sigmas
        if args.validation_sigmas is not None
        else [0.1, 0.4, 0.7]
    )
    val_ctx = val_ctx_cls(
        dataloader=val_dataloader,
        sigmas=validation_sigmas,
        steps=validation_steps,
        total_steps=validation_steps * len(validation_sigmas),
        train_loss_recorder=loss_recorder,
        original_t_min=args.t_min,
        original_t_max=args.t_max,
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

        # network.train() is invoked here
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
        )
        state.progress_bar.unpause()
        state.optimizer_train_fn()

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

        # preprocess batch for each model
        trainer.on_step_start(state.train_ctx, batch, is_train=True)

        # CUDAGraphs (reduce-overhead / max-autotune) need an explicit
        # iteration boundary for inductor's cudagraph_trees. Without this
        # call, the "pending, uninvoked backwards" fast-path check fails
        # every step and cudagraphs silently fall back to the eager path —
        # you pay compile latency and keep launch overhead. Must be called
        # before the forward on every step.
        #
        # Also clear Python references to last-step gate/σ tensors *before*
        # marking — those tensors live in the cudagraph memory pool, and a
        # lingering self._last_gate/self._sigma reference keeps the pool
        # pinned regardless of the mark call, which defeats the whole point.
        if trainer._cudagraph_mark_step:
            net_unwrapped = accelerator.unwrap_model(network)
            if hasattr(net_unwrapped, "clear_step_caches"):
                net_unwrapped.clear_step_caches()
            torch.compiler.cudagraph_mark_step_begin()

        if state.profile_started:
            torch.cuda.nvtx.range_push("forward")
        loss = trainer.process_batch(state.train_ctx, batch, is_train=True)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        # Split-backward path (APEX) backwards both branches inline inside
        # process_batch and returns a detached scalar for logging. Skip the
        # outer backward in that case so we don't double-step or crash on a
        # no-grad tensor during warmup.
        if getattr(trainer, "_split_backward_consumed", False):
            trainer._split_backward_consumed = False
        else:
            if state.profile_started:
                torch.cuda.nvtx.range_push("backward")
            accelerator.backward(loss)
            if state.profile_started:
                torch.cuda.nvtx.range_pop()

        if accelerator.sync_gradients:
            # HydraLoRA "best-expert" warmup: keep grads only on top-k experts
            # by per-expert grad-norm during warmup. No-op unless
            # expert_best_warmup_ratio > 0. Runs before clip_grad_norm so
            # clipping sees the masked grads.
            net_unwrapped = accelerator.unwrap_model(network)
            if hasattr(net_unwrapped, "step_expert_best_warmup_post_backward"):
                net_unwrapped.step_expert_best_warmup_post_backward(
                    int(getattr(trainer, "_hydra_warmup_step", 0)),
                    int(getattr(args, "max_train_steps", 0) or 0),
                )
            # Snapshot Hydra up-weight grad norms before zero_grad wipes them.
            # The metric ``hydra_up_grad`` reads this stash later in the step.
            # Also runs pre-clip so absolute magnitudes aren't distorted by
            # the global rescale (clipping preserves the below/above ratio).
            # Skip on non-log steps — the metric only fires at log cadence,
            # so capturing every step burns kernels whose output is never
            # read. global_step increments below, so predict the
            # post-increment value.
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
        torch.cuda.nvtx.range_pop()  # close per-step NVTX range
    if state.profile_started and state.global_step >= state.profile_range[1]:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        state.accelerator.print(f"\n[profiler] stopped at step {state.global_step}")
        state.accelerator.print(
            "[profiler] open the .nsys-rep with the Nsight Systems GUI\n"
        )
        state.profile_started = False
        state.profile_range = None  # don't re-trigger


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
    )
    state.progress_bar.unpause()


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
    should_log_step = (state.global_step % log_every == 0) or (
        state.global_step >= args.max_train_steps
    )

    current_loss = loss.detach().item()
    state.loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
    avr_loss: float = state.loss_recorder.moving_average
    logs = {"avr_loss": avr_loss}
    _unwrapped_net = state.accelerator.unwrap_model(state.network)
    # Refresh router_H only on log cadence — get_router_entropy → full
    # get_router_stats compute (with D2H syncs) is wasted if the only
    # consumer is the progress-bar postfix. Cache last value on trainer
    # so tqdm shows a stale value harmlessly between log steps.
    if getattr(_unwrapped_net, "_use_hydra", False) and should_log_step:
        _router_H = _unwrapped_net.get_router_entropy()
        if _router_H is not None:
            trainer._last_router_H_postfix = _router_H
    _router_H_cached = getattr(trainer, "_last_router_H_postfix", None)
    if _router_H_cached is not None:
        logs["router_H"] = f"{_router_H_cached:.3f}"
    state.progress_bar.set_postfix(refresh=False, **{**max_mean_logs, **logs})

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
        trainer._run_validation(
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
        trainer._run_validation(
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
