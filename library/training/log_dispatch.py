"""Log-dispatch: fan an assembled ``logs`` dict out to its sinks.

Sends a ``logs`` dict to every configured Accelerate tracker (tensorboard /
wandb / others) plus the :class:`~library.training.progress.ProgressSink`.
This is the *output* end of the metrics pipeline — the values are produced via
the :mod:`library.training.metrics` collector protocol, assembled into a dict,
then handed here. Distinct from :mod:`library.log`, which configures Python's
stdlib console logging.

``AnimaTrainer`` keeps thin ``step_logging`` / ``epoch_logging`` /
``val_logging`` wrappers that call :func:`dispatch_logs` with the trainer's
``progress_sink``.
"""

from __future__ import annotations

from typing import Optional


def generate_step_logs(
    args,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
    mean_grad_norm=None,
    mean_combined_norm=None,
    *,
    vr_state: Optional[dict] = None,
) -> dict:
    """Assemble the per-step ``logs`` dict (loss, norms, per-group LRs).

    The *input* end of the metrics pipeline — the dict built here is handed to
    :func:`dispatch_logs`. ``vr_state`` is the trainer's ``RuntimeState.vr``
    dict (λ tracking for the variance-reduced FM loss); pass ``None`` when VR
    is off.
    """
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/max_key_norm"] = maximum_norm
    if mean_norm is not None:
        logs["norm/avg_key_norm"] = mean_norm
    if mean_grad_norm is not None:
        logs["norm/avg_grad_norm"] = mean_grad_norm
    if mean_combined_norm is not None:
        logs["norm/avg_combined_norm"] = mean_combined_norm

    if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0 and vr_state:
        lambda_ema = vr_state.get("lambda_ema")
        lambda_batch = vr_state.get("lambda_batch")
        if isinstance(lambda_ema, float):
            logs["vr/lambda_ema"] = lambda_ema
        if isinstance(lambda_batch, float):
            logs["vr/lambda_batch"] = lambda_batch

    lrs = lr_scheduler.get_last_lr()
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None:
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if args.network_train_unet_only else -1)
            if idx == -1:
                lr_desc = "textencoder"
            else:
                if len(lrs) > 2:
                    lr_desc = f"group{idx}"
                else:
                    lr_desc = "unet"

        logs[f"lr/{lr_desc}"] = lr

        if (
            args.optimizer_type.lower().startswith("DAdapt".lower())
            or args.optimizer_type.lower() == "Prodigy".lower()
        ):
            # tracking d*lr value
            logs[f"lr/d*lr/{lr_desc}"] = (
                lr_scheduler.optimizers[-1].param_groups[i]["d"]
                * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
            )
        if (
            args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower())
            and optimizer is not None
        ):  # tracking d*lr value of unet.
            logs["lr/d*lr"] = (
                optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
            )

    return logs


def dispatch_logs(
    accelerator,
    logs: dict,
    step_value: int,
    global_step: int,
    epoch: int,
    val_step: Optional[int] = None,
    *,
    progress_sink=None,
) -> None:
    """Send ``logs`` to all trackers and the progress sink.

    ``step_value`` is the x-axis for tensorboard; ``global_step`` / ``epoch`` /
    ``val_step`` are attached as fields for wandb. ``progress_sink`` (if given)
    receives the same dict on the main process only.
    """
    tensorboard_tracker = None
    wandb_tracker = None
    other_trackers = []
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tensorboard_tracker = accelerator.get_tracker("tensorboard")
        elif tracker.name == "wandb":
            wandb_tracker = accelerator.get_tracker("wandb")
        else:
            other_trackers.append(accelerator.get_tracker(tracker.name))

    if tensorboard_tracker is not None:
        tensorboard_tracker.log(logs, step=step_value)

    if wandb_tracker is not None:
        logs["global_step"] = global_step
        logs["epoch"] = epoch
        if val_step is not None:
            logs["val_step"] = val_step
        wandb_tracker.log(logs)

    for tracker in other_trackers:
        tracker.log(logs, step=step_value)

    if progress_sink is not None and accelerator.is_main_process:
        progress_sink.log(logs, global_step=global_step, epoch=epoch, val_step=val_step)
