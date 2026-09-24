import argparse
import json
import math
import logging
import os
import shutil
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# checkpoint filename templates
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

CHECKPOINT_STATE_NAME = "{}-checkpoint-state"
CHECKPOINT_FILE_NAME = "{}-checkpoint"


def default_if_none(value, default):
    return default if value is None else value


def _ensure_parent_dir(path: str) -> None:
    """makedirs the directory that will contain ``path`` (the per-run subdir)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# Trajectory (step/epoch) checkpoints + their resumable -state dirs live in a
# per-run subdir ``<output_dir>/<model_name>/`` so they don't clutter
# output_dir; the final ``<output_name>.<ext>`` and the rolling
# ``<output_name>-checkpoint`` resumable family stay at the output_dir root
# (where inference / merge / auto_resume look). Returned names are relative to
# output_dir and carry the subdir prefix, so save and remove stay symmetric.
def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return os.path.join(model_name, EPOCH_FILE_NAME.format(model_name, epoch_no) + ext)


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return os.path.join(model_name, STEP_FILE_NAME.format(model_name, step_no) + ext)


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def save_sd_model_on_epoch_end_or_stepwise_common(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = (
            epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        )
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    else:
        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        _ensure_parent_dir(ckpt_file)  # create the per-run subdir
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

    else:
        if on_epoch_end:
            out_dir = os.path.join(
                args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no)
            )
        else:
            out_dir = os.path.join(
                args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step)
            )

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

    if args.save_state:
        if on_epoch_end:
            save_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_state_stepwise(args, accelerator, global_step)


def save_state_on_epoch_end(args: argparse.Namespace, accelerator, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")

    # Trajectory -state dirs live alongside their weights in the per-run subdir.
    state_dir = os.path.join(
        args.output_dir, model_name, EPOCH_STATE_NAME.format(model_name, epoch_no)
    )
    _ensure_parent_dir(state_dir)
    accelerator.save_state(state_dir)


def save_state_stepwise(args: argparse.Namespace, accelerator, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")

    # Trajectory -state dirs live alongside their weights in the per-run subdir.
    state_dir = os.path.join(
        args.output_dir, model_name, STEP_STATE_NAME.format(model_name, step_no)
    )
    _ensure_parent_dir(state_dir)
    accelerator.save_state(state_dir)


def get_checkpoint_state_dir(args: argparse.Namespace):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return os.path.join(args.output_dir, CHECKPOINT_STATE_NAME.format(model_name))


def get_checkpoint_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return CHECKPOINT_FILE_NAME.format(model_name) + ext


def save_checkpoint_state(args: argparse.Namespace, accelerator):
    state_dir = get_checkpoint_state_dir(args)

    logger.info("")
    logger.info(f"saving checkpoint state to {state_dir} (overwriting)")
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)

    accelerator.save_state(state_dir)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)


def save_sd_model_on_train_end_common(
    args: argparse.Namespace,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)


def resume_skip_plan(
    resume_step: int, num_batches: int, gradient_accumulation_steps: int
) -> tuple[int, int]:
    """Split an optimizer-step resume offset into ``(epoch_to_start,
    skip_batches)``: whole epochs to skip and the residual batch count inside
    the resumed epoch (for ``accelerator.skip_first_batches``).

    accelerate syncs on the last batch of the dataloader, so an epoch is
    ``ceil(num_batches / ga)`` optimizer steps. Every residual step is a full
    ``ga``-batch step (only the last step of an epoch can be partial, and the
    residual is strictly less than an epoch), so the batch conversion is exact.
    """
    ga = max(1, int(gradient_accumulation_steps))
    steps_per_epoch = max(1, math.ceil(num_batches / ga))
    epoch_to_start = resume_step // steps_per_epoch
    skip_batches = (resume_step % steps_per_epoch) * ga
    return epoch_to_start, skip_batches


class CheckpointSaver:
    """Owns every save / remove operation across a training run.

    ``metadata`` is a shared mutable dict — the trainer also writes
    ``ss_epoch`` between saves; the saver only writes during a save.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        accelerator,
        save_dtype,
        metadata: dict,
        minimum_metadata: dict,
        get_sai_model_spec_fn: Callable[[argparse.Namespace], dict],
        current_epoch,
        current_step,
        progress_sink=None,
    ):
        self.args = args
        self.accelerator = accelerator
        self.save_dtype = save_dtype
        self.metadata = metadata
        self.minimum_metadata = minimum_metadata
        self.get_sai_model_spec_fn = get_sai_model_spec_fn
        self.current_epoch = current_epoch
        self.current_step = current_step
        # Optional structured-progress sink. When set, every
        # checkpoint write emits a ``ckpt`` event.
        self.progress_sink = progress_sink
        # Set by the load_state pre-hook when resuming. Read by train() to
        # decide initial_step.
        self.steps_from_state: Optional[int] = None

    def register_hooks(self, network: Any) -> None:
        """Install accelerator save/load pre-hooks that persist epoch/step
        state to ``train_state.json`` and strip non-network models from the
        save list (we only want the adapter weights, not the frozen DiT)."""
        accelerator = self.accelerator
        unwrap_type = type(accelerator.unwrap_model(network))

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, unwrap_type):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)

            train_state_file = os.path.join(output_dir, "train_state.json")
            logger.info(
                f"save train state to {train_state_file} at epoch "
                f"{self.current_epoch.value} step {self.current_step.value + 1}"
            )
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "current_epoch": self.current_epoch.value,
                        "current_step": self.current_step.value + 1,
                    },
                    f,
                )

        def load_model_hook(models, input_dir):
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, unwrap_type):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)

            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    def auto_resume(self) -> None:
        """If ``checkpointing_epochs`` is enabled and a resumable checkpoint
        exists below ``max_train_steps``, point ``args.resume`` at it and
        force ``skip_until_initial_step``. No-op when ``args.resume`` is
        already set or no checkpoint exists."""
        args = self.args
        if not getattr(args, "checkpointing_epochs", None) or args.resume:
            return
        checkpoint_state_dir = get_checkpoint_state_dir(args)
        if not os.path.exists(checkpoint_state_dir):
            return
        train_state_file = os.path.join(checkpoint_state_dir, "train_state.json")
        if not os.path.exists(train_state_file):
            return
        with open(train_state_file, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
        ckpt_step = ckpt_data.get("current_step", 0)
        if ckpt_step < args.max_train_steps:
            args.resume = checkpoint_state_dir
            args.skip_until_initial_step = True
            logger.info(
                f"auto-resuming from checkpoint at step {ckpt_step}: {checkpoint_state_dir}"
            )
        else:
            logger.info(
                f"checkpoint already reached max_train_steps "
                f"({ckpt_step} >= {args.max_train_steps}), starting fresh"
            )

    def save(self, ckpt_name: str, network: Any, steps: int, epoch_no: int) -> None:
        """Write a network checkpoint with up-to-date training metadata."""
        args = self.args
        accelerator = self.accelerator
        unwrapped_nw = accelerator.unwrap_model(network)

        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        # ckpt_name carries a per-run subdir for trajectory (step/epoch) saves;
        # final + resumable names are bare and resolve to output_dir directly.
        _ensure_parent_dir(ckpt_file)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        self.metadata["ss_training_finished_at"] = str(time.time())
        self.metadata["ss_steps"] = str(steps)
        self.metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = self.minimum_metadata if args.no_metadata else self.metadata
        sai_metadata = self.get_sai_model_spec_fn(args)
        metadata_to_save.update(sai_metadata)

        unwrapped_nw.save_weights(ckpt_file, self.save_dtype, metadata_to_save)

        if self.progress_sink is not None:
            self.progress_sink.ckpt(global_step=steps, path=ckpt_file)

    def maybe_save_step(self, network: Any, global_step: int, epoch: int) -> None:
        """Step-cadence save. ``global_step`` must already be incremented."""
        args = self.args
        accelerator = self.accelerator
        if (
            args.save_every_n_steps is None
            or global_step % args.save_every_n_steps != 0
        ):
            return
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_name = get_step_ckpt_name(args, "." + args.save_model_as, global_step)
        self.save(ckpt_name, network, global_step, epoch)
        if args.save_state:
            save_state_stepwise(args, accelerator, global_step)

    def maybe_save_epoch(
        self, network: Any, global_step: int, epoch: int, num_train_epochs: int
    ) -> None:
        """Epoch-cadence save. ``epoch`` is 0-indexed; saver writes ``epoch+1``."""
        args = self.args
        accelerator = self.accelerator
        if args.save_every_n_epochs is None:
            return
        epoch_no = epoch + 1
        saving = (
            epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        )
        if not saving or not accelerator.is_main_process:
            return
        ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
        self.save(ckpt_name, network, global_step, epoch_no)
        if args.save_state:
            save_state_on_epoch_end(args, accelerator, epoch_no)

    def maybe_save_resumable(
        self, network: Any, global_step: int, epoch: int, num_train_epochs: int
    ) -> None:
        """``checkpointing_epochs``-cadence resumable save. Overwrites the
        same ``<output_name>-checkpoint`` file each time. ``epoch`` is 0-indexed."""
        args = self.args
        accelerator = self.accelerator
        if not (
            args.checkpointing_epochs is not None and args.checkpointing_epochs > 0
        ):
            return
        epoch_no = epoch + 1
        if not (
            epoch_no % args.checkpointing_epochs == 0 and epoch_no < num_train_epochs
        ):
            return
        if accelerator.is_main_process:
            ckpt_name = get_checkpoint_ckpt_name(args, "." + args.save_model_as)
            self.save(ckpt_name, network, global_step, epoch_no)
        save_checkpoint_state(args, accelerator)

    def save_release_pause(self, network: Any, global_step: int, epoch: int) -> str:
        """Release-pause save: the resumable ``<output_name>-checkpoint`` weights
        + state dir, exactly what ``maybe_save_resumable`` writes on its cadence,
        but on demand and regardless of ``checkpointing_epochs``. ``epoch`` is
        0-indexed. Returns the state dir (all ranks take part in
        ``save_state``)."""
        args = self.args
        accelerator = self.accelerator
        if accelerator.is_main_process:
            ckpt_name = get_checkpoint_ckpt_name(args, "." + args.save_model_as)
            self.save(ckpt_name, network, global_step, epoch + 1)
        save_checkpoint_state(args, accelerator)
        accelerator.wait_for_everyone()
        return get_checkpoint_state_dir(args)

    def resumed_from_release(self) -> bool:
        """True when this run was relaunched from a release-pause state dir
        (``--resume`` pointing at ``<output_name>-checkpoint-state``)."""
        resume = getattr(self.args, "resume", None)
        if not resume:
            return False
        return os.path.normpath(str(resume)) == os.path.normpath(
            get_checkpoint_state_dir(self.args)
        )

    def cleanup_resumable(self) -> None:
        """At training end, remove the resumable checkpoint state dir + ckpt
        file. Main-process only; no-op when ``checkpointing_epochs`` is unset,
        unless the run was itself resumed from a release-pause state."""
        args = self.args
        if not getattr(args, "checkpointing_epochs", None):
            if not self.resumed_from_release():
                return
        if not self.accelerator.is_main_process:
            return
        checkpoint_state_dir = get_checkpoint_state_dir(args)
        if os.path.exists(checkpoint_state_dir):
            logger.info(
                f"training complete, removing checkpoint state: {checkpoint_state_dir}"
            )
            shutil.rmtree(checkpoint_state_dir)
        checkpoint_ckpt = os.path.join(
            args.output_dir,
            get_checkpoint_ckpt_name(args, "." + args.save_model_as),
        )
        if os.path.exists(checkpoint_ckpt):
            logger.info(f"removing checkpoint weights: {checkpoint_ckpt}")
            os.remove(checkpoint_ckpt)

    def save_final(self, network: Any, global_step: int, num_train_epochs: int) -> None:
        """Write the final ``<output_name>.<ext>`` checkpoint. Main-process only."""
        if not self.accelerator.is_main_process:
            return
        args = self.args
        ckpt_name = get_last_ckpt_name(args, "." + args.save_model_as)
        self.save(ckpt_name, network, global_step, num_train_epochs)
        logger.info("model saved.")
