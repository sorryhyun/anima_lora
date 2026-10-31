"""``AnimaTrainer``: the run orchestrator.

``train()`` calls the setup modules in lazy-loading order (datasets → TE/VAE
→ caches → DiT → network → optimizer → accelerator) and hands the result to
``library.training.loop.run_training_loop``. The hook methods below are the
surface the loop and validation call back into; the per-step forward lives
in :class:`~library.anima.trainer.step.AnimaStep` (``self.step``).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from typing import Any, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from library import train_util
from library.anima import strategy as strategy_anima
from library.anima import text_strategies
from library.anima import training as anima_train_utils
from library.anima.trainer import data, models, network as network_setup, optim
from library.anima.trainer.step import AnimaStep
from library.config.cli_args import verify_training_args
from library.datasets import debug_dataset
from library.log import setup_logging
from library.runtime import noise as noise_utils
from library.runtime.accelerator import (
    prepare_accelerator,
    prepare_dtype,
    resolve_run_log_dir,
    resume_from_local_or_hf_if_specified,
)
from library.runtime.device import clean_memory_on_device
from library.training import (
    CheckpointSaver,
    TrainCtx,
    add_dataset_metadata,
    add_model_hash_metadata,
    build_training_metadata,
    finalize_metadata,
    save_state_on_train_end,
)
from library.training.checkpoints import resume_skip_plan
from library.training.log_dispatch import (
    dispatch_logs,
    generate_step_logs as _generate_step_logs,
)
from library.training.loop import build_loop_state, run_training_loop
from library.training.method_adapter import SetupCtx, resolve_adapters
from library.training.pause import TrainingPaused
from library.training.progress import ProgressSink, run_scope
from library.training.sampling_config import normalize_sample_args

logger = logging.getLogger(__name__)


class AnimaTrainer:
    def __init__(self):
        self.step = AnimaStep()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks = False
        self.progress_sink = None
        # Whether inductor will have CUDAGraphs active -- governs whether the
        # loop calls torch.compiler.cudagraph_mark_step_begin() each step.
        self._cudagraph_mark_step = False

    # region loop hooks

    def generate_step_logs(
        self,
        args: argparse.Namespace,
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
    ):
        # Thin wrapper: assembly lives in log_dispatch; the trainer
        # contributes only its VR λ state.
        return _generate_step_logs(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            mean_grad_norm,
            mean_combined_norm,
            vr_state=self.step.state.vr,
        )

    def step_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        dispatch_logs(
            accelerator,
            logs,
            global_step,
            global_step,
            epoch,
            progress_sink=self.progress_sink,
        )

    def epoch_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        dispatch_logs(
            accelerator,
            logs,
            epoch,
            global_step,
            epoch,
            progress_sink=self.progress_sink,
        )

    def val_logging(
        self,
        accelerator: Accelerator,
        logs: dict,
        global_step: int,
        epoch: int,
        val_step: int,
    ):
        dispatch_logs(
            accelerator,
            logs,
            global_step + val_step,
            global_step,
            epoch,
            val_step,
            progress_sink=self.progress_sink,
        )

    def get_noise_scheduler(
        self, args: argparse.Namespace, device: torch.device
    ) -> Any:
        return noise_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
        network=None,
    ):
        # With TE caching on, sample prompts come from sample_prompts_te_outputs.
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )
        qwen3_te = None if args.cache_text_encoder_outputs else text_encoders[0]

        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            text_strategies.TokenizeStrategy.get_strategy(),
            text_strategies.TextEncodingStrategy.get_strategy(),
            self.sample_prompts_te_outputs,
            network=network,
        )

    def on_step_start(self, ctx: TrainCtx, batch, *, is_train: bool = True):
        self.step.on_step_start(ctx, batch, is_train=is_train)

    def process_batch(self, ctx: TrainCtx, batch, *, is_train=True) -> torch.Tensor:
        return self.step.process_batch(ctx, batch, is_train=is_train)

    def run_after_backward(self, ctx: TrainCtx):
        self.step.run_after_backward(ctx)

    def on_validation_step_end(self, ctx: TrainCtx, batch):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            ctx.accelerator.unwrap_model(ctx.unet).prepare_block_swap_before_forward()

    @staticmethod
    def _parse_profile_steps(args) -> tuple[int, int] | None:
        """Parse --profile_steps 'start-end' into (start, end) or None.

        When set, the loop calls ``torch.cuda.profiler.start()`` at ``start``
        and ``stop()`` after ``end``, so pair this with::

            nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
                accelerate launch ... train.py --profile_steps 3-5
        """
        raw = getattr(args, "profile_steps", None)
        if not raw:
            return None
        if "-" in raw:
            a, b = raw.split("-", 1)
            return int(a), int(b)
        n = int(raw)
        return n, n + 2

    @staticmethod
    def _switch_rng_state(
        seed: int,
    ) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()

        torch.manual_seed(seed)
        random.seed(seed)

        return (cpu_rng_state, gpu_rng_state, python_rng_state)

    @staticmethod
    def _restore_rng_state(
        rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple],
    ):
        cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state)
        random.setstate(python_rng_state)

    # endregion

    @staticmethod
    def _sai_model_spec(args):
        return train_util.get_sai_model_spec_dataclass(
            args, lora=True
        ).to_metadata_dict()

    @staticmethod
    def _update_metadata(metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        if args.weighting_scheme == "min_snr":
            metadata["ss_min_snr_gamma"] = getattr(args, "min_snr_gamma", 5.0)
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_sigmoid_bias"] = getattr(args, "sigmoid_bias", 0.0)
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        if getattr(args, "int8", "off") != "off":
            metadata["ss_int8_base"] = args.int8
            metadata["ss_int8_base_id"] = args.int8_base_id

    def train(self, args):
        from library.runtime.backend import warn_if_cuda_unavailable

        warn_if_cuda_unavailable(torch)
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        normalize_sample_args(args)
        verify_training_args(args)
        if args.int8 != "off":
            from library.anima.int8 import check_plan

            check_plan(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        # The Qwen3 text encoder is never trained: the adapter goes on the DiT.
        if getattr(args, "network_train_text_encoder_only", False):
            raise ValueError(
                "--network_train_text_encoder_only is not supported: the Anima "
                "trainer trains DiT adapters only."
            )
        if not getattr(args, "network_train_unet_only", True):
            logger.warning(
                "network_train_unet_only=false is ignored: the text encoder is "
                "never trained (configs/base.toml sets it true)."
            )

        # Free-fit requires compile_dynamic_seq (else a static N-graph compile
        # cascade); auto-enable whenever torch_compile is on.
        if getattr(args, "torch_compile", False):
            if not getattr(args, "compile_dynamic_seq", False):
                logger.info(
                    "auto-enabling --compile_dynamic_seq "
                    "(free-fit shapes need the single-graph dynamic-seq path)"
                )
                args.compile_dynamic_seq = True

        cache_latents = args.cache_latents

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # --deterministic: closes flash-attn's backward atomic-add order plus
        # standard torch determinism knobs. Must precede CUDA/cublas init and
        # the first (possibly compiled) forward — the flash flag is read at
        # trace time. Bespoke loops (turbo/spd/mod) do NOT inherit this.
        if getattr(args, "deterministic", False):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            from networks import attention_dispatch

            attention_dispatch.set_deterministic(True)
            logger.info(
                "deterministic mode: flash-attn deterministic backward + "
                "torch.use_deterministic_algorithms(warn_only) + cudnn "
                "deterministic (CUBLAS_WORKSPACE_CONFIG=%s)",
                os.environ["CUBLAS_WORKSPACE_CONFIG"],
            )

        self._cudagraph_mark_step = bool(
            getattr(args, "torch_compile", False)
            and getattr(args, "compile_inductor_mode", None)
            in ("reduce-overhead", "max-autotune")
        )
        self.is_swapping_blocks = (
            args.blocks_to_swap is not None and args.blocks_to_swap > 0
        )

        # Build + install the strategy singletons. Must run before
        # build_datasets — dataset init reads them. TE-OUTPUTS caching
        # strategy is installed separately below, after the feature checks.
        strategies = strategy_anima.setup_training_strategies(args)
        tokenize_strategy = strategies.tokenize
        text_encoding_strategy = strategies.text_encoding
        tokenizers = [tokenize_strategy.qwen3_tokenizer]

        ds = data.build_datasets(args)
        train_dataset_group = ds.train_group
        val_dataset_group = ds.val_group

        token_budget = data.derive_token_budget(
            args, train_dataset_group, val_dataset_group
        )

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images)"
            )
            return

        if cache_latents:
            assert train_dataset_group.is_latent_cacheable(), (
                "when caching latents, either color_aug or random_crop cannot be used"
            )
            if val_dataset_group is not None:
                assert val_dataset_group.is_latent_cacheable(), (
                    "when caching latents, either color_aug or random_crop cannot be used"
                )

        data.configure_dataset_features(args, train_dataset_group, val_dataset_group)

        # Install the caching strategy now: after configure_dataset_features
        # (which may flip cache_llm_adapter_outputs) and before the model load,
        # so the cache-completeness probe below can decide if Qwen3 needs loading.
        strategy_anima.setup_text_encoder_outputs_caching_strategy(args)

        # When caching is enabled the caches MUST already be complete on disk
        # (the trainer does not encode on the fly); skip loading the encoders
        # entirely when nothing else needs them. `cache_latents = false` is a
        # separate, explicit live-encoding mode, not a fallback.
        sampling_enabled = bool(
            args.sample_prompts
            and (
                args.sample_at_first
                or args.sample_every_n_steps
                or args.sample_every_n_epochs
            )
        )

        def _latents_complete(group):
            return group is None or group.is_latents_cache_complete()

        def _te_complete(group):
            return group is None or group.is_text_encoder_outputs_cache_complete()

        if cache_latents and not (
            _latents_complete(train_dataset_group)
            and _latents_complete(val_dataset_group)
        ):
            raise RuntimeError(
                "Latent cache is incomplete. train.py requires a completed "
                "preprocess pass — run `make preprocess` (or set "
                "use_vae_cache = false for live VAE encoding)."
            )

        if args.cache_text_encoder_outputs and not (
            _te_complete(train_dataset_group) and _te_complete(val_dataset_group)
        ):
            raise RuntimeError(
                "Text-encoder cache is incomplete. train.py requires a completed "
                "preprocess pass — run `make preprocess` (or set "
                "use_text_cache = false for live encoding)."
            )

        # CMMD validation decodes samples through the VAE; reads cached TE
        # outputs, so it needs the VAE but not the text encoder.
        cmmd_validation = val_dataset_group is not None and getattr(
            args, "use_cmmd", True
        )
        vae_needed = (not cache_latents) or sampling_enabled or cmmd_validation
        qwen3_needed = (not args.cache_text_encoder_outputs) or bool(
            args.sample_prompts
        )

        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        weight_dtype, save_dtype = prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        qwen3, vae = models.load_text_encoder_and_vae(
            args, weight_dtype, load_qwen3=qwen3_needed, load_vae=vae_needed
        )
        if vae_dtype is None:
            vae_dtype = vae.dtype if vae is not None else weight_dtype
            logger.info(
                f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false"
            )
        text_encoders = [qwen3]

        # When vae is None, latents are already fully cached — new_cache_latents
        # still runs to populate each ImageInfo.latents_npz path, but forms
        # no encode batches so the absent VAE is never touched.
        if cache_latents:
            if vae is not None:
                vae.to(accelerator.device, dtype=vae_dtype)
                vae.requires_grad_(False)
                vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)
            if val_dataset_group is not None:
                val_dataset_group.new_cache_latents(vae, accelerator)

            if vae is not None:
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # The encoding strategy was installed with the others up top.
        self.sample_prompts_te_outputs = data.cache_text_encoder_outputs(
            args, accelerator, text_encoders, [train_dataset_group, val_dataset_group]
        )

        unet = models.load_dit(
            args, weight_dtype, accelerator, swapping=self.is_swapping_blocks
        )

        # Variance-reduced FM loss: the "frozen reference" is the trainable
        # DiT itself with ``network.set_multiplier(0)`` during the no-grad
        # forward (works since base weights are frozen and LoRA-family
        # adapters are additive). Saves ~5 GB VRAM vs a second DiT copy.
        if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
            logger.info(
                f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
                f"using trainable DiT with multiplier=0 as the control variate"
            )
        self.step.state.mem_tracker = models.build_mem_tracker(
            args, swapping=self.is_swapping_blocks
        )

        # Stage the T5("") sidecar once if caption dropout is on — dropped
        # rows then get the same crossattn embedding Anima feeds at
        # CFG-uncond inference instead of all-zeros (which is out-of-dist).
        self.step.state.caption_dropout_enabled = ds.caption_dropout_enabled
        if ds.caption_dropout_enabled:
            from library.preprocess.uncond import ensure_uncond_crossattn

            self.step.state.uncond_crossattn_1 = ensure_uncond_crossattn(
                qwen3_path=args.qwen3,
                dit_path=args.pretrained_model_name_or_path,
                t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None),
                device=accelerator.device,
                dtype=weight_dtype,
                existing=self.step.state.uncond_crossattn_1,
            )

        # Before the network exists: a grad_svd seed needs the task gradient of
        # the *base* DiT. No-op otherwise.
        network_setup.sketch_grad_basis(
            args,
            accelerator,
            unet,
            train_dataset_group,
            weight_dtype,
            swapping=self.is_swapping_blocks,
        )

        net = network_setup.create_and_apply_network(
            args,
            accelerator,
            vae,
            text_encoders,
            unet,
            weight_dtype,
            step=self.step,
            token_budget=token_budget,
        )
        if net is None:
            return
        network = net.network

        # Resolve and run on_network_built for each method adapter (EasyControl,
        # …); validates its runtime contract and sets up state before
        # optimizer/accelerator wiring.
        self.step.adapters = resolve_adapters(args, network)
        if self.step.adapters:
            setup_ctx = SetupCtx(
                args=args,
                accelerator=accelerator,
                network=network,
                unet=unet,
                text_encoders=text_encoders,
                weight_dtype=weight_dtype,
            )
            for adapter in self.step.adapters:
                adapter.on_network_built(setup_ctx)

        opt = optim.build_optimizer_and_dataloaders(
            args,
            accelerator,
            network,
            train_dataset_group,
            val_dataset_group,
            ds.collator,
        )

        acc = optim.prepare_with_accelerator(
            args,
            accelerator,
            network,
            opt,
            unet,
            text_encoders,
            vae,
            vae_dtype,
            weight_dtype,
            cache_latents,
            swapping=self.is_swapping_blocks,
        )
        network = acc.network
        train_dataloader = acc.train_dataloader
        unet = acc.unet
        text_encoders = acc.text_encoders

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Structured progress sink: a JSONL event stream next to the
        # checkpoint the GUI/daemon can tail instead of regex-parsing tqdm.
        self.progress_sink = None
        if is_main_process:
            progress_path = ProgressSink.resolve_path(args)
            if progress_path is not None:
                self.progress_sink = ProgressSink(
                    progress_path,
                    run=args.output_name or "run",
                    method=getattr(args, "method", None),
                    preset=getattr(args, "preset", None),
                    t0=training_started_at,
                )
                self.progress_sink.run_start(
                    total_steps=args.max_train_steps,
                    total_epochs=num_train_epochs,
                    pid=os.getpid(),
                    log_dir=resolve_run_log_dir(args),
                )
                # Mirror WARNING+ log records into the stream.
                self.progress_sink.attach_log_mirror()

        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = (
                math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
            )

        total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        accelerator.print("running training")
        accelerator.print("  num train images * repeats")
        accelerator.print("  num validation images * repeats")
        accelerator.print("  num reg images")
        accelerator.print("  num batches per epoch")
        accelerator.print("  num epochs")
        accelerator.print("  batch size per device")
        accelerator.print("  gradient accumulation steps")
        accelerator.print("  total optimization steps")

        metadata = build_training_metadata(
            args,
            session_id=session_id,
            training_started_at=training_started_at,
            text_encoder_lr=opt.text_encoder_lr,
            optimizer_name=opt.optimizer_name,
            optimizer_args=opt.optimizer_args,
            model_version="anima",
            num_train_images=train_dataset_group.num_train_images,
            num_val_images=val_dataset_group.num_train_images
            if val_dataset_group is not None
            else 0,
            num_reg_images=train_dataset_group.num_reg_images,
            num_batches_per_epoch=len(train_dataloader),
            num_train_epochs=num_train_epochs,
        )
        self._update_metadata(metadata, args)
        add_dataset_metadata(
            metadata,
            train_dataset_group,
            args,
            use_user_config=ds.use_user_config,
            use_dreambooth_method=ds.use_dreambooth_method,
            total_batch_size=total_batch_size,
        )
        add_model_hash_metadata(metadata, args)
        metadata, minimum_metadata = finalize_metadata(
            metadata, net_kwargs=net.net_kwargs if args.network_args else None
        )

        # Saver owns every save / remove operation plus the accelerator
        # save/load pre-hooks that persist train_state.json. Hooks must be
        # registered before resume_from_local_or_hf_if_specified() so the
        # load hook fires and populates saver.steps_from_state.
        saver = CheckpointSaver(
            args=args,
            accelerator=accelerator,
            save_dtype=save_dtype,
            metadata=metadata,
            minimum_metadata=minimum_metadata,
            get_sai_model_spec_fn=self._sai_model_spec,
            current_epoch=ds.current_epoch,
            current_step=ds.current_step,
            progress_sink=self.progress_sink,
        )
        saver.register_hooks(network)

        # auto-resume from the resumable checkpoint if one exists
        saver.auto_resume()

        resume_from_local_or_hf_if_specified(accelerator, args)
        steps_from_state = saver.steps_from_state

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader)
                    / accelerator.num_processes
                    / args.gradient_accumulation_steps
                )
        else:
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert args.max_train_steps > initial_step, (
                "max_train_steps should be greater than initial step"
            )

        # Units: ``resume_step`` / ``initial_step`` are optimizer steps here.
        # accelerate syncs on the last batch of the dataloader, so an epoch is
        # ceil(batches / ga) steps; full epochs are skipped by ``epoch_to_start``
        # and only the residual inside the resumed epoch is turned into a batch
        # count for ``skip_first_batches``. (Dividing a ga-multiplied count by
        # steps-per-epoch here used to over-skip by ga², see GH #103.)
        resume_step = initial_step
        epoch_to_start = 0
        if initial_step > 0:
            epoch_to_start, skip_batches = resume_skip_plan(
                initial_step, len(train_dataloader), args.gradient_accumulation_steps
            )
            if args.skip_until_initial_step:
                if not args.resume:
                    logger.info(
                        "initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                    )
                logger.info(
                    f"skipping {initial_step} steps: {epoch_to_start} full epochs"
                    f" + {skip_batches} batches of epoch {epoch_to_start + 1}"
                )
                initial_step = skip_batches
            else:
                initial_step = 0  # do not skip

        # Drop the train dataset-group locals before loop entry — the
        # dataloader already holds the data it needs. Keep val_dataset_group
        # alive: CMMD validation enumerates its image_data to pair held-out
        # references with generated samples.
        current_epoch, current_step = ds.current_epoch, ds.current_step
        del train_dataset_group, ds

        # sigma_lowres step-span gate: seed the train-forward counter with the
        # resume offset (already in forward units) so it covers every resume
        # path, not just re-deriving from args inside the loop.
        self.step.sigma.span_step = resume_step * args.gradient_accumulation_steps

        loop_state = build_loop_state(
            self,
            args=args,
            accelerator=accelerator,
            saver=saver,
            network=network,
            unet=unet,
            text_encoders=text_encoders,
            vae=vae,
            tokenizers=tokenizers,
            training_model=acc.training_model,
            train_dataloader=train_dataloader,
            val_dataloader=acc.val_dataloader,
            val_dataset_group=val_dataset_group,
            optimizer=acc.optimizer,
            lr_scheduler=acc.lr_scheduler,
            lr_descriptions=opt.lr_descriptions,
            optimizer_train_fn=opt.optimizer_train_fn,
            optimizer_eval_fn=opt.optimizer_eval_fn,
            weight_dtype=weight_dtype,
            unet_weight_dtype=acc.unet_weight_dtype,
            vae_dtype=vae_dtype,
            text_encoding_strategy=text_encoding_strategy,
            tokenize_strategy=tokenize_strategy,
            current_epoch=current_epoch,
            current_step=current_step,
            num_train_epochs=num_train_epochs,
            epoch_to_start=epoch_to_start,
            initial_step=initial_step,
            resume_step=resume_step,
            metadata=metadata,
        )

        # Emits the matching run_end (ok/stopped/error) on exit; run_start
        # already fired when the sink was constructed above.
        with run_scope(
            self.progress_sink,
            final_step=lambda: loop_state.global_step,
            extra_fields=lambda: {
                **self.step.liveness.run_end_fields(),
                **(
                    {
                        "token_step_hist": {
                            str(k): v
                            for k, v in sorted(self.step.token_step_hist.items())
                        }
                    }
                    if self.step.token_step_hist
                    else {}
                ),
            },
        ):
            try:
                run_training_loop(self, loop_state)
            except TrainingPaused as paused:
                # Release-pause: state is on disk, skip the final save/cleanup
                # (the resumed run does them) but still flush the trackers.
                accelerator.end_training()
                logger.info(
                    f"training paused at step {paused.global_step}; resume with "
                    f"--resume {paused.state_dir}"
                )
                raise

            accelerator.end_training()
            opt.optimizer_eval_fn()

            # Catch-all sample decode for any latents not already decoded
            # inline (block-swapping runs defer the whole batch to here).
            # Park the DiT on CPU first so the VAE decode gets the full budget.
            if is_main_process and args.sample_prompts:
                try:
                    accelerator.unwrap_model(loop_state.unet).to("cpu")
                except Exception:
                    pass
                clean_memory_on_device(accelerator.device)
                anima_train_utils.decode_pending_samples(accelerator, args, vae)

            if is_main_process and (args.save_state or args.save_state_on_train_end):
                save_state_on_train_end(args, accelerator)

            saver.cleanup_resumable()
            saver.save_final(network, loop_state.global_step, num_train_epochs)

        if is_main_process and loop_state.global_step < 2:
            _cleanup_short_log_dir(args)


def _cleanup_short_log_dir(args) -> None:
    """Delete the TensorBoard log dir when a run completed fewer than 2 steps."""
    import shutil

    log_dir = resolve_run_log_dir(args)
    if log_dir is None:
        return
    try:
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
    except Exception as e:
        print(
            f"warn: could not remove short-run log dir {log_dir}: {e}", file=sys.stderr
        )
