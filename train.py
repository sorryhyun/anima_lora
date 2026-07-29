# Anima LoRA training script (merged standalone)

import importlib
import argparse
import math
import os
import typing
from typing import Any, Union, Optional
import sys
import random
import time
from multiprocessing import Value

# Windows: suppress per-kernel ptxas.exe / cl.exe console flashes from
# torch.compile + Triton. Must run before any subprocess.Popen call (i.e.
# before torch import on Windows where inductor may prefetch toolchain).
from library.runtime.proc import install_no_window_default

install_no_window_default()

# Allocator default must land before torch initializes the CUDA caching
# allocator: free-fit varies seq_len per step and fragments the reserved pool
# without expandable segments (issue #58). Opt out: ANIMA_EXPANDABLE_SEGMENTS=0.
from library.runtime.allocator import default_expandable_segments

if default_expandable_segments():
    print(
        "Anima: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "(default; set ANIMA_EXPANDABLE_SEGMENTS=0 to disable)"
    )

import torch
import torch.nn as nn
from library.runtime.device import clean_memory_on_device

from accelerate.utils import set_seed
from accelerate import Accelerator
from library import (
    train_util,
)
from library.anima import (
    models as anima_models,
    training as anima_train_utils,
    weights as anima_utils,
    strategy as strategy_anima,
    text_strategies,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.models import sai_spec as sai_model_spec
from library.runtime import noise as noise_utils
from library.config import loader as config_util
from library.config.loader import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.training.method_adapter import (
    ComputeLossCtx,
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
    resolve_adapters,
)
from library.config.io import (
    load_dataset_config_from_base,
    read_config_from_file,
)
from library.datasets import (
    DatasetGroup,
    MinimalDataset,
    collator_class,
    debug_dataset,
    load_arbitrary_dataset,
)
from library.datasets import base as _datasets_base
from library.runtime.accelerator import (
    prepare_accelerator,
    prepare_dtype,
    resolve_run_log_dir,
    resume_from_local_or_hf_if_specified,
)
from library.training import (
    AcceleratedBundle,
    CheckpointSaver,
    DatasetBundle,
    LivenessLedger,
    LossContext,
    NetworkBundle,
    OptimizerBundle,
    SAMPLER_REGISTRY,
    RuntimeState,
    SamplerContext,
    TrainCtx,
    add_custom_train_arguments,
    add_dataset_metadata,
    add_model_hash_metadata,
    build_loss_composer,
    build_training_metadata,
    finalize_metadata,
    get_huber_threshold_if_needed,
    get_optimizer,
    get_optimizer_train_eval_fn,
    get_scheduler_fix,
    save_state_on_train_end,
)
from library.config.cli_args import (
    add_dataset_arguments,
    add_dit_training_arguments,
    add_masked_loss_arguments,
    add_network_arguments,
    add_optimizer_arguments,
    add_sd_models_arguments,
    add_train_misc_arguments,
    add_training_arguments,
    add_validation_arguments,
    verify_command_line_training_args,
    verify_training_args,
)
from library.training.loop import build_loop_state, run_training_loop
from library.training.sampling_config import normalize_sample_args
from library.training.log_dispatch import (
    dispatch_logs,
    generate_step_logs as _generate_step_logs,
)
from library.training.progress import ProgressSink, run_scope
from library.training.mem_reweight import (
    MemGapTracker,
    adapted_logmse,
    measure_base_logmse,
    measure_grid_delta,
)
from library.training.forward import (
    ForwardConditioning,
    apply_router_conditioning,
    build_forward_conditioning,
    compute_inversion_func_loss,
    prepare_text_conds,
    run_vr_reference_forward,
)
from library.log import setup_logging, add_logging_arguments

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


class AnimaTrainer:
    def __init__(self):
        self.sample_prompts_te_outputs = None
        self._padding_mask_cache = {}
        # Per-method extensions (EasyControl, IP-Adapter, …). Resolved
        # from args+network in train() right after _create_and_apply_network.
        self._adapters: list[MethodAdapter] = []
        # Feature-specific per-run state — see ``RuntimeState``.
        self._state = RuntimeState()
        # Liveness ledger (issues.md P1.1): counts aux consumption per
        # skip-if-missing loss; the loop audits it (step-25 early check +
        # run end) and flags configured-but-dead features with `LIVENESS:`.
        self._liveness = LivenessLedger()
        # Realized patch-token histogram over train steps ({tokens: examples},
        # post σ-demote swap) — per-arm FLOPs accounting for the sigma_lowres
        # E4 A/B; merged into the run_end progress event.
        self._token_step_hist: dict[int, int] = {}

    # region logging helpers

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
        # Thin wrapper (same shape as step_logging/epoch_logging below): the
        # loop calls this on the trainer; the assembly lives in log_dispatch,
        # with the trainer contributing only its VR λ state.
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
            vr_state=self._state.vr,
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
            progress_sink=getattr(self, "progress_sink", None),
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
            progress_sink=getattr(self, "progress_sink", None),
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
            progress_sink=getattr(self, "progress_sink", None),
        )

    # endregion

    # region Anima-specific methods (from AnimaNetworkTrainer overrides)

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        # use_text_cache → cache_text_encoder_outputs{,_to_disk} is expanded in
        # verify_training_args (runs first); just read the derived flag here.
        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), (
                "when caching Text Encoder output, token_warmup_step or caption_tag_dropout_rate cannot be used"
            )
            if getattr(args, "cache_llm_adapter_outputs", False):
                # Adapter output caching is only valid when the adapter is frozen (no LoRA on adapter).
                if args.network_args is not None and any(
                    "train_llm_adapter" in a and "true" in a.lower()
                    for a in args.network_args
                ):
                    raise ValueError(
                        "--cache_llm_adapter_outputs is incompatible with --network_args train_llm_adapter=True"
                    )
        elif getattr(args, "cache_llm_adapter_outputs", False):
            # Adapter-output caching writes into the TE cache; with text caching
            # off there is nothing to write into (the caching strategy is None and
            # adapter outputs are computed live), so the flag is a harmless no-op.
            # Auto-disable it instead of crashing — this combination is easy to
            # hit from the GUI, where use_text_cache and cache_llm_adapter_outputs
            # are independent toggles while methods default the latter to true.
            logger.warning(
                "cache_llm_adapter_outputs=true has no effect without text-encoder "
                "caching (use_text_cache=false / live text encoding); disabling it."
            )
            args.cache_llm_adapter_outputs = False

        assert args.network_train_unet_only or not args.cache_text_encoder_outputs, (
            "network for Text Encoder cannot be trained with caching Text Encoder outputs"
        )

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning(
                    "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled"
                )
                args.gradient_checkpointing = True
            assert args.blocks_to_swap is None or args.blocks_to_swap == 0, (
                "blocks_to_swap is not supported with unsloth_offload_checkpointing"
            )

        # Propagate inversion_dir to datasets for functional-loss supervision (postfix-func).
        inversion_dir = getattr(args, "inversion_dir", None)
        if inversion_dir:
            num_runs = getattr(args, "functional_loss_num_runs", 3)
            for dataset in train_dataset_group.datasets:
                dataset.inversion_dir = inversion_dir
                dataset.inversion_num_runs = num_runs
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.inversion_dir = inversion_dir
                    dataset.inversion_num_runs = num_runs

        # Propagate BYG per-image edit-tuple cache dir so datasets load
        # {stem}_byg.safetensors into batch["byg_{role}_emb"]/["byg_{role}_mask"].
        if getattr(args, "use_byg", False):
            byg_text_dir = getattr(args, "byg_text_dir", None) or os.path.join(
                "post_image_dataset", "byg"
            )
            for dataset in train_dataset_group.datasets:
                dataset.byg_text_dir = byg_text_dir
                kept, dropped = dataset.restrict_to_byg_tuples()
                if dropped:
                    logger.info(
                        f"BYG: kept {kept} images with edit-tuple sidecars, "
                        f"dropped {dropped} without (no swappable tag in caption)."
                    )
            # restrict_to_byg_tuples re-buckets each member, shrinking its length;
            # refresh the ConcatDataset cumulative_sizes or global indices overflow.
            train_dataset_group.refresh_concat_state()
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.byg_text_dir = byg_text_dir
                    dataset.restrict_to_byg_tuples()
                val_dataset_group.refresh_concat_state()

        # REPA v2: load cached PE-Spatial patch tokens into batches when
        # use_repa is set. The flag rides the network kwargs; read the resolved
        # merged view (--network_args + top-level TOML keys) rather than
        # re-scanning both intake paths.
        net_kwargs = resolve_network_kwargs(args)
        if net_kwargs.get("use_repa", "").lower() in ("true", "1", "yes"):
            repa_encoder = net_kwargs.get("repa_encoder") or "pe_spatial"
            for dataset in train_dataset_group.datasets:
                dataset.load_repa_pe = True
                dataset.repa_pe_encoder = repa_encoder
            # Probe PE sidecar coverage now. A missing PE cache makes the REPA
            # alignment term a silent no-op — the loss skips any batch without
            # repa_pe_features (library/training/repa.py) — so a run with
            # use_repa but no `make preprocess-pe` would train as if REPA were
            # off, with no error. Fail fast on a fully-absent cache; warn on a
            # partial one (the all-or-nothing collate tolerates per-batch gaps).
            present, total = train_dataset_group.count_repa_pe_sidecars()
            if total > 0 and present == 0:
                raise RuntimeError(
                    f"use_repa is enabled but none of the {total} training "
                    f"images have a {repa_encoder} PE feature cache "
                    f"(*_anima_{repa_encoder}.safetensors) — the REPA "
                    f"alignment loss would be a silent no-op. Run "
                    f"`make preprocess-pe ARGS='--encoder {repa_encoder}'` "
                    f"first, or disable use_repa."
                )
            if present < total:
                logger.warning(
                    f"REPA: only {present}/{total} training images have a "
                    f"{repa_encoder} PE sidecar; the alignment term is skipped "
                    f"for batches missing one. Run `make preprocess-pe "
                    f"ARGS='--encoder {repa_encoder}'` to cover the rest."
                )
            logger.info(
                f"REPA: PE feature loading enabled (encoder={repa_encoder}); "
                f"batches carry repa_pe_features ({present}/{total} cached)."
            )

        # Soft-tokens contrastive negatives. The objective's knobs live in
        # ``network_args`` (see configs/methods/soft_tokens.toml); preview them
        # via the resolved kwargs view to decide whether
        # the dataset should surface cached negative text embeddings. Off unless
        # contrastive_weight > 0. See docs/proposal/soft_tokens_contrastive.md.
        if str(getattr(args, "network_module", "") or "") == (
            "networks.methods.soft_tokens"
        ):
            con_weight = float(net_kwargs.get("contrastive_weight", 0.0) or 0.0)
            if con_weight > 0.0:
                con_k = int(net_kwargs.get("contrastive_k", 1) or 1)
                con_mode = str(net_kwargs.get("contrastive_negative_mode", "shuffled"))
                # The negative grouping always comes from the shared caption
                # index `make caption-index` writes — not a user knob.
                con_index = "post_image_dataset/captions/caption_index.json"
                if not os.path.exists(con_index):
                    raise FileNotFoundError(
                        f"contrastive_index not found: {con_index}. "
                        f"Run `make caption-index`."
                    )
                if not getattr(args, "cache_llm_adapter_outputs", False):
                    raise ValueError(
                        "soft_tokens contrastive requires "
                        "cache_llm_adapter_outputs=true (negatives are cached "
                        "crossattn_emb swapped off disk)."
                    )
                # Negatives only feed the training-step contrastive forward; the
                # validation FM-MSE stays a clean baseline, so val datasets are
                # left untouched.
                for dataset in train_dataset_group.datasets:
                    dataset.setup_contrastive_negatives(
                        con_index, k=con_k, mode=con_mode, is_validation=False
                    )
                logger.info(
                    f"Soft-tokens contrastive: weight={con_weight} k={con_k} "
                    f"mode={con_mode} index={con_index}"
                )

    def load_target_model(
        self, args, weight_dtype, accelerator, load_qwen3=True, load_vae=True
    ):
        self.is_swapping_blocks = (
            args.blocks_to_swap is not None and args.blocks_to_swap > 0
        )

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy).
        # Skipped when every text-encoder output is already cached and no live
        # encoding (sampling / TE training / cache disabled) needs it.
        if load_qwen3:
            logger.info("Loading Qwen3 text encoder...")
            qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
                args.qwen3, dtype=weight_dtype, device="cpu"
            )
            qwen3_text_encoder.eval()
        else:
            logger.info(
                "Skipping Qwen3 text encoder load: all text-encoder outputs cached."
            )
            qwen3_text_encoder = None

        # Load VAE. Skipped when every latent is already cached and no sampling
        # (which decodes latents) is configured.
        if load_vae:
            logger.info("Loading Anima VAE...")
            vae = qwen_image_autoencoder_kl.load_vae(
                args.vae,
                device="cpu",
                disable_mmap=True,
                spatial_chunk_size=args.vae_chunk_size,
                disable_cache=args.vae_disable_cache,
            )
            vae.to(weight_dtype)
            vae.eval()
        else:
            logger.info("Skipping VAE load: all latents cached and no sampling.")
            vae = None

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(
        self, args, weight_dtype, accelerator, text_encoders
    ) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        if attn_mode == "flash4":
            # Flash Attention 4 (flash-attention-sm120) is not supported yet.
            raise RuntimeError(
                "attn_mode='flash4' is not supported yet -- the flash-attention-sm120 "
                "kernel is disabled in this build. Use 'flash', 'torch', 'flex', "
                "or 'sageattn' instead."
            )
        elif attn_mode == "flash":
            from networks.attention_dispatch import flash_attn, flash_attn_func

            if flash_attn_func is not None:
                logger.info(
                    f"Using Flash Attention 2 (flash_attn {flash_attn.__version__})"
                )
            else:
                raise RuntimeError(
                    "attn_mode='flash' requested but flash_attn is not available."
                )
        else:
            logger.info(f"Using attention mode: {attn_mode}")

        # Frozen LoRA: merged into DiT weights at load time (no runtime hooks).
        # Used by postfix runs that train on top of a fixed LoRA.
        lora_weights_list = None
        lora_multipliers = None
        if getattr(args, "lora_path", None):
            from safetensors.torch import load_file

            logger.info(
                f"merging frozen LoRA from {args.lora_path} into DiT weights "
                f"(multiplier={args.lora_multiplier})"
            )
            lora_sd = load_file(args.lora_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
            lora_weights_list = [lora_sd]
            lora_multipliers = [args.lora_multiplier]

        # Load DiT
        attn_softmax_scale = getattr(args, "attn_softmax_scale", None)
        logger.info(
            f"Loading Anima DiT model with attn_softmax_scale: {attn_softmax_scale}..."
        )
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            loading_device,
            loading_dtype,
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
            attn_softmax_scale=attn_softmax_scale,
        )

        # Mod-aware training: install the distilled pooled_text_proj so every
        # training forward runs with the pooled-text t-embedding injection
        # active — an adaln LoRA then trains against the operating point
        # mod-guidance perturbs at inference. Frozen implicitly (the LoRA
        # factory excludes pooled_text_proj; the DiT-wide requires_grad_(False)
        # covers it). Loaded on CPU first — the params are meta tensors when
        # absent from the pretrained checkpoint — then moved to the model's
        # loading placement. The injection lives in Anima.forward outside the
        # blocks, so compile_blocks is unaffected.
        if getattr(args, "pooled_text_proj", None):
            anima_utils.load_pooled_text_proj(model, args.pooled_text_proj, "cpu")
            model.pooled_text_proj.to(device=loading_device, dtype=loading_dtype)

        # NOTE: torch.compile (compile_blocks) is intentionally NOT done here.
        # It must run AFTER the adapter's apply_to monkey-patches the targeted
        # Linears, or dynamo traces the un-adapted forward — see the compile
        # ordering in library/runtime/harness.py. compile is lazy, so the old
        # compile-here-apply-later ordering happened to work as long as no DiT
        # forward ran in the window; moved to _create_and_apply_network (after
        # apply_to + load_weights + grad-ckpt) so the invariant holds by
        # construction rather than by luck.

        # Store unsloth preference so that when the base trainer calls
        # dit.enable_gradient_checkpointing(), we can override to use unsloth.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        # Block swap
        self.is_swapping_blocks = (
            args.blocks_to_swap is not None and args.blocks_to_swap > 0
        )
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        # Variance-reduced FM loss: the "frozen reference" is the trainable
        # DiT itself with ``network.set_multiplier(0)`` during the no-grad
        # forward — works because base weights are frozen and LoRA-family
        # adapters are additive. See ``get_noise_pred_and_target`` for the
        # bypass. Saves ~5 GB VRAM vs holding a second DiT copy.
        if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
            logger.info(
                f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
                f"using trainable DiT with multiplier=0 as the control variate"
            )

        # Online memorization Δ-gap tracker (same set_multiplier(0) trick as
        # VR — _archive/proposals/memorization_lowsigma_reweight.md). The
        # measurement is a second per-step DiT forward, unaudited against the
        # block-swap offloader (cf.
        # [[project_blockswap_extra_forwards_gradcache]]) — raise, not warn,
        # same policy as the register-tokens guard.
        mem_mode = str(getattr(args, "mem_reweight_mode", "") or "")
        if mem_mode:
            if self.is_swapping_blocks:
                raise ValueError(
                    "--mem_reweight_mode requires blocks_to_swap=0 — the "
                    "Δ-gap measurement forward has not been validated against "
                    "the block-swap offloader and can silently desync it. "
                    "Use block compile for memory instead."
                )
            save_path = os.path.join(
                args.output_dir,
                f"{getattr(args, 'output_name', None) or 'lora'}_memgap.json",
            )
            self._state.mem_tracker = MemGapTracker(args, save_path)
            logger.info(
                f"memorization Δ-gap tracker enabled (mode={mem_mode}, "
                f"σ≤{args.mem_sigma_max}, K={args.mem_measure_every}) — "
                f"state → {save_path}"
            )

        return model, text_encoders

    # Strategy construction + singleton installation lives in
    # library/anima/strategy.py (setup_training_strategies /
    # setup_text_encoder_outputs_caching_strategy) — the training-side
    # counterpart of library/inference/text.py::ensure_text_strategies.

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_noise_scheduler(
        self, args: argparse.Namespace, device: torch.device
    ) -> Any:
        noise_scheduler = noise_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        return noise_scheduler

    # ------------------------------------------------------------------
    # Per-step forward phases (issues.md P2.1)
    #
    # ``get_noise_pred_and_target`` is a flat sequence of named phases; the
    # conditional logic of each phase lives INSIDE it, never as lexical
    # nesting around it. "Always per step" is therefore structurally evident
    # at the call site — the silent-REPA dispatch bug was exactly an
    # always-phase written inside a sometimes-branch.
    # ------------------------------------------------------------------

    def _step_ctx(self, ctx: TrainCtx) -> StepCtx:
        return StepCtx(
            args=ctx.args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )

    def _prime_adapters(self, ctx: TrainCtx, batch, latents, *, is_train) -> None:
        """ALWAYS per step. Method-adapter pre-forward priming.

        IP-Adapter encodes the reference image and primes per-block K/V;
        EasyControl runs the cond pre-pass and primes per-block (K_c, V_c).
        Both run on the 4D latent layout the patched DiT forward expects. The
        patched cross-attn / self-attn closures consume the primed tensors
        during attention."""
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            adapter.prime_for_forward(step_ctx, batch, latents, is_train=is_train)

    def _paired_step_generators(self, args, device, is_train):
        """Common-random-numbers mode (``--paired_step_rng``): per-train-step
        ``(g_sigma, g_noise)`` generators seeded from (seed, step counter),
        decoupled from the global torch stream. Arms sharing a seed see the
        identical σ sequence and identical noise on same-shape steps, so
        checkpoint deltas isolate the intervention (the noise-lottery control
        for the sigma_lowres threshold sweep). ``(None, None)`` when off."""
        if not is_train or not getattr(args, "paired_step_rng", False):
            return None, None
        counter = getattr(self, "_paired_step_counter", 0) + 1
        self._paired_step_counter = counter
        base = (int(getattr(args, "seed", 0) or 0) * 1_000_003 + counter) * 2
        mask = (1 << 62) - 1
        g_sigma = torch.Generator(device=device).manual_seed(base & mask)
        g_noise = torch.Generator(device=device).manual_seed((base + 1) & mask)
        return g_sigma, g_noise

    @staticmethod
    def _sigma_route(args):
        """Parsed ``--sigma_lowres_route`` as ``(native_edge, demote_edge)``.

        Defaults to the measured-safe 1024:896 (``SIGMA_DEMOTE_ROUTE``);
        validates the shape once per call site.
        """
        raw = str(getattr(args, "sigma_lowres_route", None) or "1024:896")
        try:
            native, demote = (int(x) for x in raw.split(":"))
        except ValueError:
            raise ValueError(
                f"--sigma_lowres_route expects NATIVE:DEMOTE (e.g. 1024:896), "
                f"got {raw!r}"
            ) from None
        if not (native > demote > 0):
            raise ValueError(
                f"--sigma_lowres_route needs NATIVE > DEMOTE > 0, got {raw!r}"
            )
        return native, demote

    @staticmethod
    def _yarnsig_params(args):
        """Parsed ``--sigma_lowres_yarnsig`` as ``(alpha, beta, center, gamma)``,
        or None when off. Validates once; cached on the args namespace."""
        raw = getattr(args, "sigma_lowres_yarnsig", None)
        if not raw:
            return None
        cached = getattr(args, "_yarnsig_parsed", None)
        if cached is not None:
            return cached
        try:
            alpha, beta, center, gamma = (float(v) for v in raw.split(","))
        except ValueError:
            raise ValueError(
                f"--sigma_lowres_yarnsig expects ALPHA,BETA,CENTER,GAMMA, got {raw!r}"
            )
        if not (0.0 <= alpha < beta and 0.0 < center < 1.0 and gamma > 0.0):
            raise ValueError(
                "--sigma_lowres_yarnsig needs 0<=ALPHA<BETA, 0<CENTER<1, "
                f"GAMMA>0, got {raw!r}"
            )
        args._yarnsig_parsed = (alpha, beta, center, gamma)
        return args._yarnsig_parsed

    def _maybe_sigma_demote(
        self, ctx: TrainCtx, batch, latents, is_train, generator=None
    ):
        """sigma_lowres Phase 1b (σ > threshold → demote-tier latent).

        Returns ``(latents, sigmas_flat)``: possibly-swapped latents plus the
        pre-drawn flat σ to feed the sampler (None → sampler draws internally,
        the untouched default path). Active only when --sigma_lowres is on,
        the batch carries ``demoted_latents`` (train datasets with the sidecar
        enabled and the emit present), and this is a train step. ``generator``
        (paired-step-RNG mode) sources the σ draw.

        Side effect: ``self._yarnsig_step`` is (re)set every call — a
        ``(h_scale, w_scale, alpha, beta, mu)`` tuple on a demoted step under
        ``--sigma_lowres_yarnsig``, else None — consumed (and cleared) by the
        primary forward.
        """
        args = ctx.args
        self._yarnsig_step = None
        if not is_train or not getattr(args, "sigma_lowres", False):
            return latents, None
        demoted = batch.get("demoted_latents")
        if demoted is None:
            return latents, None
        unsafe = [
            a.name for a in self._adapters if not getattr(a, "sigma_demote_safe", False)
        ]
        if unsafe:
            raise ValueError(
                f"--sigma_lowres is unsupported under method adapter(s) "
                f"{unsafe} — fixed-grid cond/extra-forward streams need their "
                "own operating-point probe first (sigma_lowres Q5). "
                "Grid-agnostic adapters (repa) are allowed."
            )
        from library.runtime.noise import draw_flat_sigmas

        sigmas_flat = draw_flat_sigmas(
            args,
            latents.shape[0],
            latents.shape[-2],
            latents.shape[-1],
            ctx.accelerator.device,
            generator=generator,
        )
        if sigmas_flat is None:
            if not getattr(self, "_sigma_lowres_warned", False):
                self._sigma_lowres_warned = True
                logger.warning(
                    "sigma_lowres: timestep_sampling=%r has no flat-σ draw — "
                    "training native throughout (no demotion).",
                    getattr(args, "timestep_sampling", None),
                )
            return latents, None
        threshold = float(getattr(args, "sigma_lowres_threshold", 0.5))
        total = getattr(self, "_sigma_lowres_seen", 0) + 1
        self._sigma_lowres_seen = total
        if bool((sigmas_flat > threshold).all()):
            native_hw = tuple(latents.shape[-2:])
            latents = demoted.to(device=latents.device, dtype=latents.dtype)
            yarn = self._yarnsig_params(args)
            if yarn is not None:
                alpha, beta, center, gamma = yarn
                # Patch-grid units (patch_spatial=2 on the latent grid) — the
                # probe's per-axis stretch: demoted patch i sits at
                # i · (native_patches / demoted_patches), spanning the native
                # coordinate range. μ from the batch-min σ: the sample nearest
                # the gate is the one the low-σ liability was measured on.
                s = min(max(float(sigmas_flat.min()), 1e-6), 1.0 - 1e-6)
                mu = 1.0 / (
                    1.0
                    + math.exp(
                        -gamma
                        * (math.log(s / (1.0 - s)) - math.log(center / (1.0 - center)))
                    )
                )
                self._yarnsig_step = (
                    (native_hw[0] // 2) / (latents.shape[-2] // 2),
                    (native_hw[1] // 2) / (latents.shape[-1] // 2),
                    alpha,
                    beta,
                    mu,
                )
            demoted_n = getattr(self, "_sigma_lowres_demoted", 0) + 1
            self._sigma_lowres_demoted = demoted_n
            if demoted_n == 1:
                logger.info(
                    "sigma_lowres: first demoted step — latent grid %s → %s at σ=%s%s",
                    native_hw,
                    tuple(latents.shape[-2:]),
                    [round(float(s), 3) for s in sigmas_flat],
                    (
                        f" (yarnsig μ={self._yarnsig_step[-1]:.3f})"
                        if self._yarnsig_step is not None
                        else ""
                    ),
                )
        else:
            demoted_n = getattr(self, "_sigma_lowres_demoted", 0)
        if total % 500 == 0:
            logger.info(
                "sigma_lowres: demoted %d/%d eligible steps (%.1f%%)",
                demoted_n,
                total,
                100.0 * demoted_n / total,
            )
        return latents, sigmas_flat

    def _sample_noisy_input(
        self, ctx: TrainCtx, latents, noise, *, is_train, sigmas=None
    ):
        """ALWAYS per step. Draw (noisy input, timesteps, sigmas) via the
        sampler registry (M1) and run per-step network router conditioning
        (timestep masks, σ/FEI routers, balance-loss warmup). ``sigmas``
        (pre-drawn flat σ, sigma_lowres's σ-first path) skips the in-sampler
        draw."""
        args = ctx.args
        sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
        sampler_out = sampler_fn(
            SamplerContext(
                args=args,
                noise_scheduler=ctx.noise_scheduler,
                latents=latents,
                noise=noise,
                device=ctx.accelerator.device,
                weight_dtype=ctx.weight_dtype,
                sigmas=sigmas,
            )
        )
        # timesteps are [0,1]-scaled, float32.
        self._hydra_warmup_step = apply_router_conditioning(
            network=ctx.network,
            noisy_model_input=sampler_out.noisy_input,
            timesteps=sampler_out.timesteps,
            is_train=is_train,
            warmup_step=int(getattr(self, "_hydra_warmup_step", 0)),
            max_train_steps=int(getattr(args, "max_train_steps", 0) or 0),
            gradient_accumulation_steps=int(
                getattr(args, "gradient_accumulation_steps", 1) or 1
            ),
        )
        return sampler_out.noisy_input, sampler_out.timesteps, sampler_out.sigmas

    def _prepare_conditioning(
        self, ctx: TrainCtx, batch, text_encoder_conds, noisy_model_input
    ):
        """ALWAYS per step. Returns the device-resident ``PreparedTextConds``
        (both text-conditioning modes; normalized to one uniform
        ``ForwardConditioning`` at the forward call site); fires the
        text-conditioned routers (each gated internally on cached crossattn);
        marks grad-checkpointing inputs."""
        args = ctx.args
        network = ctx.network

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            # Only require grads for text conditions when training the text encoder.
            # When using cached text encoder outputs (or training DiT-only), requiring grads here adds backward work.
            if self.is_train_text_encoder(args) and not args.cache_text_encoder_outputs:
                for t in text_encoder_conds:
                    if t is not None and t.dtype.is_floating_point:
                        t.requires_grad_(True)

        # Unpack text encoder conditions, H2D move, and on-device caption dropout.
        tc = prepare_text_conds(
            text_encoder_conds=text_encoder_conds,
            batch=batch,
            text_encoding_strategy=ctx.text_encoding_strategy,
            network=network,
            device=ctx.accelerator.device,
            weight_dtype=ctx.weight_dtype,
            uncond_crossattn_emb=self._state.uncond_crossattn_1,
        )

        # ChimeraHydra global content router (chimera with
        # ``content_router_source="crossattn"``): fire ONCE per step on the
        # pooled crossattn_emb. apply_router_conditioning ran before text
        # conds were materialized, so the content router lives outside that
        # helper. No-op on non-chimera networks or per-Linear chimera.
        if (
            getattr(network, "use_content_router", False)
            and tc.crossattn_emb is not None
            and hasattr(network, "set_content")
        ):
            network.set_content(tc.crossattn_emb)

        # Network-level GlobalRouter routed on pooled text
        # (``router_source="crossattn_emb"``, route_per_layer=False). Same
        # timing rationale as the content router above — fires once per step
        # on the materialized cross-attn text features. No-op otherwise.
        if (
            getattr(network, "use_crossattn_router", False)
            and tc.crossattn_emb is not None
            and hasattr(network, "set_crossattn_routing")
        ):
            network.set_crossattn_routing(tc.crossattn_emb)

        return tc

    def _get_padding_mask(self, latents, *, weight_dtype, device):
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask_key = (bs, h_latent, w_latent, weight_dtype, device)
        padding_mask = self._padding_mask_cache.get(padding_mask_key)
        if padding_mask is None:
            padding_mask = torch.zeros(
                bs, 1, h_latent, w_latent, dtype=weight_dtype, device=device
            )
            self._padding_mask_cache[padding_mask_key] = padding_mask
        return padding_mask

    def _run_primary_forward(
        self, ctx: TrainCtx, *, anima, noisy_model_input, timesteps, tc, padding_mask
    ):
        """ALWAYS per step. Single, branch-free forward call site (issues.md
        P2.3): both text-conditioning modes normalize to ONE uniform
        ``ForwardConditioning`` (cond, kw) bundle first — the mode split is
        data prep in ``build_forward_conditioning``, not control flow around
        the call. The normalization (postfix splice runs learned modules)
        must happen inside the primary forward's autocast / grad scope, which
        is why it lives here rather than in ``_prepare_conditioning``.
        Returns ``(model_pred, cond)``; ``cond`` is also consumed by the
        aux-loss and adapter-dispatch phases after the forward."""
        cond = build_forward_conditioning(
            network=ctx.network, tc=tc, timesteps=timesteps
        )
        model_pred = anima(
            noisy_model_input,
            timesteps,
            cond.cond,
            padding_mask=padding_mask,
            **cond.kw,
        )
        return model_pred, cond

    def _attach_aux_losses(
        self,
        ctx: TrainCtx,
        *,
        anima,
        batch,
        latents,
        noise,
        sigmas,
        timesteps,
        noisy_model_input,
        cond: ForwardConditioning,
        padding_mask,
        is_train,
    ) -> None:
        """Trainer-owned aux-loss producers riding the primary forward (func
        inversion loss, VR control variate). Every gate lives INSIDE this
        phase — including the cached-text requirement (``cond.crossattn_emb
        is not None``), which used to be implied by lexical position inside
        the else-branch. Must run inside the primary forward's autocast /
        grad scope (extra ``anima(...)`` calls)."""
        args = ctx.args

        # Functional MSE loss against a sampled stochastic inversion run.
        # The captures dict is populated by trainer-owned forward hooks
        # on cross_attn.output_proj at ``self._func_blocks``.
        self._func_loss = None
        if (
            is_train
            and getattr(self, "_func_blocks", None)
            and cond.crossattn_emb is not None
        ):
            self._func_loss = compute_inversion_func_loss(
                anima_call=anima,
                captures=self._func_captures,
                block_indices=self._func_blocks,
                batch=batch,
                noisy_model_input=noisy_model_input,
                timesteps=timesteps,
                padding_mask=padding_mask,
                has_postfix=cond.has_postfix,
                kw=cond.kw,
                device=ctx.accelerator.device,
                dtype=ctx.weight_dtype,
            )

        # Variance-reduced FM control variate (AsymFlow §5.2). Stash the
        # residual `z` so the loss composer can blend `(y + λ·z)²`.
        if (
            is_train
            and float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0
            and cond.crossattn_emb is not None
        ):
            z_residual = run_vr_reference_forward(
                anima_call=anima,
                network=ctx.network,
                latents=latents,
                noise=noise,
                sigmas=sigmas,
                timesteps=timesteps,
                crossattn_emb=cond.crossattn_emb,
                padding_mask=padding_mask,
                forward_kwargs=cond.kw,
                weight_dtype=ctx.weight_dtype,
                fei_sigma_low_div=float(args.vr_fei_sigma_low_div),
            )
            self._state.extras_for_step["vr"] = {
                "z": z_residual.detach(),
                "state": self._state.vr,
            }

        # Online memorization Δ-gap (mem_reweight.py). Producer half: the
        # causal per-item weights (from state BEFORE this step's measurement)
        # plus, on measurement steps with any σ ≤ sigma_max draw, the base
        # forward's per-sample log-MSE on the identical (x_t, ε, σ). The
        # consumer half (EMA update + loss_weights multiply) lives at the
        # loss site in ``_process_batch_inner`` where model_pred/target exist.
        tracker = self._state.mem_tracker
        if (
            is_train
            and tracker is not None
            and cond.crossattn_emb is not None
            and "image_keys" in batch
        ):
            sig = sigmas.reshape(sigmas.shape[0], -1)[:, 0].float()
            low_mask = sig <= tracker.sigma_max
            keys = list(batch["image_keys"])
            base_logmse = None
            grid_deltas = None
            if tracker.extra_sigmas:
                # Multi-draw mode: fixed σ grid × antithetic ε, every visit
                # (not gated on the train draw's σ) — the Δ is computed fully
                # here; the consumer only folds it into the EMA.
                if tracker.should_measure():
                    grid_deltas = measure_grid_delta(
                        anima_call=anima,
                        network=ctx.network,
                        latents=latents,
                        crossattn_emb=cond.crossattn_emb,
                        padding_mask=padding_mask,
                        forward_kwargs=cond.kw,
                        model_dtype=noisy_model_input.dtype,
                        sigmas=tracker.extra_sigmas,
                        n_noise=tracker.extra_noise,
                        delta=tracker.delta,
                        generator=tracker.noise_generator(latents.device),
                    )
            elif tracker.should_measure() and bool(low_mask.any()):
                base_logmse = measure_base_logmse(
                    anima_call=anima,
                    network=ctx.network,
                    noisy_model_input=noisy_model_input,
                    timesteps=timesteps,
                    crossattn_emb=cond.crossattn_emb,
                    padding_mask=padding_mask,
                    forward_kwargs=cond.kw,
                    noise=noise,
                    latents=latents,
                    delta=tracker.delta,
                )
            self._state.extras_for_step["mem_gap"] = {
                "keys": keys,
                "weights": tracker.weights(keys, low_mask.tolist()),
                "low_mask": low_mask,
                "base_logmse": base_logmse,
                "grid_deltas": grid_deltas,
            }

    def _dispatch_adapter_extras(
        self, ctx: TrainCtx, primary: ForwardArtifacts
    ) -> None:
        """ALWAYS per step — both text-conditioning paths. Method-adapter
        extra forwards (soft-tokens, REPA, …).

        This dispatch used to live inside the cached-crossattn else-branch
        only, which silently skipped every adapter's aux loss on the in-model
        text path (crossattn_emb=None — EasyControl's default; REPA trained
        as baseline). Each adapter sees the primary forward's inputs + 5D
        output and may run additional anima(...) calls inside the same
        autocast / grad scope, returning aux loss tensors keyed for the
        LossComposer."""
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            out = adapter.extra_forwards(step_ctx, primary)
            if out:
                self._state.extras_for_step.update(out)

    def get_noise_pred_and_target(
        self,
        ctx: TrainCtx,
        latents,
        batch,
        text_encoder_conds,
        *,
        is_train=True,
    ):
        accelerator = ctx.accelerator
        anima: anima_models.Anima = ctx.unet

        # Reset per-step adapter aux so stale tensors from a prior step can't
        # leak into the loss composer.
        self._state.extras_for_step = {}

        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        # Paired-step RNG (CRN): dedicated per-step generators for σ + noise,
        # so A/B arms sharing a seed stay noise-locked. (None, None) when off.
        g_sigma, g_noise = self._paired_step_generators(
            ctx.args, ctx.accelerator.device, is_train
        )

        # sigma_lowres Phase 1b: σ-first draw + latent swap. When the batch
        # carries a demote sibling and EVERY sample's σ clears the gate, the
        # whole step (input, target, masks, REPA grid) runs on the demoted
        # grid — exactly the probe's measured-safe arm. The σ marginal is
        # untouched (drawn unconditionally from the same density, merely
        # before the noise), and a native step at any σ is always valid, so
        # the all-samples rule is exact at train_batch_size=1 and
        # conservative (fewer demotes, never an unsafe one) above.
        latents, sigmas_flat = self._maybe_sigma_demote(
            ctx, batch, latents, is_train, generator=g_sigma
        )
        if is_train:
            # Realized patch-token histogram (per-arm FLOPs accounting, E4):
            # counts examples at the grid the step ACTUALLY ran on (post
            # demote swap). Emitted with the run_end progress event.
            tok = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)
            self._token_step_hist[tok] = self._token_step_hist.get(tok, 0) + int(
                latents.shape[0]
            )
        if sigmas_flat is None and g_sigma is not None:
            # Paired mode on a batch the demote path didn't draw for (no
            # --sigma_lowres, or an off-route/un-emitted batch): still take σ
            # from the paired stream so every arm shares the σ sequence.
            from library.runtime.noise import draw_flat_sigmas

            sigmas_flat = draw_flat_sigmas(
                ctx.args,
                latents.shape[0],
                latents.shape[-2],
                latents.shape[-1],
                ctx.accelerator.device,
                generator=g_sigma,
            )

        self._prime_adapters(ctx, batch, latents, is_train=is_train)
        if g_noise is not None:
            noise = torch.randn(
                latents.shape,
                generator=g_noise,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = self._sample_noisy_input(
            ctx, latents, noise, is_train=is_train, sigmas=sigmas_flat
        )
        tc = self._prepare_conditioning(
            ctx, batch, text_encoder_conds, noisy_model_input
        )
        padding_mask = self._get_padding_mask(
            latents, weight_dtype=ctx.weight_dtype, device=accelerator.device
        )
        noisy_model_input = noisy_model_input.unsqueeze(
            2
        )  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]

        # yarnsig: expose this demoted step's banded-rope params on the model
        # for exactly the span of its forward(s); cleared in the finally so a
        # later native/val/sample forward can never inherit them.
        if getattr(self, "_yarnsig_step", None) is not None:
            anima._sigma_lowres_yarn = self._yarnsig_step
        try:
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred, cond = self._run_primary_forward(
                    ctx,
                    anima=anima,
                    noisy_model_input=noisy_model_input,
                    timesteps=timesteps,
                    tc=tc,
                    padding_mask=padding_mask,
                )
                self._attach_aux_losses(
                    ctx,
                    anima=anima,
                    batch=batch,
                    latents=latents,
                    noise=noise,
                    sigmas=sigmas,
                    timesteps=timesteps,
                    noisy_model_input=noisy_model_input,
                    cond=cond,
                    padding_mask=padding_mask,
                    is_train=is_train,
                )
                self._dispatch_adapter_extras(
                    ctx,
                    ForwardArtifacts(
                        anima_call=anima,
                        noisy_model_input=noisy_model_input,
                        timesteps=timesteps,
                        crossattn_emb=cond.crossattn_emb,
                        padding_mask=padding_mask,
                        forward_kwargs=cond.kw,
                        model_pred=model_pred,
                        noise=noise,
                        latents=latents,
                        is_train=is_train,
                    ),
                )
        finally:
            anima._sigma_lowres_yarn = None
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Note: do NOT clear timestep mask here -- gradient checkpointing recomputes the forward
        # pass during backward, so the mask must remain set. It gets overwritten on the next step.

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=ctx.args.weighting_scheme, sigmas=sigmas
        )

        return model_pred, target, timesteps, weighting

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
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()
        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
            network=network,
        )

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # Re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            return accelerator.prepare(unet)

        model = unet
        model = accelerator.prepare(
            model, device_placement=[not self.is_swapping_blocks]
        )
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(
            accelerator.device
        )
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model

    def on_validation_step_end(self, ctx: TrainCtx, batch):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            ctx.accelerator.unwrap_model(ctx.unet).prepare_block_swap_before_forward()

    def process_batch(
        self,
        ctx: TrainCtx,
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        """Override base process_batch to surface caption_dropout_rates for on-device dropout."""

        # The cached text-encoder outputs list arrives as
        # [..., caption_dropout_rates] from the dataset (see strategy.py
        # cache layout). Split the trailing rates tensor off so the inner
        # path sees the canonical 4- or 5-element conds list, and stash the
        # rates on the batch -- get_noise_pred_and_target applies the dropout
        # in-place after the H2D transfer. Doing it here on CPU would clone
        # prompt_embeds / crossattn_emb on the critical path before the H2D
        # copy, blocking the main thread.
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            encoder_outputs = text_encoder_outputs_list[:-1]
            # Shallow copy so the original list (with rates appended) stays
            # intact for validation's per-sigma loop that reuses the batch.
            batch = {
                **batch,
                "text_encoder_outputs_list": encoder_outputs,
                "caption_dropout_rates": caption_dropout_rates,
            }

        return self._process_batch_inner(ctx, batch, is_train=is_train)

    def _process_batch_inner(
        self,
        ctx: TrainCtx,
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        """
        Process a batch for the network (original NetworkTrainer.process_batch logic)
        """
        args = ctx.args
        accelerator = ctx.accelerator
        network = ctx.network
        vae = ctx.vae
        text_encoders = ctx.text_encoders
        text_encoding_strategy = ctx.text_encoding_strategy
        tokenize_strategy = ctx.tokenize_strategy
        noise_scheduler = ctx.noise_scheduler
        vae_dtype = ctx.vae_dtype
        weight_dtype = ctx.weight_dtype
        train_text_encoder = ctx.train_text_encoder
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(
                    torch.FloatTensor, batch["latents"].to(accelerator.device)
                )
            else:
                if (
                    args.vae_batch_size is None
                    or len(batch["images"]) <= args.vae_batch_size
                ):
                    latents = vae.encode_pixels_to_latents(
                        batch["images"].to(accelerator.device, dtype=vae_dtype)
                    )
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size]
                        for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = vae.encode_pixels_to_latents(
                                chunk.to(accelerator.device, dtype=vae_dtype)
                            )
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(
                        torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents)
                    )

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = (
                text_encoder_outputs_list  # List of text encoder outputs
            )

        if (
            len(text_encoder_conds) == 0
            or all(c is None for c in text_encoder_conds)
            or train_text_encoder
        ):
            with (
                torch.set_grad_enabled(is_train and train_text_encoder),
                accelerator.autocast(),
            ):
                if args.weighted_captions:
                    input_ids_list, weights_list = (
                        tokenize_strategy.tokenize_with_weights(batch["captions"])
                    )
                    encoded_text_encoder_conds = (
                        text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(
                                args, accelerator, text_encoders
                            ),
                            input_ids_list,
                            weights_list,
                        )
                    )
                else:
                    input_ids = [
                        ids.to(accelerator.device) for ids in batch["input_ids_list"]
                    ]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(
                            args, accelerator, text_encoders
                        ),
                        input_ids,
                    )

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # Step-owning adapter override: a method with no `target = noise -
        # latents` and its own multi-forward objective (BYG) computes the whole
        # scalar loss here, bypassing get_noise_pred_and_target + LossComposer.
        owners = [a for a in self._adapters if a.owns_training_step(args)]
        if owners:
            assert len(owners) == 1, (
                f"at most one adapter may own the training step; got {len(owners)}: "
                f"{[a.name for a in owners]}"
            )
            return owners[0].compute_loss(
                ComputeLossCtx(
                    args=args,
                    accelerator=accelerator,
                    network=getattr(self, "_network", network),
                    unet=ctx.unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    batch=batch,
                    latents=latents,
                    text_encoder_conds=text_encoder_conds,
                    is_train=is_train,
                )
            )

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            ctx,
            latents,
            batch,
            text_encoder_conds,
            is_train=is_train,
        )

        huber_c = get_huber_threshold_if_needed(args, timesteps, noise_scheduler)

        # Assemble aux dict for the composer: extra_forwards returns from each
        # method adapter plus the trainer-owned functional-loss capture.
        loss_aux: dict = dict(self._state.extras_for_step)

        func_loss = getattr(self, "_func_loss", None)
        if func_loss is not None:
            loss_aux["func_loss"] = func_loss

        # Online memorization Δ-gap, consumer half (producer: the mem_gap
        # block in ``_attach_aux_losses``). Fold this step's paired Δ into the
        # per-item EMA (measurement steps only), then apply the CAUSAL weights
        # (computed from pre-step state) onto the per-sample loss_weights.
        loss_weights = batch["loss_weights"]
        mem_gap = loss_aux.pop("mem_gap", None)
        if mem_gap is not None:
            tracker = self._state.mem_tracker
            if mem_gap.get("grid_deltas") is not None:
                # Multi-draw Δ was fully computed in the producer, for every
                # batch item regardless of the train draw's σ.
                tracker.update(mem_gap["keys"], mem_gap["grid_deltas"].tolist())
            elif mem_gap["base_logmse"] is not None:
                with torch.no_grad():
                    d = mem_gap["base_logmse"] - adapted_logmse(
                        noise_pred, target, tracker.delta
                    )
                low = mem_gap["low_mask"].tolist()
                tracker.update(
                    [k for k, m in zip(mem_gap["keys"], low) if m],
                    [v for v, m in zip(d.tolist(), low) if m],
                )
            w = mem_gap["weights"]
            if any(x != 1.0 for x in w):
                loss_weights = loss_weights * torch.tensor(
                    w, device=loss_weights.device, dtype=loss_weights.dtype
                )

        composer = build_loss_composer(
            args, getattr(self, "_network", network), ledger=self._liveness
        )

        def _build_loss_ctx(aux: dict) -> LossContext:
            return LossContext(
                args=args,
                batch=batch,
                model_pred=noise_pred,
                target=target,
                timesteps=timesteps,
                weighting=weighting,
                huber_c=huber_c,
                loss_weights=loss_weights,
                network=getattr(self, "_network", network),
                aux=aux,
                is_train=is_train,
            )

        return composer.compose(_build_loss_ctx(loss_aux))

    # endregion

    # region Methods only in NetworkTrainer (not overridden by Anima)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        self._network = (
            network  # composer reads _network for ortho / balance regularizers
        )
        # Aux-loss gating convention (library/training/losses.py docstring):
        # handlers read network._<name>_weight. functional's weight is a
        # top-level training arg, so the trainer stamps it here.
        network._functional_loss_weight = float(
            getattr(args, "functional_loss_weight", 0.0) or 0.0
        )
        self._func_loss = None
        self._func_hooks = []
        self._func_captures = {}
        self._func_blocks = []
        if getattr(args, "functional_loss_weight", 0.0) > 0.0 and getattr(
            args, "inversion_dir", None
        ):
            blocks_str = getattr(args, "functional_loss_blocks", "8,12,16,20")
            try:
                self._func_blocks = sorted(
                    int(b.strip()) for b in blocks_str.split(",") if b.strip()
                )
            except ValueError as e:
                raise ValueError(
                    f"functional_loss_blocks must be comma-separated integers, got {blocks_str!r}"
                ) from e

            def _make_hook(block_idx: int):
                def _hook(_module, _inputs, output):
                    # Save the cross_attn.output_proj output for this block.
                    # Hook fires twice per step (main forward + inversion forward);
                    # the main forward runs first, we snapshot before second forward overwrites.
                    self._func_captures[block_idx] = output

                return _hook

            blocks_list = unet.blocks  # nn.ModuleList of 28 Anima DiT blocks
            num_blocks = len(blocks_list)
            for bi in self._func_blocks:
                if not (0 <= bi < num_blocks):
                    raise ValueError(
                        f"functional_loss_blocks contains out-of-range index {bi} (model has {num_blocks} blocks)"
                    )
                module = blocks_list[bi].cross_attn.output_proj
                h = module.register_forward_hook(_make_hook(bi))
                self._func_hooks.append(h)
            logger.info(
                f"Functional loss enabled: hooks on cross_attn.output_proj at blocks {self._func_blocks}, "
                f"weight={args.functional_loss_weight}, num_runs={args.functional_loss_num_runs}"
            )

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(
            args, lora=True
        ).to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_sigmoid_bias"] = getattr(args, "sigmoid_bias", 0.0)
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def get_text_encoders_train_flags(self, args, text_encoders):
        return (
            [True] * len(text_encoders)
            if self.is_train_text_encoder(args)
            else [False] * len(text_encoders)
        )

    def on_step_start(self, ctx: TrainCtx, batch, *, is_train: bool = True):
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            adapter.on_step_start(step_ctx, batch, is_train=is_train)

    def run_after_backward(self, ctx: TrainCtx):
        """Dispatch the post-backward hook to adapters (between
        ``accelerator.backward`` and gradient clipping)."""
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            adapter.after_backward(step_ctx)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator: Accelerator,
        text_encoders,
        dataset: DatasetGroup,
    ):
        if not args.cache_text_encoder_outputs:
            # Live-encoding mode (e.g. IP-Adapter cache_text_encoder_outputs=false):
            # move the text encoder to device for per-step encoding.
            text_encoders[0].to(accelerator.device)
            return

        # With caching on, the on-disk cache is guaranteed complete (asserted in
        # train(), including the LLM adapter's crossattn_emb outputs, which
        # preprocess writes). The dataset thus never needs encoding here — run
        # the pass with no model purely to populate
        # ImageInfo.text_encoder_outputs_npz (forms no batches).
        dataset.new_cache_text_encoder_outputs([None], accelerator)

        # The text encoder is in memory only to encode sample prompts (TE
        # training is mutually exclusive with caching). It is None when no
        # sample prompts are configured — nothing left to do.
        if text_encoders[0] is not None and args.sample_prompts is not None:
            logger.info(
                f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}"
            )
            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
            text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [
                        prompt_dict.get("prompt", ""),
                        prompt_dict.get("negative_prompt", ""),
                    ]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"  cache TE outputs for: {p}")
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = (
                                text_encoding_strategy.encode_tokens(
                                    tokenize_strategy,
                                    text_encoders,
                                    tokens_and_masks,
                                )
                            )
            self.sample_prompts_te_outputs = sample_prompts_te_outputs

            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # endregion

    # region Main training loop

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

    def _prepare_dataset(self, args) -> DatasetBundle:
        """Build train/val dataset groups and the collator shared by both loaders."""
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(
                ConfigSanitizer(support_dropout=True)
            )
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                base_ds = load_dataset_config_from_base(
                    overrides=vars(args),
                    method=getattr(args, "method", None),
                    methods_subdir=getattr(args, "methods_subdir", None) or "methods",
                    config_file=(
                        getattr(args, "config_file", None)
                        if getattr(args, "method", None) is None
                        else None
                    ),
                )
                if base_ds is not None:
                    if getattr(args, "method", None) is None and getattr(
                        args, "config_file", None
                    ):
                        logger.info("Loading dataset config from config_file")
                    else:
                        logger.info("Loading dataset config from configs/base.toml")
                    user_config = base_ds
                    use_user_config = True
                elif use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            # Global --sample_ratio override (used by the `[half]` preset and
            # the GUI's data-scope field). 1.0 is inert — base.toml ships it as
            # the visible default, and skipping the no-op keeps any per-subset
            # sample_ratio in a custom blueprint authoritative.
            sample_ratio = getattr(args, "sample_ratio", None)
            if sample_ratio is not None and sample_ratio != 1.0:
                for ds in user_config.get("datasets", []):
                    for sub in ds.get("subsets", []):
                        sub["sample_ratio"] = sample_ratio
                logger.info(f"Applied --sample_ratio={sample_ratio} to all subsets")

            # --artists_shard k_N: restrict training to one round-robin shard of
            # the artist subdirs, expanded into each subset's path_pattern before
            # the blueprint is built (so _derive_token_budget's path_pattern-
            # filtered count and validation thresholds all see the shard).
            artists_shard = getattr(args, "artists_shard", None)
            if artists_shard:
                if getattr(args, "path_pattern", None):
                    raise ValueError(
                        "--artists_shard and --path_pattern are mutually "
                        "exclusive (the shard expands to a path_pattern)."
                    )
                from library.datasets.artist_shard import apply_artist_shard
                from library.env import resolve_under_home

                shard_info = apply_artist_shard(
                    user_config, artists_shard, resolve=resolve_under_home
                )
                for image_dir, meta in shard_info.items():
                    logger.info(
                        "Applied --artists_shard=%s to %s: %d/%d artists [%s]",
                        artists_shard,
                        image_dir,
                        meta["n_shard"],
                        meta["n_artists"],
                        ", ".join(meta["artists"]),
                    )

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = (
                config_util.generate_dataset_group_by_blueprint(
                    blueprint.dataset_group,
                    # Free-fit (the only resize mode): the predefined bucket set is
                    # the union of the on-disk resized sizes, so every cached latent
                    # exact-matches its own (W, H) and nothing AR-snaps. target_res
                    # is preprocess-only and inert here — the on-disk caches decide
                    # which tiers/shapes are present, not this list.
                    target_res=getattr(args, "target_res", None),
                )
            )

            rates = [
                subset.caption_dropout_rate
                for ds in train_dataset_group.datasets
                for subset in ds.subsets
            ]
            self._state.caption_dropout_enabled = bool(rates) and any(
                r > 0 for r in rates
            )
            if self._state.caption_dropout_enabled:
                logger.info(f"caption dropout ENABLED -- per-subset rates: {rates}")
            else:
                logger.info("caption dropout DISABLED (rate=0.0 on all subsets)")
        else:
            # use arbitrary dataset class
            train_dataset_group = load_arbitrary_dataset(args)
            val_dataset_group = (
                None  # placeholder until validation dataset supported for arbitrary
            )

        # sigma_lowres Phase 1b: activate the σ-demote sidecar on the TRAIN
        # datasets only — validation stays native so val loss is comparable
        # across the A/B arms the gate requires.
        if getattr(args, "sigma_lowres", False):
            route = self._sigma_route(args)

            enabled_on = 0
            for ds in getattr(train_dataset_group, "datasets", []):
                if hasattr(ds, "enable_sigma_demote"):
                    ds.enable_sigma_demote(*route)
                    enabled_on += 1
            logger.info(
                "sigma_lowres ENABLED: %d→%d demote at σ > %s on %d train "
                "dataset(s); validation stays native.",
                route[0],
                route[1],
                getattr(args, "sigma_lowres_threshold", 0.5),
                enabled_on,
            )
            yarn = self._yarnsig_params(args)  # validates the format up front
            if yarn is not None:
                logger.info(
                    "sigma_lowres yarnsig ENABLED: banded rope on demoted "
                    "steps, α,β=%s,%s, μ=sigmoid(%s·[logit(σ)−logit(%s)]).",
                    *(yarn[0], yarn[1], yarn[3], yarn[2]),
                )
        elif getattr(args, "sigma_lowres_yarnsig", None):
            raise ValueError(
                "--sigma_lowres_yarnsig requires --sigma_lowres (it only "
                "changes rope on demoted steps, which need the demote sidecar)."
            )

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        return DatasetBundle(
            train_group=train_dataset_group,
            val_group=val_dataset_group,
            current_epoch=current_epoch,
            current_step=current_step,
            collator=collator,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
        )

    def _derive_token_budget(self, args, train_group, val_group):
        """(n_token_families, seq_range) from the buckets the datasets populate.

        Reads each dataset's ``bucket_manager.resos`` (the buckets at least one
        selected image landed in) and reduces to the set of distinct token counts,
        unioned with the token counts the sample prompts will request (see
        ``_sample_prompt_token_counts``). This sizes ``compile_blocks``' dynamo
        cache to exactly the tiers on disk for this run — independent of
        ``args.target_res``. Returns ``(None, None)`` when no bucketed resos are
        available (e.g. a MinimalDataset), leaving compile_blocks on its own
        defaults.
        """
        from library.datasets.buckets import token_counts_for_resos

        resos: set = set()
        for group in (train_group, val_group):
            if group is None:
                continue
            for dataset in getattr(group, "datasets", []):
                bm = getattr(dataset, "bucket_manager", None)
                if bm is not None:
                    resos.update(bm.resos)
        if not resos:
            return None, None
        counts = token_counts_for_resos(resos) | self._sample_prompt_token_counts(args)
        # sigma_lowres: demoted forwards run at the demote tier's token counts,
        # which must sit inside the compiled dynamic-seq range (same failure
        # mode as #42's out-of-range sample prompts).
        if getattr(args, "sigma_lowres", False):
            from library.datasets.buckets import demoted_token_counts

            counts |= demoted_token_counts(resos, *self._sigma_route(args))
        return len(counts), (min(counts), max(counts))

    def _sample_prompt_token_counts(self, args) -> set:
        """Token counts the sample prompts will request; empty when sampling is off.

        Sample generation runs through the same compiled blocks as training, so a
        sample resolution outside the training buckets (e.g. ``--w 1024 --h 1536``
        over 1024-tier data) would land outside the dynamic-seq mark_dynamic range
        and crash the run mid-training with a ConstraintViolationError (#42).
        Folding the prompt resolutions into the budget compiles for them up front.
        Prompts are re-read from disk at every sample event, so resolutions added
        to the file mid-run are NOT covered here — those are skipped with a
        warning at sample time instead (``_sample_image_inference``).
        """
        from library.datasets.buckets import token_counts_for_sample_prompts

        if not getattr(args, "sample_prompts", None):
            return set()
        will_sample = (
            getattr(args, "sample_at_first", False)
            or getattr(args, "sample_every_n_steps", None)
            or getattr(args, "sample_every_n_epochs", None)
        )
        if not will_sample:
            return set()
        try:
            prompts = train_util.load_prompts(args.sample_prompts)
        except Exception as e:
            logger.warning(
                f"Could not parse sample prompts ({args.sample_prompts}) for the "
                f"compile token budget: {e}. Sample resolutions outside the "
                "training buckets may be skipped under torch_compile."
            )
            return set()
        return token_counts_for_sample_prompts(prompts)

    def _create_and_apply_network(
        self,
        args,
        accelerator,
        vae,
        text_encoder,
        unet,
        text_encoders,
        weight_dtype,
    ) -> Optional[NetworkBundle]:
        """Import network module, merge base weights, build LoRA, apply to the model."""
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                if (
                    args.base_weights_multiplier is None
                    or len(args.base_weights_multiplier) <= i
                ):
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(
                    f"merging module: {weight_path} with multiplier {multiplier}"
                )

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(
                    text_encoder,
                    unet,
                    weights_sd,
                    weight_dtype,
                    accelerator.device if args.lowram else "cpu",
                )

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network — one resolved view of both config-intake paths
        # (--network_args + allowlisted top-level keys). Copied so the dropout
        # default below stays a factory-call detail, not part of the cached
        # ``args._network_kwargs`` view other consumers read.
        net_kwargs = dict(resolve_network_kwargs(args))

        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(
                1, args.network_weights, vae, text_encoder, unet, **net_kwargs
            )
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(
            network, "apply_max_norm_regularization"
        ):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it"
            )
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # Token-adding adapters (register tokens) do mid-stack token surgery
        # via block pre-hooks — unaudited against the block-swap offloader
        # (cf. [[project_blockswap_extra_forwards_gradcache]] for how the
        # offloader desyncs on unexpected per-step forward patterns).
        if (
            int(getattr(network, "extra_seq_tokens", 0) or 0) > 0
            and args.blocks_to_swap is not None
            and args.blocks_to_swap > 0
        ):
            raise ValueError(
                "Register tokens (extra_seq_tokens>0) require blocks_to_swap=0 — "
                "the mid-stack token insertion pre-hooks have not been validated "
                "against the block-swap offloader and can silently desync it. "
                "Use block compile for memory instead."
            )

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(
                f"load network weights from {args.network_weights}: {info}"
            )

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(
                text_encoders, self.get_text_encoders_train_flags(args, text_encoders)
            ):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()  # may have no effect

        # Native-shape flattening + per-block torch.compile. COMPILE LAST —
        # after apply_to + load_weights (above) so dynamo traces the adapter's
        # monkey-patched Linear forwards, not the bare DiT. The full sequence
        # (partitioner activation-memory budget → per-signature cache
        # isolation → compile_blocks → EasyControl cond-stream compile) lives
        # in library/runtime/harness.py with the other compile entry points.
        # Matches the harness order: block-swap → grad-ckpt → compile.
        if args.torch_compile:
            from library.runtime.harness import compile_blocks_for_training

            # Token-family budget derived from the buckets the dataset actually
            # populated (see _derive_token_budget) — not args.target_res, which is
            # a preprocess-only knob and inert at train time.
            n_token_families, seq_range = getattr(
                self, "_compile_token_budget", (None, None)
            )
            # Token-adding adapters (register tokens) grow the seq by a constant K,
            # so widen the dynamic-seq mark_dynamic bound's MAX by K or the compiled
            # block's bound is violated (ConstraintViolationError). The min stays:
            # with mid-stack insertion (insert_block > 0) blocks before the insert
            # still run at the bare seq, so one graph must cover [lo, hi+K]. The
            # family COUNT is unchanged (K is added uniformly), so n stays.
            extra_seq = int(getattr(network, "extra_seq_tokens", 0) or 0)
            if extra_seq and seq_range is not None:
                seq_range = (seq_range[0], seq_range[1] + extra_seq)
            compile_blocks_for_training(
                unet,
                network,
                backend=args.dynamo_backend,
                mode=getattr(args, "compile_inductor_mode", None),
                n_token_families=n_token_families,
                seq_range=seq_range,
                dynamic_seq=bool(getattr(args, "compile_dynamic_seq", False)),
                activation_memory_budget=float(
                    getattr(args, "activation_memory_budget", 1.0) or 1.0
                ),
                partitioner_recompute_views=bool(
                    getattr(args, "partitioner_recompute_views", False)
                ),
                partitioner_aggressive_recomputation=bool(
                    getattr(args, "partitioner_aggressive_recomputation", False)
                ),
                grad_ckpt=bool(getattr(args, "gradient_checkpointing", False)),
                logger=logger,
            )

        return NetworkBundle(
            network=network,
            net_kwargs=net_kwargs,
            train_unet=train_unet,
            train_text_encoder=train_text_encoder,
        )

    def _setup_optimizer_and_dataloader(
        self,
        args,
        accelerator,
        network,
        train_dataset_group,
        val_dataset_group,
        collator,
    ) -> OptimizerBundle:
        """Build optimizer, dataloaders, and LR scheduler; finalize max_train_steps."""
        accelerator.print("prepare optimizer, data loader etc.")

        # make backward compatibility for text_encoder_lr
        support_multiple_lrs = hasattr(
            network, "prepare_optimizer_params_with_multiple_te_lrs"
        )
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            if (
                args.text_encoder_lr is None
                or isinstance(args.text_encoder_lr, float)
                or isinstance(args.text_encoder_lr, int)
            ):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = (
                    None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
                )
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            else:
                results = network.prepare_optimizer_params(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError:
            trainable_params = network.prepare_optimizer_params(
                text_encoder_lr, args.unet_lr
            )
            lr_descriptions = None

        optimizer_name, optimizer_args, optimizer = get_optimizer(
            args, trainable_params
        )
        optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(
            optimizer, args
        )

        # prepare dataloader
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        persistent_workers = args.persistent_data_loader_workers and n_workers > 0

        dataloader_kwargs = {
            "batch_size": 1,
            "collate_fn": collator,
            "num_workers": n_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        if n_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            shuffle=True,
            **dataloader_kwargs,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            **dataloader_kwargs,
        )

        # Calculate training steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader)
                / accelerator.num_processes
                / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is"
            )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr scheduler
        lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

        return OptimizerBundle(
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            optimizer_train_fn=optimizer_train_fn,
            optimizer_eval_fn=optimizer_eval_fn,
            text_encoder_lr=text_encoder_lr,
            lr_descriptions=lr_descriptions,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
        )

    def _prepare_with_accelerator(
        self,
        args,
        accelerator,
        network,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        unet,
        text_encoders,
        text_encoder,
        vae,
        vae_dtype,
        weight_dtype,
        train_unet,
        train_text_encoder,
        cache_latents,
    ) -> AcceleratedBundle:
        """Cast model dtypes, run accelerator.prepare, flip train/eval, optional torch.compile."""
        unet_weight_dtype = te_weight_dtype = weight_dtype

        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            # None when the TE was never loaded (cache_text_encoder_outputs with
            # no sample prompts / val / TE-training -- qwen3_needed=False).
            if t_enc is None:
                continue
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp16/bf16
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)

        # accelerator preparation (no deepspeed)
        if train_unet:
            unet = self.prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            unet.to(
                accelerator.device,
                dtype=unet_weight_dtype,
            )
        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(
                    text_encoders,
                    self.get_text_encoders_train_flags(args, text_encoders),
                )
            ]
            if len(text_encoders) > 1:
                text_encoder = text_encoders
            else:
                text_encoder = text_encoders[0]
        # else: text_encoder is unchanged; device and dtype are already set above

        network, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
            accelerator.prepare(
                network, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
        )
        training_model = network

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for i, (t_enc, frag) in enumerate(
                zip(
                    text_encoders,
                    self.get_text_encoders_train_flags(args, text_encoders),
                )
            ):
                if t_enc is None:
                    continue
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            unet.eval()
            for t_enc in text_encoders:
                if t_enc is None:
                    continue
                t_enc.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        return AcceleratedBundle(
            network=network,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
            training_model=training_model,
            unet=unet,
            text_encoders=text_encoders,
            text_encoder=text_encoder,
            unet_weight_dtype=unet_weight_dtype,
        )

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        normalize_sample_args(args)
        verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        # Free-fit is the only resize mode and it requires compile_dynamic_seq: a
        # free-fit pool populates many distinct (W, H) within one tier's token
        # band, which would explode the static N-graph compile cascade. dynamic_seq
        # marks only the seq axis dynamic over the band → a single graph per tier.
        # Auto-enable it whenever compile is on (no-op if torch_compile is off).
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

        # --deterministic: close the one un-seedable noise source (flash-attn
        # backward atomic-add order) plus the standard torch determinism knobs,
        # so two runs of the identical command are bit-exact and paired A/B
        # endpoint deltas are pure treatment. Must precede CUDA/cublas init and
        # the first (possibly compiled) forward — the flash flag is read at
        # trace time. warn_only: unexpected nondeterministic ops log rather
        # than kill a run mid-flight. NB bespoke loops (turbo/spd/mod) do not
        # inherit this — mirror explicitly if a paired A/B needs it there.
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

        # Whether inductor will have CUDAGraphs active -- governs whether the
        # training loop needs to call torch.compiler.cudagraph_mark_step_begin()
        # each step (see the call site inside the accumulate block).
        self._cudagraph_mark_step = bool(
            getattr(args, "torch_compile", False)
            and getattr(args, "compile_inductor_mode", None)
            in ("reduce-overhead", "max-autotune")
        )

        # Build + install the strategy singletons (tokenize / latents-caching /
        # text-encoding). Must run before _prepare_dataset — dataset init reads
        # the tokenize + latents-caching strategies. The TE-OUTPUTS caching
        # strategy is installed separately below, after assert_extra_args has
        # had its chance to mutate cache_llm_adapter_outputs.
        strategies = strategy_anima.setup_training_strategies(args)
        tokenize_strategy = strategies.tokenize
        text_encoding_strategy = strategies.text_encoding
        tokenizers = [
            tokenize_strategy.qwen3_tokenizer
        ]  # will be removed after sample_image is refactored

        ds = self._prepare_dataset(args)
        train_dataset_group = ds.train_group
        val_dataset_group = ds.val_group
        current_epoch = ds.current_epoch
        current_step = ds.current_step
        collator = ds.collator
        use_user_config = ds.use_user_config
        use_dreambooth_method = ds.use_dreambooth_method

        # Derive the torch.compile token-family budget from the buckets the
        # selected (path_pattern-filtered) images actually populate — NOT from
        # args.target_res. The on-disk caches are the source of truth for which
        # tiers are present, so this can't drift from preprocess, and a filtered
        # run sizes the dynamo cache to only the families it really uses. Sample
        # prompt resolutions are folded in (when sampling is enabled) so sample
        # generation outside the training buckets compiles instead of crashing.
        self._compile_token_budget = self._derive_token_budget(
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

        self.assert_extra_args(
            args, train_dataset_group, val_dataset_group
        )  # may change some args

        # Install the text-encoder-outputs caching strategy now: after
        # assert_extra_args (which may flip cache_llm_adapter_outputs, read by
        # the strategy ctor) and before the model load, so the
        # cache-completeness probe below can use it to decide whether the
        # Qwen3 text encoder needs loading at all.
        strategy_anima.setup_text_encoder_outputs_caching_strategy(args)

        # Decide whether the heavy encoders are actually needed. When caching is
        # enabled the caches MUST already be complete on disk (run `make
        # preprocess` first) — train.py no longer encodes missing latents / TE
        # outputs on the fly. With complete caches and nothing else needing them
        # we skip loading the encoders entirely (saves the disk read, RAM, and
        # the GPU round-trip). `cache_latents = false` (e.g. IP-Adapter) is a
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

        # CMMD validation generates samples and decodes them through the VAE
        # (see library/training/validation.py). It reads cached TE outputs, so
        # it needs the VAE but not the text encoder.
        cmmd_validation = val_dataset_group is not None and getattr(
            args, "use_cmmd", True
        )
        # VAE: needed only to live-encode (caching off), to decode training
        # samples, or to decode CMMD validation samples. With caching on the
        # cache is guaranteed complete above, so no encode pass is required.
        vae_needed = (not cache_latents) or sampling_enabled or cmmd_validation

        # Qwen3 TE: needed only to live-encode (caching off), to encode sample
        # prompts, or when the text encoder itself is being trained.
        qwen3_needed = (
            (not args.cache_text_encoder_outputs)
            or bool(args.sample_prompts)
            or self.is_train_text_encoder(args)
        )

        # Prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precision dtype
        weight_dtype, save_dtype = prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # load target models: unet may be None for lazy loading
        model_version, text_encoder, vae, unet = self.load_target_model(
            args,
            weight_dtype,
            accelerator,
            load_qwen3=qwen3_needed,
            load_vae=vae_needed,
        )
        if vae_dtype is None:
            vae_dtype = vae.dtype if vae is not None else weight_dtype
            logger.info(
                f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false"
            )

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )

        # prepare dataset for latents caching if needed. When vae is None the
        # latents are already fully cached -- new_cache_latents still runs to
        # populate each ImageInfo.latents_npz path the dataloader reads, but
        # forms no encode batches so the (absent) VAE is never touched.
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

        # cache text encoder outputs if needed: Text Encoder is moved to cpu or
        # gpu (the encoding strategy was installed with the others up top).
        self.cache_text_encoder_outputs_if_needed(
            args,
            accelerator,
            text_encoders,
            train_dataset_group,
        )
        if val_dataset_group is not None:
            self.cache_text_encoder_outputs_if_needed(
                args,
                accelerator,
                text_encoders,
                val_dataset_group,
            )

        if unet is None:
            # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
            unet, text_encoders = self.load_unet_lazily(
                args, weight_dtype, accelerator, text_encoders
            )

        # Stage the T5("") sidecar once if caption dropout is on — dropped
        # rows then get the same crossattn embedding Anima feeds at
        # CFG-uncond inference instead of all-zeros (which is out-of-dist).
        if self._state.caption_dropout_enabled:
            from library.preprocess.uncond import ensure_uncond_crossattn

            self._state.uncond_crossattn_1 = ensure_uncond_crossattn(
                qwen3_path=args.qwen3,
                dit_path=args.pretrained_model_name_or_path,
                t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None),
                device=accelerator.device,
                dtype=weight_dtype,
                existing=self._state.uncond_crossattn_1,
            )

        net = self._create_and_apply_network(
            args, accelerator, vae, text_encoder, unet, text_encoders, weight_dtype
        )
        if net is None:
            return
        network = net.network
        net_kwargs = net.net_kwargs
        train_unet = net.train_unet
        train_text_encoder = net.train_text_encoder

        # Resolve and run on_network_built for each method adapter (EasyControl,
        # IP-Adapter, …). Each adapter validates its runtime contract and
        # logs/sets up auxiliary state before optimizer / accelerator wiring.
        self._adapters = resolve_adapters(args, network)
        if self._adapters:
            setup_ctx = SetupCtx(
                args=args,
                accelerator=accelerator,
                network=network,
                unet=unet,
                text_encoders=text_encoders,
                weight_dtype=weight_dtype,
            )
            for adapter in self._adapters:
                adapter.on_network_built(setup_ctx)

        opt = self._setup_optimizer_and_dataloader(
            args,
            accelerator,
            network,
            train_dataset_group,
            val_dataset_group,
            collator,
        )
        optimizer = opt.optimizer
        optimizer_name = opt.optimizer_name
        optimizer_args = opt.optimizer_args
        optimizer_train_fn = opt.optimizer_train_fn
        optimizer_eval_fn = opt.optimizer_eval_fn
        text_encoder_lr = opt.text_encoder_lr
        lr_descriptions = opt.lr_descriptions
        train_dataloader = opt.train_dataloader
        val_dataloader = opt.val_dataloader
        lr_scheduler = opt.lr_scheduler

        acc = self._prepare_with_accelerator(
            args,
            accelerator,
            network,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
            unet,
            text_encoders,
            text_encoder,
            vae,
            vae_dtype,
            weight_dtype,
            train_unet,
            train_text_encoder,
            cache_latents,
        )
        network = acc.network
        optimizer = acc.optimizer
        train_dataloader = acc.train_dataloader
        val_dataloader = acc.val_dataloader
        lr_scheduler = acc.lr_scheduler
        training_model = acc.training_model
        unet = acc.unet
        text_encoders = acc.text_encoders
        text_encoder = acc.text_encoder
        unet_weight_dtype = acc.unet_weight_dtype

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Structured progress sink (Phase 0): a JSONL event stream next to the
        # checkpoint that the GUI / daemon can tail instead of regex-parsing
        # tqdm. Main-process only; default on, gated by --progress_jsonl.
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
                # Mirror WARNING+ records into the stream so a reader debugging
                # the run gets them structured instead of buried in tqdm stdout.
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
            text_encoder_lr=text_encoder_lr,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            model_version=model_version,
            num_train_images=train_dataset_group.num_train_images,
            num_val_images=val_dataset_group.num_train_images
            if val_dataset_group is not None
            else 0,
            num_reg_images=train_dataset_group.num_reg_images,
            num_batches_per_epoch=len(train_dataloader),
            num_train_epochs=num_train_epochs,
        )
        self.update_metadata(metadata, args)  # architecture specific metadata
        add_dataset_metadata(
            metadata,
            train_dataset_group,
            args,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
            total_batch_size=total_batch_size,
        )
        add_model_hash_metadata(metadata, args)
        metadata, minimum_metadata = finalize_metadata(
            metadata, net_kwargs=net_kwargs if args.network_args else None
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
            get_sai_model_spec_fn=self.get_sai_model_spec,
            current_epoch=current_epoch,
            current_step=current_step,
            progress_sink=self.progress_sink,
        )
        saver.register_hooks(network)

        # auto-resume from the resumable checkpoint if one exists
        saver.auto_resume()

        # resume
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

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                if not args.resume:
                    logger.info(
                        "initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                    )
                logger.info(f"skipping {initial_step} steps")
                initial_step *= args.gradient_accumulation_steps

                epoch_to_start = initial_step // math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
            else:
                epoch_to_start = initial_step // math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
                initial_step = 0  # do not skip

        # Drop the train dataset-group local before loop entry — the
        # dataloader already holds the data it needs. Keep val_dataset_group
        # alive: CMMD validation enumerates its image_data to pair held-out
        # references with generated samples.
        del train_dataset_group

        loop_state = build_loop_state(
            self,
            args=args,
            accelerator=accelerator,
            saver=saver,
            network=network,
            unet=unet,
            text_encoder=text_encoder,
            text_encoders=text_encoders,
            vae=vae,
            tokenizers=tokenizers,
            training_model=training_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            val_dataset_group=val_dataset_group,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_descriptions=lr_descriptions,
            optimizer_train_fn=optimizer_train_fn,
            optimizer_eval_fn=optimizer_eval_fn,
            weight_dtype=weight_dtype,
            unet_weight_dtype=unet_weight_dtype,
            vae_dtype=vae_dtype,
            text_encoding_strategy=text_encoding_strategy,
            tokenize_strategy=tokenize_strategy,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            current_epoch=current_epoch,
            current_step=current_step,
            num_train_epochs=num_train_epochs,
            epoch_to_start=epoch_to_start,
            initial_step=initial_step,
            metadata=metadata,
        )

        # run_scope emits the matching run_end (ok / stopped / error) on exit;
        # run_start already fired when the sink was constructed above.
        with run_scope(
            self.progress_sink,
            final_step=lambda: loop_state.global_step,
            extra_fields=lambda: {
                **self._liveness.run_end_fields(),
                **(
                    {
                        "token_step_hist": {
                            str(k): v for k, v in sorted(self._token_step_hist.items())
                        }
                    }
                    if self._token_step_hist
                    else {}
                ),
            },
        ):
            run_training_loop(self, loop_state)

            accelerator.end_training()
            optimizer_eval_fn()

            # Catch-all sample decode for any latents not already decoded inline
            # (block-swapping runs defer the whole batch to here, and a latent
            # that failed an inline decode is left on disk for retry). The VAE
            # gets the full budget now that the loop has freed its activation /
            # block-swap memory; no-op when inline decode already drained them.
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

        # Remove the TensorBoard log dir for runs shorter than 2 steps — they
        # add noise to the runs list (e.g. aborted starts, dry-runs) and carry
        # no useful loss curves.
        if is_main_process and loop_state.global_step < 2:
            _cleanup_short_log_dir(args)

    # endregion


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


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    add_dataset_arguments(parser, True, True, True)
    add_training_arguments(parser, True)
    add_masked_loss_arguments(parser)
    add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    add_network_arguments(parser)
    add_validation_arguments(parser)
    add_train_misc_arguments(parser)
    return parser


from library.config import schema as _config_schema  # noqa: E402
from networks import all_network_kwargs as _all_network_kwargs  # noqa: E402


# Network-module-consumed flags (networks.lora_anima / networks.methods.*).
# These don't flow through argparse directly because `create_network` reads
# them from ``kwargs``. Sourced from the flat ``NETWORK_KWARGS`` allowlist in
# ``networks/__init__.py`` so adding a key there automatically registers it
# here.
NETWORK_KWARG_ALLOWLIST: tuple[str, ...] = _all_network_kwargs()

# Top-level training args that aren't network kwargs but still flow through
# ``net_kwargs`` because a network module reads them. Kept explicit -- any
# growth here should be reviewed, since the right answer is usually to
# expose the value as a proper argparse flag the network module reads
# directly rather than tunneling it through kwargs.
_EXTRA_FORWARDED_TOP_LEVEL_ARGS: tuple[str, ...] = (
    # Postfix contrastive resets its intra-step reference set on step
    # boundary, so it needs the grad-accum window.
    "gradient_accumulation_steps",
)


def resolve_network_kwargs(args) -> dict[str, str]:
    """The single intake for network kwargs, merging both config paths.

    A network kwarg can arrive as ``--network_args k=v`` (CLI / method TOML
    ``network_args`` list) or as an allowlisted top-level config key landing
    on ``args``; CLI ``--network_args`` win on overlap. Consumers outside the
    network factory (e.g. the REPA dataset-sidecar enable in
    ``assert_extra_args``) must see the same merged view the factory gets, so
    the result is cached on ``args._network_kwargs`` — read a kwarg from here
    rather than scanning ``args.network_args`` with a ``getattr(args, …)``
    fallback. All values are strings, as ``create_network(**kwargs)`` expects.
    """
    cached = getattr(args, "_network_kwargs", None)
    if cached is not None:
        return cached

    net_kwargs: dict[str, str] = {}
    for net_arg in getattr(args, "network_args", None) or []:
        key, value = net_arg.split("=", 1)
        net_kwargs[key] = value

    # Forward known network-arg keys from top-level config (TOML). Source of
    # truth: `networks.all_network_kwargs()` (the flat `NETWORK_KWARGS`
    # allowlist), plus a small tail of top-level training args the network
    # modules still want to read (e.g. postfix contrastive's step-boundary
    # window).
    for key in NETWORK_KWARG_ALLOWLIST + _EXTRA_FORWARDED_TOP_LEVEL_ARGS:
        if (
            key not in net_kwargs
            and hasattr(args, key)
            and getattr(args, key) is not None
        ):
            net_kwargs[key] = str(getattr(args, key))

    args._network_kwargs = net_kwargs
    return net_kwargs


def build_network_extras() -> dict[str, _config_schema.ConfigKey]:
    return {
        k: _config_schema.ConfigKey(name=k, type="str", source="network_module")
        for k in NETWORK_KWARG_ALLOWLIST
    }


def _install_crash_reporter(argv: list[str]) -> None:
    """Record a fatal startup/training exception into ``--progress_jsonl``.

    The daemon launches us windowless under ``pythonw.exe``; that interpreter
    drops the child's stdout/stderr (only the ``accelerate launch`` *parent*'s
    output reaches ``stdout.log``), so an uncaught traceback here is lost and the
    daemon falls back to a generic "process exited (code=1)" with nothing
    actionable. ``progress.jsonl`` is written by path, not via the dead std
    streams, so it survives — and it's what the daemon already reads to diagnose
    a job (``manager._finalize_from_exit`` → ``run_end.error``).

    ``run_scope`` already emits ``run_end(error=…)`` for failures inside the
    training loop, but only *after* ``ProgressSink.run_start`` has fired — late
    in ``train()``. Errors before that (latent/TE cache incomplete, config or
    dataset build, model load) escape it entirely. This excepthook is the
    catch-all: it appends a ``run_end`` error event for any uncaught exception,
    wherever it's raised, so the GUI's finish banner shows the real cause.
    """
    path = None
    for i, tok in enumerate(argv):
        if tok == "--progress_jsonl" and i + 1 < len(argv):
            path = argv[i + 1]
        elif tok.startswith("--progress_jsonl="):
            path = tok.split("=", 1)[1]
    if not path or path.strip().lower() in ("", "none", "off"):
        return

    import json as _json

    prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        # KeyboardInterrupt is a clean stop, handled by run_scope/the daemon's
        # stop_requested path — don't mislabel it an error.
        if not issubclass(exc_type, KeyboardInterrupt):
            try:
                # Dedupe: run_scope may already have written the terminal event
                # for an in-loop failure; don't append a second one.
                already_ended = False
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                last = line
                    try:
                        already_ended = _json.loads(last).get("ev") == "run_end"
                    except (NameError, ValueError):
                        already_ended = False
                if not already_ended:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(
                            _json.dumps(
                                {
                                    "ev": "run_end",
                                    "status": "error",
                                    "final_step": -1,
                                    "error": f"{exc_type.__name__}: {exc}",
                                }
                            )
                            + "\n"
                        )
            except Exception:  # noqa: BLE001 — reporting must never mask the crash
                pass
        prev_hook(exc_type, exc, tb)

    sys.excepthook = _hook


if __name__ == "__main__":
    _install_crash_reporter(sys.argv)
    parser = setup_parser()
    _config_schema.populate_schema(parser, extras=build_network_extras())

    args = parser.parse_args()
    verify_command_line_training_args(args)
    args = read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    artist = getattr(args, "artist_filter", None)
    if artist:
        _datasets_base.set_artist_filter(artist)
        slug = artist.lstrip("@")
        args.output_dir = "output/ckpt-artist"
        args.output_name = f"{args.output_name}_{slug}"
        logger.info(
            f"artist_filter active: '{artist}' → output_dir={args.output_dir}, "
            f"output_name={args.output_name}"
        )

    trainer = AnimaTrainer()
    trainer.train(args)
