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

# Windows: suppress per-kernel ptxas.exe/cl.exe console flashes. Must run
# before any subprocess.Popen call (before torch import on Windows).
from library.runtime.proc import install_no_window_default

install_no_window_default()

# Must land before torch initializes the CUDA caching allocator: free-fit
# varies seq_len per step and fragments the pool without expandable segments
# (issue #58). Opt out: ANIMA_EXPANDABLE_SEGMENTS=0.
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
        # Per-method extensions (EasyControl, …). Resolved from args+network
        # in train() right after _create_and_apply_network.
        self._adapters: list[MethodAdapter] = []
        # Feature-specific per-run state — see ``RuntimeState``.
        self._state = RuntimeState()
        # Counts aux consumption per skip-if-missing loss; the loop audits it
        # and flags configured-but-dead features with `LIVENESS:`.
        self._liveness = LivenessLedger()
        # Realized patch-token histogram over train steps (post σ-demote
        # swap) — per-arm FLOPs accounting; merged into run_end.
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
            # Adapter-output caching writes into the TE cache; with text
            # caching off there's nothing to write into, so it's a harmless
            # no-op — auto-disable instead of crashing (easy to hit from the
            # GUI, where these are independent toggles).
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
        # {stem}_byg.safetensors into the batch.
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
            # re-buckets each member, shrinking its length; refresh the
            # ConcatDataset cumulative_sizes or global indices overflow.
            train_dataset_group.refresh_concat_state()
            if val_dataset_group is not None:
                for dataset in val_dataset_group.datasets:
                    dataset.byg_text_dir = byg_text_dir
                    dataset.restrict_to_byg_tuples()
                val_dataset_group.refresh_concat_state()

        # REPA v2: load cached PE-Spatial patch tokens into batches when
        # use_repa is set (rides the resolved network kwargs).
        net_kwargs = resolve_network_kwargs(args)
        if net_kwargs.get("use_repa", "").lower() in ("true", "1", "yes"):
            repa_encoder = net_kwargs.get("repa_encoder") or "pe_spatial"
            for dataset in train_dataset_group.datasets:
                dataset.load_repa_pe = True
                dataset.repa_pe_encoder = repa_encoder
            # A missing PE cache makes the REPA alignment term a silent no-op
            # (the loss skips any batch without repa_pe_features), so probe
            # coverage now: fail fast on a fully-absent cache, warn on partial.
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

        # Soft-tokens contrastive negatives (configs/methods/soft_tokens.toml).
        # Preview the resolved kwargs to decide whether the dataset should
        # surface cached negative text embeddings. Off unless contrastive_weight > 0.
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

        # Skipped when every text-encoder output is already cached and no
        # live encoding (sampling / TE training) needs it.
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

        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(
        self, args, weight_dtype, accelerator, text_encoders
    ) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        from library.runtime.backend import (
            needs_rocm_attention_fallback,
            resolve_attention_mode,
        )

        requested_attn_mode = args.attn_mode
        attn_mode = resolve_attention_mode(requested_attn_mode, torch)
        if needs_rocm_attention_fallback(requested_attn_mode, torch):
            logger.warning(
                "ROCm detected: using PyTorch SDPA instead of CUDA Flash Attention"
            )

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
            # Sample prompts tokenized through the pack need its rows behind
            # the ext ids (same table the TE caches were built with).
            vocab_pack=getattr(args, "vocab_pack", None),
        )

        # Mod-aware training: install the distilled pooled_text_proj so an
        # adaln LoRA trains against the operating point mod-guidance perturbs
        # at inference. Frozen implicitly (excluded from the LoRA factory).
        if getattr(args, "pooled_text_proj", None):
            anima_utils.load_pooled_text_proj(model, args.pooled_text_proj, "cpu")
            model.pooled_text_proj.to(device=loading_device, dtype=loading_dtype)

        # NOTE: torch.compile (compile_blocks) is intentionally NOT done here —
        # it must run AFTER the adapter's apply_to monkey-patches the targeted
        # Linears, or dynamo traces the un-adapted forward. Done in
        # _create_and_apply_network instead (after apply_to + load_weights +
        # grad-ckpt) — see library/runtime/harness.py for the ordering.

        # So dit.enable_gradient_checkpointing() can override to use unsloth.
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
        # forward (works since base weights are frozen and LoRA-family
        # adapters are additive) — see get_noise_pred_and_target for the
        # bypass. Saves ~5 GB VRAM vs a second DiT copy.
        if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
            logger.info(
                f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
                f"using trainable DiT with multiplier=0 as the control variate"
            )

        # Online memorization Δ-gap tracker (same set_multiplier(0) trick as
        # VR). The measurement is a second per-step DiT forward, unaudited
        # against the block-swap offloader — raise, not warn.
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

    # Strategy construction lives in library/anima/strategy.py
    # (setup_training_strategies / setup_text_encoder_outputs_caching_strategy).

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

    # Per-step forward phases: ``get_noise_pred_and_target`` is a flat
    # sequence of named phases; conditional logic lives INSIDE each phase,
    # never as lexical nesting around it, so "always per step" is
    # structurally evident at the call site.

    def _step_ctx(self, ctx: TrainCtx) -> StepCtx:
        return StepCtx(
            args=ctx.args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )

    def _prime_adapters(self, ctx: TrainCtx, batch, latents, *, is_train) -> None:
        """ALWAYS per step. Method-adapter pre-forward priming (e.g. EasyControl
        runs the cond pre-pass and primes per-block (K_c, V_c)) — consumed by
        the patched cross-attn/self-attn closures during attention."""
        if not self._adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self._adapters:
            adapter.prime_for_forward(step_ctx, batch, latents, is_train=is_train)

    def _paired_step_generators(self, args, device, is_train):
        """Common-random-numbers mode (``--paired_step_rng``): per-train-step
        ``(g_sigma, g_noise)`` generators seeded from (seed, step counter),
        decoupled from the global torch stream, so arms sharing a seed see
        identical σ/noise on same-shape steps and checkpoint deltas isolate
        the intervention. ``(None, None)`` when off."""
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

    # The probe's operating point, and the value `--sigma_lowres` implies.
    YARNSIG_OPERATING_POINT = "1,4,0.35,2"
    _YARNSIG_OFF = frozenset({"", "0", "off", "no", "none", "false"})

    @staticmethod
    def _yarnsig_params(args):
        """Parsed ``--sigma_lowres_yarnsig`` as ``(alpha, beta, center, gamma)``,
        or None when off. Validates once; cached on the args namespace.
        **Default-on with ``--sigma_lowres``** (docs/optimizations/sigma_lowres.md):
        an unset flag resolves to the operating point, not off. Opt out with
        ``--sigma_lowres_yarnsig off``. Written back onto ``args`` so the
        snapshot records what actually ran.
        """
        raw = getattr(args, "sigma_lowres_yarnsig", None)
        if raw is None and getattr(args, "sigma_lowres", False):
            raw = AnimaTrainer.YARNSIG_OPERATING_POINT
            args.sigma_lowres_yarnsig = raw
        if raw is None or str(raw).strip().lower() in AnimaTrainer._YARNSIG_OFF:
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

    @staticmethod
    def _sigma_rule_suffix(rule):
        if rule not in (1, 2):
            raise ValueError(f"sigma_lowres rule must be 1 or 2, got {rule!r}")
        return "" if rule == 1 else "2"

    @staticmethod
    def _sigma_rule_cfg(args, rule):
        """``(threshold, threshold_max, span_spec)`` for one rule.

        Written as LITERAL ``getattr(args, "…")`` calls, one branch per rule,
        not a computed ``f"sigma_lowres_threshold{sfx}"`` — the H2 drift guard
        (``tests/config_closure.py``) AST-scans for literal names only, so a
        computed name would be invisible to it.
        """
        if rule == 1:
            return (
                getattr(args, "sigma_lowres_threshold", 0.5),
                getattr(args, "sigma_lowres_threshold_max", None),
                getattr(args, "sigma_lowres_span", None),
            )
        if rule == 2:
            return (
                getattr(args, "sigma_lowres_threshold2", None),
                getattr(args, "sigma_lowres_threshold2_max", None),
                getattr(args, "sigma_lowres_span2", None),
            )
        raise ValueError(f"sigma_lowres rule must be 1 or 2, got {rule!r}")

    @staticmethod
    def _sigma_route2(args):
        """Parsed ``--sigma_lowres_route2`` as ``(native, demote)``, or None
        when the secondary rule is off. Same shape/validation as
        ``_sigma_route`` (which always has its 1024:896 default)."""
        raw = getattr(args, "sigma_lowres_route2", None)
        if not raw:
            return None
        try:
            native, demote = (int(x) for x in str(raw).split(":"))
        except ValueError:
            raise ValueError(
                f"--sigma_lowres_route2 expects NATIVE:DEMOTE (e.g. 1024:768), "
                f"got {raw!r}"
            ) from None
        if not (native > demote > 0):
            raise ValueError(
                f"--sigma_lowres_route2 needs NATIVE > DEMOTE > 0, got {raw!r}"
            )
        return native, demote

    @staticmethod
    def _sigma_span_params(args, rule=1):
        """Parsed ``--sigma_lowres_span[2]`` as ``(mode, frac)``, or None when
        off. ``mode`` ∈ early|late|spread, 0 < frac < 1 (default 0.5).
        Validates once; cached on the args namespace per rule."""
        sfx = AnimaTrainer._sigma_rule_suffix(rule)
        raw = AnimaTrainer._sigma_rule_cfg(args, rule)[2]
        if not raw:
            return None
        cache_attr = f"_sigma_span{sfx}_parsed"
        cached = getattr(args, cache_attr, None)
        if cached is not None:
            return cached
        parts = str(raw).split(":")
        mode, frac = parts[0], 0.5
        ok = mode in ("early", "late", "spread") and len(parts) <= 2
        if ok and len(parts) == 2:
            try:
                frac = float(parts[1])
            except ValueError:
                ok = False
        if not ok or not (0.0 < frac < 1.0):
            raise ValueError(
                f"--sigma_lowres_span{sfx} expects early|late|spread[:FRAC] "
                f"with 0<FRAC<1, got {raw!r}"
            )
        setattr(args, cache_attr, (mode, frac))
        return getattr(args, cache_attr)

    @staticmethod
    def _sigma_gate_allows(args, sigmas_flat, rule=1):
        """σ gate: every σ in the batch above --sigma_lowres_threshold[2]
        and, when --sigma_lowres_threshold[2]_max is set, below it too — the
        measured demote-safe region is a per-route σ WINDOW, not a
        half-line."""
        threshold, threshold_max, _ = AnimaTrainer._sigma_rule_cfg(args, rule)
        # Rule 2's bounds have no defaults — _validate_sigma_rules() refuses a
        # route2 without both, so an unset bound here can only be rule 1's.
        threshold = 0.5 if threshold is None else float(threshold)
        if not bool((sigmas_flat > threshold).all()):
            return False
        if threshold_max is not None:
            return bool((sigmas_flat < float(threshold_max)).all())
        return True

    @staticmethod
    def _sigma_span_allows(args, step_idx, rule=1):
        """Step-span gate: may step ``step_idx`` (1-based train-forward index)
        demote? True when no span is set.

        early/late split total train forwards (max_train_steps × grad-accum)
        at the FRAC point; spread is a per-step coin seeded from
        (--seed, step_idx) alone — deterministic and touches neither the
        global nor the paired RNG streams.
        """
        span = AnimaTrainer._sigma_span_params(args, rule)
        if span is None:
            return True
        mode, frac = span
        total = int(getattr(args, "max_train_steps", 0) or 0) * int(
            getattr(args, "gradient_accumulation_steps", 1) or 1
        )
        if total <= 0:
            raise ValueError("--sigma_lowres_span needs a finalized max_train_steps")
        if mode == "early":
            return step_idx <= round(frac * total)
        if mode == "late":
            # Boundary at (1-frac)·T, same rounding as early's — so
            # early:f and late:1-f partition [1, T] exactly for any T.
            return step_idx > round((1.0 - frac) * total)
        seed = int(getattr(args, "seed", 0) or 0)
        return random.Random(seed * 1_000_003 + step_idx).random() < frac

    @staticmethod
    def _validate_sigma_rules(args):
        """Validate the whole sigma_lowres surface UP FRONT, at setup — not on
        the first train forward (after model load/adapter attach/caching).
        Rejects ``--sigma_lowres_route2`` without both σ bounds: rule 2 has
        PRIORITY, so an unbounded gate would shadow the primary rule across
        the whole σ>0.5 half-line. Also validates span specs up front.
        """
        route = AnimaTrainer._sigma_route(args)
        route2 = AnimaTrainer._sigma_route2(args)
        if route2 is not None:
            lo, hi, _ = AnimaTrainer._sigma_rule_cfg(args, 2)
            missing = [
                flag
                for flag, val in (
                    ("--sigma_lowres_threshold2", lo),
                    ("--sigma_lowres_threshold2_max", hi),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"--sigma_lowres_route2 {route2[0]}:{route2[1]} needs an "
                    f"explicit σ WINDOW — missing {', '.join(missing)}. Rule 2 "
                    "has priority over rule 1, so an unbounded gate demotes on "
                    "the deep route everywhere it fires (shadowing the primary "
                    "rule and silencing yarnsig, which is primary-only) — "
                    "including σ below the route's certified window, where the "
                    "measured demote gap is large. The shipped 1024:768 window "
                    "is 0.65 < σ < 0.95."
                )
            if not (0.0 <= float(lo) < float(hi) <= 1.0):
                raise ValueError(
                    f"--sigma_lowres_threshold2/_max must satisfy "
                    f"0 <= lo < hi <= 1, got {lo} / {hi}"
                )
        hi1 = AnimaTrainer._sigma_rule_cfg(args, 1)[1]
        if hi1 is not None and not (
            0.0 <= float(getattr(args, "sigma_lowres_threshold", 0.5)) < float(hi1)
        ):
            raise ValueError(
                f"--sigma_lowres_threshold_max ({hi1}) must exceed "
                f"--sigma_lowres_threshold "
                f"({getattr(args, 'sigma_lowres_threshold', 0.5)})"
            )
        # Parse both spans and yarnsig here so a typo raises at setup.
        AnimaTrainer._sigma_span_params(args, 1)
        AnimaTrainer._sigma_span_params(args, 2)
        AnimaTrainer._yarnsig_params(args)
        return route, route2

    @staticmethod
    def _sigma_demote_choice(args, sigmas_flat, step_idx):
        """Which demote rule fires this step: 2, 1, or None.

        The secondary rule (route2 + its own gate/span) takes precedence —
        stacked-router semantics: demote DEEPER (e.g. 768) when σ is inside
        the deep route's measured window, else the primary rule (e.g. σ>0.5 →
        896), else native.
        """
        if AnimaTrainer._sigma_route2(args) is not None:
            if AnimaTrainer._sigma_gate_allows(
                args, sigmas_flat, rule=2
            ) and AnimaTrainer._sigma_span_allows(args, step_idx, rule=2):
                return 2
        if AnimaTrainer._sigma_gate_allows(args, sigmas_flat) and (
            AnimaTrainer._sigma_span_allows(args, step_idx)
        ):
            return 1
        return None

    def _sigma_rule_dead_warn(self, args, rule):
        """Warn once per rule when its gate passed but the sibling latent for
        that route is absent from the batch — the rule is configured but dead
        (would otherwise silently fall through and train as a plain
        single-route arm).
        """
        warned = getattr(self, "_sigma_rule_dead_warned", None)
        if warned is None:
            warned = self._sigma_rule_dead_warned = set()
        if rule in warned:
            return
        warned.add(rule)
        route = (
            self._sigma_route2(args) if rule == 2 else self._sigma_route(args)
        ) or ("?", "?")
        logger.warning(
            "sigma_lowres: rule %d (%s:%s) passed its σ gate but the batch "
            "carries no sibling latent for that route — those steps fall "
            "through to %s. Emit it with `make preprocess-demote "
            'ARGS="--sigma_demote %s:%s"` (or list both routes in '
            "configs/preprocess.toml's sigma_demote). Warned once per rule.",
            rule,
            route[0],
            route[1],
            "rule 1 / native" if rule == 2 else "native",
            route[0],
            route[1],
        )

    def _maybe_sigma_demote(
        self, ctx: TrainCtx, batch, latents, is_train, generator=None
    ):
        """sigma_lowres Phase 1b (σ > threshold → demote-tier latent).

        Returns ``(latents, sigmas_flat)``: possibly-swapped latents plus the
        pre-drawn flat σ (None → sampler draws internally). Active only when
        --sigma_lowres is on, the batch carries ``demoted_latents``, and this
        is a train step; ``--sigma_lowres_span`` further restricts by training
        progress. Side effect: sets ``self._yarnsig_step`` (consumed/cleared
        by the primary forward).
        """
        args = ctx.args
        self._yarnsig_step = None
        if not is_train or not getattr(args, "sigma_lowres", False):
            return latents, None
        # Train-forward index for the step-span gate. Pre-seeded with the
        # resume offset — else a resumed run would measure the span boundary
        # from the resume point, not absolute T.
        step_idx = getattr(self, "_sigma_span_step", 0) + 1
        self._sigma_span_step = step_idx
        demoted = batch.get("demoted_latents")
        demoted2 = batch.get("demoted_latents2")
        if demoted is None and demoted2 is None:
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
        total = getattr(self, "_sigma_lowres_seen", 0) + 1
        self._sigma_lowres_seen = total
        # Priority router: rule 2 (deep route, own gate/span) > rule 1 >
        # native. A rule whose sibling latent is missing falls through to
        # the next (partial-emit degrade, mirroring the single-rule path).
        choice = self._sigma_demote_choice(args, sigmas_flat, step_idx)
        if choice == 2 and demoted2 is None:
            # Un-emitted deep route: falls through to rule 1 and looks
            # healthy, so warn explicitly.
            self._sigma_rule_dead_warn(args, 2)
            choice = (
                1
                if self._sigma_gate_allows(args, sigmas_flat)
                and self._sigma_span_allows(args, step_idx)
                else None
            )
        if choice == 1 and demoted is None:
            self._sigma_rule_dead_warn(args, 1)
            choice = None
        if choice is not None:
            native_hw = tuple(latents.shape[-2:])
            swapped = demoted2 if choice == 2 else demoted
            latents = swapped.to(device=latents.device, dtype=latents.dtype)
            # yarnsig belongs to the PRIMARY rule only — the deep route's
            # window was measured on plain demotion.
            yarn = self._yarnsig_params(args) if choice == 1 else None
            if yarn is not None:
                alpha, beta, center, gamma = yarn
                # Patch-grid units: demoted patch i sits at
                # i · (native_patches / demoted_patches). μ from the
                # batch-min σ (the sample nearest the gate).
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
        sampler registry and run per-step network router conditioning
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

        # ChimeraHydra global content router: fire ONCE per step on the pooled
        # crossattn_emb. apply_router_conditioning ran before text conds were
        # materialized, so this lives outside that helper. No-op otherwise.
        if (
            getattr(network, "use_content_router", False)
            and tc.crossattn_emb is not None
            and hasattr(network, "set_content")
        ):
            network.set_content(tc.crossattn_emb)

        # Network-level GlobalRouter routed on pooled text. Same timing
        # rationale as the content router above. No-op otherwise.
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
        """ALWAYS per step. Single, branch-free forward call site: both
        text-conditioning modes normalize to ONE uniform
        ``ForwardConditioning`` bundle first, in ``build_forward_conditioning``,
        not as control flow here. Must run inside the primary forward's
        autocast/grad scope (the postfix splice runs learned modules), hence
        not in ``_prepare_conditioning``. Returns ``(model_pred, cond)``.
        """
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
        phase, including the cached-text requirement
        (``cond.crossattn_emb is not None``). Must run inside the primary
        forward's autocast/grad scope (extra ``anima(...)`` calls)."""
        args = ctx.args

        # Functional MSE loss against a sampled stochastic inversion run. The
        # captures dict is populated by forward hooks on cross_attn.output_proj
        # at ``self._func_blocks``.
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

        # Online memorization Δ-gap (mem_reweight.py). Producer half: causal
        # per-item weights plus, on measurement steps, the base forward's
        # per-sample log-MSE. Consumer half (EMA + loss_weights multiply)
        # lives in ``_process_batch_inner``.
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
                # (not gated on the train draw's σ); the consumer only folds
                # the Δ into the EMA.
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
        extra forwards (soft-tokens, REPA, …). Must dispatch on BOTH paths —
        gating it inside the cached-crossattn branch only would silently skip
        every adapter's aux loss on the in-model text path (crossattn_emb=None
        — EasyControl's default)."""
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

        # sigma_lowres: σ-first draw + latent swap. When EVERY sample's σ
        # clears the gate, the whole step runs on the demoted grid. The σ
        # marginal is untouched (drawn merely before the noise), so the
        # all-samples rule is exact at train_batch_size=1 and conservative above.
        latents, sigmas_flat = self._maybe_sigma_demote(
            ctx, batch, latents, is_train, generator=g_sigma
        )
        if is_train and getattr(ctx.args, "sigma_lowres", False):
            # Realized patch-token histogram: counts examples at the grid the
            # step ACTUALLY ran on (post demote swap), for per-arm FLOPs
            # accounting. Emitted with the run_end progress event.
            tok = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)
            self._token_step_hist[tok] = self._token_step_hist.get(tok, 0) + int(
                latents.shape[0]
            )
        if sigmas_flat is None and g_sigma is not None:
            # Paired mode on a batch the demote path didn't draw for: still
            # take σ from the paired stream so every arm shares the sequence.
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

        # Cached text-encoder outputs arrive as [..., caption_dropout_rates].
        # Split the rates off; get_noise_pred_and_target applies the dropout
        # in-place after the H2D transfer (doing it here on CPU would clone
        # prompt_embeds/crossattn_emb before the copy, blocking the main thread).
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
        """Process a batch for the network."""
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

        # Online memorization Δ-gap, consumer half (producer: _attach_aux_losses).
        # Fold this step's paired Δ into the per-item EMA (measurement steps
        # only), then apply the CAUSAL weights (from pre-step state) onto
        # per-sample loss_weights.
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
        # handlers read network._<name>_weight; the trainer stamps it here
        # since functional's weight is a top-level training arg.
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
                    # Fires twice per step (main forward + inversion forward);
                    # main runs first, so this snapshots before the 2nd overwrites.
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
        # A LoRA trained through a CJK vocab pack is coupled to that pack's
        # rows + routing (ids ≥ 32128 in every cached caption). Stamp the
        # digest so a mismatch at load time is detectable, never silent.
        from library.anima.vocab_pack import load_vocab_pack

        pack = load_vocab_pack(getattr(args, "vocab_pack", None))
        if pack is not None:
            metadata.update(pack.checkpoint_metadata())

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

        # With caching on, the on-disk cache is guaranteed complete (asserted
        # in train()), so no encoding is needed here — run the pass with no
        # model purely to populate ImageInfo.text_encoder_outputs_npz.
        dataset.new_cache_text_encoder_outputs([None], accelerator)

        # In memory only to encode sample prompts; None when no sample
        # prompts are configured.
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
        from library.datasets.subsets import resolve_configured_mask_dir

        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        # `mask_dir` rides the config chain (configs/preprocess.toml → preset →
        # method → --mask_dir) and reaches every subset that doesn't name one
        # of its own via the BlueprintGenerator's argparse fallback. Gate it on
        # the directory existing first — the config value names where `make
        # mask` *would* write, and a maskless checkout must keep falling back
        # to the legacy auto-resolution instead of flipping alpha_mask on.
        args.mask_dir = resolve_configured_mask_dir(getattr(args, "mask_dir", None))

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

            # Global --sample_ratio override (`[half]` preset / GUI data-scope
            # field). 1.0 is inert so any per-subset sample_ratio in a custom
            # blueprint stays authoritative.
            sample_ratio = getattr(args, "sample_ratio", None)
            if sample_ratio is not None and sample_ratio != 1.0:
                for ds in user_config.get("datasets", []):
                    for sub in ds.get("subsets", []):
                        sub["sample_ratio"] = sample_ratio
                logger.info(f"Applied --sample_ratio={sample_ratio} to all subsets")

            # --artists_shard k_N: restrict training to one round-robin shard of
            # the artist subdirs, expanded into each subset's path_pattern
            # before the blueprint is built (so downstream path_pattern-
            # filtered logic sees the shard).
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
                    # Free-fit: the bucket set is the union of on-disk resized
                    # sizes, so every cached latent exact-matches its own
                    # (W, H). target_res is preprocess-only and inert here.
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

        # sigma_lowres: activate the σ-demote sidecar on TRAIN datasets only
        # — validation stays native so val loss is comparable across arms.
        if getattr(args, "sigma_lowres", False):
            route, route2 = self._validate_sigma_rules(args)

            enabled_on = 0
            for ds in getattr(train_dataset_group, "datasets", []):
                if hasattr(ds, "enable_sigma_demote"):
                    ds.enable_sigma_demote(*route)
                    if route2 is not None:
                        ds.enable_sigma_demote2(*route2)
                    enabled_on += 1
            logger.info(
                "sigma_lowres ENABLED: %d→%d demote at σ > %s on %d train "
                "dataset(s); validation stays native.",
                route[0],
                route[1],
                getattr(args, "sigma_lowres_threshold", 0.5),
                enabled_on,
            )
            if route2 is not None:
                logger.info(
                    "sigma_lowres rule2 ENABLED (priority): %d→%d demote at "
                    "%s < σ < %s%s.",
                    route2[0],
                    route2[1],
                    *self._sigma_rule_cfg(args, 2)[:2],
                    (
                        f", span {self._sigma_rule_cfg(args, 2)[2]}"
                        if self._sigma_rule_cfg(args, 2)[2]
                        else ""
                    ),
                )
            yarn = self._yarnsig_params(args)
            if yarn is not None:
                logger.info(
                    "sigma_lowres yarnsig ENABLED: banded rope on demoted "
                    "steps, α,β=%s,%s, μ=sigmoid(%s·[logit(σ)−logit(%s)]).",
                    *(yarn[0], yarn[1], yarn[3], yarn[2]),
                )
        elif self._yarnsig_params(args) is not None:
            # `off` (and friends) parse to None, so opting out in a config that
            # never enables sigma_lowres stays inert instead of hard-failing.
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
        """(n_token_families, seq_range, seq_bands) from the buckets the
        datasets populate, unioned with sample-prompt token counts
        (``_sample_prompt_token_counts``). Sizes ``compile_blocks``' dynamo
        cache to exactly the tiers on disk, independent of ``args.target_res``.
        ``seq_bands`` is the per-tier clustering of the same count set
        (``cluster_token_bands``), consumed only under ``--compile_seq_bands``.
        ``(None, None, None)`` when no bucketed resos are available.
        """
        from library.datasets.buckets import (
            cluster_token_bands,
            token_counts_for_resos,
        )

        resos: set = set()
        for group in (train_group, val_group):
            if group is None:
                continue
            for dataset in getattr(group, "datasets", []):
                bm = getattr(dataset, "bucket_manager", None)
                if bm is not None:
                    resos.update(bm.resos)
        if not resos:
            return None, None, None
        counts = token_counts_for_resos(resos) | self._sample_prompt_token_counts(args)
        # sigma_lowres: demoted forwards run at the demote tier's token
        # counts, which must sit inside the compiled dynamic-seq range.
        if getattr(args, "sigma_lowres", False):
            from library.datasets.buckets import demoted_token_counts

            counts |= demoted_token_counts(resos, *self._sigma_route(args))
            route2 = self._sigma_route2(args)
            if route2 is not None:
                counts |= demoted_token_counts(resos, *route2)
        return len(counts), (min(counts), max(counts)), cluster_token_bands(counts)

    def _sample_prompt_token_counts(self, args) -> set:
        """Token counts the sample prompts will request; empty when sampling
        is off. Sample generation runs through the same compiled blocks as
        training, so an out-of-bucket resolution would crash mid-training
        (ConstraintViolationError) unless folded into the compile budget up
        front. Resolutions added to the file mid-run are NOT covered — skipped
        with a warning at sample time instead.
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

        # Copied so the dropout default below stays a factory-call detail,
        # not part of the cached ``args._network_kwargs`` view other
        # consumers read.
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
        # via block pre-hooks — unaudited against the block-swap offloader,
        # which desyncs on unexpected per-step forward patterns.
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
        # after apply_to + load_weights so dynamo traces the adapter's
        # monkey-patched forwards, not the bare DiT (see harness.py).
        if args.torch_compile:
            from library.runtime.harness import compile_blocks_for_training

            # Token-family budget from buckets the dataset actually populated
            # (_derive_token_budget) — not args.target_res (preprocess-only).
            n_token_families, seq_range, seq_bands = getattr(
                self, "_compile_token_budget", (None, None, None)
            )
            if not getattr(args, "compile_seq_bands", False):
                seq_bands = None
            # Register tokens grow the seq by a constant K, so widen the
            # dynamic-seq bound's MAX by K or the compiled block's bound is
            # violated (the min/family-count stay: mid-stack insertion still
            # runs blocks before the insert at the bare seq). Per-band mode
            # widens each band's hi; widen_bands raises if K would make
            # adjacent bands touch (silent merging would un-tighten them).
            extra_seq = int(getattr(network, "extra_seq_tokens", 0) or 0)
            if extra_seq and seq_range is not None:
                seq_range = (seq_range[0], seq_range[1] + extra_seq)
                if seq_bands:
                    from library.datasets.buckets import widen_bands

                    seq_bands = widen_bands(seq_bands, extra_seq)
            compile_blocks_for_training(
                unet,
                network,
                backend=args.dynamo_backend,
                mode=getattr(args, "compile_inductor_mode", None),
                n_token_families=n_token_families,
                seq_range=seq_range,
                seq_bands=seq_bands,
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
        from library.runtime.backend import warn_if_cuda_unavailable

        warn_if_cuda_unavailable(torch)
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        normalize_sample_args(args)
        verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        # Free-fit requires compile_dynamic_seq: a free-fit pool populates many
        # distinct (W, H) within one tier's token band, which would explode
        # the static N-graph compile cascade. Auto-enable whenever compile is
        # on (no-op if torch_compile is off).
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

        # Whether inductor will have CUDAGraphs active -- governs whether the
        # loop needs to call torch.compiler.cudagraph_mark_step_begin() each step.
        self._cudagraph_mark_step = bool(
            getattr(args, "torch_compile", False)
            and getattr(args, "compile_inductor_mode", None)
            in ("reduce-overhead", "max-autotune")
        )

        # Build + install the strategy singletons. Must run before
        # _prepare_dataset — dataset init reads them. TE-OUTPUTS caching
        # strategy is installed separately below, after assert_extra_args.
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

        # Derive the compile token-family budget from buckets the selected
        # images actually populate — NOT args.target_res. Sample prompt
        # resolutions are folded in so out-of-bucket generation compiles.
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

        # Install the caching strategy now: after assert_extra_args (which may
        # flip cache_llm_adapter_outputs) and before the model load, so the
        # cache-completeness probe below can decide if Qwen3 needs loading.
        strategy_anima.setup_text_encoder_outputs_caching_strategy(args)

        # When caching is enabled the caches MUST already be complete on disk
        # (train.py no longer encodes on the fly); skip loading the encoders
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
        # …); validates its runtime contract and sets up state before
        # optimizer/accelerator wiring.
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

        # sigma_lowres step-span gate: seed the train-forward counter with the
        # resume offset (already in forward units) so it covers every resume
        # path, not just re-deriving from args inside the loop.
        self._sigma_span_step = initial_step

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

        # Emits the matching run_end (ok/stopped/error) on exit; run_start
        # already fired when the sink was constructed above.
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


# Network-module-consumed flags. Don't flow through argparse directly since
# `create_network` reads them from ``kwargs``; sourced from the flat
# ``NETWORK_KWARGS`` allowlist in ``networks/__init__.py``.
NETWORK_KWARG_ALLOWLIST: tuple[str, ...] = _all_network_kwargs()

# Top-level training args that flow through ``net_kwargs`` though they aren't
# network kwargs. Kept explicit; growth here should be reviewed.
_EXTRA_FORWARDED_TOP_LEVEL_ARGS: tuple[str, ...] = (
    # Postfix contrastive resets its intra-step reference set on step
    # boundary, so it needs the grad-accum window.
    "gradient_accumulation_steps",
)


def resolve_network_kwargs(args) -> dict[str, str]:
    """The single intake for network kwargs, merging both config paths
    (``--network_args k=v`` and allowlisted top-level config keys; CLI wins
    on overlap). Cached on ``args._network_kwargs`` — read from here rather
    than re-scanning ``args.network_args``. Values are strings.
    """
    cached = getattr(args, "_network_kwargs", None)
    if cached is not None:
        return cached

    net_kwargs: dict[str, str] = {}
    for net_arg in getattr(args, "network_args", None) or []:
        key, value = net_arg.split("=", 1)
        net_kwargs[key] = value

    # Forward known network-arg keys from top-level config (TOML), plus a
    # small tail of top-level training args network modules still want.
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

    The daemon launches us windowless under ``pythonw.exe``, which drops the
    child's stdout/stderr, so an uncaught traceback here is otherwise lost and
    the daemon falls back to a generic "process exited (code=1)".
    ``progress.jsonl`` is written by path, so it survives.

    ``run_scope`` already emits ``run_end(error=…)`` for in-loop failures, but
    only *after* ``ProgressSink.run_start`` fires — late in ``train()``.
    Errors before that (cache incomplete, config/dataset build, model load)
    escape it entirely; this excepthook is the catch-all.
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
