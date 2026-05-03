# LoRANetwork: the module-assembly / training-orchestration core of the LoRA
# adapter stack for Anima. Targets DiT blocks (and optionally text-encoder
# attention) with pluggable per-module classes supplied by a NetworkSpec.

import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import torch

from library.log import setup_logging
from library.training.metrics import MetricContext
from networks import NETWORK_REGISTRY, NetworkSpec, lora_save
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import (
    _parse_reft_layers,
    _refuse_split_hydra_keys,
    _refuse_unfused_attn_lora_keys,
    _stack_lora_ups,
)
from networks.lora_modules import (
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAExpModule,
    OrthoLoRAExpModule,
    ReFTModule,
    _sigma_sinusoidal_features,
)

setup_logging()
logger = logging.getLogger(__name__)

_BLOCK_IDX_RE = re.compile(r"blocks\.(\d+)\.")


class LoRANetwork(torch.nn.Module):
    # Target modules: DiT blocks, embedders, final layer. embedders and final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = [
        "Block",
        "PatchEmbed",
        "TimestepEmbedding",
        "FinalLayer",
    ]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Qwen3Attention",
        "Qwen3MLP",
        "Qwen3SdpaAttention",
        "Qwen3FlashAttention2",
    ]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        cfg: LoRANetworkCfg,
        *,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Mutable runtime state — explicitly NOT in cfg. ``set_multiplier`` and
        # ``set_loraplus_lr_ratio`` write these post-construction; per-step
        # diagnostics (hit counters, σ caches) accumulate during training.
        self.multiplier = multiplier
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None
        self._channel_scale_misses: List[str] = []
        self._channel_scale_hits: int = 0
        self._sigma_router_hits: int = 0
        self._hydra_router_hits: int = 0
        self._hydra_router_misses: int = 0
        self._last_sigma: Optional[torch.Tensor] = None
        # Per-expert pick fraction across hydra modules on the most recent
        # warmup step (random or best-by-grad path). None outside the warmup
        # window — the metric in metrics.py drops out of the dashboard then.
        self._last_expert_warmup_picks: Optional[torch.Tensor] = None
        # Hydra up-weight grad-norm snapshot (T-LoRA / σ-bucket conflict
        # diagnostic). Filled by ``capture_up_grad_stats`` between backward
        # and ``optimizer.zero_grad``; consumed by the ``hydra_up_grad``
        # metric. Values stay on-device until ``get_up_grad_stats`` runs the
        # D2H — capture happens every sync step but the metric only reads on
        # log steps, so the sync was the per-step bottleneck.
        self._last_up_grad_stats: Dict[str, object] = {}
        # Per-step cache for ``get_router_stats`` — both the progress-bar
        # postfix and the metrics layer call it on log steps. Cleared in
        # ``clear_step_caches`` so the next forward recomputes.
        self._router_stats_cache: Optional[Dict[str, object]] = None

        # Local aliases for the closure body and the post-closure ReFT block.
        # Reading via `cfg.foo` works too; aliases just keep the diff small.
        module_class = cfg.module_class
        modules_dim = cfg.modules_dim
        modules_alpha = cfg.modules_alpha
        dropout = cfg.dropout
        rank_dropout = cfg.rank_dropout
        module_dropout = cfg.module_dropout
        verbose = cfg.verbose
        alpha = cfg.alpha
        lora_dim = cfg.lora_dim
        train_llm_adapter = cfg.train_llm_adapter
        add_reft = cfg.add_reft
        reft_dim = cfg.reft_dim
        reft_alpha = cfg.reft_alpha
        reft_layers = cfg.reft_layers

        # Either regex (fresh-from-kwargs path) or explicit name set
        # (from-weights path, detected from checkpoint keys). Explicit set wins.
        self._sigma_router_names = (
            set(cfg.sigma_router_names) if cfg.sigma_router_names else None
        )
        self._sigma_router_re = (
            re.compile(cfg.sigma_router_layers)
            if (
                cfg.use_sigma_router
                and cfg.sigma_router_layers
                and self._sigma_router_names is None
            )
            else None
        )

        # Per-module HydraLoRA gating. Matching modules get the Hydra class;
        # non-matching modules fall back to plain LoRA / OrthoLoRAExp so MoE
        # capacity is concentrated where specialization is actually learnable.
        # Fresh path: regex over `original_name`. From-weights path: explicit
        # name set detected from checkpoint keys (mirrors sigma_router_names).
        # Explicit set wins. None on both = apply MoE everywhere (legacy).
        self._hydra_router_names = (
            set(cfg.hydra_router_names) if cfg.hydra_router_names else None
        )
        self._hydra_router_re = (
            re.compile(cfg.hydra_router_layers)
            if cfg.hydra_router_layers and self._hydra_router_names is None
            else None
        )

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
            )
            logger.info(
                f"neuron dropout: p={dropout}, rank dropout: p={rank_dropout}, module dropout: p={module_dropout}"
            )

        # compile regular expression if specified
        def str_to_re_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_pattern = re.compile(pattern)
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
                        continue
                    re_patterns.append(re_pattern)
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(cfg.exclude_patterns)
        include_re_patterns = str_to_re_patterns(cfg.include_patterns)

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[LoRAModule], List[str]]:
            prefix = (
                self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            )

            # First pass: collect candidate modules
            candidates = []
            for name, module in root_module.named_modules():
                if (
                    target_replace_modules is None
                    or module.__class__.__name__ in target_replace_modules
                ):
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = isinstance(child_module, torch.nn.Linear)
                        is_conv2d = isinstance(child_module, torch.nn.Conv2d)
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            # Strip torch.compile wrapper from module path
                            original_name = original_name.replace("_orig_mod.", "")
                            lora_name = f"{prefix}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = any(
                                pattern.fullmatch(original_name)
                                for pattern in exclude_re_patterns
                            )
                            included = any(
                                pattern.fullmatch(original_name)
                                for pattern in include_re_patterns
                            )
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            # layer range filter: skip blocks outside [layer_start, layer_end)
                            if is_unet and (
                                cfg.layer_start is not None or cfg.layer_end is not None
                            ):
                                block_match = _BLOCK_IDX_RE.match(original_name)
                                if block_match:
                                    block_idx = int(block_match.group(1))
                                    if (
                                        cfg.layer_start is not None
                                        and block_idx < cfg.layer_start
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} < {cfg.layer_start})"
                                            )
                                        continue
                                    if (
                                        cfg.layer_end is not None
                                        and block_idx >= cfg.layer_end
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} >= {cfg.layer_end})"
                                            )
                                        continue

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            else:
                                if cfg.reg_dims is not None:
                                    for reg, d in cfg.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = alpha
                                            logger.info(
                                                f"Module {original_name} matched with regex '{reg}' -> dim: {dim}"
                                            )
                                            break
                                if dim is None:
                                    if is_linear or is_conv2d_1x1:
                                        dim = (
                                            default_dim
                                            if default_dim is not None
                                            else lora_dim
                                        )
                                        alpha_val = alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    candidates.append(
                                        (
                                            lora_name,
                                            None,
                                            None,
                                            None,
                                            original_name,
                                            True,
                                        )
                                    )  # skipped
                                continue

                            candidates.append(
                                (
                                    lora_name,
                                    child_module,
                                    dim,
                                    alpha_val,
                                    original_name,
                                    False,
                                )
                            )

                    if target_replace_modules is None:
                        break

            # Second pass: create LoRA modules with progress bar
            from tqdm import tqdm

            loras = []
            skipped = []
            non_skipped = [
                (ln, cm, d, a, on) for ln, cm, d, a, on, skip in candidates if not skip
            ]
            skipped = [ln for ln, cm, d, a, on, skip in candidates if skip]

            label = (
                "DiT"
                if is_unet
                else f"TE{text_encoder_idx + 1}"
                if text_encoder_idx is not None
                else "model"
            )
            for lora_name, child_module, dim, alpha_val, original_name in tqdm(
                non_skipped, desc=f"Creating {label} LoRA", leave=False
            ):
                # Per-module class resolution: when the network's nominal class
                # is Hydra (MoE), narrow it to only the layers in the hydra
                # filter. Non-matching layers fall back to plain LoRA /
                # OrthoLoRAExp so router overhead + balance-loss pressure are
                # concentrated on sites where specialization is learnable.
                effective_module_class = module_class
                if (
                    module_class in (HydraLoRAModule, OrthoHydraLoRAExpModule)
                    and is_unet
                ):
                    if self._hydra_router_names is not None:
                        hydra_on = lora_name in self._hydra_router_names
                    elif self._hydra_router_re is not None:
                        hydra_on = bool(self._hydra_router_re.search(original_name))
                    else:
                        hydra_on = True
                    if hydra_on:
                        self._hydra_router_hits += 1
                    else:
                        self._hydra_router_misses += 1
                        effective_module_class = (
                            OrthoLoRAExpModule
                            if module_class is OrthoHydraLoRAExpModule
                            else LoRAModule
                        )

                extra_kwargs = {}
                if effective_module_class == OrthoLoRAExpModule:
                    pass  # no extra kwargs — SVD init reads from org_module directly
                elif effective_module_class == OrthoHydraLoRAExpModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                elif effective_module_class == HydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    if cfg.expert_init_std > 0.0:
                        extra_kwargs["expert_init_std"] = cfg.expert_init_std

                # Hard σ-band expert partition: applied to every Hydra/
                # OrthoHydra module (independent of the σ-feature router
                # regex). Each module owns the partition; the network-level
                # ``set_sigma`` propagates ``_sigma`` to enable per-step band
                # selection. Validation (E % N == 0) lives in cfg parsing.
                if (
                    cfg.specialize_experts_by_sigma_buckets
                    and effective_module_class
                    in (HydraLoRAModule, OrthoHydraLoRAExpModule)
                    and is_unet
                ):
                    extra_kwargs["specialize_experts_by_sigma_buckets"] = True
                    extra_kwargs["num_sigma_buckets"] = cfg.num_sigma_buckets
                    if cfg.sigma_bucket_boundaries is not None:
                        extra_kwargs["sigma_bucket_boundaries"] = (
                            cfg.sigma_bucket_boundaries
                        )

                # σ-conditional router: only widen the router input with
                # sinusoidal(σ) features on modules whose name matches the
                # layer filter (cross_attn.q / self_attn.qkv by default — see
                # B0 pre-analysis in timestep-hydra.md). From-weights path uses
                # an explicit name set; fresh-from-kwargs path uses a regex
                # over original_name. Gated on the effective class so a
                # hydra-excluded module can't pick up σ either.
                if (
                    cfg.use_sigma_router
                    and effective_module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAExpModule,
                    )
                    and is_unet
                ):
                    if self._sigma_router_names is not None:
                        enable = lora_name in self._sigma_router_names
                    elif self._sigma_router_re is not None:
                        enable = bool(self._sigma_router_re.search(original_name))
                    else:
                        enable = True
                    if enable:
                        extra_kwargs["sigma_feature_dim"] = cfg.sigma_feature_dim
                        extra_kwargs["sigma_hidden_dim"] = cfg.sigma_hidden_dim
                        self._sigma_router_hits += 1

                # Per-channel scaling is DiT-only: the bench script hooks DiT
                # linears, text encoder activations are never calibrated.
                if cfg.channel_scales_dict is not None and is_unet:
                    _cs = cfg.channel_scales_dict.get(lora_name)
                    if _cs is not None:
                        extra_kwargs["channel_scale"] = _cs
                        self._channel_scale_hits += 1
                    else:
                        self._channel_scale_misses.append(lora_name)

                lora = effective_module_class(
                    lora_name,
                    child_module,
                    self.multiplier,
                    dim,
                    alpha_val,
                    dropout=dropout,
                    rank_dropout=rank_dropout,
                    module_dropout=module_dropout,
                    **extra_kwargs,
                )
                lora.original_name = original_name
                loras.append(lora)

            return loras, skipped

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        # Skip for OrthoLoRA since SVD init is expensive and TE modules are discarded in apply_to anyway
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        if text_encoders is not None and module_class not in (
            OrthoLoRAExpModule,
            OrthoHydraLoRAExpModule,
        ):
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create LoRA for Text Encoder {i + 1}:")
                te_loras, te_skipped = create_modules(
                    False,
                    i,
                    text_encoder,
                    LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                )
                logger.info(
                    f"create LoRA for Text Encoder {i + 1}: {len(te_loras)} modules."
                )
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)

        logger.info(f"create LoRA for Anima DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        if cfg.channel_scales_dict is not None:
            logger.info(
                f"per_channel_scaling: {self._channel_scale_hits} DiT modules "
                f"received calibration-based input scaling"
            )
            if self._channel_scale_misses:
                logger.warning(
                    f"per_channel_scaling: {len(self._channel_scale_misses)} DiT modules "
                    f"have no calibration stats (first: {self._channel_scale_misses[:3]}). "
                    f"These will train without input rebalancing — regenerate the stats "
                    f"file with `python archive/bench/analyze_lora_input_channels.py "
                    f"--dump_channel_stats <path>` if this is unexpected."
                )

        # Create ReFT modules on the DiT residual stream (block outputs), following
        # Wu et al. (2024) §3.3 — one intervention per selected block, not per
        # internal Linear. Selection is controlled by ``reft_layers``.
        self.unet_refts: List[ReFTModule] = []
        self.text_encoder_refts: List[ReFTModule] = []
        if add_reft:
            dit_blocks = getattr(unet, "blocks", None)
            if dit_blocks is None or len(dit_blocks) == 0:
                raise ValueError(
                    "add_reft=True but DiT has no .blocks attribute to wrap. "
                    "Block-level ReFT requires a transformer with a `blocks` ModuleList."
                )
            num_blocks = len(dit_blocks)
            selected_indices = _parse_reft_layers(reft_layers, num_blocks)

            reft_alpha_value = reft_alpha if reft_alpha is not None else alpha
            for idx in selected_indices:
                block = dit_blocks[idx]
                block_embed_dim = getattr(block, "x_dim", None)
                if block_embed_dim is None:
                    raise ValueError(
                        f"Block {idx} ({type(block).__name__}) has no `x_dim`; "
                        "cannot infer embed_dim for ReFT."
                    )
                reft_name = f"reft_unet_blocks_{idx}"
                reft = ReFTModule(
                    reft_name,
                    block,
                    embed_dim=block_embed_dim,
                    multiplier=multiplier,
                    reft_dim=reft_dim,
                    alpha=reft_alpha_value,
                    dropout=dropout,
                    module_dropout=module_dropout,
                )
                reft.original_name = f"blocks.{idx}"
                self.unet_refts.append(reft)
            logger.info(
                f"create ReFT for Anima DiT: {len(self.unet_refts)}/{num_blocks} "
                f"blocks (reft_dim={reft_dim}, layers={reft_layers!r})"
            )

        # assertion: no duplicate names
        names = set()
        for lora in (
            self.text_encoder_loras
            + self.unet_loras
            + self.text_encoder_refts
            + self.unet_refts
        ):
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

        # Alias each sigma-aware module's ``_sigma`` / ``_sigma_features``
        # buffer to a single network-level shared tensor. ``set_sigma`` then
        # updates the shared tensor in place once and every aliased module
        # buffer sees the new value through shared storage — instead of
        # ~56 per-module ``copy_`` calls per training step.
        self._wire_shared_sigma_buffers()

    def _wire_shared_sigma_buffers(self) -> None:
        """Replace each HydraLoRA / OrthoHydraLoRA module's ``_sigma`` and
        ``_sigma_features`` buffers with references to a single network-level
        tensor (per sigma_feature_dim for the features). Modules then read the
        same tensor object as their own attribute, so an in-place ``copy_`` on
        the network's shared buffer flows to every module without a Python
        propagation loop.

        Run once at the end of ``__init__`` — before any forward fires, so
        Dynamo / cudagraphs capture the aliased data pointer on first compile
        and never see a per-module pointer-mismatch event.
        """
        sigma_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_sigma" not in lora._buffers:
                continue
            sigma_loras.append(lora)
            d = int(getattr(lora, "sigma_feature_dim", 0))
            if d > 0 and "_sigma_features" in lora._buffers:
                by_dim.setdefault(d, []).append(lora)
        self._sigma_aware_loras = sigma_loras
        self._sigma_aware_loras_by_dim = by_dim
        if not sigma_loras:
            self._shared_sigma = None
            self._shared_sigma_features: Dict[int, torch.Tensor] = {}
            return

        # Pick the first module's placeholder buffer as the canonical shared
        # tensor; rebind every other module's buffer to the same object. The
        # placeholder is shape (1,) / (1, dim) — set_sigma replaces it with a
        # full-shape tensor on the first call (and re-aliases at the same time).
        shared_sigma = sigma_loras[0]._buffers["_sigma"]
        for lora in sigma_loras:
            lora._buffers["_sigma"] = shared_sigma
        self._shared_sigma = shared_sigma

        self._shared_sigma_features = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_sigma_features"]
            for lora in loras:
                lora._buffers["_sigma_features"] = shared_feat
            self._shared_sigma_features[dim] = shared_feat

    def prepare_network(self, args):
        if getattr(args, "lora_fp32_accumulation", False):
            logger.warning(
                "--lora_fp32_accumulation is deprecated and has no effect; "
                "fp32 accumulation is now unconditional in LoRA/Hydra/ReFT "
                "bottleneck matmuls. Remove the flag from your config."
            )

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def fuse_weights(self):
        """Merge all LoRA deltas into base model weights for zero-overhead inference."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.fuse_weight()

    def unfuse_weights(self):
        """Remove all LoRA deltas from base model weights."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unfuse_weight()

    def set_timestep_mask(self, timesteps: torch.Tensor, max_timestep: float = 1.0):
        """Compute and set timestep-dependent rank mask on all modules."""
        if not self.cfg.use_timestep_mask:
            return

        max_rank = self.cfg.lora_dim
        # Reuse a single GPU-resident mask to avoid ~200 CPU→GPU transfers per step
        mask = getattr(self, "_shared_timestep_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, max_rank, device=timesteps.device)
            self._shared_timestep_mask = mask
            self._timestep_mask_arange = torch.arange(max_rank, device=timesteps.device)
            for lora in self.text_encoder_loras + self.unet_loras:
                lora._timestep_mask = mask

        # Compute threshold r entirely on device — avoids GPU→CPU .item() sync and
        # keeps the effective rank as a tensor so the mask build stays static-shape.
        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = (
            frac.pow(self.cfg.alpha_rank_scale) * (max_rank - self.cfg.min_rank)
            + self.cfg.min_rank
        )
        r = r.clamp(max=float(max_rank))
        mask.copy_((self._timestep_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def set_reft_timestep_mask(
        self, timesteps: torch.Tensor, max_timestep: float = 1.0
    ):
        """Compute and set timestep-dependent mask on ReFT modules."""
        if not self.cfg.use_timestep_mask:
            return
        refts = self.text_encoder_refts + self.unet_refts
        if not refts:
            return
        reft_dim = self.cfg.reft_dim

        mask = getattr(self, "_shared_reft_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, reft_dim, device=timesteps.device)
            self._shared_reft_mask = mask
            self._reft_mask_arange = torch.arange(reft_dim, device=timesteps.device)
            for reft in refts:
                reft._timestep_mask = mask

        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = frac.pow(self.cfg.alpha_rank_scale) * (reft_dim - 1) + 1
        r = r.clamp(max=float(reft_dim))
        mask.copy_((self._reft_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def clear_timestep_mask(self):
        """Restore full-rank masks on every LoRA / ReFT module.

        Each module's ``_timestep_mask`` is a Tensor by construction (default
        all-ones buffer at init, rebound to the shared live-updated mask when
        ``set_timestep_mask`` runs). Clearing fills the shared masks with ones
        in place — modules that were rebound immediately see the neutral mask
        via the shared reference; modules with local defaults are already
        neutral. Never set to None: the always-a-Tensor invariant is what
        keeps the adapter forward free of a None-vs-Tensor guard under
        ``compile_mode=full``.
        """
        shared = getattr(self, "_shared_timestep_mask", None)
        if shared is not None:
            shared.fill_(1.0)
        shared_reft = getattr(self, "_shared_reft_mask", None)
        if shared_reft is not None:
            shared_reft.fill_(1.0)

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        """Stash per-sample σ on every HydraLoRA module whose router accepts σ.

        Mirrors ``set_timestep_mask`` — one call per training step. σ and the
        sinusoidal-features tensor are stored in network-level shared buffers
        whose storage is aliased into every sigma-aware module's ``_sigma`` /
        ``_sigma_features`` (see ``_wire_shared_sigma_buffers``), so the
        update is one in-place ``copy_`` per shared tensor instead of a
        per-module Python loop.

        IMPORTANT: write in place rather than rebinding. Inductor captures
        the buffers as static cudagraph inputs and re-records the whole graph
        if the data pointer changes — rebinding every step caused per-step
        re-record under ``compile_inductor_mode=reduce-overhead``
        (cudagraph_trees log: "static input data pointer changed"). Pointer
        only changes on the first call (placeholder → full-shape) and on a
        rare batch-shape change; both re-alias every module to the new tensor.
        """
        sigmas = sigmas.detach()
        self._last_sigma = sigmas
        # Either path needs per-module ``_sigma``: σ-feature concat router
        # (sigma_feature_dim>0) and hard σ-band expert partition. Skip the
        # propagation entirely when neither is configured.
        if not (
            self.cfg.use_sigma_router or self.cfg.specialize_experts_by_sigma_buckets
        ):
            return
        sigma_loras = self._sigma_aware_loras
        if not sigma_loras:
            return

        shared_sigma = self._shared_sigma
        target_dtype = shared_sigma.dtype
        cast = sigmas.to(dtype=target_dtype, device=shared_sigma.device)
        if shared_sigma.shape == cast.shape:
            shared_sigma.copy_(cast)
        else:
            new_sigma = cast.detach().clone()
            for lora in sigma_loras:
                lora._buffers["_sigma"] = new_sigma
            self._shared_sigma = new_sigma
            shared_sigma = new_sigma

        for dim, loras in self._sigma_aware_loras_by_dim.items():
            feat = _sigma_sinusoidal_features(shared_sigma, dim).detach()
            shared_feat = self._shared_sigma_features[dim]
            cast_feat = feat.to(dtype=shared_feat.dtype, device=shared_feat.device)
            if shared_feat.shape == cast_feat.shape:
                shared_feat.copy_(cast_feat)
            else:
                new_feat = cast_feat.clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat

    def clear_sigma(self) -> None:
        """Reset cached σ to zeros.

        Never set to None: ``_sigma`` stays a Tensor so the unconditional
        sinusoidal path in ``_compute_gate`` has no None-vs-Tensor guard to
        recompile on under ``compile_mode=full``. Used in eval / validation
        and by inference teardown (``clear_hydra_sigma``). Zero in place to
        keep the cudagraph data pointer stable (see ``set_sigma`` note).

        Operates on the shared buffers populated by
        ``_wire_shared_sigma_buffers`` — one zero per shared tensor flows to
        every aliased module.
        """
        self._last_sigma = None
        if not self._sigma_aware_loras:
            return
        if self._shared_sigma is not None:
            self._shared_sigma.zero_()
            for dim, shared_feat in self._shared_sigma_features.items():
                zero_feat = _sigma_sinusoidal_features(self._shared_sigma, dim)
                cast_feat = zero_feat.to(
                    dtype=shared_feat.dtype, device=shared_feat.device
                )
                if shared_feat.shape == cast_feat.shape:
                    shared_feat.copy_(cast_feat)
                else:
                    new_feat = cast_feat.detach().clone()
                    for lora in self._sigma_aware_loras_by_dim[dim]:
                        lora._buffers["_sigma_features"] = new_feat
                    self._shared_sigma_features[dim] = new_feat

    def clear_step_caches(self) -> None:
        """Drop per-step tensor references (``_last_gate``) between training
        steps.

        ``_last_gate`` caches a tensor produced inside the compiled forward —
        under ``torch.compile(mode='reduce-overhead')`` that tensor lives in
        the inductor cudagraph memory pool. Holding a Python reference across
        the step boundary prevents ``cudagraph_trees`` from reclaiming pool
        memory and silently demotes the run to the eager fallback path. Call
        this right before ``torch.compiler.cudagraph_mark_step_begin()`` so
        the pool is free to reuse memory on the next iteration.

        ``_sigma`` is intentionally *not* cleared: it's rebound by
        ``set_sigma`` before every forward, the caller passes a tensor from
        outside the compiled region (the flow-matching sampler's ``timesteps``,
        not a pool-allocated intermediate), and keeping it a Tensor at all
        times is what lets the adapter ``_compute_gate`` drop the None-vs-
        Tensor guard under ``compile_mode=full``.

        Safe to call unconditionally — consumers (balance loss, router stats)
        read ``_last_gate`` only within the step that wrote it.
        """
        self._last_sigma = None
        self._router_stats_cache = None
        for lora in self.unet_loras + self.text_encoder_loras:
            if hasattr(lora, "_last_gate"):
                lora._last_gate = None

    def step_balance_loss_warmup(self, global_step: int, max_train_steps: int) -> None:
        """Activate the MoE load-balance penalty once training crosses the
        warmup window. Step function: ``_balance_loss_weight`` holds at 0
        during the first ``_balance_loss_warmup_ratio`` of steps, then flips
        to ``_balance_loss_target_weight``. No-op unless both attributes are
        attached (hydra post_init) and the ratio is > 0.

        Letting the router specialize before the penalty kicks in avoids
        pinning it to uniform at init; flipping the penalty on after warmup
        keeps a diverged router from collapsing to a single expert.
        """
        target = float(getattr(self, "_balance_loss_target_weight", 0.0) or 0.0)
        ratio = float(getattr(self, "_balance_loss_warmup_ratio", 0.0) or 0.0)
        if ratio <= 0.0 or max_train_steps <= 0 or target <= 0.0:
            return
        warmup_steps = int(max_train_steps * ratio)
        self._balance_loss_weight = 0.0 if global_step < warmup_steps else target

    def step_expert_warmup(self, global_step: int, max_train_steps: int) -> None:
        """Per-step random expert-gradient masking during the warmup window.

        Forward stays full MoE (every expert contributes via the learned gate),
        but only one randomly-sampled expert per module receives gradient this
        step. Breaks the zero-init expert symmetry without ever training an
        expert in isolation — each expert always sees the other experts'
        contribution when it updates. See ``docs/methods/hydra-lora.md``
        §Expert-warmup for motivation.

        Expert index is sampled independently per module so that different
        modules can route the same sample to different experts. After the
        warmup window the mask is filled with ones, so the unconditional
        ``up*m + up.detach()*(1-m)`` term in the adapter forward collapses to
        identity — every expert receives gradient normally.

        State lives entirely in the ``_expert_grad_mask`` buffer; dynamo
        treats buffer mutations as dynamic (no recompile per step and no
        recompile at the warmup→post-warmup transition).
        """
        if self.cfg.expert_warmup_ratio <= 0.0 or max_train_steps <= 0:
            return
        warmup_steps = int(max_train_steps * self.cfg.expert_warmup_ratio)
        in_warmup = global_step < warmup_steps
        k = self.cfg.expert_warmup_k
        pick_counts: Optional[torch.Tensor] = None
        n_modules = 0
        for lora in self.unet_loras + self.text_encoder_loras:
            if not hasattr(lora, "_expert_grad_mask"):
                continue
            mask = lora._expert_grad_mask
            if in_warmup:
                k_eff = max(1, min(k, lora.num_experts))
                if k_eff >= lora.num_experts:
                    mask.fill_(1.0)
                else:
                    idx = torch.randperm(lora.num_experts, device=mask.device)[:k_eff]
                    mask.zero_()
                    mask[idx] = 1.0
                if pick_counts is None or pick_counts.numel() < lora.num_experts:
                    new = torch.zeros(lora.num_experts, device=mask.device)
                    if pick_counts is not None:
                        new[: pick_counts.numel()] = pick_counts
                    pick_counts = new
                pick_counts[: lora.num_experts] += mask[: lora.num_experts].detach()
                n_modules += 1
            else:
                mask.fill_(1.0)
        if in_warmup and pick_counts is not None and n_modules > 0:
            self._last_expert_warmup_picks = pick_counts / n_modules
        else:
            self._last_expert_warmup_picks = None

    def step_expert_best_warmup_post_backward(
        self, global_step: int, max_train_steps: int
    ) -> None:
        """Greedy counterpart to ``step_expert_warmup``: during the warmup
        window, keep gradient only on the top-k experts ranked by per-expert
        grad-norm; zero the rest. Forward stays full MoE (mask all-ones), so
        every expert produces a proper gradient — we then drop the experts
        whose update would do least to lower loss this step. Score is
        ``‖∂L/∂P_e‖_F`` where ``P_e`` is the per-expert parameter
        (``lora_up_weight[e]`` for plain Hydra, ``S_p[e]`` for OrthoHydra);
        grad-norm is a first-order proxy for the loss decrease an SGD step on
        that expert would buy.

        Must be called AFTER backward (so .grad is populated) and BEFORE
        optimizer.step. Outside the warmup window this is a no-op.

        Mutually exclusive with ``expert_warmup_ratio``: that path zeros the
        forward mask so non-selected experts have zero grad anyway, which
        would make this top-k selection redundant. ``factory.py`` warns if
        both are set; behaviorally the random path's pre-forward masking
        wins because it produces zero grads that this method then sees.
        """
        ratio = self.cfg.expert_best_warmup_ratio
        if ratio <= 0.0 or max_train_steps <= 0:
            return
        warmup_steps = int(max_train_steps * ratio)
        if global_step >= warmup_steps:
            self._last_expert_warmup_picks = None
            return
        k = self.cfg.expert_warmup_k
        pick_counts: Optional[torch.Tensor] = None
        n_modules = 0
        for lora in self.unet_loras + self.text_encoder_loras:
            param = None
            if hasattr(lora, "lora_up_weight") and lora.lora_up_weight.dim() == 3:
                param = lora.lora_up_weight
            elif hasattr(lora, "S_p") and lora.S_p.dim() == 3:
                param = lora.S_p
            if param is None or param.grad is None:
                continue
            E = param.shape[0]
            k_eff = max(1, min(k, E))
            if k_eff >= E:
                continue
            norms = param.grad.detach().reshape(E, -1).norm(dim=-1)
            topk = torch.topk(norms, k_eff).indices
            keep = torch.zeros(E, dtype=param.grad.dtype, device=param.grad.device)
            keep[topk] = 1.0
            param.grad.mul_(keep.view(E, *([1] * (param.dim() - 1))))
            if pick_counts is None or pick_counts.numel() < E:
                new = torch.zeros(E, device=keep.device)
                if pick_counts is not None:
                    new[: pick_counts.numel()] = pick_counts
                pick_counts = new
            pick_counts[:E] += keep.float()
            n_modules += 1
        if pick_counts is not None and n_modules > 0:
            self._last_expert_warmup_picks = pick_counts / n_modules
        else:
            self._last_expert_warmup_picks = None

    def get_expert_warmup_pick_stats(self) -> Optional[List[float]]:
        """Per-expert pick fraction (0.0–1.0) across hydra modules on the most
        recent warmup step. None when not in warmup. ``metrics.py`` consumes
        this and emits ``hydra/expert_warmup_pick/{i}`` scalars."""
        picks = self._last_expert_warmup_picks
        if picks is None:
            return None
        return picks.detach().cpu().tolist()

    @staticmethod
    def _switch_balance(gate: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer balance: E · Σ_i frac_i · mean_gate_i. Scalar."""
        num_experts = gate.shape[-1]
        expert_idx = gate.argmax(dim=-1)  # (B,)
        frac = torch.zeros(num_experts, device=gate.device, dtype=gate.dtype)
        frac.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=gate.dtype))
        frac = frac / gate.shape[0]
        gate_mean = gate.mean(dim=0)  # (num_experts,)
        return num_experts * (frac * gate_mean).sum()

    def get_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer load-balancing loss averaged over HydraLoRA modules.

        Global term aggregates gates over the full batch. When σ-conditional
        routing is on, also adds a per-σ-bucket term so global balance can't
        mask per-bucket collapse (expert i only at high σ, expert j only at
        low σ: globally balanced but per-bucket one-hot). Buckets are fixed
        thresholds on σ∈[0,1]; for N=3 that's [1/3, 2/3]. Under logit-normal
        σ sampling this is ~30/40/30 — close enough to equal-frequency for v1.
        """
        total = None
        per_bucket_total = None
        count = 0
        per_bucket_count = 0

        sigma = self._last_sigma  # (B,) or None
        num_buckets = self.cfg.num_sigma_buckets
        bucket_w = float(self.cfg.per_bucket_balance_weight or 0.0)
        want_per_bucket = (
            self.cfg.use_sigma_router
            and sigma is not None
            and num_buckets > 1
            and bucket_w > 0.0
        )
        if want_per_bucket:
            thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
                1:-1
            ]
            bucket_ids = torch.bucketize(sigma.float(), thresholds)  # (B,) in [0, N)

        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            term = self._switch_balance(gate)
            total = term if total is None else total + term
            count += 1

            if want_per_bucket and getattr(lora, "sigma_feature_dim", 0) > 0:
                # Only penalize per-bucket collapse on modules that actually
                # have σ-conditional routing capacity to collapse.
                module_bucket_sum = None
                module_bucket_count = 0
                for b in range(num_buckets):
                    mask = bucket_ids == b
                    if int(mask.sum()) < 2:
                        # Not enough samples to meaningfully measure balance
                        # in this bucket on this step; skip.
                        continue
                    bterm = self._switch_balance(gate[mask])
                    module_bucket_sum = (
                        bterm
                        if module_bucket_sum is None
                        else module_bucket_sum + bterm
                    )
                    module_bucket_count += 1
                if module_bucket_sum is not None:
                    per_bucket_total = (
                        module_bucket_sum / module_bucket_count
                        if per_bucket_total is None
                        else per_bucket_total + module_bucket_sum / module_bucket_count
                    )
                    per_bucket_count += 1

        if total is None:
            return torch.tensor(0.0)
        out = total / count
        if per_bucket_total is not None and per_bucket_count > 0:
            out = out + bucket_w * (per_bucket_total / per_bucket_count)
        return out

    def get_router_entropy(self) -> Optional[float]:
        """Mean per-sample normalized entropy of hydra router gates, averaged
        across modules. Returns None when no hydra module has cached a gate
        this step. Thin wrapper over :meth:`get_router_stats` kept for the
        progress-bar postfix path; prefer ``get_router_stats`` for logging.
        """
        stats = self.get_router_stats()
        return stats.get("entropy_mean") if stats else None

    def get_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float], List[List[float]], List[int]]]:
        """Per-step router diagnostics aggregated across hydra modules.

        Returns a dict with:
          - entropy_mean / entropy_p05 / entropy_p50 / entropy_p95: normalized
            per-module entropy (0 = one-hot collapse, 1 = uniform), pooled
            across modules.
          - margin_mean: mean top1-top2 softmax gap, averaged over batch then
            modules. High margin = confident routing; near-zero = effectively
            random.
          - expert_usage: length-E vector of argmax frequency averaged across
            modules. Sums to ~1.0. Flat distribution = balanced; a column
            near 0 means that expert is never picked (collapse).
          - expert_usage_per_bucket: (num_buckets, E) list of argmax frequency
            per σ-bucket (low→high σ), averaged across modules. Empty buckets
            (no batch samples in that σ range this step) report zeros.
          - bucket_counts: per-bucket sample count (length num_buckets). Useful
            sanity for the per-bucket usage row — a bucket with 0 samples this
            step has a meaningless usage row.

          Per-bucket entries omitted when σ wasn't set this step or
          ``num_sigma_buckets <= 1``.

        Empty dict when no hydra module cached a gate this step.

        Vectorized: per-module gates with matching shape are stacked into one
        ``(M, B, E)`` tensor and reduced in a single pass per metric. The
        previous per-module Python loop emitted ~9 small kernels per Hydra
        module (clamp / log / sum / topk(2) / argmax / scatter_add_ / ones_like
        / div), stalling the post-step boundary by ~500 launches at the
        56-module default stack. This implementation issues a constant ~10
        launches regardless of module count (see
        ``docs/optimizations/nsys_analysis_0503.md``).

        Result is memoized on ``self._router_stats_cache`` and invalidated by
        ``clear_step_caches`` so the progress-bar postfix and the logging
        metric share one computation per step.
        """
        if self._router_stats_cache is not None:
            return self._router_stats_cache

        # Collect gates with matching expert count. Modules with mismatched E
        # are skipped (aggregating usage vectors of different length isn't
        # meaningful) — same policy as the previous per-module loop.
        gates: List[torch.Tensor] = []
        E_ref: Optional[int] = None
        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            E = gate.shape[-1]
            if E <= 1:
                continue
            if E_ref is None:
                E_ref = E
            elif E != E_ref:
                continue
            gates.append(gate)

        if not gates:
            return {}

        g = torch.stack(gates, dim=0)  # (M, B, E)
        M, B, E = g.shape

        sigma = self._last_sigma  # (B,) or None
        num_buckets = int(self.cfg.num_sigma_buckets)
        want_per_bucket = sigma is not None and num_buckets > 1
        # When ``specialize_experts_by_sigma_buckets`` is on, each sample can
        # only route to its band's ``E / num_buckets`` experts (others masked
        # to -inf pre-softmax). Normalizing entropy by ``log(E)`` then caps
        # the achievable max at ``log(experts_per_band) / log(E)`` (e.g. 0.44
        # for E=12, num_buckets=4) — making "uniform within band" look like
        # collapse. Normalize by the actual reachable support instead.
        band_partition_active = bool(
            self.cfg.specialize_experts_by_sigma_buckets and num_buckets > 1
        )
        effective_E = (E // num_buckets) if band_partition_active else E
        norm = math.log(effective_E) if effective_E > 1 else 1.0

        p = g.float().clamp_min(1e-12)
        # (M,) per-module mean entropy, normalized to [0, 1] over reachable support
        H_per_module = -(p * p.log()).sum(dim=-1).mean(dim=-1) / norm
        # (M, B, 2) top-2 in one batched topk → (M,) mean margin
        top2 = p.topk(2, dim=-1).values
        margin_per_module = (top2[..., 0] - top2[..., 1]).mean(dim=-1)
        # Argmax usage: one_hot + sum → (M, E) histograms in one pass
        expert_idx = g.argmax(dim=-1)  # (M, B)
        usage_per_module = torch.nn.functional.one_hot(expert_idx, num_classes=E).to(
            g.dtype
        ).sum(dim=1) / float(B)  # (M, E)

        H_per_module = H_per_module.detach()
        q_probs = torch.tensor(
            [0.05, 0.5, 0.95], device=H_per_module.device, dtype=H_per_module.dtype
        )
        q = torch.quantile(H_per_module, q_probs)  # (3,)
        # Single packed summary: [mean_H, p05, p50, p95, margin_mean]. One DtoH.
        summary = torch.stack(
            [H_per_module.mean(), q[0], q[1], q[2], margin_per_module.detach().mean()]
        ).cpu()
        usage_mean = usage_per_module.detach().mean(dim=0).cpu().tolist()
        out: Dict[str, Union[float, List[float], List[List[float]], List[int]]] = {
            "entropy_mean": float(summary[0]),
            "entropy_p05": float(summary[1]),
            "entropy_p50": float(summary[2]),
            "entropy_p95": float(summary[3]),
            "margin_mean": float(summary[4]),
            "expert_usage": usage_mean,
        }

        if want_per_bucket and sigma is not None:
            thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
                1:-1
            ]
            bucket_ids = torch.bucketize(sigma.float(), thresholds).clamp(
                0, num_buckets - 1
            )  # (B,)
            bucket_counts_t = torch.zeros(
                num_buckets, device=sigma.device, dtype=torch.long
            )
            bucket_counts_t.scatter_add_(
                0, bucket_ids, torch.ones_like(bucket_ids, dtype=torch.long)
            )
            # Per-bucket argmax frequency, normalized within each bucket so
            # each row sums to ~1 (or 0 for empty buckets). Flat scatter_add
            # over (M, num_buckets * E) avoids a per-module loop.
            bucket_ids_dev = bucket_ids.to(expert_idx.device)
            flat_idx = bucket_ids_dev[None, :] * E + expert_idx  # (M, B)
            bu = torch.zeros(M, num_buckets * E, device=g.device, dtype=g.dtype)
            bu.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=g.dtype))
            bu = bu.view(M, num_buckets, E)
            bc = bucket_counts_t.to(g.dtype).clamp_min(1).view(1, num_buckets, 1)
            bucket_usage_mean = (bu / bc).detach().mean(dim=0).cpu().tolist()
            out["expert_usage_per_bucket"] = bucket_usage_mean
            out["bucket_counts"] = bucket_counts_t.cpu().tolist()

        self._router_stats_cache = out
        return out

    def capture_up_grad_stats(self) -> None:
        """Snapshot per-expert grad-norm on Hydra up-weights.

        Diagnoses the T-LoRA × σ-bucket interaction: under the σ-band
        partition, a high-σ-band expert only fires at high σ, where T-LoRA
        clamps the rank to ``min_rank``. Rank columns ``[min_rank, R)`` of
        that expert's ``lora_up`` should then accumulate near-zero gradient
        — those columns are dead capacity. Reading ``lora_up_weight.grad``
        and splitting the L2 norm at the ``min_rank`` boundary makes the
        effect directly visible.

        Must be called between ``accelerator.backward(loss)`` and
        ``optimizer.zero_grad`` — once ``zero_grad(set_to_none=True)`` has
        run, ``.grad`` is ``None``.

        Stash format (read by ``library/training/metrics.py``):
          ``below`` : (E,) Σ_modules ‖grad[e, :, :min_rank]‖²  (only when T-LoRA active)
          ``above`` : (E,) Σ_modules ‖grad[e, :, min_rank:]‖²  (only when T-LoRA active)
          ``total`` : (E,) Σ_modules ‖grad[e, ...]‖²
          ``sp_total`` : (E,) Σ_modules ‖S_p.grad[e]‖²  (OrthoHydra)
          ``below_band`` / ``above_band`` / ``total_band`` / ``sp_total_band``
            (B,) per-σ-band sums (sum over experts assigned to band b).
            Only present when ``specialize_experts_by_sigma_buckets`` is on.
          ``min_rank`` : float, snapshot of ``cfg.min_rank`` for context.
          ``num_buckets`` : float, snapshot of ``cfg.num_sigma_buckets``.

        Square-norms (sum-of-squares) are reported, not L2 norms — the
        metric layer takes ``sqrt`` after summation. This keeps aggregation
        across modules correct (concatenation-of-grads norm = sqrt of
        sum-of-squares per chunk).
        """
        if not getattr(self, "_use_hydra", False):
            self._last_up_grad_stats = {}
            return

        use_tlora = bool(self.cfg.use_timestep_mask)
        min_rank = int(self.cfg.min_rank) if use_tlora else 0
        max_rank = int(self.cfg.lora_dim)
        # Clamp min_rank: a misconfig like min_rank > lora_dim would make the
        # "above" slice empty and the "below" slice full-rank, silently turning
        # the diagnostic into a no-op. Pin to [0, R].
        min_rank = max(0, min(min_rank, max_rank))
        has_tlora_split = use_tlora and 0 < min_rank < max_rank

        # Collect grads first; reduce in a few fused passes at the end. The
        # naive per-module loop launched ~4–7 tiny kernels per Hydra module
        # and stalled the post-backward / pre-optimizer boundary by hundreds
        # of ms on log steps (see docs/optimizations/nsys_analysis_0503.md).
        up_grads: List[torch.Tensor] = []  # each (E, out_i, R)
        sp_grads: List[torch.Tensor] = []  # each (E, r, r)
        expert_band_ref: Optional[torch.Tensor] = None

        for lora in self.unet_loras + self.text_encoder_loras:
            up = getattr(lora, "lora_up_weight", None)
            sp = getattr(lora, "S_p", None)
            up_grad = up.grad if isinstance(up, torch.nn.Parameter) else None
            sp_grad = sp.grad if isinstance(sp, torch.nn.Parameter) else None
            if up_grad is not None:
                up_grads.append(up_grad.detach())
            if sp_grad is not None and sp_grad.dim() == 3:
                # (E, r, r) — OrthoHydra rotation generator. No clean rank-col
                # split (Cayley couples all entries), so we report total only.
                # Plain OrthoLoRA's S_p is (r, r) with no expert axis — skipped
                # by the dim==3 check, since this diagnostic is per-expert.
                sp_grads.append(sp_grad.detach())
            if expert_band_ref is None:
                band = getattr(lora, "_expert_band", None)
                if band is not None:
                    expert_band_ref = band.detach()

        if not up_grads and not sp_grads:
            self._last_up_grad_stats = {}
            return

        total_per_exp: Optional[torch.Tensor] = None
        below_per_exp: Optional[torch.Tensor] = None
        above_per_exp: Optional[torch.Tensor] = None
        sp_total_per_exp: Optional[torch.Tensor] = None
        device_ref: Optional[torch.device] = None

        if up_grads:
            # All entries share E and R; only out_i varies. Cat along the out
            # axis into one (E, sum_out, R) tensor and reduce in one pass.
            big_up = torch.cat(up_grads, dim=1).float()
            sq_up = big_up.square()
            total_per_exp = sq_up.sum(dim=(1, 2))
            device_ref = total_per_exp.device
            if has_tlora_split:
                # Slices are views into ``sq_up``; sum along (out, rank-slice).
                below_per_exp = sq_up[:, :, :min_rank].sum(dim=(1, 2))
                above_per_exp = sq_up[:, :, min_rank:].sum(dim=(1, 2))

        if sp_grads:
            # All entries share (E, r, r). Stack into (M, E, r, r) and reduce
            # over modules + r×r in one pass.
            big_sp = torch.stack(sp_grads, dim=0).float()
            sp_total_per_exp = big_sp.square().sum(dim=(0, 2, 3))
            if device_ref is None:
                device_ref = sp_total_per_exp.device

        # Stash on-device tensors only — the D2H happens in
        # ``get_up_grad_stats`` so non-log steps avoid the
        # ``cudaStreamSynchronize`` that .cpu().tolist() forces.
        out: Dict[str, object] = {
            "min_rank": [float(min_rank)],
            "num_buckets": [float(self.cfg.num_sigma_buckets)],
        }
        if total_per_exp is not None:
            out["total"] = total_per_exp
        if below_per_exp is not None and above_per_exp is not None:
            out["below"] = below_per_exp
            out["above"] = above_per_exp
        if sp_total_per_exp is not None:
            out["sp_total"] = sp_total_per_exp

        # Per-band aggregation: scatter the per-expert sum-of-squares along
        # _expert_band. Only meaningful when σ-bucket partition is active —
        # otherwise the band assignment is undefined and per-band rows would
        # be misleading.
        if (
            expert_band_ref is not None
            and bool(self.cfg.specialize_experts_by_sigma_buckets)
            and int(self.cfg.num_sigma_buckets) > 1
        ):
            B = int(self.cfg.num_sigma_buckets)
            band = expert_band_ref.to(device_ref)

            def _scatter_to_band(per_exp: torch.Tensor) -> torch.Tensor:
                buf = torch.zeros(B, device=per_exp.device, dtype=per_exp.dtype)
                buf.scatter_add_(0, band, per_exp)
                return buf

            if total_per_exp is not None:
                out["total_band"] = _scatter_to_band(total_per_exp)
            if below_per_exp is not None and above_per_exp is not None:
                out["below_band"] = _scatter_to_band(below_per_exp)
                out["above_band"] = _scatter_to_band(above_per_exp)
            if sp_total_per_exp is not None:
                out["sp_total_band"] = _scatter_to_band(sp_total_per_exp)

        self._last_up_grad_stats = out

    def get_up_grad_stats(self) -> Dict[str, List[float]]:
        """Materialize the on-device stash from ``capture_up_grad_stats``.

        D2H is deferred to here so non-log steps don't pay the sync — the
        capture must run between backward and zero_grad (when ``.grad`` is
        live), but the metric only consumes the result on log steps.
        """
        raw = self._last_up_grad_stats
        if not raw:
            return {}
        materialized: Dict[str, List[float]] = {}
        for k, v in raw.items():
            if torch.is_tensor(v):
                materialized[k] = v.detach().cpu().tolist()
            else:
                materialized[k] = list(v)  # type: ignore[arg-type]
        return materialized

    def get_ortho_regularization(self) -> torch.Tensor:
        """Sum orthogonality regularization from all OrthoLoRA and ReFT modules."""
        total_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "regularization"):
                p_reg, q_reg = lora.regularization()
                total_reg = total_reg + p_reg + q_reg
                count += 1
        for reft in self.text_encoder_refts + self.unet_refts:
            total_reg = total_reg + reft.regularization()
            count += 1
        return total_reg / max(count, 1)

    def metrics(self, ctx: MetricContext) -> dict[str, float]:
        """Emit log-step keys owned by the LoRA network.

        Covers ortho regularization, hydra balance loss, router stats, hydra
        up-weight grad-norm diagnostics, and the per-expert warmup pick
        distribution. Each block returns nothing if its driver is off
        (``_ortho_reg_weight == 0``, ``_use_hydra == False``, etc.) so the
        cost on inactive paths is one attr check.
        """
        out: dict[str, float] = {}

        # Ortho regularization magnitude.
        ortho_w = float(getattr(self, "_ortho_reg_weight", 0.0) or 0.0)
        if ortho_w > 0.0:
            v = self.get_ortho_regularization()
            if torch.is_tensor(v):
                v = v.detach().item()
            out["reg/ortho"] = float(v)
            out["reg/ortho_weighted"] = float(ortho_w * v)

        # Hydra balance loss magnitude.
        bal_w = float(getattr(self, "_balance_loss_weight", 0.0) or 0.0)
        if bal_w > 0.0:
            v = self.get_balance_loss()
            if torch.is_tensor(v):
                v = v.detach().item()
            out["reg/balance"] = float(v)
            out["reg/balance_weighted"] = float(bal_w * v)

        if not getattr(self, "_use_hydra", False):
            return out

        # Router diagnostics.
        stats = self.get_router_stats()
        if stats:
            out["hydra/router_entropy"] = float(stats["entropy_mean"])
            out["hydra/router_entropy_p05"] = float(stats["entropy_p05"])
            out["hydra/router_entropy_p50"] = float(stats["entropy_p50"])
            out["hydra/router_entropy_p95"] = float(stats["entropy_p95"])
            out["hydra/router_margin"] = float(stats["margin_mean"])
            for i, v in enumerate(stats.get("expert_usage", [])):
                out[f"hydra/expert_usage/{i}"] = float(v)
            for b, row in enumerate(stats.get("expert_usage_per_bucket", [])):
                for i, v in enumerate(row):
                    out[f"hydra/expert_usage_b{b}/{i}"] = float(v)
            for b, c in enumerate(stats.get("bucket_counts", [])):
                out[f"hydra/bucket_count/{b}"] = float(c)

        # Hydra up-weight grad norms by rank region and σ-band.
        up = self.get_up_grad_stats()
        if up:
            eps = 1e-12

            def _emit_per_expert(prefix: str, sq: list[float]) -> None:
                for i, v in enumerate(sq):
                    out[f"hydra/up_grad/{prefix}/exp{i}"] = float(v) ** 0.5

            def _emit_per_band(prefix: str, sq: list[float]) -> None:
                for b, v in enumerate(sq):
                    out[f"hydra/up_grad/{prefix}/band{b}"] = float(v) ** 0.5

            if "total" in up:
                _emit_per_expert("total", up["total"])
            if "below" in up and "above" in up:
                _emit_per_expert("below", up["below"])
                _emit_per_expert("above", up["above"])
                for i, (b_, a_) in enumerate(zip(up["below"], up["above"])):
                    out[f"hydra/up_grad/above_below_ratio/exp{i}"] = float(
                        a_
                    ) ** 0.5 / (float(b_) ** 0.5 + eps)
            if "sp_total" in up:
                _emit_per_expert("sp_total", up["sp_total"])
            if "total_band" in up:
                _emit_per_band("total", up["total_band"])
            if "below_band" in up and "above_band" in up:
                _emit_per_band("below", up["below_band"])
                _emit_per_band("above", up["above_band"])
                for b, (bv, av) in enumerate(zip(up["below_band"], up["above_band"])):
                    out[f"hydra/up_grad/above_below_ratio/band{b}"] = float(
                        av
                    ) ** 0.5 / (float(bv) ** 0.5 + eps)
            if "sp_total_band" in up:
                _emit_per_band("sp_total", up["sp_total_band"])

        # Per-expert pick distribution during expert warmup.
        picks = self.get_expert_warmup_pick_stats()
        if picks is not None:
            for i, v in enumerate(picks):
                out[f"hydra/expert_warmup_pick/{i}"] = float(v)

        return out

    @staticmethod
    def _strip_orig_mod_keys(state_dict):
        """Strip torch.compile '_orig_mod_' from state_dict keys for compat with old checkpoints."""
        new_sd = {}
        for key, val in state_dict.items():
            new_key = re.sub(r"(?<=_)_orig_mod_", "", key)
            new_sd[new_key] = val
        return new_sd

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state_dict = self._strip_orig_mod_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        for key in list(weights_sd.keys()):
            if key.endswith(".dora_scale"):
                new_key = key.replace(".dora_scale", ".magnitude")
                weights_sd[new_key] = weights_sd.pop(key)

        # Stack per-expert hydra ups into fused lora_up_weight (training form).
        weights_sd = _stack_lora_ups(weights_sd)
        # Refuse split hydra attn keys BEFORE the regular refuser: hydra splits
        # carry no lora_up.weight, so the regular path would skip them anyway,
        # but running hydra first means any non-hydra attention still goes
        # through the normal code path cleanly.
        weights_sd = _refuse_split_hydra_keys(weights_sd)
        # Refuse unfused attn projections (inverse of save_weights defusing).
        weights_sd = _refuse_unfused_attn_lora_keys(weights_sd)

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(
                f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules"
            )
        else:
            self.text_encoder_loras = []
            self.text_encoder_refts = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []
            self.unet_refts = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # ReFT wraps each selected DiT Block's forward, so the chain is:
        #   Block.__call__ -> ReFT.forward -> original Block.forward
        #   (inside which LoRA-wrapped Linears still fire normally).
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.apply_to()
            self.add_module(reft.lora_name, reft)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for DiT")
        else:
            self.unet_loras = []

        # Pre-group checkpoint keys by LoRA module prefix (avoid O(modules * keys) scan)
        # Keys are "{module_name}.{param}" where module_name has no dots (dots → underscores)
        grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in weights_sd.items():
            prefix, dot, suffix = key.partition(".")
            if not dot:
                continue
            if prefix not in grouped_sd:
                grouped_sd[prefix] = {}
            grouped_sd[prefix][suffix] = value

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = grouped_sd.get(lora.lora_name, {})
            if sd_for_lora:
                lora.merge_to(sd_for_lora, dtype, device)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
    ):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(
            f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}"
        )
        logger.info(
            f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}"
        )

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        if text_encoder_lr is None or (
            isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0
        ):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass  # already a list with one element

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}, "router": {}}
            reg_groups = {}
            reg_lrs_list = (
                list(self.cfg.reg_lrs.items()) if self.cfg.reg_lrs is not None else []
            )
            router_scale = float(self.cfg.router_lr_scale)

            def _is_router_param(pname: str) -> bool:
                # named_parameters() yields top-level names like "router.weight"
                # — no leading dot. σ features live inside router.weight now
                # (columns [lora_dim:] of the weight), so there's a single path.
                return pname.startswith("router.")

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(
                            f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}"
                        )
                        break

                for name, param in lora.named_parameters():
                    is_router = _is_router_param(name)
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {
                                "lora": {},
                                "plus": {},
                                "router": {},
                                "lr": reg_lr,
                            }
                        if is_router:
                            reg_groups[group_key]["router"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        elif loraplus_ratio is not None and (
                            "lora_up" in name
                            or "p_layer" in name
                            or "learned_source" in name
                        ):
                            reg_groups[group_key]["plus"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        else:
                            reg_groups[group_key]["lora"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        continue

                    if is_router:
                        param_groups["router"][f"{lora.lora_name}.{name}"] = param
                    elif loraplus_ratio is not None and (
                        "lora_up" in name
                        or "p_layer" in name
                        or "learned_source" in name
                    ):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus", "router"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = (
                            reg_lr * loraplus_ratio
                            if loraplus_ratio is not None
                            else reg_lr
                        )
                    elif key == "router":
                        param_data["lr"] = reg_lr * router_scale
                    else:
                        param_data["lr"] = reg_lr
                    if (
                        param_data.get("lr", None) == 0
                        or param_data.get("lr", None) is None
                    ):
                        logger.info("NO LR skipping!")
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(
                        desc
                        + (
                            " plus"
                            if key == "plus"
                            else (" router" if key == "router" else "")
                        )
                    )

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    elif key == "router":
                        param_data["lr"] = lr * router_scale
                    else:
                        param_data["lr"] = lr
                if (
                    param_data.get("lr", None) == 0
                    or param_data.get("lr", None) is None
                ):
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append(
                    "plus" if key == "plus" else ("router" if key == "router" else "")
                )
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = (
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            )
            te1_loras = [
                lora
                for lora in self.text_encoder_loras
                if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER)
            ]
            if len(te1_loras) > 0:
                logger.info(
                    f"Text Encoder 1 (Qwen3): {len(te1_loras)} modules, LR {text_encoder_lr[0]}"
                )
                params, descriptions = assemble_params(
                    te1_loras, text_encoder_lr[0], loraplus_ratio
                )
                all_params.extend(params)
                lr_descriptions.extend(
                    ["textencoder 1" + (" " + d if d else "") for d in descriptions]
                )

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["unet" + (" " + d if d else "") for d in descriptions]
            )

        if self.text_encoder_refts:
            params, descriptions = assemble_params(
                self.text_encoder_refts,
                text_encoder_lr[0],
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["reft textencoder" + (" " + d if d else "") for d in descriptions]
            )

        if self.unet_refts:
            params, descriptions = assemble_params(
                self.unet_refts,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["reft unet" + (" " + d if d else "") for d in descriptions]
            )

        # HydraLoRA per-module routers are submodules of HydraLoRAModule instances,
        # so they are already captured by the unet_loras param group above.

        if getattr(self, "apex_condition_shift", None) is not None:
            shift_params = list(self.apex_condition_shift.parameters())
            if len(shift_params) > 0:
                shift_lr_scale = float(getattr(self, "_apex_shift_lr_scale", 0.1))
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("APEX ConditionShift: no base LR, skipping param group")
                else:
                    shift_lr = float(base_lr) * shift_lr_scale
                    all_params.append({"params": shift_params, "lr": shift_lr})
                    lr_descriptions.append("apex condition_shift")
                    logger.info(
                        f"APEX ConditionShift param group: lr={shift_lr:.2e} "
                        f"({shift_lr_scale}x of unet_lr={base_lr})"
                    )

        if getattr(self, "repa_head", None) is not None:
            repa_params = list(self.repa_head.parameters())
            if len(repa_params) > 0:
                repa_lr_scale = float(getattr(self, "_repa_lr_scale", 1.0))
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("REPA head: no base LR, skipping param group")
                else:
                    repa_lr = float(base_lr) * repa_lr_scale
                    all_params.append({"params": repa_params, "lr": repa_lr})
                    lr_descriptions.append("repa head")
                    logger.info(
                        f"REPA head param group: lr={repa_lr:.2e} "
                        f"({repa_lr_scale}x of unet_lr={base_lr})"
                    )

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        spec: NetworkSpec = getattr(self, "_network_spec", NETWORK_REGISTRY["lora"])
        if metadata is None:
            metadata = {}
        if metadata:
            metadata["ss_network_spec"] = spec.name

        # Hard σ-band partition lives in non-persistent buffers (`_expert_band`)
        # and a Python attr (`_sigma_band_partition`); nothing of it survives
        # the state_dict write. Emit the two scalars needed to re-register the
        # partition at load time so inference (`make test`) and the ComfyUI
        # node can reconstruct the per-sample band mask. Only stamped when the
        # partition is on, so older non-band checkpoints stay byte-identical.
        if self.cfg.specialize_experts_by_sigma_buckets:
            metadata["ss_specialize_experts_by_sigma_buckets"] = "true"
            metadata["ss_num_sigma_buckets"] = str(int(self.cfg.num_sigma_buckets))
            if self.cfg.sigma_bucket_boundaries is not None:
                import json as _json

                metadata["ss_sigma_bucket_boundaries"] = _json.dumps(
                    list(self.cfg.sigma_bucket_boundaries)
                )

        state_dict = self.state_dict()
        lora_save.save_network_weights(
            state_dict,
            file=file,
            dtype=dtype,
            metadata=metadata,
            save_variant=spec.save_variant,
        )

    def backup_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                org_module._lora_org_weight = org_module.weight.detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                if not org_module._lora_restored:
                    org_module.weight.data.copy_(org_module._lora_org_weight)
                    org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                lora_weight = lora.get_weight().to(
                    org_module.weight.device, dtype=org_module.weight.dtype
                )
                org_module.weight.data.add_(lora_weight)

                org_module._lora_restored = False
                lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (
                    (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(
                    down.permute(1, 0, 2, 3), up
                ).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
