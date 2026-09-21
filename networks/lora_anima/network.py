# LoRANetwork: module-assembly / training-orchestration core of the LoRA adapter
# stack. Targets DiT blocks (+ optional TE attention) via per-module classes
# supplied by a NetworkSpec.

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch

from library.log import setup_logging
from networks import NETWORK_REGISTRY, NetworkSpec, lora_save
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import (
    _refuse_split_hydra_keys,
    _refuse_unfused_attn_lora_keys,
    _stack_lora_ups,
)
from networks.lora_modules import (
    HydraLoRAModule,
    LoRAModule,
    StepExpertLoRAModule,
    _sigma_sinusoidal_features,
)
from networks.lora_anima.network_metrics import _NetworkMetricsMixin

# Re-exported from routers.py.
from networks.lora_anima.routers import (  # noqa: F401
    CROSSATTN_EMB_DIM,
    GlobalRouter,
)

setup_logging()
logger = logging.getLogger(__name__)

_BLOCK_IDX_RE = re.compile(r"blocks\.(\d+)\.")


class LoRANetwork(_NetworkMetricsMixin, torch.nn.Module):
    # embedders + final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = [
        "Block",
        "PatchEmbed",
        "TimestepEmbedding",
        "FinalLayer",
    ]
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
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

        # Mutable runtime state — explicitly NOT in cfg (written post-construction
        # / accumulated during training).
        self.multiplier = multiplier
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None
        self._channel_scale_misses: List[str] = []
        self._channel_scale_hits: int = 0
        self._grad_basis_misses: List[str] = []
        self._grad_basis_hits: int = 0
        self._sigma_router_hits: int = 0
        self._hydra_router_hits: int = 0
        self._hydra_router_misses: int = 0
        self._last_sigma: Optional[torch.Tensor] = None
        # Hydra up-weight grad-norm snapshot; filled by capture_up_grad_stats,
        # stays on-device until get_up_grad_stats runs the D2H.
        self._last_up_grad_stats: Dict[str, object] = {}
        # Per-step cache for get_router_stats; cleared in clear_step_caches.
        self._router_stats_cache: Optional[Dict[str, object]] = None
        # State-dict prefixes of training-only submodules (e.g. REPA head);
        # save_weights strips them so attaching an aux head is inference-safe.
        self._training_only_prefixes: set = set()

        # Local aliases read by the closure body.
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

        # cfg.router_targets is the single regex governing which Linears are
        # routed (Hydra/σ/FEI share it); from-weights supplies an explicit
        # per-family name set that wins instead.
        _router_re = re.compile(cfg.router_targets) if cfg.router_targets else None

        self._sigma_router_names = (
            set(cfg.sigma_router_names) if cfg.sigma_router_names else None
        )
        self._sigma_router_re = (
            _router_re
            if (
                cfg.router_source == "sigma"
                and _router_re is not None
                and self._sigma_router_names is None
            )
            else None
        )

        self._fei_router_names = (
            set(cfg.fei_router_names) if cfg.fei_router_names else None
        )
        self._fei_router_re = (
            _router_re
            if (
                cfg.router_source == "fei"
                and _router_re is not None
                and self._fei_router_names is None
            )
            else None
        )
        self._fei_router_hits = 0
        # use_global_router=True modules (shared_A + route_per_layer=False):
        # per-layer router skipped, gates from the network GlobalRouter instead.
        self._global_router_hits = 0
        # Read via getattr by library/inference/adapters.py.
        self.use_fei_router = cfg.router_source == "fei"
        self.use_sigma_router = cfg.router_source == "sigma"
        # Shared-A Hydra + network-level router: Hydra modules skip their own
        # router and consume GlobalRouter gates.
        self._use_global_router_for_hydra = (
            cfg.use_moe_style == "shared_A"
            and not cfg.route_per_layer
            and cfg.router_source != "none"
        )

        # Per-module HydraLoRA gating: matched → Hydra class, else plain
        # LoRA. Fresh build: regex over original_name. From-weights:
        # explicit name set wins. None on both = MoE everywhere.
        self._hydra_router_names = (
            set(cfg.hydra_router_names) if cfg.hydra_router_names else None
        )
        self._hydra_router_re = (
            _router_re
            if (_router_re is not None and self._hydra_router_names is None)
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
                                            logger.debug(
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
                                # Per-pattern alpha override (reg_dims/reg_lrs
                                # trio); applies regardless of how dim was set.
                                if cfg.reg_alphas is not None and dim:
                                    for reg, a in cfg.reg_alphas.items():
                                        if re.fullmatch(reg, original_name):
                                            alpha_val = a
                                            logger.debug(
                                                f"Module {original_name} matched with regex '{reg}' -> alpha: {a}"
                                            )
                                            break

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
                # Nominal Hydra class narrows to hydra-filter-matched layers;
                # non-matching layers fall back to plain LoRA.
                effective_module_class = module_class
                if module_class is HydraLoRAModule and is_unet:
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
                        effective_module_class = LoRAModule

                extra_kwargs = {}
                if effective_module_class == StepExpertLoRAModule:
                    # Shared down + K step-indexed up-heads, selected per
                    # forward via set_step_index / the turbo coordinator.
                    extra_kwargs["step_expert_K"] = cfg.step_expert_K
                elif effective_module_class == HydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    if cfg.expert_init_std > 0.0:
                        extra_kwargs["expert_init_std"] = cfg.expert_init_std
                    if self._use_global_router_for_hydra:
                        extra_kwargs["use_global_router"] = True
                        self._global_router_hits += 1

                # Hard σ-band expert partition (independent of the σ-router
                # regex); set_sigma propagates _sigma for per-step band
                # selection. E % N == 0 validated in cfg.
                if (
                    cfg.specialize_experts_by_sigma_buckets
                    and effective_module_class is HydraLoRAModule
                    and is_unet
                ):
                    extra_kwargs["specialize_experts_by_sigma_buckets"] = True
                    extra_kwargs["num_sigma_buckets"] = cfg.num_sigma_buckets
                    if cfg.sigma_bucket_boundaries is not None:
                        extra_kwargs["sigma_bucket_boundaries"] = (
                            cfg.sigma_bucket_boundaries
                        )

                # σ-conditional router: widen the router input with sinusoidal(σ)
                # on layer-filter-matched modules. Skipped under use_global_router
                # (network router consumes σ once, per-Linear cat dead).
                if (
                    cfg.router_source == "sigma"
                    and effective_module_class is HydraLoRAModule
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._sigma_router_names is not None:
                        enable = lora_name in self._sigma_router_names
                    elif self._sigma_router_re is not None:
                        enable = bool(self._sigma_router_re.search(original_name))
                    else:
                        enable = True
                    if enable:
                        extra_kwargs["sigma_feature_dim"] = cfg.sigma_feature_dim
                        self._sigma_router_hits += 1

                # FEI-conditional router: same gating as σ, widens
                # the router input with the per-sample FEI simplex (set_fei).
                if (
                    cfg.router_source == "fei"
                    and effective_module_class is HydraLoRAModule
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._fei_router_names is not None:
                        enable_fei = lora_name in self._fei_router_names
                    elif self._fei_router_re is not None:
                        enable_fei = bool(self._fei_router_re.search(original_name))
                    else:
                        enable_fei = True
                    if enable_fei:
                        extra_kwargs["fei_feature_dim"] = cfg.fei_feature_dim
                        self._fei_router_hits += 1

                # SVD-Down init — plain two-factor LoRAModule only; gate so the
                # kwarg never reaches Hydra/StepExpert.
                if cfg.down_init != "kaiming" and effective_module_class is LoRAModule:
                    extra_kwargs["down_init"] = cfg.down_init
                    if cfg.down_init == "weight_svd" and cfg.svd_slice:
                        extra_kwargs["svd_slice"] = int(cfg.svd_slice)
                    # Gradient-SVD modes carry a per-layer basis; DiT-only (the
                    # sketch never ran on the TE) and a missing key means that
                    # module keeps Kaiming, counted for the summary below.
                    if cfg.grad_basis_dict is not None:
                        if is_unet:
                            _gb = cfg.grad_basis_dict.get(lora_name)
                            extra_kwargs["grad_basis"] = _gb
                            if _gb is None:
                                self._grad_basis_misses.append(lora_name)
                            else:
                                self._grad_basis_hits += 1
                        else:
                            extra_kwargs["grad_basis"] = None

                # Per-channel scaling is DiT-only — TE activations are never calibrated.
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

        # Qwen3 TE, typically not trained.
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        if text_encoders is not None:
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
                f"channel_scaling: {self._channel_scale_hits} DiT modules "
                f"received calibration-based input scaling"
            )
            if self._channel_scale_misses:
                logger.warning(
                    f"channel_scaling: {len(self._channel_scale_misses)} DiT modules "
                    f"have no calibration stats (first: {self._channel_scale_misses[:3]}). "
                    f"These will train without input rebalancing — regenerate the vendored "
                    f"calibration with `python scripts/calibration/analyze_lora_input_channels.py "
                    f"--per_artist --dump_channel_stats networks/calibration/channel_stats.safetensors` "
                    f"if this is unexpected."
                )

        if cfg.grad_basis_dict is not None:
            logger.info(
                f"down_init={cfg.down_init}: {self._grad_basis_hits} DiT modules "
                f"seeded from the gradient basis"
            )
            if self._grad_basis_misses:
                logger.warning(
                    f"down_init={cfg.down_init}: {len(self._grad_basis_misses)} DiT "
                    f"modules have no basis entry (first: {self._grad_basis_misses[:3]}) "
                    f"and keep Kaiming. A basis is depth-baked and built from the same "
                    f"target enumeration — a large count means the artifact does not "
                    f"match this checkpoint."
                )

        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

        # Alias each module's σ/FEI/routing buffers to one network-level shared
        # tensor so set_* updates in place once (vs a per-module copy_ loop).
        self._wire_shared_sigma_buffers()
        self._wire_shared_fei_buffers()
        self._wire_shared_routing_buffers()

        # Network-level GlobalRouter when cfg selects MoE without per-Linear
        # routers; shared_A modules built with use_global_router=True consume
        # its broadcast gates.
        self.global_router: Optional[GlobalRouter] = None
        # Tells call sites to fire set_crossattn_routing with the pooled text
        # tensor each forward (broadcasts to _routing_weights).
        self.use_crossattn_router: bool = False
        if cfg.use_moe_style is not False and not cfg.route_per_layer:
            router_layer_norm = False
            if cfg.router_source == "fei":
                router_input_dim = int(cfg.fei_feature_dim)
            elif cfg.router_source == "sigma":
                router_input_dim = int(cfg.sigma_feature_dim)
            elif cfg.router_source == "crossattn_emb":
                # Pooled post-LLM-adapter text feature (DiT's cross-attn K/V).
                # LN on by default — wide T5-space variance budget.
                router_input_dim = CROSSATTN_EMB_DIM
                router_layer_norm = True
            else:
                router_input_dim = 0
            if router_input_dim > 0 and cfg.num_experts > 1:
                self.global_router = GlobalRouter(
                    input_dim=router_input_dim,
                    num_experts=int(cfg.num_experts),
                    hidden_dim=int(cfg.router_hidden_dim),
                    tau=float(cfg.router_tau),
                    apply_layer_norm=router_layer_norm,
                )
                self.use_crossattn_router = cfg.router_source == "crossattn_emb"
                logger.info(
                    f"GlobalRouter: source={cfg.router_source!r}, "
                    f"input_dim={router_input_dim}, "
                    f"num_experts={cfg.num_experts}, "
                    f"hidden={cfg.router_hidden_dim}, τ={cfg.router_tau:.2f}, "
                    f"LN={router_layer_norm}, "
                    f"routing-aware modules={len(self._routing_aware_loras)}"
                )

        # Depth of the DiT this adapter is being trained against, stamped into
        # save_weights metadata as ss_num_blocks. Read here rather than derived
        # from module names later, which layer_start/layer_end filtering would
        # under-count.
        self._trained_num_blocks = len(unet.blocks) if hasattr(unet, "blocks") else 0

    def _wire_shared_sigma_buffers(self) -> None:
        """Alias each Hydra module's ``_sigma``/``_sigma_features``
        buffers to one network-level tensor, so a ``copy_`` on the shared
        buffer flows to every module without a Python propagation loop.

        Must run before any forward fires (end of ``__init__``), so Dynamo /
        cudagraphs capture the aliased data pointer on first compile.
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

        # First module's placeholder buffer is canonical; rebind the rest to it.
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

    def _wire_shared_fei_buffers(self) -> None:
        """Mirror of ``_wire_shared_sigma_buffers`` for ``_fei`` (per feature
        dim). GOTCHA: ``Module._apply`` (``.to(device)``) reallocates buffers
        independently and silently breaks the alias if callers don't
        identity-check before writing — see ``set_sigma`` / ``set_fei``.
        """
        fei_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            d = int(getattr(lora, "fei_feature_dim", 0))
            if d <= 0:
                continue
            if "_fei" not in lora._buffers:
                continue
            fei_loras.append(lora)
            by_dim.setdefault(d, []).append(lora)
        self._fei_aware_loras = fei_loras
        self._fei_aware_loras_by_dim = by_dim
        if not fei_loras:
            self._shared_fei: Dict[int, torch.Tensor] = {}
            return

        self._shared_fei = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_fei"]
            for lora in loras:
                lora._buffers["_fei"] = shared_feat
            self._shared_fei[dim] = shared_feat

    def _wire_shared_broadcast_buffer(
        self, buffer_name: str, aware_attr: str, shared_attr: str
    ) -> None:
        """Alias every module carrying ``buffer_name`` to one shared ``(1, E)``
        tensor — broadcast scaffold behind the routing gate buffer. No per-dim split (unlike ``_shared_fei``): all such modules
        share one ``num_experts`` by construction.
        """
        loras = [
            lora
            for lora in self.unet_loras + self.text_encoder_loras
            if buffer_name in lora._buffers
        ]
        setattr(self, aware_attr, loras)
        canonical = loras[0]._buffers[buffer_name] if loras else None
        for lora in loras:
            lora._buffers[buffer_name] = canonical
        setattr(self, shared_attr, canonical)

    def _wire_shared_routing_buffers(self) -> None:
        self._wire_shared_broadcast_buffer(
            "_routing_weights", "_routing_aware_loras", "_shared_routing_weights"
        )

    def prepare_network(self, args):
        if getattr(args, "lora_fp32_accumulation", False):
            logger.warning(
                "--lora_fp32_accumulation is deprecated and has no effect; "
                "fp32 accumulation is now unconditional in LoRA/Hydra "
                "bottleneck matmuls. Remove the flag from your config."
            )

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def set_step_index(self, step: int) -> None:
        """Select the active step-expert up-head on every adapted module.

        No-op on non-step-expert modules (no ``set_step``). Mirrors the turbo
        coordinator's ``set_student_step``; both reach ``StepExpertLoRAModule._step``.
        """
        for lora in self.text_encoder_loras + self.unet_loras:
            set_step = getattr(lora, "set_step", None)
            if callable(set_step):
                set_step(step)

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

        # ONE shared mask PER DISTINCT RANK, not one per network. GOTCHA: a
        # per-pattern rank override (reg_dims, e.g. adaln_rank) leaves a
        # mixed-rank module set — a single (1, cfg.lora_dim) mask broadcast
        # against a smaller rank raises RuntimeError at the first mismatched
        # block. Group by each module's own lora_dim to keep every multiply
        # shape-exact.
        masks = getattr(self, "_shared_timestep_masks", None)
        if masks is None or any(m.device != timesteps.device for m in masks.values()):
            masks = {}
            self._timestep_mask_aranges = {}
            for lora in self.text_encoder_loras + self.unet_loras:
                rank = int(getattr(lora, "lora_dim", self.cfg.lora_dim))
                if rank not in masks:
                    masks[rank] = torch.zeros(1, rank, device=timesteps.device)
                    self._timestep_mask_aranges[rank] = torch.arange(
                        rank, device=timesteps.device
                    )
                lora._timestep_mask = masks[rank]
            self._shared_timestep_masks = masks

        # Compute threshold r entirely on device — avoids a GPU→CPU .item() sync.
        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        frac = frac.pow(self.cfg.alpha_rank_scale)
        for rank, mask in masks.items():
            # Each group masks the same FRACTION of its own rank (so an r16
            # override follows the same schedule shape as an r32 bulk); floor
            # is clamped into the group's own range.
            floor = min(float(self.cfg.min_rank), float(rank))
            r = (frac * (rank - floor) + floor).clamp(max=float(rank))
            mask.copy_(
                (self._timestep_mask_aranges[rank] < r).to(mask.dtype).unsqueeze(0)
            )

    def clear_timestep_mask(self):
        """Restore full-rank masks on every LoRA module (fill in place, never
        set to None — the always-a-Tensor invariant is what keeps the adapter
        forward free of a None-vs-Tensor guard under ``torch.compile``).
        """
        for shared in (getattr(self, "_shared_timestep_masks", None) or {}).values():
            shared.fill_(1.0)

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        """Stash per-sample σ on every HydraLoRA module whose router accepts σ.

        Writes in place (not by rebinding) to keep the data pointer stable
        under cudagraph reduce-overhead. GOTCHA: ``Module._apply`` (``.to(device)``)
        reallocates each buffer independently and orphans ``_shared_sigma`` —
        every call identity-checks the canonical buffer and rebinds if broken,
        else ``copy_`` writes to a stale tensor and every module reads zeros
        (manifested only at B=1).
        """
        sigmas = sigmas.detach()
        self._last_sigma = sigmas
        # Skip propagation when neither the σ-router nor sigma-bucket
        # partition is configured.
        if not (
            self.cfg.router_source == "sigma"
            or self.cfg.specialize_experts_by_sigma_buckets
        ):
            return
        sigma_loras = self._sigma_aware_loras
        if not sigma_loras:
            return

        canonical = sigma_loras[0]._buffers["_sigma"]
        cast = sigmas.to(dtype=canonical.dtype, device=canonical.device)
        # Rebind when the shared attr lost identity with canonical or the
        # shape changed (placeholder → full batch).
        needs_rebind = (
            self._shared_sigma is not canonical or canonical.shape != cast.shape
        )
        if needs_rebind:
            new_sigma = cast.detach().clone()
            for lora in sigma_loras:
                lora._buffers["_sigma"] = new_sigma
            self._shared_sigma = new_sigma
            shared_sigma = new_sigma
        else:
            canonical.copy_(cast)
            shared_sigma = canonical

        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            feat = _sigma_sinusoidal_features(shared_sigma, dim).detach()
            cast_feat = feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            feat_needs_rebind = (
                self._shared_sigma_features.get(dim) is not canonical_feat
                or canonical_feat.shape != cast_feat.shape
            )
            if feat_needs_rebind:
                new_feat = cast_feat.clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat
            else:
                canonical_feat.copy_(cast_feat)

    def clear_sigma(self) -> None:
        """Reset cached σ to zeros (eval / validation / inference teardown).

        Never None: ``_sigma`` stays a Tensor so ``_compute_gate`` has no
        None-vs-Tensor guard to recompile on. Zero in place; same aliasing
        recovery as ``set_sigma``.
        """
        self._last_sigma = None
        if not self._sigma_aware_loras:
            return
        sigma_loras = self._sigma_aware_loras
        canonical = sigma_loras[0]._buffers["_sigma"]
        if self._shared_sigma is not canonical:
            for lora in sigma_loras:
                lora._buffers["_sigma"] = canonical
            self._shared_sigma = canonical
        canonical.zero_()
        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            if self._shared_sigma_features.get(dim) is not canonical_feat:
                for lora in loras:
                    lora._buffers["_sigma_features"] = canonical_feat
                self._shared_sigma_features[dim] = canonical_feat
            zero_feat = _sigma_sinusoidal_features(canonical, dim)
            cast_feat = zero_feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            if canonical_feat.shape == cast_feat.shape:
                canonical_feat.copy_(cast_feat)
            else:
                new_feat = cast_feat.detach().clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat

    def set_fei(self, fei: torch.Tensor) -> None:
        """Stash per-sample FEI ``[B, fei_dim]`` on every FEI-aware module.

        Parallel to ``set_sigma`` — one call per step, same shared-buffer
        aliasing recovery. ``fei`` is
        ``(B, fei_feature_dim)``, computed by
        ``library.runtime.fei.compute_fei_2band``. When a ``GlobalRouter`` is
        wired (``route_per_layer=False``), it fires here too and broadcasts
        gates via ``set_routing_weights`` in the same call.
        """
        fei = fei.detach()
        # Fast-path: nothing to do with no per-Linear FEI consumer and no
        # global router.
        has_per_layer_fei = bool(getattr(self, "_fei_aware_loras", None))
        global_fei_router = (
            self.global_router
            if (
                self.global_router is not None
                and self.cfg.router_source == "fei"
                and not self.cfg.route_per_layer
            )
            else None
        )
        if not (has_per_layer_fei or global_fei_router is not None):
            return
        if not (self.use_fei_router or global_fei_router is not None):
            return

        # Per-layer FEI broadcast (per-Linear FEI routers).
        if has_per_layer_fei:
            for dim, loras in self._fei_aware_loras_by_dim.items():
                canonical = loras[0]._buffers["_fei"]
                cast = fei.to(dtype=canonical.dtype, device=canonical.device)
                if cast.dim() == 1:
                    cast = cast.unsqueeze(0)
                if cast.shape[-1] != dim:
                    raise ValueError(
                        f"set_fei: fei.shape[-1]={cast.shape[-1]} != fei_feature_dim={dim}"
                    )
                current_shared = self._shared_fei.get(dim)
                needs_rebind = (
                    current_shared is not canonical or canonical.shape != cast.shape
                )
                if needs_rebind:
                    new_fei = cast.detach().clone()
                    for lora in loras:
                        lora._buffers["_fei"] = new_fei
                    self._shared_fei[dim] = new_fei
                else:
                    canonical.copy_(cast)

        # Global router: fires WITH grad so L_denoise reaches the
        # GlobalRouter params (set_routing_weights keeps the live grad_fn).
        if global_fei_router is not None:
            gates = global_fei_router(fei)
            self.set_routing_weights(gates)

    def clear_fei(self) -> None:
        """Reset cached FEI to zeros without rebinding pointers (same
        in-place-zero pattern as ``clear_sigma``)."""
        if not getattr(self, "_fei_aware_loras", None):
            return
        for dim, loras in self._fei_aware_loras_by_dim.items():
            canonical = loras[0]._buffers["_fei"]
            current_shared = self._shared_fei.get(dim)
            if current_shared is not canonical:
                for lora in loras:
                    lora._buffers["_fei"] = canonical
                self._shared_fei[dim] = canonical
            canonical.zero_()

    def _broadcast_gate(
        self,
        weights: torch.Tensor,
        aware_attr: str,
        buffer_name: str,
        shared_attr: str,
    ) -> None:
        """Slot-assign a ``(B, E)`` gate tensor to every module's ``buffer_name``.

        Assigns the SAME live ``weights`` reference (NO detach, NO copy_) so the
        buffer carries the router's grad_fn — that autograd path is what trains
        the router. cudagraph pointer stability is deliberately traded away
        here: gates are a tiny tensor and the gradient path is the point.
        """
        loras = getattr(self, aware_attr, None)
        if not loras:
            return
        canonical_buf = loras[0]._buffers[buffer_name]
        w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        for lora in loras:
            setattr(lora, buffer_name, w)  # buffer slot reassign, grad_fn kept
        setattr(self, shared_attr, w)

    def _reset_gate(self, aware_attr: str, buffer_name: str, shared_attr: str) -> None:
        """Reset a broadcast gate buffer to uniform ``1/E`` in place.

        Pointer stays stable for cudagraph capture; re-aliases if
        ``Module._apply`` (``.to(device)``) broke the shared link.
        """
        loras = getattr(self, aware_attr, None)
        if not loras:
            return
        canonical = loras[0]._buffers[buffer_name]
        if getattr(self, shared_attr) is not canonical:
            for lora in loras:
                lora._buffers[buffer_name] = canonical
            setattr(self, shared_attr, canonical)
        E = int(canonical.shape[-1])
        canonical.fill_(1.0 / max(E, 1))

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast a ``(B, E)`` gate tensor to every routing-aware module.

        Fired internally by ``set_fei`` (GlobalRouter, FEI source) or externally
        by inference callers pushing pre-computed gates. See ``_broadcast_gate``.
        """
        self._broadcast_gate(
            weights,
            "_routing_aware_loras",
            "_routing_weights",
            "_shared_routing_weights",
        )

    def clear_routing_weights(self) -> None:
        """Reset GlobalRouter gates to uniform ``1/E`` (between steps / teardown)."""
        self._reset_gate(
            "_routing_aware_loras", "_routing_weights", "_shared_routing_weights"
        )

    def set_crossattn_routing(self, crossattn_emb: torch.Tensor) -> None:
        """Fire the network-level GlobalRouter on a pooled text vector.

        Used when ``cfg.router_source="crossattn_emb"``. ``crossattn_emb`` is
        ``(B, L, D)`` (raw, pooled here) or ``(B, D)`` (pre-pooled). Runs WITH
        grad, broadcasts via :meth:`set_routing_weights`. Call BEFORE each
        forward, separately for cond/uncond branches at inference — gates
        depend on the caption.
        """
        if self.global_router is None or not getattr(
            self, "use_crossattn_router", False
        ):
            return
        gates = self.global_router(crossattn_emb)
        self.set_routing_weights(gates)

    def clear_step_caches(self) -> None:
        """Drop per-step tensor references (``_last_gate``) and invalidate
        memoized router-stats caches between training steps.

        Called unconditionally before each forward. GOTCHA (1): under
        ``torch.compile(mode='reduce-overhead')`` ``_last_gate`` lives in the
        inductor cudagraph memory pool — holding a Python reference across the
        step boundary blocks ``cudagraph_trees`` reclamation and silently
        demotes the run to eager. Call must precede ``cudagraph_mark_step_begin()``.
        (2) the router-stats caches must be invalidated each step or TB shows
        frozen usage/entropy values.

        ``_sigma`` is intentionally *not* cleared here: it's rebound by
        ``set_sigma`` every forward from outside the compiled region, and
        staying a Tensor at all times is what lets ``_compute_gate`` drop its
        None-vs-Tensor guard under ``torch.compile``.
        """
        self._last_sigma = None
        self._router_stats_cache = None
        for lora in self.unet_loras + self.text_encoder_loras:
            if hasattr(lora, "_last_gate"):
                lora._last_gate = None
        # Drop GlobalRouter per-step transients for the same
        # cudagraph-pool-reclamation reason.
        if self.global_router is not None:
            self.global_router._last_gates = None
            self.global_router._last_input = None
            self.global_router._last_fei = None

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

        # GOTCHA: save_network_weights relays adaln keys to the ComfyUI layout
        # (adaln_up_{br} → adaln_modulation_{br}_2); a resume/init load must
        # rename them back or every adaln module lands in missing_keys and
        # silently trains from scratch. Mirrors factory.py. See adaln.md.
        from networks.lora_utils import (
            has_comfy_adaln_keys,
            relayout_adaln_comfy_to_runtime,
        )

        if has_comfy_adaln_keys(weights_sd):
            weights_sd = relayout_adaln_comfy_to_runtime(weights_sd)

        # Stack per-expert hydra ups into fused lora_up_weight.
        weights_sd = _stack_lora_ups(weights_sd)
        # Hydra attn before the regular refuser: hydra splits carry no
        # lora_up.weight, so non-hydra attention still goes through cleanly.
        weights_sd = _refuse_split_hydra_keys(weights_sd)
        # Refuse unfused attn projections (inverse of save_weights defusing).
        weights_sd = _refuse_unfused_attn_lora_keys(weights_sd)

        self._reabsorb_baked_inv_scale(weights_sd)

        info = self.load_state_dict(weights_sd, False)
        return info

    def _reabsorb_baked_inv_scale(self, weights_sd: Dict[str, torch.Tensor]) -> None:
        """Resume guard for baked (inv_scale-folded) checkpoints.

        ``save_network_weights`` bakes ``inv_scale`` into ``lora_down`` and
        drops the key, so a baked checkpoint has no ``inv_scale``. GOTCHA: on
        resume, a freshly-built module has its own ``inv_scale`` buffer and an
        init ``down`` baked with ``s_norm`` — loading the raw ``down`` straight
        over it would leave the forward applying ``1/s_norm`` with nothing
        absorbing it. Re-absorb: move the incoming ``down`` back into training
        space (``down *= s_norm``) and re-inject ``inv_scale``.

        No-op for inference (no channel scaling) and legacy checkpoints that
        still carry ``inv_scale``.
        """
        for lora in self.unet_loras + self.text_encoder_loras:
            if not getattr(lora, "_has_channel_scale", False):
                continue
            name = lora.lora_name
            down_key = f"{name}.lora_down.weight"
            if f"{name}.inv_scale" in weights_sd or down_key not in weights_sd:
                continue
            inv_scale = lora.inv_scale  # (in,) fp32, == 1/s_norm
            down = weights_sd[down_key]
            s_norm = (
                inv_scale.to(device=down.device, dtype=torch.float)
                .clamp_min(1e-12)
                .reciprocal()
            )
            weights_sd[down_key] = (down.to(torch.float) * s_norm.unsqueeze(0)).to(
                down.dtype
            )
            weights_sd[f"{name}.inv_scale"] = inv_scale.clone()

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(
                f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules"
            )
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

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

        # Pre-group keys by module prefix (avoid O(modules*keys) scan); keys are
        # "{module_name}.{param}" with module_name dot-free.
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
            router_lr_mult = router_scale

            def _is_router_param(pname: str) -> bool:
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
                        param_data["lr"] = reg_lr * router_lr_mult
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
                        param_data["lr"] = lr * router_lr_mult
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

        # (HydraLoRA per-module routers are HydraLoRAModule submodules, already
        # captured by the unet_loras group above.)

        # GlobalRouter (route_per_layer=False) lives on the network, not on
        # per-Linear modules, so assemble_params misses it — add it explicitly
        # at unet_lr × router_lr_scale.
        if getattr(self, "global_router", None) is not None:
            gr_params = list(self.global_router.parameters())
            if len(gr_params) > 0:
                router_scale = float(self.cfg.router_lr_scale)
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("GlobalRouter: no base LR, skipping param group")
                else:
                    gr_lr = float(base_lr) * router_scale
                    all_params.append({"params": gr_params, "lr": gr_lr})
                    lr_descriptions.append("global router")
                    logger.info(
                        f"GlobalRouter param group: lr={gr_lr:.2e} "
                        f"({router_scale}x of unet_lr={base_lr})"
                    )

        # REPA v2 projection-head param group (absolute mode only). LR =
        # repa_lr_scale × unet_lr. Training-only — stripped by lora_save.
        if getattr(self, "repa_head", None) is not None:
            rh_params = list(self.repa_head.parameters())
            if len(rh_params) > 0:
                repa_scale = float(getattr(self, "_repa_lr_scale", 1.0))
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("REPA head: no base LR, skipping param group")
                else:
                    rh_lr = float(base_lr) * repa_scale
                    all_params.append({"params": rh_params, "lr": rh_lr})
                    lr_descriptions.append("repa head")
                    logger.info(
                        f"REPA head param group: lr={rh_lr:.2e} "
                        f"({repa_scale}x repa_lr_scale of unet_lr={base_lr})"
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
            # Adapters are depth-specific: module names carry the block index, so
            # a 40-block (Anima-2.9B) adapter merged onto the 28-block base drops
            # its tail blocks with only a "not all LoRA keys are used" warning.
            # Stamp the depth so the mismatch is machine-detectable.
            num_blocks = getattr(self, "_trained_num_blocks", 0)
            if num_blocks:
                metadata["ss_num_blocks"] = str(num_blocks)

        # Hard σ-band partition lives in non-persistent buffers; nothing
        # survives the state_dict write, so stamp the scalars the loader
        # needs to re-register it (only when on, for byte-identical non-band
        # checkpoints).
        if self.cfg.specialize_experts_by_sigma_buckets:
            metadata["ss_specialize_experts_by_sigma_buckets"] = "true"
            metadata["ss_num_sigma_buckets"] = str(int(self.cfg.num_sigma_buckets))
            if self.cfg.sigma_bucket_boundaries is not None:
                import json as _json

                metadata["ss_sigma_bucket_boundaries"] = _json.dumps(
                    list(self.cfg.sigma_bucket_boundaries)
                )

        # Three-axis routing config (`lora-routing` skill). Stamped every
        # save so the loader reconstructs the router layout without key-sniffing.
        if self.cfg.use_moe_style is not False:
            metadata["ss_use_moe_style"] = str(self.cfg.use_moe_style)
            metadata["ss_route_per_layer"] = (
                "true" if self.cfg.route_per_layer else "false"
            )
            metadata["ss_router_source"] = str(self.cfg.router_source)

        # Informational — which lora_down seed this plain LoRA got. The slice is
        # what a merge tool would compare: two adapters on the same weight_svd
        # slice share an input subspace, different slices are orthogonal.
        if getattr(self.cfg, "down_init", "kaiming") != "kaiming":
            metadata["ss_down_init"] = str(self.cfg.down_init)
            if self.cfg.down_init == "weight_svd":
                metadata["ss_svd_slice"] = str(int(getattr(self.cfg, "svd_slice", 0)))

        # Scalars the loader needs to size the FEI router input.
        if self.cfg.router_source == "fei" and self.cfg.fei_feature_dim > 0:
            metadata["ss_fei_feature_dim"] = str(int(self.cfg.fei_feature_dim))
            metadata["ss_fei_sigma_low_div"] = str(float(self.cfg.fei_sigma_low_div))

        state_dict = self.state_dict()
        # Training-only submodules (e.g. the REPA head) never belong in the
        # inference artifact; attach-side registers its prefix here.
        for prefix in getattr(self, "_training_only_prefixes", ()):
            for key in [k for k in state_dict if k.startswith(prefix)]:
                del state_dict[key]
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
