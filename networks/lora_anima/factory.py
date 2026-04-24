# Build a LoRANetwork from either create_network kwargs (fresh training) or
# from the tensors in a saved checkpoint (warm-start / inference). Centralises
# all the kwarg parsing and checkpoint key-sniffing that used to live in
# lora_anima.py alongside LoRANetwork itself.

import ast
import logging
import os
import re
from typing import Dict, List, Optional

import torch

from library.log import setup_logging
from networks import NETWORK_REGISTRY, resolve_network_spec
from networks.condition_shift import ConditionShift
from networks.lora_anima.loading import (
    _refuse_split_hydra_keys,
    _refuse_unfused_attn_lora_keys,
    _stack_lora_ups,
)
from networks.lora_anima.network import LoRANetwork
from networks.lora_modules import LoRAInfModule

setup_logging()
logger = logging.getLogger(__name__)


def _maybe_attach_apex_shift(network: "LoRANetwork", kwargs: Dict[str, object]) -> None:
    """Attach ConditionShift (APEX, arXiv:2604.12322) if enabled via kwargs.

    Called from both create_network (fresh) and create_network_from_weights
    (warm start / dim_from_weights). Kwargs come from train.py's net_kwargs
    which stringifies everything, so values are parsed defensively.
    """
    mode = kwargs.get("apex_condition_shift_mode", None)
    if mode is None or str(mode).lower() in ("", "none"):
        return
    init_a = float(kwargs.get("apex_condition_shift_init_a", -1.0))
    init_b = float(kwargs.get("apex_condition_shift_init_b", 0.5))
    shift_lr_scale = float(kwargs.get("apex_shift_lr_scale", 0.1))
    shift_dim = int(kwargs.get("apex_condition_shift_dim", 1024))
    cs = ConditionShift(
        dim=shift_dim,
        mode=str(mode),
        init_a=init_a,
        init_b=init_b,
    )
    network.apex_condition_shift = cs
    network._apex_shift_lr_scale = shift_lr_scale
    logger.info(
        f"APEX ConditionShift attached: mode={mode} "
        f"(a={init_a}, b={init_b}), lr_scale={shift_lr_scale}"
    )


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    train_llm_adapter = kwargs.get("train_llm_adapter", "false")
    if train_llm_adapter is not None:
        train_llm_adapter = True if train_llm_adapter.lower() == "true" else False

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        try:
            exclude_patterns = ast.literal_eval(exclude_patterns)
            if not isinstance(exclude_patterns, list):
                exclude_patterns = [exclude_patterns]
        except (ValueError, SyntaxError):
            exclude_patterns = [exclude_patterns]

    # layer range filtering (e.g., layer_start=4 layer_end=28 to train only blocks 4-27)
    layer_start = kwargs.get("layer_start", None)
    layer_end = kwargs.get("layer_end", None)
    if layer_start is not None:
        layer_start = int(layer_start)
    if layer_end is not None:
        layer_end = int(layer_end)

    exclude_patterns.append(
        r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|pooled_text_proj).*"
    )

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        try:
            include_patterns = ast.literal_eval(include_patterns)
            if not isinstance(include_patterns, list):
                include_patterns = [include_patterns]
        except (ValueError, SyntaxError):
            include_patterns = [include_patterns]

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    use_dora = kwargs.get("use_dora", "false")
    if use_dora is not None:
        use_dora = True if use_dora.lower() == "true" else False

    use_ortho = kwargs.get("use_ortho", "false")
    if use_ortho is not None:
        use_ortho = True if use_ortho.lower() == "true" else False

    use_timestep_mask = kwargs.get("use_timestep_mask", "false")
    if use_timestep_mask is not None:
        use_timestep_mask = True if use_timestep_mask.lower() == "true" else False
    min_rank = kwargs.get("min_rank", None)
    min_rank = int(min_rank) if min_rank is not None else 1
    alpha_rank_scale = kwargs.get("alpha_rank_scale", None)
    alpha_rank_scale = float(alpha_rank_scale) if alpha_rank_scale is not None else 1.0

    add_reft = kwargs.get("add_reft", "false")
    if add_reft is not None:
        add_reft = True if add_reft.lower() == "true" else False
    reft_dim = kwargs.get("reft_dim", None)
    reft_dim = int(reft_dim) if reft_dim is not None else network_dim
    reft_alpha = kwargs.get("reft_alpha", None)
    reft_alpha = float(reft_alpha) if reft_alpha is not None else None
    # reft_layers: which DiT blocks get a ReFT intervention. See _parse_reft_layers.
    reft_layers = kwargs.get("reft_layers", "all")

    use_hydra = kwargs.get("use_hydra", "false")
    if use_hydra is not None:
        use_hydra = True if use_hydra.lower() == "true" else False
    num_experts = kwargs.get("num_experts", None)
    num_experts = int(num_experts) if num_experts is not None else 4
    balance_loss_weight = kwargs.get("balance_loss_weight", None)
    balance_loss_weight = (
        float(balance_loss_weight) if balance_loss_weight is not None else 0.01
    )
    # Fraction of training steps during which only one randomly-chosen expert
    # per module receives gradient (forward still uses all experts via the
    # learned gate, so each expert learns in the full MoE context). 0.0 =
    # off; 0.1 = warmup for the first 10% of steps. This is the per-expert
    # symmetry-breaker: zero-init experts (plain Hydra `lora_up_weight`,
    # OrthoHydra-narrow `S_p`) start identical, and warmup forces each to
    # accumulate distinct gradients before the router can collapse onto a
    # subset. Disjoint OrthoHydra bases break symmetry structurally and need
    # no warmup, but warmup still helps the router avoid early starvation.
    expert_warmup_ratio = float(kwargs.get("expert_warmup_ratio", 0.0))

    # σ-conditional HydraLoRA router (Track B, docs: timestep-hydra.md).
    # When on, each matching HydraLoRAModule gets a tiny sinusoidal(σ)→E MLP
    # that adds to the gate logits. Identity to base router at init (zero-init
    # on final layer); σ-dependence only emerges if gradients push it.
    use_sigma_router = kwargs.get("use_sigma_router", "false")
    if use_sigma_router is not None:
        use_sigma_router = str(use_sigma_router).lower() == "true"
    # 16 = 8 cos/sin freq pairs — over-resolves σ on [0,1]. Must be even.
    sigma_feature_dim = int(kwargs.get("sigma_feature_dim", 16))
    sigma_hidden_dim = int(kwargs.get("sigma_hidden_dim", 128))
    # Regex applied to each candidate module's `original_name`. Default picks
    # the layer types the B0 pre-analysis showed carry σ-correlation signal
    # (cross_attn.q, self_attn.qkv); MLPs are skipped since their σ-JS is flat.
    sigma_router_layers = kwargs.get(
        "sigma_router_layers", r".*(cross_attn\.q_proj|self_attn\.qkv_proj)$"
    )
    # Regex applied to each candidate module's `original_name` to decide which
    # target Linears actually get the HydraLoRA MoE wrapper. Non-matching
    # modules fall back to plain LoRA (or OrthoLoRAExp, if use_ortho=true).
    # None = apply MoE to every target (backward-compat, wasteful). A narrow
    # regex concentrates the ~256 default routers onto the layers where
    # semantic specialization is actually learnable (cross-attn, MLP).
    hydra_router_layers = kwargs.get("hydra_router_layers", None)
    per_bucket_balance_weight = kwargs.get("per_bucket_balance_weight", None)
    per_bucket_balance_weight = (
        float(per_bucket_balance_weight)
        if per_bucket_balance_weight is not None
        else 0.3
    )
    num_sigma_buckets = int(kwargs.get("num_sigma_buckets", 3))

    # Per-channel input pre-scaling (SmoothQuant-style). Requires a calibration file
    # produced by `bench/analyze_lora_input_channels.py --dump_channel_stats <path>`.
    # See `bench/channel_dominance_analysis.md` for motivation.
    per_channel_scaling = kwargs.get("per_channel_scaling", "false")
    if per_channel_scaling is not None:
        per_channel_scaling = per_channel_scaling.lower() == "true"

    # Memory-saving down-projection autograd (classic LoRA only). Saves the
    # low-precision x instead of the fp32-cast input; fp32 bottleneck matmul
    # and gradients are preserved bitwise. See `networks/lora_modules/custom_autograd.py`.
    use_custom_down_autograd = kwargs.get("use_custom_down_autograd", "false")
    if isinstance(use_custom_down_autograd, str):
        use_custom_down_autograd = use_custom_down_autograd.lower() == "true"
    else:
        use_custom_down_autograd = bool(use_custom_down_autograd)
    channel_stats_path = kwargs.get("channel_stats_path", None)
    channel_scaling_alpha = kwargs.get("channel_scaling_alpha", None)
    channel_scaling_alpha = (
        float(channel_scaling_alpha) if channel_scaling_alpha is not None else 0.5
    )

    channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None
    if per_channel_scaling:
        if not channel_stats_path:
            raise ValueError(
                "per_channel_scaling=true requires channel_stats_path. Generate one with:\n"
                "  python bench/analyze_lora_input_channels.py --dump_channel_stats <path.safetensors>"
            )
        if not os.path.isfile(channel_stats_path):
            raise FileNotFoundError(
                f"channel_stats_path does not exist: {channel_stats_path}"
            )
        from safetensors.torch import load_file as _load_channel_stats_file

        raw_stats = _load_channel_stats_file(channel_stats_path)
        channel_scales_dict = {}
        for _lora_name, _mean_abs in raw_stats.items():
            _s = _mean_abs.float().clamp_min(1e-6).pow(channel_scaling_alpha)
            _s = _s / _s.mean().clamp_min(1e-12)
            channel_scales_dict[_lora_name] = _s
        logger.info(
            f"Per-channel input pre-scaling: alpha={channel_scaling_alpha}, "
            f"stats={channel_stats_path} ({len(channel_scales_dict)} calibrated modules)"
        )

    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if verbose.lower() == "true" else False

    def parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, float]:
        """
        Parse a string of key-value pairs separated by commas.
        """
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                logger.warning(f"Invalid format: {pair}, expected 'key=value'")
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                pairs[key] = int(value) if is_int else float(value)
            except ValueError:
                logger.warning(f"Invalid value for {key}: {value}")
        return pairs

    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    if network_reg_lrs is not None:
        reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False)
    else:
        reg_lrs = None

    router_lr_scale = kwargs.get("network_router_lr_scale", None)
    router_lr_scale = float(router_lr_scale) if router_lr_scale is not None else 1.0
    if router_lr_scale != 1.0:
        logger.info(
            f"HydraLoRA router LR scale: {router_lr_scale}x unet_lr (applies to .router.* params — σ features live in router.weight columns)"
        )

    network_reg_dims = kwargs.get("network_reg_dims", None)
    if network_reg_dims is not None:
        reg_dims = parse_kv_pairs(network_reg_dims, is_int=True)
    else:
        reg_dims = None

    spec = resolve_network_spec(kwargs)
    module_class = spec.module_class

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
        layer_start=layer_start,
        layer_end=layer_end,
        add_reft=add_reft,
        reft_dim=reft_dim,
        reft_alpha=reft_alpha,
        reft_layers=reft_layers,
        num_experts=num_experts,
        channel_scales_dict=channel_scales_dict,
        use_sigma_router=use_sigma_router,
        sigma_feature_dim=sigma_feature_dim,
        sigma_hidden_dim=sigma_hidden_dim,
        sigma_router_layers=sigma_router_layers,
        hydra_router_layers=hydra_router_layers,
        expert_warmup_ratio=expert_warmup_ratio,
        router_lr_scale=router_lr_scale,
    )

    # Set timestep mask config (variant-agnostic)
    network._use_timestep_mask = use_timestep_mask
    network._min_rank = min_rank
    network._max_rank = network_dim
    network._alpha_rank_scale = alpha_rank_scale
    network._add_reft = add_reft
    network._reft_dim = reft_dim
    network._reft_alpha = reft_alpha
    network._reft_layers = reft_layers

    # Variant-specific defaults — overridden by spec.post_init for the matching variant.
    network._use_hydra = False
    network._balance_loss_weight = 0.0
    network._per_bucket_balance_weight = per_bucket_balance_weight
    network._num_sigma_buckets = num_sigma_buckets

    # Stamp the resolved spec; save_weights keys off this to pick the save pipeline.
    network._network_spec = spec
    if spec.post_init is not None:
        spec.post_init(network, kwargs)

    _maybe_attach_apex_shift(network, kwargs)

    if use_custom_down_autograd:
        _hits = 0
        _skipped = 0
        for mod in network.text_encoder_loras + network.unet_loras:
            if hasattr(mod, "use_custom_down_autograd"):
                mod.use_custom_down_autograd = True
                _hits += 1
            else:
                _skipped += 1
        logger.info(
            f"use_custom_down_autograd: enabled on {_hits} LoRA-family modules"
            + (f" ({_skipped} unsupported skipped)" if _skipped else "")
            + " (saves ~32-128 MiB/Linear of fp32 activation per step)"
        )

    if use_timestep_mask:
        logger.info(
            f"Timestep-dependent rank masking: min_rank={min_rank}, alpha={alpha_rank_scale}"
        )
    if use_sigma_router and network._sigma_router_hits > 0:
        logger.info(
            f"σ-conditional HydraLoRA router: {network._sigma_router_hits} modules "
            f"with sinusoidal(σ) concatenated to router input (feat={sigma_feature_dim}), "
            f"per-bucket balance w={per_bucket_balance_weight}, buckets={num_sigma_buckets}"
        )
    elif use_sigma_router:
        logger.warning(
            "use_sigma_router=true but no modules matched sigma_router_layers "
            f"regex {sigma_router_layers!r} — σ-routing is inactive"
        )
    if spec.name == "ortho_hydra":
        logger.info(
            f"OrthoHydraLoRA: Cayley + MoE, num_experts={num_experts}, "
            f"balance_loss_weight={network._balance_loss_weight}"
        )
    elif spec.name == "ortho":
        logger.info("OrthoLoRA: Cayley parameterization + SVD-informed init")
    elif spec.name == "hydra":
        logger.info(
            f"HydraLoRA: num_experts={num_experts}, balance_loss_weight={network._balance_loss_weight}"
        )
    if spec.name in ("hydra", "ortho_hydra") and (
        network._hydra_router_re is not None or network._hydra_router_names is not None
    ):
        fallback_name = (
            "OrthoLoRAExp" if spec.name == "ortho_hydra" else "LoRA"
        )
        logger.info(
            f"HydraLoRA layer filter: {network._hydra_router_hits} MoE modules, "
            f"{network._hydra_router_misses} fell back to plain {fallback_name} "
            f"(regex={hydra_router_layers!r})"
        )
        if network._hydra_router_hits == 0:
            logger.warning(
                "hydra_router_layers regex matched zero modules — no MoE routing "
                "is active, every target became plain LoRA."
            )
    if add_reft:
        _reft_alpha_str = (
            f"{reft_alpha}" if reft_alpha is not None else f"{network_alpha} (from network_alpha)"
        )
        logger.info(
            f"ReFT: reft_dim={reft_dim}, reft_alpha={_reft_alpha_str}, "
            f"layers={reft_layers!r}"
        )
    if layer_start is not None or layer_end is not None:
        logger.info(
            f"Layer range: training blocks [{layer_start or 0}, {layer_end or '...'})"
        )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    loraplus_unet_lr_ratio = (
        float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    )
    loraplus_text_encoder_lr_ratio = (
        float(loraplus_text_encoder_lr_ratio)
        if loraplus_text_encoder_lr_ratio is not None
        else None
    )
    if (
        loraplus_lr_ratio is not None
        or loraplus_unet_lr_ratio is not None
        or loraplus_text_encoder_lr_ratio is not None
    ):
        network.set_loraplus_lr_ratio(
            loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
        )

    return network


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # Strip torch.compile '_orig_mod_' from old checkpoint keys
    weights_sd = LoRANetwork._strip_orig_mod_keys(weights_sd)

    # HydraLoRA moe files: stack per-expert ups and fuse split q/k/v first so
    # the fused training-runtime keys are what the regular attention refuser
    # and the downstream detection loop see.
    weights_sd = _stack_lora_ups(weights_sd)
    weights_sd = _refuse_split_hydra_keys(weights_sd)
    # Refuse unfused attn projections so modules_dim reflects the runtime (qkv/kv fused).
    weights_sd = _refuse_unfused_attn_lora_keys(weights_sd)

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    has_dora = False
    has_ortho = False
    has_ortho_hydra = False
    has_hydra = False
    hydra_num_experts = 0
    has_reft = False
    reft_dim = None
    reft_block_indices: set[int] = set()
    # Per-module hydra flag: which lora_names were trained as MoE (Hydra) vs
    # plain LoRA / OrthoLoRAExp. Populated below by key sniff, then passed
    # through as `hydra_router_names` so create_modules can pick the right
    # class per module in mixed checkpoints (result of hydra_router_layers).
    hydra_module_names: set[str] = set()
    plain_module_names: set[str] = set()
    # Block-level ReFT key pattern: reft_unet_blocks_<idx>.<...>
    _reft_block_re = re.compile(r"^reft_unet_blocks_(\d+)$")
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]

        # Old-format global HydraLoRA router — incompatible with per-module routing.
        if key.startswith("_hydra_router"):
            raise RuntimeError(
                "This checkpoint uses the old global HydraLoRA router "
                "(_hydra_router.*). The router is now per-module and layer-local; "
                "the old format cannot be loaded. Retrain the LoRA to get the new "
                "per-module router weights."
            )

        # ReFT keys use "reft_" prefix (block-level: reft_unet_blocks_<idx>.*)
        if lora_name.startswith("reft_"):
            has_reft = True
            m = _reft_block_re.match(lora_name)
            if m is None:
                raise RuntimeError(
                    f"ReFT key {key!r} does not match the block-level scheme "
                    "'reft_unet_blocks_<idx>.*'. This checkpoint was likely trained "
                    "with the old per-Linear ReFT wiring and cannot be loaded by the "
                    "current block-level implementation."
                )
            reft_block_indices.add(int(m.group(1)))
            if "rotate_layer" in key and "weight" in key:
                reft_dim = value.size()[0]
            continue

        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_up_weight" in key:
            has_hydra = True
            hydra_num_experts = max(hydra_num_experts, value.size(0))
            hydra_module_names.add(lora_name)
        elif key.endswith(".lora_up.weight"):
            # Plain (non-stacked) LoRA up — either vanilla LoRA or the
            # plain-fallback leg of a mixed hydra_router_layers checkpoint.
            plain_module_names.add(lora_name)
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
        elif key.endswith(".S_p"):
            if value.dim() == 3:
                # OrthoHydraLoRA: S_p is (num_experts, r, r)
                has_ortho_hydra = True
                hydra_num_experts = max(hydra_num_experts, value.size(0))
                modules_dim[lora_name] = value.size(1)
                hydra_module_names.add(lora_name)
            else:
                # OrthoLoRA: S_p is (r, r) — either pure ortho or the
                # plain-fallback leg of a mixed ortho_hydra checkpoint.
                has_ortho = True
                modules_dim[lora_name] = value.size(0)
                plain_module_names.add(lora_name)
        elif "dora_scale" in key:
            has_dora = True

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    # has_hydra / has_ortho_hydra win over for_inference: the router is
    # sample-dependent and can't be folded into a LoRAInfModule static-merge path.
    # The dynamic forward-hook path works in eval mode too.
    if has_ortho_hydra:
        spec = NETWORK_REGISTRY["ortho_hydra"]
        module_class = spec.module_class
    elif has_hydra:
        spec = NETWORK_REGISTRY["hydra"]
        module_class = spec.module_class

    # Legacy σ-router refusal: the additive-bias sigma_mlp design is gone;
    # σ is now a direct input to a wider router (router.weight columns
    # [lora_dim:] take sinusoidal(σ) features). Old checkpoints carrying
    # .sigma_mlp.* can't be reshaped into the new form.
    _legacy_sigma_keys = [k for k in weights_sd if ".sigma_mlp." in k]
    if _legacy_sigma_keys:
        raise RuntimeError(
            f"Checkpoint contains {len(_legacy_sigma_keys)} legacy σ-router "
            f"keys (sigma_mlp.*). The σ-conditional router is now a direct "
            f"concat of sinusoidal(σ) into the router input; the old "
            f"additive-bias MLP path is unsupported. Retrain the LoRA to "
            f"produce the new router shape. First legacy key: "
            f"{_legacy_sigma_keys[0]!r}."
        )

    # Old-format per-module router — was Linear(in_dim, E). Current router is
    # Linear(lora_dim + sigma_feature_dim, E); width >= lora_dim, with any
    # excess = σ feature dim. The old broken shape (width ≈ in_dim, often
    # thousands) is caught by a sanity cap on excess width.
    sigma_feature_dim_detected: Optional[int] = None
    if has_hydra or has_ortho_hydra:
        _SIGMA_FEATURE_CAP = 1024
        for k, v in weights_sd.items():
            if not k.endswith(".router.weight"):
                continue
            lora_name = k[: -len(".router.weight")]
            expected_rank = modules_dim.get(lora_name)
            if expected_rank is None or v.ndim != 2:
                continue
            width = v.size(1)
            if width < expected_rank:
                raise RuntimeError(
                    f"router.weight at {k!r} has width {width} < expected "
                    f"rank {expected_rank}; checkpoint is malformed."
                )
            extra = width - expected_rank
            if extra == 0:
                continue
            if extra > _SIGMA_FEATURE_CAP:
                raise RuntimeError(
                    f"router.weight at {k!r} has shape {tuple(v.shape)}; "
                    f"expected rank {expected_rank} with optional σ features "
                    f"appended (≤ {_SIGMA_FEATURE_CAP}). The excess width "
                    f"{extra} is most likely an old-format router trained "
                    "on raw layer input (see docs/methods/hydra-lora.md "
                    "§Fixes). There is no salvage path — retrain the LoRA."
                )
            if sigma_feature_dim_detected is None:
                sigma_feature_dim_detected = extra
            elif sigma_feature_dim_detected != extra:
                raise RuntimeError(
                    f"Inconsistent σ-feature dims across modules: expected "
                    f"{sigma_feature_dim_detected}, found {extra} at {k!r}."
                )
    elif for_inference:
        spec = NETWORK_REGISTRY["lora"]  # inference uses the merge-capable LoRAInfModule
        module_class = LoRAInfModule
    elif has_dora:
        spec = NETWORK_REGISTRY["dora"]
        module_class = spec.module_class
    elif has_ortho:
        spec = NETWORK_REGISTRY["ortho"]
        module_class = spec.module_class
    else:
        spec = NETWORK_REGISTRY["lora"]
        module_class = spec.module_class

    # Detect baked-in per-channel input scaling. We pass a placeholder ones
    # tensor so each affected module registers the `inv_scale` buffer at init;
    # load_state_dict then overwrites it with the trained values. The absorption
    # step in _absorb_channel_scale is a no-op with s=ones, and the subsequent
    # weight load fully replaces the init values anyway.
    channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None
    _scale_keys = [k for k in weights_sd.keys() if k.endswith(".inv_scale")]
    if _scale_keys:
        channel_scales_dict = {}
        for _k in _scale_keys:
            _lora_name = _k.rsplit(".inv_scale", 1)[0]
            channel_scales_dict[_lora_name] = torch.ones_like(weights_sd[_k])
        logger.info(
            f"Detected per-channel input scaling in checkpoint: "
            f"{len(channel_scales_dict)} modules with baked-in inv_scale"
        )

    # σ-conditional router names: derived from router.weight widths above.
    # A module has σ routing iff its router.weight width > expected rank —
    # the excess columns are the sinusoidal(σ) feature slice. List is empty
    # when sigma_feature_dim_detected is None (no σ routing in this ckpt).
    sigma_router_names: List[str] = []
    if (has_hydra or has_ortho_hydra) and sigma_feature_dim_detected is not None:
        for k, v in weights_sd.items():
            if not k.endswith(".router.weight") or v.ndim != 2:
                continue
            lora_name = k[: -len(".router.weight")]
            expected_rank = modules_dim.get(lora_name)
            if expected_rank is None:
                continue
            if v.size(1) - expected_rank == sigma_feature_dim_detected:
                sigma_router_names.append(lora_name)

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        add_reft=has_reft,
        reft_dim=reft_dim if reft_dim is not None else 4,
        reft_layers=sorted(reft_block_indices) if has_reft else "all",
        num_experts=hydra_num_experts if (has_hydra or has_ortho_hydra) else 4,
        channel_scales_dict=channel_scales_dict,
        use_sigma_router=bool(sigma_router_names),
        sigma_feature_dim=sigma_feature_dim_detected or 128,
        sigma_hidden_dim=128,  # unused — kept for API compat with LoRANetwork
        sigma_router_names=sigma_router_names or None,
        # Per-module Hydra selection from the checkpoint: if the file contains
        # *both* hydra-style and plain-LoRA-style leaves, we're reloading a
        # mixed hydra_router_layers result and need to build each leaf with its
        # original class. If every module is hydra, leave as None (= apply the
        # nominal hydra class everywhere, legacy behaviour).
        hydra_router_names=(
            sorted(hydra_module_names)
            if (
                (has_hydra or has_ortho_hydra)
                and plain_module_names
                and hydra_module_names
            )
            else None
        ),
        # from-weights path is inference/eval; warmup is a train-time schedule.
        expert_warmup_ratio=0.0,
    )
    # Mirror create_network's variant-specific post-build attribute attachment.
    # Defaults first, then spec.post_init overrides for the matching variant.
    network._use_hydra = False
    network._balance_loss_weight = 0.0
    network._network_spec = spec
    if spec.post_init is not None:
        spec.post_init(network, kwargs)

    _maybe_attach_apex_shift(network, kwargs)

    return network, weights_sd
