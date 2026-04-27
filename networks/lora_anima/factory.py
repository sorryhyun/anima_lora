# Build a LoRANetwork from either create_network kwargs (fresh training) or
# from the tensors in a saved checkpoint (warm-start / inference). Centralises
# all the kwarg parsing and checkpoint key-sniffing that used to live in
# lora_anima.py alongside LoRANetwork itself.

import logging
import os
import re
from typing import Dict, List, Optional

import torch

from library.log import setup_logging
from networks import NETWORK_REGISTRY, resolve_network_spec
from networks.condition_shift import ConditionShift
from networks.lora_anima.config import LoRANetworkCfg
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


def _load_channel_scales(
    kwargs: Dict[str, object],
) -> Optional[Dict[str, torch.Tensor]]:
    """Load per-channel input pre-scaling stats from disk, if requested.

    SmoothQuant-style. Requires a calibration file produced by
    ``archive/bench/analyze_lora_input_channels.py --dump_channel_stats <path>``.
    See ``archive/bench/channel_dominance_analysis.md`` for motivation.
    """
    per_channel_scaling = kwargs.get("per_channel_scaling", "false")
    if per_channel_scaling is not None:
        per_channel_scaling = str(per_channel_scaling).lower() == "true"
    if not per_channel_scaling:
        return None

    channel_stats_path = kwargs.get("channel_stats_path", None)
    channel_scaling_alpha = kwargs.get("channel_scaling_alpha", None)
    channel_scaling_alpha = (
        float(channel_scaling_alpha) if channel_scaling_alpha is not None else 0.5
    )

    if not channel_stats_path:
        raise ValueError(
            "per_channel_scaling=true requires channel_stats_path. Generate one with:\n"
            "  python archive/bench/analyze_lora_input_channels.py --dump_channel_stats <path.safetensors>"
        )
    if not os.path.isfile(channel_stats_path):
        raise FileNotFoundError(
            f"channel_stats_path does not exist: {channel_stats_path}"
        )
    from safetensors.torch import load_file as _load_channel_stats_file

    raw_stats = _load_channel_stats_file(channel_stats_path)
    out: Dict[str, torch.Tensor] = {}
    for _lora_name, _mean_abs in raw_stats.items():
        _s = _mean_abs.float().clamp_min(1e-6).pow(channel_scaling_alpha)
        _s = _s / _s.mean().clamp_min(1e-12)
        out[_lora_name] = _s
    logger.info(
        f"Per-channel input pre-scaling: alpha={channel_scaling_alpha}, "
        f"stats={channel_stats_path} ({len(out)} calibrated modules)"
    )
    return out


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
    spec = resolve_network_spec(kwargs)

    # Memory-saving down-projection autograd (classic LoRA only). Saves the
    # low-precision x instead of the fp32-cast input; fp32 bottleneck matmul
    # and gradients are preserved bitwise. See `networks/lora_modules/custom_autograd.py`.
    use_custom_down_autograd = kwargs.get("use_custom_down_autograd", "false")
    if isinstance(use_custom_down_autograd, str):
        use_custom_down_autograd = use_custom_down_autograd.lower() == "true"
    else:
        use_custom_down_autograd = bool(use_custom_down_autograd)

    channel_scales_dict = _load_channel_scales(kwargs)

    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=network_dim,
        network_alpha=network_alpha,
        neuron_dropout=neuron_dropout,
        module_class=spec.module_class,
        channel_scales_dict=channel_scales_dict,
    )

    if cfg.router_lr_scale != 1.0:
        logger.info(
            f"HydraLoRA router LR scale: {cfg.router_lr_scale}x unet_lr (applies to .router.* params — σ features live in router.weight columns)"
        )

    network = LoRANetwork(text_encoders, unet, cfg, multiplier=multiplier)

    # Variant-specific defaults — overridden by spec.post_init for the matching variant.
    network._use_hydra = False
    network._balance_loss_weight = 0.0

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

    if cfg.use_timestep_mask:
        logger.info(
            f"Timestep-dependent rank masking: min_rank={cfg.min_rank}, alpha={cfg.alpha_rank_scale}"
        )
    if cfg.use_sigma_router and network._sigma_router_hits > 0:
        logger.info(
            f"σ-conditional HydraLoRA router: {network._sigma_router_hits} modules "
            f"with sinusoidal(σ) concatenated to router input (feat={cfg.sigma_feature_dim}), "
            f"per-bucket balance w={cfg.per_bucket_balance_weight}, buckets={cfg.num_sigma_buckets}"
        )
    elif cfg.use_sigma_router:
        logger.warning(
            "use_sigma_router=true but no modules matched sigma_router_layers "
            f"regex {cfg.sigma_router_layers!r} — σ-routing is inactive"
        )
    if spec.name == "ortho_hydra":
        logger.info(
            f"OrthoHydraLoRA: Cayley + MoE, num_experts={cfg.num_experts}, "
            f"balance_loss_weight={network._balance_loss_weight}"
        )
    elif spec.name == "ortho":
        logger.info("OrthoLoRA: Cayley parameterization + SVD-informed init")
    elif spec.name == "hydra":
        logger.info(
            f"HydraLoRA: num_experts={cfg.num_experts}, balance_loss_weight={network._balance_loss_weight}"
        )
    if spec.name in ("hydra", "ortho_hydra") and (
        network._hydra_router_re is not None or network._hydra_router_names is not None
    ):
        fallback_name = "OrthoLoRAExp" if spec.name == "ortho_hydra" else "LoRA"
        logger.info(
            f"HydraLoRA layer filter: {network._hydra_router_hits} MoE modules, "
            f"{network._hydra_router_misses} fell back to plain {fallback_name} "
            f"(regex={cfg.hydra_router_layers!r})"
        )
        if network._hydra_router_hits == 0:
            logger.warning(
                "hydra_router_layers regex matched zero modules — no MoE routing "
                "is active, every target became plain LoRA."
            )
    if cfg.add_reft:
        _reft_alpha_str = (
            f"{cfg.reft_alpha}"
            if cfg.reft_alpha is not None
            else f"{cfg.alpha} (from network_alpha)"
        )
        logger.info(
            f"ReFT: reft_dim={cfg.reft_dim}, reft_alpha={_reft_alpha_str}, "
            f"layers={cfg.reft_layers!r}"
        )
    if cfg.layer_start is not None or cfg.layer_end is not None:
        logger.info(
            f"Layer range: training blocks [{cfg.layer_start or 0}, {cfg.layer_end or '...'})"
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
        spec = NETWORK_REGISTRY[
            "lora"
        ]  # inference uses the merge-capable LoRAInfModule
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

    # Per-module Hydra selection from the checkpoint: if the file contains
    # *both* hydra-style and plain-LoRA-style leaves, we're reloading a mixed
    # hydra_router_layers result and need to build each leaf with its original
    # class. If every module is hydra, leave as None (= apply the nominal
    # hydra class everywhere, legacy behaviour).
    hydra_router_names = (
        sorted(hydra_module_names)
        if (
            (has_hydra or has_ortho_hydra) and plain_module_names and hydra_module_names
        )
        else None
    )

    cfg = LoRANetworkCfg.from_weights(
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        has_reft=has_reft,
        reft_dim=reft_dim,
        reft_block_indices=reft_block_indices,
        is_hydra_or_ortho_hydra=has_hydra or has_ortho_hydra,
        hydra_num_experts=hydra_num_experts,
        sigma_feature_dim_detected=sigma_feature_dim_detected,
        sigma_router_names=sigma_router_names or None,
        hydra_router_names=hydra_router_names,
        channel_scales_dict=channel_scales_dict,
    )

    network = LoRANetwork(text_encoders, unet, cfg, multiplier=multiplier)
    # Mirror create_network's variant-specific post-build attribute attachment.
    # Defaults first, then spec.post_init overrides for the matching variant.
    network._use_hydra = False
    network._balance_loss_weight = 0.0
    network._network_spec = spec
    if spec.post_init is not None:
        spec.post_init(network, kwargs)

    _maybe_attach_apex_shift(network, kwargs)

    return network, weights_sd
