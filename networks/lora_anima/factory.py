# Build a LoRANetwork from create_network kwargs (fresh training) or from a
# saved checkpoint's tensors (warm-start / inference). Centralises the kwarg
# parsing and checkpoint key-sniffing.

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from library.env import resolve_under_home
from library.log import setup_logging
from networks import NETWORK_REGISTRY, resolve_network_spec
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import (
    _refuse_split_chimera_keys,
    _refuse_split_hydra_keys,
    _refuse_split_stacked_experts_keys,
    _refuse_unfused_attn_lora_keys,
    _stack_chimera_lora_ups,
    _stack_lora_ups,
)
from networks.lora_anima.network import LoRANetwork

setup_logging()
logger = logging.getLogger(__name__)


# Vendored SmoothQuant-style calibration — ships in-tree (~3.5 MB) so deploys
# (including custom_nodes/*/_vendor/ trees) work without a separate download.
# Regenerate via `scripts/calibration/analyze_lora_input_channels.py`.
_CHANNEL_STATS_PATH = (
    Path(__file__).resolve().parent.parent / "calibration" / "channel_stats.safetensors"
)


def _load_channel_scales(
    kwargs: Dict[str, object],
) -> Optional[Dict[str, torch.Tensor]]:
    """Load per-channel input pre-scaling stats, gated on ``channel_scaling_alpha``.

    SmoothQuant-style; sole knob is 0.0 (off) .. 1.0 (fully flattened),
    base.toml ships 0.5. Inert on frozen-basis ortho variants — see
    docs/optimizations/channel_scaling.md.
    """
    raw_alpha = kwargs.get("channel_scaling_alpha", 0.0)
    channel_scaling_alpha = float(raw_alpha) if raw_alpha is not None else 0.0
    if channel_scaling_alpha == 0.0:
        return None

    if not _CHANNEL_STATS_PATH.is_file():
        raise FileNotFoundError(
            f"vendored channel stats missing at {_CHANNEL_STATS_PATH}. "
            f"Regenerate with:\n"
            f"  python scripts/calibration/analyze_lora_input_channels.py "
            f"--per_artist --dump_channel_stats {_CHANNEL_STATS_PATH}"
        )
    from safetensors.torch import load_file as _load_channel_stats_file

    raw_stats = _load_channel_stats_file(str(_CHANNEL_STATS_PATH))
    out: Dict[str, torch.Tensor] = {}
    for _lora_name, _mean_abs in raw_stats.items():
        _s = _mean_abs.float().clamp_min(1e-6).pow(channel_scaling_alpha)
        _s = _s / _s.mean().clamp_min(1e-12)
        out[_lora_name] = _s
    logger.info(
        f"channel_scaling: alpha={channel_scaling_alpha}, "
        f"stats={_CHANNEL_STATS_PATH.name} ({len(out)} calibrated modules)"
    )
    return out


def _load_grad_basis(
    kwargs: Dict[str, object],
    unet,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load the gradient-SVD ``lora_down`` basis, gated on ``down_init``.

    Both gradient modes read the same artifact: ``basis_file`` points at a
    shipped/universal basis, and ``grad_svd`` points at the per-run sketch
    ``train.py`` wrote just before the network was built. Depth is validated in
    ``grad_basis.load_basis`` — a basis is per checkpoint arch.
    """
    down_init = str(kwargs.get("down_init", "kaiming") or "kaiming")
    if down_init not in ("grad_svd", "basis_file"):
        return None

    basis_file = kwargs.get("grad_basis_file")
    if not basis_file:
        raise ValueError(
            f"down_init={down_init!r} needs network_args grad_basis_file=<path>. "
            "For 'grad_svd' that path is written by train.py's own sketch pass — "
            "seeing this means the network was built outside train.py."
        )
    from networks.grad_basis import dit_num_blocks, load_basis

    basis, meta = load_basis(
        resolve_under_home(str(basis_file)),
        num_blocks=dit_num_blocks(unet) if unet is not None else None,
    )
    logger.info(
        f"down_init={down_init}: gradient basis {Path(str(basis_file)).name} "
        f"({len(basis)} layers, rank={meta.get('rank', '?')})"
    )
    return basis


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

    # Deprecated 2026-06-10 (accepted so old snapshot TOMLs replay): the
    # fp32-bottleneck down-projection autograd was removed; GEMMs now run in
    # org_forwarded.dtype, bit-identical to the old path. See bench/lora_fp32_bottleneck.
    if str(kwargs.get("use_custom_down_autograd", "false")).strip().lower() in (
        "true",
        "1",
    ):
        logger.info(
            "use_custom_down_autograd is deprecated and ignored "
            "(fp32-bottleneck path removed; compute-dtype GEMMs are "
            "bit-identical under the trainer's autocast)"
        )

    channel_scales_dict = _load_channel_scales(kwargs)
    grad_basis_dict = _load_grad_basis(kwargs, unet)

    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=network_dim,
        network_alpha=network_alpha,
        neuron_dropout=neuron_dropout,
        module_class=spec.module_class,
        channel_scales_dict=channel_scales_dict,
        grad_basis_dict=grad_basis_dict,
    )

    if cfg.router_lr_scale != 1.0:
        logger.info(
            f"HydraLoRA router LR scale: {cfg.router_lr_scale}x unet_lr (applies to .router.* params — σ features live in router.weight columns)"
        )

    network = LoRANetwork(text_encoders, unet, cfg, multiplier=multiplier)

    # Variant-specific defaults — overridden by spec.post_init for the matching variant.
    network._use_hydra = False
    network._balance_loss_weight = 0.0

    # save_weights keys off the resolved spec to pick the save pipeline.
    network._network_spec = spec
    if spec.post_init is not None:
        spec.post_init(network, kwargs)

    # REPA v2 aux alignment loss (off unless use_repa). Config stashed on the
    # network so REPAMethodAdapter + losses._repa_loss read it without new args;
    # head built for absolute mode only (relational/Gram has none). See
    # library/training/repa.py.
    from networks.lora_anima.config import _as_bool

    if _as_bool(kwargs.get("use_repa")):
        network._repa_mode = str(kwargs.get("repa_mode", "relational")).lower()
        network._repa_weight = float(kwargs.get("repa_weight", 0.05) or 0.0)
        network._repa_layer = int(kwargs.get("repa_layer", 8))
        network._repa_encoder = str(kwargs.get("repa_encoder", "pe_spatial"))
        network._repa_lr_scale = float(kwargs.get("repa_lr_scale", 1.0) or 1.0)
        network._repa_anneal_steps = float(kwargs.get("repa_anneal_steps", 0.0) or 0.0)
        network._repa_spatial_norm = _as_bool(kwargs.get("repa_spatial_norm"))
        network._repa_timestep_weighting = float(
            kwargs.get("repa_timestep_weighting", 0.0) or 0.0
        )
        network._repa_grad_heatmap = float(kwargs.get("repa_grad_heatmap", 0) or 0)
        # REPA-DoG target band-pass: replaces spatial_norm in relational target
        # preprocess. Default-on (A/B-validated); repa_target_dog=false reverts.
        network._repa_target_dog = _as_bool(kwargs.get("repa_target_dog"), default=True)
        network._repa_dog_sigma1_div = float(
            kwargs.get("repa_dog_sigma1_div", 16.0) or 16.0
        )
        network._repa_dog_sigma2_div = float(
            kwargs.get("repa_dog_sigma2_div", 0.0) or 0.0
        )
        network._repa_dog_norm_std = float(kwargs.get("repa_dog_norm_std", 0.0) or 0.0)
        from library.vision.encoders import get_encoder_info

        enc_dim = get_encoder_info(network._repa_encoder).d_enc
        dit_dim = int(unet.model_channels)
        if network._repa_mode == "absolute":
            from library.training.repa import REPAHead

            network.repa_head = REPAHead(dit_dim, dit_dim, enc_dim)
            # Training-only: save_weights strips registered prefixes.
            network._training_only_prefixes.add("repa_head.")
        logger.info(
            f"REPA[{network._repa_mode}]: weight={network._repa_weight}, "
            f"layer={network._repa_layer}, encoder={network._repa_encoder}, "
            f"anneal_steps={network._repa_anneal_steps:g}, "
            f"spatial_norm={network._repa_spatial_norm}, "
            f"target_dog={network._repa_target_dog}"
        )
    else:
        network._repa_weight = 0.0

    if cfg.use_timestep_mask:
        logger.info(
            f"Timestep-dependent rank masking: min_rank={cfg.min_rank}, alpha={cfg.alpha_rank_scale}"
        )
    if cfg.router_source == "sigma" and network._global_router_hits > 0:
        logger.info(
            f"GlobalRouter (σ) → Hydra: {network._global_router_hits} "
            f"shared-A modules consume gates from the network-level router "
            f"on sinusoidal(σ) (feat={cfg.sigma_feature_dim}). Per-layer "
            f"routers are disabled; balance loss is inert in this mode."
        )
    elif cfg.router_source == "sigma" and network._sigma_router_hits > 0:
        logger.info(
            f"σ-conditional HydraLoRA router: {network._sigma_router_hits} modules "
            f"with sinusoidal(σ) concatenated to router input (feat={cfg.sigma_feature_dim}), "
            f"per-bucket balance w={cfg.per_bucket_balance_weight}, buckets={cfg.num_sigma_buckets}"
        )
    elif cfg.router_source == "sigma":
        logger.warning(
            "router_source='sigma' but no modules matched router_targets "
            f"regex {cfg.router_targets!r} — σ-routing is inactive"
        )
    routing_aware_count = len(getattr(network, "_routing_aware_loras", []))
    if cfg.router_source == "fei" and network._global_router_hits > 0:
        logger.info(
            f"GlobalRouter (FEI) → Hydra: {network._global_router_hits} "
            f"shared-A modules consume gates from the network-level router "
            f"on FEI ({cfg.fei_feature_dim}-band simplex, σ_low_div={cfg.fei_sigma_low_div}). "
            f"Per-layer routers are disabled; balance loss is inert in this "
            f"mode (gates arrive detached)."
        )
    elif cfg.router_source == "fei" and network._fei_router_hits > 0:
        logger.info(
            f"FEI-conditional HydraLoRA router: {network._fei_router_hits} modules "
            f"with FEI ({cfg.fei_feature_dim}-band simplex) concatenated to router input "
            f"(σ_low_div={cfg.fei_sigma_low_div}). FeRA-style content-aware routing."
        )
    elif (
        cfg.router_source == "fei"
        and cfg.use_moe_style == "independent_A"
        and routing_aware_count > 0
    ):
        logger.info(
            f"GlobalRouter (FEI) → StackedExperts: {routing_aware_count} "
            f"independent-A modules consume gates from the network-level router "
            f"on FEI ({cfg.fei_feature_dim}-band simplex, σ_low_div={cfg.fei_sigma_low_div})."
        )
    elif cfg.router_source == "fei":
        logger.warning(
            "router_source='fei' but no modules matched router_targets "
            f"regex {cfg.router_targets!r} — FEI-routing is inactive"
        )
    if cfg.specialize_experts_by_sigma_buckets:
        experts_per_band = cfg.num_experts // cfg.num_sigma_buckets
        edges_str = (
            f"custom edges {cfg.sigma_bucket_boundaries}"
            if cfg.sigma_bucket_boundaries is not None
            else "uniform edges"
        )
        logger.info(
            f"Hard σ-band expert partition ON: {cfg.num_experts} experts split "
            f"into {cfg.num_sigma_buckets} bands of {experts_per_band} experts "
            f"(interleaved layout, {edges_str}). "
            "Out-of-band logits are masked to -inf before softmax — soft routing "
            "operates only within each σ band."
        )
    if spec.name == "ortho_hydra":
        logger.info(
            f"OrthoHydraLoRA: Cayley + MoE, num_experts={cfg.num_experts}, "
            f"balance_loss_weight={network._balance_loss_weight}"
        )
    elif spec.name == "chimera_hydra":
        logger.info(
            f"ChimeraHydra: dual-pool additive, K_c={cfg.num_experts_content}, "
            f"K_f={cfg.num_experts_freq}, balance(w_c={network._balance_w_content}, "
            f"w_f={network._balance_w_freq}), outer={network._balance_loss_weight}"
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
        fallback_name = "OrthoLoRA" if spec.name == "ortho_hydra" else "LoRA"
        logger.info(
            f"HydraLoRA layer filter: {network._hydra_router_hits} MoE modules, "
            f"{network._hydra_router_misses} fell back to plain {fallback_name} "
            f"(regex={cfg.router_targets!r})"
        )
        if network._hydra_router_hits == 0:
            logger.warning(
                "router_targets regex matched zero modules — no MoE routing "
                "is active, every target became plain LoRA."
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
    metadata: Optional[Dict[str, str]] = None,
    **kwargs,
):
    # load_file() drops safetensors __metadata__, so a caller passing weights_sd=
    # would lose the three-axis stamps and trip from_weights' raise (blaming the
    # checkpoint, not the call site). Precedence: metadata= > file= > {}.
    file_metadata: Dict[str, str] = dict(metadata) if metadata else {}
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            from safetensors import safe_open

            weights_sd = load_file(file)
            if not file_metadata:
                with safe_open(file, framework="pt") as f:
                    file_metadata = dict(f.metadata() or {})
        else:
            weights_sd = torch.load(file, map_location="cpu")
    elif (
        file and not file_metadata and os.path.splitext(str(file))[1] == ".safetensors"
    ):
        # Tensors supplied directly; stamps recovered from the on-disk file anyway.
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            file_metadata = dict(f.metadata() or {})

    # Strip torch.compile '_orig_mod_' from old checkpoint keys
    weights_sd = LoRANetwork._strip_orig_mod_keys(weights_sd)

    # ComfyUI-native adaln LoRAs carry keys in the comfy layout
    # (adaln_modulation_{br}_2); rename to runtime names (adaln_up_{br}) so
    # create_modules attaches them. Presence-gated. See adaln.md.
    from networks.lora_utils import (
        has_comfy_adaln_keys,
        relayout_adaln_comfy_to_runtime,
    )

    if has_comfy_adaln_keys(weights_sd):
        weights_sd = relayout_adaln_comfy_to_runtime(weights_sd)
        logger.info(
            "Renamed ComfyUI-layout adaln LoRA keys to in-repo runtime names "
            "(adaln_modulation_{br}_2 → adaln_up_{br}) for loading."
        )

    # MoE files: stack per-expert ups (and downs, for StackedExperts); chimera
    # dual-A files have their own per-pool ups handled by a separate pair.
    weights_sd = _stack_lora_ups(weights_sd)
    weights_sd = _stack_chimera_lora_ups(weights_sd)
    weights_sd = _refuse_split_stacked_experts_keys(weights_sd)
    weights_sd = _refuse_split_hydra_keys(weights_sd)
    weights_sd = _refuse_split_chimera_keys(weights_sd)
    # Refuse unfused attn projections so modules_dim reflects the runtime (qkv/kv fused).
    weights_sd = _refuse_unfused_attn_lora_keys(weights_sd)

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    has_ortho = False
    has_ortho_hydra = False
    has_hydra = False
    # StackedExperts: 3-D lora_down_weight (E,r,in) discriminates it from Hydra's
    # 2-D shared lora_down.weight. Plan-2 stamps are canonical; this key-sniff is
    # a fallback for unstamped/legacy artifacts.
    has_stacked_experts = False
    hydra_num_experts = 0
    # MoE (Hydra) vs plain lora_names, passed as hydra_router_names so
    # create_modules picks the right class per module in mixed checkpoints.
    hydra_module_names: set[str] = set()
    plain_module_names: set[str] = set()
    # A module with .lora_up_c_weight is chimera, NOT plain Hydra.
    chimera_dual_a_modules: set[str] = set()
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]

        # Old global HydraLoRA router — incompatible with per-module routing.
        if key.startswith("_hydra_router"):
            raise RuntimeError(
                "This checkpoint uses the old global HydraLoRA router "
                "(_hydra_router.*). The router is now per-module and layer-local; "
                "the old format cannot be loaded. Retrain the LoRA to get the new "
                "per-module router weights."
            )

        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif key.endswith(".lora_up_c_weight") or key.endswith(".lora_up_f_weight"):
            # Chimera dual-A per-pool stacked ups; modules_dim filled by the
            # matching ``.lora_down_{c,f}.weight`` branch below.
            chimera_dual_a_modules.add(lora_name)
        elif key.endswith(".lora_down_c.weight") or key.endswith(".lora_down_f.weight"):
            # Chimera dual-A per-pool down (both keys overwrite modules_dim with
            # the same r → safe).
            chimera_dual_a_modules.add(lora_name)
            modules_dim[lora_name] = value.size(0)
        elif key.endswith(".lora_down_weight") and value.dim() == 3:
            # StackedExperts (independent-A) per-expert down (E, r, in) —
            # discriminator vs Hydra, flips the spec to stacked_experts_global_fei.
            has_stacked_experts = True
            hydra_num_experts = max(hydra_num_experts, value.size(0))
            modules_dim[lora_name] = value.size(1)
            hydra_module_names.add(lora_name)
        elif "lora_up_weight" in key:
            # Stacked-3-D in both Hydra and SE; the down discriminates (above for
            # SE, below for Hydra). Defer the has_hydra decision until after loop.
            hydra_num_experts = max(hydra_num_experts, value.size(0))
            hydra_module_names.add(lora_name)
        elif key.endswith(".lora_up.weight"):
            # Plain LoRA up — vanilla, or the plain-fallback leg of a mixed
            # router_targets checkpoint.
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
                # OrthoLoRA: S_p is (r, r) — pure ortho or the plain-fallback
                # leg of a mixed ortho_hydra checkpoint.
                has_ortho = True
                modules_dim[lora_name] = value.size(0)
                plain_module_names.add(lora_name)
        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    # Finalize the MoE shape post-scan: up_weight (3-D) with no matching
    # down_weight (3-D) is Hydra (shared down); both 3-D means StackedExperts.
    if not has_stacked_experts and hydra_module_names and not has_ortho_hydra:
        has_hydra = True

    # MoE keys but no metadata usually means load_file() dropped __metadata__
    # and weights_sd= was passed without file=/metadata= — surface the real
    # cause before from_weights raises blaming the checkpoint.
    if (has_hydra or has_ortho_hydra or has_stacked_experts) and not file_metadata:
        logger.warning(
            "MoE checkpoint keys detected but no safetensors metadata was "
            "available — the three-axis routing stamps (ss_use_moe_style / "
            "ss_route_per_layer / ss_router_source) live in __metadata__, which "
            "load_file() drops. If you passed a pre-loaded weights_sd=, also "
            "pass file=<path> or metadata=<dict> to create_network_from_weights "
            "so the stamps survive; otherwise loading will fail."
        )

    # MoE wins over for_inference: the router is sample-dependent and can't fold
    # into a static merge; the dynamic forward-hook path works in eval too.
    if has_stacked_experts:
        spec = NETWORK_REGISTRY["stacked_experts_global_fei"]
        module_class = spec.module_class
    elif has_ortho_hydra:
        spec = NETWORK_REGISTRY["ortho_hydra"]
        module_class = spec.module_class
    elif has_hydra:
        spec = NETWORK_REGISTRY["hydra"]
        module_class = spec.module_class

    # Legacy σ-router refusal: the additive-bias sigma_mlp design is gone (σ is
    # now a direct concat into a wider router); .sigma_mlp.* can't be reshaped.
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

    # Router is Linear(lora_dim + sigma_feature_dim + fei_feature_dim, E); the
    # FEI slice is stamped (ss_fei_feature_dim), subtract it to recover σ. Old
    # broken shape (width ≈ in_dim) is caught by a sanity cap on excess width.
    new_router_source = str(file_metadata.get("ss_router_source", "")).strip()
    use_fei_router_meta = new_router_source == "fei"
    fei_feature_dim_detected: Optional[int] = (
        int(file_metadata["ss_fei_feature_dim"])
        if use_fei_router_meta and "ss_fei_feature_dim" in file_metadata
        else None
    )
    fei_sigma_low_div_meta: Optional[float] = (
        float(file_metadata["ss_fei_sigma_low_div"])
        if use_fei_router_meta and "ss_fei_sigma_low_div" in file_metadata
        else None
    )
    sigma_feature_dim_detected: Optional[int] = None
    if has_hydra or has_ortho_hydra:
        _SIGMA_FEATURE_CAP = 1024
        fei_slice = int(fei_feature_dim_detected or 0)
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
            extra = width - expected_rank - fei_slice
            if extra < 0:
                raise RuntimeError(
                    f"router.weight at {k!r} has width {width}; expected "
                    f"rank {expected_rank} + fei_feature_dim {fei_slice}. "
                    "Metadata fei_feature_dim does not match the saved router "
                    "shape — checkpoint is malformed."
                )
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
    elif has_stacked_experts:
        # MoE wins over for_inference: per-expert weights can't fold into a
        # static merge, so don't downgrade to plain LoRA.
        pass
    elif for_inference:
        # Force plain LoRA spec even for ortho — merge_to/fuse_weight wants flat
        # down/up, and ortho checkpoints are distilled to LoRA shape at save.
        spec = NETWORK_REGISTRY["lora"]
        module_class = spec.module_class
    elif has_ortho:
        spec = NETWORK_REGISTRY["ortho"]
        module_class = spec.module_class
    else:
        spec = NETWORK_REGISTRY["lora"]
        module_class = spec.module_class

    # Detect baked-in per-channel input scaling. Pass a placeholder ones tensor
    # so each module registers the `inv_scale` buffer at init; load_state_dict
    # then overwrites it (absorption with s=ones is a no-op).
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

    # Register tokens: K from register_tokens key's shape; insert block has no
    # tensor footprint so it rides the ss_register_insert_block metadata stamp.
    num_registers = 0
    register_insert_block = 8
    _reg_tokens = weights_sd.get("register_tokens")
    if _reg_tokens is not None:
        num_registers = int(_reg_tokens.shape[0])
        register_insert_block = int(file_metadata.get("ss_register_insert_block", 8))
        logger.info(
            f"Detected register tokens in checkpoint: K={num_registers}, "
            f"insert_block={register_insert_block} — network stays kept-live "
            "(registers cannot merge into DiT weights)."
        )

    # σ-router names: a module has σ routing iff router.weight width > rank (the
    # excess is the σ feature slice). Empty when sigma_feature_dim_detected is None.
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

    # Per-module Hydra selection: mixed hydra+plain leaves (router_targets
    # subset) build each leaf with its original class; all-hydra → None.
    # Chimera dual-A uses chimera_dual_a_modules so OrthoLoRA-fallback Linears
    # aren't mis-typed as chimera.
    _is_chimera_meta = (
        str(file_metadata.get("ss_use_chimera_hydra", "")).strip().lower() == "true"
    )
    if _is_chimera_meta and chimera_dual_a_modules:
        hydra_router_names = sorted(chimera_dual_a_modules)
    else:
        hydra_router_names = (
            sorted(hydra_module_names)
            if (
                (has_hydra or has_ortho_hydra)
                and plain_module_names
                and hydra_module_names
            )
            else None
        )

    # Hard σ-band partition is non-persistent at the tensor level; recover it
    # from the metadata stamped by save_weights. Older checkpoints lack the
    # metadata and stay on the soft-routing path.
    band_partition_on = (
        str(file_metadata.get("ss_specialize_experts_by_sigma_buckets", "")).lower()
        == "true"
    )
    band_num_buckets = (
        int(file_metadata["ss_num_sigma_buckets"])
        if band_partition_on and "ss_num_sigma_buckets" in file_metadata
        else 0
    )
    if band_partition_on and not (has_hydra or has_ortho_hydra):
        logger.warning(
            "Checkpoint metadata declares specialize_experts_by_sigma_buckets "
            "but no Hydra/OrthoHydra keys were detected — ignoring."
        )
        band_partition_on = False
        band_num_buckets = 0
    if band_partition_on and hydra_num_experts % band_num_buckets != 0:
        raise RuntimeError(
            "Checkpoint metadata declares "
            f"specialize_experts_by_sigma_buckets with num_sigma_buckets="
            f"{band_num_buckets}, but the Hydra stack has "
            f"num_experts={hydra_num_experts} which is not divisible. "
            "Checkpoint is malformed."
        )
    band_boundaries: Optional[List[float]] = None
    if band_partition_on and "ss_sigma_bucket_boundaries" in file_metadata:
        try:
            parsed = json.loads(file_metadata["ss_sigma_bucket_boundaries"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                "ss_sigma_bucket_boundaries metadata is malformed: "
                f"{file_metadata['ss_sigma_bucket_boundaries']!r} ({exc})"
            ) from exc
        if not isinstance(parsed, list) or len(parsed) != band_num_buckets + 1:
            raise RuntimeError(
                "ss_sigma_bucket_boundaries metadata length does not match "
                f"num_sigma_buckets+1={band_num_buckets + 1}: {parsed!r}"
            )
        band_boundaries = [float(v) for v in parsed]

    # FEI router presence is metadata-stamped; per-module list falls back to
    # the σ-router set (or all hydra modules) — same regex as σ.
    fei_router_names: Optional[List[str]] = None
    if use_fei_router_meta and (has_hydra or has_ortho_hydra):
        fei_router_names = sigma_router_names or sorted(hydra_module_names) or None

    # Three-axis stamps from save_weights. All three required for MoE checkpoints
    # (Hydra/OrthoHydra/StackedExperts); from_weights raises otherwise.
    new_use_moe_style: Optional[str] = file_metadata.get("ss_use_moe_style") or None
    raw_route_per_layer = file_metadata.get("ss_route_per_layer")
    new_route_per_layer: Optional[bool] = (
        (str(raw_route_per_layer).strip().lower() == "true")
        if raw_route_per_layer is not None
        else None
    )
    new_router_source_stamp: Optional[str] = (
        new_router_source if new_router_source else None
    )
    # OrthoHydra centered-gate: threaded into the runtime combine so the
    # distilled _moe form subtracts 1/E like training.
    ortho_centered_gate: bool = (
        str(file_metadata.get("ss_ortho_centered_gate", "")).strip().lower() == "true"
    )

    # ss_use_chimera_hydra="true" flips to the chimera spec; FreqRouter input
    # dim needs the chimera-specific FEI/σ stamps below.
    is_chimera_hydra = (
        str(file_metadata.get("ss_use_chimera_hydra", "")).strip().lower() == "true"
    )
    chimera_num_experts_content: Optional[int] = (
        int(file_metadata["ss_num_experts_content"])
        if is_chimera_hydra and "ss_num_experts_content" in file_metadata
        else None
    )
    chimera_num_experts_freq: Optional[int] = (
        int(file_metadata["ss_num_experts_freq"])
        if is_chimera_hydra and "ss_num_experts_freq" in file_metadata
        else None
    )
    chimera_fei_feature_dim: Optional[int] = (
        int(file_metadata["ss_chimera_fei_feature_dim"])
        if is_chimera_hydra and "ss_chimera_fei_feature_dim" in file_metadata
        else None
    )
    chimera_sigma_feature_dim: Optional[int] = (
        int(file_metadata["ss_chimera_sigma_feature_dim"])
        if is_chimera_hydra and "ss_chimera_sigma_feature_dim" in file_metadata
        else None
    )
    chimera_fei_sigma_low_div: Optional[float] = (
        float(file_metadata["ss_chimera_fei_sigma_low_div"])
        if is_chimera_hydra and "ss_chimera_fei_sigma_low_div" in file_metadata
        else None
    )
    # Default false when absent — pre-LN checkpoints trained on raw concat(FEI,σ).
    chimera_freq_router_layer_norm: bool = (
        is_chimera_hydra
        and str(file_metadata.get("ss_chimera_freq_router_layer_norm", ""))
        .strip()
        .lower()
        == "true"
    )
    # Freq routing mode. Absent ⇒ "learned"; "fei" rebuilds the hardwired
    # path (no FreqRouter).
    chimera_freq_router_mode: str = (
        str(file_metadata.get("ss_chimera_freq_router_mode", "learned")).strip().lower()
        if is_chimera_hydra
        else "learned"
    ) or "learned"
    chimera_freq_router_tau: float = (
        float(file_metadata.get("ss_chimera_freq_router_tau", 1.0))
        if is_chimera_hydra
        else 1.0
    )
    # Content routing is always the network-level ContentRouter on pooled
    # crossattn_emb; the retired per-Linear path is rejected, not silently
    # mis-loaded.
    if is_chimera_hydra:
        _content_src = (
            str(file_metadata.get("ss_chimera_content_router_source", "input"))
            .strip()
            .lower()
        )
        if _content_src not in ("crossattn", "crossattn_emb"):
            raise RuntimeError(
                "Chimera checkpoint uses the retired per-Linear content router "
                f"(ss_chimera_content_router_source={_content_src!r}); only the "
                "network-level crossattn_emb ContentRouter is supported now. "
                "Retrain to produce a crossattn_emb chimera checkpoint."
            )
    chimera_content_router_layer_norm: bool = (
        is_chimera_hydra
        and str(file_metadata.get("ss_chimera_content_router_layer_norm", ""))
        .strip()
        .lower()
        == "true"
    )
    if is_chimera_hydra:
        # On-disk: per-pool distilled chimera (lora_down_{c,f} + stacked
        # lora_up_{c,f}_weight + content router). 1-A legacy fallback removed.
        if not chimera_dual_a_modules:
            raise RuntimeError(
                "Checkpoint is stamped ss_use_chimera_hydra=true but contains "
                "no dual-A chimera keys (.lora_up_c_weight / .lora_up_f_weight). "
                "The 1-A chimera format is no longer supported — retrain to "
                "produce the dual-A format."
            )
        spec = NETWORK_REGISTRY["chimera_hydra"]
        from networks.lora_modules import ChimeraHydraInferenceModule

        module_class = ChimeraHydraInferenceModule
        # Chimera dual-A keys are NOT Hydra; clear the auto-set has_hydra so
        # from_weights doesn't demand the three-axis stamps via the MoE branch.
        has_hydra = False
        if (
            chimera_num_experts_content is not None
            and chimera_num_experts_freq is not None
        ):
            hydra_num_experts = chimera_num_experts_content + chimera_num_experts_freq
        # Without these the loader falls back to sigma_feature_dim_detected
        # (default 128) and the FreqRouter gets the wrong input width.
        if chimera_sigma_feature_dim is not None:
            sigma_feature_dim_detected = chimera_sigma_feature_dim
        if chimera_fei_feature_dim is not None:
            fei_feature_dim_detected = chimera_fei_feature_dim
        if chimera_fei_sigma_low_div is not None:
            fei_sigma_low_div_meta = chimera_fei_sigma_low_div

    cfg = LoRANetworkCfg.from_weights(
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        is_hydra_or_ortho_hydra=has_hydra or has_ortho_hydra,
        hydra_num_experts=hydra_num_experts,
        sigma_feature_dim_detected=sigma_feature_dim_detected,
        sigma_router_names=sigma_router_names or None,
        hydra_router_names=hydra_router_names,
        channel_scales_dict=channel_scales_dict,
        specialize_experts_by_sigma_buckets=band_partition_on,
        num_sigma_buckets=band_num_buckets if band_partition_on else None,
        sigma_bucket_boundaries=band_boundaries if band_partition_on else None,
        fei_feature_dim=int(fei_feature_dim_detected or 0),
        fei_sigma_low_div=fei_sigma_low_div_meta,
        fei_router_names=fei_router_names,
        is_stacked_experts=has_stacked_experts,
        new_use_moe_style=new_use_moe_style,
        new_route_per_layer=new_route_per_layer,
        new_router_source=new_router_source_stamp,
        ortho_centered_gate=ortho_centered_gate,
        is_chimera_hydra=is_chimera_hydra,
        num_experts_content=chimera_num_experts_content,
        num_experts_freq=chimera_num_experts_freq,
        freq_router_layer_norm=chimera_freq_router_layer_norm,
        freq_router_mode=chimera_freq_router_mode,
        freq_router_tau=chimera_freq_router_tau,
        content_router_layer_norm=chimera_content_router_layer_norm,
        num_registers=num_registers,
        register_insert_block=register_insert_block,
    )

    network = LoRANetwork(text_encoders, unet, cfg, multiplier=multiplier)
    # Mirror create_network: defaults first, then spec.post_init overrides.
    network._use_hydra = False
    network._balance_loss_weight = 0.0
    network._network_spec = spec
    if spec.post_init is not None:
        spec.post_init(network, kwargs)

    if band_partition_on:
        experts_per_band = hydra_num_experts // band_num_buckets
        edges_str = (
            f"custom edges {band_boundaries}"
            if band_boundaries is not None
            else "uniform edges"
        )
        logger.info(
            f"Hard σ-band expert partition reconstructed from metadata: "
            f"{hydra_num_experts} experts / {band_num_buckets} bands "
            f"({experts_per_band} per band, interleaved layout, {edges_str}) "
            "— out-of-band logits masked at inference."
        )

    return network, weights_sd
