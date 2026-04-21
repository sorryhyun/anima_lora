# LoRA network module for Anima
import ast
import os
import re
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from library.log import setup_logging
from networks.lora_modules import (
    LoRAModule,
    LoRAInfModule,
    OrthoLoRAExpModule,
    OrthoHydraLoRAExpModule,
    ReFTModule,
    HydraLoRAModule,
)
from networks.condition_shift import ConditionShift
from networks import NETWORK_REGISTRY, NetworkSpec, resolve_network_spec
from networks import lora_save

import logging

setup_logging()
logger = logging.getLogger(__name__)

_BLOCK_IDX_RE = re.compile(r"blocks\.(\d+)\.")


def _parse_reft_layers(spec, num_blocks: int) -> List[int]:
    """Resolve a ``reft_layers`` spec to a sorted list of block indices.

    Accepted forms:
      - None / "all" / ""  -> every block
      - "last_N"           -> last N blocks
      - "first_N"          -> first N blocks
      - "stride_K"         -> every K-th block starting at 0
      - "3,7,11" or [3,7]  -> explicit indices (string or list[int])
    """
    if spec is None or spec == "all" or spec == "":
        return list(range(num_blocks))
    if isinstance(spec, (list, tuple)):
        indices = [int(i) for i in spec]
    elif isinstance(spec, str):
        s = spec.strip()
        if s.startswith("last_"):
            n = int(s.split("_", 1)[1])
            return list(range(max(0, num_blocks - n), num_blocks))
        if s.startswith("first_"):
            n = int(s.split("_", 1)[1])
            return list(range(min(n, num_blocks)))
        if s.startswith("stride_"):
            k = int(s.split("_", 1)[1])
            if k <= 0:
                raise ValueError(f"reft_layers stride must be positive: {spec!r}")
            return list(range(0, num_blocks, k))
        indices = [int(x) for x in s.split(",") if x.strip()]
    else:
        raise ValueError(f"unrecognized reft_layers spec: {spec!r}")

    bad = [i for i in indices if i < 0 or i >= num_blocks]
    if bad:
        raise ValueError(
            f"reft_layers out of range [0,{num_blocks}): {bad}"
        )
    return sorted(set(indices))


# Load-time inverse of the qkv/kv split performed by LoRANetwork.save_weights().
# The training runtime uses fused self_attn.qkv_proj and cross_attn.kv_proj, but saved
# checkpoints are defused to separate q_proj/k_proj/v_proj for ComfyUI compatibility.
# Without this step, reloading such a checkpoint into the live LoRA module path silently
# drops the attention LoRA keys (they don't match the fused runtime names). This helper
# reassembles the fused LoRA matrices so load_state_dict hits every module.
#
# Fusion math (n components, each with rank r, out dim `out`):
#   down_fused = cat([down_i], dim=0)                       # [n*r, in]
#   up_fused   = block_diag([up_i * (alpha_i / r)])          # [n*out, n*r]
#   alpha_fused = n * r                                      # -> LoRAModule scale = 1
# The per-component alpha is folded into up_fused so the block-diagonal structure
# reproduces each per-component delta exactly.
_LORA_ATTN_FUSE_SPECS = (
    ("self_attn", "qkv", ("q", "k", "v")),
    ("cross_attn", "kv", ("k", "v")),
)


def _stack_lora_ups(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stack per-expert `.lora_ups.N.weight` keys into fused `.lora_up_weight`
    parameters (training-runtime HydraLoRA form). In-place; returns the same dict.
    """
    ups_prefixes: Dict[str, Dict[int, torch.Tensor]] = {}
    for key in list(state_dict.keys()):
        if ".lora_ups." in key and key.endswith(".weight"):
            prefix = key.split(".lora_ups.")[0]
            idx = int(key.split("lora_ups.")[1].split(".")[0])
            ups_prefixes.setdefault(prefix, {})[idx] = state_dict.pop(key)
    for prefix, experts in ups_prefixes.items():
        stacked = torch.stack([experts[i] for i in sorted(experts.keys())])
        state_dict[f"{prefix}.lora_up_weight"] = stacked
    return state_dict


def _refuse_split_hydra_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inverse of the hydra q/k/v split performed in save_weights.

    Mirrors `_refuse_unfused_attn_lora_keys` but for the HydraLoRA key shape:
    stacked `lora_up_weight` of shape (num_experts, out_dim, rank), shared
    `lora_down.weight` / `alpha` / `router.weight` / `router.bias` /
    optional `inv_scale`. Must run AFTER `_stack_lora_ups` so per-expert ups
    have already been folded into `lora_up_weight`.

    At save time, shared tensors (down/alpha/router.*/inv_scale) are cloned
    across q/k/v because routing is driven by the same layer input. Here we
    pick the first component (all three are identical) and concat per-expert
    `lora_up_weight` along the out_dim axis in q,k,v order.
    """
    for attn_type, fused_letters, suffixes in _LORA_ATTN_FUSE_SPECS:
        first_key_suffix = f"_{attn_type}_{suffixes[0]}_proj.lora_up_weight"
        shared_prefixes: List[str] = []
        for key in list(state_dict.keys()):
            if not key.endswith(first_key_suffix):
                continue
            # key == "{shared_prefix}{suffixes[0]}_proj.lora_up_weight"
            shared_prefix = key[: -len(f"{suffixes[0]}_proj.lora_up_weight")]
            shared_prefixes.append(shared_prefix)

        for shared_prefix in shared_prefixes:
            ups: List[torch.Tensor] = []
            downs: List[torch.Tensor] = []
            alphas: List[Optional[torch.Tensor]] = []
            routers_w: List[Optional[torch.Tensor]] = []
            routers_b: List[Optional[torch.Tensor]] = []
            inv_scales: List[Optional[torch.Tensor]] = []
            # Collect any sigma_mlp.* keys per component — they were cloned
            # across q/k/v at save, so picking the first component is correct.
            sigma_mlp_groups: List[Dict[str, torch.Tensor]] = []
            complete = True
            for suf in suffixes:
                cp = f"{shared_prefix}{suf}_proj"
                uk = f"{cp}.lora_up_weight"
                dk = f"{cp}.lora_down.weight"
                if uk not in state_dict or dk not in state_dict:
                    complete = False
                    break
                ups.append(state_dict[uk])
                downs.append(state_dict[dk])
                alphas.append(state_dict.get(f"{cp}.alpha"))
                routers_w.append(state_dict.get(f"{cp}.router.weight"))
                routers_b.append(state_dict.get(f"{cp}.router.bias"))
                inv_scales.append(state_dict.get(f"{cp}.inv_scale"))
                sigma_mlp_groups.append(
                    {
                        k: state_dict[k]
                        for k in list(state_dict.keys())
                        if k.startswith(f"{cp}.sigma_mlp.")
                    }
                )
            if not complete:
                continue

            e0, _, r0 = ups[0].shape
            if not all(
                u.ndim == 3 and u.shape[0] == e0 and u.shape[2] == r0 for u in ups
            ):
                logger.warning(
                    f"hydra attn fuse: inconsistent up shapes at {shared_prefix}*, skipping"
                )
                continue

            # Per-expert concat along out_dim axis: (E, sum_out, rank).
            up_fused = torch.cat(ups, dim=1).contiguous()
            down = downs[0]
            alpha = alphas[0]
            router_w = routers_w[0]
            router_b = routers_b[0]
            inv_scale = inv_scales[0]

            fused_prefix = f"{shared_prefix}{fused_letters}_proj"
            state_dict[f"{fused_prefix}.lora_up_weight"] = up_fused
            state_dict[f"{fused_prefix}.lora_down.weight"] = down
            if alpha is not None:
                state_dict[f"{fused_prefix}.alpha"] = alpha
            if router_w is not None:
                state_dict[f"{fused_prefix}.router.weight"] = router_w
            if router_b is not None:
                state_dict[f"{fused_prefix}.router.bias"] = router_b
            if inv_scale is not None:
                state_dict[f"{fused_prefix}.inv_scale"] = inv_scale
            # sigma_mlp.* cloned across q/k/v at save time — take the first
            # component's copy and rehome under the fused prefix.
            for orig_key, v in sigma_mlp_groups[0].items():
                first_cp = f"{shared_prefix}{suffixes[0]}_proj."
                state_dict[f"{fused_prefix}.{orig_key[len(first_cp):]}"] = v

            for suf in suffixes:
                cp = f"{shared_prefix}{suf}_proj"
                for subk in (
                    "lora_up_weight",
                    "lora_down.weight",
                    "alpha",
                    "router.weight",
                    "router.bias",
                    "inv_scale",
                ):
                    state_dict.pop(f"{cp}.{subk}", None)
                for sk in list(state_dict.keys()):
                    if sk.startswith(f"{cp}.sigma_mlp."):
                        state_dict.pop(sk, None)
    return state_dict


def _refuse_unfused_attn_lora_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rewrite unfused q/k/v LoRA keys in-place to match the fused runtime.

    Returns the same dict for chaining. Incomplete or shape-inconsistent groups
    are left untouched (load_state_dict will report them as unexpected).
    """
    for attn_type, fused_letters, suffixes in _LORA_ATTN_FUSE_SPECS:
        first_key_suffix = f"_{attn_type}_{suffixes[0]}_proj.lora_down.weight"

        shared_prefixes = []
        for key in list(state_dict.keys()):
            if not key.endswith(first_key_suffix):
                continue
            # key == "{shared_prefix}{suffixes[0]}_proj.lora_down.weight"
            # shared_prefix ends with "_{attn_type}_"
            shared_prefix = key[: -len(f"{suffixes[0]}_proj.lora_down.weight")]
            shared_prefixes.append(shared_prefix)

        for shared_prefix in shared_prefixes:
            downs: List[torch.Tensor] = []
            ups: List[torch.Tensor] = []
            alphas: List[Optional[torch.Tensor]] = []
            mags: List[Optional[torch.Tensor]] = []
            complete = True
            for suf in suffixes:
                dk = f"{shared_prefix}{suf}_proj.lora_down.weight"
                uk = f"{shared_prefix}{suf}_proj.lora_up.weight"
                ak = f"{shared_prefix}{suf}_proj.alpha"
                mk = f"{shared_prefix}{suf}_proj.magnitude"
                if dk not in state_dict or uk not in state_dict:
                    complete = False
                    break
                downs.append(state_dict[dk])
                ups.append(state_dict[uk])
                alphas.append(state_dict.get(ak))
                mags.append(state_dict.get(mk))
            if not complete:
                continue

            n = len(suffixes)
            r = downs[0].shape[0]
            in_dim = downs[0].shape[1]
            out = ups[0].shape[0]
            if not all(d.shape == (r, in_dim) for d in downs):
                logger.warning(
                    f"attn LoRA fuse: inconsistent down shapes at {shared_prefix}*, skipping"
                )
                continue
            if not all(u.shape == (out, r) for u in ups):
                logger.warning(
                    f"attn LoRA fuse: inconsistent up shapes at {shared_prefix}*, skipping"
                )
                continue

            dtype = ups[0].dtype
            device = ups[0].device

            # Pre-fused detection. When save_weights splits a previously-fused
            # module it clones the *full* fused down into every per-component
            # key (see "Split fused projections" in save_weights). If we ran
            # the block-diagonal path on that, rank would inflate r -> n*r per
            # round trip (and n^k*r after k cycles). Identical downs + equal
            # alphas across components is the reliable signature of that case;
            # independently-trained per-component LoRAs (e.g. tlora warm-start)
            # never produce bit-identical down tensors.
            def _a(a):
                return a.item() if torch.is_tensor(a) else float(a)

            pre_fused = (
                n >= 2
                and all(torch.equal(downs[0], d) for d in downs[1:])
                and all(a is not None for a in alphas)
                and all(_a(a) == _a(alphas[0]) for a in alphas[1:])
            )

            if pre_fused:
                # Saved alpha is the fused-module alpha, so pass ups through
                # unscaled and keep the runtime scale = alpha/rank intact.
                alpha_value = _a(alphas[0])
                down_fused = downs[0].contiguous()
                up_fused = torch.cat(ups, dim=0).contiguous()
                alpha_fused = torch.tensor(float(alpha_value))
            else:
                per_block_scales: List[float] = []
                for a in alphas:
                    if a is None:
                        # LoRAModule default: alpha = lora_dim -> scale = 1.
                        per_block_scales.append(1.0)
                    else:
                        per_block_scales.append(_a(a) / r)

                down_fused = torch.cat(downs, dim=0).contiguous()
                up_fused = torch.zeros((n * out, n * r), dtype=dtype, device=device)
                for i, (u, s) in enumerate(zip(ups, per_block_scales)):
                    up_fused[i * out : (i + 1) * out, i * r : (i + 1) * r] = u * s
                # alpha_fused = n*r so LoRAModule's scale = (n*r) / (n*r) = 1
                alpha_fused = torch.tensor(float(n * r))

            fused_prefix = f"{shared_prefix}{fused_letters}_proj"
            state_dict[f"{fused_prefix}.lora_down.weight"] = down_fused
            state_dict[f"{fused_prefix}.lora_up.weight"] = up_fused
            state_dict[f"{fused_prefix}.alpha"] = alpha_fused

            # DoRA magnitude is per-output-row; concat matches the fused qkv/kv out dim.
            if all(m is not None for m in mags):
                state_dict[f"{fused_prefix}.magnitude"] = torch.cat(mags, dim=0)
            elif any(m is not None for m in mags):
                logger.warning(
                    f"attn LoRA fuse: partial DoRA magnitude at {shared_prefix}*, "
                    "dropping DoRA on fused module"
                )

            for suf in suffixes:
                for subk in (
                    "lora_down.weight",
                    "lora_up.weight",
                    "alpha",
                    "magnitude",
                ):
                    state_dict.pop(f"{shared_prefix}{suf}_proj.{subk}", None)

    return state_dict


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
    # Break per-expert symmetry at init — without this, zero-init ups (or S_p)
    # give every expert identical outputs and identical gradients under a
    # near-uniform router, so neither experts nor router ever diverge (MoE
    # cold-start deadlock). 1e-4 keeps ΔW ~ std·‖down‖·‖x‖ negligible at init
    # while giving the router a distinct direction per expert to latch onto.
    # Set to 0.0 to restore legacy zero-init.
    expert_init_std = float(kwargs.get("expert_init_std", 1e-4))
    # Fraction of training steps during which only one randomly-chosen expert
    # per module receives gradient (forward still uses all experts via the
    # learned gate, so each expert learns in the full MoE context). 0.0 =
    # off; 0.1 = warmup for the first 10% of steps. Complements
    # expert_init_std: the init perturb gives the router distinct directions
    # to latch onto, warmup then forces experts to actually specialize.
    expert_warmup_ratio = float(kwargs.get("expert_warmup_ratio", 0.0))

    # σ-conditional HydraLoRA router (Track B, docs: timestep-hydra.md).
    # When on, each matching HydraLoRAModule gets a tiny sinusoidal(σ)→E MLP
    # that adds to the gate logits. Identity to base router at init (zero-init
    # on final layer); σ-dependence only emerges if gradients push it.
    use_sigma_router = kwargs.get("use_sigma_router", "false")
    if use_sigma_router is not None:
        use_sigma_router = str(use_sigma_router).lower() == "true"
    sigma_feature_dim = int(kwargs.get("sigma_feature_dim", 128))
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
            f"HydraLoRA router LR scale: {router_lr_scale}x unet_lr (applies to .router. + .sigma_mlp. params)"
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
        expert_init_std=expert_init_std,
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

    if use_timestep_mask:
        logger.info(
            f"Timestep-dependent rank masking: min_rank={min_rank}, alpha={alpha_rank_scale}"
        )
    if use_sigma_router and network._sigma_router_hits > 0:
        logger.info(
            f"σ-conditional HydraLoRA router: {network._sigma_router_hits} modules "
            f"with sigma_mlp (feat={sigma_feature_dim}, hidden={sigma_hidden_dim}), "
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

    # Old-format per-module router — was Linear(in_dim, E). Current router is
    # Linear(lora_dim, E); the old shape carries a router that never trained
    # (see docs/methods/hydra-lora.md §Fixes) and cannot be reshaped into the
    # new module, so refuse at load with an explicit retrain message.
    if has_hydra or has_ortho_hydra:
        for k, v in weights_sd.items():
            if not k.endswith(".router.weight"):
                continue
            lora_name = k[: -len(".router.weight")]
            expected_rank = modules_dim.get(lora_name)
            if expected_rank is None:
                continue
            if v.ndim != 2 or v.size(1) != expected_rank:
                raise RuntimeError(
                    f"This checkpoint has an old-shape HydraLoRA router at "
                    f"{k!r} (shape {tuple(v.shape)}); the current router is "
                    f"Linear(lora_dim={expected_rank}, num_experts). Old routers "
                    "never received meaningful gradient under the previous "
                    "mean-pool-over-raw-input path (see "
                    "docs/methods/hydra-lora.md §Fixes); there is no salvage "
                    "path — retrain the LoRA to produce a router in the new "
                    "rank-R input space."
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

    # Detect σ-conditional router from checkpoint: any .sigma_mlp.* key means
    # that module was trained with σ-routing. Extract per-module names and the
    # feature dim from the first such key's shape.
    sigma_router_names: List[str] = []
    sigma_feature_dim_detected: Optional[int] = None
    sigma_hidden_dim_detected: Optional[int] = None
    for k, v in weights_sd.items():
        if ".sigma_mlp." not in k:
            continue
        lora_name = k.split(".sigma_mlp.", 1)[0]
        if lora_name not in sigma_router_names:
            sigma_router_names.append(lora_name)
        # sigma_mlp.0 is the first Linear: weight shape (hidden, feat)
        if k.endswith(".sigma_mlp.0.weight"):
            sigma_hidden_dim_detected = v.shape[0]
            sigma_feature_dim_detected = v.shape[1]

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
        sigma_hidden_dim=sigma_hidden_dim_detected or 128,
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
        # load_state_dict overwrites the random init, so skip the perturb to
        # avoid transient CPU noise on (E, out, r) experts during build.
        expert_init_std=0.0,
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
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_llm_adapter: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        verbose: Optional[bool] = False,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        add_reft: bool = False,
        reft_dim: int = 4,
        reft_alpha: Optional[float] = None,
        reft_layers: object = "all",
        num_experts: int = 4,
        channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None,
        use_sigma_router: bool = False,
        sigma_feature_dim: int = 128,
        sigma_hidden_dim: int = 128,
        sigma_router_layers: Optional[str] = None,
        sigma_router_names: Optional[List[str]] = None,
        hydra_router_layers: Optional[str] = None,
        hydra_router_names: Optional[List[str]] = None,
        expert_init_std: float = 1e-4,
        expert_warmup_ratio: float = 0.0,
        router_lr_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_llm_adapter = train_llm_adapter
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs
        self.router_lr_scale = float(router_lr_scale)
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_experts = num_experts
        self.expert_init_std = float(expert_init_std)
        self.expert_warmup_ratio = float(expert_warmup_ratio)
        self.channel_scales_dict = channel_scales_dict
        self._channel_scale_misses: List[str] = []
        self._channel_scale_hits: int = 0
        self.use_sigma_router = bool(use_sigma_router)
        self.sigma_feature_dim = int(sigma_feature_dim)
        self.sigma_hidden_dim = int(sigma_hidden_dim)
        # Either regex (fresh-from-kwargs path) or explicit name set
        # (from-weights path, detected from checkpoint keys). Explicit set wins.
        self._sigma_router_names = (
            set(sigma_router_names) if sigma_router_names else None
        )
        self._sigma_router_re = (
            re.compile(sigma_router_layers)
            if (
                self.use_sigma_router
                and sigma_router_layers
                and self._sigma_router_names is None
            )
            else None
        )
        self._sigma_router_hits: int = 0
        self._last_sigma: Optional[torch.Tensor] = None

        # Per-module HydraLoRA gating. Matching modules get the Hydra class;
        # non-matching modules fall back to plain LoRA / OrthoLoRAExp so MoE
        # capacity is concentrated where specialization is actually learnable.
        # Fresh path: regex over `original_name`. From-weights path: explicit
        # name set detected from checkpoint keys (mirrors sigma_router_names).
        # Explicit set wins. None on both = apply MoE everywhere (legacy).
        self._hydra_router_names = (
            set(hydra_router_names) if hydra_router_names else None
        )
        self._hydra_router_re = (
            re.compile(hydra_router_layers)
            if hydra_router_layers and self._hydra_router_names is None
            else None
        )
        self._hydra_router_hits: int = 0
        self._hydra_router_misses: int = 0

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
            )
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
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

        exclude_re_patterns = str_to_re_patterns(exclude_patterns)
        include_re_patterns = str_to_re_patterns(include_patterns)

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
                                self.layer_start is not None
                                or self.layer_end is not None
                            ):
                                block_match = _BLOCK_IDX_RE.match(original_name)
                                if block_match:
                                    block_idx = int(block_match.group(1))
                                    if (
                                        self.layer_start is not None
                                        and block_idx < self.layer_start
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} < {self.layer_start})"
                                            )
                                        continue
                                    if (
                                        self.layer_end is not None
                                        and block_idx >= self.layer_end
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} >= {self.layer_end})"
                                            )
                                        continue

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            else:
                                if self.reg_dims is not None:
                                    for reg, d in self.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = self.alpha
                                            logger.info(
                                                f"Module {original_name} matched with regex '{reg}' -> dim: {dim}"
                                            )
                                            break
                                if dim is None:
                                    if is_linear or is_conv2d_1x1:
                                        dim = (
                                            default_dim
                                            if default_dim is not None
                                            else self.lora_dim
                                        )
                                        alpha_val = self.alpha

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
                if module_class in (HydraLoRAModule, OrthoHydraLoRAExpModule) and is_unet:
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
                    extra_kwargs["num_experts"] = self.num_experts
                    extra_kwargs["expert_init_std"] = self.expert_init_std
                elif effective_module_class == HydraLoRAModule:
                    extra_kwargs["num_experts"] = self.num_experts
                    extra_kwargs["expert_init_std"] = self.expert_init_std

                # σ-conditional router: only build sigma_mlp on modules whose
                # name matches the layer filter (cross_attn.q / self_attn.qkv
                # by default — see B0 pre-analysis in timestep-hydra.md).
                # From-weights path uses an explicit name set; fresh-from-kwargs
                # path uses a regex over original_name. Gated on the effective
                # class so a hydra-excluded module can't pick up σ either.
                if self.use_sigma_router and effective_module_class in (
                    HydraLoRAModule,
                    OrthoHydraLoRAExpModule,
                ) and is_unet:
                    if self._sigma_router_names is not None:
                        enable = lora_name in self._sigma_router_names
                    elif self._sigma_router_re is not None:
                        enable = bool(self._sigma_router_re.search(original_name))
                    else:
                        enable = True
                    if enable:
                        extra_kwargs["sigma_feature_dim"] = self.sigma_feature_dim
                        extra_kwargs["sigma_hidden_dim"] = self.sigma_hidden_dim
                        self._sigma_router_hits += 1

                # Per-channel scaling is DiT-only: the bench script hooks DiT
                # linears, text encoder activations are never calibrated.
                if self.channel_scales_dict is not None and is_unet:
                    _cs = self.channel_scales_dict.get(lora_name)
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
        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
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

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
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

        if self.channel_scales_dict is not None:
            logger.info(
                f"per_channel_scaling: {self._channel_scale_hits} DiT modules "
                f"received calibration-based input scaling"
            )
            if self._channel_scale_misses:
                logger.warning(
                    f"per_channel_scaling: {len(self._channel_scale_misses)} DiT modules "
                    f"have no calibration stats (first: {self._channel_scale_misses[:3]}). "
                    f"These will train without input rebalancing — regenerate the stats "
                    f"file with `python bench/analyze_lora_input_channels.py "
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
        if not getattr(self, "_use_timestep_mask", False):
            return

        # Reuse a single GPU-resident mask to avoid ~200 CPU→GPU transfers per step
        mask = getattr(self, "_shared_timestep_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, self._max_rank, device=timesteps.device)
            self._shared_timestep_mask = mask
            self._timestep_mask_arange = torch.arange(
                self._max_rank, device=timesteps.device
            )
            for lora in self.text_encoder_loras + self.unet_loras:
                lora._timestep_mask = mask

        # Compute threshold r entirely on device — avoids GPU→CPU .item() sync and
        # keeps the effective rank as a tensor so the mask build stays static-shape.
        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = frac.pow(self._alpha_rank_scale) * (self._max_rank - self._min_rank) + self._min_rank
        r = r.clamp(max=float(self._max_rank))
        mask.copy_(
            (self._timestep_mask_arange < r).to(mask.dtype).unsqueeze(0)
        )

    def set_reft_timestep_mask(
        self, timesteps: torch.Tensor, max_timestep: float = 1.0
    ):
        """Compute and set timestep-dependent mask on ReFT modules."""
        if not getattr(self, "_use_timestep_mask", False):
            return
        refts = self.text_encoder_refts + self.unet_refts
        if not refts:
            return
        reft_dim = getattr(self, "_reft_dim", 4)

        mask = getattr(self, "_shared_reft_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, reft_dim, device=timesteps.device)
            self._shared_reft_mask = mask
            self._reft_mask_arange = torch.arange(reft_dim, device=timesteps.device)
            for reft in refts:
                reft._timestep_mask = mask

        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = frac.pow(self._alpha_rank_scale) * (reft_dim - 1) + 1
        r = r.clamp(max=float(reft_dim))
        mask.copy_((self._reft_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def clear_timestep_mask(self):
        """Remove timestep mask (use full rank)."""
        self._shared_timestep_mask = None
        for lora in self.text_encoder_loras + self.unet_loras:
            lora._timestep_mask = None
        self._shared_reft_mask = None
        for reft in self.text_encoder_refts + self.unet_refts:
            reft._timestep_mask = None

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        """Stash per-sample σ on every HydraLoRA module that has a ``sigma_mlp``.

        Mirrors ``set_timestep_mask`` — one call per training step, propagates
        σ to every module that opts in. Modules without a ``sigma_mlp`` just
        ignore it. The network also caches σ on itself so the per-σ-bucket
        balance loss can bucket the cached gates at loss-compute time.
        """
        if not self.use_sigma_router:
            return
        self._last_sigma = sigmas.detach()
        for lora in self.unet_loras + self.text_encoder_loras:
            if getattr(lora, "sigma_mlp", None) is not None:
                lora._sigma = sigmas

    def clear_sigma(self) -> None:
        """Drop cached σ from every HydraLoRA module. Used in eval / validation."""
        self._last_sigma = None
        for lora in self.unet_loras + self.text_encoder_loras:
            if hasattr(lora, "_sigma"):
                lora._sigma = None

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
        warmup window, ``_warmup_active`` is False everywhere and the gradient
        mask branch compiles out — every expert receives gradient normally.

        The per-module state is split into a python bool (``_warmup_active``,
        toggled at most twice per run) and a buffer (``_expert_grad_mask``,
        re-written each step). Keeping the rotating index out of a plain int
        attribute is load-bearing: torch.compile treats int module attributes
        as static and blows its recompile limit as experts rotate 0→1→2→3.
        """
        if self.expert_warmup_ratio <= 0.0 or max_train_steps <= 0:
            return
        warmup_steps = int(max_train_steps * self.expert_warmup_ratio)
        in_warmup = global_step < warmup_steps
        for lora in self.unet_loras + self.text_encoder_loras:
            if not hasattr(lora, "_expert_grad_mask"):
                continue
            lora._warmup_active = in_warmup
            if in_warmup:
                idx = int(torch.randint(0, lora.num_experts, (1,)).item())
                mask = lora._expert_grad_mask
                mask.zero_()
                mask[idx] = 1.0

    @staticmethod
    def _switch_balance(gate: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer balance: E · Σ_i frac_i · mean_gate_i. Scalar."""
        num_experts = gate.shape[-1]
        expert_idx = gate.argmax(dim=-1)  # (B,)
        frac = torch.zeros(num_experts, device=gate.device, dtype=gate.dtype)
        frac.scatter_add_(
            0, expert_idx, torch.ones_like(expert_idx, dtype=gate.dtype)
        )
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
        num_buckets = int(getattr(self, "_num_sigma_buckets", 3))
        bucket_w = float(getattr(self, "_per_bucket_balance_weight", 0.0) or 0.0)
        want_per_bucket = (
            self.use_sigma_router
            and sigma is not None
            and num_buckets > 1
            and bucket_w > 0.0
        )
        if want_per_bucket:
            thresholds = torch.linspace(
                0.0, 1.0, num_buckets + 1, device=sigma.device
            )[1:-1]
            bucket_ids = torch.bucketize(sigma.float(), thresholds)  # (B,) in [0, N)

        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            term = self._switch_balance(gate)
            total = term if total is None else total + term
            count += 1

            if want_per_bucket and getattr(lora, "sigma_mlp", None) is not None:
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
                        else per_bucket_total
                        + module_bucket_sum / module_bucket_count
                    )
                    per_bucket_count += 1

        if total is None:
            return torch.tensor(0.0)
        out = total / count
        if per_bucket_total is not None and per_bucket_count > 0:
            out = out + bucket_w * (per_bucket_total / per_bucket_count)
        return out

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
                list(self.reg_lrs.items()) if self.reg_lrs is not None else []
            )
            router_scale = float(getattr(self, "router_lr_scale", 1.0))

            def _is_router_param(pname: str) -> bool:
                # named_parameters() yields top-level names like "router.weight"
                # / "sigma_mlp.0.weight" — no leading dot.
                return pname.startswith("router.") or pname.startswith("sigma_mlp.")

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
                        + (" plus" if key == "plus" else (" router" if key == "router" else ""))
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

        state_dict = self.state_dict()
        lora_save.save_network_weights(
            state_dict,
            file=file,
            dtype=dtype,
            metadata=metadata,
            save_variant=spec.save_variant,
        )

    def backup_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                org_module._lora_org_weight = org_module.weight.detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                if not org_module._lora_restored:
                    org_module.weight.data.copy_(org_module._lora_org_weight)
                    org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
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
