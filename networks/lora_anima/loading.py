# State-dict surgery: helpers that massage saved checkpoints into the shape
# the training runtime expects (fused qkv/kv projections, stacked per-expert
# hydra ups). Pulled out of lora_anima.py so the module-building code and
# the factory don't both carry this load/save detail.

import logging
from typing import Dict, List, Optional

import torch

from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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
