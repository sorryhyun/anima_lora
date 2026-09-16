# State-dict surgery: massage saved checkpoints into the training-runtime shape
# (fused qkv/kv projections, stacked per-expert hydra ups).

import logging
from typing import Dict, List, Optional

import torch

from library.log import setup_logging
from networks.attn_fuse import iter_split_groups

setup_logging()
logger = logging.getLogger(__name__)


# Load-time inverse of the qkv/kv split performed by LoRANetwork.save_weights().
# The training runtime uses fused self_attn.qkv_proj and cross_attn.kv_proj, but saved
# checkpoints are defused to separate q_proj/k_proj/v_proj for ComfyUI compatibility.
# Without this step, reloading such a checkpoint silently drops the attention LoRA keys
# (they don't match the fused runtime names).
#
# Fusion math (n components, each with rank r, out dim `out`):
#   down_fused = cat([down_i], dim=0)                       # [n*r, in]
#   up_fused   = block_diag([up_i * (alpha_i / r)])          # [n*out, n*r]
#   alpha_fused = n * r                                      # -> LoRAModule scale = 1
# The per-component alpha is folded into up_fused so the block-diagonal structure
# reproduces each per-component delta exactly.
#
# Component lists / fragment naming live in ``attn_fuse.ATTN_FUSE_SPECS`` (shared
# with ``lora_save.py``); the scanners below iterate over it via
# ``iter_split_groups``.


def _stack_lora_ups(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stack per-expert ``.lora_ups.N.weight`` / ``.lora_downs.N.weight`` keys
    into fused ``.lora_up_weight`` / ``.lora_down_weight`` parameters
    (training-runtime form). In-place; returns the same dict.

    Two consumers:
      * Hydra (shared lora_down): only the ``.lora_ups.N.weight`` keys are
        per-expert on disk; the down is written under ``.lora_down.weight``
        and stays untouched.
      * StackedExperts (independent-A): both ``.lora_ups.N.weight`` and
        ``.lora_downs.N.weight`` are per-expert on disk and stack into the
        runtime ``lora_up_weight`` / ``lora_down_weight`` Parameters.
    """
    ups_prefixes: Dict[str, Dict[int, torch.Tensor]] = {}
    downs_prefixes: Dict[str, Dict[int, torch.Tensor]] = {}
    for key in list(state_dict.keys()):
        if ".lora_ups." in key and key.endswith(".weight"):
            prefix = key.split(".lora_ups.")[0]
            idx = int(key.split("lora_ups.")[1].split(".")[0])
            ups_prefixes.setdefault(prefix, {})[idx] = state_dict.pop(key)
        elif ".lora_downs." in key and key.endswith(".weight"):
            prefix = key.split(".lora_downs.")[0]
            idx = int(key.split("lora_downs.")[1].split(".")[0])
            downs_prefixes.setdefault(prefix, {})[idx] = state_dict.pop(key)
    for prefix, experts in ups_prefixes.items():
        stacked = torch.stack([experts[i] for i in sorted(experts.keys())])
        state_dict[f"{prefix}.lora_up_weight"] = stacked
    for prefix, experts in downs_prefixes.items():
        stacked = torch.stack([experts[i] for i in sorted(experts.keys())])
        state_dict[f"{prefix}.lora_down_weight"] = stacked
    return state_dict


def _stack_chimera_lora_ups(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Stack per-expert chimera dual-A ``.lora_ups_c.N.weight`` /
    ``.lora_ups_f.N.weight`` keys into the runtime ``.lora_up_c_weight`` /
    ``.lora_up_f_weight`` Parameters. In-place; returns the same dict.

    Chimera-only mirror of :func:`_stack_lora_ups`. Both pools have their
    own per-expert axis on disk (``_c`` for content, ``_f`` for freq) and
    fold into separate stacked Parameters in
    :class:`ChimeraHydraInferenceModule`.
    """
    ups_c_prefixes: Dict[str, Dict[int, torch.Tensor]] = {}
    ups_f_prefixes: Dict[str, Dict[int, torch.Tensor]] = {}
    for key in list(state_dict.keys()):
        if ".lora_ups_c." in key and key.endswith(".weight"):
            prefix = key.split(".lora_ups_c.")[0]
            idx = int(key.split("lora_ups_c.")[1].split(".")[0])
            ups_c_prefixes.setdefault(prefix, {})[idx] = state_dict.pop(key)
        elif ".lora_ups_f." in key and key.endswith(".weight"):
            prefix = key.split(".lora_ups_f.")[0]
            idx = int(key.split("lora_ups_f.")[1].split(".")[0])
            ups_f_prefixes.setdefault(prefix, {})[idx] = state_dict.pop(key)
    for prefix, experts in ups_c_prefixes.items():
        stacked = torch.stack([experts[i] for i in sorted(experts.keys())])
        state_dict[f"{prefix}.lora_up_c_weight"] = stacked
    for prefix, experts in ups_f_prefixes.items():
        stacked = torch.stack([experts[i] for i in sorted(experts.keys())])
        state_dict[f"{prefix}.lora_up_f_weight"] = stacked
    return state_dict


def _refuse_split_chimera_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inverse of the chimera per-pool q/k/v split performed in
    :meth:`ChimeraHydraLoRAModule.build_moe_state_dict`.

    Each chimera Linear gets ``lora_down_{c,f}.weight`` (cloned across
    q/k/v) plus per-pool stacked ups ``lora_up_{c,f}_weight`` (concatenated
    along out_dim across q/k/v) plus a shared ``router.{weight,bias}`` /
    ``alpha`` / optional ``inv_scale``. Refuse step picks the first
    component for cloned tensors and re-concats the per-pool ups.

    Must run AFTER :func:`_stack_chimera_lora_ups`.
    """
    # Detect chimera fused groups via .lora_up_c_weight (one per chimera Linear).
    for shared_prefix, spec in iter_split_groups(state_dict, ".lora_up_c_weight"):
        suffixes = spec.component_letters
        ups_c: List[torch.Tensor] = []
        ups_f: List[torch.Tensor] = []
        downs_c: List[torch.Tensor] = []
        downs_f: List[torch.Tensor] = []
        alphas: List[Optional[torch.Tensor]] = []
        routers_w: List[Optional[torch.Tensor]] = []
        routers_b: List[Optional[torch.Tensor]] = []
        inv_scales: List[Optional[torch.Tensor]] = []
        complete = True
        for suf in suffixes:
            cp = f"{shared_prefix}{suf}_proj"
            ukc = f"{cp}.lora_up_c_weight"
            ukf = f"{cp}.lora_up_f_weight"
            dkc = f"{cp}.lora_down_c.weight"
            dkf = f"{cp}.lora_down_f.weight"
            if any(k not in state_dict for k in (ukc, ukf, dkc, dkf)):
                complete = False
                break
            ups_c.append(state_dict[ukc])
            ups_f.append(state_dict[ukf])
            downs_c.append(state_dict[dkc])
            downs_f.append(state_dict[dkf])
            alphas.append(state_dict.get(f"{cp}.alpha"))
            routers_w.append(state_dict.get(f"{cp}.router.weight"))
            routers_b.append(state_dict.get(f"{cp}.router.bias"))
            inv_scales.append(state_dict.get(f"{cp}.inv_scale"))
        if not complete:
            continue

        # Per-pool concat along out_dim axis.
        up_c_fused = torch.cat(ups_c, dim=1).contiguous()
        up_f_fused = torch.cat(ups_f, dim=1).contiguous()
        down_c = downs_c[0]
        down_f = downs_f[0]
        alpha = alphas[0]
        router_w = routers_w[0]
        router_b = routers_b[0]
        inv_scale = inv_scales[0]

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
        state_dict[f"{fused_prefix}.lora_up_c_weight"] = up_c_fused
        state_dict[f"{fused_prefix}.lora_up_f_weight"] = up_f_fused
        state_dict[f"{fused_prefix}.lora_down_c.weight"] = down_c
        state_dict[f"{fused_prefix}.lora_down_f.weight"] = down_f
        if alpha is not None:
            state_dict[f"{fused_prefix}.alpha"] = alpha
        if router_w is not None:
            state_dict[f"{fused_prefix}.router.weight"] = router_w
        if router_b is not None:
            state_dict[f"{fused_prefix}.router.bias"] = router_b
        if inv_scale is not None:
            state_dict[f"{fused_prefix}.inv_scale"] = inv_scale

        for suf in suffixes:
            cp = f"{shared_prefix}{suf}_proj"
            for subk in (
                "lora_up_c_weight",
                "lora_up_f_weight",
                "lora_down_c.weight",
                "lora_down_f.weight",
                "alpha",
                "router.weight",
                "router.bias",
                "inv_scale",
            ):
                state_dict.pop(f"{cp}.{subk}", None)
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
    for shared_prefix, spec in iter_split_groups(state_dict, ".lora_up_weight"):
        suffixes = spec.component_letters
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
        if not all(u.ndim == 3 and u.shape[0] == e0 and u.shape[2] == r0 for u in ups):
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

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
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
            state_dict[f"{fused_prefix}.{orig_key[len(first_cp) :]}"] = v

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


def _refuse_split_stacked_experts_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inverse of the StackedExperts q/k/v split.

    Mirrors :func:`_refuse_split_hydra_keys` but for the independent-A
    layout: BOTH ``lora_up_weight`` (E, out_i, r) AND ``lora_down_weight``
    (E, r, in) are per-expert stacked Parameters. The discriminator vs
    Hydra is the presence of ``lora_down_weight`` (stacked) versus
    ``lora_down.weight`` (shared scalar).

    Must run AFTER ``_stack_lora_ups`` (which collapses per-expert
    ``lora_downs.{i}.weight`` / ``lora_ups.{i}.weight`` into the stacked
    runtime Parameters).

    StackedExperts has no per-Linear ``router`` / ``sigma_mlp`` keys to
    refuse — the router is a single network-level GlobalRouter, written
    under ``global_router.*`` (top-level, no q/k/v split).
    """
    for shared_prefix, spec in iter_split_groups(state_dict, ".lora_up_weight"):
        suffixes = spec.component_letters
        # Skip groups that are Hydra-form (shared lora_down.weight present).
        first_cp = f"{shared_prefix}{suffixes[0]}_proj"
        if f"{first_cp}.lora_down.weight" in state_dict:
            continue
        if f"{first_cp}.lora_down_weight" not in state_dict:
            continue

        ups: List[torch.Tensor] = []
        downs: List[torch.Tensor] = []
        alphas: List[Optional[torch.Tensor]] = []
        complete = True
        for suf in suffixes:
            cp = f"{shared_prefix}{suf}_proj"
            uk = f"{cp}.lora_up_weight"
            dk = f"{cp}.lora_down_weight"
            if uk not in state_dict or dk not in state_dict:
                complete = False
                break
            ups.append(state_dict[uk])
            downs.append(state_dict[dk])
            alphas.append(state_dict.get(f"{cp}.alpha"))
        if not complete:
            continue

        e0, _, r0 = ups[0].shape
        if not all(u.ndim == 3 and u.shape[0] == e0 and u.shape[2] == r0 for u in ups):
            logger.warning(
                f"stacked-experts attn fuse: inconsistent up shapes at "
                f"{shared_prefix}*, skipping"
            )
            continue
        if not all(
            d.ndim == 3 and d.shape[0] == e0 and d.shape[1] == r0 for d in downs
        ):
            logger.warning(
                f"stacked-experts attn fuse: inconsistent down shapes at "
                f"{shared_prefix}*, skipping"
            )
            continue

        # Per-expert concat along out_dim axis: (E, sum_out, rank).
        up_fused = torch.cat(ups, dim=1).contiguous()
        # Downs are cloned across q/k/v at save (they share the fused input);
        # take the first component.
        down_fused = downs[0]
        alpha = alphas[0]

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
        state_dict[f"{fused_prefix}.lora_up_weight"] = up_fused
        state_dict[f"{fused_prefix}.lora_down_weight"] = down_fused
        if alpha is not None:
            state_dict[f"{fused_prefix}.alpha"] = alpha

        for suf in suffixes:
            cp = f"{shared_prefix}{suf}_proj"
            for subk in (
                "lora_up_weight",
                "lora_down_weight",
                "alpha",
            ):
                state_dict.pop(f"{cp}.{subk}", None)
    return state_dict


def _refuse_unfused_attn_lora_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rewrite unfused q/k/v LoRA keys in-place to match the fused runtime.

    Returns the same dict for chaining. Incomplete or shape-inconsistent groups
    are left untouched (load_state_dict will report them as unexpected).
    """
    for shared_prefix, spec in iter_split_groups(state_dict, ".lora_down.weight"):
        suffixes = spec.component_letters
        downs: List[torch.Tensor] = []
        ups: List[torch.Tensor] = []
        alphas: List[Optional[torch.Tensor]] = []
        inv_scales: List[Optional[torch.Tensor]] = []
        complete = True
        for suf in suffixes:
            dk = f"{shared_prefix}{suf}_proj.lora_down.weight"
            uk = f"{shared_prefix}{suf}_proj.lora_up.weight"
            ak = f"{shared_prefix}{suf}_proj.alpha"
            ik = f"{shared_prefix}{suf}_proj.inv_scale"
            if dk not in state_dict or uk not in state_dict:
                complete = False
                break
            downs.append(state_dict[dk])
            ups.append(state_dict[uk])
            alphas.append(state_dict.get(ak))
            inv_scales.append(state_dict.get(ik))
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

        # Pre-fused detection: save_weights splitting a previously-fused module
        # clones the full fused down into every per-component key. Running the
        # block-diagonal path on that would inflate rank r→n*r per round trip.
        # Identical downs + equal alphas is the reliable signature (independently-
        # trained per-component LoRAs never produce bit-identical downs).
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

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
        state_dict[f"{fused_prefix}.lora_down.weight"] = down_fused
        state_dict[f"{fused_prefix}.lora_up.weight"] = up_fused
        state_dict[f"{fused_prefix}.alpha"] = alpha_fused

        # inv_scale: q/k/v share the Linear input so save cloned the [in_dim]
        # vector; pick the first. Warn on a partial set (mixed channel-scaled /
        # non-scaled per-component training the fused path can't reconcile).
        if all(s is not None for s in inv_scales):
            state_dict[f"{fused_prefix}.inv_scale"] = inv_scales[0]
        elif any(s is not None for s in inv_scales):
            logger.warning(
                f"attn LoRA fuse: partial inv_scale at {shared_prefix}*, "
                "dropping channel scaling on fused module"
            )

        for suf in suffixes:
            for subk in (
                "lora_down.weight",
                "lora_up.weight",
                "alpha",
                "inv_scale",
            ):
                state_dict.pop(f"{shared_prefix}{suf}_proj.{subk}", None)

    return state_dict
