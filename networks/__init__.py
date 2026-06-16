"""NetworkSpec registry for LoRA adapter-method dispatch.

Replaces the flag-cascade in ``networks.lora_anima.create_network`` with a
declarative map. Each entry pairs an adapter variant name with the module
class it instantiates and a ``save_variant`` label consumed by
``networks.lora_save``.

Flag precedence (evaluated top to bottom, first match wins):

    use_chimera_hydra                    → chimera_hydra
    use_moe_style="independent_A"        → stacked_experts_global_fei
    use_moe_style="shared_A" + use_ortho → ortho_hydra
    use_moe_style="shared_A"             → hydra
    use_ortho_init                       → ortho_init
    use_ortho                            → ortho
    (none)                               → lora

The legacy ``use_hydra`` / ``use_sigma_router`` / ``use_fei_router``
kwargs were retired in plan2 task #6 — see ``LoRANetworkCfg.from_kwargs``
for the rejection message. ``use_dora`` was retired alongside the
``lora_deprecated`` module; DoRA is no longer trained, saved, or loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

from networks.lora_modules import (
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoInitLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
    StepExpertLoRAModule,
)


@dataclass(frozen=True)
class NetworkSpec:
    """Descriptor for one adapter variant.

    Attributes:
        name: Stable identifier stamped on the network and written to
            metadata as ``ss_network_spec``. Also the key into
            ``NETWORK_REGISTRY``.
        module_class: Concrete ``LoRAModule`` subclass the network will
            instantiate per target module.
        save_variant: Label keyed into ``networks.lora_save.SAVE_HANDLERS``
            — selects the serialization pipeline for this variant.
        post_init: Optional hook run after the network is built; receives
            ``(network, kwargs)``. Used for variant-specific attribute
            attachment (e.g. hydra balance loss weight).
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None


# Single flat allowlist of every TOML key the LoRA family forwards into
# ``create_network`` as a string kwarg. Two jobs: (1) it lets these keys pass
# the config-schema validator (so a typo like ``num_exprets`` still warns as an
# unknown key), and (2) ``train.py`` copies exactly these keys off ``args`` into
# ``net_kwargs``. It is a closed mirror of what ``LoRANetworkCfg.from_kwargs``
# (+ ``_post_init_hydra``) actually read via ``kwargs.get(...)`` — keep the two
# in sync: adding a ``kwargs.get("foo")`` there means adding ``"foo"`` here.
#
# There is no per-variant partition: every variant's ``from_kwargs`` reads the
# whole set unconditionally (a plain-LoRA run with ``num_experts`` in its TOML
# forwards it; the cfg just ignores it), so splitting by variant bought nothing.
# Grouped by concern below purely for readability.
NETWORK_KWARGS: frozenset[str] = frozenset(
    {
        # --- core targeting / knobs (every variant) ---
        "train_llm_adapter",
        "exclude_patterns",
        "include_patterns",
        "layer_start",
        "layer_end",
        "rank_dropout",
        "module_dropout",
        "verbose",
        # Regex-driven per-module rank / lr overrides
        "network_reg_dims",
        "network_reg_lrs",
        # HydraLoRA router (+ σ-router MLP) LR multiplier on top of unet_lr / reg_lr
        "network_router_lr_scale",
        # LoRA+
        "loraplus_lr_ratio",
        "loraplus_unet_lr_ratio",
        "loraplus_text_encoder_lr_ratio",
        # T-LoRA (timestep-dependent rank masking)
        "use_timestep_mask",
        "min_rank",
        "alpha_rank_scale",
        # Per-channel input pre-scaling (SmoothQuant-style). Gated by alpha:
        # 0.0 disables; 0.5 = sqrt balance; 1.0 fully flattens. Calibration is
        # vendored at `networks/calibration/channel_stats.safetensors`.
        "channel_scaling_alpha",
        # DEPRECATED no-op (fp32-bottleneck path removed 2026-06-10); kept so
        # old snapshot TOMLs replay. The factory logs and ignores it.
        "use_custom_down_autograd",
        # Variant selectors (read by resolve_network_spec)
        "use_ortho",
        # OrthoInit: top-r SVD of W0 as trainable init (no frozen-subspace cap).
        # Mutually exclusive with use_ortho; non-MoE / non-chimera only for now.
        "use_ortho_init",
        # PSOFT-style Cayley-init magnitude (consumed by OrthoHydra +
        # StackedExperts in ortho mode).
        "ortho_init_std",
        # Three-axis routing config (see plan2.md §three-axis-config). Drives
        # `LoRANetworkCfg.from_kwargs` translation; `resolve_network_spec` also
        # dispatches on `use_moe_style="independent_A"` → `stacked_experts_global_fei`.
        "use_moe_style",
        "route_per_layer",
        "router_source",
        # GlobalRouter knobs (consumed only when route_per_layer=False).
        "router_hidden_dim",
        "router_tau",
        # FECL knobs (FeRA auxiliary loss; opt-in via fera_fecl_weight > 0).
        "fera_fecl_weight",
        "fera_num_bands",
        # --- Hydra / OrthoHydra MoE ---
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "expert_init_std",
        # OrthoHydra centered-gate init: recenter gate to (g_e - 1/E) + zero-init
        # router + start λ at ortho_lambda_init so the router gets a step-0 gradient
        # while ΔW stays 0. Consumed by LoRANetworkCfg.from_kwargs.
        "ortho_centered_gate",
        "ortho_lambda_init",
        # Unified layer filter — scopes which Linears participate in routed
        # adaptation (Hydra MoE leaves + σ / FEI feature concatenation).
        "router_targets",
        # σ-conditional router add-on (router_source="sigma")
        "sigma_feature_dim",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "specialize_experts_by_sigma_buckets",
        "sigma_bucket_boundaries",
        # FEI-conditional router (router_source="fei"; FeRA-style content-aware)
        "fei_feature_dim",
        "fei_sigma_low_div",
        # --- Step-expert (turbo DP-DMD student) ---
        # Number of step-indexed up-heads on the shared down-proj (= student_steps).
        # Presence (>1) selects the step_expert spec in resolve_network_spec.
        "step_expert_K",
        # --- ChimeraHydra dual-pool routing ---
        "use_chimera_hydra",
        "num_experts_content",
        "num_experts_freq",
        # Per-pool balance weights. Fall back to balance_loss_weight when unset.
        "balance_w_content",
        "balance_w_freq",
        # FreqRouter init magnitude (small N(0, std)) — non-zero so the freq
        # pool differentiates at step 0.
        "freq_router_init_std",
        # Per-modality LayerNorm on FreqRouter input. Active only when both
        # FEI and σ feature blocks are enabled — equalizes the variance budget
        # so the higher-dim σ block doesn't fan-in-overpower the 2-D FEI simplex.
        "freq_router_layer_norm",
        # Freq-pool routing mode: "learned" (FreqRouter MLP) or "fei" (hardwire
        # π_f = normalize(FEI ** (1/τ)), no params / no σ-features / no freq
        # balance loss; requires num_experts_freq == fei_feature_dim).
        "freq_router_mode",
        "freq_router_tau",
        # Per-pool router LR multipliers — stack on top of network_router_lr_scale.
        "network_content_router_lr_scale",
        "network_freq_router_lr_scale",
        # Parameterless LN on the network-level ContentRouter's pooled crossattn_emb
        # input. Consumed by ``LoRANetworkCfg.from_kwargs``.
        "content_router_layer_norm",
        # ContentRouter output-layer init magnitude (default 0.0 = zero-init).
        "content_router_init_std",
        # Centered-gate λ init for BOTH chimera pools (always-on).
        "chimera_lambda_init",
        # Per-expert capability levers (frozen-Cayley chimera; distill away).
        "chimera_expert_basis_mult",
        "chimera_expert_diag",
        # REPA v2 auxiliary alignment loss (docs/experimental/repa.md).
        # Off by default; the factory builds the head (absolute mode only) and
        # stashes the config on the network for REPAMethodAdapter + _repa_loss.
        "use_repa",
        "repa_mode",  # "relational" (Gram, no head) | "absolute" (patchwise + head)
        "repa_weight",
        "repa_layer",
        "repa_encoder",  # vision-encoder registry name (default pe_spatial)
        "repa_lr_scale",  # head LR multiplier (absolute mode)
        # Phase-1 operating-point levers (docs/experimental/repa.md §"Annealing plan").
        "repa_anneal_steps",  # hard cutoff: (0,1] = fraction of run, >1 = opt steps
        "repa_spatial_norm",  # iREPA target-side spatial standardization (relational)
        # Timestep reweighting (signed; 0 = uniform): >0 emphasizes high noise
        # (cond-utilization regime), <0 emphasizes low noise. See repa.py.
        "repa_timestep_weighting",
        # Lever-3 gate diagnostic: probe the alignment-gradient heatmap every N
        # micro-steps (0 = off); dumps <output_name>_repa_grad_heatmap.npz.
        "repa_grad_heatmap",
        # REPA-DoG target band-pass (docs/proposal/repa_dog_target.md): a broader
        # low-band strip than spatial_norm's DC removal. Off by default; when on
        # it replaces the spatial_norm block in the relational target preprocess.
        "repa_target_dog",  # false = off (no-op); true ⇒ DoG band-pass the target
        "repa_dog_sigma1_div",  # σ₁ = min(gh,gw)/div (outer, broad low band removed)
        "repa_dog_sigma2_div",  # 0 ⇒ σ₂ off (low-band strip only); >div1 ⇒ band-pass
        "repa_dog_norm_std",  # 0 ⇒ empirical std (matches spatial_norm); >0 = fixed
    }
)


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    target = float(blw) if blw is not None else 0.01
    warmup = kwargs.get("balance_loss_warmup_ratio")
    warmup_ratio = float(warmup) if warmup is not None else 0.0
    network._balance_loss_target_weight = target
    network._balance_loss_warmup_ratio = warmup_ratio
    # Hold the balance penalty at 0 during the warmup window so the router can
    # specialize before load-balancing kicks in; flipped to `target` by
    # LoRANetwork.step_balance_loss_warmup once global_step crosses the ratio.
    network._balance_loss_weight = 0.0 if warmup_ratio > 0.0 else target
    network._use_hydra = True
    # FECL weight surface: ``_fera_fecl_loss`` reads
    # ``ctx.network.fecl_weight``. Mirror the cfg value here (and fall back
    # to the kwarg when no cfg is present — keeps unit tests that hand a
    # network without a full cfg working).
    cfg_weight = getattr(getattr(network, "cfg", None), "fera_fecl_weight", None)
    if cfg_weight is not None:
        network.fecl_weight = float(cfg_weight)
    else:
        network.fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0) or 0.0)

    # ChimeraHydra: stamp the chimera flag + per-pool balance weights for
    # ``get_balance_loss`` to consume. Falls back to the shared
    # ``balance_loss_weight`` (the OrthoHydra default) when the user didn't
    # set explicit per-pool weights — matches the proposal §"Balance loss"
    # ("``w_c`` keeps the current ortho-hydra value; ``w_f`` starts at the
    # same value, tunable").
    cfg = getattr(network, "cfg", None)
    if cfg is not None and getattr(cfg, "use_chimera_hydra", False):
        network._use_chimera_hydra = True
        w_c = cfg.balance_w_content if cfg.balance_w_content is not None else target
        w_f = cfg.balance_w_freq if cfg.balance_w_freq is not None else target
        network._balance_w_content = float(w_c)
        # Hardwired-FEI freq gate has no router params and is a fixed function
        # of z_t, so a Switch balance penalty on it is a constant w.r.t. the
        # trained params (zero gradient) — force w_f=0 to keep the loss scalar
        # honest and the logs clean.
        if str(getattr(cfg, "freq_router_mode", "learned")).lower() == "fei":
            network._balance_w_freq = 0.0
        else:
            network._balance_w_freq = float(w_f)
    else:
        network._use_chimera_hydra = False


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {
    "lora": NetworkSpec(
        name="lora",
        module_class=LoRAModule,
        save_variant="standard",
    ),
    "ortho": NetworkSpec(
        name="ortho",
        module_class=OrthoLoRAModule,
        save_variant="ortho_to_lora",
    ),
    # OrthoInit: top-r SVD of W0 as *trainable* init (no Cayley, no frozen
    # subspace). Full LoRA expressivity with a W0-aligned warm start; distills
    # to standard LoRA at save time (sqrt-split λ → down/up), so the on-disk
    # form is identical to a distilled OrthoLoRA. See OrthoInitLoRAModule.
    "ortho_init": NetworkSpec(
        name="ortho_init",
        module_class=OrthoInitLoRAModule,
        save_variant="ortho_to_lora",
    ),
    "hydra": NetworkSpec(
        name="hydra",
        module_class=HydraLoRAModule,
        save_variant="hydra_moe",
        post_init=_post_init_hydra,
    ),
    "ortho_hydra": NetworkSpec(
        name="ortho_hydra",
        module_class=OrthoHydraLoRAModule,
        save_variant="ortho_hydra_to_hydra",
        post_init=_post_init_hydra,
    ),
    # ChimeraHydra: dual-pool additive routing on the OrthoHydra Cayley
    # parameterization (proposal: docs/proposal/chimera_hydra.md). Training
    # builds Cayley params via ``ChimeraHydraLoRAModule``; save distills
    # them to the Hydra-MoE on-disk layout (shared ``lora_down`` + per-expert
    # ``lora_ups.{i}.weight``, q/k/v defused, top-level ``freq_router.*``)
    # written to a ``*_chimera.safetensors`` sibling. Load goes through
    # ``HydraLoRAModule`` with ``num_experts_content > 0`` (the dual-pool
    # runtime form added in this commit), so the Cayley class is training-
    # only — checkpoint resume silently loses the orthogonal parameterization
    # (matches the OrthoHydra → Hydra trade-off).
    "chimera_hydra": NetworkSpec(
        name="chimera_hydra",
        module_class=ChimeraHydraLoRAModule,
        save_variant="chimera_hydra_moe",
        post_init=_post_init_hydra,
    ),
    # Step-expert: shared down-proj + K step-indexed up-heads, hard selection
    # by diffusion step counter (no router). Turbo DP-DMD student only; the
    # student LoRA is kept-live at inference (K heads can't fold into one DiT
    # weight), so save_variant is bespoke (written by TurboDMDNetwork.save_student,
    # not save_network_weights). Selected via the ``step_expert_K`` kwarg.
    "step_expert": NetworkSpec(
        name="step_expert",
        module_class=StepExpertLoRAModule,
        save_variant="step_expert",
    ),
    # FeRA paper-faithful: independent-A stacked experts, single network-level
    # router fed by FEI(z_t). See plan2.md §three-axis-config — selected via
    # ``use_moe_style="independent_A"``. Save handler stub: task #4 wires the
    # real serialization path; until then this spec is reachable only via the
    # gui-methods/fera.toml variant (also a task #6 deliverable).
    "stacked_experts_global_fei": NetworkSpec(
        name="stacked_experts_global_fei",
        module_class=StackedExpertsLoRAModule,
        save_variant="stacked_experts_global_fei",
        post_init=_post_init_hydra,
    ),
}


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the LoRA-family TOML allowlist (``NETWORK_KWARGS``), sorted.

    Single source of truth for train.py — populates the argparse schema and
    the TOML → ``net_kwargs`` forwarding list, so adding a key to
    ``NETWORK_KWARGS`` automatically makes it visible to training without
    touching train.py.
    """
    return tuple(sorted(NETWORK_KWARGS))


def _parse_bool_flag(kwargs: Mapping[str, Any], key: str) -> bool:
    v = kwargs.get(key, False)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() == "true"


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs.

    Precedence is deterministic and documented in the module docstring.
    Raises on mutually-exclusive combinations.

    Honors the ``use_moe_style`` axis (plan2.md §three-axis-config):
    ``"independent_A"`` routes to ``stacked_experts_global_fei`` (FeRA);
    ``"shared_A"`` plus ``use_ortho`` routes to ``ortho_hydra``; bare
    ``"shared_A"`` routes to ``hydra``. The legacy ``use_hydra`` kwarg was
    retired in plan2 task #6 — ``LoRANetworkCfg.from_kwargs`` raises if a
    TOML still carries it.

    ``use_chimera_hydra=True`` short-circuits to the chimera variant. The
    chimera config requires ``use_moe_style="shared_A"`` semantics under
    the hood (OrthoHydra parameterization), but uses K_c + K_f instead of
    a single ``num_experts`` — the user only sets the chimera flag.
    """
    use_ortho = _parse_bool_flag(kwargs, "use_ortho")
    use_ortho_init = _parse_bool_flag(kwargs, "use_ortho_init")
    use_chimera = _parse_bool_flag(kwargs, "use_chimera_hydra")
    if use_ortho and use_ortho_init:
        raise ValueError(
            "use_ortho and use_ortho_init are mutually exclusive: ortho freezes "
            "the SVD basis (Cayley-rotates within it); ortho_init trains the SVD "
            "basis (no cap). Pick one."
        )
    if use_chimera:
        # OrthoInit composes with chimera: ``use_ortho_init=True`` swaps each
        # pool's frozen-basis + Cayley parameterization for trainable SVD-seeded
        # bases (threaded to ``ChimeraHydraLoRAModule`` via ``cfg.use_ortho_init``
        # in ``network.py``). Same chimera_hydra spec — it distills to the
        # identical free-form ``*_chimera.safetensors`` layout either way.
        return NETWORK_REGISTRY["chimera_hydra"]

    # Step-expert (turbo per-step head split) short-circuits when step_expert_K
    # is set and > 1. K==1 collapses to plain LoRA, so don't pay the ModuleList
    # plumbing for it.
    raw_step_K = kwargs.get("step_expert_K")
    if raw_step_K is not None and int(raw_step_K) > 1:
        return NETWORK_REGISTRY["step_expert"]

    raw_moe = kwargs.get("use_moe_style")
    if isinstance(raw_moe, str):
        moe_style = raw_moe.strip()
        if moe_style.lower() in ("false", "none", ""):
            moe_style = ""
    elif raw_moe is False or raw_moe is None:
        moe_style = ""
    else:
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )
    if moe_style not in ("", "shared_A", "independent_A"):
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )

    if use_ortho_init and moe_style:
        raise NotImplementedError(
            "use_ortho_init does not yet compose with use_moe_style — the "
            "orthoinit MoE pool is a separate family member (not implemented)."
        )
    if moe_style == "independent_A":
        return NETWORK_REGISTRY["stacked_experts_global_fei"]
    if moe_style == "shared_A":
        return (
            NETWORK_REGISTRY["ortho_hydra"] if use_ortho else NETWORK_REGISTRY["hydra"]
        )
    if use_ortho_init:
        return NETWORK_REGISTRY["ortho_init"]
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


__all__ = [
    "NetworkSpec",
    "NETWORK_REGISTRY",
    "NETWORK_KWARGS",
    "all_network_kwargs",
    "resolve_network_spec",
]
