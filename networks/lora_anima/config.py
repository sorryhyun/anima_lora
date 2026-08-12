"""Frozen configuration object for ``LoRANetwork``.

Replaces the 25-arg ``LoRANetwork.__init__`` and the per-kwarg parse pile in
``factory.create_network`` / ``create_network_from_weights``. Two construction
sites — ``from_kwargs`` (fresh training; absorbs the str→bool/int/float casts
that train.py's ``net_kwargs`` produces) and ``from_weights`` (warm-start /
inference; values come from checkpoint key sniffing).

Frozen by intent: every field here is fixed for the run. Mutable runtime
state (``multiplier``, LoRA+ ratios, hit counters, σ caches, post-build attrs
written by ``spec.post_init``) stays as plain attributes on the network.
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, Union

import torch

from networks.lora_modules import LoRAModule

# Three-axis routing config (see plan2.md §three-axis-config).
MoEStyle = Union[Literal[False], Literal["shared_A"], Literal["independent_A"]]
RouterSource = Literal["input", "sigma", "fei", "crossattn_emb", "none"]

logger = logging.getLogger(__name__)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a kwarg that may arrive as ``"true"`` / ``"false"`` / bool / None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _as_moe_style(value: Any) -> MoEStyle:
    """Parse the three-valued ``use_moe_style`` kwarg.

    Accepts: ``False`` / ``None`` / ``"false"`` / ``""`` → ``False``;
    the literal strings ``"shared_A"`` / ``"independent_A"`` pass through.
    """
    if value is None or value is False:
        return False
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in ("false", "none", ""):
            return False
        if v in ("shared_A", "independent_A"):
            return v
    raise ValueError(
        f"use_moe_style={value!r}: expected False, 'shared_A', or 'independent_A'."
    )


def _as_router_source(value: Any) -> RouterSource:
    """Parse the ``router_source`` kwarg. Empty / None → ``"none"``.

    ``"crossattn_emb"`` routes the network-level GlobalRouter on the pooled
    post-LLM-adapter text features the DiT cross-attends to (route_per_layer
    must be False — there is no per-Linear crossattn signal).
    """
    if value is None:
        return "none"
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return "none"
        if v in ("input", "sigma", "fei", "crossattn_emb", "none"):
            return v  # type: ignore[return-value]
    raise ValueError(
        f"router_source={value!r}: expected 'input', 'sigma', 'fei', "
        "'crossattn_emb', or 'none'."
    )


def _as_str_list(value: Any) -> Optional[List[str]]:
    """Parse a kwarg that's either a python-literal list, single string, or None."""
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return [value] if isinstance(value, str) else None
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _as_float_list(value: Any) -> Optional[List[float]]:
    """Parse a kwarg that's either a TOML list, python-literal list string, or None.

    TOML arrays come through as native lists; CLI-stringified lists parse via
    ast.literal_eval. Raises on malformed input rather than silently dropping
    it, since a wrong σ-bucket boundary list would change band assignments
    without surfacing an error.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Could not parse list-of-floats kwarg: {value!r} ({exc})"
            ) from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Expected list of floats, got {type(value).__name__}: {value!r}"
        )
    return [float(v) for v in value]


def _validate_sigma_bucket_boundaries(
    boundaries: List[float], num_sigma_buckets: int
) -> None:
    """Validate a custom σ-bucket boundary list. Raises ValueError on any
    violation: wrong length, non-zero start, non-one end, or non-strictly-
    increasing edges.
    """
    if len(boundaries) != num_sigma_buckets + 1:
        raise ValueError(
            "sigma_bucket_boundaries must have length num_sigma_buckets + 1 = "
            f"{num_sigma_buckets + 1}, got {len(boundaries)}."
        )
    if abs(boundaries[0]) > 1e-6:
        raise ValueError(
            f"sigma_bucket_boundaries[0] must be 0.0, got {boundaries[0]}."
        )
    if abs(boundaries[-1] - 1.0) > 1e-6:
        raise ValueError(
            f"sigma_bucket_boundaries[-1] must be 1.0, got {boundaries[-1]}."
        )
    for i in range(len(boundaries) - 1):
        if boundaries[i + 1] <= boundaries[i]:
            raise ValueError(
                "sigma_bucket_boundaries must be strictly increasing; "
                f"violated at index {i}: {boundaries[i]} >= {boundaries[i + 1]}."
            )


def _parse_kv_pairs(kv_pair_str: str, *, is_int: bool) -> Dict[str, Any]:
    """Parse "key1=val1,key2=val2" into a dict, casting values to int/float."""
    pairs: Dict[str, Any] = {}
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


# Default exclude regex appended to user-supplied excludes in `from_kwargs`.
# Skips embedders / norms / modulation projectors. NB `adaln_up_` is rescued
# back into the target set by `train_adaln` (on by default in base.toml) via
# include_patterns, so it is excluded here but adapted on a stock run.
_DEFAULT_EXCLUDE = (
    r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|"
    r"pooled_text_proj).*"
)


@dataclass(frozen=True)
class LoRANetworkCfg:
    """Run-fixed configuration for a ``LoRANetwork``.

    Field groupings mirror the comment blocks in ``factory.create_network``:
    core / targeting / dropouts / regex overrides / T-LoRA / Hydra /
    σ-router / channel scaling / logging.
    """

    lora_dim: int = 4
    alpha: float = 1.0
    module_class: Type = LoRAModule
    # warm-start path supplies these from the checkpoint; fresh path leaves None
    modules_dim: Optional[Dict[str, int]] = None
    modules_alpha: Optional[Dict[str, float]] = None

    train_llm_adapter: bool = False
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: Optional[List[str]] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None

    dropout: Optional[float] = None
    rank_dropout: Optional[float] = None
    module_dropout: Optional[float] = None

    reg_dims: Optional[Dict[str, int]] = None
    reg_lrs: Optional[Dict[str, float]] = None
    # Per-pattern alpha override (regex fullmatch on the module path, like
    # reg_dims). Applies independently of reg_dims — a matched module keeps its
    # resolved dim and takes this alpha instead of the network alpha.
    reg_alphas: Optional[Dict[str, float]] = None

    use_timestep_mask: bool = False
    min_rank: int = 1
    alpha_rank_scale: float = 1.0

    num_experts: int = 4
    # Gaussian perturb std applied to fused per-expert `lora_up_weight` at
    # init in plain HydraLoRA only (NOT OrthoHydra disjoint or fallback) —
    # paper baseline knob; production training should leave at 0.0.
    expert_init_std: float = 0.0
    # OrthoHydra centered-gate init: gate recentered to ``g_e - 1/E``, router
    # fully zero-init, λ starts at ``ortho_lambda_init`` (nonzero). Gives ΔW=0
    # at init yet nonzero router gradient (disjoint P_e survive mean subtraction).
    # Off + ``ortho_lambda_init=0.0`` = legacy zero-init-λ (router gated until λ ramps).
    ortho_centered_gate: bool = False
    ortho_lambda_init: float = 0.0
    router_lr_scale: float = 1.0
    # Single regex scoping which Linears are routed; matched → Hydra leaves,
    # else fall back to plain LoRA/OrthoLoRA. σ and FEI routers share this set
    # (no separate sub-filter). ``None`` = apply MoE everywhere.
    router_targets: Optional[str] = None
    hydra_router_names: Optional[List[str]] = None
    per_bucket_balance_weight: float = 0.3
    num_sigma_buckets: int = 3
    # Hard expert/timestep partition: split E into num_sigma_buckets interleaved
    # bands (expert e → band ``e mod num_sigma_buckets``); a σ-in-band-b sample
    # masks out-of-band logits to -inf before softmax (soft routing within a band).
    # Composes with the σ-feature router. Requires num_experts % num_sigma_buckets == 0.
    specialize_experts_by_sigma_buckets: bool = False
    # Custom σ-bucket boundaries: length num_sigma_buckets+1, strictly increasing,
    # 0.0→1.0; None = uniform linspace. Lets you spend more capacity on a σ regime
    # (e.g. ``[0.0, 0.5, 0.8, 1.0]`` = wide low-σ band) at equal experts/band.
    sigma_bucket_boundaries: Optional[List[float]] = None

    # Three-axis routing config (see plan2.md). ``use_moe_style`` picks the
    # expert layout — ``False`` (no experts), ``"shared_A"`` (Hydra: one
    # ``lora_down`` + per-expert ``lora_up``), ``"independent_A"`` (FeRA:
    # stacked per-expert ``lora_down`` and ``lora_up``). ``route_per_layer``
    # picks router location: ``True`` (today's Hydra per-Linear default) or
    # ``False`` (one network-level router, FeRA-style). ``router_source``
    # picks the gate input: ``"input"`` (per-Linear input vector — only valid
    # with ``route_per_layer=True``), ``"sigma"`` (sinusoidal σ features),
    # ``"fei"`` (FEI(z_t) simplex), or ``"none"``.
    use_moe_style: MoEStyle = False
    route_per_layer: bool = False
    router_source: RouterSource = "none"

    # PSOFT-style Cayley/SVD parameterization (per-module bool). Selects
    # ``ortho`` mode on ``StackedExpertsLoRAModule`` when paired with
    # ``use_moe_style="independent_A"``; for the non-MoE / shared_A cells the
    # ``ortho``-ness is already encoded in the chosen module class
    # (``OrthoLoRA`` / ``OrthoHydra``) and this field is informational.
    use_ortho: bool = False
    ortho_init_std: float = 0.02
    # OrthoInit: top-r SVD of W0 as *trainable* init (no frozen-subspace cap).
    # Selects ``OrthoInitLoRAModule`` via ``resolve_network_spec``; mutually
    # exclusive with ``use_ortho`` (validated in the resolver). Non-MoE only.
    use_ortho_init: bool = False

    # SVD-Down: ``lora_down`` initialization for plain LoRA — ``"kaiming"``
    # (default) or ``"weight_svd"`` (seed input basis from W0's top-r right
    # singular vectors, scale-matched). Plain two-factor LoRAModule only;
    # ignored by ortho/Hydra/Chimera classes. docs/methods/svd-down-lora.md.
    down_init: str = "kaiming"

    # σ-conditional router parameters (consumed when ``router_source="sigma"``).
    # Layer scope is shared with Hydra and FEI via ``router_targets`` above.
    sigma_feature_dim: int = 16
    sigma_router_names: Optional[List[str]] = None

    # FEI-conditional router parameters (consumed when ``router_source="fei"``).
    # ``fei_feature_dim`` defaults to 2 = the simplex ``(e_low, e_high)`` from
    # ``library.runtime.fei.compute_fei_2band``. Default
    # ``fei_sigma_low_div=4.0`` for σ_low scaling — chosen by the
    # 2026-05-13 dataset sweep on real training latents (highest
    # std(e_low) at low/mid t). 8.0 remains a Pareto choice. See
    # ``[[project_fera_probe_2band_decision]]``.
    fei_feature_dim: int = 2
    fei_sigma_low_div: float = 4.0
    fei_router_names: Optional[List[str]] = None

    # GlobalRouter parameters (consumed when ``route_per_layer=False``).
    # Two-layer MLP feeding softmax/τ — same shape as FeRA's
    # ``SoftFrequencyRouter``. Final layer is zero-init so step-0 gates are
    # uniform; combined with zero-init expert ups this guarantees ΔW=0 at
    # the first optimizer step.
    router_hidden_dim: int = 64
    router_tau: float = 0.7

    # FECL (Frequency-Energy Consistency Loss) — opt-in FeRA aux loss; 0.0 off
    # (2-band path collapses to a content-free scalar; bench at 3 bands).
    # ``library/training/losses.py::_fera_fecl_loss`` reads ``network.fecl_weight``.
    fera_fecl_weight: float = 0.0
    fera_num_bands: int = 3

    # ChimeraHydra dual-pool additive routing (docs/proposal/chimera_hydra.md):
    # content pool (K_c, per-layer router) + freq pool (K_f, network FreqRouter
    # on FEI+σ), E = K_c + K_f. Per-pool balance weights tracked separately — a
    # combined term would let one pool flatten while the other concentrates.
    use_chimera_hydra: bool = False
    num_experts_content: int = 3
    num_experts_freq: int = 3
    balance_w_content: Optional[float] = None  # falls back to balance_loss_weight
    balance_w_freq: Optional[float] = None  # falls back to balance_loss_weight
    # FreqRouter init magnitude. Non-zero so the freq pool differentiates
    # immediately as FEI/σ vary across the batch — zero-weight init would
    # be a fixed point under the additive composition (see proposal §"Init").
    freq_router_init_std: float = 0.1
    # Per-modality LayerNorm on the FreqRouter input. Active only when both
    # FEI and σ feature blocks are enabled (variance balance is the whole
    # point — with one modality off LN either no-ops or destroys the 2-D
    # FEI simplex's magnitude). Parameterless (``elementwise_affine=False``)
    # so the state_dict format is unchanged; the on/off semantics live in
    # the ``ss_chimera_freq_router_layer_norm`` metadata stamp.
    freq_router_layer_norm: bool = True
    # Freq-pool routing MODE. "learned" (default): FreqRouter MLP over
    # concat(FEI, σ-features) (paper-faithful). "fei": hardwire
    # ``π_f = normalize(FEI ** (1/τ))`` — no params/σ-input/freq-balance,
    # requires num_experts_freq == fei_feature_dim. FEI is already the
    # load-bearing per-prompt simplex at low σ (σ doesn't discriminate), so
    # hardwiring keeps 100% of the signal. Stamped to ss_chimera_freq_router_mode.
    freq_router_mode: str = "learned"
    # Temperature on the hardwired FEI gate (freq_router_mode="fei" only).
    # τ=1.0 = raw-FEI passthrough; τ<1 sharpens crossover, τ>1 flattens.
    # Inert under "learned".
    freq_router_tau: float = 1.0
    # Per-pool router LR multipliers (chimera-only), stacked on router_lr_scale:
    # effective LR = unet_lr × router_lr_scale × <pool>_router_lr_scale. Bumping
    # ``content`` to 5–10× is a faster lever than balance_w_content when the
    # content pool stays near-uniform.
    content_router_lr_scale: float = 1.0
    freq_router_lr_scale: float = 1.0
    # ChimeraHydra content router: one network-level ``ContentRouter`` on pooled
    # ``crossattn_emb`` (1024-D for Anima); π_c broadcast via
    # ``_content_routing_weights`` like π_f. Only content-routing mode (per-Linear
    # ``lx_c`` softmax removed). ``content_router_layer_norm`` = parameterless LN.
    content_router_layer_norm: bool = True
    # ContentRouter output-layer init magnitude. 0.0 (zero-init) = uniform π_c at
    # step 0 → ΔW_c=0. Zero is NOT a fixed point here (disjoint P_bases_c·λ_c +
    # per-prompt input break symmetry), so nonzero is purely a plateau-kick for
    # the "usage uniform but content_margin≈0" regime — it loses the exact-identity
    # start and seeds a usage skew the balance loss must undo. Keep small, opt-in.
    content_router_init_std: float = 0.0

    # ChimeraHydra centered-gate λ init (BOTH pools), always on: gates recentered
    # to ``π - 1/K``, routers zero-init, λ_c/λ_f start at chimera_lambda_init.
    # Gives ΔW=0 at init yet nonzero router gradient (disjoint P-subspaces survive
    # mean subtraction); balance loss sees the RAW simplex. ≤0 floored to 1e-2.
    chimera_lambda_init: float = 1e-2

    # Per-expert capability levers (frozen-Cayley chimera only — orthogonality-
    # PRESERVING alternative to use_ortho_init). Both distill to the standard
    # up-stack (inference/on-disk unchanged); mutually exclusive with use_ortho_init.
    #   * ``chimera_expert_basis_mult`` (m≥1): over-complete ``(out, m·r)`` frozen
    #     pool from a DISJOINT U-slice + m·r Cayley; forward selects an r-dim
    #     Stiefel subspace (trainable colspace, disjoint across experts, no
    #     collapse). Auto-downgrades per-layer on overflow; m=1 = canonical r-slice.
    #   * ``chimera_expert_diag``: per-expert ``(K, r)`` trainable diagonal σ
    #     (init 1) — the singular spectrum the ortho-only path lacks. ΔW=0 still holds.
    chimera_expert_basis_mult: int = 1
    chimera_expert_diag: bool = False

    # Step-expert (turbo per-step head split). >1 → ``StepExpertLoRAModule``:
    # shared ``lora_down`` + K up-heads selected by diffusion step (no router).
    # 0/1 = inactive. See ``networks/lora_modules/step_expert.py``.
    step_expert_K: int = 0

    # SmoothQuant-style per-channel input pre-scaling
    channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None

    # DSR-style learnable register tokens trained jointly with the LoRA
    # (docs/proposal/headroom_register_tokens.md; shared machinery with the
    # standalone register method via networks/register_injection.py). K tokens
    # enter the self-attn sequence at block ``register_insert_block`` (DSR's
    # "starting block" — Tab. 9 sweet spot 8, 0 = stack entry), ride to the
    # end of the stack, and are stripped before unpatchify. 0 = off (default).
    # Registers can't merge into DiT weights → the checkpoint becomes
    # kept-live at inference (is_mergeable() False, static merge refused).
    num_registers: int = 0
    register_insert_block: int = 8
    # Registers get their own optimizer group at unet_lr × this scale — they
    # must grow ~14–24× the median patch norm to become sinks, which a
    # LoRA-scale lr rarely reaches (proposal §metrics).
    register_lr_scale: float = 100.0
    # E25b explicit resolution conditioning (--sigma_lowres_res_cond): one
    # zero-init projection (t_emb_dim, 256) on the network under the dot-free
    # key "sigma_lowres_res_cond_proj" (register_tokens precedent — LoRA
    # key-sniffers and merge grouping skip it; merge.py refuses it as
    # non-bakeable). Trained by ordinary backprop at unet_lr; the trainer
    # attaches (proj, s) on the DiT per forward.
    sigma_lowres_res_cond: bool = False
    # E25e re-centered variant: the delta is W·(φ(s) − φ(0)) so native (s=0)
    # forwards are bit-identical to control for ANY projection value (and the
    # projection gets zero gradient from them). Same tensor footprint; the
    # variants are distinguished only by the ss_sigma_lowres_res_cond
    # metadata stamp ("centered" vs "true").
    sigma_lowres_res_cond_centered: bool = False
    register_init_std: float = 0.02

    verbose: bool = False

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Mapping[str, Any],
        *,
        network_dim: Optional[int],
        network_alpha: Optional[float],
        neuron_dropout: Optional[float],
        module_class: Type,
        channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "LoRANetworkCfg":
        """Build cfg from train.py's stringified ``net_kwargs`` dict."""
        if network_dim is None:
            network_dim = 4
        if network_alpha is None:
            network_alpha = 1.0

        train_llm_adapter = _as_bool(kwargs.get("train_llm_adapter"))

        exclude_patterns = _as_str_list(kwargs.get("exclude_patterns")) or []
        exclude_patterns.append(_DEFAULT_EXCLUDE)
        include_patterns = _as_str_list(kwargs.get("include_patterns"))

        # adaln convenience knobs (mirror the turbo distill surface): train_adaln
        # adds the adaln_up_{branch} Linears to the target set — they sit in
        # _DEFAULT_EXCLUDE, so this rescues them via include_patterns (an
        # exclude-override, not a whitelist, so the default attn+MLP set is
        # untouched). adaln_rank / adaln_alpha give them their own rank / alpha
        # (0/absent rank = the network's; 0/absent alpha = derived from
        # network_dim/network_alpha by the √r law, below). Translates to the exact
        # include_patterns / network_reg_dims / network_reg_alphas primitives the
        # turbo harness builds by hand (networks/methods/turbo_dmd.py); injected
        # into reg_dims / reg_alphas after those strings are parsed, below.
        train_adaln = _as_bool(kwargs.get("train_adaln"))
        adaln_rank_raw = kwargs.get("adaln_rank")
        adaln_rank = int(adaln_rank_raw) if adaln_rank_raw is not None else 0
        adaln_alpha_raw = kwargs.get("adaln_alpha")
        adaln_alpha = float(adaln_alpha_raw) if adaln_alpha_raw is not None else 0.0
        if adaln_rank > 0 and not train_adaln:
            raise ValueError("adaln_rank > 0 requires train_adaln = true")
        if adaln_alpha > 0 and not train_adaln:
            raise ValueError("adaln_alpha > 0 requires train_adaln = true")

        layer_start = kwargs.get("layer_start")
        layer_start = int(layer_start) if layer_start is not None else None
        layer_end = kwargs.get("layer_end")
        layer_end = int(layer_end) if layer_end is not None else None

        rank_dropout = kwargs.get("rank_dropout")
        rank_dropout = float(rank_dropout) if rank_dropout is not None else None
        module_dropout = kwargs.get("module_dropout")
        module_dropout = float(module_dropout) if module_dropout is not None else None

        use_timestep_mask = _as_bool(kwargs.get("use_timestep_mask"))
        min_rank = kwargs.get("min_rank")
        min_rank = int(min_rank) if min_rank is not None else 1
        alpha_rank_scale = kwargs.get("alpha_rank_scale")
        alpha_rank_scale = (
            float(alpha_rank_scale) if alpha_rank_scale is not None else 1.0
        )

        num_experts = kwargs.get("num_experts")
        num_experts = int(num_experts) if num_experts is not None else 4
        expert_init_std = float(kwargs.get("expert_init_std", 0.0))

        ortho_centered_gate = _as_bool(kwargs.get("ortho_centered_gate"))
        ortho_lambda_init = float(kwargs.get("ortho_lambda_init", 0.0))
        if ortho_centered_gate and ortho_lambda_init <= 0.0:
            # Centering with λ0=0 is a no-op (router-logit grad ∝ (P_k-mean)diag(λ0)ℓ
            # vanishes); 1e-2 fires the mechanism and survives bf16 (1e-6 lost to noise).
            ortho_lambda_init = 1e-2
            logger.info(
                "ortho_centered_gate=True with ortho_lambda_init<=0; "
                "defaulting ortho_lambda_init=1e-2 (centering needs λ0>0)."
            )

        router_lr_scale = kwargs.get("network_router_lr_scale")
        router_lr_scale = float(router_lr_scale) if router_lr_scale is not None else 1.0

        down_init = str(kwargs.get("down_init", "kaiming"))
        if down_init not in ("kaiming", "weight_svd"):
            raise ValueError(
                f"down_init={down_init!r}: expected 'kaiming' or 'weight_svd'."
            )

        _legacy_router_keys = [
            k
            for k in ("hydra_router_layers", "sigma_router_layers", "fei_router_layers")
            if k in kwargs
        ]
        if _legacy_router_keys:
            raise ValueError(
                f"{_legacy_router_keys} are no longer supported — the three "
                "router layer filters were consolidated into a single "
                "`router_targets` regex. Replace them with one `router_targets = "
                "...` entry in your method TOML."
            )
        router_targets = kwargs.get("router_targets", None)
        per_bucket_balance_weight = kwargs.get("per_bucket_balance_weight")
        per_bucket_balance_weight = (
            float(per_bucket_balance_weight)
            if per_bucket_balance_weight is not None
            else 0.3
        )
        num_sigma_buckets = int(kwargs.get("num_sigma_buckets", 3))
        specialize_experts_by_sigma_buckets = _as_bool(
            kwargs.get("specialize_experts_by_sigma_buckets")
        )
        sigma_bucket_boundaries = _as_float_list(kwargs.get("sigma_bucket_boundaries"))
        if specialize_experts_by_sigma_buckets:
            if num_sigma_buckets <= 1:
                raise ValueError(
                    "specialize_experts_by_sigma_buckets requires num_sigma_buckets > 1, "
                    f"got num_sigma_buckets={num_sigma_buckets}."
                )
            if num_experts % num_sigma_buckets != 0:
                raise ValueError(
                    "specialize_experts_by_sigma_buckets requires num_experts to be "
                    f"divisible by num_sigma_buckets, got num_experts={num_experts}, "
                    f"num_sigma_buckets={num_sigma_buckets}."
                )
            if sigma_bucket_boundaries is not None:
                _validate_sigma_bucket_boundaries(
                    sigma_bucket_boundaries, num_sigma_buckets
                )
        elif sigma_bucket_boundaries is not None:
            logger.warning(
                "sigma_bucket_boundaries set but "
                "specialize_experts_by_sigma_buckets is off — boundaries ignored."
            )
            sigma_bucket_boundaries = None

        sigma_feature_dim = int(kwargs.get("sigma_feature_dim", 16))

        fei_feature_dim = int(kwargs.get("fei_feature_dim", 2))
        fei_sigma_low_div = float(kwargs.get("fei_sigma_low_div", 4.0))

        # GlobalRouter knobs (only consumed when ``route_per_layer=False``).
        router_hidden_dim = int(
            kwargs.get("router_hidden_dim", kwargs.get("router_hidden", 64))
        )
        router_tau = float(kwargs.get("router_tau", 0.7))

        use_ortho = _as_bool(kwargs.get("use_ortho"))
        ortho_init_std = float(kwargs.get("ortho_init_std", 0.02))
        use_ortho_init = _as_bool(kwargs.get("use_ortho_init"))

        # FECL knobs. Default off; turning it on requires `num_bands >= 3`
        # to be a meaningful objective (see compute_fecl docstring).
        fera_fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0))
        fera_num_bands = int(kwargs.get("fera_num_bands", kwargs.get("num_bands", 3)))

        # ChimeraHydra knobs. ``num_experts`` (parent Hydra cfg) is treated
        # as a derived value when ``use_chimera_hydra=True`` — recomputed
        # below so users only set K_c / K_f.
        use_chimera_hydra = _as_bool(kwargs.get("use_chimera_hydra"))
        num_experts_content = int(kwargs.get("num_experts_content", 3))
        num_experts_freq = int(kwargs.get("num_experts_freq", 3))
        balance_w_content_raw = kwargs.get("balance_w_content")
        balance_w_content = (
            float(balance_w_content_raw) if balance_w_content_raw is not None else None
        )
        balance_w_freq_raw = kwargs.get("balance_w_freq")
        balance_w_freq = (
            float(balance_w_freq_raw) if balance_w_freq_raw is not None else None
        )
        freq_router_init_std = float(kwargs.get("freq_router_init_std", 0.1))
        freq_router_layer_norm = _as_bool(kwargs.get("freq_router_layer_norm", True))
        freq_router_mode = (
            str(kwargs.get("freq_router_mode", "learned")).strip().lower() or "learned"
        )
        if freq_router_mode not in ("learned", "fei"):
            raise ValueError(
                f"freq_router_mode={freq_router_mode!r}: expected 'learned' or 'fei'."
            )
        freq_router_tau = float(kwargs.get("freq_router_tau", 1.0))
        content_router_lr_scale = float(
            kwargs.get("network_content_router_lr_scale", 1.0)
        )
        freq_router_lr_scale = float(kwargs.get("network_freq_router_lr_scale", 1.0))
        content_router_layer_norm = _as_bool(
            kwargs.get("content_router_layer_norm", True), default=True
        )
        content_router_init_std = float(kwargs.get("content_router_init_std", 0.0))
        # Chimera is always centered-gate; centering with λ0=0 is a no-op (each
        # router's logit gradient ∝ (P_k - mean)·diag(λ0)·ℓ vanishes at λ0=0),
        # so floor to a small nonzero default.
        chimera_lambda_init = float(kwargs.get("chimera_lambda_init", 1e-2))
        if use_chimera_hydra and chimera_lambda_init <= 0.0:
            chimera_lambda_init = 1e-2
            logger.info(
                "chimera_lambda_init<=0 floored to 1e-2 (centering needs λ0>0)."
            )
        chimera_expert_basis_mult = int(kwargs.get("chimera_expert_basis_mult", 1))
        chimera_expert_diag = _as_bool(kwargs.get("chimera_expert_diag"))
        if use_chimera_hydra:
            if num_experts_content <= 0 or num_experts_freq <= 0:
                raise ValueError(
                    "use_chimera_hydra=True requires num_experts_content > 0 "
                    f"and num_experts_freq > 0 (got K_c={num_experts_content}, "
                    f"K_f={num_experts_freq})."
                )
            if use_ortho_init and (
                chimera_expert_basis_mult > 1 or chimera_expert_diag
            ):
                raise ValueError(
                    "chimera_expert_basis_mult/chimera_expert_diag require the "
                    "frozen-Cayley path (use_ortho_init=false) — they are the "
                    "orthogonality-preserving alternative to ortho_init, which "
                    "already frees the bases."
                )
            if chimera_expert_basis_mult < 1:
                raise ValueError(
                    "chimera_expert_basis_mult must be >= 1 "
                    f"(got {chimera_expert_basis_mult})."
                )
            if freq_router_mode == "fei" and num_experts_freq != fei_feature_dim:
                raise ValueError(
                    "freq_router_mode='fei' hardwires the freq gate to the FEI "
                    "band-simplex, so num_experts_freq must equal fei_feature_dim "
                    f"(got K_f={num_experts_freq}, fei_feature_dim={fei_feature_dim}). "
                    "Either set num_experts_freq=fei_feature_dim, or use "
                    "freq_router_mode='learned' for an MLP that maps any input "
                    "width to K_f experts."
                )
        # Three-axis routing resolution (plan2.md §three-axis-config). The
        # legacy ``use_hydra`` / ``use_sigma_router`` / ``use_fei_router``
        # kwargs were retired in plan2 task #6 — every shipped TOML uses the
        # new keys, and old `.safetensors` files (with ``ss_use_hydra`` etc.)
        # stop loading by design (no legacy compat shim).
        raw_moe_style = kwargs.get("use_moe_style")
        raw_route_per_layer = kwargs.get("route_per_layer")
        raw_router_source = kwargs.get("router_source")

        for legacy_key in ("use_hydra", "use_sigma_router", "use_fei_router"):
            if kwargs.get(legacy_key) is not None:
                raise ValueError(
                    f"Legacy router kwarg {legacy_key!r} is no longer "
                    "supported. Use the three-axis keys instead: "
                    "`use_moe_style` (False / 'shared_A' / 'independent_A'), "
                    "`route_per_layer` (true / false), and `router_source` "
                    "('none' / 'input' / 'sigma' / 'fei' / 'crossattn_emb'). "
                    "See plan2.md §three-axis-config."
                )

        use_moe_style: MoEStyle = (
            _as_moe_style(raw_moe_style) if raw_moe_style is not None else False
        )

        if raw_router_source is not None:
            router_source: RouterSource = _as_router_source(raw_router_source)
        elif use_moe_style is not False:
            # Hydra's default router input is the per-Linear input vector.
            router_source = "input"
        else:
            router_source = "none"

        if raw_route_per_layer is not None:
            route_per_layer = _as_bool(raw_route_per_layer)
        else:
            # no-MoE = no router; Hydra defaults to per-layer
            route_per_layer = use_moe_style is not False

        # ChimeraHydra: pin the three-axis cells to (shared_A, per-layer,
        # input) regardless of TOML wiring. The chimera content router IS
        # a per-layer shared_A Hydra router on pooled lx; the freq router
        # adds a second routing source on top via a dedicated network-level
        # mechanism. Stamping these three values means the save metadata
        # flows through the standard MoE branch and the loader can detect
        # the chimera-specific stamps without a parallel three-axis path.
        if use_chimera_hydra:
            if use_moe_style not in (False, "shared_A"):
                raise ValueError(
                    "use_chimera_hydra=True is only compatible with "
                    "use_moe_style='shared_A' (or unset); got "
                    f"use_moe_style={use_moe_style!r}."
                )
            if raw_route_per_layer is not None and not _as_bool(raw_route_per_layer):
                raise ValueError(
                    "use_chimera_hydra=True requires route_per_layer=True "
                    "(content router is per-layer)."
                )
            if raw_router_source is not None and raw_router_source != "input":
                raise ValueError(
                    "use_chimera_hydra=True requires router_source='input' "
                    "(content router reads pooled lx); σ/FEI are owned by "
                    "the network-level FreqRouter."
                )
            use_moe_style = "shared_A"
            route_per_layer = True
            router_source = "input"

        # SVD-Down (down_init="weight_svd") targets the plain LoRAModule only —
        # the ortho/Hydra/Chimera classes own their own SVD seeding and the
        # network.py pass-through is gated to LoRAModule, so a non-plain variant
        # would silently ignore it. Fail loudly instead of no-op'ing. T-LoRA
        # (use_timestep_mask) is fine — it stays on LoRAModule.
        if down_init != "kaiming" and (
            use_ortho
            or use_ortho_init
            or use_moe_style is not False
            or use_chimera_hydra
        ):
            raise ValueError(
                f"down_init={down_init!r} only applies to plain LoRA, but a "
                "non-plain variant is selected (use_ortho / use_ortho_init / "
                "use_moe_style / use_chimera_hydra). Disable those to use SVD-Down, "
                "or keep down_init='kaiming'."
            )

        # Validate impossible combos.
        if use_moe_style is False and (route_per_layer or router_source != "none"):
            raise ValueError(
                "Routing config requires use_moe_style != False; got "
                f"use_moe_style={use_moe_style!r}, route_per_layer={route_per_layer}, "
                f"router_source={router_source!r}."
            )
        if not route_per_layer and router_source == "input":
            raise ValueError(
                "router_source='input' requires route_per_layer=True — no "
                "network-level 'input' signal exists per DiT forward."
            )
        if route_per_layer and router_source == "crossattn_emb":
            raise ValueError(
                "router_source='crossattn_emb' requires route_per_layer=False — "
                "the pooled cross-attention text feature is a single per-sample "
                "vector routed by one network-level GlobalRouter, with no "
                "per-Linear variant."
            )

        step_expert_K_raw = kwargs.get("step_expert_K")
        step_expert_K = int(step_expert_K_raw) if step_expert_K_raw is not None else 0

        reg_dims_str = kwargs.get("network_reg_dims")
        reg_dims = _parse_kv_pairs(reg_dims_str, is_int=True) if reg_dims_str else None
        reg_lrs_str = kwargs.get("network_reg_lrs")
        reg_lrs = _parse_kv_pairs(reg_lrs_str, is_int=False) if reg_lrs_str else None
        reg_alphas_str = kwargs.get("network_reg_alphas")
        reg_alphas = (
            _parse_kv_pairs(reg_alphas_str, is_int=False) if reg_alphas_str else None
        )

        if train_adaln:
            _adaln_pat = ".*adaln_up_.*"
            include_patterns = (include_patterns or []) + [_adaln_pat]
            if adaln_rank > 0:
                reg_dims = {**(reg_dims or {}), _adaln_pat: adaln_rank}
            if adaln_alpha <= 0:
                # Derive from the network's own rank/alpha instead of inheriting
                # network_alpha at a smaller rank (which runs the adaln modules
                # network_dim/adaln_rank hotter in alpha/rank). √r law
                # (alpha ∝ √r — docs/methods/adaln.md), matching
                # networks/methods/easycontrol.py. adaln_rank = 0 shares the
                # network rank, so the factor is 1 and this is a no-op.
                _r = adaln_rank if adaln_rank > 0 else network_dim
                adaln_alpha = network_alpha * math.sqrt(_r / max(network_dim, 1))
            reg_alphas = {**(reg_alphas or {}), _adaln_pat: adaln_alpha}

        # DSR register tokens (LoRA + registers trained jointly). Bounds of
        # register_insert_block are validated at network build (needs n_blocks).
        num_registers = int(kwargs.get("num_registers", 0) or 0)
        if num_registers < 0:
            raise ValueError(f"num_registers must be >= 0, got {num_registers}")
        register_insert_block = int(kwargs.get("register_insert_block", 8))
        register_lr_scale = float(kwargs.get("register_lr_scale", 100.0))
        register_init_std = float(kwargs.get("register_init_std", 0.02))

        # E25b explicit resolution conditioning (train.py injects this from
        # --sigma_lowres_res_cond; string-bool via network_args also works).
        # E25e: the value "centered" selects the re-centered W·(φ(s) − φ(0))
        # variant — one string, so the two bools can never disagree.
        _res_cond_raw = str(kwargs.get("sigma_lowres_res_cond", "")).strip().lower()
        sigma_lowres_res_cond = _res_cond_raw in ("true", "1", "yes", "centered")
        sigma_lowres_res_cond_centered = _res_cond_raw == "centered"

        verbose = _as_bool(kwargs.get("verbose"))

        return cls(
            lora_dim=network_dim,
            alpha=network_alpha,
            module_class=module_class,
            train_llm_adapter=train_llm_adapter,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            layer_start=layer_start,
            layer_end=layer_end,
            dropout=neuron_dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            reg_dims=reg_dims,
            reg_lrs=reg_lrs,
            reg_alphas=reg_alphas,
            use_timestep_mask=use_timestep_mask,
            min_rank=min_rank,
            alpha_rank_scale=alpha_rank_scale,
            num_experts=num_experts,
            expert_init_std=expert_init_std,
            ortho_centered_gate=ortho_centered_gate,
            ortho_lambda_init=ortho_lambda_init,
            router_lr_scale=router_lr_scale,
            router_targets=router_targets,
            per_bucket_balance_weight=per_bucket_balance_weight,
            num_sigma_buckets=num_sigma_buckets,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            sigma_bucket_boundaries=sigma_bucket_boundaries,
            use_moe_style=use_moe_style,
            route_per_layer=route_per_layer,
            router_source=router_source,
            sigma_feature_dim=sigma_feature_dim,
            fei_feature_dim=fei_feature_dim,
            fei_sigma_low_div=fei_sigma_low_div,
            router_hidden_dim=router_hidden_dim,
            router_tau=router_tau,
            use_ortho=use_ortho,
            ortho_init_std=ortho_init_std,
            use_ortho_init=use_ortho_init,
            down_init=down_init,
            fera_fecl_weight=fera_fecl_weight,
            fera_num_bands=fera_num_bands,
            use_chimera_hydra=use_chimera_hydra,
            num_experts_content=num_experts_content,
            num_experts_freq=num_experts_freq,
            balance_w_content=balance_w_content,
            balance_w_freq=balance_w_freq,
            freq_router_init_std=freq_router_init_std,
            freq_router_layer_norm=freq_router_layer_norm,
            freq_router_mode=freq_router_mode,
            freq_router_tau=freq_router_tau,
            content_router_lr_scale=content_router_lr_scale,
            freq_router_lr_scale=freq_router_lr_scale,
            content_router_layer_norm=content_router_layer_norm,
            content_router_init_std=content_router_init_std,
            chimera_lambda_init=chimera_lambda_init,
            chimera_expert_basis_mult=chimera_expert_basis_mult,
            chimera_expert_diag=chimera_expert_diag,
            step_expert_K=step_expert_K,
            channel_scales_dict=channel_scales_dict,
            num_registers=num_registers,
            register_insert_block=register_insert_block,
            register_lr_scale=register_lr_scale,
            register_init_std=register_init_std,
            sigma_lowres_res_cond=sigma_lowres_res_cond,
            sigma_lowres_res_cond_centered=sigma_lowres_res_cond_centered,
            verbose=verbose,
        )

    @classmethod
    def from_weights(
        cls,
        *,
        modules_dim: Dict[str, int],
        modules_alpha: Dict[str, float],
        module_class: Type,
        train_llm_adapter: bool,
        is_hydra_or_ortho_hydra: bool,
        hydra_num_experts: int,
        sigma_feature_dim_detected: Optional[int],
        sigma_router_names: Optional[List[str]],
        hydra_router_names: Optional[List[str]],
        channel_scales_dict: Optional[Dict[str, torch.Tensor]],
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: Optional[int] = None,
        sigma_bucket_boundaries: Optional[List[float]] = None,
        fei_feature_dim: int = 0,
        fei_sigma_low_div: Optional[float] = None,
        fei_router_names: Optional[List[str]] = None,
        is_stacked_experts: bool = False,
        # Three-axis stamps from save metadata. All three must be present
        # for MoE checkpoints — pre-plan2 artifacts stop loading by design.
        new_use_moe_style: Optional[str] = None,
        new_route_per_layer: Optional[bool] = None,
        new_router_source: Optional[str] = None,
        ortho_centered_gate: bool = False,
        # ChimeraHydra stamps. Present only on chimera checkpoints — when
        # set the loader builds ``ChimeraHydraLoRAModule`` instead of
        # ``OrthoHydraLoRAModule`` and the network attaches a FreqRouter.
        is_chimera_hydra: bool = False,
        num_experts_content: Optional[int] = None,
        num_experts_freq: Optional[int] = None,
        freq_router_layer_norm: bool = False,
        freq_router_mode: str = "learned",
        freq_router_tau: float = 1.0,
        content_router_layer_norm: bool = True,
        step_expert_K: int = 0,
        # Register tokens: K sniffed from the ``register_tokens`` key's shape,
        # insert block from the ``ss_register_insert_block`` metadata stamp.
        num_registers: int = 0,
        register_insert_block: int = 8,
        # E25b res-cond: presence of the dot-free "sigma_lowres_res_cond_proj"
        # key (factory sniff) — the param must exist before load_state_dict.
        # E25e centered: from the ss_sigma_lowres_res_cond metadata stamp
        # value ("centered") — no tensor footprint distinguishes the variants.
        sigma_lowres_res_cond: bool = False,
        sigma_lowres_res_cond_centered: bool = False,
    ) -> "LoRANetworkCfg":
        """Build cfg from a checkpoint key-sniff (warm-start / inference path).

        Mirrors the ``LoRANetwork(...)`` call previously embedded in
        ``create_network_from_weights``. Per-module dims / alphas come from
        ``modules_dim`` / ``modules_alpha``, so ``lora_dim`` / ``alpha`` here
        are placeholders. Training-time schedules (warmup, T-LoRA) stay off
        in the warm-start path.

        ``specialize_experts_by_sigma_buckets`` / ``num_sigma_buckets`` /
        ``sigma_bucket_boundaries`` come from safetensors metadata stamped by
        ``save_weights`` — the partition leaves no tensor footprint
        (``_expert_band`` / ``_sigma_edges`` are non-persistent) so it has to
        be reconstructed from those scalars at load time.

        For non-MoE checkpoints (plain LoRA / OrthoLoRA / T-LoRA) the
        three-axis stamps are not stamped at save time; absence is taken as
        ``(False, False, "none")``. MoE checkpoints (Hydra / OrthoHydra /
        StackedExperts) must carry all three stamps — plan2 task #6 retired
        the legacy ``ss_use_hydra`` / ``ss_use_fei_router`` fallback.
        """
        if (
            new_use_moe_style is not None
            and new_route_per_layer is not None
            and new_router_source is not None
        ):
            use_moe_style: MoEStyle = _as_moe_style(new_use_moe_style)
            route_per_layer = bool(new_route_per_layer)
            router_source: RouterSource = _as_router_source(new_router_source)
        elif is_hydra_or_ortho_hydra or is_stacked_experts:
            raise RuntimeError(
                "MoE checkpoint is missing the three-axis routing stamps "
                "(ss_use_moe_style / ss_route_per_layer / ss_router_source). "
                "Two common causes: (1) it is a pre-plan2 checkpoint, which "
                "stops loading by design — retrain the adapter to produce the "
                "new metadata; or (2) you passed a pre-loaded weights_sd= to "
                "create_network_from_weights without file= or metadata=. "
                "load_file() drops safetensors __metadata__, so the stamps "
                "vanish — pass file=<path> or metadata=<dict> so they survive."
            )
        else:
            use_moe_style = False
            route_per_layer = False
            router_source = "none"

        # ChimeraHydra requires both pool sizes to be stamped at save time;
        # absence on a flagged checkpoint indicates malformed metadata.
        if is_chimera_hydra:
            if num_experts_content is None or num_experts_freq is None:
                raise RuntimeError(
                    "ChimeraHydra checkpoint missing ss_num_experts_content / "
                    "ss_num_experts_freq metadata — checkpoint is malformed."
                )
            if (
                hydra_num_experts
                and hydra_num_experts != num_experts_content + num_experts_freq
            ):
                raise RuntimeError(
                    "ChimeraHydra checkpoint K_c + K_f mismatch: stamped "
                    f"K_c={num_experts_content}, K_f={num_experts_freq}, "
                    f"detected num_experts={hydra_num_experts}."
                )

        return cls(
            lora_dim=4,
            alpha=1.0,
            module_class=module_class,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            train_llm_adapter=train_llm_adapter,
            num_experts=(
                hydra_num_experts
                if (is_hydra_or_ortho_hydra or is_stacked_experts)
                else 4
            ),
            channel_scales_dict=channel_scales_dict,
            use_moe_style=use_moe_style,
            route_per_layer=route_per_layer,
            router_source=router_source,
            ortho_centered_gate=bool(ortho_centered_gate),
            sigma_feature_dim=(
                sigma_feature_dim_detected
                if sigma_feature_dim_detected is not None
                else 128
            ),
            sigma_router_names=sigma_router_names,
            hydra_router_names=hydra_router_names,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            num_sigma_buckets=(int(num_sigma_buckets) if num_sigma_buckets else 3),
            sigma_bucket_boundaries=sigma_bucket_boundaries,
            fei_feature_dim=int(fei_feature_dim),
            fei_sigma_low_div=(
                float(fei_sigma_low_div) if fei_sigma_low_div is not None else 4.0
            ),
            fei_router_names=fei_router_names,
            use_chimera_hydra=is_chimera_hydra,
            num_experts_content=(
                int(num_experts_content) if num_experts_content is not None else 3
            ),
            num_experts_freq=(
                int(num_experts_freq) if num_experts_freq is not None else 3
            ),
            freq_router_layer_norm=bool(freq_router_layer_norm),
            freq_router_mode=(
                str(freq_router_mode).strip().lower() if freq_router_mode else "learned"
            ),
            freq_router_tau=float(freq_router_tau),
            content_router_layer_norm=bool(content_router_layer_norm),
            step_expert_K=int(step_expert_K),
            sigma_lowres_res_cond=bool(sigma_lowres_res_cond),
            sigma_lowres_res_cond_centered=bool(sigma_lowres_res_cond_centered),
            num_registers=int(num_registers),
            register_insert_block=int(register_insert_block),
        )
