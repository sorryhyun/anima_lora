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
    LoRAInfModule,
    LoRAModule,
    OrthoHydraLoRAExpModule,
    OrthoLoRAExpModule,
    ReFTModule,
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
                    f"file with `python bench/archive/analyze_lora_input_channels.py "
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

        Mirrors ``set_timestep_mask`` — one call per training step, propagates
        σ to every module that opts in (``sigma_feature_dim > 0``). Other
        modules ignore it. The network also caches σ on itself so the per-
        σ-bucket balance loss can bucket the cached gates at loss-compute
        time, and so router stats can report per-σ-bucket expert usage even
        when ``use_sigma_router=False`` (sigma-independent gating can still
        develop sigma-correlated usage from data drift).
        """
        sigmas = sigmas.detach()
        self._last_sigma = sigmas
        if not self.cfg.use_sigma_router:
            return
        for lora in self.unet_loras + self.text_encoder_loras:
            if getattr(lora, "sigma_feature_dim", 0) > 0:
                lora._sigma = sigmas

    def clear_sigma(self) -> None:
        """Reset cached σ on every HydraLoRA module to zeros.

        Never set to None: ``_sigma`` stays a Tensor so the unconditional
        sinusoidal path in ``_compute_gate`` has no None-vs-Tensor guard to
        recompile on under ``compile_mode=full``. Used in eval / validation
        and by inference teardown (``clear_hydra_sigma``).
        """
        self._last_sigma = None
        for lora in self.unet_loras + self.text_encoder_loras:
            sigma = getattr(lora, "_sigma", None)
            if sigma is not None:
                lora._sigma = torch.zeros_like(sigma)

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

        Must be called AFTER backward (so .grad is populated) and AFTER DDP
        all-reduce (so norms reflect the global gradient direction), but
        BEFORE optimizer.step. Outside the warmup window this is a no-op.

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

    def get_router_stats(self) -> Dict[str, Union[float, List[float], List[List[float]], List[int]]]:
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

        Empty dict when no hydra module cached a gate this step. One reduction
        per module on (B, E) gates — cheap.
        """
        per_H: list[float] = []
        per_margin: list[float] = []
        per_usage: list[torch.Tensor] = []
        per_bucket_usage: list[torch.Tensor] = []  # each (num_buckets, E)
        E_ref: Optional[int] = None

        sigma = self._last_sigma  # (B,) or None
        num_buckets = int(self.cfg.num_sigma_buckets)
        want_per_bucket = sigma is not None and num_buckets > 1
        bucket_ids: Optional[torch.Tensor] = None
        bucket_counts_t: Optional[torch.Tensor] = None
        if want_per_bucket:
            thresholds = torch.linspace(
                0.0, 1.0, num_buckets + 1, device=sigma.device
            )[1:-1]
            bucket_ids = torch.bucketize(sigma.float(), thresholds)  # (B,) in [0, N)
            bucket_counts_t = torch.zeros(
                num_buckets, device=sigma.device, dtype=torch.long
            )
            bucket_counts_t.scatter_add_(
                0, bucket_ids, torch.ones_like(bucket_ids, dtype=torch.long)
            )

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
                # Skip modules with mismatched expert count — aggregating
                # usage vectors of different length isn't meaningful.
                continue

            p = gate.float().clamp_min(1e-12)
            H = -(p * p.log()).sum(dim=-1)  # (B,)
            per_H.append((H.mean() / math.log(E)).detach().item())

            top2 = p.topk(2, dim=-1).values  # (B, 2)
            per_margin.append((top2[:, 0] - top2[:, 1]).mean().detach().item())

            expert_idx = gate.argmax(dim=-1)  # (B,)
            usage = torch.zeros(E, device=gate.device, dtype=gate.dtype)
            usage.scatter_add_(
                0, expert_idx, torch.ones_like(expert_idx, dtype=gate.dtype)
            )
            per_usage.append((usage / gate.shape[0]).detach())

            if want_per_bucket and bucket_ids is not None:
                # Per-bucket argmax frequency, normalized within each bucket
                # so each row sums to ~1 (or 0 for empty buckets).
                bu = torch.zeros(
                    num_buckets, E, device=gate.device, dtype=gate.dtype
                )
                # Combined index = bucket * E + expert -> flat scatter_add
                flat_idx = bucket_ids.to(expert_idx.device) * E + expert_idx
                bu.view(-1).scatter_add_(
                    0, flat_idx, torch.ones_like(flat_idx, dtype=gate.dtype)
                )
                bc = bucket_counts_t.to(gate.dtype).clamp_min(1).unsqueeze(-1)
                per_bucket_usage.append((bu / bc).detach())

        if not per_H:
            return {}

        H_t = torch.tensor(per_H)
        q = torch.quantile(H_t, torch.tensor([0.05, 0.5, 0.95]))
        usage_mean = torch.stack(per_usage).mean(dim=0).cpu().tolist()
        out: Dict[str, Union[float, List[float], List[List[float]], List[int]]] = {
            "entropy_mean": float(H_t.mean().item()),
            "entropy_p05": float(q[0].item()),
            "entropy_p50": float(q[1].item()),
            "entropy_p95": float(q[2].item()),
            "margin_mean": float(sum(per_margin) / len(per_margin)),
            "expert_usage": usage_mean,
        }
        if per_bucket_usage and bucket_counts_t is not None:
            bucket_usage_mean = (
                torch.stack(per_bucket_usage).mean(dim=0).cpu().tolist()
            )
            out["expert_usage_per_bucket"] = bucket_usage_mean
            out["bucket_counts"] = bucket_counts_t.cpu().tolist()
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
