# LoRA network module for Anima
import ast
import os
import re
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from library.utils import setup_logging
from networks.lora_modules import LoRAModule, LoRAInfModule, DoRAModule, OrthoLoRAModule, ReFTModule, HydraLoRAModule

import logging

setup_logging()
logger = logging.getLogger(__name__)

_BLOCK_IDX_RE = re.compile(r"blocks\.(\d+)\.")


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

    # train LLM adapter
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

    # add default exclude patterns
    exclude_patterns.append(r".*(_modulation|_norm|_embedder|final_layer).*")

    # regular expression for module selection: exclude and include
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        try:
            include_patterns = ast.literal_eval(include_patterns)
            if not isinstance(include_patterns, list):
                include_patterns = [include_patterns]
        except (ValueError, SyntaxError):
            include_patterns = [include_patterns]

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # DoRA mode
    use_dora = kwargs.get("use_dora", "false")
    if use_dora is not None:
        use_dora = True if use_dora.lower() == "true" else False

    # OrthoLoRA mode
    use_ortho = kwargs.get("use_ortho", "false")
    if use_ortho is not None:
        use_ortho = True if use_ortho.lower() == "true" else False
    sig_type = kwargs.get("sig_type", "last")
    ortho_reg_weight = kwargs.get("ortho_reg_weight", None)
    ortho_reg_weight = float(ortho_reg_weight) if ortho_reg_weight is not None else 0.01

    # Timestep-dependent rank masking
    use_timestep_mask = kwargs.get("use_timestep_mask", "false")
    if use_timestep_mask is not None:
        use_timestep_mask = True if use_timestep_mask.lower() == "true" else False
    min_rank = kwargs.get("min_rank", None)
    min_rank = int(min_rank) if min_rank is not None else 1
    alpha_rank_scale = kwargs.get("alpha_rank_scale", None)
    alpha_rank_scale = float(alpha_rank_scale) if alpha_rank_scale is not None else 1.0

    # ReFT (additive representation fine-tuning)
    add_reft = kwargs.get("add_reft", "false")
    if add_reft is not None:
        add_reft = True if add_reft.lower() == "true" else False
    reft_dim = kwargs.get("reft_dim", None)
    reft_dim = int(reft_dim) if reft_dim is not None else network_dim

    # HydraLoRA (MoE-style multi-head routing)
    use_hydra = kwargs.get("use_hydra", "false")
    if use_hydra is not None:
        use_hydra = True if use_hydra.lower() == "true" else False
    num_experts = kwargs.get("num_experts", None)
    num_experts = int(num_experts) if num_experts is not None else 12
    balance_loss_weight = kwargs.get("balance_loss_weight", None)
    balance_loss_weight = float(balance_loss_weight) if balance_loss_weight is not None else 0.01

    # verbose
    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if verbose.lower() == "true" else False

    # regex-specific learning rates / dimensions
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

    network_reg_dims = kwargs.get("network_reg_dims", None)
    if network_reg_dims is not None:
        reg_dims = parse_kv_pairs(network_reg_dims, is_int=True)
    else:
        reg_dims = None

    if use_hydra:
        module_class = HydraLoRAModule
    elif use_dora:
        module_class = DoRAModule
    elif use_ortho:
        module_class = OrthoLoRAModule
    else:
        module_class = LoRAModule

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
        sig_type=sig_type,
        layer_start=layer_start,
        layer_end=layer_end,
        add_reft=add_reft,
        reft_dim=reft_dim,
        num_experts=num_experts,
    )

    # Set timestep mask and ortho regularization config
    network._use_timestep_mask = use_timestep_mask
    network._min_rank = min_rank
    network._max_rank = network_dim
    network._alpha_rank_scale = alpha_rank_scale
    network._ortho_reg_weight = ortho_reg_weight if use_ortho else 0.0

    network._add_reft = add_reft
    network._reft_dim = reft_dim

    # HydraLoRA router and config
    network._use_hydra = use_hydra
    network._balance_loss_weight = balance_loss_weight if use_hydra else 0.0
    if use_hydra:
        network._hydra_router = torch.nn.Linear(1024, num_experts, bias=True, dtype=unet.dtype)
        torch.nn.init.xavier_uniform_(network._hydra_router.weight)
        torch.nn.init.zeros_(network._hydra_router.bias)

    if use_timestep_mask:
        logger.info(
            f"Timestep-dependent rank masking: min_rank={min_rank}, alpha={alpha_rank_scale}"
        )
    if use_ortho:
        logger.info(
            f"OrthoLoRA: sig_type={sig_type}, ortho_reg_weight={ortho_reg_weight}"
        )
    if use_hydra:
        logger.info(
            f"HydraLoRA: num_experts={num_experts}, balance_loss_weight={balance_loss_weight}"
        )
    if add_reft:
        logger.info(f"ReFT: reft_dim={reft_dim}")
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

    modules_dim = {}
    modules_alpha = {}
    reft_modules_dim = {}
    reft_modules_alpha = {}
    train_llm_adapter = False
    has_dora = False
    has_ortho = False
    has_hydra = False
    hydra_num_experts = 0
    has_reft = False
    reft_dim = None
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]

        # Skip router keys
        if "_hydra_router" in key:
            continue

        # ReFT keys use "reft_" prefix
        if lora_name.startswith("reft_"):
            has_reft = True
            if "alpha" in key:
                reft_modules_alpha[lora_name] = value
            elif "rotate_layer" in key and "weight" in key:
                reft_dim = value.size()[0]
                reft_modules_dim[lora_name] = reft_dim
            continue

        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_up_weight" in key:
            has_hydra = True
            hydra_num_experts = max(hydra_num_experts, value.size(0))
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
        elif "q_layer" in key and "weight" in key and "base_" not in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            has_ortho = True
        elif "dora_scale" in key:
            has_dora = True

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    if for_inference:
        module_class = LoRAInfModule
    elif has_hydra:
        module_class = HydraLoRAModule
    elif has_dora:
        module_class = DoRAModule
    elif has_ortho:
        module_class = OrthoLoRAModule
    else:
        module_class = LoRAModule

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
        reft_modules_dim=reft_modules_dim if has_reft else None,
        reft_modules_alpha=reft_modules_alpha if has_reft else None,
        num_experts=hydra_num_experts if has_hydra else 12,
    )
    # Recreate router if loading HydraLoRA weights
    if has_hydra and "_hydra_router.weight" in weights_sd:
        router_dim = weights_sd["_hydra_router.weight"].shape[0]
        router_dtype = weights_sd["_hydra_router.weight"].dtype
        network._hydra_router = torch.nn.Linear(1024, router_dim, bias=True, dtype=router_dtype)
        network._use_hydra = True
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
        sig_type: str = "last",
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        add_reft: bool = False,
        reft_dim: int = 4,
        reft_modules_dim: Optional[Dict[str, int]] = None,
        reft_modules_alpha: Optional[Dict[str, int]] = None,
        num_experts: int = 12,
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
        self.sig_type = sig_type
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_experts = num_experts

        self._hydra_router = None  # set by create_network() if use_hydra=True

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
                            if is_unet and (self.layer_start is not None or self.layer_end is not None):
                                block_match = _BLOCK_IDX_RE.match(original_name)
                                if block_match:
                                    block_idx = int(block_match.group(1))
                                    if self.layer_start is not None and block_idx < self.layer_start:
                                        if verbose:
                                            logger.info(f"layer_range exclude: {original_name} (block {block_idx} < {self.layer_start})")
                                        continue
                                    if self.layer_end is not None and block_idx >= self.layer_end:
                                        if verbose:
                                            logger.info(f"layer_range exclude: {original_name} (block {block_idx} >= {self.layer_end})")
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
                extra_kwargs = {}
                if module_class == OrthoLoRAModule:
                    extra_kwargs["sig_type"] = self.sig_type
                elif module_class == HydraLoRAModule:
                    extra_kwargs["num_experts"] = self.num_experts

                lora = module_class(
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
        if text_encoders is not None and module_class != OrthoLoRAModule:
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

        # Create ReFT modules (additive representation intervention, orthogonal to LoRA)
        self.unet_refts: List[ReFTModule] = []
        self.text_encoder_refts: List[ReFTModule] = []
        if add_reft:
            from tqdm import tqdm

            def create_reft_for_loras(
                loras: list,
                prefix_from: str,
                prefix_to: str,
                reft_modules_dim_: Optional[Dict[str, int]],
                reft_modules_alpha_: Optional[Dict[str, int]],
            ) -> List[ReFTModule]:
                refts = []
                for lora in tqdm(loras, desc=f"Creating ReFT ({prefix_to})", leave=False):
                    # Skip Conv2d modules
                    if not isinstance(lora.org_module, torch.nn.Linear):
                        continue
                    reft_name = lora.lora_name.replace(prefix_from, prefix_to, 1)
                    if reft_modules_dim_ is not None:
                        if reft_name not in reft_modules_dim_:
                            continue
                        dim = reft_modules_dim_[reft_name]
                        a = reft_modules_alpha_.get(reft_name, dim)
                    else:
                        dim = reft_dim
                        a = alpha
                    reft = ReFTModule(
                        reft_name,
                        lora.org_module,
                        multiplier=multiplier,
                        reft_dim=dim,
                        alpha=a,
                        dropout=dropout,
                        module_dropout=module_dropout,
                    )
                    reft.original_name = lora.original_name
                    refts.append(reft)
                return refts

            self.unet_refts = create_reft_for_loras(
                self.unet_loras,
                self.LORA_PREFIX_ANIMA,
                "reft_unet",
                reft_modules_dim,
                reft_modules_alpha,
            )
            logger.info(f"create ReFT for Anima DiT: {len(self.unet_refts)} modules (reft_dim={reft_dim})")

            if self.text_encoder_loras:
                self.text_encoder_refts = create_reft_for_loras(
                    self.text_encoder_loras,
                    self.LORA_PREFIX_TEXT_ENCODER,
                    "reft_te",
                    reft_modules_dim,
                    reft_modules_alpha,
                )
                logger.info(f"create ReFT for Text Encoder: {len(self.text_encoder_refts)} modules")

        # assertion: no duplicate names
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras + self.text_encoder_refts + self.unet_refts:
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

    def prepare_network(self, args):
        if getattr(args, "lora_fp32_accumulation", False):
            logger.info("enabling fp32 accumulation for LoRA modules")
            for lora in self.text_encoder_loras + self.unet_loras:
                lora.fp32_accumulation = True

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
        t = timesteps.float().mean().item()
        r = (
            int(
                ((max_timestep - t) / max_timestep) ** self._alpha_rank_scale
                * (self._max_rank - self._min_rank)
            )
            + self._min_rank
        )
        r = min(r, self._max_rank)  # clamp

        # Reuse a single GPU-resident mask to avoid ~200 CPU→GPU transfers per step
        mask = getattr(self, "_shared_timestep_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, self._max_rank, device=timesteps.device)
            self._shared_timestep_mask = mask
            for lora in self.text_encoder_loras + self.unet_loras:
                lora._timestep_mask = mask
        mask.zero_()
        mask[:, :r] = 1.0

    def set_reft_timestep_mask(self, timesteps: torch.Tensor, max_timestep: float = 1.0):
        """Compute and set timestep-dependent mask on ReFT modules."""
        if not getattr(self, "_use_timestep_mask", False):
            return
        refts = self.text_encoder_refts + self.unet_refts
        if not refts:
            return
        reft_dim = getattr(self, "_reft_dim", 4)
        t = timesteps.float().mean().item()
        r = (
            int(
                ((max_timestep - t) / max_timestep) ** self._alpha_rank_scale
                * (reft_dim - 1)
            )
            + 1
        )
        r = min(r, reft_dim)

        mask = getattr(self, "_shared_reft_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, reft_dim, device=timesteps.device)
            self._shared_reft_mask = mask
            for reft in refts:
                reft._timestep_mask = mask
        mask.zero_()
        mask[:, :r] = 1.0

    def clear_timestep_mask(self):
        """Remove timestep mask (use full rank)."""
        self._shared_timestep_mask = None
        for lora in self.text_encoder_loras + self.unet_loras:
            lora._timestep_mask = None
        self._shared_reft_mask = None
        for reft in self.text_encoder_refts + self.unet_refts:
            reft._timestep_mask = None

    def set_hydra_gate(self, crossattn_emb: torch.Tensor):
        """Compute gate weights from crossattn_emb and propagate to all HydraLoRA modules.

        Args:
            crossattn_emb: (B, seq_len, 1024) cross-attention embeddings
        """
        if self._hydra_router is None:
            return
        # Max pool over sequence dimension: (B, 1024)
        pooled = crossattn_emb.max(dim=1).values
        # Router: (B, num_experts)
        gate = torch.softmax(self._hydra_router(pooled), dim=-1)
        # Store for balance loss
        self._last_hydra_gate = gate
        # Propagate to all HydraLoRA modules
        for lora in self.unet_loras:
            if hasattr(lora, "_hydra_gate"):
                lora._hydra_gate = gate

    def get_balance_loss(self) -> torch.Tensor:
        """Switch Transformer-style load-balancing loss for HydraLoRA routing."""
        gate = getattr(self, "_last_hydra_gate", None)
        if gate is None:
            return torch.tensor(0.0)
        num_experts = gate.shape[-1]
        # frac_i: fraction of samples where expert i has highest gate
        expert_idx = gate.argmax(dim=-1)  # (B,)
        frac = torch.zeros(num_experts, device=gate.device)
        frac.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=frac.dtype))
        frac = frac / gate.shape[0]
        # gate_mean_i: mean gate value for expert i across batch
        gate_mean = gate.mean(dim=0)  # (num_experts,)
        return num_experts * (frac * gate_mean).sum()

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

        # Rename dora_scale → magnitude for DoRA modules
        for key in list(weights_sd.keys()):
            if key.endswith(".dora_scale"):
                new_key = key.replace(".dora_scale", ".magnitude")
                weights_sd[new_key] = weights_sd.pop(key)

        # Migrate old lora_ups.N.weight → fused lora_up_weight
        ups_prefixes: dict[str, dict[int, torch.Tensor]] = {}
        for key in list(weights_sd.keys()):
            if ".lora_ups." in key and key.endswith(".weight"):
                prefix = key.split(".lora_ups.")[0]
                idx = int(key.split("lora_ups.")[1].split(".")[0])
                ups_prefixes.setdefault(prefix, {})[idx] = weights_sd.pop(key)
        for prefix, experts in ups_prefixes.items():
            stacked = torch.stack([experts[i] for i in sorted(experts.keys())])
            weights_sd[f"{prefix}.lora_up_weight"] = stacked

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

        # Apply LoRA first
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # Apply ReFT after LoRA (wraps LoRA's forward, so chain is: ReFT -> LoRA -> original)
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
            param_groups = {"lora": {}, "plus": {}}
            reg_groups = {}
            reg_lrs_list = (
                list(self.reg_lrs.items()) if self.reg_lrs is not None else []
            )

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
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {
                                "lora": {},
                                "plus": {},
                                "lr": reg_lr,
                            }
                        if loraplus_ratio is not None and (
                            "lora_up" in name or "p_layer" in name or "learned_source" in name
                        ):
                            reg_groups[group_key]["plus"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        else:
                            reg_groups[group_key]["lora"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        continue

                    if loraplus_ratio is not None and (
                        "lora_up" in name or "p_layer" in name or "learned_source" in name
                    ):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = (
                            reg_lr * loraplus_ratio
                            if loraplus_ratio is not None
                            else reg_lr
                        )
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
                    descriptions.append(desc + (" plus" if key == "plus" else ""))

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr
                if (
                    param_data.get("lr", None) == 0
                    or param_data.get("lr", None) is None
                ):
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
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

        # ReFT modules (grouped alongside their LoRA counterparts)
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

        # HydraLoRA router parameters
        if self._hydra_router is not None:
            router_lr = unet_lr if unet_lr is not None else default_lr
            router_params = {"params": list(self._hydra_router.parameters())}
            if router_lr is not None:
                router_params["lr"] = router_lr
            all_params.append(router_params)
            lr_descriptions.append("hydra_router")

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
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        # OrthoLoRA → standard LoRA conversion for ComfyUI compatibility
        # Compute ΔW = P·diag(λ)·Q - P_base·diag(λ_base)·Q_base, then SVD → lora_up/lora_down
        ortho_prefixes = set()
        for key in state_dict.keys():
            if key.endswith(".base_lambda"):
                ortho_prefixes.add(key[: -len(".base_lambda")])

        for prefix in ortho_prefixes:
            P = state_dict[f"{prefix}.p_layer.weight"]  # (out_dim, rank)
            Q = state_dict[f"{prefix}.q_layer.weight"]  # (rank, in_dim)
            lam = state_dict[f"{prefix}.lambda_layer"]  # (1, rank)
            P_base = state_dict[f"{prefix}.base_p_weight"]  # (out_dim, rank)
            Q_base = state_dict[f"{prefix}.base_q_weight"]  # (rank, in_dim)
            lam_base = state_dict[f"{prefix}.base_lambda"]  # (1, rank)
            alpha = state_dict.get(f"{prefix}.alpha")
            rank = Q.shape[0]

            # ΔW = P·diag(λ)·Q - P_base·diag(λ_base)·Q_base is rank ≤ 2r.
            # Instead of materializing the full (out_dim × in_dim) matrix and running SVD on it,
            # work in the small 2r-dimensional column/row space:
            #   ΔW = [P|P_base] @ M @ [Q; Q_base]  where M is (2r, 2r)
            # Then SVD of M gives us the decomposition cheaply.
            svd_device = "cuda" if torch.cuda.is_available() else "cpu"
            save_dtype = dtype if dtype is not None else P.dtype

            P_cat = torch.cat([P, P_base], dim=1).float().to(svd_device)       # (out, 2r)
            Q_cat = torch.cat([Q, Q_base], dim=0).float().to(svd_device)       # (2r, in)
            lam_diag = torch.diag(lam.squeeze(0).float().to(svd_device))       # (r, r)
            lam_base_diag = torch.diag(lam_base.squeeze(0).float().to(svd_device))

            # M = block_diag(diag(λ), -diag(λ_base))  — the middle (2r, 2r) matrix
            M = torch.zeros(2 * rank, 2 * rank, device=svd_device)
            M[:rank, :rank] = lam_diag
            M[rank:, rank:] = -lam_base_diag

            # QR-orthogonalize the tall/wide factors to get thin SVD via small matrix
            Qp, Rp = torch.linalg.qr(P_cat)   # Qp: (out, 2r), Rp: (2r, 2r)
            Qq, Rq = torch.linalg.qr(Q_cat.T)  # Qq: (in, 2r), Rq: (2r, 2r)

            # Core (2r × 2r) matrix whose SVD gives us the answer
            core = Rp @ M @ Rq.T
            Uc, Sc, Vhc = torch.linalg.svd(core)  # all (2r, 2r)

            # Map back: U_full = Qp @ Uc, Vh_full = Vhc @ Qq.T
            lora_up = (
                (Qp @ Uc[:, :rank] * Sc[:rank].sqrt().unsqueeze(0))
                .to(save_dtype)
                .cpu()
                .contiguous()
            )
            lora_down = (
                (Sc[:rank].sqrt().unsqueeze(1) * Vhc[:rank, :] @ Qq.T)
                .to(save_dtype)
                .cpu()
                .contiguous()
            )

            # Remove OrthoLoRA keys
            for suffix in [
                "p_layer.weight",
                "q_layer.weight",
                "lambda_layer",
                "base_p_weight",
                "base_q_weight",
                "base_lambda",
            ]:
                state_dict.pop(f"{prefix}.{suffix}", None)

            # Add standard LoRA keys
            state_dict[f"{prefix}.lora_up.weight"] = lora_up
            state_dict[f"{prefix}.lora_down.weight"] = lora_down
            if alpha is not None:
                state_dict[f"{prefix}.alpha"] = alpha

        # HydraLoRA: save full multi-head format alongside baked-down version
        hydra_prefixes = set()
        for key in list(state_dict.keys()):
            if key.endswith(".lora_up_weight"):
                hydra_prefixes.add(key.removesuffix(".lora_up_weight"))

        if hydra_prefixes:
            # Save full multi-head format (experts + router) for custom node inference
            # Expand fused lora_up_weight back to per-expert lora_ups.N.weight keys
            hydra_file = os.path.splitext(file)[0] + "_hydra.safetensors"
            hydra_sd = {}
            for k, v in state_dict.items():
                v = v.detach().clone().to("cpu")
                if k.endswith(".lora_up_weight"):
                    prefix = k.removesuffix(".lora_up_weight")
                    for i in range(v.size(0)):
                        hydra_sd[f"{prefix}.lora_ups.{i}.weight"] = v[i]
                else:
                    hydra_sd[k] = v
            if dtype is not None:
                hydra_sd = {k: v.to(dtype) for k, v in hydra_sd.items()}
            from safetensors.torch import save_file as sf_save
            sf_save(hydra_sd, hydra_file, metadata or {})
            logger.info(f"HydraLoRA full format saved to {hydra_file}")

        # HydraLoRA → standard LoRA: average expert up-projections for ComfyUI compatibility
        for prefix in hydra_prefixes:
            fused_key = f"{prefix}.lora_up_weight"
            if fused_key not in state_dict:
                continue
            # lora_up_weight is (num_experts, out_dim, lora_dim) → average to (out_dim, lora_dim)
            state_dict[f"{prefix}.lora_up.weight"] = state_dict.pop(fused_key).mean(dim=0)

        # Remove HydraLoRA router keys (not needed for baked-down inference)
        for key in list(state_dict.keys()):
            if "_hydra_router" in key:
                del state_dict[key]

        # DoRA: rename magnitude → dora_scale for ComfyUI, remove internal buffers
        for key in list(state_dict.keys()):
            if key.endswith(".magnitude"):
                new_key = key.replace(".magnitude", ".dora_scale")
                state_dict[new_key] = state_dict.pop(key)
            elif key.endswith("._org_weight_norm"):
                del state_dict[key]

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(
                state_dict, metadata
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

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
