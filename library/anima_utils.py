# Anima model loading/saving utilities

import os
import re
from typing import Dict, List, Optional, Union
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from accelerate import init_empty_weights

from networks.lora_utils import load_safetensors_with_lora
from library import anima_models
from library.safetensors_utils import WeightTransformHooks
from .utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _strip_net_prefix(key: str) -> str:
    return key[len("net.") :] if key.startswith("net.") else key


# Regex patterns for fused projection key remapping (compiled once)
# Only match DiT blocks (blocks.N.), not LLM adapter blocks (llm_adapter.blocks.N.)
_SELF_ATTN_QKV_RE = re.compile(
    r"(blocks\.\d+\.self_attn)\.(q_proj|k_proj|v_proj)(\.weight)"
)
_CROSS_ATTN_KV_RE = re.compile(r"(blocks\.\d+\.cross_attn)\.(k_proj|v_proj)(\.weight)")
_ADALN_DOWN_RE = re.compile(
    r"(blocks\.\d+)\.adaln_modulation_(self_attn|cross_attn|mlp)\.1(\.weight)"
)
_ADALN_UP_RE = re.compile(
    r"(blocks\.\d+)\.adaln_modulation_(self_attn|cross_attn|mlp)\.2\."
)

_SELF_ATTN_QKV_ORDER = ("q_proj", "k_proj", "v_proj")
_CROSS_ATTN_KV_ORDER = ("k_proj", "v_proj")
_ADALN_BRANCH_ORDER = ("self_attn", "cross_attn", "mlp")


def _dit_rename_hook(key: str) -> str:
    """1:1 key renames: strip net. prefix and remap AdaLN up-projection keys."""
    k = _strip_net_prefix(key)
    # blocks.N.adaln_modulation_{branch}.2.weight -> blocks.N.adaln_up_{branch}.weight
    k = _ADALN_UP_RE.sub(lambda m: f"{m.group(1)}.adaln_up_{m.group(2)}.", k)
    return k


def _dit_concat_hook(
    key: str, tensors: "dict[str, torch.Tensor] | None"
) -> "tuple[str | None, torch.Tensor | None]":
    """Many-to-one key fusions for fused projections.

    Phase 1 (tensors=None): return the fused target key.
    Phase 2 (tensors=dict): concatenate tensors in correct order and return.
    """
    clean = _strip_net_prefix(key)

    # Self-attention QKV: q_proj + k_proj + v_proj -> qkv_proj
    m = _SELF_ATTN_QKV_RE.match(clean)
    if m:
        fused_key = f"{m.group(1)}.qkv_proj{m.group(3)}"
        if tensors is None:
            return fused_key, None
        prefix = m.group(1)
        ordered = []
        for proj in _SELF_ATTN_QKV_ORDER:
            for orig_key, t in tensors.items():
                if _strip_net_prefix(orig_key) == f"{prefix}.{proj}{m.group(3)}":
                    ordered.append(t)
                    break
        return fused_key, torch.cat(ordered, dim=0)

    # Cross-attention KV: k_proj + v_proj -> kv_proj
    m = _CROSS_ATTN_KV_RE.match(clean)
    if m:
        fused_key = f"{m.group(1)}.kv_proj{m.group(3)}"
        if tensors is None:
            return fused_key, None
        prefix = m.group(1)
        ordered = []
        for proj in _CROSS_ATTN_KV_ORDER:
            for orig_key, t in tensors.items():
                if _strip_net_prefix(orig_key) == f"{prefix}.{proj}{m.group(3)}":
                    ordered.append(t)
                    break
        return fused_key, torch.cat(ordered, dim=0)

    # AdaLN fused down: 3 separate down-projections -> one fused
    m = _ADALN_DOWN_RE.match(clean)
    if m:
        fused_key = f"{m.group(1)}.adaln_fused_down.1{m.group(3)}"
        if tensors is None:
            return fused_key, None
        block_prefix = m.group(1)
        ordered = []
        for branch in _ADALN_BRANCH_ORDER:
            for orig_key, t in tensors.items():
                if (
                    _strip_net_prefix(orig_key)
                    == f"{block_prefix}.adaln_modulation_{branch}.1{m.group(3)}"
                ):
                    ordered.append(t)
                    break
        return fused_key, torch.cat(ordered, dim=0)

    return None, None


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    attn_softmax_scale: Optional[float] = None,
) -> anima_models.Anima:
    """
    Load Anima model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, weights are loaded as-is from the state_dict; otherwise they are cast to this dtype.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    if dit_weight_dtype is None:
        dit_weight_dtype = torch.bfloat16

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # We currently support fixed DiT config for Anima models
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
        "attn_softmax_scale": attn_softmax_scale,
    }
    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    rename_hooks = WeightTransformHooks(
        rename_hook=_dit_rename_hook,
        concat_hook=_dit_concat_hook,
    )
    sd = load_safetensors_with_lora(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        weight_transform_hooks=rename_hooks,
    )

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [
            k
            for k in missing
            if not any(
                buf_name in k
                for buf_name in (
                    "seq",
                    "dim_spatial_range",
                    "dim_temporal_range",
                    "inv_freq",
                    "pooled_text_proj",
                )
            )
        ]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}"
            )
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(
            f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    logger.info(
        f"Loaded DiT model from {dit_path}, unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
    )

    # Move non-checkpoint modules (RoPE embeddings, pooled_text_proj) to the correct device.
    # These are created on CPU during __init__ and not present in the checkpoint,
    # so load_state_dict(assign=True) doesn't move them — they remain as meta tensors.
    # Use to_empty() + init to materialize meta tensors, then move to loading_device.
    if hasattr(model, "pos_embedder"):
        model.pos_embedder.to(loading_device)
    if hasattr(model, "pooled_text_proj"):
        model.pooled_text_proj.to_empty(device=loading_device)
        # Re-init all layers: to_empty() leaves uninitialized memory (may contain NaN).
        for m in model.pooled_text_proj:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        # Zero-init output layer so the module is a no-op at init (same as init_weights)
        torch.nn.init.zeros_(model.pooled_text_proj[-1].weight)
        torch.nn.init.zeros_(model.pooled_text_proj[-1].bias)

    return model


def load_pooled_text_proj(
    model: anima_models.Anima,
    path: str,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Load trained pooled_text_proj weights into the model."""
    from safetensors.torch import load_file

    state = load_file(path, device=str(device))
    model.pooled_text_proj.load_state_dict(state, assign=True)
    logger.info(f"Loaded pooled_text_proj from {path}")


def load_llm_adapter(
    dit_path: Optional[str],
    llm_adapter_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
) -> anima_models.LLMAdapter:
    """Load only the LLM adapter weights.

    This is useful for caching adapter outputs without loading the full DiT model.

    Args:
        dit_path: Path to the DiT safetensors file (used when llm_adapter_path is None).
        llm_adapter_path: Optional path to a separate adapter safetensors file.
        dtype: Target dtype.
        device: Target device.
    """
    weight_path = llm_adapter_path or dit_path
    if weight_path is None:
        raise ValueError("Either dit_path or llm_adapter_path must be provided")
    if os.path.splitext(weight_path)[1].lower() != ".safetensors":
        raise ValueError(
            f"LLM adapter weights must be a .safetensors file, got: {weight_path}"
        )

    # Detect prefix style
    with safe_open(weight_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    if any(k.startswith("llm_adapter.") for k in keys):
        prefix = "llm_adapter."
    elif any(k.startswith("net.llm_adapter.") for k in keys):
        prefix = "net.llm_adapter."
    else:
        prefix = None  # adapter-only weights file (no prefix)

    state_dict: Dict[str, torch.Tensor] = {}
    with safe_open(weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if prefix is not None:
                if not key.startswith(prefix):
                    continue
                stripped = key[len(prefix) :]
            else:
                stripped = key
            state_dict[stripped] = f.get_tensor(key)

    if prefix is not None and len(state_dict) == 0:
        raise ValueError(f"No llm_adapter weights found in {weight_path}")

    adapter = anima_models.LLMAdapter(
        source_dim=1024,
        target_dim=1024,
        model_dim=1024,
        num_layers=6,
        self_attn=True,
    )
    missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning(
            f"Unexpected keys in LLM adapter weights: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
    if missing:
        logger.warning(
            f"Missing keys in LLM adapter weights: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    return adapter


def _is_qwen3_5(path: str) -> bool:
    """Detect whether a path refers to a Qwen3.5 model (vs Qwen3)."""
    basename = os.path.basename(path).lower()
    return "qwen_3_5" in basename or "qwen3_5" in basename or "qwen3.5" in basename


def _get_qwen_config_dir(qwen3_path: str) -> str:
    """Return the appropriate bundled config directory for the given model path."""
    configs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    if _is_qwen3_5(qwen3_path):
        config_dir = os.path.join(configs_root, "qwen3_5_08b")
        model_name = "Qwen3.5-0.8B"
    else:
        config_dir = os.path.join(configs_root, "qwen3_06b")
        model_name = "Qwen3-0.6B"

    if not os.path.exists(config_dir):
        raise FileNotFoundError(
            f"{model_name} config directory not found at {config_dir}. "
            f"Expected config.json, tokenizer.json, etc. "
            f"You can download these from the {model_name} HuggingFace repository."
        )
    return config_dir


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3/Qwen3.5 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, auto-detects Qwen3 vs Qwen3.5 from filename and uses
                     the appropriate bundled config directory.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = _get_qwen_config_dir(qwen3_path)
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Load Qwen3 or Qwen3.5 text encoder.

    Auto-detects Qwen3.5 from filename (looks for 'qwen_3_5', 'qwen3_5', or 'qwen3.5').

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    is_qwen35 = _is_qwen3_5(qwen3_path)
    model_label = "Qwen3.5" if is_qwen35 else "Qwen3"
    logger.info(f"Loading {model_label} text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            qwen3_path, torch_dtype=dtype, local_files_only=True
        ).model
    else:
        # Single safetensors file - use bundled config
        config_dir = _get_qwen_config_dir(qwen3_path)

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

        if is_qwen35:
            qwen_config = transformers.Qwen3_5TextConfig.from_pretrained(
                config_dir, local_files_only=True
            )
            model = transformers.Qwen3_5ForCausalLM(qwen_config).model
        else:
            qwen_config = transformers.Qwen3Config.from_pretrained(
                config_dir, local_files_only=True
            )
            model = transformers.Qwen3ForCausalLM(qwen_config).model

        # Load weights
        if qwen3_path.endswith(".safetensors"):
            if lora_weights is None:
                state_dict = load_file(qwen3_path, device="cpu")
            else:
                state_dict = load_safetensors_with_lora(
                    model_files=qwen3_path,
                    lora_weights_list=lora_weights,
                    lora_multipliers=lora_multipliers,
                    calc_device=device,
                    move_to_device=True,
                    dit_weight_dtype=None,
                )
        else:
            assert lora_weights is None, (
                "LoRA weights merging is only supported for safetensors checkpoints"
            )
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model.") :]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded {model_label} state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(
        f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs", "t5_old"
    )
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


def save_anima_model(
    save_path: str,
    dit_state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, any],
    dtype: Optional[torch.dtype] = None,
):
    """Save Anima DiT model with 'net.' prefix for ComfyUI compatibility.

    Args:
        save_path: Output path (.safetensors)
        dit_state_dict: State dict from dit.state_dict()
        metadata: Metadata dict to include in the safetensors file
        dtype: Optional dtype to cast to before saving
    """
    prefixed_sd = {}
    for k, v in dit_state_dict.items():
        if dtype is not None:
            # v = v.to(dtype)
            v = (
                v.detach().clone().to("cpu").to(dtype)
            )  # Reduce GPU memory usage during save
        prefixed_sd["net." + k] = v.contiguous()

    if metadata is None:
        metadata = {}
    metadata["format"] = "pt"  # For compatibility with the official .safetensors file

    save_file(
        prefixed_sd, save_path, metadata=metadata
    )  # safetensors.save_file cosumes a lot of memory, but Anima is small enough
    logger.info(f"Saved Anima model to {save_path}")
