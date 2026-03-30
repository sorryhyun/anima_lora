# Anima model loading/saving utilities

import os
from typing import Dict, List, Optional, Union
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from accelerate import init_empty_weights

from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library import anima_models
from library.safetensors_utils import WeightTransformHooks
from .utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# Original Anima high-precision keys. Kept for reference, but not used currently.
# # Keys that should stay in high precision (float32/bfloat16, not quantized)
# KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks", ""]
# ".embed." excludes Embedding in LLMAdapter
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_embedder", "norm", "adaln", "final_layer", ".embed."]


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
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
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (
        not fp8_scaled and dit_weight_dtype is not None
    ) or dit_weight_dtype is None, (
        "dit_weight_dtype should be None when fp8_scaled is True"
    )

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
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": 28,
        "num_heads": 16,
        "extra_per_block_abs_pos_emb": False,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": True,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
        "attn_softmax_scale": attn_softmax_scale,
    }
    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    rename_hooks = WeightTransformHooks(
        rename_hook=lambda k: k[len("net.") :] if k.startswith("net.") else k
    )
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=rename_hooks,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

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

    # Move non-checkpoint buffers (RoPE embeddings) to the correct device.
    # These are created on CPU during __init__ and not present in the checkpoint,
    # so load_state_dict(assign=True) doesn't move them.
    if hasattr(model, "pos_embedder"):
        model.pos_embedder.to(loading_device)
    if hasattr(model, "extra_pos_embedder"):
        model.extra_pos_embedder.to(loading_device)

    return model


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
                state_dict = load_safetensors_with_lora_and_fp8(
                    model_files=qwen3_path,
                    lora_weights_list=lora_weights,
                    lora_multipliers=lora_multipliers,
                    fp8_optimization=False,
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
