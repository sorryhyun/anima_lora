"""Anima Prefix/Postfix conditioning node for ComfyUI.

Loads learned prefix or postfix vectors from a safetensors file and applies
them to T5-compatible conditioning embeddings. Prefix vectors are prepended
before text tokens; postfix vectors are inserted after the last real text token.

Only apply to positive conditioning -- do NOT connect to negative conditioning.
"""

import logging
from typing import Dict, Tuple

import torch

import folder_paths

logger = logging.getLogger(__name__)

# Cache: path -> (mode, embeds_tensor, num_tokens)
_weight_cache: Dict[str, Tuple[str, torch.Tensor, int]] = {}


def _load_weights(file_path: str) -> Tuple[str, torch.Tensor, int]:
    """Load and auto-detect prefix/postfix weights from safetensors."""
    if file_path in _weight_cache:
        return _weight_cache[file_path]

    from safetensors.torch import load_file

    weights_sd = load_file(file_path)

    if "prefix_embeds" in weights_sd:
        mode = "prefix"
        embeds = weights_sd["prefix_embeds"]
    elif "postfix_embeds" in weights_sd:
        mode = "postfix"
        embeds = weights_sd["postfix_embeds"]
    else:
        raise ValueError(
            f"Unsupported weight file (keys: {list(weights_sd.keys())}). "
            f"Expected 'prefix_embeds' or 'postfix_embeds'."
        )

    num_tokens = embeds.shape[0]
    result = (mode, embeds, num_tokens)
    _weight_cache[file_path] = result
    logger.info(
        f"Loaded {mode} weights: {num_tokens} tokens, dim {embeds.shape[1]} from {file_path}"
    )
    return result


def _prepend_prefix(
    crossattn_emb: torch.Tensor, prefix_embeds: torch.Tensor
) -> torch.Tensor:
    """Prepend prefix vectors, trimming trailing padding to maintain seq length."""
    K = prefix_embeds.shape[0]
    B = crossattn_emb.shape[0]
    prefix = (
        prefix_embeds.unsqueeze(0)
        .expand(B, -1, -1)
        .to(dtype=crossattn_emb.dtype, device=crossattn_emb.device)
    )
    return torch.cat(
        [prefix, crossattn_emb[:, : crossattn_emb.shape[1] - K]], dim=1
    )


def _append_postfix(
    crossattn_emb: torch.Tensor, postfix_embeds: torch.Tensor
) -> torch.Tensor:
    """Insert postfix vectors right after real text tokens (overwriting zero-padding)."""
    K = postfix_embeds.shape[0]
    B, S, D = crossattn_emb.shape
    postfix = (
        postfix_embeds.unsqueeze(0)
        .expand(B, -1, -1)
        .to(dtype=crossattn_emb.dtype, device=crossattn_emb.device)
    )
    # Detect real token count from zero-padding
    mask = crossattn_emb.abs().sum(dim=-1) > 0  # [B, S]
    seqlens = mask.long().sum(dim=-1)  # [B]

    offsets = seqlens.unsqueeze(1) + torch.arange(
        K, device=crossattn_emb.device
    )  # [B, K]
    offsets = offsets.clamp(max=S - 1)
    idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
    return crossattn_emb.scatter(1, idx, postfix)


class AnimaPrefixPostfix:
    """Apply learned prefix or postfix vectors to Anima T5-compatible conditioning.

    Mode (prefix vs postfix) is auto-detected from the safetensors weight keys.
    Connect to POSITIVE conditioning only.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "weight_file": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "Safetensors file with prefix_embeds or postfix_embeds."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength multiplier for the prefix/postfix vectors.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Apply learned prefix or postfix vectors to Anima conditioning. "
        "Auto-detects mode from safetensors keys. "
        "Connect to positive conditioning only -- do not use on negative."
    )

    def apply(self, conditioning, weight_file, strength=1.0):
        if strength == 0:
            return (conditioning,)

        file_path = folder_paths.get_full_path("loras", weight_file)
        mode, embeds, num_tokens = _load_weights(file_path)

        embeds_scaled = embeds * strength

        out = []
        for cond_tensor, cond_dict in conditioning:
            modified = cond_tensor.clone()
            if mode == "prefix":
                modified = _prepend_prefix(modified, embeds_scaled)
            else:
                modified = _append_postfix(modified, embeds_scaled)
            out.append((modified, cond_dict.copy()))

        return (out,)


NODE_CLASS_MAPPINGS = {
    "AnimaPrefixPostfix": AnimaPrefixPostfix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaPrefixPostfix": "Anima Prefix/Postfix",
}
