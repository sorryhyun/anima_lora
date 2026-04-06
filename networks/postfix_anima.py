# Postfix tuning network module for Anima LLM Adapter
#
# Learns N continuous postfix vectors appended to the adapter's cross-attention
# source (Qwen3 side). These discover quality signals in embedding space that
# improve generation across all artist tags.
#
# Modes:
#   "hidden"    — postfix injected into adapter's source (Qwen3 hidden-state space)
#   "embedding" — postfix injected into Qwen3 input embeddings (goes through all layers)
#   "cfg"       — two postfix sets (pos + neg) trained with CFG-aware loss
#   "dual"      — postfix on BOTH Qwen3 (KV) and T5 (query) sides of adapter cross-attention
#   "prefix"    — learned vectors prepended to cached adapter output (T5-compatible space);
#                  compatible with cache_llm_adapter_outputs, no adapter needed at train time

import os
from typing import Optional

import torch
import torch.nn as nn

from library.utils import setup_logging

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Default Qwen3 hidden dimension
DEFAULT_EMBED_DIM = 1024


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
    num_postfix_tokens = network_dim if network_dim is not None else 8

    # Allow override via network_kwargs
    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))
    mode = kwargs.get("mode", "hidden")
    cfg_scale = float(kwargs.get("cfg_scale", 4.0))
    num_t5_postfix_tokens = int(kwargs.get("num_t5_postfix_tokens", num_postfix_tokens))

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        cfg_scale=cfg_scale,
        num_t5_postfix_tokens=num_t5_postfix_tokens,
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

    # Detect mode from keys (also check safetensors metadata as fallback)
    metadata_mode = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_mode = meta.get("ss_mode")

    if "prefix_embeds" in weights_sd:
        mode = "prefix"
        postfix_weight = weights_sd["prefix_embeds"]
    elif "postfix_pos" in weights_sd:
        mode = "cfg"
        postfix_weight = weights_sd["postfix_pos"]
    elif "postfix_t5" in weights_sd:
        mode = "dual"
        postfix_weight = weights_sd.get("postfix", weights_sd.get("prefix"))
    elif metadata_mode == "prefix":
        mode = "prefix"
        postfix_weight = weights_sd.get("prefix_embeds")
    else:
        mode = metadata_mode or "hidden"
        postfix_weight = weights_sd.get("postfix", weights_sd.get("prefix"))
    if postfix_weight is None:
        raise ValueError(
            f"Not a postfix/prefix weight file (keys: {list(weights_sd.keys())[:10]}). "
            f"Expected 'prefix_embeds', 'postfix', 'postfix_pos', or 'prefix' key."
        )

    num_postfix_tokens, embed_dim = postfix_weight.shape
    num_t5_postfix_tokens = num_postfix_tokens
    if mode == "dual":
        num_t5_postfix_tokens = weights_sd["postfix_t5"].shape[0]

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        num_t5_postfix_tokens=num_t5_postfix_tokens,
    )
    return network, weights_sd


class PostfixNetwork(nn.Module):
    def __init__(
        self,
        num_postfix_tokens: int,
        embed_dim: int,
        multiplier: float = 1.0,
        mode: str = "hidden",
        cfg_scale: float = 4.0,
        num_t5_postfix_tokens: int = None,
    ):
        super().__init__()
        self.num_postfix_tokens = num_postfix_tokens
        self.embed_dim = embed_dim
        self.multiplier = multiplier
        self.mode = mode
        self.cfg_scale = cfg_scale
        self.num_t5_postfix_tokens = num_t5_postfix_tokens or num_postfix_tokens

        # Match init scale to the target embedding space:
        #   "embedding" mode → Qwen3 input embeddings (std ≈ 0.03)
        #   "prefix"         → adapter output (post-RMSNorm, std ≈ 1.0)
        #   "hidden"/"cfg"   → Qwen3 output hidden states (std ≈ 3.5)
        init_std = 0.02 if mode == "embedding" else 1.0

        if mode == "prefix":
            self.prefix_embeds = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            logger.info(
                f"PostfixNetwork: prefix mode — {num_postfix_tokens} tokens in T5-compatible space, "
                f"dim {embed_dim}, init_std={init_std}, {self.prefix_embeds.numel()} params"
            )
        elif mode == "cfg":
            self.postfix_pos = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            self.postfix_neg = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            total_params = self.postfix_pos.numel() + self.postfix_neg.numel()
            logger.info(
                f"PostfixNetwork: {num_postfix_tokens} tokens × 2 (pos+neg), dim {embed_dim}, "
                f"cfg_scale={cfg_scale}, init_std={init_std}, {total_params} params"
            )
        elif mode == "dual":
            # Qwen3 side (KV): hidden-state space
            self.postfix = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            # T5 side (query): same embedding space (adapter's in_proj is Identity for 1024→1024)
            self.postfix_t5 = nn.Parameter(
                torch.randn(self.num_t5_postfix_tokens, embed_dim) * init_std
            )
            total_params = self.postfix.numel() + self.postfix_t5.numel()
            logger.info(
                f"PostfixNetwork: dual mode — {num_postfix_tokens} Qwen3 tokens + "
                f"{self.num_t5_postfix_tokens} T5 tokens, dim {embed_dim}, "
                f"init_std={init_std}, {total_params} params"
            )
        else:
            self.postfix = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            logger.info(
                f"PostfixNetwork: {num_postfix_tokens} tokens, dim {embed_dim}, mode={mode}, init_std={init_std}, {self.postfix.numel()} params"
            )

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if self.mode == "prefix":
            # No monkey-patching needed — training loop prepends prefix_embeds to crossattn_emb
            logger.info(
                f"Prefix mode: {self.num_postfix_tokens} learned tokens will be prepended to "
                f"cached adapter output (T5-compatible space)"
            )
            return
        if self.mode == "cfg":
            if not hasattr(unet, "llm_adapter") or unet.llm_adapter is None:
                raise ValueError(
                    "unet does not have an llm_adapter — cfg postfix requires the LLM adapter"
                )
            # Set positive postfix as default (training loop swaps for neg pass)
            unet.llm_adapter.postfix_embeds = self.postfix_pos
            logger.info(
                f"Attached {self.num_postfix_tokens} postfix embeddings (pos+neg) to LLM adapter "
                f"(cfg mode, scale={self.cfg_scale})"
            )
        elif self.mode == "dual":
            if not hasattr(unet, "llm_adapter") or unet.llm_adapter is None:
                raise ValueError(
                    "unet does not have an llm_adapter — dual postfix requires the LLM adapter"
                )
            unet.llm_adapter.postfix_embeds = self.postfix
            unet.llm_adapter.postfix_t5_embeds = self.postfix_t5
            logger.info(
                f"Attached dual postfix: {self.num_postfix_tokens} Qwen3 + "
                f"{self.num_t5_postfix_tokens} T5 tokens to LLM adapter"
            )
        elif self.mode == "embedding":
            if text_encoders is None:
                raise ValueError(
                    "embedding mode requires live text encoder (cannot use --cache_text_encoder_outputs)"
                )
            te = (
                text_encoders[0]
                if isinstance(text_encoders, (list, tuple))
                else text_encoders
            )
            self._wrap_text_encoder(te)
            logger.info(
                f"Attached {self.num_postfix_tokens} postfix embeddings to text encoder (embedding mode)"
            )
        else:
            if not hasattr(unet, "llm_adapter") or unet.llm_adapter is None:
                raise ValueError(
                    "unet does not have an llm_adapter — postfix tuning requires the LLM adapter"
                )
            unet.llm_adapter.postfix_embeds = self.postfix
            logger.info(
                f"Attached {self.num_postfix_tokens} postfix embeddings to LLM adapter"
            )

    def _wrap_text_encoder(self, text_encoder):
        original_forward = text_encoder.forward
        postfix = self.postfix

        def wrapped_forward(*args, **kwargs):
            input_ids = kwargs.pop("input_ids", None)
            if input_ids is None and args:
                input_ids = args[0]
                args = args[1:]

            if input_ids is not None:
                inputs_embeds = text_encoder.embed_tokens(input_ids)
                B = inputs_embeds.shape[0]
                extra = (
                    postfix.unsqueeze(0)
                    .expand(B, -1, -1)
                    .to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                )
                inputs_embeds = torch.cat([inputs_embeds, extra], dim=1)
                kwargs["inputs_embeds"] = inputs_embeds

                attention_mask = kwargs.get("attention_mask")
                if attention_mask is not None:
                    extra_mask = torch.ones(
                        B,
                        postfix.shape[0],
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    kwargs["attention_mask"] = torch.cat(
                        [attention_mask, extra_mask], dim=1
                    )

            return original_forward(*args, **kwargs)

        text_encoder.forward = wrapped_forward
        self._original_forward = original_forward

    def prepend_prefix(self, crossattn_emb: torch.Tensor) -> torch.Tensor:
        """Prepend learned prefix vectors to crossattn_emb, trimming trailing padding to maintain seq length."""
        K = self.num_postfix_tokens
        B = crossattn_emb.shape[0]
        prefix = (
            self.prefix_embeds.unsqueeze(0)
            .expand(B, -1, -1)
            .to(dtype=crossattn_emb.dtype, device=crossattn_emb.device)
        )
        # Trim K trailing positions (zero-padding) to keep total length unchanged
        return torch.cat([prefix, crossattn_emb[:, :crossattn_emb.shape[1] - K]], dim=1)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def is_mergeable(self):
        return False

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        if self.mode == "prefix":
            return [self.prefix_embeds]
        if self.mode == "cfg":
            return [self.postfix_pos, self.postfix_neg]
        elif self.mode == "dual":
            return [self.postfix, self.postfix_t5]
        return [self.postfix]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        lr = unet_lr or default_lr
        if self.mode == "prefix":
            params = [{"params": [self.prefix_embeds], "lr": lr}]
            descriptions = ["prefix_embeds"]
        elif self.mode == "cfg":
            params = [{"params": [self.postfix_pos, self.postfix_neg], "lr": lr}]
            descriptions = ["postfix_pos+neg"]
        elif self.mode == "dual":
            params = [{"params": [self.postfix, self.postfix_t5], "lr": lr}]
            descriptions = ["postfix_qwen3+t5"]
        else:
            params = [{"params": [self.postfix], "lr": lr}]
            descriptions = ["postfix"]
        return params, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        lr = unet_lr or default_lr
        if self.mode == "prefix":
            return [{"params": [self.prefix_embeds], "lr": lr}]
        if self.mode == "cfg":
            return [{"params": [self.postfix_pos, self.postfix_neg], "lr": lr}]
        elif self.mode == "dual":
            return [{"params": [self.postfix, self.postfix_t5], "lr": lr}]
        return [{"params": [self.postfix], "lr": lr}]

    def save_weights(self, file, dtype, metadata):
        dtype = dtype or torch.bfloat16
        if self.mode == "prefix":
            state_dict = {
                "prefix_embeds": self.prefix_embeds.detach().clone().cpu().to(dtype),
            }
        elif self.mode == "cfg":
            state_dict = {
                "postfix_pos": self.postfix_pos.detach().clone().cpu().to(dtype),
                "postfix_neg": self.postfix_neg.detach().clone().cpu().to(dtype),
            }
        elif self.mode == "dual":
            state_dict = {
                "postfix": self.postfix.detach().clone().cpu().to(dtype),
                "postfix_t5": self.postfix_t5.detach().clone().cpu().to(dtype),
            }
        else:
            state_dict = {"postfix": self.postfix.detach().clone().cpu().to(dtype)}

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            metadata["ss_network_module"] = "networks.postfix_anima"
            metadata["ss_num_postfix_tokens"] = str(self.num_postfix_tokens)
            metadata["ss_embed_dim"] = str(self.embed_dim)
            metadata["ss_mode"] = self.mode
            if self.mode == "cfg":
                metadata["ss_cfg_scale"] = str(self.cfg_scale)

            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(
                state_dict, metadata
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        if self.mode == "prefix":
            if "prefix_embeds" in weights_sd:
                self.prefix_embeds.data.copy_(weights_sd["prefix_embeds"])
                logger.info(f"Loaded prefix weights: {self.prefix_embeds.shape}")
            else:
                raise ValueError(
                    "No 'prefix_embeds' key found in weights file for prefix mode"
                )
        elif self.mode == "cfg":
            if "postfix_pos" in weights_sd:
                self.postfix_pos.data.copy_(weights_sd["postfix_pos"])
                self.postfix_neg.data.copy_(weights_sd["postfix_neg"])
                logger.info(
                    f"Loaded cfg postfix weights: pos={self.postfix_pos.shape}, neg={self.postfix_neg.shape}"
                )
            else:
                raise ValueError(
                    "No 'postfix_pos' key found in weights file for cfg mode"
                )
        elif self.mode == "dual":
            weight = weights_sd.get("postfix", weights_sd.get("prefix"))
            if weight is None:
                raise ValueError("No 'postfix' key found in weights file for dual mode")
            self.postfix.data.copy_(weight)
            if "postfix_t5" not in weights_sd:
                raise ValueError(
                    "No 'postfix_t5' key found in weights file for dual mode"
                )
            self.postfix_t5.data.copy_(weights_sd["postfix_t5"])
            logger.info(
                f"Loaded dual postfix weights: qwen3={self.postfix.shape}, t5={self.postfix_t5.shape}"
            )
        else:
            weight = weights_sd.get("postfix", weights_sd.get("prefix"))
            if weight is not None:
                self.postfix.data.copy_(weight)
                logger.info(f"Loaded postfix weights: {self.postfix.shape}")
            else:
                raise ValueError(
                    "No 'postfix' (or legacy 'prefix') key found in weights file"
                )
