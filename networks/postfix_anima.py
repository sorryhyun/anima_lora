# Prefix/Postfix tuning network module for Anima LLM Adapter
#
# Learns N continuous vectors injected into the cached adapter output (T5-compatible
# space). These discover quality signals in embedding space that improve generation
# across all artist tags.
#
# Modes:
#   "postfix" (default) — postfix appended to cached adapter output; splice position
#                          controlled by splice_position kwarg ("front_of_padding"
#                          legacy, "end_of_sequence" default). Compatible with
#                          cache_llm_adapter_outputs, no adapter needed at train time.
#   "prefix"            — learned vectors prepended to cached adapter output
#                          (T5-compatible space); compatible with
#                          cache_llm_adapter_outputs, no adapter needed at train time.
#   "cond"              — caption-conditional postfix: mean-pool content slots ->
#                          2-layer MLP -> per-sample K×D postfix vectors. Strictly
#                          more expressive than "postfix". Last layer zero-inited so
#                          training starts from baseline behavior.

import os
from typing import Optional

import torch
import torch.nn as nn

from library.log import setup_logging

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
    mode = kwargs.get("mode", "postfix")
    splice_position = kwargs.get("splice_position", "end_of_sequence")
    cond_hidden_dim = int(kwargs.get("cond_hidden_dim", 256))

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        splice_position=splice_position,
        cond_hidden_dim=cond_hidden_dim,
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
    metadata_splice = None
    metadata_cond_hidden = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_mode = meta.get("ss_mode")
            metadata_splice = meta.get("ss_splice_position")
            metadata_cond_hidden = meta.get("ss_cond_hidden_dim")

    has_cond = any(k.startswith("cond_mlp.") for k in weights_sd.keys())
    if has_cond or metadata_mode == "cond":
        mode = "cond"
        postfix_weight = None
    elif "prefix_embeds" in weights_sd:
        mode = "prefix"
        postfix_weight = weights_sd["prefix_embeds"]
    elif "postfix_embeds" in weights_sd:
        mode = "postfix"
        postfix_weight = weights_sd["postfix_embeds"]
    elif metadata_mode == "prefix":
        mode = "prefix"
        postfix_weight = weights_sd.get("prefix_embeds")
    else:
        mode = metadata_mode or "postfix"
        postfix_weight = weights_sd.get("postfix_embeds")

    if mode == "cond":
        # Infer shapes from MLP weights
        # cond_mlp.0.weight: [hidden, embed_dim]
        # cond_mlp.2.weight: [K * embed_dim, hidden]
        w0 = weights_sd.get("cond_mlp.0.weight")
        w2 = weights_sd.get("cond_mlp.2.weight")
        if w0 is None or w2 is None:
            raise ValueError(
                f"cond mode requires cond_mlp.0.weight and cond_mlp.2.weight (got keys: "
                f"{[k for k in weights_sd.keys() if 'cond_mlp' in k]})"
            )
        cond_hidden_dim = w0.shape[0]
        embed_dim = w0.shape[1]
        num_postfix_tokens = w2.shape[0] // embed_dim
    else:
        if postfix_weight is None:
            raise ValueError(
                f"Not a postfix/prefix weight file (keys: {list(weights_sd.keys())[:10]}). "
                f"Expected 'prefix_embeds', 'postfix_embeds', or cond_mlp.* keys."
            )
        num_postfix_tokens, embed_dim = postfix_weight.shape
        cond_hidden_dim = int(metadata_cond_hidden) if metadata_cond_hidden else 256

    splice_position = metadata_splice or "end_of_sequence"

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        splice_position=splice_position,
        cond_hidden_dim=cond_hidden_dim,
    )
    return network, weights_sd


class PostfixNetwork(nn.Module):
    def __init__(
        self,
        num_postfix_tokens: int,
        embed_dim: int,
        multiplier: float = 1.0,
        mode: str = "postfix",
        splice_position: str = "end_of_sequence",
        cond_hidden_dim: int = 256,
    ):
        super().__init__()
        if mode not in ("postfix", "prefix", "cond"):
            raise ValueError(
                f"mode must be 'postfix', 'prefix', or 'cond', got {mode!r}"
            )
        self.num_postfix_tokens = num_postfix_tokens
        self.embed_dim = embed_dim
        self.multiplier = multiplier
        self.mode = mode
        if splice_position not in ("front_of_padding", "end_of_sequence"):
            raise ValueError(
                f"splice_position must be 'front_of_padding' or 'end_of_sequence', got {splice_position!r}"
            )
        self.splice_position = splice_position
        self.cond_hidden_dim = cond_hidden_dim

        # Init scale matches the T5-compatible adapter output space (post-RMSNorm, std ≈ 1.0).
        init_std = 1.0

        if mode == "prefix":
            self.prefix_embeds = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            logger.info(
                f"PostfixNetwork: prefix mode — {num_postfix_tokens} tokens in T5-compatible space, "
                f"dim {embed_dim}, init_std={init_std}, {self.prefix_embeds.numel()} params"
            )
        elif mode == "cond":
            # Caption-conditional: pool content slots -> 2-layer MLP -> K*D postfix.
            # Zero-init the last layer so training starts from exact baseline behavior
            # (empty postfix at end-of-sequence overwrites zero padding with zeros).
            self.cond_mlp = nn.Sequential(
                nn.Linear(embed_dim, cond_hidden_dim),
                nn.GELU(),
                nn.Linear(cond_hidden_dim, num_postfix_tokens * embed_dim),
            )
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)
            total_params = sum(p.numel() for p in self.cond_mlp.parameters())
            logger.info(
                f"PostfixNetwork: cond mode — {num_postfix_tokens} tokens × dim {embed_dim}, "
                f"hidden {cond_hidden_dim}, splice={self.splice_position}, "
                f"{total_params} params (last layer zero-inited)"
            )
        else:
            # Default: T5-compatible postfix (appended to cached adapter output)
            self.postfix_embeds = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            logger.info(
                f"PostfixNetwork: postfix mode — {num_postfix_tokens} tokens in T5-compatible space, "
                f"dim {embed_dim}, init_std={init_std}, splice={self.splice_position}, "
                f"{self.postfix_embeds.numel()} params"
            )

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        # No monkey-patching needed — training loop handles prefix/postfix on cached crossattn_emb
        kind = "prepended to" if self.mode == "prefix" else "appended to"
        logger.info(
            f"{self.mode} mode: {self.num_postfix_tokens} learned tokens will be {kind} "
            f"cached adapter output (T5-compatible space)"
        )

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
        return torch.cat(
            [prefix, crossattn_emb[:, : crossattn_emb.shape[1] - K]], dim=1
        )

    def append_postfix(
        self, crossattn_emb: torch.Tensor, crossattn_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Splice learned postfix vectors into crossattn_emb (overwrites zero-padding slots).

        Splice position controlled by self.splice_position:
          - "end_of_sequence": place at [S-K, S). Caption-position-agnostic; preserves the
            strongest front-of-padding sinks intact.
          - "front_of_padding": place at [seqlens[i], seqlens[i]+K). Caption-position-aware;
            displaces the strongest sinks. Legacy behavior.

        In "cond" mode the postfix vectors are computed per-sample from a mean-pool of
        content slots through a 2-layer MLP. In default mode they come from a single
        learned parameter tensor shared across the batch.

        Args:
            crossattn_emb: [B, S, D] cached adapter output (zero-padded after real tokens)
            crossattn_seqlens: [B] number of real text tokens per batch element
        """
        K = self.num_postfix_tokens
        B, S, D = crossattn_emb.shape

        if self.mode == "cond":
            # Mean-pool over content slots (positions < seqlen[i])
            pos = torch.arange(S, device=crossattn_emb.device).unsqueeze(0)  # [1, S]
            mask = (pos < crossattn_seqlens.unsqueeze(1)).to(
                crossattn_emb.dtype
            )  # [B, S]
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (crossattn_emb * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
            postfix = self.cond_mlp(pooled).view(B, K, D).to(crossattn_emb.dtype)
        else:
            postfix = (
                self.postfix_embeds.unsqueeze(0)
                .expand(B, -1, -1)
                .to(dtype=crossattn_emb.dtype, device=crossattn_emb.device)
            )

        if self.splice_position == "end_of_sequence":
            # Overwrite the last K slots of the zero-padding region with the postfix.
            # torch.cat preserves autograd on both sides.
            return torch.cat([crossattn_emb[:, : S - K, :], postfix], dim=1)

        # front_of_padding: place K postfix tokens at [seqlens[i], seqlens[i]+K) per sample
        offsets = crossattn_seqlens.long().unsqueeze(1) + torch.arange(
            K, device=crossattn_emb.device
        )  # [B, K]
        idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        return crossattn_emb.scatter(1, idx, postfix)

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
        if self.mode == "cond":
            return list(self.cond_mlp.parameters())
        return [self.postfix_embeds]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        lr = unet_lr or default_lr
        if self.mode == "prefix":
            params = [{"params": [self.prefix_embeds], "lr": lr}]
            descriptions = ["prefix_embeds"]
        elif self.mode == "cond":
            params = [{"params": list(self.cond_mlp.parameters()), "lr": lr}]
            descriptions = ["cond_mlp"]
        else:
            params = [{"params": [self.postfix_embeds], "lr": lr}]
            descriptions = ["postfix_embeds"]
        return params, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        lr = unet_lr or default_lr
        if self.mode == "prefix":
            return [{"params": [self.prefix_embeds], "lr": lr}]
        if self.mode == "cond":
            return [{"params": list(self.cond_mlp.parameters()), "lr": lr}]
        return [{"params": [self.postfix_embeds], "lr": lr}]

    def save_weights(self, file, dtype, metadata):
        dtype = dtype or torch.bfloat16
        if self.mode == "prefix":
            state_dict = {
                "prefix_embeds": self.prefix_embeds.detach().clone().cpu().to(dtype),
            }
        elif self.mode == "cond":
            state_dict = {
                f"cond_mlp.{k}": v.detach().clone().cpu().to(dtype)
                for k, v in self.cond_mlp.state_dict().items()
            }
        else:
            state_dict = {
                "postfix_embeds": self.postfix_embeds.detach().clone().cpu().to(dtype)
            }

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            metadata["ss_network_module"] = "networks.postfix_anima"
            metadata["ss_network_spec"] = "postfix"
            metadata["ss_num_postfix_tokens"] = str(self.num_postfix_tokens)
            metadata["ss_embed_dim"] = str(self.embed_dim)
            metadata["ss_mode"] = self.mode
            metadata["ss_splice_position"] = self.splice_position
            if self.mode == "cond":
                metadata["ss_cond_hidden_dim"] = str(self.cond_hidden_dim)

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
        elif self.mode == "cond":
            mlp_sd = {
                k[len("cond_mlp.") :]: v
                for k, v in weights_sd.items()
                if k.startswith("cond_mlp.")
            }
            if not mlp_sd:
                raise ValueError(
                    "No 'cond_mlp.*' keys found in weights file for cond mode"
                )
            missing, unexpected = self.cond_mlp.load_state_dict(mlp_sd, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"cond_mlp load_state_dict mismatch: missing={missing}, unexpected={unexpected}"
                )
            logger.info(
                f"Loaded cond_mlp weights: {sum(p.numel() for p in self.cond_mlp.parameters())} params"
            )
        else:
            weight = weights_sd.get("postfix_embeds")
            if weight is not None:
                self.postfix_embeds.data.copy_(weight)
                logger.info(f"Loaded postfix weights: {self.postfix_embeds.shape}")
            else:
                raise ValueError(
                    "No 'postfix_embeds' key found in weights file for postfix mode"
                )
