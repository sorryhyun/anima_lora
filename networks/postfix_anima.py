# Prefix/Postfix tuning network module for Anima LLM Adapter
#
# Learns N continuous vectors injected into the cached adapter output (T5-compatible
# space). These discover quality signals in embedding space that improve generation
# across all artist tags.
#
# Modes:
#   (default)   — postfix appended to cached adapter output; splice position controlled by
#                  splice_position kwarg ("front_of_padding" legacy, "end_of_sequence" default).
#                  Compatible with cache_llm_adapter_outputs, no adapter needed at train time
#   "cond"      — caption-conditional postfix: mean-pool content slots -> 2-layer MLP ->
#                  per-sample K×D postfix vectors. Strictly more expressive than "postfix".
#                  Last layer zero-inited so training starts from baseline behavior.
#   "prefix"    — learned vectors prepended to cached adapter output (T5-compatible space);
#                  compatible with cache_llm_adapter_outputs, no adapter needed at train time
#   "cfg"       — two postfix sets (pos + neg) trained with CFG-aware loss
#   "dual"      — postfix on BOTH Qwen3 (KV) and T5 (query) sides of adapter cross-attention
#   "embedding" — postfix injected into Qwen3 input embeddings (goes through all layers)

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
    mode = kwargs.get("mode", "postfix")
    cfg_scale = float(kwargs.get("cfg_scale", 4.0))
    num_t5_postfix_tokens = int(kwargs.get("num_t5_postfix_tokens", num_postfix_tokens))
    splice_position = kwargs.get("splice_position", "end_of_sequence")
    cond_hidden_dim = int(kwargs.get("cond_hidden_dim", 256))

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        cfg_scale=cfg_scale,
        num_t5_postfix_tokens=num_t5_postfix_tokens,
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
        mode = metadata_mode or "postfix"
        postfix_weight = weights_sd.get("postfix", weights_sd.get("prefix"))

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
        num_t5_postfix_tokens = num_postfix_tokens
    else:
        if postfix_weight is None:
            raise ValueError(
                f"Not a postfix/prefix weight file (keys: {list(weights_sd.keys())[:10]}). "
                f"Expected 'prefix_embeds', 'postfix_embeds', 'postfix_pos', 'postfix_t5', "
                f"or cond_mlp.* keys."
            )
        num_postfix_tokens, embed_dim = postfix_weight.shape
        num_t5_postfix_tokens = num_postfix_tokens
        if mode == "dual":
            num_t5_postfix_tokens = weights_sd["postfix_t5"].shape[0]
        cond_hidden_dim = int(metadata_cond_hidden) if metadata_cond_hidden else 256

    splice_position = metadata_splice or "end_of_sequence"

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        num_t5_postfix_tokens=num_t5_postfix_tokens,
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
        cfg_scale: float = 4.0,
        num_t5_postfix_tokens: int = None,
        splice_position: str = "end_of_sequence",
        cond_hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_postfix_tokens = num_postfix_tokens
        self.embed_dim = embed_dim
        self.multiplier = multiplier
        self.mode = mode
        self.cfg_scale = cfg_scale
        self.num_t5_postfix_tokens = num_t5_postfix_tokens or num_postfix_tokens
        if splice_position not in ("front_of_padding", "end_of_sequence"):
            raise ValueError(
                f"splice_position must be 'front_of_padding' or 'end_of_sequence', got {splice_position!r}"
            )
        self.splice_position = splice_position
        self.cond_hidden_dim = cond_hidden_dim

        # Match init scale to the target embedding space:
        #   "embedding" mode → Qwen3 input embeddings (std ≈ 0.03)
        #   "prefix"/(default) → adapter output (post-RMSNorm, std ≈ 1.0)
        #   "cfg"            → Qwen3 output hidden states (std ≈ 3.5)
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
        elif mode == "embedding":
            self.postfix = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            logger.info(
                f"PostfixNetwork: embedding mode — {num_postfix_tokens} tokens, dim {embed_dim}, "
                f"init_std={init_std}, {self.postfix.numel()} params"
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
        if self.mode in ("prefix", "postfix", "hidden", "cond"):
            # No monkey-patching needed — training loop handles prefix/postfix on cached crossattn_emb
            kind = "prepended to" if self.mode == "prefix" else "appended to"
            logger.info(
                f"{self.mode} mode: {self.num_postfix_tokens} learned tokens will be {kind} "
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
        if self.mode == "cfg":
            return [self.postfix_pos, self.postfix_neg]
        elif self.mode == "dual":
            return [self.postfix, self.postfix_t5]
        elif self.mode == "embedding":
            return [self.postfix]
        elif self.mode == "cond":
            return list(self.cond_mlp.parameters())
        return [self.postfix_embeds]

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
        elif self.mode == "embedding":
            params = [{"params": [self.postfix], "lr": lr}]
            descriptions = ["postfix_embedding"]
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
        if self.mode == "cfg":
            return [{"params": [self.postfix_pos, self.postfix_neg], "lr": lr}]
        elif self.mode == "dual":
            return [{"params": [self.postfix, self.postfix_t5], "lr": lr}]
        elif self.mode == "embedding":
            return [{"params": [self.postfix], "lr": lr}]
        elif self.mode == "cond":
            return [{"params": list(self.cond_mlp.parameters()), "lr": lr}]
        return [{"params": [self.postfix_embeds], "lr": lr}]

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
        elif self.mode == "embedding":
            state_dict = {"postfix": self.postfix.detach().clone().cpu().to(dtype)}
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
            metadata["ss_num_postfix_tokens"] = str(self.num_postfix_tokens)
            metadata["ss_embed_dim"] = str(self.embed_dim)
            metadata["ss_mode"] = self.mode
            metadata["ss_splice_position"] = self.splice_position
            if self.mode == "cfg":
                metadata["ss_cfg_scale"] = str(self.cfg_scale)
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
        elif self.mode == "embedding":
            weight = weights_sd.get("postfix", weights_sd.get("prefix"))
            if weight is not None:
                self.postfix.data.copy_(weight)
                logger.info(f"Loaded embedding postfix weights: {self.postfix.shape}")
            else:
                raise ValueError(
                    "No 'postfix' key found in weights file for embedding mode"
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
            weight = weights_sd.get("postfix_embeds", weights_sd.get("postfix"))
            if weight is not None:
                self.postfix_embeds.data.copy_(weight)
                logger.info(f"Loaded postfix weights: {self.postfix_embeds.shape}")
            else:
                raise ValueError(
                    "No 'postfix_embeds' key found in weights file for postfix mode"
                )
