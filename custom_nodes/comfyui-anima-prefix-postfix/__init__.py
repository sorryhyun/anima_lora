"""Anima Prefix/Postfix model-patch node for ComfyUI.

Patches the Anima diffusion model's `forward` so learned prefix/postfix/cond
vectors are spliced into the T5-compatible crossattn embedding **after** the
LLM adapter runs and after its pad-to-512 step — i.e. the same space as
anima_lora's training and reference inference. Positive-only routing is done
by reading `cond_or_uncond` from transformer_options, so CFG is preserved.

Supported modes (auto-detected from safetensors keys/metadata):
  - prefix  : learned vectors prepended; last K padding slots trimmed
  - postfix : static learned vectors spliced after real text tokens
  - cond    : prompt-adaptive postfix computed per-sample by an MLP
              over mean-pooled content tokens
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import folder_paths

logger = logging.getLogger(__name__)


def _build_cond_mlp(embed_dim: int, hidden_dim: int, num_tokens: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_tokens * embed_dim),
    )


# Cache: path -> (mode, payload, num_tokens, splice_position)
# payload is a Tensor for prefix/postfix, (nn.Sequential, num_tokens, embed_dim) for cond.
_weight_cache: Dict[str, Tuple[str, object, int, str]] = {}


def _load_weights(file_path: str) -> Tuple[str, object, int, str]:
    if file_path in _weight_cache:
        return _weight_cache[file_path]

    from safetensors import safe_open
    from safetensors.torch import load_file

    weights_sd = load_file(file_path)

    metadata_mode: Optional[str] = None
    metadata_splice: Optional[str] = None
    metadata_cond_hidden: Optional[str] = None
    with safe_open(file_path, framework="pt") as f:
        meta = f.metadata() or {}
        metadata_mode = meta.get("ss_mode")
        metadata_splice = meta.get("ss_splice_position")
        metadata_cond_hidden = meta.get("ss_cond_hidden_dim")

    has_cond = any(k.startswith("cond_mlp.") for k in weights_sd)
    splice_position = metadata_splice or "end_of_sequence"

    if has_cond or metadata_mode == "cond":
        w0 = weights_sd.get("cond_mlp.0.weight")
        w2 = weights_sd.get("cond_mlp.2.weight")
        if w0 is None or w2 is None:
            raise ValueError(
                f"cond mode requires cond_mlp.0.weight and cond_mlp.2.weight "
                f"(got keys: {[k for k in weights_sd if 'cond_mlp' in k]})"
            )
        hidden_dim = w0.shape[0]
        embed_dim = w0.shape[1]
        num_tokens = w2.shape[0] // embed_dim
        if metadata_cond_hidden:
            hidden_dim = int(metadata_cond_hidden)

        mlp = _build_cond_mlp(embed_dim, hidden_dim, num_tokens)
        mlp_sd = {
            k[len("cond_mlp.") :]: v
            for k, v in weights_sd.items()
            if k.startswith("cond_mlp.")
        }
        missing, unexpected = mlp.load_state_dict(mlp_sd, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"cond_mlp load mismatch: missing={missing}, unexpected={unexpected}"
            )
        mlp.eval()
        for p in mlp.parameters():
            p.requires_grad_(False)
        result = ("cond", (mlp, num_tokens, embed_dim), num_tokens, splice_position)
    elif "prefix_embeds" in weights_sd:
        embeds = weights_sd["prefix_embeds"]
        result = ("prefix", embeds, embeds.shape[0], splice_position)
    elif "postfix_embeds" in weights_sd:
        embeds = weights_sd["postfix_embeds"]
        result = ("postfix", embeds, embeds.shape[0], splice_position)
    else:
        raise ValueError(
            f"Unsupported weight file (keys: {list(weights_sd.keys())[:10]}). "
            f"Expected 'prefix_embeds', 'postfix_embeds', or 'cond_mlp.*'."
        )

    _weight_cache[file_path] = result
    logger.info(
        f"Loaded {result[0]} weights: {result[2]} tokens from {file_path} "
        f"(splice={splice_position})"
    )
    return result


def _prepend_prefix(ctx: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
    K = prefix.shape[0]
    B, S, _ = ctx.shape
    prefix = (
        prefix.unsqueeze(0).expand(B, -1, -1).to(dtype=ctx.dtype, device=ctx.device)
    )
    return torch.cat([prefix, ctx[:, : S - K, :]], dim=1)


def _splice_postfix(
    ctx: torch.Tensor, postfix: torch.Tensor, splice_position: str
) -> torch.Tensor:
    B, S, D = ctx.shape
    K = postfix.shape[1]
    postfix = postfix.to(dtype=ctx.dtype, device=ctx.device)
    if splice_position == "end_of_sequence":
        return torch.cat([ctx[:, : S - K, :], postfix], dim=1)
    mask = ctx.abs().sum(dim=-1) > 0
    seqlens = mask.long().sum(dim=-1)
    offsets = seqlens.unsqueeze(1) + torch.arange(K, device=ctx.device)
    offsets = offsets.clamp(max=S - 1)
    idx = offsets.unsqueeze(-1).expand(-1, -1, D)
    return ctx.scatter(1, idx, postfix)


def _apply_cfg(
    ctx: torch.Tensor, mode: str, payload, splice_position: str, strength: float
) -> torch.Tensor:
    if strength == 0:
        return ctx
    if mode == "prefix":
        return _prepend_prefix(ctx, payload * strength)
    if mode == "cond":
        mlp, num_tokens, embed_dim = payload
        mlp.to(device=ctx.device, dtype=torch.float32)
        mask = (ctx.abs().sum(dim=-1) > 0).to(torch.float32)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (ctx.float() * mask.unsqueeze(-1)).sum(dim=1) / denom
        with torch.no_grad():
            out = mlp(pooled)
        postfix = out.view(ctx.shape[0], num_tokens, embed_dim) * strength
        return _splice_postfix(ctx, postfix, splice_position)
    # static postfix
    B = ctx.shape[0]
    postfix = (payload * strength).unsqueeze(0).expand(B, -1, -1)
    return _splice_postfix(ctx, postfix, splice_position)


def _make_forward_wrapper(dit, prev_forward, mode, payload, splice_position, strength):
    """Build a replacement for Anima.forward that splices post-adapter.

    If `prev_forward` is the unmodified bound method, the outermost wrapper runs
    `preprocess_text_embeds` itself (which executes the LLM adapter + pad-to-512)
    and then pops `t5xxl_ids`/`t5xxl_weights` before delegating downstream — so
    inner wrappers and the eventual real `Anima.forward` skip the adapter pass.
    """

    def new_forward(x, timesteps, context, **kwargs):
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        t5xxl_weights = kwargs.pop("t5xxl_weights", None)

        if t5xxl_ids is not None:
            context = dit.preprocess_text_embeds(
                context, t5xxl_ids, t5xxl_weights=t5xxl_weights
            )

        transformer_options = kwargs.get("transformer_options") or {}
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        B = context.shape[0]
        if cond_or_uncond:
            per_group = max(B // len(cond_or_uncond), 1)
            rows = [
                j
                for i, kind in enumerate(cond_or_uncond)
                if kind == 0
                for j in range(i * per_group, (i + 1) * per_group)
            ]
        else:
            rows = list(range(B))

        if rows:
            idx = torch.tensor(rows, device=context.device, dtype=torch.long)
            sub = context.index_select(0, idx)
            sub = _apply_cfg(sub, mode, payload, splice_position, strength)
            context = context.index_copy(0, idx, sub)

        return prev_forward(x, timesteps, context, **kwargs)

    return new_forward


class AnimaPrefixPostfix:
    """Patch the Anima DiT to splice learned prefix/postfix/cond vectors.

    Mode and splice position are auto-detected from the safetensors file.
    Only positive-batch rows are affected (read from transformer_options),
    so CFG is preserved. Stack multiple instances to combine modes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "weight_file": (
                    folder_paths.get_filename_list("loras"),
                    {
                        "tooltip": "Safetensors file with prefix_embeds, postfix_embeds, or cond_mlp.* keys."
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength multiplier for the learned vectors.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches"
    DESCRIPTION = (
        "Patch the Anima DiT to splice learned prefix/postfix/cond vectors "
        "after the LLM adapter. Auto-detects mode and splice position from "
        "the safetensors file. Applies to positive-batch rows only (CFG-safe)."
    )

    def apply(self, model, weight_file, strength=1.0):
        if strength == 0:
            return (model,)

        file_path = folder_paths.get_full_path("loras", weight_file)
        mode, payload, _, splice_position = _load_weights(file_path)

        m = model.clone()
        dit = m.model.diffusion_model

        prev = m.object_patches.get("diffusion_model.forward")
        if prev is None:
            prev = dit.forward  # bound method — preserves self

        new_fn = _make_forward_wrapper(
            dit, prev, mode, payload, splice_position, strength
        )
        m.add_object_patch("diffusion_model.forward", new_fn)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaPrefixPostfix": AnimaPrefixPostfix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaPrefixPostfix": "Anima Prefix/Postfix",
}
