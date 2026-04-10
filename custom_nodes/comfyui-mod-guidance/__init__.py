"""Anima Mod Guidance: drop-in modulation guidance for Anima models.

Single node that steers generation via learned pooled-text modulation.
Type positive tags directly in the node, connect existing negative conditioning.
Loads a distilled pooled_text_proj adapter and patches the model forward pass.

Workflow:
  Load Checkpoint --MODEL--> [Anima Mod Guidance] --MODEL--> KSampler
                  --CLIP--^         ^                           ^ ^
  CLIPTextEncode neg ---------------+                           | |
  CLIPTextEncode pos -------------------------------------------+ |
  CLIPTextEncode neg ---------------------------------------------+
"""

import logging
import os
import threading

import torch
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.patcher_extension
import folder_paths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# pooled_text_proj state_dict keys: Sequential(Linear, SiLU, Linear)
PROJ_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")

WRAPPER_KEY = "anima_mod_guidance"
STATE_KEY = "anima_mod_guidance_state"

# ---------------------------------------------------------------------------
# Adapter loading & caching
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_CPU_CACHE: dict = {}
_TYPED_CACHE: dict = {}


def _load_cpu(path: str) -> dict:
    path = os.path.abspath(path)
    with _LOCK:
        if path in _CPU_CACHE:
            return _CPU_CACHE[path]

    from safetensors.torch import load_file

    raw = load_file(path)

    missing = [k for k in PROJ_KEYS if k not in raw]
    if missing:
        raise RuntimeError(
            f"Adapter missing keys: {', '.join(missing)}. "
            f"Expected pooled_text_proj format with keys: {PROJ_KEYS}"
        )

    state = {}
    for k in PROJ_KEYS:
        state[k] = raw[k].detach().float().cpu().contiguous()

    with _LOCK:
        _CPU_CACHE[path] = state
    return state


def _validate_adapter(path: str, diffusion_model) -> dict:
    path = os.path.abspath(path)
    state = _load_cpu(path)

    model_channels = getattr(diffusion_model, "model_channels", None)
    if model_channels is None:
        raise RuntimeError("Model missing model_channels")

    w0 = state["0.weight"]  # (model_channels, pooled_dim)
    if w0.shape[0] != model_channels:
        raise RuntimeError(
            f"Adapter output dim ({w0.shape[0]}) != model_channels ({model_channels})"
        )

    pooled_dim = w0.shape[1]
    return {"path": path, "model_channels": model_channels, "pooled_dim": pooled_dim}


def _get_typed(path: str, device, dtype):
    path = os.path.abspath(path)
    key = (path, str(device), str(dtype))
    with _LOCK:
        if key in _TYPED_CACHE:
            return _TYPED_CACHE[key]

    cpu = _load_cpu(path)
    typed = {k: v.to(device=device, dtype=dtype) for k, v in cpu.items()}
    with _LOCK:
        _TYPED_CACHE[key] = typed
    return typed


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def _get_diffusion_model(model):
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Invalid MODEL input")

    cfg = getattr(base, "model_config", None)
    if cfg is not None:
        im = cfg.unet_config.get("image_model", None)
        if im != "anima":
            raise RuntimeError(
                f"Mod Guidance requires Anima model (got image_model='{im}')"
            )

    dm = getattr(base, "diffusion_model", None)
    if dm is None:
        raise RuntimeError("Model missing diffusion_model")
    return dm


# ---------------------------------------------------------------------------
# Pooled embedding extraction
# ---------------------------------------------------------------------------


def _encode_pooled(clip, text: str) -> torch.Tensor:
    """Encode text via the workflow's CLIP and return pooled embedding [1, dim].

    If the encoder provides a native pooled_output (e.g. CLIP-L/G), that is used.
    Otherwise falls back to max-pooling the sequence embeddings, which matches
    how the Anima model internally derives pooled text from T5 cross-attention.
    """
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    pooled = output.get("pooled_output")

    if pooled is None:
        pooled = cond.max(dim=1).values

    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)

    return pooled[:1].detach().float().cpu().contiguous()


def _extract_pooled_from_cond(conditioning) -> torch.Tensor:
    """Extract pooled output from a CONDITIONING input.

    Falls back to max-pooling the sequence embeddings if pooled_output
    is not present (e.g. T5-based encoders).
    """
    if not conditioning or len(conditioning) == 0:
        raise RuntimeError("Negative conditioning is empty")

    cond_tensor, meta = conditioning[0][0], conditioning[0][1]
    pooled = meta.get("pooled_output") if isinstance(meta, dict) else None

    if pooled is None:
        pooled = cond_tensor.max(dim=1).values

    if not torch.is_tensor(pooled):
        raise RuntimeError(f"pooled_output must be tensor, got {type(pooled)}")

    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)

    return pooled[:1].detach().float().cpu().contiguous()


# ---------------------------------------------------------------------------
# Forward wrapper — patches DiT forward to inject pooled-text modulation
# ---------------------------------------------------------------------------


def _project(pooled, state):
    """Project pooled embedding through the 2-layer MLP (Linear -> SiLU -> Linear)."""
    x = F.linear(pooled, state["0.weight"], state["0.bias"])
    x = F.silu(x)
    x = F.linear(x, state["2.weight"], state["2.bias"])
    return x


def _expand_pooled(pooled, batch_size, device, dtype):
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    if pooled.shape[0] == 1:
        pooled = pooled.expand(batch_size, -1)
    return pooled.to(device=device, dtype=dtype)



def _extract_transformer_options(args, kwargs):
    to = kwargs.get("transformer_options")
    if isinstance(to, dict):
        return to
    if len(args) > 0 and isinstance(args[-1], dict):
        return args[-1]
    if len(args) > 1 and isinstance(args[-2], dict):
        return args[-2]
    return {}


def _wrapper(executor, *args, **kwargs):
    to = _extract_transformer_options(args, kwargs)
    state = to.get(STATE_KEY)
    if state is None:
        return executor(*args, **kwargs)

    x = args[0]
    timesteps = args[1]
    context = args[2]
    fps = args[3] if len(args) > 3 else kwargs.get("fps")
    padding_mask = args[4] if len(args) > 4 else kwargs.get("padding_mask")

    return _forward_modulated(
        executor.class_obj, x, timesteps, context, fps, padding_mask, to, state
    )


def _forward_modulated(dit, x, timesteps, context, fps, padding_mask, to, state):
    if not hasattr(dit, "blocks") or not hasattr(dit, "prepare_embedded_sequence"):
        raise RuntimeError("Model not compatible with Mod Guidance")

    orig_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(
        x, (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial)
    )

    x_BT, rope, extra_pos = dit.prepare_embedded_sequence(
        x, fps=fps, padding_mask=padding_mask
    )

    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)

    t_emb, adaln = dit.t_embedder[1](dit.t_embedder[0](timesteps).to(x_BT.dtype))
    t_emb = dit.t_embedding_norm(t_emb)

    if adaln is None:
        raise RuntimeError("Model did not produce AdaLN-LoRA embeddings")

    # --- Modulation guidance (proposal Eq. 3): ---
    # emb = t_emb + proj(main) + w * (proj(pos) - proj(neg))
    adapter_state = _get_typed(
        state["adapter_path"], device=t_emb.device, dtype=t_emb.dtype
    )
    B = t_emb.shape[0]

    # proj(main): pool from the actual prompt's cross-attention embeddings
    main_pooled = context.max(dim=1).values  # (B, 1024)
    proj_main = _project(main_pooled, adapter_state)

    # w * (proj(pos) - proj(neg)): project separately, then difference
    pp = _expand_pooled(state["pos_pooled"], B, t_emb.device, t_emb.dtype)
    pn = _expand_pooled(state["neg_pooled"], B, t_emb.device, t_emb.dtype)
    guidance = float(state["w"]) * (_project(pp, adapter_state) - _project(pn, adapter_state))

    t_emb_mod = t_emb + proj_main.unsqueeze(1) + guidance.unsqueeze(1)
    # ----------------------------------------------------------------------

    dit.affline_scale_log_info = {"t_embedding_B_T_D": t_emb_mod.detach()}
    dit.affline_emb = t_emb_mod
    dit.crossattn_emb = context

    if extra_pos is not None and x_BT.shape != extra_pos.shape:
        raise RuntimeError(
            f"Positional embedding shape mismatch: {x_BT.shape} vs {extra_pos.shape}"
        )

    if x_BT.dtype == torch.float16:
        x_BT = x_BT.float()

    block_kw = {
        "rope_emb_L_1_1_D": rope.unsqueeze(1).unsqueeze(0),
        "extra_per_block_pos_emb": extra_pos,
        "transformer_options": to,
    }

    for block in dit.blocks:
        x_BT = block(x_BT, t_emb_mod, context, adaln_lora_B_T_3D=adaln, **block_kw)

    out = dit.final_layer(
        x_BT.to(context.dtype), t_emb_mod, adaln_lora_B_T_3D=adaln
    )
    return dit.unpatchify(out)[
        :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
    ]


def _register_wrapper(patcher, adapter_path, pos, neg, w):
    patcher.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY
    )

    opts = patcher.model_options.setdefault("transformer_options", {})
    opts[STATE_KEY] = {
        "adapter_path": adapter_path,
        "pos_pooled": pos,
        "neg_pooled": neg,
        "w": float(w),
    }

    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _wrapper
    )


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------


class AnimaModGuidance:
    """Drop-in modulation guidance for Anima.

    Type positive tags directly, connect existing negative conditioning.
    Loads a distilled pooled_text_proj adapter from the loras folder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "Anima model from Load Checkpoint."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIP from Load Checkpoint (reuse existing)."},
                ),
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": "Negative conditioning from CLIPTextEncode "
                        "(reuse existing negative prompt).",
                    },
                ),
                "adapter": (
                    folder_paths.get_filename_list("loras"),
                    {
                        "tooltip": "pooled_text_proj safetensors file "
                        "(from distill-mod or make sync).",
                    },
                ),
                "positive_tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Tags/prompt to steer generation toward.",
                    },
                ),
                "w": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Modulation guidance strength.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model with modulation guidance applied.",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"
    DESCRIPTION = (
        "Steers Anima generation via modulation guidance. "
        "Type positive tags directly, connect negative from existing CLIPTextEncode. "
        "Select a distilled pooled_text_proj adapter from the loras folder. "
        "Drop-in: connect MODEL+CLIP from loader, output MODEL to KSampler."
    )

    def apply(self, model, clip, negative, adapter, positive_tags, w):
        dm = _get_diffusion_model(model)

        # Resolve adapter from loras folder
        adapter_path = folder_paths.get_full_path("loras", adapter)
        meta = _validate_adapter(adapter_path, dm)

        # Encode positive tags via CLIP, extract negative pooled from conditioning
        pos = _encode_pooled(clip, positive_tags) if positive_tags.strip() else None
        neg = _extract_pooled_from_cond(negative)

        # If no positive tags, use negative as anchor (w * (neg - neg) = 0, no-op)
        if pos is None:
            pos = neg.clone()

        # Check pooled dim compatibility
        for name, p in [("positive", pos), ("negative", neg)]:
            if p.shape[1] != meta["pooled_dim"]:
                raise RuntimeError(
                    f"{name} pooled dim ({p.shape[1]}) != adapter "
                    f"input dim ({meta['pooled_dim']})"
                )

        # Patch model
        patched = model.clone()
        _register_wrapper(patched, adapter_path, pos, neg, w)

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "AnimaModGuidance": AnimaModGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaModGuidance": "Anima Mod Guidance",
}
