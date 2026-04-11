"""Modulation guidance: adapter loading, projection, DIFFUSION_MODEL wrapper."""

import logging
import os
import threading
from typing import Optional

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import folder_paths

logger = logging.getLogger(__name__)

PROJ_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")
MOD_WRAPPER_KEY = "spectrum_mod_guidance"
MOD_STATE_KEY = "spectrum_mod_guidance_state"

_ADAPTER_LOCK = threading.Lock()
_ADAPTER_CPU_CACHE: dict = {}
_ADAPTER_TYPED_CACHE: dict = {}


def _load_adapter_cpu(path: str) -> dict:
    path = os.path.abspath(path)
    with _ADAPTER_LOCK:
        if path in _ADAPTER_CPU_CACHE:
            return _ADAPTER_CPU_CACHE[path]

    from safetensors.torch import load_file

    raw = load_file(path)
    missing = [k for k in PROJ_KEYS if k not in raw]
    if missing:
        raise RuntimeError(
            f"Adapter missing keys: {', '.join(missing)}. "
            f"Expected pooled_text_proj format with keys: {PROJ_KEYS}"
        )
    state = {k: raw[k].detach().float().cpu().contiguous() for k in PROJ_KEYS}
    with _ADAPTER_LOCK:
        _ADAPTER_CPU_CACHE[path] = state
    return state


def _get_adapter_typed(path: str, device, dtype):
    path = os.path.abspath(path)
    key = (path, str(device), str(dtype))
    with _ADAPTER_LOCK:
        if key in _ADAPTER_TYPED_CACHE:
            return _ADAPTER_TYPED_CACHE[key]
    cpu = _load_adapter_cpu(path)
    typed = {k: v.to(device=device, dtype=dtype) for k, v in cpu.items()}
    with _ADAPTER_LOCK:
        _ADAPTER_TYPED_CACHE[key] = typed
    return typed


def _project(pooled, adapter_state):
    """Project pooled embedding through the 2-layer MLP (Linear -> SiLU -> Linear)."""
    x = F.linear(pooled, adapter_state["0.weight"], adapter_state["0.bias"])
    x = F.silu(x)
    x = F.linear(x, adapter_state["2.weight"], adapter_state["2.bias"])
    return x


class ModGuidanceState:
    """Holds raw tensors for lazy guidance delta computation on first forward."""

    def __init__(self, adapter_path: str, w: float,
                 pos_raw: torch.Tensor, pos_t5_ids: torch.Tensor,
                 pos_t5_weights: Optional[torch.Tensor],
                 neg_raw: torch.Tensor, neg_t5_ids: Optional[torch.Tensor],
                 neg_t5_weights: Optional[torch.Tensor]):
        self.adapter_path = adapter_path
        self.w = w
        self.pos_raw = pos_raw
        self.pos_t5_ids = pos_t5_ids
        self.pos_t5_weights = pos_t5_weights
        self.neg_raw = neg_raw
        self.neg_t5_ids = neg_t5_ids
        self.neg_t5_weights = neg_t5_weights
        # Computed lazily on first forward when model is on GPU
        self.guidance_delta: Optional[torch.Tensor] = None
        # Persistent hook state: set before each forward, cleared after
        self._current_t_emb_delta: Optional[torch.Tensor] = None
        self._hook_handle = None

    def ensure_guidance_delta(self, dit, device, dtype):
        """Compute guidance delta using the LLM adapter. Runs once."""
        if self.guidance_delta is not None:
            return

        adapter_state = _get_adapter_typed(self.adapter_path, device, dtype)

        with torch.no_grad():
            # Positive: quality tags through LLM adapter
            pos_adapted = dit.preprocess_text_embeds(
                self.pos_raw.unsqueeze(0).to(device=device, dtype=dtype),
                self.pos_t5_ids.unsqueeze(0).to(device=device),
                t5xxl_weights=(
                    self.pos_t5_weights.unsqueeze(0).unsqueeze(-1).to(device=device, dtype=dtype)
                    if self.pos_t5_weights is not None else None
                ),
            )
            pos_pooled = pos_adapted.max(dim=1).values  # (1, pooled_dim)

            # Negative: from conditioning through LLM adapter
            neg_adapted = dit.preprocess_text_embeds(
                self.neg_raw.unsqueeze(0).to(device=device, dtype=dtype),
                self.neg_t5_ids.unsqueeze(0).to(device=device) if self.neg_t5_ids is not None else None,
                t5xxl_weights=(
                    self.neg_t5_weights.unsqueeze(0).unsqueeze(-1).to(device=device, dtype=dtype)
                    if self.neg_t5_weights is not None else None
                ),
            )
            neg_pooled = neg_adapted.max(dim=1).values  # (1, pooled_dim)

        proj_pos = _project(pos_pooled, adapter_state)
        proj_neg = _project(neg_pooled, adapter_state)
        self.guidance_delta = (self.w * (proj_pos - proj_neg)).detach()
        logger.info(
            f"Mod guidance: delta computed (w={self.w}, "
            f"\u2016\u0394\u2016={self.guidance_delta.norm().item():.4f})"
        )


def _extract_transformer_options(args, kwargs):
    to = kwargs.get("transformer_options")
    if isinstance(to, dict):
        return to
    if len(args) > 0 and isinstance(args[-1], dict):
        return args[-1]
    if len(args) > 1 and isinstance(args[-2], dict):
        return args[-2]
    return {}


def _mod_wrapper(executor, *args, **kwargs):
    """DIFFUSION_MODEL wrapper: injects base proj + guidance delta into t_emb."""
    to = _extract_transformer_options(args, kwargs)
    mod_state = to.get(MOD_STATE_KEY)
    if mod_state is None:
        return executor(*args, **kwargs)

    dit = executor.class_obj
    context = args[2]  # post-LLM-adapter from extra_conds
    device = context.device
    dtype = context.dtype

    # Lazy-compute guidance delta on first forward (model is on GPU now)
    mod_state.ensure_guidance_delta(dit, device, dtype)

    adapter_state = _get_adapter_typed(mod_state.adapter_path, device, dtype)

    # Base: proj(max_pool(context)) — makes t_emb text-aware per sample
    pooled = context.max(dim=1).values  # (B, pooled_dim)
    proj_base = _project(pooled, adapter_state)  # (B, model_channels)

    # Guidance delta: w * (proj(pos) - proj(neg)) — constant quality steering
    delta = mod_state.guidance_delta.to(dtype=proj_base.dtype, device=device)
    if delta.shape[0] != proj_base.shape[0]:
        delta = delta.expand(proj_base.shape[0], -1)

    combined = proj_base + delta

    # Cache for Spectrum fast-forward on cached steps
    dit._mod_pooled_proj = combined.detach()

    # Set delta for the persistent hook (installed once in setup_mod_guidance).
    # Avoids per-step register/remove that causes torch.compile recompilation.
    mod_state._current_t_emb_delta = combined.unsqueeze(1).to(dtype)
    try:
        result = executor(*args, **kwargs)
    finally:
        mod_state._current_t_emb_delta = None

    return result


def setup_mod_guidance(model_clone, clip, negative, adapter_name, quality_tags, w):
    """Encode quality tags, extract negative conditioning, register wrapper.

    Called from the KSampler node's sample() before sampling starts.
    Raw tensors are stored on CPU; the LLM adapter runs lazily on first forward.
    """
    adapter_path = folder_paths.get_full_path("loras", adapter_name)
    if adapter_path is None:
        raise RuntimeError(f"Adapter not found: {adapter_name}")

    # Validate adapter against model
    dm = model_clone.model.diffusion_model
    adapter_cpu = _load_adapter_cpu(adapter_path)
    model_channels = getattr(dm, "model_channels", None)
    if model_channels is None:
        raise RuntimeError("Model missing model_channels")
    if adapter_cpu["0.weight"].shape[0] != model_channels:
        raise RuntimeError(
            f"Adapter output dim ({adapter_cpu['0.weight'].shape[0]}) "
            f"!= model_channels ({model_channels})"
        )

    # Encode positive quality tags via CLIP
    tokens = clip.tokenize(quality_tags)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    pos_raw = output["cond"][0].detach().cpu()  # (seq, dim) — single sample
    pos_t5_ids = output.get("t5xxl_ids")
    if pos_t5_ids is not None:
        pos_t5_ids = pos_t5_ids.detach().cpu()
    pos_t5_weights = output.get("t5xxl_weights")
    if pos_t5_weights is not None:
        pos_t5_weights = pos_t5_weights.detach().cpu()

    # Extract negative raw + t5 IDs from CONDITIONING
    neg_cond_tensor = negative[0][0]  # (1, seq, dim) or (seq, dim)
    neg_meta = negative[0][1]
    neg_raw = neg_cond_tensor[0].detach().cpu() if neg_cond_tensor.ndim == 3 else neg_cond_tensor.detach().cpu()
    neg_t5_ids = neg_meta.get("t5xxl_ids")
    if neg_t5_ids is not None:
        neg_t5_ids = neg_t5_ids.detach().cpu()
    neg_t5_weights = neg_meta.get("t5xxl_weights")
    if neg_t5_weights is not None:
        neg_t5_weights = neg_t5_weights.detach().cpu()

    mod_state = ModGuidanceState(
        adapter_path=adapter_path, w=w,
        pos_raw=pos_raw, pos_t5_ids=pos_t5_ids, pos_t5_weights=pos_t5_weights,
        neg_raw=neg_raw, neg_t5_ids=neg_t5_ids, neg_t5_weights=neg_t5_weights,
    )

    # Install persistent t_emb hook once (not per-step) to avoid
    # torch.compile recompilation from hook set changes.
    dm = model_clone.model.diffusion_model

    def _persistent_t_emb_hook(module, input, output):
        delta = mod_state._current_t_emb_delta
        if delta is not None:
            return output + delta
        return output

    mod_state._hook_handle = dm.t_embedding_norm.register_forward_hook(
        _persistent_t_emb_hook
    )

    # Register DIFFUSION_MODEL wrapper
    model_clone.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MOD_WRAPPER_KEY
    )
    opts = model_clone.model_options.setdefault("transformer_options", {})
    opts[MOD_STATE_KEY] = mod_state
    model_clone.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MOD_WRAPPER_KEY, _mod_wrapper
    )
