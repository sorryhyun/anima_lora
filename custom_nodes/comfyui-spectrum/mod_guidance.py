"""Modulation guidance: adapter loading, projection, APPLY_MODEL wrapper.

The wrapper runs *outside* the torch.compile region (at the APPLY_MODEL layer,
before `_apply_model` dispatches to the compiled `diffusion_model`). All tensor
math — projection, guidance delta, per-batch assembly — is precomputed once per
sample and only a cached tensor is written to `t_embedding_norm._anima_mod_delta`
on the hot path. Inside compile, the forward hook does nothing more than
`output + delta`, which dynamo inlines cleanly.
"""

import logging
import os
import threading
import urllib.request
import weakref
from typing import List, Optional

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import folder_paths

logger = logging.getLogger(__name__)

PROJ_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")
MOD_APPLY_WRAPPER_KEY = "spectrum_mod_guidance_apply"
MOD_DIFFUSION_WRAPPER_KEY = "spectrum_mod_guidance"  # legacy key for cleanup
MOD_STATE_KEY = "spectrum_mod_guidance_state"

AUTO_ADAPTER_SENTINEL = "(auto-download default)"
DEFAULT_ADAPTER_FILENAME = "pooled_text_proj_0413.safetensors"
DEFAULT_ADAPTER_URL = (
    "https://github.com/sorryhyun/anima_lora/releases/download/"
    "mod_guidance/pooled_text_proj_0413.safetensors"
)
DEFAULT_ADAPTER_SUBDIR = "anima_mod_guidance"

_ADAPTER_LOCK = threading.Lock()
_ADAPTER_CPU_CACHE: dict = {}
_ADAPTER_TYPED_CACHE: dict = {}
_DOWNLOAD_LOCK = threading.Lock()


def get_default_adapter_path() -> str:
    """Return local path to the default pooled_text_proj adapter, downloading if missing."""
    target_dir = os.path.join(folder_paths.models_dir, DEFAULT_ADAPTER_SUBDIR)
    target_path = os.path.join(target_dir, DEFAULT_ADAPTER_FILENAME)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    with _DOWNLOAD_LOCK:
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path
        os.makedirs(target_dir, exist_ok=True)
        tmp_path = target_path + ".download"
        logger.info(
            f"Mod guidance: downloading default adapter (~12MB) from {DEFAULT_ADAPTER_URL}"
        )
        try:
            urllib.request.urlretrieve(DEFAULT_ADAPTER_URL, tmp_path)
            os.replace(tmp_path, target_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        logger.info(f"Mod guidance: saved adapter to {target_path}")
        return target_path


def _resolve_adapter_path(adapter_name: Optional[str]) -> str:
    if adapter_name in (None, "", AUTO_ADAPTER_SENTINEL):
        return get_default_adapter_path()
    path = folder_paths.get_full_path("loras", adapter_name)
    if path is None:
        raise RuntimeError(f"Adapter not found: {adapter_name}")
    return path


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


class _PerBlockState:
    """Runtime state read by per-block / final-layer pre-hooks.

    Lives on `dit._anima_mod_block_state` for the duration of a single
    `_mod_apply_wrapper` call. Hooks check for `None` to fast-no-op.
    """

    __slots__ = ("delta_unit", "schedule", "final_w")

    def __init__(self, delta_unit: torch.Tensor, schedule: List[float], final_w: float):
        self.delta_unit = delta_unit  # (1, C) — proj_tag - proj_neg
        self.schedule = schedule       # length num_blocks; w_l per block
        self.final_w = final_w         # scalar w applied to final_layer


class ModGuidanceState:
    """Per-sample state. Holds raw text tensors at construction time and caches
    the precomputed base projections + steering delta after the first forward.

    All hot-path work happens in the APPLY_MODEL wrapper, outside torch.compile.
    The compiled DiT forward only sees a single tensor read inside the t_emb hook
    (base projection) plus per-block pre-hooks that add `w(ℓ) * delta_unit`.
    """

    def __init__(
        self,
        adapter_path: str,
        w: float,
        tag_raw: torch.Tensor,
        tag_t5_ids: Optional[torch.Tensor],
        tag_t5_weights: Optional[torch.Tensor],
        neg_raw: torch.Tensor,
        neg_t5_ids: Optional[torch.Tensor],
        neg_t5_weights: Optional[torch.Tensor],
        pos_raw: torch.Tensor,
        pos_t5_ids: Optional[torch.Tensor],
        pos_t5_weights: Optional[torch.Tensor],
        dit,
        start_layer: int = 0,
        end_layer: int = -1,
        taper: int = 0,
        taper_scale: float = 0.25,
        final_w: float = 0.0,
    ):
        self.adapter_path = adapter_path
        self.w = w
        self.tag_raw = tag_raw
        self.tag_t5_ids = tag_t5_ids
        self.tag_t5_weights = tag_t5_weights
        self.neg_raw = neg_raw
        self.neg_t5_ids = neg_t5_ids
        self.neg_t5_weights = neg_t5_weights
        self.pos_raw = pos_raw
        self.pos_t5_ids = pos_t5_ids
        self.pos_t5_weights = pos_t5_weights
        self.dit = dit
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.taper = int(taper)
        self.taper_scale = float(taper_scale)
        self.final_w = float(final_w)
        # Computed lazily on first forward when DiT is on GPU
        self.cond_combined: Optional[torch.Tensor] = None      # (1, C) base proj_pos
        self.uncond_combined: Optional[torch.Tensor] = None    # (1, C) base proj_neg
        self.delta_unit: Optional[torch.Tensor] = None         # (1, C) proj_tag - proj_neg
        self.per_block_schedule: Optional[List[float]] = None  # length num_blocks

    def _encode_pool(self, raw, t5_ids, t5_weights, device, dtype):
        adapted = self.dit.preprocess_text_embeds(
            raw.unsqueeze(0).to(device=device, dtype=dtype),
            t5_ids.unsqueeze(0).to(device=device) if t5_ids is not None else None,
            t5xxl_weights=(
                t5_weights.unsqueeze(0).unsqueeze(-1).to(device=device, dtype=dtype)
                if t5_weights is not None
                else None
            ),
        )
        return adapted.max(dim=1).values  # (1, pooled_dim)

    def ensure_precomputed(self, device, dtype):
        """Run LLM adapter + projection for pos / neg / tag once, cache base
        projections and unit steering delta. Schedule built off live block count."""
        if self.cond_combined is not None:
            return

        adapter_state = _get_adapter_typed(self.adapter_path, device, dtype)
        with torch.no_grad():
            tag_pooled = self._encode_pool(
                self.tag_raw, self.tag_t5_ids, self.tag_t5_weights, device, dtype
            )
            neg_pooled = self._encode_pool(
                self.neg_raw, self.neg_t5_ids, self.neg_t5_weights, device, dtype
            )
            pos_pooled = self._encode_pool(
                self.pos_raw, self.pos_t5_ids, self.pos_t5_weights, device, dtype
            )
            proj_tag = _project(tag_pooled, adapter_state)
            proj_neg = _project(neg_pooled, adapter_state)
            proj_pos = _project(pos_pooled, adapter_state)
            # Base projections — added uniformly via t_embedding_norm hook,
            # matches the line-1640 training-time injection.
            self.cond_combined = proj_pos.detach()       # (1, C)
            self.uncond_combined = proj_neg.detach()     # (1, C)
            # Unit steering direction — scaled per-block by `self.per_block_schedule`.
            self.delta_unit = (proj_tag - proj_neg).detach()
        self._build_schedule()
        logger.info(
            f"Mod guidance: precomputed "
            f"(w={self.w}, start={self.start_layer}, end={self.end_layer}, "
            f"taper={self.taper}, taper_scale={self.taper_scale}, final_w={self.final_w})"
        )

    def _build_schedule(self):
        num_blocks = len(self.dit.blocks)
        end = num_blocks if self.end_layer < 0 else min(self.end_layer, num_blocks)
        start = max(0, min(self.start_layer, end))
        sched = [0.0] * num_blocks
        w = float(self.w)
        for i in range(start, end):
            sched[i] = w
        if self.taper > 0 and end > start:
            taper_start = max(start, end - self.taper)
            taper_w = w * float(self.taper_scale)
            for i in range(taper_start, end):
                sched[i] = taper_w
        self.per_block_schedule = sched


def _t_emb_forward_hook(module, input, output):
    """Module-singleton forward hook — reads ambient delta from the module itself.

    Registered exactly once per t_embedding_norm instance. Per-sample state is
    passed through the `_anima_mod_delta` attribute that `_mod_apply_wrapper` sets/clears
    around each compiled forward. Keeping the hook identity stable across runs
    lets torch.compile's dynamo cache survive between samples.
    """
    delta = getattr(module, "_anima_mod_delta", None)
    if delta is not None:
        return output + delta
    return output


def _ensure_t_emb_hook(dm) -> None:
    t_norm = dm.t_embedding_norm
    if getattr(t_norm, "_anima_mod_hook_installed", False):
        return
    # Pre-initialize so the attribute always exists; dynamo specializes on
    # `is None` vs tensor. Setting it here once means the None branch only
    # gets traced when mod guidance is not active for this forward.
    t_norm._anima_mod_delta = None
    t_norm.register_forward_hook(_t_emb_forward_hook)
    t_norm._anima_mod_hook_installed = True


def _ensure_block_hooks(dm) -> None:
    """Install pre-forward hooks on each block + final_layer that apply
    `w(ℓ) * delta_unit` to the t_embedding argument. Hooks read state from
    `dm._anima_mod_block_state`; when None they fast-no-op."""
    if getattr(dm, "_anima_mod_block_hooks_installed", False):
        return

    dm_ref = weakref.ref(dm)

    def _make_block_prehook(idx: int):
        def _prehook(module, args, kwargs):
            owner = dm_ref()
            if owner is None:
                return None
            st = getattr(owner, "_anima_mod_block_state", None)
            if st is None or st.schedule is None:
                return None
            if idx >= len(st.schedule):
                return None
            w_l = st.schedule[idx]
            if w_l == 0.0:
                return None
            if len(args) < 2:
                return None
            t_emb = args[1]
            delta = st.delta_unit
            if delta.device != t_emb.device or delta.dtype != t_emb.dtype:
                delta = delta.to(device=t_emb.device, dtype=t_emb.dtype)
                st.delta_unit = delta
            new_t_emb = t_emb + (w_l * delta).unsqueeze(1)
            new_args = (args[0], new_t_emb) + tuple(args[2:])
            return new_args, kwargs
        return _prehook

    def _final_prehook(module, args, kwargs):
        owner = dm_ref()
        if owner is None:
            return None
        st = getattr(owner, "_anima_mod_block_state", None)
        if st is None or st.final_w == 0.0:
            return None
        if len(args) < 2:
            return None
        t_emb = args[1]
        delta = st.delta_unit
        if delta.device != t_emb.device or delta.dtype != t_emb.dtype:
            delta = delta.to(device=t_emb.device, dtype=t_emb.dtype)
            st.delta_unit = delta
        new_t_emb = t_emb + (st.final_w * delta).unsqueeze(1)
        new_args = (args[0], new_t_emb) + tuple(args[2:])
        return new_args, kwargs

    for idx, blk in enumerate(dm.blocks):
        blk.register_forward_pre_hook(_make_block_prehook(idx), with_kwargs=True)
    dm.final_layer.register_forward_pre_hook(_final_prehook, with_kwargs=True)
    dm._anima_mod_block_state = None
    dm._anima_mod_block_hooks_installed = True


def _mod_apply_wrapper(executor, *args, **kwargs):
    """APPLY_MODEL wrapper: runs outside torch.compile and writes the prebuilt
    combined tensor to `t_embedding_norm._anima_mod_delta` before the compiled
    DiT forward fires.

    Positional args (from `BaseModel.apply_model`):
        (x, t, c_concat, c_crossattn, control, transformer_options, **extra_conds)
    """
    transformer_options = args[5] if len(args) > 5 and isinstance(args[5], dict) else kwargs.get("transformer_options", {})
    mod_state = transformer_options.get(MOD_STATE_KEY)
    if mod_state is None:
        return executor(*args, **kwargs)

    x = args[0]
    device = x.device
    model = executor.class_obj  # BaseModel
    dtype = model.get_dtype_inference()
    dit = mod_state.dit  # original diffusion_model ref captured at setup

    # Lazy one-shot: run LLM adapter + projection for pos/neg/tag and build
    # the two (1, C) combined tensors. Subsequent calls early-out.
    mod_state.ensure_precomputed(device, dtype)

    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
    cond_c = mod_state.cond_combined.to(device=device, dtype=dtype)
    uncond_c = mod_state.uncond_combined.to(device=device, dtype=dtype)
    pieces = [cond_c if cou == 0 else uncond_c for cou in cond_or_uncond]
    combined = torch.cat(pieces, dim=0)  # (B, C)

    # Expose for Spectrum fast-forward (eager, cached steps) and for the
    # t_emb hook (compiled path). Both writes are outside torch.compile.
    dit._mod_pooled_proj = combined.detach()
    t_norm = dit.t_embedding_norm
    t_norm._anima_mod_delta = combined.unsqueeze(1)

    # Per-block / final-layer hooks read this. delta_unit cast to runtime dtype
    # once here; hooks lazy-recast if needed.
    delta_unit_typed = mod_state.delta_unit.to(device=device, dtype=dtype)
    dit._anima_mod_block_state = _PerBlockState(
        delta_unit=delta_unit_typed,
        schedule=mod_state.per_block_schedule,
        final_w=mod_state.final_w,
    )
    try:
        return executor(*args, **kwargs)
    finally:
        t_norm._anima_mod_delta = None
        dit._anima_mod_block_state = None


def _extract_raw_and_t5(conditioning):
    """Pull raw text-encoder output + optional t5 IDs/weights from the first
    entry of a CONDITIONING list and return them as CPU tensors.
    """
    cond_tensor = conditioning[0][0]  # (1, seq, dim) or (seq, dim)
    meta = conditioning[0][1]
    raw = (
        cond_tensor[0].detach().cpu()
        if cond_tensor.ndim == 3
        else cond_tensor.detach().cpu()
    )
    t5_ids = meta.get("t5xxl_ids")
    if t5_ids is not None:
        t5_ids = t5_ids.detach().cpu()
    t5_weights = meta.get("t5xxl_weights")
    if t5_weights is not None:
        t5_weights = t5_weights.detach().cpu()
    return raw, t5_ids, t5_weights


def setup_mod_guidance(
    model_clone,
    clip,
    positive,
    negative,
    adapter_name,
    quality_tags,
    w,
    *,
    start_layer: int = 0,
    end_layer: int = -1,
    taper: int = 0,
    taper_scale: float = 0.25,
    final_w: float = 0.0,
):
    """Capture raw tensors for quality tags / positive / negative, install the
    t_emb / per-block / final-layer hooks, and register the APPLY_MODEL wrapper.

    Called from the KSampler node's sample() before sampling starts. The LLM
    adapter and projection run lazily on the first compiled forward (when the
    DiT is on GPU) and produce the cached `cond_combined` / `uncond_combined`
    base projections plus `delta_unit` steering direction stored on
    `ModGuidanceState`.

    Schedule params (per-block guidance shape):
        start_layer:  inclusive; first block to receive `w * delta_unit`.
        end_layer:    exclusive; last block + 1. -1 means num_blocks.
        taper:        number of late slots inside [start, end) to scale by `taper_scale`.
        taper_scale:  multiplier on tapered slots (default 0.25).
        final_w:      `w` applied at `final_layer` (default 0 = don't disturb).
    """
    adapter_path = _resolve_adapter_path(adapter_name)

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

    # Encode quality tags via CLIP (same text encoder the sampler uses)
    tokens = clip.tokenize(quality_tags)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    tag_raw = output["cond"][0].detach().cpu()
    tag_t5_ids = output.get("t5xxl_ids")
    if tag_t5_ids is not None:
        tag_t5_ids = tag_t5_ids.detach().cpu()
    tag_t5_weights = output.get("t5xxl_weights")
    if tag_t5_weights is not None:
        tag_t5_weights = tag_t5_weights.detach().cpu()

    # Extract positive (user prompt) and negative raw embeddings from CONDITIONING
    pos_raw, pos_t5_ids, pos_t5_weights = _extract_raw_and_t5(positive)
    neg_raw, neg_t5_ids, neg_t5_weights = _extract_raw_and_t5(negative)

    mod_state = ModGuidanceState(
        adapter_path=adapter_path,
        w=w,
        tag_raw=tag_raw,
        tag_t5_ids=tag_t5_ids,
        tag_t5_weights=tag_t5_weights,
        neg_raw=neg_raw,
        neg_t5_ids=neg_t5_ids,
        neg_t5_weights=neg_t5_weights,
        pos_raw=pos_raw,
        pos_t5_ids=pos_t5_ids,
        pos_t5_weights=pos_t5_weights,
        dit=dm,
        start_layer=start_layer,
        end_layer=end_layer,
        taper=taper,
        taper_scale=taper_scale,
        final_w=final_w,
    )

    # Install the t_emb forward hook exactly once per DiT instance. Module-level
    # hook reads its state from `t_embedding_norm._anima_mod_delta`, which is
    # set / cleared by `_mod_apply_wrapper` outside the compile boundary.
    _ensure_t_emb_hook(dm)
    # Install per-block + final-layer pre-hooks for scheduled steering delta.
    _ensure_block_hooks(dm)

    # Clean up any legacy DIFFUSION_MODEL wrapper from previous versions
    model_clone.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MOD_DIFFUSION_WRAPPER_KEY
    )

    # Register APPLY_MODEL wrapper — runs outside torch.compile
    model_clone.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL, MOD_APPLY_WRAPPER_KEY
    )
    opts = model_clone.model_options.setdefault("transformer_options", {})
    opts[MOD_STATE_KEY] = mod_state
    model_clone.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        MOD_APPLY_WRAPPER_KEY,
        _mod_apply_wrapper,
    )
