"""LoRA / HydraLoRA / ReFT loading and application for the Anima adapter node.

A single safetensors file may contain any combination of the three; each is
auto-detected from key patterns. Plain LoRA goes through ComfyUI's
weight-patch path. HydraLoRA and ReFT are applied as per-Linear / per-block
``forward_hook``s swapped in via ``ModelPatcher.add_object_patch``
(overriding ``forward`` strands block weights on CPU under ComfyUI's
cast-weights path). Hydra hooks reproduce the trained
``HydraLoRAModule.forward`` exactly — per-sample router gate, per-expert
``lora_up`` blend — so style separation actually fires at inference time
instead of being averaged out by a uniform bake.
"""

import logging
import math
import re
from collections import OrderedDict
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Cache: path -> parsed adapter bundle ({"lora": dict|None, "hydra": dict|None, "reft": dict|None}).
_adapter_cache: Dict[str, dict] = {}

_REFT_KEY_RE = re.compile(r"^reft_unet_blocks_(\d+)\.(.+)$")


def _parse_reft(weights_sd: Dict[str, torch.Tensor]) -> Optional[Dict[int, dict]]:
    """Group ReFT keys by block index. Returns None if no ReFT keys present."""
    by_idx: Dict[int, Dict[str, torch.Tensor]] = {}
    for key, value in weights_sd.items():
        m = _REFT_KEY_RE.match(key)
        if m is None:
            continue
        idx = int(m.group(1))
        by_idx.setdefault(idx, {})[m.group(2)] = value

    if not by_idx:
        return None

    reft: Dict[int, dict] = {}
    for idx, d in by_idx.items():
        if "rotate_layer.weight" not in d or "learned_source.weight" not in d:
            logger.warning(
                f"ReFT block {idx} is missing rotate_layer.weight or "
                f"learned_source.weight — skipping."
            )
            continue
        rotate = d["rotate_layer.weight"]
        reft_dim = rotate.size(0)
        alpha_t = d.get("alpha")
        if alpha_t is None:
            alpha = float(reft_dim)
        else:
            alpha = float(alpha_t.item() if hasattr(alpha_t, "item") else alpha_t)
        reft[idx] = {
            "rotate": rotate,                              # (reft_dim, x_dim)
            "source_w": d["learned_source.weight"],        # (reft_dim, x_dim)
            "source_b": d["learned_source.bias"],          # (reft_dim,)
            "scale": alpha / reft_dim,
        }
    return reft or None


def _parse_hydra(weights_sd: Dict[str, torch.Tensor]) -> Optional[dict]:
    """Group Hydra multi-head keys. Returns None if no per-expert ups found.

    Captures router (``router.weight`` / ``router.bias``). σ-conditional
    routing is driven by the router input directly — sinusoidal(σ) is
    concatenated onto the pooled rank-R vector, so the router weight is
    ``Linear(rank + sigma_feature_dim, E)``; σ dim is recovered downstream
    from ``router_w.shape[1] - lora_down.shape[0]``. The legacy additive
    ``sigma_mlp.*`` bias path was removed on the training side (see
    ``docs/methods/hydra-lora.md`` §Fixes); no current checkpoint writes
    those keys, so we ignore them.

    Only prefixes with per-expert ``lora_ups`` are returned; prefixes that
    are plain LoRA (``lora_up.weight`` singular) are left to the
    ``_extract_lora_sd`` path.
    """
    modules: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if key.startswith("reft_"):
            continue
        parts = key.split(".")
        prefix = parts[0]
        rest = ".".join(parts[1:])
        mod = modules.setdefault(prefix, {})
        if rest == "lora_down.weight":
            mod["lora_down"] = value
        elif rest.startswith("lora_ups.") and rest.endswith(".weight"):
            idx = int(rest.split(".")[1])
            mod.setdefault("lora_ups", {})[idx] = value
        elif rest == "alpha":
            mod["alpha"] = value
        elif rest == "inv_scale":
            mod["inv_scale"] = value
        elif rest == "router.weight":
            mod["router_w"] = value
        elif rest == "router.bias":
            mod["router_b"] = value

    # Keep only real hydra modules (have per-expert ups). Prefixes that only
    # carry plain LoRA keys (lora_up.weight, singular) flow through the
    # standard LoRA path; including them here would just produce noisy
    # "missing lora_ups" skip warnings.
    hydra_only: Dict[str, dict] = {
        prefix: mod for prefix, mod in modules.items() if "lora_ups" in mod
    }
    if not hydra_only:
        return None
    num_experts = max(max(m["lora_ups"].keys()) + 1 for m in hydra_only.values())
    return {"num_experts": num_experts, "modules": hydra_only}


def _extract_lora_sd(
    weights_sd: Dict[str, torch.Tensor],
) -> Optional[Dict[str, torch.Tensor]]:
    """Pull standard LoRA keys (lora_down/lora_up/alpha/dora_scale).
    Returns None if no lora_up.weight keys are present.

    Hydra-prefix keys are excluded *entirely* — not just the per-expert
    ``.lora_ups.*`` ones. A hydra module's ``lora_down.weight`` / ``router.*``
    / ``alpha`` are orphans from ComfyUI's ``load_lora`` perspective and
    would surface as "lora key not loaded" warnings if passed through; they
    belong to the hydra path.
    """
    hydra_prefixes = {
        key.rsplit(".lora_ups.", 1)[0]
        for key in weights_sd
        if ".lora_ups." in key
    }

    out: Dict[str, torch.Tensor] = {}
    has_up = False
    for key, value in weights_sd.items():
        if key.startswith("reft_"):
            continue
        if key.endswith(".lora_up_weight"):
            continue  # Hydra stacked-ups runtime form (shouldn't appear post-save)
        prefix = key.split(".", 1)[0]
        if prefix in hydra_prefixes:
            continue  # Hydra module — handled via _parse_hydra
        out[key] = value
        if key.endswith(".lora_up.weight"):
            has_up = True
    return out if has_up else None


def load_adapter(file_path: str) -> dict:
    """Parse an Anima adapter file once, cache by path."""
    if file_path in _adapter_cache:
        return _adapter_cache[file_path]

    from safetensors.torch import load_file

    weights_sd = load_file(file_path)

    if any(k.startswith("_hydra_router") for k in weights_sd.keys()):
        raise ValueError(
            f"{file_path} uses the deprecated global HydraLoRA router format. "
            "Retrain with the current codebase."
        )

    bundle = {
        "path": file_path,
        "lora": _extract_lora_sd(weights_sd),
        "hydra": _parse_hydra(weights_sd),
        "reft": _parse_reft(weights_sd),
    }
    _adapter_cache[file_path] = bundle

    summary = []
    if bundle["lora"] is not None:
        summary.append(
            f"{sum(1 for k in bundle['lora'] if k.endswith('.lora_up.weight'))} LoRA modules"
        )
    if bundle["hydra"] is not None:
        summary.append(
            f"Hydra({bundle['hydra']['num_experts']} experts, "
            f"{len(bundle['hydra']['modules'])} modules)"
        )
    if bundle["reft"] is not None:
        summary.append(f"ReFT({len(bundle['reft'])} blocks)")
    logger.info(
        f"Loaded Anima adapter: {', '.join(summary) or 'empty'} from {file_path}"
    )
    return bundle


def _sigma_sinusoidal_features(
    sigma: torch.Tensor, sigma_feature_dim: int
) -> torch.Tensor:
    """Sinusoidal σ features matching ``networks/lora_modules.py`` verbatim.

    Trained σ-conditional bias is sensitive to this functional form, so any
    drift between training and inference shows up directly as wrong gates.
    """
    t = sigma.flatten().float()
    half_dim = sigma_feature_dim // 2
    exponent = (
        -math.log(10000)
        * torch.arange(half_dim, dtype=torch.float32, device=t.device)
        / max(half_dim, 1)
    )
    freqs = torch.exp(exponent)
    angles = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def _make_hydra_hook(params: dict, strength: float, sigma_state: dict):
    """Forward hook reproducing ``HydraLoRAModule.forward`` per Linear.

    Lazy-moves loaded tensors to the input's device on first call (saved
    dtype is preserved; bottleneck matmuls upcast to fp32 to match the CLI
    precision policy — see ``LoRAModule.forward`` rationale). ``sigma_state``
    is shared across all hydra hooks for this checkpoint; the
    diffusion-forward wrapper writes ``sigma_state["sigma"]`` once per
    denoising step, and each hook reads it to build sinusoidal(σ) features
    that are concatenated onto the pooled rank-R router input when
    ``sigma_feature_dim > 0``.
    """
    state = {
        "lora_down": params["lora_down"],
        "lora_ups": params["lora_ups"],          # (E, out, rank)
        "router_w": params["router_w"],          # (E, rank + sigma_feature_dim)
        "router_b": params["router_b"],          # (E,)
        "inv_scale": params.get("inv_scale"),    # (in_dim,) or None
        "scale": params["scale"],
        "sigma_feature_dim": int(params.get("sigma_feature_dim", 0)),
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        for k in ("lora_down", "lora_ups", "router_w", "router_b", "inv_scale"):
            if state[k] is not None:
                state[k] = state[k].to(device=x.device)
        state["device"] = x.device

    def hydra_hook(module, inputs, output):
        x = inputs[0]
        _ensure_on_device(x)

        x_lora = x.float()
        if state["inv_scale"] is not None:
            x_lora = x_lora * state["inv_scale"].float()

        # down projection (B, *, rank), fp32 — feeds both the router and the
        # gate-weighted bmm downstream.
        lx = torch.nn.functional.linear(x_lora, state["lora_down"].float())

        # router gate — RMS pool the rank-R signal over the sequence dim, then
        # optionally concat sinusoidal(σ) before the router linear. Mirrors
        # HydraLoRAModule._compute_gate after the σ-input rewiring (see
        # docs/methods/hydra-lora.md §Fixes): previously σ entered as an
        # additive sigma_mlp bias on the logits; the router now reads it as
        # input, so the σ-feature columns of router.weight train on the same
        # chain rule as the content columns.
        B = lx.shape[0]
        if lx.dim() >= 3:
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        if state["sigma_feature_dim"] > 0 and sigma_state.get("sigma") is not None:
            sigma_feat = _sigma_sinusoidal_features(
                sigma_state["sigma"], state["sigma_feature_dim"]
            ).to(device=pooled.device, dtype=pooled.dtype)
            # Broadcast σ features across the batch when σ is shape (1,) but
            # pooled is CFG-doubled.
            if sigma_feat.shape[0] == 1 and pooled.shape[0] != 1:
                sigma_feat = sigma_feat.expand(pooled.shape[0], -1)
            pooled = torch.cat([pooled, sigma_feat], dim=-1)
        elif state["sigma_feature_dim"] > 0:
            # σ-conditional router but no σ captured yet (shouldn't happen
            # under the wrapper, but stay safe): zero-pad to keep shape.
            pooled = torch.nn.functional.pad(
                pooled, (0, state["sigma_feature_dim"])
            )
        logits = torch.nn.functional.linear(
            pooled, state["router_w"].float(), state["router_b"].float()
        )
        gate = torch.softmax(logits, dim=-1)

        # gate-weighted combined ups (B, out, rank)
        combined = torch.einsum("be,eor->bor", gate, state["lora_ups"].float())

        # apply via batched matmul
        orig_shape = lx.shape
        lx_3d = lx.reshape(B, -1, orig_shape[-1])
        delta = torch.bmm(lx_3d, combined.transpose(1, 2)).reshape(
            *orig_shape[:-1], -1
        )
        return output + (delta * (state["scale"] * strength)).to(output.dtype)

    return hydra_hook


def _make_sigma_pre_hook(sigma_state: dict):
    """Forward pre-hook that records the diffusion-step ``timesteps`` arg.

    Each hydra hook reads ``sigma_state["sigma"]`` to compute the
    σ-conditional router input. ``args[1]`` is ``timesteps`` from
    ``BaseModel._apply_model`` (``self.diffusion_model(xc, t, ...)``) — the
    only call site of the DiT in inference. Hook is a pure dict store; no
    args are modified, so we return ``None``.

    Why a pre-hook rather than overriding ``diffusion_model.forward``:
    replacing ``forward`` via ``add_object_patch`` strands sub-Linears
    (e.g. cosmos ``x_embedder.proj``) on CPU under ComfyUI's lowvram-aware
    load path — exactly the failure mode that retired the old
    ``block.forward`` override in favor of ``_forward_hooks``. A pre-hook
    leaves ``forward`` untouched and torch.compile traces cleanly through it
    (with the dynamo-disable guard below for safety on the dict store).
    """

    @torch._dynamo.disable
    def sigma_pre_hook(module, args):
        if len(args) >= 2:
            sigma_state["sigma"] = args[1]

    return sigma_pre_hook


def _resolve_module(model, dotted_path: str):
    """Walk attribute / index path under ``model.model``."""
    obj = model.model
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _apply_hydra_live_to_model(
    model, hydra_data: dict, strength: float
) -> int:
    """Install live-routing forward hooks on each Hydra-adapted Linear.

    Replaces the previous uniform-bake fallback. Per-Linear hooks reproduce
    the trained ``HydraLoRAModule.forward`` (per-sample router from layer
    input, per-expert ``lora_up`` blend) so the multi-head specialization
    fires at inference. σ-conditional router bias is captured via a forward
    pre-hook on ``diffusion_model`` that records ``timesteps`` into shared
    state read by each hook.

    Returns number of hooks installed.
    """
    import comfy.lora

    if strength == 0:
        return 0

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})

    sigma_state: dict = {}

    # Install a forward pre-hook on diffusion_model to record σ. Patch
    # _forward_pre_hooks (an OrderedDict) via add_object_patch so it's
    # reverted on ModelPatcher.unpatch_model. Composes with any prior
    # diffusion_model.forward object_patch (postfix wraps forward; the
    # pre-hook fires before that wrapper sees args).
    diffusion_model = model.get_model_object("diffusion_model")
    sigma_pre_hook = _make_sigma_pre_hook(sigma_state)
    new_pre_hooks = OrderedDict(diffusion_model._forward_pre_hooks)
    new_pre_hooks[id(sigma_pre_hook)] = sigma_pre_hook
    model.add_object_patch(
        "diffusion_model._forward_pre_hooks", new_pre_hooks
    )

    patched = 0
    skipped: list[str] = []
    for prefix, mod in hydra_data["modules"].items():
        if "lora_down" not in mod or "lora_ups" not in mod:
            skipped.append(f"{prefix}: missing lora_down/lora_ups")
            continue
        if "router_w" not in mod or "router_b" not in mod:
            skipped.append(f"{prefix}: missing router")
            continue

        comfy_sd_key = key_map.get(prefix)
        if comfy_sd_key is None:
            skipped.append(f"{prefix}: not in ComfyUI key_map")
            continue
        module_path = (
            comfy_sd_key[: -len(".weight")]
            if comfy_sd_key.endswith(".weight")
            else comfy_sd_key
        )

        try:
            linear = _resolve_module(model, module_path)
        except (AttributeError, IndexError, ValueError) as e:
            skipped.append(f"{prefix}: resolve {module_path} failed ({e})")
            continue

        ups_dict = mod["lora_ups"]
        ups_stacked = torch.stack(
            [ups_dict[i] for i in sorted(ups_dict.keys())], dim=0
        )
        rank = mod["lora_down"].shape[0]

        # Router input is either rank (σ off) or rank + sigma_feature_dim
        # (σ concatenated onto the pooled rank-R vector — see
        # HydraLoRAModule._compute_gate).
        router_in = mod["router_w"].shape[1]
        sigma_feature_dim = router_in - rank
        if sigma_feature_dim < 0:
            skipped.append(
                f"{prefix}: router input {router_in} < rank {rank} "
                f"(shape {tuple(mod['router_w'].shape)}) — checkpoint malformed"
            )
            continue
        alpha_t = mod.get("alpha")
        alpha = (
            float(alpha_t.item() if hasattr(alpha_t, "item") else alpha_t)
            if alpha_t is not None
            else float(rank)
        )

        params = {
            "lora_down": mod["lora_down"],
            "lora_ups": ups_stacked,
            "router_w": mod["router_w"],
            "router_b": mod["router_b"],
            "inv_scale": mod.get("inv_scale"),
            "scale": alpha / rank,
            "sigma_feature_dim": sigma_feature_dim,
        }

        hook = _make_hydra_hook(params, strength, sigma_state)
        new_hooks = OrderedDict(linear._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(f"{module_path}._forward_hooks", new_hooks)
        patched += 1

    if skipped:
        logger.warning(
            f"Hydra live-routing skipped {len(skipped)} prefix(es); "
            f"first few: {skipped[:5]}"
        )
    has_sigma = any(
        "router_w" in m
        and "lora_down" in m
        and m["router_w"].shape[1] > m["lora_down"].shape[0]
        for m in hydra_data["modules"].values()
    )
    logger.info(
        f"Hydra live-routing installed {patched} hooks "
        f"(strength={strength}, σ-conditional={'yes' if has_sigma else 'no'})"
    )
    return patched


def _apply_lora_sd_to_model(model, lora_sd: Dict[str, torch.Tensor], strength: float):
    """Apply a standard LoRA state_dict via ComfyUI's weight patching."""
    import comfy.lora
    import comfy.lora_convert

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    model.add_patches(loaded, strength)


def _make_reft_hook(params: dict, strength: float):
    """Forward hook that adds ReFT's residual edit to the block output.

    Hook params (rotate / source_w / source_b) are moved/cast to the input's
    device/dtype on first call and cached for subsequent calls.
    """
    scale = params["scale"]
    state = {
        "rotate": params["rotate"],
        "source_w": params["source_w"],
        "source_b": params["source_b"],
        "ready": False,
    }

    def reft_hook(module, inputs, output):
        h = output
        if (
            not state["ready"]
            or state["rotate"].device != h.device
            or state["rotate"].dtype != h.dtype
        ):
            state["rotate"] = state["rotate"].to(device=h.device, dtype=h.dtype)
            state["source_w"] = state["source_w"].to(device=h.device, dtype=h.dtype)
            state["source_b"] = state["source_b"].to(device=h.device, dtype=h.dtype)
            state["ready"] = True
        delta = torch.nn.functional.linear(h, state["source_w"], state["source_b"])
        edit = torch.nn.functional.linear(delta, state["rotate"].T)
        return h + edit * (strength * scale)

    return reft_hook


def _apply_reft_to_model(model, reft_blocks: Dict[int, dict], strength: float) -> int:
    """Install per-block ReFT edits as ComfyUI object patches.

    Uses a ``forward_hook`` per block (swapped in by replacing the block's
    ``_forward_hooks`` OrderedDict via ``add_object_patch``) instead of
    overriding ``block.forward``. Replacing ``forward`` interferes with
    ComfyUI's weight-loading path — the block's Linears were ending up with
    ``comfy_cast_weights=False`` and their weights left on CPU, producing a
    device mismatch when the block ran. A forward hook leaves ``forward``
    (and ComfyUI's view of it) untouched, and torch.compile traces cleanly
    through it.

    Returns the number of blocks actually patched. The original
    ``_forward_hooks`` dict is restored on ``ModelPatcher.unpatch_model``.
    """
    diffusion = model.get_model_object("diffusion_model")
    if not hasattr(diffusion, "blocks"):
        raise ValueError(
            "ReFT adapter requires a DiT with `.blocks` ModuleList "
            f"(got {type(diffusion).__name__})."
        )
    num_blocks = len(diffusion.blocks)

    patched = 0
    for idx, params in reft_blocks.items():
        if idx < 0 or idx >= num_blocks:
            logger.warning(
                f"ReFT block index {idx} out of range [0, {num_blocks}); skipping"
            )
            continue
        block = diffusion.blocks[idx]
        hook = _make_reft_hook(params, strength)
        new_hooks = OrderedDict(block._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(
            f"diffusion_model.blocks.{idx}._forward_hooks", new_hooks
        )
        patched += 1
    return patched


def apply_adapter(
    model, file_path: str, strength_lora: float, strength_reft: float
) -> bool:
    """Apply LoRA / Hydra / ReFT components from ``file_path`` to ``model``
    in place. ``model`` must already be a clone. Returns True if anything
    was applied.
    """
    bundle = load_adapter(file_path)
    applied_any = False

    if bundle["hydra"] is not None:
        n = _apply_hydra_live_to_model(model, bundle["hydra"], strength_lora)
        if n > 0:
            applied_any = True
    if bundle["lora"] is not None:
        # Plain LoRA — apply directly. Hydra + plain-LoRA coexist in the same
        # file when ``hydra_router_layers`` is a subset regex (e.g. mlp only):
        # hydra handles mlp prefixes, plain LoRA handles cross_attn / self_attn.
        # ``_parse_hydra`` filters itself to hydra-only prefixes and
        # ``_extract_lora_sd`` skips ``.lora_ups.*`` keys, so the two paths
        # target disjoint modules — no double-patching.
        _apply_lora_sd_to_model(model, bundle["lora"], strength_lora)
        applied_any = True

    if bundle["reft"] is not None:
        n = _apply_reft_to_model(model, bundle["reft"], strength_reft)
        if n > 0:
            applied_any = True

    if not applied_any:
        logger.warning(
            f"Anima adapter at {file_path} contained no recognizable "
            "LoRA, Hydra, or ReFT keys."
        )
    return applied_any
