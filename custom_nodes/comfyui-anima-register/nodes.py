"""Anima register-token adapter — ComfyUI inference node (consolidated).

One node: takes a stock Anima ``MODEL`` + a register ``adapter.safetensors``
and returns a patched ``MODEL`` a normal KSampler runs with the registers live.
Two accepted layouts:

* the standalone register method (``train.py --method register`` /
  ``networks/methods/register.py``) — registers + trained QKV surface;
* a LoRA-family checkpoint trained with ``num_registers > 0`` (one top-level
  ``register_tokens`` tensor next to standard ``lora_unet_*`` keys) — this
  node applies BOTH halves: the registers via the eager register forward and
  the LoRA ΔW via ComfyUI's standard weight patching. Do NOT also load the
  same file through a LoRA loader (the ΔW would apply twice).

The adapter (see ``register_apply.py``) is comfy-native — it patches the split
``q_proj``/``k_proj``/``v_proj`` of ``MiniTrainDIT`` (no fused ``qkv_proj``) and
reimplements the forward to inject/strip register tokens. Patching is applied
around **each** forward call against the resident ``diffusion_model`` and
restored before returning, so it survives ComfyUI's clone/reload/recompile and
never leaves the shared model mutated (no cross-workflow contamination, no
stale-strength reuse, no "memory leak with model" GC warnings).

Constraints:
* Do NOT stack with the Block Compile node — the register mechanism runs eager.
* Loads the canonical split ``.safetensors`` (config in the header metadata).
"""

from __future__ import annotations

import json
import logging
import weakref
from pathlib import Path

from .register_apply import RegisterComfyAdapter

log = logging.getLogger("comfyui-anima-register")

HERE = Path(__file__).resolve().parent
# In-tree install: .../anima_lora/custom_nodes/comfyui-anima-register -> repo root.
REPO_ROOT = HERE.parents[1]
# Dirs scanned for the dropdown (kept tight so we only stat a handful of headers).
_SCAN_ROOTS = ("output/ckpt", "output/temp", "_archive/bench/headroom/results")
_NO_ADAPTERS = "<no register adapters found — use path_override>"


def _is_register_safetensors(path: Path) -> bool:
    """Cheap header-only check that a .safetensors is a register adapter."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        return meta.get("ss_network_spec") == "register" or "ss_num_registers" in meta
    except Exception:
        return False


def _list_register_adapters() -> list[str]:
    """Repo-relative paths of every register adapter under the scan roots."""
    found = []
    for rel in _SCAN_ROOTS:
        root = REPO_ROOT / rel
        if not root.is_dir():
            continue
        for p in root.rglob("*.safetensors"):
            if _is_register_safetensors(p):
                found.append(p.relative_to(REPO_ROOT).as_posix())
    return sorted(set(found))


def _fold_inv_scale(lora_sd: dict) -> dict:
    """Fold per_channel_scaling ``inv_scale`` into ``lora_down`` and drop it.

    Mirrors the Anima LoRA loader / ``LoRAModule.merge_to``: the trained
    forward is ``F.linear(x * inv_scale, down)``, so ``down *= inv_scale``
    column-wise before handing the dict to ComfyUI's patcher (which would
    otherwise warn on the unknown ``.inv_scale`` suffix and apply a delta
    off by ``s_norm`` per input column). Returns a new dict.
    """
    inv_keys = [k for k in lora_sd if k.endswith(".inv_scale")]
    if not inv_keys:
        return lora_sd
    out = dict(lora_sd)
    for inv_key in inv_keys:
        prefix = inv_key[: -len(".inv_scale")]
        down_key = f"{prefix}.lora_down.weight"
        inv_scale = out.pop(inv_key)
        down = out.get(down_key)
        if down is None or down.dim() != 2:
            continue
        out[down_key] = down.float() * inv_scale.float().unsqueeze(0)
    return out


def _apply_lora_half(model, lora_sd: dict, strength: float) -> None:
    """Apply the ``lora_unet_*`` half via ComfyUI's standard weight patching."""
    import comfy.lora
    import comfy.lora_convert

    lora_sd = _fold_inv_scale(lora_sd)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora_sd), key_map)
    model.add_patches(loaded, strength)


def _load_adapter(
    path: str, strength: float
) -> tuple[RegisterComfyAdapter, dict | None]:
    """Load a register ``.safetensors`` into a ready-to-apply adapter.

    The checkpoint carries the split QKV surface + ``ss_*`` config in the header
    metadata (written by ``AdapterNetworkBase.save_weights`` / register's
    ``metadata_fields``).

    Returns ``(register_adapter, lora_sd)`` — ``lora_sd`` is the standard
    ``lora_unet_*`` state dict for a LoRA-family checkpoint (``None`` for the
    standalone register method, which has no LoRA half).
    """
    from safetensors import safe_open

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"register adapter not found: {p}")

    state_dict, meta = {}, {}
    with safe_open(str(p), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    if "ss_num_registers" not in meta:
        raise ValueError(
            f"{p} is not a register adapter (no ss_num_registers metadata)"
        )

    lora_sd = None
    if "ss_target_blocks" in meta:
        # Standalone register method (`--method register`): registers + a
        # trained QKV surface, keys `register` + `qkv.{bi}.*`.
        config = {
            "num_registers": int(meta["ss_num_registers"]),
            "arm": meta.get("ss_arm", "B"),
            "qkv_mode": meta.get("ss_qkv_mode", "unfrozen"),
            "target_blocks": json.loads(meta["ss_target_blocks"]),
            "scale": float(meta.get("ss_scale", 1.0)),
            # pre-insert_block checkpoints trained with entry insertion — default 0
            "insert_block": int(meta.get("ss_insert_block", 0)),
        }
    else:
        # LoRA-family checkpoint trained with `num_registers > 0`: one
        # top-level `register_tokens` tensor rides alongside standard
        # `lora_unet_*` keys. The registers go through the eager register
        # forward; the LoRA half is returned for standard weight patching
        # (registers were trained jointly with the ΔW — apply both).
        if "register_tokens" not in state_dict:
            raise ValueError(
                f"{p} stamps ss_num_registers but carries neither a register "
                "method layout (ss_target_blocks) nor a `register_tokens` key"
            )
        lora_sd = {k: v for k, v in state_dict.items() if k != "register_tokens"}
        state_dict = {"register": state_dict["register_tokens"]}
        config = {
            "num_registers": int(meta["ss_num_registers"]),
            "arm": "B",
            "qkv_mode": "unfrozen",
            "target_blocks": [],  # no QKV surface — ΔW rides the lora_sd half
            "scale": 1.0,
            "insert_block": int(meta.get("ss_register_insert_block", 8)),
        }
    log.info(
        "loaded register adapter %s (%s, lora_keys=%d)",
        p.name,
        config,
        len(lora_sd) if lora_sd else 0,
    )
    return RegisterComfyAdapter(state_dict, config, strength=strength), lora_sd


class AnimaRegisterAdapter:
    """Load + apply a register adapter onto a MODEL in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _list_register_adapters() or [_NO_ADAPTERS]
        return {
            "required": {
                "model": ("MODEL",),
                "adapter_name": (choices,),
            },
            "optional": {
                # Non-empty wins over the dropdown — for a path outside the roots.
                "path_override": ("STRING", {"default": "", "multiline": False}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "anima"

    def apply(
        self, model, adapter_name: str, path_override: str = "", strength: float = 1.0
    ):
        sel = (path_override or "").strip() or adapter_name
        if sel == _NO_ADAPTERS:
            raise ValueError(
                f"no register adapter selected; scanned {_SCAN_ROOTS} under "
                f"{REPO_ROOT} — set path_override to an adapter.safetensors"
            )

        m = model.clone()
        reg_adapter, lora_sd = _load_adapter(sel, strength)
        # LoRA-family checkpoint: apply the ΔW half eagerly via ComfyUI's
        # weight patcher (it participates in the normal patch/unpatch cycle);
        # only the register injection needs the per-call patching below.
        if lora_sd:
            _apply_lora_half(m, lora_sd, strength)

        # Weakref only: a wrapper closing over its own patcher forms a cycle
        # through model_options, and patcher clones that die only in a full gc
        # batch strand their LoadedModel entry (ComfyUI's parent-rescue is
        # weakref-based and CPython clears weakrefs before finalizers run) →
        # "memory leak with model Anima" warnings on every later prompt.
        m_ref = weakref.ref(m)

        def unet_wrapper(apply_model, args):
            # Patch whatever diffusion_model is resident *right now* and restore
            # it before returning — the shared module is never left mutated.
            # Prefer the live BaseModel actually running this forward (survives
            # downstream rebuilds, e.g. AnimaBlockCompile's fresh clone).
            owner = getattr(apply_model, "__self__", None)
            dm = getattr(owner, "diffusion_model", None)
            if dm is None:
                dm = m_ref().model.diffusion_model  # m is alive while sampling
            with reg_adapter.applied(dm):
                return apply_model(args["input"], args["timestep"], **args["c"])

        m.set_model_unet_function_wrapper(unet_wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"AnimaRegisterAdapter": AnimaRegisterAdapter}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaRegisterAdapter": "Anima Register Adapter"}
