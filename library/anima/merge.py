"""Bake a LoRA adapter into the base DiT and save as a standalone safetensors.

The merged output is a standalone DiT checkpoint (ComfyUI-compatible, `net.`
prefixed) that reproduces LoRA+base inference without needing the adapter at
load time.

Supported: plain LoRA, OrthoLoRA, T-LoRA. (T-LoRA's timestep mask is
training-only — inference already runs full rank, so baking is bit-equivalent.)

Not supported (refuse by default; ``allow_partial=True`` to drop and proceed):
  - HydraLoRA moe     (layer-local router can't be baked under static weights)
  - postfix / prefix  (cross-attn KV splice, not a weight delta)
  - register tokens   (ride the self-attn sequence; kept-live inference only)
  - res-cond proj     (E25b t-embedding conditioning input, not a ΔW)

Same merge path as train.py's --base_weights warm-start. The CLI shell over
this lives in ``scripts/toolkits/merge_to_dit.py``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import torch

from library.anima import weights as anima_weights

logger = logging.getLogger(__name__)


# Non-bakeable marker (substring of a safetensors key) → human-readable kind.
_NON_BAKEABLE_MARKERS: dict[str, str] = {
    ".lora_up_weight": "HydraLoRA stacked (per-layer router)",
    ".lora_ups.": "HydraLoRA split (per-layer router) / step-expert turbo (per-step heads)",
    "postfix_": "postfix (cross-attn KV splice)",
    "prefix_": "prefix (cross-attn KV splice)",
    "register_tokens": "register tokens (ride the self-attn sequence, not a weight delta)",
    "sigma_lowres_res_cond_proj": "sigma_lowres res-cond projection (E25b conditioning input, not a weight delta)",
}

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class NonBakeableError(RuntimeError):
    """Raised when an adapter carries components that can't be folded into DiT
    Linear weights and ``allow_partial`` was not set."""


def pick_latest_adapter(adapter_dir: Path) -> Path:
    """Latest `*.safetensors` in adapter_dir that is bakeable.

    Skips ``*_moe.safetensors`` (HydraLoRA router-live), ``*.bak.*`` (backups),
    and any file whose name contains ``postfix`` / ``prefix`` (those are
    separate non-weight-delta adapters).
    """
    candidates = sorted(
        (
            f
            for f in adapter_dir.glob("*.safetensors")
            if not f.name.endswith("_moe.safetensors")
            and ".bak." not in f.name
            and "postfix" not in f.name.lower()
            and "prefix" not in f.name.lower()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No bakeable *.safetensors found in {adapter_dir} "
            "(excludes *_moe, *postfix*, *prefix*, and *.bak.*)"
        )
    return candidates[0]


def scan_non_bakeable_keys(weights_sd: dict) -> dict[str, int]:
    """Return ``{kind: count}`` for any key that matches a non-bakeable marker."""
    found: dict[str, int] = {}
    for key in weights_sd.keys():
        for marker, kind in _NON_BAKEABLE_MARKERS.items():
            if marker in key:
                found[kind] = found.get(kind, 0) + 1
                break
    return found


def merge_adapter_into_dit(
    adapter: Path,
    dit: Path,
    out: Path | None = None,
    *,
    multiplier: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
    allow_partial: bool = False,
    network_module: str = "networks.lora_anima",
) -> Path:
    """Bake ``adapter`` into the base ``dit`` and write a standalone checkpoint.

    Returns the output path written. Raises :class:`NonBakeableError` if the
    adapter carries Hydra-moe / postfix / prefix components and ``allow_partial``
    is False.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"adapter: {adapter}")

    from safetensors.torch import load_file

    weights_sd = load_file(str(adapter))
    non_bakeable = scan_non_bakeable_keys(weights_sd)
    if non_bakeable:
        parts = [f"{count} {kind}" for kind, count in non_bakeable.items()]
        msg = "Non-bakeable keys detected: " + ", ".join(parts) + "."
        if not allow_partial:
            raise NonBakeableError(
                msg
                + " Re-run with allow_partial to drop them and bake the LoRA portion, "
                "or retrain without these components. These cannot be folded into DiT Linear weights."
            )
        logger.warning(
            msg
            + " allow_partial set; these components will be absent from the merged DiT."
        )

    logger.info(f"loading base DiT: {dit}")
    unet = anima_weights.load_anima_model(
        device=device,
        dit_path=str(dit),
        attn_mode="torch",  # merge never runs a forward pass
        loading_device=device,
        dit_weight_dtype=dtype,
    )

    logger.info(f"building adapter network from weights (multiplier={multiplier})")
    network_mod = importlib.import_module(network_module)
    network, weights_sd = network_mod.create_network_from_weights(
        multiplier, str(adapter), None, None, unet, for_inference=True
    )

    logger.info("merging adapter into DiT")
    network.merge_to(None, unet, weights_sd, dtype, device)

    out = out or adapter.with_name(adapter.stem + "_merged.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ss_merged_from": adapter.name,
        "ss_merge_multiplier": str(multiplier),
        "ss_base_dit": dit.name,
    }
    logger.info(f"saving merged DiT: {out}")
    anima_weights.save_anima_model(str(out), unet.state_dict(), metadata, dtype=dtype)
    logger.info("done")
    return out
