#!/usr/bin/env python3
"""Bake a LoRA adapter into the base DiT and save as a new safetensors file.

The merged output is a standalone DiT checkpoint (ComfyUI-compatible, `net.`
prefixed) that reproduces LoRA+base inference without needing the adapter at
load time.

Supported: plain LoRA, OrthoLoRA, DoRA, T-LoRA. (T-LoRA's timestep mask is
training-only — inference already runs full rank, so baking is bit-equivalent.)

Not supported (refuse by default; --allow-partial to drop and proceed):
  - ReFT              (block-level hook, not a Linear weight delta)
  - HydraLoRA moe     (layer-local router can't be baked under static weights)
  - postfix / prefix  (cross-attn KV splice, not a weight delta)

Same merge path as train.py:1499's --base_weights warm-start.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.anima import weights as anima_weights  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


# Marker → human-readable kind. Substring match on safetensors keys.
_NON_BAKEABLE_MARKERS: dict[str, str] = {
    "reft_": "ReFT (block-level hook)",
    ".lora_up_weight": "HydraLoRA stacked (per-layer router)",
    ".lora_ups.": "HydraLoRA split (per-layer router)",
    "postfix_": "postfix (cross-attn KV splice)",
    "prefix_": "prefix (cross-attn KV splice)",
}

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bake a LoRA adapter into the base DiT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=Path("output/ckpt"),
        help="Directory to pick the latest adapter from (ignored if --adapter is set).",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Explicit adapter .safetensors path (overrides --adapter_dir).",
    )
    parser.add_argument(
        "--dit",
        type=Path,
        default=Path("models/diffusion_models/anima-preview3-base.safetensors"),
        help="Base DiT safetensors.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="LoRA strength to bake in.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to <adapter-stem>_merged.safetensors next to the adapter.",
    )
    parser.add_argument("--dtype", choices=list(_DTYPE_MAP), default="bf16")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the merge math.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Drop unsupported keys (ReFT / Hydra moe / postfix / prefix) and bake the rest. "
        "The merged DiT will not reproduce those components.",
    )
    parser.add_argument(
        "--network_module",
        default="networks.lora_anima",
        help="Network module providing create_network_from_weights.",
    )
    args = parser.parse_args()

    adapter = args.adapter or pick_latest_adapter(args.adapter_dir)
    logger.info(f"adapter: {adapter}")

    from safetensors.torch import load_file

    weights_sd = load_file(str(adapter))
    non_bakeable = scan_non_bakeable_keys(weights_sd)
    if non_bakeable:
        parts = [f"{count} {kind}" for kind, count in non_bakeable.items()]
        msg = "Non-bakeable keys detected: " + ", ".join(parts) + "."
        if not args.allow_partial:
            logger.error(
                msg
                + " Re-run with --allow-partial to drop them and bake the LoRA portion, "
                "or retrain without these components. These cannot be folded into DiT Linear weights."
            )
            return 2
        logger.warning(
            msg + " --allow-partial set; these components will be absent from the merged DiT."
        )

    dtype = _DTYPE_MAP[args.dtype]

    logger.info(f"loading base DiT: {args.dit}")
    unet = anima_weights.load_anima_model(
        device=args.device,
        dit_path=str(args.dit),
        attn_mode="torch",  # merge never runs a forward pass
        split_attn=False,
        loading_device=args.device,
        dit_weight_dtype=dtype,
    )

    logger.info(
        f"building adapter network from weights (multiplier={args.multiplier})"
    )
    network_module = importlib.import_module(args.network_module)
    network, weights_sd = network_module.create_network_from_weights(
        args.multiplier, str(adapter), None, None, unet, for_inference=True
    )

    logger.info("merging adapter into DiT")
    network.merge_to(None, unet, weights_sd, dtype, args.device)

    out = args.out or adapter.with_name(adapter.stem + "_merged.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ss_merged_from": adapter.name,
        "ss_merge_multiplier": str(args.multiplier),
        "ss_base_dit": args.dit.name,
    }
    logger.info(f"saving merged DiT: {out}")
    anima_weights.save_anima_model(
        str(out), unet.state_dict(), metadata, dtype=dtype
    )
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
