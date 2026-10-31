"""Anima's side of the int8 base (``library/int8/``): its convert targets, its checkpoint
keys, and the conversions the trainer and ``inference.py`` run.

Training (``--int8 fwd|full``) converts after ``create_network``, so ``weight_svd`` /
``grad_svd`` init still read bf16, and before ``apply_to``, so the LoRA modules capture
the int8 forward as ``org_forward``, and ``compile_blocks``. Generation (``--int8``)
converts in mode ``infer`` with the model on the GPU, before the adapters attach. LoRAs
then ride the P-GRAFT hook path, since a static merge would fold them into the bf16
weight the conversion drops: the base runs ``int8(x, W_q) + B·A·x``, the function the
trainer trained.

The forward copies are ``--int8_file``, else ``<dit>_int8_gptq.safetensors`` beside the
DiT, else round to nearest (``--int8_file rtn`` forces that). User doc:
``docs/anima/optimizations/int8.md``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch.nn as nn
from safetensors import safe_open

from library.anima.weights import _SELF_ATTN_QKV_ORDER
from library.env import resolve_under_home
from library.int8 import linear as _linear
from library.int8.linear import convert_

logger = logging.getLogger(__name__)

ROUND_TO_NEAREST = "rtn"  # --int8_file, and ss_int8_base_id on a LoRA
# per block: self qkv/out, cross q/out, MLP 1/2. Cross kv (M = context length) and
# adaLN (M = batch) stay bf16; phase 0 measured int8 slower there.
CONVERT_SUFFIXES = (
    "self_attn.qkv_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
)
_DIT_BLOCK = re.compile(r"^blocks\.\d+\.")


def ckpt_keys(name: str) -> list[str]:
    """A converted module's weight keys in the checkpoint (``net.`` prefix, the fused
    ``qkv_proj`` split back into q / k / v)."""
    if name.endswith("self_attn.qkv_proj"):
        base = name.removesuffix("qkv_proj")
        return [f"net.{base}{p}.weight" for p in _SELF_ATTN_QKV_ORDER]
    return [f"net.{name}.weight"]


def load_int8_file(path: str, name_keys=ckpt_keys) -> tuple[dict, dict]:
    """``({module name: (wq, s)}, metadata)`` from a ``dit_gptq.py`` file (q / k / v
    rows concatenated into ``qkv_proj``)."""
    return _linear.load_int8_file(path, name_keys)


def convert_targets(dit: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Every DiT block's large linears (not the llm_adapter's)."""
    return [
        (name, mod)
        for name, mod in dit.named_modules()
        if isinstance(mod, nn.Linear)
        and _DIT_BLOCK.match(name)
        and name.endswith(CONVERT_SUFFIXES)
    ]


def base_fingerprint(dit: nn.Module, targets=None) -> str:
    """``linear.base_fingerprint`` over ``targets``, by default the DiT's."""
    return _linear.base_fingerprint(targets or convert_targets(dit))


def convert_dit_(dit: nn.Module, mode: str, int8_file: str | None = None) -> list[str]:
    """Convert the DiT's large linears; returns the converted names. ``int8_file``
    supplies the forward copies, and must cover every target and match the loaded
    checkpoint's ``base_fingerprint``."""
    copies = {}
    if int8_file:
        copies, meta = load_int8_file(int8_file)
        if meta.get("base_fingerprint") != base_fingerprint(dit):
            raise ValueError(
                f"{int8_file} was made from {meta.get('source')}, not the loaded DiT"
            )
    names = []
    for name, mod in convert_targets(dit):
        if int8_file and name not in copies:
            raise ValueError(f"{int8_file} has no forward copy for {name}")
        convert_(mod, mode, copies.get(name))
        names.append(name)
    return names


def forward_file(int8_file: str | None, dit_path: str) -> str:
    """The file of the forward copies: ``int8_file``, else the one beside the DiT, which
    on the catalog's DiT is fetched on first use; empty = round to nearest."""
    if int8_file == ROUND_TO_NEAREST:
        return ""
    if int8_file:
        return str(resolve_under_home(int8_file))
    dit = Path(resolve_under_home(dit_path))
    beside = dit.with_name(f"{dit.stem}_int8_gptq.safetensors")
    if beside.is_file():
        return str(beside)
    from library import downloads

    asset = downloads.by_id()["anima_dit_int8"]
    if beside.resolve() != (asset.dest / downloads.ANIMA_DIT_INT8_FILE).resolve():
        return ""
    logger.info(f"int8 base: fetching {asset.repo} -> {asset.dest}")
    downloads.fetch(asset)
    return str(beside)


def _convert(dit: nn.Module, mode: str, int8_file: str | None, dit_path: str) -> str:
    """Convert and log; returns the id LoRAs are stamped with."""
    path = forward_file(int8_file, dit_path)
    names = convert_dit_(dit, mode, path or None)
    if not names:
        raise SystemExit("int8 base: no DiT linear was converted")
    base_id = ROUND_TO_NEAREST
    if path:
        with safe_open(path, framework="pt") as f:
            base_id = (f.metadata() or {})["int8_id"]
    logger.info(
        f"int8 {mode}: converted {len(names)} linears, forward copies "
        f"{path or 'round to nearest'}"
    )
    return base_id


def check_plan(args) -> None:
    """The training plans an int8 base cannot run; before anything loads."""
    if args.blocks_to_swap:
        raise SystemExit(
            "int8 base: blocks_to_swap > 0 is refused (not validated on Anima's "
            "offloader); run --int8 off for a swap plan"
        )
    if args.base_weights and forward_file(
        args.int8_file, args.pretrained_model_name_or_path
    ):
        raise SystemExit(
            "int8 base: --base_weights change the weights the forward copies were "
            f"calibrated on; pass --int8_file {ROUND_TO_NEAREST}"
        )


def convert_for_training(unet: nn.Module, network, args) -> None:
    """Convert in mode ``args.int8`` and set ``args.int8_base_id``, the LoRA's stamp."""
    from networks.lora_anima.network import LoRANetwork

    if not isinstance(network, LoRANetwork):
        raise SystemExit(
            f"int8 base: {type(network).__name__} is not a LoRANetwork; run --int8 off"
        )
    args.int8_base_id = _convert(
        unet, args.int8, args.int8_file, args.pretrained_model_name_or_path
    )


def convert_for_inference(model: nn.Module, args) -> None:
    """Mode ``infer``, then each LoRA's stamps against this base, so a LoRA trained on
    another base is visible."""
    if getattr(model, "blocks_to_swap", None):
        raise SystemExit("int8 base: blocks_to_swap > 0 is refused")
    if getattr(args, "lora_cutoff_step", None) is not None:
        raise SystemExit("int8 base: --lora_cutoff_step is not supported")
    base_id = _convert(model, "infer", getattr(args, "int8_file", None), args.dit)
    for path in args.lora_weight or []:
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        mode = meta.get("ss_int8_base", "none (bf16 base)")
        lora_id = meta.get("ss_int8_base_id", "")
        match = "matches" if lora_id == base_id else "differs from"
        logger.info(
            f"int8 infer: {path} trained on int8 base {mode} {lora_id[:16] or '-'}, "
            f"{match} this base"
        )
