"""MOSS-Image's side of the int8 base: its 252 target linears, the GPTQ file's forward
copies, and the conversions ``library.moss.train`` and ``library.moss.generate`` run
(``TrainRequest.int8``, ``GenerateRequest.int8``).

``train`` converts after the network's ``__init__``, so ``down_init weight_svd*`` still
reads bf16, and before ``apply_to``, whose patched forward captures
``module.__class__.forward``, and ``compile_blocks``. ``generate`` converts after the
network's ``__init__`` too (LoKr reads the bf16 weight's shape) and before ``apply_to``;
with no LoRA, right after the DiT loads.

Each block's ``proj.q`` / ``k`` / ``v`` become one GEMM, ``block.qkv``, with their rows
concatenated. They read the same ``y``, so the kernel rotates and quantizes it once. The
forward is bit-identical to three calls. ``wq_t`` shares one scale per column of the
weight, and dY one per token, across the three. The three linears stay as
:class:`QKVPart`, which the adapters patch.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors import safe_open

from library.env import resolve_under_home
from library.int8.linear import (
    base_fingerprint,
    close_stream,
    convert_,
    load_int8_file,
    stream_compacted_,
)
from library.int8.stream import WqtStream
from library.moss.requests import GenerateRequest, TrainRequest

ROUND_TO_NEAREST = "rtn"  # the requests' int8_file, and int8_base_id on a LoRA
QKV = ("q", "k", "v")
REST = ("o", "ffn_up_gate", "ffn_down")
_TARGET = re.compile(TrainRequest.TARGETS)
_BF16_PLAN = "--config configs/moss/lora_bf16.toml is the bf16 plan"


def moss_targets(tr) -> list:
    return [(n, m) for n, m in tr.named_modules() if _TARGET.fullmatch(n)]


def moss_keys(name: str) -> list[str]:
    """A converted module's weight key in the Diffusers checkpoint."""
    return [f"{name}.weight"]


def load_copies(int8_file: str) -> tuple[dict, dict]:
    """``({module name: (wq, s)}, metadata)`` from a ``dit_gptq_moss.py`` file."""
    return load_int8_file(str(resolve_under_home(int8_file)), moss_keys)


def check_base(meta: dict, by_name: dict) -> None:
    """Refuse a file made from another DiT; before any conversion. ``by_name``:
    module name → linear, holding at least the two ``base_fingerprint`` hashes."""
    ends = ("layers.0.proj.q", "layers.41.proj.ffn_down")
    if any(n not in by_name for n in ends):
        raise SystemExit("int8 base: the file's fingerprint linears are not targeted")
    if meta.get("base_fingerprint") != base_fingerprint(
        [(n, by_name[n]) for n in ends]
    ):
        raise SystemExit(f"int8 base: the file was made from {meta.get('source')}")


def _empty_weight(mod) -> None:
    mod.register_buffer(
        "weight",
        torch.empty(0, dtype=torch.bfloat16, device=torch.device("cuda")),
        persistent=False,
    )


def convert_moss_(mod, mode: str, forward_copy=None) -> None:
    """``convert_`` on the card. Modes that drop the weight keep an empty bf16
    ``weight`` buffer: the block forward reads ``proj["q"].weight.dtype``."""
    mod.to(torch.device("cuda"))
    convert_(mod, mode, forward_copy)
    if mode != "fwd":
        _empty_weight(mod)


class QKVPart(nn.Linear):
    """``proj.q`` / ``k`` / ``v`` once ``block.qkv`` runs their rows: ``out`` is this
    linear's share of that output, for the adapters to add to."""

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return out


def fuse_qkv_(block, mode: str, copies: dict, name: str) -> None:
    """Build ``block.qkv`` in mode ``mode`` from the three linears of block ``name``,
    with their forward copies from ``copies`` when it has them."""
    parts = [block.proj[p] for p in QKV]
    forward_copy = None
    if copies:
        wq, s = zip(*(copies[f"{name}.proj.{p}"] for p in QKV))
        forward_copy = torch.cat(wq), torch.cat(s)
    weight = torch.cat([part.weight.detach() for part in parts])
    fused = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    fused.weight = nn.Parameter(weight, requires_grad=False)
    convert_moss_(fused, mode, forward_copy)
    for part in parts:
        del part.weight
        _empty_weight(part)
        part.__class__ = QKVPart
    block.qkv = fused


def convert_blocks_(transformer, mode: str, copies: dict) -> None:
    """Every block's linears: ``block.qkv`` and the other three."""
    for i, block in enumerate(transformer.layers):
        fuse_qkv_(block, mode, copies, f"layers.{i}")
        for p in REST:
            convert_moss_(block.proj[p], mode, copies.get(f"layers.{i}.proj.{p}"))


def stream_targets(transformer) -> WqtStream:
    """``wq_t`` to the host, block by block, in the block's forward order."""
    return stream_compacted_(
        [[b.qkv, *(b.proj[p] for p in REST)] for b in transformer.layers]
    )


def check_plan(req: TrainRequest) -> None:
    """The plans an int8 base cannot run; before anything loads."""
    if req.algo == "loha":
        raise SystemExit(
            f"int8 base: loha builds W + ΔW against the base weight; {_BF16_PLAN}"
        )


def forward_file(req: TrainRequest | GenerateRequest) -> str:
    """The file of the forward copies: ``int8_file``, else on Base the catalog's,
    fetched on first use; empty = round to nearest (SFT has no file)."""
    if req.int8_file == ROUND_TO_NEAREST:
        return ""
    if req.int8_file:
        return req.int8_file
    if req.model != "base":
        return ""
    from library import downloads

    asset = downloads.by_id()["moss_dit_int8"]
    target = asset.dest / downloads.MOSS_DIT_INT8_FILE
    if not target.is_file():
        print(f"int8 base: fetching {asset.repo} -> {asset.dest}", flush=True)
        downloads.fetch(asset)
    return str(target)


@dataclass
class TrainBase:
    """A converted DiT: its LoRA stamps and, in mode ``stream``, the ``wq_t`` mover."""

    mode: str
    base_id: str
    stream: WqtStream | None = None

    def metadata(self) -> dict[str, str]:
        return {"int8_base": self.mode, "int8_base_id": self.base_id}

    def close(self) -> None:
        """The stream's counters, then its host buffers."""
        if self.stream is not None:
            close_stream()


def convert_for_training(transformer, req: TrainRequest) -> TrainBase:
    """Convert the 252 linears in mode ``req.int8``; ``stream`` is ``full`` with
    ``wq_t`` then moved to the host."""
    copies, meta, base_id = {}, {}, ROUND_TO_NEAREST
    int8_file = forward_file(req)
    if int8_file:
        copies, meta = load_copies(int8_file)
        base_id = meta["int8_id"]
    targets = moss_targets(transformer)
    if copies:
        check_base(meta, dict(targets))
    convert_blocks_(transformer, "full" if req.int8 == "stream" else req.int8, copies)
    stream = stream_targets(transformer) if req.int8 == "stream" else None
    torch.cuda.empty_cache()
    print(
        f"int8 {req.int8}: converted {len(targets)} linears, forward copies "
        f"{base_id[:16] if copies else 'round to nearest'}",
        flush=True,
    )
    return TrainBase(req.int8, base_id, stream)


def generate_base(req: GenerateRequest) -> Callable[[torch.nn.Module], None]:
    """The renders an int8 base cannot run, before anything loads; returns the
    conversion to mode ``infer`` (forward copy only, bf16 weight dropped)."""
    if req.blocks_to_swap:
        raise SystemExit(
            "int8 base: --blocks_to_swap 0 is required (generation on this base "
            "has run no swap)"
        )
    copies, meta, base_id = {}, {}, ROUND_TO_NEAREST
    int8_file = forward_file(req)
    if int8_file:
        copies, meta = load_copies(int8_file)
        base_id = meta["int8_id"]
    if req.lora:
        with safe_open(str(resolve_under_home(req.lora)), framework="pt") as f:
            lora_meta = f.metadata() or {}
        if lora_meta.get("algo") == "loha":
            raise SystemExit("int8 base: loha builds W + ΔW against the base weight")
        lora_id = lora_meta.get("int8_base_id", "")
        print(
            f"int8 infer: {req.lora} trained on "
            f"{lora_meta.get('int8_base', 'a bf16 base')} {lora_id[:16] or '-'}, "
            f"{'matches' if lora_id == base_id else 'differs from'} this base",
            flush=True,
        )

    def convert(transformer) -> None:
        targets = moss_targets(transformer)
        if copies:
            check_base(meta, dict(targets))
        convert_blocks_(transformer, "infer", copies)
        torch.cuda.empty_cache()
        print(
            f"int8 infer: converted {len(targets)} linears, base {base_id[:16]}",
            flush=True,
        )

    return convert
