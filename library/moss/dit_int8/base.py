"""MOSS-Image's side of the int8 base: its 252 target linears, the GPTQ file's forward
copies, and the conversion ``library.moss.train`` runs (``TrainRequest.int8``).

``train`` converts after the network's ``__init__``, so ``down_init weight_svd*`` still
reads bf16, and before ``apply_to``, whose patched forward captures
``module.__class__.forward``, and ``compile_blocks``. A converted linear that drops its
weight keeps an empty bf16 ``weight`` buffer, since the block forward reads
``proj["q"].weight.dtype``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from library.env import resolve_under_home
from library.int8.linear import (
    base_fingerprint,
    convert_,
    load_int8_file,
    stream_wqt_,
)
from library.int8.stream import WqtStream
from library.moss.requests import TrainRequest

ROUND_TO_NEAREST = "rtn"  # TrainRequest.int8_file, and int8_base_id on a LoRA
BLOCK_LINEARS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
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


def convert_moss_(mod, mode: str, forward_copy=None) -> None:
    """``convert_`` on the card. Modes that drop the weight keep an empty bf16
    ``weight`` buffer: the block forward reads its dtype."""
    device = torch.device("cuda")
    mod.to(device)
    convert_(mod, mode, forward_copy)
    if mode != "fwd":
        mod.register_buffer(
            "weight",
            torch.empty(0, dtype=torch.bfloat16, device=device),
            persistent=False,
        )


def stream_targets(by_name: dict, n_blocks: int) -> WqtStream:
    """``wq_t`` to the host, block by block, in the block's forward order."""
    blocks = [
        [by_name[f"layers.{b}.proj.{p}"] for p in BLOCK_LINEARS]
        for b in range(n_blocks)
    ]
    stream = stream_wqt_(blocks)
    # what stays sat between the dropped weights and wq_t and would pin their pages
    device = blocks[0][0].wq.device
    mods = [m for block in blocks for m in block]
    kept = [(m.held_wq, "weight") for m in mods]
    kept += [(m, k) for m in mods for k in ("s", "s_t")]
    for m, k in kept:
        m._buffers[k] = m._buffers[k].cpu()
    torch.cuda.empty_cache()
    for m, k in kept:
        m._buffers[k] = m._buffers[k].to(device)
    return stream


def check_plan(req: TrainRequest) -> None:
    """The plans an int8 base cannot run; before anything loads."""
    if req.algo == "loha":
        raise SystemExit(
            f"int8 base: loha builds W + ΔW against the base weight; {_BF16_PLAN}"
        )


def forward_file(req: TrainRequest) -> str:
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
        if self.stream is None:
            return
        s = self.stream
        print(
            f"wq_t stream: {s.entered} blocks entered, {s.cold} cold, "
            f"{s.wait_s:.2f} s waited",
            flush=True,
        )
        s.close()


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
    mode = "full" if req.int8 == "stream" else req.int8
    for name, mod in targets:
        convert_moss_(mod, mode, copies.get(name))
    stream = None
    if req.int8 == "stream":
        stream = stream_targets(dict(targets), len(transformer.layers))
    torch.cuda.empty_cache()
    print(
        f"int8 {req.int8}: converted {len(targets)} linears, forward copies "
        f"{base_id[:16] if copies else 'round to nearest'}",
        flush=True,
    )
    return TrainBase(req.int8, base_id, stream)
