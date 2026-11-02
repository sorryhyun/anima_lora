"""MOSS-Image's adapters: ``library/lines/adapters.py`` bound to this DiT.

Default targets are the six bias-free linears of every block,
``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`` (42 × 6 = 252). The fused
``ffn_up_gate`` gets one adapter over its full 2·ffn_dim output, so the hidden
and the gate halves share the rank.

``ref_rank`` (``TrainRequest.ref_adapters = kv``, ``docs/moss/proposal/reference_adapters.md``
plan A) adds one export adapter per ``proj.k`` / ``proj.v``, saved as
``ref_layers_N_proj_{k,v}``. They are not patched onto the linears: ``apply_to`` hands
them to the block as ``ref_bank``, the reference rows run on Base (``_Block.ref_rows``)
and the deltas reach only the keys and values the image and text rows read.
"""

from __future__ import annotations

import math

import torch

from library.lines.adapters import (  # noqa: F401  (the line's adapter surface)
    ALGOS,
    DOWN_INITS,
    SAVED_DTYPES,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    load_channel_stats,
)
from library.lines.adapters import LoRANetwork as _LoRANetwork
from library.moss.requests import TrainRequest

DEFAULT_TARGETS = TrainRequest.TARGETS


class LoRANetwork(_LoRANetwork):
    BASE_MODEL = "MOSS-Image"
    BLOCKS = "layers"
    DEFAULT_TARGETS = DEFAULT_TARGETS

    def __init__(self, model, *args, ref_rank: int = 0, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.ref_rank = ref_rank
        self._ref_banks = []
        if not ref_rank:
            return
        # mod_rank's alpha rule. Forked so the main adapters' init and the training
        # noise match a run without the bank.
        alpha = self.alpha * math.sqrt(ref_rank / self.rank)
        build = (
            ref_rank,
            alpha,
            kwargs.get("dtype", torch.float32),
            kwargs.get("channel_stats"),
            kwargs.get("channel_mean"),
        )
        with torch.random.fork_rng(devices=[]):
            for i, block in enumerate(model.layers):
                pair = []
                for key in ("k", "v"):
                    name = f"layers.{i}.proj.{key}"
                    adapter = self._build(name, block.proj[key], *build)
                    self.adapters["ref_" + name.replace(".", "_")] = adapter
                    pair.append(adapter)
                self._ref_banks.append((block, tuple(pair)))

    @classmethod
    def _meta_kwargs(cls, meta: dict) -> dict:
        return {"ref_rank": int(meta.get("ref_rank", 0))}

    def apply_to(self) -> int:
        patched = super().apply_to()
        for block, pair in self._ref_banks:
            block.ref_bank = pair
        return patched

    def restore(self) -> None:
        super().restore()
        for block, _pair in self._ref_banks:
            block.ref_bank = None


load_network = LoRANetwork.from_file


def ref_settings(path) -> tuple[float | None, bool]:
    """``(ref_timestep, ref_isolate)`` a reference LoRA trained with, as
    ``MOSSImageTransformer`` takes them. ``ref_timestep`` None (stamped negative or
    absent) = the image's own; ``ref_isolate`` is stamped ``"1"`` or ``"True"``,
    absent = off."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}
    t = meta.get("ref_timestep")
    return (
        None if t is None or float(t) < 0 else float(t),
        meta.get("ref_isolate") in ("1", "True"),
    )


def load_for_inference(
    transformer, path, multiplier: float = 1.0, before_apply=None
) -> LoRANetwork:
    """``LoRANetwork.for_inference``, plus the reference settings the LoRA trained
    with (``ref_settings``) set on ``transformer``."""
    network = LoRANetwork.for_inference(transformer, path, multiplier, before_apply)
    transformer.ref_timestep, transformer.ref_isolate = ref_settings(path)
    return network
