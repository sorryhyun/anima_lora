"""MOSS-Image's adapters: ``library/lines/adapters.py`` bound to this DiT.

Default targets are the six bias-free linears of every block,
``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`` (42 × 6 = 252). The fused
``ffn_up_gate`` gets one adapter over its full 2·ffn_dim output, so the hidden
and the gate halves share the rank.
"""

from __future__ import annotations

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
