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

``ref_adapters = all`` (plan B) builds a full second bank at ``ref_rank`` on the ``targets``,
saved as ``ref_layers_N_proj_*``, and patches both onto each linear (``_RowBanks``): the
reference rows run this bank, image and text rows the main one, and the modulation
adapter stays off the reference. Under grad the rows share one forward and
``set_row_mask(main, ref)`` splits them; ``generate`` runs the reference stream alone
(``RefCache``) inside ``reference()``.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager

import torch

from library.lines.adapters import (  # noqa: F401  (the line's adapter surface)
    ALGOS,
    DOWN_INITS,
    SAVED_DTYPES,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    load_channel_stats,
    _patched_forward,
)
from library.lines.adapters import LoRANetwork as _LoRANetwork
from library.moss.requests import TrainRequest

DEFAULT_TARGETS = TrainRequest.TARGETS


class _RowBanks:
    """The two adapters of one linear. ``side`` picks them: ``main`` and ``reference``
    for a forward holding only those rows, ``rows`` for both under their row masks."""

    def __init__(self, main, ref):
        self.main, self.ref = main, ref
        self.side = "main"

    def patched(self, module, base, x):
        out = base(module, x)
        if self.side == "main":
            return self.main(x, out)
        if self.side == "reference":
            return self.ref(x, out)
        return self.ref(x, self.main(x, out))


class LoRANetwork(_LoRANetwork):
    BASE_MODEL = "MOSS-Image"
    BLOCKS = "layers"
    DEFAULT_TARGETS = DEFAULT_TARGETS

    def __init__(self, model, *args, ref_rank: int = 0, ref_adapters="kv", **kwargs):
        super().__init__(model, *args, **kwargs)
        self.ref_rank = ref_rank
        self.ref_adapters = ref_adapters if ref_rank else "off"
        # A tuple: as an attribute the DiT would register as this network's submodule.
        self._holder = ()
        self._ref_banks = []
        self._row_banks = []
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
            if ref_adapters == "all":
                self._holder = (model,)
                extra = [re.compile(p) for p in self.rank_pattern]
                self._pairs = []
                for name, module in self._targets:
                    if any(p.fullmatch(name) for p in extra):
                        continue
                    key = name.replace(".", "_")
                    adapter = self._build(name, module, *build)
                    self.adapters["ref_" + key] = adapter
                    self._pairs.append((module, self.adapters[key], adapter))
                return
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
        return {
            "ref_rank": int(meta.get("ref_rank", 0)),
            "ref_adapters": meta.get("ref_adapters", "kv"),
        }

    def apply_to(self) -> int:
        patched = super().apply_to()
        for block, pair in self._ref_banks:
            block.ref_bank = pair
        for model in self._holder:
            model.ref_scope = self.reference
            self._row_banks = []
            for module, main, ref in self._pairs:
                banks = _RowBanks(main, ref)
                module.forward = _patched_forward(module, banks)
                self._row_banks.append(banks)
        return patched

    def restore(self) -> None:
        super().restore()
        for block, _pair in self._ref_banks:
            block.ref_bank = None
        for model in self._holder:
            model.ref_scope = None
            self._row_banks = []

    def _set_side(self, side: str) -> None:
        for banks in self._row_banks:
            banks.side = side

    @contextmanager
    def reference(self):
        """The forwards inside hold reference rows alone, on the reference bank."""
        side = self._row_banks[0].side
        self._set_side("reference")
        try:
            yield
        finally:
            self._set_side(side)

    def set_row_mask(self, mask, ref_mask=None) -> None:
        """With a reference bank, ``mask`` goes to the main adapters and ``ref_mask`` to
        that bank (both ``(1, L, 1)``, ``training.data.bank_masks``); None takes them off."""
        if self.ref_adapters != "all":
            return super().set_row_mask(mask)
        for name, adapter in self.adapters.items():
            adapter.row_mask = ref_mask if name.startswith("ref_") else mask
        self._set_side("main" if mask is None else "rows")


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
