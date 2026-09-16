"""Producer-protocol metric collection.

Every metric source — the LoRA network, the postfix network, each method
adapter — implements ``metrics(ctx) -> dict[str, float]``. The trainer
collects from a flat list of producers at log-step cadence.

Adding a metric means editing the owner that already holds the underlying
state — same file as the loss / forward / scheduler that produces the value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import torch


@dataclass
class MetricContext:
    """What every producer is handed.

    ``network`` is included so adapters can read network-level stashes that
    the loss code writes (the loss code has no adapter handle). Producers
    ignore fields they don't need.
    """

    args: object
    network: object


class MetricProducer(Protocol):
    """Anything that can emit log keys from ``metrics(ctx)``.

    Implementers: ``LoRANetwork``, ``PostfixNetwork``, ``MethodAdapter``
    subclasses. Returning an empty dict is fine and cheap — the trainer
    collects on every log step regardless of which methods are active.
    """

    def metrics(self, ctx: MetricContext) -> dict[str, float]: ...


def collect_metrics(
    producers: Iterable[MetricProducer], ctx: MetricContext
) -> dict[str, float]:
    """Run each producer under ``no_grad`` and merge results.

    A producer raising is contained — metrics must never kill a training
    step — and silent; diagnose by calling the producer directly.
    """
    out: dict[str, float] = {}
    with torch.no_grad():
        for producer in producers:
            fn = getattr(producer, "metrics", None)
            if fn is None:
                continue
            try:
                values = fn(ctx)
            except Exception:
                continue
            if values:
                out.update(values)
    return out
