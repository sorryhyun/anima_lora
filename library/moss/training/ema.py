"""EMA of the adapter weights."""

from __future__ import annotations

import torch


def ema_decay(span: float, updates: int) -> float:
    """Decay whose EMA averages ``span`` of the run's ``updates`` (1/(1 - decay))."""
    return 1.0 - 1.0 / max(1.0, span * updates)


def ema_step(
    shadows: list[list], staging: list, params: list, decays: list[float]
) -> None:
    """Copy ``params`` into the pinned CPU ``staging`` buffers once and lerp each
    fp32 shadow toward them at its decay; a shadow's first call (empty) seeds it."""
    with torch.no_grad():
        for buf, p in zip(staging, params):
            buf.copy_(p)
        for shadow, decay in zip(shadows, decays):
            if shadow:
                torch._foreach_lerp_(shadow, staging, 1.0 - decay)
            else:
                shadow.extend(buf.clone() for buf in staging)


def ema_step_card(
    shadows: list[list], staging: list, params: list, decays: list[float]
) -> None:
    """:func:`ema_step` with the shadows on the card: no host copy per update, and
    ``staging`` serves only the saves."""
    with torch.no_grad():
        for shadow, decay in zip(shadows, decays):
            if shadow:
                torch._foreach_lerp_(shadow, params, 1.0 - decay)
            else:
                shadow.extend(p.detach().clone() for p in params)
