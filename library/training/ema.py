"""Model-EMA helpers for the bespoke distillation/finetune loops, and the
adapter EMA of the LoRA trainer (:class:`AdapterEMA`).

Loop-internal machinery: import by submodule path, not via the
``library.training`` facade.
"""

import contextlib
import copy

import torch


def make_ema(model):
    """Return a frozen eval-mode deep copy of ``model`` to track as its EMA."""
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema, model, decay):
    """One EMA step: params lerp toward ``model`` at rate ``1 - decay``, buffers copy."""
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def ema_decay(span: float, updates: int) -> float:
    """Decay whose EMA averages ``span`` of the run's ``updates`` (1/(1 - decay))."""
    return 1.0 - 1.0 / max(1.0, span * updates)


def parse_ema_spans(window: float, extra: str) -> list[float]:
    """``[window, *extra]`` (``window`` dropped when 0), each in (0, 1]."""
    try:
        spans = [float(x) for x in (extra or "").split(",") if x.strip()]
    except ValueError:
        raise SystemExit(
            f"ema_extra must be comma-separated floats: {extra!r}"
        ) from None
    spans = ([window] if window else []) + spans
    if not all(0.0 < s <= 1.0 for s in spans):
        raise SystemExit(f"EMA spans must be in (0, 1], got {spans}")
    return spans


class AdapterEMA:
    """fp32 EMA shadows of the adapter's trainable parameters, one per span.

    ``window`` is the main span: checkpoint saves write it in place of the live
    weights once it has started (:meth:`swapped_in`); ``extra`` spans are kept
    beside it. Updates start at ``start``
    (the end of lr warmup); the first update seeds the shadows. Shadows are not
    part of the resumable state, so a resumed run re-seeds at its first update.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        window: float,
        extra: str,
        updates: int,
        start: int,
        on_card: bool = True,
    ):
        self.params = params
        self.spans = spans = parse_ema_spans(window, extra)
        # Index of the shadow the checkpoints carry; None = they carry the live
        # weights and only the extra spans are written beside the final file.
        self.main = 0 if window else None
        self.decays = [ema_decay(s, updates) for s in spans]
        self.start = start
        self.on_card = on_card
        self.shadows: list[list[torch.Tensor]] = [[] for _ in spans]
        self.n_updates = 0
        self.staging = (
            []
            if on_card
            else [
                torch.empty(
                    p.shape,
                    dtype=torch.float32,
                    pin_memory=torch.cuda.is_available(),
                )
                for p in params
            ]
        )

    def describe(self) -> str:
        return (
            f"EMA from update {self.start}, {'card' if self.on_card else 'CPU'} shadow: "
            + ", ".join(
                f"span {s:g} = decay {d:.4f} (~{1 / (1 - d):.0f} updates)"
                for s, d in zip(self.spans, self.decays)
            )
        )

    @torch.no_grad()
    def step(self, rate: float = 1.0) -> None:
        """Call once per optimizer update.

        ``rate`` is this update's LR over the mean LR of the schedule: the decay
        becomes ``decay ** rate``, so a span is a fraction of the run's summed LR
        and not of its update count. 1.0 (a constant LR) is the plain EMA.
        """
        self.n_updates += 1
        if self.n_updates < self.start:
            return
        if self.on_card:
            src = [p.detach().float() for p in self.params]
        else:
            for buf, p in zip(self.staging, self.params):
                buf.copy_(p)
            src = self.staging
        for shadow, decay in zip(self.shadows, self.decays):
            if shadow:
                if rate != 1.0:
                    decay = decay**rate
                torch._foreach_lerp_(shadow, src, 1.0 - decay)
            else:
                shadow.extend(t.clone() for t in src)

    @property
    def started(self) -> bool:
        return bool(self.shadows) and bool(self.shadows[0])

    def metadata(self, k: int = 0) -> dict[str, str]:
        return {
            "ss_ema_window": f"{self.spans[k]:g}",
            "ss_ema_decay": f"{self.decays[k]:.6f}",
            "ss_ema_updates": str(max(0, self.n_updates - self.start + 1)),
        }

    @contextlib.contextmanager
    def swapped_in(self, k: int = 0):
        """Shadow ``k`` in the live parameters for the block; the live weights back
        after. No-op while the shadow is empty."""
        shadow = self.shadows[k]
        if not shadow:
            yield False
            return
        with torch.no_grad():
            backup = [p.detach().to("cpu", copy=True) for p in self.params]
            for p, s in zip(self.params, shadow):
                p.copy_(s)
        try:
            yield True
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)
