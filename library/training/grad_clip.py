"""Gradient clip rules for the training loop: a fixed cap, or a cap at ``k ×`` the
median pre-clip norm of recent updates, applied to the whole update or to each
micro-batch before it joins the accumulated sum.

Every quantity stays on the card (ring buffer, median, clip coefficient), so a
clip adds no host sync. docs/anima/proposal/lora_batch_clip_ema.md.
"""

from __future__ import annotations

from typing import Optional

import torch


class GradClipper:
    """``mode="fixed"``: cap ``max_norm`` (0 = no clip). ``mode="median"``: cap
    ``k ×`` the median of the last ``window`` pre-clip norms, excluding the one
    being clipped; ``max_norm`` until ``warmup`` norms are recorded. A clip
    scales the gradient to the cap and keeps its direction."""

    def __init__(
        self,
        mode: str = "fixed",
        max_norm: float = 1.0,
        k: float = 2.0,
        window: int = 50,
        warmup: int = 10,
    ):
        if mode not in ("fixed", "median"):
            raise ValueError(f"clip_mode must be 'fixed' or 'median', got {mode!r}")
        if mode == "median" and k <= 0:
            raise ValueError(f"clip_k must be > 0, got {k}")
        if window < 1 or warmup < 1 or warmup > window:
            raise ValueError(
                f"need 1 <= clip warmup ({warmup}) <= clip_window ({window})"
            )
        self.mode = mode
        self.max_norm = float(max_norm)
        self.k = float(k)
        self.window = int(window)
        self.warmup = int(warmup)
        self._buf: Optional[torch.Tensor] = None
        self._n = 0

    def cap(self, device) -> Optional[torch.Tensor]:
        """Current cap as a 0-d fp32 tensor, or None when nothing is clipped."""
        if self.mode == "median" and self._n >= self.warmup:
            filled = self._buf[: min(self._n, self.window)]
            return filled.median() * self.k
        if self.max_norm == 0.0:
            return None
        return torch.tensor(self.max_norm, device=device, dtype=torch.float32)

    def _record(self, norm: torch.Tensor) -> None:
        if self.mode != "median":
            return
        if self._buf is None:
            self._buf = torch.zeros(
                self.window, device=norm.device, dtype=torch.float32
            )
        self._buf[self._n % self.window] = norm
        self._n += 1

    @torch.no_grad()
    def clip_(
        self, grads: list[torch.Tensor], scale: float = 1.0
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Clip ``grads`` in place. ``scale`` converts their norm to the unit the
        cap is in (``accum`` for one micro-batch's share of an accumulated
        update). Returns the pre-clip norm (in cap units) and the cap used."""
        norm = torch.nn.utils.get_total_norm(grads).float() * scale
        cap = self.cap(norm.device)
        if cap is not None:
            coef = torch.clamp(cap / (norm + 1e-6), max=1.0)
            torch._foreach_mul_(grads, coef.to(grads[0].dtype))
        self._record(norm)
        return norm, cap


class MicroAccumulator:
    """Per-micro clip under gradient accumulation: after each micro-batch's
    backward, clip that micro's gradient and move it into an fp32 sum, leaving
    ``p.grad`` empty for the next backward; on the update's last micro, write
    the sum back to ``p.grad`` for the optimizer."""

    def __init__(self, clipper: GradClipper, accum: int):
        self.clipper = clipper
        self.accum = int(accum)
        self._sum: dict[torch.nn.Parameter, torch.Tensor] = {}

    @torch.no_grad()
    def absorb(
        self, params: list[torch.nn.Parameter], last: bool
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        live = [p for p in params if p.grad is not None]
        norm = cap = None
        if live:
            norm, cap = self.clipper.clip_([p.grad for p in live], scale=self.accum)
            for p in live:
                acc = self._sum.get(p)
                if acc is None:
                    self._sum[p] = p.grad.float()
                else:
                    acc.add_(p.grad)
                p.grad = None
        if last:
            for p, acc in self._sum.items():
                p.grad = acc.to(p.dtype)
            self._sum.clear()
        return norm, cap
