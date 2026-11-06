"""Gradient clip rules for the training loop: a fixed cap, or a cap at ``k ×`` the
median pre-clip norm of recent updates, applied to the whole update or to each
micro-batch before it joins the accumulated sum.

With σ bands, the median clip keeps one window per band of the unit's σ, so a
draw is capped against draws at a similar noise level (the grad norm is
U-shaped in σ). Every quantity stays on the card (ring buffers, band lookup,
median, clip coefficient), so a clip adds no host sync.
docs/anima/findings/lora_recipe.md.
"""

from __future__ import annotations

from typing import Optional

import torch


class GradClipper:
    """``mode="fixed"``: cap ``max_norm`` (0 = no clip). ``mode="median"``: cap
    ``k ×`` the median of the last ``window`` pre-clip norms, excluding the one
    being clipped; ``max_norm`` until ``warmup`` norms are recorded. A clip
    scales the gradient to the cap and keeps its direction.

    ``sigma_edges`` (median only): inner band edges in (0, 1). A unit clipped
    with its ``sigma`` is capped at ``k ×`` the median of its band's last
    ``window`` norms once the band holds ``band_warmup``, else at the global
    cap. Every norm also enters the global window."""

    band_warmup = 5

    def __init__(
        self,
        mode: str = "fixed",
        max_norm: float = 1.0,
        k: float = 2.0,
        window: int = 50,
        warmup: int = 10,
        sigma_edges: Optional[list[float]] = None,
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
        edges = sorted(float(e) for e in (sigma_edges or []))
        if edges and mode != "median":
            raise ValueError("clip σ bands need clip_mode median")
        if any(not 0.0 < e < 1.0 for e in edges) or len(set(edges)) != len(edges):
            raise ValueError(f"clip σ band edges must be distinct, in (0, 1): {edges}")
        self.sigma_edges = edges
        self._edges: Optional[torch.Tensor] = None
        self._bands: Optional[torch.Tensor] = None  # (bands, window), NaN = empty
        self._band_n: Optional[torch.Tensor] = None

    def cap(self, device) -> Optional[torch.Tensor]:
        """Current cap as a 0-d fp32 tensor, or None when nothing is clipped."""
        if self.mode == "median" and self._n >= self.warmup:
            filled = self._buf[: min(self._n, self.window)]
            return filled.median() * self.k
        if self.max_norm == 0.0:
            return None
        return torch.tensor(self.max_norm, device=device, dtype=torch.float32)

    def _band(self, sigma: torch.Tensor, device) -> torch.Tensor:
        if self._edges is None:
            nb = len(self.sigma_edges) + 1
            self._edges = torch.tensor(self.sigma_edges, device=device)
            self._bands = torch.full(
                (nb, self.window), float("nan"), device=device, dtype=torch.float32
            )
            self._band_n = torch.zeros(nb, device=device, dtype=torch.long)
        s = sigma.detach().to(device=device, dtype=torch.float32).mean()
        return torch.bucketize(s, self._edges, right=True)

    def _band_cap(
        self, b: torch.Tensor, fallback: Optional[torch.Tensor]
    ) -> torch.Tensor:
        med = self._bands[b].nanmedian() * self.k
        if fallback is None:
            fallback = torch.full_like(med, float("inf"))
        return torch.where(self._band_n[b] >= self.band_warmup, med, fallback)

    def _record_band(self, b: torch.Tensor, norm: torch.Tensor) -> None:
        self._bands.index_put_((b, self._band_n[b] % self.window), norm)
        self._band_n.index_add_(0, b.reshape(1), torch.ones_like(b).reshape(1))

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
        self,
        grads: list[torch.Tensor],
        scale: float = 1.0,
        sigma: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Clip ``grads`` in place. ``scale`` converts their norm to the unit the
        cap is in (``accum`` for one micro-batch's share of an accumulated
        update); ``sigma`` (the unit's σ, averaged when it holds several) picks
        the band when σ bands are on. Returns the pre-clip norm (in cap units)
        and the cap used."""
        norm = torch.nn.utils.get_total_norm(grads).float() * scale
        cap = self.cap(norm.device)
        b = None
        if self.sigma_edges and sigma is not None:
            b = self._band(sigma, norm.device)
            cap = self._band_cap(b, cap)
        if cap is not None:
            coef = torch.clamp(cap / (norm + 1e-6), max=1.0)
            torch._foreach_mul_(grads, coef.to(grads[0].dtype))
        self._record(norm)
        if b is not None:
            self._record_band(b, norm)
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
        self,
        params: list[torch.nn.Parameter],
        last: bool,
        sigma: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        live = [p for p in params if p.grad is not None]
        norm = cap = None
        if live:
            norm, cap = self.clipper.clip_(
                [p.grad for p in live], scale=self.accum, sigma=sigma
            )
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


class GradSigmaLog:
    """``--grad_sigma_log``: one JSONL row per clipped unit, pairing its pre-clip
    norm and cap with the σ, loss and image key of the micro-batches it holds
    (one under ``clip_per_micro``, ``accum`` otherwise). Rows stay on the card
    until :meth:`flush`, which runs every ``flush_every`` rows and at run end.
    docs/anima/findings/lora_recipe.md."""

    def __init__(self, path, flush_every: int = 256):
        self.path = path
        self.flush_every = int(flush_every)
        self._pending: list[tuple] = []
        self._rows: list[tuple] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def add_micro(self, sigma, loss, keys) -> None:
        self._pending.append((sigma, loss, keys))

    def add_norm(self, step: int, norm, cap) -> None:
        self._rows.append((step, self._pending, norm, cap))
        self._pending = []
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        import json

        if not self._rows:
            return
        with self.path.open("a") as f:
            for step, micros, norm, cap in self._rows:
                row = {
                    "step": step,
                    "norm": float(norm),
                    "cap": None if cap is None else float(cap),
                    "sigma": [
                        x for s, _, _ in micros if s is not None for x in s.tolist()
                    ],
                    "loss": [float(lo) for _, lo, _ in micros],
                    "keys": [k for _, _, ks in micros for k in ks],
                }
                f.write(json.dumps(row) + "\n")
        self._rows.clear()
