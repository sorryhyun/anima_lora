"""Named GPU-side scalar accumulators with a single-sync flush.

The distill / training loops accumulate per-step metrics on-device so the hot
path never forces a CUDA sync, then read them all back once per log boundary.

:class:`ScalarAccumulator` owns that bookkeeping. Callers ``add()`` named
scalars (or ``add_at()`` into a fixed-width vector entry) on-device, then
``flush()`` reads everything back in ONE ``.tolist()`` and returns a
name-keyed dict.
"""

from __future__ import annotations

from typing import Union

import torch

Number = Union[float, int, torch.Tensor]


class ScalarAccumulator:
    """Named on-device accumulators flushed with a single CUDA sync.

    Each named entry is a device tensor — a 0-dim scalar (``add``) or a
    fixed-width 1-D vector (``add_at``). Accumulation stays on-device (no host
    sync); :meth:`flush` concatenates every entry, reads them back in one
    ``.tolist()``, and unpacks by the widths it recorded.

    Entries are insertion-ordered, so :meth:`flush` returns keys in first-add
    order. ``flush()`` returns raw summed values — apply any per-key reduction
    (divide by the interval, ``sqrt`` a squared-norm sum) on the returned
    Python floats, which costs no further sync.
    """

    def __init__(self, device: torch.device, *, dtype: torch.dtype = torch.float32):
        self._device = device
        self._dtype = dtype
        self._acc: dict[str, torch.Tensor] = {}

    def _ensure(self, name: str, width: int | None) -> torch.Tensor:
        t = self._acc.get(name)
        if t is None:
            shape: tuple[int, ...] = () if width is None else (width,)
            t = torch.zeros(shape, device=self._device, dtype=self._dtype)
            self._acc[name] = t
        elif (width is not None) != (t.dim() == 1) or (
            width is not None and t.numel() != width
        ):
            kind = f"vector(width={width})" if width is not None else "scalar"
            raise ValueError(
                f"accumulator {name!r} already exists with shape {tuple(t.shape)}; "
                f"cannot reuse it as a {kind}"
            )
        return t

    def add(self, name: str, value: Number) -> "ScalarAccumulator":
        """Accumulate ``value`` into the 0-dim scalar entry ``name``."""
        self._ensure(name, None).add_(
            value if isinstance(value, torch.Tensor) else float(value)
        )
        return self

    def add_at(
        self, name: str, index: int, value: Number, *, width: int
    ) -> "ScalarAccumulator":
        """Accumulate ``value`` into slot ``index`` of vector entry ``name``.

        ``width`` fixes the vector length on first touch; later calls must pass
        the same ``width``. Use for per-bucket metrics keyed by a Python index
        in the hot loop (e.g. per-stage loss sums) — the index stays on-device.
        """
        self._ensure(name, width)[index] += (
            value if isinstance(value, torch.Tensor) else float(value)
        )
        return self

    def flush(self) -> dict[str, float | list[float]]:
        """One CUDA sync: concat every entry, read once.

        Returns a name-keyed dict — scalar entries map to ``float``, vector
        entries to ``list[float]``. Does not reset; call :meth:`reset` (or
        :meth:`flush_reset`) to zero the accumulators for the next interval.
        """
        if not self._acc:
            return {}
        names = list(self._acc)
        packed = torch.cat([self._acc[n].reshape(-1) for n in names]).tolist()
        out: dict[str, float | list[float]] = {}
        off = 0
        for n in names:
            t = self._acc[n]
            ln = t.numel()
            chunk = packed[off : off + ln]
            out[n] = chunk[0] if t.dim() == 0 else chunk
            off += ln
        return out

    def reset(self) -> None:
        """Zero every accumulator in place (entries and widths are kept)."""
        for t in self._acc.values():
            t.zero_()

    def flush_reset(self) -> dict[str, float | list[float]]:
        """:meth:`flush` then :meth:`reset` — the usual log-boundary call."""
        out = self.flush()
        self.reset()
        return out

    def __contains__(self, name: str) -> bool:
        return name in self._acc
