"""Flow-matching forward-path input prep shared by the distillation loops.

Every distillation / probe loop that re-noises a clean latent and feeds it to
the DiT repeats the same handful of mechanical steps: build ``x_t = (1-t)·x +
t·ε``, sample the noise level ``t``/``σ``, insert the singleton temporal axis to
reach the DiT's 5D layout, and hand the model a zero padding mask. They lived
copy-pasted across ``project/finished/mod_guidance``, ``scripts/distill_turbo``
and a dozen ``bench/`` probes (the second copy,
``distill_turbo/primitives.py``, is what motivated promoting them here). This is
the model-agnostic half — no DiT call, just tensor math — so it sits in
``library/training/forward`` next to the other per-step composables.

5D-latent invariant (repo CLAUDE.md): the DiT operates on 5D
``(B, C, T=1, H, W)`` with the singleton at **dim 2**; everything around it is
4D ``(B, C, H, W)``. :func:`to_dit_5d` / :func:`from_dit_5d` are the only
sanctioned boundary moves — they target dim 2 explicitly and assert the shape,
so a stray ``squeeze()`` can't silently collapse the batch axis when ``B == 1``
(the exact class of bug that bit FreeText repeatedly).
"""

from __future__ import annotations

import torch


def renoise(x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """``x_t = (1 - t)·x + t·ε`` — the flow-matching forward path at level ``t``.

    ``t`` is per-batch (shape ``(B,)``); it is broadcast against ``x``'s trailing
    dims, so this works for 4D ``(B,C,H,W)`` or 5D ``(B,C,1,H,W)`` ``x`` alike.
    ``eps`` must match ``x``'s shape.
    """
    t_e = t.view(-1, *([1] * (x.dim() - 1)))
    return (1.0 - t_e) * x + t_e * eps


def sample_sigma(
    B: int,
    *,
    distribution: str = "sigmoid",
    sigmoid_scale: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample ``B`` noise levels on ``device``.

    ``"uniform"`` → ``U[0, 1)``; ``"sigmoid"`` → ``sigmoid(sigmoid_scale·N(0,1))``
    (the logit-normal level distribution Anima's samplers use). These are the
    only two strategies the distillation configs accept.
    """
    if distribution == "uniform":
        return torch.rand(B, device=device, dtype=dtype)
    return torch.sigmoid(sigmoid_scale * torch.randn(B, device=device, dtype=dtype))


def to_dit_5d(x: torch.Tensor) -> torch.Tensor:
    """4D ``(B, C, H, W)`` → 5D ``(B, C, 1, H, W)`` — insert the frame axis at dim 2."""
    if x.dim() != 4:
        raise ValueError(
            f"to_dit_5d expects a 4D (B,C,H,W) latent; got dim()={x.dim()}, "
            f"shape={tuple(x.shape)}."
        )
    return x.unsqueeze(2)


def from_dit_5d(x: torch.Tensor) -> torch.Tensor:
    """5D ``(B, C, 1, H, W)`` → 4D ``(B, C, H, W)`` — drop the dim-2 frame axis.

    Targets dim 2 explicitly (never a bare ``squeeze()``, which would also
    collapse the batch axis when ``B == 1``).
    """
    if x.dim() != 5 or x.shape[2] != 1:
        raise ValueError(
            "from_dit_5d expects a 5D (B,C,1,H,W) latent with the singleton at "
            f"dim 2; got shape={tuple(x.shape)}."
        )
    return x.squeeze(2)


def make_padding_mask(ref: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """All-zeros ``(B, 1, H, W)`` padding mask (zeros ⇒ "no padding").

    Sized and placed from ``ref`` (a 4D or 5D latent — only ``shape[0]`` and the
    trailing two dims are read). For a hot loop that re-issues the same few
    shapes, prefer :class:`PadCache` to recycle the allocation.
    """
    return torch.zeros(
        ref.shape[0], 1, ref.shape[-2], ref.shape[-1], dtype=dtype, device=ref.device
    )


class PadCache:
    """Per-spatial-shape zero padding mask, recycled across forwards.

    Constant-token bucketing keeps the spatial shape stable within a step (and
    constant in single-prompt mode), so we recycle the ``(B, 1, H, W)`` tensor
    keyed on ``(B, H, W)`` instead of re-allocating each forward.
    """

    def __init__(self, dtype: torch.dtype):
        self._dtype = dtype
        self._cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def get(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.shape[0], x.shape[-2], x.shape[-1])
        pad = self._cache.get(key)
        if pad is None or pad.dtype != self._dtype or pad.device != x.device:
            pad = torch.zeros(
                x.shape[0],
                1,
                x.shape[-2],
                x.shape[-1],
                dtype=self._dtype,
                device=x.device,
            )
            self._cache[key] = pad
        return pad
