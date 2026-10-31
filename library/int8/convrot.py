"""ConvRot (arXiv 2512.03673): each linear's weight is rotated per 256-column group by a
normalised regular Hadamard ``h``, then stored as int8 with one fp32 scale per output
row (amax / 127). ``h`` is symmetric and its own inverse, so the same group rotation
maps the input into the weight's basis and maps a rotated weight back."""

from __future__ import annotations

import torch

GROUP = 256
INT8_MAX = 127.0


def rotate(x: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """``x`` (…, K) times ``blockdiag(h)`` over its last dim; its own inverse."""
    from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight

    shape = x.shape
    return rotate_int8_convrot_weight(x.reshape(-1, shape[-1]), group).reshape(shape)


def row_scale(w_rot: torch.Tensor) -> torch.Tensor:
    return w_rot.abs().amax(1).clamp(min=1e-12) / INT8_MAX
