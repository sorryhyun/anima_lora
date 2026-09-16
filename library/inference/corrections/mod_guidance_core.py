"""Pure-compute core for modulation guidance — single source of truth.

Two pieces of math are shared verbatim between the library's mod-guidance
(``library/inference/corrections/mod_guidance.py`` + the model method
``library/anima/models.py::Anima._pooled_text_delta``) and the ComfyUI node
(``ComfyUI-Spectrum-KSampler/mod_guidance.py``):

* ``project_pooled`` — the pooled-text → modulation-delta projection: a 2-layer
  MLP (Linear → SiLU → Linear), optionally σ-FiLM-modulated between the two
  linears (``h * (1 + scale) + shift``, where ``(scale, shift)`` is a linear of
  the normed time embedding). This is the exact op of ``_pooled_text_delta``.
* ``build_block_schedule`` — the per-block ``w(ℓ)`` profile from the
  start/end/taper knobs. Mirrored by the library's ``build_mod_schedule`` and
  the node's ``ModGuidanceState._build_schedule``.

torch/stdlib only — no ``comfy``, no anima-model imports. Each linear casts its
input to the weight dtype, so a caller running the (tiny) head in fp32 against
bf16 activations gets correct results; when dtypes already match the casts are
no-ops and the result is bit-identical to a plain ``nn.Sequential`` call.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def project_pooled(
    pooled: torch.Tensor,
    w_in: torch.Tensor,
    b_in: Optional[torch.Tensor],
    w_out: torch.Tensor,
    b_out: Optional[torch.Tensor],
    *,
    film_w: Optional[torch.Tensor] = None,
    film_b: Optional[torch.Tensor] = None,
    t_emb: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Project a pooled text vector through the 2-layer mod head.

    ``Linear(w_in, b_in) → [σ-FiLM] → SiLU → Linear(w_out, b_out)``.

    σ-FiLM is applied only when both ``film_w`` and ``t_emb`` are provided: the
    hidden between the linears is modulated by ``h * (1 + scale) + shift`` where
    ``(scale, shift) = chunk(Linear(t_emb; film_w, film_b))``. Without them this
    is the plain σ-flat head — bit-for-bit ``Linear → SiLU → Linear``.

    Each linear casts its input to the weight dtype (no-op when matched), so the
    head can run at a different precision than its inputs. ``out_dtype`` casts the
    final result (e.g. back to the block compute dtype); ``None`` leaves it in
    ``w_out``'s dtype.
    """
    h = F.linear(pooled.to(w_in.dtype), w_in, b_in)
    if t_emb is not None and film_w is not None:
        film = F.linear(t_emb.to(film_w.dtype), film_w, film_b)
        scale, shift = film.chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
    h = F.silu(h)
    out = F.linear(h.to(w_out.dtype), w_out, b_out)
    return out if out_dtype is None else out.to(out_dtype)


def build_block_schedule(
    num_blocks: int,
    w: float,
    start_layer: int,
    end_layer: int,
    taper: int = 0,
    taper_scale: float = 0.25,
) -> List[float]:
    """Build the per-block ``w(ℓ)`` modulation schedule.

    Blocks ``[start, end)`` receive ``w``; the last ``taper`` slots inside that
    window are scaled to ``w * taper_scale``. ``end_layer < 0`` means
    ``num_blocks``; ``start`` is clamped into ``[0, end]``. Returns a list of
    length ``num_blocks``.
    """
    end = num_blocks if end_layer < 0 else min(end_layer, num_blocks)
    start = max(0, min(start_layer, end))
    sched = [0.0] * num_blocks
    w = float(w)
    for i in range(start, end):
        sched[i] = w
    if taper > 0 and end > start:
        taper_start = max(start, end - taper)
        taper_w = w * float(taper_scale)
        for i in range(taper_start, end):
            sched[i] = taper_w
    return sched
