"""loss — the box-share FM loss with a logarithmic glyph curve.

The probe's ``weighted_fm_loss`` (``train/stage.py``) gives a scene item the
in-box share ``s = min(ρ · n_glyphs, cap)`` — linear in the glyph count, so a
2-glyph piece takes twice a single's share and a 4-glyph one hits the 0.75
cap. Here the share rises in ``log n``:

    s(n) = s1 + (s_cap − s1) · min(1, ln n / ln n_cap)

``s1`` (``box_share``) is a single glyph's share, ``s_cap`` (``box_share_cap``)
the ceiling, reached at ``n_cap`` glyphs (``box_share_glyphs``). The count is
glyphs, not tokens — a one-token piece is paid by its glyphs (user,
2026-09-23). Everything else is the probe's form: per item
``s · mean_in + (1 − s) · mean_out`` over the latent cells under ``rec['box']``,
averaged over the batch; an item without a box is plain MSE. The probe's
``_box_mask`` draws the mask so the two losses agree on the box.
"""

from __future__ import annotations

import math

import torch


def glyph_count(text: str) -> int:
    return max(1, sum(not c.isspace() for c in text))


def box_share_of(n_glyphs: int, s1: float, s_cap: float, n_cap: float) -> float:
    """The in-box share for ``n_glyphs`` glyphs: ``s1`` at one glyph, up to
    ``s_cap`` at ``n_cap`` glyphs, logarithmic in between."""
    if n_cap <= 1.0 or n_glyphs <= 1:
        return float(s1)
    u = min(1.0, math.log(n_glyphs) / math.log(n_cap))
    return float(s1 + (s_cap - s1) * u)


def box_share_fm_loss(pred, target, recs, s1: float, s_cap: float, n_cap: float):
    """MSE on the flow target with each item's in-box share from
    ``box_share_of``; ``s1 <= 0`` is plain MSE."""
    from train.stage import _box_mask

    se = (pred.float() - target.float()) ** 2
    if s1 <= 0.0:
        return se.mean()
    m = _box_mask(se.shape, recs, se.device)
    per_cell = se.mean(dim=1, keepdim=True)  # (B, 1, h, w)
    n_in = m.sum(dim=(1, 2, 3))
    n_out = (1.0 - m).sum(dim=(1, 2, 3))
    mean_in = (per_cell * m).sum(dim=(1, 2, 3)) / n_in.clamp(min=1)
    mean_out = (per_cell * (1.0 - m)).sum(dim=(1, 2, 3)) / n_out.clamp(min=1)
    s = torch.tensor(
        [box_share_of(glyph_count(r["text"]), s1, s_cap, n_cap) for r in recs],
        device=se.device,
        dtype=se.dtype,
    )
    s = torch.where(n_in > 0, s, torch.zeros_like(s))  # no box: plain mean
    s = torch.where(n_out > 0, s, torch.ones_like(s))
    return (s * mean_in + (1.0 - s) * mean_out).mean()
