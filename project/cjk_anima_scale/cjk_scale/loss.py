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
averaged over the batch; an item without a box is plain MSE. The mask is the
probe's ``_box_mask`` rule (pixel box → latent cells, 8×) so the two losses
agree on a scene box.

``grid_box`` (``[train].grid_box``, off by default): a grid item — ``layout ==
"grid"``, ``rec['boxes']`` one per cell — takes the **union** of its cell
boxes as its box, with the share from the joined text's glyph count (so a
2×2 of 2-glyph pieces is 8 glyphs, the cap). Every row in the item shares
the one mask; a row learns from its own cell because the position clause
binds it there, not because the loss pairs them (k masks would be k
backwards). Off, a grid item is the plain canvas mean — a 24–32 px cell is
then ≈ 0.3–0.8 % of the loss, the 15–50 × per-draw gap of
``reports/conflict_joint_2026_09_25.md``'s price table. A flat 1×1
(``layout == "flat"``) stays plain either way.
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


def item_boxes(rec: dict, grid_box: bool) -> list:
    """The pixel boxes the loss weights for one record: a scene item's
    ``box``; a grid item's ``boxes`` (all cells) when ``grid_box``; nothing
    for a flat item or a grid item with ``grid_box`` off (plain MSE)."""
    layout = rec.get("layout") or ("scene" if "box" in rec else "")
    if layout == "scene" and rec.get("box"):
        return [rec["box"]]
    if layout == "grid" and grid_box and rec.get("boxes"):
        return list(rec["boxes"])
    return []


def box_mask(se_shape, recs, device, grid_box: bool = False):
    """``(B, 1, h, w)``: 1 under each of the record's ``item_boxes`` (latent
    cells, the probe's 8× rule), else 0."""
    B, _C, h, w = se_shape
    m = torch.zeros(B, 1, h, w, device=device)
    for b, r in enumerate(recs):
        for x0, y0, x1, y1 in item_boxes(r, grid_box):
            m[b, :, y0 // 8 : -(-y1 // 8), x0 // 8 : -(-x1 // 8)] = 1.0
    return m


def box_share_fm_loss(
    pred,
    target,
    recs,
    s1: float,
    s_cap: float,
    n_cap: float,
    grid_box: bool = False,
):
    """MSE on the flow target with each item's in-box share from
    ``box_share_of``; ``s1 <= 0`` is plain MSE. ``grid_box`` gives grid items
    their cells' union as the box (module docstring)."""
    se = (pred.float() - target.float()) ** 2
    if s1 <= 0.0:
        return se.mean()
    m = box_mask(se.shape, recs, se.device, grid_box)
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
