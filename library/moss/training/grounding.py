"""``clause_hw``: RoPE ``(h, w)`` for the tokens of a caption's position clauses.

A position clause (``On the top right, <tags>``) names a cell of a 3×3 grid. Its
tokens take that cell's centre on RoPE's ``h`` / ``w`` axes during training
(``MOSSImageTransformer.text_hw``); every other token stays at ``(0, 0)`` and ``f``
is untouched. Training only: generation runs without it
(``docs/moss/proposal/cdpe_text_grounding.md``, after SpaG-DiT's CDPE).
"""

from __future__ import annotations

import re

import torch

_ORDINALS = ("second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
_FROM_LEFT = re.compile(r"(\w+) from left")


def _column(position: str) -> int | None:
    """1-based column of a ``leftmost / second from left / … / rightmost`` name
    (``rightmost``: -1); None for the 3×3 words."""
    words = position.split()
    if "leftmost" in words:
        return 1
    if "rightmost" in words:
        return -1
    if m := _FROM_LEFT.search(position):
        word = m.group(1)
        if word in _ORDINALS:
            return _ORDINALS.index(word) + 2
        if word[:-2].isdigit():
            return int(word[:-2])
    return None


def _row(position: str) -> int:
    words = position.split()
    return -1 if "top" in words else 1 if "bottom" in words else 0


def clause_cells(positions: list[str]) -> list[tuple[float, float]]:
    """``(O_h, O_w)`` in ``[-1, 1]²`` for each clause position of one caption
    (``left`` → ``(0, -1)``, ``top right`` → ``(-1, 1)``). A row named by columns
    spreads them evenly from -1 (``leftmost``) to 1 (``rightmost``)."""
    named: dict[int, list[int]] = {}
    for p in positions:
        col = _column(p)
        if col is not None:
            named.setdefault(_row(p), []).append(col)
    # A column short of ``rightmost`` implies one more to its right.
    counts = {row: max(len(cols), max(cols) + 1, 2) for row, cols in named.items()}
    out = []
    for p in positions:
        row, col, words = _row(p), _column(p), p.split()
        if col is not None:
            n = counts[row]
            k = n if col < 0 else col
            out.append((float(row), -1.0 + 2.0 * (k - 1) / (n - 1)))
        else:
            out.append(
                (float(row), -1.0 if "left" in words else float("right" in words))
            )
    return out


def clause_spans(caption: str) -> list[tuple[int, int, float, float]]:
    """``(start, end, O_h, O_w)`` character ranges of the caption's position
    clauses, header to last tag; text clauses are left out."""
    from anime_tools.captions.position_clauses import parse_caption, tag_spans

    clauses = parse_caption(caption).clauses
    bounds: dict[int, list[int]] = {}
    for span in tag_spans(caption):
        if span.clause >= 0:
            lo_hi = bounds.setdefault(span.clause, [span.start, span.end])
            lo_hi[1] = span.end
    spatial = [i for i, c in enumerate(clauses) if c.position and i in bounds]
    cells = clause_cells([clauses[i].position for i in spatial])
    return [(*bounds[i], oh, ow) for i, (oh, ow) in zip(spatial, cells)]


def clause_text_hw(
    tok, caption: str, grid_hw: tuple[int, int], scale: float
) -> torch.Tensor | None:
    """``(S, 2)`` fp32 ``(h, w)`` for the prompt's tokens: ``scale`` × the clause's
    cell centre, a third of the grid side per cell. None without a position clause.
    ``tok`` is the tokenizer the cache templated with (prompt tail included)."""
    from library.moss.loader import prompt_text

    spans = clause_spans(caption)
    if not spans:
        return None
    formatted = prompt_text(tok, caption)
    base = formatted.find(caption)
    if base < 0:
        raise ValueError("the chat template rewrote the caption")
    offsets = tok(formatted, return_offsets_mapping=True).offset_mapping
    out = torch.zeros(len(offsets), 2)
    step = (scale * grid_hw[0] / 3, scale * grid_hw[1] / 3)
    for start, end, oh, ow in spans:
        rows = [
            i for i, (a, b) in enumerate(offsets) if b > base + start and a < base + end
        ]
        out[rows] = torch.tensor([oh * step[0], ow * step[1]])
    return out
