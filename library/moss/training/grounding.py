"""RoPE ``(h, w)`` for the tokens of a caption that name an image position.

``clause_hw``: a position clause (``On the top right, <tags>``) names a cell of a
3×3 grid; its tags take that cell's centre, or with ``clause_centres`` the
centre of the subject the position stage found for it. The clause's header
(``On the top right``) stays at ``(0, 0)`` unless ``clause_header`` moves it. ``text_box_hw``: a boxed text line
(``"こいつ…" [y0,x0,y1,x1]``, ``abs_centralized`` pixels on the cache canvas) puts
the tokens inside its quotes at the box's centre. Both on RoPE's ``h`` / ``w`` axes
(``MOSSImageTransformer.text_hw``); every other token stays at ``(0, 0)`` and ``f``
is untouched (``docs/moss/proposal/cdpe_text_grounding.md``, after SpaG-DiT's CDPE).
"""

from __future__ import annotations

import re

import torch

# A boxed text line: the quoted string, then its box (``box_text_clauses``).
_BOXED = re.compile(r'"((?:[^"\\]|\\.)*)" \[(-?\d+),(-?\d+),(-?\d+),(-?\d+)\]')
PATCH = 16
"""Pixels per latent cell (the VAE's 8× and the 2×2 packing)."""

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


def clause_ranges(caption: str, header: bool = True) -> list[tuple[int, int, str]]:
    """``(start, end, position)`` character ranges of the caption's position
    clauses, header (``On the left``) to last tag, or first tag to last without
    ``header``; text clauses are left out."""
    from anime_tools.captions.position_clauses import parse_caption, tag_spans

    clauses = parse_caption(caption).clauses
    bounds: dict[int, list[int]] = {}
    for span in tag_spans(caption):
        if span.clause >= 0 and (header or span.kind != "header"):
            lo_hi = bounds.setdefault(span.clause, [span.start, span.end])
            lo_hi[1] = span.end
    return [
        (*bounds[i], c.position)
        for i, c in enumerate(clauses)
        if c.position and i in bounds
    ]


def clause_spans(caption: str) -> list[tuple[int, int, float, float]]:
    """``(start, end, O_h, O_w)``: :func:`clause_ranges` with each clause's cell
    (:func:`clause_cells`)."""
    ranges = clause_ranges(caption)
    cells = clause_cells([p for _, _, p in ranges])
    return [(a, b, oh, ow) for (a, b, _), (oh, ow) in zip(ranges, cells)]


Centres = dict[str, tuple[float, float]]
"""Position name → the subject's ``(h, w)`` in ``[-0.5, 0.5]²``, fractions of the
image's sides from its centre."""


def subject_centres(report_dir) -> dict[str, Centres]:
    """Per image stem, where each position clause's subject is: the centroid of
    its SAM3 mask in the position stage's analysis tree
    (``<report_dir>/analysis/**/<stem>.{json,png}``), its box centre when the label
    map holds no pixel of it. Images whose record names one position twice are
    left out."""
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    out: dict[str, Centres] = {}
    for record in sorted((Path(report_dir) / "analysis").rglob("*.json")):
        row = json.loads(record.read_text())
        instances = row.get("instances") or []
        names = [i.get("position") for i in instances]
        if row.get("labels") != "instances" or not all(names):
            continue
        if len(set(names)) != len(names):
            continue
        labels = np.asarray(Image.open(record.with_suffix(".png")))
        height, width = labels.shape
        centres: Centres = {}
        for index, inst in enumerate(instances):
            ys, xs = np.nonzero(labels == index + 1)
            if len(ys):
                cy, cx = ys.mean() + 0.5, xs.mean() + 0.5
            else:
                x0, y0, x1, y1 = inst["box"]
                cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            centres[inst["position"]] = (
                float(cy / height - 0.5),
                float(cx / width - 0.5),
            )
        out[record.stem] = centres
    return out


def box_spans(caption: str) -> list[tuple[int, int, float, float]]:
    """``(start, end, c_h, c_w)``: the character range inside the quotes of each
    boxed text line and its box centre in latent cells (``abs_centralized`` pixels
    / ``PATCH``, which is where RoPE's centred image grid puts that pixel)."""
    out = []
    for m in _BOXED.finditer(caption):
        if not m.group(1):
            continue
        y0, x0, y1, x1 = (int(v) for v in m.group(2, 3, 4, 5))
        out.append((m.start(1), m.end(1), (y0 + y1) / 2 / PATCH, (x0 + x1) / 2 / PATCH))
    return out


def text_hw(
    tok,
    caption: str,
    grid_hw: tuple[int, int],
    clause: float = 0.0,
    box: float = 0.0,
    centres: Centres | None = None,
    header: bool = False,
) -> torch.Tensor | None:
    """``(S, 2)`` fp32 ``(h, w)`` for the prompt's tokens: ``clause`` × each position
    clause's cell centre (a third of the grid side per cell), or × its subject's
    centre where ``centres`` names the clause's position; ``box`` × each boxed
    text line's box centre. A clause's ``On the left`` stays at ``(0, 0)`` and its
    tags move; ``header`` moves it with them. None when neither moves a token. ``tok`` is the
    tokenizer the cache templated with (prompt tail included)."""
    from library.moss.loader import prompt_text

    spans = []
    if clause:
        ranges = clause_ranges(caption, header)
        cells = clause_cells([p for _, _, p in ranges])
        for (a, b, position), (oh, ow) in zip(ranges, cells):
            # A cell centre is a third of the side from the image's centre.
            fh, fw = (centres or {}).get(position, (oh / 3, ow / 3))
            spans.append((a, b, clause * fh * grid_hw[0], clause * fw * grid_hw[1]))
    if box:
        spans += [(a, b, box * ch, box * cw) for a, b, ch, cw in box_spans(caption)]
    if not spans:
        return None
    formatted = prompt_text(tok, caption)
    base = formatted.find(caption)
    if base < 0:
        raise ValueError("the chat template rewrote the caption")
    offsets = tok(formatted, return_offsets_mapping=True).offset_mapping
    out = torch.zeros(len(offsets), 2)
    for start, end, h, w in spans:
        rows = [
            i for i, (a, b) in enumerate(offsets) if b > base + start and a < base + end
        ]
        out[rows] = torch.tensor([h, w], dtype=out.dtype)
    return out
