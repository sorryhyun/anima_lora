"""``join_cjk`` — the PP-OCRv6 column-joiner, kept here for the PP spotting records.

Retired from ``anime_tools.ocr._text`` on 2026-09-07 with PP-OCRv6 itself (the
package's OCR is the AnimeText block detector + the VL reader, whose boxes are
blocks already and must not be joined). The PP spotting records this research
tree was built from are still one box per column, so the joiner lives on
beside the builder, verbatim.
"""

from __future__ import annotations

from collections.abc import Sequence

from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.ocr._text import char_count, is_latin_only

VERTICAL_RATIO = 1.5
GAP_RATIO = 0.6
OVERLAP_RATIO = 0.4
JOIN_SEP = " "
SIZE_RATIO = 2.5


def _vertical(line: OcrLine) -> bool:
    return line.height >= VERTICAL_RATIO * max(line.width, 1)


def _adjacent(a: OcrLine, b: OcrLine) -> bool:
    """Whether two boxes are neighbouring columns (or rows) of one block.

    Same orientation, facing each other along their length, close along their
    thickness, and of a comparable thickness. Mixed orientations never join: a
    column beside a row is a sfx over dialogue, not its continuation.
    """
    vertical = _vertical(a)
    if vertical != _vertical(b):
        return False
    if vertical:
        span = (a.box[1], a.box[3], b.box[1], b.box[3])
        near = (a.box[0], a.box[2], b.box[0], b.box[2])
        thick = (a.width, b.width)
        length = (a.height, b.height)
    else:
        span = (a.box[0], a.box[2], b.box[0], b.box[2])
        near = (a.box[1], a.box[3], b.box[1], b.box[3])
        thick = (a.height, b.height)
        length = (a.width, b.width)

    overlap = min(span[1], span[3]) - max(span[0], span[2])
    if overlap < OVERLAP_RATIO * max(1, min(length)):
        return False
    gap = max(near[0], near[2]) - min(near[1], near[3])
    if gap > GAP_RATIO * max(1, max(thick)):
        return False
    lo, hi = min(thick), max(thick)
    return hi <= SIZE_RATIO * max(1, lo)


def _clusters(lines: Sequence[OcrLine]) -> list[list[int]]:
    """Indices grouped by adjacency — a plain union-find over every pair.

    At most :attr:`~anime_tools.ocr._onnx.OcrEngine.max_boxes` boxes reach here,
    so the quadratic pass costs nothing worth avoiding.
    """
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _adjacent(lines[i], lines[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _merge(lines: Sequence[OcrLine]) -> OcrLine:
    """One block's boxes as a single record, read in its own direction.

    Columns read right to left, rows top to bottom; the parts are joined with
    :data:`JOIN_SEP` (one space). Japanese sets no separator between the columns
    of a sentence, but a joined block is as often a list — a profile card's
    ``名前 / 身長 / 好きなもの`` rows — and glued (``椎名真昼ちゃん身長：156cm``)
    the boundary is lost for good, while a space inside a sentence costs a
    reader nothing. The box is the block's bound and the score the
    per-character mean of the parts, so a long confident column is not outvoted
    by the two glyphs beside it.
    """
    if len(lines) == 1:
        return lines[0]
    if _vertical(lines[0]):
        ordered = sorted(lines, key=lambda ln: -ln.box[0])
    else:
        ordered = sorted(lines, key=lambda ln: ln.box[1])
    weight = sum(char_count(ln.text) for ln in ordered)
    score = (
        sum(ln.score * char_count(ln.text) for ln in ordered) / weight
        if weight
        else min(ln.score for ln in ordered)
    )
    return OcrLine(
        seq=0,
        box=(
            min(ln.box[0] for ln in ordered),
            min(ln.box[1] for ln in ordered),
            max(ln.box[2] for ln in ordered),
            max(ln.box[3] for ln in ordered),
        ),
        score=score,
        text=JOIN_SEP.join(ln.text.strip() for ln in ordered),
    )


def join_cjk(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Merge each block of neighbouring CJK boxes into one line.

    Only lines carrying a non-ASCII character are candidates
    (:func:`is_latin_only`): English is set with spaces and wraps for width, so
    two English boxes stacked in a balloon are two lines and stay two. Everything
    else passes through untouched and in place — the caller re-sorts into reading
    order afterwards, since a merged block sits where neither part did.
    """
    joinable = [i for i, ln in enumerate(lines) if not is_latin_only(ln.text)]
    if len(joinable) < 2:
        return list(lines)

    out: list[tuple[int, OcrLine]] = [
        (i, ln) for i, ln in enumerate(lines) if is_latin_only(ln.text)
    ]
    candidates = [lines[i] for i in joinable]
    for group in _clusters(candidates):
        out.append((joinable[group[0]], _merge([candidates[k] for k in group])))
    return [ln for _, ln in sorted(out, key=lambda pair: pair[0])]
