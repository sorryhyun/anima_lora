"""P1's read terms over a pool item (proposal § 3.3), every asked string scored.

``r_seq(targets, reads)``: each target string is matched to its own bubble
(one bubble per target, the assignment that maximises the sum), and scores
LCS(target, read) / max(len target, len read) against the best of that
bubble's reads: each box alone, and the bubble's boxes joined in column order
(right to left) or line order (top to bottom). A target with no bubble left
scores 0; ``r_seq`` is the mean over targets. No whole-page read, no join
across bubbles. With one target it is P0's per-box r_seq.

``g_p(asked, reads)``: the asked strings' letters among all letters drawn
(``eval.ruler.score.score_page``), with every string the caption asks for,
dialogue and SFX, as the asked bag.

Reads are the reader's records (``common.readers.Readers.read_images``):
``{"box": [x0, y0, x1, y1], "sfx": text, ...}``. Callers put the reseed
package on ``sys.path`` first.
"""

from __future__ import annotations

BUBBLE_GAP = 0.5  # boxes this share of their short side apart are one bubble


def lcs(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def bubbles(boxes: list) -> list:
    """Box reads grouped into bubbles: two boxes join when their gap on both
    axes is within ``BUBBLE_GAP`` of the smaller box's short side (union-find)."""
    n = len(boxes)
    up = list(range(n))

    def root(i):
        while up[i] != i:
            up[i] = up[up[i]]
            i = up[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i]["box"], boxes[j]["box"]
            m = BUBBLE_GAP * min(
                min(a[2] - a[0], a[3] - a[1]), min(b[2] - b[0], b[3] - b[1])
            )
            gx = max(a[0], b[0]) - min(a[2], b[2])
            gy = max(a[1], b[1]) - min(a[3], b[3])
            if gx <= m and gy <= m:
                up[root(i)] = root(j)
    out: dict = {}
    for i in range(n):
        out.setdefault(root(i), []).append(boxes[i])
    return list(out.values())


def bubble_reads(reads: list) -> list:
    """Per bubble, its candidate reads (normalised): each box, and the boxes
    joined in column and in line order."""
    from common.text import norm

    boxes = [r for r in reads if not r.get("whole") and r.get("sfx")]
    cx = lambda r: (r["box"][0] + r["box"][2]) / 2  # noqa: E731
    out = []
    for g in bubbles(boxes):
        cs = [norm(r["sfx"]) for r in g]
        if len(g) > 1:
            for key in (
                lambda r: (-cx(r), r["box"][1]),
                lambda r: (r["box"][1], r["box"][0]),
            ):
                cs.append("".join(norm(r["sfx"]) for r in sorted(g, key=key)))
        out.append([c for c in cs if c])
    return [cs for cs in out if cs]


def seq_score(t: str, cs: list) -> float:
    return max((lcs(t, c) / max(len(t), len(c)) for c in cs), default=0.0)


def match(targets: list, reads: list) -> list:
    """Per target: its score and the bubble it took (``None``: none left)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from common.text import norm

    ts = [norm(t) for t in targets]
    bs = bubble_reads(reads)
    out = [(0.0, None)] * len(ts)
    if not ts or not bs:
        return out
    m = np.array([[seq_score(t, cs) if t else 0.0 for cs in bs] for t in ts])
    for i, j in zip(*linear_sum_assignment(m, maximize=True)):
        out[i] = (float(m[i, j]), int(j))
    return out


def r_seq(targets: list, reads: list) -> float:
    if not targets:
        return 0.0
    return sum(s for s, _ in match(targets, reads)) / len(targets)


def g_p(asked: list, reads: list) -> float:
    from eval.ruler.score import score_page

    return score_page("".join(asked), reads, reads)["g_p"]
