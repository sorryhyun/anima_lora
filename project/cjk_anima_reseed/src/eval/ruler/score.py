"""One render's scores against its string and its EN ref: the text
(``score_text``), the glyphs and regions drawn (``score_page``), the page's
similarity to the EN ref (``AtSim``, ``outside_mask``, ``flat_white``)."""

from __future__ import annotations

import re
from pathlib import Path

BOOL = ("official", "exact", "contained", "le1", "le2", "dup")
# the page scores (user, 10-06: what is drawn, not only whether the string
# reads whole) — ``score_page``; a real is ``None`` where it has no meaning
# (kanji recall on a kana-only string) and is left out of a mean / a pair
PAGE = (
    "g_p",
    "g_r",
    "g_f1",
    "g_r_kanji",
    "g_r_kana",
    "drawn",
    "a_p",
    "text_area",
    "iou_en",
)
REAL = ("cer", *PAGE, "en_cls", "en_match", "en_tok_out", "fw_over_en")


def candidates(reads: list, reader: str) -> list:
    """One reader's reads: every box (the whole page too), and the boxes
    joined in column order (right to left) and in line order (top to bottom)
    — a bubble set in two columns is one string (user, 10-05)."""
    from common.text import norm

    boxes = [r for r in reads if not r.get("whole")]
    out = [norm(r.get(reader) or "") for r in reads]
    if len(boxes) > 1:
        cx = lambda r: (r["box"][0] + r["box"][2]) / 2  # noqa: E731
        for key in (
            lambda r: (-cx(r), r["box"][1]),
            lambda r: (r["box"][1], r["box"][0]),
        ):
            out.append(
                "".join(norm(r.get(reader) or "") for r in sorted(boxes, key=key))
            )
    return [c for c in out if c]


def score_text(text: str, reads: list) -> dict:
    from common.text import lev, norm

    t = norm(text)
    by = {x: candidates(reads, x) for x in ("sfx", "vl")}
    allc = by["sfx"] + by["vl"]
    best = min(allc, key=lambda c: (lev(c, t), len(c)), default="")
    d = lev(best, t) if best else len(t)
    doubled = lambda s: {s[i] for i in range(len(s) - 1) if s[i] == s[i + 1]}  # noqa: E731
    return {
        "best": best,
        "official": t in by["sfx"] and t in by["vl"],
        "exact": t in allc,
        "contained": any(t in c for c in allc),
        "le1": len(t) > 2 and d <= 1,
        "le2": len(t) > 2 and d <= 2,
        "cer": min(1.0, d / max(1, len(t))),
        "dup": bool(best) and (bool(doubled(best) - doubled(t)) or len(best) > len(t)),
    }


LETTER_RE = re.compile(r"[ぁ-ゟァ-ヺー一-鿿]")  # kana, ー, kanji (NFKC)
KANJI = re.compile(r"[一-鿿]")
READERS = ("sfx", "vl")
ON_SHARE = 0.5  # a box is on the target: this share of its letters are the target's


def letters(s: str | None) -> list:
    import unicodedata

    return [c for c in unicodedata.normalize("NFKC", s or "") if LETTER_RE.match(c)]


def box_mask(boxes: list, hw: tuple):
    import numpy as np

    H, W = hw
    m = np.zeros((H, W), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        m[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)] = True
    return m


def score_page(text: str, reads: list, en_reads: list) -> dict:
    """What the page draws against the string (user, 10-06), every text box
    read, each reader on its own and the two averaged:

    - glyphs (kana, ー, kanji; a bag — order and box free): ``g_p`` = the
      string's letters among all letters drawn (low: much text that is not
      the string), ``g_r`` = the string's letters drawn, ``g_f1``;
      ``g_r_kanji`` / ``g_r_kana`` the recall over its kanji / kana only;
      ``drawn`` = letters drawn;
    - regions: a box is on the string when ``ON_SHARE`` of its letters are
      the string's (and it holds two of them, one for a one-letter string);
      ``a_p`` = the on boxes' area over all text area, ``text_area`` = text
      area over the page, ``iou_en`` = the text area's IoU with the EN ref's
      (its layout — the EN page letters its other bubbles too, so not the
      string's place)."""
    from collections import Counter

    whole = next(r for r in reads if r.get("whole"))
    hw = (int(whole["box"][3]), int(whole["box"][2]))
    boxes = [r for r in reads if not r.get("whole")]
    G = Counter(letters(text))
    nG = sum(G.values())
    gk = Counter({c: n for c, n in G.items() if KANJI.match(c)})
    ga = G - gk
    text_m = box_mask([r["box"] for r in boxes], hw)
    en_m = box_mask([r["box"] for r in en_reads if not r.get("whole")], hw)
    t_area = int(text_m.sum())
    per = []
    for rd in READERS:
        D, on = Counter(), []
        for r in boxes:
            b = Counter(letters(r.get(rd)))
            D += b
            hit, nb = sum((b & G).values()), sum(b.values())
            if nb and hit >= max(min(2, nG), ON_SHARE * nb):
                on.append(r["box"])
        hit, nD = sum((D & G).values()), sum(D.values())
        pr = hit / nD if nD else 0.0
        rc = hit / nG if nG else 0.0
        on_area = int((box_mask(on, hw) & text_m).sum())
        per.append(
            {
                "g_p": pr,
                "g_r": rc,
                "g_f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                "g_r_kanji": sum((D & gk).values()) / sum(gk.values()) if gk else None,
                "g_r_kana": sum((D & ga).values()) / sum(ga.values()) if ga else None,
                "drawn": nD,
                "a_p": on_area / t_area if t_area else 0.0,
            }
        )
    out = {
        k: None if per[0][k] is None else sum(x[k] for x in per) / len(per)
        for k in per[0]
    }
    union = int((text_m | en_m).sum())
    return out | {
        "text_area": t_area / (hw[0] * hw[1]),
        "iou_en": int((text_m & en_m).sum()) / union if union else 0.0,
    }


def flat_white(file: str) -> float:
    """16² patches with std < 6 and mean > 225 (sigma_split's ``placement``)."""
    import numpy as np
    from PIL import Image

    im = np.asarray(Image.open(file).convert("L"), np.float32)
    H, W = im.shape
    p = im[: H // 16 * 16, : W // 16 * 16].reshape(H // 16, 16, W // 16, 16)
    return float(((p.std(axis=(1, 3)) < 6) & (p.mean(axis=(1, 3)) > 225)).mean())


def outside_mask(pe, n: int, hw, boxes):
    """``EnRef._outside_mask`` on the encoder's own patch grid: PE buckets
    the aspect (``pick_bucket``), so the grid is not ``√(n·W/H)`` off the
    square (800×736 → 1024 tokens)."""
    import math

    import torch

    from library.vision.encoder import pick_bucket

    H, W = hw
    gh, gw = pick_bucket(H, W, pe.bundle.bucket_spec)
    assert gh * gw == n, f"{n} tokens is no {gh}×{gw} grid for {W}x{H}"
    keep = torch.ones(gh, gw, dtype=torch.bool)
    for b in boxes:
        x0, y0 = int(b[0] / W * gw), int(b[1] / H * gh)
        x1, y1 = math.ceil(b[2] / W * gw), math.ceil(b[3] / H * gh)
        keep[max(0, y0) : min(gh, y1), max(0, x0) : min(gw, x1)] = False
    return keep.flatten()


class AtSim:
    """EN-ref similarity as ``anime_tools.grouping`` scores a near-twin pair
    (user, 10-05): the page stretched to PE-Spatial's 512² bucket, the CLS
    cosine (``en_cls``) and the dense grid match — mutual NN + ratio test over
    G×G pooled cells, the inlier fraction (``en_match``) — at the package's
    defaults."""

    def __init__(self, device="cuda"):
        from anime_tools.grouping import groups
        from anime_tools.grouping.embedder import pe_spatial_embedder

        self.emb = pe_spatial_embedder(device)
        self.g = groups.DEFAULT_GRID
        self.cell_min = groups.DEFAULT_CELL_MATCH_MIN
        self.ratio = groups.DEFAULT_RATIO
        self._memo: dict = {}

    def feats(self, path: Path):
        import torch
        from anime_tools.grouping.features import _load_512
        from anime_tools.grouping.matching import pool_cells_batch

        if path not in self._memo:
            cls, g16 = self.emb(_load_512(path)[None])
            cells = pool_cells_batch(
                torch.from_numpy(g16.astype("float32")).to(self.emb.device), self.g
            )
            self._memo[path] = (torch.from_numpy(cls[0]), cells)
        return self._memo[path]

    def __call__(self, f: Path, ref: Path) -> dict:
        from anime_tools.grouping.matching import match_fracs

        (ca, ga), (cb, gb) = self.feats(f), self.feats(ref)
        return {
            "en_cls": float(ca @ cb),
            "en_match": float(match_fracs(ga, gb, self.cell_min, self.ratio)[0]),
        }
