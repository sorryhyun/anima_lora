"""The pilot helpers this line's OCR code reads, copied from
``project/finished/cjk_aware_anima/datasets/``: ``deskew_crop`` / ``MangaOCR`` /
``OCR_MODEL`` from ``manga_text.py`` and ``norm`` / ``sim`` / ``is_runaway`` from
``build_ocr_records.py`` (bodies unchanged). ``manga109.pilot_manga_text()`` and
``manga109.pilot_records()`` both return this module.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

OCR_MODEL = "kha-white/manga-ocr-base"


def _on_page_window(rot, cx: float, cy: float, cw: int, ch: int, iw: int, ih: int):
    """``(top, bottom, left, right)`` of the largest window of a ``cw`` × ``ch``
    ``getRectSubPix`` crop of the ``rot``-warped page whose every pixel samples
    the page itself — not ``warpAffine``'s replicated border, not
    ``getRectSubPix``'s. The valid set is convex (a rectangle ∩ an affine image
    of one), so a window whose four edges are valid is valid inside, and
    peeling the worst edge one line at a time ends at such a window."""
    import cv2
    import numpy as np

    u = np.arange(cw, dtype=np.float64) + cx - (cw - 1) * 0.5
    v = np.arange(ch, dtype=np.float64) + cy - (ch - 1) * 0.5
    xr, yr = np.meshgrid(u, v)
    inv = cv2.invertAffineTransform(rot)
    xo = inv[0, 0] * xr + inv[0, 1] * yr + inv[0, 2]
    yo = inv[1, 0] * xr + inv[1, 1] * yr + inv[1, 2]
    ok = (
        (xr >= 0)
        & (xr <= iw - 1)
        & (yr >= 0)
        & (yr <= ih - 1)
        & (xo >= 0)
        & (xo <= iw - 1)
        & (yo >= 0)
        & (yo <= ih - 1)
    )
    t, b, left, r = 0, ch, 0, cw
    while t < b and left < r:
        bad = {
            "t": 1 - ok[t, left:r].mean(),
            "b": 1 - ok[b - 1, left:r].mean(),
            "l": 1 - ok[t:b, left].mean(),
            "r": 1 - ok[t:b, r - 1].mean(),
        }
        edge = max(bad, key=bad.get)
        if bad[edge] == 0:
            return t, b, left, r
        if edge == "t":
            t += 1
        elif edge == "b":
            b -= 1
        elif edge == "l":
            left += 1
        else:
            r -= 1
    return None


def deskew_crop(
    img, poly: list[float], pad_frac: float, min_side: int, *, trim: bool = True
):
    """Rotate a polygon upright and return the tight crop.

    Two rules that matter for manga-ocr:

    * **Orientation is preserved.** ``minAreaRect`` reports an angle in
      ``[0, 90)``; taking it verbatim would transpose a vertical text column
      into a horizontal strip. manga-ocr reads vertical Japanese natively, so we
      pick whichever of ``angle`` / ``angle - 90`` is the smaller rotation and
      only correct the tilt.
    * **Padding.** The polygons hug the glyphs, and the recogniser was trained
      on crops with a margin; a zero-pad crop clips strokes on the outer
      characters.
    * **The margin stops at the page** (``trim``, 2026-11-08). Where the padded
      box runs past the page edge, the lines that would sample outside it are
      cut off, as the shipped reader's ``pad_box`` clips its box to the image.
      ``trim=False`` is the old cut: those lines came back as ``BORDER_REPLICATE``
      streaks (13 % of COO SFX crops; a full-height SFX got 180 px of them).
      A crop that never leaves the page is byte-identical either way.
    """
    import cv2
    import numpy as np

    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if angle > 45:
        # OpenCV >= 4.5 reports angle in (0, 90] with (w, h) measured along the
        # rotated axes: an axis-aligned 30x120 box comes back as (120, 30) @ 90°.
        # Taking angle-90 without swapping the extents cropped a transposed
        # rectangle (found 2026-09-06 on the plan_ocr sincos gate; every
        # axis-aligned box was hit) — swap so the extents follow the rotation.
        angle -= 90
        w, h = h, w
    w = max(w, 1.0)
    h = max(h, 1.0)
    pad = pad_frac * max(w, h)
    w, h = w + 2 * pad, h + 2 * pad

    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    ih, iw = img.shape[:2]
    rotated = cv2.warpAffine(
        img,
        rot,
        (iw, ih),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    crop = cv2.getRectSubPix(rotated, (int(round(w)), int(round(h))), (cx, cy))
    if crop is None or crop.size == 0:
        return None, None
    if trim:
        win = _on_page_window(rot, cx, cy, crop.shape[1], crop.shape[0], iw, ih)
        if win is None:
            return None, None
        t, b, left, r = win
        crop = crop[t:b, left:r].copy()
    ch, cw = crop.shape[:2]
    if min(ch, cw) < min_side:
        return None, None
    orient = (
        "vertical" if ch > cw * 1.3 else ("horizontal" if cw > ch * 1.3 else "square")
    )
    return crop, orient


class MangaOCR:
    """manga-ocr (kha-white) with a vocab-only decoder.

    The published tokenizer is ``BertJapaneseTokenizer``, whose *encoder* needs
    MeCab/fugashi -- but generation only ever **decodes**, and decoding is plain
    WordPiece over ``vocab.txt``. Reading the vocab directly keeps two native
    dependencies out of the project for a pilot that never tokenises Japanese
    input.
    """

    def __init__(self, model_id: str = OCR_MODEL, device: str = "cuda") -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(device).eval()

        local = Path(model_id) / "vocab.txt"  # a fine-tuned dir (plan_ocr O2)
        vp = local if local.is_file() else Path(hf_hub_download(model_id, "vocab.txt"))
        vocab = vp.read_text(encoding="utf-8")
        self.vocab = vocab.splitlines()
        self.specials = {
            i for i, t in enumerate(self.vocab) if t.startswith("[") and t.endswith("]")
        }

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in self.specials or i >= len(self.vocab):
                continue
            tok = self.vocab[i]
            out.append(tok[2:] if tok.startswith("##") else tok)
        return "".join(out).replace(" ", "")

    def read(self, crops: list, batch_size: int = 32, max_new_tokens: int = 48):
        """Return ``[(text, mean_token_logprob), ...]``.

        The logprob is the whole point of running greedy with scores: it is the
        only per-region quality signal available without ground truth, and the
        ``hard_neg`` control tells us whether it actually separates.
        """
        import numpy as np
        from PIL import Image

        torch = self.torch
        results: list[tuple[str, float]] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            images = [Image.fromarray(c[:, :, ::-1]).convert("RGB") for c in batch]
            pixel = self.processor(images, return_tensors="pt").pixel_values.to(
                self.device
            )
            with torch.no_grad():
                out = self.model.generate(
                    pixel,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                scores = self.model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True
                )
            seqs = out.sequences[:, -scores.shape[-1] :]
            for row, sc, seq in zip(out.sequences, scores, seqs):
                text = self.decode(row.tolist())
                keep = [
                    float(s)
                    for s, t in zip(sc.tolist(), seq.tolist())
                    if int(t) not in self.specials and not math.isinf(s)
                ]
                results.append((text, float(np.mean(keep)) if keep else float("-inf")))
        return results


_STRIP_RE = re.compile(r"[\s。、．，,.・…‥「」『』!！?？~～〜❤♥♡♪☆★()（）\-ー—–|｜]")


def norm(s: str) -> str:
    """The A/B's comparison key: NFKC, then punctuation / symbols / spaces gone."""
    return _STRIP_RE.sub("", unicodedata.normalize("NFKC", s))


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    return SequenceMatcher(None, a, b).ratio() if a or b else 1.0


def is_runaway(text: str, *, ngram_repeats: int = 3, run: int = 8) -> bool:
    """The VL failure class PP does not have: ``ぉぉぉ…``×100, ``ふくっ``×100.

    A glyph run of ``run`` or more, or any 3-gram occurring ``ngram_repeats``
    or more times. ``ぱんぱん`` (one repeat) and ``おおおん`` pass.
    """
    t = "".join(text.split())
    if re.search(r"(.)\1{%d,}" % (run - 1), t):
        return True
    if len(t) >= 9:
        grams = defaultdict(int)
        for i in range(len(t) - 2):
            grams[t[i : i + 3]] += 1
        if max(grams.values()) >= ngram_repeats:
            return True
    return False
