#!/usr/bin/env python3
"""plan_render S0 — text boxes → reads → bubbles → panels on the paired-edition corpus.

The corpus is never pathed here (plan principle 9): pass ``--corpus <root>`` or
set ``ANIMA_RENDER_CORPUS``. Everything this script writes lands under
``output/render/<name>/`` (gitignored), keyed by the page's path *relative to
the corpus root*.

Stages, each idempotent and resumable per page::

    # GPU (daemon): detect on a per-work page sample, then read every box
    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima_dit/render/corpus_boxes.py det --corpus <root> --editions en ja --per_work 4"
    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima_dit/render/corpus_boxes.py read --corpus <root> --editions en ja"
    # CPU: bubbles (box clusters + script guard), panels (XY-cut + bubble assignment), the hand-check sheet
    python project/cjk_aware_anima_dit/render/corpus_boxes.py bubbles --corpus <root> --editions en ja
    python project/cjk_aware_anima_dit/render/corpus_boxes.py panels --editions en ja
    python project/cjk_aware_anima_dit/render/corpus_boxes.py sheet --corpus <root> --editions ja --n 40

* ``det`` — ``anime_tools.ocr.animetext.AnimeTextDetector`` (yolo12l @ 640,
  conf 0.25, ``inner`` nesting — the shipped defaults) over every non-promo page
  of each edition, or ``--per_work N`` seeded pages per work (the same page
  numbers across editions when their page counts agree), plus the Manga109
  ``frame`` detector for panels → ``boxes_<ed>.jsonl``. NB the text detector emits *block* boxes for horizontal text (a whole balloon,
  2–3 lines), so a box is usually already a bubble.
* ``read`` — every box, padded 12 %, read by ``stock`` (``ocr/pseudo_label.py``
  ``Vl16Sweeper``; PaddleOCR-VL-1.6, spacing-preserving) → ``reads_<ed>.jsonl``.
* ``bubbles`` — union-find over the page's boxes: two boxes share a bubble when
  they stack (rows: horizontal overlap ≥ 0.3 and vertical gap ≤ 0.8 × the
  thinner one's height; columns: the transpose). Guards, in order (the first
  hit is the ``reason``): every line non-empty (``empty_line``);
  script-consistent with the edition — EN ASCII with a letter, JA kana/kanji,
  KO hangul-dominant (``script``); ≥ ``--min_chars`` (``short``); a JA bubble
  whose every line is SFX by ``anime_tools.captions.ocr_sfx.line_kind``
  (``sfx``, speech only in v0); the S0b guards — a read that is mostly one
  unit repeated over ≥ 20 characters (``repeat``: ``ウウウ…`` / ``O O O O``
  reader loops, ``びゅる`` ×20 bursts; a typeset ``AH! AH! AH!`` stays), a JA kana-only read with ≤ ``--min_kana`` content kana
  (``short_kana``: ``あっ`` / ``くー〜〜`` moans, the hand-lettered majority),
  no balloon around the box (``no_balloon``: the band just outside each member
  box — the middle half of each side, corners excluded — is neither mostly
  white nor mostly flat, *and* the box interior is not mostly white either
  (text that fills its balloon to the outline); hand-lettered SFX sit on
  artwork; needs ``--corpus``), and finally ``overlap`` — a bubble whose
  member box overlaps another bubble's (IoU > 0.1 or one inside the other)
  is rejected whatever the neighbour's verdict, because the padded read crop
  saw both (merged-neighbour reads). → ``bubbles_<ed>.jsonl`` with ``ok`` +
  ``reason`` + ``kind`` + ``ring`` on every bubble. Multi-line reads are
  flattened (EN with a space, JA/KO joined). S0b changes no box, so S0's reads
  (one per box) stay valid — no GPU pass.
* ``panels`` — the ``frame`` boxes the det stage also wrote (``deepghs/
  manga109_yolo``, the Manga109 YOLO11-l ONNX export at its F1 threshold; a page
  with no frame is one whole-page panel — white-gutter XY-cut was tried first
  and does not survive this corpus's full-bleed colour pages). Frame fixes:
  a frame under ``--min_area`` of the page is dropped; a frame that contains
  ≥ 2 other frames (a whole-page box over real panels) is dropped; a bubble
  goes to the *smallest* frame holding its centre (nested frames no longer
  double-assign); the panel then grows to contain every accepted bubble it
  holds (never a holed box half outside the crop). Bubbles are ordered in
  manga reading order — rows by a band one median bubble tall, right→left
  inside a row (EN scanlations keep it) → ``samples_<ed>.jsonl``: one row per
  panel with ≥ 1 accepted bubble (``lines`` in order, ``intact`` = the
  rejected bubbles inside it, copied through un-holed, ``slant`` = the panel
  overlaps a sibling frame by > 10 % of its area — the slanted-gutter
  cut-through proxy, counted, not fixed).
* ``sheet`` — ``sheet_<ed>.png`` + ``sheet_<ed>.tsv``: panel crops with the
  accepted bubbles outlined red (index = caption order) and the intact
  rejects outlined blue with their reason, reads wrapped underneath — the
  hand pass before anything is cut.

The cut stage (panel crops → ``post_image_dataset/render/<ed>/``) is a separate
script; nothing here writes outside ``output/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parent
REPO = LINE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LINE / "ocr"))  # pseudo_label (the stock reader wrapper)

EDITIONS = ("en", "ja", "ko")
OUT_ROOT = REPO / "output" / "render"
PAD = 0.12
"""Crop padding for the reader, as a fraction of the box's longer side (the
K1 screen's value)."""
VERTICAL_RATIO = 1.5
"""``anime_tools.ocr._text.VERTICAL_RATIO``: taller than this × width = a column."""
STACK_OVERLAP = 0.3
STACK_GAP = 0.8
FRAMES_DIR = REPO / "models" / "manga109_yolo"
"""``deepghs/manga109_yolo`` ``v2023.12.07_l_yv11`` (ONNX; classes body / face /
frame / text; F1 0.92 on Manga109-s). Fetched ad hoc for this probe, not a
catalog row."""
FRAME_CLASS = 2
FRAME_CONF = 0.373
"""The card's F1 threshold (``threshold.json``)."""

HANGUL = re.compile(r"[가-힣]")
KANA_KANJI = re.compile(r"[぀-ヿ一-鿿]")
KANA = re.compile(r"[぀-ヿ]")
KANJI = re.compile(r"[一-鿿]")
FILLER = re.compile(r"[\s\W_ーｰ〜～っッ]", re.UNICODE)
"""Stripped before a read is measured for repetition / kana content: spaces,
punctuation and symbols (・…！？♡), long-vowel marks and the small tsu — the
characters a moan is padded with."""
RING_FRAC, RING_MIN, RING_MAX = 0.15, 5, 24
"""Balloon probe: the ring outside a text box is this fraction of the box's
short side wide, clamped to [RING_MIN, RING_MAX] px."""
RING_WHITE, RING_FLAT_TOL, RING_OK = 200, 15, 0.6
"""A ring passes when ≥ RING_OK of its pixels are ≥ RING_WHITE (a white
balloon) or within ± RING_FLAT_TOL of its median (a tinted / screentoned
balloon on a colour page — cells 2 and 13 of the S0 JA sheet)."""
FILL_OK = 0.7
"""Second chance for a box whose text fills its balloon to the outline (tall
narration bubbles, boxed narration): the box interior itself is ≥ FILL_OK
white. Balloon text sits on white at 0.75–0.9 (typeset EN a little lower);
hand-lettered SFX on art at 0.4–0.6."""


# --------------------------------------------------------------------------- corpus


def corpus_root(arg: str | None) -> Path:
    root = arg or os.environ.get("ANIMA_RENDER_CORPUS")
    if not root:
        sys.exit("pass --corpus <root> or set ANIMA_RENDER_CORPUS (never a default)")
    root = Path(root).expanduser().resolve()
    if not (root / "retrieved").is_dir():
        sys.exit(f"{root}: no retrieved/ tree")
    return root


def iter_pages(
    root: Path,
    ed: str,
    artists: set[str] | None,
    per_work: int = 0,
    seed: int = 0,
):
    """``(rel, path)`` for every non-promo, present page of edition ``ed``,
    artist dir → work → page order. ``rel`` is POSIX, relative to ``root``.

    ``per_work`` > 0 keeps a seeded sample of that many pages per work; the rng
    is seeded by ``(seed, work id)`` so two editions with the same page count
    draw the same page numbers."""
    for artist_dir in sorted((root / "retrieved").iterdir()):
        if not artist_dir.is_dir():
            continue
        if artists and artist_dir.name not in artists:
            continue
        for work_dir in sorted(artist_dir.iterdir()):
            side = work_dir / ed
            files = side / "files.json"
            if not files.is_file():
                continue
            try:
                recs = json.loads(files.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            pages = []
            for rec in sorted(recs, key=lambda r: r.get("page", 0)):
                name = rec.get("file")
                if not name or rec.get("promo") or rec.get("status", "ok") != "ok":
                    continue
                p = side / name
                if p.is_file() and p.stat().st_size:
                    pages.append((p.relative_to(root).as_posix(), p))
            if per_work and len(pages) > per_work:
                rng = random.Random(f"{seed}:{work_dir.name}")
                idx = sorted(rng.sample(range(len(pages)), per_work))
                pages = [pages[i] for i in idx]
            yield from pages


def read_bgr(path: Path):
    import cv2
    import numpy as np

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:  # avif and friends: PIL (with its plugins) as the fallback
        from PIL import Image

        try:
            with Image.open(path) as im:
                bgr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            return None
    return bgr


# --------------------------------------------------------------------------- jsonl


def out_dir(name: str) -> Path:
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_jsonl(path: Path, key: str | None = "rel"):
    if not path.is_file():
        return {} if key else []
    rows = {} if key else []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if key:
                rows[r[key]] = r
            else:
                rows.append(r)
    return rows


def append_jsonl(fh, row: dict) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def _pages(a, root: Path, ed: str):
    return list(iter_pages(root, ed, a.artists, per_work=a.per_work, seed=a.seed))


# --------------------------------------------------------------------------- det


class FrameDetector:
    """Manga panels (``frame``) off the Manga109 YOLO11-l ONNX export, run
    through onnxruntime on the AnimeText letterbox; the head's 4 box rows plus
    the ``frame`` score row go through ``animetext.decode`` unchanged."""

    def __init__(self, device: str, conf: float = FRAME_CONF, nms: float = 0.5):
        import onnxruntime as ort

        onnx = FRAMES_DIR / "model.onnx"
        if not onnx.is_file():
            raise FileNotFoundError(
                f"{onnx}: fetch deepghs/manga109_yolo v2023.12.07_l_yv11/model.onnx"
            )
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", *providers]
        self.sess = ort.InferenceSession(str(onnx), providers=providers)
        self.conf, self.nms = conf, nms

    def detect_scored(self, bgr) -> list:
        import numpy as np

        from anime_tools.ocr import animetext

        canvas, r = animetext.letterbox(bgr, 640)
        out = self.sess.run(None, {"images": animetext.to_tensor(canvas)})[0][0]
        head = np.concatenate([out[:4], out[4 + FRAME_CLASS : 5 + FRAME_CLASS]], 0)
        h, w = bgr.shape[:2]
        return animetext.decode(head, r, w, h, conf=self.conf, nms=self.nms)


def run_det(a) -> None:
    from anime_tools.ocr import animetext

    root = corpus_root(a.corpus)
    det = animetext.AnimeTextDetector.load(
        device=a.device, conf=a.det_conf, nest="inner"
    )
    frames = FrameDetector(a.device)
    for ed in a.editions:
        path = out_dir(a.name) / f"boxes_{ed}.jsonl"
        done = load_jsonl(path) if not a.overwrite else {}
        pages = _pages(a, root, ed)
        todo = [(rel, p) for rel, p in pages if rel not in done]
        if a.limit:
            todo = todo[: a.limit]
        print(
            f"[{ed}] {len(pages)} pages, {len(done)} done, {len(todo)} to detect "
            f"on {det.device}",
            flush=True,
        )
        t0, n_boxes, n_frames, n_bad = time.time(), 0, 0, 0
        mode = "w" if a.overwrite else "a"
        with path.open(mode, encoding="utf-8") as fh:
            for i, (rel, p) in enumerate(todo, 1):
                bgr = read_bgr(p)
                if bgr is None:
                    n_bad += 1
                    continue
                scored = det.detect_scored(bgr)
                fr = frames.detect_scored(bgr)
                n_boxes += len(scored)
                n_frames += len(fr)
                append_jsonl(
                    fh,
                    {
                        "rel": rel,
                        "size": [int(bgr.shape[1]), int(bgr.shape[0])],
                        "boxes": [list(map(int, b[:4])) for b in scored],
                        "scores": [round(float(b[4]), 4) for b in scored],
                        "frames": [list(map(int, b[:4])) for b in fr],
                        "frame_scores": [round(float(b[4]), 4) for b in fr],
                    },
                )
                if i % 100 == 0:
                    rate = i / (time.time() - t0)
                    print(
                        f"  [{ed}] {i}/{len(todo)} pages, {n_boxes} boxes, "
                        f"{rate:.1f} pages/s",
                        flush=True,
                    )
        print(
            f"[{ed}] det done: {len(todo)} pages, {n_boxes} boxes, {n_frames} frames, "
            f"{n_bad} unreadable, "
            f"{time.time() - t0:.0f}s → {path}",
            flush=True,
        )


# --------------------------------------------------------------------------- read


def pad_crop(bgr, box, pad: float = PAD, min_side: int = 8):
    ih, iw = bgr.shape[:2]
    x0, y0, x1, y1 = box
    p = pad * max(x1 - x0, y1 - y0)
    X0, Y0 = max(0, int(round(x0 - p))), max(0, int(round(y0 - p)))
    X1, Y1 = min(iw, int(round(x1 + p))), min(ih, int(round(y1 + p)))
    if min(X1 - X0, Y1 - Y0) < min_side:
        return None
    return bgr[Y0:Y1, X0:X1]


def run_read(a) -> None:
    import pseudo_label as pl

    root = corpus_root(a.corpus)
    reader = pl.SWEEPERS["stock"](None, a.device)
    for ed in a.editions:
        boxes = load_jsonl(out_dir(a.name) / f"boxes_{ed}.jsonl")
        path = out_dir(a.name) / f"reads_{ed}.jsonl"
        done = load_jsonl(path) if not a.overwrite else {}
        todo = [r for rel, r in boxes.items() if rel not in done and r["boxes"]]
        if a.limit:
            todo = todo[: a.limit]
        print(
            f"[{ed}] {len(boxes)} pages boxed, {len(done)} read, {len(todo)} to read",
            flush=True,
        )
        t0, n_lines = time.time(), 0
        mode = "w" if a.overwrite else "a"
        with path.open(mode, encoding="utf-8") as fh:
            batch_pages: list[tuple[dict, list[int], list]] = []
            n_crops = 0

            def flush() -> None:
                nonlocal batch_pages, n_crops, n_lines
                crops = [c for _, _, cs in batch_pages for c in cs]
                texts: list[str] = []
                for s in range(0, len(crops), a.bs):
                    texts += [t for t, _, _ in reader.read(crops[s : s + a.bs])]
                k = 0
                for row, idx, cs in batch_pages:
                    out = [""] * len(row["boxes"])
                    for j in idx:
                        out[j] = texts[k]
                        k += 1
                    n_lines += len(idx)
                    append_jsonl(fh, {"rel": row["rel"], "texts": out})
                batch_pages, n_crops = [], 0

            for i, row in enumerate(todo, 1):
                bgr = read_bgr(root / row["rel"])
                if bgr is None:
                    continue
                idx, cs = [], []
                for j, b in enumerate(row["boxes"]):
                    c = pad_crop(bgr, b)
                    if c is not None:
                        idx.append(j)
                        cs.append(c)
                batch_pages.append((row, idx, cs))
                n_crops += len(cs)
                if n_crops >= a.bs:
                    flush()
                if i % 50 == 0:
                    rate = n_lines / max(time.time() - t0, 1e-6)
                    print(
                        f"  [{ed}] {i}/{len(todo)} pages, {n_lines} lines, "
                        f"{rate:.1f} lines/s",
                        flush=True,
                    )
            if batch_pages:
                flush()
        print(
            f"[{ed}] read done: {len(todo)} pages, {n_lines} lines, "
            f"{time.time() - t0:.0f}s → {path}",
            flush=True,
        )


# --------------------------------------------------------------------------- bubbles


def _is_column(box) -> bool:
    x0, y0, x1, y1 = box
    return (y1 - y0) > VERTICAL_RATIO * max(x1 - x0, 1)


def _overlap(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    return inter / max(1, min(a1 - a0, b1 - b0))


def _stacked(a, b) -> bool:
    """Whether boxes ``a`` and ``b`` sit in one bubble: consecutive rows or
    consecutive columns."""
    ca, cb = _is_column(a), _is_column(b)
    if ca != cb:
        return False
    if not ca:  # rows: share horizontal extent, small vertical gap
        if _overlap(a[0], a[2], b[0], b[2]) < STACK_OVERLAP:
            return False
        gap = max(a[1], b[1]) - min(a[3], b[3])
        return gap <= STACK_GAP * min(a[3] - a[1], b[3] - b[1])
    if _overlap(a[1], a[3], b[1], b[3]) < STACK_OVERLAP:
        return False
    gap = max(a[0], b[0]) - min(a[2], b[2])
    return gap <= STACK_GAP * min(a[2] - a[0], b[2] - b[0])


def group_bubbles(boxes: list) -> list[list[int]]:
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _stacked(boxes[i], boxes[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        if _is_column(boxes[idx[0]]):
            idx.sort(key=lambda k: -boxes[k][2])  # right → left
        else:
            idx.sort(key=lambda k: boxes[k][1])  # top → bottom
        out.append(idx)
    # page order: top → bottom by the bubble's top edge, ties right → left
    out.sort(key=lambda g: (min(boxes[k][1] for k in g), -max(boxes[k][2] for k in g)))
    return out


def script_ok(ed: str, text: str) -> bool:
    if ed == "ko":
        import pseudo_label as pl

        return pl.hangul_dominant(text)
    if ed == "en":
        # Latin letters required; non-ASCII allowed only as punctuation / symbols
        # (♡, —, fullwidth ！？) — a kana or kanji anywhere rejects the bubble
        # (EN scanlations leave ~a third of SFX / moans in Japanese).
        if not any(c.isascii() and c.isalpha() for c in text):
            return False
        return all(c.isascii() or unicodedata.category(c)[0] in "PSZ" for c in text)
    if ed == "ja":
        return bool(KANA_KANJI.search(text))
    return True


def _flatten(ed: str, text: str) -> str:
    """A multi-line read as one caption line: EN joins with a space, JA / KO
    with nothing (JA does not space words; the KO teacher already spaces)."""
    parts = [t.strip() for t in text.splitlines() if t.strip()]
    return (" " if ed == "en" else "").join(parts)


def content(text: str) -> str:
    return FILLER.sub("", text)


def repetitive(
    text: str, min_span: int = 20, min_frac: float = 0.6, max_unique: float = 0.34
) -> bool:
    """A reader loop, not a typeset moan: one 1–3-character unit repeated back
    to back over ≥ ``min_span`` stripped characters that are ≥ ``min_frac`` of
    the read (``ウウウ…`` × 60, ``びゅる`` × 20, ``O O O O`` × 95); a read
    without Latin letters that long additionally fails on fewer than
    ``max_unique`` distinct characters per character. ``AH! AH! AH!`` inside
    a balloon (16 chars) is speech and stays."""
    s = content(text)
    if len(s) < min_span:
        return False
    span = max((len(m.group(0)) for m in re.finditer(r"(.{1,3}?)\1{2,}", s)), default=0)
    if span >= min_span and span >= min_frac * len(s):
        return True
    latin = any(c.isascii() and c.isalpha() for c in s)
    return not latin and len(set(s)) / len(s) < max_unique


def kana_only(text: str) -> bool:
    return (
        bool(KANA.search(text))
        and not KANJI.search(text)
        and not any(c.isascii() and c.isalnum() for c in text)
    )


def ring_score(gray, box) -> tuple[float, float]:
    """``(white, flat)`` fractions of the band just outside ``box`` — the
    middle half of each of the four sides, corners excluded (a typeset block
    hugs its balloon on the sides, so the corners of a full ring fall on the
    outline and the art beyond it; the middle of each side is still inside
    the balloon's curve)."""
    import numpy as np

    H, W = gray.shape
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    m = int(min(RING_MAX, max(RING_MIN, RING_FRAC * min(w, h))))
    qx, qy = w // 4, h // 4
    strips = [
        gray[max(0, y0 - m) : y0, x0 + qx : x1 - qx],  # top
        gray[y1 : min(H, y1 + m), x0 + qx : x1 - qx],  # bottom
        gray[y0 + qy : y1 - qy, max(0, x0 - m) : x0],  # left
        gray[y0 + qy : y1 - qy, x1 : min(W, x1 + m)],  # right
    ]
    r = np.concatenate([st.ravel() for st in strips if st.size])
    if r.size == 0:
        return 1.0, 1.0
    med = np.median(r)
    return (
        float((r >= RING_WHITE).mean()),
        float((np.abs(r.astype(np.int16) - med) <= RING_FLAT_TOL).mean()),
    )


def fill_white(gray, box) -> float:
    x0, y0, x1, y1 = box
    r = gray[y0:y1, x0:x1]
    return float((r >= RING_WHITE).mean()) if r.size else 1.0


def _iou_cont(a, b) -> tuple[float, float]:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0, 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter), inter / max(1, min(aa, bb))


def boxes_collide(a, b, iou: float = 0.1, cont: float = 0.9) -> bool:
    u, c = _iou_cont(a, b)
    return u > iou or c >= cont


def run_bubbles(a) -> None:
    import cv2

    from anime_tools.captions.ocr_sfx import line_kind

    root = corpus_root(a.corpus)

    for ed in a.editions:
        boxes = load_jsonl(out_dir(a.name) / f"boxes_{ed}.jsonl")
        reads = load_jsonl(out_dir(a.name) / f"reads_{ed}.jsonl")
        path = out_dir(a.name) / f"bubbles_{ed}.jsonl"
        n_pages = n_bub = n_ok = 0
        reasons: dict[str, int] = {}
        with path.open("w", encoding="utf-8") as fh:
            for rel, row in boxes.items():
                if rel not in reads:
                    continue
                n_pages += 1
                texts = reads[rel]["texts"]
                gray = cv2.imread(str(root / rel), cv2.IMREAD_GRAYSCALE)
                page: list[dict] = []
                for k, idx in enumerate(group_bubbles(row["boxes"])):
                    lines = [_flatten(ed, texts[i]) for i in idx]
                    bx = [row["boxes"][i] for i in idx]
                    kind = "speech"
                    if ed == "ja" and all(line_kind(ln) == "sfx" for ln in lines if ln):
                        kind = "sfx"
                    thick = [
                        (b[2] - b[0]) if _is_column(b) else (b[3] - b[1]) for b in bx
                    ]
                    thick.sort()
                    ring = fill = 1.0
                    if gray is not None:
                        ring = min(max(ring_score(gray, b)) for b in bx)
                        fill = min(fill_white(gray, b) for b in bx)
                    reason = ""
                    if any(not ln for ln in lines):
                        reason = "empty_line"
                    elif any(not script_ok(ed, ln) for ln in lines):
                        reason = "script"
                    elif sum(len(ln.replace(" ", "")) for ln in lines) < a.min_chars:
                        reason = "short"
                    elif kind == "sfx":
                        reason = "sfx"
                    elif any(repetitive(ln) for ln in lines):
                        reason = "repeat"
                    elif (
                        ed == "ja"
                        and all(kana_only(ln) for ln in lines)
                        and sum(len(content(ln)) for ln in lines) <= a.min_kana
                    ):
                        reason = "short_kana"
                    elif ring < RING_OK and fill < FILL_OK:
                        reason = "no_balloon"
                    union = [
                        min(b[0] for b in bx),
                        min(b[1] for b in bx),
                        max(b[2] for b in bx),
                        max(b[3] for b in bx),
                    ]
                    page.append(
                        {
                            "rel": rel,
                            "k": k,
                            "size": row["size"],
                            "box": union,
                            "boxes": bx,
                            "scores": [row["scores"][i] for i in idx],
                            "lines": lines,
                            "column": _is_column(bx[0]),
                            "line_h": thick[len(thick) // 2],
                            "ring": round(ring, 3),
                            "fill": round(fill, 3),
                            "ok": not reason,
                            "reason": reason,
                            "kind": kind,
                        }
                    )
                # overlap: an accepted bubble touching any other bubble's box
                for i, bi in enumerate(page):
                    if not bi["ok"]:
                        continue
                    for j, bj in enumerate(page):
                        if i != j and any(
                            boxes_collide(p, q)
                            for p in bi["boxes"]
                            for q in bj["boxes"]
                        ):
                            bi["ok"], bi["reason"] = False, "overlap"
                            break
                for b in page:
                    n_bub += 1
                    n_ok += b["ok"]
                    if b["reason"]:
                        reasons[b["reason"]] = reasons.get(b["reason"], 0) + 1
                    append_jsonl(fh, b)
        print(
            f"[{ed}] bubbles: {n_pages} pages, {n_bub} bubbles, {n_ok} ok "
            f"({100 * n_ok / max(n_bub, 1):.1f} %), rejects {reasons} → {path}",
            flush=True,
        )


# --------------------------------------------------------------------------- panels


def _reading_order(bubbles: list[dict]) -> list[dict]:
    """Manga order inside a panel: rows first, right → left inside a row. A
    row is a greedy band: sorted by top edge, a bubble starts a new row when
    its top is more than one median bubble height below the row's first top
    (S0 used half a height and a fixed grid, which sorted y-first)."""
    if not bubbles:
        return []
    med_h = sorted(b["box"][3] - b["box"][1] for b in bubbles)[len(bubbles) // 2]
    band = max(1, med_h)
    rows: list[list[dict]] = []
    for b in sorted(bubbles, key=lambda b: b["box"][1]):
        if rows and b["box"][1] - rows[-1][0]["box"][1] <= band:
            rows[-1].append(b)
        else:
            rows.append([b])
    out: list[dict] = []
    for row in rows:
        out += sorted(row, key=lambda b: (-b["box"][2], b["box"][1]))
    return out


def _centre_in(box, panel) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return panel[0] <= cx < panel[2] and panel[1] <= cy < panel[3]


def _area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _clip(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def clean_frames(frames: list, size, min_area: float) -> tuple[list, dict]:
    """Drop frames under ``min_area`` of the page and frames containing ≥ 2
    other frames (a whole-page box over the real panels). Returns the kept
    frames and the drop counts."""
    W, H = size
    drops = {"small": 0, "container": 0}
    keep = []
    for f in frames:
        if _area(f) < min_area * W * H:
            drops["small"] += 1
            continue
        keep.append(f)
    out = []
    for f in keep:
        inside = sum(
            1
            for g in keep
            if g is not f and _iou_cont(f, g)[1] >= 0.8 and _area(g) < _area(f)
        )
        if inside >= 2:
            drops["container"] += 1
            continue
        out.append(f)
    return out, drops


def _grow(panel, boxes, size, margin: float = 0.02):
    W, H = size
    x0, y0, x1, y1 = panel
    for b in boxes:
        x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
    grown = [x0, y0, x1, y1] != list(panel)
    if grown:
        mx, my = int(margin * (x1 - x0)), int(margin * (y1 - y0))
        x0, y0, x1, y1 = (
            max(0, x0 - mx),
            max(0, y0 - my),
            min(W, x1 + mx),
            min(H, y1 + my),
        )
    return [x0, y0, x1, y1], grown


def run_panels(a) -> None:
    for ed in a.editions:
        boxes = load_jsonl(out_dir(a.name) / f"boxes_{ed}.jsonl")
        bubbles = load_jsonl(out_dir(a.name) / f"bubbles_{ed}.jsonl", key=None)
        by_page: dict[str, list[dict]] = {}
        for b in bubbles:
            by_page.setdefault(b["rel"], []).append(b)
        path = out_dir(a.name) / f"samples_{ed}.jsonl"
        n_pages = n_panels = n_samples = n_orphan = n_whole = 0
        multi = grown = slant = 0
        drops = {"small": 0, "container": 0}
        t0 = time.time()
        with path.open("w", encoding="utf-8") as fh:
            for i, (rel, bl) in enumerate(sorted(by_page.items()), 1):
                size = boxes[rel]["size"]
                frames, d = clean_frames(
                    boxes[rel].get("frames") or [], size, a.min_area
                )
                for key in drops:
                    drops[key] += d[key]
                panels = frames or [[0, 0, size[0], size[1]]]
                n_pages += 1
                n_whole += not frames
                n_panels += len(panels)
                # each bubble → the smallest panel holding its centre
                assign: dict[int, list[dict]] = {k: [] for k in range(len(panels))}
                for b in bl:
                    hits = [
                        k for k, pn in enumerate(panels) if _centre_in(b["box"], pn)
                    ]
                    if hits:
                        assign[min(hits, key=lambda k: _area(panels[k]))].append(b)
                    elif b["ok"]:
                        n_orphan += 1
                for k, pn in enumerate(panels):
                    inside = assign[k]
                    ok = _reading_order([b for b in inside if b["ok"]])
                    if not ok:
                        continue
                    crop, g = _grow(pn, [b["box"] for b in ok], size)
                    grown += g
                    sl = any(
                        _iou_cont(crop, q)[0] > 0
                        and _area(_clip(crop, q)) > 0.1 * _area(crop)
                        for j, q in enumerate(panels)
                        if j != k
                    )
                    slant += sl
                    n_samples += 1
                    multi += len(ok) > 1
                    intact = [b for b in inside if not b["ok"]]
                    append_jsonl(
                        fh,
                        {
                            "rel": rel,
                            "k": k,
                            "size": size,
                            "panel": crop,
                            "frame": pn,
                            "grown": g,
                            "slant": sl,
                            "n_panels": len(panels),
                            "bubbles": [b["k"] for b in ok],
                            "boxes": [b["box"] for b in ok],
                            "lines": [" ".join(b["lines"]) for b in ok],
                            "line_h": [b["line_h"] for b in ok],
                            "n_rejected": len(intact),
                            "intact": [
                                {"k": b["k"], "box": b["box"], "reason": b["reason"]}
                                for b in intact
                            ],
                        },
                    )
                if i % 200 == 0:
                    print(f"  [{ed}] {i}/{len(by_page)} pages", flush=True)
        print(
            f"[{ed}] panels: {n_pages} pages, {n_panels} panels "
            f"({n_panels / max(n_pages, 1):.1f}/page, {n_whole} pages without a frame "
            f"= whole page; dropped {drops}), {n_samples} samples with text "
            f"({multi} multi-bubble, {grown} grown to hold a bubble, {slant} "
            f"overlap a sibling frame > 10 %), {n_orphan} ok bubbles outside every "
            f"panel, {time.time() - t0:.0f}s → {path}",
            flush=True,
        )


# --------------------------------------------------------------------------- sheet


def _font(size: int):
    import subprocess

    from PIL import ImageFont

    try:
        hit = subprocess.run(
            ["fc-list", ":", "file"], capture_output=True, text=True, check=False
        ).stdout
        for pat in ("NotoSansCJK-Regular", "NotoSansCJK", "DejaVuSans"):
            for line in hit.splitlines():
                f = line.split(":")[0].strip()
                if pat in f and f.endswith((".ttc", ".otf", ".ttf")):
                    return ImageFont.truetype(f, size)
    except Exception:
        pass
    return ImageFont.load_default()


def _wrap(draw, text: str, font, width: int) -> list[str]:
    """Greedy wrap by rendered width (a CJK glyph is ~2 Latin columns; S0
    wrapped by character count and the JA captions overprinted)."""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def run_sheet(a) -> None:
    from PIL import Image, ImageDraw

    root = corpus_root(a.corpus)
    rng = random.Random(a.seed)
    for ed in a.editions:
        rows = load_jsonl(out_dir(a.name) / f"samples_{ed}.jsonl", key=None)
        pick = rng.sample(rows, min(a.n, len(rows)))
        cell_w, cell_h, cap_h = 400, 400, 96
        cols = 4
        n_rows = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, n_rows * (cell_h + cap_h)), "white")
        draw = ImageDraw.Draw(sheet)
        font = _font(15)
        tsv = out_dir(a.name) / f"sheet_{ed}.tsv"
        with tsv.open("w", encoding="utf-8") as fh:
            fh.write("i\trel\tk\tn_bubbles\tn_rejected\tline_h\tlines\thand\n")
            for i, r in enumerate(pick):
                x0, y0, x1, y1 = r["panel"]
                with Image.open(root / r["rel"]) as im:
                    crop = im.convert("RGB").crop((x0, y0, x1, y1))
                d = ImageDraw.Draw(crop)
                lw = max(2, (x1 - x0) // 300)
                for it in r.get("intact", []):
                    b = it["box"]
                    d.rectangle(
                        (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0),
                        outline="blue",
                        width=lw,
                    )
                    d.text(
                        (b[0] - x0 + 4, b[1] - y0 + 2), it["reason"], "blue", _font(22)
                    )
                for j, b in enumerate(r["boxes"]):
                    d.rectangle(
                        (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0),
                        outline="red",
                        width=lw,
                    )
                    d.text((b[0] - x0 + 4, b[1] - y0 + 2), str(j), "red", _font(28))
                crop.thumbnail((cell_w - 8, cell_h - 8))
                cx, cy = (i % cols) * cell_w, (i // cols) * (cell_h + cap_h)
                sheet.paste(crop, (cx + 4, cy + 4))
                label = f"{i}: " + " | ".join(
                    f"[{j}] {ln}" for j, ln in enumerate(r["lines"])
                )
                for li, seg in enumerate(_wrap(draw, label, font, cell_w - 8)[:4]):
                    draw.text((cx + 4, cy + cell_h + 2 + 20 * li), seg, "black", font)
                fh.write(
                    f"{i}\t{r['rel']}\t{r['k']}\t{len(r['lines'])}\t{r['n_rejected']}\t"
                    f"{r['line_h']}\t{' | '.join(r['lines'])}\t\n"
                )
        png = out_dir(a.name) / f"sheet_{ed}.png"
        sheet.save(png)
        print(
            f"[{ed}] sheet: {len(pick)} of {len(rows)} samples → {png}, {tsv}",
            flush=True,
        )


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("det", "read", "bubbles", "panels", "sheet"))
    ap.add_argument(
        "--corpus", default=None, help="corpus root (or ANIMA_RENDER_CORPUS)"
    )
    ap.add_argument("--name", default="s0", help="output dir under output/render/")
    ap.add_argument("--editions", nargs="+", default=["en", "ja"], choices=EDITIONS)
    ap.add_argument("--artists", nargs="*", default=None, help="artist dirs to keep")
    ap.add_argument(
        "--per_work", type=int, default=0, help="seeded pages per work (0 = all)"
    )
    ap.add_argument("--limit", type=int, default=0, help="pages per edition (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto)")
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--bs", type=int, default=16, help="reader batch")
    ap.add_argument("--min_chars", type=int, default=2)
    ap.add_argument(
        "--min_kana",
        type=int,
        default=2,
        help="bubbles: JA kana-only reads with ≤ this many content kana are moans → short_kana",
    )
    ap.add_argument(
        "--min_area",
        type=float,
        default=0.03,
        help="panels: drop frames under this fraction of the page area",
    )
    ap.add_argument("--n", type=int, default=40, help="sheet: samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.artists = set(a.artists) if a.artists else None
    if a.device is None:
        import torch

        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    {
        "det": run_det,
        "read": run_read,
        "bubbles": run_bubbles,
        "panels": run_panels,
        "sheet": run_sheet,
    }[a.stage](a)


if __name__ == "__main__":
    main()
