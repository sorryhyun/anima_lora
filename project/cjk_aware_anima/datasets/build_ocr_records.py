#!/usr/bin/env python3
"""Hybrid OCR records: PP-OCRv6 v2 records + PaddleOCR-VL-1.6 as detector and second reader.

plan_base1.md B0. PP-OCRv6 stays the recognizer of record; VL-1.6 does two jobs
only — (a) *detector*: page ``Spotting:`` adds lines PP's DB head never boxed
(the 44 masked-but-no-line sincos pages are the point); (b) *second reader on
PP's own boxes* where PP is weak (rule 1b): score < ``--weak_score``, a symbol
(``♡ ー ～``) in dispute with the matched Spotting line, or a line the SFX rule
flags. Not a swap, not prompt engineering.

Two stages, so the GPU pass runs once and the merge is re-runnable on CPU::

    # GPU (daemon): Spotting on every page of the shard + ``OCR:`` on every PP box
    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima/datasets/build_ocr_records.py --stage spot"
    # CPU: merge, kind, reading order, gate numbers
    python project/cjk_aware_anima/datasets/build_ocr_records.py --stage merge

Merge rules (``merge_page``):

* VL Spotting quads → axis bounds on the page; VL columns are joined into
  blocks with ``anime_tools.ocr._text.join_cjk`` first, because the PP records
  are already column-joined and a per-column quad never reaches IoU 0.5 with a
  three-column balloon box.
* A VL block is the *same line* as a PP record when IoU ≥ ``--iou`` (0.3) or
  one box holds ≥ ``--contain`` (0.5) of the smaller one, or the boxes touch and
  the texts agree (``--text_sim`` 0.75). The plan's IoU 0.5 was measured too
  strict on sincos: a 30 px column that VL and PP box 12 px apart is IoU 0.42
  with identical text.
* Rule 1b replaces PP's text with the VL ``OCR:`` read of PP's box only when
  the read passes the repetition guard (no 3-gram repeated ≥ 3×, no glyph run
  ≥ 8), is at most 2× PP's length and survives ``keep_line``. ``pp_text`` is
  kept on the record for the sheet.
* Unmatched VL blocks enter with ``engine=vl16_spotting`` after the floors
  (``min_chars`` 3, ASCII-only, tally — the same ``keep_line`` the PP engine
  applies) and the runaway / symbol-only garbage gate.
* Every record gets ``kind ∈ {speech, sfx, chrome}`` — B0 uses the v1 text rule
  (``ocr_sfx.line_kind``) plus a chrome word list; B1 re-labels with hand labels.
  Reading order is recomputed over the merged page with
  ``anime_tools.ocr.reading_order`` (page-aware, right-to-left).

Output: ``post_image_dataset/cjk_unmask/ocr_records_<shard>_hybrid.jsonl`` and
``output/tests/ocr_hybrid/<shard>_report.md`` with the two gate numbers (the
masked-but-no-line floor, PP alone vs hybrid; best-match similarity to manga-ocr
on the 40 A/B pages, PP alone vs hybrid).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))  # ocr_sfx (sibling, torch-free)

from anime_tools.captions.ocr_sidecar import OcrLine  # noqa: E402
from anime_tools.ocr import reading_order  # noqa: E402
from anime_tools.ocr._text import join_cjk, keep_line  # noqa: E402
from ocr_sfx import line_kind  # noqa: E402

VL_MODEL = REPO / "models/paddleocr_vl_1.6"
SPOT_MAX_PIXELS = 2048 * 28 * 28
CROP_MAX_PIXELS = 1280 * 28 * 28
UPSCALE_BELOW = 1500
CROP_PAD = 4

LOC_RE = re.compile(r"<\|LOC_(\d+)\|>")

# UI chrome seen in the sincos records (screenshot pages): neither speech nor
# SFX. B1 grows this list against the hand labels; B0 only needs the ones the
# eyeball named so the mirror builder can start dropping them.
CHROME_WORDS: tuple[str, ...] = (
    "ツイート",
    "ポスト",
    "完了にする",
    "お気に入り",
    "リツイート",
    "フォロー",
    "いいね",
    "返信",
)
CHROME_RE = re.compile(r"^\d{1,2}:\d{2}$|^[\d.,]+\s*(cm|kg|%)$", re.IGNORECASE)

SYMBOLS = frozenset("♡♥❤ー～〜♪☆★")
_STRIP_RE = re.compile(r"[\s。、．，,.・…‥「」『』!！?？~～〜❤♥♡♪☆★()（）\-ー—–|｜]")


# --------------------------------------------------------------------------- text
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


def is_symbol_only(text: str) -> bool:
    return not any(ch.isalnum() for ch in text)


def symbol_dispute(pp: str, vl: str) -> bool:
    """Same letters, different symbols: PP dropped a ``♡`` or spelled ``ー`` as
    ``1`` where the matched Spotting read has the mark (or the reverse)."""
    if norm(pp) != norm(vl):
        return False
    return {c for c in pp if c in SYMBOLS} != {c for c in vl if c in SYMBOLS}


def record_kind(text: str) -> str:
    if text.strip() in CHROME_WORDS or CHROME_RE.match(text.strip()):
        return "chrome"
    return line_kind(text)


# --------------------------------------------------------------------------- geometry
def parse_spotting(text: str, width: int, height: int) -> list[OcrLine]:
    """``line<|LOC_x0|><|LOC_y0|>…×8`` per row, the grid 0–1000 on the fed image,
    → axis-aligned boxes in the page's own pixels (the grid is a fraction, so
    the ×2 upscale cancels). Rows without exactly 8 LOC tokens are skipped."""
    out: list[OcrLine] = []
    for raw in text.splitlines():
        locs = [int(v) for v in LOC_RE.findall(raw)]
        body = LOC_RE.sub("", raw).strip()
        if len(locs) != 8 or not body:
            continue
        xs = [locs[i] / 1000 * width for i in (0, 2, 4, 6)]
        ys = [locs[i] / 1000 * height for i in (1, 3, 5, 7)]
        box = (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1)
        out.append(OcrLine(seq=len(out), box=box, score=-1.0, text=body))
    return out


def _area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap(a, b) -> tuple[float, float]:
    """``(iou, containment)`` — containment is the intersection over the
    smaller box, so a column quad inside a joined balloon box scores ~1."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0, 0.0
    aa, ab = _area(a), _area(b)
    return inter / (aa + ab - inter), inter / max(1, min(aa, ab))


# --------------------------------------------------------------------------- merge
@dataclass
class Rec:
    stem: str
    text: str
    score: float | None
    box: list[int]
    engine: str
    post: str = "hybrid_b0"
    kind: str = "speech"
    pp_text: str | None = None  # PP's read when rule 1b replaced it
    vl_text: str | None = None  # the matched Spotting read (audit / dispute)
    rule1b: str | None = None  # why the second reader ran: weak | symbol | sfx
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return {k: v for k, v in d.items() if v is not None}


HEARTS = frozenset("♡♥❤")


def _normalize_read(text: str) -> str:
    """``❤️`` (emoji + variation selector) → ``♥``: the symbol block has rows for
    ``♡`` / ``♥``, not for the emoji sequence."""
    return text.replace("\ufe0f", "").replace("❤", "♥")


_LETTERS_RE = re.compile(r"[\s。、．，,.・…‥「」『』!！?？~～〜❤♥♡♪☆★()（）]")


def letters(s: str) -> str:
    """Like :func:`norm` but keeps ``ー`` / ``一`` / ``-``: the letters a symbol
    dispute must leave alone."""
    return _LETTERS_RE.sub("", unicodedata.normalize("NFKC", s))


def accept_second_read(
    pp_text: str,
    vl_read: str,
    min_chars: int,
    corroborate: str | None = None,
    reason: str | None = None,
) -> str | None:
    """Rule 1b's guard: the VL ``OCR:`` read of PP's box, or ``None``.

    Rejects a runaway, an over-long read (> 2× PP), one under the floors, and
    one that *loses* a heart PP had (PP drops hearts, it never invents them —
    measured 2026-09-05: 3 of 9 symbol disputes went the wrong way without
    this). With ``corroborate`` (the matched Spotting read) the crop read must
    agree with that second, independent VL reading (sim ≥ 0.5); a weak PP
    line whose two VL readings disagree keeps PP's text. A ``symbol`` dispute
    may only move symbols: a read that changes a letter (``ご主人様`` →
    ``ごー主人様``, ``一発`` → ``ー発``) is rejected.
    """
    read = _normalize_read("".join(vl_read.split()))  # columns come ``\\n``-split
    if not read or is_runaway(read):
        return None
    if len(read) > 2 * max(1, len(pp_text)):
        return None
    if not keep_line(read, min_chars=min_chars, skip_en=True):
        return None
    if {c for c in pp_text if c in HEARTS} - {c for c in read if c in HEARTS}:
        return None
    if corroborate is not None and sim(read, corroborate) < 0.5:
        return None
    if reason == "symbol" and letters(read) != letters(pp_text):
        return None  # a symbol dispute may only move symbols, never letters
    return read


def merge_page(
    stem: str,
    pp: list[dict],
    spotting: list[OcrLine],
    crop_reads: list[str] | None,
    *,
    page_size: tuple[int, int] | None = None,
    iou_thr: float,
    contain_thr: float,
    text_sim: float = 0.75,
    weak_score: float,
    min_chars: int,
) -> tuple[list[Rec], dict]:
    """One page's PP records + VL Spotting blocks (+ VL crop reads of the PP
    boxes, index-aligned with ``pp``) → hybrid records in reading order."""
    stats = defaultdict(int)
    blocks = join_cjk(spotting)
    blocks = [b for b in blocks if not is_runaway(b.text)]
    if page_size is not None:
        # a quad over most of the page is Spotting hallucinating a caption for
        # the artwork (``だなんか`` on [0, 0, 704, 1487]), not a text line
        page_area = max(1, page_size[0] * page_size[1])
        kept = [b for b in blocks if _area(b.box) < 0.4 * page_area]
        stats["vl_fullpage_dropped"] = len(blocks) - len(kept)
        blocks = kept
    stats["vl_blocks"] = len(blocks)

    # match: each VL block to at most one PP record (best IoU)
    matched_vl: dict[int, int] = {}  # block idx -> pp idx
    pp_best: dict[int, tuple[float, int]] = {}
    for bi, blk in enumerate(blocks):
        best = None
        for pi, r in enumerate(pp):
            iou, cont = overlap(blk.box, tuple(r["box"]))
            same_text = cont > 0 and sim(blk.text, r["text"]) >= text_sim
            if iou >= iou_thr or cont >= contain_thr or same_text:
                key = max(iou, cont, 1.0 if same_text else 0.0)
                if best is None or key > best[0]:
                    best = (key, pi)
        if best is not None:
            matched_vl[bi] = best[1]
            if best[1] not in pp_best or best[0] > pp_best[best[1]][0]:
                pp_best[best[1]] = (best[0], bi)

    recs: list[Rec] = []
    for pi, r in enumerate(pp):
        rec = Rec(
            stem=stem,
            text=r["text"],
            score=r.get("score"),
            box=list(r["box"]),
            engine=r.get("engine", "ppocr_v6"),
        )
        if pi in pp_best:
            rec.vl_text = blocks[pp_best[pi][1]].text
            stats["pp_matched"] += 1
        reason = None
        if rec.score is not None and rec.score < weak_score:
            reason = "weak"
        elif rec.vl_text and symbol_dispute(rec.text, rec.vl_text):
            reason = "symbol"
        elif line_kind(rec.text) == "sfx":
            reason = "sfx"
        if reason and crop_reads is not None and pi < len(crop_reads):
            rec.rule1b = reason
            stats[f"rule1b_{reason}"] += 1
            new = accept_second_read(
                rec.text,
                crop_reads[pi],
                min_chars,
                corroborate=rec.vl_text if reason == "weak" else None,
                reason=reason,
            )
            if new is not None and new != rec.text:
                rec.pp_text = rec.text
                rec.text = new
                rec.engine = "ppocr_v6+vl16_crop"
                stats["replaced"] += 1
            elif new is None:
                stats["second_read_rejected"] += 1
        recs.append(rec)

    seen_vl: list[tuple[str, tuple]] = []
    for bi, blk in enumerate(blocks):
        if bi in matched_vl:
            continue
        text = _normalize_read("".join(blk.text.split()))
        if any(norm(text) == t and overlap(blk.box, b)[1] >= 0.7 for t, b in seen_vl):
            stats["vl_only_dup"] += 1  # the same quad emitted twice
            continue
        seen_vl.append((norm(text), blk.box))
        if not keep_line(text, min_chars=min_chars, skip_en=True):
            stats["vl_only_floor"] += 1
            continue
        if is_symbol_only(text):
            stats["vl_only_garbage"] += 1
            continue
        recs.append(
            Rec(
                stem=stem,
                text=text,
                score=None,
                box=list(blk.box),
                engine="vl16_spotting",
            )
        )
        stats["vl_only"] += 1

    for rec in recs:
        rec.kind = record_kind(rec.text)
        stats[f"kind_{rec.kind}"] += 1

    # reading order over the merged page
    lines = [
        OcrLine(seq=i, box=tuple(r.box), score=r.score or 0.0, text=r.text)
        for i, r in enumerate(recs)
    ]
    order = [ln.seq for ln in reading_order(lines)]
    return [recs[i] for i in order], dict(stats)


# --------------------------------------------------------------------------- GPU stage
def run_spot_stage(
    pages: list[Path],
    pp_by_stem: dict[str, list[dict]],
    raw_path: Path,
    *,
    batch_size: int,
    crop_batch: int,
    model_dir: Path,
) -> None:
    """Spotting on every page + ``OCR:`` on every PP box. Appends one JSON row
    per page to ``raw_path``; pages already there are skipped (resumable)."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    done = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["stem"])
    todo = [p for p in pages if p.stem not in done]
    print(f"spot stage: {len(todo)} pages to do ({len(done)} cached)", flush=True)
    if not todo:
        return

    t0 = time.time()
    model = (
        AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    proc = AutoProcessor.from_pretrained(str(model_dir))
    min_edge = proc.image_processor.size["shortest_edge"]
    print(f"model loaded {time.time() - t0:.1f}s", flush=True)

    def gen(images, prompt, max_pixels, max_new):
        texts = [
            proc.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for _ in images
        ]
        inputs = proc(
            text=texts,
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            images_kwargs={
                "size": {"shortest_edge": min_edge, "longest_edge": max_pixels}
            },
        ).to("cuda")
        n = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            o = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True
            )
        skip = {proc.tokenizer.eos_token_id, proc.tokenizer.pad_token_id}
        return [
            proc.tokenizer.decode(
                [i for i in row[n:].tolist() if i not in skip]
            ).strip()
            for row in o
        ]

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    out = raw_path.open("a", encoding="utf-8")
    for k in range(0, len(todo), batch_size):
        chunk = todo[k : k + batch_size]
        ims = [Image.open(p).convert("RGB") for p in chunk]
        fed = [
            im.resize((im.width * 2, im.height * 2), Image.Resampling.LANCZOS)
            if max(im.size) < UPSCALE_BELOW
            else im
            for im in ims
        ]
        t = time.time()
        spots = gen(fed, "Spotting:", SPOT_MAX_PIXELS, 1024)
        dt_spot = time.time() - t
        # crop reads on PP's boxes, pooled across the chunk
        crops, owners = [], []
        for ci, (p, im) in enumerate(zip(chunk, ims)):
            for r in pp_by_stem.get(p.stem, []):
                x0, y0, x1, y1 = r["box"]
                W, H = im.size
                crops.append(
                    im.crop(
                        (
                            max(0, x0 - CROP_PAD),
                            max(0, y0 - CROP_PAD),
                            min(W, x1 + CROP_PAD),
                            min(H, y1 + CROP_PAD),
                        )
                    )
                )
                owners.append(ci)
        reads = []
        t = time.time()
        for j in range(0, len(crops), crop_batch):
            reads += gen(crops[j : j + crop_batch], "OCR:", CROP_MAX_PIXELS, 256)
        dt_crop = time.time() - t
        per_page: dict[int, list[str]] = defaultdict(list)
        for ci, rd in zip(owners, reads):
            per_page[ci].append(rd)
        for ci, (p, im) in enumerate(zip(chunk, ims)):
            row = {
                "stem": p.stem,
                "size": list(im.size),
                "fed": list(fed[ci].size),
                "spotting": spots[ci],
                "crop_ocr": per_page.get(ci, []),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        print(
            f"[{k + len(chunk)}/{len(todo)}] spot {dt_spot:.1f}s crops {len(crops)} "
            f"{dt_crop:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GB",
            flush=True,
        )
    out.close()


# --------------------------------------------------------------------------- merge stage
def load_records(path: Path) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["stem"]].append(r)
    return by


def floor_count(
    masks_dir: Path, by_stem: dict[str, list]
) -> tuple[int, int, list[str]]:
    masked = {p.name[: -len("_mask.png")] for p in masks_dir.glob("*_mask.png")}
    missing = sorted(s for s in masked if not by_stem.get(s))
    return len(masked), len(missing), missing


def best_match_sim(
    ref_by_stem: dict[str, list[dict]], by_stem: dict[str, list], stems: list[str]
) -> tuple[float, int, int]:
    """Mean best-match similarity of each manga-ocr reference line to the
    records of the same page (the A/B metric), plus the ≥ 0.9 count."""
    sims = []
    for s in stems:
        texts = [
            (r["text"] if isinstance(r, dict) else r.text) for r in by_stem.get(s, [])
        ]
        for ref in ref_by_stem.get(s, []):
            sims.append(max((sim(ref["text"], t) for t in texts), default=0.0))
    if not sims:
        return 0.0, 0, 0
    return sum(sims) / len(sims), sum(1 for v in sims if v >= 0.9), len(sims)


def run_merge_stage(opts, pages, pp_by_stem, raw_path, out_path, report_path):
    raw = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            raw[r["stem"]] = r
    missing_raw = [p.stem for p in pages if p.stem not in raw]
    if missing_raw:
        print(
            f"WARN: {len(missing_raw)} pages without a Spotting row: {missing_raw[:8]}"
        )

    hybrid: dict[str, list[Rec]] = {}
    totals = defaultdict(int)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            stem = p.stem
            row = raw.get(stem)
            pp = pp_by_stem.get(stem, [])
            if row is None:
                spotting, crop_reads = [], None
            else:
                W, H = row["size"]
                spotting = parse_spotting(row["spotting"], W, H)
                crop_reads = row.get("crop_ocr")
                if crop_reads is not None and len(crop_reads) != len(pp):
                    print(
                        f"WARN {stem}: {len(crop_reads)} crop reads for {len(pp)} PP lines"
                    )
                    crop_reads = None
            recs, stats = merge_page(
                stem,
                pp,
                spotting,
                crop_reads,
                page_size=tuple(row["size"]) if row is not None else None,
                iou_thr=opts.iou,
                contain_thr=opts.contain,
                text_sim=opts.text_sim,
                weak_score=opts.weak_score,
                min_chars=opts.min_chars,
            )
            for k, v in stats.items():
                totals[k] += v
            if recs:
                hybrid[stem] = recs
                for r in recs:
                    f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")

    # ---- gate numbers
    masks_dir = REPO / "post_image_dataset/masks" / opts.shard
    n_masked, floor_pp, miss_pp = floor_count(masks_dir, pp_by_stem)
    _, floor_hy, miss_hy = floor_count(masks_dir, hybrid)
    recovered = sorted(set(miss_pp) - set(miss_hy))

    ref_path = (
        REPO / "post_image_dataset/cjk_unmask" / f"ocr_records_{opts.shard}.jsonl"
    )
    ab_path = REPO / "output/tests/vl16_ab/ab.jsonl"
    sim_lines = []
    if ref_path.exists() and ab_path.exists():
        ref = load_records(ref_path)
        ab_stems = [
            json.loads(row)["stem"]
            for row in ab_path.read_text(encoding="utf-8").splitlines()
        ]
        m_pp, hi_pp, n_ref = best_match_sim(ref, pp_by_stem, ab_stems)
        m_hy, hi_hy, _ = best_match_sim(ref, hybrid, ab_stems)
        # replaced lines only: did the second reader move us toward manga-ocr?
        rep_pp, rep_hy = [], []
        for s in ab_stems:
            refs = ref.get(s, [])
            for r in hybrid.get(s, []):
                if r.pp_text is not None and refs:
                    rep_pp.append(max(sim(x["text"], r.pp_text) for x in refs))
                    rep_hy.append(max(sim(x["text"], r.text) for x in refs))
        sim_lines = [
            f"| best-match sim to manga-ocr, {n_ref} ref lines on {len(ab_stems)} A/B pages | "
            f"{m_pp:.3f} ({hi_pp} ≥ 0.9) | {m_hy:.3f} ({hi_hy} ≥ 0.9) |",
        ]
        if rep_pp:
            sim_lines.append(
                f"| replaced lines only ({len(rep_pp)}), PP read vs VL read | "
                f"{sum(rep_pp) / len(rep_pp):.3f} | {sum(rep_hy) / len(rep_hy):.3f} |"
            )

    n_pp = sum(len(v) for v in pp_by_stem.values())
    n_hy = sum(len(v) for v in hybrid.values())
    md = [
        f"# Hybrid OCR records — {opts.shard} (B0, {time.strftime('%Y-%m-%d %H:%M')})\n",
        f"records: `{out_path.relative_to(REPO)}`; raw VL outputs: `{raw_path.relative_to(REPO)}`\n",
        "| | PP-OCRv6 v2 | hybrid |",
        "|---|---|---|",
        f"| pages ({len(pages)} in shard) with any line | {len(pp_by_stem)} | {len(hybrid)} |",
        f"| lines | {n_pp} | {n_hy} |",
        f"| **masked-but-no-line floor** ({n_masked} masked) | **{floor_pp}** | **{floor_hy}** |",
        *sim_lines,
        "",
        f"- VL Spotting blocks (after column join + runaway drop): {totals.get('vl_blocks', 0)}; "
        f"matched to a PP record: {totals.get('pp_matched', 0)} PP lines; "
        f"VL-only lines kept: {totals.get('vl_only', 0)} "
        f"(floor-dropped {totals.get('vl_only_floor', 0)}, symbol-only {totals.get('vl_only_garbage', 0)}, "
        f"duplicate quad {totals.get('vl_only_dup', 0)}, full-page quad {totals.get('vl_fullpage_dropped', 0)})",
        f"- rule 1b second reads: weak {totals.get('rule1b_weak', 0)} · symbol {totals.get('rule1b_symbol', 0)} · "
        f"sfx {totals.get('rule1b_sfx', 0)} → replaced {totals.get('replaced', 0)}, "
        f"rejected by the guard {totals.get('second_read_rejected', 0)}",
        f"- kind: speech {totals.get('kind_speech', 0)} · sfx {totals.get('kind_sfx', 0)} · "
        f"chrome {totals.get('kind_chrome', 0)} (v1 text rule + chrome list; B1 re-labels)",
        f"- floor pages recovered by VL ({len(recovered)}): {' '.join(recovered) or '—'}",
        f"- floor pages still empty ({len(miss_hy)}): {' '.join(miss_hy) or '—'}",
        "",
        "## VL-only lines (engine=vl16_spotting)\n",
        "| stem | kind | text | box |",
        "|---|---|---|---|",
    ]
    for stem, recs in hybrid.items():
        for r in recs:
            if r.engine == "vl16_spotting":
                md.append(f"| {stem} | {r.kind} | {r.text} | {r.box} |")
    md += [
        "",
        "## Rule 1b replacements\n",
        "| stem | why | PP-OCRv6 | VL-1.6 crop | Spotting |",
        "|---|---|---|---|---|",
    ]
    for stem, recs in hybrid.items():
        for r in recs:
            if r.pp_text is not None:
                md.append(
                    f"| {stem} | {r.rule1b} | {r.pp_text} | {r.text} | {r.vl_text or ''} |"
                )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[: 12 + len(sim_lines)]))
    print(f"\nwrote {out_path} ({n_hy} lines) and {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shard", default="sincos")
    ap.add_argument("--stage", choices=["spot", "merge", "all"], default="all")
    ap.add_argument(
        "--records",
        type=Path,
        default=None,
        help="PP-OCRv6 records (default: …_ppocr_v2.jsonl)",
    )
    ap.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw VL outputs jsonl (default: ocr_raw_vl16_<shard>.jsonl)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=VL_MODEL)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop_batch", type=int, default=32)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--contain", type=float, default=0.5)
    ap.add_argument("--text_sim", type=float, default=0.75)
    ap.add_argument("--weak_score", type=float, default=0.85)
    ap.add_argument("--min_chars", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N pages only (smoke)")
    opts = ap.parse_args()

    base = REPO / "post_image_dataset/cjk_unmask"
    records = opts.records or base / f"ocr_records_{opts.shard}_ppocr_v2.jsonl"
    raw_path = opts.raw or base / f"ocr_raw_vl16_{opts.shard}.jsonl"
    out_path = opts.out or base / f"ocr_records_{opts.shard}_hybrid.jsonl"
    report_path = (
        opts.report or REPO / "output/tests/ocr_hybrid" / f"{opts.shard}_report.md"
    )
    pages = sorted((REPO / "post_image_dataset/resized" / opts.shard).glob("*.png"))
    if opts.limit:
        pages = pages[: opts.limit]
    pp_by_stem = load_records(records)
    print(
        f"{len(pages)} pages, {sum(len(v) for v in pp_by_stem.values())} PP lines on {len(pp_by_stem)} stems"
    )

    if opts.stage in ("spot", "all"):
        run_spot_stage(
            pages,
            pp_by_stem,
            raw_path,
            batch_size=opts.batch_size,
            crop_batch=opts.crop_batch,
            model_dir=opts.model,
        )
    if opts.stage in ("merge", "all"):
        run_merge_stage(opts, pages, pp_by_stem, raw_path, out_path, report_path)


if __name__ == "__main__":
    main()
