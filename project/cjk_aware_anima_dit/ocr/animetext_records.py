#!/usr/bin/env python3
"""D1 (``plan_det.md``): OCR records for the sincos shard on the AnimeText
detector — boxes from ``anime_tools.ocr.animetext`` (the D0 package module),
every box read by the SFX reader, ``kind`` from the hand labels by overlap else
the text rule, reading order from the package, and a **record-level dedupe** of
the nested reads YOLO's block-and-columns emit.

    # CPU: detect (ONNX, ~0.5 s/page), apply the cached reads, the regression table, the side row, the hand-pass sheet
    python project/cjk_aware_anima_dit/ocr/animetext_records.py --stage det apply side hand
    # GPU (daemon), only if the det stage found boxes the O6 probe never read:
    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima_dit/ocr/animetext_records.py --stage read"

Stages:

* ``det`` — ``AnimeTextDetector`` (yolo12l @ 640, conf 0.25, NMS 0.5, ``inner``
  nesting; ``--det_conf`` overrides) over every page →
  ``output/tests/ocr_animetext/boxes_pkg.jsonl`` (box + score).
* ``read`` — the SFX reader over every box not in the probe's read cache
  (``output/tests/ocr_animetext/reads.jsonl``, keyed ``(stem, box)``; the D0
  gate showed the package reproduces the probe's boxes exactly, so this is
  normally a no-op).
* ``apply`` — reads → records (guard-passed, ≥ ``--min_chars`` after ``norm``,
  carries a letter), ``kind`` (hand label whose box covers the record's — IoU ≥
  0.3 or containment ≥ 0.5, best IoU wins — else ``record_kind``), the dedupe
  (a record whose normalized text sits inside another record's on the same
  page, its box ≥ 0.85 inside that record's box, is dropped — the column read
  inside its block read), reading order →
  ``post_image_dataset/cjk_unmask/ocr_records_sincos_animetext.jsonl`` and the
  O6 regression table (floor / manga-ocr best-match / hand-SFX exact) against
  ``hybrid_vl`` and the O6 ``inner c0.25`` row, plus a ``join_cjk`` row (the
  OCR stage's sidecar default) so a cross-balloon join would show.
* ``side`` — decision 2, once: PP-OCRv6's recognizer on the same boxes (the
  engine's ``crop_quad`` path, score ≥ 0.6) → the same table row, written to
  ``output/tests/ocr_animetext/records_animetext_ppocr.jsonl`` (never the
  records dir).
* ``hand`` — 60 random *new* records (boxes no ``hybrid_vl`` or PP v3 record
  covers) → ``assets/animetext_new_lines_sincos.tsv`` (``real_text`` blank, for
  the hand pass) + crop sheets under ``output/tests/ocr_animetext/hand_pass/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

REPO = m109.REPO
ASSETS = m109.ASSETS
OUT = REPO / "output/tests/ocr_animetext"
RECORDS_DIR = REPO / "post_image_dataset/cjk_unmask"
SHARD = "sincos"

Box = tuple[int, int, int, int]

DEDUPE_CONTAINMENT = 0.85
"""A record's box this far inside another's, with its text inside the other's
text, is the same read twice."""


def _pkg():
    """``anime_tools.ocr.{animetext, reread, sfx, _text}`` — the sibling checkout
    when the installed pin predates the detector."""
    try:
        from anime_tools.ocr import animetext  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO.parent / "anime_tools"))
        for k in [k for k in sys.modules if k.startswith("anime_tools")]:
            del sys.modules[k]
    from anime_tools.ocr import _text, animetext, reread, sfx

    return animetext, reread, sfx, _text


def _load_jsonl(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by[r["stem"]].append(r)
    return dict(by)


def _write_jsonl(path: Path, by: dict[str, list[dict]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for stem in sorted(by):
            for r in by[stem]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    return n


def load_hand(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return [
        {
            "stem": r.stem,
            "box": tuple(json.loads(r.box)),
            "kind": r.kind_hand or r.kind_rec,
            "text": r.text_hand,
        }
        for _, r in df.iterrows()
        if r.stem
    ]


def pages_of(opts) -> list[Path]:
    pages = sorted((REPO / "post_image_dataset/resized" / SHARD).glob("*.png"))
    return pages[: opts.limit] if opts.limit else pages


# --------------------------------------------------------------------------- det


def run_det(opts, pages: list[Path]) -> None:
    import cv2

    animetext, *_ = _pkg()
    path = OUT / "boxes_pkg.jsonl"
    if path.exists() and not opts.overwrite:
        print(f"{path.name}: exists, skip (--overwrite to redo)", flush=True)
        return
    det = animetext.AnimeTextDetector.load(
        device=opts.device, conf=opts.det_conf, nms=opts.nms, nest=opts.nest
    )
    t0, n = time.time(), 0
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pages, 1):
            bgr = cv2.imread(str(p))
            scored = det.detect_scored(bgr)
            n += len(scored)
            f.write(
                json.dumps(
                    {
                        "stem": p.stem,
                        "size": [bgr.shape[1], bgr.shape[0]],
                        "boxes": [list(map(int, b[:4])) for b in scored],
                        "scores": [round(b[4], 4) for b in scored],
                    }
                )
                + "\n"
            )
            if i % 50 == 0:
                print(f"  {i}/{len(pages)} pages, {n} boxes", flush=True)
    print(
        f"det: {len(pages)} pages, {n} boxes ≥ {opts.det_conf} ({opts.nest}), "
        f"{time.time() - t0:.1f}s on {opts.device}",
        flush=True,
    )


def load_boxes() -> dict[str, list[tuple[Box, float]]]:
    by = {}
    for line in (OUT / "boxes_pkg.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by[r["stem"]] = [
            (tuple(b), s) for b, s in zip(r["boxes"], r["scores"], strict=True)
        ]
    return by


def load_reads() -> dict[str, str | None]:
    reads: dict[str, str | None] = {}
    path = OUT / "reads.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            reads[f"{r['stem']}:{r['box']}"] = r["text"]
    return reads


def _key(stem: str, box: Box) -> str:
    return f"{stem}:{list(box)}"


# --------------------------------------------------------------------------- read


def run_read(opts, pages: list[Path]) -> None:
    import cv2

    _, _, sfx, _ = _pkg()
    reads = load_reads()
    boxes = load_boxes()
    todo = {
        p.stem: [b for b, _ in boxes.get(p.stem, []) if _key(p.stem, b) not in reads]
        for p in pages
    }
    todo = {s: b for s, b in todo.items() if b}
    n = sum(len(v) for v in todo.values())
    print(f"read: {n} boxes on {len(todo)} pages not in the cache", flush=True)
    if not n:
        return
    reader = sfx.SfxReader.load(device=opts.device, batch_size=opts.bs)
    with (OUT / "reads.jsonl").open("a", encoding="utf-8") as f:
        for p in pages:
            bs = todo.get(p.stem)
            if not bs:
                continue
            bgr = cv2.imread(str(p))
            for b, t in zip(bs, reader.read_boxes(bgr, bs), strict=True):
                f.write(
                    json.dumps(
                        {"stem": p.stem, "box": list(b), "text": t}, ensure_ascii=False
                    )
                    + "\n"
                )
            f.flush()
    print("read: done", flush=True)


# --------------------------------------------------------------------------- apply


def kind_for(stem: str, box: Box, text: str, hand_by_stem, rec, reread) -> tuple[str, str]:
    """``(kind, source)``: the covering hand label with the best IoU, else the rule."""
    best, arg = 0.0, None
    for h in hand_by_stem.get(stem, []):
        iou, cont = rec.overlap(box, h["box"])
        if (iou >= reread.COVERED_IOU or cont >= reread.COVERED_CONTAINMENT) and iou > best:
            best, arg = iou, h
    if arg is not None:
        return arg["kind"], "hand"
    return rec.record_kind(text), "rule"


def dedupe_page(records: list[dict], rec, stats: Counter) -> list[dict]:
    """Drop a record whose normalized text sits inside another's on the page
    while its box sits ≥ :data:`DEDUPE_CONTAINMENT` inside the other's box —
    a column read repeated by the block read that holds it. Largest boxes are
    the keepers; a dropped record never shields another."""
    order = sorted(
        records,
        key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]),
        reverse=True,
    )
    kept: list[dict] = []
    for r in order:
        nr = rec.norm(r["text"])
        dup = False
        for k in kept:
            _, cont = rec.overlap(tuple(r["box"]), tuple(k["box"]))
            inside = cont >= DEDUPE_CONTAINMENT and (
                (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])
                <= (k["box"][2] - k["box"][0]) * (k["box"][3] - k["box"][1])
            )
            if inside and nr and nr in rec.norm(k["text"]):
                dup = True
                stats["dedupe_dropped"] += 1
                break
        if not dup:
            kept.append(r)
    return kept


def build_records(
    opts, pages, *, reads, boxes, hand, rec, reread, engine: str, stats: Counter
) -> dict[str, list[dict]]:
    from anime_tools.captions.ocr_sidecar import OcrLine
    from anime_tools.ocr import reading_order

    hand_by_stem: dict[str, list[dict]] = defaultdict(list)
    for h in hand:
        hand_by_stem[h["stem"]].append(h)
    out: dict[str, list[dict]] = {}
    for p in pages:
        rs = []
        for b, score in boxes.get(p.stem, []):
            stats["boxes"] += 1
            t = reads.get(_key(p.stem, b))
            if t is None:
                stats["guard_rejected" if _key(p.stem, b) in reads else "unread"] += 1
                continue
            if len(rec.norm(t)) < opts.min_chars or not reread.has_script(t):
                stats["under_floor"] += 1
                continue
            kind, src = kind_for(p.stem, b, t, hand_by_stem, rec, reread)
            rs.append(
                {
                    "stem": p.stem,
                    "text": t,
                    "score": None,
                    "box": list(b),
                    "det_score": score,
                    "engine": engine,
                    "kind": kind,
                    "kind_src": src,
                    "sfx_guard": "ok",
                }
            )
        if opts.dedupe:
            rs = dedupe_page(rs, rec, stats)
        if not rs:
            continue
        lines = [
            OcrLine(seq=i, box=tuple(r["box"]), score=0.0, text=r["text"])
            for i, r in enumerate(rs)
        ]
        out[p.stem] = [rs[ln.seq] for ln in reading_order(lines)]
        for r in out[p.stem]:
            stats[f"kind_{r['kind']}"] += 1
            stats[f"kind_src_{r['kind_src']}"] += 1
    return out


def metrics(by_stem, *, rec, reread, hand, ref, ab_stems, masks_dir) -> dict:
    n_masked, floor, miss = rec.floor_count(masks_dir, by_stem)
    m_sim, hi, n_ref = rec.best_match_sim(ref, by_stem, ab_stems)
    ex = sims = n_h = 0
    for h in hand:
        if h["kind"] != "sfx":
            continue
        n_h += 1
        best, arg = 0.0, None
        for r in by_stem.get(h["stem"], []):
            iou, _ = rec.overlap(h["box"], tuple(r["box"]))
            if iou > best:
                best, arg = iou, r
        if arg is None:
            continue
        s = rec.sim(h["text"], arg["text"])
        sims += s
        ex += rec.norm(h["text"]) == rec.norm(arg["text"])
    return {
        "records": sum(len(v) for v in by_stem.values()),
        "pages_with_line": len(by_stem),
        "floor": floor,
        "floor_pages": miss,
        "n_masked": n_masked,
        "best_match_sim": round(m_sim, 3),
        "best_match_ge_0.9": hi,
        "n_ref": n_ref,
        "hand_sfx_exact": ex,
        "hand_sfx_mean_sim": round(sims / max(n_h, 1), 3),
        "hand_sfx_n": n_h,
    }


def joined_view(by_stem, join_cjk, reading_order, OcrLine) -> tuple[dict, int]:
    """The records as the OCR stage's sidecar would carry them (``join_cjk``
    on): a ``{stem: [{"text", "box"}]}`` view and how many joins happened."""
    out, merges = {}, 0
    for stem, rs in by_stem.items():
        lines = [
            OcrLine(seq=i, box=tuple(r["box"]), score=0.0, text=r["text"])
            for i, r in enumerate(rs)
        ]
        joined = join_cjk(lines)
        merges += len(lines) - len(joined)
        out[stem] = [
            {"text": ln.text, "box": list(ln.box)} for ln in reading_order(joined)
        ]
    return out, merges


def _row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['records']} / {m['pages_with_line']} | **{m['floor']}** | "
        f"{m['best_match_sim']} / {m['best_match_ge_0.9']} | "
        f"{m['hand_sfx_exact']} / {m['hand_sfx_mean_sim']} |"
    )


def run_apply(opts, pages: list[Path]) -> None:
    animetext, reread, sfx, _text = _pkg()
    from anime_tools.captions.ocr_sidecar import OcrLine
    from anime_tools.ocr import reading_order

    rec = m109.pilot_records()
    hand = load_hand(ASSETS / "sfx_labels_sincos.tsv")
    ref = _load_jsonl(RECORDS_DIR / f"ocr_records_{SHARD}.jsonl")
    ab_stems = [
        json.loads(r)["stem"]
        for r in (REPO / "output/tests/vl16_ab/ab.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    masks_dir = REPO / "post_image_dataset/masks" / SHARD
    hyb = _load_jsonl(RECORDS_DIR / f"ocr_records_{SHARD}_hybrid_vl.jsonl")
    reads, boxes = load_reads(), load_boxes()
    kw = dict(rec=rec, reread=reread, hand=hand, ref=ref, ab_stems=ab_stems, masks_dir=masks_dir)

    stats: Counter = Counter()
    recs = build_records(
        opts, pages, reads=reads, boxes=boxes, hand=hand, rec=rec, reread=reread,
        engine="animetext+sfx_reader", stats=stats,
    )
    if stats["unread"]:
        print(f"WARNING: {stats['unread']} boxes have no cached read — run --stage read first")
    out_path = opts.out or RECORDS_DIR / f"ocr_records_{SHARD}_animetext.jsonl"
    n = _write_jsonl(out_path, recs)

    raw_stats: Counter = Counter()
    opts_raw = argparse.Namespace(**{**vars(opts), "dedupe": False})
    raw = build_records(
        opts_raw, pages, reads=reads, boxes=boxes, hand=hand, rec=rec, reread=reread,
        engine="animetext+sfx_reader", stats=raw_stats,
    )
    m_hyb = metrics(hyb, **kw)
    m_raw = metrics(raw, **kw)
    m_dd = metrics(recs, **kw)
    jview, merges = joined_view(recs, _text.join_cjk, reading_order, OcrLine)
    m_join = metrics(jview, **kw)

    md = [
        f"# AnimeText records — {SHARD} (plan_det D1, {time.strftime('%Y-%m-%d %H:%M')})\n",
        f"Detector: `anime_tools.ocr.animetext` yolo12l @ 640, conf {opts.det_conf}, NMS {opts.nms}, "
        f"nest `{opts.nest}` → {stats['boxes']} boxes on {len(pages)} pages. Reader: the SFX reader "
        f"(cached reads, guard rejected {stats['guard_rejected']}, under floor {stats['under_floor']}). "
        f"Dedupe dropped {stats['dedupe_dropped']} nested reads. Output `{out_path.relative_to(REPO)}` ({n} records).\n",
        "| records | lines / pages | floor | best-match / ≥0.9 | hand-SFX exact / mean sim |",
        "|---|---|---|---|---|",
        _row("hybrid_vl (3-layer stack, O4)", m_hyb),
        _row("O6 probe, yolo12l 640 inner c0.25 (findings)", {
            "records": 1001, "pages_with_line": 163, "floor": 4, "best_match_sim": 0.844,
            "best_match_ge_0.9": 45, "hand_sfx_exact": 63, "hand_sfx_mean_sim": 0.860}),
        _row("animetext, no dedupe", m_raw),
        _row("**animetext, dedupe** (the file)", m_dd),
        _row(f"animetext, dedupe + join_cjk ({merges} joins)", m_join),
        "",
        f"- gate (plan_det D1): floor ≤ 4 → {'PASS' if m_dd['floor'] <= 4 else 'FAIL'} ({m_dd['floor']}); "
        f"best-match ≥ 0.84 → {'PASS' if m_dd['best_match_sim'] >= 0.84 else 'FAIL'} ({m_dd['best_match_sim']}); "
        f"hand-SFX exact ≥ 64 → {'PASS' if m_dd['hand_sfx_exact'] >= 64 else 'FAIL'} ({m_dd['hand_sfx_exact']} / {m_dd['hand_sfx_n']})",
        f"- kind: speech {stats['kind_speech']} · sfx {stats['kind_sfx']} · chrome {stats['kind_chrome']} "
        f"(hand label by overlap {stats['kind_src_hand']}, rule {stats['kind_src_rule']})",
        f"- floor pages still empty: {' '.join(m_dd['floor_pages']) or '—'}",
        f"- hand-SFX reference: {m_dd['hand_sfx_n']} rows; manga-ocr reference: {m_dd['n_ref']} lines on {len(ab_stems)} A/B pages",
        "",
        "## Dedupe drops\n",
        "| stem | dropped (box) | kept by (box) |",
        "|---|---|---|",
    ]
    for stem in sorted(raw):
        kept_keys = {tuple(r["box"]) for r in recs.get(stem, [])}
        for r in raw[stem]:
            if tuple(r["box"]) in kept_keys:
                continue
            keeper = next(
                (k for k in recs.get(stem, [])
                 if rec.overlap(tuple(r["box"]), tuple(k["box"]))[1] >= DEDUPE_CONTAINMENT
                 and rec.norm(r["text"]) in rec.norm(k["text"])),
                None,
            )
            md.append(
                f"| {stem} | {r['text']} {r['box']} | "
                f"{(keeper['text'] + ' ' + str(keeper['box'])) if keeper else '?'} |"
            )
    report = OUT / "records_report.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[: md.index("## Dedupe drops\n")]))
    print(f"wrote {out_path} ({n}) and {report}")
    (OUT / "records_metrics.json").write_text(
        json.dumps(
            {"hybrid_vl": m_hyb, "animetext_raw": m_raw, "animetext": m_dd,
             "animetext_join": m_join, "joins": merges, "stats": dict(stats)},
            ensure_ascii=False, indent=1,
        )
    )


# --------------------------------------------------------------------------- side


def run_side(opts, pages: list[Path]) -> None:
    """PP-OCRv6's recognizer on the AnimeText boxes (decision 2, once)."""
    import cv2

    animetext, reread, _, _text = _pkg()
    from anime_tools.ocr._onnx import TextRecognizer, crop_quad

    rec = m109.pilot_records()
    hand = load_hand(ASSETS / "sfx_labels_sincos.tsv")
    ref = _load_jsonl(RECORDS_DIR / f"ocr_records_{SHARD}.jsonl")
    ab_stems = [
        json.loads(r)["stem"]
        for r in (REPO / "output/tests/vl16_ab/ab.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    masks_dir = REPO / "post_image_dataset/masks" / SHARD
    boxes = load_boxes()
    t0 = time.time()
    recog = TextRecognizer.load(device=opts.device, batch_size=16)
    print(f"PP-OCRv6 recognizer loaded {time.time() - t0:.1f}s", flush=True)
    reads: dict[str, str | None] = {}
    t0, n = time.time(), 0
    for p in pages:
        bs = boxes.get(p.stem, [])
        if not bs:
            continue
        bgr = cv2.imread(str(p))
        crops, keys = [], []
        for b, _ in bs:
            crop = crop_quad(bgr, animetext.as_quad(b))
            if crop is None or not crop.size:
                continue
            crops.append(crop)
            keys.append(_key(p.stem, b))
        for k, (text, score) in zip(keys, recog.recognize(crops), strict=True):
            reads[k] = text.strip() if text.strip() and score >= opts.min_score else None
        n += len(crops)
    print(f"side: {n} crops recognized in {time.time() - t0:.1f}s", flush=True)
    stats: Counter = Counter()
    recs = build_records(
        opts, pages, reads=reads, boxes=boxes, hand=hand, rec=rec, reread=reread,
        engine="animetext+ppocr_v6", stats=stats,
    )
    path = OUT / "records_animetext_ppocr.jsonl"
    _write_jsonl(path, recs)
    m = metrics(recs, rec=rec, reread=reread, hand=hand, ref=ref, ab_stems=ab_stems, masks_dir=masks_dir)
    row = _row(f"animetext + PP-OCRv6 rec (side row, score ≥ {opts.min_score})", m)
    print(row)
    report = OUT / "records_report.md"
    if report.exists():
        s = report.read_text(encoding="utf-8")
        if "PP-OCRv6 rec (side row" not in s:
            s = s.replace("\n\n- gate (plan_det D1)", f"\n{row}\n\n- gate (plan_det D1)", 1)
            report.write_text(s, encoding="utf-8")
    (OUT / "records_metrics_side.json").write_text(json.dumps({"animetext_ppocr": m, "stats": dict(stats)}, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------- hand pass


def run_hand(opts, pages: list[Path]) -> None:
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    _, reread, sfx, _ = _pkg()
    rec = m109.pilot_records()
    recs = _load_jsonl(opts.out or RECORDS_DIR / f"ocr_records_{SHARD}_animetext.jsonl")
    hyb = _load_jsonl(RECORDS_DIR / f"ocr_records_{SHARD}_hybrid_vl.jsonl")
    pp = _load_jsonl(RECORDS_DIR / f"ocr_records_{SHARD}_ppocr_v3.jsonl")
    new = []
    for stem, rs in recs.items():
        known = [tuple(r["box"]) for r in hyb.get(stem, [])] + [tuple(r["box"]) for r in pp.get(stem, [])]
        for r in rs:
            if not reread.covered(tuple(r["box"]), known):
                new.append(r)
    rng = np.random.default_rng(opts.seed)
    pick = [new[i] for i in sorted(rng.choice(len(new), min(opts.n_hand, len(new)), replace=False))]
    print(f"hand: {len(new)} new records (no hybrid_vl / PP v3 box covers them); drew {len(pick)}")

    tsv = ASSETS / f"animetext_new_lines_{SHARD}.tsv"
    if tsv.exists() and not opts.overwrite:
        print(f"{tsv} exists — keeping it (--overwrite to redraw; hand labels would be lost)")
    else:
        lines = ["row\tstem\tbox\tdet_score\tkind\ttext\treal_text\tnote"]
        for i, r in enumerate(pick):
            lines.append(
                f"{i}\t{r['stem']}\t{json.dumps(r['box'])}\t{r['det_score']}\t{r['kind']}\t{r['text']}\t\t"
            )
        tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {tsv}")

    # crop sheets: 4 × 5 tiles per sheet, crop (pad 0.12) fitted into 280×200 + caption
    d = OUT / "hand_pass"
    d.mkdir(parents=True, exist_ok=True)
    import subprocess

    fpath = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK JP"], capture_output=True, text=True).stdout.strip()
    font = ImageFont.truetype(fpath, 20)
    small = ImageFont.truetype(fpath, 15)
    TW, TH, CAP = 300, 210, 70
    cols, rows_per = 4, 5
    per = cols * rows_per
    cache: dict[str, np.ndarray] = {}
    for s in range(0, len(pick), per):
        sheet = Image.new("RGB", (cols * TW, rows_per * (TH + CAP)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(pick[s : s + per]):
            i = s + j
            bgr = cache.get(r["stem"])
            if bgr is None:
                bgr = cache[r["stem"]] = cv2.imread(str(REPO / "post_image_dataset/resized" / SHARD / f"{r['stem']}.png"))
            crop = sfx.crop_box(bgr, r["box"], 0.12)
            im = Image.fromarray(crop[:, :, ::-1])
            sc = min((TW - 10) / im.width, (TH - 10) / im.height)
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
            x0, y0 = (j % cols) * TW, (j // cols) * (TH + CAP)
            sheet.paste(im, (x0 + (TW - im.width) // 2, y0 + (TH - im.height) // 2))
            draw.rectangle([x0, y0, x0 + TW - 1, y0 + TH + CAP - 1], outline=(200, 200, 200))
            draw.text((x0 + 6, y0 + TH + 2), f"#{i}  {r['kind']}  det {r['det_score']:.2f}", fill=(90, 90, 90), font=small)
            draw.text((x0 + 6, y0 + TH + 22), r["text"][:22], fill=(0, 0, 0), font=font)
            draw.text((x0 + 6, y0 + TH + 48), f"{r['stem']} {r['box']}", fill=(140, 140, 140), font=small)
        p = d / f"sheet_{s // per:02d}.png"
        sheet.save(p)
        print(f"  {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=["det", "read", "apply", "side", "hand"], default=["det", "apply"])
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--nest", default="inner", choices=["inner", "outer", "raw"])
    ap.add_argument("--min_chars", type=int, default=2, help="records: norm(text) length floor")
    ap.add_argument("--min_score", type=float, default=0.6, help="side row: PP-OCRv6 score floor")
    ap.add_argument("--no_dedupe", dest="dedupe", action="store_false")
    ap.add_argument("--out", type=Path, default=None, help="records file (default: cjk_unmask/ocr_records_sincos_animetext.jsonl)")
    ap.add_argument("--n_hand", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="det/side: onnxruntime device; read: torch device")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    opts = ap.parse_args()
    if opts.out is not None and not opts.out.is_absolute():
        opts.out = REPO / opts.out
    pages = pages_of(opts)
    print(f"{len(pages)} pages; stages {opts.stage}", flush=True)
    if "det" in opts.stage:
        run_det(opts, pages)
    if "read" in opts.stage:
        run_read(opts, pages)
    if "apply" in opts.stage:
        run_apply(opts, pages)
    if "side" in opts.stage:
        run_side(opts, pages)
    if "hand" in opts.stage:
        run_hand(opts, pages)


if __name__ == "__main__":
    main()
