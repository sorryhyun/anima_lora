#!/usr/bin/env python3
"""O4 (``plan_ocr.md``): the SFX reader wired into the hybrid OCR records.

**Superseded 2026-09-06 (plan_det D3).** The three-layer stack this re-reads
(PP-OCRv6 DB → VL Spotting boxes → MIT-mask components) is retired; the
records default is now ``ocr_records_<shard>_animetext.jsonl`` from
``animetext_records.py`` (one detector, every box read by the SFX reader).
Kept as-is so the C10/C11 ``_hybrid_vl`` files stay reproducible.

The third reader — ``anime_tools.ocr.sfx.SfxReader``, the B′ weights (VL-1.6
LoRA + fine-tuned tower) with the decode guard built in — re-reads the hybrid
records and reads what the MIT text mask boxed that no detector did.

    # GPU (daemon): read every crop once, cache the raw + guarded reads
    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima_dit/ocr/reread_records.py --stage read"
    # CPU: apply the reads, re-derive kind / reading order, the floor numbers
    python project/cjk_aware_anima_dit/ocr/reread_records.py --stage apply              # --reread all → …_hybrid_vl.jsonl (default)
    python project/cjk_aware_anima_dit/ocr/reread_records.py --stage apply --reread sfx  # arm C11's SFX-only file → …_hybrid_sfx.jsonl

What the read stage crops (all of it goes through the reader once; ``apply``
picks):

* **every record** of the input file (``--reread`` decides at apply time
  whether only ``kind: sfx`` records or all of them take the new read — the
  user's "just run all of OCR through VL" is the ``all`` arm, measured against
  ``sfx`` on the hand labels and **the default since 2026-09-06**: manga-ocr
  best-match 0.810 vs 0.786, hearts restored, and the user's eyeball of the
  184 changed lines — no C arm, the user's call);
* **mask components**: on every masked page (``--mask_pages all``; ``floor`` =
  only the masked-but-no-line pages the plan names) the MIT text-pixel mask's
  connected components after a closing pass — bbox min side ≥
  ``--comp_min_side``, not already covered by a record, at most ``--comp_max``
  per page by area — become crops too. A read that passes the guard and the
  line floors becomes a record with ``engine: sfx_reader``.

``kind`` comes from the hand labels (``--kind_labels``, B1's file: a record
whose ``(stem, box)`` is labelled takes ``kind_hand``) and from the v1 text
rule + chrome list for everything else (mask-component records included).

Records keep the audit trail: ``prev_text`` (what the record said before),
``sfx_raw`` (the unguarded decode), ``sfx_guard`` = ``ok | rejected``; a
rejected read leaves the text alone. Output default
``ocr_records_<shard>_hybrid_vl.jsonl`` (``_hybrid_sfx`` under ``--reread
sfx``) + a report under ``output/tests/ocr_hybrid/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

REPO = m109.REPO
LINE = m109.LINE
ASSETS = m109.ASSETS

CLOSE_FRAC = 0.025
"""Closing kernel as a fraction of the page width: merges the glyphs of one
column (and usually the columns of one block) into one component."""
FULLPAGE_FRAC = 0.4


def _pkg_sfx():
    """``anime_tools.ocr.sfx`` — from the installed package, else the sibling
    checkout (dev loop before the pinned rev carries it)."""
    try:
        from anime_tools.ocr import sfx
    except ImportError:
        sys.path.insert(0, str(REPO.parent / "anime_tools"))
        for k in [k for k in sys.modules if k.startswith("anime_tools")]:
            del sys.modules[k]
        from anime_tools.ocr import sfx
    return sfx


# --------------------------------------------------------------------------- io


def load_records(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["stem"]].append(r)
    return dict(by)


def load_kind_labels(path: Path) -> dict[tuple[str, tuple[int, ...]], str]:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for _, r in df.iterrows():
        if r.kind_hand:
            out[(r.stem, tuple(json.loads(r.box)))] = r.kind_hand
    return out


# --------------------------------------------------------------------------- mask components


def mask_components(
    mask: np.ndarray, *, min_side: int, close_frac: float = CLOSE_FRAC
) -> list[tuple[int, int, int, int]]:
    """Axis-aligned boxes of the text-pixel components of an ignore mask
    (``0`` = text, ``255`` = trained on), largest first."""
    import cv2

    text = (mask == 0).astype(np.uint8)
    k = max(3, int(round(close_frac * mask.shape[1])) | 1)
    closed = cv2.morphologyEx(
        text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    page_area = mask.shape[0] * mask.shape[1]
    boxes = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if min(w, h) < min_side or w * h >= FULLPAGE_FRAC * page_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def covered(box, records: list[dict], rec) -> bool:
    """A component a record already boxes (containment ≥ 0.5 either way, or
    IoU ≥ 0.3) needs no second crop."""
    for r in records:
        iou, cont = rec.overlap(tuple(box), tuple(r["box"]))
        if iou >= 0.3 or cont >= 0.5:
            return True
    return False


# --------------------------------------------------------------------------- read stage


def run_read_stage(
    opts, pages: list[Path], by_stem: dict[str, list[dict]], raw_path: Path
) -> None:
    import cv2

    rec = m109.pilot_records()
    sfx = _pkg_sfx()
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
    masks_dir = REPO / "post_image_dataset/masks" / opts.shard
    masked = {p.name[: -len("_mask.png")] for p in masks_dir.glob("*_mask.png")}

    todo = [p for p in pages if p.stem not in done]
    print(f"read stage: {len(todo)} pages to do ({len(done)} cached)", flush=True)
    if not todo:
        return
    t0 = time.time()
    reader = sfx.SfxReader.load(
        device=opts.device,
        base_dir=REPO / "models/paddleocr_vl_1.6",
        adapter_dir=Path(opts.sfx_reader) if opts.sfx_reader else None,
        batch_size=opts.bs,
    )
    print(f"reader loaded {time.time() - t0:.1f}s", flush=True)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    out = raw_path.open("a", encoding="utf-8")
    n_crops = n_comp = 0
    t0 = time.time()
    for k in range(0, len(todo), opts.pages_per_batch):
        chunk = todo[k : k + opts.pages_per_batch]
        crops: list = []
        rows = []
        for p in chunk:
            bgr = cv2.imread(str(p))
            H, W = bgr.shape[:2]
            recs = by_stem.get(p.stem, [])
            items = [
                {"src": f"rec:{i}", "box": list(r["box"])} for i, r in enumerate(recs)
            ]
            want_mask = p.stem in masked and (
                opts.mask_pages == "all" or (opts.mask_pages == "floor" and not recs)
            )
            if want_mask:
                mask = cv2.imread(str(masks_dir / f"{p.stem}_mask.png"), 0)
                if mask.shape != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                comps = [
                    b
                    for b in mask_components(mask, min_side=opts.comp_min_side)
                    if not covered(b, recs, rec)
                ][: opts.comp_max]
                items += [{"src": "mask", "box": list(b)} for b in comps]
                n_comp += len(comps)
            for it in items:
                crop = sfx.crop_box(bgr, it["box"], opts.pad)
                if crop is None or not crop.size:
                    it["raw"], it["text"] = "", None
                    continue
                it["crop_wh"] = [int(crop.shape[1]), int(crop.shape[0])]
                crops.append(crop)
            rows.append({"stem": p.stem, "size": [W, H], "items": items})
        raws = reader.read_raw(crops)
        # scatter back in crop order
        cursor = 0
        for row in rows:
            for it in row["items"]:
                if "crop_wh" not in it:
                    continue
                raw = raws[cursor]
                cursor += 1
                it["raw"] = raw
                it["text"] = sfx.guard(raw, it["crop_wh"][0], it["crop_wh"][1])
        assert cursor == len(raws)
        n_crops += len(crops)
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        el = time.time() - t0
        print(
            f"[{k + len(chunk)}/{len(todo)}] crops {n_crops} (mask comps {n_comp}) "
            f"{n_crops / max(el, 1e-9):.1f} crops/s",
            flush=True,
        )
    out.close()


# --------------------------------------------------------------------------- apply stage


def apply_page(
    stem: str,
    recs: list[dict],
    row: dict | None,
    *,
    reread: str,
    kind_labels: dict,
    rec,
    sfx,
    min_chars: int,
    stats: dict,
) -> list[dict]:
    from anime_tools.captions.ocr_sidecar import OcrLine
    from anime_tools.ocr import reading_order
    from anime_tools.ocr._text import keep_line

    out: list[dict] = []
    items = row["items"] if row else []
    # the guard runs here, on the cached raw decode, so a guard change never
    # costs a GPU pass
    for it in items:
        if "crop_wh" in it:
            it["text"] = sfx.guard(it.get("raw", ""), *it["crop_wh"])
    by_src = {it["src"]: it for it in items}
    for i, r in enumerate(recs):
        r = dict(r)
        r.pop("post", None)
        key = (stem, tuple(r["box"]))
        if key in kind_labels:
            r["kind"] = kind_labels[key]
            r["kind_src"] = "hand"
        else:
            r["kind"] = rec.record_kind(r["text"])
            r["kind_src"] = "rule"
        it = by_src.get(f"rec:{i}")
        take = reread == "all" or (reread == "sfx" and r["kind"] == "sfx")
        if it is not None and take:
            r["sfx_raw"] = it.get("raw", "")
            stats[f"reread_{r['kind']}"] += 1
            if it.get("text"):
                if it["text"] != r["text"]:
                    r["prev_text"] = r["text"]
                    r["text"] = it["text"]
                    r["engine"] = r["engine"] + "+sfx_reader"
                    stats["replaced"] += 1
                r["sfx_guard"] = "ok"
            else:
                r["sfx_guard"] = "rejected"
                stats["guard_rejected"] += 1
        out.append(r)
    seen = [tuple(r["box"]) for r in out]
    for it in items:
        if it["src"] != "mask":
            continue
        stats["mask_comps"] += 1
        text = it.get("text")
        if not text:
            stats["mask_rejected"] += 1
            continue
        if not keep_line(text, min_chars=min_chars, skip_en=True):
            stats["mask_floor"] += 1
            continue
        if rec.is_symbol_only(text):
            stats["mask_symbol_only"] += 1
            continue
        if any(rec.overlap(tuple(it["box"]), b)[1] >= 0.7 for b in seen):
            stats["mask_dup"] += 1
            continue
        seen.append(tuple(it["box"]))
        out.append(
            {
                "stem": stem,
                "text": text,
                "score": None,
                "box": list(it["box"]),
                "engine": "sfx_reader",
                "kind": rec.record_kind(text),
                "kind_src": "rule",
                "sfx_raw": it.get("raw", ""),
                "sfx_guard": "ok",
            }
        )
        stats["mask_added"] += 1
    for r in out:
        stats[f"kind_{r['kind']}"] += 1
    lines = [
        OcrLine(seq=i, box=tuple(r["box"]), score=r.get("score") or 0.0, text=r["text"])
        for i, r in enumerate(out)
    ]
    order = [ln.seq for ln in reading_order(lines)]
    return [{k: v for k, v in out[i].items() if v is not None} for i in order]


def run_apply_stage(opts, pages, by_stem, raw_path: Path, out_path: Path, report: Path):
    rec = m109.pilot_records()
    sfx = _pkg_sfx()
    raw = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            raw[r["stem"]] = r
    kind_labels = load_kind_labels(opts.kind_labels) if opts.kind_labels else {}
    stats: dict = defaultdict(int)
    new: dict[str, list[dict]] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            recs = apply_page(
                p.stem,
                by_stem.get(p.stem, []),
                raw.get(p.stem),
                reread=opts.reread,
                kind_labels=kind_labels,
                rec=rec,
                sfx=sfx,
                min_chars=opts.min_chars,
                stats=stats,
            )
            if recs:
                new[p.stem] = recs
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    masks_dir = REPO / "post_image_dataset/masks" / opts.shard
    n_masked, floor_in, miss_in = rec.floor_count(masks_dir, by_stem)
    _, floor_out, miss_out = rec.floor_count(masks_dir, new)
    recovered = sorted(set(miss_in) - set(miss_out))
    n_in = sum(len(v) for v in by_stem.values())
    n_out = sum(len(v) for v in new.values())
    md = [
        f"# Hybrid + SFX reader records — {opts.shard} (O4, reread={opts.reread}, "
        f"mask_pages={opts.mask_pages}, {time.strftime('%Y-%m-%d %H:%M')})\n",
        f"in: `{opts.records_in.relative_to(REPO)}` → out: `{out_path.relative_to(REPO)}`; "
        f"raw reads: `{raw_path.relative_to(REPO)}`\n",
        "| | hybrid (in) | + SFX reader |",
        "|---|---|---|",
        f"| pages ({len(pages)} in shard) with any line | {len(by_stem)} | {len(new)} |",
        f"| lines | {n_in} | {n_out} |",
        f"| **masked-but-no-line floor** ({n_masked} masked) | **{floor_in}** | **{floor_out}** |",
        "",
        f"- re-read records: sfx {stats.get('reread_sfx', 0)} · speech {stats.get('reread_speech', 0)} · "
        f"chrome {stats.get('reread_chrome', 0)} → text replaced {stats.get('replaced', 0)}, "
        f"guard rejected {stats.get('guard_rejected', 0)}",
        f"- mask components cropped {stats.get('mask_comps', 0)} → added {stats.get('mask_added', 0)} "
        f"(guard {stats.get('mask_rejected', 0)}, floor {stats.get('mask_floor', 0)}, "
        f"symbol-only {stats.get('mask_symbol_only', 0)}, duplicate {stats.get('mask_dup', 0)})",
        f"- kind: speech {stats.get('kind_speech', 0)} · sfx {stats.get('kind_sfx', 0)} · "
        f"chrome {stats.get('kind_chrome', 0)} "
        f"({'hand labels for matched records, ' if kind_labels else ''}v1 rule elsewhere)",
        f"- floor pages recovered ({len(recovered)}): {' '.join(recovered) or '—'}",
        f"- floor pages still empty ({len(miss_out)}): {' '.join(miss_out) or '—'}",
        "",
        "## Mask-component lines (engine=sfx_reader)\n",
        "| stem | kind | text | box |",
        "|---|---|---|---|",
    ]
    for stem, recs in new.items():
        for r in recs:
            if r["engine"] == "sfx_reader":
                md.append(f"| {stem} | {r['kind']} | {r['text']} | {r['box']} |")
    md += [
        "",
        "## Re-read replacements\n",
        "| stem | kind | before | SFX reader | engine |",
        "|---|---|---|---|---|",
    ]
    for stem, recs in new.items():
        for r in recs:
            if "prev_text" in r:
                md.append(
                    f"| {stem} | {r['kind']} | {r['prev_text']} | {r['text']} | {r['engine']} |"
                )
    md += [
        "",
        "## Guard-rejected reads\n",
        "| stem | kind | kept text | raw decode |",
        "|---|---|---|---|",
    ]
    for stem, recs in new.items():
        for r in recs:
            if r.get("sfx_guard") == "rejected":
                md.append(
                    f"| {stem} | {r['kind']} | {r['text']} | {r.get('sfx_raw', '')[:40].replace('|', '\\|')} |"
                )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:14]))
    print(f"\nwrote {out_path} ({n_out} lines) and {report}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shard", default="sincos")
    ap.add_argument("--stage", choices=["read", "apply", "all"], default="all")
    ap.add_argument("--records_in", type=Path, default=None)
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--sfx_reader", default=None, help="adapter dir (default: catalog)")
    ap.add_argument(
        "--reread",
        choices=["sfx", "all", "none"],
        default="all",
        help="all (default since 2026-09-06, user's call): every record takes the VL read; "
        "sfx: only kind=sfx records (arm C11's file); none: mask components only",
    )
    ap.add_argument("--mask_pages", choices=["all", "floor", "none"], default="all")
    ap.add_argument("--comp_min_side", type=int, default=32)
    ap.add_argument("--comp_max", type=int, default=16)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--min_chars", type=int, default=2)
    ap.add_argument(
        "--kind_labels",
        type=Path,
        default=ASSETS / "sfx_labels_sincos.tsv",
        help="hand labels (stem, box → kind_hand); '' to use the rule only",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--pages_per_batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    opts = ap.parse_args()
    if opts.kind_labels is not None and str(opts.kind_labels) in ("", "."):
        opts.kind_labels = None
    for name in ("records_in", "raw", "out", "report"):  # relative = under the repo
        v = getattr(opts, name)
        if v is not None and not v.is_absolute():
            setattr(opts, name, REPO / v)

    base = REPO / "post_image_dataset/cjk_unmask"
    opts.records_in = opts.records_in or base / f"ocr_records_{opts.shard}_hybrid.jsonl"
    raw_path = opts.raw or base / f"ocr_raw_sfx_{opts.shard}.jsonl"
    out_path = opts.out or base / (
        f"ocr_records_{opts.shard}_hybrid_vl.jsonl"
        if opts.reread == "all"
        else f"ocr_records_{opts.shard}_hybrid_sfx.jsonl"
    )
    report = (
        opts.report
        or REPO
        / "output/tests/ocr_hybrid"
        / f"{opts.shard}_sfx_{opts.reread}_report.md"
    )
    pages = sorted((REPO / "post_image_dataset/resized" / opts.shard).glob("*.png"))
    if opts.limit:
        pages = pages[: opts.limit]
    by_stem = load_records(opts.records_in)
    print(
        f"{len(pages)} pages, {sum(len(v) for v in by_stem.values())} records on {len(by_stem)} stems"
    )
    if opts.stage in ("read", "all"):
        run_read_stage(opts, pages, by_stem, raw_path)
    if opts.stage in ("apply", "all"):
        run_apply_stage(opts, pages, by_stem, raw_path, out_path, report)


if __name__ == "__main__":
    main()
