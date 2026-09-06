#!/usr/bin/env python3
"""Re-base the sincos hand labels onto the AnimeText boxes (one-shot, ran 2026-09-06).

The pre-2026-09-06 ``assets/sfx_labels_sincos.tsv`` carried PP-OCRv6 /
VL-spotting boxes, loose around hand lettering; it and the D1 hand-pass file
``animetext_new_lines_sincos.tsv`` were retired the same day (recover either
with ``git show ee01658f:project/cjk_aware_anima_dit/assets/<name>``), and the
file this script wrote **is** today's ``sfx_labels_sincos.tsv``. The D1 records
(``post_image_dataset/cjk_unmask/ocr_records_sincos_animetext.jsonl``) sit on
the AnimeText detector's boxes. This builds a draft label file on those boxes:

* every AnimeText record becomes a row; the old hand label that covers it
  (IoU ≥ 0.3 or the record ≥ 0.5 inside it, best IoU wins — the D1 ``apply``
  rule) donates ``kind_hand`` / ``text_hand``. A 1:1 match at IoU ≥ 0.5 is
  ``checked`` (same lettering, tighter box); a 1:1 match below that, or a hand
  box the detector split into several records, is ``draft`` with the old row
  and its text in ``note`` (the text may need re-splitting).
* a record no hand label covers is ``draft`` with the reader's read as
  ``text_hand`` and the rule's ``kind`` as ``kind_hand``; rows the D1 hand
  pass (``animetext_new_lines_sincos.tsv``, ``real_text = y``) already saw are
  noted.
* an old hand row no record covers is appended on its old box with
  ``engine = hand_only`` and ``status = unchecked`` (dropped by the eval unless
  ``--include_unchecked``) so nothing is lost.

    git show ee01658f:project/cjk_aware_anima_dit/assets/sfx_labels_sincos.tsv > /tmp/old_labels.tsv
    python project/cjk_aware_anima_dit/ocr/relabel_animetext.py --old /tmp/old_labels.tsv --sheets --overwrite

The output has the eval's schema (``stem, box, kind_hand, text_hand, status``
+ ``engine, kind_rec, text_rec, det_score, src_row, note``) so
``eval_sfx.py --labels <tsv>`` and ``ocr_eval_sheet.py --labels <tsv>`` read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402

REPO = m109.REPO
ASSETS = m109.ASSETS
SHARD = "sincos"
NEW_LINES = ASSETS / f"animetext_new_lines_{SHARD}.tsv"  # retired; optional if restored
RECORDS = (
    REPO / "post_image_dataset/cjk_unmask" / f"ocr_records_{SHARD}_animetext.jsonl"
)
SHEETS = REPO / "output/tests/ocr_animetext/relabel_sheets"
MATCH_IOU, MATCH_CONTAIN, CHECKED_IOU = 0.3, 0.5, 0.5

Box = tuple[int, int, int, int]


def _area(b: Box) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    i = _inter(a, b)
    return i / (_area(a) + _area(b) - i + 1e-9)


def contained(a: Box, b: Box) -> float:
    """Fraction of ``a`` inside ``b``."""
    return _inter(a, b) / (_area(a) + 1e-9)


def match(rec_box: Box, hands: list[dict]) -> dict | None:
    best, best_iou = None, -1.0
    for h in hands:
        v = iou(rec_box, h["box"])
        if v >= MATCH_IOU or contained(rec_box, h["box"]) >= MATCH_CONTAIN:
            if v > best_iou:
                best, best_iou = h, v
    return best


def build(
    records: list[dict], old: list[dict], new_lines: dict[tuple, dict]
) -> tuple[list[dict], Counter]:
    by_page: dict[str, list[dict]] = defaultdict(list)
    for h in old:
        by_page[h["stem"]].append(h)
    # first pass: which hand row each record maps to, so 1:many splits are known
    owner = [match(tuple(r["box"]), by_page.get(r["stem"], [])) for r in records]
    fan = Counter(id(h) for h in owner if h is not None)
    stats: Counter = Counter()
    rows: list[dict] = []
    for r, h in zip(records, owner, strict=True):
        box = tuple(r["box"])
        row = {
            "stem": r["stem"],
            "box": json.dumps(list(box)),
            "det_score": r.get("det_score", ""),
            "engine": r["engine"],
            "kind_rec": r["kind"],
            "text_rec": r["text"],
            "src_row": "",
            "note": "",
        }
        if h is None:
            row.update(kind_hand=r["kind"], text_hand=r["text"], status="draft")
            nl = new_lines.get((r["stem"], box))
            if nl is not None:
                row["note"] = (
                    f"D1 hand pass #{nl['row']} real_text={nl['real_text'] or '?'}"
                    + (f"; {nl['note']}" if nl["note"] else "")
                )
                stats["new: seen in D1 hand pass"] += 1
            else:
                stats["new: unlabelled"] += 1
        else:
            v = iou(box, h["box"])
            row["src_row"] = h["row"]
            row["kind_hand"] = h["kind"]
            if fan[id(h)] == 1:
                row["text_hand"] = h["text"]
                if v >= CHECKED_IOU:
                    row["status"] = "checked"
                    stats["transferred 1:1 checked"] += 1
                else:
                    row["status"] = "draft"
                    row["note"] = f"box moved iou={v:.2f} from old #{h['row']}"
                    stats["transferred 1:1 draft (iou<0.5)"] += 1
                if ev.exact_key(h["text"]) != ev.exact_key(r["text"]):
                    stats["  (of which reader text != hand text)"] += 1
            else:
                row["text_hand"] = r["text"]
                row["status"] = "draft"
                row["note"] = (
                    f"split {fan[id(h)]} ways from old #{h['row']} (hand: {h['text']})"
                )
                stats["split from one hand row (draft)"] += 1
            if h["note"]:
                row["note"] = (row["note"] + "; " if row["note"] else "") + h["note"]
        rows.append(row)
    matched = {id(h) for h in owner if h is not None}
    record_rows = list(zip(records, rows, strict=True))
    for h in old:
        if id(h) in matched:
            continue
        # an old fragment box sitting inside a wider record: tell that record's row
        for r, row in record_rows:
            if (
                r["stem"] == h["stem"]
                and contained(h["box"], tuple(r["box"])) >= MATCH_CONTAIN
            ):
                row["note"] = (
                    row["note"] + "; " if row["note"] else ""
                ) + f"old #{h['row']} fragment inside: {h['text']}"
                stats["  (old fragment inside a wider record)"] += 1
        rows.append(
            {
                "stem": h["stem"],
                "box": json.dumps(list(h["box"])),
                "det_score": "",
                "engine": "hand_only",
                "kind_rec": h["kind_rec"],
                "text_rec": h["text_rec"],
                "kind_hand": h["kind"],
                "text_hand": h["text"],
                "status": "unchecked",
                "src_row": h["row"],
                "note": "no AnimeText record covers this old box"
                + (f"; {h['note']}" if h["note"] else ""),
            }
        )
        stats["old hand row without a record (unchecked)"] += 1
    rows.sort(
        key=lambda r: (r["stem"], json.loads(r["box"])[1], json.loads(r["box"])[0])
    )
    for i, r in enumerate(rows):
        r["row"] = i
    return rows, stats


def load_old(path: Path) -> list[dict]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return [
        {
            "row": r.row,
            "stem": r.stem,
            "box": tuple(json.loads(r.box)),
            "kind": r.kind_hand or r.kind_rec,
            "text": r.text_hand,
            "kind_rec": r.kind_rec,
            "text_rec": r.text_rec,
            "note": r.note,
        }
        for _, r in df.iterrows()
        if r.stem
    ]


def load_new_lines(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        (r.stem, tuple(json.loads(r.box))): dict(r) for _, r in df.iterrows() if r.stem
    }


def write_sheets(rows: list[dict], out: Path) -> None:
    """Draft rows only, 4 × 5 crops per sheet (the D1 hand-pass layout)."""
    import subprocess

    import cv2
    from PIL import Image, ImageDraw, ImageFont

    from anime_tools.ocr import sfx

    pick = [r for r in rows if r["status"] != "checked"]
    out.mkdir(parents=True, exist_ok=True)
    fpath = subprocess.run(
        ["fc-match", "-f", "%{file}", "Noto Sans CJK JP"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    font, small = ImageFont.truetype(fpath, 20), ImageFont.truetype(fpath, 14)
    TW, TH, CAP, cols, rows_per = 300, 210, 90, 4, 5
    per = cols * rows_per
    cache: dict[str, object] = {}
    for s in range(0, len(pick), per):
        sheet = Image.new("RGB", (cols * TW, rows_per * (TH + CAP)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(pick[s : s + per]):
            bgr = cache.get(r["stem"])
            if bgr is None:
                bgr = cache[r["stem"]] = cv2.imread(
                    str(
                        REPO / "post_image_dataset/resized" / SHARD / f"{r['stem']}.png"
                    )
                )
            crop = sfx.crop_box(bgr, json.loads(r["box"]), 0.12)
            im = Image.fromarray(crop[:, :, ::-1])
            sc = min((TW - 10) / im.width, (TH - 10) / im.height)
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
            x0, y0 = (j % cols) * TW, (j // cols) * (TH + CAP)
            sheet.paste(im, (x0 + (TW - im.width) // 2, y0 + (TH - im.height) // 2))
            draw.rectangle(
                [x0, y0, x0 + TW - 1, y0 + TH + CAP - 1], outline=(200, 200, 200)
            )
            det = (
                f"det {float(r['det_score']):.2f}"
                if r["det_score"] not in ("", None)
                else r["engine"]
            )
            draw.text(
                (x0 + 6, y0 + TH + 2),
                f"#{r['row']}  {r['kind_hand']}  {det}  [{r['status']}]",
                fill=(90, 90, 90),
                font=small,
            )
            draw.text(
                (x0 + 6, y0 + TH + 20), r["text_hand"][:22], fill=(0, 0, 0), font=font
            )
            draw.text(
                (x0 + 6, y0 + TH + 46), r["note"][:40], fill=(160, 60, 60), font=small
            )
            draw.text(
                (x0 + 6, y0 + TH + 64),
                f"{r['stem']} {r['box']}",
                fill=(140, 140, 140),
                font=small,
            )
        p = out / f"sheet_{s // per:02d}.png"
        sheet.save(p)
    print(
        f"{len(pick)} draft/unchecked rows → {len(range(0, len(pick), per))} sheets under {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument(
        "--old",
        type=Path,
        required=True,
        help="the PP-box hand labels (retired; see the docstring)",
    )
    ap.add_argument("--out", type=Path, default=ASSETS / f"sfx_labels_{SHARD}.tsv")
    ap.add_argument(
        "--sheets",
        action="store_true",
        help="also draw crop sheets of the non-checked rows",
    )
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.out.exists() and not a.overwrite:
        sys.exit(
            f"{a.out} exists — --overwrite to rebuild (hand edits in it would be lost)"
        )
    records = [
        json.loads(line)
        for line in a.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old = load_old(a.old)
    rows, stats = build(records, old, load_new_lines(NEW_LINES))
    cols = [
        "row",
        "stem",
        "box",
        "det_score",
        "engine",
        "kind_rec",
        "text_rec",
        "kind_hand",
        "text_hand",
        "status",
        "src_row",
        "note",
    ]
    pd.DataFrame(rows)[cols].to_csv(a.out, sep="\t", index=False)
    print(
        f"wrote {a.out}: {len(rows)} rows from {len(records)} records + {len(old)} old hand rows"
    )
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  status:", dict(Counter(r["status"] for r in rows)))
    print("  kind_hand:", dict(Counter(r["kind_hand"] for r in rows)))
    if a.sheets:
        write_sheets(rows, SHEETS)


if __name__ == "__main__":
    main()
