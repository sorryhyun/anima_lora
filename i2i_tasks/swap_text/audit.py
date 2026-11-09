#!/usr/bin/env python3
"""Proofread the staged captions against their lettering, then fold the result back
into ``<root>/ocr.json``. Three steps, run after **caption**:

1. ``crops`` — one crop per caption line of the pages in ``--keys`` (a JSON list of
   ``<lang>/<anchor>/<page>`` keys; default every staged page), cut from the staged
   target the caption describes (the box grown by 12 % of its longer side, clipped to
   the page; crops under 256 px upscaled for reading). Writes
   ``<root>/audit_crops/{id}.png`` and ``manifest.jsonl`` rows
   ``{id, stem, line, box, text}``.
2. ``sheets`` — the crops 20 to a sheet, 5 wide, each with its id and its text under
   it (line breaks as a red ``|``; ``ZOOM`` on long text). Writes
   ``<root>/audit_sheets/sheet_NNNN.png`` and ``index.json``. Proofreaders return rows
   ``{id, fix}``, ``{id, drop}`` or ``{id, unsure}``; a crop with no row is confirmed.
3. ``apply`` — read the merged labels (``--labels``: rows ``{stem, box, v5_read, label,
   status, drop_reason}``) and edit ``ocr.json``: a ``fixed`` line's row takes the label,
   a ``drop`` line's row is deleted — its text block then goes unread, so **pool** keeps
   the group JA and out of the caption instead of pasting lettering the caption does not
   name — as is an ``unsure`` row noted as Japanese lettering. Rows match on page key,
   box and text. The touched pages leave ``pool_metrics.json``; rerun
   ``--stages pool assemble caption`` after.

    .venv/bin/python -m i2i_tasks.swap_text.audit crops --root <root> --keys keys.json
    .venv/bin/python -m i2i_tasks.swap_text.audit sheets --root <root>
    .venv/bin/python -m i2i_tasks.swap_text.audit apply --root <root> --labels labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from i2i_tasks.swap_text.common import read_jsonl, write_jsonl
from library.env import resolve_under_home

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
PAD, MIN_SIDE = 0.12, 256
CW, CH, TH, COLS, PER, ZOOM = 300, 300, 150, 5, 20, 40
JA_NOTE = re.compile(r"japanese|kana|일본", re.I)


def crop_box(box, w: int, h: int) -> list[int]:
    x0, y0, x1, y1 = (int(v) for v in box)
    g = round(PAD * max(x1 - x0, y1 - y0))
    return [max(x0 - g, 0), max(y0 - g, 0), min(x1 + g, w), min(y1 + g, h)]


def stage_crops(a) -> None:
    import cv2

    staging, out = a.root / "staging", a.root / "audit_crops"
    out.mkdir(exist_ok=True)
    keys = set(json.loads(a.keys.read_text())) if a.keys else None
    plan = {r["stem"]: r for r in read_jsonl(staging / "plan.jsonl")}
    rows = []
    for p in read_jsonl(staging / "pairs.jsonl"):
        r = plan[p["stem"]]
        if keys is not None and r["key"] not in keys:
            continue
        cap = json.loads((staging / p["caption"]).read_text(encoding="utf-8"))
        # caption items are the plan's lines minus the JA ones, in order
        boxes = iter([(t.replace('"', "”").strip(), b) for t, b in r["lines"]])
        img = cv2.imread(str(staging / p["target"]))
        h, w = img.shape[:2]
        for li, it in enumerate(cap.get("items", [])):
            box = next(b for t, b in boxes if t == it["text"])
            x0, y0, x1, y1 = crop_box(box, w, h)
            c = img[y0:y1, x0:x1]
            if max(c.shape[:2]) < MIN_SIDE:
                s = MIN_SIDE / max(c.shape[:2])
                c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            rid = f"{len(rows):05d}"
            cv2.imwrite(str(out / f"{rid}.png"), c)
            rows.append(
                {
                    "id": rid,
                    "stem": p["stem"],
                    "line": li,
                    "box": [int(v) for v in box],
                    "text": it["text"],
                }
            )
    write_jsonl(out / "manifest.jsonl", rows)
    print(f"{len(rows)} caption lines")


def stage_sheets(a) -> None:
    from PIL import Image, ImageDraw, ImageFont

    crops, out = a.root / "audit_crops", a.root / "audit_sheets"
    out.mkdir(exist_ok=True)
    rows = read_jsonl(crops / "manifest.jsonl")
    font = ImageFont.truetype(FONT, 20, index=1)  # the KR face of the collection
    small = ImageFont.truetype(FONT, 15, index=1)
    index = []
    for s in range(0, len(rows), PER):
        part = rows[s : s + PER]
        n_rows = (len(part) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * CW, n_rows * (CH + TH)), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(part):
            x, y = (i % COLS) * CW, (i // COLS) * (CH + TH)
            im = Image.open(crops / f"{r['id']}.png").convert("RGB")
            im.thumbnail((CW - 10, CH - 10))
            sheet.paste(im, (x + (CW - im.width) // 2, y + (CH - im.height) // 2))
            d.rectangle([x, y, x + CW - 1, y + CH + TH - 1], outline=(180, 180, 180))
            tag = r["id"] + ("  ZOOM" if len(r["text"]) >= ZOOM else "")
            d.rectangle(
                [x + 2, y + 2, x + 8 + d.textlength(tag, font=small), y + 22],
                fill="white",
            )
            d.text((x + 5, y + 2), tag, fill=(200, 0, 0), font=small)
            toks = []
            for k, line in enumerate(r["text"].split("\n")):
                if k:
                    toks.append((" | ", (220, 0, 0)))
                toks += [(ch, (0, 0, 160)) for ch in line]
            tx, ty = x + 6, y + CH + 4
            for t, col in toks:
                wdt = d.textlength(t, font=font)
                if tx + wdt > x + CW - 6:
                    tx, ty = x + 6, ty + 24
                if ty > y + CH + TH - 24:
                    break
                d.text((tx, ty), t, fill=col, font=font)
                tx += wdt
        name = f"sheet_{s // PER:04d}.png"
        sheet.save(out / name)
        index.append({"sheet": name, "ids": [r["id"] for r in part]})
    (out / "index.json").write_text(json.dumps(index))
    print(f"{len(index)} sheets")


def stage_apply(a) -> None:
    labels = read_jsonl(a.labels)
    plan = {r["stem"]: r["key"] for r in read_jsonl(a.root / "staging" / "plan.jsonl")}
    ocr = json.loads((a.root / "ocr.json").read_text(encoding="utf-8"))
    n = {"fix": 0, "drop": 0, "unmatched": 0}
    dels: dict[str, set[int]] = {}
    touched: set[str] = set()
    for r in labels:
        if r["status"] == "fixed":
            act = "fix"
        elif r["status"] == "drop" or (
            r["status"] == "unsure" and JA_NOTE.search(r.get("drop_reason", ""))
        ):
            act = "drop"
        else:
            continue
        key = plan[r["stem"]]
        hit = [
            i
            for i, row in enumerate(ocr[key])
            if [int(v) for v in row[1]] == r["box"]
            and row[0].replace('"', "”").strip() == r["v5_read"]
        ]
        if len(hit) != 1:
            n["unmatched"] += 1
            print(f"unmatched {r.get('id')} {key} ({len(hit)} rows)")
            continue
        if act == "fix":
            ocr[key][hit[0]][0] = r["label"]
        else:
            dels.setdefault(key, set()).add(hit[0])
        n[act] += 1
        touched.add(key)
    for key, idx in dels.items():
        ocr[key] = [row for i, row in enumerate(ocr[key]) if i not in idx]
    print(n, f"pages {len(touched)}")
    if a.dry:
        return
    (a.root / "ocr.json").write_text(json.dumps(ocr, ensure_ascii=False))
    mpath = a.root / "pool_metrics.json"
    metrics = json.loads(mpath.read_text())
    for k in touched:
        metrics.pop(k, None)
    mpath.write_text(json.dumps(metrics))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=("crops", "sheets", "apply"))
    ap.add_argument("--root", default="post_image_dataset/i2i/swap_text_relax")
    ap.add_argument("--keys", type=Path, help="crops: page keys to cut (JSON list)")
    ap.add_argument("--labels", type=Path, help="apply: merged label rows (jsonl)")
    ap.add_argument("--dry", action="store_true", help="apply: count, write nothing")
    a = ap.parse_args()
    a.root = resolve_under_home(a.root)
    {"crops": stage_crops, "sheets": stage_sheets, "apply": stage_apply}[a.step](a)


if __name__ == "__main__":
    main()
