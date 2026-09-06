#!/usr/bin/env python3
"""Contact sheet for the SFX path end to end: crop → reader → record → caption clause.

``ocr_contact_sheet.py`` shows a page's records; this one shows **how each
SFX line got into the caption** (plan_ocr O4 / arm C11, the default since
``DROP_KINDS`` lost ``sfx`` 2026-09-06). One tile per page that carries a
``kind: sfx`` record:

* **left** — the resized page, every record boxed by kind (**red** = sfx,
  black = speech, grey = chrome; a **dashed blue** box is a MIT-mask
  component the reader read that no detector boxed, ``engine=sfx_reader``),
  numbered in reading order;
* **middle** — the SFX crops *as the reader saw them* (``crop_box`` with the
  package's ``CROP_PAD``), each with the guarded read, the raw decode when
  the guard changed it, the text the record held **before** the re-read
  (``prev_text`` — what PP-OCRv6 / VL Spotting made of it), the engine chain
  and whether ``kind`` came from a hand label or the text rule;
* **right** — the caption the mirror holds for the page, split by the
  grammar: flat bag (grey), position clauses (grey), the speech clause
  (black) and the **SFX clause (red)**; the SFX clause is recomputed from the
  records through the same ``text_clause`` and a difference is flagged.

    python project/cjk_aware_anima_dit/probes/sfx_caption_sheet.py            # 87 sincos pages
    python project/cjk_aware_anima_dit/probes/sfx_caption_sheet.py --only mask # pages with a mask-component read
    python project/cjk_aware_anima_dit/probes/sfx_caption_sheet.py --only rule # pages whose kind came from the rule

Writes ``output/tests/ocr_contact_sheet/<name>_pNN.png`` + ``<name>.pdf``.
CPU only (PIL + the torch-free parts of ``anime_tools``); needs a CJK font.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401  (PDF save needs the JPEG codec registered)

from anime_tools.captions.position_clauses import (
    compose_caption,
    parse_caption,
    text_clause,
)
from anime_tools.ocr.sfx import CROP_PAD, pad_box

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "project/cjk_aware_anima/datasets"))
from ocr_sfx import dedupe_sfx, sfx_groups  # noqa: E402  (torch-free; the caption builder's rule)

UNMASK = REPO / "post_image_dataset/cjk_unmask"

KIND_COLOUR = {"sfx": (210, 0, 0), "speech": (0, 0, 0), "chrome": (140, 140, 140)}
MASK_COMP = (0, 110, 255)
PREV = (200, 110, 0)
GREY = (110, 110, 110)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    fam = "Noto Sans CJK JP" + (":bold" if bold else "")
    path = subprocess.run(
        ["fc-match", "-f", "%{file}", fam], capture_output=True, text=True
    ).stdout.strip()
    return ImageFont.truetype(path, size)


def load(path: Path) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["stem"]].append(r)
    return by


def wrap(text: str, fnt, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if fnt.getlength(cur + ch) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines or [""]


def dashed_rect(d: ImageDraw.ImageDraw, box, colour, width: int, dash: int = 8):
    x0, y0, x1, y1 = box
    for ax, ay, bx, by in (
        (x0, y0, x1, y0),
        (x1, y0, x1, y1),
        (x1, y1, x0, y1),
        (x0, y1, x0, y0),
    ):
        L = max(abs(bx - ax), abs(by - ay))
        n = max(1, int(L / dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1.0, (i + 1) / n)
            d.line(
                [
                    (ax + (bx - ax) * t0, ay + (by - ay) * t0),
                    (ax + (bx - ax) * t1, ay + (by - ay) * t1),
                ],
                fill=colour,
                width=width,
            )


def clause_str(c) -> str:
    return compose_caption((), (c,))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--records", type=Path, default=UNMASK / "ocr_records_sincos_hybrid_vl.jsonl"
    )
    ap.add_argument(
        "--mirror", type=Path, default=UNMASK / "mirror_sincos_hybrid_vl_sentence"
    )
    ap.add_argument("--shard", default="sincos")
    ap.add_argument("--name", default="sfx_caption")
    ap.add_argument("--out", type=Path, default=REPO / "output/tests/ocr_contact_sheet")
    ap.add_argument(
        "--only",
        choices=["sfx", "all", "mask", "rule", "changed"],
        default="sfx",
        help="sfx: pages with a kind=sfx record (default); all: every page with records; "
        "mask: pages with a mask-component read; rule: pages with a rule-kinded SFX record; "
        "changed: pages where the re-read changed an SFX record's text",
    )
    ap.add_argument(
        "--crops",
        choices=["sfx", "changed"],
        default="sfx",
        help="which records get a crop in the middle column: sfx (default) = the kind=sfx "
        "records; changed = every record the re-read changed (prev_text), any kind",
    )
    ap.add_argument(
        "--baseline_mirror",
        type=Path,
        default=None,
        help="a second mirror whose text clauses are printed (grey, struck) under the "
        "current ones when they differ — e.g. the SFX-only re-read captions under the all-VL ones",
    )
    ap.add_argument("--limit", type=int, default=0)
    opts = ap.parse_args()
    out = opts.out
    out.mkdir(parents=True, exist_ok=True)

    resized = REPO / "post_image_dataset/resized" / opts.shard
    recs = load(opts.records)

    def is_sfx(r):
        return r.get("kind", "speech") == "sfx"

    stems = sorted(recs)
    if opts.only == "sfx":
        stems = [s for s in stems if any(is_sfx(r) for r in recs[s])]
    elif opts.only == "mask":
        stems = [s for s in stems if any(r["engine"] == "sfx_reader" for r in recs[s])]
    elif opts.only == "rule":
        stems = [
            s
            for s in stems
            if any(is_sfx(r) and r.get("kind_src") == "rule" for r in recs[s])
        ]
    elif opts.only == "changed":
        want = is_sfx if opts.crops == "sfx" else (lambda r: True)
        stems = [s for s in stems if any(want(r) and "prev_text" in r for r in recs[s])]
    if opts.limit:
        stems = stems[: opts.limit]

    F, FB, FS, FT = font(18), font(20, True), font(15), font(14)
    IMG_W, CROP_W, TXT_W, PAD = 420, 340, 560, 12
    TILE_W = IMG_W + CROP_W + TXT_W + 4 * PAD
    CROP_MAX = 110
    mismatches: list[str] = []

    def tile(stem: str) -> Image.Image:
        im = Image.open(resized / f"{stem}.png").convert("RGB")
        W, H = im.size
        s = min(IMG_W / W, 620 / H)
        page = recs[stem]
        arr = np.array(im)
        # boxes at full res, then downscale
        d = ImageDraw.Draw(im)
        for r in page:
            kind = r.get("kind", "speech")
            if r["engine"] == "sfx_reader":
                dashed_rect(d, r["box"], MASK_COMP, max(3, int(3 / s)))
            else:
                d.rectangle(
                    r["box"],
                    outline=KIND_COLOUR[kind],
                    width=max(3, int(3 / s)) if kind == "sfx" else max(2, int(2 / s)),
                )
        im = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for i, r in enumerate(page, 1):
            x0, y0 = r["box"][0] * s, r["box"][1] * s
            col = (
                MASK_COMP
                if r["engine"] == "sfx_reader"
                else KIND_COLOUR[r.get("kind", "speech")]
            )
            d.rectangle([x0, y0 - 18, x0 + 26, y0], fill=col)
            d.text((x0 + 3, y0 - 18), str(i), font=FS, fill="white")

        # ---- middle: the SFX crops as the reader saw them
        crops: list[tuple[Image.Image, list[tuple[str, object, object]]]] = []
        sfx_idx = [i for i, r in enumerate(page, 1) if is_sfx(r)]
        groups = sfx_groups([page[i - 1]["text"] for i in sfx_idx])
        dup_of = {
            i: sfx_idx[g]
            for i, g in zip(sfx_idx, groups, strict=True)
            if sfx_idx[g] != i
        }
        for i, r in enumerate(page, 1):
            if opts.crops == "sfx" and not is_sfx(r):
                continue
            if opts.crops == "changed" and "prev_text" not in r:
                continue
            kcol = KIND_COLOUR[r.get("kind", "speech")]
            x0, y0, x1, y1 = pad_box(r["box"], W, H, CROP_PAD)
            c = Image.fromarray(arr[y0:y1, x0:x1])
            cs = min(CROP_MAX / c.width, CROP_MAX / c.height, 1.0)
            c = c.resize(
                (max(1, int(c.width * cs)), max(1, int(c.height * cs))), Image.LANCZOS
            )
            rows: list[tuple[str, object, object]] = []
            src = (
                "mask comp"
                if r["engine"] == "sfx_reader"
                else r["engine"].removesuffix("+sfx_reader")
            )
            if i in dup_of:
                rows.append((f"#{i}  {r['text']}", FB, ("strike", GREY)))
                rows.append((f"repeat of #{dup_of[i]} → not in caption", FT, GREY))
            else:
                rows.append((f"#{i}  {r['text']}", FB, kcol))
            raw = r.get("sfx_raw")
            if raw is not None and raw != r["text"]:
                rows.append((f"raw: {raw}", FT, GREY))
            if "prev_text" in r:
                rows.append((f"was: {r['prev_text']}", FS, PREV))
            elif r.get("sfx_guard") == "rejected":
                rows.append(("guard REJECTED, kept old read", FS, PREV))
            elif r["engine"] == "sfx_reader":
                rows.append(("new line (no detector box)", FT, MASK_COMP))
            rows.append(
                (
                    f"{src} → sfx_reader · kind by {r.get('kind_src', '?')}"
                    + f" · {x1 - x0}×{y1 - y0}px",
                    FT,
                    GREY,
                )
            )
            crops.append((c, rows))

        # ---- right: the caption the mirror holds
        cap_path = opts.mirror / f"{stem}.txt"
        rows_r: list[tuple[str, object, object]] = []
        n_sfx = sum(is_sfx(r) for r in page)
        n_mask = sum(r["engine"] == "sfx_reader" for r in page)
        rows_r.append(
            (
                f"{stem}  {W}x{H}  lines={len(page)}  sfx={n_sfx}  mask-comp={n_mask}",
                FB,
                "black",
            )
        )
        if not cap_path.exists():
            rows_r.append(("NO CAPTION IN MIRROR", FB, KIND_COLOUR["sfx"]))
        else:
            parsed = parse_caption(cap_path.read_text(encoding="utf-8").strip())
            rows_r.append((", ".join(parsed.flat_tags), FT, GREY))
            for c in parsed.position_clauses:
                rows_r.append((clause_str(c), FT, GREY))
            sfx_clause = None
            for c in parsed.text_clauses:
                txt = clause_str(c)
                if "SFX" in c.prefix:
                    sfx_clause = txt
                    rows_r.append((txt, F, KIND_COLOUR["sfx"]))
                else:
                    rows_r.append((txt, F, "black"))
            if opts.baseline_mirror and (opts.baseline_mirror / f"{stem}.txt").exists():
                bparsed = parse_caption(
                    (opts.baseline_mirror / f"{stem}.txt")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                for bc in bparsed.text_clauses:
                    btxt = clause_str(bc)
                    if all(btxt != clause_str(c) for c in parsed.text_clauses):
                        rows_r.append(
                            (
                                f"{opts.baseline_mirror.name}: {btxt}",
                                FS,
                                ("strike", GREY),
                            )
                        )
            expect = dedupe_sfx([r["text"] for r in page if is_sfx(r)])
            expect_s = clause_str(text_clause(expect, sfx=True)) if expect else None
            if expect_s != sfx_clause:
                mismatches.append(stem)
                rows_r.append(("", FS, "black"))
                rows_r.append(("SFX CLAUSE ≠ RECORDS", FB, (180, 0, 120)))
                rows_r.append((f"records: {expect_s}", FS, (180, 0, 120)))
            elif sfx_idx:
                idx = [str(i) for i in sfx_idx if i not in dup_of]
                rows_r.append(("", FS, "black"))
                rows_r.append(
                    (f"SFX clause ← records #{', #'.join(idx)}  (matches)", FT, GREY)
                )

        # ---- compose
        out_r = [(ln, f, c) for t, f, c in rows_r for ln in wrap(t, f, TXT_W)]
        th_r = sum(f.size + 5 for _, f, _ in out_r)
        crop_blocks = []
        for c, rows in crops:
            lines = [
                (ln, f, col)
                for t, f, col in rows
                for ln in wrap(t, f, CROP_W - c.width - 8)
            ]
            hh = max(c.height, sum(f.size + 4 for _, f, _ in lines))
            crop_blocks.append((c, lines, hh))
        th_c = sum(hh + 10 for _, _, hh in crop_blocks)
        h = max(im.height, th_r, th_c) + 2 * PAD
        t = Image.new("RGB", (TILE_W, h), "white")
        t.paste(im, (PAD, PAD))
        d = ImageDraw.Draw(t)
        x_c = IMG_W + 2 * PAD
        y = PAD
        for c, lines, hh in crop_blocks:
            t.paste(c, (x_c, y))
            d.rectangle(
                [x_c - 1, y - 1, x_c + c.width, y + c.height],
                outline=KIND_COLOUR["sfx"],
            )
            yy = y
            for ln, f, col in lines:
                strike = isinstance(col, tuple) and col and col[0] == "strike"
                if strike:
                    col = col[1]
                d.text((x_c + c.width + 8, yy), ln, font=f, fill=col)
                if strike:
                    ym = yy + f.size * 0.55
                    d.line(
                        [
                            (x_c + c.width + 8, ym),
                            (x_c + c.width + 8 + f.getlength(ln), ym),
                        ],
                        fill=col,
                        width=2,
                    )
                yy += f.size + 4
            y += hh + 10
        x_r = IMG_W + CROP_W + 3 * PAD
        y = PAD
        for ln, f, c in out_r:
            strike = isinstance(c, tuple) and c and c[0] == "strike"
            if strike:
                c = c[1]
            d.text((x_r, y), ln, font=f, fill=c)
            if strike:
                ym = y + f.size * 0.55
                d.line([(x_r, ym), (x_r + f.getlength(ln), ym)], fill=c, width=1)
            y += f.size + 5
        d.rectangle([0, 0, TILE_W - 1, h - 1], outline=(200, 200, 200))
        return t

    tiles = [tile(s) for s in stems]
    COLS, MAXH = 2, 2300
    pages: list[tuple[list, int]] = []
    page: list = []
    y = 0
    row: list = []
    for t in tiles:
        row.append(t)
        if len(row) == COLS:
            rh = max(tt.height for tt in row)
            if y + rh > MAXH and page:
                pages.append((page, y))
                page, y = [], 0
            page.append((row, y, rh))
            y += rh
            row = []
    if row:
        rh = max(tt.height for tt in row)
        if y + rh > MAXH and page:
            pages.append((page, y))
            page, y = [], 0
        page.append((row, y, rh))
        y += rh
    if page:
        pages.append((page, y))
    legend = (
        f"{opts.name} · {opts.records.name} → {opts.mirror.name} · boxes: red = sfx, black = speech, "
        "grey = chrome, dashed blue = MIT-mask component read by the SFX reader · middle: crop as the reader "
        "saw it (pad 0.12), read / raw / 'was' = text before the re-read, struck = repeat dropped by dedupe_sfx · right: caption, SFX clause in red"
    )
    outs = []
    for i, (pg, ph) in enumerate(pages, 1):
        canvas = Image.new("RGB", (COLS * TILE_W, ph + 40), "white")
        d = ImageDraw.Draw(canvas)
        d.text((10, 8), f"p{i}/{len(pages)}  " + legend, font=FT, fill=(80, 80, 80))
        for r, yy, rh in pg:
            for j, t in enumerate(r):
                canvas.paste(t, (j * TILE_W, yy + 40))
        canvas.save(out / f"{opts.name}_p{i:02d}.png")
        outs.append(canvas)
    if outs:
        outs[0].save(
            out / f"{opts.name}.pdf",
            save_all=True,
            append_images=outs[1:],
            resolution=100,
        )
    print(f"{len(stems)} tiles, {len(pages)} pages -> {out / (opts.name + '.pdf')}")
    if mismatches:
        print(
            f"SFX clause ≠ records on {len(mismatches)} pages: {' '.join(mismatches)}"
        )
    else:
        print("SFX clause == records on every tile")


if __name__ == "__main__":
    main()
