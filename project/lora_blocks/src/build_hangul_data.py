"""Plan step 3 data: synthetic Korean lettering for the Hangul block (CPU).

Base pages are the Korean edition pages of swap_text v3 (``pairs_all.jsonl``, eval works
already out). On each page the OCR text boxes that sit inside a white balloon (the ring
around the padded box is near-white) are whitened, and a Korean line from another page's
OCR is lettered into the box in one of the Korean faces of ``cjk_anima_reseed``: target =
the page with the rendered lines. Condition, per page: ``blank`` (the boxes whitened) or
``orig`` (the edition's own lettering, so the edit is a Korean → Korean replacement). The
caption is swap_text's layout JSON with the new strings and their ink boxes; boxes left
alone keep their OCR item.

Writes ``post_image_dataset/i2i/hangul_block/{img/, pairs.jsonl, syllables.json,
sheet.png}``. Cache with ``project/lora_blocks/configs/hangul_cache.toml``.

    .venv/bin/python project/lora_blocks/src/build_hangul_data.py [--pages 600]
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from anime_tools.captions.layout_json import centered_box
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from library.env import resolve_under_home

STAGING = resolve_under_home("post_image_dataset/i2i/swap_text_v3/staging")
OUT = resolve_under_home("post_image_dataset/i2i/hangul_block")
FONT_DIRS = [
    resolve_under_home("project/cjk_anima_reseed/assets/fonts/kozh"),
    resolve_under_home("project/cjk_anima_reseed/assets/fonts/ko_extra"),
]
PAD = 4
RING = 5
WHITE = 215
RING_WHITE = 0.93
MIN_FS = 11
MAX_BOXES = 8
KEEP = re.compile(r"[가-힣 !?.,~…\-]")


def hangul(s: str) -> int:
    return sum("가" <= c <= "힣" for c in s)


def clean(s: str) -> str:
    s = "".join(c if KEEP.match(c) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def load_fonts() -> list[tuple[Path, set[int]]]:
    fonts = []
    for d in FONT_DIRS:
        for f in sorted(d.glob("*")):
            if f.suffix.lower() not in (".ttf", ".otf"):
                continue
            cmap = set(TTFont(str(f), fontNumber=0).getBestCmap())
            if sum(c in cmap for c in range(0xAC00, 0xD7A4)) >= 2350:
                fonts.append((f, cmap))
    return fonts


def wrap(text: str, n: int) -> list[str]:
    """``text`` in ``n`` lines of balanced length, broken at spaces when it has them."""
    words = text.split(" ")
    if n == 1:
        return [text]
    if len(words) < n:
        chars = text.replace(" ", "")
        if len(chars) < n:
            return []
        step = len(chars) / n
        return [chars[round(i * step) : round((i + 1) * step)] for i in range(n)]
    target = len(text) / n
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > target * 1.15 and len(lines) < n - 1:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    return lines if len(lines) == n else []


def fit(text, font_path, box_w, box_h, fs_cap):
    """The largest size and line split that fit ``text`` in the box, or None."""
    best = None
    for n in range(1, 8):
        lines = wrap(text, n)
        if not lines:
            continue
        lo, hi = MIN_FS, int(fs_cap)
        found = None
        while lo <= hi:
            fs = (lo + hi) // 2
            font = ImageFont.truetype(str(font_path), fs)
            w = max(font.getlength(ln) for ln in lines)
            h = n * fs * 1.18
            if w <= box_w and h <= box_h:
                found = fs
                lo = fs + 1
            else:
                hi = fs - 1
        if found and (best is None or found > best[0]):
            best = (found, lines)
    return best


def box_pixels(item, size):
    w, h = size
    y0, x0, y1, x1 = item["bbox"]
    return x0 + w / 2, y0 + h / 2, x1 + w / 2, y1 + h / 2


def in_balloon(luma: np.ndarray, x0, y0, x1, y1) -> bool:
    h, w = luma.shape
    a0, b0, a1, b1 = x0 - RING, y0 - RING, x1 + RING, y1 + RING
    if a0 < 0 or b0 < 0 or a1 > w or b1 > h:
        return False
    outer = luma[b0:b1, a0:a1].astype(np.float32)
    ring = np.ones_like(outer, dtype=bool)
    ring[RING:-RING, RING:-RING] = False
    return (outer[ring] >= WHITE).mean() >= RING_WHITE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [
        json.loads(ln)
        for ln in (STAGING / "pairs_all.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    rows = [r for r in rows if r["lang"] == "ko"]
    pool = []
    for r in rows:
        for it in json.loads((STAGING / r["caption"]).read_text())["items"]:
            s = clean(it.get("text", ""))
            if 2 <= len(s.replace(" ", "")) <= 40 and hangul(s) >= 0.7 * len(
                s.replace(" ", "")
            ):
                pool.append(s)
    pool = sorted(set(pool))
    by_len: dict[int, list[str]] = {}
    for s in pool:
        by_len.setdefault(len(s.replace(" ", "")), []).append(s)
    fonts = load_fonts()
    print(f"{len(rows)} ko pages, {len(pool)} lines, {len(fonts)} faces")

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rng.shuffle(rows)
    out_rows, syl, face_n, n_boxes = [], Counter(), Counter(), 0
    for r in rows:
        if len(out_rows) >= args.pages:
            break
        page = Image.open(STAGING / r["target"]).convert("RGB")
        luma = np.asarray(page.convert("L"))
        cap = json.loads((STAGING / r["caption"]).read_text())
        tgt = page.copy()
        blank = page.copy()
        dt, db = ImageDraw.Draw(tgt), ImageDraw.Draw(blank)
        items, done = [], 0
        for it in cap["items"]:
            x0, y0, x1, y1 = (round(v) for v in box_pixels(it, page.size))
            px0, py0, px1, py1 = x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD
            orig = clean(it.get("text", "")).replace(" ", "")
            ok = (
                done < MAX_BOXES
                and x1 - x0 >= 14
                and y1 - y0 >= 14
                and orig
                and in_balloon(luma, px0, py0, px1, py1)
            )
            rendered = None
            if ok:
                n0 = len(orig)
                g0 = ((x1 - x0) * (y1 - y0) / n0) ** 0.5
                n1 = min(n0, 30)
                lens = [n for n in by_len if 0.5 * n1 <= n <= max(1.5 * n1, 2)]
                for _ in range(6):
                    s = rng.choice(by_len[rng.choice(lens)])
                    face, cmap = rng.choice(fonts)
                    if any(ord(c) not in cmap for c in s if c != " "):
                        continue
                    got = fit(s, face, px1 - px0 - 2, py1 - py0 - 2, max(1.25 * g0, 16))
                    if got:
                        rendered = (s, face, *got)
                        break
            if not rendered:
                items.append(it)
                continue
            s, face, fs, lines = rendered
            font = ImageFont.truetype(str(face), fs)
            for d in (dt, db):
                d.rectangle([px0, py0, px1 - 1, py1 - 1], fill=(255, 255, 255))
            lh = fs * 1.18
            top = (py0 + py1) / 2 - len(lines) * lh / 2
            ink = Image.new("L", page.size, 0)
            di = ImageDraw.Draw(ink)
            for i, ln in enumerate(lines):
                cx, cy = (px0 + px1) / 2, top + (i + 0.5) * lh
                dt.text((cx, cy), ln, font=font, fill=(16, 16, 16), anchor="mm")
                di.text((cx, cy), ln, font=font, fill=255, anchor="mm")
            ib = ink.getbbox()
            items.append(
                {
                    "label": "text",
                    "caption": "text",
                    "text": " ".join(lines) if " " in s else "".join(lines),
                    "bbox": centered_box(ib, page.size),
                }
            )
            syl.update(c for c in s if "가" <= c <= "힣")
            face_n[face.name] += 1
            done += 1
        if not done:
            continue
        mode = rng.choice(("blank", "orig"))
        stem = f"hg_{r['stem']}"
        tgt.save(OUT / "img" / f"{stem}.tgt.png")
        (blank if mode == "blank" else page).save(OUT / "img" / f"{stem}.src.png")
        new_cap = dict(cap, items=items)
        (OUT / "img" / f"{stem}.txt").write_text(
            json.dumps(new_cap, ensure_ascii=False, separators=(",", ":"))
        )
        out_rows.append(
            {
                "stem": stem,
                "target": f"img/{stem}.tgt.png",
                "cond": f"img/{stem}.src.png",
                "caption": f"img/{stem}.txt",
                "base": r["stem"],
                "mode": mode,
                "rendered": done,
            }
        )
        n_boxes += done

    (OUT / "pairs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    )
    (OUT / "syllables.json").write_text(
        json.dumps(
            {"faces": dict(face_n), "syllables": dict(syl.most_common())},
            ensure_ascii=False,
            indent=0,
        )
    )
    modes = Counter(r["mode"] for r in out_rows)
    print(
        f"{len(out_rows)} pairs ({dict(modes)}), {n_boxes} lines, "
        f"{len(syl)} distinct syllables -> {OUT}"
    )

    tiles = []
    for r in out_rows[:4]:
        a = Image.open(OUT / r["cond"])
        b = Image.open(OUT / r["target"])
        t = Image.new("RGB", (a.width * 2, a.height), "white")
        t.paste(a, (0, 0))
        t.paste(b, (a.width, 0))
        tiles.append(t.resize((t.width // 2, t.height // 2)))
    sheet = Image.new("RGB", (tiles[0].width * 2, tiles[0].height * 2), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * t.width, (i // 2) * t.height))
    sheet.save(OUT / "sheet.png")


if __name__ == "__main__":
    main()
