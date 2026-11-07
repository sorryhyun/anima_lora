"""Plan step 3 data, v2: synthetic Korean lettering laid out as the edition letters it.

Changes from v1 (``build_hangul_data.py``, reports/hangul_s500.md):

- **Balloons, not boxes.** A box is used when its balloon is closed: the white region
  (luma >= 200) reached from the white pixels around the edition's text box, with its
  holes filled, is 1.2x the box to 6 % of the page. The balloon is filled with its own
  ground colour (the median of its light pixels), which removes the lettering, Japanese or
  Korean. Text on artwork and open balloons are
  left alone.
- **Base page**: half the pages are the Japanese raw (``.src``), half the Korean edition
  (``.tgt``) of a swap_text v3 twin, aligned pixel for pixel, so the edition's boxes
  locate the same balloons on both.
- **The edition's layout**: line count, glyph height and line pitch are measured from the
  edition's ink in the box. The new string is wrapped into that many lines at that height
  (shrunk only to fit the balloon) and centred where the edition's block sat.
- **One face per page**, bold dialogue faces only (v1's block lettered thinner than
  ksplit e2).
- **Partial captions**: each usable balloon is re-lettered with p 0.7 (at least one per
  page). The others keep their text, Japanese or Korean, on both sides of the pair and are
  left out of the caption, as swap_text leaves out a balloon that stays.
- Text: the edition's own line for that box (p 0.5) or another Korean OCR line within
  ±25 % of its length.

Condition, per page: ``blank`` (the re-lettered balloons emptied) or ``orig`` (the base
page as it is: Japanese raw = swap_text's task, Korean edition = Korean → Korean).

Writes ``post_image_dataset/i2i/hangul_block_v2/{img/, pairs.jsonl, stats.json,
sheet.png}``.

    .venv/bin/python project/lora_blocks/src/build_hangul_v2.py [--pages 600]
"""

import argparse
import functools
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from anime_tools.captions.layout_json import centered_box
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from library.env import resolve_under_home

STAGING = resolve_under_home("post_image_dataset/i2i/swap_text_v3/staging")
OUT = resolve_under_home("post_image_dataset/i2i/hangul_block_v2")
FONTS = resolve_under_home("project/cjk_anima_reseed/assets/fonts")
FACES = {
    "ko_extra/Pretendard-Bold.ttf": 3,
    "ko_extra/GangwonEduAll-Bold.ttf": 2,
    "ko_extra/KMU80TTFHaeongSans.ttf": 2,
    "ko_extra/Pretendard-Black.ttf": 1,
    "kozh/DoHyeon-Regular.ttf": 1,
    "kozh/Jua-Regular.ttf": 1,
}
WHITE = 200
INK = 128
MAX_BALLOON = 0.06
P_LETTER = 0.7
P_OWN = 0.5
KEEP = re.compile(r"[가-힣 !?.,~…\-]")


def clean(s: str) -> str:
    s = "".join(c if KEEP.match(c) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def nonspace(s: str) -> int:
    return len(s.replace(" ", ""))


def box_px(item, size):
    w, h = size
    y0, x0, y1, x1 = item["bbox"]
    return (
        max(0, round(x0 + w / 2)),
        max(0, round(y0 + h / 2)),
        min(w, round(x1 + w / 2)),
        min(h, round(y1 + h / 2)),
    )


def balloon(luma: np.ndarray, box) -> np.ndarray | None:
    """The filled balloon around ``box``, or None if it is not a closed white balloon."""
    x0, y0, x1, y1 = box
    h, w = luma.shape
    white = luma >= WHITE
    ring = np.zeros_like(white)
    a0, b0, a1, b1 = max(0, x0 - 3), max(0, y0 - 3), min(w, x1 + 3), min(h, y1 + 3)
    ring[b0:b1, a0:a1] = True
    ring[y0:y1, x0:x1] = False
    seeds = ring & white
    if seeds.sum() < 0.6 * ring.sum():
        return None
    lab, _ = ndimage.label(white)
    ids = np.unique(lab[seeds])
    comp = np.isin(lab, ids[ids > 0])
    area = comp.sum()
    if area > MAX_BALLOON * h * w:
        return None
    filled = ndimage.binary_fill_holes(comp)
    if filled.sum() < 1.2 * (x1 - x0) * (y1 - y0) or not filled[y0:y1, x0:x1].all():
        return None
    return filled


def layout(ink: np.ndarray):
    """(lines, glyph height, pitch) of the ink rows in a box."""
    rows = ink.sum(1) > 0
    runs, start = [], None
    for i, r in enumerate(list(rows) + [False]):
        if r and start is None:
            start = i
        elif not r and start is not None:
            runs.append((start, i))
            start = None
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged = [(a, b) for a, b in merged if b - a >= 4]
    if not merged:
        return None
    heights = sorted(b - a for a, b in merged)
    gh = heights[len(heights) // 2]
    pitch = (
        (merged[-1][0] - merged[0][0]) / (len(merged) - 1)
        if len(merged) > 1
        else gh * 1.3
    )
    return len(merged), gh, max(pitch, gh * 1.05)


def wrap(text: str, n: int) -> list[str] | None:
    if n == 1:
        return [text]
    words = text.split(" ")
    if len(words) < n:
        chars = text.replace(" ", "")
        if len(chars) < n:
            return None
        step = len(chars) / n
        return [chars[round(i * step) : round((i + 1) * step)] for i in range(n)]
    total = len(text)
    # balanced split at word boundaries: dynamic programming on line lengths
    m = len(words)
    lens = [len(w) for w in words]

    @functools.lru_cache(None)
    def go(i, k):
        if k == 1:
            ln = sum(lens[i:]) + (m - i - 1)
            return ((ln - total / n) ** 2, (m,))
        out = None
        for j in range(i + 1, m - k + 2):
            ln = sum(lens[i:j]) + (j - i - 1)
            cost, rest = go(j, k - 1)
            c = (ln - total / n) ** 2 + cost
            if out is None or c < out[0]:
                out = (c, (j,) + rest)
        return out

    _, cuts = go(0, n)
    lines, i = [], 0
    for j in cuts:
        lines.append(" ".join(words[i:j]))
        i = j
    return lines


class Face:
    def __init__(self, path: Path):
        self.path = path
        self.cmap = set(TTFont(str(path), fontNumber=0).getBestCmap())
        f = ImageFont.truetype(str(path), 100)
        b = f.getbbox("한글을", anchor="ls")
        self.ink_ratio = (b[3] - b[1]) / 100.0

    def covers(self, s: str) -> bool:
        return all(ord(c) in self.cmap for c in s if c != " ")


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
    pool = set()
    for r in rows:
        for it in json.loads((STAGING / r["caption"]).read_text())["items"]:
            s = clean(it.get("text", ""))
            if 2 <= nonspace(s) <= 40:
                pool.add(s)
    by_len: dict[int, list[str]] = {}
    for s in sorted(pool):
        by_len.setdefault(nonspace(s), []).append(s)
    faces = [(Face(FONTS / f), w) for f, w in FACES.items()]
    print(f"{len(rows)} ko twins, {len(pool)} lines, {len(faces)} faces")

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rng.shuffle(rows)
    out_rows, syl, stats = [], Counter(), Counter()
    for r in rows:
        if len(out_rows) >= args.pages:
            break
        ja = Image.open(STAGING / r["cond"]).convert("RGB")
        ko = Image.open(STAGING / r["target"]).convert("RGB")
        if ja.size != ko.size:
            stats["size_mismatch"] += 1
            continue
        base_name = rng.choice(("ja", "ko"))
        base = ja if base_name == "ja" else ko
        ko_l = np.asarray(ko.convert("L"))
        base_l = np.asarray(base.convert("L"))
        face = rng.choices([f for f, _ in faces], [w for _, w in faces])[0]
        cap = json.loads((STAGING / r["caption"]).read_text())

        plans = []
        for it in cap["items"]:
            own = clean(it.get("text", ""))
            if nonspace(own) < 1:
                continue
            box = box_px(it, ko.size)
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                continue
            bk = balloon(ko_l, box)
            bb = balloon(base_l, box) if base_name == "ja" else bk
            if bk is None or bb is None:
                stats["not_balloon"] += 1
                continue
            x0, y0, x1, y1 = box
            ink = (ko_l[y0:y1, x0:x1] < INK) & bk[y0:y1, x0:x1]
            lay = layout(ink)
            rows_hit = np.nonzero(ink.any(1))[0]
            if (
                lay is None
                or ink.mean() < 0.06
                or rows_hit[-1] - rows_hit[0] < 0.6 * (y1 - y0)
            ):
                stats["no_ink"] += 1
                continue
            plans.append((it, own, box, bb, ink, lay, bk))
        # a balloon holding more than one OCR line is left as it is
        centres = []
        for it in cap["items"]:
            x0, y0, x1, y1 = box_px(it, ko.size)
            centres.append(((y0 + y1) // 2, (x0 + x1) // 2))
        solo = []
        for p in plans:
            bk = p[6]
            if (
                sum(
                    bool(bk[min(cy, bk.shape[0] - 1), min(cx, bk.shape[1] - 1)])
                    for cy, cx in centres
                )
                == 1
            ):
                solo.append(p[:6])
            else:
                stats["shared_balloon"] += 1
        plans = solo
        if not plans:
            continue
        chosen = [p for p in plans if rng.random() < P_LETTER] or [rng.choice(plans)]
        chosen = chosen[:8]

        page_blank = base.copy()
        page_tgt = base.copy()
        items = []
        for it, own, box, bb, ink, (n, gh, pitch) in chosen:
            if rng.random() < P_OWN and face.covers(own):
                s = own
            else:
                n0 = nonspace(own)
                lens = [k for k in by_len if 0.75 * n0 <= k <= max(1.25 * n0, n0 + 1)]
                cands = [s for k in lens for s in by_len[k]]
                rng.shuffle(cands)
                s = next((c for c in cands[:50] if face.covers(c)), own)
                if not face.covers(s):
                    stats["face_gap"] += 1
                    continue
            lines = wrap(s, min(n, nonspace(s)))
            if not lines:
                stats["wrap"] += 1
                continue
            fs = max(8, round(1.05 * gh / face.ink_ratio))
            ys, xs = np.nonzero(bb)
            bw = xs.max() - xs.min()
            while fs > 8:
                font = ImageFont.truetype(str(face.path), fs)
                if max(font.getlength(ln) for ln in lines) <= 0.9 * bw:
                    break
                fs -= 1
            font = ImageFont.truetype(str(face.path), fs)
            pitch_s = pitch * fs / max(gh / face.ink_ratio, 1)

            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            mask = Image.new("L", page_tgt.size, 0)
            dm = ImageDraw.Draw(mask)
            spots = [
                (cy + (i - (len(lines) - 1) / 2) * pitch_s, ln)
                for i, ln in enumerate(lines)
            ]
            for y, ln in spots:
                dm.text((cx, y), ln, font=font, fill=255, anchor="mm")
            m = np.asarray(mask) > 0
            if (m & ~bb).sum() > 0.02 * max(m.sum(), 1):
                stats["spill"] += 1
                continue
            base_rgb = np.asarray(base)
            ground = np.median(base_rgb[bb & (base_l >= WHITE)], axis=0).astype(
                np.uint8
            )
            fill = np.asarray(page_blank).copy()
            fill[bb] = ground
            page_blank = Image.fromarray(fill)
            tfill = np.asarray(page_tgt).copy()
            tfill[bb] = ground
            page_tgt = Image.fromarray(tfill)
            dt = ImageDraw.Draw(page_tgt)
            for y, ln in spots:
                dt.text((cx, y), ln, font=font, fill=(12, 12, 12), anchor="mm")
            items.append(
                {
                    "label": "text",
                    "caption": "text",
                    "text": " ".join(lines) if " " in s else "".join(lines),
                    "bbox": centered_box(mask.getbbox(), page_tgt.size),
                }
            )
            syl.update(c for c in s if "가" <= c <= "힣")
            stats["lines"] += 1
            stats["own" if s == own else "other"] += 1
        if not items:
            continue
        mode = rng.choice(("blank", "orig"))
        stem = f"hg2_{r['stem']}"
        page_tgt.save(OUT / "img" / f"{stem}.tgt.png")
        (page_blank if mode == "blank" else base).save(OUT / "img" / f"{stem}.src.png")
        new_cap = {
            "summary": {"caption": "comic, korean text"},
            "items": items,
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        }
        (OUT / "img" / f"{stem}.txt").write_text(
            json.dumps(new_cap, ensure_ascii=False, separators=(",", ":"))
        )
        out_rows.append(
            {
                "stem": stem,
                "target": f"img/{stem}.tgt.png",
                "cond": f"img/{stem}.src.png",
                "caption": f"img/{stem}.txt",
                "base": base_name,
                "mode": mode,
                "face": face.path.name,
                "lettered": len(items),
                "balloons": len(plans),
            }
        )
        stats[f"base_{base_name}"] += 1
        stats[f"mode_{mode}"] += 1

    (OUT / "pairs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    )
    stats["syllables"] = len(syl)
    (OUT / "stats.json").write_text(
        json.dumps(
            {"stats": dict(stats), "faces": Counter(r["face"] for r in out_rows)},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"{len(out_rows)} pairs -> {OUT}")
    print(dict(stats))

    tiles = []
    for r in out_rows[:4]:
        a = Image.open(OUT / r["cond"])
        b = Image.open(OUT / r["target"])
        t = Image.new("RGB", (a.width * 2, a.height), "white")
        t.paste(a, (0, 0))
        t.paste(b, (a.width, 0))
        tiles.append(t.resize((t.width // 2, t.height // 2)))
    if tiles:
        sheet = Image.new("RGB", (tiles[0].width * 2, tiles[0].height * 2), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 2) * t.width, (i // 2) * t.height))
        sheet.save(OUT / "sheet.png")


if __name__ == "__main__":
    main()
