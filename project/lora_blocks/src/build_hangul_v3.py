"""Plan step 3 data, v3: v2's edition-laid lettering plus narration boxes, lighter faces and
the syllables swap_text's data never held.

Changes from v2 (``build_hangul_v2.py``, reports/hangul_ko10.md):

- **Flat boxes beside balloons.** A box that is not a closed balloon is used when the
  ring around it (box + margin, the box left out) is one flat ground on both sides of the
  twin: ≥ 92 % of the ring within ±20 luma of its median, and no other OCR box inside the
  margin. The box + margin is filled with the ring's ground colour. This takes the
  rectangular narration boxes and captions on flat panels v2 rejected, dark ones included
  (the ink is then light).
- **Long strings.** Lines up to 120 non-space syllables (v2: 40), so a narration box gets a
  narration-length string.
- **Alignment** read off the edition: left-aligned when the lines' left edges line up and
  their right edges do not, centred otherwise.
- **Faces**: Regular / Medium gothics first (Pretendard Regular, Nanum Gothic, Gangwon Edu
  All Bold), Pretendard Bold and KMU HaeongSans at a lower share. v2's bold faces dropped
  the short vowel strokes at small sizes (ㅏ / ㅓ / ㅔ read as ㅣ).
- **Tail syllables**: with p 0.2 a lettered string has one syllable swapped for a KS X 1001
  syllable that swap_text v3's Korean captions never hold (``끜`` came out ``끔``).

Writes ``post_image_dataset/i2i/hangul_block_v3/{img/, pairs.jsonl, stats.json,
sheet.png}``.

    .venv/bin/python project/lora_blocks/src/build_hangul_v3.py [--pages 800]
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
OUT = resolve_under_home("post_image_dataset/i2i/hangul_block_v3")
FONTS = resolve_under_home("project/cjk_anima_reseed/assets/fonts")
FACES = {
    "ko_extra/Pretendard-Regular.ttf": 3,
    "kozh/NanumGothic-Regular.ttf": 2,
    "ko_extra/GangwonEduAll-Bold.ttf": 2,
    "ko_extra/Pretendard-Bold.ttf": 1,
    "ko_extra/KMU80TTFHaeongSans.ttf": 1,
}
WHITE = 200
INK = 128
FLAT_TOL = 20
FLAT_SHARE = 0.92
INK_DELTA = 60
MAX_BALLOON = 0.06
MAX_LEN = 120
P_LETTER = 0.7
P_OWN = 0.5
P_TAIL = 0.2
KEEP = re.compile(r"[가-힣 !?.,~…\-]")
HANGUL = re.compile("[가-힣]")


def clean(s: str) -> str:
    s = "".join(c if KEEP.match(c) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def nonspace(s: str) -> int:
    return len(s.replace(" ", ""))


def ksx1001() -> list[str]:
    out = []
    for c in range(0xAC00, 0xD7A4):
        try:
            if len(chr(c).encode("euc_kr")) == 2:
                out.append(chr(c))
        except UnicodeEncodeError:
            pass
    return out


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


def margin_of(box) -> int:
    return max(6, round(0.12 * min(box[2] - box[0], box[3] - box[1])))


def flat_ring(luma: np.ndarray, box, others) -> float | None:
    """The ring's ground luma when the ring around ``box`` is one flat ground and holds no
    other OCR box, else None."""
    x0, y0, x1, y1 = box
    h, w = luma.shape
    m = margin_of(box)
    a0, b0, a1, b1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    for o in others:
        if o[0] < a1 and o[2] > a0 and o[1] < b1 and o[3] > b0:
            return None
    ring = np.ones((b1 - b0, a1 - a0), bool)
    ring[y0 - b0 : y1 - b0, x0 - a0 : x1 - a0] = False
    vals = luma[b0:b1, a0:a1][ring].astype(np.float32)
    if vals.size < 50:
        return None
    g = float(np.median(vals))
    if (np.abs(vals - g) <= FLAT_TOL).mean() < FLAT_SHARE:
        return None
    return g


def rect_mask(shape, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = shape
    m = margin_of(box)
    out = np.zeros(shape, bool)
    out[max(0, y0 - m) : min(h, y1 + m), max(0, x0 - m) : min(w, x1 + m)] = True
    return out


def layout(ink: np.ndarray):
    """(lines, glyph height, pitch, align) of the ink rows in a box."""
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
    align = "centre"
    if len(merged) > 1:
        cols = [np.nonzero(ink[a:b].any(0))[0] for a, b in merged]
        lefts = np.array([c[0] for c in cols], np.float32)
        rights = np.array([c[-1] for c in cols], np.float32)
        if lefts.std() < 0.2 * gh and rights.std() > 2 * lefts.std() + 1:
            align = "left"
    return len(merged), gh, max(pitch, gh * 1.05), align


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


def size_tier(gh: int) -> str:
    return "<20" if gh < 20 else "20-28" if gh < 28 else "28-40" if gh < 40 else "40+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [
        json.loads(ln)
        for ln in (STAGING / "pairs_all.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    rows = [r for r in rows if r["lang"] == "ko"]
    pool, seen = set(), set()
    for r in rows:
        for it in json.loads((STAGING / r["caption"]).read_text())["items"]:
            seen.update(HANGUL.findall(it.get("text", "")))
            s = clean(it.get("text", ""))
            if 2 <= nonspace(s) <= MAX_LEN:
                pool.add(s)
    tail = [c for c in ksx1001() if c not in seen]
    by_len: dict[int, list[str]] = {}
    for s in sorted(pool):
        by_len.setdefault(nonspace(s), []).append(s)
    faces = [(Face(FONTS / f), w) for f, w in FACES.items()]
    print(
        f"{len(rows)} ko twins, {len(pool)} lines, {len(tail)} tail syllables, "
        f"{len(faces)} faces"
    )

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rng.shuffle(rows)
    out_rows, syl, stats = [], Counter(), Counter()
    tail_used = Counter()
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
        boxes = [box_px(it, ko.size) for it in cap["items"]]

        plans = []
        for k, it in enumerate(cap["items"]):
            own = clean(it.get("text", ""))
            if nonspace(own) < 1:
                continue
            box = boxes[k]
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                continue
            x0, y0, x1, y1 = box
            bk = balloon(ko_l, box)
            bb = balloon(base_l, box) if base_name == "ja" else bk
            if bk is not None and bb is not None:
                kind = "balloon"
                ink = (ko_l[y0:y1, x0:x1] < INK) & bk[y0:y1, x0:x1]
                region = bb
                ground_l = float(np.median(base_l[bb & (base_l >= WHITE)]))
            else:
                others = [b for j, b in enumerate(boxes) if j != k]
                gk = flat_ring(ko_l, box, others)
                gb = flat_ring(base_l, box, others) if base_name == "ja" else gk
                if gk is None or gb is None:
                    stats["not_flat"] += 1
                    continue
                kind = "flat"
                ink = np.abs(ko_l[y0:y1, x0:x1].astype(np.float32) - gk) > INK_DELTA
                region = rect_mask(ko_l.shape, box)
                ground_l = gb
            lay = layout(ink)
            rows_hit = np.nonzero(ink.any(1))[0]
            if (
                lay is None
                or ink.mean() < 0.04
                or rows_hit[-1] - rows_hit[0] < 0.6 * (y1 - y0)
            ):
                stats["no_ink"] += 1
                continue
            plans.append((it, own, box, region, lay, kind, ground_l, bk))
        # a balloon holding more than one OCR line is left as it is
        centres = [((b[1] + b[3]) // 2, (b[0] + b[2]) // 2) for b in boxes]
        solo = []
        for p in plans:
            if p[5] == "flat":
                solo.append(p)
                continue
            bk = p[7]
            hits = sum(
                bool(bk[min(cy, bk.shape[0] - 1), min(cx, bk.shape[1] - 1)])
                for cy, cx in centres
            )
            if hits == 1:
                solo.append(p)
            else:
                stats["shared_balloon"] += 1
        plans = solo
        if not plans:
            continue
        chosen = [p for p in plans if rng.random() < P_LETTER] or [rng.choice(plans)]
        chosen = chosen[:8]

        base_rgb = np.asarray(base)
        page_blank = base.copy()
        page_tgt = base.copy()
        items = []
        for it, own, box, region, (n, gh, pitch, align), kind, ground_l, _ in chosen:
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
            swapped = None
            if rng.random() < P_TAIL:
                pos = [i for i, c in enumerate(s) if HANGUL.match(c)]
                cand = [c for c in rng.sample(tail, 20) if face.covers(c)]
                if pos and cand:
                    i = rng.choice(pos)
                    swapped = cand[0]
                    s = s[:i] + swapped + s[i + 1 :]
            lines = wrap(s, min(n, nonspace(s)))
            if not lines:
                stats["wrap"] += 1
                continue
            fs = max(8, round(1.05 * gh / face.ink_ratio))
            ys, xs = np.nonzero(region)
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
            if align == "left":
                width = max(font.getlength(ln) for ln in lines)
                lx = max(x0, min(cx - width / 2, x0 + 2))
                anchor, ax = "lm", lx
            else:
                anchor, ax = "mm", cx
            mask = Image.new("L", page_tgt.size, 0)
            dm = ImageDraw.Draw(mask)
            spots = [
                (cy + (i - (len(lines) - 1) / 2) * pitch_s, ln)
                for i, ln in enumerate(lines)
            ]
            for y, ln in spots:
                dm.text((ax, y), ln, font=font, fill=255, anchor=anchor)
            m = np.asarray(mask) > 0
            if (m & ~region).sum() > 0.02 * max(m.sum(), 1):
                stats["spill"] += 1
                continue
            if kind == "balloon":
                ground = np.median(
                    base_rgb[region & (np.asarray(base.convert("L")) >= WHITE)], axis=0
                ).astype(np.uint8)
            else:
                ring = rect_mask(region.shape, box)
                ring[y0:y1, x0:x1] = False
                ground = np.median(base_rgb[ring], axis=0).astype(np.uint8)
            fill = np.asarray(page_blank).copy()
            fill[region] = ground
            page_blank = Image.fromarray(fill)
            tfill = np.asarray(page_tgt).copy()
            tfill[region] = ground
            page_tgt = Image.fromarray(tfill)
            ink_rgb = (240, 240, 240) if ground_l < 110 else (12, 12, 12)
            dt = ImageDraw.Draw(page_tgt)
            for y, ln in spots:
                dt.text((ax, y), ln, font=font, fill=ink_rgb, anchor=anchor)
            items.append(
                {
                    "label": "text",
                    "caption": "text",
                    "text": " ".join(lines) if " " in s else "".join(lines),
                    "bbox": centered_box(mask.getbbox(), page_tgt.size),
                }
            )
            syl.update(HANGUL.findall(s))
            if swapped:
                tail_used[swapped] += 1
            stats["lines"] += 1
            stats[f"kind_{kind}"] += 1
            stats[f"align_{align}"] += 1
            stats[f"px_{size_tier(gh)}"] += 1
            stats["dark" if ground_l < 110 else "light"] += 1
            stats["own" if s == own else "other"] += 1
        if not items:
            continue
        mode = rng.choice(("blank", "orig"))
        stem = f"hg3_{r['stem']}"
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
                "regions": len(plans),
            }
        )
        stats[f"base_{base_name}"] += 1
        stats[f"mode_{mode}"] += 1

    (OUT / "pairs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    )
    stats["syllables"] = len(syl)
    stats["tail_syllables"] = len(tail_used)
    (OUT / "stats.json").write_text(
        json.dumps(
            {
                "stats": dict(stats),
                "faces": Counter(r["face"] for r in out_rows),
                "tail": dict(tail_used),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"{len(out_rows)} pairs -> {OUT}")
    print(dict(stats))

    flat = [r for r in out_rows if r["lettered"]][:2]
    flat += [
        r
        for r in out_rows
        if "kind_flat" in stats
        and any(
            len(it["text"]) > 30
            for it in json.loads((OUT / r["caption"]).read_text())["items"]
        )
    ][:2]
    tiles = []
    for r in flat[:4]:
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
