"""Plan step 3 data, v4: v3's builder over every Korean twin, with targets whose vowel strokes
stay readable at their size and vowel minimal pairs against the language prior.

Changes from v3 (``build_hangul_v3.py``, reports/hangul_v3_ft.md, the cjk_ko line's data findings):

- **Readable targets.** At the font px a line is drawn at, every Hangul syllable must differ
  from each of its vowel neighbours (``CONFUSE``: ㅏ / ㅓ / ㅔ / ㅐ / ㅕ against ㅣ and each
  other, ㅗ / ㅜ / ㅡ, ㅘ / ㅙ — the swaps on ko10) by at least ``PAIR_K``·px² of anti-aliased
  ink (at 20 px, 10 px of ink). Probe: at 16–20 px, 저 / 지 and 커 / 키 differ by 3–15 px in
  most v3 faces, 4–5 px in Gangwon Edu All Bold (weight 2 in v3); 69 % of v3's lines sit
  under 20 px of ink. From 28 px the stroke-gap rule of cjk_ko's ``legible.py`` (median gap
  ≥ 1.5 px, p10 ≥ 0.8 px) also applies. A line whose page face fails is drawn in a face that
  passes (refit to the box); with none, the face with the widest margin.
- **Vowel minimal pairs** (cjk_ko's counterfactual syllables): with p ``P_PAIR`` a lettered
  string has one syllable swapped for a vowel neighbour (KS X 1001), so the caption and not
  the word decides the vowel (정말 → 정밀).
- **Every Korean twin** (``--pages`` default 2000, the pool is ~1,700): Korean exposure is the
  bet (the user, 2026-11-08: English and Japanese lettering is clean because pretraining saw
  far more of it).

Writes ``post_image_dataset/i2i/hangul_block_v4/{img/, pairs.jsonl, stats.json, sheet.png,
sheet_small.png}``.

    .venv/bin/python project/lora_blocks/src/build_hangul_v4.py [--pages 2000]
"""

import argparse
import functools
import json
import random
from collections import Counter

import numpy as np
from anime_tools.captions.layout_json import centered_box
from build_hangul_v3 import (
    FONTS,
    HANGUL,
    INK,
    INK_DELTA,
    MAX_LEN,
    P_LETTER,
    P_OWN,
    P_TAIL,
    STAGING,
    WHITE,
    Face,
    balloon,
    box_px,
    clean,
    flat_ring,
    ksx1001,
    layout,
    nonspace,
    rect_mask,
    size_tier,
    wrap,
)
from PIL import Image, ImageDraw, ImageFont

from library.env import resolve_under_home

OUT = resolve_under_home("post_image_dataset/i2i/hangul_block_v4")
FACES = {
    "ko_extra/Pretendard-Regular.ttf": 3,
    "kozh/NanumGothic-Regular.ttf": 2,
    "ko_extra/GangwonEduAll-Bold.ttf": 2,
    "ko_extra/Pretendard-Bold.ttf": 1,
    "ko_extra/KMU80TTFHaeongSans.ttf": 1,
}
P_PAIR = 0.3
PAIR_K = 0.025
GAP_RULE = {"median": 1.5, "p10": 0.8, "from_px": 28}
# medial index (0 ㅏ … 20 ㅣ) -> the medials it is drawn as when its short strokes go
CONFUSE = {
    0: (20, 4),
    1: (5, 20),
    4: (20, 6, 0),
    5: (1, 20),
    6: (4, 20),
    20: (0, 4, 5, 1),
    8: (18, 13),
    13: (18, 8),
    18: (8, 13),
    9: (10,),
    10: (9,),
}


def neighbours(c: str) -> list[str]:
    o = ord(c) - 0xAC00
    if not 0 <= o < 11172:
        return []
    cho, jung, jong = o // 588, (o % 588) // 28, o % 28
    return [chr(0xAC00 + cho * 588 + v * 28 + jong) for v in CONFUSE.get(jung, ())]


@functools.lru_cache(maxsize=None)
def glyph(path: str, c: str, px: int) -> np.ndarray:
    font = ImageFont.truetype(path, px)
    im = Image.new("L", (px * 2, px * 2), 0)
    ImageDraw.Draw(im).text((px // 2, px // 2), c, font=font, fill=255)
    return np.asarray(im, np.float32) / 255


@functools.lru_cache(maxsize=None)
def pair_diff(path: str, c: str, px: int) -> float:
    """Least ink difference (px) between ``c`` and its vowel neighbours at ``px``; inf with none."""
    nb = neighbours(c)
    if not nb:
        return float("inf")
    g = glyph(path, c, px)
    return min(float(np.abs(g - glyph(path, n, px)).sum()) for n in nb)


@functools.lru_cache(maxsize=None)
def gaps(path: str, c: str) -> tuple:
    """``(p10, median)`` stroke gaps as shares of the font px (cjk_ko ``reseed/legible.py``)."""
    ref = 256
    font = ImageFont.truetype(path, ref, index=0)
    bb = font.getbbox(c)
    im = Image.new("L", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), 0)
    ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), c, font=font, fill=255)
    a = np.asarray(im) > 127
    runs = []
    for m in (a, a.T):
        for r in m:
            ink = np.flatnonzero(r)
            if len(ink) < 2:
                continue
            d = np.diff(r[ink[0] : ink[-1] + 1].astype(np.int8))
            runs.append(np.flatnonzero(d == 1) - np.flatnonzero(d == -1))
    g = np.concatenate(runs) / ref if runs else np.zeros(0)
    if not len(g):
        return (1.0, 1.0)
    return (float(np.percentile(g, 10)), float(np.median(g)))


def margin(face: Face, s: str, px: int) -> float:
    """How far ``s`` at ``px`` clears the readability rules (≥ 1 passes)."""
    path = str(face.path)
    syl = set(HANGUL.findall(s))
    m = min((pair_diff(path, c, px) for c in syl), default=float("inf")) / (
        PAIR_K * px * px
    )
    if px >= GAP_RULE["from_px"]:
        for c in syl:
            p10, med = gaps(path, c)
            m = min(m, med * px / GAP_RULE["median"], p10 * px / GAP_RULE["p10"])
    return m


def fit(face: Face, lines: list[str], gh: int, bw: int) -> int:
    fs = max(8, round(1.05 * gh / face.ink_ratio))
    while fs > 8:
        font = ImageFont.truetype(str(face.path), fs)
        if max(font.getlength(ln) for ln in lines) <= 0.9 * bw:
            break
        fs -= 1
    return fs


def swap_pair(s: str, rng: random.Random, common: set, face: Face):
    pos = [i for i, c in enumerate(s) if HANGUL.match(c)]
    rng.shuffle(pos)
    for i in pos:
        nb = [n for n in neighbours(s[i]) if n in common and face.covers(n)]
        if nb:
            n = rng.choice(nb)
            return s[:i] + n + s[i + 1 :], (s[i], n)
    return s, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2000)
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
    common = set(ksx1001())
    tail = [c for c in sorted(common) if c not in seen]
    by_len: dict[int, list[str]] = {}
    for s in sorted(pool):
        by_len.setdefault(nonspace(s), []).append(s)
    faces = [(Face(FONTS / f), w) for f, w in FACES.items()]
    print(
        f"{len(rows)} ko twins, {len(pool)} lines, {len(tail)} tail syllables, "
        f"{len(faces)} faces",
        flush=True,
    )

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rng.shuffle(rows)
    out_rows, syl, stats = [], Counter(), Counter()
    tail_used, pair_used, line_faces = Counter(), Counter(), Counter()
    small = []
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
        items, page_small = [], []
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
            swapped = paired = None
            if rng.random() < P_PAIR:
                s, paired = swap_pair(s, rng, common, face)
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
            ys, xs = np.nonzero(region)
            bw = xs.max() - xs.min()

            lf = face
            fs = fit(lf, lines, gh, bw)
            if margin(lf, s, fs) < 1:
                tried = []
                for f, _ in faces:
                    if f is face or not f.covers(s):
                        continue
                    fs_f = fit(f, lines, gh, bw)
                    tried.append((margin(f, s, fs_f), f, fs_f))
                ok = [t for t in tried if t[0] >= 1]
                if ok:
                    _, lf, fs = rng.choice(ok)
                    stats["repick"] += 1
                elif tried and max(t[0] for t in tried) > margin(face, s, fs):
                    _, lf, fs = max(tried, key=lambda t: t[0])
                    stats["repick_best"] += 1
                else:
                    stats["kept_below"] += 1
            font = ImageFont.truetype(str(lf.path), fs)
            pitch_s = pitch * fs / max(gh / lf.ink_ratio, 1)

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
            if gh < 20:
                page_small.append((mask.getbbox(), lf.path.name, fs, paired))
            syl.update(HANGUL.findall(s))
            if swapped:
                tail_used[swapped] += 1
            if paired:
                pair_used[f"{paired[0]}>{paired[1]}"] += 1
                stats["pair_swaps"] += 1
            line_faces[lf.path.name] += 1
            stats["lines"] += 1
            stats[f"kind_{kind}"] += 1
            stats[f"align_{align}"] += 1
            stats[f"px_{size_tier(gh)}"] += 1
            stats["dark" if ground_l < 110 else "light"] += 1
            stats["own" if s == own else "other"] += 1
        if not items:
            continue
        mode = rng.choice(("blank", "orig"))
        stem = f"hg4_{r['stem']}"
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
        small += [(stem, *x) for x in page_small]
        stats[f"base_{base_name}"] += 1
        stats[f"mode_{mode}"] += 1
        if len(out_rows) % 200 == 0:
            print(f"{len(out_rows)} pages, {stats['lines']} lines", flush=True)

    (OUT / "pairs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    )
    stats["syllables"] = len(syl)
    stats["tail_syllables"] = len(tail_used)
    (OUT / "stats.json").write_text(
        json.dumps(
            {
                "stats": dict(stats),
                "page_faces": Counter(r["face"] for r in out_rows),
                "line_faces": dict(line_faces),
                "pairs": dict(pair_used.most_common()),
                "tail": dict(tail_used),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"{len(out_rows)} pairs -> {OUT}")
    print(dict(stats))

    # small lines at 2x: target crop over the source crop, the face and px under each
    pick = [x for x in small if x[4]][:12] + [x for x in small if not x[4]][:12]
    tiles = []
    for stem, bb, fname, fs, paired in pick:
        x0, y0, x1, y1 = bb
        x0, y0, x1, y1 = x0 - 6, y0 - 6, x1 + 6, y1 + 6
        t = Image.open(OUT / "img" / f"{stem}.tgt.png").crop((x0, y0, x1, y1))
        t = t.resize((t.width * 2, t.height * 2), Image.NEAREST)
        lab = f"{fname[:14]} {fs}px" + (f" {paired[0]}>{paired[1]}" if paired else "")
        tile = Image.new("RGB", (max(t.width, 260), t.height + 24), "white")
        tile.paste(t, (0, 0))
        ImageDraw.Draw(tile).text(
            (2, t.height + 2),
            lab,
            fill=(200, 0, 0),
            font=ImageFont.truetype(str(FONTS / "ko_extra/Pretendard-Regular.ttf"), 16),
        )
        tiles.append(tile)
    if tiles:
        cols = 3
        cw = max(t.width for t in tiles)
        rh = [
            max(t.height for t in tiles[i : i + cols]) for i in range(0, len(tiles), cols)
        ]
        sheet = Image.new("RGB", (cw * cols, sum(rh)), "white")
        y = 0
        for ri, i in enumerate(range(0, len(tiles), cols)):
            for j, t in enumerate(tiles[i : i + cols]):
                sheet.paste(t, (j * cw, y))
            y += rh[ri]
        sheet.save(OUT / "sheet_small.png")


if __name__ == "__main__":
    main()
