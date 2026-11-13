#!/usr/bin/env python3
"""Contact sheets for the region staging tree — how each mask became a paint.

One row per image: ``target | mask overlay | cond (paint)`` with a footer line
(slice · augment level · position word · coverages · the positioned caption
clause when one exists). The overlay tints the girl mask red, the partner
(boy) mask blue, the face mask green, and outlines the paint in yellow, so a
gate/paint mistake is visible at a glance. Sheets are grouped ``<slice>-<level>``
(``solo-slack``, ``pair-exact`` …) with ``--per_sheet`` rows each, sampled with
a fixed seed, plus a ``drops`` sheet of gate-failed images (their raw masks,
reason in the footer). CPU only; run any time after the cond stage::

    python i2i_tasks/region/contact_sheet.py [--base …] [--per_sheet 8]

Output: ``{base}/contact_sheets/<group>[-N].png`` + ``index.html``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = 320
FOOTER = 34
GAP = 6


def _font(size: int = 13):
    for name in ("DejaVuSans.ttf", "NotoSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(img: np.ndarray, tile: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = tile / max(h, w)
    out = cv2.resize(
        img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((tile, tile, 3), 24, np.uint8)
    y, x = (tile - out.shape[0]) // 2, (tile - out.shape[1]) // 2
    canvas[y : y + out.shape[0], x : x + out.shape[1]] = out
    return canvas


def _mask(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    m = (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)
    if m.shape != (size[1], size[0]):
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return m


def _overlay(
    rgb: np.ndarray, masks: dict[str, np.ndarray | None], paint: np.ndarray | None
) -> np.ndarray:
    out = rgb.astype(np.float32) * 0.55
    colors = {
        "girl": (255, 60, 60),
        "boy": (60, 120, 255),
        "face": (60, 255, 90),
        "head": (60, 255, 90),
    }
    for name, m in masks.items():
        if m is None:
            continue
        col = np.array(colors[name], np.float32)
        out[m > 0] = out[m > 0] * 0.5 + col * 0.5
    out = out.clip(0, 255).astype(np.uint8)
    if paint is not None:
        contours, _ = cv2.findContours(
            paint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (255, 230, 0), max(2, out.shape[1] // 300))
    return out


def _paint_mask(cond: np.ndarray, paint_color: tuple[int, int, int]) -> np.ndarray:
    diff = np.abs(cond.astype(np.int16) - np.array(paint_color, np.int16)).max(axis=2)
    return (diff <= 2).astype(np.uint8)


def _clause(staging: Path, key: str) -> str:
    sidecar = staging / f"{key}.variants.txt"
    if not sidecar.is_file():
        return ""
    for raw in sidecar.read_text(encoding="utf-8").splitlines():
        if ". On the " in raw:
            return raw.split(". ", 1)[1].rstrip()
    return ""


def _row(
    base: Path, rec: dict, paint_color, *, font, drop_reason: str | None = None
) -> Image.Image:
    key = rec["image"]
    rel = Path(key)
    staging = base / "staging"
    img_path = staging / f"{key}.png"
    src = img_path if img_path.exists() else base / "resized_fallback" / f"{key}.png"
    if not src.exists():
        # dropped images have their symlink pruned; read the resized original
        src = Path("post_image_dataset/resized") / f"{key}.png"
    rgb = np.array(Image.open(src).convert("RGB"))
    size = (rgb.shape[1], rgb.shape[0])
    mname = f"{rel.stem}_mask.png"
    masks = {
        "girl": _mask(base / "masks" / rel.parent / mname, size),
        "boy": _mask(base / "masks_boy" / rel.parent / mname, size),
        "face": _mask(base / "masks_head" / rel.parent / mname, size)
        if (base / "masks_head").is_dir()
        else _mask(base / "masks_face" / rel.parent / mname, size),
    }
    cond_path = base / "cond_images" / f"{key}.png"
    cond = (
        np.array(Image.open(cond_path).convert("RGB")) if cond_path.exists() else None
    )
    paint = _paint_mask(cond, paint_color) if cond is not None else None
    tiles = [_fit(rgb, TILE), _fit(_overlay(rgb, masks, paint), TILE)]
    tiles.append(
        _fit(cond, TILE) if cond is not None else np.full((TILE, TILE, 3), 24, np.uint8)
    )
    strip = np.full((TILE + FOOTER, 3 * TILE + 2 * GAP, 3), 16, np.uint8)
    for i, t in enumerate(tiles):
        x = i * (TILE + GAP)
        strip[:TILE, x : x + TILE] = t
    im = Image.fromarray(strip)
    d = ImageDraw.Draw(im)
    if drop_reason:
        text = f"{key}  DROPPED: {drop_reason}"
    else:
        text = (
            f"{key}  {rec['slice']}·{rec['level']}  pos={rec['position']}  "
            f"girl={rec['girl_coverage']:.2f} paint={rec['paint_coverage']:.2f} "
            f"girl/paint={rec['girl_in_paint_area_ratio']:.2f}"
        )
        if rec.get("partner_visible") is not None:
            text += f" boy_visible={rec['partner_visible']:.2f}"
        clause = _clause(staging, key)
        if clause:
            text += f"   «{clause}»"
    d.text((4, TILE + 4), text[:170], fill=(230, 230, 230), font=font)
    d.text(
        (4, TILE + 19),
        "target | masks (girl red · boy blue · face green · paint outline) | cond",
        fill=(140, 140, 140),
        font=font,
    )
    return im


def _sheet(rows: list[Image.Image], title: str, font) -> Image.Image:
    w = rows[0].width
    h = 28 + sum(r.height + GAP for r in rows)
    sheet = Image.new("RGB", (w, h), (10, 10, 10))
    ImageDraw.Draw(sheet).text((6, 6), title, fill=(255, 255, 255), font=font)
    y = 28
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + GAP
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="post_image_dataset/i2i/region")
    ap.add_argument("--per_sheet", type=int, default=8)
    ap.add_argument("--max_sheets_per_group", type=int, default=1)
    ap.add_argument("--paint_color", nargs=3, type=int, default=[128, 128, 128])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default {base}/contact_sheets")
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out) if args.out else base / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))
    font = _font()
    rng = random.Random(args.seed)
    paint_color = tuple(args.paint_color)

    records = report["records"]
    drops = report.get("drops", [])
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[f"{rec['slice']}-{rec['level']}"].append(rec)
    index: list[tuple[str, str, int]] = []
    for name in sorted(groups):
        recs = groups[name]
        rng.shuffle(recs)
        n_sheets = min(
            args.max_sheets_per_group, max(1, -(-len(recs) // args.per_sheet))
        )
        for si in range(n_sheets):
            chunk = recs[si * args.per_sheet : (si + 1) * args.per_sheet]
            if not chunk:
                break
            rows = [_row(base, r, paint_color, font=font) for r in chunk]
            fname = f"{name}.png" if n_sheets == 1 else f"{name}-{si}.png"
            _sheet(rows, f"{name}  (n={len(recs)})", font).save(out_dir / fname)
            index.append((name, fname, len(recs)))
            print(f"{fname}: {len(chunk)} rows of {len(recs)}")
    if drops:
        rng.shuffle(drops)
        rows = [
            _row(
                base,
                {"image": d["image"]},
                paint_color,
                font=font,
                drop_reason=d["reason"],
            )
            for d in drops[: args.per_sheet]
        ]
        _sheet(rows, f"drops  (n={len(drops)}: {report['dropped']})", font).save(
            out_dir / "drops.png"
        )
        index.append(("drops", "drops.png", len(drops)))
    html = ["<html><body style='background:#111;color:#ddd;font-family:sans-serif'>"]
    html.append(
        f"<h2>region staging — {report['kept']} kept, levels {report['levels']}, dropped {report['dropped']}</h2>"
    )
    for name, fname, n in index:
        html.append(
            f"<h3>{name} (n={n})</h3><img src='{fname}' style='max-width:100%'>"
        )
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"→ {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
