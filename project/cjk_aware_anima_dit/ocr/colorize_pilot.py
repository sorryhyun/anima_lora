#!/usr/bin/env python3
"""O3 colorize pilot gate (``plan_ocr.md`` O3, "its own gate, before the GPU hours").

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/cjk_aware_anima_dit/ocr/colorize_pilot.py --name 1024_half_comic"
    … --name 1024 --reader vl16 --ckpt output/ocr/vl16_tower_lr1e-5/best        # the B′ reader instead of stock manga-ocr

Over every page under ``derived/colorized/<name>/`` (the pilot's output), cut
the **same** COO + speech polygons from the source spread and the colorized
copy (``deskew_crop``, 12 % pad, min side 16) and report:

(a) **stroke-mask IoU** — Otsu-binarised dark pixels of the grey source crop
    vs the grey-converted colorized crop, plain and after a 1-px dilation of
    both (tolerates sub-pixel drift from the tier round-trip). Glyphs that the
    repaint redrew show up as a low IoU; the per-crop threshold is set here.
(b) **read agreement** — the reader on source vs colorized crops of the same
    polygon: exact-agreement rate between the two reads, beside each read's
    exact / sim against the COO or ``<text>`` label. The plan's clause: the
    two reads must agree at (about) the rate the source read agrees with the
    label; a colorized read that also scores as well against the label as the
    source read does is the cleaner form of the same test.

Writes ``reports/ocr_colorize_pilot_<name>[_<reader>].md`` (numbers only) and a
side-by-side crop sheet + two full-page pairs under ``derived/colorized/<name>/pilot/``
(Manga109-s derivatives stay out of the tree).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402


from build_manga109_crops import dilate, iou, stroke_mask  # noqa: E402


def sheet(pairs, reads, path: Path, n: int = 32, cell_h: int = 96):
    """Side-by-side (source | colorized) crops, one row per line, with both reads."""
    rows = []
    for (src, col), (rs, rc, gt) in list(zip(pairs, reads))[:n]:
        tiles = []
        for c in (src, col):
            h, w = c.shape[:2]
            tiles.append(cv2.resize(c, (max(1, int(w * cell_h / h)), cell_h)))
        row = np.full((cell_h, 520 + 900, 3), 255, np.uint8)
        x = 0
        for t in tiles:
            t = t[:, : min(t.shape[1], 250)]
            row[:, x : x + t.shape[1]] = t
            x += 260
        rows.append((row, f"gt {gt} | src {rs} | col {rc}"))
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18
        )
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (1420, cell_h * len(rows)), "white")
    d = ImageDraw.Draw(canvas)
    for i, (row, text) in enumerate(rows):
        canvas.paste(Image.fromarray(row[:, :, ::-1]), (0, i * cell_h))
        d.text((530, i * cell_h + 30), text, fill="black", font=font)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="1024_half_comic", help="colorized/<name> subdir")
    ap.add_argument("--reader", default="manga_ocr", choices=sorted(ev.READERS))
    ap.add_argument("--ckpt")
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--min_side", type=int, default=16)
    ap.add_argument("--iou_thr", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()

    mt = m109.pilot_manga_text()
    root = m109.derived_root() / "colorized" / a.name
    pilot = root / "pilot"
    pilot.mkdir(exist_ok=True)
    pages = sorted(
        (p.parent.name, int(p.stem)) for p in root.glob("*/[0-9][0-9][0-9].png")
    )
    if not pages:
        raise SystemExit(f"no colorized pages under {root}")
    print(f"{len(pages)} colorized pages under {root}", flush=True)

    rows, pairs = [], []
    for book, page in pages:
        src = cv2.imread(str(m109.page_path(book, page)))
        col = cv2.imread(str(root / book / f"{page:03d}.png"))
        assert src is not None and col is not None, (book, page)
        assert src.shape[:2] == col.shape[:2], (book, page, src.shape, col.shape)
        lines = [ln for ln in m109.iter_coo(book) if ln.page == page] + [
            ln for ln in m109.iter_text(book) if ln.page == page
        ]
        for ln in lines:
            if not ln.text:
                continue
            cs, orient = mt.deskew_crop(src, list(ln.poly), a.pad, a.min_side)
            cc, _ = mt.deskew_crop(col, list(ln.poly), a.pad, a.min_side)
            if cs is None or cc is None:
                continue
            ms, mc = stroke_mask(cs), stroke_mask(cc)
            rows.append(
                dict(
                    book=book,
                    page=page,
                    id=ln.id,
                    kind=ln.kind,
                    text=ln.text,
                    orient=orient,
                    w=cs.shape[1],
                    h=cs.shape[0],
                    ink_src=float(ms.mean()),
                    ink_col=float(mc.mean()),
                    iou=iou(ms, mc),
                    iou_d1=iou(dilate(ms), dilate(mc)),
                )
            )
            pairs.append((cs, cc))
    df = pd.DataFrame(rows)
    print(f"{len(df)} crops ({df.kind.value_counts().to_dict()})", flush=True)

    reader = ev.READERS[a.reader](a.ckpt, a.device)
    t0 = time.time()
    orients = list(df.orient)
    pred_src = reader.read([p[0] for p in pairs], orients, a.bs)
    pred_col = reader.read([p[1] for p in pairs], orients, a.bs)
    wall = time.time() - t0
    s_src = ev.score(df, pred_src)
    s_col = ev.score(df, pred_col)
    df["pred_src"], df["pred_col"] = s_src.pred_norm, s_col.pred_norm
    df["exact_src"], df["exact_col"] = s_src.exact, s_col.exact
    df["sim_src"], df["sim_col"] = s_src.sim, s_col.sim
    df["agree"] = [
        ev.exact_key(x) == ev.exact_key(y)
        for x, y in zip(s_src.pred_norm, s_col.pred_norm)
    ]
    name = a.name + (f"_{a.reader}" if a.reader != "manga_ocr" else "")
    df.to_json(
        pilot / f"pilot_{name}.jsonl", orient="records", lines=True, force_ascii=False
    )

    md = [
        f"# O3 colorize pilot — `{a.name}`, reader `{a.reader}`"
        + (f" (`{a.ckpt}`)" if a.ckpt else "")
        + "\n",
        f"{len(pages)} pages, {len(df)} crops; reader wall {wall:.0f} s for {2 * len(df)} crops.\n",
        "## (a) stroke-mask IoU, source vs colorized crop\n",
        "| kind | n | IoU mean | IoU p10 | IoU d1 mean | IoU d1 p10 | d1 ≥ thr | ink src | ink col |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, g in df.groupby("kind"):
        md.append(
            f"| {k} | {len(g)} | {g.iou.mean():.3f} | {g.iou.quantile(0.1):.3f} | "
            f"{g.iou_d1.mean():.3f} | {g.iou_d1.quantile(0.1):.3f} | "
            f"{100 * (g.iou_d1 >= a.iou_thr).mean():.1f} % | {g.ink_src.mean():.3f} | {g.ink_col.mean():.3f} |"
        )
    md += [
        f"\n(thr = {a.iou_thr}; d1 = both masks dilated 1 px)\n",
        "## (b) read agreement, source vs colorized\n",
        "| kind | n | src exact | col exact | src sim | col sim | reads agree | agree given src exact | col exact given src exact |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, g in df.groupby("kind"):
        ok = g[g.exact_src]
        md.append(
            f"| {k} | {len(g)} | {100 * g.exact_src.mean():.1f} % | {100 * g.exact_col.mean():.1f} % | "
            f"{g.sim_src.mean():.3f} | {g.sim_col.mean():.3f} | {100 * g.agree.mean():.1f} % | "
            f"{100 * ok.agree.mean() if len(ok) else 0:.1f} % | {100 * ok.exact_col.mean() if len(ok) else 0:.1f} % |"
        )
    md.append(
        "\n## Worst 20 crops by IoU (d1)\n\n| book / page / id | kind | gt | src read | col read | IoU d1 |\n|---|---|---|---|---|---|"
    )
    for _, r in df.sort_values("iou_d1").head(20).iterrows():
        md.append(
            f"| {r.book} {r.page:03d} {r.id} | {r.kind} | {r.text[:20]} | {r.pred_src[:20]} | {r.pred_col[:20]} | {r.iou_d1:.2f} |"
        )
    md.append(
        "\n## Source right, colorized wrong\n\n| book / page / id | kind | gt | col read | IoU d1 |\n|---|---|---|---|---|"
    )
    bad = df[df.exact_src & ~df.exact_col]
    for _, r in bad.head(30).iterrows():
        md.append(
            f"| {r.book} {r.page:03d} {r.id} | {r.kind} | {r.text[:20]} | {r.pred_col[:24]} | {r.iou_d1:.2f} |"
        )
    md.append(
        f"\n({len(bad)} such crops; {int((~df.exact_src & df.exact_col).sum())} the other way round)\n"
    )
    rep = m109.LINE / "reports" / f"ocr_colorize_pilot_{name}.md"
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)

    # sheets: SFX first (the lever's target), then the two lowest-IoU pages side by side
    kinds, ious = list(df.kind), list(df.iou_d1)
    order = sorted(
        range(len(df)), key=lambda i: (0 if kinds[i] == "sfx" else 1, ious[i])
    )
    sfx_first = [i for i in order if kinds[i] == "sfx"][:24] + [
        i for i in order if kinds[i] == "speech"
    ][:8]
    sheet(
        [pairs[i] for i in sfx_first],
        [
            (df.pred_src.iloc[i], df.pred_col.iloc[i], df.text.iloc[i])
            for i in sfx_first
        ],
        pilot / f"sheet_{name}.png",
    )
    for book, page in pages[:2]:
        src = cv2.imread(str(m109.page_path(book, page)))
        col = cv2.imread(str(root / book / f"{page:03d}.png"))
        cv2.imwrite(
            str(pilot / f"page_{book}_{page:03d}.jpg"),
            np.hstack([src, col]),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
    print(f"sheet → {pilot}", flush=True)


if __name__ == "__main__":
    main()
