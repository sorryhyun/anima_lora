"""Stage ``scenes``, second half — keep / reject each generated scene.

Detector + readers over every image: keep exactly-one-box images whose box
reads the anchor, is at least ``--scene_min_box`` px on its short side and
whose anchor ink the composite erase removes (``--scene_max_residual``). Also
``--scene_rejudge`` (re-apply the rule from stored reads) and the report:
``scenes.jsonl``, ``scenes_all.jsonl``, the kept / rejected sheets, ``report.md``.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

from common.bubble import bubble_bbox, bubble_mask
from common.readers import Readers, contact_sheet, load_bgr
from common.render.scene import (
    anchor_residual,
    erase_lost,
    erase_uniform,
    region_offset,
)
from common.text import norm

from .stage import FRAME_OPEN_OK, FRAMES, JA_FRAMES


def filter_scenes(a, out: Path, items: list[dict], todo: list[dict] | None = None):
    """Detector + readers over ``todo`` (default: every item): keep
    exactly-one-box images whose box reads the anchor and are large enough;
    then both jsonl files, sheets and the report over all of ``items`` — a
    grown pool judges only its new renders, the stored rows stand."""
    todo = items if todo is None else todo
    if todo:
        rd = Readers(a.device)
        t0 = time.time()
        for n, it in enumerate(todo, 1):
            bgr = load_bgr(Path(it["file"]))
            reads = rd.read_image(bgr, whole=False)
            it["boxes"] = [r["box"] for r in reads]
            it["reads"] = [{"sfx": r["sfx"], "vl": r["vl"]} for r in reads]
            it["reason"] = judge(a, it, reads, bgr)
            if n % 100 == 0 or n == len(todo):
                print(
                    f"scenes read {n}/{len(todo)} {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
        del rd
    report_scenes(a, out, items)


def prune_rejected(items: list[dict]) -> int:
    """``--scene_prune``: unlink every rejected render still on disk; the row
    keeps its stored reason (``_rejudge_one`` and the sheets tolerate the
    missing file). Returns the count removed."""
    n = 0
    for it in items:
        if it.get("reason", "pass") == "pass":
            continue
        p = Path(it["file"])
        if p.exists():
            p.unlink()
            n += 1
    return n


def report_scenes(a, out: Path, items: list[dict]):
    kept = [it for it in items if it["reason"] == "pass"]
    (out / "scenes_all.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items)
    )
    keys = (
        "i",
        "file",
        "head",
        "generals",
        "tags",
        "prompt",
        "anchor",
        "frame",
        "clause_tpl",
        "box",
        "region",
        "bubble",
        "boxes_anchor",
        "regions",
        "bubbles",
        "shape",
        "seed",
        "read",
        "residual",
        "open_uniform",
        "open_lost",
        "region_offset",
        "boxes_speck",
        "speck_regions",
        "speck_bubbles",
    )
    (out / "scenes.jsonl").write_text(
        "\n".join(
            json.dumps({k: it[k] for k in keys if k in it}, ensure_ascii=False)
            for it in kept
        )
    )
    reasons = Counter(it["reason"] for it in items)
    n = len(items)
    short = [
        min(it["region"][2] - it["region"][0], it["region"][3] - it["region"][1])
        for it in kept
    ]
    lines = [
        f"# scenes `{a.scene_tag}` — {n} generated, **{len(kept)} kept ({len(kept) / max(1, n):.0%})**",
        "",
        f"prompt: `<rating, count, character, copyright, @artist, generals sorted>. <clause>`; "
        f"frames `{a.scene_frames}` ("
        + "; ".join(
            f"{f}: `{FRAMES[f][1]}`" for f in FRAMES if f in a.scene_frames.split(",")
        )
        + f"); shapes `{a.scene_shapes}` × gen scale {a.scene_gen_scale}; batch {a.scene_batch}; "
        f"{a.steps} steps cfg {a.cfg}; negative `{a.scene_negative}`; min box {a.scene_min_box} px; "
        f"max erase residual {a.scene_max_residual}; open erase lost ink <= {a.scene_open_lost}; "
        f"region offset <= {a.scene_max_offset}",
        "",
        "| reason | n | share |",
        "|---|---|---|",
    ]
    for k in (
        "pass",
        "no_box",
        "multi_box",
        "read_miss",
        "small_box",
        "open_bubble",
        "bubble_leak",
        "erase_miss",
        "speck_erase",
    ):
        lines.append(
            f"| {k} | {reasons.get(k, 0)} | {reasons.get(k, 0) / max(1, n):.0%} |"
        )
    if kept:
        short.sort()
        # tategaki pool (2026-09-16): how many headline regions are taller
        # than wide, and how tall — the sentence line's binding constraint
        ars = [
            (it["region"][3] - it["region"][1])
            / max(1, it["region"][2] - it["region"][0])
            for it in kept
        ]
        tall_h = sorted(
            it["region"][3] - it["region"][1] for it, ar in zip(kept, ars) if ar >= 1.0
        )
        lines += [
            "",
            f"kept usable-region short side (px): min {short[0]} p10 {short[len(short) // 10]} "
            f"median {short[len(short) // 2]} max {short[-1]}; "
            f"anchor bubbles per kept image {sum(len(it['regions']) for it in kept) / len(kept):.2f}; "
            f"kept with erased specks {sum(bool(it.get('boxes_speck')) for it in kept)} "
            f"({sum(len(it.get('boxes_speck', ())) for it in kept)} specks)",
            f"tall regions (AR ≥ 1.0): {sum(ar >= 1.0 for ar in ars)}/{len(kept)} "
            f"({sum(ar >= 1.0 for ar in ars) / len(kept):.0%}); AR ≥ 1.3: "
            f"{sum(ar >= 1.3 for ar in ars)}; tall-region height median "
            f"{tall_h[len(tall_h) // 2] if tall_h else 0} px",
            "",
            "kept per anchor: "
            + ", ".join(
                f"{k} {v}/{c}"
                for (k, v), c in zip(
                    sorted(Counter(it["anchor"] for it in kept).items()),
                    [
                        Counter(it["anchor"] for it in items)[k]
                        for k in sorted(Counter(it["anchor"] for it in kept))
                    ],
                )
            ),
            "kept per shape: "
            + ", ".join(
                f"{k} {v}"
                for k, v in sorted(
                    Counter("x".join(map(str, it["shape"])) for it in kept).items()
                )
            ),
            "kept per frame: "
            + ", ".join(
                f"{k} {v}/{Counter(it.get('frame', 'reads_as') for it in items)[k]}"
                for k, v in sorted(
                    Counter(it.get("frame", "reads_as") for it in kept).items()
                )
            ),
            "reject per frame: "
            + "; ".join(
                f"{f} "
                + ", ".join(
                    f"{r} {c}"
                    for r, c in Counter(
                        it["reason"]
                        for it in items
                        if it.get("frame", "reads_as") == f and it["reason"] != "pass"
                    ).most_common(3)
                )
                for f in sorted({it.get("frame", "reads_as") for it in items})
            ),
        ]
    lines += [
        "",
        "Sheets: sheet_kept.png (text box lime, region cyan, bubble yellow), sheet_rejected.png (reason / reads).",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    rng = random.Random(0)
    _sheet(rng.sample(kept, min(40, len(kept))), out / "sheet_kept.png", True)
    # pruned rejects (--scene_prune) have no render to draw
    rej = [it for it in items if it["reason"] != "pass" and Path(it["file"]).exists()]
    _sheet(rng.sample(rej, min(40, len(rej))), out / "sheet_rejected.png", False)


def _sheet(its, path: Path, kept: bool):
    from PIL import Image, ImageDraw

    if not its:
        return
    rows = []
    for it in its:
        im = Image.open(it["file"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in it["boxes"]:
            d.rectangle(b, outline="red" if not kept else "lime", width=4)
        if kept:
            for bub, reg in zip(it["bubbles"], it["regions"]):
                if bub:
                    d.rectangle(bub, outline="yellow", width=3)
                d.rectangle(reg, outline="cyan", width=3)
        r0 = it["reads"][0] if it["reads"] else {"sfx": "", "vl": ""}
        rows.append(
            (
                im,
                [
                    f"{it['i']:05d} {it['anchor']} {it.get('frame', '')[:6]} {it['reason']}",
                    f"vl {r0['vl'] or ''} / sfx {r0['sfx'] or ''}",
                    it["tags"][:40],
                    it["tags"][40:80],
                ],
            )
        )
    contact_sheet(rows, path, thumb=192, cols=8)


# ----------------------------------------------------------------------------
# judging one image


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def judge(a, it: dict, reads: list, bgr) -> str:
    """Set ``boxes_anchor`` / ``regions`` / ``bubbles`` (one per anchor
    bubble; ``box`` / ``region`` / ``bubble`` = the largest) and ``read`` on
    ``it``; return the reason. The detector boxes the *text*, not the bubble.
    Every box that reads the anchor is an anchor bubble (the base draws one
    per speaker; the data stage swaps all of them); boxes overlapping an
    anchor box are merged into it; any other box is stray text and rejects
    the image unless it is a speck (under a quarter of the anchor box's
    area), which is recorded (``boxes_speck`` / ``speck_regions`` /
    ``speck_bubbles``) for the data stage to erase. Each anchor bubble must be closed (flood fill from a ring
    outside the text box stays off the border) and its inscribed region at
    least ``--scene_min_box`` on the short side."""
    if not reads:
        return "no_box"
    anchor = norm(it["anchor"])
    if it.get("frame") in JA_FRAMES:
        # JA frame: the base's kana are garbled and get erased — every
        # detector box is an anchor bubble, no read match asked
        hits = list(reads)
    else:
        hits = [
            r for r in reads if anchor in {norm(r["vl"] or ""), norm(r["sfx"] or "")}
        ]
    if not hits:
        return "read_miss"
    boxes = []
    for r in hits:
        box = list(r["box"])
        for o in reads:
            if o is not r and o not in hits and _overlap(box, o["box"]):
                box = [
                    min(box[0], o["box"][0]),
                    min(box[1], o["box"][1]),
                    max(box[2], o["box"][2]),
                    max(box[3], o["box"][3]),
                ]
        boxes.append(box)
    area = max(1, max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes))
    others = [
        o
        for o in reads
        if o not in hits and not any(_overlap(b, o["box"]) for b in boxes)
    ]
    specks = [
        o["box"]
        for o in others
        if (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) < 0.25 * area
    ]
    if len(specks) < len(others):
        return "multi_box"
    r = hits[0]
    it["read"] = r["vl"] if norm(r["vl"] or "") == anchor else (r["sfx"] or r["vl"])
    it["boxes_anchor"], it["bubbles"], it["regions"] = boxes, [], []
    H, W = bgr.shape[:2]
    unis, losts, offs = [], [], []
    for b in boxes:
        bubble, region = bubble_region(bgr, b)
        if bubble is None:
            u, region = _open_region(a, bgr, b, region, a.scene_min_box)
            unis.append(u)
            losts.append(erase_lost(bgr, b, region))
        else:
            offs.append(region_offset(b, region))
        it["bubbles"].append(bubble)
        it["regions"].append(region)
    # a flood that leaked through an outline gap (Δ0.9, ja_comic 770): the
    # region leaves the text, the erase paints a panel strip or a figure
    it["region_offset"] = max(offs) if offs else None
    if offs and it["region_offset"] > a.scene_max_offset:
        return "bubble_leak"
    # the largest bubble is the headline record
    k = max(
        range(len(boxes)),
        key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
    )
    it["box"], it["bubble"], it["region"] = boxes[k], it["bubbles"][k], it["regions"][k]
    # no closed bubble: fine for a bubble-less frame, for --scene_allow_open,
    # or when the rectangle erase has no visible seam (≥ --scene_open_uniform)
    # and paints over no outline / art beside the text (≤ --scene_open_lost;
    # Δ0.9, s1 627: the seam test passed a rectangle through the outline)
    open_anchors = [b for b, bub in zip(boxes, it["bubbles"]) if bub is None]
    it["open_uniform"] = min(unis) if unis else None
    it["open_lost"] = max(losts) if losts else None
    open_ok = (
        a.scene_allow_open
        or it.get("frame") in FRAME_OPEN_OK
        or (
            open_anchors
            and it["open_uniform"] >= a.scene_open_uniform
            and it["open_lost"] <= a.scene_open_lost
        )
    )
    if open_anchors and not open_ok:
        return "open_bubble"
    if any(min(g[2] - g[0], g[3] - g[1]) < a.scene_min_box for g in it["regions"]):
        return "small_box"
    # the erase the data stage will run must actually remove the anchor:
    # a flood that took another blob leaves the letters under the kana
    it["residual"] = max(
        anchor_residual(bgr, b, g, open_ok=open_ok)
        for b, g in zip(boxes, it["regions"])
    )
    if it["residual"] > a.scene_max_residual:
        return "erase_miss"
    # specks (plan_synth2 Δ0.9): the small non-anchor boxes — a second
    # bubble of pseudo-text, a sign, a signature — were kept un-erased and
    # became text beside the glyph in the training target (sl1w 192: "Hav"
    # under the pasted glyph). Each gets the anchor's erase; a bubble speck
    # the flood cannot clear or that leaked, or an open one whose rectangle
    # leaves a seam or paints over art, rejects the scene.
    it["boxes_speck"], it["speck_regions"], it["speck_bubbles"] = [], [], []
    for b in specks:
        b = list(b)
        bubble, region = bubble_region(bgr, b)
        if bubble is None:
            u, region = _open_region(a, bgr, b, region, 0)
            if (
                u < a.scene_open_uniform
                or erase_lost(bgr, b, region) > a.scene_open_lost
            ):
                return "speck_erase"
        elif (
            region_offset(b, region) > a.scene_max_offset
            or anchor_residual(bgr, b, region) > a.scene_max_residual
        ):
            return "speck_erase"
        it["boxes_speck"].append(b)
        it["speck_regions"].append(region)
        it["speck_bubbles"].append(bubble)
    return "pass"


def _open_region(a, bgr, box, region, min_side: int):
    """``(seam uniformity, region)`` for a box with no closed bubble: the
    region is the text box grown 1.5×; take the smallest growth whose erase
    seam is invisible instead, so a broken outline is kept rather than
    painted over (s1 832). Growths under ``min_side`` px are skipped."""
    H, W = bgr.shape[:2]
    best = (erase_uniform(bgr, box, region), region)
    for grow in (1.2, 1.35):
        g = _grown(box, W, H, grow)
        if min(g[2] - g[0], g[3] - g[1]) < min_side:
            continue
        u = erase_uniform(bgr, box, g)
        if u >= a.scene_open_uniform:
            return u, g
    return best


def bubble_region(bgr, box):
    """``(bubble bbox or None, usable region)`` from ``common.bubble.bubble_mask``:
    the usable region is the bubble bbox's inscribed rectangle (0.72 × —
    1/√2 of an ellipse's axes), grown to hold the text box. No bubble (open
    fill, leak, a blob that does not enclose the text): the region is the
    text box grown 1.5× and clamped — bounded, so an erase stays near the
    text."""
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    m = bubble_mask(bgr, box)
    if m is None:
        return None, _grown(box, W, H)
    bx0, by0, bx1, by1 = bubble_bbox(m)
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    bw, bh = (bx1 - bx0) * 0.72, (by1 - by0) * 0.72
    region = [
        int(max(0, bcx - bw / 2)),
        int(max(0, bcy - bh / 2)),
        int(min(W, bcx + bw / 2)),
        int(min(H, bcy + bh / 2)),
    ]
    region = [
        min(region[0], x0),
        min(region[1], y0),
        max(region[2], x1),
        max(region[3], y1),
    ]
    return [bx0, by0, bx1, by1], region


def _grown(box, W, H, k: float = 1.5):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * k, (y1 - y0) * k
    return [
        int(max(0, cx - w / 2)),
        int(max(0, cy - h / 2)),
        int(min(W, cx + w / 2)),
        int(min(H, cy + h / 2)),
    ]
