#!/usr/bin/env python
"""span_reband — the seed's word-length windows moved to σ 0.85–0.95, warm (proposal_length step 1)

`proposal_length.md` § Span: the text region's span is set at σ ≥ 0.9 and no
item of the seed trains there with a word-length region — `scene_window`
(b0507, ≈ 34 px, the word filling 0.7–1.0 of its bubble along the reading
axis) trains at 0.5–0.7. `hypothesis.md` H1 + `b0305_reband`: an item
trained above where its glyphs resolve teaches its layout alone, and the
rows carry that layout to every caption. So the arm moves only the items
whose canvas layout *is* the wanted one:

- ``retrain_kana``'s b0507 ``scene_window`` items, **filtered** (the data
  check of 2026-10-01): a speech bubble (no ``sign`` frame, no open region),
  one anchor in the scene (no second, erased bubble beside the text), and a
  region that holds ≤ 2 columns / lines at the item's px — 2 589 of 4 060;
- those move to ``--band`` (0.85–0.95) and are repeated ``--repeat`` more
  times (default 3 → 10 356 of 25 167 records ≈ 2.5 epochs of them over the
  arm's steps, b0305_reband's exposure); every other record of the seed's
  mix stays at its own band (the 1 471 windows the filter drops stay at
  0.5–0.7), so the arm is one band change on top of the seed's data.

Warm from the seed rows (μ ``--mu``, default 0.02), the trainer's lr
(1e-3 cosine), ``--steps_per_row`` × the 174 kana rows (default 23 → 4 002
steps), every other row frozen at the seed. ``windows.py``'s ``SIGMA_MAX``
0.9 is the builder's; the band is overwritten on the records, as
b0305_reband did.

Legs:
- ``data`` (CPU) → ``OUT/run1001_span_reband/data[_<tag>]`` (images stay in
  ``retrain_kana/data/img``; latents and the TE cache are built on first
  train; ``sheet_reband.png`` = 24 of the rebanded items);
- ``train`` (GPU) → ``OUT/experiments/span_reband_warm[_<tag>]``;
- ``read`` (GPU): the ``sent`` ruler vs the seed's routed floor
  (``garble_replace``'s read leg: official / ≤ 1 edit / dup + box / box_h /
  flat_white / box IoU).

The σ-composite read (the arm's rows above 0.8, the seed's below) is
``shared_dir``'s ``full_s`` arm: ``shared_dir/run_exp.py --arm_rows
OUT/experiments/span_reband_warm --arms full,full_s --label …``.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/span_reband/run_exp.py \\
      --label r0 --legs data train read"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.builder import tier_of  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "span_reband"
SRC_RUN = "retrain_kana"
SRC = OUT / SRC_RUN / "data"
TIER = "bubbleN_34"  # the records of record: group b0507, recipe scene_window
BAND = (0.85, 0.95)
MU = 0.02
STEPS_PER_ROW = 23  # × 174 kana rows ≈ 4 000 steps (b0305_reband)
REPEAT = 3  # extra copies of each rebanded record
MAX_LINES = 2  # columns / lines the region holds at the item's px
# the renderer's pitch along the cross axis (scene.py's fit), as the data check read it
V_GAP, H_PITCH = 1.15, 1.2


def scenes() -> dict:
    """``(pool, i)`` → the scene row (``region`` / ``bubble`` / ``frame`` / anchors)."""
    sc = {}
    for pool in ("s1", "s1w", "sl1w", "ja_comic"):
        f = OUT / f"scenes_{pool}" / "scenes.jsonl"
        for ln in f.read_text("utf-8").splitlines():
            if ln:
                s = json.loads(ln)
                sc[(pool, s["i"])] = s
    return sc


def lines_held(r: dict, s: dict) -> int:
    """Columns (vertical) or lines (horizontal) the scene's region holds at
    the item's px."""
    x0, y0, x1, y1 = s["region"]
    px = r["px"]
    if r["horizontal"]:
        return int((y1 - y0 - px) / (px * H_PITCH)) + 1
    return int((x1 - x0 - px) / (px * V_GAP)) + 1


def keep(r: dict, s: dict) -> str | None:
    """Why a window is dropped, or None to keep it."""
    if s.get("frame") == "sign":
        return "sign"
    if s.get("bubble") is None:
        return "open_region"
    if len(s["boxes_anchor"]) > 1:
        return "second_bubble"
    if lines_held(r, s) > MAX_LINES:
        return "wide_region"
    return None


def data(dst: Path, band: tuple, repeat: int) -> dict:
    recs = [
        json.loads(ln)
        for ln in (SRC / "train.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    sc = scenes()
    out, moved, dropped = [], [], Counter()
    for r in recs:
        if tier_of(r) == TIER:
            why = keep(r, sc[(r["scene_pool"], r["scene"])])
            if why is None:
                r = r | {"band": list(band), "reband": True}
                moved.append(r)
                out += [r] * (1 + repeat)
                continue
            dropped[why] += 1
        out.append(r)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8"
    )
    for f in ("vocabs.json", "eval.json"):
        shutil.copy2(SRC / f, dst / f)
    bj = json.loads((SRC / "build.json").read_text("utf-8"))
    (dst / "build.json").write_text(
        json.dumps(
            bj
            | {
                "reband_from": str(SRC),
                "reband": {
                    "tier": TIER,
                    "band": list(band),
                    "repeat": repeat,
                    "max_lines": MAX_LINES,
                },
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    sheet(moved, dst / "sheet_reband.png")
    bands = Counter(tuple(r["band"]) for r in out)
    return {
        "from": str(SRC),
        "windows": len(moved) + sum(dropped.values()),
        "moved": len(moved),
        "dropped": dict(dropped),
        "repeat": repeat,
        "records": len(out),
        "rebanded_share": round(len(moved) * (1 + repeat) / len(out), 3),
        "bands": {f"{a}-{b}": n for (a, b), n in sorted(bands.items())},
        "moved_px_p50": sorted(r["px"] for r in moved)[len(moved) // 2],
        "moved_glyphs": dict(sorted(Counter(r["glyphs"] for r in moved).items())),
        "moved_horizontal": round(sum(r["horizontal"] for r in moved) / len(moved), 3),
        "glyph_route": bj.get("glyph_route"),
    }


def sheet(moved: list, path: Path, n: int = 24, seed: int = 0) -> None:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    pick = rng.sample(moved, min(n, len(moved)))
    tiles = []
    for r in pick:
        im = Image.open(r["file"]).convert("RGB")
        ImageDraw.Draw(im).rectangle(r["box"], outline=(255, 0, 0), width=2)
        im.thumbnail((320, 320))
        tiles.append(im)
    cols, tw, th = 6, 320, 320
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * th), "white")
    for k, im in enumerate(tiles):
        out.paste(im, ((k % cols) * tw, (k // cols) * th))
    out.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "read"]
    )
    p.add_argument("--band", type=float, nargs=2, default=list(BAND))
    p.add_argument("--mu", type=float, default=MU)
    p.add_argument("--steps_per_row", type=int, default=STEPS_PER_ROW)
    p.add_argument("--repeat", type=int, default=REPEAT)
    p.add_argument("--tag", default="", help="suffix for the data dir and the arm")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    dst = OUT / f"run1001_{NAME}" / f"data{sfx}"
    name = f"{NAME}_warm{sfx}"

    from cjk_scale.config import load_run

    rc = dataclasses.replace(load_run(SRC_RUN), name=name)
    vocabs = json.loads((SRC / "vocabs.json").read_text("utf-8"))
    steps = args.steps_per_row * len(vocabs)
    print(
        f"{name}: {TIER} filtered → band {args.band}, ×{1 + args.repeat}, "
        f"{len(vocabs)} rows × {args.steps_per_row} = {steps} steps, μ {args.mu}, "
        f"warm from {SEED_ROWS}",
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {
        "src": str(SRC),
        "band": list(args.band),
        "mu": args.mu,
        "steps_per_row": args.steps_per_row,
        "steps": steps,
        "rows": len(vocabs),
        "repeat": args.repeat,
    }
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = data(dst, tuple(args.band), args.repeat)
        print(json.dumps(metrics["data"], ensure_ascii=False), flush=True)
    if "train" in args.legs:
        from cjk_scale import train as T

        T.INIT_ANCHOR = args.mu
        T.train(
            rc,
            data=dst,
            out=OUT / "experiments" / name,
            cold=False,
            steps_per_row=args.steps_per_row,
            context=SEED_ROWS,
        )
    if "read" in args.legs:
        GR = load_experiment("garble_replace")
        GR.DATA = dst
        metrics["read"] = GR.read("warm", name)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(dst), str(OUT / "experiments" / name)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
