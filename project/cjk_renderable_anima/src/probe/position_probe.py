"""Position probe — does the frozen adapter + DiT put k quoted texts in k places?

No training, no delta. The gate for a grid step 1 (k single-unit glyphs drawn
mechanically in a 2x2 / 3x2 / 3x3 grid, k rows paid per item): the grid only
pays k draws if the base model already routes a quoted text to a cell —
through a position clause, or through list order. EN controls stand in for the
ext rows, as in `order_probe.py`.

Each item is one unit -> cell assignment, rendered under every frame x style,
so the styles are paired:

    list      `English text reads as "A", "B", "C", "D".` in reading order —
              the native multi-text grammar (export stage, `--combine_ocr`)
    pos       `On the top left, English text reads as "A". On the top right, …`
              clauses in reading order
    pos_shuf  the same clauses in a shuffled order — clause vs sequence

Frames: `bubble` (manga, multiple speech bubbles) and `flat` (white canvas).
Units: `letter` (one T5 piece) and `word` (the EN control words). Default is
k 4 and 9 — 48 captions x 2 seeds, block-compiled (`--compile 0` = eager); `--k 4,6,9` adds 3x2.

Scoring, per unit: `found` = the unit is in any read of the image; `bound` =
it is in the read of a detector box whose centre sits in the unit's cell;
`perm` = `bound` averaged over random unit -> cell assignments (what a bag of
glyphs scores); `lift` = bound - perm. `pos_shuf` also reports `order` —
`bound` against the clause *sequence* instead of the clause's position words.

    make daemon-run ARGS="--label position-probe --queue \\
        project/cjk_renderable_anima/src/probe/position_probe.py"

Outputs `output/wake_probe/position_probe/{img/,reads.json,report.md,sheet_*.png}`.

In-frame read of a trained table (plan_grid code owed 1): `--arm_dir <rows
arm> --units あいう…` renders `pos` grid captions of the table's own units with
the delta on and scores them per cell, into `<arm>/inframe/`. `found` says the
row was learned, `bound` that the credit went to the right clause — apart from
"did not transfer to a single bubble", which is the arm's `single` read.

    make daemon-run ARGS="--label inframe --queue \\
        project/cjk_renderable_anima/src/probe/position_probe.py \\
        --arm_dir output/wake_probe/rows_mix_m24_m1500 --units あいうか… --n_items 4"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common.models import generate_to, load_generator, load_vae  # noqa: E402
from common.paths import OUT  # noqa: E402
from common.prompts import EN_WORDS, GRID_FRAMES, grid_caption  # noqa: E402
from common.readers import Readers, contact_sheet, load_bgr  # noqa: E402
from common.text import norm  # noqa: E402

# k -> (cols, rows)
GRIDS = {4: (2, 2), 6: (3, 2), 9: (3, 3)}
FRAMES = tuple(GRID_FRAMES)
STYLES = ("list", "pos", "pos_shuf")
# no I / O / Q: the readers trade them for l / 0
LETTERS = list("ABCDEFGHJKLMNPRSTUVWXYZ")
WORDS = [w for w in EN_WORDS if " " not in w]
N_PERM = 200


def caption(
    frame: str, style: str, k: int, units: list[str], order: list[int], lang="english"
) -> str:
    """``units[i]`` belongs in cell ``i``; ``order`` is the clause sequence."""
    if lang != "english":
        return grid_caption(frame, *GRIDS[k], units, order, lang=lang)
    if style == "list":
        quoted = ", ".join(f'"{u}"' for u in units)
        return f"{GRID_FRAMES[frame].format(lang='english')} English text reads as {quoted}."
    return grid_caption(frame, *GRIDS[k], units, order, lang="english")


def build_items(
    n_items: int, ks: list[int], kinds: list[str], seed: int = 0, ja_units=None
):
    """``ja_units``: the in-frame read — one kind (``ja``), the ``pos`` style."""
    rng = random.Random(seed)
    items = []
    styles = ("pos",) if ja_units else STYLES
    for k in ks:
        for kind in ["ja"] if ja_units else kinds:
            pool = ja_units or (LETTERS if kind == "letter" else WORDS)
            for j in range(n_items):
                units = rng.sample(pool, k)
                shuf = list(range(k))
                while shuf == list(range(k)):
                    rng.shuffle(shuf)
                for frame in FRAMES:
                    for style in styles:
                        order = shuf if style == "pos_shuf" else list(range(k))
                        items.append(
                            {
                                "frame": frame,
                                "style": style,
                                "k": k,
                                "kind": kind,
                                "item": j,
                                "units": units,
                                "order": order,
                                "caption": caption(
                                    frame, style, k, units, order,
                                    "japanese" if ja_units else "english",
                                ),
                            }
                        )
    return items


def cell_reads(reads: list, k: int, W: int, H: int) -> dict[int, str]:
    """Normalised reads (both readers) of the detector boxes, keyed by the cell
    each box's centre sits in."""
    cols, rows = GRIDS[k]
    out = defaultdict(str)
    for r in reads:
        if r["whole"]:
            continue
        x0, y0, x1, y1 = r["box"]
        c = min(cols - 1, int((x0 + x1) / 2 / W * cols))
        rw = min(rows - 1, int((y0 + y1) / 2 / H * rows))
        out[rw * cols + c] += norm(r["sfx"] or "") + "|" + norm(r["vl"] or "") + "|"
    return out


def bound_frac(units: list[str], target: list[int], cells: dict[int, str]) -> float:
    """Share of units read inside their target cell (``target[i]`` for unit i)."""
    return sum(norm(u) in cells.get(t, "") for u, t in zip(units, target)) / len(units)


def score(m: dict, W: int, H: int, rng: random.Random):
    k, units = m["k"], m["units"]
    cells = cell_reads(m["reads"], k, W, H)
    every = "|".join(
        norm(r["sfx"] or "") + "|" + norm(r["vl"] or "") for r in m["reads"]
    )
    ident = list(range(k))
    m["n_boxes"] = sum(not r["whole"] for r in m["reads"])
    m["found"] = sum(norm(u) in every for u in units) / k
    m["bound"] = bound_frac(units, ident, cells)
    perms = []
    for _ in range(N_PERM):
        t = ident[:]
        rng.shuffle(t)
        perms.append(bound_frac(units, t, cells))
    m["perm"] = sum(perms) / N_PERM
    # clause i of the sequence is unit order[i]; "order" puts it in cell i
    seq = [m["order"].index(i) for i in ident]
    m["order_bound"] = bound_frac(units, seq, cells)
    m["cells"] = {str(c): s for c, s in cells.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--n_items", type=int, default=2, help="assignments per k x kind")
    p.add_argument("--k", default="4,9", help="grid sizes; 6 (3x2) is available")
    p.add_argument("--kinds", default="letter,word")
    p.add_argument("--compile", type=int, default=1, help="block-compile the DiT")
    p.add_argument("--arm_dir", default="", help="in-frame read: a trained rows arm")
    p.add_argument(
        "--units", default="", help="--arm_dir: its units (a string, or a comma list)"
    )
    a = p.parse_args()
    assert bool(a.arm_dir) == bool(a.units), "--arm_dir and --units go together"
    ja_units = (a.units.split(",") if "," in a.units else list(a.units)) or None

    import torch

    out = Path(a.arm_dir) / "inframe" if a.arm_dir else OUT / "position_probe"
    (out / "img").mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in a.k.split(",")]
    items = build_items(a.n_items, ks, a.kinds.split(","), ja_units=ja_units)
    print(f"{len(items)} captions x {a.seeds} seeds", flush=True)
    for it in items[: len(FRAMES) * len(STYLES)]:
        print(f"  {it['frame']:6s} {it['style']:8s} {it['caption']}")

    args, gen, device, shared = load_generator(a.size, a.steps, a.cfg, out / "img")
    shared["model"].eval()
    if a.arm_dir:
        # before compile_blocks (compile-after-apply)
        from common.hooks import ExtDelta
        from common.models import load_trained

        sd = load_trained(Path(a.arm_dir))
        assert "lora" not in sd and "out_vec" not in sd, "in-frame read: rows arms only"
        delta = ExtDelta.from_state(shared["model"], sd["delta"], device)
        delta.scale = 1.0
    if a.compile:
        # one canvas, no adapter: a single static graph per block
        shared["model"].compile_blocks(n_token_families=1)
    vae = load_vae(device)
    manifest = []
    t0 = time.time()
    for e in items:
        for seed in range(a.seeds):
            tag = f"{e['frame']}_{e['style']}_k{e['k']}_{e['kind']}_{e['item']}_s{seed}"
            fn = out / "img" / f"{tag}.png"
            generate_to(fn, args, gen, shared, vae, device, e["caption"], seed)
            manifest.append({"file": str(fn), "seed": seed, **e})
    print(
        f"gen: {len(manifest)} images in {(time.time() - t0) / 60:.1f} min", flush=True
    )
    del vae, shared
    torch.cuda.empty_cache()

    from PIL import Image

    rd = Readers(a.device)
    rng = random.Random(0)
    for m in manifest:
        bgr = load_bgr(Path(m["file"]))
        m["reads"] = rd.read_image(bgr, whole=True)
        score(m, bgr.shape[1], bgr.shape[0], rng)
    (out / "reads.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    agg = defaultdict(list)
    for m in manifest:
        agg[(m["frame"], m["style"], m["k"], m["kind"])].append(m)

    def mean(ms, key):
        return sum(m[key] for m in ms) / len(ms)

    lines = [
        f"# position probe — in-frame read of {a.arm_dir}"
        if a.arm_dir
        else "# position probe — base model, k quoted EN texts, no delta",
        "",
        f"size {a.size} steps {a.steps} cfg {a.cfg} seeds {a.seeds}, "
        f"{a.n_items} assignments per k x kind",
        "",
        "`bound` = unit read inside its cell; `perm` = the same under random "
        "assignments; `order` = bound against the clause sequence (differs from "
        "`bound` on `pos_shuf` only).",
        "",
        "| frame | style | k | kind | n | found | bound | perm | lift | order | boxes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (frame, style, k, kind), ms in agg.items():
        lines.append(
            f"| {frame} | {style} | {k} | {kind} | {len(ms)} | {mean(ms, 'found'):.2f} | "
            f"{mean(ms, 'bound'):.2f} | {mean(ms, 'perm'):.2f} | "
            f"{mean(ms, 'bound') - mean(ms, 'perm'):+.2f} | "
            f"{mean(ms, 'order_bound'):.2f} | {mean(ms, 'n_boxes'):.1f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # one sheet per frame x k x kind; a row of the sheet is one assignment x
    # seed across the three styles
    sheets = defaultdict(list)
    for m in manifest:
        sheets[(m["frame"], m["k"], m["kind"])].append(m)
    for (frame, k, kind), ms in sheets.items():
        ms.sort(key=lambda m: (m["item"], m["seed"], STYLES.index(m["style"])))
        cols = GRIDS[k][0]
        rows = []
        for m in ms:
            u = m["units"]
            grid = "/".join(
                " ".join(u[i : i + cols]) for i in range(0, k, cols)
            )
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"{m['style']} #{m['item']} s{m['seed']}",
                        grid,
                        f"found {m['found']:.2f} bound {m['bound']:.2f}",
                        f"perm {m['perm']:.2f} boxes {m['n_boxes']}",
                    ],
                )
            )
        n_styles = len({m["style"] for m in ms})
        contact_sheet(rows, out / f"sheet_{frame}_k{k}_{kind}.png", cols=n_styles)


if __name__ == "__main__":
    main()
