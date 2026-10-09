#!/usr/bin/env python
"""inject_count — does a word-length banner, put on the trajectory at σ, survive to the end?

The count read `proposal_length.md` step 0 never ran, and `cf_sense`'s
``move`` is a teacher-forced single-step projection (`findings.md` § Against:
teacher-forced reads peak below where the trajectory commits). This is the
trajectory-side read, no training: take a floor render whose banner holds a
leftover slot (``dup``: `こんにちちは`, 6 slots), redraw its banner with the
word filling the same extent (``A``: 5 slots, glyphs × 6/5), and put
``(1 − σ)·z_A + σ·ε`` on the sampler at σ ∈ ``--sigmas`` in place of its own
x_t; the rest of the trajectory runs as the floor did (seed rows, the floor's
caption and seed). ``B`` = the floor render itself re-noised the same way,
same ε: the control that re-noising at σ gives the floor back.

What it answers: at which σ the slot count is committed. If ``A`` injected
at 0.9 keeps 5 slots to the end (official / ≤ 1 edit up, dup down vs ``B``),
whatever puts the trajectory on the 5-slot banner by 0.9 wins, and a row
acting at 0.85–0.95 has a target there. If the sampler writes the sixth
slot back below 0.9, the count is re-decided lower and 0.85–0.95 alone is
not where to train it.

Canvases: ``erase`` = the read box padded, filled with its ring median;
``draw`` = the word in one line (or one column when the box is taller than
wide) at ``fs = min(0.9 · box short side, box long side / (n · 1.05))``,
glyphs spread from the box's first to its last edge, ink = the banner's own
stroke colour, a manga face that covers the text. The canvases are read too
(``canvas`` arm, no sampling): a redraw the readers cannot read is dropped.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/inject_count/run_exp.py \\
      --label ic0"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "inject_count"
SIGMAS = (0.95, 0.9, 0.85, 0.8)
PAD = 6  # px around the read box the erase covers
PITCH = 1.05  # glyph pitch / fs along the line
GROW = 1.5  # fs may exceed the box short side by this much
MAX_ITEMS = 60


def dup_items(SS, R) -> list[dict]:
    """The floor's ``sent`` renders whose main read holds a doubled glyph,
    with that read's box (the largest non-whole box whose read doubles)."""
    import re

    from common.text import lev, norm

    chars = sorted({it["text"] for it in SS.floor_items()})
    h = R.hits(SS.FLOOR_READS, chars, SS.CLAUSE)
    recs = {
        (m["text"], m["clause"], m["pi"], m["seed"]): m
        for m in json.loads(SS.FLOOR_READS.read_text("utf-8"))
    }
    out = []
    for it in SS.floor_items():
        k = (it["text"], it["clause"], it["pi"], it["seed"])
        if not h[k]["dup"] or len(it["text"]) < 3:
            continue
        boxes = [
            r
            for r in recs[k]["reads"]
            if not r.get("whole")
            and r.get("box")
            and any(re.search(r"(.)\1", r.get(x) or "") for x in ("sfx", "vl"))
        ]
        if not boxes:
            continue
        # the box whose read, doubled glyphs collapsed, is nearest the word
        # (the leftover-slot banner), the larger on a tie
        t = norm(it["text"])

        def score(r):
            best = min(
                lev(re.sub(r"(.)\1+", r"\1", norm(r.get(x) or "")), t)
                for x in ("sfx", "vl")
            )
            return (best, -(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]))

        b = min(boxes, key=score)
        if score(b)[0] > 2:
            continue
        # the main text: no other CJK box more than 1.5 × its area (a redraw
        # of a secondary line would leave the banner's own leftover in place)
        area = lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])  # noqa: E731
        others = [
            r
            for r in recs[k]["reads"]
            if not r.get("whole")
            and r.get("box")
            and re.search(r"[ぁ-ヿ一-鿿]", (r.get("sfx") or "") + (r.get("vl") or ""))
        ]
        if any(area(r) > 1.5 * area(b) for r in others):
            continue
        out.append(it | {"box": b["box"], "floor_read": b.get("sfx") or b.get("vl")})
    return out


def redraw(src: Path, dst: Path, text: str, box: list, rng: random.Random) -> dict:
    """``src`` with ``box`` erased and ``text`` drawn filling it; returns the
    drawn geometry."""
    import numpy as np
    from common.bubble import ring_median
    from common.render.flat import find_fonts, pick_font
    from PIL import Image, ImageDraw, ImageFont

    GR = load_experiment("garble_replace")
    im = Image.open(src).convert("RGB")
    arr = np.asarray(im).copy()
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    fill = tuple(int(v) for v in ring_median(arr, box))
    ink = GR.ink_color(arr, box, fill)
    n = len(text)
    vertical = (y1 - y0) > (x1 - x0)
    long_side, short_side = (
        ((y1 - y0), (x1 - x0)) if vertical else ((x1 - x0), (y1 - y0))
    )
    # n contiguous glyphs filling the span: fs from the length, allowed past
    # the box's own height up to GROW × (one leftover slot of six is × 1.2);
    # a longer box (a garble line, not a banner) gets the word centred in it
    fs = int(min(long_side / (n * PITCH), GROW * short_side))
    span = fs * (1 + (n - 1) * PITCH)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = fs * 0.6
    if vertical:
        ex0, ey0, ex1, ey1 = (
            cx - half,
            min(y0, cy - span / 2),
            cx + half,
            max(y1, cy + span / 2),
        )
    else:
        ex0, ey0, ex1, ey1 = (
            min(x0, cx - span / 2),
            cy - half,
            max(x1, cx + span / 2),
            cy + half,
        )
    ex0, ey0 = max(0, int(ex0) - PAD), max(0, int(ey0) - PAD)
    ex1, ey1 = min(W, int(ex1) + PAD), min(H, int(ey1) + PAD)
    arr[ey0:ey1, ex0:ex1] = fill
    im = Image.fromarray(arr)
    fonts = [f for f in find_fonts() if "Light" not in f] or find_fonts()
    font_path = pick_font(text, fonts, rng)
    font = ImageFont.truetype(font_path, fs)
    d = ImageDraw.Draw(im)
    start = (cy if vertical else cx) - span / 2
    for i, ch in enumerate(text):
        w = d.textlength(ch, font=font)
        pos = start + i * fs * PITCH
        if vertical:
            d.text((cx - w / 2, pos), ch, fill=ink, font=font)
        else:
            d.text((pos + fs / 2 - w / 2, cy - fs / 2), ch, fill=ink, font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    return {
        "fs": fs,
        "box_short": short_side,
        "font": Path(font_path).name,
        "vertical": vertical,
        "ink": ink,
    }


class Injector:
    """sigma_split's Splitter with x_t replaced at one step."""

    def __init__(self, SS):
        self.sp = SS.Splitter()
        self.SS = SS

    def latent(self, file: Path):
        from common.models import encode_images

        return encode_images(self.sp.vae, [str(file)], self.sp.device)[0]  # (C, H, W)

    def render(self, fn: Path, it: dict, z, sigma: float, eps_seed: int) -> float:
        """Run the floor's trajectory, replace x_t at the step nearest
        ``sigma`` with ``(1 − σ)·z + σ·ε``; returns the σ actually used."""
        import torch
        from common.models import decode_image
        from library.inference import generation as G
        from library.inference import sampling as S

        sp = self.sp
        hi, null = sp.encode(it["caption"], "seed")
        step = S.step
        used = {}

        def inj(latents, noise_pred, sigmas, i):
            new = step(latents, noise_pred, sigmas, i)
            if "j" not in used:
                used["j"] = min(
                    range(1, len(sigmas) - 1),
                    key=lambda j: abs(float(sigmas[j]) - sigma),
                )
            j = used["j"]
            if i + 1 == j:
                s = float(sigmas[j])
                used["s"] = s
                g = torch.Generator(device="cpu").manual_seed(eps_seed)
                eps = torch.randn(z.shape, generator=g).to(new.device)
                zz = z.to(new.device)
                mixed = (1 - s) * zz + s * eps
                return mixed.reshape(new.shape).to(new.dtype)
            return new

        S.step = inj
        a2 = sp._args(it["caption"], it["seed"])
        try:
            with torch.no_grad():
                lat = G.generate(
                    a2,
                    sp.gen,
                    sp.shared,
                    precomputed_text_data={"context": hi, "context_null": null},
                )
        finally:
            S.step = step
        fn.parent.mkdir(parents=True, exist_ok=True)
        decode_image(sp.vae, lat, sp.device).save(fn)
        return used["s"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    p.add_argument("--max_items", type=int, default=MAX_ITEMS)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    SS = load_experiment("sigma_split")
    from cjk_scale import reads as R

    items = dup_items(SS, R)
    rng = random.Random(0)
    rng.shuffle(items)
    items = sorted(
        items[: args.max_items], key=lambda it: (it["text"], it["pi"], it["seed"])
    )
    root = OUT / "experiments" / NAME
    print(
        f"{len(items)} dup floor renders × {len(args.sigmas)} σ × 2 (A / B); "
        f"strings {sorted({it['text'] for it in items})}",
        flush=True,
    )
    if args.dry_run:
        return
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    # canvases (CPU)
    canvas = []
    for it in items:
        key = f"p{it['pi']:02d}_{it['text']}_s{it['seed']}"
        fa = root / "canvas" / "img" / f"A_{key}.png"
        geo = redraw(Path(it["floor_file"]), fa, it["text"], it["box"], rng)
        it["canvas_a"], it["key"], it["geo"] = str(fa), key, geo
        canvas.append(
            {k: it[k] for k in ("seed", "pi", "prompt", "text", "clause", "caption")}
            | {"file": str(fa), "cond": "canvas"}
        )
    inj = Injector(SS)
    device = inj.sp.device
    t0 = time.time()
    manifests: dict = {"canvas": canvas}
    used_sigma: dict = {}
    for n, it in enumerate(items):
        zA = inj.latent(Path(it["canvas_a"]))
        zB = inj.latent(Path(it["floor_file"]))
        for sigma in args.sigmas:
            for arm, z in (("A", zA), ("B", zB)):
                name = f"{arm}_s{sigma:g}"
                fn = root / name / "img" / f"{name}_{it['key']}.png"
                eps_seed = 1000 * it["seed"] + it["pi"] * 10 + int(sigma * 100)
                if not fn.exists():
                    used_sigma[name] = inj.render(fn, it, z, sigma, eps_seed)
                manifests.setdefault(name, []).append(
                    {
                        k: it[k]
                        for k in ("seed", "pi", "prompt", "text", "clause", "caption")
                    }
                    | {"file": str(fn), "cond": name, "sigma": sigma}
                )
        if n % 10 == 0:
            print(f"  {n}/{len(items)} · {(time.time() - t0) / 60:.1f} min", flush=True)
    inj.sp.free()
    for name, man in manifests.items():
        SS.read_arm(man, root / name, device)
        print(f"read {name}", flush=True)
    chars = sorted({it["text"] for it in items})
    keys = {(it["text"], it["clause"], it["pi"], it["seed"]) for it in items}
    floor_h = {
        k: v for k, v in R.hits(SS.FLOOR_READS, chars, SS.CLAUSE).items() if k in keys
    }
    metrics: dict = {
        "items": len(items),
        "sigmas": args.sigmas,
        "used_sigma": used_sigma,
        "floor_tally": R.tally(floor_h),
        "arms": {},
    }
    print("===== floor (the dup subset)", flush=True)
    for name in manifests:
        f = root / name / "native_reads.json"
        h = R.hits(f, chars, SS.CLAUSE)
        ms = json.loads(f.read_text("utf-8"))
        ps = [SS.placement(m) for m in ms]
        print(f"===== {name}", flush=True)
        metrics["arms"][name] = {
            "tally": R.tally(h),
            "vs_floor": R.paired(h, floor_h),
            "placement": {
                k: round(st.mean(x[k] for x in ps), 4)
                for k in ("box", "box_h", "flat_white")
            },
        }
        print(f"  vs floor {metrics['arms'][name]['vs_floor']}", flush=True)
    # A vs B at each σ, paired
    for sigma in args.sigmas:
        ha = R.hits(root / f"A_s{sigma:g}" / "native_reads.json", chars, SS.CLAUSE)
        hb = R.hits(root / f"B_s{sigma:g}" / "native_reads.json", chars, SS.CLAUSE)
        metrics["arms"][f"A_s{sigma:g}"]["vs_B"] = R.paired(ha, hb)
        print(f"  A vs B at σ {sigma:g}: {R.paired(ha, hb)}", flush=True)
    sheet(items, manifests, args.sigmas, root / "sheet.png")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def sheet(items: list, manifests: dict, sigmas: list, path: Path, n: int = 16) -> None:
    """Rows = items, columns = floor | A canvas | A at each σ | B at each σ."""
    from common.render.flat import find_fonts
    from PIL import Image, ImageDraw, ImageFont

    by = {
        name: {(m["text"], m["pi"], m["seed"]): m for m in man}
        for name, man in manifests.items()
    }
    cols = (
        ["floor", "canvas"]
        + [f"A_s{s:g}" for s in sigmas]
        + [f"B_s{s:g}" for s in sigmas]
    )
    T = 192
    font = ImageFont.truetype(find_fonts()[0], 11)  # Noto Serif CJK: full coverage
    out = Image.new("RGB", (len(cols) * T, min(n, len(items)) * (T + 14) + 16), "white")
    d = ImageDraw.Draw(out)
    for c, name in enumerate(cols):
        d.text((c * T + 4, 2), name, fill="black", font=font)
    for r, it in enumerate(items[:n]):
        k = (it["text"], it["pi"], it["seed"])
        y = 16 + r * (T + 14)
        for c, name in enumerate(cols):
            f = it["floor_file"] if name == "floor" else by[name][k]["file"]
            im = Image.open(f).convert("RGB")
            im.thumbnail((T, T))
            out.paste(im, (c * T, y))
            if name != "floor":
                m = by[name][k]
                rd = next(
                    (x.get("sfx") for x in m.get("reads", []) if not x.get("whole")), ""
                )
                d.text((c * T + 2, y + T), (rd or "")[:12], fill="black", font=font)
        d.text(
            (2, y + T),
            f"{it['text']} p{it['pi']} s{it['seed']}",
            fill="black",
            font=font,
        )
    out.save(path)


if __name__ == "__main__":
    main()
