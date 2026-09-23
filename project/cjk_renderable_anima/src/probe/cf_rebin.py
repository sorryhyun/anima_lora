#!/usr/bin/env python
"""cf_rebin — plan_band Stage A.0: rebin ``cf_sense.pt`` reads by glyph px,
ink per glyph, font and bubble, with no GPU.

Runs since the Stage A knobs record ``px`` / ``font`` / ``bubble`` / ``ink_a``
/ ``glyphs`` on every item. The Gate 0 runs (2026-09-22) recorded none of
them, so for an EN run without ``px`` the layout and font draws are replayed:
``cf_sense`` seeds ``random.Random(30_000 + seed)``, builds the pairs, then
draws one ``sample_layout`` + ``pick_font`` per item in order, and every one
of those draws is still consumed the same way (``_render_pair``), so the
replay is the run's own sequence. The replayed pair texts are checked
against the saved items. Ink is read from the saved renders inside the saved
box (a JA run gets ink and glyphs this way too; its px stays unknown).

    .venv/bin/python project/cjk_renderable_anima/src/probe/cf_rebin.py \\
        output/wake_probe/rows_step1_0921_s30k/cf_sense_en/cf_sense.pt \\
        output/wake_probe/rows_step1_0921m_merge/cf_sense_en_piece/cf_sense.pt

Bins: px {<40, 40–60, 60–100, 100–140, ≥140}, ink/glyph cells² {<4, 4–8,
8–16, 16–32, 32–64, ≥64} (the plan's § 1 edges), font (exact), bubble.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PX_EDGES = (40, 60, 100, 140)
INK_EDGES = (4, 8, 16, 32, 64)


def _bin(v, edges):
    for i, e in enumerate(edges):
        if v < e:
            return f"{'<' if i == 0 else f'{edges[i - 1]}–'}{e}"
    return f"≥{edges[-1]}"


def _replay(d: dict, seed: int, size) -> None:
    """Fill ``px`` / ``font`` / ``bubble`` on a pre-knob EN run's items."""
    from common.render.flat import find_fonts, pick_font, sample_layout
    from eval.cf_sense import _en_pairs

    items = d["items"]
    meta = d.get("meta") or {}
    n_id = sum(it["kind"] == "id" for it in items)
    # the single run's ``order`` items are two-word strings too: read piece
    # off the ``id`` items only
    piece = meta.get("rows") == "piece" or any(
        " " in it["a"] for it in items if it["kind"] == "id"
    )
    rng = random.Random(30_000 + seed)
    pairs = _en_pairs(rng, n_id, piece)
    assert len(pairs) == len(items), f"replay: {len(pairs)} pairs vs {len(items)} items"
    fonts = find_fonts()
    for p, it in zip(pairs, items, strict=True):
        assert (p["a"], p["b"]) == (it["a"], it["b"]), (
            f"replay drifted at {it['a']}/{it['b']} (got {p['a']}/{p['b']}): "
            "different seed, pair count or font set"
        )
        lay = sample_layout(len(p["a"]), rng, size)
        font = pick_font(p["a"] + p["b"], fonts, rng)
        it["px"], it["font"], it["bubble"] = (
            int(lay["fs"]),
            Path(font).stem,
            lay["bubble"],
        )


def _fill_ink(items) -> None:
    from PIL import Image

    from common.render.ink import glyph_count, ink_pixels

    for it in items:
        if "ink_a" in it:
            continue
        x0, y0, x1, y1 = it["box"]
        with Image.open(it["file_a"]) as im:
            it["ink_a"] = ink_pixels(im, (8 * x0, 8 * y0, 8 * x1, 8 * y1))
        it["glyphs"] = glyph_count(it["a"])


def rebin(d: dict, by: str) -> list[str]:
    import torch

    pos, sigmas, items = d["pos"], d["sigmas"], d["items"]
    mv = pos[0, :, :, 0] - pos[0, :, :, 1]  # (items, σ), first cond
    keyf = {
        "px": lambda it: _bin(it["px"], PX_EDGES) if "px" in it else None,
        "ink": lambda it: _bin(it["ink_a"] / max(it["glyphs"], 1) / 64, INK_EDGES),
        "font": lambda it: it.get("font"),
        "bubble": lambda it: it.get("bubble"),
    }[by]
    keys = [keyf(it) for it in items]
    L = [
        f"### by {by}",
        "",
        f"| kind | {by} | n | " + " | ".join(f"{s:.2f}" for s in sigmas) + " | peak |",
        "|---|---|---|" + "---|" * len(sigmas) + "---|",
    ]
    for kind in sorted({it["kind"] for it in items}):
        vals = sorted(
            {
                k
                for k, it in zip(keys, items, strict=True)
                if k is not None and it["kind"] == kind
            },
            key=str,
        )
        for v in vals:
            sel = torch.tensor(
                [
                    k == v and it["kind"] == kind
                    for k, it in zip(keys, items, strict=True)
                ]
            )
            m = mv[sel].mean(dim=0)
            L.append(
                f"| {kind} | {v} | {int(sel.sum())} | "
                + " | ".join(f"{float(x):+.3f}" for x in m)
                + f" | {sigmas[int(m.argmax())]:.2f} |"
            )
    L.append("")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pts", nargs="+", type=Path, help="cf_sense.pt files")
    ap.add_argument("--seed", type=int, default=0, help="the run's --seed (replay)")
    ap.add_argument("--size", type=int, default=512, help="the run's --train_size")
    ap.add_argument("--by", default="px,ink,font,bubble")
    ap.add_argument(
        "--out", type=Path, default=None, help="write the markdown here too"
    )
    a = ap.parse_args(argv)
    import torch

    L = []
    for pt in a.pts:
        d = torch.load(pt, weights_only=False)
        items = d["items"]
        lang = (d.get("meta") or {}).get("lang") or (
            "en" if len(d["conds"]) == 1 else "ja"
        )
        if "px" not in items[0] and lang == "en":
            _replay(d, a.seed, a.size)
        _fill_ink(items)
        L += [f"## {pt.parent.parent.name}/{pt.parent.name} ({len(items)} pairs)", ""]
        for by in a.by.split(","):
            if by == "px" and "px" not in items[0]:
                L += [f"### by px — not recorded and not replayable for {lang}", ""]
                continue
            L += rebin(d, by)
    text = "\n".join(L)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
