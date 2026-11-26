#!/usr/bin/env python
"""Attention as a glyph locator — on a generated string page, does a quote token's
cross-attention say where its glyph was drawn (forward only, no training).

Pages: strings of the draw's ``val`` (``strings_file``), ``per_bin`` a length bin, on
``scenes`` × ``seeds``, rendered as ``s1.py`` renders them (the bubble frame, the rows of
``[start]``, 4 Euler steps, the pixel head). A pre-hook on blocks 8, 13, 14, 15 (``src/loc.py``) keeps, at
every step, the cross-attention probability from each image token to each quote token
that has a row, softmax over every key, per head.

Maps, the mean over the steps of ``map_steps``:

- ``b13_15``: blocks 13 – 15, head mean;
- ``heads4``: the heads (15, 1), (14, 15), (13, 7), (8, 14) (``bench/xattn_loc``).

A token's centre on the page, inside the picked box (``gd.pick``; dilated a token):

- ``raw``: the peak of its own map;
- ``share``: the peak of its map over the sum of the quote tokens' maps, on the cells
  where that sum is at or over its median in the box.

The peak is taken on a 3 × 3 mean and refined to the weighted centre of its 3 × 3 cells.
A token's window is a square of ``window`` glyph sides about the centre, a glyph side
being the root of the box's area over the string's glyphs.

Read, per map and centre (a variant), over the pages with a box:

- ``own``: the tuned reader's read of token p's window holds p's glyph (``s1.key``);
  ``own_alone``: the read is that glyph and nothing else;
- ``null``: the read of another token's window holds p's glyph, the other token being
  the one half the string away whose glyph differs;
- both split by whether the page's own read hit position p (``s1.score``);
- ``order``: on boxes at least 2.5 times as long as wide, or the reverse, with three
  tokens or more: Spearman of token order against the centre along the long side;
- ``apart``: adjacent tokens whose centres are half a glyph side apart or more.

Writes ``output/cjk_rl/attn_loc/<config stem>/`` (pages, ``maps.npz``: a page's maps at
every step in fp16, an overlay a page with the centres numbered) and
``results/<ts>-attn_loc-<config stem>/`` (``records.json``, ``result.json``).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/attn_loc.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import glyph_g0 as G
import glyph_scenes as S
import turbo_read as T

import g1  # noqa: E402  (src/, on the path through turbo_read)
import gd  # noqa: E402
import s1  # noqa: E402
from loc import SETS, TOK_PX, Recorder, centres, glyph_side, token_cols, window  # noqa: E402

LINE = Path(__file__).resolve().parents[1]
OUT = T.REPO / "output" / "cjk_rl" / "attn_loc"
CENTRES = ("raw", "share")


def spearman(a, b) -> float | None:
    import numpy as np

    if len(a) < 3 or np.ptp(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def overlay(bgr, box, pts: dict, path: Path) -> None:
    """The page four times, a variant a panel, its centres numbered."""
    from PIL import Image, ImageDraw

    page = Image.fromarray(bgr[:, :, ::-1])
    W, H = page.size
    sheet = Image.new("RGB", (2 * W, 2 * H), "white")
    for n, (name, ps) in enumerate(pts.items()):
        im = page.copy()
        d = ImageDraw.Draw(im)
        d.rectangle(box, outline=(0, 160, 255), width=2)
        for k, (x, y) in enumerate(ps):
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 40, 40))
            d.text((x - 4, y - 6), str(k), fill="white")
        d.text((6, 6), name, fill=(255, 40, 40))
        sheet.paste(im, ((n % 2) * W, (n // 2) * H))
    sheet.resize((W, H)).save(path, quality=85)


def tabulate(recs: list) -> dict:
    out = {}
    for var in [f"{s}/{c}" for s in SETS for c in CENTRES]:
        for b in [f"{lo}-{hi}" for lo, hi in s1.BINS] + ["all"]:
            xs = [r for r in recs if r["box"] and b in ("all", r["bin"])]
            n = own = alone = null = 0
            hit = [0, 0, 0]  # positions the page's read hit: count, own, null
            miss = [0, 0, 0]
            rho, apart, pairs = [], 0, 0
            for r in xs:
                v = r["vars"][var]
                for t, p in enumerate(r["pos"]):
                    o, a, z = v["own"][t], v["alone"][t], v["null"][t]
                    n += 1
                    own += o
                    alone += a
                    null += z or 0
                    cell = hit if r["hit"][p] else miss
                    cell[0] += 1
                    cell[1] += o
                    cell[2] += z or 0
                if v["rho"] is not None:
                    rho.append(v["rho"])
                apart += v["apart"]
                pairs += max(len(r["pos"]) - 1, 0)
            if not n:
                continue
            out[f"{var}/{b}"] = {
                "pages": len(xs),
                "tokens": n,
                "own": round(own / n, 3),
                "own_alone": round(alone / n, 3),
                "null": round(null / n, 3),
                "page_hit": round(hit[0] / n, 3),
                "own_on_hit": round(hit[1] / max(hit[0], 1), 3),
                "null_on_hit": round(hit[2] / max(hit[0], 1), 3),
                "own_on_miss": round(miss[1] / max(miss[0], 1), 3),
                "null_on_miss": round(miss[2] / max(miss[0], 1), 3),
                "order_rho": round(sum(rho) / len(rho), 3) if rho else None,
                "order_boxes": len(rho),
                "apart": round(apart / max(pairs, 1), 3),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="a string a bin, one scene")
    a = ap.parse_args()
    import numpy as np
    import torch

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    scenes, seeds = list(cfg["scenes"]), list(cfg["seeds"])
    per_bin = cfg["per_bin"]
    if a.smoke:
        scenes, per_bin = scenes[:1], 1
    pool = json.loads((T.REPO / cfg["strings_file"]).read_text("utf-8"))["val"]
    strings = [s for b in pool.values() for s in b[:per_bin]]
    frame = cfg["frame"]
    jobs = [(s, si) for s in strings for si in scenes]
    kana = [k for _, k, _ in g1.glyphs(cfg | {"fold_small": False}, False)]
    out = OUT / name
    (out / "pages").mkdir(parents=True, exist_ok=True)

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode([G.framed(sc[si][frame], s) for s, si in jobs])
    rdr = gd.Reader("cuda")
    at, _ = s1.row_map(pol, sc[scenes[0]][frame], kana)
    start = pol.raw.detach().clone()
    if cfg.get("start"):
        start = g1.saved_rows(cfg["start"], {k: at[s1.key(k)] for k in kana}, start)
    pol.raw.data.copy_(start)
    rec = Recorder(pol.anima)
    W, H = G.SHAPE
    gh, gw = H // TOK_PX, W // TOK_PX
    steps = list(cfg["map_steps"])
    print(
        f"{name}: {len(strings)} strings × {len(scenes)} scenes × {len(seeds)} seeds, "
        f"maps over steps {steps} (σ {[round(float(pol.sig[i]), 2) for i in steps]})",
        flush=True,
    )
    recs, saved, t0 = [], {}, time.time()
    for n, (s, si) in enumerate(jobs):
        caption = G.framed(sc[si][frame], s)
        au = s1.units(s)
        ak = [s1.key(u) for u in au]
        cols, groups, pos = token_cols(pol, caption, ak, at)
        e = pol.embed(caption)
        x = pol.noise(seeds, G.SHAPE)
        per_step = []
        rec.cols = torch.tensor(cols, device=pol.dev)
        with torch.no_grad():
            for i in range(pol.steps):
                v = pol.forward(i, x, e)
                per_step.append(rec.sets())  # (B, 2, n, Nq)
                if i < pol.steps - 1:
                    x = pol.euler(i, x, v)
            px = pol.decode(x)
        rec.cols = None
        maps = torch.stack(per_step, 1)  # (B, steps, 2, tokens, Nq)
        maps = torch.stack([maps[..., g, :].mean(-2) for g in groups], -2)
        bgrs, reads = rdr.read(px)
        for b, seed in enumerate(seeds):
            stem = f"{n:03d}_sc{si:03d}_s{seed}"
            case, r = gd.pick(s, reads[b], cfg["conf_min"])
            row = {
                "string": s,
                "bin": s1.bin_of(len(au)),
                "scene": si,
                "seed": seed,
                "stem": stem,
                "case": case,
                "pos": pos,
                "box": None,
            }
            saved[stem] = maps[b].half().cpu().numpy()
            if r is None or not pos:
                recs.append(row)
                continue
            box = [int(v) for v in r["box"]]
            hit, _, _ = s1.score(au, r["sfx"] or "")
            side = glyph_side(box, len(au), cfg["window"])
            m = maps[b, steps].mean(0)  # (2, n, Nq)
            pts = {
                f"{sn}/{c}": p
                for k, sn in enumerate(SETS)
                for c, p in centres(m[k], box, gh, gw).items()
            }
            wins = [window(c, side, W, H) for ps in pts.values() for c in ps]
            got = rdr.ocr.read_boxes_scored(bgrs[b], wins, pad=0.0)
            texts = [s1.keys(g[0]) if g else [] for g in got]
            far = []  # the token whose window is p's null
            for t, p in enumerate(pos):
                order = [(t + len(pos) // 2 + d) % len(pos) for d in range(len(pos))]
                far.append(next((u for u in order if ak[pos[u]] != ak[p]), None))
            bw, bh = box[2] - box[0], box[3] - box[1]
            axis = 1 if bh >= 2.5 * bw else 0 if bw >= 2.5 * bh else None
            row |= {"box": box, "read": r["sfx"], "hit": hit, "side": side, "vars": {}}
            for k, (var, ps) in enumerate(pts.items()):
                tx = texts[k * len(pos) : (k + 1) * len(pos)]
                want = [ak[p] for p in pos]
                rho = None
                if axis is not None and len(pos) >= 3:
                    rho = spearman(np.arange(len(pos)), np.array([c[axis] for c in ps]))
                row["vars"][var] = {
                    "centres": [[round(c[0], 1), round(c[1], 1)] for c in ps],
                    "reads": ["".join(x) for x in tx],
                    "own": [int(w in x) for w, x in zip(want, tx)],
                    "alone": [int(x == [w]) for w, x in zip(want, tx)],
                    "null": [
                        None if far[t] is None else int(w in tx[far[t]])
                        for t, w in enumerate(want)
                    ],
                    "rho": rho,
                    "apart": sum(
                        (
                            (ps[t][0] - ps[t + 1][0]) ** 2
                            + (ps[t][1] - ps[t + 1][1]) ** 2
                        )
                        ** 0.5
                        >= 0.5 * side / cfg["window"]
                        for t in range(len(ps) - 1)
                    ),
                }
            recs.append(row)
            overlay(bgrs[b], box, pts, out / "pages" / f"{stem}.webp")
        if (n + 1) % 16 == 0:
            print(
                f"  caption {n + 1} / {len(jobs)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    np.savez_compressed(out / "maps.npz", **saved)
    table = tabulate(recs)
    print(
        "| map / centre / glyphs | pages | tokens | own | own alone | null | page hit "
        "| own on hit | own on miss | null on miss | order ρ (boxes) | apart |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in table.items():
        print(
            f"| {k} | {v['pages']} | {v['tokens']} | {v['own']} | {v['own_alone']} | "
            f"{v['null']} | {v['page_hit']} | {v['own_on_hit']} | {v['own_on_miss']} | "
            f"{v['null_on_miss']} | {v['order_rho']} ({v['order_boxes']}) | {v['apart']} |",
            flush=True,
        )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"attn_loc-{name}", root=LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/attn_loc.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={
            "pages": len(recs),
            "cases": {
                c: sum(r["case"] == c for r in recs) for c in ("box", "low", "none")
            },
            "table": table,
        },
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
