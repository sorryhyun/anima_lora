#!/usr/bin/env python
"""Layout hold — what fixes a page's bubble: the (scene, seed), the string, or the steps
before σ 0.75 (forward only).

Two sets of pages on the rows of the config's ``[table]``, each read as any page is, with
the boxes' geometry and the bubble kept:

    own    every string on every read scene at ``own_seeds`` (``k = 0``).
    swap   a render's first ``k`` steps on string A and the rest on string B, A and B of
           different lengths (a short string against a long one, both ways), at
           ``swap_seeds`` and ``ks``. ``k = 2`` gives B to σ 0.75 → 0.5 → 0.

The bubble of a page is the connected bright region (every channel at ``BRIGHT`` or over,
eroded by ``ERODE`` pixels so a faint or open outline still closes it) under the picked
box (``gd.pick``), holes filled; a region over ``MAX_SHARE`` of the page is a leak into
the background and counts as no bubble. Two pages' layouts are compared by
the IoU of their bubbles.

Tables: the own pages by length; pairs of own pages of different strings by level (the
same scene and seed, the same scene at another seed, another scene) and by the two
lengths, with the picked boxes' centres beside the bubbles (the bubble misses on a
shaded or open bubble; the box does not); the picked box's way a (scene, seed); the swap
pages against A's and B's own pages of the same scene and seed.

Config: ``words_file`` (a JSON list under the repo; strings of 2 glyphs are ``short``,
of 5 or more ``long``, the rest left out), ``read_scenes``, ``own_seeds``, ``swap_seeds``,
``ks``, ``frame``, ``rows``, ``[table]``.

Pages and row sheets → ``output/cjk_rl/layout_hold/<stem>/``, result →
``results/<ts>-layout_hold-<stem>/`` (``records.json``, ``masks.npz``).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/layout_hold.py --config project/cjk_rl/configs/<name>.toml"
    .venv/bin/python project/cjk_rl/probes/layout_hold.py --tabulate project/cjk_rl/results/<ts>-layout_hold-<stem>   # CPU, writes tables.json
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from itertools import combinations
from pathlib import Path

import glyph_g0 as G
import glyph_scenes as S
import turbo_read as T
import word_read as WR

import gd  # noqa: E402  (src/, on the path through turbo_read)

ROOT = T.REPO / "output" / "cjk_rl" / "layout_hold"
G1_OUT = T.REPO / "output" / "cjk_rl" / "g1"
LINE = Path(__file__).resolve().parents[1]
BRIGHT = 245
ERODE = 5
MAX_SHARE = 0.5
EDGE = 1.15  # orient_read's
M = 96  # a stored mask's side


def bubble(bgr, box):
    """``(mask, why)``: the filled bright region under ``box`` at the page's size."""
    import cv2
    import numpy as np

    if box is None:
        return None, "no box"
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    ker = np.ones((ERODE, ERODE), np.uint8)
    n, lab = cv2.connectedComponents(
        cv2.erode((bgr.min(axis=2) >= BRIGHT).astype(np.uint8), ker), connectivity=4
    )
    ids = np.bincount(lab[y0:y1, x0:x1].ravel(), minlength=n)
    ids[0] = 0
    if ids.max() == 0:
        return None, "no bright"
    comp = cv2.dilate((lab == ids.argmax()).astype(np.uint8), ker)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(comp, cs, -1, 1, cv2.FILLED)
    if comp.mean() > MAX_SHARE:
        return None, "leak"
    return comp, "ok"


def geometry(key: str, bgr, reads: list) -> tuple[dict, object]:
    """A page's record fields and its mask at ``M`` × ``M`` (``None`` with no bubble)."""
    import cv2
    import numpy as np
    from common.text import norm

    bs = [r for r in reads if not r.get("whole")]
    case, pick = gd.pick(key, reads)
    out = {
        "boxes": G.boxes(reads),
        "geo": [
            {
                "box": [int(v) for v in r["box"]],
                "text": norm(r["sfx"]) if r.get("sfx") else "",
                "conf": float(r.get("sfx_conf") or 0.0),
            }
            for r in bs
        ],
        "case": case,
        "pick": None,
        "way": "none",
    }
    if pick is not None:
        x0, y0, x1, y1 = (int(v) for v in pick["box"])
        w, h = x1 - x0, y1 - y0
        out["pick"] = [x0, y0, x1, y1]
        out["way"] = "tall" if h > EDGE * w else "wide" if w > EDGE * h else "square"
    mask, why = bubble(bgr, out["pick"])
    out["bubble"] = why
    if mask is None:
        return out, None
    ys, xs = np.nonzero(mask)
    out |= {
        "b_share": round(float(mask.mean()), 4),
        "b_box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
    }
    small = cv2.resize(mask.astype(np.float32), (M, M), interpolation=cv2.INTER_AREA)
    return out, small >= 0.5


def med(xs) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return round(float(xs[n // 2] + xs[(n - 1) // 2]) / 2, 3)


def centre_gap(a: dict, b: dict) -> float | None:
    """The distance between two pages' picked boxes' centres over the page's side."""
    if not (a["pick"] and b["pick"]):
        return None
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a["pick"], b["pick"]
    dx, dy = (ax0 + ax1 - bx0 - bx1) / 2, (ay0 + ay1 - by0 - by1) / 2
    return (dx * dx + dy * dy) ** 0.5 / G.SHAPE[0]


def cls_of(s: str) -> str:
    return "short" if len(s) <= 2 else "long"


def tabulate(recs: list, masks) -> dict:
    """The three tables, printed and returned. ``masks[i]`` is record ``i``'s mask (all
    zero with no bubble)."""
    import numpy as np

    flat = masks.reshape(len(recs), -1).astype(np.float32)
    area = flat.sum(1)

    def iou(i: int, j: int) -> float:
        inter = float(flat[i] @ flat[j])
        return inter / max(area[i] + area[j] - inter, 1.0)

    own = [i for i, r in enumerate(recs) if r["k"] == 0]
    out = {"own": {}, "pairs": {}, "swap": {}}

    print("\nOwn pages (k = 0)")
    print(
        "| length | pages | no box | no bright | leak | boxes at 0.6 | tall | wide | square "
        "| bubble share, median | bubble h / w, median | positions hit | whole string |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ("short", "long"):
        xs = [recs[i] for i in own if cls_of(recs[i]["b"]) == c]
        if not xs:
            continue
        pg = [WR.page(r["b"], r["boxes"], True) for r in xs]
        ok = [r for r in xs if r["bubble"] == "ok"]
        cell = {
            "pages": len(xs),
            "no_box": sum(r["bubble"] == "no box" for r in xs),
            "no_bright": sum(r["bubble"] == "no bright" for r in xs),
            "leak": sum(r["bubble"] == "leak" for r in xs),
            "boxes": round(
                sum(sum(g["conf"] >= 0.6 for g in r["geo"]) for r in xs) / len(xs), 2
            ),
            "tall": sum(r["way"] == "tall" for r in xs),
            "wide": sum(r["way"] == "wide" for r in xs),
            "square": sum(r["way"] == "square" for r in xs),
            "b_share": med(r["b_share"] for r in ok),
            "b_aspect": med(
                (r["b_box"][3] - r["b_box"][1]) / (r["b_box"][2] - r["b_box"][0])
                for r in ok
            ),
            "pos": round(
                sum(sum(p["hit"]) for p in pg) / sum(len(r["b"]) for r in xs), 3
            ),
            "solo": sum(p["solo"] for p in pg),
        }
        out["own"][c] = cell
        print(
            f"| {c} | {cell['pages']} | {cell['no_box']} | {cell['no_bright']} | "
            f"{cell['leak']} | {cell['boxes']} | {cell['tall']} | {cell['wide']} | "
            f"{cell['square']} | {cell['b_share']} | {cell['b_aspect']} | {cell['pos']} | "
            f"{cell['solo']} |",
            flush=True,
        )

    print("\nPairs of own pages of different strings, both with a bubble")
    print(
        "| level | lengths | pairs | bubble IoU, median | IoU ≥ 0.5 | same way "
        "| pairs with a picked box | centres within 0.1 of the side |"
    )
    print("|---|---|---|---|---|---|---|---|")
    acc: dict = {}
    gaps: dict = {}
    for i, j in combinations(own, 2):
        a, b = recs[i], recs[j]
        if a["b"] == b["b"]:
            continue
        level = (
            "another scene"
            if a["scene"] != b["scene"]
            else "same scene and seed"
            if a["seed"] == b["seed"]
            else "same scene, another seed"
        )
        rel = "–".join(sorted((cls_of(a["b"]), cls_of(b["b"])), reverse=True))
        gap = centre_gap(a, b)
        if gap is not None:
            gaps.setdefault((level, rel), []).append(gap)
        if a["bubble"] == b["bubble"] == "ok":
            acc.setdefault((level, rel), []).append((iou(i, j), a["way"] == b["way"]))
    for level in ("same scene and seed", "same scene, another seed", "another scene"):
        for rel in ("short–short", "long–long", "short–long"):
            v = acc.get((level, rel), [])
            if not v:
                continue
            cell = {
                "pairs": len(v),
                "iou": med(x for x, _ in v),
                "iou_half": round(sum(x >= 0.5 for x, _ in v) / len(v), 3),
                "same_way": round(sum(w for _, w in v) / len(v), 3),
                "picked": len(g := gaps[(level, rel)]),
                "centre_near": round(sum(x < 0.1 for x in g) / len(g), 3),
            }
            out["pairs"][f"{level}/{rel}"] = cell
            print(
                f"| {level} | {rel} | {cell['pairs']} | {cell['iou']} | "
                f"{cell['iou_half']} | {cell['same_way']} | {cell['picked']} | "
                f"{cell['centre_near']} |",
                flush=True,
            )

    print("\nThe picked box's way on the long strings' own pages, a (scene, seed)")
    print("| scene | seed | tall | wide | square | no box |")
    print("|---|---|---|---|---|---|")
    out["cells"] = {}
    for si, seed in sorted({(recs[i]["scene"], recs[i]["seed"]) for i in own}):
        ws = [
            recs[i]["way"]
            for i in own
            if (recs[i]["scene"], recs[i]["seed"]) == (si, seed)
            and cls_of(recs[i]["b"]) == "long"
        ]
        cell = {w: ws.count(w) for w in ("tall", "wide", "square", "none")}
        out["cells"][f"{si}/{seed}"] = cell
        print(
            f"| {si} | {seed} | {cell['tall']} | {cell['wide']} | {cell['square']} | "
            f"{cell['none']} |",
            flush=True,
        )

    print("\nSwap pages against A's and B's own pages of the same scene and seed")
    print(
        "| A → B | k | pages | all three with a bubble | IoU to A's, median | IoU to B's, median "
        "| A's to B's, median | closer to A's | bubble share A's / swap / B's "
        "| B's positions, swap / own | A's positions, swap | B whole, swap / own "
        "| way as A's / as B's |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    at = {(r["b"], r["scene"], r["seed"]): i for i, r in enumerate(recs) if r["k"] == 0}
    for d in ("short → long", "long → short"):
        for k in sorted({r["k"] for r in recs} - {0}):
            idx = [
                i
                for i, r in enumerate(recs)
                if r["k"] == k and f"{cls_of(r['a'])} → {cls_of(r['b'])}" == d
            ]
            if not idx:
                continue
            tri = []
            for i in idx:
                r = recs[i]
                ia = at[(r["a"], r["scene"], r["seed"])]
                ib = at[(r["b"], r["scene"], r["seed"])]
                if all(recs[x]["bubble"] == "ok" for x in (i, ia, ib)):
                    tri.append((i, ia, ib))
            xs = [recs[i] for i in idx]
            ob = [recs[at[(r["b"], r["scene"], r["seed"])]] for r in xs]
            oa = [recs[at[(r["a"], r["scene"], r["seed"])]] for r in xs]
            pb = [WR.page(r["b"], r["boxes"], True) for r in xs]
            pa = [WR.page(r["a"], r["boxes"], True) for r in xs]
            po = [WR.page(r["b"], r["boxes"], True) for r in ob]
            nb, na = sum(len(r["b"]) for r in xs), sum(len(r["a"]) for r in xs)
            cell = {
                "pages": len(xs),
                "tri": len(tri),
                "iou_a": med(iou(i, ia) for i, ia, _ in tri),
                "iou_b": med(iou(i, ib) for i, _, ib in tri),
                "iou_ab": med(iou(ia, ib) for _, ia, ib in tri),
                "closer_a": round(
                    sum(iou(i, ia) > iou(i, ib) for i, ia, ib in tri)
                    / max(len(tri), 1),
                    3,
                ),
                "share_a": med(recs[ia]["b_share"] for _, ia, _ in tri),
                "share_swap": med(recs[i]["b_share"] for i, _, _ in tri),
                "share_b": med(recs[ib]["b_share"] for _, _, ib in tri),
                "b_pos": round(sum(sum(p["hit"]) for p in pb) / nb, 3),
                "b_pos_own": round(sum(sum(p["hit"]) for p in po) / nb, 3),
                "a_pos": round(sum(sum(p["hit"]) for p in pa) / na, 3),
                "b_solo": sum(p["solo"] for p in pb),
                "b_solo_own": sum(p["solo"] for p in po),
                "way_a": round(
                    sum(r["way"] == o["way"] for r, o in zip(xs, oa)) / len(xs), 3
                ),
                "way_b": round(
                    sum(r["way"] == o["way"] for r, o in zip(xs, ob)) / len(xs), 3
                ),
            }
            out["swap"][f"{d}/k{k}"] = cell
            print(
                f"| {d} | {k} | {cell['pages']} | {cell['tri']} | {cell['iou_a']} | "
                f"{cell['iou_b']} | {cell['iou_ab']} | {cell['closer_a']} | "
                f"{cell['share_a']} / {cell['share_swap']} / {cell['share_b']} | "
                f"{cell['b_pos']} / {cell['b_pos_own']} | {cell['a_pos']} | "
                f"{cell['b_solo']} / {cell['b_solo_own']} | "
                f"{cell['way_a']} / {cell['way_b']} |",
                flush=True,
            )
    return out


def sheets(recs: list, out: Path) -> None:
    """Row sheets for viewing, one a pair on one of its (scene, seed): A's own page, the
    swap pages by ``k``, B's own page; the picked box in red, the bubble's box in green."""
    import cv2
    import numpy as np

    at = {(r["a"], r["b"], r["k"], r["scene"], r["seed"]): r for r in recs}
    ks = sorted({r["k"] for r in recs} - {0})
    pairs = list(dict.fromkeys((r["a"], r["b"]) for r in recs if r["k"]))
    cells = sorted({(r["scene"], r["seed"]) for r in recs if r["k"]})
    if not pairs:
        return
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    names = []
    for n, (ka, kb) in enumerate(pairs):
        si, seed = cells[n % len(cells)]
        row = [at[(ka, ka, 0, si, seed)]]
        row += [at[(ka, kb, k, si, seed)] for k in ks]
        row += [at[(kb, kb, 0, si, seed)]]
        tiles = []
        for x in row:
            im = cv2.imread(x["file"])
            if x.get("b_box"):
                x0, y0, x1, y1 = x["b_box"]
                cv2.rectangle(im, (x0, y0), (x1, y1), (0, 160, 0), 3)
            if x["pick"]:
                x0, y0, x1, y1 = x["pick"]
                cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)
            tiles.append(cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA))
        cv2.imwrite(
            str(out / "sheets" / f"{n:02d}_sc{si:02d}_s{seed}.webp"),
            np.concatenate(tiles, axis=1),
            [cv2.IMWRITE_WEBP_QUALITY, 90],
        )
        names.append(f"{n:02d}: {ka} own | {ka} → {kb} at k {ks} | {kb} own")
    (out / "sheets" / "rows.txt").write_text("\n".join(names) + "\n", "utf-8")


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="two strings, one scene")
    ap.add_argument("--tabulate", help="a result folder: the tables again, on the CPU")
    a = ap.parse_args()
    if a.tabulate:
        d = Path(a.tabulate)
        tabs = tabulate(
            json.loads((d / "records.json").read_text("utf-8")),
            np.load(d / "masks.npz")["masks"],
        )
        (d / "tables.json").write_text(json.dumps(tabs, indent=1), "utf-8")
        return

    import torch
    from PIL import Image

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    scenes = list(cfg["read_scenes"])[: 1 if a.smoke else None]
    words = json.loads((T.REPO / cfg["words_file"]).read_text("utf-8"))
    short = [w for w in words if len(w) == 2]
    long = [w for w in words if len(w) >= 5]
    n = min(len(short), len(long), 1 if a.smoke else 10**6)
    short, long = short[:n], long[:n]
    ps = [(s, g) for s, g in zip(short, long)] + [(g, s) for s, g in zip(short, long)]
    own_seeds, swap_seeds = list(cfg["own_seeds"]), list(cfg["swap_seeds"])
    assert set(swap_seeds) <= set(own_seeds), "a swap page needs its own pages"
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    frame = cfg["frame"]
    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode([G.framed(sc[si][frame], w) for w in short + long for si in scenes])
    rdr = gd.Reader("cuda")
    sv = torch.load(
        G1_OUT / cfg["table"]["run"] / f"rows_{cfg['table']['save']}.pt",
        weights_only=False,
    )
    table = pol.raw.detach().float().cpu().clone()
    for i in sv["rows"].values():
        table[i] = sv["raw"][i].float()
    pol.raw.data.copy_(table)
    print(
        f"{name}: {len(short)} short + {len(long)} long strings × scenes {scenes}; own "
        f"seeds {own_seeds}, swap seeds {swap_seeds} × k {cfg['ks']}; rows {cfg['table']}",
        flush=True,
    )

    recs, masks = [], []

    def render(ea, eb, k: int, seeds: list, ka: str, kb: str, si: int) -> None:
        with torch.no_grad():
            x = pol.noise(seeds, G.SHAPE)
            for i in range(pol.steps):
                v = pol.forward(i, x, ea if i < k else eb)
                if i < pol.steps - 1:
                    x = pol.euler(i, x, v)
            px = pol.decode(x)
        bgrs, reads = rdr.read(px)
        for b, seed in enumerate(seeds):
            f = out / f"k{k}" / f"{ka}_{kb}_sc{si:02d}_s{seed}.webp"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(bgrs[b][:, :, ::-1]).save(f, quality=95)
            geo, mask = geometry(kb, bgrs[b], reads[b])
            recs.append(
                {"a": ka, "b": kb, "k": k, "scene": si, "seed": seed, "file": str(f)}
                | geo
            )
            masks.append(mask if mask is not None else np.zeros((M, M), bool))

    t0 = time.time()
    for si in scenes:
        emb = {w: pol.embed(G.framed(sc[si][frame], w)) for w in short + long}
        for w in short + long:
            render(emb[w], emb[w], 0, own_seeds, w, w, si)
        for k in cfg["ks"]:
            for ka, kb in ps:
                render(emb[ka], emb[kb], k, swap_seeds, ka, kb, si)
        print(
            f"  scene {si}: {len(recs)} pages ({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )

    masks = np.stack(masks)
    tabs = tabulate(recs, masks)
    sheets(recs, out)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"layout_hold-{name}", root=LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    np.savez_compressed(run_dir / "masks.npz", masks=masks)
    write_result(
        run_dir,
        script="project/cjk_rl/probes/layout_hold.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={"tables": tabs, "sigmas": [float(s) for s in pol.sig]},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
