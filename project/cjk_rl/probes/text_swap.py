#!/usr/bin/env python
"""Text swap — which steps' text a page's lettering follows (forward only).

A render runs its first ``k`` steps on caption A and the rest on caption B (the same
scene, another string in the quote), on the rows of the config's ``[table]`` (a G1 save
read as saved). The page is read as any page is and its closest box aligned to A and to B
(``word_read.page``). ``k = 3`` gives B to the last step alone (σ 0.5 → 0, the step G1's
backward covers), ``k = 2`` to the last two (σ 0.75 → 0.5 → 0), ``k = 0`` is B's own
render.

Config: ``singles`` (a string of glyphs) and ``words`` (a list, or ``words_file``, a JSON
list under the repo; in runs of ``bin`` strings of one length class); A is a string and B the next of its run. ``ks`` are the
arms. ``ref`` names a ``row_line.py`` result whose arm ``ref_arm`` holds the words'
``k = 0`` pages (same rows, scenes, seeds); words are then rendered at the other ``ks``
only.

Tables, per class and ``k``: pages, B alone in a box, A alone in a box, the share of B's
positions hit, of A's, glyphs in the box over B's length.

Pages → ``output/cjk_rl/text_swap/<stem>/``, result → ``results/<ts>-text_swap-<stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/text_swap.py --config project/cjk_rl/configs/<name>.toml"
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
import word_read as WR

import gd  # noqa: E402  (src/, on the path through turbo_read)

ROOT = T.REPO / "output" / "cjk_rl" / "text_swap"
G1_OUT = T.REPO / "output" / "cjk_rl" / "g1"
LINE = Path(__file__).resolve().parents[1]


def pairs(cfg: dict, smoke: bool) -> list:
    """``[(cls, a, b)]``."""
    out = []
    s = list(cfg.get("singles", ""))
    out += [("single", a, s[(n + 1) % len(s)]) for n, a in enumerate(s)]
    w, m = list(cfg.get("words", [])), cfg.get("bin", 12)
    if cfg.get("words_file"):
        w = json.loads((T.REPO / cfg["words_file"]).read_text("utf-8"))
    for i in range(0, len(w), m):
        run = w[i : i + m]
        out += [("word", a, run[(n + 1) % len(run)]) for n, a in enumerate(run)]
    if smoke:
        out = [p for p in out if p[0] == "single"][:1] + [
            p for p in out if p[0] == "word"
        ][:1]
    return out


def cls_of(cls: str, b: str) -> str:
    if cls == "single":
        return "single"
    return str(len(b)) if len(b) <= 4 else "5-8"


def main() -> None:
    import torch
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="two pairs, one scene")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    scenes = list(cfg["read_scenes"])[: 1 if a.smoke else None]
    seeds = list(cfg["read_seeds"])
    ps = pairs(cfg, a.smoke)
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    recs = []
    if cfg.get("ref"):
        (ref_file,) = sorted((LINE / "results").glob(f"*-row_line-{cfg['ref']}"))
        at = {
            (r["key"], r["scene"], r["seed"]): r
            for r in json.loads((ref_file / "records.json").read_text("utf-8"))
            if r["arm"] == cfg["ref_arm"]
        }
        for cls, ka, kb in ps:
            if cls != "word":
                continue
            for si in scenes:
                for seed in seeds:
                    r = at[(kb, si, seed)]
                    recs.append(
                        {
                            "cls": cls,
                            "a": ka,
                            "b": kb,
                            "k": 0,
                            "scene": si,
                            "seed": seed,
                        }
                        | {"boxes": r["boxes"], "file": r["file"]}
                    )

    frame = cfg["frame"]
    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode(
        [
            G.framed(sc[si][frame], k)
            for _, ka, kb in ps
            for k in (ka, kb)
            for si in scenes
        ]
    )
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
        f"{name}: {len(ps)} pairs × scenes {scenes} × seeds {seeds} × k {cfg['ks']}; "
        f"rows {cfg['table']}; {len(recs)} reference pages",
        flush=True,
    )

    t0 = time.time()
    for n, (cls, ka, kb) in enumerate(ps):
        ks = [k for k in cfg["ks"] if not (k == 0 and cls == "word" and cfg.get("ref"))]
        for si in scenes:
            ea = pol.embed(G.framed(sc[si][frame], ka))
            eb = pol.embed(G.framed(sc[si][frame], kb))
            for k in ks:
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
                    recs.append(
                        {
                            "cls": cls,
                            "a": ka,
                            "b": kb,
                            "k": k,
                            "scene": si,
                            "seed": seed,
                        }
                        | {"boxes": G.boxes(reads[b]), "file": str(f)}
                    )
        if n % 8 == 7:
            print(
                f"  pair {n + 1} / {len(ps)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )

    table_out = {}
    print(
        "| class | k | pages | B alone | A alone | B's positions | A's positions | drawn / B's length |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for c in ("single", "2", "3", "4", "5-8"):
        for k in sorted({r["k"] for r in recs}):
            xs = [r for r in recs if r["k"] == k and cls_of(r["cls"], r["b"]) == c]
            if not xs:
                continue
            pb = [WR.page(r["b"], r["boxes"], True) for r in xs]
            pa = [WR.page(r["a"], r["boxes"], True) for r in xs]
            nb, na = sum(len(r["b"]) for r in xs), sum(len(r["a"]) for r in xs)
            cell = {
                "pages": len(xs),
                "b_solo": sum(p["solo"] for p in pb),
                "a_solo": sum(p["solo"] for p in pa),
                "b_pos": round(sum(sum(p["hit"]) for p in pb) / nb, 3),
                "a_pos": round(sum(sum(p["hit"]) for p in pa) / na, 3),
                "drawn_over_b": round(sum(p["drawn"] for p in pb) / nb, 2),
            }
            table_out[f"{c}/k{k}"] = cell
            print(
                f"| {c} | {k} | {cell['pages']} | {cell['b_solo']} | {cell['a_solo']} | "
                f"{cell['b_pos']} | {cell['a_pos']} | {cell['drawn_over_b']} |",
                flush=True,
            )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"text_swap-{name}", root=LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/text_swap.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={"table": table_out, "sigmas": [float(s) for s in pol.sig]},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
