"""builder — stage config → ``data_scale_<stage>_<tag>/`` (``img/``,
``train.jsonl``, ``eval.json``, ``build.json``, ``sheet_<recipe>.png``).

The gate (design § 4)::

    stage band B
    for each recipe r in the mix (with its share):
        item = r.draw()                      # unit(s), px, layout
        w = window(kind_of(units), px, layout)
        keep iff B ⊆ w (or |B ∩ w| / |B| ≥ min_overlap)   # gate = contain | overlap
        re-draw otherwise (px is what the re-draw moves)

``gate = "none"`` (stage0309, the mixed consolidation pass) keeps every
render. A recipe with no source under the run's inventory (no corpus line
on a 24-row run, no piece) is dropped and the remaining shares renormalise
— ``build.json`` ``dropped`` — the stage file is never edited for it.
Rendering is forked over ``workers`` processes, each on its own seeded
stream drawn from the build seed (deterministic per seed × workers). Every kept record carries ``recipe`` / ``layout`` / ``px`` /
``window`` and the probe's ``glyphs`` / ``ink`` / ``box_area`` ink stats;
``build.json`` holds the per-recipe px medians (the ± 20 % launch gate),
the rejection counts and the config.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import statistics as st
import time
from collections import Counter
from pathlib import Path

from .config import StageConfig
from .paths import data_dir
from .recipes import RECIPES, Pools, build_pools, missing_source
from .windows import covers, kind_of, window

# forked workers inherit an HF tokenizer; its thread pool must not be live
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


# eval.json group order (the probe's, minus the groups this line never draws)
EVAL_ORDER = (
    "single",
    "en",
    "word",
    "single_kanji",
    "single_ext",
    "single_small",
    "single_extra",
    "phrase",
    "phrase_held",
    "short",
    "short_held",
)


def build(
    cfg: StageConfig,
    tag: str,
    n_items: int | None = None,
    seed: int | None = None,
    workers: int | None = None,
) -> Path:
    from common.prompts import TPL_BUBBLE, TPL_EN
    from data.stage import _ink_stats

    t0 = time.time()
    out = data_dir(cfg.stage, tag)
    (out / "img").mkdir(parents=True, exist_ok=True)
    seed = int(cfg.data["seed"] if seed is None else seed)
    n = int(cfg.data["n_items"] if n_items is None else n_items)
    workers = default_workers() if workers is None else max(1, int(workers))
    rng = random.Random(seed)
    pools = build_pools(cfg, out, rng)
    dropped = {
        m.name: r for m in cfg.mix if (r := missing_source(m.name, m.params, pools))
    }
    live = [m for m in cfg.mix if m.name not in dropped]
    assert live, f"every recipe dropped: {dropped}"
    counts = _counts([(m.name, m.share) for m in live], n)
    print(
        f"build {cfg.stage} → {out.name}: band {cfg.band[0]:.2f}–{cfg.band[1]:.2f}, "
        f"gate {cfg.gate}, {n} items {counts}, {workers} workers"
        + (f"; dropped {dropped}" if dropped else ""),
        flush=True,
    )
    recs: list = []
    report: dict = {}
    for m in live:
        got, rep = _build_recipe(
            cfg, m, counts[m.name], pools, rng, out, len(recs), workers
        )
        recs += got
        report[m.name] = rep
    assert recs, "no item survived the gate"
    stats = _ink_stats(recs)
    _px_gate(cfg, recs, report)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
        encoding="utf-8",
    )
    ev = [
        {
            "group": g,
            "text": s,
            "caption": (TPL_EN if g == "en" else TPL_BUBBLE).format(s),
        }
        for g in EVAL_ORDER
        for s in pools.inv.evals.get(g, ())
    ]
    (out / "eval.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    shapes = Counter("x".join(map(str, r["shape"])) for r in recs)
    build = {
        "stage": cfg.stage,
        "tag": tag,
        "band": list(cfg.band),
        "gate": cfg.gate,
        "min_overlap": cfg.min_overlap,
        "seed": seed,
        "n_items": n,
        "config": str(cfg.path),
        "run": cfg.run.name if cfg.run else None,
        "run_config": str(cfg.run.path) if cfg.run else None,
        "data": cfg.data,
        "mix": [{"recipe": m.name, "share": m.share, **m.params} for m in cfg.mix],
        "dropped": dropped,
        "shares": {m.name: round(counts[m.name] / n, 3) for m in live},
        "workers": workers,
        "recipes": report,
        "ink_stats": {k: {"px": v[0], "ink": v[1]} for k, v in stats.items()},
        "shapes": dict(sorted(shapes.items())),
        "eval": Counter(e["group"] for e in ev),
        "n_train": len(recs),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "build.json").write_text(
        json.dumps(build, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"data: {len(recs)} train items "
        f"{dict(Counter(r['recipe'] for r in recs))}; eval {len(ev)} prompts "
        f"{dict(build['eval'])}; shapes {build['shapes']}; {build['minutes']} min",
        flush=True,
    )
    return out


def _counts(shares: list, n: int) -> dict:
    """Items per recipe from ``[(name, share)]`` — the shares of the live
    recipes, renormalised (a dropped recipe's share goes to the rest)."""
    total = sum(w for _, w in shares)
    counts = {name: int(n * w / total) for name, w in shares}
    top = max(shares, key=lambda x: x[1])[0]
    counts[top] += n - sum(counts.values())
    return counts


# fork-inherited job state (set before the pool forks; never pickled)
_JOB: dict = {}


def _worker(job):
    w, n, first, seed = job
    return _draw_loop(
        _JOB["cfg"],
        _JOB["m"],
        n,
        _JOB["pools"],
        random.Random(seed),
        _JOB["out"],
        first,
    )


def _build_recipe(
    cfg, m, n: int, pools: Pools, rng, out: Path, first: int, workers: int = 1
):
    t0 = time.time()
    if workers <= 1 or n < 4 * workers:
        kept, rejects, px_seen, tries = _draw_loop(cfg, m, n, pools, rng, out, first)
    else:
        per = [n // workers + (i < n % workers) for i in range(workers)]
        jobs = [
            (i, per[i], first + sum(per[:i]), rng.randrange(2**31))
            for i in range(workers)
        ]
        _JOB.update(cfg=cfg, m=m, pools=pools, out=out)
        try:
            with mp.get_context("fork").Pool(workers) as pool:
                parts = pool.map(_worker, jobs)
        finally:
            _JOB.clear()
        kept, rejects, px_seen, tries = [], Counter(), [], 0
        for k, rj, px, t in parts:
            kept += k
            rejects.update(rj)
            px_seen += px
            tries += t
        pools.used.update(Counter(r["scene"] for r in kept if "scene" in r))
    if len(kept) < n:
        print(
            f"  {m.name}: WARNING {len(kept)}/{n} items after {tries} tries "
            f"(rejects {dict(rejects)})",
            flush=True,
        )
    q = _quantiles([r["px"] for r in kept]) if kept else None
    drawn = _quantiles(px_seen) if px_seen else None
    rep = {
        "n": len(kept),
        "planned": n,
        "tries": tries,
        "rejects": dict(rejects),
        "px_kept": q,
        "px_drawn": drawn,
        "windows": dict(Counter(json.dumps(r["window"]) for r in kept)),
        "horizontal": _n_horizontal(kept),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(
        f"  {m.name}: {len(kept)}/{n} in {tries} tries, rejects {dict(rejects)}; "
        f"px kept {_fmt(q)} (drawn {_fmt(drawn)}); windows {rep['windows']}",
        flush=True,
    )
    _sheet(rng, kept, out / f"sheet_{m.name}.png")
    return kept, rep


def _draw_loop(cfg, m, n: int, pools: Pools, rng, out: Path, first: int):
    """Draw ``n`` items of recipe ``m`` through the band gate; files are
    numbered from ``first``. Returns (kept, rejects, drawn px, tries)."""
    draw = RECIPES[m.name]
    band = cfg.band
    kept, rejects = [], Counter()
    px_seen: list = []
    tries, max_tries = 0, 4 * n + 50
    while len(kept) < n and tries < max_tries:
        tries += 1
        item = draw(pools, rng, m.params)
        if item is None:
            rejects["render"] += 1
            continue
        px = item.px()
        px_seen.append(px)
        kind = kind_of(item.units, pools.n_tokens)
        w = window(kind, px, item.layout)
        if cfg.gate != "none" and not covers(band, w, cfg.min_overlap):
            rejects["no_window" if w is None else "band"] += 1
            continue
        i = first + len(kept)
        fn = out / "img" / f"{m.name}_{i:06d}.png"
        item.image.save(fn)
        rec = {
            "file": str(fn),
            "text": item.text,
            "caption": item.caption,
            "src": item.src,
            "kind": m.name,
            "recipe": m.name,
            "layout": item.layout,
            "units": item.units,
            "shape": list(item.shape),
            "px": round(px, 1),
            "law_kind": kind,
            "window": list(w.band) if w else None,
            **item.extra,
        }
        if item.layout == "scene":
            rec["box"] = item.boxes[0]
        else:
            rec["boxes"] = item.boxes
        kept.append(rec)
    return kept, rejects, px_seen, tries


def _n_horizontal(recs) -> dict:
    """Items / cells drawn as left-to-right lines (``horizontal_frac``):
    scene items carry a bool, grid items the list of line cells; single
    glyphs have no orientation and are counted under ``no_orientation``."""
    n = Counter()
    for r in recs:
        h = r.get("horizontal")
        if isinstance(h, list):
            n["cells"] += len(r["units"])
            n["cells_horizontal"] += len(h)
            n["cells_no_orientation"] += sum(len(u) == 1 for u in r["units"])
        elif len(r["units"]) == 1 and len(r["units"][0]) == 1:
            n["no_orientation"] += 1
        else:
            n["items"] += 1
            n["items_horizontal"] += int(bool(h))
    return dict(n)


def _px_gate(cfg, recs, report):
    """The ± 20 % launch gate (the reads' contract, data/stage.py): a recipe
    that declares ``px_target`` must *draw* its median px within 20 % of it.
    Read on the drawn px, before the band gate — the gate truncates the kept
    distribution (stage0709 keeps the ≥ 40 px side of the bubble-fit draw),
    and the contract is about what the recipe renders."""
    bad = []
    for m in cfg.mix:
        t = m.params.get("px_target")
        if m.name not in report or not t or not report[m.name]["px_drawn"]:
            continue
        med = report[m.name]["px_drawn"]["median"]
        report[m.name]["px_target"] = t
        report[m.name]["px_ok"] = abs(med - t) <= 0.2 * t
        if not report[m.name]["px_ok"]:
            bad.append(f"{m.name}: drawn median px {med:.0f} vs target {t} ± 20 %")
    if bad:
        print("px gate FAILED — " + "; ".join(bad), flush=True)
        raise SystemExit("px gate: " + "; ".join(bad))


def _quantiles(xs):
    xs = sorted(xs)
    return {
        "median": round(st.median(xs), 1),
        "p10": round(xs[int(0.1 * (len(xs) - 1))], 1),
        "p90": round(xs[int(0.9 * (len(xs) - 1))], 1),
    }


def _fmt(q) -> str:
    return "–" if not q else f"{q['median']:.0f} ({q['p10']:.0f}–{q['p90']:.0f})"


def _sheet(rng, recs, path: Path, n: int = 24):
    from common.readers import contact_sheet
    from PIL import Image, ImageDraw

    if not recs:
        return
    tiles = []
    for r in rng.sample(recs, min(n, len(recs))):
        im = Image.open(r["file"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in r.get("boxes") or [r["box"]]:
            d.rectangle(b, outline=(0, 255, 0), width=2)
        h = r.get("horizontal")
        tag = " H" if (h is True or (isinstance(h, list) and h)) else ""
        tiles.append((im, [r["text"][:24], f"{r['px']:.0f} px {r['layout']}{tag}"]))
    contact_sheet(tiles, path, thumb=192, cols=6)
