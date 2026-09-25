"""builder — a run's vocabs → ``<run>/data/`` (``img/``, ``train.jsonl``,
``eval.json``, ``vocabs.json``, ``build.json``, ``sheet_<group>_<recipe>.png``).

The recipe table is by kind (plan.md § 2): which tiers run is decided by
which kinds the run's vocabs hold — no shares to set, no stage. A tier is a
recipe drawn for one band; a **band group** is the tiers of one kind at one
band. Every item is stamped with its band — ``windows.window(kind, px,
layout)`` at build time — and the trainer draws its σ inside it::

    for each band group g of the vocabs' kinds (n = ITEMS_PER_VOCAB × vocabs of
    the kind × g.share, split over g's tiers by weight):
        item = tier.draw()                          # vocab(s), px, layout
        w = window(kind_of(vocabs), px, layout)
        keep iff g.band ⊆ w (or |g.band ∩ w| / |g.band| ≥ MIN_OVERLAP); re-draw otherwise
        item.band = w.band                          # the band the trainer draws σ in

The tiers and their weights are the old stage files' mixes (``_archive/configs/
stage0709|0507|0305.toml``) with the stage gone, so a band group draws the
item stream that stage's build drew: each group restarts from the same pool
state and the same ``Random(SEED)`` state (the stage builds each started
from ``Random(seed)`` + a fresh ``build_pools``). Rendering forks over
``workers`` processes, each on a seeded stream drawn from the group's rng
(deterministic per seed × workers). A tier whose source is empty (three
pieces fill no 2×2 grid) is skipped and says so — its items go nowhere else.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import statistics as st
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import SEED, RunConfig, phrase_file
from .paths import SEED_ROWS, data_dir
from .recipes import RECIPES, Pools, build_pools, missing_source
from .windows import covers, kind_of, window

# forked workers inherit an HF tokenizer; its thread pool must not be live
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# items per vocab: run0925_300f's two stage dirs of 10 000 over its 300 piece
# vocabs (≈ 67; plan.md § 2 "~70"), so its re-run (plan.md § 6-3) draws the same volume
ITEMS_PER_VOCAB = 20_000 / 300
# the gate's overlap threshold (every stage file of record: gate = contain, min_overlap 0.8)
MIN_OVERLAP = 0.8


@dataclass(frozen=True)
class Tier:
    recipe: str
    weight: float  # within the group, as the stage file's share (renormalised over the group)
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Group:
    name: str  # file prefix + build.json key
    kind: str  # the vocab kind that brings the group in
    band: tuple
    share: float  # of the kind's items
    tiers: tuple


TABLE = (
    # single vocabs: stage0709 — glyph identity at ≥ 48 px, the bubble fit and
    # the 1×1–3×3 grids (band_b1 B.1, step1_0921; band law § 3)
    Group(
        "b0709",
        "single",
        (0.7, 0.9),
        1.0,
        (
            Tier("scene_single", 0.5, {"fill": 0.7, "min_glyph": 28, "px_target": 50}),
            Tier(
                "grid_single",
                0.5,
                {
                    "grids": "1x1:1,2x2:1,3x3:1,2x3:1,3x2:1",
                    "fill": [
                        0.3,
                        0.8,
                    ],  # × cell short side: 3×3 → 51–136 px, 1×1 → 150–400
                    "bubble_frac": 0.5,
                    "mark_horizontal": True,
                },
            ),
        ),
    ),
    # piece vocabs, the large tier: stage0507 — one piece per bubble ≈ 35–48 px,
    # pieces in 24–32 px word cells, 2–5-piece lines ≈ 32 px (micro_cf_0922;
    # grid cells: user 2026-09-24). stage0507's small-single tiers are gone
    # with the single kind's own band (plan.md § 2).
    Group(
        "b0507",
        "piece",
        (0.5, 0.7),
        0.5,
        (
            Tier(
                "scene_piece",
                0.4,
                {"fill": [0.7, 1.0], "min_glyph": 28, "px_target": 40},
            ),
            Tier(
                "grid_string",
                0.15,
                {
                    "grids": "2x2,2x3,3x2",
                    "glyph_px": [
                        27,
                        36,
                    ],  # font px; the ink px the gate reads ≈ 0.9 of it → 24–32
                    "source": "pieces",
                    "max_glyphs": 8,
                    "bubble_frac": 0.5,
                },
            ),
            Tier("scene_short", 0.2, {"min_glyph": 28, "fill": 0.7, "max_lines": 1}),
        ),
    ),
    # piece vocabs, the small tier: stage0305 — 12–24 px text: dialogue lines,
    # one piece per small bubble, small word cells (cf_band_a1 A.2; design § 2).
    # The two scene_piece px tiers stay two tiers (plan.md "Open": the small
    # tier priced ≥ the large — grid_box report § 2).
    Group(
        "b0305",
        "piece",
        (0.3, 0.5),
        0.5,
        (
            Tier(
                "scene_sentence",
                0.4,
                {
                    "glyph_px": [14, 22],
                    "min_glyph": 12,
                    "fill": 0.9,
                    "max_lines": 2,
                    "px_target": 18,
                },
            ),
            Tier(
                "scene_piece",
                0.3,
                {
                    "glyph_px": [12, 24],
                    "min_glyph": 12,
                    "fill": 0.9,
                    "fill_min": 0.5,  # small text in small bubbles, not floating in a big one
                    "px_target": 18,
                },
            ),
            Tier(
                "grid_string",
                0.3,
                {
                    "grids": "2x2,2x3,3x2",
                    "glyph_px": [12, 24],
                    "source": "both",
                    "max_glyphs": 8,
                    "bubble_frac": 0.5,
                },
            ),
        ),
    ),
)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


# eval.json group order (the stages', minus the groups this line never draws)
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


def vocab_kinds(pools: Pools) -> dict:
    """kind → the run's distinct vocabs of that kind (``multi`` = the small
    digraphs a vocab spec like ``small`` brings; no tier draws them)."""
    return {
        "single": list(dict.fromkeys(pools.singles)),
        "piece": list(dict.fromkeys(pools.pieces)),
        "multi": list(dict.fromkeys(pools.digraphs)),
    }


def plan_groups(kinds: dict) -> list:
    """``[(group, n_items)]`` for the kinds present (the volume rule)."""
    return [
        (g, int(round(ITEMS_PER_VOCAB * len(kinds[g.kind]) * g.share)))
        for g in TABLE
        if kinds.get(g.kind)
    ]


def build(rc: RunConfig, workers: int | None = None) -> Path:
    from common.prompts import TPL_BUBBLE, TPL_EN
    from data.stage import _ink_stats

    t0 = time.time()
    out = data_dir(rc.name)
    (out / "img").mkdir(parents=True, exist_ok=True)
    workers = default_workers() if workers is None else max(1, int(workers))
    # the pools every group restarts from (piece vocabs bring the corpus lines)
    rng = random.Random(SEED)
    pools = build_pools(rc.vocab_specs(), SEED_ROWS, phrase_file, rng)
    snap = (rng.getstate(), pools.shapes.rng.getstate())
    kinds = vocab_kinds(pools)
    groups = plan_groups(kinds)
    assert groups, f"{rc.path}: no single or piece vocab — nothing to draw"
    if kinds["multi"]:
        print(
            f"build: {len(kinds['multi'])} multi vocabs ({' '.join(kinds['multi'][:10])}"
            f"{' …' if len(kinds['multi']) > 10 else ''}) — no tier draws them",
            flush=True,
        )
    print(
        f"build {rc.name} → {out}: {sum(len(v) for v in kinds.values())} vocabs "
        f"({', '.join(f'{k} {len(v)}' for k, v in kinds.items() if v)}); "
        + ", ".join(
            f"{g.name} σ {g.band[0]:.1f}–{g.band[1]:.1f} {n}" for g, n in groups
        )
        + f"; {workers} workers",
        flush=True,
    )
    recs: list = []
    report: dict = {}
    skipped: dict = {}
    for g, n in groups:
        _restart(pools, rng, snap)
        live = []
        for t in g.tiers:
            why = missing_source(t.recipe, t.params, pools)
            if why:
                skipped[f"{g.name}/{t.recipe}"] = why
                print(f"  {g.name}/{t.recipe}: skipped — {why}", flush=True)
            else:
                live.append(t)
        counts = _counts([(t.recipe, t.weight) for t in g.tiers], n)
        for t in live:
            got, rep = _build_tier(
                g, t, counts[t.recipe], pools, rng, out, len(recs), workers
            )
            recs += got
            report[f"{g.name}/{t.recipe}"] = rep
    assert recs, "no item survived the gate"
    stats = _ink_stats(recs)
    _px_gate(groups, report)
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
    vocabs = [v for k in ("single", "piece", "multi") for v in kinds[k]]
    (out / "vocabs.json").write_text(
        json.dumps(vocabs, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    shapes = Counter("x".join(map(str, r["shape"])) for r in recs)
    build = {
        "run": rc.name,
        "run_config": str(rc.path),
        "vocabs": rc.vocabs if isinstance(rc.vocabs, str) else list(rc.vocabs),
        "vocab_specs": rc.vocab_specs(),
        "n_vocabs": {k: len(v) for k, v in kinds.items()},
        "seed": SEED,
        "seed_rows": str(SEED_ROWS),
        "items_per_vocab": ITEMS_PER_VOCAB,
        "min_overlap": MIN_OVERLAP,
        "groups": {
            g.name: {
                "kind": g.kind,
                "band": list(g.band),
                "n_items": n,
                "tiers": [
                    {"recipe": t.recipe, "weight": t.weight, **t.params}
                    for t in g.tiers
                ],
            }
            for g, n in groups
        },
        "skipped": skipped,
        "workers": workers,
        "tiers": report,
        "ink_stats": {k: {"px": v[0], "ink": v[1]} for k, v in stats.items()},
        "bands": dict(Counter(json.dumps(r["band"]) for r in recs)),
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
        f"{dict(Counter(r['group'] + '/' + r['recipe'] for r in recs))}; bands {build['bands']}; "
        f"eval {len(ev)} prompts {dict(build['eval'])}; {build['minutes']} min",
        flush=True,
    )
    return out


def _restart(pools: Pools, rng, snap) -> None:
    """Put the draw state back where ``build_pools`` left it: the main rng,
    the canvas-shape rng, the scene-use counts, the decks, the balanced-line
    counts — what a stage build of record started each stage from."""
    rng.setstate(snap[0])
    pools.shapes.rng.setstate(snap[1])
    pools.used, pools.decks, pools.balanced = Counter(), {}, {}


def _counts(shares: list, n: int) -> dict:
    """Items per tier from ``[(name, weight)]`` — the weights renormalised
    over the group (int floor, the remainder to the heaviest)."""
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
        _JOB["g"],
        _JOB["t"],
        n,
        _JOB["pools"],
        random.Random(seed),
        _JOB["out"],
        first,
    )


def _build_tier(
    g: Group,
    t: Tier,
    n: int,
    pools: Pools,
    rng,
    out: Path,
    first: int,
    workers: int = 1,
):
    t0 = time.time()
    if workers <= 1 or n < 4 * workers:
        kept, rejects, px_seen, tries = _draw_loop(g, t, n, pools, rng, out, first)
    else:
        per = [n // workers + (i < n % workers) for i in range(workers)]
        jobs = [
            (i, per[i], first + sum(per[:i]), rng.randrange(2**31))
            for i in range(workers)
        ]
        _JOB.update(g=g, t=t, pools=pools, out=out)
        try:
            with mp.get_context("fork").Pool(workers) as pool:
                parts = pool.map(_worker, jobs)
        finally:
            _JOB.clear()
        kept, rejects, px_seen, tries = [], Counter(), [], 0
        for k, rj, px, tr in parts:
            kept += k
            rejects.update(rj)
            px_seen += px
            tries += tr
        pools.used.update(Counter(r["scene"] for r in kept if "scene" in r))
    name = f"{g.name}/{t.recipe}"
    if len(kept) < n:
        print(
            f"  {name}: WARNING {len(kept)}/{n} items after {tries} tries "
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
        "px_target": t.params.get("px_target"),
        "windows": dict(Counter(json.dumps(r["window"]) for r in kept)),
        "horizontal": _n_horizontal(kept),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(
        f"  {name}: {len(kept)}/{n} in {tries} tries, rejects {dict(rejects)}; "
        f"px kept {_fmt(q)} (drawn {_fmt(drawn)}); windows {rep['windows']}",
        flush=True,
    )
    _sheet(rng, kept, out / f"sheet_{g.name}_{t.recipe}.png")
    return kept, rep


def _draw_loop(g: Group, t: Tier, n: int, pools: Pools, rng, out: Path, first: int):
    """Draw ``n`` items of tier ``t`` through the group's band gate; files are
    numbered from ``first``. Returns (kept, rejects, drawn px, tries)."""
    draw = RECIPES[t.recipe]
    kept, rejects = [], Counter()
    px_seen: list = []
    tries, max_tries = 0, 4 * n + 50
    while len(kept) < n and tries < max_tries:
        tries += 1
        item = draw(pools, rng, t.params)
        if item is None:
            rejects["render"] += 1
            continue
        px = item.px()
        px_seen.append(px)
        kind = kind_of(item.vocabs, pools.n_tokens)
        w = window(kind, px, item.layout)
        if not covers(g.band, w, MIN_OVERLAP):
            rejects["no_window" if w is None else "band"] += 1
            continue
        i = first + len(kept)
        fn = out / "img" / f"{g.name}_{t.recipe}_{i:06d}.png"
        item.image.save(fn)
        rec = {
            "file": str(fn),
            "text": item.text,
            "caption": item.caption,
            "src": item.src,
            "kind": t.recipe,
            "recipe": t.recipe,
            "group": g.name,
            "layout": item.layout,
            "units": item.vocabs,  # on-disk key, frozen
            "shape": list(item.shape),
            "px": round(px, 1),
            "law_kind": kind,
            "window": list(w.band),
            "band": list(w.band),  # σ is drawn per item inside it (train.py)
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


def _px_gate(groups, report):
    """The ± 20 % launch gate (the reads' contract, data/stage.py): a tier
    that declares ``px_target`` must *draw* its median px within 20 % of it.
    Read on the drawn px, before the band gate — the gate truncates the kept
    distribution (the single tier keeps the ≥ 40 px side of the bubble-fit
    draw), and the contract is about what the recipe renders."""
    bad = []
    for g, _n in groups:
        for t in g.tiers:
            key = f"{g.name}/{t.recipe}"
            target = t.params.get("px_target")
            if key not in report or not target or not report[key]["px_drawn"]:
                continue
            med = report[key]["px_drawn"]["median"]
            report[key]["px_ok"] = abs(med - target) <= 0.2 * target
            if not report[key]["px_ok"]:
                bad.append(
                    f"{key}: drawn median px {med:.0f} vs target {target} ± 20 %"
                )
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
        tiles.append(
            (
                im,
                [
                    r["text"][:24],
                    f"{r['px']:.0f} px {r['layout']}{tag} σ {r['band'][0]}–{r['band'][1]}",
                ],
            )
        )
    contact_sheet(tiles, path, thumb=192, cols=6)
