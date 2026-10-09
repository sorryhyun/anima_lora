"""builder — a run's vocabs → ``<run>/data/`` (``img/``, ``train.jsonl``,
``eval.json``, ``vocabs.json``, ``build.json``, ``sheet_<tier>.png``).

The recipe table is by kind (plan.md § 2): which tiers run is decided by
which kinds the run's vocabs hold — no shares to set, no stage. A **tier** is
an item pool: a recipe at one glyph size, named ``<form>_<px>`` — the form
(``lone`` a glyph alone on its canvas, ``grid`` one glyph per cell,
``bubble1`` one glyph in a scene bubble, ``bubbleN`` a 2–6 glyph window in
one; ``piece_*`` / ``line_*`` for the piece kind) and the median ink px its
items were built at (``TIER_PX``). The name is the file prefix, the record's
``tier`` and the ``build.json`` key; until 2026-10-02 an item was named by
its band group and recipe (``b0507/scene_window``) — ``tier_of`` reads those
records. A **group** is the tiers of one kind drawn at one band from one rng
restart; it has no name. Every item is stamped with its band —
``windows.window(kind, px, layout)`` at build time — and the trainer draws
its σ inside it::

    for each group g of the vocabs' kinds (n = ITEMS_PER_VOCAB × vocabs of
    the kind × g.share, split over g's tiers by weight):
        item = tier.draw()                          # vocab(s), px, layout
        w = window(kind_of(vocabs), px, layout)
        keep iff g.band ⊆ w (or |g.band ∩ w| / |g.band| ≥ MIN_OVERLAP); re-draw otherwise
        (a tier with ``gate = "group"`` skips the gate: w = g.band)
        item.tier = tier.name                       # a 1×1 from a mixed deck: tier.flat
        item.band = w.band                          # the band the trainer draws σ in

The tiers and their weights are the old stage files' mixes (``_archive/configs/
stage0709|0507|0305.toml``) with the stage gone, so a group draws the
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

from .config import SEED, RunConfig, dataset_ja_lines, phrase_file
from .paths import data_dir
from .recipes import RECIPES, Pools, build_pools, missing_source
from .windows import Window, covers, kind_of, window

# forked workers inherit an HF tokenizer; its thread pool must not be live
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# items per vocab: run0925_300f's two stage dirs of 10 000 over its 300 piece
# vocabs (≈ 67; plan.md § 2 "~70"), so its re-run (plan.md § 6-3) draws the same volume
ITEMS_PER_VOCAB = 20_000 / 300
# the gate's overlap threshold (every stage file of record: gate = contain, min_overlap 0.8)
MIN_OVERLAP = 0.8


@dataclass(frozen=True)
class Tier:
    name: str  # the item pool, <form>_<px>: file prefix, the record's `tier`, build.json key
    recipe: str
    weight: float  # within the group, as the stage file's share (renormalised over the group)
    params: dict = field(default_factory=dict)
    # the name the tier's 1×1 items take when its deck deals 1×1 beside the
    # grids (one draw, two pools: splitting the draw would change the stream)
    flat: str = ""

    def name_of(self, layout: str) -> str:
        return self.flat if self.flat and layout == "flat" else self.name

    @property
    def names(self) -> tuple:
        return (self.name, self.flat) if self.flat else (self.name,)


@dataclass(frozen=True)
class Group:
    """The tiers of one kind drawn at one band, from one rng restart."""

    kind: str  # the vocab kind that brings the group in
    band: tuple
    share: float  # of the kind's items
    tiers: tuple

    @property
    def label(self) -> str:
        return " + ".join(n for t in self.tiers for n in t.names)


TABLE = (
    # single vocabs: stage0709 — glyph identity at ≥ 48 px, the bubble fit and
    # the 1×1–3×3 grids (band_b1 B.1, step1_0921; band law § 3). The lone
    # tier at share 0.5 beside the two in-word groups below: P1b's 1 : 2
    # (retrain_experiments § 3; C1: lone alone composes nothing)
    Group(
        "single",
        (0.7, 0.9),
        0.5,
        (
            Tier(
                "bubble1_52",
                "bubble1",
                0.5,
                {"fill": 0.7, "min_glyph": 28, "px_target": 50},
            ),
            Tier(
                "grid_82",
                "grid",
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
                flat="lone_190",  # a fifth of the deck's deals
            ),
        ),
    ),
    # single vocabs in words (retrain_experiments § 3, P1b / C2 / C3): windows of
    # dialogue lines at the piece tiers' px and bands (Stage B's
    # ``scene_spelled`` took the piece ``scene_piece`` params), routed
    # captions; the 0.5–0.7 group carries the count tier at 0.3 (Stage B)
    Group(
        "single",
        (0.5, 0.7),
        0.5,
        (
            Tier(
                "bubbleN_34",
                "bubbleN",
                0.7,
                {"fill": [0.7, 1.0], "min_glyph": 28, "px_target": 40},
            ),
            Tier(
                "bubble1_32",
                "bubble1",
                0.3,
                {"glyph_px": [28, 40], "fill": [0.2, 0.4], "min_glyph": 12},
            ),
        ),
    ),
    Group(
        "single",
        (0.3, 0.5),
        0.5,
        (
            Tier(
                "bubbleN_18",
                "bubbleN",
                1.0,
                {
                    "glyph_px": [12, 24],
                    "min_glyph": 12,
                    "fill": 0.9,
                    "fill_min": 0.5,
                    "px_target": 18,
                },
            ),
        ),
    ),
    # piece vocabs, the large tier: stage0507 — one piece per bubble ≈ 35–48 px,
    # pieces in 24–32 px word cells, 2–5-piece lines ≈ 32 px (micro_cf_0922;
    # grid cells: user 2026-09-24). stage0507's small-single tiers are gone
    # with the single kind's own band (plan.md § 2).
    Group(
        "piece",
        (0.5, 0.7),
        0.5,
        (
            Tier(
                "piece_bubble_38",
                "scene_piece",
                0.4,
                {"fill": [0.7, 1.0], "min_glyph": 28, "px_target": 40},
            ),
            Tier(
                "piece_grid_29",
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
            Tier(
                "line_bubble_32",
                "scene_short",
                0.2,
                {"min_glyph": 28, "fill": 0.7, "max_lines": 1},
            ),
        ),
    ),
    # piece vocabs, the small tier: stage0305 — 12–24 px text: dialogue lines,
    # one piece per small bubble, small word cells (cf_band_a1 A.2; design § 2).
    # The two scene_piece px tiers stay two tiers (plan.md "Open": the small
    # tier priced ≥ the large — grid_box report § 2).
    Group(
        "piece",
        (0.3, 0.5),
        0.5,
        (
            Tier(
                "line_bubble_19",
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
                "piece_bubble_19",
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
                "piece_grid_17",
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

# The number in a tier's name: the ink px (√(box area / glyphs)) its items were
# built at, p10 / median / p90, on the kana run — a fixed label, not
# recomputed per build (a kanji run draws the large tiers larger:
# retrain_kanji_b4 grid_82 → 100, lone_190 → 235; ``build.json`` ``tiers``
# has every build's own ``px_kept``). Single kind: ``retrain_kana/data``,
# and ``run1002_grid_lone/data`` for the small grid / lone tiers the
# experiments draw (``experiments/grid_small``, ``grid_lone``), and
# ``experiments/grid_44``'s build for its 44 px pair; piece kind:
# ``run0925_300f``'s stage builds (median, p10–p90 from its build log).
TIER_PX = {
    "lone_190": (119, 191, 268),
    "grid_82": (55, 82, 118),
    "bubble1_52": (42, 52, 75),
    "grid_44": (41, 44, 49),
    "lone_44": (34, 44, 52),
    "bubbleN_34": (29, 34, 45),
    "bubble1_32": (27, 32, 38),
    "grid_29": (25, 29, 33),
    "lone_28": (22, 28, 35),
    "bubbleN_18": (14, 18, 22),
    "grid_16": (13, 16, 20),
    "lone_16": (12, 16, 21),
    "piece_bubble_38": (30, 38, 53),
    "line_bubble_32": (29, 32, 39),
    "piece_grid_29": (26, 29, 32),
    "piece_bubble_19": (15, 19, 23),
    "line_bubble_19": (16, 19, 23),
    "piece_grid_17": (13, 17, 21),
}

# (group, recipe) of a record built before 2026-10-02 → its tier
LEGACY_TIERS = {
    ("b0709", "scene_single"): "bubble1_52",
    ("b0709", "grid_single"): "grid_82",  # its 1×1 (layout flat): lone_190
    ("b0507", "scene_window"): "bubbleN_34",
    ("b0507", "scene_single_small"): "bubble1_32",
    ("b0305", "scene_window"): "bubbleN_18",
    ("g0507", "grid_single"): "grid_29",
    ("g0305", "grid_single"): "grid_16",
    ("l0507", "grid_single"): "lone_28",
    ("l0305", "grid_single"): "lone_16",
    ("b0507", "scene_piece"): "piece_bubble_38",
    ("b0507", "grid_string"): "piece_grid_29",
    ("b0507", "scene_short"): "line_bubble_32",
    ("b0305", "scene_sentence"): "line_bubble_19",
    ("b0305", "scene_piece"): "piece_bubble_19",
    ("b0305", "grid_string"): "piece_grid_17",
}


def tier_of(rec: dict) -> str:
    """A ``train.jsonl`` record's tier: its ``tier``, or — a record built
    before 2026-10-02 — the tier its ``group`` / ``recipe`` became
    (``<group>_<recipe>``, its file prefix, when an experiment's own table
    drew it)."""
    if "tier" in rec:
        return rec["tier"]
    lo, hi = rec["band"]
    # a pre-collapse stage record has no group: its stage was its band
    group = rec.get("group") or f"b{round(lo * 10):02d}{round(hi * 10):02d}"
    key = (group, rec.get("recipe") or rec.get("kind"))
    if key == ("b0709", "grid_single") and rec.get("layout") == "flat":
        return "lone_190"
    return LEGACY_TIERS.get(key) or f"{key[0]}_{key[1]}"


def tiers(*names: str, table: tuple | None = None) -> tuple:
    """``table``'s (``TABLE``'s) tiers of these names, in the order asked."""
    by = {t.name: t for g in (TABLE if table is None else table) for t in g.tiers}
    return tuple(by[n] for n in names)


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


def plan_groups(kinds: dict, table: tuple = TABLE, budget: float | dict = 1.0) -> list:
    """``[(group, n_items)]`` for the kinds present (the volume rule);
    ``budget`` = the run's factor, or vocab → factor (``budget.run_budget``;
    a vocab it lacks counts 1): a kind's items are ITEMS_PER_VOCAB × Σ its
    vocabs' factors × the group's share, so items keep pace with the steps."""

    def mass(vs) -> float:
        if isinstance(budget, dict):
            return sum(budget.get(v, 1.0) for v in vs)
        return budget * len(vs)

    return [
        (g, int(round(ITEMS_PER_VOCAB * mass(kinds[g.kind]) * g.share)))
        for g in table
        if kinds.get(g.kind)
    ]


def _weigh(pool: list, w: dict) -> list:
    """``pool`` with each vocab repeated by its draw weight (in place order;
    all weights 1 → the pool itself)."""
    if all(x == 1 for x in w.values()):
        return pool
    return [v for v in pool for _ in range(w.get(v, 1))]


def context_singles(rc: RunConfig) -> set:
    """The singles trained along ``rc``'s context chain (each context run's
    ``vocabs.json``, else its vocabs file): cold-retrained rows a window may
    carry beside the run's own. A glyph whose row is still the seed's stays
    out of the windows (plan_retrain: that row is what the retrain replaces)."""
    from data.vocabs import _list_file

    from .merge import idx_source

    out: set = set()
    for run in rc.context_chain():
        f = idx_source(run)
        vs = (
            json.loads(f.read_text(encoding="utf-8"))
            if f.suffix == ".json"
            else _list_file(str(f))
        )
        out |= {v for v in vs if len(v) == 1}
    return out


def build(
    rc: RunConfig,
    workers: int | None = None,
    table: tuple = TABLE,
    prepare=None,
) -> Path:
    """Build the run's data dir. ``table`` is ``TABLE`` for every run;
    ``experiments/`` pass another one to validate a mix before it becomes
    the rule — ``scale.py`` never does. ``prepare(pools, rc)`` (experiments
    only) edits the pools once the windows are in — a window rule, a pool of
    its own — and returns what ``build.json`` records under ``prepare``."""
    from common.prompts import TPL_BUBBLE, TPL_EN
    from data.inventory import qwen_pieces
    from data.stage import _ink_stats

    from .budget import draw_weights, run_budget, seed_ids

    t0 = time.time()
    out = data_dir(rc.name)
    (out / "img").mkdir(parents=True, exist_ok=True)
    workers = default_workers() if workers is None else max(1, int(workers))
    # the pools every group restarts from (piece vocabs bring the corpus lines)
    rng = random.Random(SEED)
    ctx = rc.context_rows()
    pools = build_pools(rc.vocab_specs(), ctx, lambda: phrase_file(rc), rng)
    snap = (rng.getstate(), pools.shapes.rng.getstate())
    kinds = vocab_kinds(pools)
    budget = run_budget(
        [v for k in ("single", "piece", "multi") for v in kinds[k]],
        qwen_pieces(char_rows=True),
        seed_ids(ctx),
    )
    weights = draw_weights(budget)
    pools.singles = _weigh(pools.singles, weights)
    groups = plan_groups(kinds, table, budget)
    assert groups, f"{rc.path}: no single or piece vocab — nothing to draw"
    names = [n for g, _n in groups for t in g.tiers for n in t.names]
    assert len(set(names)) == len(names), f"{rc.path}: two tiers of one name ({names})"
    route = any(t.recipe == "bubbleN" for g, _n in groups for t in g.tiers)
    win = _windows(rc, pools, out, weights) if route else {}
    prepared = prepare(pools, rc) if prepare else None
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
            f"{g.label} σ {g.band[0]:.1f}–{g.band[1]:.1f} {n}" for g, n in groups
        )
        + (
            f"; budget {_budget_summary(budget)}"
            if set(budget.values()) - {1.0}
            else ""
        )
        + (f"; context {rc.context}" if rc.context else "")
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
                skipped[t.name] = why
                print(f"  {t.name}: skipped — {why}", flush=True)
            else:
                live.append(t)
        counts = _counts([(t.name, t.weight) for t in g.tiers], n)
        for t in live:
            got, rep = _build_tier(
                g, t, counts[t.name], pools, rng, out, len(recs), workers
            )
            recs += got
            report[t.name] = rep
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
        "seed_rows": str(ctx),
        "context": rc.context,
        "items_per_vocab": ITEMS_PER_VOCAB,
        "budget_factor": _budget_summary(budget),
        "draw_weights": _budget_summary(weights),
        "glyph_route": route,  # train.py routes the run's captions per glyph
        "windows": win,
        **({"prepare": prepared} if prepare else {}),
        "min_overlap": MIN_OVERLAP,
        "groups": [
            {
                "kind": g.kind,
                "band": list(g.band),
                "n_items": n,
                "tiers": [
                    {
                        "name": t.name,
                        **({"flat": t.flat} if t.flat else {}),
                        "recipe": t.recipe,
                        "weight": t.weight,
                        **t.params,
                    }
                    for t in g.tiers
                ],
            }
            for g, n in groups
        ],
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
        f"{dict(Counter(r['tier'] for r in recs))}; bands {build['bands']}; "
        f"eval {len(ev)} prompts {dict(build['eval'])}; {build['minutes']} min",
        flush=True,
    )
    return out


def _windows(rc: RunConfig, pools: Pools, out: Path, weights: dict) -> dict:
    """The windowed word pool on ``pools.windows`` (glyph → windows): the
    run's letter singles, with the context chain's trained ones beside them
    (``context_singles``), over the dialogue lines, the read strings held out
    by trigram, every window routed to its glyphs' single rows or dropped.
    The lines are the dialogue pool plus the training set's own JA text
    (``dataset_ja_lines``, plan_retrain § 2b).
    Keys are the run's glyphs only (a draw picks one of them, weighted by
    ``weights`` on ``pools.window_keys``, then one of its windows); a
    context glyph rides frozen at its context row. Writes ``windows.json``;
    returns the stats for ``build.json``."""
    from .recipes import WINDOW_LEN, routed_windows, window_glyphs, window_pool

    glyphs = window_glyphs(pools.singles)
    ctx = window_glyphs(context_singles(rc)) - glyphs
    lines = [
        ln.split("\t")[0]
        for ln in Path(phrase_file(rc)).read_text(encoding="utf-8").splitlines()
    ]
    ds = dataset_ja_lines()
    held = rc.held_strings()
    ws = window_pool(glyphs | ctx, lines + ds, held, WINDOW_LEN)
    ws = [w for w in ws if any(c in glyphs for c in w)]
    ok, _ids = routed_windows(ws, glyphs | ctx)
    pools.windows = {g: [w for w in ok if g in w] for g in sorted(glyphs)}
    pools.windows = {g: v for g, v in pools.windows.items() if v}
    pools.window_keys = _weigh(list(pools.windows), weights)
    n = sorted(len(v) for v in pools.windows.values())
    stats = {
        "length": list(WINDOW_LEN),
        "held": list(rc.read),
        "held_file": (rc.held, len(held) - len(rc.read)) if rc.held else None,
        "context_glyphs": len(ctx),
        "lines": {"dialogue": len(lines), "dataset": len(ds)},
        "phrases": phrase_file(rc),
        "n": len(ok),
        "dropped_by_encoding": len(ws) - len(ok),
        "glyphs": len(pools.windows),
        "glyphs_without": sorted(glyphs - set(pools.windows)),
        "per_glyph_min": n[0] if n else 0,
        "per_glyph_median": n[len(n) // 2] if n else 0,
        "by_length": dict(sorted(Counter(map(len, ok)).items())),
    }
    (out / "windows.json").write_text(
        json.dumps(ok, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(
        f"windows: {len(ok)} ({stats['dropped_by_encoding']} dropped by the "
        f"encoding check), {len(pools.windows)} glyphs, per glyph min {stats['per_glyph_min']} "
        f"median {stats['per_glyph_median']}; none for "
        f"{''.join(stats['glyphs_without']) or '-'} (lone only)",
        flush=True,
    )
    return stats


def _budget_summary(f: dict) -> dict:
    """factor → how many vocabs take it (``build.json``)."""
    return {f"{x:g}": n for x, n in sorted(Counter(f.values()).items())}


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
    name = t.name
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
        **({"items": dict(Counter(r["tier"] for r in kept))} if t.flat else {}),
        "horizontal": _n_horizontal(kept),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(
        f"  {name}: {len(kept)}/{n} in {tries} tries, rejects {dict(rejects)}; "
        f"px kept {_fmt(q)} (drawn {_fmt(drawn)}); windows {rep['windows']}",
        flush=True,
    )
    _sheet(rng, kept, out / f"sheet_{t.name}.png")
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
        kind = kind_of(item.law_vocabs(), pools.n_tokens)
        w = window(kind, px, item.layout)
        if t.params.get("gate") == "group":
            # a lone glyph's px is its own ink box, so a thin or small one
            # (ー 0.38 × its font px, っ 0.58; the median glyph 0.83) reads a
            # band below its font size and the gate drops the row from the
            # tier (grid_lone, 10-02). Such a tier is drawn at a grid tier's
            # font px and takes that tier's band — the group's — as it is
            w = Window(*g.band, "the group's band (tier gate = group)")
        elif not covers(g.band, w, MIN_OVERLAP):
            rejects["no_window" if w is None else "band"] += 1
            continue
        i = first + len(kept)
        name = t.name_of(item.layout)
        fn = out / "img" / f"{name}_{i:06d}.png"
        item.image.save(fn)
        rec = {
            "file": str(fn),
            "text": item.text,
            "caption": item.caption,
            "src": item.src,
            "kind": t.recipe,
            "recipe": t.recipe,
            "tier": name,
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
            key = t.name
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
