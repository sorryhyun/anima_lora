"""builder — a run's rows → ``<run>/data/`` (``img/``, ``train.jsonl``,
``eval.json``, ``vocabs.json``, ``windows.json``, ``sentences.json`` (a
``sent`` tier drawn), ``build.json``,
``sheet_<tier>.png``), in one pass::

    for each tier t of run.table() (n = ITEMS_PER_ROW × rows × t.share × frac):
        item = t.recipe()               # re-drawn on a render miss or a px outside t.px_keep
        item.tier, item.band = t.name, t.band

Rendering forks over ``workers`` processes, each on a seed drawn from the
build's rng (deterministic per seed × workers).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import statistics as st
import time
from collections import Counter
from pathlib import Path

from . import table as T
from .config import Run
from .pools import (
    add_sentences,
    add_windows,
    build_pools,
    focus_pools,
    lang_caption,
)
from .recipes import RECIPES

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# eval.json group order (the stages'; the trainer encodes these captions too)
EVAL_ORDER = (
    "single",
    "en",
    "single_kanji",
    "single_ext",
    "single_small",
    "single_extra",
)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def build(run: Run, workers: int | None = None, frac: float = 1.0) -> Path:
    """``frac`` < 1 draws that share of every tier (a look at the sizes)."""
    from common.prompts import TPL_BUBBLE, TPL_EN
    from data.stage import _ink_stats

    t0 = time.time()
    out = run.data
    (out / "img").mkdir(parents=True, exist_ok=True)
    workers = default_workers() if workers is None else max(1, int(workers))
    rng = random.Random(T.SEED)
    pools = build_pools(list(run.rows), rng, run.lang)
    phrase = run.phrase_file()
    held = run.held_strings()
    win = add_windows(pools, run.read, out, phrase, workers, held)
    focus = focus_pools(pools, run.focus, run.focus_kanji[1]) if run.focus else None
    n_rows = len(run.focus) if run.focus else len(pools.singles)
    table = run.table()
    plan = [(t, int(round(T.ITEMS_PER_ROW * n_rows * t.share * frac))) for t in table]
    plan = [(t, n) for t, n in plan if n]  # a share-0 tier is not drawn
    if run.lang:
        plan, dropped = _lang_plan(run, plan, pools, n_rows, frac)
    sent = [t.params["lengths"] for t, _n in plan if t.recipe == "sent"]
    sents = (
        add_sentences(
            pools,
            run.read,
            (min(a for a, _ in sent), max(b for _, b in sent)),
            out,
            phrase,
            held,
            run.focus,
            run.focus_kanji[0] if run.focus else 0,
        )
        if sent
        else None
    )
    print(
        f"build {run.name} → {out}: {len(pools.singles)} rows"
        + (f" ({n_rows} focus)" if run.focus else "")
        + "; "
        + ", ".join(f"{t.name} σ {t.band[0]:g}–{t.band[1]:g} {n}" for t, n in plan)
        + f"; {workers} workers",
        flush=True,
    )
    recs: list = []
    report: dict = {}
    for t, n in plan:
        got, report[t.name] = _build_tier(t, n, pools, rng, out, len(recs), workers)
        recs += got
    assert recs, "nothing drawn"
    _ink_stats(recs, workers)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
        encoding="utf-8",
    )
    ev = [
        {
            "group": g,
            "text": s,
            "caption": (
                TPL_EN.format(s)
                if g == "en"
                else lang_caption(pools, TPL_BUBBLE.format(s), s)
            ),
        }
        for g in EVAL_ORDER
        for s in pools.inv.evals.get(g, ())
    ]
    (out / "eval.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "vocabs.json").write_text(
        json.dumps(pools.singles, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    build = {
        "run": run.name,
        "run_config": str(run.path),
        "rows": list(run.rows),
        "n_rows": len(pools.singles),
        "seed": T.SEED,
        "seed_rows": str(run.seed_rows()),
        "pack": os.environ.get("ANIMA_VOCAB_PACK", ""),
        "frac": frac,
        "glyph_route": True,
        "lines": phrase,  # the windows: train.py routes the captions per glyph
        "windows": win,
        **({"focus": focus} if focus else {}),
        **({"held": run.held} if run.held else {}),
        **({"lang": run.lang, "dropped_tiers": dropped} if run.lang else {}),
        **({"sentences": sents} if sents else {}),
        "scenes": {
            "pools": T.SCENES,
            "small": T.SMALL_POOL,
            "mono_share": T.MONO_SHARE,
            "horizontal_frac": T.HORIZONTAL_FRAC,
            "scene_cap": T.SCENE_CAP,
            "bubble_edge_min": T.BUBBLE_EDGE_MIN,
            "erase_left_max": T.ERASE_LEFT_MAX,
        },
        "table": [
            {
                "name": t.name,
                "recipe": t.recipe,
                "share": t.share,
                "band": list(t.band),
                "px_keep": list(t.px_keep),
                **t.params,
            }
            for t in table
        ],
        "workers": workers,
        "tiers": report,
        "shapes": dict(Counter("x".join(map(str, r["shape"])) for r in recs)),
        "n_train": len(recs),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "build.json").write_text(
        json.dumps(build, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"data: {len(recs)} items {dict(Counter(r['tier'] for r in recs))}; "
        f"eval {len(ev)}; {build['minutes']} min",
        flush=True,
    )
    return out


def _lang_plan(run: Run, plan: list, pools, n_rows: int, frac: float):
    """A ``lang`` run's plan: a ``bubbleN`` tier with no window for any row
    drops (no dialogue line spells KO / ZH rows), the other tiers' shares
    scaled back to the plan's Σ so the items per row stay. No ``sent`` tier:
    its lines are JA dialogue. Returns ``(plan, dropped tier names)``."""
    assert not any(t.recipe == "sent" for t, _n in plan), (
        f"{run.name}: a lang run draws no sent tier (no KO / ZH dialogue lines)"
    )
    drop = {t.name for t, _n in plan if t.recipe == "bubbleN" and not pools.windows}
    if not drop:
        return plan, []
    total = sum(t.share for t, _n in plan)
    kept = [t for t, _n in plan if t.name not in drop]
    k = total / sum(t.share for t in kept)
    print(f"lang: no window for any row — {sorted(drop)} out, shares × {k:.3f}")
    return [
        (t, int(round(T.ITEMS_PER_ROW * n_rows * t.share * k * frac))) for t in kept
    ], sorted(drop)


# fork-inherited job state (set before the pool forks; never pickled)
_JOB: dict = {}


def _worker(job):
    n, first, seed = job
    return _draw_loop(
        _JOB["t"], n, _JOB["pools"], random.Random(seed), _JOB["out"], first
    )


def _build_tier(t: T.Tier, n: int, pools, rng, out: Path, first: int, workers: int):
    t0 = time.time()
    if workers <= 1 or n < 4 * workers:
        kept, rejects, px_seen, tries = _draw_loop(t, n, pools, rng, out, first)
    else:
        per = [n // workers + (i < n % workers) for i in range(workers)]
        jobs = [
            (per[i], first + sum(per[:i]), rng.randrange(2**31)) for i in range(workers)
        ]
        _JOB.update(t=t, pools=pools, out=out)
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
        j_of = {(sc["pool"], sc["i"]): j for j, sc in enumerate(pools.scenes)}
        pools.used.update(
            Counter(j_of[r["scene_pool"], r["scene"]] for r in kept if "scene" in r)
        )
    if len(kept) < n:
        print(
            f"  {t.name}: WARNING {len(kept)}/{n} after {tries} tries "
            f"(rejects {dict(rejects)})",
            flush=True,
        )
    rep = {
        "n": len(kept),
        "planned": n,
        "tries": tries,
        "rejects": dict(rejects),
        "px_kept": _quantiles([r["px"] for r in kept]),
        "px_drawn": _quantiles(px_seen),
        "mono": sum(bool(r.get("mono")) for r in kept),
        "horizontal": sum(r.get("horizontal") is True for r in kept),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(
        f"  {t.name}: {len(kept)}/{n} in {tries} tries, rejects {dict(rejects)}; "
        f"px kept {_fmt(rep['px_kept'])} (drawn {_fmt(rep['px_drawn'])})",
        flush=True,
    )
    _sheet(rng, kept, out / f"sheet_{t.name}.png")
    return kept, rep


def _draw_loop(t: T.Tier, n: int, pools, rng, out: Path, first: int):
    draw = RECIPES[t.recipe]
    lo, hi = t.px_keep
    pools.tier_used = Counter()
    pools.scene_cap = max(1, math.ceil(T.SCENE_CAP * n))
    kept, rejects, px_seen = [], Counter(), []
    tries, max_tries = 0, 4 * n + 50
    while len(kept) < n and tries < max_tries:
        tries += 1
        item = draw(pools, rng, t.params)
        if item is None:
            rejects["render"] += 1
            continue
        px = item.px()
        px_seen.append(px)
        if (lo is not None and px < lo) or (hi is not None and px > hi):
            rejects["px"] += 1
            continue
        fn = out / "img" / f"{t.name}_{first + len(kept):06d}.png"
        item.image.save(fn)
        rec = {
            "file": str(fn),
            "text": item.text,
            "caption": item.caption,
            "src": item.src,
            "kind": t.recipe,
            "recipe": t.recipe,
            "tier": t.name,
            "layout": item.layout,
            "units": item.vocabs,  # the on-disk key the trainer's readers know
            "shape": list(item.shape),
            "px": round(px, 1),
            "band": list(t.band),
            **item.extra,
        }
        if item.layout == "scene":
            rec["box"] = item.boxes[0]
        else:
            rec["boxes"] = item.boxes
        kept.append(rec)
    return kept, rejects, px_seen, tries


def _quantiles(xs):
    if not xs:
        return None
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
        tag = " H" if r.get("horizontal") is True else ""
        tiles.append((im, [r["text"][:24], f"{r['px']:.0f} px {r['layout']}{tag}"]))
    contact_sheet(tiles, path, thumb=192, cols=6)
