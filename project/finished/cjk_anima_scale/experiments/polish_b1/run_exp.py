#!/usr/bin/env python
"""polish_b1 — ``retrain_kanji_b1``'s merged rows polished on scenes, every band (2026-09-28)

A pilot of the polish that follows ``retrain_kanji_b3`` (plan_retrain § 3),
run on b1's ``trained.pt`` — ``retrain_kana``'s rows with b1's 329 kanji on
top (the chain's merged save) — so the trained rows are all 503 singles
(174 kana + 329 kanji), warm, every other row frozen at b1's file. A side
branch: b2 still chains on the unpolished b1.

The model is ``step2_band2`` (the wake line's ``reports/step2_band2_2026_09_22.md``, ``project/finished/``):
5 k steps, μ 0.3, the band keyed on the item. Data is ``builder.TABLE``'s
singles groups with the grid dropped, plus whole dialogue lines (the
sentence tier step 2 had; ``plan_retrain`` § 4 doubling asks windows 2–6 vs
whole lines):

    band      tier             share of items
    0.7–0.9   bubble1_52       0.30
    0.5–0.7   bubbleN_34       0.20
              bubble1_32       0.05
              line_bubble_31   0.15   (26–36 px)
    0.3–0.5   bubbleN_18       0.10
              line_bubble_18   0.20   (14–22 px)

(The data of record names its items ``p0709`` / ``p0507`` / ``p0305`` +
``scene_single`` / ``scene_window`` / ``scene_single_small`` /
``scene_line``; the tiers were named by px on 2026-10-02.)

Lone 0.35 (step 2's next-arm share after the 30 k run's length collapse),
windows 0.30, lines 0.35. The singles / window tiers take TABLE's params.
``scene_line`` is a dialogue line of ``LINE_LEN`` letters, ``normalize``d
(``･･･`` → ``…``, symbol runs collapsed), every other character a trained
single (the 503, punctuation included), no glyph doubled in a row, no
trigram of a read word, routed to its glyphs' rows or dropped. Lines are
drawn evenly (user, 2026-09-28): each line about once, not per glyph —
lone and window tiers already carry per-row exposure.

Volume = what 5 k steps consume: ``STEPS_PER_ROW`` × 503 rows × batch 4
items, one epoch (user: data and caches only for that).

Legs: ``data`` (CPU) → ``OUT/run0928_polish_b1/data``; ``train`` (GPU) →
``OUT/experiments/polish_b1``. ``--dry_run`` builds the line pool and prints
the plan, renders nothing.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      .venv/bin/python project/cjk_anima_scale/experiments/polish_b1/run_exp.py \\
      --label p1 --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # windows and lines are routed
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

EXP = OUT / "experiments"
NAME = "polish_b1"
DATA = "run0928_polish_b1"
BASE = "retrain_kanji_b1"  # the rows this pass warms from and is read against
CONTEXT = OUT / BASE / "trained.pt"
N_ROWS = 503  # retrain_kana's 174 + b1's 329
STEPS_PER_ROW = 10  # 5 030 steps (user: 5 k)
BATCH = 4  # train.BATCH
N_ITEMS = STEPS_PER_ROW * N_ROWS * BATCH
INIT_ANCHOR = 0.3  # step2_band2's μ; --mu for another arm (polish_b1_mu<μ>)
LINE_LEN = (7, 16)  # letters; windows cover 2–6
LINES: dict = {}  # glyph → its lines (coverage stats only)
LINE_ORDER: list = []  # the pool, shuffled once (fork-inherited by the build workers)
LINE_DECK: dict = {}  # (tier, worker) → what is left of its share, per process
SLOT = {"w": 0, "W": 1}  # this process's build worker index, the worker count


_ORIG_WORKER: list = []


def _slot_worker(job):
    """``builder._worker`` that first tells ``scene_line`` which worker it is
    (module-level: ``Pool.map`` pickles it by name)."""
    SLOT["w"] = job[0]
    LINE_DECK.clear()
    return _ORIG_WORKER[0](job)


def patch_workers(workers: int) -> None:
    from cjk_scale import builder

    SLOT["W"] = workers
    _ORIG_WORKER[:] = [builder._worker]
    builder._worker = _slot_worker


def _tier(name: str, weight: float):
    """``builder.TABLE``'s tier ``name`` at this table's weight."""
    import dataclasses

    from cjk_scale.builder import tiers

    return dataclasses.replace(tiers(name)[0], weight=weight)


def table():
    from cjk_scale.builder import Group, Tier

    return (
        Group("single", (0.7, 0.9), 0.30, (_tier("bubble1_52", 0.30),)),
        Group(
            "single",
            (0.5, 0.7),
            0.40,
            (
                _tier("bubbleN_34", 0.20),
                _tier("bubble1_32", 0.05),
                Tier(
                    "line_bubble_31",
                    "scene_line",
                    0.15,
                    {
                        "glyph_px": [26, 36],
                        "min_glyph": 24,
                        "fill": 0.9,
                        "max_lines": 2,
                        "px_target": 31,
                    },
                ),
            ),
        ),
        Group(
            "single",
            (0.3, 0.5),
            0.30,
            (
                _tier("bubbleN_18", 0.10),
                Tier(
                    "line_bubble_18",
                    "scene_line",
                    0.20,
                    {
                        "glyph_px": [14, 22],
                        "min_glyph": 12,
                        "fill": 0.9,
                        "max_lines": 2,
                        "px_target": 18,
                    },
                ),
            ),
        ),
    )


def rc_of(name: str):
    from cjk_scale.config import RunConfig, load_run

    kana, b1 = load_run("retrain_kana"), load_run(BASE)
    return RunConfig(
        name=name,
        path=Path(__file__),
        vocabs=tuple(kana.vocab_specs() + b1.vocab_specs()),
        read=tuple(kana.read) + tuple(b1.read),
        context=BASE,
    )


def trained_singles() -> set:
    out: set = set()
    for run in ("retrain_kana", BASE):
        out |= set(json.loads((OUT / run / "data" / "vocabs.json").read_text("utf-8")))
    assert len(out) == N_ROWS and all(len(v) == 1 for v in out), len(out)
    return out


SYMBOL_RUN = "！？、。〜～ー・…"  # a run of one of these collapses to one (！！ → ！)
# no row, T5 encodes it as its pretrained "..." (routed or not); the renderer
# turns it a quarter in a column (scene.V_ROTATE)
NATIVE = {"…"}


def normalize(t: str) -> str:
    """Untrained symbols to what a page would carry (user, 2026-09-28):
    Manga109's ellipsis ``･･･`` → ``…``; ``〰`` → ``〜``; a run of one
    symbol → one. A doubled letter (いい, ここ) stays and the line drops,
    as a window does."""
    import re

    t = re.sub(r"･+", "…", t).replace("〰", "〜")
    t = re.sub(f"([{SYMBOL_RUN}])\\1+", r"\1", t)
    return t.lstrip("、。")


def routed_lines(lines: list, glyphs: set) -> list:
    """``recipes.routed_windows`` with ``NATIVE`` characters allowed: a line
    encodes to its trained glyphs' single rows and no other ext row."""
    from transformers import AutoTokenizer

    from cjk_scale.recipes import routed_windows
    from common.models import checkpoints
    from library.anima import ext_vocab
    from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder
    from library.anima.vocab_pack import resolve_pack_prefix
    from library.env import resolve_under_home

    _ok, ids = routed_windows([], glyphs)  # asserts one row per glyph, routed = not
    t5 = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/t5_old")
    )
    qw = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    _, mapping = ext_vocab.load_ext_assets(
        resolve_pack_prefix(checkpoints().vocab_pack)
    )
    enc = HybridT5Encoder.from_mapping(t5, qw, mapping, glyph_route=True)

    def ext(text: str) -> list:
        e, mask = enc.encode(f'Japanese text reads as "{text}".', 512)
        return [i - T5_TABLE_SIZE for i, m in zip(e, mask) if m and i >= T5_TABLE_SIZE]

    return [ln for ln in lines if ext(ln) == [ids[c] for c in ln if c not in NATIVE]]


def line_pool(held) -> tuple[list, dict]:
    """Whole dialogue lines over the trained singles (module docstring),
    ``normalize``d first."""
    from cjk_scale.config import phrase_file

    chars = trained_singles() | NATIVE
    grams = set()
    for h in held:
        n = min(3, len(h))
        grams |= {h[i : i + n] for i in range(len(h) - n + 1)}
    why, cands = Counter(), set()
    for ln in Path(phrase_file()).read_text(encoding="utf-8").splitlines():
        t = normalize(ln.split("\t")[0].strip())
        n = sum(unicodedata.category(c) in ("Lo", "Lm") for c in t)
        if not t or any(c not in chars for c in t):
            why["untrained char"] += 1
        elif not LINE_LEN[0] <= n <= LINE_LEN[1]:
            why["length"] += 1
        elif any(a == b for a, b in zip(t, t[1:])):
            why["doubled glyph"] += 1
        elif any(g in t for g in grams):
            why["held trigram"] += 1
        else:
            cands.add(t)
    ok = routed_lines(sorted(cands), set("".join(cands)) - NATIVE)
    why["encoding"] = len(cands) - len(ok)
    return ok, dict(why)


def scene_line(pools, rng: random.Random, p: dict):
    """The next line of this (tier, worker)'s share of the pool: the pool is
    shuffled once and dealt into disjoint shares, so a line is drawn once
    until its share runs out (then the share is dealt again)."""
    from cjk_scale.recipes import _draw_scene, _target

    tier = 0 if p["px_target"] > 24 else 1
    key = (tier, SLOT["w"])
    if not LINE_DECK.get(key):
        n = 2 * SLOT["W"]
        LINE_DECK[key] = LINE_ORDER[tier * SLOT["W"] + SLOT["w"] :: n]
        rng.shuffle(LINE_DECK[key])
    text = LINE_DECK[key].pop()
    fill = float(p.get("fill", 0.9))
    item = _draw_scene(
        pools,
        rng,
        text,
        min_glyph=int(p.get("min_glyph", 12)),
        fill=fill,
        max_lines=int(p.get("max_lines", 2)),
        fewest_lines=True,
        target_px=_target(rng, p),
        fill_max=fill,
        fill_min=float(p.get("fill_min", 0)),
    )
    if item is None:
        return None
    assert item.caption.count(f'"{text}"') == 1, item.caption
    return item


def set_lines(lines: list) -> dict:
    LINES.clear()
    LINE_ORDER[:] = lines
    random.Random(0).shuffle(LINE_ORDER)
    for ln in lines:
        for c in set(ln):
            if unicodedata.category(c) in ("Lo", "Lm"):
                LINES.setdefault(c, []).append(ln)
    per = sorted(len(v) for v in LINES.values())
    return {
        "n": len(lines),
        "by_length": dict(
            sorted(
                Counter(
                    sum(unicodedata.category(c) in ("Lo", "Lm") for c in ln)
                    for ln in lines
                ).items()
            )
        ),
        "glyphs": len(LINES),
        "per_glyph_min": per[0],
        "per_glyph_median": per[len(per) // 2],
        "glyphs_without": "".join(
            sorted(
                c
                for c in trained_singles() - set(LINES)
                if unicodedata.category(c) in ("Lo", "Lm")
            )
        ),
    }


def plan_rows(groups) -> list:
    return [
        {
            "tier": t.name,
            "band": list(g.band),
            "recipe": t.recipe,
            "items": round(N_ITEMS * t.weight),
            "px": t.params.get("glyph_px") or t.params.get("px_target"),
        }
        for g in groups
        for t in g.tiers
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--legs", nargs="+", default=["data"], choices=["data", "train"])
    p.add_argument("--workers", type=int)
    p.add_argument("--mu", type=float, default=INIT_ANCHOR, help="the anchor μ")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    name = NAME if args.mu == INIT_ANCHOR else f"{NAME}_mu{args.mu:g}"

    from cjk_scale import builder, recipes
    from cjk_scale import train as T

    rc = rc_of(DATA)
    groups = table()
    shares = sum(t.weight for g in groups for t in g.tiers)
    assert abs(shares - 1.0) < 1e-9 and all(
        abs(g.share - sum(t.weight for t in g.tiers)) < 1e-9 for g in groups
    ), "tier weights are shares of all items"
    plan = plan_rows(groups)
    lines, why = line_pool(rc.read)
    stats = set_lines(lines)
    print(
        f"{NAME}: {N_ROWS} rows warm from {CONTEXT}, {STEPS_PER_ROW} steps/row → "
        f"{STEPS_PER_ROW * N_ROWS} steps × batch {BATCH} = {N_ITEMS} items, μ {args.mu} → {EXP / name}",
        flush=True,
    )
    for r in plan:
        print(
            f"  {r['tier']:<15} σ {r['band'][0]}–{r['band'][1]}  {r['recipe']:<11} "
            f"{r['items']:>6}  px {r['px']}",
            flush=True,
        )
    print(f"lines: {json.dumps(stats, ensure_ascii=False)}; dropped {why}", flush=True)
    metrics: dict = {
        "base": BASE,
        "n_rows": N_ROWS,
        "steps_per_row": STEPS_PER_ROW,
        "n_items": N_ITEMS,
        "init_anchor": args.mu,
        "arm": name,
        "plan": plan,
        "lines": stats,
        "lines_dropped": why,
        "held": list(rc.read),
    }
    if args.dry_run:
        rng = random.Random(0)
        for ln in rng.sample(lines, 12):
            print(f"  {ln}", flush=True)
        return
    run_dir = make_run_dir(
        "polish_b1",
        label=args.label,
        root=LINE / "experiments" / "polish_b1" / "results",
    )
    data = OUT / DATA / "data"
    if "data" in args.legs:
        recipes.RECIPES["scene_line"] = scene_line
        # group → items: its tiers' shares of N_ITEMS (the table's volume, not ITEMS_PER_VOCAB)
        builder.plan_groups = lambda kinds, table, budget=1.0: [
            (g, round(N_ITEMS * g.share)) for g in table
        ]
        workers = builder.default_workers() if args.workers is None else args.workers
        patch_workers(workers)
        builder.build(rc, workers=workers, table=groups)
        recs = [
            json.loads(ln)
            for ln in (data / "train.jsonl").read_text("utf-8").splitlines()
            if ln
        ]
        metrics["data"] = {
            "n": len(recs),
            "by_tier": dict(Counter(r["tier"] for r in recs)),
            "bands": dict(Counter(json.dumps(r["band"]) for r in recs)),
            "lines_distinct": len(
                {r["text"] for r in recs if r["recipe"] == "scene_line"}
            ),
            "lines_items": sum(r["recipe"] == "scene_line" for r in recs),
        }
        print(f"data: {json.dumps(metrics['data'], ensure_ascii=False)}", flush=True)
    if "train" in args.legs:
        assert CONTEXT.exists(), CONTEXT
        T.INIT_ANCHOR = args.mu
        T.train(
            rc_of(name),
            data=data,
            out=EXP / name,
            cold=False,
            steps_per_row=STEPS_PER_ROW,
            context=CONTEXT,
        )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / DATA), str(EXP / name)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
