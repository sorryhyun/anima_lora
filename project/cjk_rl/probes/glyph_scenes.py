#!/usr/bin/env python
"""glyph_scenes — the scene pool of the glyph stage, picked by render (after
``reports/glyph_g0_2026_11_25.md``; the user's call, 11-25: scenes of artists whose
captions hold little text, or captions that never had an OCR clause).

A candidate is a training caption with no text clause and no lettering tag (so nothing
is taken out of it; ``glyph_g0.scene_tags`` leaves it as it is). ``low`` candidates
come from the artist folders whose captions carry a text clause at most ``LOW_CLAUSE``
of the time and a lettering tag at most ``LOW_TEXTISH``; ``other`` candidates from the
remaining folders. Frames, stack, decode and reads as ``glyph_g0.py``, Euler, seeds
0 – 3, 768².

    screen  ``N_LOW`` + ``N_OTHER`` candidates: the no-clause render in both frames
            (what the scene letters by itself), and ``stop`` (rows off) and う (rows
            on) in both frames. A scene is ``clean`` when none of its 8 no-clause
            renders has a read letter. ``--round 2`` and up screen ``N_MORE`` fresh
            candidates of the ``low`` folders each (scene ids go on from the round
            before).
    select  (CPU) the selected prompts from every round's reads on disk →
            ``assets/glyph_scenes.json``. The rule (the user's call, 11-25): the plain
            no-clause renders letter nothing, the bubble ones on at most
            ``BUBBLE_MAX`` of 4 seeds, and the two probe strings read exactly on
            ``LAND_MIN`` of their 8 bubble renders or more.
    cross   the ``N_CROSS`` clean scenes with the most exact reads of the two probe
            strings in the bubble frame (one scene an artist): 24 hiragana, the 32
            frequent kanji and 8 EN words on every one of them, bubble frame. Per
            string: ``solo`` out of 16, ``extra``; the terms' spread inside a
            (string, scene) group of 4 seeds.

Renders: ``output/cjk_rl/glyph_scenes/m15v11_px/``. Result →
``results/<ts>-glyph_scenes-<stage>/`` (``scenes.json``: every candidate with its
numbers). ``assets/glyph_scenes.json`` is gitignored (``project/**/assets/*.json``).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/glyph_scenes.py screen"
    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/glyph_scenes.py screen --round 2"
    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/glyph_scenes.py cross"
    .venv/bin/python project/cjk_rl/probes/glyph_scenes.py select
    .venv/bin/python project/cjk_rl/probes/glyph_scenes.py screen --tabulate     # CPU
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import glyph_g0 as G
import turbo_read as T

OUT = T.REPO / "output" / "cjk_rl" / "glyph_scenes"
ASSET = T.LINE / "assets" / "glyph_scenes.json"  # the selected prompts
SEED = 1126
SEEDS = (0, 1, 2, 3)
N_LOW, N_OTHER, N_CROSS = 44, 20, 4
N_MORE = 64  # candidates of a later round, low folders only
BUBBLE_MAX, LAND_MIN = 1, 2  # the selection rule
LOW_CLAUSE, LOW_TEXTISH = 0.05, 0.12
PER_ARTIST = 4  # candidates a folder
PROBES = (("en", "stop"), ("hira", "う"))
EN_CROSS = G.EN_WORDS[:8]


# ---------------------------------------------------------------- candidates (CPU)


def candidates(rounds: int = 1) -> list:
    """``[{i, round, stem, artist, group, clause_rate, textish_rate, plain, bubble}]``
    of rounds 1 … ``rounds``: the ruler's images and the ``$CHARACTER_REF_EXCLUDE``
    tags held out. A folder's text-free captions are shuffled once; round r takes its
    next ``PER_ARTIST`` (round 1) or as many as it needs, so no caption comes twice."""
    from anime_tools.captions.position_clauses import (
        TEXT_PREFIXES,
        compose_caption,
        parse_caption,
    )

    from eval.ruler import RULER
    from eval.ruler.build import CAPTIONS, DROP_TAGS
    from i2i_tasks.tools.character_ref_pairs import exclude_tags
    from library.env import load_dotenv

    load_dotenv()
    held = {
        m["stem"]
        for m in json.loads((RULER / "ruler.json").read_text("utf-8"))["items"]
    }
    excluded = exclude_tags()

    def lettering(t: str) -> bool:
        return t in DROP_TAGS or t in G.BUBBLES or bool(G.TEXTISH.search(t))

    rate = defaultdict(lambda: [0, 0, 0])  # folder → captions, with a clause, textish
    free = defaultdict(list)  # folder → (stem, parsed) of the text-free captions
    for f in sorted(CAPTIONS.rglob("*.txt")):
        if f.name.endswith((".variants.txt", ".history.txt")):
            continue
        p = parse_caption(f.read_text(encoding="utf-8"))
        artist = f.parent.relative_to(CAPTIONS).parts[0]
        clause = any(cl.prefix in TEXT_PREFIXES for cl in p.clauses)
        tags = [*p.flat_tags, *(t for cl in p.clauses for t in cl.tags)]
        textish = clause or any(lettering(t) for t in tags)
        r = rate[artist]
        r[0] += 1
        r[1] += clause
        r[2] += textish
        if not textish and f.stem not in held and not excluded.intersection(tags):
            free[artist].append((f.stem, p))
    rng = random.Random(SEED)
    out, turns = [], []
    rest: dict = {}  # folder → its shuffled captions past round 1's
    for group, n in (("low", N_LOW), ("other", N_OTHER)):
        artists = sorted(
            a
            for a, (k, c, t) in rate.items()
            if ((c / k <= LOW_CLAUSE and t / k <= LOW_TEXTISH) == (group == "low"))
            and free[a]
        )
        rng.shuffle(artists)
        picks = []
        for a in artists:
            xs = list(free[a])
            rng.shuffle(xs)
            picks.append([(a, x) for x in xs[:PER_ARTIST]])
            if group == "low":
                rest[a] = [(a, x) for x in xs[PER_ARTIST:]]
        # one a folder in turn, so the candidates spread over the folders
        turn = [p[k] for k in range(PER_ARTIST) for p in picks if k < len(p)]
        if group == "low":  # round 1's turn that did not fit goes first in round 2
            for a, x in reversed(turn[n:]):
                rest[a].insert(0, (a, x))
        turns += [(1, group, x) for x in turn[:n]]
    # later rounds: the low folders' remaining captions, one a folder in turn
    depth = max((len(v) for v in rest.values()), default=0)
    more = [rest[a][k] for k in range(depth) for a in sorted(rest) if k < len(rest[a])]
    for r in range(2, rounds + 1):
        turns += [(r, "low", x) for x in more[(r - 2) * N_MORE : (r - 1) * N_MORE]]
    for rnd, group, (a, (stem, p)) in turns:
        k, c, t = rate[a]
        out.append(
            {
                "i": len(out),
                "round": rnd,
                "stem": stem,
                "artist": a,
                "group": group,
                "clause_rate": round(c / k, 3),
                "textish_rate": round(t / k, 3),
                "plain": compose_caption(list(p.flat_tags), p.clauses),
                "bubble": compose_caption([*p.flat_tags, "speech bubble"], p.clauses),
            }
        )
    return out


def job(part, arm, c, frame, seed, key=None, cls=None) -> dict:
    stem = f"sc{c['i']:02d}_{frame}_s{seed}"
    return {
        "part": part,
        "arm": arm,
        "stem": stem if key is None else f"{key}_{stem}",
        "caption": c[frame] if key is None else G.framed(c[frame], key),
        "seed": seed,
        "shape": G.SHAPE,
        "frame": frame,
        "scene": c["i"],
        "key": key,
        "cls": cls,
    }


def screen_jobs(cs: list) -> list:
    out = []
    for c in cs:
        for frame in G.FRAMES:
            for seed in SEEDS:
                out.append(job("anchor", G.ROWS, c, frame, seed))
                for cls, key in PROBES:
                    arm = T.EN if cls == "en" else G.ROWS
                    out.append(job("grid", arm, c, frame, seed, key, cls))
    return out


def cross_jobs(cs: list, chosen: list) -> list:
    st = G.strings()
    keys = [("hira", k) for k in st["hira"]] + [("kanji", k) for k in st["kanji"]]
    keys += [("en", k) for k in EN_CROSS]
    return [
        job("cross", T.EN if cls == "en" else G.ROWS, cs[i], "bubble", seed, key, cls)
        for cls, key in keys
        for i in chosen
        for seed in SEEDS
    ]


# ---------------------------------------------------------------- tables


def letters(reads: list) -> int:
    return sum(ch.isalnum() for b in G.boxes(reads) for ch in b)


def screen_table(cs: list, js: list, reads: dict) -> tuple[list, dict]:
    rd = lambda j: reads[(j["part"], j["arm"])][j["stem"]]  # noqa: E731
    rows = {c["i"]: {k: c[k] for k in c if k not in G.FRAMES} for c in cs}
    for j in js:
        r = rows[j["scene"]]
        if j["part"] == "anchor":
            n = letters(rd(j))
            r[f"anchor_{j['frame']}"] = r.get(f"anchor_{j['frame']}", 0) + n
            r[f"lettered_{j['frame']}"] = r.get(f"lettered_{j['frame']}", 0) + (n > 0)
        else:
            t = G.page_terms(j["key"], rd(j))
            k = f"{j['key']}_{j['frame']}"
            r[f"solo_{k}"] = r.get(f"solo_{k}", 0) + t["solo"]
            r[f"extra_{k}"] = r.get(f"extra_{k}", 0) + t["extra"] / len(SEEDS)
    for r in rows.values():
        r["clean"] = not (r["lettered_plain"] or r["lettered_bubble"])
        r["clean_plain"] = not r["lettered_plain"]
        r["clean_bubble"] = not r["lettered_bubble"]
        r["score"] = r["solo_stop_bubble"] + r["solo_う_bubble"]
        r["selected"] = (
            r["clean_plain"]
            and r["lettered_bubble"] <= BUBBLE_MAX
            and r["score"] >= LAND_MIN
        )
    rows = list(rows.values())
    summary = {}
    for name, xs in (
        ("low", [r for r in rows if r["group"] == "low"]),
        ("other", [r for r in rows if r["group"] == "other"]),
        ("clean", [r for r in rows if r["clean"]]),
        ("not clean", [r for r in rows if not r["clean"]]),
    ):
        n = len(xs)
        if not n:
            continue
        summary[name] = {
            "scenes": n,
            "selected": sum(r["selected"] for r in xs),
            "clean": sum(r["clean"] for r in xs),
            "clean_plain": sum(r["clean_plain"] for r in xs),
            "clean_bubble": sum(r["clean_bubble"] for r in xs),
            "anchor_letters": {
                f: round(sum(r[f"anchor_{f}"] for r in xs) / n / len(SEEDS), 2)
                for f in G.FRAMES
            },
            "solo": {
                f"{key} {f}": round(
                    sum(r[f"solo_{key}_{f}"] for r in xs) / n / len(SEEDS), 4
                )
                for _, key in PROBES
                for f in G.FRAMES
            },
            "extra": {
                f"{key} {f}": round(sum(r[f"extra_{key}_{f}"] for r in xs) / n, 2)
                for _, key in PROBES
                for f in G.FRAMES
            },
        }
    return rows, summary


def write_asset(cs: list, rows: list) -> None:
    full = {c["i"]: c for c in cs}
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    ASSET.write_text(
        json.dumps(
            {
                "probe": "project/cjk_rl/probes/glyph_scenes.py",
                "rule": f"plain no-clause renders letter nothing; bubble ones on at "
                f"most {BUBBLE_MAX} of {len(SEEDS)} seeds; stop / う read exactly on "
                f"{LAND_MIN} of their 8 bubble renders or more",
                "screened": len(rows),
                "cross": choose(rows),
                "scenes": [
                    r | {f: full[r["i"]][f] for f in G.FRAMES}
                    for r in rows
                    if r["selected"]
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"selected {sum(r['selected'] for r in rows)} of {len(rows)} → {ASSET}",
        flush=True,
    )


def rounds_on_disk(root: Path) -> int:
    """The rounds whose screen has been read (their stems are in the reads)."""
    f = root / "anchor" / G.ROWS / "reads.json"
    got = json.loads(f.read_text("utf-8")) if f.exists() else {}
    r = 1
    while any(
        f"sc{c['i']:02d}_plain_s0" in got
        for c in candidates(r + 1)
        if c["round"] == r + 1
    ):
        r += 1
    return r


def choose(rows: list) -> list:
    """The ``N_CROSS`` clean scenes of round 1 with the most exact reads of the probe
    strings in the bubble frame, one a folder."""
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: (-r["score"], r["i"])):
        if r["round"] != 1:
            continue
        if r["clean"] and r["artist"] not in seen:
            out.append(r["i"])
            seen.add(r["artist"])
        if len(out) == N_CROSS:
            break
    return out


def cross_table(js: list, reads: dict, img_terms: dict) -> dict:
    recs = defaultdict(list)
    for j in js:
        t = G.page_terms(j["key"], reads[(j["part"], j["arm"])][j["stem"]])
        m_init, fw = img_terms[f"{j['part']}/{j['arm']}/{j['stem']}"]
        recs[(j["cls"], j["key"])].append(
            t | {"m_init": m_init, "fw": fw, "scene": j["scene"]}
        )
    keys = {
        f"{cls} {key}": {
            "solo": sum(r["solo"] for r in xs),
            "on_page": sum(r["on_page"] for r in xs),
            "n": len(xs),
            "extra": round(G.mean(r["extra"] for r in xs), 2),
            "no_text": sum(r["drawn"] == 0 for r in xs),
        }
        for (cls, key), xs in recs.items()
    }
    out = {"keys": keys, "by_class": {}, "by_scene": {}, "spread": {}}
    for cls in ("hira", "kanji", "en"):
        xs = [r for (c, _), g in recs.items() if c == cls for r in g]
        out["by_class"][cls] = {
            "n": len(xs),
            "solo": round(G.mean(r["solo"] for r in xs), 4),
            "on_page": round(G.mean(r["on_page"] for r in xs), 4),
            "extra": round(G.mean(r["extra"] for r in xs), 2),
            "no_text": round(G.mean(r["drawn"] == 0 for r in xs), 4),
            "only_it": round(G.mean(r["solo"] and r["extra"] == 0 for r in xs), 4),
        }
        for s in sorted({r["scene"] for r in xs}):
            ys = [r for r in xs if r["scene"] == s]
            out["by_scene"][f"{cls} sc{s:02d}"] = {
                "solo": sum(r["solo"] for r in ys),
                "n": len(ys),
                "extra": round(G.mean(r["extra"] for r in ys), 2),
            }
        # groups: a string on a scene, 4 seeds; and a string over its 16 renders
        gs = defaultdict(list)
        for (c, key), g in recs.items():
            if c == cls:
                for r in g:
                    gs[(key, r["scene"])].append(r)
        wide = [g for (c, _), g in recs.items() if c == cls]
        out["spread"][cls] = {
            "groups_of_4": len(gs),
            "not_constant_4": round(
                G.mean(len({round(r["r_seq"], 6) for r in g}) > 1 for g in gs.values()),
                4,
            ),
            "not_constant_16": round(
                G.mean(len({round(r["r_seq"], 6) for r in g}) > 1 for g in wide), 4
            ),
            "std_4": {
                t: round(G.mean(G.std(r[t] for r in g) for g in gs.values()), 4)
                for t in ("r_seq", "g_p", "m_init", "fw")
            },
        }
    return out


def sheets(js: list, reads: dict, root: Path, out: Path, stage: str, chosen) -> None:
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    cell = (
        lambda j, head: (  # noqa: E731
            Image.open(T.img(j, root)).convert("RGB"),
            [head, " / ".join(G.boxes(reads[(j["part"], j["arm"])][j["stem"]]))[:26]],
        )
    )
    if stage == "screen":
        # a scene a row: its no-clause bubble renders, then stop and う in a bubble
        scenes = sorted({j["scene"] for j in js})
        for a in range(0, len(scenes), 8):
            rows = []
            for s in scenes[a : a + 8]:
                for key in (None, "stop", "う"):
                    rows += [
                        cell(j, f"sc{s:02d} {key or 'no clause'} s{j['seed']}")
                        for j in js
                        if (j["scene"], j["key"], j["frame"]) == (s, key, "bubble")
                    ]
            contact_sheet(rows, out / f"screen_bubble_{a:02d}.png", thumb=160, cols=12)
        return
    keys = list(dict.fromkeys((j["cls"], j["key"]) for j in js))
    for cls in dict.fromkeys(c for c, _ in keys):
        ks = [k for c, k in keys if c == cls]
        for a in range(0, len(ks), 8):
            rows = [
                cell(j, f"{j['key']} sc{j['scene']:02d} s{j['seed']}")
                for k in ks[a : a + 8]
                for j in js
                if j["key"] == k
            ]
            contact_sheet(rows, out / f"cross_{cls}_{a:02d}.png", thumb=128, cols=16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("screen", "cross", "select"))
    p.add_argument("--round", type=int, default=1, help="screen: which candidates")
    p.add_argument("--tabulate", action="store_true", help="CPU: the reads on disk")
    a = p.parse_args()
    root = OUT / "m15v11_px"
    gpu = not a.tabulate
    if a.stage == "select":
        cs = candidates(rounds_on_disk(root))
        js = screen_jobs(cs)
        rows, summary = screen_table(cs, js, T.read(js, root, gpu=False))
        for k, v in summary.items():
            print(f"  {k:<10} {v}", flush=True)
        write_asset(cs, rows)
        return
    cs = candidates(a.round if a.stage == "screen" else 1)
    js = screen_jobs([c for c in cs if c["round"] == a.round or a.stage == "cross"])
    if a.stage == "cross":  # round 1's reads decide the scenes
        rows, _ = screen_table(cs, js, T.read(js, root, gpu=False))
        chosen = choose(rows)
        print(f"cross on scenes {chosen}", flush=True)
        js = cross_jobs(cs, chosen)
    info = {}
    if gpu:
        info = T.render(js, root, "pixel_head")
        print(f"render: {info}", flush=True)
    reads = T.read(js, root, gpu=gpu)
    from bench._common import make_run_dir, write_result

    metrics: dict = {"render": info}
    if a.stage == "screen":
        cs = [c for c in cs if c["round"] == a.round]
        rows, metrics["summary"] = screen_table(cs, js, reads)
        chosen = choose(rows)
        metrics["chosen"] = chosen
        for k, v in metrics["summary"].items():
            print(f"  {k:<10} {v}", flush=True)
        print(f"  chosen {chosen}", flush=True)
    else:
        img_terms = G.image_terms(js, root, gpu=gpu, parts=("cross",))
        metrics["cross"] = cross_table(js, reads, img_terms)
        metrics["chosen"] = chosen
        for k in ("by_class", "by_scene", "spread"):
            for name, v in metrics["cross"][k].items():
                print(f"  {k:<9} {name:<12} {v}", flush=True)
    run_dir = make_run_dir(
        "cjk_rl", label=f"glyph_scenes-{a.stage}", root=T.LINE / "results"
    )
    sheets(js, reads, root, run_dir / "sheets", a.stage, chosen)
    if a.stage == "screen":
        full = {c["i"]: c for c in cs}
        rows = [r | {f: full[r["i"]][f] for f in G.FRAMES} for r in rows]
        (run_dir / "scenes.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print("selected prompts: run `glyph_scenes.py select`", flush=True)
    write_result(
        run_dir,
        script="project/cjk_rl/probes/glyph_scenes.py",
        args=vars(a) | {"model": T.MODEL, "seeds": SEEDS, "renders": str(root)},
        label=a.stage,
        metrics=metrics,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
