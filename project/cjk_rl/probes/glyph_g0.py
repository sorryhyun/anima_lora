#!/usr/bin/env python
"""glyph_g0 — the forward probes of ``archive/proposal_glyph.md`` § 4 (G0a – c): the tagless
frame on ``m15v11`` (4 steps, cfg 1, pixel-head decode), no update.

Frame: ``<scene>. Text reads as "<string>".`` at 768². A scene is a training caption
with its text clauses, its text tags and its bubble tags out (``scene_tags``); the
``bubble`` frame puts ``speech bubble`` back, the ``plain`` frame does not. The anchor
of a render is its scene with no clause, same frame, seed and sampler.

    G0a  24 hiragana and the 32 frequent kanji (jp_v1's rows, ``seed_1008``) and 24
         one-token EN words (rows off) × the two frames × 8 seeds, Euler; string n of a
         class sits on scene n mod 8. Per class and frame: a box reads exactly the
         string (``solo``), the string is in a box (``on_page``), letters drawn beside
         it (``extra``); the null is the next string of the class on the same scene,
         scored on this page. The no-clause render of the 8 scenes and of 16 more:
         what a scene letters by itself.
    G0b  per group of 8 seeds: the within-group std of r_seq, g_p, m_init and fw, and
         the share of groups whose seeds do not all score alike. On Euler (160 groups)
         and on P0's SDE, a = 0.7 (16 groups: G-R's sampler).
    G0c  4 strings × 2 frames × 8 seeds, Euler and SDE: the same latents decoded by
         the pixel head and by the VAE, read by the same reader; paired terms and a
         sign test.

``g_p`` here counts every letter and digit the reader returns (``str.isalnum`` after
``norm``), so Latin lettering on a page counts as drawn; ``eval.ruler.score``'s counts
kana and kanji only. ``r_seq`` is ``src/reward.py``'s.

Renders: ``output/cjk_rl/glyph_g0/<part>/<arm>/`` (a render on disk is not drawn
again), reads and image terms cached beside them. Result →
``results/<ts>-glyph_g0-<label>/`` with the sheets (per class, 4 strings a sheet: a
string's plain row above its bubble row, 8 seeds across).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/glyph_g0.py --label g0"
    .venv/bin/python project/cjk_rl/probes/glyph_g0.py --label g0 --tabulate    # CPU
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import turbo_read as T  # the pack, routing, the reseed path, the render and the reads

import reward  # noqa: E402

LINE = T.LINE
OUT = T.REPO / "output" / "cjk_rl" / "glyph_g0"
SEED = 1125
ROWS = "seed_1008"  # jp_v1
SEEDS = tuple(range(8))
SHAPE = (768, 768)
N_GRID, N_EXTRA = 8, 16  # scenes under the grid; scenes rendered with no clause only
FRAMES = ("plain", "bubble")
SDE = "sde07"
EN_WORDS = (
    "love stop yes no help home open exit sale cafe book star moon sun rain snow "
    "fire water cat dog fish bird tree rose"
).split()
# (class, index) of the strings also sampled with the SDE; the first four also get the
# VAE decode (G0c)
SDE_KEYS = (
    ("hira", 0),
    ("kanji", 0),
    ("kanji", 1),
    ("en", 0),
    ("hira", 1),
    ("hira", 2),
    ("kanji", 3),
    ("en", 3),
)
VAE_KEYS = SDE_KEYS[:4]
# tags that name lettering: out of a scene with the text tags and the bubbles
TEXTISH = re.compile(r"( text| writing| username| logo| name)$|^(name tag|sign|logo)$")
BUBBLES = ("speech bubble", "thought bubble")


# ---------------------------------------------------------------- build (CPU)


def scene_tags(tags) -> list:
    from eval.ruler.build import DROP_TAGS

    return [
        t
        for t in tags
        if t not in DROP_TAGS and t not in BUBBLES and not TEXTISH.search(t)
    ]


def scenes() -> list:
    """``N_GRID + N_EXTRA`` scenes drawn from the training captions (the ruler's images
    and the ``$CHARACTER_REF_EXCLUDE`` tags held out, as ``src/pool.py``):
    ``{stem, plain, bubble}``, the two no-clause captions."""
    from anime_tools.captions.position_clauses import (
        TEXT_PREFIXES,
        PositionClause,
        compose_caption,
        parse_caption,
    )

    from eval.ruler import RULER
    from eval.ruler.build import CAPTIONS
    from i2i_tasks.tools.character_ref_pairs import exclude_tags
    from library.env import load_dotenv

    load_dotenv()
    held = {
        m["stem"]
        for m in json.loads((RULER / "ruler.json").read_text("utf-8"))["items"]
    }
    excluded = exclude_tags()
    files = [
        f
        for f in sorted(CAPTIONS.rglob("*.txt"))
        if not f.name.endswith((".variants.txt", ".history.txt")) and f.stem not in held
    ]
    random.Random(SEED).shuffle(files)
    out = []
    for f in files:
        p = parse_caption(f.read_text(encoding="utf-8"))
        if excluded.intersection(p.flat_tags):
            continue
        tags = scene_tags(p.flat_tags)
        pos = [
            PositionClause(
                position=cl.position, tags=tuple(scene_tags(cl.tags)), prefix=cl.prefix
            )
            for cl in p.clauses
            if cl.prefix not in TEXT_PREFIXES
        ]
        out.append(
            {
                "stem": f.stem,
                "plain": compose_caption(tags, pos),
                "bubble": compose_caption([*tags, "speech bubble"], pos),
            }
        )
        if len(out) == N_GRID + N_EXTRA:
            break
    return out


def strings() -> dict:
    """class → its strings. Every JA glyph must reach a row in the tagless frame and
    every EN word must be one T5 token (asserted)."""
    from common.models import checkpoints
    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.anima.weights import load_t5_tokenizer
    from library.inference.text import ensure_text_strategies

    out = {
        "hira": list(T.KANA),
        "kanji": T.kanji_sets()["kanji frequent"],
        "en": EN_WORDS,
    }
    t5 = load_t5_tokenizer()
    for w in EN_WORDS:
        assert len(t5(w, add_special_tokens=False)["input_ids"]) == 1, w
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    for cls in ("hira", "kanji"):
        for k in out[cls]:
            ids = tok.tokenize(f'1girl, solo. Text reads as "{k}".')[2][0].tolist()
            assert sum(i >= T5_TABLE_SIZE for i in ids) == 1, (k, "no single row")
    return out


def framed(scene: str, key: str) -> str:
    return f'{scene.rstrip(".")}. Text reads as "{key}".'


def jobs(smoke: bool) -> tuple[list, list]:
    """``(jobs, scenes)``. Parts: ``grid`` / ``anchor`` (Euler), ``grid_sde`` /
    ``anchor_sde``; a job with ``vae`` also has its ``<part>_vae`` copy."""
    sc, st = scenes(), strings()
    seeds = SEEDS[:2] if smoke else SEEDS
    out = []
    for cls, keys in st.items():
        arm = T.EN if cls == "en" else ROWS
        for n, key in enumerate(keys):
            if smoke and n >= 2:
                continue
            si = n % N_GRID
            nxt = keys[(n + N_GRID) % len(keys)]  # the null: same scene, another string
            for frame in FRAMES:
                for part, sampler in (("grid", "euler"), ("grid_sde", SDE)):
                    if part == "grid_sde" and (cls, n) not in SDE_KEYS:
                        continue
                    for seed in seeds:
                        out.append(
                            {
                                "part": part,
                                "arm": arm,
                                "stem": f"{key}_{frame}_s{seed}",
                                "caption": framed(sc[si][frame], key),
                                "seed": seed,
                                "shape": SHAPE,
                                "sampler": sampler,
                                "vae": (cls, n) in VAE_KEYS,
                                "cls": cls,
                                "n": n,
                                "key": key,
                                "null": nxt,
                                "frame": frame,
                                "scene": si,
                            }
                        )
    sde_scenes = {j["scene"] for j in out if j["part"] == "grid_sde"}
    for si, s in enumerate(sc):
        if smoke and si >= 2:
            continue
        for frame in FRAMES:
            for part, sampler in (("anchor", "euler"), ("anchor_sde", SDE)):
                if part == "anchor_sde" and si not in sde_scenes:
                    continue
                for seed in seeds:
                    out.append(
                        {
                            "part": part,
                            "arm": ROWS,
                            "stem": f"sc{si:02d}_{frame}_s{seed}",
                            "caption": s[frame],
                            "seed": seed,
                            "shape": SHAPE,
                            "sampler": sampler,
                            "frame": frame,
                            "scene": si,
                        }
                    )
    return out, sc


def anchor_of(j: dict) -> dict:
    part = "anchor_sde" if j["part"].startswith("grid_sde") else "anchor"
    return j | {
        "part": part,
        "arm": ROWS,
        "stem": f"sc{j['scene']:02d}_{j['frame']}_s{j['seed']}",
    }


# ---------------------------------------------------------------- terms


def boxes(reads: list) -> list:
    from common.text import norm

    out = [norm(r["sfx"]) for r in reads if not r.get("whole") and r.get("sfx")]
    return [b for b in out if b]


def page_terms(key: str, reads: list) -> dict:
    from common.text import norm

    k, bs = norm(key), boxes(reads)
    drawn = Counter(c for b in bs for c in b if c.isalnum())
    n = sum(drawn.values())
    hit = sum((drawn & Counter(k)).values())
    return {
        "solo": int(k in bs),
        "on_page": int(any(k in b for b in bs)),
        "extra": n - hit,
        "drawn": n,
        "g_p": hit / n if n else 0.0,
        "r_seq": reward.r_seq([key], reads),
    }


def image_terms(js: list, root: Path, gpu: bool, parts=("grid", "grid_sde")) -> dict:
    """``{part/arm/stem: [m_init, fw]}`` for the renders of ``parts`` (pixel head): the
    ``AtSim`` match to the anchor render and the flat-white share over the anchor's,
    as ``src/grpo.py`` computes them; cached in ``image_terms.json``."""
    from PIL import Image

    f = root / "image_terms.json"
    got = json.loads(f.read_text("utf-8")) if f.exists() else {}
    name = lambda j: f"{j['part']}/{j['arm']}/{j['stem']}"  # noqa: E731
    todo = [j for j in js if j["part"] in parts and name(j) not in got]
    if todo:
        assert gpu, f"{len(todo)} renders without image terms (run on the daemon)"
        import grpo
        from eval.ruler.score import AtSim

        at = AtSim("cuda")
        load = lambda j: Image.open(T.img(j, root)).convert("RGB")  # noqa: E731
        for i in range(0, len(todo), 8):
            g = todo[i : i + 8]
            ims, ans = [load(j) for j in g], [load(anchor_of(j)) for j in g]
            for j, m, im, an in zip(g, grpo.match_init(at, ims, ans), ims, ans):
                got[name(j)] = [m, grpo.flat_white_img(im) - grpo.flat_white_img(an)]
        f.write_text(json.dumps(got), "utf-8")
    return got


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def std(xs) -> float:
    xs = list(xs)
    m = mean(xs)
    return (
        (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    )


def tabulate(js: list, reads: dict, img_terms: dict) -> tuple[dict, list]:
    """The tables and the per-render records."""
    from scipy.stats import binomtest

    rd = lambda j: reads[(j["part"], j["arm"])][j["stem"]]  # noqa: E731
    recs = []
    for j in js:
        if not j["part"].startswith("grid"):
            continue
        r = rd(j)
        t = page_terms(j["key"], r)
        nl = page_terms(j["null"], r)
        m_init, fw = img_terms.get(f"{j['part']}/{j['arm']}/{j['stem']}", (None, None))
        recs.append(
            {k: j[k] for k in ("part", "cls", "n", "key", "frame", "scene", "seed")}
            | t
            | {"solo_null": nl["solo"], "on_page_null": nl["on_page"]}
            | {"m_init": m_init, "fw": fw, "boxes": boxes(r)}
        )
    out: dict = {}

    # G0a: per class and frame
    for part in ("grid", "grid_sde"):
        tab = {}
        for cls in ("hira", "kanji", "en"):
            for frame in FRAMES:
                xs = [
                    r
                    for r in recs
                    if (r["part"], r["cls"], r["frame"]) == (part, cls, frame)
                ]
                if not xs:
                    continue
                tab[f"{cls} | {frame}"] = {
                    "n": len(xs),
                    "solo": round(mean(r["solo"] for r in xs), 4),
                    "solo_null": round(mean(r["solo_null"] for r in xs), 4),
                    "on_page": round(mean(r["on_page"] for r in xs), 4),
                    "on_page_null": round(mean(r["on_page_null"] for r in xs), 4),
                    "extra": round(mean(r["extra"] for r in xs), 2),
                    "no_text": round(mean(r["drawn"] == 0 for r in xs), 4),
                    "only_it": round(
                        mean(r["solo"] and r["extra"] == 0 for r in xs), 4
                    ),
                    "r_seq": round(mean(r["r_seq"] for r in xs), 4),
                    "g_p": round(mean(r["g_p"] for r in xs), 4),
                }
        out["g0a" if part == "grid" else "g0a_sde"] = tab
    keys = defaultdict(dict)
    for r in recs:
        if r["part"] == "grid":
            d = keys[f"{r['cls']} {r['key']}"].setdefault(
                r["frame"], {"solo": 0, "on_page": 0, "extra": 0.0, "scene": r["scene"]}
            )
            d["solo"] += r["solo"]
            d["on_page"] += r["on_page"]
            d["extra"] += r["extra"] / len(SEEDS)
    out["g0a_keys"] = keys

    # the no-clause renders: what a scene letters by itself
    anc = defaultdict(list)
    for j in js:
        if j["part"] == "anchor":
            bs = boxes(rd(j))
            anc[(j["scene"], j["frame"])].append(
                sum(c.isalnum() for b in bs for c in b)
            )
    out["anchors"] = {
        frame: {
            "scenes": sum(f == frame for _, f in anc),
            "renders_with_text": round(
                mean(v > 0 for (_, f), xs in anc.items() if f == frame for v in xs), 4
            ),
            "letters_a_render": round(
                mean(v for (_, f), xs in anc.items() if f == frame for v in xs), 2
            ),
            # scenes lettering on half their seeds or more
            "scenes_lettering": sorted(
                s
                for (s, f), xs in anc.items()
                if f == frame and mean(v > 0 for v in xs) >= 0.5
            ),
            "by_scene": {
                s: [round(mean(v > 0 for v in xs), 2), round(mean(xs), 1)]
                for (s, f), xs in sorted(anc.items())
                if f == frame
            },
        }
        for frame in FRAMES
    }

    # G0b: within-group spread of each term
    terms = ("r_seq", "g_p", "m_init", "fw")
    groups = defaultdict(list)
    for r in recs:
        groups[(r["part"], r["cls"], r["key"], r["frame"])].append(r)
    g0b = {}
    for part in ("grid", "grid_sde"):
        for side in ("ja", "en"):
            for frame in FRAMES:
                gs = [
                    g
                    for (p, cls, _, f), g in groups.items()
                    if p == part and f == frame and (cls == "en") == (side == "en")
                ]
                if not gs or any(r["m_init"] is None for g in gs for r in g):
                    continue
                sd = {t: mean(std(r[t] for r in g) for g in gs) for t in terms}
                w = {
                    "beta": sd["r_seq"] / sd["g_p"] if sd["g_p"] else 0.0,
                    "gamma": sd["r_seq"] / sd["m_init"] if sd["m_init"] else 0.0,
                    "delta": sd["r_seq"] / sd["fw"] if sd["fw"] else 0.0,
                }
                rs = [
                    [
                        r["r_seq"]
                        + w["beta"] * r["g_p"]
                        - w["gamma"] * (1 - r["m_init"])
                        - w["delta"] * max(0.0, r["fw"])
                        for r in g
                    ]
                    for g in gs
                ]
                g0b[f"{part} | {side} | {frame}"] = {
                    "groups": len(gs),
                    "std": {t: round(v, 4) for t, v in sd.items()},
                    "nondegenerate": {
                        t: round(
                            mean(len({round(r[t], 6) for r in g}) > 1 for g in gs), 4
                        )
                        for t in ("r_seq", "g_p")
                    },
                    "weights": {k: round(v, 3) for k, v in w.items()},
                    "r_std": round(mean(std(x) for x in rs), 4),
                    "r_seq_mean": round(mean(r["r_seq"] for g in gs for r in g), 4),
                }
    out["g0b"] = g0b

    # G0c: the pixel head against the VAE on the same latents
    g0c = {}
    for part in ("grid", "grid_sde"):
        pairs = []
        for j in js:
            if j["part"] == part and j.get("vae"):
                v = reads[(part + "_vae", j["arm"])][j["stem"]]
                pairs.append((page_terms(j["key"], rd(j)), page_terms(j["key"], v)))
        if not pairs:
            continue
        tab = {"pairs": len(pairs)}
        for t in ("r_seq", "g_p", "solo", "on_page", "drawn"):
            up = sum(a[t] > b[t] for a, b in pairs)
            dn = sum(a[t] < b[t] for a, b in pairs)
            tab[t] = {
                "pixel_head": round(mean(a[t] for a, _ in pairs), 4),
                "vae": round(mean(b[t] for _, b in pairs), 4),
                "head_higher": up,
                "vae_higher": dn,
                "sign_p": round(float(binomtest(up, up + dn).pvalue), 4)
                if up + dn
                else 1.0,
            }
        g0c[part] = tab
    out["g0c"] = g0c
    return out, recs


def sheets(js: list, reads: dict, root: Path, out: Path) -> None:
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    cell = (
        lambda j, head: (  # noqa: E731
            Image.open(T.img(j, root)).convert("RGB"),
            [head, " / ".join(boxes(reads[(j["part"], j["arm"])][j["stem"]]))[:26]],
        )
    )
    grid = [j for j in js if j["part"] == "grid"]
    for cls in dict.fromkeys(j["cls"] for j in grid):
        ns = sorted({j["n"] for j in grid if j["cls"] == cls})
        for a in range(0, len(ns), 4):
            rows = [
                cell(j, f"{j['key']} {j['frame']} s{j['seed']} sc{j['scene']}")
                for n in ns[a : a + 4]
                for frame in FRAMES
                for j in grid
                if (j["cls"], j["n"], j["frame"]) == (cls, n, frame)
            ]
            contact_sheet(rows, out / f"grid_{cls}_{a:02d}.png", thumb=192, cols=8)
    for frame in FRAMES:
        rows = [
            cell(j, f"sc{j['scene']:02d} {frame} s{j['seed']}")
            for j in js
            if j["part"] == "anchor" and j["frame"] == frame and j["seed"] < 4
        ]
        if rows:
            contact_sheet(rows, out / f"anchor_{frame}.png", thumb=192, cols=8)
    for part in ("grid", "grid_sde"):
        for frame in FRAMES:
            rows = []
            for j in js:
                if j["part"] == part and j.get("vae") and j["frame"] == frame:
                    v = j | {"part": part + "_vae"}
                    rows += [cell(j, f"{j['key']} s{j['seed']} head"), cell(v, "vae")]
            if rows:
                contact_sheet(rows, out / f"g0c_{part}_{frame}.png", thumb=192, cols=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--smoke", action="store_true", help="a few items, its own dir")
    p.add_argument("--tabulate", action="store_true", help="CPU: the reads on disk")
    a = p.parse_args()
    root = OUT / ("_smoke" if a.smoke else "m15v11_px")
    js, sc = jobs(a.smoke)
    info = {}
    if not a.tabulate:
        info = T.render(js, root, "pixel_head")
        print(f"render: {info}", flush=True)
    vae = [j | {"part": j["part"] + "_vae"} for j in js if j.get("vae")]
    reads = T.read(js + vae, root, gpu=not a.tabulate)
    img_terms = image_terms(js, root, gpu=not a.tabulate)
    from bench._common import make_run_dir, write_result

    metrics, recs = tabulate(js, reads, img_terms)
    for k in ("g0a", "g0a_sde", "anchors", "g0b", "g0c"):
        for name, v in metrics[k].items():
            v = {x: y for x, y in v.items() if x != "by_scene"}
            print(f"  {k:<8} {name:<28} {v}", flush=True)
    run_dir = make_run_dir("cjk_rl", label=f"glyph_g0-{a.label}", root=LINE / "results")
    sheets(js, reads, root, run_dir / "sheets")
    (run_dir / "records.json").write_text(
        json.dumps({"scenes": sc, "records": recs}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/glyph_g0.py",
        args=vars(a) | {"model": T.MODEL, "seeds": SEEDS, "renders": str(root)},
        label=a.label,
        metrics={"render": info} | metrics,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
