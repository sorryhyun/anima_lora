#!/usr/bin/env python
"""Plain scenes — which scenes letter an EN word with no bubble tag.

The screen of ``glyph_scenes.py`` picked scenes by the bubble frame. This one reads
the plain frame only (the scene with no ``speech bubble`` tag): per candidate scene
its no-clause render and ``<scene>. Text reads as "<word>".`` for the config's EN
``words``, at the config's ``seeds``, on ``m15v11`` (4 steps, cfg 1, pixel-head
decode; ``turbo_read``'s render and read). An EN word holds no trained row, so the
count is the model's own lettering of the scene.

Candidates come from ``glyph_scenes.free_pool`` (the text-free captions, the ruler's
images and the ``$CHARACTER_REF_EXCLUDE`` tags held out), less the stems an earlier
screen holds (``results/*-glyph_scenes-screen/scenes.json``, ``results/*-plain_scenes-*
/scenes.json``): first every caption of the folders named in ``assets/plain_pool.json``
(``{"whole": [folder, …]}``, local; none when the file is absent), then ``per_folder``
a folder in turn over the shuffled folders, up to ``n``. They are written once to
``assets/<config stem>.json`` (local) and read from there afterwards; ids start at
``id0``.

Per scene: ``lettered`` (no-clause renders with a read letter), ``en_solo`` (renders
where a box reads exactly the word), ``en_on_page``, ``en_none`` (renders with no
letter read), ``extra`` a render, and ``solo_<word>``. Sheets hold a scene a row, the
scenes in the order of ``en_solo``: its no-clause renders, then each word's.

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Renders → ``output/cjk_rl/plain_scenes/
<stem>/``, result → ``results/<ts>-plain_scenes-<stem>/`` (``scenes.json``: the table
with the captions).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/plain_scenes.py --config project/cjk_rl/configs/<name>.toml"
    .venv/bin/python project/cjk_rl/probes/plain_scenes.py --config … --tabulate   # CPU
    .venv/bin/python project/cjk_rl/probes/plain_scenes.py --config … --select 8   # CPU

``--select N`` writes the run's scenes with ``en_solo`` of N or more and no lettered
no-clause render to ``assets/plain_scenes.json`` (local), which ``glyph_scenes.scenes``
reads beside the first screen's selection.
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import defaultdict
from pathlib import Path

import glyph_g0 as G
import glyph_scenes as S
import turbo_read as T

OUT = T.REPO / "output" / "cjk_rl" / "plain_scenes"
POOL = T.LINE / "assets" / "plain_pool.json"
FRAME = "plain"
GROUND = {"simple background", "white background"}
FIGURES = {"2girls", "3girls", "multiple girls", "1boy", "2boys", "multiple boys"}


def screened() -> set:
    """The stems earlier screens hold."""
    out = set()
    for pat in ("*-glyph_scenes-screen", "*-plain_scenes-*"):
        for f in (T.LINE / "results").glob(f"{pat}/scenes.json"):
            out |= {r["stem"] for r in json.loads(f.read_text("utf-8"))}
    return out


def candidates(cfg: dict, asset: Path) -> list:
    if asset.exists():
        return json.loads(asset.read_text("utf-8"))
    from anime_tools.captions.position_clauses import compose_caption

    rate, free = S.free_pool()
    seen = screened()
    whole = json.loads(POOL.read_text("utf-8"))["whole"] if POOL.exists() else []
    rng = random.Random(cfg["pool_seed"])
    picks = [(a, x, True) for a in whole for x in free[a] if x[0] not in seen]
    folders = sorted(a for a in free if a not in whole)
    rng.shuffle(folders)
    per = []
    for a in folders:
        xs = [x for x in free[a] if x[0] not in seen]
        rng.shuffle(xs)
        per.append([(a, x, False) for x in xs[: cfg["per_folder"]]])
    picks += [p[k] for k in range(cfg["per_folder"]) for p in per if k < len(p)]
    out = []
    for a, (stem, p), w in picks[: cfg["n"]]:
        k, c, t = rate[a]
        tags = set(p.flat_tags) | {t for cl in p.clauses for t in cl.tags}
        out.append(
            {
                "i": cfg["id0"] + len(out),
                "stem": stem,
                "artist": a,
                "whole": w,
                "clause_rate": round(c / k, 3),
                "textish_rate": round(t / k, 3),
                "ground": bool(tags & GROUND),
                "figures": bool(tags & FIGURES),
                "prose": bool(p.clauses),
                "tags": len(tags),
                "plain": compose_caption(list(p.flat_tags), p.clauses),
                "bubble": compose_caption([*p.flat_tags, "speech bubble"], p.clauses),
            }
        )
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return out


def jobs(cfg: dict, cs: list) -> list:
    out = []
    for c in cs:
        for seed in cfg["seeds"]:
            out.append(S.job("anchor", G.ROWS, c, FRAME, seed))
            for w in cfg["words"]:
                out.append(S.job("grid", T.EN, c, FRAME, seed, w, "en"))
    return out


def table(cfg: dict, cs: list, js: list, reads: dict) -> tuple[list, dict]:
    rows = {c["i"]: dict(c) for c in cs}
    acc = defaultdict(lambda: defaultdict(float))
    for j in js:
        rd, a = reads[(j["part"], j["arm"])][j["stem"]], acc[j["scene"]]
        if j["part"] == "anchor":
            a["lettered"] += S.letters(rd) > 0
            continue
        t = G.page_terms(j["key"], rd)
        a["en_solo"] += t["solo"]
        a["en_on_page"] += t["on_page"]
        a["en_none"] += t["drawn"] == 0
        a["extra"] += t["extra"]
        a[f"solo_{j['key']}"] += t["solo"]
    n_en = len(cfg["words"]) * len(cfg["seeds"])
    for i, a in acc.items():
        rows[i] |= {k: int(v) for k, v in a.items() if k != "extra"}
        rows[i]["extra"] = round(a["extra"] / n_en, 2)
        for w in cfg["words"]:
            rows[i].setdefault(f"solo_{w}", 0)
    rows = sorted(rows.values(), key=lambda r: (-r["en_solo"], r["i"]))

    def part(xs: list) -> dict:
        n = len(xs)
        return {
            "scenes": n,
            "en_solo": round(sum(r["en_solo"] for r in xs) / n / n_en, 3),
            "en_none": round(sum(r["en_none"] for r in xs) / n / n_en, 3),
            "extra": round(sum(r["extra"] for r in xs) / n, 2),
            "lettered": round(
                sum(r["lettered"] for r in xs) / n / len(cfg["seeds"]), 3
            ),
            "half_or_more": sum(r["en_solo"] * 2 >= n_en for r in xs),
        }

    groups = {"all": rows}
    for name, key in (("ground", "ground"), ("figures", "figures"), ("prose", "prose")):
        groups[f"{name}"] = [r for r in rows if r[key]]
        groups[f"no {name}"] = [r for r in rows if not r[key]]
    groups["whole folders"] = [r for r in rows if r["whole"]]
    summary = {k: part(v) for k, v in groups.items() if v}
    summary["en_solo histogram"] = {
        str(k): sum(r["en_solo"] == k for r in rows) for k in range(n_en + 1)
    }
    summary["words"] = {w: sum(r[f"solo_{w}"] for r in rows) for w in cfg["words"]}
    return rows, summary


def sheets(cfg: dict, rows: list, js: list, reads: dict, root: Path, out: Path) -> None:
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    at = defaultdict(list)
    for j in js:
        at[j["scene"]].append(j)
    cols = len(cfg["seeds"]) * (1 + len(cfg["words"]))
    for a in range(0, len(rows), 8):
        cells = []
        for r in rows[a : a + 8]:
            for key in (None, *cfg["words"]):
                for j in at[r["i"]]:
                    if j["key"] != key:
                        continue
                    rd = reads[(j["part"], j["arm"])][j["stem"]]
                    head = f"sc{r['i']} {key or 'no clause'} s{j['seed']}"
                    cells.append(
                        (
                            Image.open(T.img(j, root)).convert("RGB"),
                            [head, " / ".join(G.boxes(rd))[:22]],
                        )
                    )
        contact_sheet(cells, out / f"plain_{a:03d}.png", thumb=128, cols=cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="the first two candidates")
    ap.add_argument("--tabulate", action="store_true", help="CPU: the reads on disk")
    ap.add_argument("--select", type=int, help="CPU: write the selection at this count")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    stem = Path(a.config).stem
    if a.select is not None:
        (run,) = sorted((T.LINE / "results").glob(f"*-plain_scenes-{stem}"))
        rows = json.loads((run / "scenes.json").read_text("utf-8"))
        n_en = len(cfg["words"]) * len(cfg["seeds"])
        keep = [r for r in rows if r["en_solo"] >= a.select and not r["lettered"]]
        S.ASSET_PLAIN.write_text(
            json.dumps(
                {
                    "probe": "project/cjk_rl/probes/plain_scenes.py",
                    "run": stem,
                    "rule": f"plain no-clause renders letter nothing; {cfg['words']} "
                    f"read exactly on {a.select} of their {n_en} plain renders or more",
                    "screened": len(rows),
                    "scenes": keep,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"selected {len(keep)} of {len(rows)} → {S.ASSET_PLAIN}", flush=True)
        return
    cs = candidates(cfg, T.LINE / "assets" / f"{stem}.json")
    if a.smoke:
        cs = cs[:2]
    name = ("smoke_" if a.smoke else "") + stem
    root = OUT / name
    js = jobs(cfg, cs)
    print(
        f"{name}: {len(cs)} scenes of {len({c['artist'] for c in cs})} folders "
        f"({sum(c['whole'] for c in cs)} from whole folders) × (no clause + "
        f"{cfg['words']}) × seeds {cfg['seeds']}: {len(js)} renders",
        flush=True,
    )
    info = {}
    if not a.tabulate:
        info = T.render(js, root, "pixel_head")
        print(f"render: {info}", flush=True)
    reads = T.read(js, root, gpu=not a.tabulate)
    rows, summary = table(cfg, cs, js, reads)
    for k, v in summary.items():
        print(f"  {k:<18} {v}", flush=True)
    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_rl", label=f"plain_scenes-{name}", root=T.LINE / "results"
    )
    sheets(cfg, rows, js, reads, root, run_dir / "sheets")
    (run_dir / "scenes.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/plain_scenes.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "renders": str(root)},
        label=name,
        metrics={"render": info, "summary": summary},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
