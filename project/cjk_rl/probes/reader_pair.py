#!/usr/bin/env python
"""reader_pair — hayai-ocr v2.7 Supernova beside the tuned reader on the glyph stage's
pages (forward only; no render, no update).

Pages on disk are read again: AnimeText boxes, then every box's crop
(``common.readers.crop``, 12 % a side) by both readers. A set is a ``records.json`` of
``src/g1.py`` (``arm`` / ``key`` / ``scene`` / ``seed`` / ``file``) or a directory of
``<arm>/<key>_sc<NN>_<frame>_s<seed>.webp`` (``glyph_scenes.py cross``).

Per box: the tuned reader's text and confidence (``Readers.read_images``); hayai's text
(``model.ocr``, ``max_patches="auto"``, ``escalate=False``, greedy, the labeling pass's
call) and a confidence of the same kind, the mean probability of its own read's tokens
under a teacher-forced forward. On a ``ce`` set, both readers' teacher-forced
cross-entropy of the asked string on the crop (``gd.Reader.ce``; hayai's is a mean over
the string's tokens and the EOS as well).

Tables: per arm and per glyph ``solo`` / ``on_page`` / ``extra`` (``glyph_g0.page_terms``)
under each reader; the two readers' agreement per box and per page; the share of reads
under ``conf_min`` among boxes that read the asked string and among the others; the rank
correlation of the two cross-entropies. Sheets: the crops one reader reads as the asked
string and the other does not, numbered, in a shuffled order, with the asked string
under each (``sheets/key.json`` holds each cell's case and both reads).

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Result →
``results/<ts>-reader_pair-<config stem>/`` (``reads.json`` a page, ``result.json``).

    make daemon-run ARGS="--queue --stall-timeout 600 project/cjk_rl/probes/reader_pair.py --config project/cjk_rl/configs/<name>.toml"
    .venv/bin/python project/cjk_rl/probes/reader_pair.py --tabulate project/cjk_rl/results/<run>   # CPU
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import tomllib
from collections import defaultdict
from math import comb
from pathlib import Path

import glyph_g0 as G
import turbo_read as T

import gd  # noqa: E402  (src/, on the path through turbo_read)

STEM = re.compile(r"^(?P<key>.+)_sc(?P<scene>\d+)_(?:[a-z]+_)?s(?P<seed>\d+)$")
CELLS = 48  # crops a sheet


def pages_of(st: dict, smoke: bool) -> list:
    """``[{set, arm, key, scene, seed, file}]`` of a config ``[[sets]]`` entry."""
    out = []
    if "records" in st:
        for r in json.loads((T.REPO / st["records"]).read_text("utf-8")):
            out.append({k: r[k] for k in ("arm", "key", "scene", "seed", "file")})
    else:
        for f in sorted((T.REPO / st["dir"]).glob("*/*.webp")):
            m = STEM.match(f.stem)
            assert m, f
            out.append(
                {
                    "arm": f.parent.name,
                    "key": m["key"],
                    "scene": int(m["scene"]),
                    "seed": int(m["seed"]),
                    "file": str(f),
                }
            )
    if smoke:
        out = out[:: max(1, len(out) // 32)][:32]
    return [{"set": st["name"]} | p for p in out]


def run(cfg: dict, smoke: bool, out: Path) -> dict:
    """Reads every page of every set; writes ``reads.json``; returns the timings."""
    import math

    import numpy as np
    import torch
    from common.readers import crop, load_bgr
    from PIL import Image

    rdr = gd.Reader("cuda")
    hy = gd.Hayai("cuda", cfg["hayai"], cfg["revision"])
    W, H = G.SHAPE
    sec = defaultdict(float)
    n_box = defaultdict(int)
    first = {"same": 0, "n": 0}
    pages = []

    def lap(key: str, t0: float) -> float:
        torch.cuda.synchronize()
        sec[key] += time.time() - t0
        return time.time()

    for st in cfg["sets"]:
        ps = pages_of(st, smoke)
        t_set = time.time()
        for i in range(0, len(ps), T.READ_PAGES):
            chunk = ps[i : i + T.READ_PAGES]
            bgrs = [load_bgr(Path(p["file"])) for p in chunk]
            t = time.time()
            reads = rdr.rd.read_images(bgrs)
            t = lap(f"{st['name']}/det_vl", t)
            owners = [
                (b, r) for b, rs in enumerate(reads) for r in rs if not r.get("whole")
            ]
            crops = [
                Image.fromarray(
                    np.ascontiguousarray(crop(bgrs[b], r["box"])[:, :, ::-1])
                )
                for b, r in owners
            ]
            texts = hy.read(crops)
            t = lap(f"{st['name']}/hayai", t)
            n_box[st["name"]] += len(crops)
            for (b, r), im, text in zip(owners, crops, texts):
                r["hy"], r["hy_conf"] = text, None
                if text:
                    pr, top = hy.probs(im, text)
                    r["hy_conf"] = sum(pr[:-1]) / max(1, len(pr) - 1)
                    first["n"] += 1
                    ids = hy.tok(text, add_special_tokens=False).input_ids
                    first["same"] += int(bool(ids) and top == ids[0])
                if st.get("ce"):
                    pr, _ = hy.probs(im, chunk[b]["key"])
                    r["hy_ce"] = -sum(math.log(max(x, 1e-30)) for x in pr) / len(pr)
            t = lap(f"{st['name']}/hayai_tf", t)
            if st.get("ce"):
                for b, p in enumerate(chunk):
                    bs = [r for r in reads[b] if not r.get("whole")]
                    if not bs:
                        continue
                    p01 = (
                        torch.from_numpy(bgrs[b][:, :, ::-1].copy())
                        .permute(2, 0, 1)
                        .float()
                        .div(255)
                        .cuda()
                    )
                    with torch.no_grad():
                        ce = rdr.ce(
                            [p01] * len(bs),
                            [bgrs[b]] * len(bs),
                            [gd.crop_box(r["box"], W, H) for r in bs],
                            [p["key"]] * len(bs),
                        )
                    for r, c in zip(bs, ce.tolist()):
                        r["vl_ce"] = c
                lap(f"{st['name']}/vl_ce", t)
            for p, rs in zip(chunk, reads):
                pages.append(p | {"reads": rs})
        print(
            f"  {st['name']}: {len(ps)} pages, {n_box[st['name']]} boxes "
            f"({(time.time() - t_set) / 60:.1f} min)",
            flush=True,
        )
    (out / "reads.json").write_text(json.dumps(pages, ensure_ascii=False), "utf-8")
    return {
        "seconds": {k: round(v, 2) for k, v in sec.items()},
        "boxes": dict(n_box),
        "hayai_tf_first_token": first,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


# ---------------------------------------------------------------- tables


def as_reader(reads: list, name: str) -> list:
    """``reads`` with ``sfx`` / ``sfx_conf`` holding ``name``'s read (``vl`` | ``hy``)."""
    if name == "vl":
        return reads
    return [r | {"sfx": r.get("hy"), "sfx_conf": r.get("hy_conf")} for r in reads]


def sign_p(up: int, down: int) -> float:
    n, k = up + down, min(up, down)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def spearman(xs: list, ys: list) -> float | None:
    def rank(v):
        o = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r

    n = len(xs)
    if n < 3:
        return None
    a, b = rank(xs), rank(ys)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if not va or not vb:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def tabulate(pages: list, cfg: dict) -> dict:
    from common.text import norm

    cm = cfg["conf_min"]
    m = {}
    for name in dict.fromkeys(p["set"] for p in pages):
        ps = [p for p in pages if p["set"] == name]
        arms = list(dict.fromkeys(p["arm"] for p in ps))
        keys = list(dict.fromkeys(p["key"] for p in ps))
        terms = {}  # (reader, arm, key) → [page_terms]
        for p in ps:
            for rd in ("vl", "hy"):
                t = G.page_terms(p["key"], as_reader(p["reads"], rd))
                case, r = gd.pick(p["key"], as_reader(p["reads"], rd), cm)
                t |= {"case": case, "ce": r and r.get(f"{rd}_ce")}
                p[rd] = t
                terms.setdefault((rd, p["arm"], p["key"]), []).append(t)

        def tot(rd, arm, x, key=None):
            return sum(
                t[x]
                for (r, a, k), ts in terms.items()
                if r == rd and a == arm and key in (None, k)
                for t in ts
            )

        by_arm = {}
        for arm in arms:
            n = sum(p["arm"] == arm for p in ps)
            by_arm[arm] = {"pages": n} | {
                rd: {
                    "solo": tot(rd, arm, "solo"),
                    "on_page": tot(rd, arm, "on_page"),
                    "extra_a_page": round(tot(rd, arm, "extra") / n, 2),
                    "low": sum(p[rd]["case"] == "low" for p in ps if p["arm"] == arm),
                }
                for rd in ("vl", "hy")
            }
            by_arm[arm]["no_box"] = sum(
                p["vl"]["case"] == "none" for p in ps if p["arm"] == arm
            )
            # pages by (the tuned reader's solo, hayai's solo)
            by_arm[arm]["solo_2x2"] = {
                f"vl{a}_hy{b}": sum(
                    p["vl"]["solo"] == a and p["hy"]["solo"] == b
                    for p in ps
                    if p["arm"] == arm
                )
                for a in (1, 0)
                for b in (1, 0)
            }
        by_key = {
            key: {
                arm: {rd: tot(rd, arm, "solo", key) for rd in ("vl", "hy")}
                | {"pages": len(terms.get(("vl", arm, key), []))}
                for arm in arms
                if ("vl", arm, key) in terms
            }
            for key in keys
        }
        boxes = [(p, r) for p in ps for r in p["reads"] if not r.get("whole")]
        nv = [norm(r["sfx"] or "") for _, r in boxes]
        nh = [norm(r.get("hy") or "") for _, r in boxes]
        nk = [norm(p["key"]) for p, _ in boxes]
        box = {
            "n": len(boxes),
            "vl_empty": sum(not v for v in nv),
            "hy_empty": sum(not h for h in nh),
            "same_text": sum(v == h and bool(v) for v, h in zip(nv, nh)),
            "both_key": sum(v == k and h == k for v, h, k in zip(nv, nh, nk)),
            "vl_only_key": sum(v == k and h != k for v, h, k in zip(nv, nh, nk)),
            "hy_only_key": sum(v != k and h == k for v, h, k in zip(nv, nh, nk)),
            "neither_key": sum(v != k and h != k for v, h, k in zip(nv, nh, nk)),
        }
        conf = {}
        for rd, texts, ck in (("vl", nv, "sfx_conf"), ("hy", nh, "hy_conf")):
            hit = [r[ck] for (_, r), t, k in zip(boxes, texts, nk) if t and t == k]
            miss = [r[ck] for (_, r), t, k in zip(boxes, texts, nk) if t and t != k]
            conf[rd] = {
                "key_n": len(hit),
                "key_under": sum((c or 0) < cm for c in hit),
                "other_n": len(miss),
                "other_under": sum((c or 0) < cm for c in miss),
            }
        d = {"by_arm": by_arm, "by_key": by_key, "box": box, "conf": conf}
        if len(arms) == 2 and all(len(v) == 2 for v in by_key.values()):
            a, b = arms
            d["paired"] = {}
            for rd in ("vl", "hy"):
                up = sum(v[b][rd] > v[a][rd] for v in by_key.values())
                dn = sum(v[b][rd] < v[a][rd] for v in by_key.values())
                d["paired"][rd] = {"up": up, "down": dn, "p": round(sign_p(up, dn), 4)}
        ce = [(r["vl_ce"], r["hy_ce"]) for _, r in boxes if "vl_ce" in r]
        if ce:
            d["ce"] = {
                "boxes": len(ce),
                "spearman": round(spearman(*zip(*ce)), 3),
                # the picked crop's cross-entropy, each reader on its own pick
                "picked_mean": {
                    arm: {
                        rd: round(
                            G.mean(
                                p[rd]["ce"]
                                for p in ps
                                if p["arm"] == arm and p[rd]["ce"] is not None
                            ),
                            3,
                        )
                        for rd in ("vl", "hy")
                    }
                    for arm in arms
                },
            }
        m[name] = d
    return m


def show(m: dict) -> None:
    for name, d in m.items():
        print(f"\n== {name}")
        for arm, v in d["by_arm"].items():
            print(
                f"  {arm:>10} {v['pages']:4d} pages, no box {v['no_box']:3d} | "
                + " | ".join(
                    f"{rd} solo {v[rd]['solo']:3d} on_page {v[rd]['on_page']:3d} "
                    f"extra {v[rd]['extra_a_page']:.2f} low {v[rd]['low']:3d}"
                    for rd in ("vl", "hy")
                )
                + f" | {v['solo_2x2']}"
            )
        print(f"  boxes {d['box']}")
        print(f"  conf {d['conf']}")
        if "paired" in d:
            print(f"  paired {d['paired']}")
        if "ce" in d:
            print(f"  ce {d['ce']}")
        arms = list(d["by_arm"])
        print("  solo by key (" + " | ".join(arms) + "), vl / hy:")
        for key, v in d["by_key"].items():
            print(
                f"    {key:<8} "
                + " | ".join(f"{v[a]['vl']:2d} / {v[a]['hy']:2d}" for a in v)
            )


def sheets(pages: list, out: Path, seed: int = 1125) -> dict:
    """The crops one reader reads as the asked string and the other does not."""
    from common.readers import contact_sheet, crop, load_bgr
    from common.text import norm
    from PIL import Image

    cells = []
    for p in pages:
        k = norm(p["key"])
        for r in p["reads"]:
            if r.get("whole"):
                continue
            v, h = norm(r["sfx"] or "") == k, norm(r.get("hy") or "") == k
            if v != h:
                cells.append((p, r, "vl_only" if v else "hy_only"))
    random.Random(seed).shuffle(cells)
    out.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for n, (p, r, case) in enumerate(cells):
        c = crop(load_bgr(Path(p["file"])), r["box"])
        im = Image.fromarray(c[:, :, ::-1])
        s = 256 / max(im.size)
        im = im.resize(
            (max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS
        )
        rows.append((im, [f"#{n}  {p['key']}"]))
        key[n] = {
            "case": case,
            "set": p["set"],
            "arm": p["arm"],
            "key": p["key"],
            "file": Path(p["file"]).name,
            "box": r["box"],
            "vl": r["sfx"],
            "vl_conf": r["sfx_conf"],
            "hy": r.get("hy"),
            "hy_conf": r.get("hy_conf"),
        }
    for i in range(0, len(rows), CELLS):
        contact_sheet(
            rows[i : i + CELLS], out / f"disagree_{i // CELLS:02d}.png", cols=8
        )
    (out / "key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), "utf-8"
    )
    return {
        "cells": len(cells),
        "vl_only": sum(c == "vl_only" for *_, c in cells),
        "hy_only": sum(c == "hy_only" for *_, c in cells),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", help="configs/<name>.toml; its stem names the run")
    p.add_argument("--smoke", action="store_true", help="32 pages a set")
    p.add_argument("--tabulate", help="CPU: a result dir's reads, tables and sheets")
    a = p.parse_args()

    if a.tabulate:
        d = Path(a.tabulate)
        res = json.loads((d / "result.json").read_text("utf-8"))
        pages = json.loads((d / "reads.json").read_text("utf-8"))
        m = tabulate(pages, res["args"]["cfg"])
        show(m)
        res["metrics"] = {
            "run": res["metrics"]["run"],
            "sets": m,
            "sheets": sheets(pages, d / "sheets"),
        }
        (d / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), "utf-8"
        )
        return

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    label = ("smoke_" if a.smoke else "") + Path(a.config).stem

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_rl", label=f"reader_pair-{label}", root=T.LINE / "results"
    )
    info = run(cfg, a.smoke, run_dir)
    pages = json.loads((run_dir / "reads.json").read_text("utf-8"))
    m = tabulate(pages, cfg)
    show(m)
    print(f"\nrun {info}", flush=True)
    write_result(
        run_dir,
        script="project/cjk_rl/probes/reader_pair.py",
        args=vars(a) | {"cfg": cfg},
        label=label,
        metrics={"run": info, "sets": m, "sheets": sheets(pages, run_dir / "sheets")},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
