#!/usr/bin/env python
"""G1 — sixteen rows trained on the reader's loss through G-D's path (``plan.md`` item 2).

Policy and path: ``gd.Policy`` (``m15v11``, 4 Euler steps, pixel-head decode, rows
``seed_1008``) and ``gd.Reader``. A group is one glyph on one training scene
(``<scene, bubble frame>. Text reads as "<glyph>".``) at 4 seeds:

- steps 0 – 2 for the 4 renders in one forward each, no graph;
- per render, the last step once with a graph (it is the render's own last step); the
  page is read; a page with a box takes the reader's cross-entropy of the glyph on the
  picked crop (``gd.pick``), a page with none −log of the detector's top score (the
  detector's backward runs on those pages only); the loss is pulled to the glyph's row;
- the render's row gradient is clipped to ``clip`` and added to the row's sum.

A row steps when ``accum`` of its renders are in: the mean gradient joins the row's
momentum (``beta``), and the row moves ``eta`` row norms against the momentum's
direction. A pass is every (glyph, training scene) once, shuffled; pass p draws seeds
``train_seed0 + 4p …``.

The read (after training, same process): every glyph on the read scenes at the read
seeds, start rows and trained rows, each page read as training reads it. Per glyph
``solo`` / ``on_page`` / ``extra`` (``glyph_g0.page_terms``), the next glyph's ``solo``
on the same pages (the null), the reader's cross-entropy on the picked crop, cases.
Sheets: per glyph the two arms as rows ``A`` / ``B`` in an order drawn per glyph
(``sheets/key.json`` holds it).

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Writes ``output/cjk_rl/g1/<config stem>/``
(``log.jsonl`` a row step, ``rows.pt``, the read's pages) and
``results/<ts>-g1-<config stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/src/g1.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import tomllib
from math import comb
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE / "probes"))

import glyph_g0 as G  # noqa: E402
import glyph_scenes as S  # noqa: E402
import turbo_read as T  # noqa: E402

import gd  # noqa: E402

ROOT = T.REPO / "output" / "cjk_rl" / "g1"
CLASSES = ("hira", "kanji")


def glyphs(cfg: dict, smoke: bool) -> list:
    """``[(cls, key, alt)]``; ``alt`` is the next glyph of the class."""
    out = []
    for cls in CLASSES:
        keys = list(cfg[cls])
        for n, key in enumerate(keys):
            if smoke and n:
                continue
            out.append((cls, key, keys[(n + 1) % len(keys)]))
    return out


def train(pol, rdr, cfg: dict, gl: list, sc: dict, scenes: list, out: Path) -> dict:
    import torch

    W, H = G.SHAPE
    frame, accum = cfg["frame"], cfg["accum"]
    dim = pol.raw.shape[1]
    init = pol.raw.detach().clone()
    cap = {(k, si): G.framed(sc[si][frame], k) for _, k, _ in gl for si in scenes}
    row = {k: pol.row_of(cap[(k, scenes[0])]) for _, k, _ in gl}
    acc = {k: torch.zeros(dim, device=pol.dev) for k in row}
    mom = {k: torch.zeros(dim, device=pol.dev) for k in row}
    prev: dict = {}
    since = {k: [] for k in row}  # the renders since the row's last step
    steps = dict.fromkeys(row, 0)
    rng = random.Random(cfg["shuffle_seed"])
    log = open(out / "log.jsonl", "a")
    t_run, n_groups, phase = time.time(), 0, {}

    def lap(name: str, t0: float) -> float:
        torch.cuda.synchronize()
        phase[name] = phase.get(name, 0.0) + time.time() - t0
        return time.time()

    for p in range(cfg["passes"]):
        seeds = [cfg["train_seed0"] + 4 * p + k for k in range(4)]
        order = [(k, si) for k in row for si in scenes]
        rng.shuffle(order)
        for key, si in order:
            t = time.time()
            e_graph = pol.embed(cap[(key, si)], grad=True)
            e = e_graph.detach()
            x3 = pol.head_steps(pol.noise(seeds, G.SHAPE), e)
            t = lap("head", t)
            for b in range(len(seeds)):
                px, ei = pol.last(x3[b : b + 1], e)
                t = lap("last_fwd", t)
                img = px.detach()
                bgrs, reads = rdr.read(img)
                t = lap("read", t)
                case, r = gd.pick(key, reads[0], cfg["conf_min"])
                leaf = img.clone().requires_grad_(True)
                p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
                if r is None:
                    top = rdr.det_scores(p01).max(1).values
                    loss = top.clamp_min(1e-6).log().neg().sum()
                else:
                    box = gd.crop_box(r["box"], W, H)
                    loss = rdr.ce([p01[0]], bgrs, [box], [key]).sum()
                g_px = torch.autograd.grad(loss, leaf)[0]
                t = lap("loss", t)
                g_e = torch.autograd.grad((px * g_px).sum(), ei)[0]
                g = pol.row_grads(e_graph, g_e)[0, row[key]]
                del px, ei
                t = lap("last_bwd", t)
                n = float(g.norm())
                acc[key] += g * min(1.0, cfg["clip"] / max(n, 1e-30))
                since[key].append(
                    {
                        "case": case,
                        "loss": float(loss.detach()),
                        "norm": n,
                        "exact": bool(r and r["sfx"] and G.boxes([r]) == [_norm(key)]),
                    }
                )
            del e_graph
            n_groups += 1
            if len(since[key]) >= accum:
                rs = since[key]
                gbar = acc[key] / len(rs)
                mom[key] = cfg["beta"] * mom[key] + gbar
                d = mom[key] / mom[key].norm().clamp_min(1e-30)
                with torch.no_grad():
                    pol.raw[row[key]] -= cfg["eta"] * pol.row_scale * d
                steps[key] += 1
                cos = (
                    float(torch.nn.functional.cosine_similarity(gbar, prev[key], dim=0))
                    if key in prev
                    else None
                )
                prev[key] = gbar.clone()
                crop = [x for x in rs if x["case"] != "none"]
                rec = {
                    "pass": p,
                    "key": key,
                    "step": steps[key],
                    "renders": len(rs),
                    "gbar": float(gbar.norm()),
                    "cos_prev": cos,
                    "drift": float((pol.raw[row[key]] - init[row[key]]).norm())
                    / pol.row_scale,
                    "cases": {
                        c: sum(x["case"] == c for x in rs)
                        for c in ("box", "low", "none")
                    },
                    "exact": sum(x["exact"] for x in rs),
                    "ce": sum(x["loss"] for x in crop) / len(crop) if crop else None,
                    "clipped": sum(x["norm"] > cfg["clip"] for x in rs),
                    "groups": n_groups,
                    "min": round((time.time() - t_run) / 60, 2),
                }
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                acc[key].zero_()
                since[key] = []
            if n_groups % 16 == 0:
                print(
                    f"  pass {p} group {n_groups} / {cfg['passes'] * len(order)} "
                    f"steps {sum(steps.values())} peak "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                    f"({(time.time() - t_run) / 60:.1f} min)",
                    flush=True,
                )
    log.close()
    trained = pol.raw.detach().clone()
    torch.save(
        {
            "ext_ids": pol.r.delta.ext_ids,
            "rows": {k: row[k] for k in row},
            "raw": trained.cpu(),  # effective units, every ext row
            "init": init.cpu(),
            "row_scale": pol.row_scale,
            "cfg": cfg,
        },
        out / "rows.pt",
    )
    info = {
        "groups": n_groups,
        "minutes": round((time.time() - t_run) / 60, 2),
        "s_per_group": {k: round(v / n_groups, 3) for k, v in phase.items()},
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "steps": steps,
        "drift": {
            k: round(float((trained[row[k]] - init[row[k]]).norm()) / pol.row_scale, 4)
            for k in row
        },
    }
    return {"init": init, "trained": trained, "info": info}


def _norm(s: str) -> str:
    from common.text import norm

    return norm(s)


def read(pol, rdr, cfg: dict, gl: list, sc: dict, scenes: list, arms: dict, out: Path):
    """``[{arm, cls, key, scene, seed, case, solo, on_page, extra, null, ce, reads}]``
    and the pages on disk."""
    import torch
    from PIL import Image

    W, H = G.SHAPE
    seeds = list(cfg["read_seeds"])
    recs = []
    t0 = time.time()
    for arm, table in arms.items():
        pol.raw.data.copy_(table)
        for si in scenes:
            for cls, key, alt in gl:
                caption = G.framed(sc[si][cfg["frame"]], key)
                _, px = pol.rollout(pol.noise(seeds, G.SHAPE), pol.embed(caption))
                bgrs, reads = rdr.read(px)
                p01 = (px.clamp(-1, 1) + 1) / 2
                for b, seed in enumerate(seeds):
                    case, r = gd.pick(key, reads[b], cfg["conf_min"])
                    ce = None
                    if r is not None:
                        with torch.no_grad():
                            ce = float(
                                rdr.ce(
                                    [p01[b]],
                                    [bgrs[b]],
                                    [gd.crop_box(r["box"], W, H)],
                                    [key],
                                )
                            )
                    f = out / "read" / arm / f"{key}_sc{si:02d}_s{seed}.webp"
                    f.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(bgrs[b][:, :, ::-1]).save(f, quality=95)
                    recs.append(
                        {"arm": arm, "cls": cls, "key": key, "scene": si, "seed": seed}
                        | {"case": case, "ce": ce, "file": str(f)}
                        | {
                            x: G.page_terms(key, reads[b])[x]
                            for x in ("solo", "on_page", "extra", "r_seq")
                        }
                        | {"null": G.page_terms(alt, reads[b])["solo"]}
                        | {"boxes": G.boxes(reads[b])}
                    )
        print(f"  read {arm} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return recs


def sign_p(up: int, down: int) -> float:
    """Two-sided sign test."""
    n, k = up + down, min(up, down)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def tabulate(recs: list, arms: list) -> dict:
    a, b = arms
    keys = list(dict.fromkeys(r["key"] for r in recs))

    def cell(arm, key):
        xs = [r for r in recs if r["arm"] == arm and r["key"] == key]
        crop = [r["ce"] for r in xs if r["ce"] is not None]
        return {
            "n": len(xs),
            "solo": sum(r["solo"] for r in xs),
            "on_page": sum(r["on_page"] for r in xs),
            "null": sum(r["null"] for r in xs),
            "extra": round(sum(r["extra"] for r in xs) / len(xs), 2),
            "r_seq": round(sum(r["r_seq"] for r in xs) / len(xs), 3),
            "none": sum(r["case"] == "none" for r in xs),
            "low": sum(r["case"] == "low" for r in xs),
            "ce": round(sum(crop) / len(crop), 3) if crop else None,
        }

    per = {k: {arm: cell(arm, k) for arm in arms} for k in keys}
    out = {"per_glyph": per, "paired": {}, "total": {}}
    for m in ("solo", "on_page", "null", "extra", "r_seq", "none", "low", "ce"):
        pairs = [
            (per[k][a][m], per[k][b][m])
            for k in keys
            if per[k][a][m] is not None and per[k][b][m] is not None
        ]
        up = sum(y > x for x, y in pairs)
        down = sum(y < x for x, y in pairs)
        out["paired"][m] = {
            a: round(sum(x for x, _ in pairs) / len(pairs), 3),
            b: round(sum(y for _, y in pairs) / len(pairs), 3),
            "up": up,
            "down": down,
            "p": round(sign_p(up, down), 3),
            "glyphs": len(pairs),
        }
    for arm in arms:
        xs = [r for r in recs if r["arm"] == arm]
        out["total"][arm] = {
            "n": len(xs),
            "solo": sum(r["solo"] for r in xs),
            "on_page": sum(r["on_page"] for r in xs),
            "null": sum(r["null"] for r in xs),
            "none": sum(r["case"] == "none" for r in xs),
            **{
                f"solo_{c}": sum(r["solo"] for r in xs if r["cls"] == c)
                for c in CLASSES
            },
        }
    return out


def sheets(recs: list, arms: list, scenes: list, out: Path, seed: int) -> None:
    """Per glyph: rows ``A`` and ``B`` for each pair of scenes, 4 seeds a scene."""
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key_of = {}
    at = {(r["arm"], r["key"], r["scene"], r["seed"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    for key in dict.fromkeys(r["key"] for r in recs):
        order = list(arms)
        rng.shuffle(order)
        key_of[key] = dict(zip("AB", order))
        rows = []
        for i in range(0, len(scenes), 2):
            for name, arm in zip("AB", order):
                for si in scenes[i : i + 2]:
                    for s in seeds:
                        r = at[(arm, key, si, s)]
                        rows.append(
                            (
                                Image.open(r["file"]).convert("RGB"),
                                [
                                    f"{key} {name} sc{si:02d} s{s}",
                                    " / ".join(r["boxes"])[:26],
                                ],
                            )
                        )
        contact_sheet(rows, out / f"{key}.png", thumb=256, cols=2 * len(seeds))
    (out / "key.json").write_text(
        json.dumps(key_of, ensure_ascii=False, indent=1), "utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="two glyphs, two scenes")
    a = ap.parse_args()

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    asset = json.loads(S.ASSET.read_text("utf-8"))["scenes"]
    sc = {s["i"]: s for s in asset}
    tr, rd_sc = list(cfg["train_scenes"]), list(cfg["read_scenes"])
    assert not {sc[i]["artist"] for i in tr} & {sc[i]["artist"] for i in rd_sc}, (
        "a read scene's folder is trained on"
    )
    if a.smoke:
        tr, rd_sc = tr[:2], rd_sc[:1]
        cfg = cfg | {"passes": 1, "accum": 4}
    gl = glyphs(cfg, a.smoke)
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.jsonl").unlink(missing_ok=True)

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    frame = cfg["frame"]
    pol.encode([G.framed(sc[i][frame], k) for _, k, _ in gl for i in tr + rd_sc])
    rdr = gd.Reader("cuda")
    print(
        f"{name}: {len(gl)} glyphs × {len(tr)} training scenes × {cfg['passes']} "
        f"passes; read on {rd_sc}; row_scale {pol.row_scale:.2f}",
        flush=True,
    )
    res = train(pol, rdr, cfg, gl, sc, tr, out)
    print(f"train: {res['info']}", flush=True)
    arms = {"start": res["init"], "trained": res["trained"]}
    recs = read(pol, rdr, cfg, gl, sc, rd_sc, arms, out)
    pol.raw.data.copy_(res["trained"])
    table = tabulate(recs, list(arms))
    for k, v in table["per_glyph"].items():
        s, t = v["start"], v["trained"]
        print(
            f"  {k} solo {s['solo']:>2} → {t['solo']:>2} / {s['n']}  on_page "
            f"{s['on_page']:>2} → {t['on_page']:>2}  null {s['null']} → {t['null']}  "
            f"extra {s['extra']} → {t['extra']}  ce {s['ce']} → {t['ce']}  none "
            f"{s['none']} → {t['none']}  drift {res['info']['drift'][k]}",
            flush=True,
        )
    for m, v in table["paired"].items():
        print(f"  paired {m:<8} {v}", flush=True)
    print(f"  total {table['total']}", flush=True)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"g1-{name}", root=LINE / "results")
    sheets(recs, list(arms), rd_sc, run_dir / "sheets", cfg["shuffle_seed"])
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    steps = [
        json.loads(x) for x in (out / "log.jsonl").read_text("utf-8").splitlines() if x
    ]
    cs = [s["cos_prev"] for s in steps if s["cos_prev"] is not None]
    write_result(
        run_dir,
        script="project/cjk_rl/src/g1.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={
            "train": res["info"]
            | {
                "row_steps": len(steps),
                # successive mean gradients of a row
                "cos_prev_mean": round(sum(cs) / len(cs), 4) if cs else None,
                "cos_prev_n": len(cs),
            },
            "read": table,
        },
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
