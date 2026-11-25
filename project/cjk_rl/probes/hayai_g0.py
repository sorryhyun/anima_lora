#!/usr/bin/env python
"""G0h — hayai's loss beside the tuned reader's on G1's path (no update).

G1's group as ``src/g1.py`` runs it (start rows, the last step with a graph, one render
a forward, the tuned reader's read and pick). On a render with a crop, two losses of
the asked glyph on the same crop: the tuned reader's cross-entropy (``gd.Reader.ce``)
and hayai's (``gd.Hayai.ce``). Each is pulled to the image and then, through the last
step, to the glyph's row.

Per render: both losses, both gradients' norms on the image and on the row, their
cosine on the image and on the row, whether each reader reads the glyph on the crop.
Tables: the row-norm ratio hayai / tuned (all, and by which reader reads the glyph);
the row cosine of the two terms on one render against the cosine across two renders
of a group (the tuned term of one, hayai's of another) and the tuned term's own across
two renders; the seconds of hayai's term; the tensor path's distance from hayai's
processor.

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Result → ``results/<ts>-hayai_g0-<stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 600 project/cjk_rl/probes/hayai_g0.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tomllib
from itertools import combinations
from pathlib import Path

import glyph_g0 as G
import glyph_scenes as S
import turbo_read as T

import gd  # noqa: E402  (src/, on the path through turbo_read)


def med(xs) -> float | None:
    xs = [float(x) for x in xs]
    return float(f"{statistics.median(xs):.4g}") if xs else None


def cos(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def run(cfg: dict, smoke: bool) -> tuple[list, dict]:
    import numpy as np
    import torch
    from common.readers import crop
    from common.text import norm
    from PIL import Image

    sc = {s["i"]: s for s in json.loads(S.ASSET.read_text("utf-8"))["scenes"]}
    keys = list(cfg["hira"] + cfg["kanji"])
    scenes = list(cfg["scenes"])
    if smoke:
        keys, scenes = keys[:2], scenes[:1]
    seeds = list(cfg["seeds"])
    W, H = G.SHAPE
    cap = {(k, si): G.framed(sc[si][cfg["frame"]], k) for k in keys for si in scenes}

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode(list(cap.values()))
    rdr = gd.Reader("cuda")
    hy = gd.Hayai("cuda")
    check, sec, recs, vec = {}, {}, [], {}

    def lap(name: str, t0: float) -> float:
        torch.cuda.synchronize()
        sec.setdefault(name, []).append(time.time() - t0)
        return time.time()

    t_run = time.time()
    for n, ((key, si), caption) in enumerate(cap.items()):
        e_graph = pol.embed(caption, grad=True)
        e = e_graph.detach()
        row = pol.row_of(caption)
        x3 = pol.head_steps(pol.noise(seeds, G.SHAPE), e)
        for b, seed in enumerate(seeds):
            px, ei = pol.last(x3[b : b + 1], e)
            img = px.detach()
            bgrs, reads = rdr.read(img)
            case, r = gd.pick(key, reads[0], cfg["conf_min"])
            rec = {"key": key, "scene": si, "seed": seed, "case": case}
            if r is not None:
                box = gd.crop_box(r["box"], W, H)
                g_px, loss = {}, {}
                t = time.time()
                for name in ("vl", "hy"):
                    leaf = img.clone().requires_grad_(True)
                    p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
                    if name == "vl":
                        val = rdr.ce([p01[0]], bgrs, [box], [key]).sum()
                    else:
                        val = hy.ce([p01[0]], [box], [key]).sum()
                    g_px[name] = torch.autograd.grad(val, leaf)[0]
                    loss[name] = float(val.detach())
                    t = lap(f"loss_{name}", t)
                with torch.no_grad():  # the path's check, outside the timing
                    hy.ce([(img[0].clamp(-1, 1) + 1) / 2], [box], [key], check)
                g_row = {}
                for name in ("vl", "hy"):
                    g_e = torch.autograd.grad(
                        (px * g_px[name]).sum(), ei, retain_graph=name == "vl"
                    )[0]
                    g_row[name] = pol.row_grads(e_graph, g_e)[0, row].cpu()
                im = Image.fromarray(
                    np.ascontiguousarray(crop(bgrs[0], r["box"])[:, :, ::-1])
                )
                hy_read = hy.read([im])[0]
                k = norm(key)
                rec |= {
                    "box": list(box),
                    "vl_read": r["sfx"],
                    "vl_conf": r["sfx_conf"],
                    "hy_read": hy_read,
                    "vl_exact": norm(r["sfx"] or "") == k,
                    "hy_exact": norm(hy_read) == k,
                    "ce_vl": loss["vl"],
                    "ce_hy": loss["hy"],
                    "px_vl": float(g_px["vl"].norm()),
                    "px_hy": float(g_px["hy"].norm()),
                    "row_vl": float(g_row["vl"].norm()),
                    "row_hy": float(g_row["hy"].norm()),
                    "cos_px": cos(g_px["vl"], g_px["hy"]),
                    "cos_row": cos(g_row["vl"], g_row["hy"]),
                }
                vec[len(recs)] = g_row
            del px, ei
            recs.append(rec)
        del e_graph
        print(
            f"  {n + 1} / {len(cap)} {key} sc{si:02d} "
            f"{[r['case'] for r in recs[-len(seeds) :]]} "
            f"({(time.time() - t_run) / 60:.1f} min)",
            flush=True,
        )
    # cosines across two renders of a group
    cross = {"vl_hy": [], "vl_vl": [], "hy_hy": []}
    for i, j in combinations(sorted(vec), 2):
        a, b = recs[i], recs[j]
        if (a["key"], a["scene"]) != (b["key"], b["scene"]):
            continue
        cross["vl_hy"] += [
            cos(vec[i]["vl"], vec[j]["hy"]),
            cos(vec[j]["vl"], vec[i]["hy"]),
        ]
        cross["vl_vl"].append(cos(vec[i]["vl"], vec[j]["vl"]))
        cross["hy_hy"].append(cos(vec[i]["hy"], vec[j]["hy"]))
    info = {
        "seconds_a_render": {k: round(statistics.median(v), 4) for k, v in sec.items()},
        "check": check,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "minutes": round((time.time() - t_run) / 60, 2),
        "cross": {
            k: {
                "n": len(v),
                "mean": round(sum(v) / len(v), 4) if v else None,
                "median": med(v),
            }
            for k, v in cross.items()
        },
    }
    return recs, info


def summarize(recs: list, info: dict, cfg: dict) -> dict:
    cr = [r for r in recs if r["case"] != "none"]
    w = cfg["weight"]

    def cell(rs: list) -> dict:
        ratio = [r["row_hy"] / max(r["row_vl"], 1e-30) for r in rs]
        return {
            "n": len(rs),
            "ce_vl": med(r["ce_vl"] for r in rs),
            "ce_hy": med(r["ce_hy"] for r in rs),
            "row_vl": med(r["row_vl"] for r in rs),
            "row_hy": med(r["row_hy"] for r in rs),
            "row_ratio": med(ratio),  # hayai / tuned, a render
            "px_ratio": med(r["px_hy"] / max(r["px_vl"], 1e-30) for r in rs),
            # renders where the weighted hayai term is the larger one on the row
            "hy_larger_at_w": sum(w * x > 1 for x in ratio),
            "cos_row": med(r["cos_row"] for r in rs),
            "cos_px": med(r["cos_px"] for r in rs),
        }

    by = {
        "all": cr,
        "both read it": [r for r in cr if r["vl_exact"] and r["hy_exact"]],
        "tuned only": [r for r in cr if r["vl_exact"] and not r["hy_exact"]],
        "hayai only": [r for r in cr if not r["vl_exact"] and r["hy_exact"]],
        "neither": [r for r in cr if not r["vl_exact"] and not r["hy_exact"]],
    }
    return {
        "renders": len(recs),
        "cases": {c: sum(r["case"] == c for r in recs) for c in ("box", "low", "none")},
        "weight": w,
        "by": {k: cell(v) for k, v in by.items()},
        "cos_row_mean": round(sum(r["cos_row"] for r in cr) / max(1, len(cr)), 4),
    } | info


def show(m: dict) -> None:
    print(f"\nrenders {m['renders']} cases {m['cases']} weight {m['weight']}")
    for k, v in m["by"].items():
        print(f"  {k:<13} {v}")
    print(f"  cos_row mean {m['cos_row_mean']}; across renders {m['cross']}")
    print(
        f"  seconds a render {m['seconds_a_render']}; check {m['check']}; "
        f"peak {m['peak_gib']} GiB; {m['minutes']} min",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="its stem names the run")
    p.add_argument("--smoke", action="store_true", help="two glyphs, one scene")
    a = p.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    label = ("smoke_" if a.smoke else "") + Path(a.config).stem
    recs, info = run(cfg, a.smoke)
    m = summarize(recs, info, cfg)
    show(m)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"hayai_g0-{label}", root=T.LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/hayai_g0.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL},
        label=label,
        metrics=m,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
