#!/usr/bin/env python
"""gd_g0d — G0d of ``plan.md`` item 1: the gradient G-D's path (``src/gd.py``) puts on a
row, measured; no update.

A group is one glyph on one scene (``<scene, bubble frame>. Text reads as "<glyph>".``,
768², ``m15v11``, Euler) at the config's seeds, one forward for the group. Per render:

- the ordinary read picks the crop (``gd.pick``): ``box`` (a read at ``conf_min`` or
  over), ``low`` (boxes, none at ``conf_min``), ``none`` (no box);
- four terms, each on the render alone: ``ce`` (the reader's cross-entropy of the asked
  glyph on the crop), ``ce_alt`` (the same crop, the next glyph of the class as the
  label), ``page`` (pixel MSE to the anchor outside the crop; the anchor is the scene
  with no clause, plain frame, same noise), ``det`` (−log of the detector's top text
  score on the page). ``ce`` and ``ce_alt`` need a crop; ``page`` covers the whole page
  where there is none;
- each term's gradient on the glyph's row, kept per sampling step: the last step alone
  is one backward, all four are the sum over steps.

Tables: the row gradient's size against Adam's ε (rms per element, effective units, as
``src/grpo.py`` trains the rows) by term, depth and case; cosines between seeds,
between scenes, between glyphs (the null), between the asked and the other label, and
of ``det`` and ``page`` to ``ce``; each step's share; seconds by phase and peak VRAM.
Checks of the path: the sampling loop against ``generate_body``, the tensor crop
against the reader's processor, the tensor letterbox against the detector's.

The run's settings, question and gate are its config's header
(``configs/g0d_<mmdd>_<hhmm>.toml``). Result → ``results/<ts>-gd_g0d-<config stem>/``
(``records.json`` a render, ``grads.pt`` the row gradients); pages →
``output/cjk_rl/gd_g0d/<config stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/gd_g0d.py --config project/cjk_rl/configs/<name>.toml"
    .venv/bin/python project/cjk_rl/probes/gd_g0d.py --tabulate project/cjk_rl/results/<run>   # CPU
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

OUT = T.REPO / "output" / "cjk_rl" / "gd_g0d"
P1_LOG = T.REPO / "output" / "cjk_rl" / "grpo" / "grpo_1113_1940" / "log.jsonl"
EPS = 1e-8  # Adam's
TERMS = ("ce", "ce_alt", "page", "det")
CLASSES = ("hira", "kanji")


def groups(cfg: dict, smoke: bool) -> list:
    """``[{cls, key, alt, scene, caption, anchor}]``; ``alt`` is the next glyph of the
    class."""
    sc = {s["i"]: s for s in json.loads(S.ASSET.read_text("utf-8"))["scenes"]}
    out = []
    for si in cfg["scenes"][:1] if smoke else cfg["scenes"]:
        for cls in CLASSES:
            keys = list(cfg[cls])
            for n, key in enumerate(keys):
                if smoke and n not in (0, len(keys) - 1):
                    continue
                out.append(
                    {
                        "cls": cls,
                        "key": key,
                        "alt": keys[(n + 1) % len(keys)],
                        "scene": si,
                        "caption": G.framed(sc[si][cfg["frame"]], key),
                        "anchor": sc[si]["plain"],
                    }
                )
    return out


class Clock:
    def __init__(self):
        self.t: dict = {}

    def lap(self, key: str, t0: float) -> float:
        import torch

        torch.cuda.synchronize()
        self.t.setdefault(key, []).append(time.time() - t0)
        return time.time()


def check_loop(pol, g: dict, seeds: list) -> dict:
    """``Policy.rollout`` against ``generate_body`` + ``pixel_decode`` on one group."""
    import copy

    import torch

    from library.inference.generation import generate_body
    from library.inference.pixel_decoder import ensure_pixel_decoder, pixel_decode

    W, H = G.SHAPE
    e = pol.embed(g["caption"])
    x0 = pol.noise(seeds, G.SHAPE)
    _, px = pol.rollout(x0, e)
    a2 = copy.deepcopy(pol.r.args)
    a2.prompt, a2.seed, a2.image_size = g["caption"], seeds[0], (H, W)
    with torch.no_grad():
        ensure_pixel_decoder(a2, pol.anima, pol.dev)
        generate_body(
            a2,
            pol.anima,
            {"embed": [e.cpu()], "prompt": g["caption"]},
            None,
            pol.dev,
            seeds,
            latents=x0,
        )
        ref = pixel_decode(a2, pol.anima)
    d = (px.clamp(-1, 1).cpu() - ref).abs()
    return {"px_max": float(d.max()), "px_mean": float(d.mean())}


def run_group(pol, rdr, g, seeds, anchor, cfg, ck: Clock, check: dict, deep: bool):
    """``(records, {term: (B, steps, dim) row gradients}, pages)`` for one group.
    ``deep``: the reader's and the detector's tensor paths are also checked against
    their own preprocessing (one more forward each)."""
    import torch

    from common.text import norm

    W, H = G.SHAPE
    B, n = len(seeds), pol.steps
    t = time.time()
    e_graph = pol.embed(g["caption"], grad=True)
    e = e_graph.detach()
    xs, px = pol.rollout(pol.noise(seeds, G.SHAPE), e)
    t = ck.lap("roll", t)
    bgrs, reads = rdr.read(px)
    t = ck.lap("read", t)
    picks = [gd.pick(g["key"], r, cfg["conf_min"]) for r in reads]
    idx = [b for b, (case, _) in enumerate(picks) if case != "none"]
    boxes = {b: gd.crop_box(picks[b][1]["box"], W, H) for b in idx}

    # the terms' cotangents on the image, one graph alive at a time
    torch.cuda.reset_peak_memory_stats()
    img = px.detach()

    def cot_of(fn, sl=slice(None)):
        """``fn(clamped image, the same in [0, 1])`` → a loss a render; its gradient
        on the head's image and its values."""
        leaf = img[sl].clone().requires_grad_(True)
        pc = gd.ste_clamp(leaf, -1.0, 1.0)
        val = fn(pc, (pc + 1) / 2)
        return torch.autograd.grad(val.sum(), leaf)[0], val.detach()

    cots, loss = {}, {}
    for term, text in (("ce", g["key"]), ("ce_alt", g["alt"])):
        if not idx:
            break
        cots[term], ce = cot_of(
            lambda pc, p01: rdr.ce(
                [p01[b] for b in idx],
                [bgrs[b] for b in idx],
                [boxes[b] for b in idx],
                [text] * len(idx),
                check if deep and term == "ce" else None,
            )
        )
        loss[term] = dict(zip(idx, ce.tolist()))
        t = ck.lap("px_ce", t)
    mask = torch.ones(B, 1, H, W, device=px.device)
    for b, (x0, y0, x1, y1) in boxes.items():
        mask[b, :, y0:y1, x0:x1] = 0
    cots["page"], page = cot_of(
        lambda pc, p01: (
            (((pc - anchor) ** 2) * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3)))
        )
    )
    t = ck.lap("px_page", t)
    if deep:
        with torch.no_grad():
            u8 = torch.stack(
                [torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1) for x in bgrs]
            )
            rdr.det_scores(u8.to(px.device).float() / 255, bgrs, check)
        t = time.time()
    tops, gs = [], []
    for b in range(B):  # a page a graph: the detector's activations are the large part
        gb, top_b = cot_of(
            lambda pc, p01: (
                rdr.det_scores(p01).max(1).values.clamp_min(1e-6).log().neg()
            ),
            slice(b, b + 1),
        )
        gs.append(gb)
        tops.append(float(torch.exp(-top_b)))
    cots["det"], top = torch.cat(gs), tops
    t = ck.lap("px_det", t)
    peak_px = torch.cuda.max_memory_allocated() / 2**30

    # each term back through the steps; the embedding's cotangent a step
    torch.cuda.reset_peak_memory_stats()
    row = pol.row_of(g["caption"])
    vecs = {}
    for term, cot in cots.items():
        ges, gx = [None] * n, cot
        for i in reversed(range(n)):
            ges[i], gx = pol.vjp(i, xs[i], e, gx)
            t = ck.lap("vjp_last" if i == n - 1 else "vjp_step", t)
        steps = []
        for i in range(n):
            rg = pol.row_grads(e_graph, ges[i])  # (B, rows, dim)
            if "other_rows" not in check:
                rest = rg.clone()
                rest[:, row] = 0
                check["other_rows"] = float(rest.abs().max())
            steps.append(rg[:, row].cpu())
        vecs[term] = torch.stack(steps, 1)  # (B, steps, dim)
        t = ck.lap("rows", t)
    peak_vjp = torch.cuda.max_memory_allocated() / 2**30

    recs = []
    k = norm(g["key"])
    for b, seed in enumerate(seeds):
        case, r = picks[b]
        rec = {x: g[x] for x in ("cls", "key", "alt", "scene")} | {
            "seed": seed,
            "case": case,
            "boxes": sum(not x.get("whole") for x in reads[b]),
            "page": float(page[b]),
            "det_top": float(top[b]),
            "peak_px": round(peak_px, 2),
            "peak_vjp": round(peak_vjp, 2),
        }
        if r is not None:
            x0, y0, x1, y1 = boxes[b]
            lf = img[b, :, y0:y1, x0:x1]
            sat = lf.abs() > 1
            gm = cots["ce"][b, :, y0:y1, x0:x1].abs()
            rec |= {
                "box": list(boxes[b]),
                "read": r["sfx"],
                "conf": r["sfx_conf"],
                "exact": bool(r["sfx"]) and norm(r["sfx"]) == k,
                "ce": loss["ce"][b],
                "ce_alt": loss["ce_alt"][b],
                "sat": float(sat.float().mean()),  # crop pixels past the clamp
                "sat_grad": float(gm[sat].sum() / gm.sum().clamp_min(1e-30)),
            }
        recs.append(rec)
    return recs, vecs, bgrs


# ---------------------------------------------------------------- tables (CPU)


def med(xs) -> float | None:
    xs = [float(x) for x in xs]
    return round(statistics.median(xs), 4) if xs else None


def summarize(recs: list, V: dict, info: dict) -> dict:
    """``V[term]``: ``(N, steps, dim)`` in the records' order."""
    import torch

    N = len(recs)
    dim = V["ce"].shape[-1]
    has = [r["case"] != "none" for r in recs]
    vec = {
        (t, d): (V[t][:, -1] if d == "last" else V[t].sum(1))
        for t in TERMS
        for d in ("last", "all")
    }

    def cases(term):
        if term in ("ce", "ce_alt"):
            return {
                "exact": [i for i in range(N) if has[i] and recs[i]["exact"]],
                "box other": [
                    i
                    for i in range(N)
                    if recs[i]["case"] == "box" and not recs[i]["exact"]
                ],
                "low": [
                    i
                    for i in range(N)
                    if recs[i]["case"] == "low" and not recs[i]["exact"]
                ],
                "any crop": [i for i in range(N) if has[i]],
            }
        return {
            "crop": [i for i in range(N) if has[i]],
            "none": [i for i in range(N) if not has[i]],
        }

    size = {}
    for (term, d), v in vec.items():
        rms = v.norm(dim=1) / dim**0.5
        for name, ids in cases(term).items():
            if not ids:
                continue
            x = rms[ids]
            size[f"{term} | {d} | {name}"] = {
                "n": len(ids),
                "rms_median": float(x.median()),
                "rms_p10": float(x.quantile(0.1)),
                "decades_over_eps": round(float(torch.log10(x.median() / EPS)), 2),
                "p10_decades": round(float(torch.log10(x.quantile(0.1) / EPS)), 2),
            }

    def cos(a, b) -> float:
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    def stat(xs) -> dict:
        return {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 4) if xs else None,
            "median": med(xs),
        }

    cosine = {}
    for d in ("last", "all"):
        v = vec[("ce", d)]
        pairs = {"seed": [], "scene": [], "glyph (null)": []}
        for i, j in combinations([i for i in range(N) if has[i]], 2):
            a, b = recs[i], recs[j]
            if a["key"] == b["key"]:
                same = a["scene"] == b["scene"]
                pairs["seed" if same else "scene"].append(cos(v[i], v[j]))
            elif (a["cls"], a["scene"], a["seed"]) == (b["cls"], b["scene"], b["seed"]):
                pairs["glyph (null)"].append(cos(v[i], v[j]))
        for name, xs in pairs.items():
            cosine[f"ce {name} | {d}"] = stat(xs)
        ids = cases("ce")
        miss = set(ids["box other"]) | set(ids["low"])
        cosine[f"ce seed, neither exact | {d}"] = stat(
            [
                cos(v[i], v[j])
                for i, j in combinations(sorted(miss), 2)
                if (recs[i]["key"], recs[i]["scene"])
                == (recs[j]["key"], recs[j]["scene"])
            ]
        )
        # a group's mean over its seeds (the renders with a crop that miss the glyph):
        # the same glyph on the other scene, under the asked label and under the other
        alt = vec[("ce_alt", d)]
        mean_of = {}
        for i in sorted(miss):
            mean_of.setdefault((recs[i]["key"], recs[i]["scene"]), []).append(i)
        own, crossed = [], []
        for (key, si), a_ids in mean_of.items():
            for (key2, sj), b_ids in mean_of.items():
                if key2 == key and sj > si:
                    own.append(cos(v[a_ids].mean(0), v[b_ids].mean(0)))
                    crossed.append(cos(v[a_ids].mean(0), alt[b_ids].mean(0)))
        # the part of a render's gradient that the label sets: asked minus other
        dv = v - alt
        lab = {"same glyph": [], "two glyphs of a class (null)": []}
        for i, j in combinations(sorted(miss), 2):
            if recs[i]["key"] == recs[j]["key"]:
                lab["same glyph"].append(cos(dv[i], dv[j]))
            elif recs[i]["cls"] == recs[j]["cls"]:
                lab["two glyphs of a class (null)"].append(cos(dv[i], dv[j]))
        for name, xs in lab.items():
            cosine[f"ce − ce_alt, {name} | {d}"] = stat(xs)
        cosine[f"group mean, other scene: ce · ce | {d}"] = stat(own)
        cosine[f"group mean, other scene: ce · ce_alt | {d}"] = stat(crossed)
        for name in ("any crop", "exact", "box other", "low"):
            for other in ("ce_alt", "det", "page"):
                w = vec[(other, d)]
                cosine[f"ce · {other} | {d} | {name}"] = stat(
                    [cos(v[i], w[i]) for i in ids[name]]
                )
        w = vec[("det", d)]
        none = [i for i in range(N) if not has[i]]
        cosine[f"det seed | {d} | none"] = stat(
            [
                cos(w[i], w[j])
                for i, j in combinations(none, 2)
                if (recs[i]["key"], recs[i]["scene"])
                == (recs[j]["key"], recs[j]["scene"])
            ]
        )

    share = {}
    for term in TERMS:
        ids = cases(term).get("any crop") or list(range(N))
        if term == "det":
            ids = list(range(N))
        x = V[term][ids].norm(dim=2)  # (n, steps)
        tot = V[term][ids].sum(1).norm(dim=1, keepdim=True).clamp_min(1e-30)
        share[term] = [round(float(s), 3) for s in (x / tot).median(0).values]

    crop = [r for r in recs if r["case"] != "none"]
    loss = {
        name: {
            "n": len(xs),
            "ce": med(r["ce"] for r in xs),
            "ce_alt": med(r["ce_alt"] for r in xs),
            "conf": med(r["conf"] or 0 for r in xs),
            "sat": med(r["sat"] for r in xs),
            "sat_grad": med(r["sat_grad"] for r in xs),
        }
        for name, xs in (
            ("exact", [r for r in crop if r["exact"]]),
            ("box other", [r for r in crop if r["case"] == "box" and not r["exact"]]),
            ("low", [r for r in crop if r["case"] == "low" and not r["exact"]]),
        )
    }
    out = {
        "n": N,
        "dim": dim,
        "cases": {c: sum(r["case"] == c for r in recs) for c in ("box", "low", "none")},
        "exact": sum(bool(r.get("exact")) for r in recs),
        "det_top": {
            c: med(r["det_top"] for r in recs if (r["case"] == "none") == (c == "none"))
            for c in ("crop", "none")
        },
        "page_mse": med(r["page"] for r in recs),
        "loss": loss,
        "size": size,
        "cosine": cosine,
        "step_share": share,
    } | info
    if P1_LOG.exists():  # P1's row gradient in the same unit
        L = [json.loads(x) for x in P1_LOG.read_text().splitlines() if x.strip()]
        out["p1_rms_median"] = statistics.median(
            r["gnorm"] / (r["rows"] * dim) ** 0.5 for r in L if r["rows"]
        )
    return out


def show(m: dict) -> None:
    print(f"cases {m['cases']} exact {m['exact']} det_top {m['det_top']}", flush=True)
    for k, v in m["loss"].items():
        print(f"  loss {k:<10} {v}", flush=True)
    for k, v in m["size"].items():
        print(
            f"  size {k:<28} n {v['n']:>3} rms {v['rms_median']:.3e} "
            f"({v['decades_over_eps']:+.2f} decades; p10 {v['p10_decades']:+.2f})",
            flush=True,
        )
    for k, v in m["cosine"].items():
        print(f"  cos  {k:<44} {v}", flush=True)
    print(f"  step share {m['step_share']}", flush=True)
    for k in ("seconds", "peak_gib", "check", "grad_bs", "p1_rms_median"):
        if k in m:
            print(f"  {k}: {m[k]}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="configs/<name>.toml; its stem names the run")
    p.add_argument("--smoke", action="store_true", help="four groups, its own dirs")
    p.add_argument("--tabulate", help="CPU: a result dir's records and gradients")
    a = p.parse_args()

    import torch

    if a.tabulate:
        d = Path(a.tabulate)
        res = json.loads((d / "result.json").read_text("utf-8"))
        old = res["metrics"]
        info = {
            k: old[k] for k in ("seconds", "peak_gib", "check", "grad_bs") if k in old
        }
        m = summarize(
            json.loads((d / "records.json").read_text("utf-8")),
            torch.load(d / "grads.pt"),
            info,
        )
        show(m)
        res["metrics"] = m
        (d / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), "utf-8"
        )
        return

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    label = ("smoke_" if a.smoke else "") + Path(a.config).stem
    seeds = list(cfg["seeds"])
    gs = groups(cfg, a.smoke)
    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=cfg["grad_bs"])
    pol.encode([g["caption"] for g in gs] + [g["anchor"] for g in gs])
    rdr = gd.Reader("cuda")
    print(
        f"{len(gs)} groups × {len(seeds)} seeds; steps {pol.steps} shift "
        f"{pol.r.args.flow_shift} σ {[round(float(s), 4) for s in pol.sig]}; "
        f"rows {tuple(pol.raw.shape)} row_scale {pol.row_scale:.4f}",
        flush=True,
    )
    check = {"loop": check_loop(pol, gs[0], seeds)}
    print(f"check loop: {check['loop']}", flush=True)

    root = OUT / label
    root.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    from library.inference.output import pixels_to_pil

    anchors = {}
    for g in gs:
        if g["scene"] not in anchors:
            _, px = pol.rollout(pol.noise(seeds, G.SHAPE), pol.embed(g["anchor"]))
            anchors[g["scene"]] = px.clamp(-1, 1)
            for b, s in enumerate(seeds):
                pixels_to_pil(px[b]).save(
                    root / f"anchor_sc{g['scene']:02d}_s{s}.webp", quality=95
                )

    ck, recs, V = Clock(), [], {t: [] for t in TERMS}
    dim = pol.raw.shape[1]
    t_run = time.time()
    for n, g in enumerate(gs):
        rs, vecs, bgrs = run_group(
            pol, rdr, g, seeds, anchors[g["scene"]], cfg, ck, check, deep=n < 4
        )
        recs += rs
        for t in TERMS:
            V[t].append(vecs.get(t, torch.zeros(len(seeds), pol.steps, dim)))
        for b, s in enumerate(seeds):
            Image.fromarray(bgrs[b][:, :, ::-1]).save(
                root / f"{g['key']}_sc{g['scene']:02d}_s{s}.webp", quality=95
            )
        print(
            f"  {n + 1} / {len(gs)} {g['key']} sc{g['scene']:02d} "
            f"{[r['case'] + ('*' if r.get('exact') else '') for r in rs]} "
            f"grad_bs {pol.grad_bs} peak {rs[0]['peak_px']} / {rs[0]['peak_vjp']} GiB "
            f"({(time.time() - t_run) / 60:.1f} min)",
            flush=True,
        )
    V = {t: torch.cat(v) for t, v in V.items()}
    info = {
        # a phase's median seconds for a group of len(seeds) renders (the median
        # drops the first group's compile); px_ce is one label's forward and backward
        "seconds": {k: round(statistics.median(v), 3) for k, v in ck.t.items()},
        "peak_gib": {
            "px": max(r["peak_px"] for r in recs),
            "vjp": max(r["peak_vjp"] for r in recs),
            "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        },
        "check": check,
        "grad_bs": pol.grad_bs,
        "row_scale": pol.row_scale,
        "sigmas": [float(s) for s in pol.sig],
        "minutes": round((time.time() - t_run) / 60, 2),
    }
    m = summarize(recs, V, info)
    show(m)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"gd_g0d-{label}", root=T.LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    torch.save(V, run_dir / "grads.pt")
    write_result(
        run_dir,
        script="project/cjk_rl/probes/gd_g0d.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "pages": str(root)},
        label=label,
        metrics=m,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
