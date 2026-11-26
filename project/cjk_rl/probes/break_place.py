#!/usr/bin/env python
"""Break place — does a page's own wrap, used as the label, tell a break's place apart
(no training).

Pages: train strings with no break in the quote, on the training scenes at 4 seeds, as
``s1.py`` renders them. A page is kept when the picked box's read has two lines or more
and every line end of it follows a glyph the read hit: those places are the page's
``true`` places, a place ``h`` being a break after the first ``h`` glyphs.

The page's final latent is the target. It is noised again to the σ of step ``i``
(``steps``; a fresh draw, the same for every caption) and the step's forward gives
x̂₀ = x_σ − σ·v under a caption with a break at ``h``; the loss ``L(h)`` is the mean
squared error to the target inside the picked box. Three spellings of the break:

- ``row``: one row on the T5 side (``stand``'s place in the table, set to the pack's row
  of ``init``, as ``break_grad.py``) and a line feed on the Qwen side;
- ``lf``: the line feed on the Qwen side alone;
- ``lit``: backslash and ``n``, as a dataset caption spells a break.

Forward only, every place of the string, ``draws`` noise draws:

- ``auc``: the share of (true, other) place pairs with ``L(true) < L(other)``;
- ``null``: the same with the true places of a page of another string, scaled to this
  string's length, standing in for the true ones (this page's own left out of the
  comparison): what a place's position in the string alone gives;
- ``auc − null`` ± a jackknife over strings.

With a graph, the ``row`` spelling, one draw: the break row's gradient of ``L`` at a
true place and at two other places. ``d`` = true − other, ``n`` = other − other. Per
step and summed: the mean cosine of ``d`` between pages of different strings
(``across``), ± a jackknife over strings, the same for ``n`` and for the true place's
gradient alone.

Writes ``output/cjk_rl/break_place/<config stem>/pages/`` and
``results/<ts>-break_place-<config stem>/`` (``records.json``, ``grads.pt``).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/break_place.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from pathlib import Path

import glyph_g0 as G
import glyph_scenes as S
import turbo_read as T

import g1  # noqa: E402  (src/, on the path through turbo_read)
import gd  # noqa: E402
import s1  # noqa: E402

LINE = Path(__file__).resolve().parents[1]
OUT = T.REPO / "output" / "cjk_rl" / "break_place"
ARMS = ("row", "lf", "lit")
PAD = 2  # latent cells about the box


def auc(L: dict, true: set, other: list) -> float | None:
    """The share of (true, other) pairs with the true place's loss the lower."""
    pairs = [(L[str(t)], L[str(w)]) for t in true for w in other]
    if not pairs:
        return None
    return sum((a < b) + 0.5 * (a == b) for a, b in pairs) / len(pairs)


def jackknife(vals: list, groups: list) -> tuple:
    """Mean of ``vals`` and its jackknife standard error over ``groups``."""
    m = sum(vals) / len(vals)
    gs = sorted(set(groups))
    if len(gs) < 2:
        return m, float("nan")
    jk = []
    for q in gs:
        v = [x for x, g in zip(vals, groups) if g != q]
        jk.append(sum(v) / len(v))
    mj = sum(jk) / len(jk)
    return m, ((len(jk) - 1) / len(jk) * sum((x - mj) ** 2 for x in jk)) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="six strings")
    a = ap.parse_args()
    import torch
    from PIL import Image

    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.anima.vocab_pack import resolve_active_pack

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    tr = list(cfg["train_scenes"])
    lo, hi = cfg["lengths"]
    pool = json.loads((T.REPO / cfg["strings_file"]).read_text("utf-8"))["train"]
    strings = [s for s in pool[cfg["skip"] :] if lo <= len(s1.units(s)) <= hi]
    strings = strings[: 6 if a.smoke else cfg["strings"]]
    frame, stand = cfg["frame"], cfg["stand"]
    steps, draws = list(cfg["steps"]), int(cfg["draws"])
    kana = [k for _, k, _ in g1.glyphs(cfg | {"fold_small": False}, False)]
    out = OUT / name / "pages"
    out.mkdir(parents=True, exist_ok=True)

    def caps(s: str, si: int, h: int) -> tuple:
        au = s1.units(s)
        x, y = "".join(au[:h]), "".join(au[h:])
        f = sc[si][frame]
        return (
            G.framed(f, x + "\\n" + y),  # a line feed after process_escape
            G.framed(f, x + stand + y),
            G.framed(f, x + "\\\\n" + y),  # backslash and n
        )

    def plain(s: str, si: int) -> str:
        return G.framed(sc[si][frame], s)

    jobs = [(s, tr[n % len(tr)]) for n, s in enumerate(strings)]
    probe = G.framed(sc[tr[0]][frame], "あ" + cfg["init"] + "い")
    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    every = [probe]
    for s, si in jobs:
        every.append(plain(s, si))
        for h in range(1, len(s1.units(s))):
            every += caps(s, si, h)
    t0 = time.time()
    pol.encode(every)
    print(f"{len(set(every))} captions encoded, {time.time() - t0:.0f} s", flush=True)
    rdr = gd.Reader("cuda")
    at, _ = s1.row_map(pol, sc[tr[0]][frame], kana)
    start = pol.raw.detach().clone()
    if cfg.get("start"):
        rows = {k: at[s1.key(k)] for k in kana}
        if cfg["start"].get("marks"):  # an s1 run's save: the marks' rows are trained
            rows |= {m: at[s1.key(m)] for m in s1.MARKS}
        start = g1.saved_rows(cfg["start"], rows, start)
    pol.raw.data.copy_(start)

    def ext(caption) -> list:
        return [
            int(v) - T5_TABLE_SIZE
            for v in pol.hid[caption][2].tolist()
            if v >= T5_TABLE_SIZE
        ]

    # the break's row: ``stand``'s place in the table, the pack's row of ``init`` in it
    s0, si0 = jobs[0]
    (e_st,) = set(ext(caps(s0, si0, 1)[1])) - set(ext(plain(s0, si0)))
    ib = pol.r.delta.index[e_st]
    table = resolve_active_pack(pol.r.args).table.float()
    _, e_in, _ = ext(probe)
    assert e_in not in pol.r.delta.index, "init's row is the table's"
    rs = float(pol.r.delta.row_scale)
    with torch.no_grad():
        pol.raw[ib] = ((table[e_in] - table[e_st]) / rs).to(pol.raw)
    # ``row``: the Qwen stream of the line feed, the T5 ids of the stand-in. A place
    # where the line feed changes the string's own T5 ids (before a dot run) is left out
    valid: dict = {}
    for s, si in jobs:
        base = ext(plain(s, si))
        valid[s] = []
        for h in range(1, len(s1.units(s))):
            lf, sym, _ = caps(s, si, h)
            es = ext(sym)
            if ext(lf) != base or len(es) != len(base) + 1:
                continue
            k = es.index(e_st)
            if es[:k] + es[k + 1 :] != base:
                continue
            valid[s].append(h)
            pol.hid[("row", s, si, h)] = pol.hid[lf][:2] + pol.hid[sym][2:]
    left = sum(len(s1.units(s)) - 1 - len(v) for s, v in valid.items())
    print(f"places left out for their T5 ids: {left}", flush=True)

    def key(arm: str, s: str, si: int, h: int):
        lf, _, lit = caps(s, si, h)
        return {"row": ("row", s, si, h), "lf": lf, "lit": lit}[arm]

    W, H = G.SHAPE
    sig = [float(v) for v in pol.sig]
    print(
        f"{name}: {len(jobs)} strings × 4 seeds; steps {steps} (σ "
        f"{[round(sig[i], 3) for i in steps]}), {draws} draws; the break is ext {e_st} "
        f"({stand}) at the pack's {e_in} ({cfg['init']!r})",
        flush=True,
    )

    def loss_of(i: int, x, e, tgt, mask):
        v = pol.forward(i, x, e)
        x0 = x.float() - sig[i] * v.float()
        d = ((x0 - tgt) ** 2 * mask).flatten(1).sum(1)
        return d / (mask.sum() * tgt.shape[1])

    recs, grads, t0 = [], [], time.time()
    for n, (s, si) in enumerate(jobs):
        au = s1.units(s)
        ak = [s1.key(u) for u in au]
        nu = len(au)
        seeds = [cfg["seed0"] + 4 * n + x for x in range(4)]
        e = pol.embed(plain(s, si))
        x = pol.noise(seeds, G.SHAPE)
        with torch.no_grad():
            for i in range(pol.steps):
                v = pol.forward(i, x, e)
                if i < pol.steps - 1:
                    x = pol.euler(i, x, v)
            px = pol.decode(x)
            x_end = pol.euler(pol.steps - 1, x, v)
        pol.feat = None
        bgrs, reads = rdr.read(px)
        for b, seed in enumerate(seeds):
            case, r = gd.pick(s, reads[b], cfg["conf_min"])
            row = {"string": s, "group": n, "scene": si, "seed": seed, "case": case}
            stem = f"{n:02d}_sc{si:03d}_s{seed}"
            Image.fromarray(bgrs[b][:, :, ::-1]).save(out / f"{stem}.webp", quality=85)
            if r is None:
                recs.append(row | {"lines": 0, "true": [], "kept": 0})
                continue
            read = r["sfx"] or ""
            ru, br = s1.read_units(read)
            pairs = s1.align(ak, [s1.key(u) for u in ru])
            true = sorted({i + 1 for i, j in pairs if j in br and i + 1 < nu})
            places = valid[s]
            kept = int(
                case == "box"
                and bool(br)
                and len(true) == len(br)
                and all(h in places for h in true)
                and len(pairs) * 2 >= nu
                and len(places) - len(true) >= 2
            )
            row |= {
                "read": read,
                "lines": len(br) + 1,
                "hit": len(pairs) / nu,
                "true": true,
                "kept": kept,
                "glyphs": nu,
                "box": [int(v) for v in r["box"]],
            }
            if not kept:
                recs.append(row)
                continue
            tgt = x_end[b : b + 1].float()
            x0, y0, x1, y1 = (int(v) for v in gd.crop_box(r["box"], W, H))
            mask = torch.zeros_like(tgt[:, :1])
            mask[
                ...,
                max(0, y0 // 8 - PAD) : y1 // 8 + 1 + PAD,
                max(0, x0 // 8 - PAD) : x1 // 8 + 1 + PAD,
            ] = 1.0
            gen = torch.Generator(device="cpu")
            gen.manual_seed(100_000 + seed)
            eps = torch.randn((draws, *tgt.shape[1:]), generator=gen).to(tgt)
            L = {arm: {} for arm in ARMS}
            L0 = {}
            with torch.no_grad():
                embs = {
                    (arm, h): pol.embed(key(arm, s, si, h))
                    for arm in ARMS
                    for h in places
                }
                embs[("plain", 0)] = e
                order = list(embs)
                for i in steps:
                    acc = {k: 0.0 for k in order}
                    for m in range(draws):
                        xs = ((1 - sig[i]) * tgt + sig[i] * eps[m : m + 1]).to(x.dtype)
                        for c in range(0, len(order), 4):
                            ks = order[c : c + 4]
                            ks4 = ks + [ks[-1]] * (4 - len(ks))
                            eb = torch.cat([embs[k] for k in ks4])
                            ls = loss_of(i, xs.expand(4, *xs.shape[1:]), eb, tgt, mask)
                            for k, v in zip(ks, ls.tolist()):
                                acc[k] += v / draws
                    for arm in ARMS:
                        L[arm][i] = {str(h): acc[(arm, h)] for h in places}
                    L0[i] = acc[("plain", 0)]
            pol.feat = None
            del embs
            other = [h for h in places if h not in true]
            far = [h for h in other if min(abs(h - t) for t in true) >= 2]
            rng = random.Random(seed)
            w = rng.sample(far, 2) if len(far) >= 2 else rng.sample(other, 2)
            g = {"group": n, "seed": seed}
            for i in steps:
                xs = ((1 - sig[i]) * tgt + sig[i] * eps[:1]).to(x.dtype)
                for tag, h in (("t", true[0]), ("w1", w[0]), ("w2", w[1])):
                    e_graph = pol.embed(("row", s, si, h), grad=True)
                    ei = e_graph.detach().float().clone().requires_grad_(True)
                    with torch.enable_grad():
                        loss = loss_of(i, xs, ei, tgt, mask).sum()
                        (ge,) = torch.autograd.grad(loss, ei)
                    pol.feat = None
                    g[f"{tag}_{i}"] = pol.row_grads(e_graph, ge)[0, ib].cpu()
                    del e_graph, ei, loss, ge
            grads.append(g)
            recs.append(
                row
                | {
                    "wrong": w,
                    "L": {arm: {str(i): L[arm][i] for i in steps} for arm in ARMS},
                    "L_plain": {str(i): L0[i] for i in steps},
                }
            )
        if (n + 1) % 4 == 0:
            print(
                f"  string {n + 1} / {len(jobs)}, kept {sum(r['kept'] for r in recs)} "
                f"of {len(recs)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )

    kept = [r for r in recs if r["kept"]]
    res = {
        "renders": len(recs),
        "box": sum(r["case"] == "box" for r in recs),
        "two_lines": sum(r["lines"] > 1 for r in recs),
        "kept": len(kept),
        "kept_strings": len({r["group"] for r in kept}),
    }
    print({k: v for k, v in res.items()}, flush=True)

    def null_auc(p: dict, Lp: dict) -> float | None:
        nu, true = p["glyphs"], set(p["true"])
        places = [int(h) for h in Lp]
        vals = []
        for q in kept:
            if q["group"] == p["group"]:
                continue
            fake = {min(nu - 1, max(1, round(h / q["glyphs"] * nu))) for h in q["true"]}
            if fake & true or not fake <= set(places):
                continue
            rest = [h for h in places if h not in fake and h not in true]
            v = auc(Lp, fake, rest)
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    print("| spelling | step (σ) | pages | auc | null | auc − null | ± | gap / plain |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for i in steps + ["mean"]:
            rows = []
            for p in kept:
                ii = steps if i == "mean" else [i]
                au_, nu_, gap = [], [], []
                for j in ii:
                    Lp = p["L"][arm][str(j)]
                    oth = [int(h) for h in Lp if int(h) not in p["true"]]
                    au_.append(auc(Lp, set(p["true"]), oth))
                    nu_.append(null_auc(p, Lp))
                    lt = sum(Lp[str(t)] for t in p["true"]) / len(p["true"])
                    lw = sum(Lp[str(h)] for h in oth) / len(oth)
                    gap.append((lw - lt) / p["L_plain"][str(j)])
                if any(v is None for v in au_ + nu_):
                    continue
                rows.append(
                    (
                        p["group"],
                        sum(au_) / len(au_),
                        sum(nu_) / len(nu_),
                        sum(gap) / len(gap),
                    )
                )
            if not rows:
                continue
            gsr = [r[0] for r in rows]
            d, se = jackknife([r[1] - r[2] for r in rows], gsr)
            cell = {
                "pages": len(rows),
                "auc": round(sum(r[1] for r in rows) / len(rows), 4),
                "null": round(sum(r[2] for r in rows) / len(rows), 4),
                "diff": round(d, 4),
                "se": round(se, 4),
                "gap": round(sum(r[3] for r in rows) / len(rows), 5),
            }
            res[f"score/{arm}/{i}"] = cell
            lab = i if i == "mean" else f"{i} ({sig[i]:.2f})"
            print(
                f"| {arm} | {lab} | {cell['pages']} | {cell['auc']} | {cell['null']} "
                f"| {cell['diff']} | ± {cell['se']} | {cell['gap']} |",
                flush=True,
            )

    def unit(v):
        return v / v.norm().clamp_min(1e-30)

    if len(grads) >= 3:
        gq = torch.tensor([g["group"] for g in grads])
        same = gq[:, None] == gq[None, :]
        print("| vector | step (σ) | pages | across | ± | norm, median |")
        print("|---|---|---|---|---|---|")
        for i in steps + ["sum"]:
            ii = steps if i == "sum" else [i]

            def vec(g, tag):
                return sum(g[f"{tag}_{j}"] for j in ii)

            kinds = {
                "true − other": [vec(g, "t") - vec(g, "w1") for g in grads],
                "other − other": [vec(g, "w2") - vec(g, "w1") for g in grads],
                "true alone": [vec(g, "t") for g in grads],
            }
            for kind, vs in kinds.items():
                U = torch.stack([unit(v) for v in vs])
                C = U @ U.T
                full = float(C[~same].mean())
                jk = []
                for q in gq.unique():
                    m = (gq != q)[:, None] & (gq != q)[None, :] & ~same
                    jk.append(float(C[m].mean()))
                mj = sum(jk) / len(jk)
                se = ((len(jk) - 1) / len(jk) * sum((x - mj) ** 2 for x in jk)) ** 0.5
                norms = sorted(float(v.norm()) for v in vs)
                cell = {
                    "across": round(full, 4),
                    "se": round(se, 4),
                    "norm_median": norms[len(norms) // 2],
                }
                res[f"grad/{kind}/{i}"] = cell
                lab = i if i == "sum" else f"{i} ({sig[i]:.2f})"
                print(
                    f"| {kind} | {lab} | {len(vs)} | {cell['across']} | ± {cell['se']} "
                    f"| {cell['norm_median']:.3e} |",
                    flush=True,
                )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"break_place-{name}", root=LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    torch.save(grads, run_dir / "grads.pt")
    write_result(
        run_dir,
        script="project/cjk_rl/probes/break_place.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics=res,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
