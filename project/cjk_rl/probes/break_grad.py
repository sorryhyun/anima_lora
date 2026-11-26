#!/usr/bin/env python
"""Break gradient — does the reader's line break give a break row a direction (no
training).

A quote ``<first n // 2 glyphs><break><the rest>`` of a train string, on the training
scenes at 4 seeds. The break is one row on the T5 side and a line feed on the Qwen side
(what a pack fold of the line feed onto a row would give): the caption is encoded with a
line feed for the Qwen stream and with ``stand`` (a glyph whose row the table holds and
no string asks for) for the T5 ids, and ``stand``'s row is set to the pack's own row of
``init`` (a character the pack has a row for and the table does not), so the break
starts as an untrained pack row; with ``init_t5`` it is set to that id's row of the
adapter's own T5 table instead (3: the space piece).

A render's losses, on the box picked for the string (``gd.pick``), with the target the
string with a line break at the asked place:

- ``brk``: the tuned reader's cross-entropy of the target's line-break token alone;
- ``str``: its mean over the target's tokens and the EOS.

Each is backed through all four steps, one step's graph at a time (``gd.Policy.vjp``),
and the break row's gradient kept a step: ``s3`` (σ 0.5 → 0), ``s2``, ``s1``, ``s0``,
with ``last2`` = s3 + s2 (what ``back_steps = 2`` trains on) and ``all``.

Per loss and part: the mean cosine between two renders of different strings (``across``;
the break row is every string's, so this is what a step of it averages) with a jackknife
standard error over strings, between two seeds of one string (``within``), and the
median norm. Beside them: the pages whose read breaks after the asked glyph, and the
break token's cross-entropy.

Result → ``results/<ts>-break_grad-<config stem>/`` (``grads.pt`` holds the vectors).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/break_grad.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
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
STEPS = ("s3", "s2", "s1", "s0")  # the last step first
PARTS = STEPS + ("last2", "all")
LOSSES = ("brk", "str")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="two strings")
    a = ap.parse_args()
    import torch

    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.anima.vocab_pack import resolve_active_pack

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    tr = list(cfg["train_scenes"])
    lo, hi = cfg["lengths"]
    pool = json.loads((T.REPO / cfg["strings_file"]).read_text("utf-8"))["train"]
    strings = [s for s in pool[cfg["skip"] :] if lo <= len(s1.units(s)) <= hi]
    strings = strings[: 2 if a.smoke else cfg["strings"]]
    frame, stand = cfg["frame"], cfg["stand"]
    kana = [k for _, k, _ in g1.glyphs(cfg | {"fold_small": False}, False)]

    def caps(s: str, si: int) -> tuple:
        au = s1.units(s)
        h = len(au) // 2
        x, y = "".join(au[:h]), "".join(au[h:])
        f = sc[si][frame]
        return G.framed(f, x + "\\n" + y), G.framed(f, x + stand + y), G.framed(f, s)

    jobs = [(s, tr[n % len(tr)]) for n, s in enumerate(strings)]
    t5_id = cfg.get("init_t5")
    probe = G.framed(sc[tr[0]][frame], "あ" + (stand if t5_id else cfg["init"]) + "い")
    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode([c for j in jobs for c in caps(*j)] + [probe])
    rdr = gd.Reader("cuda")
    at, _ = s1.row_map(pol, sc[tr[0]][frame], kana)
    start = pol.raw.detach().clone()
    if cfg.get("start"):
        start = g1.saved_rows(cfg["start"], {k: at[s1.key(k)] for k in kana}, start)
    pol.raw.data.copy_(start)

    def ext(caption: str) -> list:
        return [
            int(v) - T5_TABLE_SIZE
            for v in pol.hid[caption][2].tolist()
            if v >= T5_TABLE_SIZE
        ]

    # the break's row: ``stand``'s place in the table, the pack's row of ``init`` in it
    lf0, sym0, plain0 = caps(*jobs[0])
    (e_st,) = set(ext(sym0)) - set(ext(plain0))
    ib = pol.r.delta.index[e_st]
    table = resolve_active_pack(pol.r.args).table.float()
    if t5_id:
        row0 = pol.anima.llm_adapter.embed.weight[t5_id].detach().float().cpu()
        what = f"the T5 table's id {t5_id}"
    else:
        _, e_in, _ = ext(probe)
        assert e_in not in pol.r.delta.index, "init's row is the table's"
        row0, what = table[e_in], f"the pack's {e_in} ({cfg['init']!r})"
    rs = float(pol.r.delta.row_scale)
    with torch.no_grad():
        pol.raw[ib] = ((row0 - table[e_st]) / rs).to(pol.raw)
    for s, si in jobs:  # the Qwen stream of the line feed, the T5 ids of the stand-in
        lf, sym, plain = caps(s, si)
        assert ext(lf) == ext(plain) and len(ext(sym)) == len(ext(plain)) + 1
        pol.hid[lf] = pol.hid[lf][:2] + pol.hid[sym][2:]
    W, H = G.SHAPE
    print(
        f"{name}: {len(jobs)} strings × 4 seeds; the break is ext {e_st} ({stand}) at "
        f"{what}, norm {float(row0.norm()):.1f}",
        flush=True,
    )
    recs, t0 = [], time.time()
    for n, (s, si) in enumerate(jobs):
        seeds = [cfg["seed0"] + 4 * n + x for x in range(4)]
        key = caps(s, si)[0]
        au = s1.units(s)
        ak = [s1.key(u) for u in au]
        h = len(au) // 2
        target = "".join(au[:h]) + "\n" + "".join(au[h:])
        c = len("".join(au[:h]))  # the break's place in the target
        e_graph = pol.embed(key, grad=True)
        e = e_graph.detach()
        xs = pol.inputs(pol.noise(seeds, G.SHAPE), e, pol.steps)
        for b in range(4):
            px, ei, xi = pol.last(xs[-1][b : b + 1], e, wrt_x=True)
            img = px.detach()
            bgrs, reads = rdr.read(img)
            case, r = gd.pick(s, reads[0], cfg["conf_min"])
            row = {"string": s, "group": n, "seed": seeds[b], "case": case}
            if r is None:
                del px, ei, xi
                recs.append(row | {"at": 0, "lines": 0, "hit": 0, "ce": None})
                continue
            read = r["sfx"] or ""
            ru, br = s1.read_units(read)
            pairs = s1.align(ak, [s1.key(u) for u in ru])
            row |= {
                "at": int(h - 1 in {i for i, j in pairs if j in br}),
                "lines": len(br) + 1,
                "hit": len(pairs) / len(au),
            }
            leaf = img.clone().requires_grad_(True)
            p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
            tk: list = []
            rdr.ce([p01[0]], bgrs, [gd.crop_box(r["box"], W, H)], [target], tokens=tk)
            each, off = tk[0]
            keep = [j for j, (x, y) in enumerate(off or []) if x < c + 1 and y > c]
            if not keep:  # the tokens' spans unknown
                del px, ei, xi, tk, each
                recs.append(row | {"ce": None})
                continue
            brk = each[keep].sum()
            g_px = {
                "brk": torch.autograd.grad(brk, leaf, retain_graph=True)[0],
                "str": torch.autograd.grad(each.mean(), leaf)[0],
            }
            row["ce"] = float(brk.detach())
            del tk, each, brk
            for k, loss in enumerate(LOSSES):
                if k:  # the last step again, with a graph of its own
                    ge, gx = pol.vjp(pol.steps - 1, xs[-1][b : b + 1], e, g_px[loss])
                else:
                    ge, gx = torch.autograd.grad((px * g_px[loss]).sum(), (ei, xi))
                    del px, ei, xi
                gs = [pol.row_grads(e_graph, ge)[0, ib].cpu()]
                for st in range(2, pol.steps + 1):
                    ge, gx = pol.vjp(pol.steps - st, xs[-st][b : b + 1], e, gx)
                    gs.append(pol.row_grads(e_graph, ge)[0, ib].cpu())
                row |= {f"{loss}_{p}": g for p, g in zip(STEPS, gs)}
                row[f"{loss}_last2"] = gs[0] + gs[1]
                row[f"{loss}_all"] = sum(gs)
            recs.append(row)
        del e_graph
        if (n + 1) % 4 == 0:
            print(
                f"  string {n + 1} / {len(jobs)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )

    def unit(v):
        return v / v.norm().clamp_min(1e-30)

    xs = [r for r in recs if r.get("ce") is not None]
    g = torch.tensor([r["group"] for r in xs])
    same = g[:, None] == g[None, :]
    off_d = ~torch.eye(len(xs), dtype=torch.bool)
    out = {
        "renders": len(recs),
        "backed": len(xs),
        "box": sum(r["case"] == "box" for r in recs),
        "at": sum(r["at"] for r in recs),
        "two_lines": sum(r["lines"] > 1 for r in recs),
        "ce_median": sorted(r["ce"] for r in xs)[len(xs) // 2],
    }
    print(
        "| loss | part | renders | across | ± (jackknife over strings) | within | norm, median |"
    )
    print("|---|---|---|---|---|---|---|")
    for loss in LOSSES:
        for part in PARTS:
            U = torch.stack([unit(r[f"{loss}_{part}"]) for r in xs])
            C = U @ U.T
            full = float(C[~same].mean())
            jk = []
            for q in g.unique():
                m = (g != q)[:, None] & (g != q)[None, :] & ~same
                jk.append(float(C[m].mean()))
            mj = sum(jk) / len(jk)
            se = ((len(jk) - 1) / len(jk) * sum((x - mj) ** 2 for x in jk)) ** 0.5
            norms = sorted(float(r[f"{loss}_{part}"].norm()) for r in xs)
            cell = {
                "across": round(full, 4),
                "se": round(se, 4),
                "within": round(float(C[same & off_d].mean()), 4),
                "norm_median": round(norms[len(norms) // 2], 4),
            }
            out[f"{loss}/{part}"] = cell
            print(
                f"| {loss} | {part} | {len(xs)} | {cell['across']} | ± {cell['se']} | "
                f"{cell['within']} | {cell['norm_median']} |",
                flush=True,
            )
    for part in ("last2", "all"):
        c = [
            float(torch.dot(unit(r[f"brk_{part}"]), unit(r[f"str_{part}"]))) for r in xs
        ]
        out[f"brk_str_cos/{part}"] = round(sum(c) / len(c), 4)
    print({k: v for k, v in out.items() if not isinstance(v, dict)}, flush=True)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"break_grad-{name}", root=LINE / "results")
    torch.save(recs, run_dir / "grads.pt")
    write_result(
        run_dir,
        script="project/cjk_rl/probes/break_grad.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL},
        label=name,
        metrics=out,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
