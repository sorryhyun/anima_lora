#!/usr/bin/env python
"""Sentence gradient — do a row's renders agree on a direction (no training).

For each row of the config's ``keys``, at the rows ``s1.py`` starts from, two kinds of
group on the same scenes and seeds:

- ``glyph``: the quote holds the row's glyph alone (the glyph stage's caption);
- ``string``: the quote holds a string of the draw that has the glyph (``strings_file``,
  from index ``skip`` on).

A render's loss is ``s1.py``'s (the picked box, the tuned reader's cross-entropy with the
read's breaks, hayai's at ``hayai_weight``, the detector's term on a page with no box).
Its gradient on the row is kept in three parts: through the last step alone (``last``),
through the step before (``prev``, the last step's input cotangent carried back), and
their sum (``sum``, what ``back_steps = 2`` trains on).

Per kind and part, pooled over the rows: the mean cosine between two renders of one row
from different groups (``across``) and from the same group (``within``), the median norm,
the share over ``clip``, and the cosine of ``last`` with ``prev`` on the same render. A
row's step is the mean of ``accum`` renders, so its direction is signal when ``across``
is over 0.

Result → ``results/<ts>-sent_grad-<config stem>/`` (``grads.pt`` holds the vectors).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/sent_grad.py --config project/cjk_rl/configs/<name>.toml"
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
PARTS = ("last", "prev", "sum")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="one row, two groups")
    a = ap.parse_args()
    import torch

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    tr = list(cfg["train_scenes"])
    n_groups = 2 if a.smoke else cfg["groups_a_row"]
    rows = list(cfg["keys"])[: 1 if a.smoke else None]
    pool = json.loads((T.REPO / cfg["strings_file"]).read_text("utf-8"))["train"]
    pool = pool[cfg["skip"] :]
    frame = cfg["frame"]
    jobs = []  # (kind, row glyph, group, scene, quote)
    for k in rows:
        have = [s for s in pool if s1.key(k) in s1.keys(s)][:n_groups]
        assert len(have) == n_groups, k
        for j in range(n_groups):
            jobs.append(("glyph", k, j, tr[j % len(tr)], k))
            jobs.append(("string", k, j, tr[j % len(tr)], have[j]))
    kana = [k for _, k, _ in g1.glyphs(cfg | {"fold_small": False}, False)]

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode([G.framed(sc[si][frame], q) for _, _, _, si, q in jobs])
    rdr = gd.Reader("cuda")
    if cfg.get("reader_compile"):
        rdr.compile()
    hy = gd.Hayai("cuda") if cfg.get("hayai_weight") else None
    at, _ = s1.row_map(pol, sc[tr[0]][frame], kana)
    start = pol.raw.detach().clone()
    if cfg.get("start"):
        start = g1.saved_rows(cfg["start"], {k: at[s1.key(k)] for k in kana}, start)
    pol.raw.data.copy_(start)
    W, H = G.SHAPE
    print(f"{name}: {len(rows)} rows × {n_groups} groups × 2 kinds", flush=True)
    recs, t0 = [], time.time()
    for n, (kind, k, j, si, q) in enumerate(jobs):
        seeds = [cfg["seed0"] + 4 * j + x for x in range(4)]
        caption = G.framed(sc[si][frame], q)
        au = s1.units(q)
        i = at[s1.key(k)]
        e_graph = pol.embed(caption, grad=True)
        e = e_graph.detach()
        xs = pol.inputs(pol.noise(seeds, G.SHAPE), e, 2)
        for b in range(4):
            px, ei, xi = pol.last(xs[-1][b : b + 1], e, wrt_x=True)
            img = px.detach()
            bgrs, reads = rdr.read(img)
            case, r = gd.pick(q, reads[0], cfg["conf_min"])
            leaf = img.clone().requires_grad_(True)
            p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
            hit = [False] * len(au)
            if r is None:
                top = rdr.det_scores(p01).max(1).values
                total = top.clamp_min(1e-6).log().neg().sum()
            else:
                hit, _, target = s1.score(au, r["sfx"] or "")
                box = gd.crop_box(r["box"], W, H)
                total = rdr.ce([p01[0]], bgrs, [box], [target]).sum()
                if hy is not None:
                    total = (
                        total + cfg["hayai_weight"] * hy.ce([p01[0]], [box], [q]).sum()
                    )
            g_px = torch.autograd.grad(total, leaf)[0]
            ge_last, g_x = torch.autograd.grad((px * g_px).sum(), (ei, xi))
            del px, ei, xi
            ge_prev, _ = pol.vjp(pol.steps - 2, xs[-2][b : b + 1], e, g_x)
            parts = {
                "last": pol.row_grads(e_graph, ge_last)[0, i].cpu(),
                "prev": pol.row_grads(e_graph, ge_prev)[0, i].cpu(),
            }
            parts["sum"] = parts["last"] + parts["prev"]
            pos = [p for p, u in enumerate(au) if s1.key(u) == s1.key(k)]
            recs.append(
                {"kind": kind, "key": k, "group": j, "seed": seeds[b], "quote": q}
                | {"case": case, "hit": sum(hit[p] for p in pos), "pos": len(pos)}
                | parts
            )
        del e_graph
        if (n + 1) % 16 == 0:
            print(
                f"  group {n + 1} / {len(jobs)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )

    def unit(v):
        return v / v.norm().clamp_min(1e-30)

    out = {}
    print(
        "| kind | part | across | its spread over rows | within | norm, median | over clip | hit share |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for kind in ("glyph", "string"):
        for part in PARTS:
            across, within, norms = [], [], []
            for k in rows:
                xs = [r for r in recs if r["kind"] == kind and r["key"] == k]
                U = torch.stack([unit(r[part]) for r in xs])
                C = U @ U.T
                g = torch.tensor([r["group"] for r in xs])
                same = g[:, None] == g[None, :]
                off = ~torch.eye(len(xs), dtype=torch.bool)
                across.append(float(C[~same].mean()))
                within.append(float(C[same & off].mean()))
                norms += [float(r[part].norm()) for r in xs]
            xs = [r for r in recs if r["kind"] == kind]
            m = sum(across) / len(across)
            sd = (sum((x - m) ** 2 for x in across) / max(len(across) - 1, 1)) ** 0.5
            cell = {
                "across": round(m, 4),
                "across_se": round(sd / len(across) ** 0.5, 4),
                "across_by_row": dict(zip(rows, (round(x, 4) for x in across))),
                "within": round(sum(within) / len(within), 4),
                "norm_median": round(sorted(norms)[len(norms) // 2], 4),
                "over_clip": round(sum(x > cfg["clip"] for x in norms) / len(norms), 3),
                "hit": round(
                    sum(r["hit"] for r in xs) / max(sum(r["pos"] for r in xs), 1), 3
                ),
            }
            out[f"{kind}/{part}"] = cell
            print(
                f"| {kind} | {part} | {cell['across']} | ± {cell['across_se']} | "
                f"{cell['within']} | {cell['norm_median']} | {cell['over_clip']} | {cell['hit']} |",
                flush=True,
            )
        xs = [r for r in recs if r["kind"] == kind]
        c = [float(torch.dot(unit(r["last"]), unit(r["prev"]))) for r in xs]
        out[f"{kind}/last_prev_cos"] = round(sum(c) / len(c), 4)
        print(
            f"  {kind}: cosine of last with prev on a render {out[f'{kind}/last_prev_cos']}"
        )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"sent_grad-{name}", root=LINE / "results")
    torch.save(recs, run_dir / "grads.pt")
    write_result(
        run_dir,
        script="project/cjk_rl/probes/sent_grad.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL},
        label=name,
        metrics=out,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
