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

With ``own = true`` a ``string`` render has a second loss, backed the same way into
``own_last`` / ``own_prev`` / ``own_sum``: the cross-entropy of the row's own glyph
positions alone (``s1.own_ce``, both readers; the other glyphs', the line breaks' and the
EOS's tokens left out). A page with no box has no such part.

With ``win = true`` a ``string`` render has a third loss, ``win_last`` / ``win_prev`` /
``win_sum``: for each position of the row's glyph, the cross-entropy of that glyph alone
(its tokens, the EOS left out; both readers) on the window that the position's
cross-attention gives (``src/loc.py``: blocks 13 – 15, the steps σ 0.75 and 0.5, the
peak inside the picked box, a square of ``window`` glyph sides), summed over the
positions. A page with no box has no such part. With ``wall = true`` a fourth,
``wall_*``: the same sum over every glyph of the string that has a row, the row's own
or not (what one backward a render would give every row).

Per kind and part, pooled over the rows: the mean cosine between two renders of one row
from different groups (``across``) and from the same group (``within``), the median norm,
the share over ``clip``, and the cosine of ``last`` with ``prev`` on the same render. A
row's step is the mean of ``accum`` renders, so its direction is signal when ``across``
is over 0. Per own part also its cosine with the whole loss's part on the same render
and the median of its norm over that part's.

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
import loc  # noqa: E402
import s1  # noqa: E402

LINE = Path(__file__).resolve().parents[1]
PARTS = ("last", "prev", "sum")
TAGS = ("own_", "win_", "wall_")  # a string render's other losses, by config key


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
    want_own = bool(cfg.get("own"))
    want_win = bool(cfg.get("win") or cfg.get("wall"))
    EXTRA = tuple(f"{t}{p}" for t in TAGS if cfg.get(t[:-1]) for p in PARTS)
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
    rec = loc.Recorder(pol.anima) if want_win else None
    print(f"{name}: {len(rows)} rows × {n_groups} groups × 2 kinds", flush=True)
    recs, t0 = [], time.time()
    for n, (kind, k, j, si, q) in enumerate(jobs):
        seeds = [cfg["seed0"] + 4 * j + x for x in range(4)]
        caption = G.framed(sc[si][frame], q)
        au = s1.units(q)
        i = at[s1.key(k)]
        e_graph = pol.embed(caption, grad=True)
        e = e_graph.detach()
        mapped = want_win and kind == "string"
        if mapped:  # the quote tokens' maps at the two steps the backward covers
            ak = [s1.key(u) for u in au]
            cols, groups, upos = loc.token_cols(pol, caption, ak, at)
            rec.cols = torch.tensor(cols, device=pol.dev)
        xs = pol.inputs(pol.noise(seeds, G.SHAPE), e, 2)
        m_prev = rec.sets()[:, 0] if mapped else None  # (4, tokens, Nq), σ 0.75
        for b in range(4):
            if mapped:
                rec.cols = torch.tensor(cols, device=pol.dev)
            px, ei, xi = pol.last(xs[-1][b : b + 1], e, wrt_x=True)
            if mapped:
                m = (m_prev[b] + rec.sets()[0, 0]) / 2
                m = torch.stack([m[g].mean(0) for g in groups])  # a glyph a map
                rec.cols = None
            img = px.detach()
            bgrs, reads = rdr.read(img)
            case, r = gd.pick(q, reads[0], cfg["conf_min"])
            leaf = img.clone().requires_grad_(True)
            p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
            hit = [False] * len(au)
            pos = [p for p, u in enumerate(au) if s1.key(u) == s1.key(k)]
            g_more = {}  # tag -> the image's cotangent of another loss
            if r is None:
                top = rdr.det_scores(p01).max(1).values
                total = top.clamp_min(1e-6).log().neg().sum()
            else:
                hit, _, target = s1.score(au, r["sfx"] or "")
                box = gd.crop_box(r["box"], W, H)
                hw = cfg["hayai_weight"] if hy is not None else 0.0
                total = rdr.ce([p01[0]], bgrs, [box], [target]).sum()
                if hy is not None:
                    total = total + hw * hy.ce([p01[0]], [box], [q]).sum()
                if want_own and kind == "string":
                    # a second forward of the readers: their compiled backward runs
                    # once a graph
                    leaf2 = img.clone().requires_grad_(True)
                    q01 = (gd.ste_clamp(leaf2, -1.0, 1.0) + 1) / 2
                    tk, htk = [], []
                    rdr.ce([q01[0]], bgrs, [box], [target], tokens=tk)
                    mine = s1.own_ce(tk[0], s1.glyph_spans(au, target), pos)
                    if hy is not None and mine is not None:
                        hy.ce([q01[0]], [box], [q], tokens=htk)
                        mine = mine + hw * s1.own_ce(htk[0], s1.glyph_spans(au, q), pos)
                    if mine is not None:
                        g_more["own_"] = torch.autograd.grad(mine, leaf2)[0]
                    del tk, htk, mine
                if mapped:
                    rb = [int(v) for v in r["box"]]
                    side = loc.glyph_side(rb, len(au), cfg["window"])
                    cs = loc.centres(m, rb, H // loc.TOK_PX, W // loc.TOK_PX)["raw"]
                    g_at = {}  # a glyph position -> the image's cotangent of its window
                    for p in upos if cfg.get("wall") else pos:
                        # a graph a window, freed at once: the readers' graphs of a
                        # whole string do not fit beside the last step's
                        leaf3 = img.clone().requires_grad_(True)
                        w01 = (gd.ste_clamp(leaf3, -1.0, 1.0) + 1) / 2
                        win = loc.window(cs[upos.index(p)], side, W, H)
                        u, tk, htk = au[p], [], []
                        rdr.ce([w01[0]], bgrs, [win], [u], tokens=tk)
                        seen = s1.own_ce(tk[0], [(0, len(u))], [0])
                        if hy is not None:
                            hy.ce([w01[0]], [win], [u], tokens=htk)
                            seen = seen + hw * s1.own_ce(htk[0], [(0, len(u))], [0])
                        g_at[p] = torch.autograd.grad(seen, leaf3)[0]
                        del seen, tk, htk, w01, leaf3
                    if cfg.get("win"):
                        g_more["win_"] = sum(g_at[p] for p in pos)
                    if cfg.get("wall"):
                        g_more["wall_"] = sum(g_at.values())
                    del g_at
            g_px = torch.autograd.grad(total, leaf)[0]
            ge_last, g_x = torch.autograd.grad((px * g_px).sum(), (ei, xi))
            del px, ei, xi
            lasts = {"": (ge_last, g_x)}
            for tag, g in g_more.items():  # the last step again, a graph each
                lasts[tag] = pol.vjp(pol.steps - 1, xs[-1][b : b + 1], e, g)
            parts = dict.fromkeys(EXTRA) if kind == "string" else {}
            for tag, (ge_last, g_x) in lasts.items():
                ge_prev, _ = pol.vjp(pol.steps - 2, xs[-2][b : b + 1], e, g_x)
                parts[tag + "last"] = pol.row_grads(e_graph, ge_last)[0, i].cpu()
                parts[tag + "prev"] = pol.row_grads(e_graph, ge_prev)[0, i].cpu()
                parts[tag + "sum"] = parts[tag + "last"] + parts[tag + "prev"]
            del lasts
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

    def med(v):
        return round(sorted(v)[len(v) // 2], 4)

    out = {}
    print(
        "| kind | part | renders | across | its spread over rows | within | norm, median | over clip | hit share |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for kind in ("glyph", "string"):
        for part in PARTS + (EXTRA if kind == "string" else ()):
            across, within, norms = [], [], []
            for k in rows:
                xs = [
                    r
                    for r in recs
                    if r["kind"] == kind and r["key"] == k and r[part] is not None
                ]
                U = torch.stack([unit(r[part]) for r in xs])
                C = U @ U.T
                g = torch.tensor([r["group"] for r in xs])
                same = g[:, None] == g[None, :]
                off = ~torch.eye(len(xs), dtype=torch.bool)
                across.append(float(C[~same].mean()))
                within.append(float(C[same & off].mean()))
                norms += [float(r[part].norm()) for r in xs]
            xs = [r for r in recs if r["kind"] == kind and r[part] is not None]
            m = sum(across) / len(across)
            sd = (sum((x - m) ** 2 for x in across) / max(len(across) - 1, 1)) ** 0.5
            cell = {
                "renders": len(xs),
                "across": round(m, 4),
                "across_se": round(sd / len(across) ** 0.5, 4),
                "across_by_row": dict(zip(rows, (round(x, 4) for x in across))),
                "within": round(sum(within) / len(within), 4),
                "norm_median": med(norms),
                "over_clip": round(sum(x > cfg["clip"] for x in norms) / len(norms), 3),
                "hit": round(
                    sum(r["hit"] for r in xs) / max(sum(r["pos"] for r in xs), 1), 3
                ),
            }
            if part in EXTRA:  # against the whole loss's part on the same render
                whole = part.rsplit("_", 1)[1]
                cell["cos_whole"] = round(
                    sum(float(torch.dot(unit(r[part]), unit(r[whole]))) for r in xs)
                    / len(xs),
                    4,
                )
                cell["norm_over_whole"] = med(
                    [
                        float(r[part].norm() / r[whole].norm().clamp_min(1e-30))
                        for r in xs
                    ]
                )
            out[f"{kind}/{part}"] = cell
            print(
                f"| {kind} | {part} | {cell['renders']} | {cell['across']} | ± {cell['across_se']} | "
                f"{cell['within']} | {cell['norm_median']} | {cell['over_clip']} | {cell['hit']} |",
                flush=True,
            )
            if part in EXTRA:
                print(
                    f"  {kind} / {part}: cosine with `{whole}` on a render "
                    f"{cell['cos_whole']}, its norm over that one's {cell['norm_over_whole']}",
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
