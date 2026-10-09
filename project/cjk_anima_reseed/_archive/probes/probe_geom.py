#!/usr/bin/env python
"""probe_geom — what each loss term and AdamW do to a row, in the row's own
coordinates (2026-10-06)

The question (user, 10-06, after ``structure_candidate.md``): does the
box-share loss move a row's norm or turn it where no glyph asks for it, and
is the turn ``rand_turn`` priced (``reports/sent_ball_2026_10_05.md`` § 6)
the loss's or the optimizer's? No training: the rows sit still and the
gradient is read.

- ``grad`` (GPU): the trainer's setup for a run (``--run``, its data dir and
  table) with the rows at ``--rows`` (a merged ``trained.pt``), the run's
  first ``--batches`` batches in its own order (SEED 0, the trainer's
  ``Batcher``); per batch one forward, the in-box term ``mean(s · mean_in)``
  and the out-box term ``mean((1 − s) · mean_out)`` of ``box_share_fm_loss``
  taken apart and each one's gradient on the rows read
  (``autograd.grad``, eager — no compile, no step) →
  ``output/cjk_anima_reseed/probe_geom/<label>/grads.pt``: per batch the
  live rows it touches and their two gradients.
- ``read`` (CPU): each row's gradient in ``structure_candidate.md``'s terms
  — stick ``s``, init ``q_i``, glyph ``e_i``, the rest (a new direction) and
  the row's own radius — signal against noise by split halves of its
  draws, and the gradients replayed through AdamW's recursion (the
  trainer's betas, f0's lr and kana ``row_lr``) against plain SGD →
  ``…/<label>/read.json``. A linearised replay: the gradients are read at
  the start rows, so it says which way the first steps go, not where the
  run ends.

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_geom.py grad --label s080"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_geom.py read --label s080
    .venv/bin/python project/cjk_anima_reseed/probes/probe_geom.py tiers --label s080

``tiers`` (CPU): the draws split by the item that holds the glyph — σ, tier,
and bubbleN against sent per row → ``…/<label>/tiers.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()

from reseed import OUT  # noqa: E402

PROBE = OUT / "probe_geom"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"


def split_terms(pred, target, recs, s1, s_cap, n_cap, grid_box):
    """``box_share_fm_loss`` as its two summands: (in-box, out-box); their
    sum is the loss the trainer backpropagates."""
    import torch
    from cjk_scale.loss import box_mask, box_share_of, glyph_count

    se = (pred.float() - target.float()) ** 2
    m = box_mask(se.shape, recs, se.device, grid_box)
    per_cell = se.mean(dim=1, keepdim=True)
    n_in = m.sum(dim=(1, 2, 3))
    n_out = (1.0 - m).sum(dim=(1, 2, 3))
    mean_in = (per_cell * m).sum(dim=(1, 2, 3)) / n_in.clamp(min=1)
    mean_out = (per_cell * (1.0 - m)).sum(dim=(1, 2, 3)) / n_out.clamp(min=1)
    s = torch.tensor(
        [box_share_of(glyph_count(r["text"]), s1, s_cap, n_cap) for r in recs],
        device=se.device,
        dtype=se.dtype,
    )
    s = torch.where(n_in > 0, s, torch.zeros_like(s))
    s = torch.where(n_out > 0, s, torch.ones_like(s))
    return (s * mean_in).mean(), ((1.0 - s) * mean_out).mean()


def grad(run_name: str, rows_path: str, batches: int, label: str) -> None:
    import os
    import time
    from types import SimpleNamespace

    import torch
    from cjk_scale import train as T
    from common.models import checkpoints, dit_forward, gen_args
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.anima.vocab_pack import strategy_pack
    from reseed import REPO
    from reseed.config import load
    from train.stage import Batcher, LatentStore, _encode_text

    from cjk_scale.rows import Rows

    run = load(run_name)
    run.use_pack()
    rows_path = str(rows_path if Path(rows_path).is_absolute() else REPO / rows_path)
    assert Path(rows_path).is_file(), rows_path
    out = PROBE / label
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(T.SEED)
    data = run.data
    recs, ev, vocabs = T.load_items(data)
    keep = list(range(len(recs)))
    assert all(r["src"] == "scene" for r in recs), "a scene table: the box share is on"
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    device = get_generation_settings(args).device
    cache, touched, _ = _encode_text(recs, ev, device, out, te_cache=data / "te_cache")
    rc = run.scale_config()
    p = T.plan(rc, data, recs, vocabs, touched, Path(rows_path))
    ns = SimpleNamespace(seed=T.SEED, batch=T.BATCH, train_size=512)
    lat = LatentStore(ns, data, recs, keep, device)
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    rows = Rows(
        anima,
        device,
        p.idx,
        pack,
        warm=Path(rows_path),
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
        touched=p.touched,
        frozen=p.frozen,
        context=Path(rows_path),
    )
    live = ~rows.frozen_mask
    anima.train()
    batcher = Batcher(ns, recs, lat)
    raw = rows.delta.raw
    got = []
    t0 = time.time()
    for step in range(1, batches + 1):
        idx = batcher.next(step)
        latents = lat[idx].to(device)
        noise = torch.randn_like(latents)
        brecs = [recs[i] for i in idx]
        noisy, ts, target = T.noisy_by_band(
            latents, noise, [tuple(r["band"]) for r in brecs], device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dit_forward(
                anima, noisy, ts, cache, [r["caption"] for r in brecs], device
            )
        l_in, l_out = split_terms(
            pred,
            target,
            brecs,
            T.BOX_SHARE,
            T.BOX_SHARE_CAP,
            T.BOX_SHARE_GLYPHS,
            T.GRID_BOX,
        )
        g_in = torch.autograd.grad(l_in, raw, retain_graph=True)[0]
        g_out = torch.autograd.grad(l_out, raw)[0]
        nz = (
            (live & ((g_in.abs().sum(1) + g_out.abs().sum(1)) > 0)).nonzero().squeeze(1)
        )
        got.append(
            {
                "step": step,
                "items": [int(i) for i in idx],
                "sigma": [float(t) for t in ts.flatten()],
                "rows": nz.cpu(),
                "g_in": g_in[nz].float().cpu(),
                "g_out": g_out[nz].float().cpu(),
                "l_in": float(l_in.detach()),
                "l_out": float(l_out.detach()),
            }
        )
        if step % 50 == 0 or step == 1:
            print(
                f"step {step}/{batches}: {len(nz)} rows, l_in {float(l_in.detach()):.4f} "
                f"l_out {float(l_out.detach()):.4f}, {(time.time() - t0) / step:.2f} s/batch",
                flush=True,
            )
    st = rows.delta.state_dict()
    torch.save(
        {
            "run": run_name,
            "rows": rows_path,
            "ext_ids": [int(e) for e in st["ext_ids"]],
            "live": live.cpu(),
            "raw": st["raw"].float().cpu(),
            "row_scale": rows.row_scale,
            "pack_rows": rows.pack_rows.cpu(),
            "batches": got,
        },
        out / "grads.pt",
    )
    print(f"→ {out / 'grads.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


K_PC = 64  # the glyph subspace: the family's top e_i PCs (isotropic share 64 / 1024)
F0 = "output/cjk_anima_reseed/sent_kanji_f0/trained.pt"


def _moved(ids, off, end_rows):
    """Per row: the end file's offset less the probe's start offset
    (effective units) — what the trained run actually did."""
    import numpy as np
    import torch
    from reseed import REPO

    p = Path(end_rows) if Path(end_rows).is_absolute() else REPO / end_rows
    d = torch.load(p, map_location="cpu", weights_only=False)["delta"]
    end = dict(
        zip(
            [int(e) for e in d["ext_ids"]],
            (d["raw"].float() * float(d["row_scale"])).numpy(),
        )
    )
    return {
        k: (end[e] - off[k]) if e in end else np.zeros(off.shape[1])
        for k, e in enumerate(ids)
    }


def read(label: str, lr: float, kana_scale: float, end_rows: str = F0) -> None:
    import numpy as np
    import torch

    sd = torch.load(PROBE / label / "grads.pt", map_location="cpu", weights_only=False)
    ids = sd["ext_ids"]
    rs = float(sd["row_scale"])
    off = sd["raw"].numpy() * rs  # offsets, effective units
    pack = sd["pack_rows"].numpy()
    live = sd["live"].numpy()
    fam = _families(ids)
    n_all, dim = off.shape

    # the decomposition's basis per family (structure_candidate.md)
    basis = {}
    for f, members in fam.items():
        m = np.array([i for i in members if live[i]])
        o, pk = off[m], pack[m]
        s = o.mean(0)
        q = pk - pk.mean(0)
        qh = q / np.linalg.norm(q, axis=1, keepdims=True)
        e = o - s
        e = e - np.sum(e * qh, 1, keepdims=True) * qh
        _, _, vt = np.linalg.svd(e - e.mean(0), full_matrices=False)
        basis[f] = dict(
            members=m,
            s=s / np.linalg.norm(s),
            qh=qh,
            pcs=vt[:K_PC],
            eh=e / np.linalg.norm(e, axis=1, keepdims=True),
            rh=(pk + o) / np.linalg.norm(pk + o, axis=1, keepdims=True),
        )
    pos = {int(i): (f, k) for f, b in basis.items() for k, i in enumerate(b["members"])}
    moved = _moved(ids, off, end_rows) if end_rows else None

    # per row: the draws' gradients, in step order
    draws = {i: [] for i in pos}
    for b in sd["batches"]:
        for j, i in enumerate(b["rows"].tolist()):
            if i in draws:
                draws[i].append(
                    (b["step"], b["g_in"][j].numpy(), b["g_out"][j].numpy())
                )
    n_steps = len(sd["batches"])

    def parts(v, f, k):
        """Energy shares of v on s, q_i, e_i (orthonormalised in that order)
        and the rest; and the signed cos with s, e_i, the row."""
        B = basis[f]
        vecs = [B["s"], B["qh"][k], B["eh"][k]]
        Q, _ = np.linalg.qr(np.stack(vecs, 1))
        c = Q.T @ v
        tot = float(v @ v) + 1e-30
        sh = [float(x * x) / tot for x in c]
        return {
            "share_s": sh[0],
            "share_q": sh[1],
            "share_e": sh[2],
            "share_rest": 1 - sum(sh),
            "cos_s": float(v @ B["s"]) / np.sqrt(tot),
            "cos_e": float(v @ B["eh"][k]) / np.sqrt(tot),
            "cos_q": float(v @ B["qh"][k]) / np.sqrt(tot),
            "cos_row": float(v @ B["rh"][k]) / np.sqrt(tot),
            "share_glyph_pcs": float(np.sum((B["pcs"] @ v) ** 2)) / tot,
        }

    def adam(seq, scale):
        """AdamW (betas 0.9 / 0.99, eps 1e-8, wd 0) over the probe's steps,
        the row's gradient where drawn and 0 elsewhere; returns the summed
        step (−lr · m̂ / (√v̂ + ε)) in effective units."""
        m = np.zeros(dim)
        v = np.zeros(dim)
        disp = np.zeros(dim)
        by = {s: g for s, g in seq}
        for t in range(1, n_steps + 1):
            g = by.get(t)
            g = np.zeros(dim) if g is None else g
            m = 0.9 * m + 0.1 * g
            v = 0.99 * v + 0.01 * g * g
            mh, vh = m / (1 - 0.9**t), v / (1 - 0.99**t)
            disp += -lr * scale * mh / (np.sqrt(vh) + 1e-8)
        return disp * rs  # raw → effective

    rows_out = {}
    for i, (f, k) in pos.items():
        d = draws[i]
        if len(d) < 4:
            continue
        gi = np.stack(
            [x[1] for x in d]
        )  # d loss / d raw: the same direction as on the effective row
        go = np.stack([x[2] for x in d])
        gt = gi + go
        rec = {"family": f, "draws": len(d)}
        for name, G in (("in", gi), ("out", go), ("total", gt)):
            h1, h2 = G[0::2].sum(0), G[1::2].sum(0)
            rel = float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-30))
            mean = G.mean(0)
            rec[name] = {
                "norm_mean_draw": float(np.linalg.norm(G, axis=1).mean()),
                "norm_of_mean": float(np.linalg.norm(mean)),
                "split_half_cos": rel,
                "step": parts(-mean, f, k),  # a descent step's direction
            }
        rec["cos_in_out"] = float(
            gi.mean(0)
            @ go.mean(0)
            / (np.linalg.norm(gi.mean(0)) * np.linalg.norm(go.mean(0)) + 1e-30)
        )
        scale = kana_scale if f == "kana" else 1.0
        da = adam([(x[0], x[1] + x[2]) for x in d], scale)
        sgd = -gt.mean(0)
        a1 = adam([(x[0], x[1] + x[2]) for x in d[0::2]], scale)
        a2 = adam([(x[0], x[1] + x[2]) for x in d[1::2]], scale)
        rec["adam"] = {
            "disp": float(np.linalg.norm(da)),
            "cos_with_sgd": float(
                da @ sgd / (np.linalg.norm(da) * np.linalg.norm(sgd) + 1e-30)
            ),
            "split_half_cos": float(
                a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-30)
            ),
            "step": parts(da, f, k),
            "spike_turn_deg": _turn(da, basis[f], off, i),
        }
        if moved is not None and np.linalg.norm(moved[i]) > 0:
            mv = moved[i]

            def cm(v, mv=mv):
                return float(v @ mv / (np.linalg.norm(v) * np.linalg.norm(mv) + 1e-30))

            rec["vs_end"] = {
                "cos_sgd": cm(sgd),
                "cos_adam": cm(da),
                "cos_in": cm(-gi.mean(0)),
                "cos_out": cm(-go.mean(0)),
                "moved": float(np.linalg.norm(mv)),
            }
        rows_out[i] = rec

    summary = _summarise(rows_out)
    (PROBE / label / "read.json").write_text(
        json.dumps(
            {
                "label": label,
                "lr": lr,
                "kana_scale": kana_scale,
                "n_steps": n_steps,
                "summary": summary,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    _print(summary)


def _turn(disp, B, off, i):
    """Degrees the row's offset spike turns under the replayed displacement
    (stick held, as the decomposition reads it)."""
    import numpy as np

    sp = off[i] - (off[B["members"]].mean(0))
    new = sp + disp - disp @ B["s"] * B["s"]
    c = float(sp @ new / (np.linalg.norm(sp) * np.linalg.norm(new) + 1e-30))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _families(ids):
    import json as _j

    from reseed import REPO

    pk = REPO / "models/vocab_packs/anima_cjk_vocab_pack_preview51"
    j = _j.loads(
        (pk.parent / "anima_cjk_vocab_pack_preview51.json").read_text(encoding="utf-8")
    )
    tr = _j.loads(
        (pk / "anima_cjk_vocab_pack_preview51_trained.json").read_text(encoding="utf-8")
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(REPO / "library/anima/configs/qwen3_06b"))
    c2r = dict(j["char"])
    for q, r in j["qwen"].items():
        s = tok.decode([int(q)])
        if len(s) == 1 and s not in c2r:
            c2r[s] = r
    at = {int(e): k for k, e in enumerate(ids)}
    out = {}
    for f, chars in (("kana", tr["hiragana"] + tr["katakana"]), ("kanji", tr["kanji"])):
        out[f] = [at[c2r[c]] for c in chars if c in c2r and c2r[c] in at]
    return out


def _summarise(rows_out):
    import numpy as np

    out = {}
    for f in ("kana", "kanji"):
        R = [r for r in rows_out.values() if r["family"] == f]
        if not R:
            continue
        n = np.array([r["draws"] for r in R])
        qs = np.quantile(n, [0.25, 0.5, 0.75])
        groups = {
            "all": R,
            "rare (Q1 draws)": [r for r in R if r["draws"] <= qs[0]],
            "frequent (Q4 draws)": [r for r in R if r["draws"] > qs[2]],
        }
        out[f] = {"rows": len(R), "draws_q": [float(x) for x in qs]}
        for g, rr in groups.items():
            if not rr:
                continue
            d = {"n": len(rr)}
            for term in ("in", "out", "total"):
                d[term] = {
                    k: float(np.median([r[term][k] for r in rr]))
                    for k in ("norm_mean_draw", "norm_of_mean", "split_half_cos")
                }
                d[term]["step"] = {
                    k: float(np.median([r[term]["step"][k] for r in rr]))
                    for k in rr[0][term]["step"]
                }
            d["out_over_in_draw"] = float(
                np.median(
                    [r["out"]["norm_mean_draw"] / r["in"]["norm_mean_draw"] for r in rr]
                )
            )
            d["cos_in_out"] = float(np.median([r["cos_in_out"] for r in rr]))
            d["adam"] = {
                k: float(np.median([r["adam"][k] for r in rr]))
                for k in ("disp", "cos_with_sgd", "split_half_cos", "spike_turn_deg")
            }
            d["adam"]["step"] = {
                k: float(np.median([r["adam"]["step"][k] for r in rr]))
                for k in rr[0]["adam"]["step"]
            }
            ve = [r["vs_end"] for r in rr if "vs_end" in r]
            if ve:
                d["vs_end"] = {k: float(np.median([v[k] for v in ve])) for k in ve[0]}
            out[f][g] = d
    return out


def _print(summary):
    for f, s in summary.items():
        print(f"\n== {f}: {s['rows']} rows, draws Q1/med/Q3 {s['draws_q']}")
        for g in ("all", "rare (Q1 draws)", "frequent (Q4 draws)"):
            if g not in s:
                continue
            d = s[g]
            print(
                f" -- {g} (n {d['n']}); |out|/|in| per draw {d['out_over_in_draw']:.2f}; cos(mean in, mean out) {d['cos_in_out']:+.3f}"
            )
            for term in ("in", "out", "total"):
                t = d[term]
                st = t["step"]
                print(
                    f"   {term:5s} split-half cos {t['split_half_cos']:+.3f} | step shares s {st['share_s']:.3f} q {st['share_q']:.3f} e {st['share_e']:.3f} rest {st['share_rest']:.3f} | cos s {st['cos_s']:+.3f} e {st['cos_e']:+.3f} q {st['cos_q']:+.3f} row {st['cos_row']:+.3f}"
                )
            a = d["adam"]
            st = a["step"]
            if "vs_end" in d:
                v = d["vs_end"]
                print(
                    f"   vs the run's own move (|Δ| {v['moved']:.1f}): cos sgd {v['cos_sgd']:+.3f} adam {v['cos_adam']:+.3f} in {v['cos_in']:+.3f} out {v['cos_out']:+.3f}"
                )
            print(
                f"   glyph-PC share (iso {K_PC / 1024:.3f}): in {d['in']['step']['share_glyph_pcs']:.3f} out {d['out']['step']['share_glyph_pcs']:.3f} adam {d['adam']['step']['share_glyph_pcs']:.3f}"
            )
            print(
                f"   adam  cos(adam, sgd) {a['cos_with_sgd']:+.3f} split-half {a['split_half_cos']:+.3f} turn {a['spike_turn_deg']:.2f}° | shares s {st['share_s']:.3f} q {st['share_q']:.3f} e {st['share_e']:.3f} rest {st['share_rest']:.3f} | cos e {st['cos_e']:+.3f} row {st['cos_row']:+.3f}"
            )


def tiers(label: str, run_name: str) -> None:
    """Per draw, the item that holds the row's glyph (draws whose glyph sits
    in one item of the batch only): agreement with the row's other draws
    (leave-one-out cos) by family × σ decile / tier, within-tier, and the
    bubbleN-sum against the sent-sum per row."""
    import collections

    import numpy as np
    import torch
    from reseed.config import load

    sd = torch.load(PROBE / label / "grads.pt", map_location="cpu", weights_only=False)
    recs = [
        json.loads(ln)
        for ln in (load(run_name).data / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln
    ]
    ids = sd["ext_ids"]
    fam = _families(ids)
    famof = {k: f for f, ks in fam.items() for k in ks}
    r2c = {v: c for c, v in _char_rows().items()}
    draws: dict = {}
    for b in sd["batches"]:
        for jj, k in enumerate(b["rows"].tolist()):
            ch = r2c.get(ids[k])
            if k not in famof or not ch:
                continue
            its = [t for t, i in enumerate(b["items"]) if ch in recs[i]["text"]]
            if len(its) != 1:
                continue
            r = recs[b["items"][its[0]]]
            g = (b["g_in"][jj] + b["g_out"][jj]).numpy()
            draws.setdefault(k, []).append((b["sigma"][its[0]], r["tier"], g))

    def loo(G):
        S = G.sum(0)
        return [
            float(g @ (S - g) / (np.linalg.norm(g) * np.linalg.norm(S - g) + 1e-30))
            for g in G
        ]

    by = collections.defaultdict(list)
    for k, d in draws.items():
        if len(d) < 20:
            continue
        for x, c in zip(d, loo(np.stack([x[2] for x in d]))):
            by[(famof[k], "σ", round(min(int(x[0] * 10), 9) / 10, 1))].append(c)
            by[(famof[k], "tier", x[1])].append(c)
    out = {
        "loo_by": {
            f"{a}|{b}|{c}": [len(v), float(np.mean(v))]
            for (a, b, c), v in sorted(by.items(), key=str)
        }
    }
    for key, v in out["loo_by"].items():
        print(f"  {key:28s} n {v[0]:5d}  cos(draw, LOO) {v[1]:+.3f}")
    for f in ("kana", "kanji"):
        for name in ("bubbleN", "sent"):
            cs = []
            for k, d in draws.items():
                G = [x[2] for x in d if x[1].startswith(name)]
                if famof[k] == f and len(G) >= 8:
                    cs += loo(np.stack(G))
            out[f"{f}|within|{name}"] = [len(cs), float(np.mean(cs)) if cs else None]
        cc = []
        for k, d in draws.items():
            A = [x[2] for x in d if x[1].startswith("bubbleN")]
            B = [x[2] for x in d if x[1].startswith("sent")]
            if famof[k] == f and len(A) >= 8 and len(B) >= 8:
                a, b = np.sum(A, 0), np.sum(B, 0)
                cc.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
        out[f"{f}|bubbleN_vs_sent"] = [len(cc), float(np.median(cc)) if cc else None]
        print(
            f"  {f}: within bubbleN {out[f'{f}|within|bubbleN']}, within sent {out[f'{f}|within|sent']}, "
            f"cos(bubbleN sum, sent sum) median {out[f'{f}|bubbleN_vs_sent']}"
        )
    (PROBE / label / "tiers.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False)
    )


def _char_rows() -> dict:
    import json as _j

    from reseed import REPO
    from transformers import AutoTokenizer

    j = _j.loads(
        (REPO / "models/vocab_packs/anima_cjk_vocab_pack_preview51.json").read_text(
            encoding="utf-8"
        )
    )
    tok = AutoTokenizer.from_pretrained(str(REPO / "library/anima/configs/qwen3_06b"))
    c2r = dict(j["char"])
    for q, r in j["qwen"].items():
        s = tok.decode([int(q)])
        if len(s) == 1 and s not in c2r:
            c2r[s] = r
    return c2r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["grad", "read", "tiers"])
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument(
        "--rows", default=STICK080, help="merged trained.pt, repo-relative or absolute"
    )
    p.add_argument("--batches", type=int, default=1200)
    p.add_argument("--label", default="s080")
    p.add_argument("--lr", type=float, default=2e-4, help="read: replay lr (f0's peak)")
    p.add_argument(
        "--kana_scale", type=float, default=0.12, help="read: f0's kana row_lr"
    )
    p.add_argument(
        "--end",
        default=F0,
        help="read: the trained rows the probe's start went to ('' = skip)",
    )
    a = p.parse_args()
    if a.verb == "grad":
        grad(a.run, a.rows, a.batches, a.label)
    elif a.verb == "tiers":
        tiers(a.label, "sent_kanji")
    else:
        read(a.label, a.lr, a.kana_scale, a.end)


if __name__ == "__main__":
    main()
