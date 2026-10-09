#!/usr/bin/env python
"""probe_pres — the page-preservation term's gradient on the rows (2026-10-06)

The question (user, 10-06): `product_criteria.md` Axis 2 — the JA render
differs from the EN render of the same page only inside the text box, one
glyph or a sentence alike — as a training term. The form:

    L_pres = mean_out (v_θ(x_σ, c_JA; rows) − sg[v_base(x_σ, c_EN)])²

the student is the DiT with the rows under the item's own caption, the
teacher the same frozen DiT at the same x_σ (same latent, same ε) under the
caption with its JA string swapped for an EN line (``english text``,
``English text reads as``); the EN caption holds no ext id, so the teacher
never sees a row. ``mean_out`` averages the cells outside the item's text box
dilated by ``DIL`` latent cells (the EN line need not sit on the JA box's
cells). The data term's out-box residual is ~82 % the draw's own (σ, ε)
(`reports/probe_scene_2026_10_06.md`); here student and teacher share them,
so only what the rows change against EN is left.

No training: the rows sit still and the gradient is read.

- ``grad`` (GPU): f0's start rows (``--rows``), the trainer's first
  ``--batches`` batches in its own order (as `probe_geom`), each batch at
  one σ off ``SIGMAS`` in turn (the bands are not used: the point is the
  layout σ above them). Per batch the data term's in-box and out-box
  summands (`probe_geom.split_terms`) and L_pres, each one's gradient on
  the rows; per item L_pres's value →
  ``output/cjk_anima_reseed/probe_pres/<label>/grads.pt``.
- ``read`` (CPU): per row and term, the per-draw norm, split-half cos and
  the per-draw signal share ρ (Spearman–Brown), the step's share on the
  family stick; the family-common share of each term (‖mean over rows‖² /
  mean ‖row‖² of the rows' mean gradients); all of it by σ; per item
  L_pres by σ × glyph count → ``…/<label>/read.json``.

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_pres.py grad --label s080"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_pres.py read --label s080
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

PROBE = OUT / "probe_pres"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"
SIGMAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# the term's pieces live with the trainer's loss (cjk_scale.train(pres=))
from cjk_scale.loss import _len_bin, en_caption, out_mask  # noqa: E402, F401
from cjk_scale.loss import PRES_DIL as DIL  # noqa: E402


def grad(run_name: str, rows_path: str, batches: int, label: str) -> None:
    import os
    import time
    from types import SimpleNamespace

    import torch
    from cjk_scale import train as T
    from cjk_scale.loss import glyph_count
    from common.models import (
        checkpoints,
        dit_forward,
        encode_captions,
        ext_ids_of,
        gen_args,
    )
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from probe_geom import split_terms
    from reseed import REPO
    from reseed.config import load
    from train.stage import Batcher, LatentStore

    from cjk_scale.rows import Rows

    run = load(run_name)
    run.use_pack()
    rows_path = str(rows_path if Path(rows_path).is_absolute() else REPO / rows_path)
    assert Path(rows_path).is_file(), rows_path
    out = PROBE / label
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(T.SEED)
    data = run.data
    recs, _ev, vocabs = T.load_items(data)
    keep = list(range(len(recs)))
    assert all(r["src"] == "scene" for r in recs), "a scene table: the box share is on"
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    # compiled below, after the rows: one dynamic-seq graph over the data's
    # token counts, not gen_args' static one-graph-per-count cascade
    args.compile_blocks = False
    device = get_generation_settings(args).device
    ns = SimpleNamespace(seed=T.SEED, batch=T.BATCH, train_size=512)
    lat = LatentStore(ns, data, recs, keep, device)

    # the batches first (the trainer's order); only their captions are
    # encoded, JA and EN, into the probe's own dir (the data dir's te_cache
    # holds the whole run's and is left alone)
    batcher = Batcher(ns, recs, lat)
    order = [list(batcher.next(step)) for step in range(1, batches + 1)]
    used = sorted({i for b in order for i in b})
    en_of = {i: en_caption(recs[i], i) for i in used}
    ja = sorted({recs[i]["caption"] for i in used})
    cache = encode_captions(ja + sorted(set(en_of.values())), device, out / "te")
    touched = ext_ids_of({c: cache[c] for c in ja})
    print(
        f"captions: {len(ja)} JA + {len(set(en_of.values()))} EN for {len(used)} items, "
        f"{len(touched)} ext rows touched",
        flush=True,
    )
    rc = run.scale_config()
    p = T.plan(rc, data, recs, vocabs, touched, Path(rows_path))

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
    seqs = [(w // 16) * (h // 16) for w, h in (recs[i]["shape"] for i in used)]
    # three autograd.grad calls on one graph (retain_graph) — donated buffers forbid it
    torch._functorch.config.donated_buffer = False
    anima.compile_blocks(dynamic_seq=True, seq_range=(min(seqs), max(seqs)))
    anima.train()
    raw = rows.delta.raw
    got = []
    t0 = time.time()
    for step, idx in enumerate(order, start=1):
        sigma = SIGMAS[(step - 1) % len(SIGMAS)]
        latents = lat[idx].to(device)
        noise = torch.randn_like(latents)
        brecs = [recs[i] for i in idx]
        ts = torch.full((len(idx),), sigma, device=device, dtype=torch.float32)
        noisy = ((1.0 - sigma) * latents.float() + sigma * noise.float()).to(
            torch.bfloat16
        )
        target = noise.float() - latents.float()
        # grad mode on (a no_grad forward guards its own graphs): the EN
        # caption holds no ext id, so ExtDelta stays out and nothing here
        # needs grad — detach is the stop-gradient
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teach = (
                dit_forward(anima, noisy, ts, cache, [en_of[i] for i in idx], device)
                .detach()
                .float()
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
        mo = out_mask(pred.shape, brecs, pred.device, T.GRID_BOX)
        per_cell = ((pred.float() - teach) ** 2).mean(dim=1, keepdim=True)
        pres_item = (per_cell * mo).sum(dim=(1, 2, 3)) / mo.sum(dim=(1, 2, 3)).clamp(
            min=1
        )
        l_pres = pres_item.mean()
        g_in = torch.autograd.grad(l_in, raw, retain_graph=True)[0]
        g_out = torch.autograd.grad(l_out, raw, retain_graph=True)[0]
        g_pres = torch.autograd.grad(l_pres, raw)[0]
        nz = (
            (live & ((g_in.abs().sum(1) + g_pres.abs().sum(1)) > 0))
            .nonzero()
            .squeeze(1)
        )
        got.append(
            {
                "step": step,
                "sigma": sigma,
                "items": [int(i) for i in idx],
                "glyphs": [glyph_count(r["text"]) for r in brecs],
                "tiers": [r["tier"] for r in brecs],
                "pres_item": pres_item.detach().float().cpu().tolist(),
                "out_frac": (mo.mean(dim=(1, 2, 3))).float().cpu().tolist(),
                "rows": nz.cpu(),
                "g_in": g_in[nz].float().cpu(),
                "g_out": g_out[nz].float().cpu(),
                "g_pres": g_pres[nz].float().cpu(),
            }
        )
        if step % 50 == 0 or step == 1:
            print(
                f"step {step}/{batches} σ {sigma}: {len(nz)} rows, l_in {float(l_in.detach()):.4f} "
                f"l_out {float(l_out.detach()):.4f} l_pres {float(l_pres.detach()):.5f}, "
                f"{(time.time() - t0) / step:.2f} s/batch",
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
            "sigmas": list(SIGMAS),
            "dil": DIL,
            "batches": got,
        },
        out / "grads.pt",
    )
    print(f"→ {out / 'grads.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


TERMS = ("in", "out", "pres")


def _rho(c: float, n: int) -> float:
    """Per-draw signal share from the split-half cos of two halves of n / 2
    draws (Spearman–Brown inverted)."""
    h = n / 2
    return float(c / (h * (1 - c) + c)) if c < 1 else 1.0


def read(label: str) -> None:
    import collections

    import numpy as np
    import torch
    from probe_geom import _families

    sd = torch.load(PROBE / label / "grads.pt", map_location="cpu", weights_only=False)
    ids = sd["ext_ids"]
    off = sd["raw"].numpy() * float(sd["row_scale"])
    live = sd["live"].numpy()
    fam = _families(ids)
    stick = {}
    for f, members in fam.items():
        m = np.array([i for i in members if live[i]])
        s = off[m].mean(0)
        stick[f] = s / np.linalg.norm(s)
    famof = {i: f for f, ks in fam.items() for i in ks if live[i]}

    # per row × σ-group: the draws of each term
    groups = {"all": None, "σ ≤ 0.7": (0.0, 0.75), "σ ≥ 0.8": (0.75, 1.0)}
    groups.update({f"σ {s}": (s - 0.01, s + 0.01) for s in sd["sigmas"]})
    draws = collections.defaultdict(lambda: collections.defaultdict(list))
    for b in sd["batches"]:
        for j, i in enumerate(b["rows"].tolist()):
            if i not in famof:
                continue
            for g, rng in groups.items():
                if rng is None or rng[0] <= b["sigma"] <= rng[1]:
                    draws[g][i].append({t: b[f"g_{t}"][j].numpy() for t in TERMS})

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    summary = {}
    for g in groups:
        summary[g] = {}
        for f in ("kana", "kanji"):
            R = {i: d for i, d in draws[g].items() if famof[i] == f and len(d) >= 6}
            if not R:
                continue
            rec = {
                "rows": len(R),
                "draws_med": float(np.median([len(d) for d in R.values()])),
            }
            means = {t: [] for t in TERMS}
            for t in TERMS:
                nd, sh, rho, cs, ss = [], [], [], [], []
                for i, d in R.items():
                    G = np.stack([x[t] for x in d])
                    h1, h2 = G[0::2].sum(0), G[1::2].sum(0)
                    c = cos(h1, h2)
                    mean = G.mean(0)
                    means[t].append(mean / (np.linalg.norm(mean) + 1e-30))
                    nd.append(float(np.linalg.norm(G, axis=1).mean()))
                    sh.append(c)
                    rho.append(_rho(c, len(G)))
                    cs.append(cos(-mean, stick[f]))
                    ss.append(cos(-mean, stick[f]) ** 2)
                M = np.stack(means[t])
                rec[t] = {
                    "norm_draw": float(np.median(nd)),
                    "split_half": float(np.median(sh)),
                    "rho": float(np.median(rho)),
                    "cos_step_stick": float(np.median(cs)),
                    "share_stick": float(np.median(ss)),
                    # family-common share of the rows' unit mean gradients
                    "common": float(np.linalg.norm(M.mean(0)) ** 2),
                    "common_cos_stick": cos(-M.mean(0), stick[f]),
                }
            rec["pres_over_in"] = float(
                np.median(
                    [
                        np.linalg.norm(np.stack([x["pres"] for x in d]), axis=1).mean()
                        / (
                            np.linalg.norm(
                                np.stack([x["in"] for x in d]), axis=1
                            ).mean()
                            + 1e-30
                        )
                        for d in R.values()
                    ]
                )
            )
            rec["cos_pres_in"] = float(
                np.median(
                    [
                        cos(
                            np.stack([x["pres"] for x in d]).mean(0),
                            np.stack([x["in"] for x in d]).mean(0),
                        )
                        for d in R.values()
                    ]
                )
            )
            summary[g][f] = rec

    # the term's value per item: σ × glyph-count bin
    val = collections.defaultdict(list)
    for b in sd["batches"]:
        for n, v in zip(b["glyphs"], b["pres_item"]):
            val[(b["sigma"], _len_bin(n))].append(v)
    by_len = {
        f"{s}|{lb}": [len(val[(s, lb)]), float(np.mean(val[(s, lb)]))]
        for s in sd["sigmas"]
        for lb in ("one", "word", "line")
        if val[(s, lb)]
    }
    out = {
        "label": label,
        "n_steps": len(sd["batches"]),
        "summary": summary,
        "pres_by_sigma_len": by_len,
    }
    (PROBE / label / "read.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False)
    )
    _print(out)


def _print(out):
    for g, fams in out["summary"].items():
        for f, r in fams.items():
            print(
                f"\n== {g} · {f}: {r['rows']} rows, draws med {r['draws_med']:.0f}; "
                f"|pres|/|in| per draw {r['pres_over_in']:.3f}, cos(mean pres, mean in) {r['cos_pres_in']:+.3f}"
            )
            for t in TERMS:
                x = r[t]
                print(
                    f"   {t:5s} |g| {x['norm_draw']:.2e}  split-half {x['split_half']:+.3f}  ρ {x['rho']:.3f}  "
                    f"cos(step, stick) {x['cos_step_stick']:+.3f}  common {x['common']:.3f} "
                    f"(cos stick {x['common_cos_stick']:+.3f})"
                )
    print("\nL_pres per item, σ | length bin: n, mean")
    for k, (n, v) in out["pres_by_sigma_len"].items():
        print(f"  {k:12s} {n:5d}  {v:.5f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["grad", "read"])
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument(
        "--rows", default=STICK080, help="merged trained.pt, repo-relative or absolute"
    )
    p.add_argument("--batches", type=int, default=1200)
    p.add_argument("--label", default="s080")
    a = p.parse_args()
    if a.verb == "grad":
        grad(a.run, a.rows, a.batches, a.label)
    else:
        read(a.label)


if __name__ == "__main__":
    main()
