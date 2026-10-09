#!/usr/bin/env python
"""probe_accum — per-row accumulation against the trainer's AdamW, a smoke
(2026-10-06)

The question (user, 10-06, after ``reports/probe_scene_2026_10_06.md``): a
draw's gradient on a row is ~97 % its own; does stepping a row only once it
has summed N draws beat stepping on every draw? Under AdamW, N draws summed
into one step at lr × √N move a row by the same signal and the same noise
as the N steps at lr, in expectation and to first order — so the pair isolates
the one thing that can differ: the noise walk moving the row, and the later
draws then read at a moved row (drift).

- ``select`` (CPU): ``--rows`` kanji of f0's at mid frequency (items holding
  the glyph in ``--band``, evenly spaced by count), the f0 data dir's items
  holding any of them, the multi-glyph lines split by text into train / held
  out (``--hold`` of the texts; single glyphs always train) → printed, and
  ``…/<label>/select.json``.
- ``run`` (GPU, one job): the trainer's setup on f0's data dir and start rows
  (``seed_fixed_1005_stick080``), only the picked rows live, every other row
  frozen at the start; two arms on the **same** batches, σ and ε (SEED 0
  re-drawn per arm), ``--steps`` each, constant lr, compiled:

  - ``plain``: ``torch.optim.AdamW`` (the trainer's betas, wd 0) at ``--lr``;
  - ``accum``: per row, the gradient summed over its draws until ``--accum``
    items have held the glyph, then one AdamW step on that row alone at
    ``--lr`` × √``--accum`` (its own m / v / t; rows between events do not
    move), what is left summed in one last step at the end.

  ``--lr_accum`` sets accum's lr outright (a16: lr × √N moved the rows 0.46×
  plain's — AdamW's v on a sparse row averages in the absent steps' zeros, so
  a plain draw steps ~1 / √(draw rate) larger); ``--reuse <label>`` takes
  start / plain and their evals from that label (same picks, draws, seeds).

  Then the held-out items at start / plain / accum under the same σ, ε per
  item (``--evals`` draws each): the box-share loss and its in-box term
  → ``…/<label>/{rows.pt, eval.json, train_log.json}``, and ``read``.
- ``read`` (CPU): per row the held-out loss change (plain − start, accum −
  start, accum − plain; the items holding the row), the sign over rows, the
  rows' displacement and its cos with f0's (f0's move sums ~360 draws per
  kanji: mostly signal) → ``…/<label>/read.json``.

    .venv/bin/python project/cjk_anima_reseed/probes/probe_accum.py select --label a16
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_accum.py run --label a16"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_accum.py read --label a16
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_accum.py look --label a16m"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()

from reseed import OUT, REPO  # noqa: E402

PROBE = OUT / "probe_accum"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"
F0 = "output/cjk_anima_reseed/sent_kanji_f0/trained.pt"
BETAS, EPS = (0.9, 0.99), 1e-8  # cjk_scale.train's AdamW, torch's eps


def _held(text: str, hold: float) -> bool:
    return len(text) > 1 and zlib.crc32(text.encode()) % 1000 < hold * 1000


def select(run_name: str, label: str, n_rows: int, band: tuple, hold: float) -> dict:
    from probe_geom import _char_rows
    from reseed.config import load

    run = load(run_name)
    kanji = next(
        s for s in run.rows if not any("぀" <= c <= "ヿ" for c in s)
    ).removeprefix("chars:")
    recs = [
        json.loads(ln)
        for ln in (run.data / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    count: dict = {}
    for r in recs:
        for c in set(r["text"]):
            if c in kanji:
                count[c] = count.get(c, 0) + 1
    lo, hi = band
    elig = sorted((n, c) for c, n in count.items() if lo <= n <= hi)
    assert len(elig) >= n_rows, f"{len(elig)} kanji in {band}, want {n_rows}"
    step = len(elig) / n_rows
    picked = [elig[int(k * step)][1] for k in range(n_rows)]
    c2r = _char_rows()
    assert all(c in c2r for c in picked), [c for c in picked if c not in c2r]
    on = set(picked)
    items = [i for i, r in enumerate(recs) if on & set(r["text"])]
    held = [i for i in items if _held(recs[i]["text"], hold)]
    train = [i for i in items if not _held(recs[i]["text"], hold)]
    draws = {c: sum(c in recs[i]["text"] for i in train) for c in picked}
    sel = {
        "run": run_name,
        "chars": picked,
        "rows": [int(c2r[c]) for c in picked],
        "items_per_char": {c: count[c] for c in picked},
        "train_items_per_char": draws,
        "train": train,
        "held": held,
        "band": list(band),
        "hold": hold,
    }
    out = PROBE / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "select.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    per_item = [len(on & set(recs[i]["text"])) for i in train]
    print(
        f"{n_rows} kanji ({''.join(picked)}), items {lo}–{hi} each in {run_name}'s data\n"
        f"items holding one: {len(items)} → train {len(train)}, held out "
        f"{len(held)} ({len({recs[i]['text'] for i in held})} texts)\n"
        f"picked glyphs per train item: mean {sum(per_item) / len(per_item):.2f}; "
        f"train items per row: min {min(draws.values())}, median "
        f"{sorted(draws.values())[n_rows // 2]}, max {max(draws.values())}\n"
        f"→ {out / 'select.json'}",
        flush=True,
    )
    return sel


def item_terms(pred, target, recs):
    """``box_share_fm_loss`` per item: (loss, in-box mean) as (B,) vectors."""
    import torch
    from cjk_scale import train as T
    from cjk_scale.loss import box_mask, box_share_of, glyph_count

    se = (pred.float() - target.float()) ** 2
    m = box_mask(se.shape, recs, se.device, T.GRID_BOX)
    per_cell = se.mean(dim=1, keepdim=True)
    n_in = m.sum(dim=(1, 2, 3))
    n_out = (1.0 - m).sum(dim=(1, 2, 3))
    mean_in = (per_cell * m).sum(dim=(1, 2, 3)) / n_in.clamp(min=1)
    mean_out = (per_cell * (1.0 - m)).sum(dim=(1, 2, 3)) / n_out.clamp(min=1)
    s = torch.tensor(
        [
            box_share_of(
                glyph_count(r["text"]), T.BOX_SHARE, T.BOX_SHARE_CAP, T.BOX_SHARE_GLYPHS
            )
            for r in recs
        ],
        device=se.device,
        dtype=se.dtype,
    )
    s = torch.where(n_in > 0, s, torch.zeros_like(s))
    s = torch.where(n_out > 0, s, torch.ones_like(s))
    return s * mean_in + (1.0 - s) * mean_out, mean_in


def run(
    run_name: str,
    label: str,
    steps: int,
    accum: int,
    lr: float,
    evals: int,
    lr_accum: float | None = None,
    reuse: str = "",
) -> None:
    import os
    import time
    from types import SimpleNamespace

    import torch
    from cjk_scale import train as T
    from cjk_scale.loss import box_share_fm_loss
    from cjk_scale.rows import Rows
    from common.models import checkpoints, dit_forward, gen_args
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.harness import compile_blocks_for_training
    from reseed.config import load
    from train.stage import Batcher, LatentStore, _encode_text

    out = PROBE / label
    sel = json.loads((out / "select.json").read_text(encoding="utf-8"))
    assert sel["run"] == run_name, (sel["run"], run_name)
    run_ = load(run_name)
    run_.use_pack()
    start = REPO / STICK080
    data = run_.data
    recs_all, ev, vocabs = T.load_items(data)
    keep = sorted(sel["train"] + sel["held"])
    pos = {i: k for k, i in enumerate(keep)}
    recs = [recs_all[i] for i in keep]
    tr = [pos[i] for i in sel["train"]]
    ho = [pos[i] for i in sel["held"]]
    assert all(r["src"] == "scene" for r in recs), "a scene table: the box share is on"
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    device = get_generation_settings(args).device
    cache, touched, _ = _encode_text(recs, ev, device, out, te_cache=data / "te_cache")
    p = T.plan(run_.scale_config(), data, recs, vocabs, touched, start)
    live_ids = set(sel["rows"])
    assert live_ids <= p.idx, "a picked row outside f0's rows"
    ns = SimpleNamespace(seed=T.SEED, batch=T.BATCH, train_size=512)
    lat = LatentStore(ns, data, recs_all, keep, device)
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    rows = Rows(
        anima,
        device,
        live_ids,
        pack,
        warm=start,
        init_anchor=0.0,
        free_residual=0.0,
        lr=lr,
        touched=live_ids & p.touched,
        frozen=(p.idx | p.frozen) - live_ids,
        context=start,
    )
    raw = rows.delta.raw
    live = ~rows.frozen_mask
    ext = [int(e) for e in rows.delta.ext_ids]
    at = {e: k for k, e in enumerate(ext)}
    char_at = {c: at[r] for c, r in zip(sel["chars"], sel["rows"])}
    assert int(live.sum()) == len(live_ids), (int(live.sum()), len(live_ids))
    raw0 = raw.detach().clone()
    anima.train()
    compile_blocks_for_training(
        anima, None, backend="inductor", n_token_families=lat.n_families
    )
    trecs = [recs[i] for i in tr]

    class Sub:  # the Batcher over the train items; LatentStore indexes keep-positions
        row_of = lat.row_of

        @staticmethod
        def shape_of(k):
            return lat.shape_of(tr[k])

    def holders(brecs) -> torch.Tensor:
        n = torch.zeros(len(ext), device=device)
        for r in brecs:
            for c in set(r["text"]):
                if c in char_at:
                    n[char_at[c]] += 1
        return n

    def forward(idx_keep, brecs):
        latents = lat[idx_keep].to(device)
        noise = torch.randn_like(latents)
        noisy, ts, target = T.noisy_by_band(
            latents, noise, [tuple(r["band"]) for r in brecs], device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dit_forward(
                anima, noisy, ts, cache, [r["caption"] for r in brecs], device
            )
        return pred, target, ts

    def train_arm(arm: str) -> dict:
        with torch.no_grad():
            raw.copy_(raw0)
        torch.manual_seed(T.SEED)
        batcher = Batcher(ns, trecs, Sub)
        opt = (
            torch.optim.AdamW([raw], lr=lr, weight_decay=0.0, betas=BETAS)
            if arm == "plain"
            else None
        )
        acc = torch.zeros_like(raw)
        cnt = torch.zeros(len(ext), device=device)
        m = torch.zeros_like(raw)
        v = torch.zeros_like(raw)
        t = torch.zeros(len(ext), device=device)
        lr_b = lr_accum or lr * math.sqrt(accum)
        draws = torch.zeros(len(ext), device=device)
        events = torch.zeros(len(ext), device=device)
        miss = 0
        log = []
        t0 = time.time()

        def event(ready):
            g = acc[ready] / cnt[ready, None]
            t[ready] += 1
            m[ready] = BETAS[0] * m[ready] + (1 - BETAS[0]) * g
            v[ready] = BETAS[1] * v[ready] + (1 - BETAS[1]) * g * g
            mh = m[ready] / (1 - BETAS[0] ** t[ready])[:, None]
            vh = v[ready] / (1 - BETAS[1] ** t[ready])[:, None]
            raw[ready] -= lr_b * mh / (vh.sqrt() + EPS)
            acc[ready] = 0
            cnt[ready] = 0
            events[ready] += 1

        for step in range(1, steps + 1):
            b = batcher.next(step)
            brecs = [trecs[k] for k in b]
            pred, target, _ = forward([tr[k] for k in b], brecs)
            loss = box_share_fm_loss(
                pred,
                target,
                brecs,
                T.BOX_SHARE,
                T.BOX_SHARE_CAP,
                T.BOX_SHARE_GLYPHS,
                T.GRID_BOX,
            )
            raw.grad = None
            loss.backward()
            # a draw = an item whose glyph reached the row: the text's count
            # where the gradient agrees (a word piece can swallow a glyph;
            # a caption tag can carry one the lettered text lacks)
            n = holders(brecs)
            hit = (raw.grad.abs().sum(1) > 0) & live
            miss += int(((n > 0) != hit).sum())
            n = torch.where(hit, n.clamp(min=1), torch.zeros_like(n))
            draws += n
            if arm == "plain":
                opt.step()
            else:
                with torch.no_grad():
                    acc += raw.grad.float()
                    cnt += n
                    ready = (cnt >= accum) & live
                    if bool(ready.any()):
                        event(ready)
            if step % 100 == 0 or step == 1:
                d = ((raw.detach() - raw0)[live] * rows.row_scale).norm(dim=1)
                rec = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "disp_mean": float(d.mean()),
                    "events_mean": float(events[live].mean()),
                    "s_per_step": (time.time() - t0) / step,
                }
                log.append(rec)
                print(f"{arm} {rec}", flush=True)
        if arm == "accum":
            with torch.no_grad():
                left = (cnt > 0) & live
                if bool(left.any()):
                    event(left)
        print(f"{arm}: {miss} row-batch presences off the text's", flush=True)
        return {
            "miss": miss,
            "raw": raw.detach()[live].float().cpu().clone(),
            "draws": draws[live].cpu(),
            "events": events[live].cpu(),
            "log": log,
            "min": (time.time() - t0) / 60,
        }

    def evaluate(tag: str) -> list:
        """Held-out items in shape-batches of BATCH (the compiled graphs'
        batch), each batch's σ / ε seeded by its index — the same per arm;
        grad mode on (no recompile), nothing stepped."""
        by: dict = {}
        for k in ho:
            by.setdefault(lat.shape_of(k), []).append(k)
        batches = [
            ks[j : j + T.BATCH]
            for _s, ks in sorted(by.items())
            for j in range(0, len(ks) - T.BATCH + 1, T.BATCH)
        ]
        got = []
        for e in range(evals):
            for bi, ks in enumerate(batches):
                torch.manual_seed(100_000 * (e + 1) + bi)
                brecs = [recs[k] for k in ks]
                pred, target, ts = forward(ks, brecs)
                lo, li = item_terms(pred, target, brecs)
                for j, k in enumerate(ks):
                    got.append(
                        {
                            "item": keep[k],
                            "e": e,
                            "sigma": float(ts.flatten()[j]),
                            "loss": float(lo[j]),
                            "in": float(li[j]),
                        }
                    )
                del pred, target, lo, li
        print(f"eval {tag}: {len(got)} item-draws", flush=True)
        return got

    t_all = time.time()
    arms, evs = {}, {}
    if reuse:  # start / plain from that label: the same picks, draws and eval seeds
        R0 = torch.load(PROBE / reuse / "rows.pt", weights_only=False)
        c0 = R0["config"]
        assert (c0["steps"], c0["lr"], c0["evals"]) == (steps, lr, evals), c0
        assert R0["ext_ids"] == [e for e, x in zip(ext, live.tolist()) if x]
        assert torch.equal(R0["start"], raw0[live].float().cpu()), "start rows moved"
        arms["plain"] = {
            **R0["plain"],
            "log": json.loads((PROBE / reuse / "train_log.json").read_text())["plain"],
        }
        ev0 = json.loads((PROBE / reuse / "eval.json").read_text(encoding="utf-8"))
        evs.update(start=ev0["start"], plain=ev0["plain"])
        print(f"start / plain from {reuse}", flush=True)
    else:
        arms["plain"] = train_arm("plain")
    arms["accum"] = train_arm("accum")
    for tag, src in (("start", raw0), ("plain", None), ("accum", None)):
        if tag in evs:
            continue
        with torch.no_grad():
            if src is None:
                raw.copy_(raw0)
                raw[live] = arms[tag]["raw"].to(raw.dtype).to(device)
            else:
                raw.copy_(src)
        evs[tag] = evaluate(tag)
    torch.save(
        {
            "ext_ids": [e for e, x in zip(ext, live.tolist()) if x],
            "chars": [
                next(c for c, r in zip(sel["chars"], sel["rows"]) if r == e)
                for e, x in zip(ext, live.tolist())
                if x
            ],
            "row_scale": rows.row_scale,
            "start": raw0[live].float().cpu(),
            **{a: {k: v for k, v in r.items() if k != "log"} for a, r in arms.items()},
            "config": {
                "steps": steps,
                "accum": accum,
                "lr": lr,
                "lr_accum": lr_accum or lr * math.sqrt(accum),
                "evals": evals,
                "reuse": reuse,
            },
        },
        out / "rows.pt",
    )
    (out / "train_log.json").write_text(
        json.dumps({a: r["log"] for a, r in arms.items()}, indent=1)
    )
    (out / "eval.json").write_text(json.dumps(evs))
    print(
        f"→ {out} ({(time.time() - t_all) / 60:.1f} min: plain "
        f"{arms['plain']['min']:.1f}, accum {arms['accum']['min']:.1f})",
        flush=True,
    )
    read(label)


def read(label: str) -> None:
    import numpy as np
    import torch
    from scipy.stats import binomtest

    out = PROBE / label
    sel = json.loads((out / "select.json").read_text(encoding="utf-8"))
    R = torch.load(out / "rows.pt", weights_only=False)
    evs = json.loads((out / "eval.json").read_text(encoding="utf-8"))
    from reseed.config import load

    held = set(sel["held"])
    lines = (load(sel["run"]).data / "train.jsonl").read_text(encoding="utf-8")
    recs_text = {
        i: json.loads(ln)["text"]
        for i, ln in enumerate(lines.splitlines())
        if i in held
    }

    def eff(path: str) -> dict:
        d = torch.load(REPO / path, map_location="cpu", weights_only=False)["delta"]
        k = {int(e): j for j, e in enumerate(d["ext_ids"])}
        rs = float(d["row_scale"])
        return {e: d["raw"][k[e]].float() * rs for e in R["ext_ids"]}

    s0, f0 = eff(STICK080), eff(F0)
    rs = float(R["row_scale"])
    start = R["start"] * rs
    for j, e in enumerate(R["ext_ids"]):
        assert torch.allclose(start[j], s0[e], atol=1e-3 * float(s0[e].norm())), (
            f"start row {e} is not stick080's"
        )

    def cos(u, v) -> float:
        return float(u @ v / (u.norm() * v.norm() + 1e-30))

    per_row = []
    for j, (e, c) in enumerate(zip(R["ext_ids"], R["chars"])):
        dp = R["plain"]["raw"][j] * rs - start[j]
        da = R["accum"]["raw"][j] * rs - start[j]
        df = f0[e] - s0[e]
        per_row.append(
            {
                "char": c,
                "draws": int(R["plain"]["draws"][j]),
                "events": int(R["accum"]["events"][j]),
                "disp_plain": float(dp.norm()),
                "disp_accum": float(da.norm()),
                "disp_f0": float(df.norm()),
                "cos_plain_f0": cos(dp, df),
                "cos_accum_f0": cos(da, df),
                "cos_plain_accum": cos(dp, da),
            }
        )
    # held-out: paired per item-draw, then per row over the items holding it
    key = {(r["item"], r["e"]): r for r in evs["start"]}
    diffs = []
    by_tag = {t: {(r["item"], r["e"]): r for r in evs[t]} for t in evs}
    for k, r0 in key.items():
        rp, ra = by_tag["plain"][k], by_tag["accum"][k]
        diffs.append(
            {
                "item": k[0],
                "sigma": r0["sigma"],
                **{
                    f"{m}_{a}": rx[m] - r0[m]
                    for m in ("loss", "in")
                    for a, rx in (("plain", rp), ("accum", ra))
                },
            }
        )
    for row in per_row:
        c = row["char"]
        ds = [d for d in diffs if c in recs_text[d["item"]]]
        row["held_draws"] = len(ds)
        for m in ("loss", "in"):
            for a in ("plain", "accum"):
                row[f"d{m}_{a}"] = (
                    float(np.mean([d[f"{m}_{a}"] for d in ds])) if ds else None
                )
            row[f"d{m}_accum_minus_plain"] = (
                row[f"d{m}_accum"] - row[f"d{m}_plain"] if ds else None
            )

    def summary(m: str) -> dict:
        rel = [r for r in per_row if r["held_draws"]]
        x = np.array([r[f"d{m}_accum_minus_plain"] for r in rel])
        it = np.array([d[f"{m}_accum"] - d[f"{m}_plain"] for d in diffs])
        base = np.array([d[f"{m}_plain"] for d in diffs])
        wins = int((x < 0).sum())
        return {
            "rows": len(rel),
            "accum_lower_rows": wins,
            "sign_p": binomtest(wins, len(rel)).pvalue,
            "row_median": float(np.median(x)),
            "item_mean": float(it.mean()),
            "item_se": float(it.std(ddof=1) / math.sqrt(len(it))),
            "plain_minus_start_item_mean": float(base.mean()),
            "plain_minus_start_item_se": float(base.std(ddof=1) / math.sqrt(len(base))),
            "accum_minus_start_item_mean": float(
                np.mean([d[f"{m}_accum"] for d in diffs])
            ),
        }

    res = {
        "config": R["config"],
        "held_item_draws": len(diffs),
        "held": {m: summary(m) for m in ("in", "loss")},
        "geometry_median": {
            k: float(np.median([r[k] for r in per_row]))
            for k in (
                "draws",
                "events",
                "disp_plain",
                "disp_accum",
                "disp_f0",
                "cos_plain_f0",
                "cos_accum_f0",
                "cos_plain_accum",
            )
        },
        "rows": per_row,
    }
    (out / "read.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("char draws ev | d_in plain accum a−p | cos→f0 plain accum | disp p a f0")
    for r in per_row:
        print(
            f"{r['char']} {r['draws']:4d} {r['events']:3d} | "
            + (
                f"{r['din_plain']:+.5f} {r['din_accum']:+.5f} {r['din_accum_minus_plain']:+.5f}"
                if r["held_draws"]
                else "   (no held-out item)   "
            )
            + f" | {r['cos_plain_f0']:+.3f} {r['cos_accum_f0']:+.3f} | "
            f"{r['disp_plain']:.2f} {r['disp_accum']:.2f} {r['disp_f0']:.2f}"
        )


LOOK = (52143, 60407, 73807, 40837)  # held-out sent_34 lines: 北 街 曜 勉


def look(label: str, items: tuple, seeds: tuple) -> None:
    """A look, not a read (user, 10-06): a few held-out lines rendered with
    plain's and accum's rows (the start's for every other row) on the ruler's
    path — its sampler, the punct pack, routed — each line on its own
    item's caption and canvas, read by the ruler's readers and scored as its
    glyphs are (``ruler.score_text`` / ``score_page``, no EN ref) →
    ``…/<label>/look/`` (+ ``sheet.png``, ``look.json``)."""
    import os

    import torch
    from PIL import Image, ImageDraw, ImageFont
    from reseed.config import load

    out = PROBE / label
    sel = json.loads((out / "select.json").read_text(encoding="utf-8"))
    R = torch.load(out / "rows.pt", weights_only=False)
    load(sel["run"]).use_pack()
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    import ruler as RU

    RU.PACK = "punct"
    lines = (load(sel["run"]).data / "train.jsonl").read_text(encoding="utf-8")
    lines = lines.splitlines()
    held = set(sel["held"])
    assert set(items) <= held, f"not held out: {set(items) - held}"
    its = sorted(
        ({"i": i, **json.loads(lines[i])} for i in items),
        key=lambda m: tuple(m["shape"]),
    )
    ids, tabs = RU.tables(["seed_fixed_1005_stick080"])
    pos = {e: k for k, e in enumerate(ids)}
    arms = {}
    for a in ("plain", "accum"):
        t = tabs["seed_fixed_1005_stick080"].clone()
        for j, e in enumerate(R["ext_ids"]):
            t[pos[e]] = R[a]["raw"][j].float() * float(R["row_scale"])
        arms[a] = t
    d = out / "look"

    def fn(a, m, sd):
        return d / a / f"i{m['i']}_s{sd}.png"

    todo = [(a, m, sd) for a in arms for m in its for sd in seeds]
    if not all(fn(*x).exists() for x in todo):
        r = RU.Renderer()
        assert r.ids == ids, "the ruler's union of rows moved"
        for a, tab in arms.items():
            r.set_arm(tab)
            for m in its:
                for sd in seeds:
                    r.render(fn(a, m, sd), m["caption"], sd, m["shape"])
                    print(f"  {a} i{m['i']} s{sd}", flush=True)
        del r
        torch.cuda.empty_cache()
    from common.readers import Readers, load_bgr

    rd = Readers("cuda")
    got = []
    for a, m, sd in todo:
        reads = rd.read_image(load_bgr(fn(a, m, sd)), whole=True)
        got.append(
            {
                "arm": a,
                "item": m["i"],
                "seed": sd,
                "text": m["text"],
                **RU.score_text(m["text"], reads),
                **RU.score_page(m["text"], reads, []),
            }
        )
    (d / "look.json").write_text(json.dumps(got, ensure_ascii=False, indent=1))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18
        )
    except OSError:
        font = ImageFont.load_default()
    H = 300
    cells = []
    for m in its:
        for sd in seeds:
            row = []
            for a in arms:
                im = Image.open(fn(a, m, sd)).convert("RGB")
                im = im.resize((int(im.width * H / im.height), H))
                g = next(
                    x
                    for x in got
                    if (x["arm"], x["item"], x["seed"]) == (a, m["i"], sd)
                )
                row.append((im, f"{a} s{sd} F1 {g['g_f1']:.2f} | {g['best']}"))
            cells.append((m["text"], row))
    W = max(sum(im.width for im, _ in row) + 10 * len(row) for _, row in cells)
    sheet = Image.new("RGB", (W, len(cells) * (H + 52)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (text, row) in enumerate(cells):
        y = k * (H + 52)
        dr.text((4, y + 2), text, fill="black", font=font)
        x = 0
        for im, cap in row:
            sheet.paste(im, (x, y + 26))
            dr.text((x + 4, y + 28 + H), cap, fill="black", font=font)
            x += im.width + 10
    sheet.save(d / "sheet.png")
    print("arm    item   seed  g_f1  g_p   g_r   best | text")
    for g in got:
        print(
            f"{g['arm']:<6} {g['item']:<6} {g['seed']}  {g['g_f1']:.2f}  {g['g_p']:.2f}  "
            f"{g['g_r']:.2f}  {g['best']} | {g['text']}"
        )
    for a in arms:
        xs = [g for g in got if g["arm"] == a]
        print(
            f"{a}: g_f1 {sum(g['g_f1'] for g in xs) / len(xs):.3f}, exact "
            f"{sum(g['exact'] for g in xs)} / {len(xs)}"
        )
    print(f"→ {d / 'sheet.png'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["select", "run", "read", "look"])
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument("--label", default="a16")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--band", default="80,130", help="items per kanji, lo,hi")
    p.add_argument("--hold", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--evals", type=int, default=2)
    p.add_argument("--lr_accum", type=float, default=None, help="default lr × √accum")
    p.add_argument(
        "--reuse", default="", help="a label: its start / plain arm and evals"
    )
    p.add_argument("--items", default=",".join(map(str, LOOK)), help="look: item ids")
    p.add_argument("--seeds", default="0,1", help="look: render seeds")
    a = p.parse_args()
    if a.verb == "select":
        lo, hi = (int(x) for x in a.band.split(","))
        select(a.run, a.label, a.rows, (lo, hi), a.hold)
    elif a.verb == "run":
        run(a.run, a.label, a.steps, a.accum, a.lr, a.evals, a.lr_accum, a.reuse)
    elif a.verb == "look":
        look(
            a.label,
            tuple(int(x) for x in a.items.split(",")),
            tuple(int(x) for x in a.seeds.split(",")),
        )
    else:
        read(a.label)


if __name__ == "__main__":
    main()
