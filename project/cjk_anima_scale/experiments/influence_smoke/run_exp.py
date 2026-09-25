#!/usr/bin/env python
"""influence_smoke — first contact for idea.md's validation influence.

Does a dev-set-anchored influence score rank the recipe cells the way the
rulers did, and does the linearization it rests on hold across a real
training delta? Everything runs on run0925_300f's built data dirs and its
trained table, so the ground truth is a run whose ruler verdict is known
(nothing bought at drift 1.4/1.7; the comparator that did buy hits was
scene_piece-only — next.md § 4a).

Three reads, one job:

E1  v_s sanity — v_s (in-box FM gradient on the rows over dev composites of
    string s, at the seed table) concentrates on s's own rows.
E2  linearization — pred ΔL_dev[s] = v_sᵀ (moved − seed) vs the measured
    paired loss change between the two tables (same item, σ, noise). Sign
    agreement per string is the gate; the magnitude ratio is the curvature
    read.
E3  ranking — per cell c (recipe × stage band), I_raw[c,s] = v_sᵀ ḡ_c and
    I_adam[c,s] = v_sᵀ (m̂_c / (√v̂_c + ε)). The rulers say scene_piece
    bought hits and the sentence/short/grid-heavy mix did not; if I does
    not reproduce that ordering, the surrogate fails the smoke and the
    full bank is not worth building (idea.md § Calibrate before trusting).

The bank is taken at TWO table points (seed and the 300f trained table);
cos(ḡ_c^seed, ḡ_c^moved) per cell is the drift read idea.md asks for.

Dev items are existing scene_piece composites whose text is the dev string
(the "right text rendered into a clean latent"), excluded from the bank
sample. Dev strings are run0925_72's dev pieces — disjoint from the
acceptance strings (product_criteria.md § Dev vs acceptance).

Caveat stated up front: dev FM loss is not a hit count. If E2 passes and
E3 fails, the verdict is "influence tracks the loss but the loss does not
track the ruler" — that kills the surrogate as specced, which is exactly
what the smoke is for.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap  # noqa: E402
from cjk_scale.paths import legacy_data_dir as data_dir  # noqa: E402  (old stage dirs, records)

bootstrap()

from bench._common import make_run_dir, write_result  # noqa: E402

BINS = ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9))
CELLS = (
    ("stage0507", "scene_piece"),
    ("stage0507", "scene_short"),
    ("stage0507", "grid_string"),
    ("stage0305", "scene_piece"),
    ("stage0305", "scene_sentence"),
    ("stage0305", "grid_string"),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--tag", default="run0925_300f", help="data dirs' tag")
    p.add_argument(
        "--seed_table",
        default=str(OUT / "rows_step1_0921_merged" / "trained.pt"),
    )
    p.add_argument(
        "--moved_table",
        default=str(OUT / "rows_joint0507_0305_run0925_300f" / "trained.pt"),
    )
    p.add_argument("--items_per_cell", type=int, default=192)
    p.add_argument(
        "--dev_strings", default="ありがとう,それを,すごい,して,です"
    )
    p.add_argument("--dev_items", type=int, default=24, help="per string, both tiers")
    p.add_argument("--dev_noises", type=int, default=2, help="noises per item per σ bin")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def load_recs(stage: str, tag: str) -> list[dict]:
    d = data_dir(stage, tag)
    assert (d / "train.jsonl").exists(), f"no data dir {d}"
    return [
        json.loads(ln)
        for ln in (d / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]


def sample_plan(args, recs_by_stage, rng):
    """(bank_pick, dev_pick): bank_pick[(stage, recipe)] and
    dev_pick[string] are lists of (stage, index-into-that-stage's-recs);
    dev items are excluded from the bank sample."""
    dev_strings = [s for s in args.dev_strings.split(",") if s]
    dev_pick: dict[str, list] = {}
    dev_idx: set[tuple] = set()
    for s in dev_strings:
        cands = [
            (st, i)
            for st, recs in recs_by_stage.items()
            for i, r in enumerate(recs)
            if r.get("recipe") == "scene_piece" and r.get("text") == s
        ]
        assert len(cands) >= 4, f"dev string {s!r}: only {len(cands)} scene_piece items"
        pick = rng.sample(cands, min(args.dev_items, len(cands)))
        dev_pick[s] = sorted(pick)
        dev_idx |= set(pick)
    bank_pick: dict[tuple, list] = {}
    for st, rcp in CELLS:
        pool = [
            (st, i)
            for i, r in enumerate(recs_by_stage[st])
            if r.get("recipe") == rcp and (st, i) not in dev_idx
        ]
        assert pool, f"cell {st}/{rcp}: no items"
        bank_pick[(st, rcp)] = sorted(
            rng.sample(pool, min(args.items_per_cell, len(pool)))
        )
    return bank_pick, dev_pick


def load_overlay(path: Path, ext_ids: list[int], row_scale: float, base):
    """base with the rows present in ``path`` replaced (rescaled to this
    table's units). Returns (table, n_present)."""
    import torch

    src = torch.load(path, map_location="cpu", weights_only=False)
    raw = src["delta"]["raw"].float()
    idx = {int(e): i for i, e in enumerate(src["delta"]["ext_ids"])}
    rs = src["delta"].get("row_scale")
    k = float(rs) / row_scale if rs is not None else 1.0
    out = base.clone()
    n = 0
    for i, e in enumerate(ext_ids):
        j = idx.get(int(e))
        if j is not None:
            out[i] = (raw[j] * k).to(out.device)
            n += 1
    return out, n


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    recs_by_stage = {st: load_recs(st, args.tag) for st, _ in CELLS}
    bank_pick, dev_pick = sample_plan(args, recs_by_stage, rng)

    print(f"influence_smoke {args.label}: tag {args.tag}")
    for cell, pick in bank_pick.items():
        print(f"  bank {cell[0]}/{cell[1]}: {len(pick)} items")
    for s, pick in dev_pick.items():
        by = defaultdict(int)
        for st, _ in pick:
            by[st] += 1
        print(f"  dev {s}: {len(pick)} items {dict(by)}")
    n_bank = sum(len(v) for v in bank_pick.values())
    n_dev = sum(len(v) for v in dev_pick.values())
    print(
        f"  reads: bank {n_bank} × 2 tables + dev {n_dev} × {len(BINS)} bins × "
        f"{args.dev_noises} noises (grad at seed + paired value at moved)"
    )
    if args.dry_run:
        return

    import torch
    import torch.nn.functional as F  # noqa: N812
    from common.models import checkpoints, dit_forward, gen_args
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.noise import fm_training_batch
    from train.stage import LatentStore, _encode_text

    from cjk_scale.legacy import load_stage as load  # the archived stage configs
    from cjk_scale.loss import box_mask, box_share_fm_loss
    from cjk_scale.rows import RowTable

    t_start = time.time()
    work = OUT / f"influence_{args.label}"
    work.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(
        "influence_smoke",
        label=args.label,
        root=LINE / "experiments" / "influence_smoke" / "results",
    )
    cfgs = {st: load(st, None) for st in recs_by_stage}
    args_gen = gen_args(512, 1000, 4.0, work)
    device = get_generation_settings(args_gen).device

    # -- per stage: the sub-sampled recs, their text + latents ---------------
    used: dict[str, list[int]] = {st: [] for st in recs_by_stage}
    for (st, _rcp), pick in bank_pick.items():
        used[st] += [i for _s, i in pick]
    for s, pick in dev_pick.items():
        for st, i in pick:
            used[st].append(i)
    ev = [
        {"text": s, "caption": f'japanese text. Japanese text reads as "{s}".'}
        for s in dev_pick
    ]
    per: dict[str, dict] = {}
    table_ext: set[int] = set()
    ev_ext_all: dict[str, list] = {}
    for st, idxs in used.items():
        idxs = sorted(set(idxs))
        sub = [recs_by_stage[st][i] for i in idxs]
        pos = {i: k for k, i in enumerate(idxs)}  # data-dir index → sub position
        sdir = work / st
        sdir.mkdir(exist_ok=True)
        cache, train_ext, ev_ext = _encode_text(
            sub, ev if not ev_ext_all else [], device, sdir, te_cache=sdir / "te_cache"
        )
        ev_ext_all = ev_ext_all or ev_ext
        table_ext |= train_ext
        ns = SimpleNamespace(
            seed=args.seed, batch=1, train_size=512, row_blocks=0, row_boost="", arm="rows"
        )
        per[st] = {
            "sub": sub,
            "pos": pos,
            "cache": cache,
            "lat": LatentStore(ns, sdir, sub, list(range(len(sub))), device),
        }
        print(f"{st}: {len(sub)} items encoded", flush=True)
    for s, ids in ev_ext_all.items():
        table_ext |= set(int(i) for i in ids)

    # -- model + table --------------------------------------------------------
    anima = load_dit_model(args_gen, device, torch.bfloat16)
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached — set ANIMA_VOCAB_PACK"
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    rows = RowTable(
        anima,
        device,
        table_ext,
        strategy_pack(tok),
        warm=Path(args.seed_table),
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
    )
    raw = rows.delta.raw
    ext_ids = [int(e) for e in rows.delta.ext_ids]
    R, D = raw.shape
    seed_tab = raw.detach().clone()
    moved_tab, n_moved = load_overlay(
        Path(args.moved_table), ext_ids, rows.row_scale, seed_tab
    )
    delta_tab = moved_tab - seed_tab
    print(
        f"table: {R} rows × {D}; moved overlays {n_moved} rows, "
        f"‖Δ‖ {float(delta_tab.norm()):.3f}",
        flush=True,
    )
    anima.train()
    g = torch.Generator(device=device).manual_seed(args.seed)

    def set_table(t):
        with torch.no_grad():
            raw.copy_(t)

    # -- bank: per cell × table point, unconditional mean + second moment ----
    points = ("seed", "moved")
    bank = {
        (cell, pt): {
            "sum_g": torch.zeros(R, D, device=device),
            "sum_g2": torch.zeros(R, D, device=device),
            "sum_norm": torch.zeros(R, device=device),
            "touched": torch.zeros(R, device=device),
            "n": 0,
        }
        for cell in bank_pick
        for pt in points
    }
    halves = {cell: torch.zeros(2, R, D, device=device) for cell in bank_pick}
    t0 = time.time()
    n_reads = 0
    for cell, pick in bank_pick.items():
        st, _rcp = cell
        cfg, p = cfgs[st], per[st]
        t = cfg.train
        bs_cfg, cap, n_cap = (
            float(t["box_share"]),
            float(t["box_share_cap"]),
            float(t["box_share_glyphs"]),
        )
        gbox = bool(int(t.get("grid_box", 0)))
        t_min, t_max = cfg.band
        for k, (st_, i) in enumerate(pick):
            r = p["sub"][p["pos"][i]]
            latents = p["lat"][[p["pos"][i]]].to(device)
            noise = torch.randn(
                latents.shape, generator=g, device=device, dtype=latents.dtype
            )
            noisy, ts, target = fm_training_batch(
                latents,
                noise,
                dtype=torch.bfloat16,
                device=device,
                t_min=t_min,
                t_max=t_max,
            )
            bs = bs_cfg if r["src"] == "scene" or (gbox and r["src"] == "grid") else 0.0
            for pt, tab in (("seed", seed_tab), ("moved", moved_tab)):
                set_table(tab)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = dit_forward(anima, noisy, ts, p["cache"], [r["caption"]], device)
                loss = box_share_fm_loss(pred, target, [r], bs, cap, n_cap, gbox)
                (gr,) = torch.autograd.grad(loss, raw)
                b = bank[(cell, pt)]
                b["sum_g"] += gr
                b["sum_g2"] += gr * gr
                nrm = gr.norm(dim=1)
                b["sum_norm"] += nrm
                b["touched"] += (nrm > 0).float()
                b["n"] += 1
                if pt == "seed":
                    halves[cell][k % 2] += gr
                n_reads += 2
            if (k + 1) % 64 == 0 or k + 1 == len(pick):
                print(
                    f"  bank {st}/{cell[1]}: {k + 1}/{len(pick)}, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )

    # -- dev: v_s at seed + paired loss at both tables -----------------------
    dev_v = {
        (s, b): torch.zeros(R, D, device=device) for s in dev_pick for b in range(len(BINS))
    }
    dev_n = defaultdict(int)
    dev_loss = defaultdict(list)  # (s, bin, "in"/"plain") → [(seed, moved)]
    for s, pick in dev_pick.items():
        for st, i in pick:
            p = per[st]
            r = p["sub"][p["pos"][i]]
            latents = p["lat"][[p["pos"][i]]].to(device)
            for b, (t_min, t_max) in enumerate(BINS):
                for _ in range(args.dev_noises):
                    noise = torch.randn(
                        latents.shape, generator=g, device=device, dtype=latents.dtype
                    )
                    noisy, ts, target = fm_training_batch(
                        latents,
                        noise,
                        dtype=torch.bfloat16,
                        device=device,
                        t_min=t_min,
                        t_max=t_max,
                    )
                    vals = {}
                    for pt, tab, grad in (("seed", seed_tab, True), ("moved", moved_tab, False)):
                        set_table(tab)
                        with torch.set_grad_enabled(grad):
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                pred = dit_forward(
                                    anima, noisy, ts, p["cache"], [r["caption"]], device
                                )
                            se = (pred.float() - target.float()) ** 2
                            m = box_mask(se.shape, [r], se.device)
                            per_cell = se.mean(dim=1, keepdim=True)
                            loss_in = (per_cell * m).sum() / m.sum().clamp(min=1)
                            loss_plain = se.mean()
                        vals[pt] = (float(loss_in.detach()), float(loss_plain.detach()))
                        if grad:
                            (gr,) = torch.autograd.grad(loss_in, raw)
                            dev_v[(s, b)] += gr
                            dev_n[(s, b)] += 1
                    dev_loss[(s, b, "in")].append((vals["seed"][0], vals["moved"][0]))
                    dev_loss[(s, b, "plain")].append((vals["seed"][1], vals["moved"][1]))
                    n_reads += 2
        print(f"  dev {s}: done, {(time.time() - t0) / 60:.1f} min", flush=True)
    set_table(seed_tab)
    print(f"{n_reads} reads in {(time.time() - t0) / 60:.1f} min", flush=True)

    # -- compute ---------------------------------------------------------------
    def dot(a, b):
        return float((a.double().flatten() @ b.double().flatten()))

    eps = 1e-8
    label_of = {}
    for s, ids in ev_ext_all.items():
        for i in ids:
            label_of[int(i)] = s
    own_rows = {s: [j for j, e in enumerate(ext_ids) if label_of.get(e) == s] for s in dev_pick}

    v_pooled = {}
    e1 = {}
    for s in dev_pick:
        vs = [dev_v[(s, b)] / max(dev_n[(s, b)], 1) for b in range(len(BINS))]
        v = torch.stack(vs).mean(0)
        v_pooled[s] = v
        own = own_rows[s]
        tot = float((v.double() ** 2).sum())
        e1[s] = float((v[own].double() ** 2).sum()) / max(tot, 1e-30) if own else 0.0

    e2 = {}
    for s in dev_pick:
        pred = -dot(v_pooled[s], delta_tab)  # improvement if positive
        rows_e2 = {"pred_improve": pred, "bins": {}}
        agree_all, meas_all = [], []
        for b in range(len(BINS)):
            pairs = dev_loss[(s, b, "in")]
            meas = [sd - mv for sd, mv in pairs]  # positive = moved improved
            mmean = sum(meas) / len(meas)
            pred_b = -dot(dev_v[(s, b)] / max(dev_n[(s, b)], 1), delta_tab)
            rows_e2["bins"][f"{BINS[b][0]}-{BINS[b][1]}"] = {
                "pred_improve": pred_b,
                "meas_improve": mmean,
                "n": len(meas),
                "sign_agree": (pred_b > 0) == (mmean > 0),
            }
            agree_all.append((pred_b > 0) == (mmean > 0))
            meas_all += meas
        rows_e2["meas_improve"] = sum(meas_all) / len(meas_all)
        rows_e2["sign_agree"] = (pred > 0) == (rows_e2["meas_improve"] > 0)
        plain = [sd - mv for b in range(len(BINS)) for sd, mv in dev_loss[(s, b, "plain")]]
        rows_e2["meas_improve_plain"] = sum(plain) / len(plain)
        e2[s] = rows_e2

    e3 = {}
    drift = {}
    for cell in bank_pick:
        key = f"{cell[0]}/{cell[1]}"
        bs_, bm_ = bank[(cell, "seed")], bank[(cell, "moved")]
        gbar = bs_["sum_g"] / max(bs_["n"], 1)
        vhat = bs_["sum_g2"] / max(bs_["n"], 1)
        adam = gbar / (vhat.sqrt() + eps)
        gbar_m = bm_["sum_g"] / max(bm_["n"], 1)
        drift[key] = float(
            F.cosine_similarity(gbar.flatten(), gbar_m.flatten(), dim=0)
        )
        h0, h1 = halves[cell][0], halves[cell][1]
        e3[key] = {
            "n": bs_["n"],
            "g_norm": float(gbar.norm()),
            "coh": float((h0 + h1).norm()) / max(float(bs_["sum_norm"].sum()), 1e-30),
            "half": float(F.cosine_similarity(h0.flatten(), h1.flatten(), dim=0)),
            "I_raw": {s: dot(v_pooled[s], gbar) for s in dev_pick},
            "I_adam": {s: dot(v_pooled[s], adam) for s in dev_pick},
        }
        e3[key]["I_raw_mean"] = sum(e3[key]["I_raw"].values()) / len(dev_pick)
        e3[key]["I_adam_mean"] = sum(e3[key]["I_adam"].values()) / len(dev_pick)

    torch.save(
        {
            "ext_ids": ext_ids,
            "cells": {
                f"{c[0]}/{c[1]}|{pt}": {
                    k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b.items()
                }
                for (c, pt), b in bank.items()
            },
            "dev_v": {f"{s}|{b}": v.cpu() for (s, b), v in dev_v.items()},
            "delta": delta_tab.cpu(),
            "args": vars(args),
        },
        work / "bank.pt",
    )

    # -- report ------------------------------------------------------------------
    rank = sorted(e3, key=lambda k: -e3[k]["I_raw_mean"])
    lines = [
        f"# influence_smoke — {args.label} ({time.strftime('%Y-%m-%d')})",
        "",
        f"Bank {n_bank} items × 2 tables, dev {n_dev} items × {len(BINS)} bins × "
        f"{args.dev_noises} noises; table {R} rows (moved overlays {n_moved}); "
        f"{(time.time() - t_start) / 60:.0f} min.",
        "",
        "## E1 — v_s concentration (‖v_s‖² share on s's own rows)",
        "",
        "| string | share |",
        "|---|---|",
        *[f"| {s} | {e1[s]:.3f} |" for s in dev_pick],
        "",
        "## E2 — linearization: pred vs measured improvement (seed → 300f table, in-box loss)",
        "",
        "Positive = the 300f delta reduced dev loss. `plain` = full-canvas loss, measured only.",
        "",
        "| string | pred | meas | sign | meas plain | per-bin sign (0.3-0.5 / 0.5-0.7 / 0.7-0.9) |",
        "|---|---|---|---|---|---|",
    ]
    for s in dev_pick:
        r = e2[s]
        bins = " / ".join(
            "✓" if r["bins"][f"{lo}-{hi}"]["sign_agree"] else "✗" for lo, hi in BINS
        )
        lines.append(
            f"| {s} | {r['pred_improve']:+.3e} | {r['meas_improve']:+.3e} | "
            f"{'✓' if r['sign_agree'] else '✗'} | {r['meas_improve_plain']:+.3e} | {bins} |"
        )
    lines += [
        "",
        "## E3 — the price with direction (at the seed table)",
        "",
        "`I_raw` = v_sᵀ ḡ_c per draw (positive: one draw of c is expected to reduce dev "
        "loss on s); `I_adam` = v_sᵀ (m̂/√v̂); `drift` = cos(ḡ_c seed, ḡ_c moved).",
        "",
        "| cell | n | ‖ḡ‖ | coh | half | I_raw mean | I_adam mean | drift | "
        + " | ".join(dev_pick)
        + " |",
        "|---|---|---|---|---|---|---|---|" + "---|" * len(dev_pick),
    ]
    for key in rank:
        c = e3[key]
        lines.append(
            f"| {key} | {c['n']} | {c['g_norm']:.3g} | {c['coh']:.2f} | {c['half']:+.2f} | "
            f"{c['I_raw_mean']:+.3e} | {c['I_adam_mean']:+.3e} | {drift[key]:+.2f} | "
            + " | ".join(f"{c['I_raw'][s]:+.2e}" for s in dev_pick)
            + " |"
        )
    lines += [
        "",
        f"Ranking by `I_raw mean`: {' > '.join(rank)}.",
        "",
        "Ruler expectation: scene_piece cells above scene_sentence / scene_short / "
        "grid_string (micro_warm_0923 bought hits scene_piece-only; the 300f mix did "
        "not). A ranking that disagrees fails the smoke; E2 sign disagreement fails "
        "the linearization the whole idea rests on. Note the caveat in the module "
        "docstring: E2 pass + E3 fail = the loss itself does not track the ruler.",
        "",
        f"Bank tensors: `{work / 'bank.pt'}`.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={
            "e1_concentration": e1,
            "e2_linearization": e2,
            "e3_cells": e3,
            "drift": drift,
            "ranking": rank,
            "n_reads": n_reads,
            "minutes": (time.time() - t_start) / 60,
        },
        artifacts=["report.md"],
        extra={"bank": str(work / "bank.pt")},
    )
    print(f"result: {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
