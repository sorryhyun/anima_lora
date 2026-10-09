#!/usr/bin/env python
"""probe_pres_train — the page-preservation term trained, on 16 hiragana
(2026-10-06)

The question (user, 10-06, after ``reports/probe_pres_2026_10_06.md``): does
L_pres make the rows carry the glyph and leave the page outside the text
alone? ``probe_pres`` read its gradient at f0's start rows (coherent, shared
across rows, not the stick; tiny at σ ≤ 0.7); here it steps them.

    loss = box_share_fm_loss (item's band) + λ · L_pres (σ ~ U(--pres_band))
    L_pres = mean_out (v_θ(x_σ, c_JA; rows) − sg[v_base(x_σ, c_EN)])²

(``probe_pres.en_caption`` / ``out_mask``: the EN line of the JA string's
length bin, the text box dilated by ``DIL``). The pres pass is a second
forward on the same items at its own σ / ε, from its own generator, so the
data term's draws are the plain arm's draw for draw.

Hiragana rows (a kana row sits in most lines; ``probe_pres``'s common
direction was twice the kanji's, 0.24 vs 0.12). The eval reads σ 0.95 apart:
there the term pulled against the in-box term (kana −0.03, kanji −0.12).

- ``data`` (CPU): the run's table (its ``data_from``'s rows, shares, lines,
  pack) at ``--n_items`` → ``…/<label>/data``; ``select`` / ``run`` / ``read``
  read a label's own build when it has one, else the run's data dir.
- ``select`` (CPU): ``--rows`` hiragana of f0's (small kana out) at items
  holding the glyph in ``--band``, evenly spaced by count; every item holding
  one; the multi-glyph texts split train / held out (``--hold``), the held
  items cut to ``--n_held`` by a text hash; the ``--steps`` batches in the
  trainer's order → ``…/<label>/select.json``. ``--keep <label> --top n``
  picks that label's rows and the ``n`` most frequent hiragana besides.
- ``run --arm <name> --lam λ`` (GPU, one arm a job): f0's data dir and start
  rows (``seed_fixed_1005_stick080``), only the picked rows live, every
  other row frozen there; plain AdamW (the trainer's betas, wd 0, constant
  ``--lr``) on the selected batches. The text cache is the probe's own
  (``…/<label>/te``: the batches' and held items' JA and EN captions — the
  data dir's whole-run cache is not touched). Then the held items:

  - ``band``: the box-share loss and its in-box term at each item's band
    (``--evals`` draws, as ``probe_accum``);
  - ``hi``: at σ 0.8 / 0.9 / 0.95 each, L_pres per item and the in-box term
    (the text's cost where the layout is set);

  the start rows' evals once per label (``eval_start.json``) →
  ``…/<label>/<arm>/{rows.pt, eval.json, train_log.json}``.
- ``read`` (CPU): each arm against start and plain, paired per item-draw:
  band in-box, hi L_pres and in-box by σ, per-row sign over the items holding
  the row; geometry — displacement, Δ(arm − plain) split into the rows'
  common part and the per-row rest, the common part's cos with
  ``probe_pres``'s common pres step (its ``s080`` grads, σ ≥ 0.8) and with
  the hiragana stick → ``…/<label>/read.json``.

    .venv/bin/python project/cjk_anima_reseed/probes/probe_pres_train.py select --label h16
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_pres_train.py run --label h16 --arm plain"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_pres_train.py run --label h16 --arm p5 --lam 5"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_pres_train.py read --label h16

h32 (10-06): h16's rows + the 16 most frequent, on its own 7 500-item build::

    .venv/bin/python project/cjk_anima_reseed/probes/probe_pres_train.py data --label h32 --n_items 7500
    .venv/bin/python project/cjk_anima_reseed/probes/probe_pres_train.py select --label h32 --keep h16 --top 16
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_pres_train.py run --label h32 --arm plain"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_pres_train.py run --label h32 --arm p10 --lam 10"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_pres_train.py run --label h32 --arm p10c09 --lam 10 --pres_band 0.8,0.9"

Renders: the ruler, the arms' rows.pt on stick080's rows (``--rows_pt``)::

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run --pack punct --rows_pt h16_plain=output/cjk_anima_reseed/probe_pres_train/h16/plain/rows.pt,h16_p5=output/cjk_anima_reseed/probe_pres_train/h16/p5/rows.pt --arms seed_fixed_1005_stick080@punct,h16_plain@punct,h16_p5@punct --label h16"
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

PROBE = OUT / "probe_pres_train"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"
BETAS = (0.9, 0.99)  # cjk_scale.train's AdamW
SMALL = set("ぁぃぅぇぉっゃゅょゎゕゖ")
HI_SIGMAS = (0.8, 0.9, 0.95)
PRES_SEED = 7_000  # the pres pass's generator, apart from the data term's


def _hira(c: str) -> bool:
    return "ぁ" <= c <= "ゖ"


def _h(s: str) -> int:
    return zlib.crc32(s.encode())


def _data_dir(run, label: str) -> Path:
    """The label's own build (``data``) when it has one, else the run's."""
    own = PROBE / label / "data"
    return own if (own / "train.jsonl").exists() else run.data


def build_data(run_name: str, label: str, n_items: int, workers) -> Path:
    """The run's table at ``n_items`` (its data run's rows, shares, lines and
    pack: f0's ``data_from``) → ``…/<label>/data``."""
    from dataclasses import replace

    from reseed import table as TB
    from reseed.builder import build
    from reseed.config import load

    run = load(run_name)
    src = load(run.data_from) if run.data_from else run
    assert "/" not in (run.data_from or ""), "a reseed run's build"
    assert all(r.startswith("chars:") for r in src.rows), src.rows
    n_rows = len({c for r in src.rows for c in r[len("chars:") :]})
    full = sum(TB.ITEMS_PER_ROW * n_rows * t.share for t in src.table())
    src.use_pack()
    b = replace(src, name=f"{PROBE.name}/{label}")
    assert b.data == PROBE / label / "data", b.data
    print(f"{src.name}'s table: {full:.0f} items in full → {n_items}", flush=True)
    return build(b, workers, n_items / full)


def select(
    run_name: str,
    label: str,
    n_rows: int,
    band: tuple,
    hold: float,
    n_held: int,
    steps: int,
    keep: str = "",
    top: int = 0,
) -> dict:
    from types import SimpleNamespace

    from cjk_scale import train as T
    from probe_geom import _char_rows
    from reseed.config import load

    run = load(run_name)
    data = _data_dir(run, label)
    recs = [
        json.loads(ln)
        for ln in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    count: dict = {}
    for r in recs:
        for c in set(r["text"]):
            if _hira(c) and c not in SMALL:
                count[c] = count.get(c, 0) + 1
    lo, hi = band
    c2r = _char_rows()
    if keep:  # another label's rows + the ``top`` most frequent hiragana besides
        picked = json.loads((PROBE / keep / "select.json").read_text("utf-8"))["chars"]
        rest = sorted(
            ((n, c) for c, n in count.items() if c not in picked and c in c2r),
            reverse=True,
        )
        assert len(rest) >= top, f"{len(rest)} hiragana besides {keep}'s, want {top}"
        picked = picked + [c for _n, c in rest[:top]]
    else:
        elig = sorted((n, c) for c, n in count.items() if lo <= n <= hi)
        assert len(elig) >= n_rows, f"{len(elig)} hiragana in {band}, want {n_rows}"
        step = len(elig) / n_rows
        picked = [elig[int(k * step)][1] for k in range(n_rows)]
    assert all(c in c2r for c in picked), [c for c in picked if c not in c2r]
    on = set(picked)
    items = [i for i, r in enumerate(recs) if on & set(r["text"])]
    held_all = [
        i
        for i in items
        if len(recs[i]["text"]) > 1 and _h(recs[i]["text"]) % 1000 < hold * 1000
    ]
    hs = set(held_all)
    train = [i for i in items if i not in hs]
    held = sorted(sorted(held_all, key=lambda i: (_h(f"{i}"), i))[:n_held])

    # the batches, in the trainer's order over the train items (shape-batched)
    shapes = {i: "x".join(map(str, recs[i]["shape"])) for i in train}

    class Sub:
        row_of = True

        @staticmethod
        def shape_of(k):
            return shapes[train[k]]

    from train.stage import Batcher

    ns = SimpleNamespace(seed=T.SEED, batch=T.BATCH, train_size=512)
    b = Batcher(ns, [recs[i] for i in train], Sub)
    order = [[train[k] for k in b.next(s)] for s in range(1, steps + 1)]
    drawn = [i for bb in order for i in bb]
    draws = {c: sum(c in recs[i]["text"] for i in drawn) for c in picked}
    sel = {
        "run": run_name,
        "data": str(data),
        "chars": picked,
        "rows": [int(c2r[c]) for c in picked],
        "items_per_char": {c: count[c] for c in picked},
        "draws_per_char": draws,
        "n_train": len(train),
        "batches": order,
        "held": held,
        "band": list(band),
        "hold": hold,
        "steps": steps,
    }
    out = PROBE / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "select.json").write_text(json.dumps(sel, ensure_ascii=False))
    print(
        f"{len(picked)} hiragana ({''.join(picked)}) in {data}\n"
        f"items holding one: {len(items)} → train {len(train)}, held out "
        f"{len(held_all)} → {len(held)} kept ({len({recs[i]['text'] for i in held})} texts)\n"
        f"{steps} batches: {len(set(drawn))} items; draws per row: "
        + " ".join(f"{c}{draws[c]}" for c in picked)
        + f"\n→ {out / 'select.json'}",
        flush=True,
    )
    return sel


def run(
    label: str,
    arm: str,
    lam: float,
    pres_band: tuple,
    lr: float,
    evals: int,
) -> None:
    import os
    import time
    from types import SimpleNamespace

    import torch
    from cjk_scale import train as T
    from cjk_scale.loss import box_share_fm_loss
    from cjk_scale.rows import Rows
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
    from library.runtime.harness import compile_blocks_for_training
    from probe_accum import item_terms
    from probe_pres import en_caption, out_mask
    from reseed.config import load
    from train.stage import LatentStore

    root = PROBE / label
    out = root / arm
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((root / "select.json").read_text(encoding="utf-8"))
    assert lam == 0 or arm != "plain", "the plain arm is λ 0"
    run_ = load(sel["run"])
    run_.use_pack()
    start = REPO / STICK080
    data = Path(sel.get("data") or run_.data)
    recs_all, _ev, vocabs = T.load_items(data)
    order = sel["batches"]
    held = sel["held"]
    keep = sorted({i for b in order for i in b} | set(held))
    pos = {i: k for k, i in enumerate(keep)}
    recs = [recs_all[i] for i in keep]
    assert all(r["src"] == "scene" for r in recs), "a scene table: the box share is on"
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    device = get_generation_settings(args).device

    # the probe's own text cache: every arm encodes the same caption set (JA
    # and EN for every kept item), so the first arm writes it and the rest reuse
    en_of = {k: en_caption(recs[k], keep[k]) for k in range(len(recs))}
    ja = sorted({r["caption"] for r in recs})
    cache = encode_captions(ja + sorted(set(en_of.values())), device, root / "te")
    touched = ext_ids_of({c: cache[c] for c in ja})
    print(
        f"text: {len(ja)} JA + {len(set(en_of.values()))} EN captions for "
        f"{len(recs)} items",
        flush=True,
    )
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
    assert int(live.sum()) == len(live_ids), (int(live.sum()), len(live_ids))
    raw0 = raw.detach().clone()
    anima.train()
    compile_blocks_for_training(
        anima, None, backend="inductor", n_token_families=lat.n_families
    )

    def at_sigma(latents, s, gen=None):
        noise = (
            torch.randn(
                latents.shape, generator=gen, device=device, dtype=torch.float32
            )
            if gen is not None
            else torch.randn_like(latents.float())
        )
        sv = s.view(-1, *([1] * (latents.dim() - 1)))
        noisy = ((1.0 - sv) * latents.float() + sv * noise).to(torch.bfloat16)
        return noisy, noise - latents.float()

    def pres_terms(ks, latents, s, gen=None):
        """(L_pres per item, student pred, target) at σ ``s`` (B,)."""
        brecs = [recs[k] for k in ks]
        noisy, target = at_sigma(latents, s, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teach = (
                dit_forward(anima, noisy, s, cache, [en_of[k] for k in ks], device)
                .detach()
                .float()
            )
            pred = dit_forward(
                anima, noisy, s, cache, [r["caption"] for r in brecs], device
            )
        mo = out_mask(pred.shape, brecs, pred.device, T.GRID_BOX)
        per_cell = ((pred.float() - teach) ** 2).mean(dim=1, keepdim=True)
        val = (per_cell * mo).sum(dim=(1, 2, 3)) / mo.sum(dim=(1, 2, 3)).clamp(min=1)
        return val, pred, target

    # ---- train
    torch.manual_seed(T.SEED)
    gen = torch.Generator(device=device).manual_seed(PRES_SEED)
    opt = torch.optim.AdamW([raw], lr=lr, weight_decay=0.0, betas=BETAS)
    lo_p, hi_p = pres_band
    log = []
    t0 = time.time()
    for step, b in enumerate(order, start=1):
        ks = [pos[i] for i in b]
        brecs = [recs[k] for k in ks]
        latents = lat[ks].to(device)
        noise = torch.randn_like(latents)
        noisy, ts, target = T.noisy_by_band(
            latents, noise, [tuple(r["band"]) for r in brecs], device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dit_forward(
                anima, noisy, ts, cache, [r["caption"] for r in brecs], device
            )
        loss = box_share_fm_loss(
            pred,
            target,
            brecs,
            T.BOX_SHARE,
            T.BOX_SHARE_CAP,
            T.BOX_SHARE_GLYPHS,
            T.GRID_BOX,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        del pred
        l_pres = None
        if lam > 0:  # second graph after the first is freed: same peak memory
            s = lo_p + (hi_p - lo_p) * torch.rand(len(ks), generator=gen, device=device)
            val, _pp, _tt = pres_terms(ks, latents, s, gen)
            l_pres = val.mean()
            (lam * l_pres).backward()
            del _pp, _tt
        opt.step()
        if step % 100 == 0 or step == 1:
            d = ((raw.detach() - raw0)[live] * rows.row_scale).norm(dim=1)
            rec = {
                "step": step,
                "loss": float(loss.detach()),
                "pres": None if l_pres is None else float(l_pres.detach()),
                "disp_mean": float(d.mean()),
                "s_per_step": (time.time() - t0) / step,
            }
            log.append(rec)
            print(f"{arm} {rec}", flush=True)
    train_min = (time.time() - t0) / 60

    # ---- eval
    hk = [pos[i] for i in held]
    by: dict = {}
    for k in hk:
        by.setdefault(lat.shape_of(k), []).append(k)
    ebatches = [
        ks[j : j + T.BATCH]
        for _s, ks in sorted(by.items())
        for j in range(0, len(ks) - T.BATCH + 1, T.BATCH)
    ]

    def evaluate(tag: str) -> dict:
        band, hi = [], []
        for e in range(evals):
            for bi, ks in enumerate(ebatches):
                torch.manual_seed(100_000 * (e + 1) + bi)
                brecs = [recs[k] for k in ks]
                latents = lat[ks].to(device)
                noise = torch.randn_like(latents)
                noisy, ts, target = T.noisy_by_band(
                    latents, noise, [tuple(r["band"]) for r in brecs], device
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = dit_forward(
                        anima, noisy, ts, cache, [r["caption"] for r in brecs], device
                    )
                lo, li = (x.detach() for x in item_terms(pred, target, brecs))
                band += [
                    {
                        "item": keep[k],
                        "e": e,
                        "sigma": float(ts.flatten()[j]),
                        "loss": float(lo[j]),
                        "in": float(li[j]),
                    }
                    for j, k in enumerate(ks)
                ]
                del pred, lo, li
        for si, sg in enumerate(HI_SIGMAS):
            for bi, ks in enumerate(ebatches):
                g = torch.Generator(device=device).manual_seed(
                    200_000 + 1_000 * si + bi
                )
                brecs = [recs[k] for k in ks]
                latents = lat[ks].to(device)
                s = torch.full((len(ks),), sg, device=device)
                val, pred, target = pres_terms(ks, latents, s, g)
                # floats out before the next forward: a live graph doubles the peak
                val = val.detach()
                li = item_terms(pred, target, brecs)[1].detach()
                del pred
                hi += [
                    {
                        "item": keep[k],
                        "sigma": sg,
                        "pres": float(val[j]),
                        "in": float(li[j]),
                    }
                    for j, k in enumerate(ks)
                ]
                del val, li
        print(f"eval {tag}: {len(band)} band / {len(hi)} hi item-draws", flush=True)
        return {"band": band, "hi": hi}

    f_start = root / "eval_start.json"
    if not f_start.exists():
        trained = raw.detach().clone()
        with torch.no_grad():
            raw.copy_(raw0)
        f_start.write_text(json.dumps(evaluate("start")))
        with torch.no_grad():
            raw.copy_(trained)
    ev = evaluate(arm)
    ext = [int(e) for e in rows.delta.ext_ids]
    lv = live.tolist()
    torch.save(
        {
            "ext_ids": [e for e, x in zip(ext, lv) if x],
            "chars": [
                next(c for c, r in zip(sel["chars"], sel["rows"]) if r == e)
                for e, x in zip(ext, lv)
                if x
            ],
            "row_scale": rows.row_scale,
            "start": raw0[live].float().cpu(),
            "raw": raw.detach()[live].float().cpu(),
            "config": {
                "arm": arm,
                "lam": lam,
                "pres_band": list(pres_band),
                "lr": lr,
                "evals": evals,
                "steps": len(order),
                "train_min": train_min,
            },
        },
        out / "rows.pt",
    )
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "eval.json").write_text(json.dumps(ev))
    print(f"→ {out} (train {train_min:.1f} min)", flush=True)


def read(label: str, s080: str) -> None:
    import numpy as np
    import torch
    from probe_geom import _families
    from scipy.stats import binomtest

    root = PROBE / label
    sel = json.loads((root / "select.json").read_text(encoding="utf-8"))
    from reseed.config import load

    lines = (Path(sel.get("data") or load(sel["run"]).data) / "train.jsonl").read_text(
        encoding="utf-8"
    )
    hs = set(sel["held"])
    text = {
        i: json.loads(ln)["text"] for i, ln in enumerate(lines.splitlines()) if i in hs
    }
    ev0 = json.loads((root / "eval_start.json").read_text(encoding="utf-8"))
    arms = sorted(  # an arm: eval.json + rows.pt (the label's data/ has an eval.json)
        d.name
        for d in root.iterdir()
        if (d / "eval.json").exists() and (d / "rows.pt").exists()
    )
    assert "plain" in arms, f"no plain arm under {root}"
    E = {a: json.loads((root / a / "eval.json").read_text()) for a in arms}
    R = {a: torch.load(root / a / "rows.pt", weights_only=False) for a in arms}
    chars = R["plain"]["chars"]
    ids = R["plain"]["ext_ids"]
    rs = float(R["plain"]["row_scale"])

    def keyed(ev, part):
        return {
            (r["item"], r.get("e", 0), r["sigma"] if part == "hi" else 0): r
            for r in ev[part]
        }

    def paired(a: str, b_ev: dict, part: str, m: str, sg=None) -> np.ndarray:
        A, B = keyed(E[a], part), keyed(b_ev, part)
        return np.array(
            [
                A[k][m] - B[k][m]
                for k in A
                if (sg is None or abs(k[2] - sg) < 1e-6) and k in B
            ]
        )

    def ms(x):
        return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))]

    def per_row_sign(a: str, b_ev: dict, part: str, m: str, sg=None) -> dict:
        A, B = keyed(E[a], part), keyed(b_ev, part)
        xs = []
        for c in chars:
            ds = [
                A[k][m] - B[k][m]
                for k in A
                if k in B and c in text[k[0]] and (sg is None or abs(k[2] - sg) < 1e-6)
            ]
            if ds:
                xs.append(float(np.mean(ds)))
        lower = sum(x < 0 for x in xs)
        return {
            "rows": len(xs),
            "lower": lower,
            "sign_p": float(binomtest(lower, len(xs)).pvalue) if xs else None,
        }

    # probe_pres's common pres step on these rows (σ ≥ 0.8)
    common_s080 = None
    gp = PROBE.parent / "probe_pres" / s080 / "grads.pt"
    if gp.exists():
        sd = torch.load(gp, map_location="cpu", weights_only=False)
        at = {e: k for k, e in enumerate(sd["ext_ids"])}
        want = {at[e]: j for j, e in enumerate(ids)}
        acc = {j: [] for j in range(len(ids))}
        for b in sd["batches"]:
            if b["sigma"] < 0.75:
                continue
            for jj, i in enumerate(b["rows"].tolist()):
                if i in want:
                    acc[want[i]].append(b["g_pres"][jj].numpy())
        U = [
            (lambda m: m / (np.linalg.norm(m) + 1e-30))(np.mean(v, 0))
            for v in acc.values()
            if len(v) >= 6
        ]
        common_s080 = -np.mean(U, 0) if U else None
    # the kana stick at the start rows (the family's mean offset, as probe_pres)
    st = torch.load(REPO / STICK080, map_location="cpu", weights_only=False)["delta"]
    off = st["raw"].float().numpy() * float(st["row_scale"])
    stick = off[_families([int(e) for e in st["ext_ids"]])["kana"]].mean(0)
    stick /= np.linalg.norm(stick)

    def cos(u, v) -> float:
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))

    start = R["plain"]["start"].numpy() * rs
    dplain = R["plain"]["raw"].numpy() * rs - start
    res = {"arms": {}}
    for a in arms:
        d = R[a]["raw"].numpy() * rs - start
        rec = {
            "config": R[a]["config"],
            "disp_median": float(np.median(np.linalg.norm(d, axis=1))),
            "vs_start": {
                "band_in": ms(paired(a, ev0, "band", "in")),
                **{
                    f"hi{sg}_{m}": ms(paired(a, ev0, "hi", m, sg))
                    for sg in HI_SIGMAS
                    for m in ("pres", "in")
                },
            },
        }
        if a != "plain":
            dd = d - dplain
            c = dd.mean(0)
            rest = dd - c
            rec["vs_plain"] = {
                "band_in": ms(paired(a, E["plain"], "band", "in")),
                "band_in_rows": per_row_sign(a, E["plain"], "band", "in"),
                **{
                    f"hi{sg}_{m}": ms(paired(a, E["plain"], "hi", m, sg))
                    for sg in HI_SIGMAS
                    for m in ("pres", "in")
                },
                **{
                    f"hi{sg}_pres_rows": per_row_sign(a, E["plain"], "hi", "pres", sg)
                    for sg in HI_SIGMAS
                },
            }
            rec["geometry"] = {
                # share of Δ(arm − plain)'s energy in the rows' common vector
                "common_share": float(len(dd) * (c @ c) / ((dd * dd).sum() + 1e-30)),
                "common_norm": float(np.linalg.norm(c)),
                "rest_norm_median": float(np.median(np.linalg.norm(rest, axis=1))),
                "cos_common_s080": None if common_s080 is None else cos(c, common_s080),
                "cos_common_stick": cos(c, stick),
                "cos_move_plain_median": float(
                    np.median([cos(d[j], dplain[j]) for j in range(len(d))])
                ),
            }
        res["arms"][a] = rec
    (root / "read.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["data", "select", "run", "read"])
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument("--label", default="h16")
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--band", default="2000,14000", help="items per hiragana, lo,hi")
    p.add_argument("--keep", default="", help="select: this label's rows, plus --top")
    p.add_argument("--top", type=int, default=0, help="select: most frequent besides")
    p.add_argument("--n_items", type=int, default=7500, help="data: the build's size")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--hold", type=float, default=0.15)
    p.add_argument("--n_held", type=int, default=400)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--arm", default="plain")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--pres_band", default="0.8,0.95")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--evals", type=int, default=2)
    p.add_argument("--s080", default="s080", help="probe_pres label for the read")
    a = p.parse_args()
    if a.verb == "data":
        build_data(a.run, a.label, a.n_items, a.workers)
    elif a.verb == "select":
        lo, hi = (int(x) for x in a.band.split(","))
        select(
            a.run, a.label, a.rows, (lo, hi), a.hold, a.n_held, a.steps, a.keep, a.top
        )
    elif a.verb == "run":
        pb = tuple(float(x) for x in a.pres_band.split(","))
        run(a.label, a.arm, a.lam, pb, a.lr, a.evals)
    else:
        read(a.label, a.s080)


if __name__ == "__main__":
    main()
