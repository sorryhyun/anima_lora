#!/usr/bin/env python
"""probe_scene — how much of a row's gradient the scene sets (2026-10-06)

The question (user, 10-06, after ``reports/probe_cf_2026_10_06.md``): the
row's per-draw gradient scatters across items; how much of that is the
scene? No training: the rows sit at ``--rows`` and the gradient is read.

A crossed design per target row: ``--lines`` dialogue lines that hold the
glyph once × ``--scenes`` scenes × ``--noises`` noise draws, every cell
rendered and read. "Scene" is what an item's scene is at training: the
image, its tag caption (``scene_caption``) and the layout ``_sent_plan``
fits there. Held fixed: one canvas shape (so noise draw k is the same σ_k,
ε_k in every cell), one font per row, the font px target (``--px``).

- ``render`` (CPU): per row, candidate lines and scenes of ``--shape``, the
  feasible (line, scene) renders, a full ``lines × scenes`` block of them
  picked greedily → ``output/cjk_anima_reseed/probe_scene/<label>/``
  ``cells.jsonl`` + PNGs, ``sheet.png``.
- ``grad`` (GPU): VAE latents, the captions encoded, then per cell and noise
  draw a batch-1 forward under the cell's caption, ``box_share_fm_loss`` on
  the drawn box, the target row's gradient from each of its two summands
  (in-box, out-box) → ``grads.pt``.
- ``read`` (CPU): per row, a three-way random-effects ANOVA on the gradient
  vectors (line, scene, noise and their interactions; variance summed over
  the dims), the row's common direction against it, and the mean cos
  between draws by what they share; four views — the box-share sum, the
  in-box term, the out-box term, plain MSE rebuilt from the two by area →
  ``read.json``.

    .venv/bin/python project/cjk_anima_reseed/probes/probe_scene.py render --label sc1
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_scene.py grad --label sc1"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_scene.py read --label sc1
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()

from reseed import OUT  # noqa: E402

PROBE = OUT / "probe_scene"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"
BAND = (0.45, 0.7)  # sent_34's


def _is_kana(c: str) -> bool:
    return "ぁ" <= c <= "ヿ"


def _is_kanji(c: str) -> bool:
    return "一" <= c <= "鿿"


def render(
    run_name: str,
    label: str,
    n_kana: int,
    n_kanji: int,
    n_lines: int,
    n_scenes: int,
    shape: tuple,
    px: float,
) -> None:
    from types import SimpleNamespace

    from common.render.flat import find_fonts, font_covers
    from common.render.scene import render_into_scene
    from data.inventory import qwen_pieces
    from data.synth import load_scenes, scene_caption
    from probe_geom import _char_rows
    from reseed import table as T
    from reseed.config import load
    from reseed.pools import whole_bubbles
    from reseed.recipes import _sent_cuts, _sent_plan

    run = load(run_name)
    run.use_pack()
    out = PROBE / label
    (out / "img").mkdir(parents=True, exist_ok=True)
    texts = sorted(
        {
            json.loads(ln)["text"]
            for ln in (run.data / "train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln and '"sent' in ln
        }
    )
    c2r = _char_rows()
    texts = [
        t
        for t in texts
        if 8 <= len(t) <= 10 and all(c in c2r or not c.isalnum() for c in t)
    ]
    scenes = [
        s
        for s in whole_bubbles(load_scenes(T.SCENES, 0.0, 0, "", T.ONE_BUBBLE))
        if tuple(s["shape"]) == shape and s.get("frame") not in T.SENT_FRAMES_OUT
    ]
    fonts = find_fonts()
    tokq = SimpleNamespace(tokq=qwen_pieces(char_rows=True))
    min_glyph = int(0.85 * px)
    by: dict = {}
    for t in texts:
        for c in set(t):
            if c in c2r and t.count(c) == 1:
                by.setdefault(c, []).append(t)
    rng = random.Random(0)
    elig = {c: v for c, v in by.items() if len(v) >= 3 * n_lines}
    targets = [
        ("kana", c)
        for c in rng.sample(
            sorted(c for c in elig if _is_kana(c) and c != "ー"), n_kana
        )
    ]
    targets += [
        ("kanji", c)
        for c in rng.sample(sorted(c for c in elig if _is_kanji(c)), n_kanji)
    ]
    print(f"{len(texts)} lines of 8–10, {len(scenes)} scenes at {shape}", flush=True)

    cells = []
    for r_i, (fam, c) in enumerate(targets):
        lines = rng.sample(elig[c], min(len(elig[c]), 8 * n_lines))
        fok = [f for f in fonts if all(font_covers(f, t) for t in lines)]
        font = random.Random(r_i).choice(fok)
        # plans first (cheap): ~11 of 191 scenes take a given line at 30 px
        plans = {}
        for li, t in enumerate(lines):
            cuts = _sent_cuts(tokq, t)
            for si, sc in enumerate(scenes):
                got = _sent_plan(sc["region"], t, cuts, px, min_glyph)
                if got is not None:
                    plans[(li, si)] = (got, cuts)
        n_take = {
            si: sum((li, si) in plans for li in range(len(lines)))
            for si in range(len(scenes))
        }
        cand = sorted(
            (si for si in n_take if n_take[si] >= n_lines), key=lambda si: -n_take[si]
        )
        cand_sc = [scenes[si] for si in cand[: 4 * n_scenes]]
        feas = {}  # (line, scene) → (image, box, caption)
        for li, t in enumerate(lines):
            for si, sc in enumerate(cand_sc):
                got = plans.get((li, cand[si]))
                if got is None:
                    continue
                (k, f), cuts = got
                drawn = render_into_scene(
                    scene=sc,
                    text=t,
                    font_path=font,
                    rng=random.Random(1000 * li + si),
                    min_glyph=min_glyph,
                    stroke=False,
                    fill_frac=f,
                    max_lines=k,
                    cuts=cuts,
                    vertical_only=True,
                    fewest_lines=True,
                    tategaki=True,
                    vert_forms=True,
                    keep_outline=True,
                )
                if drawn is not None:
                    feas[(li, si)] = (*drawn, scene_caption(sc, t))
        # a full block: scenes by how many lines they take, then the lines all of them take
        sc_order = sorted(
            range(len(cand_sc)),
            key=lambda s: -sum((li, s) in feas for li in range(len(lines))),
        )
        pick_s, pick_l = [], list(range(len(lines)))
        for s in sc_order:
            keep = [li for li in pick_l if (li, s) in feas]
            if len(keep) >= n_lines:
                pick_s.append(s)
                pick_l = keep
            if len(pick_s) == n_scenes:
                break
        if len(pick_s) < n_scenes:
            print(
                f"  {c}: no {n_lines}×{n_scenes} block ({len(cand)} scenes take ≥ {n_lines} lines), skipped",
                flush=True,
            )
            continue
        pick_l = pick_l[:n_lines]
        for a, li in enumerate(pick_l):
            for b, si in enumerate(pick_s):
                im, box, cap = feas[(li, si)]
                f = out / "img" / f"r{r_i:02d}_l{a}_s{b}.png"
                im.save(f)
                cells.append(
                    {
                        "row_i": r_i,
                        "char": c,
                        "family": fam,
                        "row": c2r[c],
                        "line": a,
                        "scene": b,
                        "text": lines[li],
                        "scene_id": [cand_sc[si]["pool"], cand_sc[si]["i"]],
                        "caption": cap,
                        "box": list(box),
                        "file": str(f),
                    }
                )
        print(f"  {c} ({fam}): {len(feas)} feasible renders, block kept", flush=True)
    (out / "cells.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cells),
        encoding="utf-8",
    )
    print(
        f"→ {out / 'cells.jsonl'}: {len(cells)} cells, {len({x['row_i'] for x in cells})} rows",
        flush=True,
    )
    if cells:
        _sheet(cells, out / "sheet.png")


def _sheet(cells, path: Path) -> None:
    """The first row's block: lines down, scenes across."""
    from PIL import Image

    r0 = cells[0]["row_i"]
    blk = [c for c in cells if c["row_i"] == r0]
    L = max(c["line"] for c in blk) + 1
    S = max(c["scene"] for c in blk) + 1
    w, h = 160, 228
    sheet = Image.new("RGB", (w * S, h * L), "white")
    for c in blk:
        sheet.paste(
            Image.open(c["file"]).resize((w, h)), (c["scene"] * w, c["line"] * h)
        )
    sheet.save(path)
    print(f"→ {path}", flush=True)


def grad(run_name: str, rows_path: str, label: str, n_noises: int) -> None:
    import os
    import time

    import numpy as np
    import torch
    from cjk_scale import train as T
    from cjk_scale.rows import Rows
    from common.models import (
        checkpoints,
        dit_forward,
        encode_images,
        gen_args,
        load_vae,
    )
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from cjk_scale.loss import box_mask, box_share_of, glyph_count
    from probe_geom import split_terms
    from reseed import REPO
    from reseed.config import load
    from train.stage import _encode_text

    run = load(run_name)
    run.use_pack()
    rows_path = str(rows_path if Path(rows_path).is_absolute() else REPO / rows_path)
    out = PROBE / label
    cells = [
        json.loads(ln)
        for ln in (out / "cells.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    data = run.data
    recs, ev, vocabs = T.load_items(data)
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    device = get_generation_settings(args).device

    t0 = time.time()
    vae = load_vae(device)
    lat = encode_images(vae, [c["file"] for c in cells], device)
    del vae
    torch.cuda.empty_cache()
    print(f"latents: {tuple(lat.shape)} in {time.time() - t0:.0f}s", flush=True)

    cache, _, _ = _encode_text(cells, [], device, out)
    # the run's rows (its data's captions decide which are live)
    _, touched, _ = _encode_text(recs, ev, device, out, te_cache=data / "te_cache")
    plan = T.plan(run.scale_config(), data, recs, vocabs, touched, Path(rows_path))
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    rows = Rows(
        anima,
        device,
        plan.idx,
        strategy_pack(tok),
        warm=Path(rows_path),
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
        touched=plan.touched,
        frozen=plan.frozen,
        context=Path(rows_path),
    )
    anima.train()
    raw = rows.delta.raw
    at = {int(e): k for k, e in enumerate(rows.delta.ext_ids)}
    gen = torch.Generator(device="cpu").manual_seed(T.SEED)
    sigmas = np.linspace(BAND[0], BAND[1], n_noises + 2)[1:-1]
    eps = torch.randn((n_noises, *lat.shape[1:]), generator=gen)
    # the two box-share summands apart: in = s · mean_in, out = (1 − s) · mean_out
    G = np.zeros((2, len(cells), n_noises, raw.shape[1]), dtype=np.float32)
    shares = np.zeros((len(cells), 2), dtype=np.float32)  # s, in-box area share a
    t0 = time.time()
    for n, c in enumerate(cells):
        k = at[int(c["row"])]
        x0 = lat[n][None].to(device)
        rec = {"layout": "scene", "box": c["box"], "text": c["text"]}
        for e in range(n_noises):
            sig = float(sigmas[e])
            ep = eps[e][None].to(device)
            x = (1 - sig) * x0 + sig * ep
            ts = torch.full((1,), sig, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dit_forward(
                    anima, x.to(torch.bfloat16), ts, cache, [c["caption"]], device
                )
            l_in, l_out = split_terms(
                pred,
                ep - x0,
                [rec],
                T.BOX_SHARE,
                T.BOX_SHARE_CAP,
                T.BOX_SHARE_GLYPHS,
                T.GRID_BOX,
            )
            G[0, n, e] = (
                torch.autograd.grad(l_in, raw, retain_graph=True)[0][k]
                .float()
                .cpu()
                .numpy()
            )
            G[1, n, e] = torch.autograd.grad(l_out, raw)[0][k].float().cpu().numpy()
        m = box_mask(pred.shape, [rec], device, False)
        shares[n] = (
            box_share_of(
                glyph_count(c["text"]), T.BOX_SHARE, T.BOX_SHARE_CAP, T.BOX_SHARE_GLYPHS
            ),
            float(m.mean()),
        )
        if (n + 1) % 36 == 0 or n == 0:
            print(
                f"cell {n + 1}/{len(cells)}: {(time.time() - t0) / (n + 1) / n_noises:.2f} s/draw",
                flush=True,
            )
    torch.save(
        {
            "sigmas": sigmas.tolist(),
            "g_in": torch.from_numpy(G[0]),
            "g_out": torch.from_numpy(G[1]),
            "shares": torch.from_numpy(shares),
            "rows": rows_path,
        },
        out / "grads.pt",
    )
    print(f"→ {out / 'grads.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def anova3(X):
    """``X`` (L, S, E, d): a crossed three-way random-effects decomposition,
    one observation per cell, variance summed over the d dims. Returns the
    variance components (negative estimates kept) and the common direction's
    energy ‖μ̂‖² less its sampling share."""

    L, S, E, _ = X.shape
    m = X.mean((0, 1, 2))
    mL, mS, mE = X.mean((1, 2)), X.mean((0, 2)), X.mean((0, 1))
    mLS, mLE, mSE = X.mean(2), X.mean(1), X.mean(0)

    def ss(a):
        return float((a**2).sum())

    SS = {
        "line": S * E * ss(mL - m),
        "scene": L * E * ss(mS - m),
        "noise": L * S * ss(mE - m),
        "line×scene": E * ss(mLS - mL[:, None] - mS[None] + m),
        "line×noise": S * ss(mLE - mL[:, None] - mE[None] + m),
        "scene×noise": L * ss(mSE - mS[:, None] - mE[None] + m),
    }
    res = (
        X
        - mLS[:, :, None]
        - mLE[:, None]
        - mSE[None]
        + mL[:, None, None]
        + mS[None, :, None]
        + mE[None, None]
        - m
    )
    SS["resid"] = ss(res)
    df = {
        "line": L - 1,
        "scene": S - 1,
        "noise": E - 1,
        "line×scene": (L - 1) * (S - 1),
        "line×noise": (L - 1) * (E - 1),
        "scene×noise": (S - 1) * (E - 1),
        "resid": (L - 1) * (S - 1) * (E - 1),
    }
    MS = {k: SS[k] / df[k] for k in SS}
    V = {"resid": MS["resid"]}
    V["line×scene"] = (MS["line×scene"] - MS["resid"]) / E
    V["line×noise"] = (MS["line×noise"] - MS["resid"]) / S
    V["scene×noise"] = (MS["scene×noise"] - MS["resid"]) / L
    V["line"] = (MS["line"] - MS["line×scene"] - MS["line×noise"] + MS["resid"]) / (
        S * E
    )
    V["scene"] = (MS["scene"] - MS["line×scene"] - MS["scene×noise"] + MS["resid"]) / (
        L * E
    )
    V["noise"] = (MS["noise"] - MS["line×noise"] - MS["scene×noise"] + MS["resid"]) / (
        L * S
    )
    var_mean = (
        V["line"] / L
        + V["scene"] / S
        + V["noise"] / E
        + V["line×scene"] / (L * S)
        + V["line×noise"] / (L * E)
        + V["scene×noise"] / (S * E)
        + V["resid"] / (L * S * E)
    )
    return V, float(m @ m) - var_mean


def read(label: str) -> None:
    import numpy as np
    import torch

    out = PROBE / label
    sd = torch.load(out / "grads.pt", map_location="cpu", weights_only=False)
    gi, go = sd["g_in"].numpy(), sd["g_out"].numpy()
    s_, a_ = (sd["shares"].numpy()[:, j][:, None, None] for j in (0, 1))
    # ∇mean_in = g_in / s, ∇mean_out = g_out / (1 − s); plain MSE weighs them by area
    views = {
        "box_share": gi + go,
        "in": gi,
        "out": go,
        "plain_mse": a_ * gi / s_ + (1 - a_) * go / (1 - s_),
    }
    print(
        f"in-box share s {float(np.median(s_)):.3f}, box area share a {float(np.median(a_)):.3f}; "
        f"|out|/|in| per draw {float(np.median(np.linalg.norm(go, axis=-1) / np.linalg.norm(gi, axis=-1))):.3f}"
    )
    result = {}
    for vname, G in views.items():
        print(f"\n######## {vname}")
        result[vname] = _read_view(G, out)
    (out / "read.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))


def _read_view(G, out):
    import numpy as np

    cells = [
        json.loads(ln)
        for ln in (out / "cells.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    by: dict = {}
    for n, c in enumerate(cells):
        by.setdefault(c["row_i"], []).append((c, n))
    KEYS = [
        "signal",
        "line",
        "scene",
        "noise",
        "line×scene",
        "line×noise",
        "scene×noise",
        "resid",
    ]
    SHARE = ["", "l", "s", "e", "ls", "le", "se"]
    rows = []
    for r_i, cs in by.items():
        L = max(c["line"] for c, _ in cs) + 1
        S = max(c["scene"] for c, _ in cs) + 1
        X = np.zeros((L, S, G.shape[1], G.shape[2]))
        for c, n in cs:
            X[c["line"], c["scene"]] = G[n]
        V, sig = anova3(X)
        tot = sig + sum(V.values())
        rec = {"char": cs[0][0]["char"], "family": cs[0][0]["family"]}
        rec["share"] = {"signal": sig / tot, **{k: v / tot for k, v in V.items()}}
        # mean cos between two draws, by the factors they share
        U = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-30)
        acc = {k: [] for k in SHARE}
        idx = list(itertools.product(range(L), range(S), range(X.shape[2])))
        for a, b in itertools.combinations(idx, 2):
            k = (
                ("l" if a[0] == b[0] else "")
                + ("s" if a[1] == b[1] else "")
                + ("e" if a[2] == b[2] else "")
            )
            acc[k].append(float(U[a] @ U[b]))
        rec["cos"] = {k: float(np.mean(v)) for k, v in acc.items() if v}
        rows.append(rec)
    summary = {}
    for f in ("kana", "kanji", "all"):
        rr = [r for r in rows if f in ("all", r["family"])]
        if not rr:
            continue
        summary[f] = {
            "rows": len(rr),
            "share": {k: float(np.median([r["share"][k] for r in rr])) for k in KEYS},
            "cos_by_shared": {
                k or "none": float(np.median([r["cos"][k] for r in rr])) for k in SHARE
            },
        }
    for f, s in summary.items():
        print(f"\n== {f} ({s['rows']} rows)")
        print(
            "  variance share: "
            + "  ".join(f"{k} {v:+.3f}" for k, v in s["share"].items())
        )
        print(
            "  cos by shared:  "
            + "  ".join(f"{k} {v:+.3f}" for k, v in s["cos_by_shared"].items())
        )
    return {"summary": summary, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["render", "grad", "read"])
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument("--rows", default=STICK080)
    p.add_argument("--label", default="sc1")
    p.add_argument("--kana", type=int, default=8)
    p.add_argument("--kanji", type=int, default=8)
    p.add_argument("--lines", type=int, default=6)
    p.add_argument("--scenes", type=int, default=6)
    p.add_argument("--noises", type=int, default=4)
    p.add_argument("--shape", default="448x640")
    p.add_argument("--px", type=float, default=30.0)
    a = p.parse_args()
    if a.verb == "render":
        w, h = (int(x) for x in a.shape.split("x"))
        render(a.run, a.label, a.kana, a.kanji, a.lines, a.scenes, (w, h), a.px)
    elif a.verb == "grad":
        grad(a.run, a.rows, a.label, a.noises)
    else:
        read(a.label)


if __name__ == "__main__":
    main()
