#!/usr/bin/env python
"""probe_cf — counterfactual-input FM against plain FM, per draw, on the
warm rows (2026-10-06)

The question (user, 10-06, after ``reports/probe_geom_2026_10_06.md``): a
draw's plain-FM gradient on a warm row is ~97 % noise; does the CF loss
(``../finished/cjk_renderable_anima/idea.md``: the input noised from a
sibling render B, the caption A's, the target the straight line to A) carry
a larger per-draw signal share on the same rows? No training: the rows sit
at ``--rows`` and the gradient is read.

- ``pairs`` (CPU): target rows drawn from the run's ``sent`` items (the
  glyph once in the line), ``--per_row`` items each; every item re-lettered
  on its own scene with its own columns / fill / font px as a pair through
  ``render_into_scene(ref_text=…)`` — A′ = the line, B = the line with the
  target glyph replaced (same fit, font, positions; asserted to differ only
  inside the union box). B alternates per draw: ``swap``
  (another glyph of the target's family) and ``dup`` (the neighbour glyph
  repeated — the repeat mode) → ``output/cjk_anima_reseed/probe_cf/<label>/``
  ``pairs.jsonl`` + PNGs.
- ``grad`` (GPU): the pairs VAE-encoded, then per pair one (σ, ε), σ ~
  U(``--t_min``, ``--t_max``): plain = A′ noised, target ε − A′; CF = B
  noised with the same ε, target (x_σ − A′)/σ. Each a batch-1 forward
  under A's caption, ``box_share_fm_loss`` on the union box, the target
  row's gradient read. Also the CF leverage λ = 1 − ⟨r, d⟩ / ‖d‖² in the
  box, d = (A′ − B)/σ the residual of a model that believes B (λ 0: the
  input wins, 1: the caption) → ``grads.pt``.
- ``read`` (CPU): per row and loss, the draws' mean pairwise cos and the
  split-half cos (mean over random splits) → Spearman–Brown per-draw ρ; the
  two losses' mean directions against each other and against the move f0
  made → ``read.json``.

    .venv/bin/python project/cjk_anima_reseed/probes/probe_cf.py pairs --label cf57
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_cf.py grad --label cf57"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_cf.py read --label cf57
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()

from reseed import OUT  # noqa: E402

PROBE = OUT / "probe_cf"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"
F0 = "output/cjk_anima_reseed/sent_kanji_f0/trained.pt"
KINDS = ("swap", "dup")


def _is_kana(c: str) -> bool:
    return "ぁ" <= c <= "ヿ"


def _is_kanji(c: str) -> bool:
    return "一" <= c <= "鿿"


def pairs(run_name: str, label: str, n_kana: int, n_kanji: int, per_row: int) -> None:
    from types import SimpleNamespace

    from common.render.flat import find_fonts, font_covers
    from common.render.scene import render_into_scene
    from data.inventory import qwen_pieces
    from data.synth import load_scenes
    from probe_geom import _char_rows
    from reseed import table as T
    from reseed.config import load
    from reseed.recipes import _sent_cuts

    run = load(run_name)
    run.use_pack()
    out = PROBE / label
    (out / "img").mkdir(parents=True, exist_ok=True)
    recs = [
        json.loads(ln)
        for ln in (run.data / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    sent = [r for r in recs if r["tier"].startswith("sent")]
    c2r = _char_rows()
    letters = {c for r in sent for c in r["text"] if c in c2r}
    kana_pool = sorted(c for c in letters if _is_kana(c) and c != "ー")
    kanji_pool = sorted(c for c in letters if _is_kanji(c))

    # items per char, the char once in the line
    by: dict = {}
    for k, r in enumerate(sent):
        for c in set(r["text"]):
            if c in c2r and r["text"].count(c) == 1:
                by.setdefault(c, []).append(k)
    rng = random.Random(0)
    elig = {c: v for c, v in by.items() if len(v) >= 2 * per_row}
    kana = rng.sample(sorted(c for c in elig if _is_kana(c) and c != "ー"), n_kana)
    kanji = rng.sample(sorted(c for c in elig if _is_kanji(c)), n_kanji)
    print(
        f"{len(sent)} sent items; eligible rows (≥ {2 * per_row} items, glyph once): "
        f"kana {sum(_is_kana(c) for c in elig)}, kanji {sum(_is_kanji(c) for c in elig)}",
        flush=True,
    )

    scenes = {}
    for tag, one in ((T.SCENES, T.ONE_BUBBLE), (T.SMALL_POOL, "")):
        for s in load_scenes(tag, 0.0, 0, "", one):
            scenes[(s["pool"], s["i"])] = s
    fonts = find_fonts()
    tokq = SimpleNamespace(tokq=qwen_pieces(char_rows=True))

    def sibling(text, i, kind, prng):
        c = text[i]
        if kind == "dup":
            for j in (i - 1, i + 1):
                if 0 <= j < len(text) and text[j] != c and text[j] in letters:
                    return text[:i] + text[j] + text[i + 1 :]
            return None
        if _is_kana(c):  # the same script: hiragana for hiragana
            hira = c <= "\u309f"
            pool = [x for x in kana_pool if (x <= "\u309f") == hira]
        else:
            pool = kanji_pool
        while True:
            b = prng.choice(pool)
            if b != c and b not in text:
                return text[:i] + b + text[i + 1 :]

    rows_out = []
    n_fail = 0
    for fam, chars in (("kana", kana), ("kanji", kanji)):
        for c in chars:
            items = list(elig[c])
            rng.shuffle(items)
            got = 0
            for k in items:
                if got == per_row:
                    break
                r = sent[k]
                text = r["text"]
                i = text.index(c)
                kind = KINDS[got % 2]
                prng = random.Random(k * 100_003 + ord(c))
                bt = sibling(text, i, kind, prng)
                if bt is None:
                    continue
                sc = scenes[(r["scene_pool"], r["scene"])]
                ok = [f for f in fonts if font_covers(f, text + bt)]
                if not ok:
                    continue
                drawn = None
                for attempt in range(4):
                    frng = random.Random(k * 100_003 + ord(c) + 7 * attempt + 1)
                    drawn = render_into_scene(
                        scene=sc,
                        text=text,
                        font_path=frng.choice(ok),
                        rng=frng,
                        min_glyph=int(0.85 * r["glyph_px"]),
                        stroke=False,
                        fill_frac=r["fill"],
                        max_lines=r["columns"],
                        cuts=_sent_cuts(tokq, text),
                        vertical_only=True,
                        fewest_lines=True,
                        tategaki=True,
                        vert_forms=True,
                        keep_outline=True,
                        ref_text=bt,
                    )
                    if drawn is not None:
                        break
                if drawn is None:
                    n_fail += 1
                    continue
                im_a, box_a, im_b, box_b = drawn
                n = len(rows_out)
                fa, fb = out / "img" / f"{n:05d}_a.png", out / "img" / f"{n:05d}_b.png"
                im_a.save(fa)
                im_b.save(fb)
                rows_out.append(
                    {
                        "n": n,
                        "char": c,
                        "family": fam,
                        "row": c2r[c],
                        "kind": kind,
                        "pos": i,
                        "text": text,
                        "text_b": bt,
                        "caption": r["caption"],
                        "tier": r["tier"],
                        "glyph_px": r["glyph_px"],
                        "src_item": k,
                        "shape": list(im_a.size),
                        "box": [
                            min(box_a[0], box_b[0]),
                            min(box_a[1], box_b[1]),
                            max(box_a[2], box_b[2]),
                            max(box_a[3], box_b[3]),
                        ],
                        "file_a": str(fa),
                        "file_b": str(fb),
                    }
                )
                got += 1
            assert got == per_row, (c, got)
    (out / "pairs.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows_out),
        encoding="utf-8",
    )
    print(
        f"→ {out / 'pairs.jsonl'}: {len(rows_out)} pairs, {len(kana)} kana + "
        f"{len(kanji)} kanji rows × {per_row} ({n_fail} render misses)",
        flush=True,
    )
    _sheet(rows_out, out / "sheet.png")


def _sheet(prs, path: Path, n: int = 12) -> None:
    """A′ | B crops around the box, ``n`` pairs."""
    from PIL import Image

    pick = random.Random(1).sample(prs, min(n, len(prs)))
    tiles = []
    for p in pick:
        x0, y0, x1, y1 = p["box"]
        pad = 16
        box = (max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad)
        a = Image.open(p["file_a"]).crop(box)
        b = Image.open(p["file_b"]).crop(box)
        t = Image.new("RGB", (a.width * 2 + 8, a.height), "red")
        t.paste(a, (0, 0))
        t.paste(b, (a.width + 8, 0))
        tiles.append(t.resize((int(t.width * 240 / t.height), 240)))
    W = max(t.width for t in tiles)
    sheet = Image.new("RGB", (W * 4, 240 * ((len(tiles) + 3) // 4)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 4) * W, (k // 4) * 240))
    sheet.save(path)
    print(f"→ {path}", flush=True)


def grad(run_name: str, rows_path: str, label: str, t_min: float, t_max: float) -> None:
    import os
    import time

    import torch
    from cjk_scale import train as T
    from common.models import checkpoints, dit_forward, encode_images, gen_args
    from common.models import load_vae
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from probe_geom import split_terms
    from reseed import REPO
    from reseed.config import load
    from train.stage import _encode_text

    from cjk_scale.rows import Rows

    run = load(run_name)
    run.use_pack()
    rows_path = str(rows_path if Path(rows_path).is_absolute() else REPO / rows_path)
    out = PROBE / label
    prs = [
        json.loads(ln)
        for ln in (out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    data = run.data
    recs, ev, vocabs = T.load_items(data)
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    device = get_generation_settings(args).device

    # VAE first, freed before the DiT
    t0 = time.time()
    vae = load_vae(device)
    lat = {}
    for p in prs:
        for side in ("a", "b"):
            lat[(p["n"], side)] = encode_images(vae, [p[f"file_{side}"]], device)[0]
    del vae
    torch.cuda.empty_cache()
    print(f"latents: {len(lat)} in {time.time() - t0:.0f}s", flush=True)

    cache, touched, _ = _encode_text(recs, ev, device, out, te_cache=data / "te_cache")
    rc = run.scale_config()
    plan = T.plan(rc, data, recs, vocabs, touched, Path(rows_path))
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    rows = Rows(
        anima,
        device,
        plan.idx,
        pack,
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
    got = []
    t0 = time.time()
    for step, p in enumerate(prs, 1):
        k = at[int(p["row"])]
        a = lat[(p["n"], "a")][None].to(device)
        b = lat[(p["n"], "b")][None].to(device)
        eps = torch.randn(a.shape, generator=gen).to(device)
        sig = t_min + (t_max - t_min) * float(torch.rand(1, generator=gen))
        ts = torch.full((1,), sig, device=device)
        rec = {"layout": "scene", "box": p["box"], "text": p["text"]}
        res = {"n": p["n"], "sigma": sig}
        for name, x0 in (("plain", a), ("cf", b)):
            x = (1 - sig) * x0 + sig * eps
            target = eps - a if name == "plain" else (x - a) / sig
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dit_forward(
                    anima, x.to(torch.bfloat16), ts, cache, [p["caption"]], device
                )
            l_in, l_out = split_terms(
                pred,
                target,
                [rec],
                T.BOX_SHARE,
                T.BOX_SHARE_CAP,
                T.BOX_SHARE_GLYPHS,
                T.GRID_BOX,
            )
            g = torch.autograd.grad(l_in + l_out, raw)[0][k]
            res[f"g_{name}"] = g.float().cpu()
            res[f"l_in_{name}"] = float(l_in.detach())
            with torch.no_grad():
                from cjk_scale.loss import box_mask

                m = box_mask(pred.shape, [rec], device, False)
                r = (pred.float() - target.float()) * m
                d = ((a - b) / sig) * m
                if name == "cf":
                    res["lambda"] = 1.0 - float((r * d).sum() / (d * d).sum())
                    res["d_norm"] = float(d.norm())
                res[f"r_norm_{name}"] = float(r.norm())
        got.append(res)
        if step % 50 == 0 or step == 1:
            print(
                f"pair {step}/{len(prs)}: σ {sig:.2f} λ {res['lambda']:+.3f} "
                f"|g| plain {res['g_plain'].norm():.3g} cf {res['g_cf'].norm():.3g}, "
                f"{(time.time() - t0) / step:.2f} s/pair",
                flush=True,
            )
    st = rows.delta.state_dict()
    torch.save(
        {
            "run": run_name,
            "rows": rows_path,
            "t_band": [t_min, t_max],
            "ext_ids": [int(e) for e in st["ext_ids"]],
            "raw": st["raw"].float().cpu(),
            "row_scale": rows.row_scale,
            "draws": got,
        },
        out / "grads.pt",
    )
    print(f"→ {out / 'grads.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _cos(u, v) -> float:
    import numpy as np

    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))


def _agreement(G, rng, splits: int = 50) -> dict:
    """``G`` (n, d): mean pairwise cos of the draws, the split-half cos (mean
    over random halves) and the per-draw ρ both imply."""
    import numpy as np

    n = len(G)
    U = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-30)
    C = U @ U.T
    pair = float((C.sum() - np.trace(C)) / (n * (n - 1)))
    hs = []
    for _ in range(splits):
        perm = rng.permutation(n)
        hs.append(_cos(G[perm[: n // 2]].sum(0), G[perm[n // 2 :]].sum(0)))
    h = float(np.mean(hs))
    m = n / 2
    rho_sb = h / (m - (m - 1) * h) if h < 1 else 1.0
    return {"pair_cos": pair, "split_half": h, "rho_sb": float(rho_sb)}


def read(label: str, end_rows: str) -> None:
    import numpy as np
    import torch
    from reseed import REPO

    sd = torch.load(PROBE / label / "grads.pt", map_location="cpu", weights_only=False)
    prs = {
        p["n"]: p
        for p in map(
            json.loads,
            (PROBE / label / "pairs.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    rs = float(sd["row_scale"])
    off = dict(zip(sd["ext_ids"], sd["raw"].numpy() * rs))
    e = torch.load(
        Path(end_rows) if Path(end_rows).is_absolute() else REPO / end_rows,
        map_location="cpu",
        weights_only=False,
    )["delta"]
    end = dict(
        zip(
            [int(x) for x in e["ext_ids"]],
            (e["raw"].float() * float(e["row_scale"])).numpy(),
        )
    )
    rng = np.random.default_rng(0)

    per_row: dict = {}
    for d in sd["draws"]:
        p = prs[d["n"]]
        per_row.setdefault(p["row"], {"p": p, "d": []})["d"].append(d)
    rows_out = []
    for row, x in per_row.items():
        D, p = x["d"], x["p"]
        rec = {"char": p["char"], "family": p["family"], "draws": len(D)}
        for name in ("plain", "cf"):
            G = np.stack([d[f"g_{name}"].numpy() for d in D])
            rec[name] = _agreement(G, rng)
            rec[name]["norm_draw"] = float(np.linalg.norm(G, axis=1).mean())
            for kind in KINDS:
                Gk = np.stack(
                    [d[f"g_{name}"].numpy() for d in D if prs[d["n"]]["kind"] == kind]
                )
                rec[name][kind] = _agreement(Gk, rng)
            rec[name]["swap_vs_dup"] = _cos(
                *[
                    np.sum(
                        [
                            d[f"g_{name}"].numpy()
                            for d in D
                            if prs[d["n"]]["kind"] == kd
                        ],
                        0,
                    )
                    for kd in KINDS
                ]
            )
            mv = end.get(int(row), off[int(row)]) - off[int(row)]
            rec[name]["cos_f0_move"] = (
                _cos(-G.mean(0), mv) if np.linalg.norm(mv) > 0 else None
            )
            rec[name]["cos_row"] = _cos(-G.mean(0), off[int(row)])
        Gp = np.stack([d["g_plain"].numpy() for d in D])
        Gc = np.stack([d["g_cf"].numpy() for d in D])
        rec["cos_plain_cf_mean"] = _cos(Gp.mean(0), Gc.mean(0))
        rec["cos_plain_cf_draw"] = float(np.mean([_cos(a, b) for a, b in zip(Gp, Gc)]))
        Gx = Gc - Gp  # the CF correction alone (same σ, ε, item)
        rec["corr"] = _agreement(Gx, rng)
        rec["lambda"] = float(np.mean([d["lambda"] for d in D]))
        rows_out.append(rec)

    def med(rr, *path):
        v = []
        for r in rr:
            x = r
            for k in path:
                x = x[k]
            if x is not None:
                v.append(x)
        return float(np.median(v)) if v else None

    summary = {}
    for f in ("kana", "kanji"):
        rr = [r for r in rows_out if r["family"] == f]
        s = {"rows": len(rr), "draws": med(rr, "draws"), "lambda": med(rr, "lambda")}
        for name in ("plain", "cf", "corr"):
            s[name] = {
                k: med(rr, name, k) for k in ("pair_cos", "split_half", "rho_sb")
            }
        for name in ("plain", "cf"):
            s[name]["norm_draw"] = med(rr, name, "norm_draw")
            s[name]["swap_vs_dup"] = med(rr, name, "swap_vs_dup")
            s[name]["cos_f0_move"] = med(rr, name, "cos_f0_move")
            s[name]["cos_row"] = med(rr, name, "cos_row")
            for kind in KINDS:
                s[name][kind] = {
                    k: med(rr, name, kind, k)
                    for k in ("pair_cos", "split_half", "rho_sb")
                }
        s["cos_plain_cf_mean"] = med(rr, "cos_plain_cf_mean")
        s["cos_plain_cf_draw"] = med(rr, "cos_plain_cf_draw")
        s["rho_ratio_cf_over_plain"] = float(
            np.median([r["cf"]["rho_sb"] / max(r["plain"]["rho_sb"], 1e-4) for r in rr])
        )
        summary[f] = s
    lam = {}
    for d in sd["draws"]:
        p = prs[d["n"]]
        key = f"{p['family']}|{p['kind']}|σ{0.5 + 0.1 * min(int((d['sigma'] - 0.5) / 0.1), 1):.1f}"
        lam.setdefault(key, []).append(d["lambda"])
    summary["lambda_by"] = {
        k: [len(v), float(np.mean(v))] for k, v in sorted(lam.items())
    }
    (PROBE / label / "read.json").write_text(
        json.dumps({"summary": summary, "rows": rows_out}, indent=1, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["pairs", "grad", "read"])
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument("--rows", default=STICK080)
    p.add_argument("--label", default="cf57")
    p.add_argument("--kana", type=int, default=24)
    p.add_argument("--kanji", type=int, default=24)
    p.add_argument("--per_row", type=int, default=24)
    p.add_argument("--t_min", type=float, default=0.5)
    p.add_argument("--t_max", type=float, default=0.7)
    p.add_argument("--end", default=F0)
    a = p.parse_args()
    if a.verb == "pairs":
        pairs(a.run, a.label, a.kana, a.kanji, a.per_row)
    elif a.verb == "grad":
        grad(a.run, a.rows, a.label, a.t_min, a.t_max)
    else:
        read(a.label, a.end)


if __name__ == "__main__":
    main()
