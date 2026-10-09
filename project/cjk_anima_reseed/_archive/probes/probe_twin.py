#!/usr/bin/env python
"""probe_twin — the twin difference: glyph identity as a subspace (`idea3.md`)

One `sent` item drawn twice: A with glyph u in slot k, B with v there — the
record's scene, columns and fill, one font, one rng, `render_into_scene(
ref_text=…)` (same fit, pixel-identical outside the text box) — under A's
caption (row u). D = g_A − g_B is where row u's effect depends on the glyph
drawn, S = (g_A + g_B) / 2 what the two share.

- ``render`` (CPU): f0's `sent` records in a seeded order; per record a slot
  k whose glyph u (kana or kanji, half each — the family alternates with
  the need) is in the text and the caption once and has a pack row; v of
  u's script (hiragana / katakana / kanji) and size (small kana for small),
  one f0's `sent` texts hold at least ``V_MIN_COUNT`` times (a glyph the
  rows were trained on, not one of the pack's rare kanji), absent from the
  text and the caption, not u's dakuten / small-kana sibling, covered by
  the font; a twin kept iff A and B differ inside one glyph's box
  (``GLYPH_SLACK`` × the record's glyph px on both axes) →
  ``output/cjk_anima_reseed/probe_twin/<label>/`` ``twins.jsonl`` + PNGs,
  ``sheet.png`` (A | B | |A − B| for the first ``SHEET`` twins).

- ``fit`` (GPU): per twin, A and B VAE-encoded (before the DiT loads), one
  σ uniform on ``SIGMA`` and one ε, then A's forward and B's — one at a
  time, the fp32 graph of each retained for its probes — under A's caption,
  with the same ``--probes`` Gaussian probes on the dilated box (in) and off
  it (out) on both (a reseeded generator): g per (twin, row) pair as
  `probe_jl`'s fit (fp32, SDPA, eager; summed over a row's occurrences),
  D = g_A − g_B and S = (g_A + g_B) / 2 per probe → ``…/<label>/fit.pt``.
  Twins alternate between fits A / B within a family (disjoint).
- ``read`` (CPU): idea3's five — f = E|D|² / E(|g_A|² + |g_B|²) (1 − cos
  for equal norms) for the own row and the cross rows (the item's other
  once-only glyphs) by σ; M_D / M_S per fit, family, own / cross and mask;
  M_D's A / B overlap (and over fit A's first n twins); M_D u = λ M_S u
  fitted on one fit, scored on the other, against r = tr(M_D) / tr(M_S);
  own against cross (M_D's top 16, against the own / own floor); the moves
  (stick, f0, pres, Δ(pres − f0)) as M_D / M_S see them →
  ``…/<label>/read{,_pair}.json``.

Stop (idea3): M_D's own in-box A / B overlap under 0.6 at k 16
(pair-weighted), or no cross-fit λ over 2 r, or own / cross overlapping as
much as own / own.

    .venv/bin/python project/cjk_anima_reseed/probes/probe_twin.py render --label t1 --kana 200 --kanji 200
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_twin.py fit --label t1"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_twin.py read --label t1 --weight pair
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()

from reseed import OUT  # noqa: E402

PROBE = OUT / "probe_twin"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"
SIGMA = (0.45, 0.8)  # sent's band and the step above it
MASKS = ("in", "out")
KS = (16, 64)
CUTS = (10, 25, 50, 100)  # fit A cut to its first n twins
SIG_BINS = ((0.45, 0.55), (0.55, 0.65), (0.65, 0.8 + 1e-9))
GLYPH_SLACK = 1.5  # the A / B difference box, × the record's glyph px
FONT_TRIES = 4  # fonts tried per record before it is dropped
V_MIN_COUNT = 20  # v's occurrences in f0's sent texts
SHEET = 12
_SMALL = dict(
    zip(
        "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶ",
        "あいうえおつやゆよわかけアイウエオツヤユヨワカケ",
    )
)


def _script(c: str) -> str:
    if "ぁ" <= c <= "ゖ":
        return "hira"
    if "ァ" <= c <= "ヺ":
        return "kata"
    if "一" <= c <= "鿿":
        return "kanji"
    return ""


def _base(c: str) -> str:
    """A glyph's base: dakuten / handakuten stripped, small kana → large."""
    b = unicodedata.normalize("NFD", c)[0]
    return _SMALL.get(b, b)


def _diff_box(im_a, im_b):
    """Pixel bbox where A and B differ, or None when they are equal."""
    from PIL import ImageChops

    return ImageChops.difference(im_a.convert("RGB"), im_b.convert("RGB")).getbbox()


def render(run_name: str, label: str, n_kana: int, n_kanji: int, seed: int) -> None:
    from collections import Counter
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
    sent = [(i, r) for i, r in enumerate(recs) if r.get("recipe") == "sent"]
    scenes = {
        (s.get("pool"), s["i"]): s
        for pool in (T.SCENES, T.SMALL_POOL)
        for s in load_scenes(pool, 0.0, 0, "", T.ONE_BUBBLE if pool == T.SCENES else "")
    }
    c2r = _char_rows()
    seen = Counter(c for _i, r in sent for c in r["text"])
    by_script: dict = {}
    for c, m in sorted(seen.items()):
        if m >= V_MIN_COUNT and c in c2r and _script(c):
            by_script.setdefault(_script(c), []).append(c)
    print(
        "v pools: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_script.items())),
        flush=True,
    )
    fonts = find_fonts()
    tokq = SimpleNamespace(tokq=qwen_pieces(char_rows=True))
    rng = random.Random(seed)
    rng.shuffle(sent)
    need = {"kana": n_kana, "kanji": n_kanji}
    kept, dropped = [], Counter()
    for i, r in sent:
        if not any(need.values()):
            break
        text, cap = r["text"], r["caption"]
        sc = scenes.get((r["scene_pool"], r["scene"]))
        if sc is None or cap.count(f'"{text}"') != 1:
            dropped["record"] += 1
            continue
        once = [
            c
            for c in text
            if text.count(c) == 1 and cap.count(c) == 1 and c in c2r and _script(c)
        ]
        fam = max(need, key=need.get)  # the family still most wanted
        cands = [c for c in once if (_script(c) == "kanji") == (fam == "kanji")]
        if not cands:
            dropped[f"{fam}/no u"] += 1
            continue
        u = rng.choice(cands)
        k = text.index(u)
        pool_v = [
            v
            for v in by_script[_script(u)]
            if v not in text
            and v not in cap
            and _base(v) != _base(u)
            and (v in _SMALL) == (u in _SMALL)
        ]
        cuts = _sent_cuts(tokq, text)
        px = float(r["glyph_px"])
        drawn = None
        for t in range(FONT_TRIES):
            v = rng.choice(pool_v)
            ref = text[:k] + v + text[k + 1 :]
            fok = [f for f in fonts if font_covers(f, text + v)]
            if not fok:
                continue
            font = rng.choice(fok)
            drawn = render_into_scene(
                scene=sc,
                text=text,
                font_path=font,
                rng=random.Random(seed * 7919 + i),
                min_glyph=int(0.85 * px),
                stroke=False,
                fill_frac=float(r["fill"]),
                max_lines=int(r["columns"]),
                cuts=cuts,
                vertical_only=True,
                fewest_lines=True,
                tategaki=True,
                vert_forms=True,
                keep_outline=True,
                ref_text=ref,
            )
            if drawn is not None:
                break
        if drawn is None:
            dropped[f"{fam}/draw"] += 1
            continue
        im_a, box_a, im_b, box_b = drawn
        d = _diff_box(im_a, im_b)
        if d is None:
            dropped[f"{fam}/same"] += 1
            continue
        dw, dh = d[2] - d[0], d[3] - d[1]
        if max(dw, dh) > GLYPH_SLACK * px:
            dropped[f"{fam}/wide"] += 1
            continue
        n = len(kept)
        fa, fb = out / "img" / f"{n:04d}_A.png", out / "img" / f"{n:04d}_B.png"
        im_a.save(fa)
        im_b.save(fb)
        union = [
            min(box_a[0], box_b[0]),
            min(box_a[1], box_b[1]),
            max(box_a[2], box_b[2]),
            max(box_a[3], box_b[3]),
        ]
        kept.append(
            {
                "n": n,
                "i": i,
                "family": fam,
                "text": text,
                "ref_text": ref,
                "slot": k,
                "u": u,
                "v": v,
                "row": c2r[u],
                "cross": [c for c in once if c != u],
                "cross_rows": [c2r[c] for c in once if c != u],
                "caption": cap,
                "scene": [r["scene_pool"], r["scene"]],
                "shape": list(im_a.size),
                "columns": int(r["columns"]),
                "fill": float(r["fill"]),
                "glyph_px": px,
                "font": Path(font).name,
                "box": union,
                "diff": list(d),
                "files": [str(fa), str(fb)],
                # what `out_mask` / `item_boxes` read
                "src": "scene",
                "layout": "scene",
            }
        )
        need[fam] -= 1
    (out / "twins.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept),
        encoding="utf-8",
    )
    fams = Counter(x["family"] for x in kept)
    print(
        f"→ {out / 'twins.jsonl'}: {len(kept)} twins ({dict(fams)}), "
        f"dropped {dict(dropped)}",
        flush=True,
    )
    if kept:
        _sheet(kept[:SHEET], out / "sheet.png")


def _sheet(twins: list, path: Path) -> None:
    """A | B | |A − B| ×4 per twin, the difference box outlined on A and B."""
    from PIL import Image, ImageChops, ImageDraw

    h = 256
    rows = []
    for x in twins:
        a, b = (Image.open(f).convert("RGB") for f in x["files"])
        diff = ImageChops.difference(a, b).point(lambda p: min(255, 4 * p))
        for im in (a, b):
            ImageDraw.Draw(im).rectangle(x["diff"], outline=(255, 0, 0), width=2)
        w = round(a.width * h / a.height)
        row = Image.new("RGB", (3 * w, h), "white")
        for j, im in enumerate((a, b, diff)):
            row.paste(im.resize((w, h)), (j * w, 0))
        ImageDraw.Draw(row).text(
            (4, 4), f"{x['n']} {x['u']}→{x['v']} k{x['slot']}", fill=(255, 0, 0)
        )
        rows.append(row)
    W = max(r.width for r in rows)
    cols = 2
    sheet = Image.new("RGB", (cols * W, h * -(-len(rows) // cols)), "white")
    for j, r in enumerate(rows):
        sheet.paste(r, ((j % cols) * W, (j // cols) * h))
    sheet.save(path)
    print(f"→ {path}", flush=True)


def _model(run_name: str, rows_path: str, out: Path, captions: list):
    """`probe_jl._setup`'s model half: the run's pack and plan for the rows
    ``captions`` touch, the DiT in fp32 / SDPA / eager with the rows at
    ``rows_path`` → ``(env, cache)``."""
    import os
    from types import SimpleNamespace

    import torch
    from cjk_scale import train as T
    from cjk_scale.rows import Rows
    from common.models import checkpoints, encode_captions, ext_ids_of, gen_args
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from reseed import REPO
    from reseed.config import load

    run = load(run_name)
    run.use_pack()
    rows_path = str(rows_path if Path(rows_path).is_absolute() else REPO / rows_path)
    assert Path(rows_path).is_file(), rows_path
    torch.manual_seed(T.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 is 20 % off fp32 here
    torch.backends.cudnn.allow_tf32 = False
    data = run.data
    recs, _ev, vocabs = T.load_items(data)
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    args.compile_blocks = False  # fp32 eager
    device = get_generation_settings(args).device
    cache = encode_captions(sorted(set(captions)), device, out / "te")
    p = T.plan(run.scale_config(), data, recs, vocabs, ext_ids_of(cache), Path(rows_path))
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    rows = Rows(
        anima,
        device,
        p.idx,
        strategy_pack(tok),
        warm=Path(rows_path),
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
        touched=p.touched,
        frozen=p.frozen,
        context=Path(rows_path),
    )
    anima.attn_mode = "torch"  # fp32 SDPA: bf16 is cos 0.3–0.6 off on the rows
    anima.float()
    env = SimpleNamespace(T=T, device=device, anima=anima, rows=rows, rows_path=rows_path)
    return env, cache


def fit(run_name: str, rows_path: str, label: str, probes: int) -> None:
    import time
    from collections import Counter

    import torch
    from cjk_scale.loss import out_mask
    from common.models import encode_images, load_vae
    from probe_jl import _forward

    out = PROBE / label
    twins = [
        json.loads(ln)
        for ln in (out / "twins.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    device = torch.device("cuda")
    # the latents first, the DiT after (lazy loading)
    vae = load_vae(device)
    lat = {x["n"]: encode_images(vae, x["files"], device) for x in twins}
    del vae
    torch.cuda.empty_cache()
    print(f"{len(twins)} twins encoded", flush=True)
    env, cache = _model(run_name, rows_path, out, [x["caption"] for x in twins])
    T, anima, rows = env.T, env.anima, env.rows
    delta = rows.delta
    n_rows, dim = delta.raw.shape
    live = ~rows.frozen_mask
    ext_of = torch.tensor(delta.ext_ids, device=device)

    cap: dict = {}

    def grab_ids(module, a):
        cap["n"] = cap.get("n", 0) + 1
        cap["ids"] = a[0]

    def grab(module, a, output):
        cap["emb"] = output

    embed = anima.llm_adapter.embed
    handles = [
        embed.register_forward_pre_hook(grab_ids, prepend=True),
        embed.register_forward_hook(grab),
    ]
    anima.train()
    got, seen = [], Counter()
    t0 = time.time()
    for n, x in enumerate(twins):
        fit_ = "AB"[seen[x["family"]] % 2]
        seen[x["family"]] += 1
        gen = torch.Generator(device=device).manual_seed(1_000_003 * (T.SEED + 1) + x["n"])
        sigma = SIGMA[0] + (SIGMA[1] - SIGMA[0]) * float(
            torch.rand((), generator=gen, device=device)
        )
        both = lat[x["n"]].to(device).float()
        noise = torch.randn(both.shape[1:], generator=gen, device=device)[None]
        ts = torch.full((1,), sigma, device=device, dtype=torch.float32)
        side = []
        for s in range(2):
            noisy = (1.0 - sigma) * both[s : s + 1] + sigma * noise
            cap.clear()
            pred = _forward(anima, noisy, ts, cache, [x["caption"]], device)
            assert cap.get("n") == 1, f"embed ran {cap.get('n')} times in one forward"
            ids, emb = cap["ids"], cap["emb"]
            mo = out_mask(pred.shape, [x], device, False)
            mi = 1.0 - mo
            ext = ids >= delta.T
            bpos, lpos = ext.nonzero(as_tuple=True)
            loc = delta.lut[
                torch.clamp(ids[bpos, lpos] - delta.T, max=delta.lut.numel() - 1)
            ]
            k = loc >= 0
            lpos, loc = lpos[k], loc[k]
            uk, inv = torch.unique(loc, return_inverse=True)

            def pair_g(v, pred=pred, emb=emb, uk=uk, inv=inv, lpos=lpos):
                g = torch.autograd.grad(pred, emb, grad_outputs=v, retain_graph=True)[0]
                return torch.zeros(len(uk), dim, device=device).index_add_(
                    0, inv, g[0, lpos].float()
                )

            if n == 0 and s == 0:  # the hook against autograd on ``raw``
                v = torch.randn(pred.shape, device=device)
                agg = pair_g(v)
                g_raw = torch.autograd.grad(
                    pred, delta.raw, grad_outputs=v, retain_graph=True
                )[0]
                per_row = torch.zeros(n_rows, dim, device=device).index_add_(0, uk, agg)
                want = per_row[live] * rows.row_scale * delta.scale
                err = float((g_raw[live] - want).norm() / g_raw[live].norm())
                print(f"self-check: {len(uk)} rows, |g_raw − Σ g · row_scale| / |g_raw| = {err:.2e}", flush=True)
                assert err < 1e-4, err
            gp = torch.Generator(device=device).manual_seed(2_000_003 * (T.SEED + 1) + x["n"])
            G = torch.empty(len(uk), len(MASKS), probes, dim)
            for j, mask in enumerate((mi, mo)):
                for q in range(probes):
                    v = torch.randn(pred.shape, generator=gp, device=device) * mask
                    G[:, j, q] = pair_g(v).cpu()
            side.append((ext_of[uk].cpu(), G))
            del pred, emb, pair_g
            cap.clear()
        (ea, Ga), (eb, Gb) = side
        assert torch.equal(ea, eb), "A and B carry different rows"
        es = ea.tolist()
        assert x["row"] in es, f"twin {x['n']}: u's row {x['row']} not in the caption's rows"
        got.append(
            {
                "n": x["n"],
                "fit": fit_,
                "family": x["family"],
                "sigma": sigma,
                "ext": ea,
                "own": es.index(x["row"]),
                "cross": [es.index(r) for r in x["cross_rows"] if r in es],
                "in_frac": float(mi.mean()),
                # (pair, D / S, mask, probe, dim)
                "G": torch.stack([Ga - Gb, (Ga + Gb) / 2], 1).to(torch.bfloat16),
            }
        )
        if (n + 1) % 25 == 0 or n == 0:
            g = got[-1]["G"][got[-1]["own"]].float()
            print(
                f"{n + 1}/{len(twins)} {x['family']}/{fit_} σ {sigma:.3f} {len(es)} rows: "
                f"own in |D| {float(g[0, 0].norm(dim=-1).mean()):.3e} "
                f"|S| {float(g[1, 0].norm(dim=-1).mean()):.3e}, "
                f"{(time.time() - t0) / (n + 1):.2f} s/twin",
                flush=True,
            )
    for h in handles:
        h.remove()
    st = delta.state_dict()
    torch.save(
        {
            "run": run_name,
            "rows": env.rows_path,
            "ext_ids": [int(e) for e in st["ext_ids"]],
            "raw": st["raw"].float().cpu(),
            "row_scale": rows.row_scale,
            "probes": probes,
            "sigma": SIGMA,
            "twins": got,
        },
        out / "fit.pt",
    )
    print(f"→ {out / 'fit.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------- read


def _f(G) -> "tuple":
    """Per pair (rows of ``G``: (pair, D / S, mask, probe, dim)) and mask:
    E|D|² and E(|g_A|² + |g_B|²) = E|D|² / 2 + 2 E|S|² over the probes."""
    e = (G.float() ** 2).sum(-1).mean(-1).numpy()  # (pair, D / S, mask)
    return e[:, 0], e[:, 0] / 2 + 2 * e[:, 1]


def read(label: str, weight: str, ridge: float) -> None:
    import numpy as np
    import torch
    from probe_geom import _char_rows
    from probe_jl import _gen, _moves, _orth, _overlap, _quad, _share, _top

    sd = torch.load(PROBE / label / "fit.pt", map_location="cpu", weights_only=False)
    tw = sd["twins"]
    P = sd["probes"]
    d = sd["raw"].shape[1]
    out = {"label": label, "weight": weight, "twins": len(tw), "probes": P}

    # 1. f by σ, own / cross, in / out (ratio of sums over pairs)
    fr: dict = {}
    for t in tw:
        b = next(j for j, (lo, hi) in enumerate(SIG_BINS) if lo <= t["sigma"] < hi)
        dd, gg = _f(t["G"])
        for kind, idx in (("own", [t["own"]]), ("cross", t["cross"])):
            for j, m in enumerate(MASKS):
                a = fr.setdefault((t["family"], kind, m, b), [0.0, 0.0, 0])
                a[0] += float(dd[idx, j].sum())
                a[1] += float(gg[idx, j].sum())
                a[2] += len(idx)
    out["f"] = {
        f"{fam}|{kind}|{m}|σ{SIG_BINS[b][0]:.2f}–{min(SIG_BINS[b][1], 0.8):.2f}": {
            "f": a[0] / max(a[1], 1e-30),
            "pairs": a[2],
        }
        for (fam, kind, m, b), a in sorted(fr.items())
    }

    # 2–4. moments per (fit, family, kind, mask): M_D, M_S
    acc: dict = {}
    cuts: dict = {}
    seen: dict = {}
    for t in tw:
        G = t["G"].float().numpy().astype(np.float64)  # (pair, 2, mask, P, d)
        if weight == "pair":  # each pair over its S's RMS, D and S alike
            rms = np.sqrt((G[:, 1] ** 2).sum(-1).mean((1, 2)))
            G = G / rms[:, None, None, None, None]
        key0 = (t["fit"], t["family"])
        seen[key0] = seen.get(key0, 0) + 1
        for kind, idx in (("own", [t["own"]]), ("cross", t["cross"])):
            if not idx:
                continue
            a = acc.setdefault((*key0, kind), {"n": 0})
            for j, m in enumerate(MASKS):
                for w_, nm in ((0, "D"), (1, "S")):
                    X = G[idx, w_, j].reshape(-1, d)
                    a[f"{nm}_{m}"] = a.get(f"{nm}_{m}", 0.0) + X.T @ X
            a["n"] += len(idx)
        if t["fit"] == "A" and seen[key0] in CUTS:
            a = acc.get(("A", t["family"], "own"))
            if a:
                cuts[(t["family"], seen[key0])] = {
                    k: (v / (a["n"] * P) if k != "n" else v) for k, v in a.items()
                }
    for a in acc.values():
        for k in list(a):
            if k != "n":
                a[k] = a[k] / (a["n"] * P)

    start = {
        int(e): (r * float(sd["row_scale"])).numpy()
        for e, r in zip(sd["ext_ids"], sd["raw"])
    }
    moves = _moves(start)
    c2r = _char_rows()
    fam_ext = {"kana": set(), "kanji": set()}
    for c, e in c2r.items():
        if len(c) == 1 and _script(c) and e in start:
            fam_ext["kanji" if _script(c) == "kanji" else "kana"].add(e)

    out["families"] = {}
    for fam in ("kana", "kanji"):
        if ("A", fam, "own") not in acc or ("B", fam, "own") not in acc:
            continue
        rec: dict = {}
        A, B = acc[("A", fam, "own")], acc[("B", fam, "own")]
        rec["pairs"] = [A["n"], B["n"]]
        pool = {
            k: (A[k] * A["n"] + B[k] * B["n"]) / (A["n"] + B["n"]) for k in A if k != "n"
        }
        rec["r_in"] = float(np.trace(pool["D_in"]) / np.trace(pool["S_in"]))
        rec["overlap"] = {
            f"{nm}_{m}@{k}": _overlap(_top(A[f"{nm}_{m}"], k)[1], _top(B[f"{nm}_{m}"], k)[1])
            for nm in ("D", "S")
            for m in MASKS
            for k in KS
        }
        rec["overlap_by_twins@16"] = {
            n: _overlap(_top(c["D_in"], 16)[1], _top(B["D_in"], 16)[1])
            for (f_, n), c in sorted(cuts.items())
            if f_ == fam and n < seen[("A", fam)]
        }
        rec["topk_trace_share"] = {
            f"D_in@{k}": float(_top(pool["D_in"], k)[0][:k].sum() / np.trace(pool["D_in"]))
            for k in KS
        }
        # 3. M_D u = λ M_S u, fitted on one fit, scored on the other
        r = rec["r_in"]
        lam: dict = {}
        for src, dst in ((A, B), (B, A)):
            w, V, _ = _gen(src["D_in"], src["S_in"], ridge)
            Bd = dst["S_in"] + ridge * np.trace(dst["S_in"]) / d * np.eye(d)
            xs = ((dst["D_in"] @ V) * V).sum(0) / ((Bd @ V) * V).sum(0)
            for k in (1, 16, 64):
                lam.setdefault(f"in_sample@{k}", []).append(float(w[:k].mean() / r))
                lam.setdefault(f"cross@{k}", []).append(float(xs[:k].mean() / r))
        rec["lambda_over_r"] = {k: float(np.mean(v)) for k, v in lam.items()}
        hv = {f: _orth(_gen(acc[(f, fam, "own")]["D_in"], acc[(f, fam, "own")]["S_in"], ridge)[1], 16) for f in "AB"}
        rec["hi_lambda_overlap@16"] = _overlap(hv["A"], hv["B"])
        # 4. own against cross
        if ("A", fam, "cross") in acc and ("B", fam, "cross") in acc:
            CA, CB = acc[("A", fam, "cross")], acc[("B", fam, "cross")]
            q = lambda M: _top(M, 16)[1]  # noqa: E731
            rec["own_cross@16"] = {
                "own_own": rec["overlap"]["D_in@16"],
                "cross_cross": _overlap(q(CA["D_in"]), q(CB["D_in"])),
                "own_cross": float(
                    np.mean(
                        [_overlap(q(A["D_in"]), q(CB["D_in"])), _overlap(q(CA["D_in"]), q(B["D_in"]))]
                    )
                ),
                "cross_pairs": [CA["n"], CB["n"]],
            }
        # 5. the moves
        w, V, _ = _gen(pool["D_in"], pool["S_in"], ridge)
        Q = {"D_in@16": _top(pool["D_in"], 16)[1], "hi_lambda@16": _orth(V, 16)}
        fe = fam_ext[fam]
        st = np.stack([start[e] for e in fe if np.linalg.norm(start[e]) > 0])
        sets = {"stick": st.mean(0, keepdims=True)}
        for name, mv in moves.items():
            xs = [mv[e] for e in mv if e in fe and np.linalg.norm(mv[e]) > 1e-6]
            if len(xs) >= 2:
                X = np.stack(xs)
                sets[name] = X
                sets[f"{name}:shared"] = X.mean(0, keepdims=True)
        rec["moves"] = {
            name: {
                "rows": int(len(X)),
                **{q_: _share(X, Qm) for q_, Qm in Q.items()},
                "lambda_over_r": float(_quad(X, pool["D_in"]) / _quad(X, pool["S_in"]) / r),
            }
            for name, X in sets.items()
        }
        rec["stop"] = {
            "overlap_D_in@16<0.6": rec["overlap"]["D_in@16"] < 0.6,
            "cross_lambda<2r": rec["lambda_over_r"]["cross@1"] < 2.0,
            "own≈cross": (
                rec["own_cross@16"]["own_cross"] >= rec["own_cross@16"]["own_own"]
                if "own_cross@16" in rec
                else None
            ),
        }
        out["families"][fam] = rec
    f = PROBE / label / ("read.json" if weight == "none" else f"read_{weight}.json")
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    _print(out)
    print(f"→ {f}", flush=True)


def _print(out: dict) -> None:
    print(f"{out['twins']} twins, {out['probes']} probes, weight {out['weight']}")
    print("f = E|D|² / E(|g_A|² + |g_B|²):")
    for k, v in out["f"].items():
        print(f"   {k:34s} {v['f']:.3f}  (pairs {v['pairs']})")
    for fam, r in out["families"].items():
        print(f"== {fam}: pairs {r['pairs']}, r = tr(M_D)/tr(M_S) in {r['r_in']:.3f}")
        print("   A/B overlap " + "  ".join(f"{k} {v:.3f}" for k, v in r["overlap"].items()))
        print("   M_D in by twins @16 " + "  ".join(f"{k}: {v:.2f}" for k, v in r["overlap_by_twins@16"].items()))
        print("   top-k trace share " + "  ".join(f"{k} {v:.3f}" for k, v in r["topk_trace_share"].items()))
        print("   λ / r " + "  ".join(f"{k} {v:.2f}" for k, v in r["lambda_over_r"].items()) + f"   hi-λ A/B @16 {r['hi_lambda_overlap@16']:.3f}")
        if "own_cross@16" in r:
            print("   own / cross @16 " + "  ".join(f"{k} {v}" if isinstance(v, list) else f"{k} {v:.3f}" for k, v in r["own_cross@16"].items()))
        for name, m in r["moves"].items():
            print(f"     {name:22s} n {m['rows']:5d}  " + "  ".join(f"{k} {v:.3f}" for k, v in m.items() if k != "rows"))
        print(f"   stop: {[k for k, v in r['stop'].items() if v]}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["render", "fit", "read"])
    p.add_argument(
        "--rows", default=STICK080, help="merged trained.pt, repo-relative or absolute"
    )
    p.add_argument("--probes", type=int, default=8, help="per mask per twin side")
    p.add_argument("--weight", choices=["none", "pair"], default="none")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument("--label", default="t0")
    p.add_argument("--kana", type=int, default=8)
    p.add_argument("--kanji", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.verb == "render":
        render(a.run, a.label, a.kana, a.kanji, a.seed)
    elif a.verb == "fit":
        fit(a.run, a.rows, a.label, a.probes)
    else:
        read(a.label, a.weight, a.ridge)


if __name__ == "__main__":
    main()
