#!/usr/bin/env python
"""xattn_loc on the generation path — does the string's alignment go with the read?

``probe.py`` read real training pages. Here the reseed line's dialogue ruler
(96 real JA strings, each in its own image's caption; ``criteria.md`` there)
is rendered on the base DiT with the shipped vocab pack, 28 steps, cfg 4, at
``--seeds``, with the recorder on the cond pass of every step. Per render the
string tokens' cross-attn maps are kept as band means over σ (hi ≥ 0.85, mid
0.6 – 0.85, low < 0.6) for three block sets (13 – 15, 8, and the four heads
``probe.py`` found: (15, 1), (14, 15), (13, 7), (8, 14)). The pages are then
read by the tuned reader (the ruler's ``Readers`` / ``score_text`` /
``score_page``) and each render gets:

- ``text``  — the string map's lift over every text box the page letters;
- ``on`` / ``off`` — over the boxes that hold the string's letters / the rest;
- ``peak``  — the share of the map in its top 5 % of image tokens (no box);
- ``rho``   — token order against the attention centroid along the largest
  elongated ``on`` box;

beside the read (``g_f1``, ``g_r``, ``g_p``, ``cer``, ``exact``, ``le2``,
``dup``, ``drawn``). The summary pairs the two: Spearman over renders, the
means by read outcome, and inside a string the seed with the higher alignment
against the seed with the higher ``g_f1``.

Compile is off: the per-block hooks read q / k in Python on every forward.

    make daemon-run ARGS="--queue --stall-timeout 600 bench/xattn_loc/gen_probe.py --seeds 0,1"
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from bench.xattn_loc.probe import TOK_PX, clause_strings, line_axis, region, spearman
from library.env import resolve_under_home

RULER = "output/cjk_anima_reseed/ruler/ruler.json"
RESEED = "project/cjk_anima_reseed"
STEPS, CFG = 28, 4.0
BANDS = (("hi", 0.85, 1.01), ("mid", 0.6, 0.85), ("low", -1.0, 0.6))
HEADS4 = ((15, 1), (14, 15), (13, 7), (8, 14))
SETS = ("b13_15", "b8", "heads4")
ON_SHARE = 0.5


class Recorder:
    def __init__(self, anima):
        import torch

        from library.anima.models import Attention

        self.torch = torch
        self.cols = None
        self.cond_ptr = None
        self.on = False
        self.band = None
        self.acc: dict = {}
        self.n: dict = {}
        self.step_maps: dict = {}
        want = {8, 13, 14, 15}
        for name, mod in anima.named_modules():
            m = re.search(r"blocks\.(\d+)\.", name)
            if isinstance(mod, Attention) and not mod.is_selfattn and m:
                if int(m.group(1)) in want:
                    mod.forward = self._wrap(mod, int(m.group(1)), mod.forward)
        orig = anima.forward

        def model_fwd(latents, timesteps, context, *a, **kw):
            if self.cond_ptr is None:
                self.cond_ptr = context.data_ptr()  # the sampler's first call
            self.on = self.cols is not None and context.data_ptr() == self.cond_ptr
            sg = float(timesteps.flatten()[0])
            self.band = next(b for b, lo, hi in BANDS if lo <= sg < hi)
            self.step_maps = {}
            out = orig(latents, timesteps, context, *a, **kw)
            if self.on:
                self._fold()
            self.on = False
            return out

        anima.forward = model_fwd

    def begin(self, cols):
        self.cols = self.torch.tensor(cols, device="cuda")
        self.cond_ptr, self.acc, self.n = None, {}, {}

    def _wrap(self, mod, b, orig):
        torch = self.torch

        def fwd(x, attn_params, context, rope_cos_sin=None):
            if self.on:
                with torch.no_grad():
                    q, k, _ = mod.compute_qkv(x, context, rope_cos_sin=None)
                    H, D = q.shape[-2:]
                    q = q.reshape(-1, H, D).float()
                    k = k.reshape(-1, H, D).float()
                    p = torch.einsum("qhd,khd->hqk", q, k) * D**-0.5
                    self.step_maps[b] = torch.softmax(p, -1)[:, :, self.cols]
            return orig(x, attn_params, context, rope_cos_sin=rope_cos_sin)

        return fwd

    def _fold(self):
        torch, sm = self.torch, self.step_maps
        cur = {
            "b13_15": torch.stack([sm[b].mean(0) for b in (13, 14, 15)]).mean(0),
            "b8": sm[8].mean(0),
            "heads4": torch.stack([sm[b][h] for b, h in HEADS4]).mean(0),
        }
        for s, v in cur.items():
            key = (self.band, s)
            self.acc[key] = v if key not in self.acc else self.acc[key] + v
            self.n[key] = self.n.get(key, 0) + 1

    def maps(self) -> dict:
        return {
            f"{b}|{s}": (v / self.n[(b, s)]).cpu().numpy().astype(np.float32)
            for (b, s), v in self.acc.items()
        }


def string_cols(encoder, cap: str, text: str):
    ids, mask, offs = encoder.encode_aligned(cap, 512)
    hit = [s for s in clause_strings(cap) if cap[s["a"] : s["b"]] == text]
    if hit:
        a, b = hit[0]["a"], hit[0]["b"]
    else:
        a = cap.find(text)
        if a < 0:
            return None
        b = a + len(text)
    cols = [j for j, (x, y) in enumerate(offs[:-1]) if y > x and x >= a and y <= b]
    return {
        "a": a,
        "b": b,
        "cols": cols,
        "offs": [list(offs[j]) for j in cols],
        "ids": ids,
    }


def render_all(a, items, out: Path):
    import torch

    from anima_lora import GenerationRequest, default_checkpoints
    from bench.xattn_loc.probe import load_encoder
    from common.models import decode_image, load_vae
    from library.inference.generation import generate, get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models
    from library.inference.text import process_escape

    encoder, pack = load_encoder()
    print(f"pack {pack.name}", flush=True)
    assert "jp_v1" in pack.name, "not the shipped pack"
    ck = default_checkpoints()
    req = GenerationRequest(
        prompt="", image_size=(896, 672), infer_steps=STEPS, guidance_scale=CFG,
        seed=0, dit=a.dit or ck.dit, vae=ck.vae, text_encoder=ck.text_encoder,
        attn_mode="flash", save_path=str(out),
    )  # fmt: skip
    args = req.to_args()
    args.compile_blocks = False
    gen = get_generation_settings(args)
    device = gen.device
    shared = load_shared_models(args)
    shared["text_encoder"].to(device)
    shared["conds_cache"] = {}
    anima = shared["model"] = load_dit_model(args, device, torch.bfloat16)
    anima.eval()
    vae = load_vae(device)
    rec = Recorder(anima)
    seeds = [int(s) for s in a.seeds.split(",")]
    meta, todo = [], [(m, s) for m in items for s in seeds]
    for n, (m, seed) in enumerate(todo):
        cap = process_escape(m["prompts"]["captioned"]["caption"])
        sc = string_cols(encoder, cap, m["text"])
        if sc is None or not sc["cols"]:
            print(f"  r{m['i']:02d}: string not in caption, skipped", flush=True)
            continue
        fn = out / "renders" / f"r{m['i']:02d}_s{seed}.png"
        fn.parent.mkdir(parents=True, exist_ok=True)
        a2 = copy.deepcopy(args)
        a2.prompt, a2.seed = cap, seed
        a2.image_size = (m["shape"][1], m["shape"][0])
        rec.begin(sc["cols"])
        with torch.no_grad():
            lat = generate(a2, gen, shared)
        got = shared["conds_cache"][cap][2].flatten().tolist()
        assert got == sc["ids"], f"r{m['i']:02d}: the sampler's T5 ids differ"
        decode_image(vae, lat, device).save(fn)
        mp = rec.maps()
        assert len(mp) == len(BANDS) * len(SETS), sorted(mp)
        np.savez_compressed(fn.with_suffix(".npz"), **mp)
        meta.append(
            {"i": m["i"], "seed": seed, "bin": m["bin"], "text": m["text"],
             "shape": m["shape"], "file": str(fn), "a": sc["a"], "b": sc["b"],
             "cols": sc["cols"], "offs": sc["offs"], "steps": dict(rec.n and {
                 f"{b}": rec.n[(b, "b8")] for b, _, _ in BANDS})}
        )  # fmt: skip
        if n % 16 == 0:
            print(f"  render {n + 1} / {len(todo)} r{m['i']:02d} s{seed}", flush=True)
    rec.cols = None
    del anima, vae, rec
    shared.clear()
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def lift(mp: np.ndarray, box: np.ndarray) -> float:
    if not box.any() or mp.sum() <= 0:
        return float("nan")
    return float(mp[box].sum() / mp.sum() / box.mean())


def read_all(meta: list, out: Path) -> list:
    from common.readers import Readers, load_bgr
    from eval.ruler.score import letters, score_page, score_text
    from collections import Counter

    rd = Readers("cuda")
    rows = []
    for n, r in enumerate(meta):
        reads = rd.read_image(load_bgr(Path(r["file"])), whole=True)
        W, H = r["shape"]
        gh, gw = H // TOK_PX, W // TOK_PX
        G = Counter(letters(r["text"]))
        nG = sum(G.values())
        boxes = [x for x in reads if not x.get("whole")]
        on = []
        for x in boxes:
            b = Counter(letters(x.get("sfx")))
            hit, nb = sum((b & G).values()), sum(b.values())
            on.append(bool(nb and hit >= max(min(2, nG), ON_SHARE * nb)))

        def mask(sel):
            m = np.zeros(gh * gw, dtype=bool)
            for x, keep in zip(boxes, sel):
                if keep:
                    m |= region(x["box"], gh, gw).numpy()
            return m

        text_m, on_m = mask([True] * len(boxes)), mask(on)
        off_m = text_m & ~on_m
        row = {
            "i": r["i"], "seed": r["seed"], "bin": r["bin"], "text": r["text"],
            "n_tok": len(r["cols"]), "n_box": len(boxes), "n_on": int(sum(on)),
            "text_area": float(text_m.mean()),
            **{k: v for k, v in score_text(r["text"], reads).items() if k != "best"},
            **{k: v for k, v in score_page(r["text"], reads, []).items()
               if k in ("g_p", "g_r", "g_f1", "drawn")},
        }  # fmt: skip
        z = np.load(Path(r["file"]).with_suffix(".npz"))
        # the largest elongated on box, for the order read
        s = None
        cand = sorted(
            (x for x, k in zip(boxes, on) if k),
            key=lambda x: -(x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]),
        )
        if cand:
            s = {"box": cand[0]["box"], "a": r["a"], "b": r["b"], "text": r["text"],
                 "cols": r["cols"]}  # fmt: skip
            if line_axis(s) is None:
                s = None
        for band, _, _ in BANDS:
            for st in SETS:
                P = z[f"{band}|{st}"].astype(np.float64)  # (Nq, n_tok)
                assert P.shape[0] == gh * gw, f"{P.shape} vs {gh}x{gw}"
                mp = P.sum(1)
                k = f"{band}_{st}"
                row[f"text_{k}"] = lift(mp, text_m)
                row[f"on_{k}"] = lift(mp, on_m)
                row[f"off_{k}"] = lift(mp, off_m)
                top = max(1, int(0.05 * mp.size))
                row[f"peak_{k}"] = float(np.sort(mp)[-top:].sum() / mp.sum())
                row[f"rho_{k}"] = float("nan")
                if s is not None:
                    axis, _ = line_axis(s)
                    bm = region(s["box"], gh, gw).numpy()
                    idx = np.nonzero(bm)[0]
                    if len(idx) >= 4:
                        x0, y0, x1, y1 = (v / TOK_PX for v in s["box"])
                        coord = (idx // gw if axis else idx % gw) + 0.5
                        lo, hi = (y0, y1) if axis else (x0, x1)
                        u = np.clip((coord - lo) / max(hi - lo, 1e-6), 0, 1)
                        Q = P[idx]
                        Q = Q / np.clip(Q.sum(0, keepdims=True), 1e-20, None)
                        row[f"rho_{k}"] = spearman(
                            np.arange(Q.shape[1]), (Q * u[:, None]).sum(0)
                        )
        rows.append(row)
        if n % 32 == 0:
            print(f"  read {n + 1} / {len(meta)}", flush=True)
    return rows


def summarize(rows: list) -> dict:
    import pandas as pd

    d = pd.DataFrame(rows)
    out: dict = {
        "renders": len(d), "strings": int(d.i.nunique()),
        "read": {
            b: {k: round(float(x[k].mean()), 3) for k in ("g_f1", "g_r", "g_p", "exact", "le2", "cer", "dup", "drawn")}
            for b, x in [("all", d), *d.groupby("bin")]
        },
    }  # fmt: skip
    good = d.g_f1 >= d.g_f1.median()
    S: dict = {}
    for m in ("text", "on", "off", "peak", "rho"):
        for band, _, _ in BANDS:
            for st in SETS:
                c = f"{m}_{band}_{st}"
                x = d[[c, "g_f1", "g_r", "cer"]].dropna()
                e: dict = {"n": len(x), "mean": _r(d[c].mean()),
                           "good": _r(d.loc[good, c].mean()),
                           "bad": _r(d.loc[~good, c].mean())}  # fmt: skip
                if len(x) >= 12:
                    for y in ("g_f1", "g_r", "cer"):
                        e[f"sp_{y}"] = _r(x[c].corr(x[y], method="spearman"))
                # inside a string: the seed with the higher alignment has the higher g_f1?
                pv = d.pivot_table(
                    index="i", columns="seed", values=[c, "g_f1"], dropna=False
                )
                if c in pv.columns.get_level_values(0) and pv[c].shape[1] >= 2:
                    s0, s1 = pv[c].columns[:2]
                    da = pv[c][s1] - pv[c][s0]
                    dg = pv["g_f1"][s1] - pv["g_f1"][s0]
                    ok = da.notna() & dg.notna() & (dg != 0) & (da != 0)
                    e["pair_n"] = int(ok.sum())
                    e["pair_agree"] = (
                        _r(float(((da > 0) == (dg > 0))[ok].mean()))
                        if ok.any()
                        else None
                    )
                    if ok.sum() >= 12:
                        e["pair_sp"] = _r(da[ok].corr(dg[ok], method="spearman"))
                S[c] = e
    out["align"] = S
    return out


def _r(v):
    return None if v is None or v != v else round(float(v), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--only", type=int, default=0, help="the first N strings (a smoke)")
    p.add_argument("--dit", default=None)
    p.add_argument("--label", default="ruler")
    p.add_argument(
        "--reread", default="", help="a run dir: read and score its renders again"
    )
    a = p.parse_args()
    home = resolve_under_home(RESEED)
    for q in (
        home,
        home / "src",
    ):  # the ruler's readers and scores; no bootstrap (it names a pack)
        sys.path.append(str(q))
    from bench._common import make_run_dir, write_result

    items = json.loads(resolve_under_home(RULER).read_text("utf-8"))["items"]
    if a.only:
        items = items[:: max(1, len(items) // a.only)][: a.only]
    if a.reread:
        out = Path(a.reread)
        meta = json.loads((out / "meta.json").read_text("utf-8"))
    else:
        out = make_run_dir("xattn_loc", f"gen-{a.label}")
        meta = render_all(a, items, out)
    rows = read_all(meta, out)
    with (out / "gen_rows.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    S = summarize(rows)
    write_result(out, script=__file__, args=a, metrics=S, label=a.label)
    print(json.dumps(S["read"], indent=1), flush=True)
    for c, e in S["align"].items():
        if "_hi_" in c or "_mid_" in c:
            print(f"  {c:<22} {e}", flush=True)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
