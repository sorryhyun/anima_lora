#!/usr/bin/env python
"""jamo_read — ``proposal_jamo`` phase 1's reads on ``jamo_j64`` / ``jamo_f64``.

``fit`` (CPU, R3): the jamo model fitted to F64's free rows (effective units,
least squares, minimum norm — 64 rows under 106 / 68 vectors), with and without
``cls``: leave-one-out cos (each trained syllable predicted from the other 63),
J64's composed rows against F64's on the 64, and J64's H rows against the
F64-regressed H rows. Writes the regressed arm ``jamo_f64_reg`` (F64's rows,
every other syllable composed from the fit) → ``OUT/jamo_f64_reg/trained.pt``.

``render`` (GPU, R1 / R2): each syllable alone in a speech bubble, KO caption,
through the ruler's ``Renderer`` (512², seed 0): the 64 under J64 and F64, H
and a few H-only words under J64 and F64-reg. No floor arm (the pack's
untrained Hangul rows; user 10-10: skip it). Sheets → ``results/<ts>-jamo-read/``,
read by eye per jamo position.

    .venv/bin/python project/cjk_anima_reseed/probes/jamo_read.py fit
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/jamo_read.py render"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

J, F, REG = "jamo_j64", "jamo_f64", "jamo_f64_reg"
WORDS = ("정리", "소리", "아래", "문제", "시장")  # H syllables only
SIZE = (512, 512)


def setup():
    from reseed.config import PACKS, load

    run = load(J)
    os.environ["ANIMA_VOCAB_PACK"] = PACKS[run.pack]
    from reseed import bootstrap

    bootstrap()
    return run


def sets() -> dict:
    return json.loads((HOME / "assets" / "jamo_sets.json").read_text())


def design(syls, cls: bool):
    """``(n, k)`` 0/1: the jamo model's vectors each syllable sums."""
    import torch
    from reseed.jamo import CLASSES, N_CHO, N_JUNG, N_JONG, codes

    nc = N_CHO * (len(CLASSES) if cls else 1)
    k = 1 + nc + N_JUNG + N_JONG - 1
    X = torch.zeros(len(syls), k)
    for i, (cho, c, jung, jong) in enumerate(codes(syls).tolist()):
        X[i, 0] = 1
        X[i, 1 + (cho * len(CLASSES) + c if cls else cho)] = 1
        X[i, 1 + nc + jung] = 1
        if jong:
            X[i, 1 + nc + N_JUNG + jong - 1] = 1
    return X


def effective(delta: dict, ext_ids):
    at = {int(e): i for i, e in enumerate(delta["ext_ids"])}
    return delta["raw"].float()[[at[int(e)] for e in ext_ids]] * float(
        delta["row_scale"]
    )


def fit() -> None:
    setup()
    import torch
    import torch.nn.functional as Fn
    from common.models import load_trained
    from reseed import OUT
    from reseed.jamo import ALL, syllable_rows

    s = sets()
    j64, held = list(s["J64"]), list(s["H"])
    ext = syllable_rows()
    dj, df = load_trained(OUT / J), load_trained(OUT / F)
    Yf = effective(df["delta"], [ext[c] for c in j64])
    Yj = effective(dj["delta"], [ext[c] for c in j64])
    Hj = effective(dj["delta"], [ext[c] for c in held])

    def cos(a, b):
        return Fn.cosine_similarity(a, b, dim=1)

    def stat(v):
        return {
            "mean": round(float(v.mean()), 3),
            "median": round(float(v.median()), 3),
        }

    out = {
        "norm": {
            "f64_64": stat(Yf.norm(dim=1)),
            "j64_64": stat(Yj.norm(dim=1)),
            "j64_H": stat(Hj.norm(dim=1)),
        },
        "cos_j64_f64_64": stat(cos(Yj, Yf)),
    }
    for cls in (True, False):
        X = design(j64, cls)
        loo = []
        for i in range(len(j64)):
            m = torch.arange(len(j64)) != i
            W = torch.linalg.pinv(X[m]) @ Yf[m]
            loo.append(float(cos((X[i : i + 1] @ W), Yf[i : i + 1])))
        W = torch.linalg.pinv(X) @ Yf
        tag = "cls" if cls else "plain"
        out[f"loo_cos_{tag}"] = stat(torch.tensor(loo))
        out[f"loo_cos_{tag}_per"] = dict(zip(j64, [round(v, 3) for v in loo]))
        out[f"fit_resid_{tag}"] = round(
            float((X @ W - Yf).norm() / Yf.norm()), 4
        )  # 0 when the 64 equations are independent
        Hr = design(held, cls) @ W
        out[f"cos_H_j64_vs_reg_{tag}"] = stat(cos(Hj, Hr))
        out[f"norm_H_reg_{tag}"] = stat(Hr.norm(dim=1))
        if cls:
            reg = design(list(ALL), True) @ W
    # the regressed arm: F64's delta, every syllable's row the fit's composition
    # (F64's 64 kept as trained; the seed rows as F64 has them)
    delta = dict(df["delta"])
    have = {int(e): i for i, e in enumerate(delta["ext_ids"])}
    raw = delta["raw"].float() * float(delta["row_scale"])
    trained = {ext[c] for c in j64}
    new_ids, new_rows = [], []
    for c, r in zip(ALL, reg):
        e = ext[c]
        if e in trained:
            continue
        if e in have:
            raw[have[e]] = r
        else:
            new_ids.append(e)
            new_rows.append(r)
    ids = [int(e) for e in delta["ext_ids"]] + new_ids
    raw = torch.cat([raw, torch.stack(new_rows)])
    order = sorted(range(len(ids)), key=ids.__getitem__)
    delta.update(ext_ids=[ids[j] for j in order], raw=raw[order], row_scale=1.0)
    dst = OUT / REG / "trained.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta": delta, "derived": f"{F} min-norm jamo fit (cls)"}, dst)
    (OUT / REG / "fit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(
        json.dumps({k: v for k, v in out.items() if not k.endswith("_per")}, indent=1)
    )
    worst = sorted(out["loo_cos_cls_per"].items(), key=lambda kv: kv[1])[:8]
    print("LOO worst (cls):", worst)
    print(f"→ {dst} ({len(new_ids)} rows added), {OUT / REG / 'fit.json'}")


def render(held: list | None) -> None:
    run = setup()
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE_KO"] = "1"
    from bench._common import make_run_dir, write_result
    from common.prompts import TPL_BUBBLE
    from common.readers import contact_sheet
    from PIL import Image
    from reseed import HOME as RH
    from reseed import OUT
    from reseed.pools import relang

    from eval.ruler import SEED_RENDER, VIEW
    from eval.ruler.arms import PACK_ARMS, tables
    from eval.ruler.render import Renderer, check

    VIEW.pack = run.pack
    s = sets()
    jobs = {  # sheet → (arms, texts)
        "trained": ((J, F), list(s["J64"])),
        "held": ((J, REG), list(s["H"]) + list(WORDS)),
    }
    if held:  # H and the words only, under these arms side by side
        jobs = {"held": (tuple(held), list(s["H"]) + list(WORDS))}
    every = list(dict.fromkeys(a for arms, _ in jobs.values() for a in arms))
    PACK_ARMS[run.pack].update({a: OUT / a for a in every})
    _, tabs = tables(every + ["retrain_kana"])
    r = Renderer()
    info = {"check_rk": check(r, tabs["retrain_kana"])}
    t0 = time.time()
    files = {}
    for arm in every:
        r.set_arm(tabs[arm])
        its = [t for arms, ts in jobs.values() if arm in arms for t in ts]
        for n, t in enumerate(its):
            fn = OUT / J / "render" / arm / f"bubble_{t}_s{SEED_RENDER}.png"
            r.render(fn, relang(TPL_BUBBLE.format(t), "korean"), SEED_RENDER, SIZE)
            files[arm, t] = fn
            print(
                f"  {arm}: {n + 1} / {len(its)} {t} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    out = make_run_dir("cjk_anima_reseed", label="jamo-read", root=RH / "results")
    sheets = []
    for name, (arms, ts) in jobs.items():
        tiles = [
            (Image.open(files[a, t]), [f"{t}  {a.removeprefix('jamo_')}"])
            for t in ts
            for a in arms
        ]
        cols = 8 if len(arms) == 2 else 3 * len(arms)  # arms side by side
        per = 4 * cols
        for k in range(0, len(tiles), per):
            fn = out / f"sheet_{name}_{k // per}.png"
            contact_sheet(tiles[k : k + per], fn, thumb=320, cols=cols)
            sheets.append(fn.name)
    info["minutes"] = round((time.time() - t0) / 60, 1)
    write_result(
        out,
        script=__file__,
        args={"held": held},
        label="jamo-read",
        metrics={"renders": len(files), **info},
        artifacts=sheets,
    )
    print(f"→ {out}: {len(files)} renders, {info['minutes']} min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fit", "render"))
    p.add_argument("--held", default="", help="render: H only, these arms (a,b,…)")
    a = p.parse_args()
    if a.mode == "fit":
        fit()
    else:
        render([x for x in a.held.split(",") if x] or None)
