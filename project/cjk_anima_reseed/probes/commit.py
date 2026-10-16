#!/usr/bin/env python
"""commit — when an arm's render leaves the EN scene, read off the denoising
trajectory (user 10-16: retrain_kana's plain-banner habit, the reseed's
motivation, against the cold arms; lens: Kim et al. 2025, "How Diffusion
Models Memorize", arXiv 2509.25705, `_archive/paper/` — early CFG
overestimation of a cond-carried template collapses the trajectory onto it).

The pack rows sit in the cond branch only (T5("") has none), so a layout a row
carries is a cond-only injection that CFG scales by g at σ ≈ 1 — the paper's
setting. Per render, on identity's render path (512², 28 steps, cfg 4, routed)
with a recorder in the TrajStatsRecorder hook (post-combine, pre-step), in the
latent scaled by the VAE's per-channel std (linear, so x_t = (1−σ)x₀ + σx_T
holds):

- ``dev``     ‖x₀(arm) − x₀(EN)‖ / ‖x₀(EN)‖ — how far the final latent left
              the EN render of the same prompt and seed (same x_T)
- ``r[t]``    ⟨x̂₀ₜ(arm) − x̂₀ₜ(EN), D⟩ / ‖D‖², D = x₀(arm) − x₀(EN): the share of
              the final departure already in the step-t estimate (1 = decided;
              > 1 at t = 0 = overshoot)
- ``inj0``    the same projection of the step-0 cond − uncond x̂₀ gap (the
              paper's Wen et al. statistic, signed along D; ×g is CFG's share)
- ``w0[t]``   least squares x_t = w0·x₀ + wT·x_T against the flow schedule
              (1 − σ, σ); ``w0_ex`` = Σₜ (w0 − (1 − σ)) — destination fixed early
- ``white`` / ``ink`` / ``sat`` — pixel fractions near white / near black, mean
              saturation: the plain page read straight off the image
- per (arm, word, prompt): seed diversity ``div[k]`` = mean per-dim variance of
              the step-k x̂₀ across the seeds, over EN's at the same prompt

Arms render ``hira_cold_hp``'s 13 ``read`` words × p00 / p01 × ``--seeds`` as
``<prompt>. Text reads as "<key>".``; EN renders ``eval.enref.enref_caption``
(the shared refs' ``hi``) with the delta off, once per prompt × seed. Every
render is drawn fresh (the trajectory is the measurement); PNGs to
``output/cjk_anima_reseed/commit/<arm>/``.

    make daemon-run ARGS="project/cjk_anima_reseed/probes/commit.py --arms retrain_kana,hira_cold_sball_lr3,hira_cold_sball_lr3_x2 --label rk_vs_x2"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, OUT, SCALE_OUT, bootstrap  # noqa: E402

bootstrap()
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed

EN = "EN"
EN_WORD = "hi"  # native_enref/512_28_4's word
PROMPTS = (0, 1)
DIV_STEPS = (3, 7, 14)
ROOT = OUT / "commit"
SRC = SCALE_OUT / "retrain_kana" / "native_r4_plain" / "native_reads.json"
ENREF = SCALE_OUT / "native_enref" / "512_28_4"


class Capture:
    """One generation's trajectory, on the GPU, in std-scaled latent units."""

    def __init__(self, g: float):
        self.g = g
        self.sig, self.xt, self.x0, self.inj0 = [], [], None, None

    def record(self, i, sigma, latents, noise_pred, uncond_noise_pred=None):
        import torch

        from library.inference.traj_stats import QWEN_LATENTS_STD

        s = float(sigma)
        xt, v = latents.float().squeeze(2), noise_pred.float().squeeze(2)
        std = torch.tensor(QWEN_LATENTS_STD, device=xt.device).view(1, -1, 1, 1)
        x0 = (xt - s * v) / std
        if i == 0:
            vu = uncond_noise_pred.float().squeeze(2)
            vc = vu + (v - vu) / self.g
            self.inj0 = (s * (vu - vc) / std).half()  # x̂₀(cond) − x̂₀(uncond)
            self.x0 = []
        self.sig.append(s)
        self.xt.append((xt / std).half())
        self.x0.append(x0.half())

    def flush(self):
        pass


class Hook:
    """Stands in for ``TrajStatsRecorder`` in ``library.inference.generation``:
    hands the loop the probe's live capture."""

    live: Capture | None = None

    @classmethod
    def from_args(cls, args, *, seed=None):
        return cls.live


WORDS = 0


def grid() -> list:
    from reseed.config import load

    prompts = {
        m["pi"]: m["prompt"] for m in json.loads(SRC.read_text(encoding="utf-8"))
    }
    return [
        {"key": k, "pi": pi, "prompt": prompts[pi]}
        for k in load("hira_cold_hp").read[: WORDS or None]
        for pi in PROMPTS
    ]


def run(r, caption: str, seed: int, fn: Path) -> dict:
    """Render once with the hook live; ``{sig, xt, x0, inj0, final}`` (CPU
    fp16 stacks, ``(28, C, H, W)``) and the pixel reads."""
    import copy

    import numpy as np
    import torch

    from common.models import decode_image
    from library.inference.generation import generate
    from library.inference.traj_stats import QWEN_LATENTS_STD

    cap = Hook.live = Capture(float(r.args.guidance_scale))
    a2 = copy.deepcopy(r.args)
    a2.prompt, a2.seed, a2.image_size = caption, seed, (512, 512)
    with torch.no_grad():
        lat = generate(a2, r.gen, r.shared)
    Hook.live = None
    img = decode_image(r.vae, lat, r.device)
    fn.parent.mkdir(parents=True, exist_ok=True)
    img.save(fn)
    f = lat.float()
    f = f.squeeze(2) if f.dim() == 5 else f
    std = torch.tensor(QWEN_LATENTS_STD, device=f.device).view(1, -1, 1, 1)
    px = np.asarray(img.convert("RGB"), np.float32) / 255.0
    hi, lo = px.max(-1), px.min(-1)
    return {
        "sig": cap.sig,
        "xt": torch.cat(cap.xt).cpu(),
        "x0": torch.cat(cap.x0).cpu(),
        "inj0": cap.inj0[0].cpu(),
        "final": (f / std)[0].half().cpu(),
        "white": float(((lo > 0.85) & (hi - lo < 0.08)).mean()),
        "ink": float((hi < 0.25).mean()),
        "sat": float((hi - lo).mean()),
    }


def self_decomp(t: dict) -> dict:
    """``x_t = w0·x₀ + wT·x_T`` per step, against ``(1 − σ, σ)``."""
    import torch

    x0, xT = t["final"].float().flatten(), t["xt"][0].float().flatten()
    A = torch.stack([x0, xT], 1)
    W = torch.linalg.lstsq(A, t["xt"].float().flatten(1).T).solution  # (2, steps)
    sig = torch.tensor(t["sig"])
    return {
        "w0": [round(float(v), 4) for v in W[0]],
        "wT": [round(float(v), 4) for v in W[1]],
        "w0_ex": round(float((W[0] - (1 - sig)).sum()), 4),
        "wT_def": round(float(-(W[1] - sig).sum()), 4),
    }


def vs_en(t: dict, e: dict) -> dict:
    D = (t["final"].float() - e["final"].float()).flatten()
    dd = max(float(D @ D), 1e-12)
    r = ((t["x0"].float() - e["x0"].float()).flatten(1) @ D) / dd
    inj = float((t["inj0"].float() - e["inj0"].float()).flatten() @ D) / dd
    return {
        "dev": round(dd**0.5 / float(e["final"].float().norm()), 4),
        "r": [round(float(v), 4) for v in r],
        "inj0": round(inj, 4),
        "wen0": round(float(t["inj0"].float().pow(2).mean().sqrt()), 4),
    }


def div(trajs: list) -> dict:
    """Mean per-dim variance of the x̂₀ estimate across seeds, at ``DIV_STEPS``
    and the end (x_t itself is the seeds' noise early on — ratio ≡ 1)."""
    import torch

    out = {}
    for k in DIV_STEPS:
        out[f"s{k}"] = float(
            torch.stack([t["x0"][k].float() for t in trajs]).var(0).mean()
        )
    out["end"] = float(torch.stack([t["final"].float() for t in trajs]).var(0).mean())
    return out


def stats(recs: list, arms: list) -> dict:
    """Paired arm − retrain_kana (Wilcoxon, same word / prompt / seed) and the
    pooled Spearman of each early read against ``dev`` / ``white``."""
    from scipy.stats import spearmanr, wilcoxon

    def val(m, k):
        if k.startswith("r"):
            return m["r"][int(k[1:])]
        return m[k]

    keys = [
        "dev",
        "white",
        "ink",
        "sat",
        "inj0",
        "wen0",
        "w0_ex",
        "wT_def",
        "r0",
        "r3",
        "r7",
    ]
    by = {(m["arm"], m["key"], m["pi"], m["seed"]): m for m in recs}
    out = {"paired": {}, "spearman": {}}
    base = "retrain_kana"
    for a in arms:
        if a == base or base not in arms:
            continue
        slots = [s[1:] for s in by if s[0] == a]
        out["paired"][a] = {}
        for k in keys:
            x = [val(by[(a, *s)], k) for s in slots]
            y = [val(by[(base, *s)], k) for s in slots]
            d = [u - v for u, v in zip(x, y)]
            p = wilcoxon(d).pvalue if any(d) else 1.0
            out["paired"][a][k] = {
                "arm": round(sum(x) / len(x), 4),
                base: round(sum(y) / len(y), 4),
                "p": round(float(p), 5),
            }
    ms = [m for m in recs if m["arm"] != EN]
    for k in ("inj0", "wen0", "w0_ex", "r0", "r3", "r7"):
        out["spearman"][k] = {}
        for tgt in ("dev", "white"):
            rho, p = spearmanr([val(m, k) for m in ms], [m[tgt] for m in ms])
            out["spearman"][k][tgt] = {
                "rho": round(float(rho), 3),
                "p": round(float(p), 5),
            }
    return out


def sheets(recs: list, arms: list, seeds: int, out: Path) -> None:
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    by = {(m["arm"], m["key"], m["pi"], m["seed"]): m for m in recs}
    for pi in PROMPTS:
        rows = [
            (
                Image.open(ROOT / EN / f"p{pi:02d}_s{s}.png").convert("RGB"),
                [f"EN p{pi:02d} s{s}", "", "", ""],
            )
            for s in range(seeds)
        ]
        for m in grid():
            if m["pi"] != pi:
                continue
            for a in arms:
                for s in range(seeds):
                    x = by[(a, m["key"], pi, s)]
                    rows.append(
                        (
                            Image.open(x["file"]).convert("RGB"),
                            [
                                f"{m['key']} s{s}",
                                a[-18:],
                                f"dev {x['dev']:.2f} w {x['white']:.2f}",
                                f"r0 {x['r'][0]:.2f} r7 {x['r'][7]:.2f}",
                            ],
                        )
                    )
        contact_sheet(rows, out / f"sheet_p{pi:02d}.png", thumb=160, cols=seeds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", required=True, help="comma list (retrain_kana first)")
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--words", type=int, default=0, help="first N words only (smoke)")
    a = p.parse_args()
    arms = list(dict.fromkeys(a.arms.split(",")))
    global WORDS
    WORDS = a.words

    import numpy as np
    import torch
    from PIL import Image

    import library.inference.generation as gen_mod
    from eval.enref import enref_caption, enref_file
    from eval.ruler.arms import arm_dirs, tables
    from eval.ruler.render import Renderer, check

    unknown = set(arms) - set(arm_dirs())
    assert not unknown, f"unknown arms {unknown}"
    gen_mod.TrajStatsRecorder = Hook
    _, tabs = tables(arms + ["retrain_kana"])
    r = Renderer()
    check(r, tabs["retrain_kana"])
    t0 = time.time()

    def tick(what, n, tot):
        print(f"  {what}: {n} / {tot} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    r.set_arm(None)
    en, recs = {}, []
    prompts = {m["pi"]: m["prompt"] for m in grid()}
    for pi in PROMPTS:
        for s in range(a.seeds):
            fn = ROOT / EN / f"p{pi:02d}_s{s}.png"
            en[(pi, s)] = run(r, enref_caption(prompts[pi], EN_WORD), s, fn)
            ref = enref_file(ENREF, pi, s)
            if ref.is_file():  # the shared EN refs, drawn on the ruler path
                d = np.abs(
                    np.asarray(Image.open(fn), np.float32)
                    - np.asarray(Image.open(ref).convert("RGB"), np.float32)
                ).mean()
                print(
                    f"check EN p{pi:02d} s{s} vs {ref.name}: mean |Δpx| {d:.3f}",
                    flush=True,
                )
                assert d < 12.0, "the EN render does not reproduce the shared ref"
            tick("EN", len(en), len(PROMPTS) * a.seeds)
    for pi in PROMPTS:
        recs.append(
            {"arm": EN, "pi": pi, "div": div([en[(pi, s)] for s in range(a.seeds)])}
        )
    for e in en.values():
        e.update(self_decomp(e))

    g = grid()
    for arm in arms:
        r.set_arm(tabs[arm])
        n = 0
        for m in g:
            trajs = []
            for s in range(a.seeds):
                fn = ROOT / arm / f"{m['key']}_p{m['pi']:02d}_s{s}.png"
                t = run(r, f'{m["prompt"]}. Text reads as "{m["key"]}".', s, fn)
                trajs.append(t)
                recs.append(
                    {
                        "arm": arm,
                        "key": m["key"],
                        "pi": m["pi"],
                        "seed": s,
                        "file": str(fn),
                        **{k: round(t[k], 4) for k in ("white", "ink", "sat")},
                        **self_decomp(t),
                        **vs_en(t, en[(m["pi"], s)]),
                    }
                )
                n += 1
                tick(arm, n, len(g) * a.seeds)
            recs.append({"arm": arm, "key": m["key"], "pi": m["pi"], "div": div(trajs)})
        torch.cuda.empty_cache()

    renders = [m for m in recs if "seed" in m]
    divs = [m for m in recs if "div" in m]
    en_div = {m["pi"]: m["div"] for m in divs if m["arm"] == EN}
    res = {"arms": {}, "en": {}, "stats": stats(renders, arms)}
    for pi in PROMPTS:
        es = [en[(pi, s)] for s in range(a.seeds)]
        res["en"][f"p{pi:02d}"] = {
            k: round(sum(e[k] for e in es) / len(es), 4)
            for k in ("white", "ink", "sat", "w0_ex")
        }
    for arm in arms:
        ms = [m for m in renders if m["arm"] == arm]
        mean = lambda f: round(sum(map(f, ms)) / len(ms), 4)  # noqa: E731
        dv = [m for m in divs if m["arm"] == arm]
        res["arms"][arm] = {
            **{
                k: mean(lambda m, k=k: m[k])
                for k in (
                    "dev",
                    "white",
                    "ink",
                    "sat",
                    "inj0",
                    "wen0",
                    "w0_ex",
                    "wT_def",
                )
            },
            "r": [mean(lambda m, t=t: m["r"][t]) for t in range(len(ms[0]["r"]))],
            "w0": [mean(lambda m, t=t: m["w0"][t]) for t in range(len(ms[0]["w0"]))],
            "div_over_en": {
                k: round(sum(d["div"][k] / en_div[d["pi"]][k] for d in dv) / len(dv), 4)
                for k in dv[0]["div"]
            },
        }
        print(
            f"  {arm:<26} {json.dumps({k: v for k, v in res['arms'][arm].items() if k not in ('r', 'w0')})}",
            flush=True,
        )
        print(
            f"    r[0,1,3,7,14] {[res['arms'][arm]['r'][t] for t in (0, 1, 3, 7, 14)]}",
            flush=True,
        )
    print(f"  stats {json.dumps(res['stats'], ensure_ascii=False)}", flush=True)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"commit-{a.label}", root=HOME / "results"
    )
    sheets(renders, arms, a.seeds, run_dir / "sheets")
    (run_dir / "renders.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_anima_reseed/probes/commit.py",
        args=vars(a),
        metrics=res,
    )
    print(f"commit → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
