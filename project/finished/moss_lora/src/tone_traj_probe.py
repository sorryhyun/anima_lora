"""Where along a real i2i render the sanitize tone drift enters.

Follow-up to Phase 0 of docs/moss/proposal/tone_xt_trust.md (the trust k is too low to
turn a single-step δ into the render's ~6×). Renders the five held-out pages through the
normal ``run_generate`` path (the sanitize generate config: CFG 3 + skip-22, ER-SDE,
int8 base, fp32 stream, compile) and, per step, on the rest tokens (pixels the v3 seed-0
render left alone, ``tone_trust_probe.page_mask``), fits on the L / a / b tone directions
(Lab units):

- ``m``: the page tone the state implies, (x_t mean)/(1−σ), minus the reference's;
- ``xhat``: the guided x̂₀ the sampler steps toward, minus the reference's.

The final latent's fit is checked against the decoded PNG's Lab difference on the same
pixels. With Phase 0's cond-branch k(σ), ``xhat − k·m`` estimates the error x̂₀ adds on the
render's own state (δ on the trajectory), to set beside δ on a correct x_t.

    make daemon-run ARGS="--stall-timeout 900 project/finished/moss_lora/src/tone_traj_probe.py"
    .venv/bin/python project/finished/moss_lora/src/tone_traj_probe.py --analyze
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from tone_probe import PAGES
from tone_trust_probe import LORAS, page_mask

import library.moss.generate as G
from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.requests import GenerateRequest

INPUTS = "output/moss_i2i_eval/sanitize/inputs"
OUT = "project/finished/moss_lora/space/tone_traj_probe.json"
CONFIG = "configs/moss/i2i/sanitize/generate.toml"
# Phase 0 cond-branch k on the held-out pages (tone_trust_probe.json), v3 EMA.
K_SIG = [0.1, 0.3, 0.45, 0.6, 0.8]
K_V3 = {"L": [0.78, 0.52, 0.37, 0.26, 0.11], "b": [0.655, 0.355, 0.23, 0.14, 0.055]}


def k_at(tone: str, sigma: float) -> float:
    return float(np.interp(sigma, [0.0, *K_SIG, 1.0], [1.0, *K_V3[tone], 0.0]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loras", default="v3,v3_raw")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--out", default=OUT)
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()
    if args.analyze:
        summarize(json.loads(resolve_under_home(args.out).read_text()))
        return

    dirs = load_file(
        resolve_under_home("library/moss/calibration/tone_dirs.safetensors")
    )
    D = torch.stack([dirs[t].float() for t in ("L", "a", "b")], 1).cuda()
    to_lab = torch.linalg.pinv(D).T

    # Rest tokens / pixels per page, keyed by the latent grid (every page differs).
    masks = {}
    for page in PAGES:
        src = moss_cache.load_rgb(resolve_under_home(f"{INPUTS}/{page}.png"))
        w, h = moss_cache.target_size(*src.size, 1024)
        px = page_mask(page, (w, h), INPUTS)
        tok = px.reshape(h // 16, 16, w // 16, 16).all((1, 3))
        masks[(h // 16, w // 16)] = (page, torch.from_numpy(tok).cuda(), px)

    rows: list[dict] = []
    state: dict = {}
    real_denoise = G.denoise
    from library.inference.sampling import ERSDESampler

    real_step = ERSDESampler.step

    # ER-SDE's step sees x_t and the guided x̂₀ of every step, Spectrum forecasts too.
    def step(self, latents, x0, i, *a, **kw):
        state["record"](i, latents, x0)
        return real_step(self, latents, x0, i, *a, **kw)

    def denoise(
        transformer, latents, cond, cond_mask, uncond, uncond_mask, sigmas, **kw
    ):
        ref, seed = kw.get("ref"), kw.get("seed")
        page, rest, _ = masks[tuple(latents.shape[-2:])]
        n = rest.sum()
        ref_dc = (ref.float() * rest).sum((2, 3))[0] / n
        sig = sigmas.float().cpu().tolist() + [0.0]
        steps = []

        def record(i, x_t, x_hat):
            s = sig[i]
            m = (x_t.float() * rest).sum((2, 3))[0] / n / max(1 - s, 1e-6)
            xh = (x_hat.float() * rest).sum((2, 3))[0] / n
            steps.append(
                dict(
                    i=i,
                    sigma=s,
                    m=((m - ref_dc) @ to_lab).tolist(),
                    xhat=((xh - ref_dc) @ to_lab).tolist(),
                )
            )

        state["record"] = record
        out = real_denoise(
            transformer, latents, cond, cond_mask, uncond, uncond_mask, sigmas, **kw
        )
        if len(steps) != len(sig) - 1:
            raise RuntimeError(f"recorded {len(steps)} of {len(sig) - 1} steps")
        fin = (out.float().to(rest.device) * rest).sum((2, 3))[0] / n
        rows.append(
            dict(
                lora=state["lora"],
                page=page,
                seed=seed,
                steps=steps,
                final=((fin - ref_dc) @ to_lab).tolist(),
            )
        )
        return out

    G.denoise = denoise
    ERSDESampler.step = step
    for name in args.loras.split(","):
        state["lora"] = name
        out_dir = f"output/tests/moss_sanitize_traj/{name}"
        req = GenerateRequest.from_argv(
            [
                "--config",
                CONFIG,
                "--lora",
                LORAS[name],
                "--ref_image",
                INPUTS,
                "--seeds",
                args.seeds,
                "--out_dir",
                out_dir,
            ]
        )
        G.run_generate(req)
        # Decoded check: Lab of the render vs the input, rest pixels.
        from bench.moss_i2i_eval.tone_map import lab

        for r in rows:
            if r["lora"] != name:
                continue
            _, _, px = next(v for v in masks.values() if v[0] == r["page"])
            png = resolve_under_home(f"{out_dir}/000_{r['page']}_base_s{r['seed']}.png")
            if not png.is_file():
                continue
            src = moss_cache.load_rgb(resolve_under_home(f"{INPUTS}/{r['page']}.png"))
            a, _ = lab(Image.open(png))
            b, _ = lab(moss_cache.fit(src, a.shape[1], a.shape[0]))
            d = (a - b)[px]
            r["decoded"] = [float(d[:, c].mean()) for c in range(3)]

    out = resolve_under_home(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dict(args=vars(args), rows=rows)
    out.write_text(json.dumps(data))
    summarize(data)
    print(f"-> {out}", flush=True)


def summarize(data: dict) -> None:
    probe_sig = [0.95, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0]
    for lora in dict.fromkeys(r["lora"] for r in data["rows"]):
        rs = [r for r in data["rows"] if r["lora"] == lora]
        fin = np.array([r["final"] for r in rs])
        dec = np.array([r.get("decoded", [np.nan] * 3) for r in rs])
        print(
            f"\n=== {lora}: {len(rs)} renders; final latent fit ΔL {fin[:, 0].mean():+.2f} "
            f"Δb {fin[:, 2].mean():+.2f} | decoded ΔL {np.nanmean(dec[:, 0]):+.2f} "
            f"Δb {np.nanmean(dec[:, 2]):+.2f}"
        )
        print(
            f"{'σ':>5} | {'m L':>6} {'m b':>6} | {'x̂₀ L':>6} {'x̂₀ b':>6} | "
            f"{'δtraj L':>7} {'δtraj b':>7}   (mean over renders)"
        )
        for target in probe_sig:
            vals = []
            for r in rs:
                st = min(r["steps"], key=lambda s: abs(s["sigma"] - target))
                s = st["sigma"]
                m, xh = np.array(st["m"]), np.array(st["xhat"])
                dl = xh[0] - k_at("L", s) * m[0]
                db = xh[2] - k_at("b", s) * m[2]
                vals.append((s, m[0], m[2], xh[0], xh[2], dl, db))
            v = np.array(vals).mean(0)
            print(
                f"{v[0]:5.3f} | {v[1]:+6.2f} {v[2]:+6.2f} | {v[3]:+6.2f} {v[4]:+6.2f} | "
                f"{v[5]:+7.2f} {v[6]:+7.2f}"
            )
        print("per page final Δb (latent fit / decoded):")
        for page in dict.fromkeys(r["page"] for r in rs):
            pr = [r for r in rs if r["page"] == page]
            print(
                f"  {page:>18}: "
                + "  ".join(
                    f"s{r['seed']} {r['final'][2]:+.2f}/{r.get('decoded', [0, 0, np.nan])[2]:+.2f}"
                    for r in pr
                )
            )


if __name__ == "__main__":
    main()
