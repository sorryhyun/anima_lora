#!/usr/bin/env python3
"""MIST render A/B — head-to-head vs the guidance variants it might replace.

Phase-1 eyeball+tone test for MIST (library/inference/corrections/mist_core.py,
reimplementing Peng et al. ICML 2026, repo-root 18927_MIST_*.pdf). The point of
this bench is the *substitution* question, not "MIST vs plain CFG": MIST occupies
the same velocity-combine slot as SMC-CFG (so it would REPLACE smc), and it's an
alternative to FSG's pre-step calibration (a different hook — they can also STACK).
So every arm runs the same noise / prompt / seed / sampler, and the only thing
that changes is the guidance treatment:

  * **baseline**   — plain deterministic CFG (reference)
  * **smc**        — SMC-CFG combine (the incumbent in the Spectrum node)
  * **cfg++**      — CFG++ σ-scheduled reweight (the shipped flow-matching baseline)
  * **fsg/cfg**    — FSG foresight calibration on the CFG substrate (shipped combo)
  * **mist**       — full MIST combine (ST + IA)
  * **fsg+mist**   — FSG calibration AND MIST combine (orthogonal-stack: does
                     FSG still add anything once MIST renormalizes the combine?)

With ``--ablate`` two more arms reproduce the paper's IA/ST split (Tables 2/9):
  * **mist-IA**    — Invariant Alignment only (DC+EM; ST off)
  * **mist-ST**    — Stability Thresholding only (TD+SS; IA off)

The hypothesis to falsify (per stored Anima findings): TD is near-inert because
Anima's guidance delta already decays front-loaded (cross-attn drive peaks at σ=1,
~0.02 floor below σ0.85), and IA's variance renorm may be redundant with what
CFG++ already does. So watch whether **mist-IA ≈ mist** (ST dead) and whether
**mist ⪅ cfg++** (no net win). A real win = mist beats BOTH smc and cfg++ on
prompt adherence without the tone collapse, and fsg+mist ≈ mist (FSG redundant).

Per-arm mean HSV saturation + RMS contrast (and Δ% vs baseline) land in
result.json and on each panel label — MIST's whole pitch is taming
over-saturation/variance-explosion, so the tone numbers are a first-order signal
(though they certify *change*, not *quality* — the contact sheet is the gate).

Sampler: ``--sampler euler`` (deterministic, isolates the operator) or
``er_sde`` (production SDE; seed shared across arms so injected noise is identical).

Run from anima_lora/::

    uv run python bench/mist/render_compare.py --num_prompts 4 --compile
    uv run python bench/mist/render_compare.py --sampler er_sde --infer_steps 28 \
        --guidance 7 --ablate --num_prompts 6 --num_seeds 2 --compile
    # sweep SS ceiling γ and the high-w regime MIST targets:
    uv run python bench/mist/render_compare.py --guidance 10 --mist_gamma 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # _archive/bench/mist/ -> repo root
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from anima_lora import decode_to_pil, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from _archive.bench.fsg.render_compare import (  # noqa: E402
    _fsg_calibrate,
    _norm,
    _sat_contrast,
)
from _archive.bench.fsg.probe_golden_path import _sample_captions, _velocity  # noqa: E402
from library.anima import models as anima_models  # noqa: E402
from _archive.bench.mist.mist_core import MISTState  # noqa: E402  (archived alongside this bench)
from library.inference.corrections.smc_cfg import SMCCFGState  # noqa: E402
from library.inference.sampling import (  # noqa: E402
    ERSDESampler,
    cfgpp_guidance_weight,
    get_timesteps_sigmas,
)
from library.inference.text import prepare_text_inputs  # noqa: E402

log = logging.getLogger("bench.mist.render")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _grid(
    images: list[Image.Image],
    labels: list[str],
    caption: str,
    *,
    ncols: int = 4,
    cell: int = 512,
) -> Image.Image:
    """2-D grid of arm renders (downscaled to ``cell`` px) — one per (prompt,seed).

    Replaces the single horizontal strip × all-prompts contact sheet, which blew
    up to ~100 MB. Each cell is thumbnailed to ``cell`` px (long edge) so a full
    8-arm 2×4 grid lands at ~1-2 MB instead of tens of MB.
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    thumbs = [im.copy() for im in images]
    for t in thumbs:
        t.thumbnail((cell, cell))
    tw = max(t.width for t in thumbs)
    th = max(t.height for t in thumbs)
    head, lab_h = 24, 18
    W = ncols * tw
    H = head + nrows * (th + lab_h)
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    cap = caption if len(caption) <= 200 else caption[:197] + "…"
    d.text((6, 4), cap[:160], fill="black")
    for k, (im, lab) in enumerate(zip(thumbs, labels)):
        r, c = divmod(k, ncols)
        x = c * tw
        y = head + r * (th + lab_h)
        d.text((x + 4, y), lab, fill="black")
        grid.paste(im, (x, y + lab_h))
    return grid


def _make_combiner(tag: str, args):
    """Fresh per-trajectory stateful combiner (None = plain CFG / handled inline).

    Sweep tags ``mist:<mode>:<strength>`` isolate the IA knobs with ST OFF (ST is
    inert on Anima — proven in the first run, and the paper's own Table 11/12 show
    it barely moves): mode ∈ {full(DC+EM), dc, em}, strength = ia_strength blend.
    ``mist_paper`` is the paper-faithful full config (ST on, γ=1.5, T_clip=1).
    """
    if tag == "smc":
        return SMCCFGState(lam=args.smc_lambda, alpha=args.smc_alpha)
    if tag == "mist" or tag == "mist_paper":
        return MISTState(gamma=args.mist_gamma, t_clip=args.mist_t_clip)
    if tag == "mist_ia":
        return MISTState(use_td=False, use_ss=False, use_dc=True, use_em=True)
    if tag == "mist_st":
        return MISTState(use_td=True, use_ss=True, use_dc=False, use_em=False,
                         gamma=args.mist_gamma, t_clip=args.mist_t_clip)
    if tag and tag.startswith("mist:"):
        _, mode, s = tag.split(":")
        return MISTState(
            use_td=False, use_ss=False,
            use_dc=mode in ("full", "dc"),
            use_em=mode in ("full", "em"),
            ia_strength=float(s),
        )
    return None


@torch.no_grad()
def _trajectory(
    anima,
    x0,
    sig,
    sigmas_t,
    embed,
    nembed,
    pad,
    guidance,
    *,
    band,
    cfgpp_lambda,
    combiner,
    fsg,
    sampler,
    er_seed,
    device,
    dtype,
):
    """One denoise trajectory. ``combiner`` (SMC/MIST) replaces the CFG combine;
    ``band``≠None grafts FSG calibration; ``cfgpp_lambda``≠None uses the CFG++
    reweight. combiner takes precedence over cfgpp (they're the same slot)."""
    x = x0.clone()
    n = len(sig) - 1
    er = (
        ERSDESampler(sigmas_t, seed=er_seed, device=device)
        if sampler == "er_sde"
        else None
    )
    for i in range(n):
        s_i = sig[i]
        if band is not None and band[0] <= s_i <= band[1]:
            x = _fsg_calibrate(
                anima,
                x,
                s_i,
                fsg["dsig"],
                fsg["gamma"],
                fsg["k"],
                embed,
                nembed,
                pad,
                i,
            )
        vc = _velocity(anima, x, s_i, embed, pad, i)
        vu = _velocity(anima, x, s_i, nembed, pad, i)
        if combiner is not None:
            # MIST consumes the step index (Temporal Decay); SMC does not.
            if isinstance(combiner, MISTState):
                noise_pred = combiner.combine(vc, vu, guidance, step_i=i)
            else:
                noise_pred = combiner.combine(vc, vu, guidance)
        elif cfgpp_lambda is not None:
            w_eff = cfgpp_guidance_weight(s_i, sig[i + 1], cfgpp_lambda)
            noise_pred = vu + w_eff * (vc - vu)
        else:
            noise_pred = vu + guidance * (vc - vu)
        if er is not None:
            denoised = x.float() - s_i * noise_pred.float()
            x = er.step(x, denoised, i).to(dtype)
        else:
            x = x - (sig[i] - sig[i + 1]) * noise_pred
    return x


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", nargs="+", default=None)
    ap.add_argument("--caption_dir", default="post_image_dataset/resized")
    ap.add_argument("--num_prompts", type=int, default=4)
    ap.add_argument("--neg_prompt", default="")
    ap.add_argument("--num_seeds", type=int, default=1)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=20)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--sampler", choices=["euler", "er_sde"], default="euler")
    ap.add_argument("--grid_cols", type=int, default=4,
                    help="Columns in the per-(prompt,seed) arm grid. Default 4 → 2×4.")
    ap.add_argument("--cell_px", type=int, default=512,
                    help="Long-edge px per grid cell (thumbnailed). Default 512.")
    # MIST knobs
    ap.add_argument(
        "--mist_gamma",
        type=float,
        default=None,
        help="SS guidance-to-structure ceiling γ. Default None → use w.",
    )
    ap.add_argument(
        "--mist_t_clip",
        type=int,
        default=2,
        help="Skip Temporal Decay for the first N steps. Default 2.",
    )
    ap.add_argument(
        "--ablate",
        action="store_true",
        help="Add mist-IA (IA only) and mist-ST (ST only) arms.",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="IA-knob sweep arm set: base, smc, paper-MIST, full/DC/EM × "
        "ia_strength. Isolates the color-collapse lever (ST off). Drops cfg++/fsg.",
    )
    # SMC knobs (incumbent)
    ap.add_argument("--smc_lambda", type=float, default=5.0)
    ap.add_argument("--smc_alpha", type=float, default=0.2)
    # CFG++ / FSG knobs
    ap.add_argument("--cfgpp_lambda", type=float, default=1.5)
    ap.add_argument("--fsg_lo", type=float, default=0.59)
    ap.add_argument("--fsg_hi", type=float, default=0.75)
    ap.add_argument("--fsg_gamma", type=float, default=None)
    ap.add_argument("--fsg_dsigma", type=float, default=0.1)
    ap.add_argument("--fsg_k", type=int, default=3)
    add_common_args(ap)
    args = ap.parse_args()

    prompts = args.prompts or _sample_captions(
        args.caption_dir, args.num_prompts, args.seed
    )
    band = (args.fsg_lo, args.fsg_hi)
    fsg = {
        "gamma": args.guidance if args.fsg_gamma is None else args.fsg_gamma,
        "dsig": args.fsg_dsigma,
        "k": args.fsg_k,
    }

    # (label, fsg_band, cfgpp_lambda, combiner_tag)
    if args.sweep:
        # IA-knob sweep: is EM the color killer? does dialing ia_strength recover
        # color while keeping the win? (ST off — proven inert.) base/smc reference.
        arms = [
            ("baseline", None, None, None),
            ("smc", None, None, "smc"),
            ("mist-paper", None, None, "mist_paper"),
            ("mist full s1.0", None, None, "mist:full:1.0"),
            ("mist full s0.5", None, None, "mist:full:0.5"),
            ("mist full s0.25", None, None, "mist:full:0.25"),
            ("mist DC s1.0", None, None, "mist:dc:1.0"),
            ("mist EM s1.0", None, None, "mist:em:1.0"),
        ]
    else:
        arms = [
            ("baseline", None, None, None),
            ("smc", None, None, "smc"),
            (f"cfg++ λ{args.cfgpp_lambda:g}", None, args.cfgpp_lambda, None),
            (f"fsg/cfg {band[0]:g}-{band[1]:g}", band, None, None),
            ("mist", None, None, "mist"),
            ("fsg+mist", band, None, "mist"),
        ]
        if args.ablate:
            arms += [
                ("mist-IA", None, None, "mist_ia"),
                ("mist-ST", None, None, "mist_st"),
            ]
    log.info(
        f"arms: {[a[0] for a in arms]}  |  guidance={args.guidance} sampler={args.sampler}"
    )

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype
    vae = load_vae(args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True)

    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sig = [float(s) for s in sigmas]
    C = anima_models.Anima.LATENT_CHANNELS
    hl, wl = args.height // 8, args.width // 8
    pad = torch.zeros(1, 1, hl, wl, dtype=dtype, device=device)

    run_dir = make_run_dir("mist", label=args.label or "render-compare")
    rows: list[dict] = []
    sat_con: dict[str, list[tuple[float, float]]] = {lab: [] for lab, *_ in arms}

    for pi, prompt in enumerate(prompts):
        ctx, ctx_null = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=prompt,
            negative_prompt=args.neg_prompt,
            text_encoder_path=args.text_encoder,
        )
        embed = ctx["embed"][0].to(device, dtype).expand(1, -1, -1)
        nembed = ctx_null["embed"][0].to(device, dtype).expand(1, -1, -1)

        for sj in range(max(1, args.num_seeds)):
            g = torch.Generator(device=device).manual_seed(args.seed + pi * 1000 + sj)
            x0 = torch.randn((1, C, 1, hl, wl), generator=g, device=device, dtype=dtype)
            er_seed = args.seed + pi * 1000 + sj

            imgs, labels, base_lat = [], [], None
            for lab, b, cfgpp, tag in arms:
                lat = _trajectory(
                    anima,
                    x0,
                    sig,
                    sigmas,
                    embed,
                    nembed,
                    pad,
                    args.guidance,
                    band=b,
                    cfgpp_lambda=cfgpp,
                    combiner=_make_combiner(tag, args),
                    fsg=fsg,
                    sampler=args.sampler,
                    er_seed=er_seed,
                    device=device,
                    dtype=dtype,
                )
                if lab == "baseline":
                    base_lat, drift = lat, 0.0
                else:
                    drift = _norm(lat - base_lat) / max(_norm(base_lat), 1e-8)
                im = decode_to_pil(vae, lat, device)
                sat, con = _sat_contrast(im)
                sat_con[lab].append((sat, con))
                imgs.append(im)
                labels.append(
                    f"{lab}  sat{sat:.3f} c{con:.3f}"
                    + (f"  Δ={drift:.3f}" if lab != "baseline" else "")
                )
                _safe = lab.replace(" ", "_").replace(".", "p").replace("/", "-")
                im.save(run_dir / f"p{pi}_s{sj}_{_safe}.png")
                rows.append(
                    {
                        "prompt_idx": pi,
                        "seed": sj,
                        "arm": lab,
                        "drift": drift,
                        "saturation": sat,
                        "rms_contrast": con,
                    }
                )
            _grid(
                imgs, labels, f"p{pi} s{sj}: {prompt}",
                ncols=args.grid_cols, cell=args.cell_px,
            ).save(run_dir / f"grid_p{pi}_s{sj}.png")
            log.info(
                f"  [{pi + 1}/{len(prompts)} seed {sj}] "
                + "  ".join(labels[k] for k in range(1, len(labels)))
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    base_sat = (
        float(np.mean([s for s, _ in sat_con["baseline"]]))
        if sat_con["baseline"]
        else 0.0
    )
    base_con = (
        float(np.mean([c for _, c in sat_con["baseline"]]))
        if sat_con["baseline"]
        else 0.0
    )
    sat_contrast: dict[str, dict] = {}
    log.info("\narm                          sat     Δsat%    contrast  Δcon%")
    for lab, *_ in arms:
        vals = sat_con[lab]
        if not vals:
            continue
        msat = float(np.mean([s for s, _ in vals]))
        mcon = float(np.mean([c for _, c in vals]))
        dsat = 100.0 * (msat - base_sat) / base_sat if base_sat else 0.0
        dcon = 100.0 * (mcon - base_con) / base_con if base_con else 0.0
        sat_contrast[lab] = {
            "saturation": msat,
            "rms_contrast": mcon,
            "dsat_pct": dsat,
            "dcon_pct": dcon,
        }
        log.info(f"{lab:<26} {msat:.4f}  {dsat:+6.2f}   {mcon:.4f}  {dcon:+6.2f}")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "arms": [a[0] for a in arms],
            "guidance": args.guidance,
            "sampler": args.sampler,
            "sat_contrast": sat_contrast,
            "prompts": prompts,
            "rows": rows,
        },
        artifacts=sorted(p.name for p in run_dir.glob("grid_*.png")),
    )
    log.info(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
