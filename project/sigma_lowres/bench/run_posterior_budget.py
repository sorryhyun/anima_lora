#!/usr/bin/env python
"""sigma_lowres E12 — posterior-budget (quantization-regime) probes, native-only.

Tests the Lottery-Prior-inspired hypothesis (``paper/action.md`` §4.4/Q5):
the denoiser at σ acts as a lossy code, so a demotion perturbation above the
code's resolution "re-rolls" posterior detail — response amplitude saturates
at the posterior-uncertainty scale (route-uniform m(σ)) while the landing
direction is a fresh draw. Two forward-only probes on the NATIVE grid:

**Probe A — posterior-trace curve.** Hutchinson estimate of tr Cov(x|z,c)
via the Divergence-is-Uncertainty identity. With z = (1−σ)x + σε and
v̂ ≈ E[ε−x|z], Tweedie gives E[x|z] = z − σv̂ and (Gaussian channel,
a = 1−σ):  D_z E[x|z] = (a/σ²)·Cov(x|z), hence

    tr Cov(x|z,c) = σ²/(1−σ) · tr(I − σ·D_z v̂).

tr(I − σ·D_z v̂) is estimated with Rademacher probes u (uᵀu = d exactly, so
only the FD-JVP term carries variance):  d − σ·uᵀ(v̂(z+δu) − v̂(z))/δ.
The identity degenerates at σ = 1 (0·∞) — the default grid tops out just
below. **Prediction**: m(σ) ∝ √tr-Cov, route-independently; shape mismatch
kills the hypothesis.

**Probe B — ε-sweep saturation.** Response amplitude vs perturbation size at
matched norms: per image, unit directions û from the repromote arms
(demote→up-resize→re-encode at the native bucket — the E9 B-intervention),
the reenc control, and a random-Gaussian control; x_pert = x + ε·û with ε on
a shared relative grid plus each direction's natural amplitude ("nat" — the
true arm). Response is E11's read: rel-L2 between K-draw mean residuals
r̄_pert vs r̄_base (shared noise seeds), with the parity split-half floor.
**Prediction**: plateau at m(σ) with reenc's natural ε below the knee and all
routes on the plateau; plateau departure locates the account's validity
domain (candidate story for 512's rotation / κ_eff ≈ 1).

Paired FD is meaningless without determinism (chaos floor ≈ 0.41 swallows
small-ε signal), so deterministic mode is ON by default. No compile — FD
needs no autograd and avoids the jvp-vs-compiled-DiT gotcha entirely.

Usage::

    make daemon-run ARGS="project/sigma_lowres/bench/run_posterior_budget.py \
        --pilot --label e12-pilot"
    make daemon-run ARGS="project/sigma_lowres/bench/run_posterior_budget.py \
        --label e12"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from project.sigma_lowres.bench.run_prior_distance import rel_l2  # noqa: E402
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    select_probe_set,
)
from project.sigma_lowres.bench.tier_routing.run_grad_probe import (  # noqa: E402
    DIT,
    VAE,
    encode_probe_latents,
)
from library.io.cache import (  # noqa: E402
    load_cached_latents,
    load_cached_text_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dit", default=DIT)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--tier", type=int, default=1280, help="native tier of the set")
    p.add_argument(
        "--data_root",
        default=str(Path(__file__).resolve().parent / "probe1280_cache"),
        help="dataset root holding lora/ + resized/ (the E11 probe cache, so "
        "the m(σ) comparison is same-corpus)",
    )
    p.add_argument("--max_per_artist", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--skip_trace", action="store_true", help="skip Probe A (trace curve)"
    )
    p.add_argument("--skip_sweep", action="store_true", help="skip Probe B (ε-sweep)")
    p.add_argument(
        "--no_deterministic",
        action="store_true",
        help="disable train.py-mirrored deterministic mode (NOT recommended: "
        "paired FD requires it — the chaos floor swallows small-δ signal)",
    )
    # Probe A
    p.add_argument("--num_images", type=int, default=40)
    p.add_argument(
        "--sigmas",
        default="0.125,0.375,0.625,0.875,0.96875",
        help="Probe A σ grid; σ=1.0 is excluded by default (the trace "
        "identity degenerates there: prefactor σ²/(1−σ) → ∞ against "
        "tr(I−σD) → 0)",
    )
    p.add_argument("--draws", type=int, default=4, help="noise draws per (image, σ)")
    p.add_argument("--hutch_k", type=int, default=8, help="Rademacher probes per draw")
    p.add_argument(
        "--fd_delta",
        type=float,
        default=0.03125,
        help="per-element FD step (power of two: with unit-scale latents and "
        "±1 Rademacher u, z+δu stays near-exactly bf16-representable); probe "
        "0 of every draw is repeated at 2δ for the linearity check",
    )
    # Probe B
    p.add_argument("--sweep_images", type=int, default=12)
    p.add_argument("--sweep_sigmas", default="0.375,0.625,0.875")
    p.add_argument(
        "--sweep_edges",
        default="896,512",
        help="repromote direction arms (mid-window route + the rotation/"
        "κ_eff≈1 route where plateau departure is predicted)",
    )
    p.add_argument("--sweep_draws", type=int, default=2)
    p.add_argument(
        "--eps_ratios",
        default="0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.8",
        help="perturbation norms as fractions of ‖x‖ (each direction's "
        "natural amplitude is always added as an extra 'nat' point)",
    )
    p.add_argument("--pilot", action="store_true", help="~20 min shape check")
    args = p.parse_args()
    if args.pilot:
        args.num_images = 8
        args.draws = 2
        args.hutch_k = 4
        args.sweep_images = 4
    return args


def forward(anima, z: torch.Tensor, sig, crossattn, pad) -> torch.Tensor:
    """(1,16,H,W) fp32 → v̂ (16,H,W) fp32. The dim-2 unsqueeze/squeeze is the
    5D DiT boundary dance."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v = anima(z.unsqueeze(2).to(torch.bfloat16), sig, crossattn, padding_mask=pad)
    return v.squeeze(2).float()[0]


def mean_residual_halves(
    anima, z_of, x_pert: torch.Tensor, sigma: float, crossattn, pad, draws, seed_base
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parity half-means of v̂(z_σ) − (ε − x_pert) over shared-seed draws."""
    h, w = x_pert.shape[-2:]
    halves = [
        torch.zeros(16, h, w, device=x_pert.device),
        torch.zeros(16, h, w, device=x_pert.device),
    ]
    for d in range(draws):
        eps, z = z_of(x_pert, sigma, d, seed_base)
        v = forward(
            anima, z, torch.full((1,), sigma, device=x_pert.device), crossattn, pad
        )
        halves[d % 2] += v - (eps - x_pert)[0]
    return halves[0] / (draws - draws // 2), halves[1] / max(draws // 2, 1)


def main() -> None:
    args = parse_args()
    if not args.no_deterministic:  # must precede any CUDA/cublas init
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        from networks import attention_dispatch

        attention_dispatch.set_deterministic(True)
        log.info("deterministic mode on (train.py-mirrored knobs)")
    start_heartbeat()
    device = torch.device("cuda")
    sigmas_a = [float(s) for s in args.sigmas.split(",") if s]
    if any(s >= 1.0 for s in sigmas_a):
        raise SystemExit("Probe A σ grid must stay below 1.0 (identity degenerates)")
    sigmas_b = [float(s) for s in args.sweep_sigmas.split(",") if s]
    edges_b = [int(e) for e in args.sweep_edges.split(",") if e]
    ratios = [float(v) for v in args.eps_ratios.split(",") if v]
    delta = args.fd_delta

    records = score_corpus(k=args.quant_k, data_root=Path(args.data_root).resolve())
    probe = select_probe_set(
        records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
    )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records under {args.data_root}")
    sweep_probe = probe[: args.sweep_images] if not args.skip_sweep else []
    log.info(
        f"probe set: {len(probe)} images ({len({r.artist for r in probe})} artists); "
        f"A: σ={sigmas_a} D={args.draws} k={args.hutch_k} δ={delta}; "
        f"B: {len(sweep_probe)} images, σ={sigmas_b}, dirs=repromote{edges_b}"
        f"+reenc+rand, ε/‖x‖={ratios}+nat"
    )

    # VAE encodes the sweep arms and frees BEFORE the DiT loads.
    extra_latents = (
        encode_probe_latents(
            sweep_probe, edges_b, args.vae, device, True, repromote=True
        )
        if sweep_probe
        else {}
    )

    from library.runtime.harness import build_anima

    args.compile = False
    args.gradient_checkpointing = False
    bundle = build_anima(args, dit_path=args.dit, adapter=None, train_mode=False)
    anima = bundle.anima

    def z_of(x: torch.Tensor, sigma: float, d: int, seed_base: int):
        gen = torch.Generator(device=device).manual_seed(seed_base + d)
        eps = torch.randn(x.shape, generator=gen, device=device)
        return eps, (1.0 - sigma) * x + sigma * eps

    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )
    rows_path = run_dir / "per_image.jsonl"
    rows: list[dict] = []
    t0 = time.time()

    for i, r in enumerate(probe):
        crossattn, _ = load_cached_text_features(r.te_path, variant=0)
        if crossattn is None:
            log.info(f"  [{i}] {r.artist}/{r.stem}: no crossattn_emb — skipped")
            continue
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        x = load_cached_latents(r.npz_path)[0].unsqueeze(0).to(device)  # (1,16,H,W)
        h, w = x.shape[-2:]
        pad = torch.zeros(1, 1, h, w, dtype=torch.bfloat16, device=device)
        d_dim = 16 * h * w
        row: dict = {"artist": r.artist, "stem": r.stem, "tokens": (h // 2) * (w // 2)}

        if not args.skip_trace:
            trace: dict = {"sigmas": sigmas_a, "d_dim": d_dim}
            tr_cov, tr_dv, lin = [], [], []
            for si, sig in enumerate(sigmas_a):
                seed_base = args.seed * 1_000_000 + i * 10_000 + si * 100
                samples, lins = [], []
                for d in range(args.draws):
                    _, z = z_of(x, sig, d, seed_base)
                    # snap the base point to the bf16 grid so base/perturbed
                    # forwards differ by (nearly) the intended step alone
                    z = z.to(torch.bfloat16).float()
                    sig_t = torch.full((1,), sig, device=device)
                    v0 = forward(anima, z, sig_t, crossattn, pad).double()
                    ugen = torch.Generator(device=device).manual_seed(
                        seed_base + 50 + d
                    )
                    for j in range(args.hutch_k):
                        u = (
                            torch.randint(
                                0, 2, x.shape, generator=ugen, device=device
                            ).float()
                            * 2.0
                            - 1.0
                        )
                        vj = forward(anima, z + delta * u, sig_t, crossattn, pad)
                        t_j = (
                            float(u.double().flatten() @ (vj.double() - v0).flatten())
                            / delta
                        )
                        samples.append(d_dim - sig * t_j)
                        if j == 0:  # linearity check: same u at 2δ
                            v2 = forward(
                                anima, z + 2 * delta * u, sig_t, crossattn, pad
                            )
                            t2 = float(
                                u.double().flatten() @ (v2.double() - v0).flatten()
                            ) / (2 * delta)
                            lins.append(t2 / t_j if t_j != 0 else float("nan"))
                s = np.array(samples)
                pref = sig**2 / (1.0 - sig)
                tr_cov.append(
                    {
                        "mean": float(pref * s.mean()),
                        "sem": float(pref * s.std(ddof=1) / np.sqrt(len(s))),
                    }
                )
                tr_dv.append(float(np.mean([(d_dim - v) / sig for v in samples])))
                lin.append(float(np.nanmean(lins)))
            trace["tr_cov"] = tr_cov
            trace["rms_post_std"] = [
                float(np.sqrt(max(c["mean"], 0.0) / d_dim)) for c in tr_cov
            ]
            trace["tr_dv_per_dim"] = [round(v / d_dim, 5) for v in tr_dv]
            trace["fd_linearity"] = [round(v, 4) for v in lin]
            row["trace"] = trace
            log.info(
                f"  [{i + 1}/{len(probe)}] {r.artist}/{r.stem} rms_post_std="
                + str([round(v, 4) for v in trace["rms_post_std"]])
                + f" lin={trace['fd_linearity']}"
            )

        if i < len(sweep_probe):
            x0 = x[0]
            x_norm = float(x0.norm())
            dirs: dict[str, torch.Tensor] = {
                f"route{e}": extra_latents[(r.stem, f"repromote{e}")].to(device) - x0
                for e in edges_b
            }
            dirs["reenc"] = extra_latents[(r.stem, "reenc")].to(device) - x0
            rgen = torch.Generator(device=device).manual_seed(args.seed + 777 + i)
            dirs["rand"] = torch.randn(x0.shape, generator=rgen, device=device)
            sweep: dict = {
                "sigmas": sigmas_b,
                "eps_ratios": ratios,
                "nat_ratio": {
                    k: (round(float(v.norm()) / x_norm, 5) if k != "rand" else None)
                    for k, v in dirs.items()
                },
            }
            for si, sig in enumerate(sigmas_b):
                seed_base = args.seed * 1_000_000 + i * 10_000 + (si + 90) * 100
                m1b, m2b = mean_residual_halves(
                    anima, z_of, x, sig, crossattn, pad, args.sweep_draws, seed_base
                )
                rb, fb = 0.5 * (m1b + m2b), rel_l2(m1b, m2b)
                sweep[f"res_norm_base_s{si}"] = round(float(rb.norm()), 3)
                for name, u_raw in dirs.items():
                    nrm = float(u_raw.norm())
                    if nrm < 1e-6 * x_norm:
                        # degenerate direction — e.g. the deterministic VAE
                        # reproduces the cached latent bit-exactly on reenc
                        continue
                    uhat = (u_raw / nrm).unsqueeze(0)
                    points = list(ratios)
                    if sweep["nat_ratio"][name]:
                        points.append(sweep["nat_ratio"][name])
                    resp, dabs, floors = [], [], []
                    for ratio in points:
                        xp = x + ratio * x_norm * uhat
                        m1p, m2p = mean_residual_halves(
                            anima,
                            z_of,
                            xp,
                            sig,
                            crossattn,
                            pad,
                            args.sweep_draws,
                            seed_base,
                        )
                        rp, fp = 0.5 * (m1p + m2p), rel_l2(m1p, m2p)
                        # rel_l2 ceilings at ~1.41 for orthogonal equal-norm
                        # arms — keep the absolute difference norm alongside
                        # so a true amplitude plateau is separable from the
                        # metric ceiling at large ε
                        resp.append(round(rel_l2(rp, rb), 5))
                        dabs.append(round(float((rp - rb).norm()), 3))
                        floors.append(round(0.5 * (fb**2 + fp**2) ** 0.5, 5))
                    sweep[f"resp_{name}_s{si}"] = resp
                    sweep[f"dabs_{name}_s{si}"] = dabs
                    sweep[f"floor_{name}_s{si}"] = floors
            row["sweep"] = sweep
            si_last = len(sigmas_b) - 1
            log.info(
                f"    sweep {r.stem}: "
                + " ".join(
                    f"{n}@nat(s-last)={sweep[f'resp_{n}_s{si_last}'][-1]}"
                    for n in dirs
                    if sweep["nat_ratio"][n] and f"resp_{n}_s{si_last}" in sweep
                )
            )

        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    if not rows:
        raise SystemExit("no rows produced")

    headline: dict = {
        "n_images": len(rows),
        "wall_time_s": round(time.time() - t0, 1),
        "dit": args.dit,
        "fd_delta": delta,
        "deterministic": not args.no_deterministic,
    }
    tr_rows = [r["trace"] for r in rows if "trace" in r]
    if tr_rows:
        v = np.array([t["rms_post_std"] for t in tr_rows])  # (n, n_sig)
        headline["probeA"] = {
            "sigmas": sigmas_a,
            "rms_post_std_mean": [round(float(x), 5) for x in v.mean(0)],
            "rms_post_std_sem": [
                round(float(x), 5) for x in v.std(0, ddof=1) / np.sqrt(v.shape[0])
            ],
            "tr_dv_per_dim_mean": [
                round(float(x), 5)
                for x in np.array([t["tr_dv_per_dim"] for t in tr_rows]).mean(0)
            ],
            "fd_linearity_mean": [
                round(float(x), 4)
                for x in np.nanmean([t["fd_linearity"] for t in tr_rows], axis=0)
            ],
        }
    sw_rows = [r["sweep"] for r in rows if "sweep" in r]
    if sw_rows:
        pb: dict = {"sigmas": sigmas_b, "eps_ratios": ratios, "n_images": len(sw_rows)}
        for name in list(sw_rows[0]["nat_ratio"]):
            have = [s for s in sw_rows if f"resp_{name}_s0" in s]
            nats = [s["nat_ratio"][name] for s in have if s["nat_ratio"][name]]
            pb[f"nat_ratio_{name}"] = round(float(np.mean(nats)), 5) if nats else None
            pb[f"n_images_{name}"] = len(have)
            for si in range(len(sigmas_b)):
                if not have:
                    continue
                m = np.array([s[f"resp_{name}_s{si}"][: len(ratios)] for s in have])
                pb[f"resp_{name}_s{si}"] = [round(float(x), 5) for x in m.mean(0)]
                if nats:
                    pb[f"resp_{name}_s{si}_nat"] = round(
                        float(
                            np.mean(
                                [
                                    s[f"resp_{name}_s{si}"][-1]
                                    for s in have
                                    if s["nat_ratio"][name]
                                ]
                            )
                        ),
                        5,
                    )
        headline["probeB"] = pb
    log.info(json.dumps(headline, indent=2))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=[rows_path],
        extra={"probe_set": [f"{r['artist']}/{r['stem']}" for r in rows]},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
