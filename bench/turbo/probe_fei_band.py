#!/usr/bin/env python3
"""Turbo FEI band-gap probe (Phase 0 of docs/anima/proposal/turbo_fei_band_on_trajectory.md).

THE QUESTION. Does the trained DP-DMD student over-blur *on its own rollout*
relative to the teacher — i.e. is there a HF deficit at the distribution the
loss now visits (the student's trajectory + the renoise-of-student DMD point)?
The 2026-05 item-2 lever falsified because it measured the band gap at renoised
*real* data, where the only gap is the trivial "one-shot x0 is noisier than the
teacher's" (project/anima_turbo/findings/turbo_fei_band_deficit_falsified.md). This probe
re-measures at the right distribution, on an ALREADY-TRAINED checkpoint — the
fast way to read the gate without burning a training run (a fresh short run is
useless: the zero-init student ≈ teacher for the first thousands of steps, so
its gap is the same trivial one the falsification warned about).

CONSTRUCTION (no training; same Euler grids as scripts/distill_turbo/distill.py).
For each cached real latent, with the checkpoint loaded as a plain LoRA:

* teacher = LoRA multiplier 0 (frozen base DiT), CFG-guided velocity;
* student = LoRA multiplier 1, cond-only velocity (CFG is baked).

**Site A** — paired trajectory FEI gap at MATCHED σ. Roll the teacher K-step CFG
anchor and the student's step-0 from the SAME ε; snapshot the teacher state at
the anchor step whose σ matches the student's post-step-0 state z1, then compare
``e_low(z1) − e_low(z_teacher_snap)``. ``gap_low > 0`` ⇒ ``student_over_low``
(over-blur symptom on the trajectory). At student_steps=4 / anchor 6/12 the match
is exact (both σ=0.9).

**Site B** — x0-scale band deficit at the DMD point (the lever's site), swept
over a fixed τ grid (the falsified headline was a *sign structure across the
schedule*, so a τ-pooled mean would hide a flip). e_S = FEI(x_pred) on the true
N-step rollout endpoint; e_T = FEI(x_renoised − τ·v_real_cfg) the teacher's
1-step x0 estimate. ``dh = relu(e_high_T − e_high_S)`` is the HF deficit →
over-blur / ``w_high`` arm; ``dl = relu(e_low_T − e_low_S)`` is the LF deficit →
the arm that killed item 2. Because ``e_low+e_high≡1``, at most one fires.

PRE-REGISTERED GATE:

* ``dh_pos`` fires (mean ≥ ``--gate_dh`` in any mid/low-τ bucket) OR Site A shows
  a persistent ``student_over_low`` gap → **GO Phase 1** (band-split grad_dm).
* ``dl_pos`` dominates, or both ≈ 0 → **CLOSE the line** (the anchor already
  fixed the over-blur, or the gap never lived at the new sites either).

Run from anima_lora/::

    uv run python bench/turbo/probe_fei_band.py \
        --adapter output/ckpt/anima_turbo_P_5k.safetensors \
        --student_steps 4 --k_anchor 6 --teacher_anchor_steps 12
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


import numpy as np
import torch

from bench._anima import (
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, write_result
from bench.soft_tokens_contrastive.reward_premise_probe import (
    EmbCache,
    _fmt,
    discover_pairs,
)
from library.anima.uncond import (
    default_uncond_path,
    load_uncond_crossattn,
    uncond_for_batch,
)
from library.inference.sampling import get_timesteps_sigmas
from library.io.cache import load_cached_latents
from library.runtime.fei import compute_fei_2band, fei_sigma_low
from library.training.forward.renoise import renoise

REPO_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("bench.turbo.fei_band")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=False, text_encoder=False)
    ap.add_argument(
        "--adapter",
        default="output/ckpt/anima_turbo_P_5k.safetensors",
        help="Trained turbo student checkpoint (plain LoRA).",
    )
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--num_samples", type=int, default=120, help="images sampled")
    ap.add_argument(
        "--num_seeds",
        type=int,
        default=2,
        help="ε draws per image; teacher/student share the draw (paired).",
    )
    # Rollout shape — MUST match the trained checkpoint's snapshot for Site A's
    # σ-match to be exact (student_steps=4 / 6 / 12 → z1 σ=0.9 == teacher idx 3).
    ap.add_argument("--student_steps", type=int, default=4)
    ap.add_argument("--k_anchor", type=int, default=6)
    ap.add_argument("--teacher_anchor_steps", type=int, default=12)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--teacher_cfg", type=float, default=4.0)
    ap.add_argument(
        "--sigma_low_div",
        type=float,
        default=16.0,
        help="σ_low = min(H_lat,W_lat)/div; 16 matches the original Phase-0 sweep.",
    )
    ap.add_argument(
        "--tau_grid",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="fixed τ_dm grid for Site B (a clean per-bucket schedule profile).",
    )
    # Pre-registered gate.
    ap.add_argument(
        "--gate_dh",
        type=float,
        default=0.01,
        help="Site-B HF deficit mean counting as 'fires' (over-blur → w_high).",
    )
    ap.add_argument(
        "--gate_tau_max",
        type=float,
        default=0.7,
        help="apply the Site-B gate in buckets with τ ≤ this (mid/low-τ).",
    )
    ap.add_argument(
        "--gate_frac_over",
        type=float,
        default=0.60,
        help="Site-A student_over_low frac (with mean gap>0) counting as persistent.",
    )
    add_common_args(ap)
    args = ap.parse_args()

    tau_grid = sorted(float(t) for t in args.tau_grid)
    # σ grids (token-count-invariant), identical to the distill loop.
    student_sigmas = get_timesteps_sigmas(args.student_steps, args.flow_shift, "cpu")[
        1
    ].tolist()
    teacher_sigmas = get_timesteps_sigmas(
        args.teacher_anchor_steps, args.flow_shift, "cpu"
    )[1].tolist()
    # Site A pairing: teacher anchor state nearest to the student z1 σ.
    z1_sigma = float(student_sigmas[1])
    snap_j = min(
        range(1, args.k_anchor + 1),
        key=lambda j: abs(float(teacher_sigmas[j]) - z1_sigma),
    )
    snap_idx = snap_j - 1  # snapshot z right after this anchor-loop iteration
    snap_sigma = float(teacher_sigmas[snap_j])
    dsig = abs(snap_sigma - z1_sigma)
    log.info(
        f"student σ={['%.3f' % s for s in student_sigmas]} | "
        f"teacher σ={['%.3f' % s for s in teacher_sigmas]}"
    )
    log.info(
        f"Site A pairing: student z1 σ={z1_sigma:.4f} ↔ teacher σ={snap_sigma:.4f} "
        f"(anchor step {snap_idx}); |Δσ|={dsig:.4f}"
        + ("" if dsig < 0.02 else "  ⚠ loose match — gap is noisier")
    )

    pairs = discover_pairs(args.data_dir)
    pool_list = sorted(pairs)
    if not pool_list:
        raise SystemExit(f"no cached pairs under {args.data_dir}")
    rng = np.random.default_rng(args.seed)
    take = min(args.num_samples, len(pool_list))
    stems = [pool_list[int(i)] for i in rng.choice(len(pool_list), take, replace=False)]
    log.info(f"{take}/{len(pairs)} cached pairs sampled")

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, network = bundle.anima, bundle.network
    device, dtype = bundle.device, bundle.dtype
    embs = EmbCache(pairs)
    uncond_base = load_uncond_crossattn(
        str(default_uncond_path()), device=device, dtype=dtype
    )
    n_seeds = max(1, args.num_seeds)
    autocast = (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda"
        else torch.no_grad()
    )

    @torch.no_grad()
    def _vel(mult: float, x4d, t_b, c):
        """Squeezed velocity at LoRA multiplier ``mult`` (0=teacher, 1=student)."""
        network.set_multiplier(mult)
        pad = torch.zeros(
            x4d.shape[0], 1, x4d.shape[-2], x4d.shape[-1], dtype=dtype, device=device
        )
        with autocast:
            v = anima.forward_mini_train_dit(
                x4d.unsqueeze(2), t_b, c, padding_mask=pad, skip_pooled_text_proj=True
            )
        return v.squeeze(2)

    @torch.no_grad()
    def _teacher_cfg_vel(x4d, t_b, c_cond, c_null):
        v_c = _vel(0.0, x4d, t_b, c_cond)
        if args.teacher_cfg == 1.0:
            return v_c.float()
        v_u = _vel(0.0, x4d, t_b, c_null)
        return v_u.float() + args.teacher_cfg * (v_c.float() - v_u.float())

    # Accumulators. Site A: paired e_low (student z1, teacher snap). Site B:
    # per-τ-bucket dh/dl (HF/LF deficit) on the true rollout endpoint.
    a_es: list[float] = []  # e_low(z1)        (student trajectory)
    a_et: list[float] = []  # e_low(z_snap)    (teacher trajectory)
    b_dh: dict[float, list[float]] = {t: [] for t in tau_grid}
    b_dl: dict[float, list[float]] = {t: [] for t in tau_grid}
    n_scored = 0

    for ai, stem in enumerate(stems):
        npz_path, _te = pairs[stem]
        emb = embs.get(stem)
        if emb is None:
            continue
        lat, _res, _oh, _ow = load_cached_latents(npz_path)
        x0 = lat.to(device, dtype).unsqueeze(0)  # (1, C, H, W)
        H, W = x0.shape[-2], x0.shape[-1]
        sig_low = fei_sigma_low(H, W, args.sigma_low_div)
        c_cond = emb.to(device, dtype).unsqueeze(0)
        c_null = uncond_for_batch(uncond_base, c_cond)

        for sj in range(n_seeds):
            g = torch.Generator(device=device).manual_seed(args.seed + ai * 1000 + sj)
            eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)

            # --- teacher K-step CFG anchor → snapshot at the matched σ ---
            z = eps
            z_snap = None
            for i in range(args.k_anchor):
                s_i, s_next = teacher_sigmas[i], teacher_sigmas[i + 1]
                t_b = torch.full((1,), s_i, device=device, dtype=dtype)
                v = _teacher_cfg_vel(z, t_b, c_cond, c_null)
                z = (z.float() - (s_i - s_next) * v).to(dtype)
                if i == snap_idx:
                    z_snap = z

            # --- student rollout (cond-only) → z1 + true endpoint x_pred ---
            x = eps
            z1 = None
            for i in range(args.student_steps):
                s_i, s_next = student_sigmas[i], student_sigmas[i + 1]
                t_b = torch.full((1,), s_i, device=device, dtype=dtype)
                v = _vel(1.0, x, t_b, c_cond)
                x = x - (s_i - s_next) * v
                if i == 0:
                    z1 = x
            x_pred = x

            # Site A: paired LF energy at the matched σ.
            a_es.append(float(compute_fei_2band(z1.float(), sig_low)[0, 0]))
            a_et.append(float(compute_fei_2band(z_snap.float(), sig_low)[0, 0]))

            # Site B: band deficit at the DMD point over the τ grid.
            e_s = compute_fei_2band(x_pred.float(), sig_low)[0]  # (e_low, e_high)
            for tau in tau_grid:
                t_b = torch.full((1,), float(tau), device=device, dtype=dtype)
                eps_dm = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                x_ren = renoise(x_pred.detach(), t_b, eps_dm)
                v_real = _teacher_cfg_vel(x_ren, t_b, c_cond, c_null)
                x0_t = x_ren.float() - float(tau) * v_real
                e_t = compute_fei_2band(x0_t, sig_low)[0]
                b_dh[tau].append(float(max(0.0, float(e_t[1] - e_s[1]))))
                b_dl[tau].append(float(max(0.0, float(e_t[0] - e_s[0]))))

        n_scored += 1
        if (ai + 1) % 20 == 0 or ai + 1 == len(stems):
            log.info(f"  [{ai + 1}/{len(stems)}] scored")
        if device.type == "cuda" and (ai + 1) % 50 == 0:
            torch.cuda.empty_cache()

    if n_scored == 0:
        raise SystemExit("no images scored — check caches")

    # ── aggregate ────────────────────────────────────────────────────────────
    es = np.asarray(a_es)
    et = np.asarray(a_et)
    gap = es - et  # >0 ⇒ student_over_low
    site_a = {
        "e_low_student_mean": float(es.mean()),
        "e_low_teacher_mean": float(et.mean()),
        "gap_low_mean": float(gap.mean()),
        "gap_low_std": float(gap.std()),
        "frac_over_low": float((gap > 0).mean()),
        "matched_sigma": z1_sigma,
        "delta_sigma": dsig,
        "n": int(len(gap)),
    }

    site_b: dict[float, dict] = {}
    for tau in tau_grid:
        dh = np.asarray(b_dh[tau])
        dl = np.asarray(b_dl[tau])
        site_b[tau] = {
            "dh_pos_mean": float(dh.mean()),
            "dl_pos_mean": float(dl.mean()),
            "frac_dh": float((dh > 0).mean()),
            "frac_dl": float((dl > 0).mean()),
            "n": int(len(dh)),
        }

    # ── pre-registered gate ──────────────────────────────────────────────────
    reasons: list[str] = []
    for tau in tau_grid:
        if tau <= args.gate_tau_max and site_b[tau]["dh_pos_mean"] >= args.gate_dh:
            reasons.append(
                f"Site B dh_pos={site_b[tau]['dh_pos_mean']:.4f} ≥ {args.gate_dh} "
                f"at τ={tau} (frac {site_b[tau]['frac_dh']:.2f}) — HF deficit / over-blur"
            )
    if site_a["gap_low_mean"] > 0 and site_a["frac_over_low"] >= args.gate_frac_over:
        reasons.append(
            f"Site A student_over_low: gap_low={site_a['gap_low_mean']:+.4f} "
            f"(frac {site_a['frac_over_low']:.2f}) at σ={z1_sigma:.2f}"
        )
    go = bool(reasons)
    # Is the LF arm the dominant signal instead (the falsified inverse)? Judge in
    # the lever's mid/low-τ region only — a dh spike at the noisy renoise end
    # (τ→1) is the trivial "teacher x0 is noisier" artifact, not the lever.
    lever_taus = [t for t in tau_grid if t <= args.gate_tau_max] or tau_grid
    dl_dominates = all(
        site_b[t]["dl_pos_mean"] >= site_b[t]["dh_pos_mean"] for t in lever_taus
    )
    verdict = (
        "GO Phase 1 (band-split grad_dm — the over-blur arm fires on-trajectory)"
        if go
        else (
            "CLOSE — LF deficit dominates again (falsified inverse persists)"
            if dl_dominates
            else "CLOSE — both arms ≈ 0 (anchor fixed the over-blur, or no gap here)"
        )
    )

    # ── render ───────────────────────────────────────────────────────────────
    L = ["# Turbo FEI band-gap probe (Phase 0)\n"]
    L.append(
        f"- adapter: `{args.adapter}`\n"
        f"- images: **{n_scored}** · {n_seeds} ε draws · "
        f"rollout N={args.student_steps}, anchor {args.k_anchor}/"
        f"{args.teacher_anchor_steps} @ flow_shift={args.flow_shift}, "
        f"teacher_cfg={args.teacher_cfg}\n"
        f"- FEI: 2-band, σ_low=min(H,W)/{args.sigma_low_div:g} "
        f"(sign: gap_low>0 ⇒ student_over_low; dh>0 ⇒ student HF-deficient/over-blur)\n"
        f"- **verdict: {verdict}**\n"
    )
    for r in reasons:
        L.append(f"  - GATE: {r}")

    L.append("\n## Site A — paired trajectory LF gap at matched σ\n")
    L.append(
        f"σ={z1_sigma:.3f} (|Δσ|={dsig:.4f}) · "
        f"e_low student {site_a['e_low_student_mean']:.5f} vs teacher "
        f"{site_a['e_low_teacher_mean']:.5f} · "
        f"**gap_low {site_a['gap_low_mean']:+.5f}** ± {site_a['gap_low_std']:.5f} · "
        f"frac_over_low {site_a['frac_over_low']:.2f} (n={site_a['n']})\n"
    )

    L.append("\n## Site B — x0-scale band deficit at the DMD point, by τ\n")
    L.append(
        "| τ | dh_pos (HF deficit / w_high) | dl_pos (LF deficit / w_low) | frac dh | frac dl | n |"
    )
    L.append("|---|---|---|---|---|---|")
    for tau in tau_grid:
        m = site_b[tau]
        L.append(
            f"| {tau:.2f} | {_fmt(m['dh_pos_mean'], '.5f')} | "
            f"{_fmt(m['dl_pos_mean'], '.5f')} | {m['frac_dh']:.2f} | "
            f"{m['frac_dl']:.2f} | {m['n']} |"
        )

    L.append("\n## Reading it\n")
    L.append(
        "- **GO** ⇒ wire Phase 1 (`[repa]`-style band-split of `grad_dm` between "
        "`dm_x0_norm` and the surrogate assembly; β=0 default). The arm to drive "
        "is `w_high` where `dh_pos` fired.\n"
        "- **CLOSE (dl dominates)** ⇒ the falsified inverse persists at the new "
        "sites too; retire the line for good (write the closing finding).\n"
        "- **CLOSE (both ≈ 0)** ⇒ the diversity anchor already fixed the over-blur; "
        "same closing action.\n"
        "- Cross-check against sample grids at `--infer_steps "
        f"{args.student_steps} --cfg 1.0` — over-blur is visible; a scalar verdict "
        "that disagrees with the grids loses.\n"
        "- Re-derive the sign against this script before trusting plots (the 2026-05 "
        "wiring inverted it via a σ mismatch — here Site A is σ-matched by "
        "construction; |Δσ| is reported above).\n"
    )

    run_dir = make_run_dir("turbo_fei_band", label=args.label or "phase0")
    (run_dir / "fei_band.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    csv = run_dir / "site_b_by_tau.csv"
    with csv.open("w") as f:
        f.write("tau,dh_pos_mean,dl_pos_mean,frac_dh,frac_dl,n\n")
        for tau in tau_grid:
            m = site_b[tau]
            f.write(
                f"{tau},{m['dh_pos_mean']},{m['dl_pos_mean']},"
                f"{m['frac_dh']},{m['frac_dl']},{m['n']}\n"
            )

    metrics = {
        "adapter": args.adapter,
        "n_images": n_scored,
        "num_seeds": n_seeds,
        "rollout": {
            "student_steps": args.student_steps,
            "k_anchor": args.k_anchor,
            "teacher_anchor_steps": args.teacher_anchor_steps,
            "flow_shift": args.flow_shift,
            "teacher_cfg": args.teacher_cfg,
        },
        "sigma_low_div": args.sigma_low_div,
        "tau_grid": tau_grid,
        "site_a": site_a,
        "site_b": {str(t): site_b[t] for t in tau_grid},
        "gate": {
            "gate_dh": args.gate_dh,
            "gate_tau_max": args.gate_tau_max,
            "gate_frac_over": args.gate_frac_over,
            "reasons": reasons,
        },
        "go": go,
        "dl_dominates": dl_dominates,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["fei_band.md", "site_b_by_tau.csv"],
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(f"  Turbo FEI band-gap probe → {run_dir}")
    log.info(
        f"  Site A  σ={z1_sigma:.2f}  gap_low {site_a['gap_low_mean']:+.5f}  "
        f"frac_over {site_a['frac_over_low']:.2f}"
    )
    for tau in tau_grid:
        m = site_b[tau]
        log.info(
            f"  Site B  τ={tau:.2f}  dh {m['dh_pos_mean']:.5f}  dl {m['dl_pos_mean']:.5f}"
        )
    log.info(f"  VERDICT: {verdict}")
    for r in reasons:
        log.info(f"    - {r}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
