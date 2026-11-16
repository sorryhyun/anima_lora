#!/usr/bin/env python3
"""How large is the L_CDM signal without CFG on the real score?

The paper's L_CDM (arXiv:2605.06376 Eq. 9) takes a cond-only teacher against the
cond-only fake; ours takes the CFG'd teacher, so the off-trajectory term carries
``(α − 1)·(v_c − v_u)`` beside the distribution-matching part. This probe loads a
run's resume bundle through the distiller's own ``build_run`` (student and fake as
trained together) and, on the draws the training loop makes, builds the CDM signal
both ways from one set of forwards. Forward only, nothing is stepped.

Per sample (prompt, rollout ε, grad step g) and per draw (t', τ̂, ε̂):

  cfg       τ̂·(v_cfg − v_fake) / den(v_cfg)        the signal as trained
  cond      τ̂·(v_c − v_fake) / den(v_c)            the paper's, our normaliser
  cond_off  τ̂·(v_c − v_fake) / mean|x̂0_real − x̂0|  the paper's, its own normaliser
            (clamped at 0.1, the official ``norm_clip_min``)
  dm / ca   the two parts of ``cfg`` over its own denominator:
            τ̂·(v_c − v_fake) and τ̂·(α − 1)·(v_c − v_u)

``den(v) = max(mean|τ̂·v|, norm_floor)`` under ``dm_x0_norm``, else 1. The same
five are taken for the on-trajectory DM term at the same (τ̂, ε̂), on ``x_pred``.

Read, fixed before the run: with ``rms(cond) / rms(cfg) < 0.1`` on the CDM rows a
cond-only L_CDM arm is a CDM-off arm and is not worth a training run; at 0.1 or
above it is a distinct arm.

    make daemon-run ARGS="--queue --stall-timeout 900 bench/turbo/probe_cdm_signal.py \
        --probe_label cdm_signal_<run> --config <run>.toml \
        --resume output/ckpt/<run>/<run>_resume.pt --no_log --validate_every_n_steps 0"

Everything after the ``--probe_*`` flags is the distiller's own CLI.
"""

from __future__ import annotations

import argparse
import logging
import tempfile

import numpy as np
import torch

from bench._common import make_run_dir, write_result
from library.anima.uncond import uncond_for_batch
from library.config.resolved import load_toml
from scripts.distill_turbo.config import build_argparser, resolve_config
from library.training.forward import renoise
from scripts.distill_turbo.primitives import cdm_extrapolate, sample_dynamic_sigmas
from scripts.distill_turbo.setup import build_run
from scripts.distill_turbo.steps import _uniform_tau

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OFFICIAL_NORM_CLIP = 0.1
TAU_BINS = ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        )
    )


def _signals(cfg, tau_e, x0, eps, v_c, v_u, v_fake) -> dict:
    """The five signals of one query and the raw velocity gaps behind them."""
    alpha = cfg.teacher_cfg
    v_c, v_u, v_fake = v_c.float(), v_u.float(), v_fake.float()
    v_cfg = v_u + alpha * (v_c - v_u)

    def den(v):
        if not cfg.dm_x0_norm:
            return 1.0
        return (
            (tau_e * v)
            .abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(cfg.norm_floor)
        )

    den_cfg = den(v_cfg)
    s_cfg = tau_e * (v_cfg - v_fake) / den_cfg
    s_cond = tau_e * (v_c - v_fake) / den(v_c)
    # x̂0_real − x̂0 = τ̂·(ε̂ − x̂0 − v_c) on z = (1 − τ̂)·x̂0 + τ̂·ε̂.
    den_off = (
        (tau_e * (eps.float() - x0.float() - v_c))
        .abs()
        .mean(dim=(1, 2, 3), keepdim=True)
        .clamp_min(OFFICIAL_NORM_CLIP)
    )
    s_cond_off = tau_e * (v_c - v_fake) / den_off
    s_dm = tau_e * (v_c - v_fake) / den_cfg
    s_ca = tau_e * (alpha - 1.0) * (v_c - v_u) / den_cfg
    return {
        "rms_cfg": _rms(s_cfg),
        "rms_cond": _rms(s_cond),
        "rms_cond_off": _rms(s_cond_off),
        "rms_dm_part": _rms(s_dm),
        "rms_ca_part": _rms(s_ca),
        "cos_cfg_cond": _cos(s_cfg, s_cond),
        "cos_dm_ca": _cos(s_dm, s_ca),
        "cos_vc_vfake": _cos(v_c, v_fake),
        "rms_vc": _rms(v_c),
        "rms_vc_minus_vfake": _rms(v_c - v_fake),
        "rms_vc_minus_vu": _rms(v_c - v_u),
    }


def _summary(rows: list[dict], term: str) -> dict:
    keys = [k for k in rows[0][term] if k.startswith(("rms_", "cos_"))]

    def agg(sel):
        out = {k: float(np.mean([r[term][k] for r in sel])) for k in keys}
        out["n"] = len(sel)
        # Ratio of the mean rms, and the mean of the per-draw ratio.
        out["cond_over_cfg"] = out["rms_cond"] / out["rms_cfg"]
        out["cond_over_cfg_per_draw"] = float(
            np.mean([r[term]["rms_cond"] / r[term]["rms_cfg"] for r in sel])
        )
        out["cond_off_over_cfg"] = out["rms_cond_off"] / out["rms_cfg"]
        return out

    res = {"all": agg(rows)}
    for lo, hi in TAU_BINS:
        sel = [r for r in rows if lo <= r["tau"] < hi]
        if sel:
            res[f"tau_{lo:.2f}_{hi:.2f}"] = agg(sel)
    return res


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--probe_samples", type=int, default=32)
    own.add_argument("--probe_draws", type=int, default=4)
    own.add_argument("--probe_label", type=str, default="cdm_signal")
    probe, rest = own.parse_known_args()

    args = build_argparser().parse_args(rest)
    if not args.resume:
        raise SystemExit("--resume <bundle> is required")
    # The run's own name stays untouched: build_run writes a config snapshot.
    args.output_dir = tempfile.mkdtemp(prefix="probe_cdm_signal_")
    args.output_name = "probe_cdm_signal"
    cfg = resolve_config(args, load_toml(args.config))
    ctx = build_run(args, cfg)
    turbo, device, dtype = ctx.turbo, ctx.device, ctx.dtype
    fwd = ctx.forward
    S, M = probe.probe_samples, probe.probe_draws
    logger.info(
        f"probe: bundle step {ctx.start_step}, S={S} M={M}, "
        f"dynamic_schedule={cfg.dynamic_schedule}, teacher_cfg={cfg.teacher_cfg}, "
        f"dm_x0_norm={cfg.dm_x0_norm}"
    )

    rows: list[dict] = []
    data_iter = iter(ctx.dataloader)
    for s in range(S):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(ctx.dataloader)
            batch = next(data_iter)
        latents = batch["latents"].to(device, dtype=dtype)
        emb = batch["crossattn_emb"].to(device, dtype=ctx.cond_dtype)
        B = latents.shape[0]
        torch.compiler.cudagraph_mark_step_begin()
        c_null = uncond_for_batch(ctx.uncond_base, emb)

        # The training loop's draw of the grid and of the grad step.
        if cfg.dynamic_schedule:
            sigmas = sample_dynamic_sigmas(2, cfg.student_steps)
        else:
            sigmas = ctx.student_sigmas
        n_steps = len(sigmas) - 1
        g = int(torch.randint(0 if cfg.dm_first_step else 1, n_steps, (1,)).item())

        def t_of(sig):
            return torch.full((B,), sig, device=device, dtype=dtype)

        x = torch.randn_like(latents)
        for i in range(g):
            v = fwd("student", x, t_of(sigmas[i]), emb, no_grad=True)
            x = x - (sigmas[i] - sigmas[i + 1]) * v
        s_g = sigmas[g]
        x_g = x
        v_g = fwd("student", x_g, t_of(s_g), emb, no_grad=True)
        x_pred = x_g.float() - s_g * v_g.float()

        for _ in range(M):
            t_off = torch.rand(B).to(device=device, dtype=dtype)
            x_off = cdm_extrapolate(x_g, v_g, s_g, t_off).to(dtype)
            v_off = fwd("student", x_off, t_off, emb, no_grad=True)
            x0_off = x_off.float() - t_off.view(B, 1, 1, 1).float() * v_off.float()

            tau = _uniform_tau(ctx, cfg, B)
            tau_e = tau.view(B, 1, 1, 1).float()
            eps = torch.randn_like(latents)
            row = {
                "sample": s,
                "g": g,
                "s_g": float(s_g),
                "t_off": float(t_off.float().mean()),
                "tau": float(tau.float().mean()),
                "tokens": int(latents.shape[-1] * latents.shape[-2] // 4),
            }
            # (term, clean estimate, σ the fake is told the sample came from)
            for term, x0, src in (("cdm", x0_off, t_off), ("dm", x_pred, s_g)):
                z = renoise(x0.to(dtype), tau, eps)
                v_c = fwd("teacher", z, tau, emb, no_grad=True)
                v_u = fwd("teacher", z, tau, c_null, no_grad=True)
                turbo.set_fake_source(src)
                v_fake = fwd("fake", z, tau, emb, no_grad=True)
                row[term] = _signals(cfg, tau_e, x0, eps, v_c, v_u, v_fake)
            rows.append(row)
        last = rows[-M:]
        logger.info(
            f"sample {s + 1}/{S}: g={g} σ_g={s_g:.3f}  cdm cond/cfg "
            f"{np.mean([r['cdm']['rms_cond'] / r['cdm']['rms_cfg'] for r in last]):.3f}  "
            f"dm cond/cfg "
            f"{np.mean([r['dm']['rms_cond'] / r['dm']['rms_cfg'] for r in last]):.3f}"
        )

    metrics = {
        "bundle_step": ctx.start_step,
        "samples": S,
        "draws": M,
        "teacher_cfg": cfg.teacher_cfg,
        "dm_x0_norm": cfg.dm_x0_norm,
        "dynamic_schedule": cfg.dynamic_schedule,
        "cdm": _summary(rows, "cdm"),
        "dm": _summary(rows, "dm"),
        "cond_cdm_over_cfg_dm": float(
            np.mean([r["cdm"]["rms_cond"] for r in rows])
            / np.mean([r["dm"]["rms_cfg"] for r in rows])
        ),
        "rows": rows,
    }
    for term in ("cdm", "dm"):
        logger.info(f"--- {term} ---")
        for name, a in metrics[term].items():
            logger.info(
                f"{name:15s} n={a['n']:3d}  rms cfg {a['rms_cfg']:.4f}  "
                f"cond {a['rms_cond']:.4f}  cond_off {a['rms_cond_off']:.4f}  "
                f"dm part {a['rms_dm_part']:.4f}  ca part {a['rms_ca_part']:.4f}  "
                f"cond/cfg {a['cond_over_cfg']:.3f}  cos(cfg,cond) "
                f"{a['cos_cfg_cond']:+.3f}  cos(dm,ca) {a['cos_dm_ca']:+.3f}  "
                f"cos(v_c,v_fake) {a['cos_vc_vfake']:.5f}"
            )
    logger.info(
        f"cond-only CDM rms over CFG'd DM rms: {metrics['cond_cdm_over_cfg_dm']:.3f}"
    )

    run_dir = make_run_dir("turbo", probe.probe_label)
    write_result(
        run_dir,
        script=__file__,
        args={**vars(probe), "config": args.config, "resume": args.resume},
        metrics=metrics,
        label=probe.probe_label,
    )
    logger.info(f"result: {run_dir / 'result.json'}")

    if cfg.int8 != "off":
        from library.anima.int8 import finish_training

        finish_training()


if __name__ == "__main__":
    main()
