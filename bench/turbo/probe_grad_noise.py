#!/usr/bin/env python3
"""Where does the student gradient's noise come from in a turbo run?

Loads a run's resume bundle through the distiller's own ``build_run`` (student,
fake, disc and the student optimizer's second moments as trained), then takes
gradients without stepping anything. For S samples (prompt, rollout ε) and M
draws of the DM query (σ_a, ε_dm) per sample it records the student gradient of
each loss term:

  anchor  div_weight · ‖v₀ − v_target‖²          one per sample
  dm      (grad_signal · x_θ).mean()             per draw; also its step-0 part
          (the share that reaches the weights through x₁, the rollout path)
  gan     gan_weight · generator loss            per draw (same σ_a, ε_dm)

and splits each term's energy into the mean over everything (the part that
accumulates), the variance between samples, and the variance between draws of
one sample. Vectors are kept as CountSketches (inner products unbiased, relative
error ~1/√K); numbers are reported for the raw gradient and for the gradient in
Adam's metric (divided by √v̂ from the bundle), which is what moves the weights.

    make daemon-run ARGS="--stall-timeout 900 bench/turbo/probe_grad_noise.py \
        --config <run>.toml --resume output/ckpt/<run>/<run>_resume.pt \
        --no_log --validate_every_n_steps 0 --probe_samples 16 --probe_draws 3"

Everything after the ``--probe_*`` flags is the distiller's own CLI.
"""

from __future__ import annotations

import argparse
import gc
import logging
import tempfile

import numpy as np
import torch

from bench._common import make_run_dir, write_result
from library.anima.uncond import uncond_for_batch
from library.config.resolved import load_toml
from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.setup import build_run
from scripts.distill_turbo.steps import (
    dmd_surrogate,
    gan_generator_term,
    mfd_surrogate,
    student_rollout,
    teacher_anchor,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SKETCH_DIM = 1 << 20


class Sketcher:
    """CountSketch of a per-parameter gradient list into ``SKETCH_DIM`` buckets."""

    def __init__(self, params, device):
        g = torch.Generator(device="cpu").manual_seed(1234)
        n_max = max(p.numel() for p in params)
        self.bucket = torch.randint(0, SKETCH_DIM, (n_max,), generator=g).to(device)
        self.sign = (
            torch.randint(0, 2, (n_max,), generator=g).float().mul_(2).sub_(1)
        ).to(device)
        self.offset = torch.randint(0, SKETCH_DIM, (len(params),), generator=g).tolist()
        self.psign = (
            (torch.randint(0, 2, (len(params),), generator=g) * 2 - 1).float().tolist()
        )
        self.device = device

    def __call__(self, grads, precond=None) -> tuple[torch.Tensor, float]:
        """Sketch (on CPU) and exact squared norm; ``precond`` divides per element."""
        out = torch.zeros(SKETCH_DIM, device=self.device)
        sq = torch.zeros((), device=self.device)
        for i, g in enumerate(grads):
            if g is None:
                continue
            f = g.detach().float().flatten()
            if precond is not None:
                f = f / precond[i]
            n = f.numel()
            idx = (self.bucket[:n] + self.offset[i]) % SKETCH_DIM
            out.index_add_(0, idx, f * (self.sign[:n] * self.psign[i]))
            sq = sq + f.square().sum()
        return out.cpu(), float(sq)


def _energy_split(X: torch.Tensor) -> dict:
    """``X``: (S, M, K) sketches. Energy of the grand mean, between samples and
    between draws, as fractions of the mean squared norm (unbiased: the mean
    terms use only inner products of distinct vectors)."""
    S, M, _ = X.shape
    flat = X.reshape(S * M, -1).double()
    G = flat @ flat.T
    sid = torch.arange(S).repeat_interleave(M)
    same = sid[:, None] == sid[None, :]
    eye = torch.eye(S * M, dtype=torch.bool)
    total = G.diagonal().mean().item()
    cross = G[~same].mean().item() if S > 1 else float("nan")
    out = {"mean_sq_norm": total, "mean_frac": cross / total}
    if M > 1:
        within = G[same & ~eye].mean().item()
        out["between_sample_frac"] = (within - cross) / total
        out["between_draw_frac"] = 1.0 - within / total
    else:
        out["noise_frac"] = 1.0 - cross / total
    return out


def _unit(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _cross_mean_cos(X: torch.Tensor, Y: torch.Tensor) -> float:
    """cos between the grand means of two (S, M, K) families, from pairs of
    distinct samples only (same-sample noise shared by X and Y stays out)."""
    S = X.shape[0]
    xm, ym = X.mean(1).double(), Y.mean(1).double()
    off = ~torch.eye(S, dtype=torch.bool)
    xy = (xm @ ym.T)[off].mean().item()
    xx = (xm @ xm.T)[off].mean().item()
    yy = (ym @ ym.T)[off].mean().item()
    if xx <= 0 or yy <= 0:
        return float("nan")
    return xy / (xx * yy) ** 0.5


def _same_sample_cos(X: torch.Tensor, Y: torch.Tensor) -> float:
    return (
        torch.nn.functional.cosine_similarity(X.double(), Y.double(), dim=-1)
        .mean()
        .item()
    )


def _zeros_for_none(grads, params):
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--probe_samples", type=int, default=16)
    own.add_argument("--probe_draws", type=int, default=3)
    own.add_argument("--probe_label", type=str, default="grad_noise")
    probe, rest = own.parse_known_args()

    args = build_argparser().parse_args(rest)
    if not args.resume:
        raise SystemExit("--resume <bundle> is required")
    # The run's own name stays untouched: build_run writes a config snapshot.
    args.output_dir = tempfile.mkdtemp(prefix="probe_grad_noise_")
    args.output_name = "probe_grad_noise"
    cfg = resolve_config(args, load_toml(args.config))
    # One rollout graph is backwarded once per term and draw; a compiled backward
    # with donated buffers refuses retain_graph.
    import torch._functorch.config as functorch_config

    functorch_config.donated_buffer = False
    if cfg.student_steps != 2 or not cfg.rollout_grad:
        raise SystemExit("probe expects a superturbo config (2 steps, rollout_grad)")
    ctx = build_run(args, cfg)
    turbo, device = ctx.turbo, ctx.device

    params = [p for grp in ctx.student_opt.param_groups for p in grp["params"]]
    grp0 = ctx.student_opt.param_groups[0]
    beta2, adam_eps = grp0["betas"][1], grp0["eps"]
    precond = []
    for p in params:
        st = ctx.student_opt.state[p]
        t = float(st["step"])
        v_hat = st["exp_avg_sq"].float() / (1.0 - beta2**t)
        precond.append((v_hat.sqrt() + adam_eps).flatten())
    n_params = sum(p.numel() for p in params)
    logger.info(
        f"probe: {len(params)} student tensors, {n_params:,} params, "
        f"bundle step {ctx.start_step}, S={probe.probe_samples} M={probe.probe_draws}"
    )
    sketch = Sketcher(params, device)
    sigmas = ctx.student_sigmas
    gan_w = cfg.gan_loss_weight_gen if turbo.disc is not None else 0.0
    surrogate = mfd_surrogate if cfg.mfd_dtau > 0.0 else dmd_surrogate

    S, M = probe.probe_samples, probe.probe_draws
    fam = {k: {"raw": [], "adam": []} for k in ("anchor", "dm", "dm_step0", "gan")}
    norms = {k: [] for k in ("anchor", "dm", "dm_step0", "gan")}
    draws = []
    data_iter = iter(ctx.dataloader)

    def take(key, grads):
        g = _zeros_for_none(grads, params)
        raw, sq = sketch(g)
        adam, _ = sketch(g, precond)
        fam[key]["raw"].append(raw)
        fam[key]["adam"].append(adam)
        norms[key].append(sq**0.5)

    for s in range(S):
        batch = next(data_iter)
        latents = batch["latents"].to(device, dtype=ctx.dtype)
        crossattn_emb = batch["crossattn_emb"].to(device, dtype=ctx.cond_dtype)
        B = latents.shape[0]
        torch.compiler.cudagraph_mark_step_begin()
        eps = torch.randn_like(latents)
        c_null = uncond_for_batch(ctx.uncond_base, crossattn_emb)
        v_target = teacher_anchor(ctx, cfg, eps, crossattn_emb, c_null, B)
        roll = student_rollout(ctx, cfg, eps, v_target, sigmas, 1, crossattn_emb, B)
        turbo.set_fake_source(roll.s_g)

        take(
            "anchor",
            torch.autograd.grad(
                cfg.div_weight * roll.div_loss,
                params,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        for m in range(M):
            dmd = surrogate(ctx, cfg, roll.x_pred, crossattn_emb, c_null, B)
            loss_dm = (dmd.grad_signal * roll.x_pred.float()).mean()
            out = torch.autograd.grad(
                loss_dm, params + [roll.x_g], retain_graph=True, allow_unused=True
            )
            take("dm", out[:-1])
            g_step0 = torch.autograd.grad(
                roll.x_g,
                params,
                grad_outputs=out[-1],
                retain_graph=True,
                allow_unused=True,
            )
            take("dm_step0", g_step0)
            del out, g_step0
            if gan_w > 0.0:
                gan_loss = gan_generator_term(
                    ctx, cfg, roll.x_pred, dmd.tau_dm, dmd.eps_dm, crossattn_emb, gan_w
                )
                take(
                    "gan",
                    torch.autograd.grad(
                        gan_w * gan_loss, params, retain_graph=True, allow_unused=True
                    ),
                )
                del gan_loss
            draws.append(
                {
                    "sample": s,
                    "sigma_a": float(dmd.tau_dm.float().mean()),
                    "grad_signal_rms": float(dmd.grad_signal.square().mean().sqrt()),
                    "dm_norm": norms["dm"][-1],
                    "dm_step0_norm": norms["dm_step0"][-1],
                    "gan_norm": norms["gan"][-1] if gan_w > 0.0 else 0.0,
                    "tokens": int(latents.shape[-1] * latents.shape[-2] // 4),
                }
            )
        # The next sample does not fit beside this rollout's saved activations,
        # and under retain_graph any leftover reference to the graph keeps them.
        # One non-retaining backward frees them whatever still points at it.
        torch.autograd.grad(roll.x_pred.float().sum(), params, allow_unused=True)
        del roll, v_target, loss_dm, dmd, eps
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            f"sample {s + 1}/{S}: latent {tuple(latents.shape)}  "
            f"|anchor| {norms['anchor'][-1]:.2f}  "
            f"|dm| {np.mean(norms['dm'][-M:]):.2f}  "
            f"|dm step0| {np.mean(norms['dm_step0'][-M:]):.2f}  "
            f"|gan| {np.mean(norms['gan'][-M:]) if gan_w > 0.0 else 0.0:.2f}  "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB  "
            f"held {torch.cuda.memory_allocated() / 2**30:.2f} GB"
        )

    metrics: dict = {
        "bundle_step": ctx.start_step,
        "n_params": n_params,
        "samples": S,
        "draws": M,
        "norms": {k: float(np.mean(v)) for k, v in norms.items() if v},
        "draw_rows": draws,
    }
    for space in ("raw", "adam"):
        A = torch.stack(fam["anchor"][space]).reshape(S, 1, -1)
        D = torch.stack(fam["dm"][space]).reshape(S, M, -1)
        D0 = torch.stack(fam["dm_step0"][space]).reshape(S, M, -1)
        D1 = D - D0
        fams = {"anchor": A, "dm": D, "dm_step0": D0, "dm_step1": D1}
        total = D + A
        if gan_w > 0.0:
            Gn = torch.stack(fam["gan"][space]).reshape(S, M, -1)
            fams["gan"] = Gn
            total = total + Gn
            fams["dm_plus_anchor"] = D + A
        fams["total"] = total
        res = {"split": {}, "split_unit": {}, "rel": {}}
        for k, X in fams.items():
            res["split"][k] = _energy_split(X)
            res["split_unit"][k] = _energy_split(_unit(X))
        Ae = A.expand(S, M, -1)
        res["rel"] = {
            "same_sample_cos(dm_step0, anchor)": _same_sample_cos(D0, Ae),
            "same_sample_cos(dm_step0, dm_step1)": _same_sample_cos(D0, D1),
            "same_sample_cos(dm, anchor)": _same_sample_cos(D, Ae),
            "mean_cos(dm, anchor)": _cross_mean_cos(D, Ae),
            "mean_cos(dm_step0, anchor)": _cross_mean_cos(D0, Ae),
            "mean_cos(dm_step0, dm_step1)": _cross_mean_cos(D0, D1),
            "share_of_total_sq_norm": {
                k: res["split"][k]["mean_sq_norm"]
                / res["split"]["total"]["mean_sq_norm"]
                for k in fams
                if k != "total"
            },
        }
        if gan_w > 0.0:
            res["rel"]["same_sample_cos(gan, dm)"] = _same_sample_cos(fams["gan"], D)
            res["rel"]["mean_cos(gan, dm)"] = _cross_mean_cos(fams["gan"], D)
        metrics[space] = res
        logger.info(f"--- {space} ---")
        for k in fams:
            sp, su = res["split"][k], res["split_unit"][k]
            logger.info(
                f"{k:15s} rms-norm {sp['mean_sq_norm'] ** 0.5:9.3f}  unit-vector split: "
                + "  ".join(
                    f"{n}={v:+.3f}" for n, v in su.items() if n != "mean_sq_norm"
                )
            )
        for k, v in res["rel"].items():
            logger.info(f"{k}: {v}")

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
