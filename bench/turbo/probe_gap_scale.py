"""Where the student / teacher feature gap sits in space, and what the paired disc reads.

On cached anchor pairs, at the FM / GAN tap of the σ_k state (frozen teacher features):

* ``keep_k``: the share of the gap's energy left after a k×k token mean pool,
  ``‖pool_k(f_s − f_t)‖² / ‖f_s − f_t‖²`` (per element). 1/k² for a gap with no
  spatial correlation, 1 for one that is constant over the k×k cell.
* ``rel_k``: the relative feature L2 of the pooled features (``_rel_feat_mse``).
* The same for ``jitter`` (v_target plus Gaussian noise of the student residual's
  rms: an equal-norm residual in a random direction) and, as the features' own
  smoothness, ``keep_k`` of ``f_t`` about its per-channel spatial mean.

With a disc in the resume bundle, per token: the logits of the teacher state, the
student state and the jitter state, their correlation over tokens, and the share
of the logit variance common to the pair (``var(m) / (var(m) + var(d))`` with
``m`` the pair mean and ``d`` half the pair difference); and ``r1_a<alpha>``, the
approximate-R1 term of the disc step at that perturbation scale.

Run with the arm's config; ``--resume auto`` loads its online student and disc::

    python bench/turbo/probe_gap_scale.py --config <turbo.toml> [--n 16]
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from bench._common import make_run_dir, write_result
from library.config.resolved import load_toml
from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.setup import build_run
from scripts.distill_turbo.steps import _rel_feat_mse

POOLS = (1, 2, 4)
R1_ALPHAS = (0.05, 0.1, 0.2)


def to_grid(f: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, T, H, W, D) or the flattened (B, 1, L, 1, D) → (B, H, W, D)."""
    f = f.float()
    f = f.reshape(f.shape[0], -1, f.shape[-1])
    if f.shape[1] < h * w:
        raise SystemExit(f"{f.shape[1]} tokens for a {h}x{w} grid")
    return f[:, : h * w].reshape(f.shape[0], h, w, f.shape[-1])


def pool(g: torch.Tensor, k: int) -> torch.Tensor:
    if k == 1:
        return g
    B, H, W, D = g.shape
    g = g[:, : H // k * k, : W // k * k]
    return g.reshape(B, H // k, k, W // k, k, D).mean((2, 4))


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--resume", default="auto")
    p.add_argument("--label", default="gap_scale")
    a = p.parse_args()

    targs = build_argparser().parse_args(["--config", a.config, "--resume", a.resume])
    cfg = resolve_config(targs, load_toml(a.config))
    ctx = build_run(targs, cfg)
    turbo = ctx.turbo
    bidx = cfg.fm_block_idx if cfg.fm_block_idx >= 0 else ctx.model.num_blocks // 2
    sig = ctx.teacher_anchor_sigmas[cfg.k_anchor]
    taps = {bidx} | (turbo.gan_feature_set or set())
    ps = ctx.model.patch_spatial

    def feats(x, c, B):
        t = torch.full((B,), sig, device=ctx.device, dtype=ctx.dtype)
        return ctx.forward(
            "teacher",
            x,
            t,
            c,
            no_grad=True,
            return_block_features=taps,
            return_features_early=True,
        )

    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for i in range(a.n):
        b = ctx.next_batch()
        c, B = b.crossattn_emb, b.latents.shape[0]
        h, w = b.latents.shape[-2] // ps, b.latents.shape[-1] // ps
        e, vt = b.anchor_eps.float(), b.anchor_v.float()
        t0 = torch.full((B,), ctx.student_sigmas[0], device=ctx.device, dtype=ctx.dtype)
        vs = ctx.forward("student", b.anchor_eps, t0, c, no_grad=True).float()
        res_rms = (vs - vt).pow(2).mean().sqrt()
        vj = vt + torch.randn(vt.shape, generator=g).to(vt.device) * res_rms

        fd = {
            "t": feats(e - (1 - sig) * vt, c, B),
            "s": feats(e - (1 - sig) * vs, c, B),
            "j": feats(e - (1 - sig) * vj, c, B),
        }
        gr = {k: to_grid(v[bidx], h, w) for k, v in fd.items()}
        row = {"mse": float((vs - vt).pow(2).mean()), "tokens": h * w}
        own = gr["t"] - gr["t"].mean((1, 2), keepdim=True)  # per channel, over space
        for k in POOLS:
            pt = pool(gr["t"], k)
            row[f"keep{k}_own"] = float(pool(own, k).pow(2).mean() / own.pow(2).mean())
            for arm, name in (("s", "student"), ("j", "jitter")):
                d = gr[arm] - gr["t"]
                row[f"keep{k}_{name}"] = float(
                    pool(d, k).pow(2).mean() / d.pow(2).mean()
                )
                row[f"rel{k}_{name}"] = float(_rel_feat_mse(pool(gr[arm], k), pt))
        if turbo.disc is not None:
            grid = (h, w) if turbo.disc.needs_grid else None

            def logits(feat_dict):
                return turbo.disc(turbo.features_in_order(feat_dict), grid).float()

            lg = {k: logits(v).flatten() for k, v in fd.items()}
            m, d = (lg["t"] + lg["s"]) / 2, (lg["t"] - lg["s"]) / 2
            row |= {
                "logit_t": float(lg["t"].mean()),
                "logit_s": float(lg["s"].mean()),
                "logit_j": float(lg["j"].mean()),
                "disc_loss": float(
                    torch.nn.functional.softplus(lg["s"]).mean()
                    + torch.nn.functional.softplus(-lg["t"]).mean()
                ),
            }
            if lg["t"].numel() > 8:  # per-token heads
                row |= {
                    "std_t": float(lg["t"].std()),
                    "std_s": float(lg["s"].std()),
                    "corr_ts": corr(lg["t"], lg["s"]),
                    "common_share": float(m.var() / (m.var() + d.var())),
                    "s_above_0": float((lg["s"] > 0).float().mean()),
                    "t_below_0": float((lg["t"] < 0).float().mean()),
                }
            # The approximate-R1 term of the paired disc step (gan.r1_weight):
            # squared logit change under x_t + alpha * eta, eta ~ N(0, I).
            x_t = e - (1 - sig) * vt
            for alpha in R1_ALPHAS:
                eta = torch.randn(x_t.shape, generator=g).to(x_t.device)
                lg_a = logits(feats(x_t + alpha * eta, c, B)).flatten()
                row[f"r1_a{alpha}"] = float((lg_a - lg["t"]).pow(2).mean())
        rows.append(row)
        print(i, json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    keys = list(rows[0])
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    median = {k: statistics.median(r[k] for r in rows) for k in keys}
    print("mean", json.dumps({k: round(v, 4) for k, v in mean.items()}))
    print("median", json.dumps({k: round(v, 4) for k, v in median.items()}))
    run_dir = make_run_dir("turbo", a.label)
    write_result(
        run_dir,
        script=__file__,
        args={**vars(a), "fm_block_idx": bidx, "sigma_k": sig},
        metrics={"mean": mean, "median": median, "rows": rows},
        label=a.label,
    )


if __name__ == "__main__":
    main()
