"""Floor and control for the anchor feature matching (``dpdmd.fm_weight``).

On cached anchor pairs, the relative feature L2 at the FM tap (σ_k state) for:

* ``own_det``: the teacher's no-grad features of x_t, twice (kernel non-determinism)
* ``own_path``: x_t on the training path (grad + block checkpoint) against no-grad
* ``student``: x_s = ε − (1 − σ_k)·v_student against x_t (what the loss sees)
* ``jitter``: v_target plus Gaussian noise of the student residual's rms, against x_t
  (a residual of equal norm in a random direction)

Run with the training config of the arm (its student init is the student read)::

    python bench/turbo/probe_fm_floor.py --config <turbo.toml> [--n 16]
"""

from __future__ import annotations

import argparse
import json

import torch

from bench._common import make_run_dir, write_result
from library.config.resolved import load_toml
from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.setup import build_run
from scripts.distill_turbo.steps import _rel_feat_mse, selective_block_grad_ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--label", default="fm_floor")
    a = p.parse_args()

    targs = build_argparser().parse_args(["--config", a.config])
    cfg = resolve_config(targs, load_toml(a.config))
    ctx = build_run(targs, cfg)
    bidx = cfg.fm_block_idx if cfg.fm_block_idx >= 0 else ctx.model.num_blocks // 2
    sig = ctx.teacher_anchor_sigmas[cfg.k_anchor]
    tap = {bidx}

    def feats(x, c, B, *, grad):
        t = torch.full((B,), sig, device=ctx.device, dtype=ctx.dtype)
        if not grad:
            return ctx.forward(
                "teacher",
                x,
                t,
                c,
                no_grad=True,
                return_block_features=tap,
                return_features_early=True,
            )[bidx].detach()
        xg = x.detach().requires_grad_()
        with selective_block_grad_ckpt(ctx.model):
            f = ctx.forward(
                "teacher",
                xg,
                t,
                c,
                no_grad=False,
                return_block_features=tap,
                return_features_early=True,
            )[bidx]
        return f.detach()

    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for i in range(a.n):
        b = ctx.next_batch()
        c, B = b.crossattn_emb, b.latents.shape[0]
        e, vt = b.anchor_eps.float(), b.anchor_v.float()
        t0 = torch.full((B,), ctx.student_sigmas[0], device=ctx.device, dtype=ctx.dtype)
        vs = ctx.forward("student", b.anchor_eps, t0, c, no_grad=True).float()
        res_rms = (vs - vt).pow(2).mean().sqrt()
        noise = torch.randn(vt.shape, generator=g).to(vt.device)
        vj = vt + noise * res_rms

        x_t = e - (1 - sig) * vt
        f_t = feats(x_t, c, B, grad=False)
        row = {
            "mse": float((vs - vt).pow(2).mean()),
            "own_det": float(_rel_feat_mse(feats(x_t, c, B, grad=False), f_t)),
            "own_path": float(_rel_feat_mse(feats(x_t, c, B, grad=True), f_t)),
            "student": float(
                _rel_feat_mse(feats(e - (1 - sig) * vs, c, B, grad=False), f_t)
            ),
            "jitter": float(
                _rel_feat_mse(feats(e - (1 - sig) * vj, c, B, grad=False), f_t)
            ),
        }
        rows.append(row)
        print(i, json.dumps({k: round(v, 5) for k, v in row.items()}), flush=True)

    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    print("mean", json.dumps({k: round(v, 5) for k, v in mean.items()}))
    run_dir = make_run_dir("turbo", a.label)
    write_result(
        run_dir,
        script=__file__,
        args={**vars(a), "fm_block_idx": bidx, "sigma_k": sig},
        metrics={"mean": mean, "rows": rows},
        label=a.label,
    )


if __name__ == "__main__":
    main()
