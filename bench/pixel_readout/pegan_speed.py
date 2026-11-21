#!/usr/bin/env python3
"""PE-Spatial discriminator step time (``x2_pegan.PEDisc``): the generator term with its
backward to the fake crop, then one discriminator update, on random 512-px crops.

Arms: ``eager`` (the tower's tokens of ``gen_term`` reused in ``step``, the two noised
crops as one batch), ``compile`` (the same with the tower's forward compiled), and with
``--old <x2_pegan.py>`` a reference implementation from that file (e.g. the version with
four separate forwards per update, from ``git show <rev>:bench/pixel_readout/x2_pegan.py``).

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/pegan_speed.py"
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import torch

from bench._common import make_run_dir, write_result
from bench.pixel_readout import x2_pegan


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--old", default=None, help="a reference x2_pegan.py to time beside")
    p.add_argument("--label", default=None)
    args = p.parse_args()
    dev = torch.device("cuda")
    c = args.crop

    arms = [("eager", x2_pegan, {}), ("compile", x2_pegan, {"compile": True})]
    if args.old:
        spec = importlib.util.spec_from_file_location("x2_pegan_old", args.old)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        arms.insert(0, ("old", old, {}))

    metrics: dict = {}
    for name, mod, kw in arms:
        torch.manual_seed(0)
        disc = mod.PEDisc(dev, **kw)
        g = torch.Generator(device=dev).manual_seed(1)
        t_gen = t_step = 0.0
        last = {}
        torch.cuda.reset_peak_memory_stats()
        t_first = None
        for i in range(args.warmup + args.iters):
            real = torch.rand(1, 3, c, c, generator=g, device=dev) * 2 - 1
            fake = real + 0.05 * torch.randn(1, 3, c, c, generator=g, device=dev)
            fake.requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            term = disc.gen_term(fake, real)
            term.backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            torch.manual_seed(100 + i)  # the R1 / R2 noise, the same in every arm
            last = disc.step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if i == 0:
                t_first = t2 - t0
            if i >= args.warmup:
                t_gen += t1 - t0
                t_step += t2 - t1
        n = args.iters
        metrics[name] = {
            "gen_ms": t_gen / n * 1e3,
            "step_ms": t_step / n * 1e3,
            "total_ms": (t_gen + t_step) / n * 1e3,
            "first_update_s": t_first,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "last_gen_term": term.item(),
            **{f"last_{k}": v for k, v in last.items()},
        }
        m = metrics[name]
        print(
            f"[{name}] gen {m['gen_ms']:.1f} ms  step {m['step_ms']:.1f} ms  total "
            f"{m['total_ms']:.1f} ms  first update {t_first:.1f} s  peak "
            f"{m['peak_gb']:.2f} GB  last gen {m['last_gen_term']:.5f} d_loss "
            f"{m['last_d_loss']:.5f} margin {m['last_d_margin']:.5f}",
            flush=True,
        )
        del disc
        torch.cuda.empty_cache()
    # the compiled tower against the eager one: same heads, same crops
    torch.manual_seed(0)
    de, dc = x2_pegan.PEDisc(dev), x2_pegan.PEDisc(dev, compile=True)
    dc.heads.load_state_dict(de.heads.state_dict())
    g = torch.Generator(device=dev).manual_seed(7)
    real = torch.rand(1, 3, c, c, generator=g, device=dev) * 2 - 1
    grads, taps, terms = [], [], []
    for d in (de, dc):
        fake = real + 0.05 * torch.randn(
            1, 3, c, c, generator=g.manual_seed(8), device=dev
        )
        fake.requires_grad_(True)
        term = d.gen_term(fake, real)
        grads.append(torch.autograd.grad(term, fake)[0].float().flatten())
        taps.append([t.float() for t in d._stash[2]])
        terms.append(term.item())
    agree = {
        "gen_term_eager": terms[0],
        "gen_term_compile": terms[1],
        "grad_cosine": float(torch.nn.functional.cosine_similarity(*grads, dim=0)),
        "grad_norm_ratio": float(grads[1].norm() / grads[0].norm()),
        "tap_rel_diff": [
            float((a - b).abs().mean() / a.abs().mean()) for a, b in zip(*taps)
        ],
    }
    metrics["compile_vs_eager"] = agree
    print(f"compile vs eager: {agree}", flush=True)
    run_dir = make_run_dir("pixel_readout", label=args.label)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=dev,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
