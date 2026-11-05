#!/usr/bin/env python3
"""Which σ band carries a LoRA's look and its artifacts: render with the LoRA off
at high σ or at low σ (``inference.py --lora_sigma_range``) beside the plain
multiplier sweep.

Arms per LoRA (P-GRAFT hooks, the ``make test`` recipe: 28 st euler, cfg 4,
shift 3, 1024²):

- ``hooks``: LoRA on at every step through the hooks (the control for the hook
  path; ``m1.0`` is the static merge);
- ``hi_off``: off at σ >= ``--hi`` (default 0.8);
- ``lo_off``: off at σ < ``--lo`` (default 0.4);
- ``mask``: the training-time T-LoRA rank mask applied per step
  (``--lora_timestep_mask``, min_rank 1, linear), where inference otherwise runs
  full rank.

The grid puts them next to existing ``m0.6`` / ``m0.8`` / ``m1.0`` renders of the
same LoRA and render seed when those folders exist (``--mult_dirs``).

    make daemon-run ARGS="--stall-timeout 900 bench/anima_abclip/sigma_gate.py"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bench._common import make_run_dir, write_result
from bench.anima_abclip.run_ab import BODIES, TAGS, render_seed
from bench.anima_ema.render_ab import INFER, _by_seed, grid
from library.env import resolve_under_home

CKPT = resolve_under_home("output/ckpt")
OUT = resolve_under_home("output/tests/sigma_gate")


def steps_in(lo: float, hi: float, steps: int = 28, shift: float = 3.0) -> int:
    n = 0
    for i in range(steps):
        t = 1 - i / steps
        s = shift * t / (1 + (shift - 1) * t)
        n += lo <= s < hi
    return n


def render(prompts: list[str], seed: int, lora: Path, out: Path, extra: list[str]):
    out.mkdir(parents=True, exist_ok=True)
    done = _by_seed(out)
    want = [render_seed(p, seed) for p in range(len(prompts))]
    if all(w in done for w in want):
        print(f"reusing {out}", flush=True)
        return done
    pf = out / "prompts.txt"
    pf.write_text(
        "".join(f"{pr} --d {render_seed(p, seed)}\n" for p, pr in enumerate(prompts)),
        encoding="utf-8",
    )
    cmd = [sys.executable, *INFER, "--lora_weight", str(lora), *extra]
    cmd += ["--from_file", str(pf), "--save_path", str(out)]
    print(f"[{out.name}] {' '.join(extra)}", flush=True)
    subprocess.run(cmd, check=True, cwd=resolve_under_home("."))
    return _by_seed(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--art", default="sweet", choices=sorted(TAGS))
    ap.add_argument("--stem", default="abclip_{art}_B_s{seed}")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--render_seed", type=int, default=3)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument(
        "--mult_dirs",
        default="base=output/tests/sigma_gate/base_nolora,"
        "m0.6=output/tests/sweet_mult/s{seed}_m0.6,"
        "m0.8=output/tests/sweet_mult/s{seed}_m0.8,"
        "m1.0=output/tests/sigband/{stem}",
        help="name=dir pattern ({seed}, {stem}) of existing renders to put in the grid",
    )
    args = ap.parse_args()

    prompts = [b.format(a=TAGS[args.art][0]) for b in BODIES]
    arms = {
        "hooks": ["--pgraft"],
        "hi_off": ["--lora_sigma_range", "0", str(args.hi)],
        "lo_off": ["--lora_sigma_range", str(args.lo), "2"],
        "mask": ["--lora_timestep_mask"],
    }
    run_dir = make_run_dir("anima_abclip", label=f"sigma_gate_{args.art}")
    grids = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        stem = args.stem.format(art=args.art, seed=seed)
        lora = CKPT / f"{stem}.safetensors"
        cols: dict[str, dict[int, Path]] = {}
        for spec in filter(None, args.mult_dirs.split(",")):
            name, pat = spec.split("=", 1)
            d = resolve_under_home(pat.format(seed=seed, stem=stem))
            if d.is_dir():
                cols[name] = _by_seed(d)
        for arm, extra in arms.items():
            cols[arm] = render(
                prompts, args.render_seed, lora, OUT / f"{stem}_{arm}", extra
            )
        keys = [render_seed(p, args.render_seed) for p in range(len(prompts))]
        names = [n for n, by in cols.items() if all(k in by for k in keys)]
        path = run_dir / f"grid_{stem}.png"
        grid(
            [[cols[n][k] for n in names] for k in keys],
            names,
            [f"p{p}" for p in range(len(keys))],
            path,
        )
        grids.append(str(path))
    write_result(
        run_dir,
        script=__file__,
        args={**vars(args), "prompts": prompts, "arms": arms},
        metrics={
            "steps_lora_off": {
                "hi_off": steps_in(args.hi, 2.0),
                "lo_off": steps_in(-1.0, args.lo),
                "of": 28,
            }
        },
        label=f"sigma_gate_{args.art}",
        artifacts=grids,
    )
    print(f"grids in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
