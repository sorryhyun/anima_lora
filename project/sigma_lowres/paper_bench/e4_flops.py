#!/usr/bin/env python3
"""E4 per-arm FLOPs from the trainer's `token_step_hist` run_end field.

Each E4 run's `run_end` progress event carries a histogram of realized
patch-token counts per train step (train.py, post σ-demote swap). This
script converts those to FLOPs exactly:

  FLOPs_fwd(run) = Σ_t hist[t] · F(t)

with F(t) measured once per distinct token count by a real DiT forward
under `torch.utils.flop_counter.FlopCounterMode` (batch 1, eager,
`attn_mode="torch"` — sdpa is FLOP-counted, flash-attn custom kernels are
not; the choice changes kernels, not FLOPs). Token count alone determines
F — the (H, W) split doesn't (linear terms are per-token, attention is
over the flattened sequence) — so each t is realized as a near-square
patch grid. Reported alongside: `flops_train ≈ 3 × flops_fwd` (the
standard fwd+bwd ≈ 3× fwd accounting; LoRA adapter + REPA overhead
excluded, identical across arms).

Usage (GPU — daemon)::

    make daemon-run ARGS="project/sigma_lowres/paper_bench/e4_flops.py \
        --jobs <job_id> [<job_id> …]"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from anima_lora import default_checkpoints  # noqa: E402

JOBS_DIR = REPO / "output" / "daemon" / "jobs"


def _grid_for(tokens: int) -> tuple[int, int]:
    """A valid near-square patch grid (ph, pw) with ph·pw == tokens, both
    ≤ 256 (the rope per-axis cap). Real bucket counts always factor this
    way; fall back to the divisor pair closest to square."""
    best = None
    for ph in range(int(tokens**0.5), 0, -1):
        if tokens % ph == 0:
            pw = tokens // ph
            if ph <= 256 and pw <= 256:
                return ph, pw
            if best is None:
                best = (ph, pw)
    raise ValueError(f"no valid grid for {tokens} tokens")


def _run_record(job_id: str) -> tuple[str, dict]:
    jdir = JOBS_DIR / job_id
    name = job_id
    job = jdir / "job.json"
    if job.exists():
        rec = json.loads(job.read_text())
        # Train jobs carry their CLI overrides in `extra`.
        argv = rec.get("extra") or rec.get("argv") or []
        if "--output_name" in argv:
            name = argv[argv.index("--output_name") + 1]
    hist = None
    with open(jdir / "progress.jsonl", encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("ev") == "run_end":
                hist = ev.get("token_step_hist")
                wall = ev.get("ts")
    if not hist:
        raise RuntimeError(f"{job_id}: no token_step_hist in run_end")
    return name, {"hist": {int(k): v for k, v in hist.items()}, "wall_s": wall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = dict(_run_record(j) for j in args.jobs)
    if len(runs) != len(args.jobs):
        raise RuntimeError(f"run-name collision: {len(args.jobs)} jobs → {list(runs)}")
    counts = sorted({t for r in runs.values() for t in r["hist"]})
    print(f"{len(runs)} runs, {len(counts)} distinct token counts", flush=True)

    from library.runtime.harness import build_anima

    bundle = build_anima(
        SimpleNamespace(device="cuda", dtype="bf16", attn_mode="torch"),
        dit_path=str(default_checkpoints().dit),
        adapter=None,
        train_mode=False,
    )
    anima, device = bundle.anima, bundle.device

    flops_per_count: dict[int, int] = {}
    ca = torch.zeros(1, 512, 1024, device=device, dtype=torch.bfloat16)
    sigma = torch.tensor([0.5], device=device)
    for t in counts:
        ph, pw = _grid_for(t)
        noisy = torch.zeros(
            1, 16, 1, 2 * ph, 2 * pw, device=device, dtype=torch.bfloat16
        )
        pad = torch.zeros(1, 1, 2 * ph, 2 * pw, device=device, dtype=torch.bfloat16)
        with (
            FlopCounterMode(display=False) as fc,
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            anima(noisy, sigma, ca, padding_mask=pad)
        flops_per_count[t] = int(fc.get_total_flops())
        print(f"  F({t}) = {flops_per_count[t] / 1e12:.3f} TFLOPs", flush=True)

    out = {
        "flops_per_count": flops_per_count,
        "note": "F(t) = one DiT forward at batch 1; flops_train ≈ 3 × flops_fwd",
        "runs": {},
    }
    for name, r in runs.items():
        fwd = sum(n * flops_per_count[t] for t, n in r["hist"].items())
        out["runs"][name] = {
            "steps": sum(r["hist"].values()),
            "flops_fwd": fwd,
            "flops_train_est": 3 * fwd,
            "wall_s": r["wall_s"],
            "pflops_fwd": round(fwd / 1e15, 3),
        }
        print(f"{name}: fwd {fwd / 1e15:.3f} PFLOPs · wall {r['wall_s']}s", flush=True)

    if args.out:
        out_dir = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
        out_dir = HERE / "runs" / f"{stamp}-e4-flops"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flops.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {out_dir / 'flops.json'}", flush=True)


if __name__ == "__main__":
    main()
