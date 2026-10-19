"""Compare Anima's SDPA and native ROCm FlashAttention forward/backward paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.backend import resolve_attention_mode  # noqa: E402
from networks.attention_dispatch import AttentionParams, dispatch_attention  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if resolve_attention_mode("flash", torch) != "flash":
        raise RuntimeError(
            "the compatible native ROCm FlashAttention wheel is required"
        )
    torch.manual_seed(42)
    data = [
        torch.randn(1, args.seq_len, 16, 128, device=device, dtype=torch.bfloat16) * 0.2
        for _ in range(3)
    ]
    results = {}
    reference = None
    for mode in ("torch", "flash"):
        params = AttentionParams(mode)
        values = [t.detach().clone().requires_grad_() for t in data]
        times = []
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(args.warmup + args.steps):
            for t in values:
                t.grad = None
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = dispatch_attention(*values, attn_params=params)
            output.float().square().mean().backward()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000
            if step >= args.warmup:
                times.append(elapsed)
        actual = [t.detach().float().cpu() for t in (output, *(v.grad for v in values))]
        if not all(torch.isfinite(t).all() for t in actual):
            raise RuntimeError(f"{mode} produced non-finite outputs or gradients")
        if reference is None:
            reference = actual
        else:
            for value, expected in zip(actual, reference):
                torch.testing.assert_close(value, expected, rtol=0.06, atol=0.015)
        results[mode] = {
            "median_forward_backward_ms": statistics.median(times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "samples_ms": times,
            "max_abs_output_gradient_diff": max(
                (a - b).abs().max().item() for a, b in zip(actual, reference)
            ),
        }
        print(json.dumps({mode: results[mode]}), flush=True)
        del output, values
        torch.cuda.empty_cache()
    run_dir = make_run_dir("rocm_flash", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=results,
        label=args.label,
        device=device,
    )
    print(f"result: {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
