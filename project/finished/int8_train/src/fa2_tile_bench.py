"""FA2 forward tile on sm_120, and CUDNN attention (step_headroom.md, candidate 1).

Three arms on the captured Anima attention calls, plus MOSS's shape on random bf16 (timing
only): FA2 128x64, FA2 128x32 (one binary, tile chosen through ``cc_spoof.c``) and SDPA on the
CUDNN backend. Timing alternates the arms over rounds, so clock drift hits them alike; the
first round is warm-up. Keep gate: forward at least 1.15x FA2 at FA2's distance from fp32.

``--fa2_dir`` runs a build whose hdim128 branch takes 128x32 on cc 12 in place of the wheel.

    make daemon-run ARGS="--stall-timeout 300 project/finished/int8_train/src/fa2_tile_bench.py"
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from library.env import resolve_under_home

SRC = Path(__file__).parent


def load_spoof(out_dir: Path) -> ctypes.CDLL:
    """Re-execs under LD_PRELOAD: an RTLD_GLOBAL load of the shim did not reach FA2's calls."""
    so = out_dir / "cc_spoof.so"
    if str(so) not in os.environ.get("LD_PRELOAD", ""):
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", "-o", so, SRC / "cc_spoof.c", "-ldl"],
            check=True,
        )
        env = {**os.environ, "LD_PRELOAD": str(so)}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return ctypes.CDLL(str(so))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument("--out", default="project/finished/int8_train/assets/fa2_tile.json")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--fa2_dir", default=None)
    ap.add_argument("--moss_tokens", type=int, default=4600)
    args = ap.parse_args()
    if args.fa2_dir:
        sys.path.insert(0, args.fa2_dir)

    shim = load_spoof(Path(resolve_under_home("output/int8_train")))

    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = torch.load(resolve_under_home(args.capture))
    dev = "cuda"

    # Capability each tile needs: the wheel takes 128x64 on the real 12.0, a patched build 128x32.
    cc = (
        {"fa2_128x64": (9, 0), "fa2_128x32": (0, 0)}
        if args.fa2_dir
        else {
            "fa2_128x64": (0, 0),
            "fa2_128x32": (8, 6),
        }
    )

    def fa2_arm(name):
        def fn(q, k, v):
            shim.cc_spoof_set(*cc[name])
            o = flash_attn_func(q, k, v)
            shim.cc_spoof_set(0, 0)
            return o

        return fn

    def cudnn(q, k, v):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
        return o.transpose(1, 2)

    arms = {
        "fa2_128x64": fa2_arm("fa2_128x64"),
        "fa2_128x32": fa2_arm("fa2_128x32"),
        "cudnn": cudnn,
    }

    def timeit(fn, iters=50):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    errs = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for s in sets:
            q, k, v = (s[n].to(dev).contiguous() for n in "qkv")
            ref = F.scaled_dot_product_attention(
                q.float().transpose(1, 2),
                k.float().transpose(1, 2),
                v.float().transpose(1, 2),
            ).transpose(1, 2)
            for name, fn in arms.items():
                err = float((fn(q, k, v).float() - ref).norm() / ref.norm())
                errs[s["kind"]][name].append(err)
    err_med = {
        kind: {a: statistics.median(x) for a, x in by.items()}
        for kind, by in errs.items()
    }

    moss = torch.randn(1, args.moss_tokens, 24, 128, device=dev, dtype=torch.bfloat16)
    inputs = {"moss_self": (moss, moss.flip(1), moss.roll(1, 2))}
    for kind in ("self", "cross"):
        s = next(x for x in sets if x["kind"] == kind and x["image"] == 0)
        inputs[kind] = tuple(s[n].to(dev).contiguous() for n in "qkv")

    speed = {}
    for kind, (q, k, v) in inputs.items():
        fwd, step = defaultdict(list), defaultdict(list)
        with torch.no_grad():
            for r in range(args.rounds):
                for name, fn in arms.items():
                    t = timeit(lambda: fn(q, k, v))
                    if r:
                        fwd[name].append(t)

        qg, kg, vg = (x.clone().requires_grad_() for x in (q, k, v))

        def fwd_bwd(fn):
            fn(qg, kg, vg).sum().backward()
            qg.grad = kg.grad = vg.grad = None

        for r in range(args.rounds):
            for name, fn in arms.items():
                t = timeit(lambda: fwd_bwd(fn), iters=20)
                if r:
                    step[name].append(t)
        speed[kind] = {
            "shape": {"q": list(q.shape), "Lk": k.shape[1]},
            "fwd_ms": {a: statistics.median(x) for a, x in fwd.items()},
            "fwd_bwd_ms": {a: statistics.median(x) for a, x in step.items()},
            "fwd_rounds": dict(fwd),
        }

    assert shim.cc_spoof_calls() > 0, "cc_spoof did not interpose FA2's capability read"
    q, k, v = inputs["self"]
    names = {}
    for name in ("fa2_128x64", "fa2_128x32"):
        with (
            torch.no_grad(),
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof,
        ):
            arms[name](q, k, v)
            torch.cuda.synchronize()
        names[name] = sorted(
            {e.name[:80] for e in prof.events() if "flash_fwd_kernel" in e.name}
        )

    import flash_attn_2_cuda

    result = {
        "fa2_so": flash_attn_2_cuda.__file__,
        "gpu": torch.cuda.get_device_name(),
        "cudnn": torch.backends.cudnn.version(),
        "err_median": err_med,
        "speed": speed,
        "kernels": names,
    }
    out = Path(resolve_under_home(args.out))
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    for kind, sp in speed.items():
        base, base_s = sp["fwd_ms"]["fa2_128x64"], sp["fwd_bwd_ms"]["fa2_128x64"]
        for a in arms:
            print(
                f"{kind} {a}: fwd {sp['fwd_ms'][a]:.3f} ms ({base / sp['fwd_ms'][a]:.3f}x), "
                f"fwd+bwd {sp['fwd_bwd_ms'][a]:.3f} ms ({base_s / sp['fwd_bwd_ms'][a]:.3f}x)"
                + (f", err {err_med[kind][a]:.5f}" if kind in err_med else "")
            )
    print(names)


if __name__ == "__main__":
    main()
