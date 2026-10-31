"""Wall time of each phase of a MOSS training step, no profiler; a sync at each boundary.

make daemon-run ARGS="project/int8_train/src/phase_timer_moss.py --cache <set> --max_steps 40"
"""

import dataclasses
import os
import statistics
import sys
import time
from library.moss.requests import TrainRequest

req = TrainRequest.from_argv(sys.argv[1:])
if req.expandable_segments:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from library.moss import train as T  # noqa: E402
from library.moss.training.muon import BatchedMuon  # noqa: E402

marks = {}
t = [0.0]


def lap(name):
    torch.cuda.synchronize()
    now = time.perf_counter()
    marks.setdefault(name, []).append((now - t[0]) * 1e3)
    t[0] = now


def wrap(obj, attr, name):
    orig = getattr(obj, attr)

    def f(*a, **k):
        lap(name + "_before")
        out = orig(*a, **k)
        lap(name)
        return out

    setattr(obj, attr, f)


wrap(torch.Tensor, "backward", "backward")
wrap(torch.nn.utils, "clip_grad_norm_", "clip")
wrap(BatchedMuon, "step", "muon")
for n in ("ema_step_card", "ema_step"):
    if hasattr(T, n):
        wrap(T, n, "ema")

try:
    T.run_train(
        dataclasses.replace(req, output="output/ckpt_moss/int8_phase.safetensors")
    )
finally:
    print("phase ms median steps 11+:")
    tot = 0
    for k, v in marks.items():
        m = statistics.median(v[10:]) if len(v) > 10 else float("nan")
        tot += m
        print(f"  {k:18}{m:8.1f}  n={len(v)}")
    print("  sum", round(tot, 1))
