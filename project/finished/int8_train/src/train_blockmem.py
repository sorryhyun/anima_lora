"""train.py with per-block memory on step 1 and step times; ANIMA_BM_SMALL=1 keeps only
ama mitsuki's ~3000-token samples, so budget 1.0 fits on 16 GB. Run through the daemon:
reports/anima_wqt_stream.md."""

import atexit
import glob
import os
import re
import runpy
import statistics
import sys
import time

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402

import library.anima.models as m  # noqa: E402
import library.datasets.base as base  # noqa: E402

if os.environ.get("ANIMA_BM_SMALL") == "1":
    small = set()
    for p in glob.glob("post_image_dataset/lora/ama_mitsuki/*_anima.npz"):
        hit = re.search(r"/([^/]+)_(\d{4})x(\d{4})_anima\.npz$", p)
        if (int(hit[2]) // 16) * (int(hit[3]) // 16) < 3400:
            txt = f"post_image_dataset/resized/ama_mitsuki/{hit[1]}.txt"
            small.add(open(txt, encoding="utf-8").read().strip())
    print(f"blockmem small captions {len(small)}", flush=True)
    base._caption_has_artist = lambda c, n: bool(c) and c.strip() in small

orig = m.Block.forward
seen = [0]
stamps = []
sizes = []


def forward(self, x, *a, **k):
    if seen[0] % 28 == 0:
        stamps.append(time.perf_counter())
        sizes.append(x.shape[2])
    out = orig(self, x, *a, **k)
    if seen[0] < 28:
        gb = torch.cuda.memory_allocated() / 1024**3
        print(f"blockmem {seen[0]} {x.shape[2]} {gb:.3f}", flush=True)
    seen[0] += 1
    return out


def report():
    dt = [b - a for a, b in zip(stamps, stamps[1:])][10:]
    if dt:
        print(
            f"blockmem steps {len(dt)} median {statistics.median(dt) * 1000:.1f} ms "
            f"mean {statistics.mean(dt) * 1000:.1f} ms "
            f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
            f"tokens {min(sizes)}..{max(sizes)}",
            flush=True,
        )


atexit.register(report)
m.Block.forward = forward
sys.argv = ["train.py", *sys.argv[1:]]
runpy.run_path("train.py", run_name="__main__")
