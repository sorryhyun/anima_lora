"""plan_render — activation_memory_budget VRAM probe for a render EasyControl arm.

Runs the exact argv ``make easycontrol EASYADAPTER=<adapter>`` would build
(descriptor blueprint + [training] overrides), but for a fixed number of
steps with ``--activation_memory_budget B`` overridden, and samples
``nvidia-smi`` memory.used every second while the child runs. Prints one
summary line at the end (baseline / peak / delta MiB, rc, wall). Submit
through the daemon::

    make daemon-run ARGS="--stall-timeout 900 \
        project/cjk_aware_anima_dit/render/vram_probe.py --budget 0.99 --queue"

``--dry-run`` prints the child argv and exits (CPU only).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.tasks._common import PY, _preset, build_method_args  # noqa: E402
from scripts.tasks.training import _easy_train_extra  # noqa: E402

_EPOCH_FLAGS = {"--max_train_epochs", "--save_every_n_epochs", "--checkpointing_epochs"}


def _strip(argv: list[str], flags: set[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in flags:
            skip = True
            continue
        out.append(tok)
    return out


def _mem_used_mib() -> int | None:
    try:
        txt = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return int(txt.strip().splitlines()[0])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="render_en")
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tag = f"amb{a.budget:.2f}".replace(".", "")
    out_dir = ROOT / "output" / "render" / "vram_probe"
    extra = _strip(_easy_train_extra(a.adapter, []), _EPOCH_FLAGS)
    extra += [
        "--max_train_steps",
        str(a.steps),
        "--activation_memory_budget",
        str(a.budget),
        "--output_name",
        f"probe_{a.adapter}_{tag}",
        "--output_dir",
        str(out_dir),
    ]
    argv = [
        PY,
        "train.py",
        *build_method_args("easycontrol", preset=_preset(), extra=extra),
    ]
    print("[vram_probe] argv:", " ".join(argv), flush=True)
    if a.dry_run:
        return 0

    baseline = _mem_used_mib()
    peak = {"v": baseline or 0}
    stop = threading.Event()

    def sampler() -> None:
        last_print = 0.0
        while not stop.is_set():
            m = _mem_used_mib()
            if m is not None and m > peak["v"]:
                peak["v"] = m
            now = time.monotonic()
            if now - last_print >= 30:
                print(f"[vram_probe] mem.used now={m} peak={peak['v']} MiB", flush=True)
                last_print = now
            stop.wait(1.0)

    t = threading.Thread(target=sampler, daemon=True)
    t0 = time.monotonic()
    t.start()
    rc = subprocess.call(argv, cwd=ROOT)
    stop.set()
    t.join(timeout=5)
    wall = time.monotonic() - t0
    delta = None if baseline is None else peak["v"] - baseline
    print(
        f"[vram_probe] adapter={a.adapter} budget={a.budget} steps={a.steps} "
        f"baseline={baseline} peak={peak['v']} delta={delta} MiB rc={rc} wall={wall:.0f}s",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
