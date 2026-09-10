#!/usr/bin/env python3
"""Launch ``<python> <script> [args…]`` under any interpreter and sample peak VRAM.

Two jobs in one, both forced by the daemon contract:

* The daemon always launches command jobs with the repo ``.venv`` python, so
  an A/B against a scratch venv needs a shim that re-launches under the other
  interpreter — argv[1] is the interpreter, the rest is passed verbatim.
* ``train.py`` logs no peak-VRAM number, so the shim samples
  ``nvidia-smi memory.used`` (whole-GPU, includes the idle baseline) every 2 s
  and prints ``VRAM_SHIM peak_used_mib=<n>`` on exit for ``run_bench.py``.

Leading ``--pin KEY=VALUE`` options (before the interpreter) set
``torch._inductor.config`` entries via the repo's ``pin_inductor_flag`` — the
helper that holds in the backward-compile context too — then run the script
through ``runpy`` so ``train.py`` sees the pin before its first compile. Use it
for knobs with no env var (``combo_kernels``).

Exit code is the child's. SIGTERM/SIGINT are forwarded so a daemon kill still
tears the child down.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading


def _sample(peak: list[int], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            out = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                .stdout.strip()
                .splitlines()[0]
            )
            peak[0] = max(peak[0], int(out))
        except Exception:  # noqa: BLE001 — sampling is best-effort
            pass
        stop.wait(2.0)


def _coerce(raw: str) -> object:
    low = raw.lower()
    if low in ("1", "true"):
        return True
    if low in ("0", "false"):
        return False
    try:
        return int(raw)
    except ValueError:
        return raw


def _pin_bootstrap(pins: list[str]) -> str:
    """Source for ``python -c``: pin each flag, then run ``sys.argv[1]`` as __main__."""
    items = [(k, _coerce(v)) for k, v in (p.split("=", 1) for p in pins)]
    return (
        "import runpy, sys; sys.path.insert(0, '.'); "
        "from library.runtime.dynamo import pin_inductor_flag as pin; "
        + "".join(f"pin({k!r}, {v!r}); " for k, v in items)
        + "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )


def main() -> None:
    argv = sys.argv[1:]
    pins: list[str] = []
    while len(argv) >= 2 and argv[0] == "--pin":
        pins.append(argv[1])
        argv = argv[2:]
    if len(argv) < 2:
        sys.exit("usage: vram_shim.py [--pin K=V …] <python> <script> [args…]")
    if pins:
        argv = [argv[0], "-c", _pin_bootstrap(pins), *argv[1:]]
    peak, stop = [0], threading.Event()
    thread = threading.Thread(target=_sample, args=(peak, stop), daemon=True)
    thread.start()
    child = subprocess.Popen(argv)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _f: child.send_signal(s))
    rc = child.wait()
    stop.set()
    thread.join(timeout=3)
    print(f"VRAM_SHIM peak_used_mib={peak[0]}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
