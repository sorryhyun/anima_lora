"""GUI launch-speed regression guards.

The GUI process must stay light (gui/CLAUDE.md): a single careless import of a
torch/cv2-backed ``library`` module adds ~2.4s warm — and 10-30s after a cold
boot — to every launch (this regressed once via ``library.datasets.subsets``).

Two layers, both in a fresh interpreter (an in-process check would be polluted
by other tests that legitimately import torch):

* ``test_gui_app_import_stays_torch_free`` — the root-cause guard. Fails the
  moment a heavy import sneaks back into the ``gui.app`` chain, regardless of
  how fast the machine is.
* ``test_gui_launch_under_budget`` — end-to-end wall clock: import ``gui.app``,
  build and show ``MainWindow`` offscreen. Budget is generous vs. the ~1.35s
  measured warm launch (Config/Preprocess eager, everything else behind
  ``LazyTabHolder``) so a loaded machine doesn't flake it, while a
  torch-sized regression — or eager-building the lazy tabs again — still
  trips.

  Imports are NOT the usual suspect when this trips: they are ~0.18s of the
  total, and construction is the rest. Split it before profiling imports::

      t0 → import gui.app → QApplication() → MainWindow() → show()

  Known construction costs, in case a future regression needs a baseline
  (measured 2026-08-17, warm, offscreen):

  * ``PreprocessingTab._refresh_status`` ~0.39s — stats the whole dataset
    (``_filtered_files`` + ``count_preprocess_caches`` + ``_count_resized``,
    ~26k ``stat`` calls). Scales with dataset size, so a big corpus makes this
    the dominant term.
  * ~0.45s of genuine Qt widget-tree construction (``addWidget``).

  Started in MOSS mode (``model_mode`` in ``gui_settings.json``) the MOSS
  panel is built too, ~0.9s of widgets (2026-11-13); its cache scan runs after
  show. The TensorBoard run list (one row per run, ~0.33s for 828) is built on
  the panel's first show.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Warm launch is ~1.35s; the headroom absorbs a loaded CI box and a dataset /
# run-log tree larger than this checkout's (both scans above scale with it).
LAUNCH_BUDGET_S = 2.5

# Heavyweight modules that must never load in the GUI process.
_FORBIDDEN = ("torch", "cv2")


def _run_in_fresh_interpreter(code: str, **env_extra: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, **env_extra},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"child interpreter failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout


@pytest.mark.core
def test_gui_app_import_stays_torch_free():
    out = _run_in_fresh_interpreter(
        "import sys\n"
        "import gui.app  # noqa: F401\n"
        f"leaked = [m for m in {_FORBIDDEN!r} if m in sys.modules]\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    line = next(ln for ln in out.splitlines() if ln.startswith("LEAKED="))
    leaked = line.removeprefix("LEAKED=")
    assert not leaked, (
        f"importing gui.app pulled in {leaked} — a heavy module re-entered the "
        "GUI import chain. Find it with: python -X importtime -c 'import gui.app'"
    )


def test_gui_launch_under_budget():
    out = _run_in_fresh_interpreter(
        "import time\n"
        "t0 = time.perf_counter()\n"
        "import sys\n"
        "from PySide6.QtWidgets import QApplication\n"
        "import gui.app as ga\n"
        "app = QApplication(sys.argv)\n"
        "ga._dark(app)\n"
        "win = ga.MainWindow()\n"
        "win.show()\n"
        "print(f'ELAPSED={time.perf_counter() - t0:.3f}')\n",
        QT_QPA_PLATFORM="offscreen",
    )
    line = next(ln for ln in out.splitlines() if ln.startswith("ELAPSED="))
    elapsed = float(line.removeprefix("ELAPSED="))
    assert elapsed < LAUNCH_BUDGET_S, (
        f"GUI launch (import + MainWindow build) took {elapsed:.2f}s, "
        f"budget is {LAUNCH_BUDGET_S}s. Profile imports with "
        "`python -X importtime -c 'import gui.app'` and window construction "
        "with cProfile around MainWindow()."
    )
