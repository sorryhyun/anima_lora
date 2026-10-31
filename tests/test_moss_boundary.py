"""``library.moss`` stays a line of its own.

Anima code never imports it, and it reaches into the rest of ``library`` only
through the list below. ``requests`` must stay torch-free: the GUI builds its
forms from it at launch.
"""

from __future__ import annotations

import subprocess
import sys

from tests._boundary import ROOT, imports_of, python_files

MOSS = ROOT / "library" / "moss"

# Where importing library.moss is allowed.
_MOSS_HOMES = (
    "library/moss/",
    "scripts/moss/",
    "scripts/calibration/moss_",
    "scripts/tasks/training.py",  # the i2i cache job, from torch-free requests
    "gui/moss/",
    "bench/moss_",
    "bench/_moss.py",
    "tests/",
)
# What library.moss may import from the rest of library.
_SHARED = (
    "library.env",
    "library.downloads",
    "library.lines",
    "library.runtime.offloading",
    "library.runtime.blockswap",
    "library.runtime.device",
    "library.runtime.dynamo",
    "library.inference.sampling",
    "library.inference.corrections.cns_core",
    "library.training.progress",
    "library.training.repa",
    "library.vision.buckets",
    "library.datasets.curation_actions",
)


def test_anima_side_never_imports_moss():
    offenders = []
    for path in python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(_MOSS_HOMES):
            continue
        if any(n.startswith("library.moss") for n in imports_of(path)):
            offenders.append(rel)
    assert not offenders, f"Anima code imports library.moss: {offenders}"


def test_moss_uses_only_shared_library_helpers():
    offenders = []
    for path in MOSS.rglob("*.py"):
        for name in imports_of(path):
            if name.startswith("networks"):
                offenders.append(f"{path.name}: {name}")
            if not name.startswith("library.") or name.startswith("library.moss"):
                continue
            if not name.startswith(_SHARED):
                offenders.append(f"{path.name}: {name}")
    assert not offenders, f"library.moss reaches past the shared helpers: {offenders}"


def test_requests_is_torch_free():
    code = "import sys, library.moss.requests; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "library.moss.requests imports torch"
