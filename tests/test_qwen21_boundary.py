"""``library.qwen21`` stays a detachable island.

Anima code never imports it, and it reaches into the rest of ``library`` only
for the shared helpers — so retiring the line is deleting its directories.
``requests`` must stay torch-free: the GUI builds its forms from it at launch.
"""

from __future__ import annotations

import subprocess
import sys

from tests._boundary import ROOT, imports_of, python_files

QWEN21 = ROOT / "library" / "qwen21"

# Where importing library.qwen21 is allowed.
_QWEN21_HOMES = (
    "library/qwen21/",
    "scripts/qwen21/",
    "gui/qwen21/",
    "project/qwen21_lora/",
    "tests/",
)
# What library.qwen21 may import from the rest of library.
_SHARED = (
    "library.env",
    "library.lines",
    "library.runtime.offloading",
    "library.runtime.blockswap",
    "library.runtime.device",
    "library.runtime.dynamo",
)


def test_anima_side_never_imports_qwen21():
    offenders = []
    for path in python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(_QWEN21_HOMES):
            continue
        if any(n.startswith("library.qwen21") for n in imports_of(path)):
            offenders.append(rel)
    assert not offenders, f"Anima code imports library.qwen21: {offenders}"


def test_qwen21_uses_only_shared_library_helpers():
    offenders = []
    for path in QWEN21.rglob("*.py"):
        for name in imports_of(path):
            if not name.startswith("library.") or name.startswith("library.qwen21"):
                continue
            if not name.startswith(_SHARED):
                offenders.append(f"{path.name}: {name}")
    assert not offenders, f"library.qwen21 reaches into Anima: {offenders}"


def test_requests_is_torch_free():
    code = "import sys, library.qwen21.requests; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "library.qwen21.requests imports torch"


def test_gui_is_torch_free():
    code = "import sys, gui.qwen21.app; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "gui.qwen21.app imports torch"
