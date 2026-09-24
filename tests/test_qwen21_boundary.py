"""``library.qwen21`` stays a detachable island.

Anima code never imports it, and it reaches into the rest of ``library`` only
for the shared helpers — so retiring the line is deleting its directories.
``requests`` must stay torch-free: the GUI builds its forms from it at launch.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    "library.runtime.offloading",
    "library.runtime.device",
    "library.runtime.dynamo",
)
_SCANNED = ("anima_lora", "library", "networks", "scripts", "gui", "bench", "train.py")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _python_files():
    for entry in _SCANNED:
        path = ROOT / entry
        yield from [path] if path.is_file() else path.rglob("*.py")


def test_anima_side_never_imports_qwen21():
    offenders = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(_QWEN21_HOMES):
            continue
        if any(n.startswith("library.qwen21") for n in _imports(path)):
            offenders.append(rel)
    assert not offenders, f"Anima code imports library.qwen21: {offenders}"


def test_qwen21_uses_only_shared_library_helpers():
    offenders = []
    for path in QWEN21.rglob("*.py"):
        for name in _imports(path):
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
