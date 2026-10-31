"""Import scanning shared by the line boundary tests."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("anima_lora", "library", "networks", "scripts", "gui", "bench", "train.py")


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def python_files() -> Iterator[Path]:
    for entry in SCANNED:
        path = ROOT / entry
        yield from [path] if path.is_file() else path.rglob("*.py")
