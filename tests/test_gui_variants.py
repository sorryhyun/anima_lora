"""Tests for the Track 2 GUI variant registry (``configs/gui-methods/*.toml``).

Built-in variants are discovered from each file's ``[variant]`` metadata
table (see ``gui/__init__.py``). This module pins:

* every built-in is loadable via ``load_method_preset(..., methods_subdir="gui-methods")``
* every built-in carries a ``[variant].family`` string
* output_name is unique across built-ins (the allowlist below covers
  families that may again ship same-model variants; hardware now composes
  via the presets.toml Hardware picker instead of per-variant file copies).

Customs under ``configs/gui-methods/custom/`` are intentionally exempt —
users name and structure those freely.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import toml

from library.config.io import load_method_preset

REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_METHODS_DIR = REPO_ROOT / "configs" / "gui-methods"

# Built-in variants in the same family may legitimately share an output_name
# when they're same-model variants under different length/quality budgets
# (hardware variants no longer exist as file copies — the GUI composes the
# presets.toml Hardware preset instead). Listed as (family, output_name) pairs.
_INTENTIONAL_OUTPUT_NAME_COLLISIONS: set[tuple[str, str]] = {
    ("lora", "anima"),
    ("tlora", "anima_tlora_ortho"),
    ("hydralora", "anima_hydra"),
}


def _builtin_files() -> list[Path]:
    """Direct children of gui-methods/ (skipping the ``custom/`` subdir)."""
    return sorted(p for p in GUI_METHODS_DIR.glob("*.toml") if p.is_file())


BUILTINS = _builtin_files()
assert BUILTINS, "no built-in gui-methods/*.toml files found"


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_builtin_has_variant_metadata(path: Path):
    """Every built-in must declare ``[variant].family``."""
    data = toml.loads(path.read_text(encoding="utf-8"))
    meta = data.get("variant")
    assert isinstance(meta, dict), f"{path.name}: missing [variant] table"
    family = meta.get("family")
    assert isinstance(family, str) and family, (
        f"{path.name}: [variant].family must be a non-empty string"
    )


@pytest.mark.core
@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_builtin_loads_clean(path: Path, caplog):
    """Every built-in must merge cleanly through the base → preset → method
    chain — no schema warnings, no missing files."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        merged = load_method_preset(path.stem, "default", methods_subdir="gui-methods")
    # Required keys must show up downstream — guarantees the merge didn't
    # silently drop everything.
    assert "network_module" in merged
    assert "output_name" in merged
    offenders = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and rec.name.startswith("library.")
    ]
    assert not offenders, f"{path.name} produced warnings: {offenders}"


def test_no_unintentional_output_name_collisions():
    """Built-in variants must each pick a unique ``output_name`` unless the
    (family, name) pair is in ``_INTENTIONAL_OUTPUT_NAME_COLLISIONS`` — i.e.
    multiple files baking the same adapter on purpose.
    """
    by_output: dict[str, list[tuple[str, str]]] = {}
    for path in BUILTINS:
        data = toml.loads(path.read_text(encoding="utf-8"))
        output_name = data.get("output_name")
        family = (data.get("variant") or {}).get("family", "?")
        if not isinstance(output_name, str):
            continue
        by_output.setdefault(output_name, []).append((family, path.stem))

    bad: list[str] = []
    for output_name, owners in by_output.items():
        if len(owners) == 1:
            continue
        families = {f for f, _ in owners}
        if (
            len(families) == 1
            and (next(iter(families)), output_name)
            in _INTENTIONAL_OUTPUT_NAME_COLLISIONS
        ):
            continue
        owner_str = ", ".join(f"{stem} (family={family})" for family, stem in owners)
        bad.append(f"output_name={output_name!r} shared by: {owner_str}")
    assert not bad, "Output name collisions:\n  " + "\n  ".join(bad)
