"""Curation → trainer dependency-direction guard.

The curation half of the tree (tagger, masking, grouping, caption polishing) is
split into ``sorryhyun/anime_tools`` (contract: ``../anime_tools/docs/contract.md``).
The one rule of that split is::

    anima_lora (trainer) ──depends on──▶ anime_tools        never the reverse

This test pins the boundary so it cannot regress: every trainer-side file that
drives the curation package must not import the DiT/VAE/adapter side. If a curation
feature needs something from the trainer, move that leaf into the curation set
(or duplicate a ≤50-line pure helper) — do not add an exemption here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Move manifest (proposal §"Move manifest"). Directories are recursive; globs
# are relative to the repo root.
CURATION_PATHS: tuple[str, ...] = (
    # Phases 1–2 (2026-08-30) moved captions / tagger / stages / masking /
    # grouping into the ``anime_tools`` package and Phase 3 deleted the
    # forwarding shims + shells. What remains here is the trainer-side
    # remainder that must keep the direction: the ``make`` task wrappers that
    # invoke ``python -m anime_tools.…``. ``anime_tools`` imports are allowed —
    # that is the dependency direction. The package's own boundary is guarded
    # by its ``tests/test_boundary.py``.
    "scripts/tasks/tagger.py",
    "scripts/tasks/masking.py",
    "scripts/tasks/curate.py",
    # The SAM mask stage's legacy-prompts → ``masks`` translation (anime_tools
    # 0.6.4), shared by ``masking.py`` and the GUI's rule cards; imports only
    # ``anime_tools``.
    "library/config/sam_masks.py",
)

# Anything under ``library.`` that is not itself in the manifest is trainer-only
# (``library.env`` / ``library.log`` / ``library.io`` / ``library.datasets`` /
# ``library.runtime`` included — the curation side uses ``anime_tools.{_env,_walk,_hf}``),
# plus ``networks`` and ``train``. Curation code imports ``anime_tools`` directly —
# the Phase 1–2 forwarding shims (``library.captioning.*`` …) are gone.
FORBIDDEN_ROOTS: tuple[str, ...] = ("library", "networks", "train")
ALLOWED_MODULES: frozenset[str] = frozenset()


_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?P<from>[\w.]+)\s+import|import\s+(?P<mod>[\w.]+))",
    re.MULTILINE,
)


def _curation_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in CURATION_PATHS:
        files.update(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def _in_manifest(module: str, manifest: set[str]) -> bool:
    rel = module.replace(".", "/")
    return rel + ".py" in manifest or any(m.startswith(rel + "/") for m in manifest)


def _forbidden(module: str, manifest: set[str]) -> bool:
    if module in ALLOWED_MODULES:
        return False
    if not any(module == r or module.startswith(r + ".") for r in FORBIDDEN_ROOTS):
        return False
    return not _in_manifest(module, manifest)


def _violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    manifest = {str(p.relative_to(REPO_ROOT)) for p in _curation_files()}
    out: list[str] = []
    for m in _IMPORT_RE.finditer(text):
        mod = m.group("from") or m.group("mod")
        if _forbidden(mod, manifest):
            line = text.count("\n", 0, m.start()) + 1
            out.append(f"{path.relative_to(REPO_ROOT)}:{line}: {m.group(0).strip()}")
    return out


def test_manifest_globs_resolve():
    """A renamed/moved file must update the manifest, not silently drop out."""
    for pattern in CURATION_PATHS:
        assert list(REPO_ROOT.glob(pattern)), (
            f"manifest entry matches nothing: {pattern}"
        )


@pytest.mark.parametrize(
    "path", _curation_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_curation_module_does_not_import_trainer(path: Path):
    violations = _violations(path)
    assert not violations, (
        "curation-side module imports the trainer (dependency direction is "
        "trainer → anime_tools, never the reverse):\n  " + "\n  ".join(violations)
    )
