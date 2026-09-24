"""Line-local tests for ``src/`` — run only as
``.venv/bin/python -m pytest project/cjk_renderable_anima/tests`` (never part of
the repo suite). ``src/`` goes on ``sys.path`` first, as ``wake_probe.py`` has it."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_runtest_setup(item):
    """Keep ``src/`` first even when another test tree has since pushed the repo
    root ahead of it (``tests/`` modules that load a script which does
    ``sys.path.insert(0, REPO)`` at import) — otherwise ``import train`` here
    resolves to the repo-root ``train.py``."""
    if sys.path and sys.path[0] != str(SRC):
        if str(SRC) in sys.path:
            sys.path.remove(str(SRC))
        sys.path.insert(0, str(SRC))
