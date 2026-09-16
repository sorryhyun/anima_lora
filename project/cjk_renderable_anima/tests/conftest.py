"""Line-local tests for ``src/`` — run only as
``.venv/bin/python -m pytest project/cjk_renderable_anima/tests`` (never part of
the repo suite). ``src/`` goes on ``sys.path`` first, as ``wake_probe.py`` has it."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
