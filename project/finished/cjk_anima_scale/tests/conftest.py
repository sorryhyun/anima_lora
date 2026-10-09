"""Line-local tests — run as
``.venv/bin/python -m pytest project/cjk_anima_scale/tests`` (never part of
the repo suite). The line dir and its own ``src/`` go on ``sys.path`` the
way ``scale.py`` puts them."""

import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]
if str(LINE) not in sys.path:
    sys.path.insert(0, str(LINE))

from cjk_scale.paths import bootstrap  # noqa: E402

bootstrap()
