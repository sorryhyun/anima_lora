"""Every GPU entry point installs the Windows no-console-window Popen default.

Without it, torch.compile's per-kernel ptxas / cl launches each flash a console
window on Windows (dozens at the start of a run).
"""

from pathlib import Path

import pytest

from library.env import resolve_under_home

ENTRY_POINTS = [
    "train.py",
    "inference.py",
    "scripts/moss/train.py",
    "scripts/moss/generate.py",
    "scripts/moss/cache.py",
    "scripts/moss/tune_swap.py",
    "scripts/moss/distill_cache.py",
    "scripts/moss/colorize_cache.py",
    "scripts/moss/captions.py",
    "scripts/qwen21/train.py",
    "scripts/qwen21/generate.py",
    "scripts/qwen21/cache.py",
    "scripts/distill_turbo/distill.py",
    "scripts/distill_turbo/anchor_cache.py",
]


@pytest.mark.parametrize("path", ENTRY_POINTS)
def test_entry_point_installs_no_window(path: str) -> None:
    src = Path(resolve_under_home(path)).read_text(encoding="utf-8")
    assert "install_no_window_default()" in src or "@no_window_main" in src
