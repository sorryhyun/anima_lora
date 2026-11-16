"""Preference-RL pick rounds on the turbo student (project/anima_turbo/pref_rl/proposal.md)."""

from __future__ import annotations

from scripts.tasks._common import PY, ROOT, run


def cmd_turbo_pref(_extra):
    run([PY, str(ROOT / "project/anima_turbo/pref_rl/app.py")])
