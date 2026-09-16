"""The wake_probe argparse surface as JSON — the golden fixture's generator.

    .venv/bin/python project/cjk_renderable_anima/tests/cli_dump.py > \
        project/cjk_renderable_anima/tests/fixtures/cli_golden.json

Regenerate only for an intended flag change; a refactor must leave it equal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def dump() -> list[dict]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from cli import build_parser
    from common.paths import REPO
    from stages import STAGES

    p = build_parser(STAGES)
    out = []
    for g in p._action_groups:
        for act in g._group_actions:
            row = {
                "group": g.title,
                "option_strings": act.option_strings,
                "dest": act.dest,
                "default": act.default,
                "choices": act.choices,
                "nargs": act.nargs,
                "type": getattr(act.type, "__name__", repr(act.type)),
                "help": act.help,
                "action": type(act).__name__,
                "metavar": act.metavar,
                "required": act.required,
            }
            # the native prompt default is an absolute path under the checkout
            text = json.dumps(row, ensure_ascii=False, default=str)
            out.append(json.loads(text.replace(str(REPO), "<REPO>")))
    return out


if __name__ == "__main__":
    json.dump(dump(), sys.stdout, ensure_ascii=False, indent=1)
    print()
