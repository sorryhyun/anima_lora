#!/usr/bin/env python
"""run_stage — run one of the line's vendored stages (``stages.py``) by hand.

The entry the scene pools grow through (README § Scene pools), and a way to
re-run a ruler on an arm dir outside ``scale.py <run> eval``:

    make daemon-run ARGS="--label scenes-s1 --stall-timeout 0 \\
        project/cjk_anima_scale/src/run_stage.py --stage scenes --scene_tag s1 …"
    .venv/bin/python project/cjk_anima_scale/src/run_stage.py --stage eval \\
        --data_path output/cjk_anima_scale/<run>/data --arm_path output/cjk_anima_scale/<run>/ctx …

Outputs land under ``output/cjk_anima_scale/`` (``common.paths.OUT``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli import build_parser  # noqa: E402
from common.paths import OUT  # noqa: E402
from stages import STAGES, run  # noqa: E402


def main():
    a = build_parser(STAGES, __doc__).parse_args()
    assert "all" not in a.stage, f"name the stages: {', '.join(STAGES)}"
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.stage:
        print(f"===== stage {s} ({a.arm})", flush=True)
        run(s, a)


if __name__ == "__main__":
    main()
