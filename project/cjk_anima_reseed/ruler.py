"""The dialogue ruler's CLI (``criteria.md``); the code is ``src/eval/ruler/``.

    .venv/bin/python project/cjk_anima_reseed/ruler.py build
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run --arms kana_up,ball_rk_bubble,stick_nlg_high --label arms"

``build`` (CPU) draws the string set; ``render`` (GPU) draws what is missing;
``read`` scores every render — text: official (both readers exact), exact
(either), contained, ≤ 1 / ≤ 2 edits, CER, dup (a doubled glyph the string
lacks, or a read longer than it), a bubble's boxes also joined in column and
line order; glyphs: ``g_p`` / ``g_r`` / ``g_f1`` and the rest; page:
``en_cls`` / ``en_match`` (``AtSim``), EN-ref token cos outside every text
box of both images (``en_tok_out``), flat-white share over the EN ref's —
and pairs each arm against the floor arms → ``results/<ts>-ruler-<label>/``.
``run`` = both. ``sample`` draws random strings of a read into larger sheets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reseed import REPO, bootstrap  # noqa: E402

bootstrap()

from eval.ruler import EN, FLOOR, MODES, VIEW  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=["build", "render", "read", "run", "sample"])
    p.add_argument(
        "--arms",
        default=",".join(FLOOR),
        help="comma list; `en` = the EN references (default: the floor)",
    )
    p.add_argument("--label", default="floor")
    p.add_argument("--prompts", choices=MODES, default=MODES[0])
    p.add_argument("--pack", default="", help="render on this base pack (reseed PACKS)")
    p.add_argument(
        "--marks", action="store_true", help="only the strings holding a mark"
    )
    p.add_argument(
        "--only", default="", help="render: these ruler indices only (comma list)"
    )
    p.add_argument(
        "--rows_pt",
        default="",
        help="name=path,…: a probe's rows.pt on stick080's rows as arm `name`",
    )
    p.add_argument("--from", dest="src", help="sample: a read's results dir")
    p.add_argument("--n", type=int, default=12, help="sample: strings drawn")
    p.add_argument("--seed", type=int, default=0, help="sample: the draw's seed")
    p.add_argument(
        "--has", default="", help="sample: only strings holding one of these chars"
    )
    a = p.parse_args()
    if a.verb == "build":
        from eval.ruler.build import build

        build()
        return
    if a.verb == "sample":
        from eval.ruler.read import sample

        sample(Path(a.src), a.n, a.seed, a.has)
        return
    VIEW.mode, VIEW.pack, VIEW.marks_only = a.prompts, a.pack, a.marks
    VIEW.only = {int(x) for x in a.only.split(",") if x}
    assert not (VIEW.only and a.verb in ("read", "run")), "--only is a look: render"
    if VIEW.pack:
        from reseed.config import PACKS

        os.environ["ANIMA_VOCAB_PACK"] = PACKS[VIEW.pack]
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed
    from eval.ruler.arms import DERIVED, PACK_ARMS, arm_dirs, rows_pt

    for spec in filter(None, a.rows_pt.split(",")):
        k, v = spec.split("=", 1)
        assert k not in DERIVED and k not in arm_dirs(), f"arm {k} exists"
        DERIVED[k] = lambda v=v: rows_pt(
            REPO / v if not Path(v).is_absolute() else Path(v)
        )
    names = [x for x in a.arms.split(",") if x]
    plain = {EN, *arm_dirs(), *DERIVED}
    pack = VIEW.pack
    known = plain | ({f"{x}@{pack}" for x in plain - {EN}} if pack else set())
    assert set(names) <= known, f"unknown arms {set(names) - known}"
    if a.verb in ("render", "run"):
        from eval.ruler.render import render

        # on a pack, render only its own arms: the raw pack's are on disk
        bad = [x for x in names if pack and "@" not in x and x not in PACK_ARMS[pack]]
        assert not bad, f"--pack {pack} renders X@{pack} or its own arms, not {bad}"
        render(names)
    if a.verb in ("read", "run"):
        from eval.ruler.read import read

        # the floor arms always ride along: every read pairs against them
        rd = [x for x in dict.fromkeys([*FLOOR[1:], *names]) if x != EN]
        read(rd, a.label, __file__)


if __name__ == "__main__":
    main()
