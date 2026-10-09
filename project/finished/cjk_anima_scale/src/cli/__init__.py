"""cli — the stage command line: flags grouped by the stage that reads them.

Vendored into ``project/cjk_anima_scale/src/`` (2026-09-25) and trimmed to the
groups the line's stages read (``eval`` / ``native`` / ``target`` /
``cf_sense`` / ``scenes``): the data / train / rows / encoder groups are gone
with the stages that read them, and (2026-09-25) so are the kept stages' dead
levers (``--out_vec*``, ``--delta_parts``, ``--native_floor``, ``--kept_*``,
``--with_c_flat``, ``--cls_*``, ``--salad_size``, the tag layout's
``--data_tag`` / ``--arm_tag``). Every kept flag keeps its name, dest and
default — the defaults are part of the ruler.

One module per reader: ``run`` (run + generation), ``eval`` (eval / native +
the ``cf_*`` flags ``cf_sense`` reads), ``scenes``.
"""

from __future__ import annotations

import argparse

from .eval import cf_args, eval_args
from .run import generation_args, run_args
from .scenes import scene_args


def build_parser(stages, description: str | None = None) -> argparse.ArgumentParser:
    """``stages``: the stage names ``--stage`` accepts."""
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    run_args(p.add_argument_group("run"), stages)
    generation_args(p.add_argument_group("generation"))
    eval_args(p.add_argument_group("eval / native"))
    cf_args(p.add_argument_group("cf_sense"))
    scene_args(p.add_argument_group("scenes"))
    return p
