"""cli — the wake_probe command line: flags grouped by the stage *and arm* that reads them.

The groups are documentation only — argparse group membership has no runtime
effect, every flag keeps its name, dest, default and help verbatim. The 09-16
regroup moved flags to the reader that actually consults them, because the old
"train: encoder arm" group held the rows arm's ``--init_rows`` / ``--pin_*``
(which every S-line run passes) and the S-line *data* group held train flags.

One module per reader: ``run`` (run + generation), ``data``, ``train``,
``eval`` (eval / native + classify), ``scenes``.
"""

from __future__ import annotations

import argparse

from .data import data_args, data_synth_args
from .eval import classify_args, eval_args
from .run import generation_args, run_args
from .scenes import scene_args
from .train import encoder_args, rows_args, train_args, train_synth_args


def build_parser(stages, description: str | None = None) -> argparse.ArgumentParser:
    """``stages``: the stage names ``--stage`` accepts (besides ``all``)."""
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    run_args(p.add_argument_group("run"), stages)
    generation_args(p.add_argument_group("generation"))
    data_args(p.add_argument_group("data"))
    data_synth_args(p.add_argument_group("data: S line (plan_synth)"))
    train_args(p.add_argument_group("train"))
    train_synth_args(p.add_argument_group("train: S line (plan_synth)"))
    rows_args(p.add_argument_group("train: rows arm"))
    encoder_args(p.add_argument_group("train: encoder arm"))
    eval_args(p.add_argument_group("eval / native"))
    classify_args(p.add_argument_group("classify / classify_str"))
    scene_args(p.add_argument_group("scenes"))
    return p
