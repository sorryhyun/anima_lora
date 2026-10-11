#!/usr/bin/env python
"""reseed front door: ``run.py <run> data | train``.

    .venv/bin/python project/cjk_anima_reseed/run.py <run> data              # CPU
    .venv/bin/python project/cjk_anima_reseed/run.py <run> data --frac 0.02  # a look at the sizes
    make daemon-run ARGS="project/cjk_anima_reseed/run.py <run> train"
    # a smoke: another data dir / out dir, a short schedule, an early stop
    make daemon-run ARGS="project/cjk_anima_reseed/run.py <run> train --data <dir> --out <dir> --steps_per_row 1 --max_steps 400"

``<run>`` is ``configs/<run>.toml`` or a path to one.
``data`` → ``output/cjk_anima_reseed/<run>/data``; ``train`` →
``…/<run>/trained.pt`` (the whole merged rows, ``reseed.trainer``: the
run's rows trained cold or warm as its config says, every other row frozen
at the run's context rows). A run is read on the dialogue ruler
(``ruler.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reseed import bootstrap  # noqa: E402

bootstrap()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("verb", choices=["data", "train"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--frac", type=float, default=1.0, help="data: share of every tier")
    p.add_argument("--data", default=None, help="train: this data dir (a smoke)")
    p.add_argument("--out", default=None, help="train: this out dir (a smoke)")
    p.add_argument(
        "--steps_per_row", type=int, default=None, help="train: the config's replaced"
    )
    p.add_argument("--max_steps", type=int, default=None, help="train: stop here")
    a = p.parse_args()
    from reseed.config import load

    run = load(a.run)
    run.use_pack()
    if a.verb == "data":
        assert not run.data_from, f"{run.name}: trains on {run.data}"
        from reseed.builder import build

        build(run, a.workers, a.frac)
    else:
        assert a.frac == 1.0, "--frac is the data verb's"
        from reseed import trainer as T

        T.train(
            run,
            data=Path(a.data) if a.data else run.data,
            out=Path(a.out) if a.out else run.dir,
            context=run.seed_rows(),
            cold=not run.rows_from,
            max_steps=a.max_steps,
            steps_per_row=None if run.focus else a.steps_per_row or run.steps_per_row,
            steps=run.steps(a.steps_per_row),
            lr=run.lr or None,
            row_step_scale=run.row_step_scale(),
            free_residual=run.free_residual,
            pres=run.pres,
            factor=run.factor,
            factor_init=run.factor_init_rows(),
        )


if __name__ == "__main__":
    main()
