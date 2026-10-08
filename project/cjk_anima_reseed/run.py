#!/usr/bin/env python
"""reseed front door: ``run.py <run> data | train | read``.

    .venv/bin/python project/cjk_anima_reseed/run.py <run> data              # CPU
    .venv/bin/python project/cjk_anima_reseed/run.py <run> data --frac 0.02  # a look at the sizes
    make daemon-run ARGS="project/cjk_anima_reseed/run.py <run> train"
    make daemon-run ARGS="project/cjk_anima_reseed/run.py <run> read"
    # a smoke: another data dir / out dir, a short schedule, an early stop
    make daemon-run ARGS="project/cjk_anima_reseed/run.py <run> train --data <dir> --out <dir> --steps_per_row 1 --max_steps 400"

``<run>`` is ``configs/<run>.toml`` or a path to one (``_archive/configs/``).
``data`` → ``output/cjk_anima_reseed/<run>/data``; ``train`` →
``…/<run>/trained.pt`` (the whole merged rows, ``cjk_scale.train``: the
run's rows trained cold or warm as its config says, every other row frozen
at the run's context rows). ``read`` (GPU) is
the scale line's ``experiments/grid_lone`` ``read_plain`` on the kana run's
13 words + 14 singles: the run paired against every reseed run read
before it and ``READ_AGAINST`` (renders cached in their dirs; the run's land in ``…/<run>/native_r4_plain/``) →
``results/<YYYYMMDD-HHMM>-<run>/result.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reseed import bootstrap  # noqa: E402

bootstrap()

# the arms of record a read pairs against (``output/cjk_anima_scale/…``)
READ_AGAINST = (
    "experiments/reseed_anchor_cold_kana_anchor",
    "experiments/reseed_recap_cold_kana_hp",
    "retrain_kana",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("verb", choices=["data", "train", "read"])
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
        assert not (run.stick_from or run.data_from), (
            f"{run.name}: trains on {run.data}"
        )
        from reseed.builder import build

        build(run, a.workers, a.frac)
    elif a.verb == "read":
        read(run)
    else:
        assert a.frac == 1.0, "--frac is the data verb's"
        from cjk_scale import train as T

        T.train(
            run.scale_config(),
            data=Path(a.data) if a.data else run.data,
            out=Path(a.out) if a.out else run.dir,
            max_steps=a.max_steps,
            cold=not (run.stick_from or run.warm or run.rows_from),
            steps_per_row=None if run.focus else a.steps_per_row or run.steps_per_row,
            steps=run.steps(a.steps_per_row),
            context=run.seed_rows(),
            drop_tiers=run.drop_tiers,
            stick_only=bool(run.stick_from),
            band=run.band,
            tag_drop=run.tag_drop,
            ball_on=run.seed_rows() if run.ball_on else None,
            lr=run.lr or None,
            row_step_scale=run.row_step_scale(),
            free_residual=run.free_residual,
            pres=run.pres,
        )


def read(run) -> None:
    import os

    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # every render is routed
    from bench._common import make_run_dir, write_result
    from cjk_scale.paths import OUT as SCALE_OUT
    from cjk_scale.paths import load_experiment
    from reseed import HOME, OUT

    # every other reseed run already read (its plain renders cached), then the
    # scale line's arms of record
    read_before = {
        d.name: d
        for d in sorted(OUT.iterdir())
        if d != run.dir and (d / "native_r4_plain" / "native_reads.json").exists()
    }
    arms = (
        {run.name: run.dir}
        | read_before
        | {Path(d).name: SCALE_OUT / d for d in READ_AGAINST}
    )
    metrics = {"read_plain": load_experiment("grid_lone").read_plain(arms, "all")}
    run_dir = make_run_dir("cjk_anima_reseed", label=run.name, root=HOME / "results")
    write_result(
        run_dir,
        script=__file__,
        args={"run": run.name, "verb": "read"},
        label=run.name,
        metrics=metrics,
        artifacts=[str(run.dir)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
