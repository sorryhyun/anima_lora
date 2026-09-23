#!/usr/bin/env python
"""scale — the JA vocab pack at scale, one stage at a time (``design.md``).

    scale.py --stage stage0709 --tag t1 --steps data train eval [--submit [--queue]]
    scale.py --stage stage0507 --tag t1 --steps train eval          # warm from stage0709/t1
    scale.py --stage stage0709 --tag t1 --steps bake
    scale.py --stage stage0507 --tag bp --steps data boxprobe --n_items 240
        --warm_from output/cjk_anima_scale/rows_step1_0921m_merge/trained.pt   # gradient read, no training
    scale.py windows                                               # the band law
    scale.py stages                                                # the configs

A stage is ``configs/<stage>.toml``; ``--tag`` names the chain — every stage
of one chain shares it, and ``warm_from = "<stage>"`` resolves to that
stage's table under the same tag. Steps: ``data`` (CPU: renders +
``train.jsonl`` / ``eval.json``), ``train`` (GPU), ``eval`` (GPU: exact /
native / cf_sense + the regression check), ``bake``.

``--submit`` enqueues this same command on the daemon (agent-launched GPU
work must go through it) and records it in ``runs/ledger.jsonl``; the
submit shell must name the pack (``ANIMA_VOCAB_PACK=…``), since a job
inherits an unset var from whichever shell booted the daemon.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjk_scale.paths import REPO, bootstrap  # noqa: E402

bootstrap()

STEPS = ("data", "train", "eval", "bake", "boxprobe")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "windows", "stages", "ledger"],
    )
    p.add_argument("--stage", help="configs/<stage>.toml")
    p.add_argument("--tag", help="the chain's tag (shared by every stage of one chain)")
    p.add_argument(
        "--steps", nargs="+", default=["data", "train", "eval"], choices=STEPS
    )
    # data
    p.add_argument("--n_items", type=int, help="override [data].n_items (smoke builds)")
    p.add_argument("--seed", type=int, help="override [data].seed / [train].seed")
    # train
    p.add_argument(
        "--warm_from",
        help="override warm_from: a stage name, a trained.pt path, or 'cold'",
    )
    p.add_argument("--train_steps", type=int, help="override [train].train_steps")
    p.add_argument("--steps_per_row", type=int, help="override [train].steps_per_row")
    p.add_argument("--init_anchor", type=float, help="override [train].init_anchor")
    p.add_argument("--batch", type=int)
    p.add_argument("--compile", type=int)
    # eval
    p.add_argument(
        "--eval_only",
        nargs="+",
        choices=["eval", "native", "cf_sense"],
        help="run a subset of the rulers",
    )
    p.add_argument(
        "--eval_extra",
        nargs=argparse.REMAINDER,
        help="verbatim flags for the probe's eval parser",
    )
    # bake
    p.add_argument("--bake_out", help="bake: output pack dir")
    # boxprobe
    p.add_argument(
        "--probe_draws", type=int, default=3, help="boxprobe: σ draws per item"
    )
    p.add_argument(
        "--probe_items",
        type=int,
        default=0,
        help="boxprobe: cap on scene items (0 = all)",
    )
    # daemon
    p.add_argument(
        "--submit", action="store_true", help="enqueue this command on the daemon"
    )
    p.add_argument(
        "--queue",
        action="store_true",
        help="with --submit: detach instead of attaching",
    )
    p.add_argument(
        "--label", help="with --submit: the job label (default scale-<stage>-<tag>)"
    )
    p.add_argument(
        "--stall_timeout", type=float, default=0.0, help="with --submit: 0 = off"
    )
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.command == "windows":
        from cjk_scale.windows import table

        print(table())
        return
    if a.command == "stages":
        from cjk_scale.config import load, stage_names

        for s in stage_names():
            c = load(s)
            print(
                f"{s}: band {c.band[0]:.2f}–{c.band[1]:.2f}, gate {c.gate}, warm_from "
                f"{c.warm_from or 'cold'}, mix {' '.join(f'{m.name}={m.share:g}' for m in c.mix)}, "
                f"{c.train['steps_per_row']} steps/row"
            )
        return
    if a.command == "ledger":
        from cjk_scale.ledger import rows

        for r in rows():
            print(
                f"{r['ts']}  {r.get('job_id', '-'):<26} {r['stage']}/{r['tag']}  {' '.join(r['steps'])}"
            )
        return
    assert a.stage and a.tag, "--stage and --tag are required"
    if a.submit:
        return submit(a)
    from cjk_scale.config import load

    cfg = load(a.stage)
    for step in a.steps:
        print(f"===== {cfg.stage} / {a.tag}: {step}", flush=True)
        if step == "data":
            from cjk_scale.builder import build

            build(cfg, a.tag, n_items=a.n_items, seed=a.seed)
        elif step in ("train", "boxprobe"):
            warm = cfg.warm_table(a.tag)
            if a.warm_from:
                warm = (
                    None
                    if a.warm_from == "cold"
                    else cfg.__class__(
                        **{**cfg.__dict__, "warm_from": a.warm_from}
                    ).warm_table(a.tag)
                )
            if warm is not None:
                assert warm.exists(), f"warm table {warm} does not exist"
            if step == "boxprobe":
                from cjk_scale.boxprobe import probe

                probe(
                    cfg,
                    a.tag,
                    warm=warm,
                    draws=a.probe_draws,
                    max_items=a.probe_items,
                    seed=a.seed or 0,
                )
                continue
            from cjk_scale.train import train

            ov = {
                k: getattr(a, k)
                for k in (
                    "train_steps",
                    "steps_per_row",
                    "init_anchor",
                    "batch",
                    "compile",
                    "seed",
                )
                if getattr(a, k) is not None
            }
            train(cfg, a.tag, warm=warm, overrides=ov)
        elif step == "eval":
            from cjk_scale.eval import run

            run(
                cfg,
                a.tag,
                which=tuple(a.eval_only or ("eval", "native", "cf_sense")),
                extra=a.eval_extra,
            )
        elif step == "bake":
            from cjk_scale.bake import bake

            bake(cfg.stage, a.tag, a.bake_out)


def submit(a) -> int:
    """Enqueue ``scale.py <this argv minus --submit/--queue/--label>`` as a
    daemon command job and record it in the ledger."""
    from anima_daemon import client as _client
    from anima_daemon import config as _dconfig
    from cjk_scale.ledger import append

    assert os.environ.get("ANIMA_VOCAB_PACK"), (
        "set ANIMA_VOCAB_PACK in the submit shell (every launch names the pack; "
        "an unset var falls through to the shell that booted the daemon)"
    )
    argv = list(sys.argv[1:])
    for flag in ("--submit", "--queue"):
        argv = [x for x in argv if x != flag]
    for flag in ("--label", "--stall_timeout"):
        if flag in argv:
            i = argv.index(flag)
            del argv[i : i + 2]
    script = str(Path(__file__).resolve().relative_to(REPO))
    label = a.label or f"scale-{a.stage}-{a.tag}-{'-'.join(a.steps)}"
    cl = _client.ensure_daemon(expected_root=_dconfig.ROOT)
    resp = cl.submit_command(
        label=label, argv=[script, *argv], stall_timeout=a.stall_timeout
    )
    job_id = resp.get("job_id")
    row = append(
        job_id=job_id,
        label=label,
        stage=a.stage,
        tag=a.tag,
        steps=a.steps,
        argv=[script, *argv],
    )
    print(f"submitted {job_id} ({label}) — ledger: {row['ts']}", flush=True)
    if a.queue:
        return 0
    rec = cl.wait(job_id)
    rc = int(rec.get("returncode") or 0)
    print(f"job {job_id} finished: {rec.get('state')} rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
