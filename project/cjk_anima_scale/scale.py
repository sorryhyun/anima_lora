#!/usr/bin/env python
"""scale — the JA vocab pack at scale: a run is one file, everything else is a rule.

    scale.py <run> data        # CPU: vocabs → items (recipe table by kind, σ band stamped per item)
    scale.py <run> train       # GPU: the vocabs' rows, everything else frozen at the seed
    scale.py <run> eval        # GPU: floor + trained on the rulers → <run>/sheet.png + reads.json
    scale.py <run> conflict    # GPU: do the run's band groups pull a row the same way (no training)
    scale.py <run> <verb> --submit [--queue]   # enqueue on the daemon (GPU verbs must)
    scale.py <run> data --workers N            # render processes (default cpu − 2)
    scale.py windows | runs | ledger           # the band law, the run files, the job ledger

A run is ``configs/runs/<run>.toml`` = ``{vocabs, read}`` (``cjk_scale/config.py``);
it lands in ``output/cjk_anima_scale/<run>/``. ``--submit`` enqueues this
same command on the daemon (agent-launched GPU work must go through it) and
records it in ``runs/ledger.jsonl``; the submit shell must name the pack
(``ANIMA_VOCAB_PACK=…``), since a job inherits an unset var from whichever
shell booted the daemon.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjk_scale.paths import REPO, bootstrap  # noqa: E402

bootstrap()

VERBS = ("data", "train", "eval", "conflict")
COMMANDS = ("windows", "runs", "ledger")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "run", help=f"configs/runs/<run>.toml, or one of {', '.join(COMMANDS)}"
    )
    p.add_argument("verb", nargs="?", choices=VERBS)
    p.add_argument(
        "--workers", type=int, help="data: render processes (default cpu − 2)"
    )
    p.add_argument(
        "--submit", action="store_true", help="enqueue this command on the daemon"
    )
    p.add_argument(
        "--queue",
        action="store_true",
        help="with --submit: detach instead of attaching",
    )
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.run in COMMANDS:
        assert a.verb is None, f"`{a.run}` takes no verb"
        return command(a.run)
    assert a.verb, f"scale.py {a.run} <{'|'.join(VERBS)}>"
    from cjk_scale.config import load_run

    rc = load_run(a.run)
    if a.submit:
        return submit(a)
    print(f"===== {rc.name}: {a.verb}", flush=True)
    if a.verb == "data":
        from cjk_scale.builder import build

        build(rc, workers=a.workers)
    elif a.verb == "train":
        from cjk_scale.train import train

        train(rc)
    elif a.verb == "eval":
        from cjk_scale.eval import run

        run(rc)
    elif a.verb == "conflict":
        from cjk_scale.conflict import probe

        probe(rc)


def command(name: str) -> None:
    if name == "windows":
        from cjk_scale.windows import table

        print(table())
    elif name == "runs":
        from cjk_scale.config import load_run, run_names

        for r in run_names():
            try:
                c = load_run(r)
            except AssertionError:  # a pre-collapse stage-shaped run file (records)
                print(f"{r}: stage-shaped run file (record only)")
                continue
            vocabs = c.vocabs if isinstance(c.vocabs, str) else ", ".join(c.vocabs)
            print(f"{r}: vocabs {vocabs}; read {' '.join(c.read) or '-'}")
    elif name == "ledger":
        from cjk_scale.ledger import rows

        for r in rows():
            what = (
                f"{r['run']} {r['verb']}"
                if "verb" in r
                else f"{r['stage']}/{r['tag']}  {' '.join(r['steps'])}"
            )
            print(f"{r['ts']}  {r.get('job_id', '-'):<26} {what}")


def submit(a) -> int:
    """Enqueue ``scale.py <run> <verb>`` as a daemon command job and record it
    in the ledger."""
    from anima_daemon import client as _client
    from anima_daemon import config as _dconfig
    from cjk_scale.ledger import append

    assert os.environ.get("ANIMA_VOCAB_PACK"), (
        "set ANIMA_VOCAB_PACK in the submit shell (every launch names the pack; "
        "an unset var falls through to the shell that booted the daemon)"
    )
    script = str(Path(__file__).resolve().relative_to(REPO))
    argv = [script, a.run, a.verb] + (
        ["--workers", str(a.workers)] if a.workers else []
    )
    label = f"scale-{a.run}-{a.verb}"
    cl = _client.ensure_daemon(expected_root=_dconfig.ROOT)
    resp = cl.submit_command(label=label, argv=argv, stall_timeout=0.0)
    job_id = resp.get("job_id")
    row = append(job_id=job_id, label=label, run=a.run, verb=a.verb, argv=argv)
    print(f"submitted {job_id} ({label}) — ledger: {row['ts']}", flush=True)
    if a.queue:
        return 0
    rec = cl.wait(job_id)
    rc = int(rec.get("returncode") or 0)
    print(f"job {job_id} finished: {rec.get('state')} rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
