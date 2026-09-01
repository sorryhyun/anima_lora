"""Client-side verbs for ``python -m anima_daemon`` — submit / wait / status / prune.

The daemon's HTTP surface was fully capable of "run this argv on the GPU queue"
long before anything on the command line could ask for it: submitting an
arbitrary command job meant writing a Python snippet against
``DaemonClient.submit_command``. These verbs are that missing front door,
kept in the daemon package (rather than ``scripts/tasks/``) so they work from a
bare checkout, a vendored node tree, or an agent shell — no ``tasks.py`` import,
no ``library.*``, stdlib only. (``prune`` is the odd one: state-dir maintenance
rather than a client call, but it belongs on the same front door and obeys the
same stdlib-only rule.)

    python -m anima_daemon submit [--label L] [--stall-timeout S] [--wait]
                                  [--hold] -- <argv…>
    python -m anima_daemon wait <job_id> [--timeout S]
    python -m anima_daemon status [job_id]
    python -m anima_daemon prune [--days N] [--keep N] [--apply]

``make daemon-run`` (``scripts/tasks/daemon.py``) is the repo-flavored wrapper
over ``submit`` with attach-by-default streaming; this module is the plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import client as _client
from . import config
from . import jobs as _jobs

VERBS = ("submit", "wait", "status", "prune")


def _label_for(argv: list[str]) -> str:
    """Derive a display label from the child argv: the script/module basename,
    plus the child's own ``--label`` when it has one.

    ``["project/x/bench/run_pair_census.py", "--limit", "5"]`` → ``run_pair_census``;
    ``["-m", "scripts.distill_turbo.distill"]`` → ``distill``. An inline
    ``python -c <src>`` has no name to take, so it stays ``command`` rather than
    becoming a slice of source code.

    ``daemon-run``'s own ``--label`` must precede the script path, but bench
    scripts take a ``--label`` of their own and muscle memory puts it after —
    labelling the bench *run dir* while the job record stayed generic, so a grid
    of N runs showed as N identical rows. When the daemon side wasn't given a
    label, borrow the child's: ``run_bench --label ko3_a`` → ``run_bench:ko3_a``.
    Display only — the child argv is passed through untouched either way.
    """
    name = None
    for i, tok in enumerate(argv):
        if tok == "-m" and i + 1 < len(argv):
            name = argv[i + 1].rsplit(".", 1)[-1]
            break
        if tok == "-c":
            return "command"
        if tok.startswith("-"):
            continue
        name = Path(tok).stem or tok
        break
    if name is None:
        return "command"
    child = _child_label(argv)
    return f"{name}:{child}" if child else name


def _child_label(argv: list[str]) -> Optional[str]:
    """The value of a ``--label`` (or ``--label=``) in the child's own argv."""
    for i, tok in enumerate(argv):
        if tok == "--label" and i + 1 < len(argv):
            return argv[i + 1] or None
        if tok.startswith("--label="):
            return tok.split("=", 1)[1] or None
    return None


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2), flush=True)


def _result_envelope(record: dict) -> Optional[dict]:
    """The bench ``result.json`` a finished job lifted, if any (§ result-lift)."""
    path = record.get("result_path")
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _exit_code_for(record: dict) -> int:
    """The job's exit code: its OS ``returncode`` when known, else state-derived.
    A signal death (negative rc) maps to the conventional ``128+N`` so ``&&``
    chains behave as they would for an inline run."""
    rc = record.get("returncode")
    if isinstance(rc, int):
        return 128 + (-rc) if rc < 0 else rc
    return 0 if record.get("state") == "done" else 1


def cmd_submit(args: argparse.Namespace) -> int:
    argv = list(args.argv or [])
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(
            "nothing to run: pass the child argv after `--`, e.g.\n"
            "  python -m anima_daemon submit -- project/x/bench/run_probe.py --limit 5",
            file=sys.stderr,
        )
        return 2
    label = args.label or _label_for(argv)
    cl = _client.ensure_daemon(expected_root=config.ROOT)
    resp = cl.submit_command(
        label=label,
        argv=argv,
        stall_timeout=args.stall_timeout,
        # `--hold` stages the job behind a paused gate; the default leaves the
        # gate alone so it runs when it reaches the front of the queue. Note this
        # is NOT `make …--queue`, which means "don't attach" — submitting here
        # never attaches, so returning immediately is already the default.
        start=False if args.hold else None,
    )
    job_id = resp.get("job_id")
    if not args.wait:
        _print_json({"job_id": job_id, "state": resp.get("state"), "base_url": cl.base})
        return 0
    return _wait_and_report(cl, job_id, timeout=args.timeout)


def cmd_wait(args: argparse.Namespace) -> int:
    cl = _client.DaemonClient()
    return _wait_and_report(cl, args.job_id, timeout=args.timeout)


def _timeout_snapshot(cl, job_id: str) -> dict:
    """Where the job stands when the wait gave up — state + last progress event.

    A bare ``timed out`` told a scripted caller nothing about whether the run was
    healthy-but-slow or wedged, so the snapshot goes to stdout as JSON (the
    timeout message itself stays on stderr).
    """
    out: dict = {"job_id": job_id, "timed_out": True}
    try:
        record = cl.job_record(job_id) or {}
    except Exception:  # noqa: BLE001 — the daemon may be the thing that's wedged
        record = {}
    for key in ("state", "started_at", "stdout_path", "progress_path"):
        if record.get(key) is not None:
            out[key] = record[key]
    # `latest` is the live field GET /jobs/{id} derives; fall back to reading the
    # stream directly so a snapshot still works with the daemon down.
    latest = record.get("latest")
    if latest is None:
        from . import tail

        latest = tail.last_event(record.get("progress_path"))
    if latest is not None:
        out["latest"] = latest
    if record.get("stale_for") is not None:
        out["stale_for"] = record["stale_for"]
    return out


def _wait_and_report(cl, job_id: str, *, timeout: Optional[float]) -> int:
    """Block on the job, print its final record (+ lifted result envelope), and
    return its exit code — ``124`` on wait timeout (matching ``timeout(1)``),
    with a snapshot of where the job stands so the caller keeps the in-flight
    status instead of an empty buffer."""
    try:
        record = cl.wait(job_id, timeout=timeout)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        _print_json(_timeout_snapshot(cl, job_id))
        return 124
    except KeyboardInterrupt:
        print(f"\ndetached (job {job_id} continues).", file=sys.stderr)
        return 130
    out = {
        "job_id": job_id,
        "state": record.get("state"),
        "returncode": record.get("returncode"),
        "error": record.get("error"),
        "ckpt_path": record.get("ckpt_path"),
        "result_path": record.get("result_path"),
        "result_summary": record.get("result_summary"),
        "stdout_path": record.get("stdout_path"),
    }
    envelope = _result_envelope(record)
    if envelope is not None:
        out["result"] = envelope
    _print_json(out)
    return _exit_code_for(record)


def cmd_status(args: argparse.Namespace) -> int:
    cl = _client.DaemonClient()
    if args.job_id:
        record = cl.job_record(args.job_id)
        if record is None:
            print(json.dumps({"error": "no such job", "job_id": args.job_id}))
            return 2
        envelope = _result_envelope(record)
        if envelope is not None:
            record = {**record, "result": envelope}
        _print_json(record)
        return 0
    health = cl.health()
    if health is None:
        _print_json({"up": False, "base_url": None})
        return 1
    _print_json({"up": True, "base_url": cl.base, **health})
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    """Sweep old terminal job dirs. Dry-run unless ``--apply``.

    Pure filesystem — it does not talk to the daemon, so it works whether or not
    one is up. With a *live* daemon the pruned jobs stay in its in-memory table
    until its next restart (harmless: they're finished history), so the boot
    sweep in ``manager._reconcile`` remains the primary path and this is the
    "I want the space back now" escape hatch.
    """
    summary = _jobs.prune_jobs(
        max_age_days=args.days,
        keep_recent=args.keep,
        dry_run=not args.apply,
    )
    summary["freed_mb"] = round(summary["freed_bytes"] / 1e6, 1)
    if not args.verbose:
        summary["pruned"] = len(summary["pruned"])
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anima_daemon",
        description="Submit / wait on / inspect jobs on the local anima daemon.",
    )
    sub = p.add_subparsers(dest="verb", required=True)

    s = sub.add_parser("submit", help="enqueue a command job (`-- <argv…>`)")
    s.add_argument("--label", help="display label (default: script/module basename)")
    s.add_argument(
        "--stall-timeout",
        type=float,
        default=None,
        dest="stall_timeout",
        help="per-job stall-watchdog budget in seconds; 0 disables it "
        "(default: the daemon's 120s command-job budget)",
    )
    s.add_argument(
        "--hold",
        action="store_true",
        help="stage the job behind a paused queue gate (Start Queue releases it) "
        "instead of letting it run when it reaches the front",
    )
    s.add_argument(
        "--wait", action="store_true", help="block until the job is terminal"
    )
    s.add_argument("--timeout", type=float, default=None, help="--wait timeout (s)")
    s.add_argument("argv", nargs=argparse.REMAINDER, help="`-- <child argv…>`")
    s.set_defaults(func=cmd_submit)

    w = sub.add_parser("wait", help="block until a job is terminal; print its record")
    w.add_argument("job_id")
    w.add_argument(
        "--timeout", type=float, default=None, help="give up after S seconds"
    )
    w.set_defaults(func=cmd_wait)

    st = sub.add_parser("status", help="daemon health, or one job record")
    st.add_argument("job_id", nargs="?", help="omit for daemon-level health")
    st.set_defaults(func=cmd_status)

    pr = sub.add_parser(
        "prune", help="delete old terminal job dirs (dry-run by default)"
    )
    pr.add_argument(
        "--days",
        type=float,
        default=None,
        help=f"age threshold in days; 0 disables (default: {config.JOB_RETENTION_DAYS:g})",
    )
    pr.add_argument(
        "--keep",
        type=int,
        default=None,
        help="always keep this many newest terminal jobs regardless of age "
        f"(default: {config.JOB_RETENTION_KEEP})",
    )
    pr.add_argument(
        "--apply", action="store_true", help="actually delete (default: dry-run)"
    )
    pr.add_argument(
        "--verbose",
        action="store_true",
        help="list every pruned job id, not just the count",
    )
    pr.set_defaults(func=cmd_prune)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)
