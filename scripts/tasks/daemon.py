"""CLI surface for the local training daemon (``make daemon*``).

Lifecycle verbs, mapped to the guarantees in ``plan.md`` Phase 1:

    daemon            start (idempotent — no-op if already up), wait /health
    daemon-attach     non-owning viewer; ctrl-C detaches only, training lives on
    daemon-kill       abort the running (or JOB=<id>) job, free GPU; daemon stays up
    daemon-terminate  shut the whole daemon down (active job dies too)
    daemon-prune      delete old terminal job dirs (dry-run unless --apply)

plus the submit/observe front door:

    daemon-run        submit an arbitrary command job (attach-by-default)
    daemon-wait       block until JOB=<id> is terminal; print record + result
    daemon-status     one-shot JSON (health + jobs, or JOB=<id>/--job <id>)

``daemon`` starts the daemon **console-detached** (see ``proc.spawn_detached``),
so the terminal's SIGINT reaches only the foreground group, never the daemon.
``daemon-attach`` is the parent of nothing, so its ctrl-C can't touch training.
Both teardown verbs verify the pidfile's ``(pid, create_time)`` before acting so
they never touch a PID-reused stranger.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from anima_daemon import client as _client
from anima_daemon import config as _cfg
from anima_daemon import proc as _proc


def _job_arg(extra) -> str | None:
    """Resolve a job id from ``JOB=<id>`` env or the first positional arg."""
    job = os.environ.get("JOB")
    if not job and extra and not extra[0].startswith("-"):
        job = extra[0]
    return job or None


def _read_result_envelope(record: dict) -> dict | None:
    """The bench ``result.json`` this job lifted, inlined (or ``None``).

    A GPU job that called ``bench/_common.write_result`` under the daemon dropped
    a ``result_path.json`` pointer in its job dir, which the daemon followed onto
    ``record["result_path"]`` (see ``manager._lift_result``). Reading it here is
    what makes "where did my run land, and what did it say" one command.
    """
    path = record.get("result_path")
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def cmd_daemon(extra):
    """Start the training daemon (idempotent). Detached + waits for /health."""
    existing = _proc.daemon_alive(_cfg.PIDFILE)
    if existing is not None:
        print(
            f"daemon already running (pid {existing.get('pid')}, "
            f"port {existing.get('port')})."
        )
        return
    try:
        cl = _client.ensure_daemon()
    except RuntimeError as e:
        print(f"failed to start daemon: {e}", file=sys.stderr)
        sys.exit(1)
    health = cl.health() or {}
    print(
        f"daemon up on {cl.base} (pid {health.get('pid')}). "
        f"Logs: {_cfg.DAEMON_LOG}\n"
        "  make daemon-attach        # follow events\n"
        "  make daemon-kill          # abort the running job\n"
        "  make daemon-terminate     # stop the daemon"
    )


# Compact per-job view daemon-status prints by default. Full records run ~1KB each
# and the history grows unboundedly — a polling agent needs "what's queued/running/
# failed", not the whole record (GET /jobs/{id} has that).
_STATUS_JOB_FIELDS = (
    "id",
    "method",
    "kind",
    "preset",
    "state",
    "submitted_at",
    "started_at",
    "ended_at",
    # Exit code. Without it a clean `done` and a "done but rc=1" row read
    # identically, so a queue-N-and-check-later batch had to open every job
    # record to find the failures. Negative = signal death (-9 = SIGKILL).
    "returncode",
    "error",
    "ckpt_path",
    # Where a finished bench/gen job landed its envelope — "where did my run
    # land" without a job-dir spelunk. Only the *pointer* here: a bench
    # `metrics` blob can run to hundreds of lines (per-pair records, per-step
    # curves), which would swamp a 15-job overview. `--job <id>` inlines the
    # whole envelope for the one job you actually want.
    "result_path",
    "chained_job_id",
)

# Default job cap for the compact view — the full history grows unboundedly, so
# a bare `daemon-status` shows only the most-recent slice (newest first). `--all`
# lifts the cap; `--limit N` sets it. `jobs_total`/`jobs_shown` in the output
# report the truncation so a capped view never reads as "that's everything".
_STATUS_DEFAULT_LIMIT = 15

# Shorthand state groups for `--running` / `--failed` / `--done`.
_STATUS_ACTIVE_STATES = frozenset({"running", "paused"})
_STATUS_FAILED_STATES = frozenset({"error", "stopped"})
# Every legal job state — `--state` is validated against this. A typo (or a job
# id passed where a state belongs) used to filter everything out and print
# `jobs_shown: 0`, which reads exactly like "the job vanished".
_STATUS_ALL_STATES = frozenset(
    {"queued", "running", "paused", "done", "error", "stopped"}
)


def _parse_status_flags(extra):
    """Parse the ``daemon-status`` filter flags out of ``extra``.

    ``--full`` (raw records) · ``--all`` (no cap) · ``--limit N`` ·
    ``--state s[,s]`` · ``--running``/``--active`` · ``--failed`` · ``--done`` ·
    ``--job <id>`` (one record, no list). Unknown tokens are ignored
    (forward-compatible with the make ARGS shim); a bad ``--state`` value is not
    — it lands in ``opts["bad_states"]`` for the caller to error out on.
    """
    extra = list(extra or [])
    opts = {
        "full": False,
        "all": False,
        "states": None,
        "bad_states": None,
        "job": None,
        "limit": _STATUS_DEFAULT_LIMIT,
    }
    i = 0
    while i < len(extra):
        a = extra[i]
        if a == "--full":
            opts["full"] = True
        elif a == "--all":
            opts["all"] = True
        elif a in ("--running", "--active"):
            opts["states"] = set(_STATUS_ACTIVE_STATES)
        elif a == "--failed":
            opts["states"] = set(_STATUS_FAILED_STATES)
        elif a == "--done":
            opts["states"] = {"done"}
        elif a == "--state" and i + 1 < len(extra):
            i += 1
            asked = {s.strip() for s in extra[i].split(",") if s.strip()}
            bad = asked - _STATUS_ALL_STATES
            if bad:
                opts["bad_states"] = sorted(bad)
            opts["states"] = asked & _STATUS_ALL_STATES
        elif a == "--job" and i + 1 < len(extra):
            i += 1
            opts["job"] = extra[i]
        elif a == "--limit" and i + 1 < len(extra):
            i += 1
            try:
                opts["limit"] = int(extra[i])
            except ValueError:
                pass
        i += 1
    return opts


def _job_target(job: dict) -> str | None:
    """Best-effort label for *what* a job operates on — the missing piece when
    skimming the queue. Command jobs (soup/preprocess/distill) carry it in
    ``argv`` (``--name`` / ``--path_pattern`` / the bench script's own
    ``--label``, else the ``-m`` module); train jobs carry it as the
    ``output_name`` override, else fall back to ``method``.

    ``--label`` is scanned last and is what separates a grid of N bench runs of
    the same script — otherwise they render as N identical ``argv[0]`` rows.
    It is derived at read time, so it also rescues jobs already on disk."""
    argv = job.get("argv") or []
    if job.get("kind") == "command":
        for flag in ("--name", "--path_pattern", "--output_name", "--label"):
            for i, tok in enumerate(argv):
                if tok == flag and i + 1 < len(argv):
                    return argv[i + 1]
                if tok.startswith(f"{flag}="):
                    return tok.split("=", 1)[1]
        if "-m" in argv:
            i = argv.index("-m")
            if i + 1 < len(argv):
                return argv[i + 1]
        return argv[0] if argv else None
    overrides = job.get("overrides") or {}
    return overrides.get("output_name") or job.get("method")


def cmd_daemon_status(extra):
    """Daemon status as one JSON object on stdout — the agent/script surface.

    ``{"up", "base_url", "pid", "port", "root", "stale_code", "paused",
    "active_job", "jobs_total", "jobs_shown", "jobs"}``. Passive: never starts a
    daemon (safe to poll); ``up: false`` + exit 1 when nothing answers
    ``/health``. ``base_url`` is resolved from the pidfile each call, so it
    follows a fallback-to-ephemeral port — read it from here rather than assuming
    8765. ``stale_code: true`` means the resident daemon is serving source older
    than the current on-disk ``anima_daemon/*`` — the next submit will eagerly
    restart it (Phase 0a).

    Jobs are compact summaries (id/state/error/ckpt_path/result_path/… plus a
    derived ``target`` = what the job operates on), **newest first** and capped to
    the most-recent ``_STATUS_DEFAULT_LIMIT``; ``jobs_total`` vs ``jobs_shown``
    report any truncation. Filter flags: ``--full`` (raw records) · ``--all`` (no
    cap) · ``--limit N`` · ``--state s[,s]`` · ``--running`` · ``--failed`` ·
    ``--done`` · ``--job <id>`` / ``JOB=<id>`` (that one record, full, with the
    bench ``result.json`` it lifted inlined under ``result`` — no
    list-then-eyeball step, and it reads the on-disk record when the daemon is
    down). An unknown ``--state`` value errors out (exit 2) instead of silently
    filtering everything away, which read like "the job vanished".
    """
    opts = _parse_status_flags(extra)
    if opts["bad_states"]:
        print(
            f"unknown job state(s): {', '.join(opts['bad_states'])}\n"
            f"  valid: {', '.join(sorted(_STATUS_ALL_STATES))}\n"
            "  (looking up one job? use --job <id> or JOB=<id>)",
            file=sys.stderr,
        )
        sys.exit(2)
    cl = _client.DaemonClient()
    job_id = opts["job"] or os.environ.get("JOB")
    if job_id:
        # Single-record mode: reads the on-disk job.json when the daemon is down,
        # so a post-mortem lookup works without resurrecting a daemon.
        record = cl.job_record(job_id)
        if record is None:
            print(
                json.dumps(
                    {
                        "up": cl.health() is not None,
                        "error": "no such job",
                        "job_id": job_id,
                    },
                    indent=2,
                )
            )
            sys.exit(2)
        envelope = _read_result_envelope(record)
        if envelope is not None:
            record = {**record, "result": envelope}
        print(json.dumps(record, indent=2))
        return
    health = cl.health()
    if health is None:
        print(json.dumps({"up": False, "base_url": None, "jobs": []}))
        sys.exit(1)
    jobs = cl.list_jobs()
    jobs.sort(key=lambda j: j.get("submitted_at") or 0, reverse=True)  # newest first
    jobs_total = len(jobs)
    if opts["states"] is not None:
        jobs = [j for j in jobs if j.get("state") in opts["states"]]
    if not opts["all"] and opts["limit"] is not None and opts["limit"] >= 0:
        jobs = jobs[: opts["limit"]]
    if opts["full"]:
        out_jobs = [{**j, "target": _job_target(j)} for j in jobs]
    else:
        out_jobs = [
            {**{k: j.get(k) for k in _STATUS_JOB_FIELDS}, "target": _job_target(j)}
            for j in jobs
        ]
    print(
        json.dumps(
            {
                "up": True,
                "base_url": cl.base,
                "pid": health.get("pid"),
                "port": health.get("port"),
                "root": health.get("root"),
                "stale_code": _client.daemon_is_stale(health),
                "paused": health.get("paused"),
                "active_job": health.get("active_job"),
                "jobs_total": jobs_total,
                "jobs_shown": len(out_jobs),
                "jobs": out_jobs,
            },
            indent=2,
        )
    )


# How often an idle attach prints "still here, still quiet". A GPU bench between
# prints and a wedged daemon look identical on a silent pipe, and the silence can
# legitimately run minutes — so say so periodically rather than leaving the reader
# (or an agent parsing the pipe) to guess.
_ATTACH_TICK_SECONDS = 30.0


def _start_attach_ticker(stop: threading.Event, last_line: list[float]) -> None:
    """Print ``[attached Ns — no output yet]`` while the stream stays silent."""
    t0 = time.time()

    def tick() -> None:
        while not stop.wait(_ATTACH_TICK_SECONDS):
            quiet = time.time() - last_line[0]
            if quiet >= _ATTACH_TICK_SECONDS:
                print(
                    f"[attached {time.time() - t0:.0f}s — no output for {quiet:.0f}s]",
                    flush=True,
                )

    threading.Thread(target=tick, daemon=True).start()


def cmd_daemon_attach(extra):
    """Read-only viewer. ``JOB=<id>`` follows that job's stdout; otherwise the
    daemon event stream. Ctrl-C detaches this terminal only — never the daemon
    or the training subprocess (we are the parent of nothing).

    Every write is flushed: stdout to a pipe is block-buffered, so an unflushed
    banner made a piped attach look like zero bytes / a hang. On a job that is
    already terminal this now returns as soon as the log is drained (the SSE
    endpoint closes the connection at ``eof``)."""
    if not _client.is_running():
        print("no daemon; `make daemon` to start.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    stream = cl.stream_logs(job) if job else cl.stream_events()
    what = f"job {job}" if job else "daemon events"
    print(f"attached to {what} ({cl.base}) — ctrl-C to detach\n", flush=True)
    stop = threading.Event()
    last_line = [time.time()]
    _start_attach_ticker(stop, last_line)
    try:
        for line in stream:
            last_line[0] = time.time()
            # The log stream's terminator is a {"ev":"eof","state":…} event. Now
            # that the connection actually closes (rather than parking), every
            # attach reaches it — so render it instead of leaking raw JSON.
            if line.startswith("{") and '"eof"' in line:
                try:
                    ev = json.loads(line)
                except ValueError:
                    ev = None
                if isinstance(ev, dict) and ev.get("ev") == "eof":
                    print(f"\n[job {job} ended: {ev.get('state')}]", flush=True)
                    break
            print(line, flush=True)
    except KeyboardInterrupt:
        print("\ndetached (training continues).", flush=True)
    except Exception as e:  # noqa: BLE001 — socket reset on daemon shutdown, etc.
        print(f"\nstream ended: {e}", flush=True)
    finally:
        stop.set()


def cmd_daemon_wait(extra):
    """Block until ``JOB=<id>`` (or the active job) is terminal, then print its
    record + any bench result envelope as JSON. Exits with the job's own exit code
    (``done`` → 0), so ``make daemon-wait JOB=… && next-step`` composes.

    The non-streaming half of "submit → wait → read the result": no log volume,
    and it reads the persisted ``job.json`` if the daemon restarts mid-wait. Ctrl-C
    detaches (exit 130) — the job keeps running. ``ARGS="--timeout 600"`` bounds the
    wait (exit 124, like ``timeout(1)``) and prints a JSON snapshot — state plus
    the last progress event and its staleness — so a scripted caller that gives up
    still learns whether the job is healthy-but-slow or wedged, rather than
    wrapping this in its own ``timeout`` and losing the status entirely.
    """
    extra = list(extra or [])
    timeout = None
    if "--timeout" in extra:
        i = extra.index("--timeout")
        if i + 1 < len(extra):
            try:
                timeout = float(extra[i + 1])
            except ValueError:
                print(f"bad --timeout value: {extra[i + 1]!r}", file=sys.stderr)
                sys.exit(2)
            del extra[i : i + 2]
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    if not job:
        health = cl.health() or {}
        job = health.get("active_job")
    if not job:
        print(
            "nothing to wait for: pass JOB=<id> (no job is active).\n"
            "  make daemon-status         # what has run / is queued",
            file=sys.stderr,
        )
        sys.exit(2)
    from anima_daemon.cli import _wait_and_report

    sys.exit(_wait_and_report(cl, job, timeout=timeout))


def split_daemon_run_args(
    extra: list[str],
) -> tuple[str | None, str | None, list[str], list[str]]:
    """Separate daemon-run's own flags from the child's argv.

    Returns ``(label, stall_timeout_raw, head, passthrough)``. Own flags
    (``--label`` / ``--stall-timeout``, space or ``=`` form) are recognized
    only BEFORE the first positional token (the script/module): bench scripts
    take ``--label`` themselves, so the same flag *after* the script path
    belongs to the child (the old anywhere-scan silently ate it). A literal
    ``--`` is a hard separator — everything after it is child argv verbatim,
    exempt from both this scan and run-mode flag resolution.

    ``stall_timeout_raw`` is returned unparsed so the caller owns the
    error/exit policy.
    """
    label: str | None = None
    stall_raw: str | None = None
    passthrough: list[str] = []
    if "--" in extra:
        cut = extra.index("--")
        extra, passthrough = extra[:cut], extra[cut + 1 :]

    head: list[str] = []
    in_own_zone = True
    i = 0
    while i < len(extra):
        tok = extra[i]
        own = in_own_zone and (
            tok in ("--label", "--stall-timeout")
            or tok.startswith(("--label=", "--stall-timeout="))
        )
        if own:
            if "=" in tok:
                flag, value = tok.split("=", 1)
            else:
                flag = tok
                if i + 1 >= len(extra):
                    print(f"{flag} needs a value", file=sys.stderr)
                    sys.exit(2)
                i += 1
                value = extra[i]
            if flag == "--label":
                label = value
            else:
                stall_raw = value
        else:
            if in_own_zone and not tok.startswith("-"):
                in_own_zone = False  # script/module reached: the rest is the child's
            head.append(tok)
        i += 1
    return label, stall_raw, head, passthrough


def cmd_daemon_run(extra):
    """Submit an arbitrary GPU command job (``ARGS="<script.py> [flags…]"``).

    The generic front door for "run this on the serial GPU queue": a bench script,
    a one-off probe, anything that would otherwise be started from a background
    shell and silently SIGKILLed by an agent harness after ~60s. Same run modes as
    the train/gen targets — attach by default (stream stdout, exit with the job's
    code, ctrl-C detaches), ``--queue`` to detach, ``--inline`` to bypass the
    daemon.

    daemon-run's own flags — ``--label NAME`` (display label, default: script
    basename) and ``--stall-timeout S`` (stall-watchdog budget, 0 = off) — must
    come BEFORE the script path; after it, every token (including ``--label``,
    which most bench scripts define themselves) is handed to the child. A
    ``--`` separator makes everything after it child argv verbatim, run-mode
    flags included.
    """
    from scripts.tasks import _common

    label, stall_raw, head, passthrough = split_daemon_run_args(list(extra or []))
    stall_timeout = None
    if stall_raw is not None:
        try:
            stall_timeout = float(stall_raw)
        except ValueError:
            print(f"bad --stall-timeout value: {stall_raw!r}", file=sys.stderr)
            sys.exit(2)
    mode, argv = _common._resolve_run_mode(head)
    argv += passthrough
    if not argv:
        print(
            'nothing to run. e.g. make daemon-run ARGS="project/x/bench/run_probe.py '
            '--limit 5"',
            file=sys.stderr,
        )
        sys.exit(2)
    from anima_daemon.cli import _label_for

    _common.run_command(
        label or _label_for(argv), argv, mode=mode, stall_timeout=stall_timeout
    )


def cmd_daemon_kill(extra):
    """Abort a job; the daemon stays up and advances to the next queued job.
    ``JOB=<id>`` targets a specific job; otherwise the running one."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    result = cl.stop(job)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(f"job {result.get('job_id')} → {result.get('state')} (daemon still up).")


def cmd_daemon_pause(extra):
    """Freeze the running job's process tree (SIGSTOP) in place — VRAM stays put,
    resume is instant. ``JOB=<id>`` targets a specific job; otherwise the active
    one. The queue does not advance past a paused job (it still owns the card)."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    result = cl.pause_job(_job_arg(extra))
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(
        f"job {result.get('job_id')} → {result.get('state')} (frozen; VRAM held). "
        f"`make daemon-resume` to thaw."
    )


def cmd_daemon_resume(extra):
    """Thaw a paused job (SIGCONT) → back to running. ``JOB=<id>`` targets a
    specific job; otherwise the active (paused) one."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    result = cl.resume_job(_job_arg(extra))
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(f"job {result.get('job_id')} → {result.get('state')} (thawed).")


def cmd_daemon_prune(extra):
    """Delete old terminal job dirs. Dry-run by default; ``ARGS="--apply"`` acts.

    The daemon already sweeps at boot (``jobs.prune_jobs`` from
    ``manager._reconcile``), so this is for reclaiming space from a daemon that's
    been up for weeks, or for previewing what the sweep would take. Forwards
    ``--days`` / ``--keep`` / ``--apply`` / ``--verbose`` straight through.
    """
    from anima_daemon import cli as _cli

    rc = _cli.main(["prune", *(extra or [])])
    if rc:
        sys.exit(rc)


def cmd_daemon_terminate(extra):
    """Stop the whole daemon. The active job tree is killed and the GPU freed."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        return
    cl = _client.DaemonClient()
    cl.shutdown(kill_jobs=True)
    print("daemon terminated (active job killed, GPU freed, queue discarded).")
