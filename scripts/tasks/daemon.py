"""CLI surface for the local training daemon (``make daemon*``).

Lifecycle verbs:

    daemon            start (idempotent — no-op if already up), wait /health
    daemon-attach     non-owning viewer; ctrl-C detaches only, training lives on
    daemon-kill       abort the running (or JOB=<id>) job, free GPU; daemon stays up
    daemon-terminate  shut the whole daemon down (active job dies too)
    daemon-prune      delete old terminal job dirs (dry-run unless --apply)

plus the submit/observe front door:

    daemon-run        submit an arbitrary command job (attach-by-default)
    daemon-wait       block until JOB=<id> is terminal; print record + result
    daemon-jobs       the job history as greppable lines, oldest first
                      (newest 15; --full = all of it; RUN=<name> = one run's jobs)
    daemon-log        a job's captured stdout, read off disk

``daemon`` starts the daemon **console-detached** (see ``proc.spawn_detached``),
so the terminal's SIGINT reaches only the foreground group, never the daemon.
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
from anima_daemon.record import (
    ACTIVE_STATES,
    ALL_STATES,
    STATE_DONE,
    STATE_ERROR,
    STATE_RUNNING,
    STATE_STOPPED,
    UNFINISHED_STATES,
    iter_records,
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _job_arg(extra) -> str | None:
    """Resolve a job id from ``JOB=<id>`` env or the first positional arg."""
    job = os.environ.get("JOB")
    if not job and extra and not extra[0].startswith("-"):
        job = extra[0]
    return job or None


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
        "  make daemon-attach        # follow the active job (or events)\n"
        "  make daemon-kill          # abort the running job\n"
        "  make daemon-terminate     # stop the daemon"
    )


# Default job cap for a bare `daemon-jobs` (most recent slice). `--limit N` sets
# it and `--full` lifts it; a trailing `N of M jobs` line reports truncation.
_STATUS_DEFAULT_LIMIT = 15
# Cap for a `--run <name>` lookup. A run's own name matches a job or two; a name
# that matches more is shared (a script path, `smoke`, a shipped config), and the
# caller gets the newest few and the count instead of hundreds of lines.
_RUN_DEFAULT_LIMIT = 5

_STATUS_FAILED_STATES = frozenset({STATE_ERROR, STATE_STOPPED})


def _parse_status_flags(extra):
    """Parse the ``daemon-jobs`` filter flags out of ``extra``.

    ``--full`` (no cap) · ``--limit N`` · ``--run NAME`` (``RUN=<name>``) ·
    ``--state s[,s]`` · ``--running``/``--active`` · ``--failed`` · ``--done``.
    Unknown tokens are ignored (forward-compatible with the make ARGS shim); a
    bad ``--state`` value is not — it lands in ``opts["bad_states"]`` for the
    caller to error out on. ``opts["limit"]`` is ``None`` until ``--limit``
    sets it to a positive number.
    """
    extra = list(extra or [])
    opts = {
        "full": False,
        "states": None,
        "bad_states": None,
        "limit": None,
        "run": os.environ.get("RUN") or None,
    }
    i = 0
    while i < len(extra):
        a = extra[i]
        if a == "--full":
            opts["full"] = True
        elif a in ("--running", "--active"):
            opts["states"] = set(ACTIVE_STATES)
        elif a == "--failed":
            opts["states"] = set(_STATUS_FAILED_STATES)
        elif a == "--done":
            opts["states"] = {STATE_DONE}
        elif a == "--state" and i + 1 < len(extra):
            i += 1
            asked = {s.strip() for s in extra[i].split(",") if s.strip()}
            bad = asked - ALL_STATES
            if bad:
                opts["bad_states"] = sorted(bad)
            opts["states"] = asked & ALL_STATES
        elif a == "--limit" and i + 1 < len(extra):
            i += 1
            try:
                opts["limit"] = max(int(extra[i]), 0) or None
            except ValueError:
                pass
        elif a == "--run" and i + 1 < len(extra):
            i += 1
            opts["run"] = extra[i]
        elif a.startswith("--run="):
            opts["run"] = a.split("=", 1)[1]
        i += 1
    return opts


# Flags that name a run outright, in scan order. ``--label`` is last: it is what
# separates a grid of N bench runs of the same script.
_RUN_NAME_FLAGS = ("--name", "--path_pattern", "--output_name", "--label", "--run")

# What a job on a shipped config writes: the LoRA file (train), the image folder
# (generate), the cache folder (cache). Its name stands in for a run name there.
_ARTIFACT_FLAGS = ("--output", "--out_dir", "--out")


def _shipped_config(config: str) -> bool:
    """True for a config under the repo's ``configs/`` tree: a recipe many jobs
    share (``configs/moss/i2i/swap_text/train.toml``), not one run's file."""
    path = Path(config)
    if path.is_absolute():
        try:
            path = path.relative_to(_cfg.ROOT)
        except ValueError:
            return False
    return path.parts[:1] == ("configs",)


def _argv_value(argv, flag: str) -> str | None:
    """The value of ``flag`` in ``argv`` (``--flag v`` or ``--flag=v``), else None."""
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if isinstance(tok, str) and tok.startswith(f"{flag}="):
            return tok.split("=", 1)[1]
    return None


def _job_run_name(job: dict) -> tuple[str | None, str | None]:
    """``(run name, what ran)`` for a job; ``daemon-jobs --run`` matches on the first.

    Command jobs, first hit wins: a naming flag in ``argv`` (``--name`` /
    ``--path_pattern`` / ``--output_name`` / the bench script's own ``--label`` /
    ``--run``, unless its value is a path), else the stem of the ``--config`` file
    (a run's config file carries its ``output_name``), else the ``-m`` module or
    the script path. A config under ``configs/`` is a shipped recipe that many
    jobs share, so there the name is what the job writes (the stem of
    ``--output``, the last folder of ``--out_dir`` / ``--out``), else the config's
    last two path parts (``swap_text/train``). Train jobs:
    the ``output_name`` override, else ``--output_name`` in ``extra``, else the
    method (``--method`` in ``extra`` wins over the record's).

    The second item is the script (command jobs) or method (train jobs), set
    only when the name came from a config stem or an ``--output_name``, where
    the name alone no longer says what ran.
    """
    if job.get("kind") == "command":
        argv = job.get("argv") or []
        for flag in _RUN_NAME_FLAGS:
            name = _argv_value(argv, flag)
            if name and not (flag == "--run" and os.sep in name):
                return name, None
        script = _argv_value(argv, "-m") or (argv[0] if argv else None)
        config = _argv_value(argv, "--config")
        if config and Path(config).stem:
            if not _shipped_config(config):
                return Path(config).stem, script
            for flag in _ARTIFACT_FLAGS:
                artifact = _argv_value(argv, flag)
                if artifact and Path(artifact).stem:
                    return Path(artifact).stem, script
            return "/".join(Path(config).with_suffix("").parts[-2:]), script
        return script, None
    extra = job.get("extra") or []
    method = _argv_value(extra, "--method") or job.get("method")
    name = (job.get("overrides") or {}).get("output_name") or _argv_value(
        extra, "--output_name"
    )
    return (name, method) if name else (method, None)


def _job_target(job: dict) -> str | None:
    """A job's run name (see ``_job_run_name``)."""
    return _job_run_name(job)[0]


def jobs_by_run_name(name: str, jobs) -> list[dict]:
    """The records in ``jobs`` whose run name is exactly ``name``, oldest first.

    An exact match on ``_job_run_name``, not a search over argv: a later job
    that passes ``--init <name>`` is a different run.
    """
    hits = [j for j in jobs if _job_run_name(j)[0] == name]
    hits.sort(key=lambda j: j.get("submitted_at") or 0)
    return hits


def _job_line(job: dict) -> str:
    """One job as one line: when · id · state · rc · run name [what ran] · error."""
    ts = job.get("started_at") or job.get("submitted_at") or 0
    when = time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "  -  "
    rc = job.get("returncode")
    end = job.get("ended_at") or (
        time.time() if job.get("state") == STATE_RUNNING else None
    )
    started = job.get("started_at")
    dur = f"{(end - started) / 60:5.1f}m" if started and end else "     -"
    err = (job.get("error") or "").splitlines()
    tail = f"  {err[0][:60]}" if err else ""
    name, ran = _job_run_name(job)
    target = f"{name}  {ran}" if name and ran else (name or "-")
    return (
        f"{when}  {job.get('id', '?'):<22} {str(job.get('state')):<8} "
        f"rc={'-' if rc is None else rc:<4} {dur}  {target}{tail}"
    )


def job_line_with_dir(job: dict, jobs_dir: Path | None = None) -> str:
    """``_job_line`` plus the job's directory as the last field (relative to the
    CWD when it sits under it), so its ``stdout.log`` opens without a lookup."""
    path = Path(jobs_dir or _cfg.JOBS_DIR).absolute() / str(job.get("id"))
    try:
        path = path.relative_to(Path.cwd())
    except ValueError:
        pass
    return f"{_job_line(job)}  {path}"


def cmd_daemon_status(extra):
    """Daemon health in one line, then every unfinished job (running, paused,
    queued) as ``daemon-jobs`` lines. ``JOB=<id>``: that job's full record +
    result envelope (``python -m anima_daemon status <id>``). Exit 1 when the
    daemon is down."""
    from anima_daemon import cli as _cli

    job = _job_arg(extra)
    if job:
        sys.exit(_cli.main(["status", job]))
    cl = _client.DaemonClient()
    health = cl.health()
    if health is None:
        print("daemon down")
        sys.exit(1)
    stale = (
        " · STALE code (next submit restarts it)"
        if _client.daemon_is_stale(health)
        else ""
    )
    print(
        f"daemon up · pid {health.get('pid')} · {cl.base}"
        f" · active {health.get('active_job') or '-'}"
        f"{' · queue PAUSED' if health.get('paused') else ''}{stale}"
    )
    jobs = [j for j in cl.list_jobs() if j.get("state") in UNFINISHED_STATES]
    jobs.sort(key=lambda j: j.get("submitted_at") or 0)
    for job in jobs:
        print(_job_line(job))
    if not jobs:
        print("no unfinished jobs")


def cmd_daemon_jobs(extra):
    """The job history as greppable lines, oldest first (``| tail -5`` = the five
    most recent): the newest 15, or the whole history with ``--full``.

    Filters: ``--limit N`` · ``--full`` · ``--running``/``--active`` ·
    ``--failed`` · ``--done`` · ``--state s[,s]``. Falls back to the on-disk
    records when the daemon is down (exit 1).

    ``--run NAME`` / ``RUN=<name>``: every job in the history whose run name is
    exactly ``NAME`` (``_job_run_name``), each line ending in its job directory.
    The newest 5 matches unless ``--limit`` / ``--full`` is given (a name that
    matches more is shared by many jobs); exit 1 when nothing matches.
    """
    opts = _parse_status_flags(extra)
    if opts["bad_states"]:
        print(
            f"unknown job state(s): {', '.join(opts['bad_states'])}\n"
            f"  valid: {', '.join(sorted(ALL_STATES))}",
            file=sys.stderr,
        )
        sys.exit(2)
    cl = _client.DaemonClient()
    up = cl.health() is not None
    jobs = cl.list_jobs() if up else list(iter_records())
    jobs.sort(key=lambda j: j.get("submitted_at") or 0)  # oldest first: tail = newest
    total = len(jobs)
    run = opts["run"]
    if run:
        jobs = jobs_by_run_name(run, jobs)
    if opts["states"] is not None:
        jobs = [j for j in jobs if j.get("state") in opts["states"]]
    matched = len(jobs)
    limit = opts["limit"] or (_RUN_DEFAULT_LIMIT if run else _STATUS_DEFAULT_LIMIT)
    if not opts["full"] and limit:
        jobs = jobs[-limit:]  # the newest N, still oldest-first
    for job in jobs:
        print(job_line_with_dir(job) if run else _job_line(job))
    shown = len(jobs)
    note = "" if up else "  (daemon down — read from disk)"
    if run:
        if not matched:
            print(f"no job with run name {run!r} in {total} jobs{note}", flush=True)
            sys.exit(1)
        rest = (
            ""
            if shown == matched
            else "  — a name many jobs share; --full for the rest"
        )
        print(f"\n{shown} of {matched} jobs named {run!r}{note}{rest}", flush=True)
        return
    print(
        f"\n{shown} of {total} jobs{note}"
        + ("" if shown == matched else "  — --full for the rest"),
        flush=True,
    )
    if not up:
        sys.exit(1)


def cmd_daemon_log(extra):
    """Dump a job's captured stdout — ``JOB=<id>``, else the most recent job.

    Reads the log file off disk, so it works on a finished job and with the
    daemon down. ``ARGS="-n 200"`` bounds the tail (default 100; ``-n 0`` = the whole file).
    """
    extra = list(extra or [])
    n = 100
    if "-n" in extra:
        i = extra.index("-n")
        if i + 1 < len(extra):
            try:
                n = int(extra[i + 1])
            except ValueError:
                pass
            del extra[i : i + 2]
    job_id = _job_arg(extra)
    cl = _client.DaemonClient()
    if not job_id:
        jobs = cl.list_jobs() if cl.health() is not None else list(iter_records())
        jobs.sort(key=lambda j: j.get("submitted_at") or 0)
        if not jobs:
            print("no jobs on record.", file=sys.stderr)
            sys.exit(1)
        job_id = jobs[-1].get("id")
    record = cl.job_record(job_id) or {}
    path = Path(record.get("stdout_path") or (_cfg.job_dir(job_id) / "stdout.log"))
    if not path.is_file():
        print(f"no stdout log for job {job_id} ({path})", file=sys.stderr)
        sys.exit(1)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = _job_line(record) if record else job_id
    shown = lines if n <= 0 else lines[-n:]
    print(f"# {head}\n# {path}  ({len(lines)} lines, showing {len(shown)})\n")
    for ln in shown:
        print(ln)


# How often an attach with no output prints a `[attached Ns — no output yet]` tick.
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
    """Read-only viewer. ``JOB=<id>`` follows that job's stdout; without it, the
    active job's stdout, or the daemon event stream when nothing is running
    (``ARGS=--events`` forces the event stream). Ctrl-C detaches this terminal
    only, since this process is the parent of nothing. On a job that is already
    terminal it returns as soon as the log is drained."""
    if not _client.is_running():
        print("no daemon; `make daemon` to start.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    if not job and "--events" not in extra:
        job = (cl.health() or {}).get("active_job")
    stream = cl.stream_logs(job) if job else cl.stream_events()
    what = f"job {job}" if job else "daemon events"
    print(f"attached to {what} ({cl.base}) — ctrl-C to detach\n", flush=True)
    stop = threading.Event()
    last_line = [time.time()]
    _start_attach_ticker(stop, last_line)
    try:
        for line in stream:
            last_line[0] = time.time()
            # The log stream's terminator is a {"ev":"eof","state":…} event;
            # render it instead of printing raw JSON.
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

    Reads the persisted ``job.json`` if the daemon restarts mid-wait. Ctrl-C
    detaches (exit 130); the job keeps running. ``ARGS="--timeout 600"`` bounds
    the wait (exit 124, like ``timeout(1)``) and prints a JSON snapshot: state,
    the last progress event and its staleness.
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
            "  make daemon-jobs           # what has run / is queued",
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
    belongs to the child. A literal ``--`` is a hard separator — everything
    after it is child argv verbatim, exempt from both this scan and run-mode
    flag resolution.

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

    For a bench script or one-off probe. Same run modes as the train/gen targets: attach by default (stream stdout, exit with
    the job's code, ctrl-C detaches), ``--queue`` to detach, ``--inline`` to
    bypass the daemon.

    Own flags: ``--label NAME`` (display label, default: script basename) and
    ``--stall-timeout S`` (stall-watchdog budget, 0 = off), both before the
    script path — see ``split_daemon_run_args``.
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
    jid, state = result.get("job_id"), result.get("state")
    if state in ("running", "paused"):
        # the reply carries the state at the kill; the job finalizes as stopped
        print(f"job {jid}: killed (was {state}); the queue moves on.")
    elif state == "stopped":
        print(f"job {jid}: cancelled.")
    else:
        print(f"job {jid}: already {state}, nothing to kill.")


def cmd_daemon_pause(extra):
    """Freeze the running job's process tree (SIGSTOP); VRAM stays put and the
    queue does not advance. ``JOB=<id>`` targets a specific job; otherwise the
    active one.

    ``RELEASE=1`` (or ``--release``, train.py jobs only): the trainer saves a
    resumable state and exits, freeing the GPU; ``make daemon-resume`` relaunches
    it. Details: ``anima_daemon/README.md``."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    extra = list(extra or [])
    release = _truthy(os.environ.get("RELEASE")) or "--release" in extra
    extra = [a for a in extra if a != "--release"]
    cl = _client.DaemonClient()
    result = cl.pause_job(_job_arg(extra), release_model=release)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    if release:
        print(
            f"job {result.get('job_id')}: release-pause requested — the trainer "
            f"saves a resumable state at its next step and exits (state → paused, "
            f"GPU free). `make daemon-resume JOB={result.get('job_id')}` relaunches it."
        )
        return
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
    if result.get("relaunch"):
        print(
            f"job {result.get('job_id')} → {result.get('state')} (relaunch "
            f"#{result['relaunch']} from the saved state, front of the queue)."
        )
        return
    print(f"job {result.get('job_id')} → {result.get('state')} (thawed).")


def cmd_daemon_prune(extra):
    """Delete old terminal job dirs. Dry-run by default; ``ARGS="--apply"`` acts.

    The daemon already sweeps at boot; this covers a daemon that has been up for
    weeks, or a preview. Takes ``--days`` / ``--keep`` / ``--apply`` / ``--verbose``.
    """
    from anima_daemon import cli as _cli

    rc = _cli.main(["prune", *(extra or [])])
    if rc:
        sys.exit(rc)


def cmd_runpod(extra):
    """RunPod pods: ``ARGS="gpus|ls|create|wait|ssh|stop|terminate …"``, forwarded
    to ``python -m anima_daemon.runpod`` (``anima_daemon/runpod/cli.py``)."""
    from anima_daemon.runpod import cli as _rp_cli

    rc = _rp_cli.main(list(extra or []))
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
