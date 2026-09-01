"""One-liner status for a training run, read from its ``progress.jsonl``.

Usage:
    python scripts/run_status.py                  # newest run under output/logs
    python scripts/run_status.py anima_turbo      # by output_name
    python scripts/run_status.py path/to/x.progress.jsonl
    python scripts/run_status.py --list           # every known run, newest first
    python scripts/run_status.py --json           # machine-readable digest

Answers "where is this run at?" without exporting the TensorBoard events file:
step N/total, rate, ETA, last losses, last checkpoint, and whether the run is
still alive (``running``), finished (``ok``), or died (``error`` / ``stopped`` /
``dead``). Works for any run that emits the sink — ``train.py`` methods and the
bespoke ``make turbo`` loop alike.

Both launch paths are covered: an inline run writes
``output/logs/<output_name>.progress.jsonl``, while a **daemon** job (the default
launch mode) is handed an explicit per-job path,
``output/daemon/jobs/<id>/progress.jsonl`` — same stream, different home and a
bare filename that carries no run name. So daemon job dirs are scanned too, and
a run name there is matched against the ``run_start`` event inside the stream
(a job id works as a target as well).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library.env import resolve_under_home  # noqa: E402
from library.training.progress import read_status  # noqa: E402

_SUFFIX = ".progress.jsonl"
# The daemon points every train job's sink here (manager._build_cmd), one dir
# per job id, so the filename is bare — no `<run>` prefix to match on.
_DAEMON_NAME = "progress.jsonl"


def _find_streams(logs_dir: Path, jobs_dir: Path | None = None) -> list[Path]:
    """Every progress stream under ``logs_dir`` + the daemon job dirs, newest first."""
    streams: list[Path] = []
    if logs_dir.is_dir():
        streams += [p for p in logs_dir.rglob(f"*{_SUFFIX}") if p.is_file()]
    if jobs_dir is not None and jobs_dir.is_dir():
        # Only train jobs get a sink; a command job's progress.jsonl never
        # materializes, so is_file() is the filter.
        streams += [p for p in jobs_dir.glob(f"*/{_DAEMON_NAME}") if p.is_file()]

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:  # pruned between the glob and the stat
            return 0.0

    return sorted(streams, key=_mtime, reverse=True)


def _stream_run_name(path: Path) -> str | None:
    """The run (``output_name``) a stream belongs to, or ``None``.

    A flat ``logs/`` stream carries it in the filename; a daemon job's stream is
    always ``<job dir>/progress.jsonl``, so the only copy of the name is the
    ``run`` field of the ``run_start`` event — the first line of the file.
    """
    if path.name.endswith(_SUFFIX):
        return path.name[: -len(_SUFFIX)]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in itertools.islice(fh, 20):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("ev") == "run_start":
                    return rec.get("run")
    except OSError:
        return None
    return None


def _job_id_for(path: Path) -> str | None:
    """The daemon job id owning this stream (its dir name), for flat-named files."""
    return path.parent.name if path.name == _DAEMON_NAME else None


def _resolve(target: str | None, logs_dir: Path, jobs_dir: Path | None) -> Path:
    """Resolve a path / run name / job id / nothing-at-all to one stream file."""
    streams = _find_streams(logs_dir, jobs_dir)
    if target:
        direct = resolve_under_home(target)
        if direct.is_file():
            return direct
        named = logs_dir / f"{target}{_SUFFIX}"
        if named.is_file():
            return named
        # A run name that doesn't sit flat in logs_dir: a nested logs/<run>/ dir,
        # or a daemon job dir (matched on the in-stream run name, or the job id).
        # Streams are newest-first, so the first hit is the latest run of a name
        # that has been trained more than once.
        hit = next(
            (
                p
                for p in streams
                if _job_id_for(p) == target or _stream_run_name(p) == target
            ),
            None,
        )
        if hit is not None:
            return hit
        where = f"{logs_dir}" + (f" and {jobs_dir}" if jobs_dir else "")
        raise SystemExit(f"no progress stream for {target!r} (looked under {where})")
    if not streams:
        raise SystemExit(
            f"no progress stream under {logs_dir}"
            + (f" or {jobs_dir}" if jobs_dir else "")
            + " — no run has started, or the sink is disabled "
            "(--no_log / --progress_jsonl off)."
        )
    return streams[0]


def _fmt_dur(seconds: float | None) -> str:
    if not seconds or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_metrics(metrics: dict, limit: int) -> str:
    out = []
    for key, val in list(metrics.items())[:limit]:
        out.append(f"{key}={val:.4g}" if isinstance(val, float) else f"{key}={val}")
    return " ".join(out)


def format_status(st: dict, *, metric_limit: int = 6) -> str:
    step = st["global_step"] if st["global_step"] is not None else "?"
    total = st["total_steps"] or "?"
    pct = f" ({st['pct']:.1f}%)" if st["pct"] is not None else ""
    rate = f"{st['rate']:.2f} it/s" if st["rate"] else "? it/s"
    # A daemon-launched run: name the job so the follow-up (`make daemon-status
    # ARGS="--job <id>"`, `make daemon-attach JOB=<id>`) needs no lookup step.
    job = _job_id_for(Path(st["path"]))
    where = f" (job {job})" if job else ""
    head = (
        f"{st['run'] or st['path']}{where} [{st['method'] or '?'}] "
        f"{st['status'].upper()}  step {step}/{total}{pct}  {rate}  "
        f"elapsed {_fmt_dur(st['elapsed'])}  ETA {_fmt_dur(st['eta'])}"
    )
    lines = [head]
    if st["metrics"]:
        lines.append(f"  last:  {_fmt_metrics(st['metrics'], metric_limit)}")
    if st["val"]:
        lines.append(f"  val:   {_fmt_metrics(_strip_ev(st['val']), metric_limit)}")
    if st["ckpt"]:
        lines.append(
            f"  ckpt:  {st['ckpt']['path']} (step {st['ckpt']['global_step']})"
        )
    if st["error"]:
        lines.append(f"  error: {st['error']}")
    if st["warnings"]:
        lines.append(f"  {st['warnings']} warning/error log event(s) in the stream")
    return "\n".join(lines)


def _strip_ev(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("ev", "ts")}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "target",
        nargs="?",
        help="run name (output_name), a .progress.jsonl path, or omit for the "
        "most recently updated run.",
    )
    p.add_argument(
        "--logs-dir",
        default="output/logs",
        help="where to look up run names (default: output/logs).",
    )
    p.add_argument(
        "--jobs-dir",
        default="output/daemon/jobs",
        help="daemon job dirs, each holding a per-job progress.jsonl "
        "(default: output/daemon/jobs). Pass '' to skip them.",
    )
    p.add_argument(
        "--list", action="store_true", help="report every run found, newest first."
    )
    p.add_argument("--json", action="store_true", help="emit the status dict as JSON.")
    args = p.parse_args(argv)

    logs_dir = resolve_under_home(args.logs_dir)
    jobs_dir = resolve_under_home(args.jobs_dir) if args.jobs_dir else None
    if args.list:
        streams = _find_streams(logs_dir, jobs_dir)
        if not streams:
            raise SystemExit(
                f"no progress stream under {logs_dir}"
                + (f" or {jobs_dir}" if jobs_dir else "")
            )
    else:
        streams = [_resolve(args.target, logs_dir, jobs_dir)]

    statuses = []
    for path in streams:
        try:
            statuses.append(read_status(str(path)))
        except (OSError, ValueError) as exc:
            if not args.list:
                raise SystemExit(f"{path}: {exc}")
            print(f"{path}: {exc}", file=sys.stderr)

    if args.json:
        json.dump(statuses if args.list else statuses[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("\n\n".join(format_status(st) for st in statuses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
