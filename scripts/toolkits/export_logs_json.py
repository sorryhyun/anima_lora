"""Export TensorBoard scalar logs from a training run to JSON.

Usage:
    python scripts/toolkits/export_logs_json.py output/logs/ip_adapter_default_20260424-1611
    python scripts/toolkits/export_logs_json.py --all output/logs
    python scripts/toolkits/export_logs_json.py output/logs/<run> --out metrics.json --jsonl

Reads every ``events.out.tfevents.*`` file under the given run directory and
dumps all scalar tags. Default output is one ``metrics.json`` per run written
next to the event files. Pass ``--jsonl`` for one-object-per-line streaming
format, or ``--stdout`` to print a single JSON blob.

Output schema (default JSON form):

    {
      "run":  "<run directory path>",
      "tags": {
        "<tag>": [[step, wall_time, value], ...],
        ...
      }
    }

Iterate ``payload["tags"].items()``; the metric value is ``row[2]``.

``--jsonl`` flattens to one ``{tag, step, wall_time, value}`` object per line.

``--summary`` prints the last value (and its step) per tag, plus the run's max
step. Add ``--stdout`` for the same thing as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _find_event_dirs(root: Path) -> list[Path]:
    """Return unique directories containing ``events.out.tfevents.*`` files."""
    dirs: set[Path] = set()
    for p in root.rglob("events.out.tfevents.*"):
        dirs.add(p.parent)
    return sorted(dirs)


def _load_scalars(event_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return {
        tag: [(e.step, e.wall_time, e.value) for e in ea.Scalars(tag)]
        for tag in ea.Tags().get("scalars", [])
    }


def _run_payload(run_root: Path) -> dict:
    event_dirs = _find_event_dirs(run_root)
    if not event_dirs:
        raise SystemExit(f"no event files under {run_root}")
    tags: dict[str, list[tuple[int, float, float]]] = {}
    for d in event_dirs:
        for tag, series in _load_scalars(d).items():
            tags.setdefault(tag, []).extend(series)
    for series in tags.values():
        series.sort(key=lambda row: (row[0], row[1]))
    return {"run": str(run_root), "tags": tags}


def _summary_payload(payload: dict) -> dict:
    """Collapse a run payload to last-value-per-tag + max step."""
    tags = {}
    for tag, series in sorted(payload["tags"].items()):
        if not series:
            continue
        step, wall_time, value = series[-1]
        tags[tag] = {
            "step": step,
            "wall_time": wall_time,
            "value": value,
            "points": len(series),
        }
    return {
        "run": payload["run"],
        "max_step": max((t["step"] for t in tags.values()), default=0),
        "tags": tags,
    }


def _print_summary(summary: dict) -> None:
    n_tags = len(summary["tags"])
    print(f"{summary['run']}  (max step {summary['max_step']}, {n_tags} tags)")
    width = max((len(t) for t in summary["tags"]), default=0)
    for tag, last in summary["tags"].items():
        print(f"  {tag:<{width}}  step {last['step']:>8}  {last['value']:>12.6g}")


def _dump(payload: dict, out: Path, jsonl: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with out.open("w") as f:
            for tag, series in payload["tags"].items():
                for step, wall_time, value in series:
                    f.write(
                        json.dumps(
                            {
                                "tag": tag,
                                "step": step,
                                "wall_time": wall_time,
                                "value": value,
                            }
                        )
                        + "\n"
                    )
    else:
        with out.open("w") as f:
            json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "runs",
        nargs="+",
        help="run directories (or a logs root combined with --all).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="treat each positional arg as a logs root and export every "
        "immediate subdirectory containing event files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (single-run only). Defaults to <run>/metrics.json "
        "(or .jsonl with --jsonl). Ignored with --all or --stdout.",
    )
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="emit one JSON object per scalar event instead of a nested dict.",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="print a single JSON array of run payloads to stdout and exit.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="print max-step + last value per tag instead of exporting the full "
        "series (combine with --stdout for the same digest as JSON).",
    )
    args = p.parse_args(argv)

    run_dirs: list[Path] = []
    if args.all:
        for root in args.runs:
            root_p = Path(root)
            if not root_p.is_dir():
                raise SystemExit(f"not a directory: {root_p}")
            for child in sorted(root_p.iterdir()):
                if child.is_dir() and _find_event_dirs(child):
                    run_dirs.append(child)
    else:
        run_dirs = [Path(r) for r in args.runs]

    if not run_dirs:
        raise SystemExit("no runs to export.")

    payloads = [_run_payload(r) for r in run_dirs]

    if args.summary:
        summaries = [_summary_payload(p) for p in payloads]
        if args.stdout:
            json.dump(
                summaries if len(summaries) > 1 else summaries[0], sys.stdout, indent=2
            )
            sys.stdout.write("\n")
        else:
            for i, summary in enumerate(summaries):
                if i:
                    print()
                _print_summary(summary)
        return 0

    if args.stdout:
        json.dump(payloads if len(payloads) > 1 else payloads[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for run, payload in zip(run_dirs, payloads):
        if args.out is not None and len(run_dirs) == 1:
            out = args.out
        else:
            suffix = ".jsonl" if args.jsonl else ".json"
            out = run / f"metrics{suffix}"
        _dump(payload, out, args.jsonl)
        n_series = len(payload["tags"])
        n_points = sum(len(v) for v in payload["tags"].values())
        shape = (
            "{tag: [[step, value], ...]}"
            if args.jsonl
            else "{'run': str, 'tags': {tag: [[step, wall_time, value], ...]}}"
        )
        print(f"{run} -> {out}  ({n_series} tags, {n_points} points)")
        print(f"  shape: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
