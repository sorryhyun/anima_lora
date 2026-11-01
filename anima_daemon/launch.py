"""Turn a ``Job`` into the argv and environment the worker spawns."""

from __future__ import annotations

import os

from .client import venv_python
from .jobs import Job


def build_cmd(job: Job) -> tuple[list[str], dict]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Force UTF-8 stdio in the job tree so a non-ASCII char (em-dash, etc.)
    # never crashes a child on a non-UTF-8 console locale (e.g. Korean
    # Windows cp949 → UnicodeEncodeError). Inherited by grandchildren.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # tqdm redraws at 0.1s cadence drown stdout.log's real lines; the GUI
    # tracker parses only the latest line and training has progress.jsonl.
    env.setdefault("TQDM_MININTERVAL", "10")

    # daemon-env ← captured_env ← extra_env: update, not setdefault, so the
    # caller's value wins; a command job's extra_env is applied after.
    env.update(job.captured_env or {})

    # Result envelope lift: bench/_common.write_result reads these to drop
    # a `result_path.json` pointer.
    env["ANIMA_DAEMON_JOB_ID"] = job.id
    env["ANIMA_DAEMON_JOB_DIR"] = str(job.dir)

    # Windowless interpreter (see client.venv_python). No --progress_jsonl: a
    # command job's loop finds its sink through ANIMA_DAEMON_JOB_DIR.
    if job.kind == "command":
        env.update(job.extra_env or {})
        return [venv_python(windowless=True), *job.argv], env

    # Lazy: the task runner's imports load only when a train job launches.
    from scripts.tasks._common import build_launch_cmd, build_method_args

    overrides = dict(job.overrides or {})
    extra = list(job.extra or [])
    # Dict overrides → --key value (unless already in extra). train.py bools
    # are `store_true`: a False override is dropped, so a caller can't force
    # a flag the config chain turns on back off here.
    for key, val in overrides.items():
        flag = f"--{key}"
        if flag in extra:
            continue
        if isinstance(val, bool):
            if val:
                extra.append(flag)
        elif key == "target_res" and isinstance(val, (list, tuple)):
            extra += [flag, *[str(v) for v in val]]
        else:
            extra += [flag, str(val)]
    if "--progress_jsonl" not in extra:
        extra += ["--progress_jsonl", job.progress_path or ""]
    if job.config_file:
        args = ["--config_file", job.config_file, *extra]
    else:
        args = build_method_args(
            job.method,
            preset=job.preset,
            methods_subdir=job.methods_subdir,
            extra=extra,
        )
    cmd = build_launch_cmd(*args, python_exe=venv_python(windowless=True))
    return cmd, env
