"""Shared helpers for ``scripts/tasks/*`` command modules.

Centralizes:
- ``ROOT`` (project root, regardless of where the calling module lives)
- ``PY`` resolution (venv-aware, pythonw.exe-safe)
- ``run`` / ``build_launch_cmd`` / ``accelerate_launch`` / ``train`` subprocess helpers
- ``latest_output`` / ``latest_lora`` / ``latest_hydra`` checkpoint pickers
- ``INFERENCE_BASE`` — shared inference.py argv prefix
- ``_path`` / ``_preset`` config-overlay helpers
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The ``anime_tools`` seam: every wrapper here was written against one version
# of the package's CLI/stdio surface (``anime_tools.contract``, stdlib-only —
# the sentinels, modes, file sets and request flags the trainer spells). The
# package bumps ``CONTRACT_VERSION`` on an incompatible change; a mismatch
# means the pinned rev in ``pyproject.toml`` moved past what this code was
# audited against (or the trainer moved and the pin did not), so fail every
# target up front instead of minutes into a GPU job.
ANIME_TOOLS_CONTRACT_VERSION = 1


def _check_anime_tools_contract() -> None:
    try:
        from anime_tools.contract import CONTRACT_VERSION
    except ImportError:
        # Not installed: each curation target fails on its own ``-m`` with the
        # package's name in the error, which is the clearer message.
        return
    if CONTRACT_VERSION != ANIME_TOOLS_CONTRACT_VERSION:
        raise SystemExit(
            f"anime_tools contract version {CONTRACT_VERSION} != "
            f"{ANIME_TOOLS_CONTRACT_VERSION} the task runner was written for. "
            "Bump the anime_tools pin in pyproject.toml ([tool.uv.sources]) and "
            "ANIME_TOOLS_CONTRACT_VERSION in scripts/tasks/_common.py together, "
            "re-auditing scripts/tasks/{masking,preprocess,curate,tagger}.py "
            "against the package's contract (docs/proposal/anime_tools_api_first.md)."
        )


_check_anime_tools_contract()

# Shared so the shipped and experimental inference modules agree on what counts
# as a REF_IMAGE for the test-* commands.
_REF_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _python_exe() -> str:
    """Resolve the venv's ``python.exe`` even if this process runs as pythonw.exe.

    pythonw children don't surface a working ``sys.stdout``/``sys.stderr`` to
    inherited pipes the way python.exe does — tqdm progress silently drops,
    breaking the GUI's progress bar. python.exe + ``CREATE_NO_WINDOW`` (set in
    ``run()``) gives both no console popup and working stdio for grandchildren.
    """
    if sys.platform == "win32":
        cand = Path(sys.executable).with_name("python.exe")
        if cand.exists():
            return str(cand)
    return sys.executable


PY = _python_exe()


def _preset(default: str = "default") -> str:
    return os.environ.get("PRESET", default)


_PATH_OVERRIDES_CACHE: dict | None = None


def _path_overrides() -> dict:
    """Top-level path scalars from base.toml → preset → method file (cached).

    Reads ``METHOD``/``METHODS_SUBDIR`` env vars so the GUI can point
    preprocess at the same variant file training will use. Missing env vars →
    just base + preset. Defers the ``library.config.io`` import so commands
    that don't touch preprocess keep the module-load surface small.
    """
    global _PATH_OVERRIDES_CACHE
    if _PATH_OVERRIDES_CACHE is not None:
        return _PATH_OVERRIDES_CACHE
    try:
        from library.config.io import (
            load_path_overrides,
            load_path_overrides_from_config,
        )

        config_file = os.environ.get("CONFIG_FILE")
        if config_file:
            _PATH_OVERRIDES_CACHE = load_path_overrides_from_config(config_file)
        else:
            _PATH_OVERRIDES_CACHE = load_path_overrides(
                preset=_preset(),
                method=os.environ.get("METHOD") or None,
                methods_subdir=os.environ.get("METHODS_SUBDIR") or "methods",
            )
    except Exception as e:  # noqa: BLE001 — fall back silently to defaults
        print(f"warn: could not read base.toml path overrides: {e}", file=sys.stderr)
        _PATH_OVERRIDES_CACHE = {}
    return _PATH_OVERRIDES_CACHE


def _path(key: str, default: str) -> str:
    """Fetch a path key from base.toml/preset overrides, with hardcoded fallback."""
    val = _path_overrides().get(key, default)
    return str(val) if val is not None else default


def bespoke_preset_flags(preset: str) -> list[str]:
    """Translate ``configs/presets.toml[<preset>]`` into CLI flags for the
    bespoke distillation loops (``scripts/distill_turbo``/``scripts/distill_cjk``)
    that bypass ``train.py``'s config merge chain.

    Honored keys: ``blocks_to_swap`` → ``--blocks_to_swap N``;
    ``gradient_checkpointing`` (bool) → ``--grad_ckpt``/``--no_grad_ckpt``
    (defaults to ``--no_grad_ckpt`` when omitted — these trainable footprints
    are tiny, so ckpt is a pure perf loss when VRAM isn't tight);
    ``sample_ratio`` → ``--sample_ratio R``. Other preset keys are dropped.
    """
    try:
        from library.config.io import load_preset_section
    except Exception as e:  # noqa: BLE001
        print(f"warn: could not import preset loader: {e}", file=sys.stderr)
        return ["--no_grad_ckpt"]

    try:
        section = load_preset_section(preset)
    except (FileNotFoundError, KeyError) as e:
        print(
            f"warn: preset '{preset}' not found ({e}); using bespoke-loop defaults",
            file=sys.stderr,
        )
        return ["--no_grad_ckpt"]

    flags: list[str] = []
    if "blocks_to_swap" in section:
        flags += ["--blocks_to_swap", str(int(section["blocks_to_swap"]))]
    if "gradient_checkpointing" in section:
        flags.append(
            "--grad_ckpt" if section["gradient_checkpointing"] else "--no_grad_ckpt"
        )
    else:
        flags.append("--no_grad_ckpt")
    if "sample_ratio" in section:
        flags += ["--sample_ratio", str(float(section["sample_ratio"]))]
    return flags


def latest_output(prefix: str = "", exclude: str | None = None) -> Path:
    """Most recently modified .safetensors file in output/ckpt/ matching prefix.

    `exclude`, if given, skips any filename containing that substring (e.g.
    disambiguating anima_postfix vs anima_postfix_exp). HydraLoRA multi-head
    sibling files (`*_moe.safetensors`) and `.bak.` files are always excluded.
    """
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("*.safetensors")
            if f.name.startswith(prefix)
            and not f.name.endswith("_moe.safetensors")
            and ".bak." not in f.name
            and (exclude is None or exclude not in f.name)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        label = f"'{prefix}*.safetensors'" if prefix else "*.safetensors"
        print(f"No {label} files found in output/ckpt/", file=sys.stderr)
        sys.exit(1)
    return outputs[0]


def latest_lora() -> Path:
    # Exclude pooled_text_proj heads: not LoRAs (resolved separately by MOD=1),
    # but a blind newest-`.safetensors` pick would grab them after a mod-guidance
    # distill (`project/finished/mod_guidance/`).
    return latest_output(exclude="pooled_text_proj")


def latest_hydra() -> Path:
    """Latest HydraLoRA multi-head file (`anima_hydra*_moe.safetensors`)."""
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("anima_hydra*_moe.safetensors")
            if ".bak." not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print(
            "No 'anima_hydra*_moe.safetensors' files found in output/ckpt/ "
            "(enable the HydraLoRA block in configs/methods/lora.toml and run `make lora`)",
            file=sys.stderr,
        )
        sys.exit(1)
    return outputs[0]


def _random_ref_image(directory: Path) -> str | None:
    """Pick a random image under ``directory`` (recursive), or ``None`` if empty.

    Recurses because source layouts nest images under per-artist subdirs.
    Shared by the EasyControl/IP-Adapter/DirectEdit test commands as the
    fallback reference when no REF_IMAGE is supplied.
    """
    if not directory.is_dir():
        return None
    pool = [p for p in directory.rglob("*") if p.suffix.lower() in _REF_IMAGE_EXTS]
    if not pool:
        return None
    pick = random.choice(pool)
    print(f"  > Random ref: {pick}")
    return str(pick)


def _has_console() -> bool:
    """True if this process is attached to a Windows console (or is non-Windows).

    Used to decide whether to suppress new console popups for child processes.
    A pythonw.exe-launched process (e.g. desktop GUI shortcut) has no console.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001 — err on the safe side: keep output visible
        return True


def run(cmd: list[str], **kwargs):
    """Run a subprocess, exit on failure.

    Prepends the venv's Scripts/bin directory to PATH (child env + our own
    lookup) so venv CLIs (``accelerate``, ``hf``, ...) resolve even when
    launched via a desktop shortcut that invokes ``pythonw.exe`` directly,
    bypassing venv activation. On Windows, ``subprocess.run`` uses the
    parent's PATH to locate the exe (setting ``env["PATH"]`` alone doesn't
    affect lookup), so we resolve the first arg with ``shutil.which`` against
    the boosted PATH instead.

    When this process has no console (pythonw.exe), Windows would pop a new
    console for any console-subsystem child; pass ``CREATE_NO_WINDOW`` to
    suppress it.
    """
    print(f"  > {' '.join(cmd)}")
    env = kwargs.pop("env", None)
    if env is None:
        env = os.environ.copy()
    # Curation stages (anime_tools) anchor bare relative defaults on
    # ANIMA_HOME; pin it to this checkout so they resolve as before the split.
    env.setdefault("ANIMA_HOME", str(ROOT))
    # Bound `hf` CLI socket timeouts so a stalled download can't hang the serial
    # daemon queue (a hung-not-failed fetch wedges every job queued behind it,
    # training included). setdefault means an explicit ANIMA_HF_TIMEOUT/HF_HUB_*
    # still wins.
    if cmd and os.path.basename(str(cmd[0])) in ("hf", "hf.exe"):
        from library.runtime.hf_download import ensure_hf_timeouts

        ensure_hf_timeouts()
        env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", os.environ["HF_HUB_DOWNLOAD_TIMEOUT"])
        env.setdefault("HF_HUB_ETAG_TIMEOUT", os.environ["HF_HUB_ETAG_TIMEOUT"])
    venv_bin = str(Path(PY).parent)
    if venv_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    # Un-buffer children's stdio so the GUI sees tqdm/training output live.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Force UTF-8 stdio regardless of console locale: on non-UTF-8 consoles
    # (e.g. Korean Windows cp949) a child printing an em-dash crashes.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = list(cmd)
    if cmd and not Path(cmd[0]).is_absolute():
        resolved = shutil.which(cmd[0], path=env["PATH"])
        if resolved:
            cmd[0] = resolved
    if sys.platform == "win32" and not _has_console():
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        # Under pythonw.exe fd 1/2 aren't exposed to children the standard way,
        # so default inheritance drops grandchild output; pass Python's wrapped
        # stdout/stderr instead, which route to our parent's pipes.
        if sys.stdout is not None:
            kwargs.setdefault("stdout", sys.stdout)
        if sys.stderr is not None:
            kwargs.setdefault("stderr", sys.stderr)
    result = subprocess.run(cmd, cwd=kwargs.pop("cwd", ROOT), env=env, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _nsys_wrapper() -> tuple[list[str], Path] | tuple[None, None]:
    """Build an ``nsys profile`` prefix when PROFILE_STEPS is set.

    Returns ``(prefix, out_path)`` when active so the caller can both wrap the
    launch AND run ``nsys stats`` afterward; ``(None, None)`` when unset.
    Honors NSYS_OUT for the report path (default ``output/nsys/profile.nsys-rep``).

    GOTCHA: must use ``--capture-range-end=stop`` (NOT ``stop-shutdown``) with
    ``--wait=primary`` — ``stop-shutdown`` SIGTERMs the accelerate launcher the
    moment ``cuProfilerStop`` fires, before it reaps the train.py worker, which
    then gets reparented and the default ``--wait=all`` hangs forever. The
    worker instead calls ``cuProfilerStop`` then voluntarily exits (see
    ``library/training/loop.py`` ``_profiler_step_end``), letting the launcher
    exit naturally so ``--wait=primary`` finalizes cleanly.
    """
    if not os.environ.get("PROFILE_STEPS"):
        return None, None
    nsys = shutil.which("nsys")
    if nsys is None:
        print(
            "warn: PROFILE_STEPS set but `nsys` not found on PATH; "
            "running without profiler wrapper",
            file=sys.stderr,
        )
        return None, None
    out = os.environ.get("NSYS_OUT", "output/nsys/profile.nsys-rep")
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  > nsys report -> {out_path}")
    # Profile config tuned for kernel + bottleneck analysis. Symbol-resolution
    # is OFF: otherwise nsys finalize stalls for minutes reaching NVIDIA's
    # symbol servers while VRAM stays reserved.
    metrics_set = os.environ.get("NSYS_GPU_METRICS_SET")
    cmd = [
        nsys,
        "profile",
        "-o",
        str(out_path.with_suffix("")),  # nsys appends .nsys-rep
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--wait=primary",
        "--trace=cuda,nvtx,cudnn,cublas",
        "--cuda-graph-trace=node",
        "--cuda-memory-usage=true",
        "--python-sampling=true",
        "--python-sampling-frequency=1000",
        "--stats=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--cudabacktrace=none",
        "--resolve-symbols=false",
    ]
    if _nsys_gpu_metrics_available(nsys):
        cmd += [
            "--gpu-metrics-devices=cuda-visible",
            "--gpu-metrics-frequency=10000",
        ]
        if metrics_set:
            cmd.append(f"--gpu-metrics-set={metrics_set}")
    else:
        print(
            "  > nsys: GPU metrics disabled (perf counters restricted to admin). "
            "To enable SM occupancy / tensor-core / memory-bandwidth counters:\n"
            "      sudo tee /etc/modprobe.d/nvidia-perf.conf <<<'options nvidia "
            '"NVreg_RestrictProfilingToAdminUsers=0"\'\n'
            "      sudo update-initramfs -u && sudo reboot\n"
            "    See https://developer.nvidia.com/ERR_NVGPUCTRPERM",
            file=sys.stderr,
        )
    return cmd, out_path


def _nsys_gpu_metrics_available(nsys: str) -> bool:
    """Probe whether nsys can collect GPU metrics on this host.

    nsys validates ``--gpu-metrics-devices`` at argv-parse time and aborts the
    whole run if the perf-counter ioctl is restricted to root (default on
    most distros — ERR_NVGPUCTRPERM). Probing first lets us skip the flag
    instead of crashing the training task.
    """
    try:
        out = subprocess.run(
            [nsys, "profile", "--gpu-metrics-devices=help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    blob = (out.stdout or "") + (out.stderr or "")
    return (
        "Insufficient privilege" not in blob
        and "None of the installed GPUs" not in blob
    )


# nsys stats reports auto-generated after profiling, tuned for kernel +
# bottleneck analysis on a per-step NVTX trace.
_NSYS_STATS_REPORTS = (
    "cuda_gpu_kern_sum",
    "nvtx_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_api_sum",
    "cuda_kern_exec_sum",
)


def _nsys_run_stats(rep_path: Path) -> None:
    """Generate textual ``nsys stats`` reports next to the .nsys-rep.

    Writes one ``<stem>_<report>.txt`` per report. Best-effort: a missing
    .nsys-rep or a stats failure just warns — shouldn't fail the training task.
    """
    if not rep_path.exists():
        print(
            f"warn: nsys report not found at {rep_path}; skipping stats",
            file=sys.stderr,
        )
        return
    nsys = shutil.which("nsys")
    if nsys is None:
        return
    out_prefix = rep_path.with_suffix("")
    cmd = [
        nsys,
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--format=column",
        "--output",
        str(out_prefix),
    ]
    for report in _NSYS_STATS_REPORTS:
        cmd += ["--report", report]
    cmd.append(str(rep_path))
    print(f"  > nsys stats -> {out_prefix.parent}/")
    # Best-effort: the .nsys-rep is the canonical artifact, so don't sys.exit.
    try:
        subprocess.run(cmd, cwd=ROOT, check=False)
    except OSError as e:
        print(f"warn: nsys stats failed: {e}", file=sys.stderr)


def build_launch_cmd(*args: str, python_exe: str | None = None) -> list[str]:
    """Build the ``train.py`` launch command list (no side effects).

    Pure command construction, extracted from ``accelerate_launch`` so other
    spawners (the training daemon under ``anima_daemon/``) can ``Popen`` the
    same command themselves — detached, own stdio/process-tree monitoring —
    instead of going through ``run()``'s blocking path. The nsys wrapper stays
    in ``accelerate_launch`` (CLI-only; the daemon never applies it).

    Single-GPU fast path (default): invoke ``train.py`` directly, skipping the
    ``accelerate launch`` bootstrap (a second full Python process importing
    accelerate/torch just to spawn one local worker). ``train.py`` builds its
    own single-process ``Accelerator()`` and reads ``mixed_precision`` from the
    config chain itself. Set ``ANIMA_ACCELERATE_LAUNCH=1`` to force the
    accelerate launcher, which multi-GPU/distributed runs genuinely need.

    ``python_exe`` overrides the launching interpreter (default ``PY`` =
    python.exe). GOTCHA: the detached daemon passes ``pythonw.exe`` here — a
    uv-venv python.exe is a trampoline that re-execs the real interpreter, and
    ``CREATE_NO_WINDOW`` doesn't survive that re-exec, so a python.exe worker
    would pop a console window that kills the job (``STATUS_CONTROL_C_EXIT``)
    when closed. pythonw.exe never allocates a console, and the daemon
    redirects stdout/stderr to a file so stdio still works.

    The accelerate path invokes ``python -m accelerate.commands.accelerate_cli
    launch`` rather than the bare ``accelerate`` console-script, so the
    launching interpreter propagates to accelerate's workers via
    ``sys.executable`` (the accelerate.exe shim hardcodes python.exe otherwise).
    """
    py = python_exe or PY
    if not os.environ.get("ANIMA_ACCELERATE_LAUNCH"):
        return [py, "train.py", *args]
    # Forward --mixed_precision to the launcher so it matches train.py's Accelerator().
    mixed_precision = "bf16"
    for i, a in enumerate(args):
        if a == "--mixed_precision" and i + 1 < len(args):
            mixed_precision = args[i + 1]
        elif a.startswith("--mixed_precision="):
            mixed_precision = a.split("=", 1)[1]
    return [
        py,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--num_cpu_threads_per_process",
        "3",
        "--mixed_precision",
        mixed_precision,
        "train.py",
        *args,
    ]


def accelerate_launch(*args: str):
    """Launch training via accelerate with extra CLI args forwarded (blocking).

    Builds the command via ``build_launch_cmd`` then runs it through ``run``.
    When PROFILE_STEPS is set, wraps the launch with ``nsys profile`` so
    ``make <method> PROFILE_STEPS=3-5`` produces a navigable Nsight report at
    ``output/nsys/profile.nsys-rep`` (override with NSYS_OUT), then generates
    textual summaries via ``nsys stats``.
    """
    cmd = build_launch_cmd(*args)
    nsys_prefix, nsys_out = _nsys_wrapper()
    if nsys_prefix is not None:
        cmd = nsys_prefix + ["--"] + cmd
    run(cmd)
    if nsys_out is not None:
        _nsys_run_stats(nsys_out)


def build_method_args(
    method: str,
    *,
    preset: str,
    methods_subdir: str | None = None,
    extra=None,
    artist: str | None = None,
    profile_steps: str | None = None,
) -> list[str]:
    """Assemble the ``["--method", m, "--preset", p, ...]`` train.py arg list.

    Pure — no env reads, no subprocess. Shared by the CLI ``train()`` path and
    the training daemon so the daemon doesn't duplicate the ARTIST/
    PROFILE_STEPS handling. ``artist``/``profile_steps`` add their flags only
    when the caller didn't already pass them in ``extra``.
    """
    extra = list(extra or [])
    args = ["--method", method, "--preset", preset]
    if methods_subdir:
        args += ["--methods_subdir", methods_subdir]
    if artist and not any(a == "--artist_filter" for a in extra):
        args += ["--artist_filter", artist]
    if profile_steps and not any(a == "--profile_steps" for a in extra):
        args += ["--profile_steps", profile_steps]
    return [*args, *extra]


# --- attach-by-default run modes ---------------------------------------------
# Every GPU-touching CLI target routes through the daemon by default: submit,
# then stream the job's stdout to this terminal (attach). Three modes,
# resolved by _resolve_run_mode:
#   attach (default) — submit + stream + exit with the job's exit code; ctrl-C
#                      DETACHES (job survives), it never kills the run.
#   detach           — submit + return (the --queue behaviour, for overnight sweeps).
#   inline           — run the child directly, no daemon (the debugging path:
#                      pdb / py-spy / faulthandler / nsys all want a direct
#                      child). Forced automatically for PROFILE_STEPS / accelerate.
_RUN_MODE_FLAGS = {
    "--inline": "inline",
    "--queue": "detach",
    "--detach": "detach",
    "--attach": "attach",
}


def _resolve_run_mode(extra: list[str]) -> tuple[str, list[str]]:
    """Pop any run-mode flag from ``extra`` and return ``(mode, extra)``.

    Precedence: explicit ``--inline/--queue/--detach/--attach`` flag > the
    ``ANIMA_RUN_MODE`` env var > attach default. When implicit,
    ``PROFILE_STEPS``/``ANIMA_ACCELERATE_LAUNCH`` force inline so the default
    attach path never silently drops them. An explicit flag always wins.
    """
    extra = list(extra)
    flagged: str | None = None
    for flag, mode in _RUN_MODE_FLAGS.items():
        while flag in extra:
            extra.remove(flag)
            flagged = mode
    if flagged is not None:
        return flagged, extra
    mode = os.environ.get("ANIMA_RUN_MODE") or "attach"
    if mode not in ("attach", "detach", "inline"):
        mode = "attach"
    if mode != "inline" and (
        os.environ.get("PROFILE_STEPS") or os.environ.get("ANIMA_ACCELERATE_LAUNCH")
    ):
        mode = "inline"
    return mode, extra


def _safe_job(cl, job_id: str) -> dict | None:
    try:
        return cl.get(job_id)
    except Exception:  # noqa: BLE001 — a down/racing daemon is expected here
        return None


def _attach_hints(job_id: str) -> str:
    return (
        f"  make daemon-attach JOB={job_id}   # re-attach to this job\n"
        f"  make daemon-kill JOB={job_id}     # stop it\n"
        f"  make daemon-status                # queue overview"
    )


# Whether this process has already printed the queued-job cheat-sheet. Queueing
# a grid in one `for` loop repeated five lines of hints per job and buried the
# one line the reader wanted ("what did the queue just get?"), so the hints go
# out once and every later submit prints only its own `queued job` line.
_QUEUED_HINTS_SHOWN = False


def _print_queued(cl, job_id: str, desc: str) -> None:
    global _QUEUED_HINTS_SHOWN
    line = f"queued job {job_id} ({desc}). daemon: {cl.base}"
    if _QUEUED_HINTS_SHOWN:
        print(line)
        return
    _QUEUED_HINTS_SHOWN = True
    print(
        f"{line}\n"
        f"  make daemon-attach JOB={job_id}   # follow this job's output\n"
        f"  make daemon-attach                # follow queue/lifecycle events\n"
        f"  make daemon-kill JOB={job_id}     # cancel it\n"
        f"  make daemon-terminate             # stop the daemon + discard queue"
    )


def _exit_code_for(cl, job_id: str, *, settle: float = 10.0) -> int:
    """The job's exit code once terminal: its OS ``returncode`` when known,
    else done→0/other→1. A signal death (negative rc) maps to ``128+N`` so
    ``&&`` chains see a normal nonzero."""
    deadline = time.time() + settle
    job = _safe_job(cl, job_id)
    while job and job.get("state") in ("queued", "running") and time.time() < deadline:
        time.sleep(0.3)
        job = _safe_job(cl, job_id)
    if job is None:
        return 1
    rc = job.get("returncode")
    if isinstance(rc, int):
        return 128 + (-rc) if rc < 0 else rc
    return 0 if job.get("state") == "done" else 1


def _attach_and_wait(cl, job_id: str) -> int:
    """Stream a submitted job's stdout to this terminal; return its exit code.

    Ctrl-C DETACHES — the job keeps running (same contract as
    ``make daemon-attach``); prints the re-attach one-liners and returns 0.
    Tolerates a daemon restart mid-stream: on a dropped stream it re-resolves
    the pidfile and re-attaches, skipping lines already printed (the on-disk
    ``stdout.log`` replays from the start).
    """
    from anima_daemon.client import DaemonClient

    # flush: stdout to a pipe is block-buffered, so an unflushed banner makes a
    # piped attach look like zero bytes (i.e. indistinguishable from a hang).
    print(
        f"\nattached to job {job_id} ({cl.base}) — ctrl-C detaches "
        "(the job keeps running)\n",
        flush=True,
    )
    emitted = 0  # log lines already printed, across reconnects
    try:
        while True:
            seen = 0
            got_eof = False
            try:
                for payload in cl.stream_logs(job_id):
                    # The tail SSE ends with a {"ev":"eof","state":…} sentinel.
                    if payload.startswith("{") and '"eof"' in payload:
                        try:
                            obj = json.loads(payload)
                        except ValueError:
                            obj = None
                        if isinstance(obj, dict) and obj.get("ev") == "eof":
                            got_eof = True
                            break
                    seen += 1
                    if seen <= emitted:
                        continue  # replayed line from before a reconnect
                    emitted = seen
                    print(payload, flush=True)
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001 — socket drop / daemon restart
                pass
            if got_eof:
                break
            # Stream ended without eof: either the job is terminal (stop) or the
            # daemon restarted. Re-resolve its port and re-check; re-attach only
            # while the job is still live.
            job = _safe_job(cl, job_id)
            if job is None:
                cl = DaemonClient()  # port may have moved on restart
                job = _safe_job(cl, job_id)
            if job is None or job.get("state") not in ("queued", "running"):
                break
            time.sleep(0.5)
            cl = DaemonClient()
    except KeyboardInterrupt:
        print(f"\ndetached (job {job_id} continues).")
        print(_attach_hints(job_id))
        return 0
    return _exit_code_for(cl, job_id)


def queue_command(label: str, argv: list[str]) -> None:
    """Enqueue a bespoke-loop distillation command on the local daemon.

    The bespoke loops (``scripts/distill_turbo``/``scripts/distill_cjk``)
    bypass ``train.py``, so they submit a plain ``kind="command"`` job
    (detach semantics) instead of riding the train ``--queue`` path. ``argv``
    is run as ``[venv_python, *argv]`` from the repo root (module form:
    ``["-m", "scripts.distill_turbo.distill", ...]``). Preset/CLI flags must
    be baked into ``argv`` here — the command-job path does no config merging.
    """
    from anima_daemon import client as _daemon_client

    cl = _daemon_client.ensure_daemon()
    resp = cl.submit_command(label=label, argv=list(argv))
    job_id = resp.get("job_id")
    print(
        f"queued job {job_id} (label={label}). daemon: {cl.base}\n"
        f"  make daemon-attach JOB={job_id}   # follow this job's output\n"
        f"  make daemon-attach                # follow queue/lifecycle events\n"
        f"  make daemon-kill JOB={job_id}     # cancel it\n"
        f"  make daemon-terminate             # stop the daemon + discard queue"
    )


def run_command(
    label: str,
    argv: list[str],
    *,
    mode: str = "attach",
    stall_timeout: float | None = None,
) -> None:
    """Run a GPU command job through the daemon, attach-by-default.

    The generic chokepoint for non-train GPU work (bench/``test-*``/``gen``):
    submit a ``kind="command"`` job and stream its stdout to this terminal,
    exiting with the job's exit code (ctrl-C detaches, the run survives).
    Same three modes as ``train`` (``_resolve_run_mode``): ``attach``
    (default), ``detach`` (``--queue``), ``inline`` (no daemon; debugging path).

    ``argv`` is the child argv **without** the interpreter — the daemon
    prepends its resolved venv python, and the inline path prepends ``PY``.
    The daemon exports ``ANIMA_DAEMON_JOB_DIR`` so a script that emits a bench
    envelope / generation manifest gets result-lifted; inline it's a plain run.

    ``stall_timeout`` overrides the daemon's 120s command-job stall budget for
    this job (0 disables it) for a loop that legitimately prints nothing for
    minutes.
    """
    if mode == "inline":
        run([PY, *argv])
        return

    from anima_daemon import client as _daemon_client

    cl = _daemon_client.ensure_daemon()
    job_id = cl.submit_command(
        label=label, argv=list(argv), stall_timeout=stall_timeout
    ).get("job_id")
    if mode == "detach":
        _print_queued(cl, job_id, f"command={label}")
        return
    rc = _attach_and_wait(cl, job_id)
    if rc:
        sys.exit(rc)


def train(
    method: str, extra, preset: str | None = None, methods_subdir: str | None = None
):
    """Launch training for a given method + preset (PRESET env overrides default).

    `methods_subdir` selects the folder under `configs/` that holds the method
    file (default ``"methods"``; pass ``"gui-methods"`` for the clean
    per-variant files used by the `lora-gui` path).

    ARTIST env var trains an artist-only LoRA — equivalent to
    `--artist_filter <name>` (filters dataset to `@<name>`-tagged captions and
    redirects output to `output/ckpt-artist/`).

    Run mode (``_resolve_run_mode``): by default the job is submitted to the
    local daemon and this terminal *attaches* to its stdout, exiting with the
    job's exit code (ctrl-C detaches, the run survives). ``--queue`` detaches
    (submit + return); ``--inline`` runs the child directly with no daemon.
    ``ANIMA_RUN_MODE`` sets the default; ``PROFILE_STEPS``/
    ``ANIMA_ACCELERATE_LAUNCH`` force inline.
    """
    preset = preset or _preset()
    extra = list(extra or [])
    artist = os.environ.get("ARTIST")
    profile_steps = os.environ.get("PROFILE_STEPS")

    mode, extra = _resolve_run_mode(extra)

    if mode == "inline":
        args = build_method_args(
            method,
            preset=preset,
            methods_subdir=methods_subdir,
            extra=extra,
            artist=artist,
            profile_steps=profile_steps,
        )
        accelerate_launch(*args)
        return

    # Daemon paths (attach/detach). Fold ARTIST/PROFILE_STEPS into extra as
    # explicit flags: the daemon's own build_method_args doesn't read env
    # vars, so without folding a queued artist run would silently train the
    # full dataset.
    if artist and "--artist_filter" not in extra:
        extra += ["--artist_filter", artist]
    if profile_steps and "--profile_steps" not in extra:
        extra += ["--profile_steps", profile_steps]

    from anima_daemon import client as _daemon_client

    cl = _daemon_client.ensure_daemon()
    job_id = cl.submit(
        method=method,
        preset=preset,
        methods_subdir=methods_subdir,
        extra=extra,
    ).get("job_id")

    if mode == "detach":
        _print_queued(cl, job_id, f"method={method}, preset={preset}")
        return
    rc = _attach_and_wait(cl, job_id)
    if rc:
        sys.exit(rc)


INFERENCE_BASE = [
    PY,
    "inference.py",
    "--dit",
    "models/diffusion_models/anima-base-v1.0.safetensors",
    "--text_encoder",
    "models/text_encoders/qwen_3_06b_base.safetensors",
    "--vae",
    "models/vae/qwen_image_vae.safetensors",
    "--vae_chunk_size",
    "64",
    "--vae_disable_cache",
    "--attn_mode",
    "flash",  # flash4 not supported yet (flash-attention-sm120 disabled)
    "--lora_multiplier",
    "1.0",
    "--prompt",
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds.",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size",
    "1024",
    "1024",
    "--infer_steps",
    "28",
    "--flow_shift",
    "3.0",
    "--sampler",
    "euler",
    "--guidance_scale",
    "4.0",
    "--seed",
    "42",
    "--save_path",
    "output/tests",
]


def override_arg(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace a ``--flag VALUE`` (or ``--flag V1 V2``) pair in argv with a
    fresh ``--flag value`` pair. Used to retarget ``INFERENCE_BASE`` defaults
    (e.g. the turbo contract: 4 steps, cfg=1.0) without rewriting the whole list.
    """
    if flag not in argv:
        return argv + [flag, value]
    i = argv.index(flag)
    return argv[:i] + [flag, value] + argv[i + 2 :]
