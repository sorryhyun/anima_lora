"""Shared helpers for ``scripts/tasks/*`` command modules.

Centralizes:
- ``ROOT`` (project root, regardless of where the calling module lives)
- ``PY`` resolution (venv-aware, pythonw.exe-safe)
- ``run`` / ``accelerate_launch`` / ``train`` subprocess helpers
- ``latest_output`` / ``latest_lora`` / ``latest_hydra`` checkpoint pickers
- ``INFERENCE_BASE`` — shared inference.py argv prefix
- ``_path`` / ``_preset`` config-overlay helpers
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _python_exe() -> str:
    """Resolve the venv's ``python.exe`` even if this process runs as pythonw.exe.

    Why python.exe and not just ``sys.executable``: when the GUI is launched
    via the desktop shortcut, sys.executable is pythonw.exe. pythonw children
    don't surface a working ``sys.stdout``/``sys.stderr`` to inherited pipes
    the way python.exe does — tqdm progress (which writes to stderr) silently
    drops, breaking the GUI's progress bar. python.exe + ``CREATE_NO_WINDOW``
    (set in ``run()`` when this process has no console) gives us both no
    console popup AND working stdio for grandchildren.
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

    Reads ``METHOD`` and ``METHODS_SUBDIR`` env vars so the GUI can point
    preprocess at the same variant file training will use (e.g.
    ``METHOD=lora METHODS_SUBDIR=gui-methods`` honors overrides written from
    ``ConfigTab``). Missing env vars → just base + preset.

    Defers the import of ``library.config.io`` so commands that don't touch
    preprocess (e.g. ``test-merge``) keep the module-load surface small.
    """
    global _PATH_OVERRIDES_CACHE
    if _PATH_OVERRIDES_CACHE is not None:
        return _PATH_OVERRIDES_CACHE
    sys.path.insert(0, str(ROOT))
    try:
        from library.config.io import load_path_overrides

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


def latest_output(prefix: str = "", exclude: str | None = None) -> Path:
    """Return the most recently modified .safetensors file in output/ckpt/ matching prefix.

    If `exclude` is given, any filename containing that substring is skipped. Useful
    to disambiguate overlapping prefixes (e.g. anima_postfix vs anima_postfix_exp).
    HydraLoRA multi-head sibling files (`*_moe.safetensors`) and backup files
    (containing `.bak.`) are always excluded.
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
    return latest_output()


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

    Prepends the venv's Scripts/bin directory to PATH (in both the child env
    and our own lookup) so venv-installed CLIs (``accelerate``, ``hf``, ...)
    resolve even when this process was started via a desktop shortcut that
    invokes ``pythonw.exe`` directly, bypassing venv activation.

    On Windows, ``subprocess.run`` uses the parent's PATH to locate the exe —
    setting ``env["PATH"]`` only affects the *child's* environment, not the
    lookup. We resolve the first arg to an absolute path with ``shutil.which``
    against the boosted PATH so the lookup works regardless.

    When this process has no console (pythonw.exe), Windows would allocate a
    new console for any console-subsystem child (python.exe, hf.exe, ...).
    We pass ``CREATE_NO_WINDOW`` to suppress that popup so GUI users don't
    see a terminal flash for every subprocess.
    """
    print(f"  > {' '.join(cmd)}")
    env = kwargs.pop("env", None)
    if env is None:
        env = os.environ.copy()
    venv_bin = str(Path(PY).parent)
    if venv_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    # Block-buffered stdio over pipes makes progress output (tqdm, training
    # logs) appear in chunks instead of streaming live. PYTHONUNBUFFERED keeps
    # children's Python stdio line-/un-buffered so the GUI sees output as it
    # happens. Inherited by grandchildren too.
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = list(cmd)
    if cmd and not Path(cmd[0]).is_absolute():
        resolved = shutil.which(cmd[0], path=env["PATH"])
        if resolved:
            cmd[0] = resolved
    if sys.platform == "win32" and not _has_console():
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        # Explicit stdio inheritance: when this process runs under pythonw.exe
        # (e.g. GUI shortcut), pythonw's fd 1/2 aren't exposed to children the
        # standard way — subprocess.run's default inheritance silently drops
        # the grandchild's output. Passing sys.stdout/sys.stderr directly hands
        # over Python's wrapped file objects, which DO route to the pipes our
        # parent (QProcess) set up. Only set when the caller hasn't.
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
    launch AND run ``nsys stats`` against the resulting report afterward.
    Returns ``(None, None)`` when PROFILE_STEPS is unset. Honors NSYS_OUT for
    the report path (default ``output/nsys/profile.nsys-rep``).

    Why ``--capture-range-end=stop`` (not ``stop-shutdown``) and ``--wait=primary``:
    the wrapped tree is ``nsys → accelerate launcher → train.py worker``. With
    ``stop-shutdown`` nsys SIGTERMs the launcher the moment ``cuProfilerStop``
    fires, the launcher dies before reaping the worker, the worker gets
    reparented to init, and the default ``--wait=all`` blocks forever waiting
    for it. Instead: the worker calls ``cuProfilerStop`` and then voluntarily
    ``sys.exit(0)`` (see ``library/training/loop.py`` ``_profiler_step_end``),
    the launcher exits naturally, and ``--wait=primary`` lets nsys finalize
    the report as soon as the launcher (its primary target) is gone — no
    leftover ``/tmp/*.qdstrm``.
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
    # Profile config tuned for kernel optimization + bottleneck analysis.
    #
    # Bottleneck-analysis additions (none of these need symbol downloads):
    #   --gpu-metrics-devices=cuda-visible  HW perf counters: SM occupancy,
    #       tensor-core util, DRAM/L2 bandwidth, warp stall reasons. The single
    #       most useful signal for "is this kernel compute- or memory-bound".
    #       nsys auto-picks the metric set (gb20x for Blackwell, ad10x for Ada,
    #       etc.); override with NSYS_GPU_METRICS_SET if needed.
    #   --gpu-metrics-frequency=10000       10 kHz sampling — fine enough to
    #       see per-step variation in a 3-step capture window.
    #   --cuda-graph-trace=node             per-node timing inside CUDA graphs
    #       (torch.compile emits these). Without it you only see the whole
    #       graph as one opaque blob.
    #   --cuda-memory-usage=true            tracks cudaMalloc/Free over time so
    #       you can correlate VRAM spikes with NVTX step ranges. Marked
    #       "significant runtime overhead" by nsys but fine inside a 3-step
    #       window — and essential for catching allocator thrash.
    #   --python-sampling=true @ 1 kHz      Python-side IP samples. Catches
    #       "Python is the bottleneck" cases (data loader, cache misses,
    #       config merging) that pure CUDA traces miss. Uses Python's own
    #       frame metadata, no debug-symbol download.
    #   --stats=true                        emit a sqlite next to the .nsys-rep
    #       so you can grep/SQL kernel timings without opening the GUI.
    #
    # Symbol-resolution is still OFF (--resolve-symbols=false + the three
    # *=none flags below). Without these, nsys finalize stalls for many
    # minutes on "Press Ctrl-C to stop symbol files downloading" reaching
    # out to NVIDIA's symbol servers — VRAM stays reserved, CPU sits at 0%.
    # The additions above are perf-counter and Python-frame data; none of
    # them need C++/CUDA-API symbol resolution.
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
    whole run if the perf-counter ioctl is restricted to root (the default on
    most distros — see ERR_NVGPUCTRPERM). Probing first lets us silently skip
    the flag instead of crashing the training task.
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
    return "Insufficient privilege" not in blob and "None of the installed GPUs" not in blob


# nsys stats reports auto-generated after profiling. Tuned for kernel
# optimization + bottleneck analysis on a per-step NVTX trace:
#   cuda_gpu_kern_sum     — top kernels by total GPU time (the "what to
#                           optimize" list)
#   nvtx_kern_sum         — kernels grouped under our `step=N` NVTX ranges
#                           (which step is slow + which kernels caused it)
#   cuda_gpu_mem_time_sum — host↔device mem ops by total time (catches
#                           transfer-bound steps, e.g. uncached latents)
#   cuda_gpu_mem_size_sum — same ops by bytes moved (cross-check time vs size
#                           to spot small-but-frequent thrash)
#   cuda_api_sum          — host-side CUDA API calls (cudaLaunchKernel,
#                           cudaStreamSynchronize blocking, etc.)
#   cuda_kern_exec_sum    — per-kernel queue/exec timings (launch overhead
#                           vs. on-GPU runtime — small kernels dominated by
#                           launch latency show up here)
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

    Writes one ``<stem>_<report>.txt`` per report into the same directory as
    the .nsys-rep. Best-effort: if the .nsys-rep didn't materialize (e.g. nsys
    aborted before finalizing) or stats fails, prints a warning and returns —
    a missing summary shouldn't fail the training task itself.
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
    out_prefix = rep_path.with_suffix("")  # strip .nsys-rep
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
    # Don't sys.exit on failure — best-effort summary, the .nsys-rep is the
    # canonical artifact and the GUI can always open it directly.
    try:
        subprocess.run(cmd, cwd=ROOT, check=False)
    except OSError as e:
        print(f"warn: nsys stats failed: {e}", file=sys.stderr)


def accelerate_launch(*args: str):
    """Launch training via accelerate with extra CLI args forwarded.

    Invoked as ``python -m accelerate.commands.accelerate_cli launch`` rather
    than the bare ``accelerate`` console-script. This keeps ``sys.executable``
    propagating from this process through to accelerate's workers — so when
    the GUI is launched via pythonw.exe (no console), the workers also run
    under pythonw.exe and don't pop terminal windows. The accelerate.exe
    shim hardcodes python.exe as the worker interpreter, defeating that.

    When PROFILE_STEPS is set, wraps the launch with ``nsys profile`` so
    ``make <method> PROFILE_STEPS=3-5`` produces a navigable Nsight report
    at ``output/nsys/profile.nsys-rep`` (override with NSYS_OUT). After the
    run, generates per-report textual summaries via ``nsys stats`` next to
    the .nsys-rep.
    """
    cmd = [
        PY,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--num_cpu_threads_per_process",
        "3",
        "--mixed_precision",
        "bf16",
        "train.py",
        *args,
    ]
    nsys_prefix, nsys_out = _nsys_wrapper()
    if nsys_prefix is not None:
        cmd = nsys_prefix + ["--"] + cmd
    run(cmd)
    if nsys_out is not None:
        _nsys_run_stats(nsys_out)


def train(
    method: str, extra, preset: str | None = None, methods_subdir: str | None = None
):
    """Launch training for a given method + preset (PRESET env overrides default).

    `methods_subdir` selects the folder under `configs/` that holds the method
    file (default ``"methods"``; pass ``"gui-methods"`` for the clean per-variant
    files used by the `lora-gui` path).

    ARTIST env var trains an artist-only LoRA — equivalent to passing
    `--artist_filter <name>` (filters dataset to `@<name>`-tagged captions and
    redirects output to `output/ckpt-artist/`).
    """
    args = ["--method", method, "--preset", preset or _preset()]
    if methods_subdir:
        args += ["--methods_subdir", methods_subdir]
    artist = os.environ.get("ARTIST")
    if artist and not any(a == "--artist_filter" for a in extra):
        args += ["--artist_filter", artist]
    profile_steps = os.environ.get("PROFILE_STEPS")
    if profile_steps and not any(a == "--profile_steps" for a in extra):
        args += ["--profile_steps", profile_steps]
    accelerate_launch(*args, *extra)


INFERENCE_BASE = [
    PY,
    "inference.py",
    "--dit",
    "models/diffusion_models/anima-preview3-base.safetensors",
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
    "masterpiece, best quality, score_7, safe, An anime girl wearing a black tank-top"
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
    "1.0",
    "--sampler",
    "er_sde",
    "--guidance_scale",
    "4.0",
    "--seed",
    "42",
    "--save_path",
    "output/tests",
]
