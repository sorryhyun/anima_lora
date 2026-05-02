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


def _nsys_wrapper() -> list[str] | None:
    """Build an ``nsys profile`` prefix when PROFILE_STEPS is set.

    Returns None unless PROFILE_STEPS is set. Honors NSYS_OUT for the report
    path (default ``output/profile.nsys-rep``). ``stop-shutdown`` makes nsys
    finalize the report and exit when ``torch.cuda.profiler.stop()`` fires
    inside ``train.py``, so the file lands on disk without waiting for the
    rest of training to complete.
    """
    if not os.environ.get("PROFILE_STEPS"):
        return None
    nsys = shutil.which("nsys")
    if nsys is None:
        print(
            "warn: PROFILE_STEPS set but `nsys` not found on PATH; "
            "running without profiler wrapper",
            file=sys.stderr,
        )
        return None
    out = os.environ.get("NSYS_OUT", "output/profile.nsys-rep")
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  > nsys report -> {out_path}")
    return [
        nsys,
        "profile",
        "-o",
        str(out_path.with_suffix("")),  # nsys appends .nsys-rep
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--trace=cuda,nvtx,osrt,cudnn,cublas",
    ]


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
    at ``output/profile.nsys-rep`` (override with NSYS_OUT).
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
    nsys_prefix = _nsys_wrapper()
    if nsys_prefix is not None:
        cmd = nsys_prefix + ["--"] + cmd
    run(cmd)


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
    "30",
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
