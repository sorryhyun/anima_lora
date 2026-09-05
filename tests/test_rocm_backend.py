import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import library.runtime.backend as backend
from library.runtime.backend import (
    diagnose_cuda_unavailable,
    is_rocm,
    needs_rocm_attention_fallback,
    resolve_attention_mode,
)
from scripts import update

ROOT = Path(__file__).resolve().parents[1]


def _torch(hip):
    return SimpleNamespace(version=SimpleNamespace(hip=hip))


def test_rocm_detection_uses_hip_version():
    assert is_rocm(_torch("7.14.0"))
    assert not is_rocm(_torch(None))


def test_rocm_flash_falls_back_to_torch_sdpa():
    assert resolve_attention_mode("flash", _torch("7.14.0")) == "torch"


def test_rocm_keeps_explicit_non_flash_modes():
    assert resolve_attention_mode("flex", _torch("7.14.0")) == "flex"
    assert resolve_attention_mode("torch", _torch("7.14.0")) == "torch"


def test_cuda_keeps_flash():
    assert resolve_attention_mode("flash", _torch(None)) == "flash"


def test_none_request_does_not_report_rocm_fallback():
    assert resolve_attention_mode(None, _torch(None)) == "torch"
    assert not needs_rocm_attention_fallback(None, _torch(None))
    assert not needs_rocm_attention_fallback(None, _torch("7.14.0"))


def test_only_explicit_rocm_flash_request_reports_fallback():
    assert needs_rocm_attention_fallback("flash", _torch("7.14.0"))
    assert not needs_rocm_attention_fallback("torch", _torch("7.14.0"))
    assert not needs_rocm_attention_fallback("flash", _torch(None))


def test_update_reuses_saved_rocm_backend(monkeypatch, tmp_path):
    (tmp_path / ".anima_backend").write_text("rocm\n", encoding="ascii")
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.delenv("ANIMA_BACKEND", raising=False)

    command, chosen = update._uv_sync_command()
    assert command == [
        "uv",
        "sync",
        "--no-group",
        "cuda-windows",
        "--group",
        "rocm-windows",
    ]
    assert chosen == "rocm"


def test_update_environment_override_wins(monkeypatch, tmp_path):
    (tmp_path / ".anima_backend").write_text("rocm\n", encoding="ascii")
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.setenv("ANIMA_BACKEND", "cuda")

    assert update._selected_windows_backend() == "cuda"


def test_update_hardware_beats_poisoned_venv(monkeypatch, tmp_path):
    """An NVIDIA box whose venv got the ROCm torch (GH #92) must pick cuda."""
    venv = tmp_path / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_text("", encoding="ascii")
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.delenv("ANIMA_BACKEND", raising=False)
    monkeypatch.setattr(update, "_detect_windows_gpu_vendor", lambda: "nvidia")

    def poisoned_probe(*a, **k):
        raise AssertionError("venv probe must not run when hardware is known")

    monkeypatch.setattr(update.subprocess, "run", poisoned_probe)
    assert update._selected_windows_backend() == "cuda"


def test_update_amd_hardware_selects_rocm(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.delenv("ANIMA_BACKEND", raising=False)
    monkeypatch.setattr(update, "_detect_windows_gpu_vendor", lambda: "amd")

    assert update._selected_windows_backend() == "rocm"


def test_update_sync_persists_backend_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.delenv("ANIMA_BACKEND", raising=False)
    monkeypatch.setattr(update, "_detect_windows_gpu_vendor", lambda: "nvidia")

    command, chosen = update._uv_sync_command()
    assert command == ["uv", "sync"]
    assert chosen == "cuda"
    assert (tmp_path / ".anima_backend").read_text(encoding="ascii").strip() == "cuda"


def _torch_full(*, available, hip=None, cuda=None):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: available),
        version=SimpleNamespace(hip=hip, cuda=cuda),
    )


def test_diagnose_silent_when_cuda_works(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    assert diagnose_cuda_unavailable(_torch_full(available=True, cuda="13.2")) is None


def test_diagnose_silent_without_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: None)
    assert diagnose_cuda_unavailable(_torch_full(available=False, hip="7.14.0")) is None


def test_diagnose_flags_rocm_build_on_nvidia(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    message = diagnose_cuda_unavailable(_torch_full(available=False, hip="7.14.0"))
    assert message is not None and "ROCm build" in message


def test_diagnose_flags_cpu_build_on_nvidia(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    message = diagnose_cuda_unavailable(_torch_full(available=False))
    assert message is not None and "CPU-only build" in message


def test_diagnose_flags_driver_problem(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    message = diagnose_cuda_unavailable(_torch_full(available=False, cuda="13.2"))
    assert message is not None and "driver" in message


def test_windows_backend_dependencies_are_isolated():
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    groups = project["dependency-groups"]
    cuda = "\n".join(groups["cuda-windows"])
    rocm = "\n".join(groups["rocm-windows"])

    assert "+cu132" in cuda
    assert "flash_attn" in cuda
    assert "torch[device-gfx1200,device-gfx1201]==2.13.0+rocm10.0.0" in rocm
    assert "torchvision[device-gfx1200,device-gfx1201]==0.28.0+rocm10.0.0" in rocm
    assert "+rocm7.14.0" not in rocm
    assert "device-gfx1200" in rocm and "device-gfx1201" in rocm
    assert "flash" not in rocm

    sources = project["tool"]["uv"]["sources"]
    assert any(
        source.get("index") == "amd-rocm-100" and source.get("group") == "rocm-windows"
        for source in sources["torch"]
    )
    indices = project["tool"]["uv"]["index"]
    rocm_index = next(index for index in indices if index["name"] == "amd-rocm-100")
    assert rocm_index["url"] == "https://stable.repo.amd.com/rocm/whl-next/"

    # GH #92: a flagless `uv sync` (old updaters) must install the CUDA stack
    # on Windows, and the legacy --extra flags must keep resolving as stubs.
    assert "cuda-windows" in project["tool"]["uv"]["default-groups"]
    extras = project["project"]["optional-dependencies"]
    assert extras["cuda-windows"] == [] and extras["rocm-windows"] == []


def test_flagless_sync_resolves_cuda_torch_on_windows():
    """GH #92 lock-level guard: the default (no-flag) resolution must give every
    platform except macOS the +cu132 torch — v1.16.1's lock resolved flagless
    win32 syncs to the ROCm torch, silently demoting NVIDIA users to CPU."""
    result = subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    torch_lines = [
        line for line in result.stdout.splitlines() if line.startswith("torch==")
    ]
    assert torch_lines, "no torch pin in the default export"
    non_darwin = [line for line in torch_lines if "== 'darwin'" not in line]
    assert non_darwin and all("torch==2.12.0+cu132" in line for line in non_darwin), (
        torch_lines
    )
    torchvision_lines = [
        line for line in result.stdout.splitlines() if line.startswith("torchvision==")
    ]
    non_darwin_vision = [
        line for line in torchvision_lines if "== 'darwin'" not in line
    ]
    assert non_darwin_vision and all(
        "torchvision==0.27.0+cu132" in line for line in non_darwin_vision
    ), torchvision_lines
    assert "+rocm" not in result.stdout
    assert "rocm-sdk" not in result.stdout
    flash_win = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("flash-attn") and "win_amd64" in line
    ]
    assert flash_win, "the default export must ship the Windows flash-attn wheel"


def test_rocm_group_resolves_pytorch_213_rocm10():
    """The ROCm group must resolve the stable Windows ROCm 10 / torch 2.13 stack."""
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-hashes",
            "--no-emit-project",
            "--no-group",
            "cuda-windows",
            "--group",
            "rocm-windows",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    torch_lines = [line for line in lines if line.startswith("torch==")]
    torchvision_lines = [line for line in lines if line.startswith("torchvision==")]

    assert any("torch==2.13.0+rocm10.0.0" in line for line in torch_lines), torch_lines
    assert any(
        "torchvision==0.28.0+rocm10.0.0" in line for line in torchvision_lines
    ), torchvision_lines
    assert not any("+rocm7.14.0" in line for line in torch_lines), torch_lines
    assert not any("+rocm7.14.0" in line for line in torchvision_lines), (
        torchvision_lines
    )
    assert any("amd-torch-device-gfx1200" in line for line in lines)
    assert any("amd-torch-device-gfx1201" in line for line in lines)
    assert not any(
        line.startswith("flash-attn") and "win_amd64" in line for line in lines
    )
