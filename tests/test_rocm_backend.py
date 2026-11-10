import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from library.runtime import fa2
from library.runtime.backend import resolve_attention_mode
from scripts import update

ROOT = Path(__file__).resolve().parents[1]


def _torch(hip):
    return SimpleNamespace(version=SimpleNamespace(hip=hip))


def test_rocm_flash_falls_back_to_torch_sdpa():
    assert resolve_attention_mode("flash", _torch("7.14.0")) == "torch"


def test_cuda_keeps_flash():
    assert resolve_attention_mode("flash", _torch(None)) == "flash"


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


@pytest.mark.core
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
    monkeypatch.delenv("ANIMA_TORCH", raising=False)
    monkeypatch.setattr(update, "_detect_windows_gpu_vendor", lambda: "nvidia")

    command, chosen = update._uv_sync_command()
    assert command == ["uv", "sync"]
    assert chosen == "cuda"
    assert (tmp_path / ".anima_backend").read_text(encoding="ascii").strip() == "cuda"


def test_update_torch_choice_is_saved_and_reused(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "linux")
    monkeypatch.setenv("ANIMA_TORCH", "2.13")

    opt_in = ["uv", "sync", "--no-default-groups", "--group", "torch-213"]
    assert update._uv_sync_command() == (opt_in, None)
    assert (tmp_path / ".anima_torch").read_text(encoding="ascii").strip() == "2.13"

    monkeypatch.delenv("ANIMA_TORCH")
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.setenv("ANIMA_BACKEND", "cuda")
    assert update._uv_sync_command() == (opt_in, "cuda")


def test_update_torch_choice_ignored_on_rocm(monkeypatch, tmp_path):
    (tmp_path / ".anima_torch").write_text("2.12\n", encoding="ascii")
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.setenv("ANIMA_BACKEND", "rocm")
    monkeypatch.delenv("ANIMA_TORCH", raising=False)

    command, _ = update._uv_sync_command()
    assert command == [
        "uv",
        "sync",
        "--no-group",
        "cuda-windows",
        "--group",
        "rocm-windows",
    ]


@pytest.mark.core
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
    assert non_darwin and all("torch==2.14.1+cu132" in line for line in non_darwin), (
        torch_lines
    )
    torchvision_lines = [
        line for line in result.stdout.splitlines() if line.startswith("torchvision==")
    ]
    non_darwin_vision = [
        line for line in torchvision_lines if "== 'darwin'" not in line
    ]
    assert non_darwin_vision and all(
        "torchvision==0.29.1+cu132" in line for line in non_darwin_vision
    ), torchvision_lines
    assert "+rocm" not in result.stdout
    assert "rocm-sdk" not in result.stdout
    flash_win = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("flash-attn") and "win_amd64" in line
    ]
    assert flash_win, "the default export must ship the Windows flash-attn wheel"


def _export(*args):
    return subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _flash_wheels(lines):
    """Map each flash-attn wheel's platform tag to its torch minor."""
    wheels = {}
    for line in lines:
        if line.startswith("flash-attn"):
            name = line.split(" ; ")[0].rsplit("/", 1)[1]
            torch_minor = name.split("torch", 1)[1].split("-", 1)[0]
            wheels[name.rsplit("-", 1)[1].removesuffix(".whl")] = torch_minor
    return wheels


ALL_PLATFORMS = {"linux_x86_64", "linux_aarch64", "win_amd64"}


@pytest.mark.parametrize(
    "args, torch_pin, vision_pin, minor, triton",
    [
        ((), "2.14.1+cu132", "0.29.1+cu132", "2.14", "3.8."),
        (
            ("--no-default-groups", "--group", "torch-213"),
            "2.13.0+cu132",
            "0.28.0+cu132",
            "2.13",
            "3.7.",
        ),
        (
            ("--no-default-groups", "--group", "torch-212"),
            "2.12.0+cu132",
            "0.27.0+cu132",
            "2.12",
            "3.7.",
        ),
        (
            ("--no-default-groups", "--group", "fa2-sm120"),
            "2.14.1+cu132",
            "0.29.1+cu132",
            "2.14",
            "3.8.",
        ),
    ],
)
def test_torch_stacks_pair_flash_attn_and_triton(
    args, torch_pin, vision_pin, minor, triton
):
    """Every selectable stack ships torch, flash-attn and triton of one minor on
    all three locked platforms; triton comes from PyPI (the AMD index carries
    local `+rocm` builds that also match torch's `triton==X` pin)."""
    lines = _export(*args)
    torch_lines = [
        line
        for line in lines
        if line.startswith("torch==") and "== 'darwin'" not in line
    ]
    assert torch_lines and all(f"torch=={torch_pin}" in line for line in torch_lines)
    vision_lines = [
        line
        for line in lines
        if line.startswith("torchvision==") and "== 'darwin'" not in line
    ]
    assert vision_lines and all(
        f"torchvision=={vision_pin}" in line for line in vision_lines
    )
    wheels = _flash_wheels(lines)
    assert set(wheels) == ALL_PLATFORMS, wheels
    # A stable-ABI wheel ("2.12plus") loads on every torch from that minor up.
    assert all(
        v == minor
        or (
            v.endswith("plus")
            and tuple(map(int, v[:-4].split("."))) <= tuple(map(int, minor.split(".")))
        )
        for v in wheels.values()
    ), wheels
    tritons = [line for line in lines if line.startswith("triton==")]
    assert tritons and all(
        line.startswith(f"triton=={triton}") and "+" not in line.split(" ; ")[0]
        for line in tritons
    ), tritons
    assert any(line.startswith(f"triton-windows=={triton}") for line in lines)


def test_fa2_sm120_group_installs_the_release_build():
    """The group's Linux x86_64 wheel is the release library/runtime/fa2.py names
    (TAG) and its Windows wheel WINDOWS_TAG's; aarch64 has no such build and keeps
    the stock wheel. The default stack stays stock."""
    rows = [
        line
        for line in _export("--no-default-groups", "--group", "fa2-sm120")
        if line.startswith("flash-attn")
    ]

    def from_release(tag):
        return [r for r in rows if f"/{fa2.REPO}/releases/download/{tag}/" in r]

    assert len(rows) == 3, rows
    linux, windows = from_release(fa2.TAG), from_release(fa2.WINDOWS_TAG)
    assert len(linux) == 1 and "linux_x86_64" in linux[0], rows
    assert len(windows) == 1 and "win_amd64" in windows[0], rows
    assert all(fa2.REPO not in line for line in rows if "linux_aarch64" in line), rows
    assert not any(fa2.REPO in line for line in _export()), "default stack is stock"


def test_update_keeps_the_fa2_sm120_stack(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "linux")
    monkeypatch.setenv("ANIMA_TORCH", "2.14-sm120")
    opt_in = ["uv", "sync", "--no-default-groups", "--group", "fa2-sm120"]
    assert update._uv_sync_command() == (opt_in, None)
    monkeypatch.delenv("ANIMA_TORCH")
    assert update._uv_sync_command() == (opt_in, None)


def test_rocm_group_resolves_pytorch_214_rocm101():
    """Select stable ROCm 10.1 with its Windows SDK and no CUDA Flash wheel."""
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

    assert any("torch==2.14.0+rocm10.1.0" in line for line in torch_lines), torch_lines
    assert any(
        "torchvision==0.29.0a0+rocm10.1.0" in line for line in torchvision_lines
    ), torchvision_lines
    assert not any("+rocm7.14.0" in line for line in torch_lines), torch_lines
    assert not any("+rocm7.14.0" in line for line in torchvision_lines), (
        torchvision_lines
    )
    assert any("amd-torch-device-gfx1200" in line for line in lines)
    assert any("amd-torch-device-gfx1201" in line for line in lines)
    assert any(line.startswith("rocm-sdk-devel==10.1.0") for line in lines)
    assert any(line.startswith("triton-windows==3.8.") for line in lines)
    assert not any(
        line.startswith("flash-attn") and "win_amd64" in line for line in lines
    )
