"""GH #105: post-install commands must preserve the synced Windows backend."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh")


@pytest.mark.skipif(
    sys.platform != "win32" or POWERSHELL is None,
    reason="requires Windows PowerShell",
)
@pytest.mark.parametrize("backend", ["rocm", "cuda"])
def test_installer_preserves_backend_through_gui_launch(backend, tmp_path):
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "tests" / "fixtures" / "install_backend_harness.ps1"),
            "-Installer",
            str(ROOT / "install.ps1"),
            "-Backend",
            backend,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls = json.loads(result.stdout)
    assert all(call["backend"] == backend for call in calls), calls
    assert calls[0]["argv"][0] == "sync"
    if backend == "rocm":
        assert calls[0]["argv"] == [
            "sync",
            "--no-group",
            "cuda-windows",
            "--group",
            "rocm-windows",
        ]
        assert any("tests/rocm_smoke_test.py" in call["argv"] for call in calls)
    else:
        assert calls[0]["argv"] == ["sync"]
        assert not any("tests/rocm_smoke_test.py" in call["argv"] for call in calls)
    assert any("gui-shortcut" in call["argv"] for call in calls)
    assert calls[-1]["kind"] == "launch"
    assert calls[-1]["argv"][-3:] == ["python", "tasks.py", "gui"]
    assert all("--no-sync" in call["argv"] for call in calls[1:]), calls
