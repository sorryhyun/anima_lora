"""GUI launch + Windows desktop shortcut."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ._common import PY, ROOT, run


def cmd_gui(_extra):
    run([PY, "-m", "gui"])


def _ensure_shortcut_icon() -> Path | None:
    """Return a ready ``.ico`` path for the desktop shortcut, or None.

    Windows .lnk files only accept ``.ico`` (not PNG). If ``icon.png`` exists at
    the project root, convert it to ``gui/icon.ico`` (rebuilt only when the .png
    is newer or the .ico is missing). Returns None if there's no source image or
    Pillow isn't available — caller falls back to the interpreter's own icon.
    """
    src = ROOT / "icon.png"
    if not src.exists():
        return None
    dst = ROOT / "gui" / "icon.ico"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    try:
        from PIL import Image
    except ImportError:
        print("warn: Pillow not available — skipping icon conversion", file=sys.stderr)
        return None
    try:
        img = Image.open(src)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # Multi-resolution ICO so Windows picks the right size for taskbar /
        # desktop / file explorer at any DPI.
        img.save(
            dst,
            format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal, just skip the icon
        print(f"warn: could not convert icon.png → icon.ico: {exc}", file=sys.stderr)
        return None
    return dst


def cmd_gui_shortcut(_extra):
    """Create a Windows desktop shortcut ('Anima LoRA GUI.lnk') that launches the GUI.

    Targets ``pythonw.exe`` from the active venv with ``tasks.py gui``, so it runs
    without flashing a console window. Honors the OneDrive-redirected Desktop when
    present.
    """
    if sys.platform != "win32":
        print("gui-shortcut is Windows-only.", file=sys.stderr)
        sys.exit(1)

    # Prefer the local Desktop. Only fall back to a OneDrive-redirected one if
    # the local folder doesn't exist (i.e., OneDrive backup has actually moved it).
    user = Path(os.environ.get("USERPROFILE", ""))
    candidates = [user / "Desktop", user / "OneDrive" / "Desktop"]
    desktop = next((d for d in candidates if d.is_dir()), None)
    if desktop is None:
        print(
            f"Could not locate a Desktop folder (checked: {', '.join(map(str, candidates))})",
            file=sys.stderr,
        )
        sys.exit(1)

    shortcut_path = desktop / "Anima LoRA GUI.lnk"
    pyw = Path(PY).with_name("pythonw.exe")
    target = pyw if pyw.exists() else Path(PY)
    icon = _ensure_shortcut_icon() or target

    # Pass paths via env vars to sidestep PowerShell quoting on user paths.
    env = {
        **os.environ,
        "_SC_PATH": str(shortcut_path),
        "_SC_TARGET": str(target),
        "_SC_ARGS": f'"{ROOT / "tasks.py"}" gui',
        "_SC_WD": str(ROOT),
        "_SC_DESC": "Anima LoRA GUI",
        "_SC_ICON": str(icon),
    }
    ps = (
        "$ws = New-Object -ComObject WScript.Shell;"
        "$sc = $ws.CreateShortcut($env:_SC_PATH);"
        "$sc.TargetPath = $env:_SC_TARGET;"
        "$sc.Arguments = $env:_SC_ARGS;"
        "$sc.WorkingDirectory = $env:_SC_WD;"
        "$sc.Description = $env:_SC_DESC;"
        "$sc.IconLocation = $env:_SC_ICON;"
        "$sc.Save()"
    )
    print(f"  > Creating shortcut: {shortcut_path}")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps], cwd=ROOT, env=env
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"  > Done. Double-click '{shortcut_path.name}' on your desktop to launch.")
