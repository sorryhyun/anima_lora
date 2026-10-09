"""Launching the ``anime_tools`` curation panel against this checkout.

The panel (``python -m anime_tools.gui``) curates in its own ``workspace/`` and
publishes only the decisions — captions, masks, the revised master — with
Export's ``sidecars_only``; the trainer resizes from ``image_dataset/`` itself.
This module seeds the panel's settings file for that and starts (or finds) the
server; the GUI's anime_tools tab renders it. It is not a daemon job: there is no
GPU work, and the server runs with ``--exit-with-window``, so it stops a few
seconds after the last page showing it (the tab, or a browser) is gone.

Qt-free so it stays headless-testable; ``anime_tools`` imports are lazy, to keep
them off the GUI's launch path.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from library.env import anima_home

# Keys of the panel's settings blob (``anime_tools.gui.dataset.SETTINGS_KEY`` and
# the per-stage form memory ``routes/jobs.py`` writes). Spelled here because
# ``anime_tools.gui.dataset`` imports the image stack.
DATASET_KEY = "dataset"
FORMS_KEY = "values"
EXPORT_STAGE = "export"

# Export options ``sidecars_only`` refuses; dropped from the saved form so the
# seeded form passes the request's validation.
_IMAGE_SHAPING = ("resize_cap", "webp")

PANEL_PORT = 8790
_PORT_TRIES = 50  # the panel's own ``pick_port`` range
LOG_NAME = "anime_tools_gui.log"
# The launcher's start line: ``anime_tools GUI → http://127.0.0.1:8790   (home: …)``.
# The arrow is not matched: a detached child writes it in the console code page.
_URL_LINE = re.compile(r"anime_tools GUI \S* *(http://\S+)")


@dataclass
class SeedResult:
    path: Path
    warnings: list[str] = field(default_factory=list)


def settings_path(home: Path) -> Path:
    from anime_tools.gui.settings import SETTINGS_NAME

    return home / SETTINGS_NAME


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _home_relative(path: Path, home: Path) -> str:
    try:
        return path.relative_to(home).as_posix()
    except ValueError:
        return str(path)


def seed_settings(
    source_image_dir: str | Path, *, home: Path | None = None
) -> SeedResult:
    """Point the panel at the trainer's source tree and default its Export to
    ``sidecars_only``. Every other saved setting is left alone.

    ``src`` is written only when the trainer's source is not the package default
    (``image_dataset``); otherwise a saved ``src`` is blanked, so the panel never
    curates a tree the trainer does not resize from.
    """
    from anime_tools import workspace as WS
    from anime_tools._json import write_json

    home = (home or anima_home()).resolve()
    path = settings_path(home)
    data = _read(path)

    src = Path(source_image_dir).expanduser()
    src = (src if src.is_absolute() else home / src).resolve()
    dataset = data.get(DATASET_KEY)
    if not isinstance(dataset, dict):
        dataset = {}
    if src == (home / WS.SOURCE_ROOT).resolve():
        if dataset.get("src"):
            dataset["src"] = ""
    else:
        dataset["src"] = _home_relative(src, home)
    if dataset:
        data[DATASET_KEY] = dataset

    forms = data.get(FORMS_KEY)
    if not isinstance(forms, dict):
        forms = {}
    export = forms.get(EXPORT_STAGE)
    if not isinstance(export, dict):
        export = {}
    export["sidecars_only"] = True
    for key in _IMAGE_SHAPING:
        export.pop(key, None)
    forms[EXPORT_STAGE] = export
    data[FORMS_KEY] = forms

    write_json(path, data)
    return SeedResult(path, _root_warnings(dataset, home))


def _root_warnings(dataset: dict, home: Path) -> list[str]:
    """A saved workspace root (``dst`` / ``masks``) inside the trainer's
    ``post_image_dataset`` would have the panel write over the trainer's trees."""
    from anime_tools import workspace as WS

    out_tree = (home / WS.EXPORT_ROOT).resolve()
    warnings = []
    for key in ("dst", "masks"):
        raw = str(dataset.get(key) or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        p = (p if p.is_absolute() else home / p).resolve()
        if p.is_relative_to(out_tree):
            warnings.append(f"{key} = {raw}")
    return warnings


def export_report_path(home: Path | None = None) -> Path:
    from anime_tools import workspace as WS

    return (home or anima_home()) / WS.REPORTS / EXPORT_STAGE / "report.json"


def export_is_stale(home: Path | None = None) -> bool:
    """Whether the panel's workspace holds a caption or mask newer than its last
    Export — i.e. TE caching would encode captions the panel has since changed.

    ``False`` when there is no workspace yet.
    """
    from anime_tools import workspace as WS

    home = home or anima_home()
    trees = (
        (home / WS.DEFAULT_ROOTS["master"], "*.txt"),
        (home / WS.RESIZED, "*.txt"),
        (home / WS.MASKS, "*.png"),
    )
    newest = 0.0
    for root, pattern in trees:
        if not root.is_dir():
            continue
        for p in root.rglob(pattern):
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                continue
    if newest == 0.0:
        return False
    report = export_report_path(home)
    # A dry run writes a report too; only an applied one published anything.
    if not _read(report).get("apply"):
        return True
    try:
        return newest > report.stat().st_mtime
    except OSError:
        return True


def _listening_ports() -> list[int]:
    """The panel's candidate ports that something listens on.

    Probing every candidate is slow on Windows, where a connect to a closed
    loopback port waits out its timeout instead of failing at once. Where the
    socket table is not readable (macOS without root) every candidate is
    returned.
    """
    candidates = range(PANEL_PORT, PANEL_PORT + _PORT_TRIES)
    try:
        listening = {
            c.laddr.port
            for c in psutil.net_connections(kind="tcp")
            if c.status == psutil.CONN_LISTEN and c.laddr
        }
    except (OSError, psutil.Error):
        return list(candidates)
    return [p for p in candidates if p in listening]


def find_running(home: Path | None = None, *, timeout: float = 0.5) -> str | None:
    """URL of a panel already serving ``home``, or ``None``."""
    home = (home or anima_home()).resolve()
    for port in _listening_ports():
        url = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(f"{url}/api/info", timeout=timeout) as resp:
                info = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if not isinstance(info, dict) or "home" not in info:
            continue
        try:
            if Path(info["home"]).resolve() == home:
                return url
        except (OSError, TypeError):
            continue
    return None


def launch_argv(home: Path | None = None) -> list[str]:
    """No window of its own — the GUI's tab shows the page — but the server
    still exits with its last page."""
    from anima_daemon.client import venv_python

    home = (home or anima_home()).resolve()
    return [
        venv_python(windowless=True),
        "-m",
        "anime_tools.gui",
        "--home",
        str(home),
        "--exit-with-window",
    ]


def log_path(home: Path | None = None) -> Path:
    return (home or anima_home()).resolve() / "output" / LOG_NAME


def launch(home: Path | None = None) -> Path:
    """Start the panel detached; its stdout is appended to
    ``output/anime_tools_gui.log`` (returned)."""
    from anima_daemon.proc import spawn_detached

    home = (home or anima_home()).resolve()
    log = log_path(home)
    spawn_detached(launch_argv(home), cwd=home, stdout_path=log)
    return log


def url_in_log(log: Path, offset: int = 0) -> str | None:
    """The URL a launch announced in ``log`` past byte ``offset`` (the log is
    appended to, so earlier launches' lines sit before it), or ``None`` yet."""
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    m = _URL_LINE.search(text)
    return m.group(1) if m else None
