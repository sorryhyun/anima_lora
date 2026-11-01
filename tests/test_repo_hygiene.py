import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.core


def test_no_tracked_symlinks():
    """Release tarballs are extracted with tarfile's data filter on every user
    machine — a committed symlink (mode 120000) with an absolute or outside
    target raises LinkOutsideDestinationError and bricks `make update` for
    everyone (v1.16.2.hotfix incident). Keep convenience links untracked."""
    result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    links = [
        line.split("\t", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("120000")
    ]
    assert not links, f"tracked symlinks would break release extraction: {links}"


def test_lock_has_no_out_of_tree_path_sources():
    """`make update`, install.sh and install.ps1 all run a flagless `uv sync`,
    which validates every lock entry — including non-default groups. A path
    source outside the repo (the old `../anime_tools` dev group) makes that
    sync fail on every machine without the sibling checkout, leaving new code
    on an old venv. Test unreleased sibling code with PYTHONPATH instead."""
    import tomllib

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    outside = [
        f"{pkg['name']} -> {src.get('editable') or src.get('directory') or src.get('path')}"
        for pkg in lock["package"]
        if (src := pkg.get("source", {}))
        and any(
            str(src.get(k, "")).startswith("..")
            for k in ("editable", "directory", "path")
        )
    ]
    assert not outside, f"uv.lock references paths outside the repo: {outside}"
