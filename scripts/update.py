#!/usr/bin/env python3
"""Update anima_lora from a GitHub release.

Downloads the release tarball from sorryhyun/anima_lora, extracts it to a
temp dir, then merges over the working tree using a 3-way reconciliation
of (baseline / user / new) sha256 hashes:

  - Datasets, outputs, models, caches, .venv: never touched.
  - Configs in configs/methods/, configs/gui-methods/, configs/base.toml,
    configs/presets.toml, configs/sam_mask.yaml: prompt on conflict
    (keep yours / overwrite / backup-and-overwrite / show diff).
  - Code files (library/, scripts/, train.py, etc.): overwritten silently
    when unmodified; user-modified versions are copied to
    .anima-update-backups/<timestamp>/ before being overwritten.

The baseline manifest lives at .anima_release.json. If it doesn't exist
(first run after upgrading from a release that predates this script), every
file that differs from upstream is treated as user-modified — so configs
will prompt, code files will be backed up. Use --yes-overwrite to default
all conflicts to backup-and-overwrite.

After file merge, runs `uv sync` (skip with --no-sync).
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

# Korean / non-UTF-8 Windows code pages (cp949, cp1252, …) can't encode the
# em-dash, arrow, and ellipsis characters used below. Force UTF-8 on stdout
# and stderr so this script works the same way under the GUI subprocess as
# it does in a UTF-8 terminal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent
REPO = "sorryhyun/anima_lora"
MANIFEST_FILE = ROOT / ".anima_release.json"
BACKUP_ROOT = ROOT / ".anima-update-backups"

# Directories never touched by update — user data, caches, env, downloads.
# Match is by leading path segments (so "archive/graft/runtime" matches
# anything under that prefix while leaving the rest of archive/ updatable).
PRESERVE_DIRS: tuple[str, ...] = (
    "image_dataset", "post_image_dataset",
    "ip-adapter-dataset", "easycontrol-dataset",
    "output", "models",
    "masks", "masks_mit", "masks_sam",
    "bench", "logs", "results",
    ".venv", ".git", ".claude",
    "test_output", "output_temp", "workflows",
    "archive/graft/runtime",
    "__pycache__", "anima_lora.egg-info",
    ".anima-update-backups",
)
PRESERVE_FILES: tuple[str, ...] = (
    ".env", ".anima_release.json",
)

# Files that prompt on conflict instead of silent overwrite. Globs match
# the path relative to ROOT with forward slashes.
CONFLICT_GLOBS: tuple[str, ...] = (
    "configs/methods/*.toml",
    "configs/gui-methods/*.toml",
    "configs/base.toml",
    "configs/presets.toml",
    "configs/sam_mask.yaml",
    "configs/datasets/*",
)


def _is_preserved(rel: str) -> bool:
    if rel in PRESERVE_FILES:
        return True
    parts = rel.split("/")
    for pres in PRESERVE_DIRS:
        pres_parts = pres.split("/")
        if parts[: len(pres_parts)] == pres_parts:
            return True
    return False


def _is_conflict_path(rel: str) -> bool:
    return any(fnmatch.fnmatchcase(rel, g) for g in CONFLICT_GLOBS)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_tree(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            yield p


def _resolve_release(version: str | None) -> tuple[str, str]:
    """Return (label, tarball_url). version=None → latest tag, "main" → main branch."""
    if version == "main":
        return ("main", f"https://github.com/{REPO}/archive/refs/heads/main.tar.gz")
    if version is None:
        api = f"https://api.github.com/repos/{REPO}/releases/latest"
    else:
        api = f"https://api.github.com/repos/{REPO}/releases/tags/{version}"
    req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"GitHub API {api} returned {e.code} {e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach GitHub: {e.reason}")
    tag = data["tag_name"]
    return tag, f"https://github.com/{REPO}/archive/refs/tags/{tag}.tar.gz"


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "anima-update"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)


def _extract_tarball(tar: Path, dest: Path) -> Path:
    """Extract tarball to dest/, return path to single top-level dir inside."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) != 1:
        sys.exit(f"unexpected tarball layout: {[p.name for p in children]}")
    return children[0]


def _load_baseline() -> tuple[str | None, dict[str, str]]:
    if not MANIFEST_FILE.exists():
        return None, {}
    data = json.loads(MANIFEST_FILE.read_text())
    return data.get("version"), data.get("files", {})


def _save_manifest(version: str, files: dict[str, str]) -> None:
    MANIFEST_FILE.write_text(
        json.dumps(
            {
                "version": version,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": dict(sorted(files.items())),
            },
            indent=2,
        )
    )


def _print_diff(user: Path, new: Path, rel: str) -> None:
    try:
        a = user.read_text().splitlines()
        b = new.read_text().splitlines()
    except UnicodeDecodeError:
        print("    (binary file — diff skipped)")
        return
    out = "\n".join(difflib.unified_diff(
        a, b, fromfile=f"a/{rel} (yours)", tofile=f"b/{rel} (new)", lineterm="",
    ))
    print(out or "    (files differ but unified diff is empty)")


def _prompt_conflict(rel: str, user_path: Path, new_path: Path) -> str:
    """Return action: 'keep' | 'overwrite' | 'backup'."""
    while True:
        choice = input(
            f"\n  conflict: {rel} (you modified it AND upstream changed it)\n"
            f"    [k]eep yours / [o]verwrite with new / [b]ackup yours then overwrite / [d]iff: "
        ).strip().lower()
        if choice in ("k", "keep"):
            return "keep"
        if choice in ("o", "overwrite"):
            return "overwrite"
        if choice in ("b", "backup", ""):
            return "backup"
        if choice in ("d", "diff"):
            _print_diff(user_path, new_path, rel)
            continue
        print("    invalid choice; please pick k/o/b/d")


def update(
    version: str | None,
    dry_run: bool,
    yes_overwrite: bool,
    no_sync: bool,
) -> int:
    print(f"anima_lora update — repo {REPO}")
    tag, tarball_url = _resolve_release(version)
    print(f"  target: {tag}")

    baseline_version, baseline_hashes = _load_baseline()
    if baseline_version:
        print(f"  current baseline: {baseline_version}")
    else:
        print("  no baseline manifest — first run; conflicts will be prompted")

    if version is None and baseline_version == tag:
        print(f"already on {tag}; nothing to do")
        return 0

    with tempfile.TemporaryDirectory(prefix="anima-update-") as td:
        tdir = Path(td)
        tar_path = tdir / "release.tar.gz"
        _download(tarball_url, tar_path)
        extracted_root = _extract_tarball(tar_path, tdir / "extracted")
        return _apply(
            extracted_root, tag, baseline_hashes,
            dry_run=dry_run,
            yes_overwrite=yes_overwrite,
            no_sync=no_sync,
        )


def _apply(
    new_root: Path,
    new_tag: str,
    baseline_hashes: dict[str, str],
    *,
    dry_run: bool,
    yes_overwrite: bool,
    no_sync: bool,
) -> int:
    new_files: dict[str, Path] = {}
    new_hashes: dict[str, str] = {}
    for p in _walk_tree(new_root):
        rel = p.relative_to(new_root).as_posix()
        if _is_preserved(rel):
            continue
        new_files[rel] = p
        new_hashes[rel] = _sha256_file(p)

    summary = {
        "wrote_new": 0,
        "overwrote_unchanged": 0,
        "no_change": 0,
        "config_kept": 0,
        "config_overwrote": 0,
        "config_backed_up": 0,
        "code_backed_up": 0,
        "deleted": 0,
        "kept_user_added": 0,
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_session = BACKUP_ROOT / timestamp

    def _backup(dest: Path) -> None:
        rel = dest.relative_to(ROOT).as_posix()
        bak = backup_session / rel
        if dry_run:
            return
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, bak)

    def _do_copy(src: Path, dest: Path) -> None:
        rel = dest.relative_to(ROOT).as_posix()
        print(f"  write  {rel}")
        if dry_run:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    for rel, src in new_files.items():
        dest = ROOT / rel
        if not dest.exists():
            _do_copy(src, dest)
            summary["wrote_new"] += 1
            continue

        user_hash = _sha256_file(dest)
        new_hash = new_hashes[rel]
        if user_hash == new_hash:
            summary["no_change"] += 1
            continue

        baseline_hash = baseline_hashes.get(rel)
        user_modified = (baseline_hash is None) or (user_hash != baseline_hash)

        if not user_modified:
            _do_copy(src, dest)
            summary["overwrote_unchanged"] += 1
            continue

        if _is_conflict_path(rel):
            action = "backup" if yes_overwrite else _prompt_conflict(rel, dest, src)
            if action == "keep":
                summary["config_kept"] += 1
                continue
            if action == "backup":
                _backup(dest)
                summary["config_backed_up"] += 1
            else:
                summary["config_overwrote"] += 1
            _do_copy(src, dest)
        else:
            _backup(dest)
            summary["code_backed_up"] += 1
            _do_copy(src, dest)

    for rel, baseline_hash in baseline_hashes.items():
        if rel in new_files:
            continue
        dest = ROOT / rel
        if not dest.exists():
            continue
        user_hash = _sha256_file(dest)
        if user_hash == baseline_hash:
            print(f"  delete {rel}")
            if not dry_run:
                dest.unlink()
            summary["deleted"] += 1
        else:
            print(f"  upstream removed but you modified: {rel} (kept)")
            summary["kept_user_added"] += 1

    print("\nsummary:")
    for k, v in summary.items():
        if v:
            print(f"  {k}: {v}")
    if backup_session.exists():
        print(f"\n  backups in {backup_session.relative_to(ROOT)}/")

    if dry_run:
        print("\n(dry run — no files written, manifest not updated)")
        return 0

    _save_manifest(new_tag, new_hashes)
    print(f"\nmanifest updated: {MANIFEST_FILE.name} → {new_tag}")

    if not no_sync:
        print("\nrunning uv sync …")
        try:
            subprocess.run(["uv", "sync"], cwd=ROOT, check=True)
        except FileNotFoundError:
            print("  uv not found on PATH; skip or run `uv sync` manually")
        except subprocess.CalledProcessError as e:
            print(f"  uv sync failed (exit {e.returncode}); rerun manually")
            return e.returncode

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Update anima_lora from GitHub release")
    ap.add_argument(
        "--version",
        help='Tag to install (e.g. "v1.0"). Default: latest release. Use "main" for the main branch tarball.',
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without touching files",
    )
    ap.add_argument(
        "--yes-overwrite", action="store_true",
        help="Non-interactive: on config conflicts, back up user file and overwrite",
    )
    ap.add_argument(
        "--no-sync", action="store_true",
        help="Skip the trailing `uv sync`",
    )
    args = ap.parse_args()
    return update(
        version=args.version,
        dry_run=args.dry_run,
        yes_overwrite=args.yes_overwrite,
        no_sync=args.no_sync,
    )


if __name__ == "__main__":
    raise SystemExit(main())
