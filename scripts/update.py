#!/usr/bin/env python3
"""Update anima_lora from a GitHub release.

Downloads the release tarball from sorryhyun/anima_lora, extracts it to a
temp dir, then merges over the working tree using a 3-way reconciliation
of (baseline / user / new) sha256 hashes:

  - Datasets, outputs, models, caches, .venv: never touched.
  - User configs (``CONFLICT_GLOBS``): prompt on conflict
    (keep yours / overwrite / backup-and-overwrite / show diff).
  - Code files (library/, scripts/, train.py, etc.) AND configs/base.toml:
    overwritten silently when unmodified; user-modified versions are copied to
    .anima-update-backups/<timestamp>/ before being overwritten. base.toml is
    overwritten even under --keep-conflicts.

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
import os
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

# Non-UTF-8 Windows code pages (cp949, cp1252, …) can't encode the em-dash /
# arrow / ellipsis below — force UTF-8 so the GUI subprocess matches a terminal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent


def _detect_windows_gpu_vendor() -> str | None:
    """Return 'nvidia' or 'amd' from the installed display adapters, or None."""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "ForEach-Object { $_.Name + ' ' + $_.AdapterCompatibility }",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    adapters = result.stdout.lower()
    # Prefer NVIDIA on mixed-vendor systems (iGPU + dGPU), matching install.ps1.
    if "nvidia" in adapters:
        return "nvidia"
    if (
        "amd" in adapters
        or "advanced micro devices" in adapters
        or "radeon" in adapters
    ):
        return "amd"
    return None


def _selected_windows_backend() -> str:
    """Keep the accelerator chosen at install time across release updates."""
    override = os.environ.get("ANIMA_BACKEND", "").strip().lower()
    if override in {"cuda", "rocm"}:
        return override

    marker = ROOT / ".anima_backend"
    if marker.is_file():
        saved = marker.read_text(encoding="ascii").strip().lower()
        if saved in {"cuda", "rocm"}:
            return saved

    # No marker yet: decide from the hardware, NOT from the venv's torch build —
    # an older extra-unaware `uv sync` installed ROCm torch on every Windows
    # machine (GH #92), so trusting `torch.version.hip` would lock NVIDIA users
    # into that state.
    vendor = _detect_windows_gpu_vendor()
    if vendor == "amd":
        return "rocm"
    if vendor == "nvidia":
        return "cuda"

    # No adapter identified (headless/VM): the venv build is the last resort.
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if python.is_file():
        try:
            result = subprocess.run(
                [str(python), "-c", "import torch; print(bool(torch.version.hip))"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip() == "True":
                return "rocm"
        except (OSError, subprocess.CalledProcessError):
            pass
    return "cuda"


def _uv_sync_command() -> tuple[list[str], str | None]:
    """Return the sync command and the Windows backend it targets (None off-Windows)."""
    command = ["uv", "sync"]
    if sys.platform != "win32":
        return command, None
    backend = _selected_windows_backend()
    # cuda-windows is a DEFAULT dependency group (GH #92), so the CUDA stack
    # needs no flags; ROCm swaps the default group out explicitly.
    if backend == "rocm":
        command.extend(["--no-group", "cuda-windows", "--group", "rocm-windows"])
    # Persist the decision so later updates (and manual `make update` runs)
    # stop re-deriving it — install.ps1 writes the same marker.
    try:
        (ROOT / ".anima_backend").write_text(backend + "\n", encoding="ascii")
    except OSError:
        pass
    return command, backend


def _verify_windows_backend(backend: str) -> None:
    """Warn loudly when the synced venv's torch build contradicts the backend."""
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        return
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import torch; print('rocm' if torch.version.hip else "
                "('cuda' if torch.version.cuda else 'cpu'))",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return
    build = result.stdout.strip()
    if build != backend:
        fix = (
            "uv sync"
            if backend == "cuda"
            else "uv sync --no-group cuda-windows --group rocm-windows"
        )
        print(
            f"\nWARNING: the installed PyTorch is a {build} build but the "
            f"selected backend is {backend} — training/caching would fall back "
            f"to CPU.\n  Fix: {fix}\n"
            f"  (backend override: set ANIMA_BACKEND=cuda|rocm or edit "
            f".anima_backend)"
        )


REPO = "sorryhyun/anima_lora"
MANIFEST_FILE = ROOT / ".anima_release.json"
BACKUP_ROOT = ROOT / ".anima-update-backups"

# Directories never touched by update (user data, caches, env, downloads).
# Matched by leading path segments, so "archive/graft/runtime" matches that
# prefix while leaving the rest of archive/ updatable. "_archive/" is
# preserved wholesale (never overwritten or pruned).
PRESERVE_DIRS: tuple[str, ...] = (
    "_archive",
    "image_dataset",
    "post_image_dataset",
    "ip-adapter-dataset",
    "easycontrol-dataset",
    "output",
    "models",
    "masks",
    "masks_mit",
    "masks_sam",
    "bench",
    "logs",
    "results",
    ".venv",
    ".git",
    ".claude",
    "test_output",
    "output_temp",
    "workflows",
    "archive/graft/runtime",
    "__pycache__",
    "anima_lora.egg-info",
    ".anima-update-backups",
)
PRESERVE_FILES: tuple[str, ...] = (
    ".env",
    ".anima_release.json",
)

# Files that prompt on conflict instead of silent overwrite (globs relative to
# ROOT). configs/base.toml is not here: it takes the code-file path (see the
# module docstring).
CONFLICT_GLOBS: tuple[str, ...] = (
    "configs/methods/*.toml",
    "configs/gui-methods/*.toml",
    "configs/preprocess.toml",
    "configs/presets.toml",
    "configs/sam_mask.yaml",
    "configs/clause_vocabulary.yaml",
    "configs/datasets/*",
    # Self-contained per-method dirs (EasyControl).
    "configs/easycontrol/*.toml",
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


def _resolve_release(version: str | None) -> tuple[str, str, str]:
    """Return (label, tarball_url, body).

    version=None → latest tag, "main" → main branch tarball (no release body
    available, returns ""). For tagged releases the GitHub API also returns
    the release notes body, which we print before the confirmation prompt.
    """
    if version == "main":
        return ("main", f"https://github.com/{REPO}/archive/refs/heads/main.tar.gz", "")
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
    body = (data.get("body") or "").strip()
    return tag, f"https://github.com/{REPO}/archive/refs/tags/{tag}.tar.gz", body


def _print_release_notes(tag: str, body: str) -> None:
    rule = "─" * 64
    print()
    print(rule)
    print(f"  Release notes — {tag}")
    print(rule)
    if body:
        for line in body.splitlines():
            print(f"  {line}" if line else "")
    else:
        print("  (release has no description)")
    print(rule)


def _confirm_update(from_tag: str | None, to_tag: str) -> bool:
    """Prompt the user before applying an update.

    Returns True if we should proceed, False to abort. When stdin isn't a
    TTY (e.g. invoked from the GUI's QProcess) we auto-proceed silently —
    the caller is expected to have shown the changelog already.
    """
    if not sys.stdin.isatty():
        print("  (stdin not a TTY — proceeding without prompt)")
        return True
    from_label = from_tag if from_tag else "(no baseline)"
    try:
        ans = input(f"\nProceed with update {from_label} → {to_tag}? [Y/n]: ")
    except EOFError:
        return True
    ans = ans.strip().lower()
    if ans in ("", "y", "yes"):
        return True
    print("aborted.")
    return False


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "anima-update"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)


def _skip_links_filter(member, path):
    """The data filter, but symlinks/hardlinks are dropped instead of fatal.

    A committed symlink in the release tarball (absolute target →
    LinkOutsideDestinationError) would otherwise abort the update; release
    content never depends on links, so skipping is safe.
    """
    if member.islnk() or member.issym():
        print(f"  skipping link in tarball: {member.name}")
        return None
    return tarfile.data_filter(member, path)


def _extract_tarball(tar: Path, dest: Path) -> Path:
    """Extract tarball to dest/, return path to single top-level dir inside."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        tf.extractall(dest, filter=_skip_links_filter)
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
    out = "\n".join(
        difflib.unified_diff(
            a,
            b,
            fromfile=f"a/{rel} (yours)",
            tofile=f"b/{rel} (new)",
            lineterm="",
        )
    )
    print(out or "    (files differ but unified diff is empty)")


def _prompt_conflict(rel: str, user_path: Path, new_path: Path) -> str:
    """Return action: 'keep' | 'overwrite' | 'backup'."""
    # Without a real TTY (e.g. the GUI's QProcess), input() would block forever
    # on stdin nobody can write to — bail with an actionable message instead.
    if not sys.stdin.isatty():
        sys.exit(
            f"\nconflict: {rel} (you modified it AND upstream changed it)\n"
            "  stdin is not a terminal — cannot prompt. Re-run with one of:\n"
            "    --keep-conflicts   keep all your modified configs as-is\n"
            "    --yes-overwrite    back up your configs, then overwrite with upstream"
        )
    while True:
        choice = (
            input(
                f"\n  conflict: {rel} (you modified it AND upstream changed it)\n"
                f"    [k]eep yours / [o]verwrite with new / [b]ackup yours then overwrite / [d]iff: "
            )
            .strip()
            .lower()
        )
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


def seed_manifest(version: str | None) -> int:
    """Write .anima_release.json for the current tree without downloading.

    Used by the bootstrap installer (install.sh / install.ps1) right after it
    extracts a release tarball, so the *first* `make update` has a correct
    baseline and treats nothing as user-modified. The non-preserved file set
    hashed here must match exactly what `_apply` records, otherwise a later
    update would see preserved files as "upstream-removed" and delete them —
    so this reuses `_is_preserved` / `_sha256_file` rather than reimplementing
    the walk in shell. Run before `uv sync` so `.venv` doesn't exist yet.
    """
    if version is None:
        version, _, _ = _resolve_release(None)
    hashes: dict[str, str] = {}
    for p in _walk_tree(ROOT):
        rel = p.relative_to(ROOT).as_posix()
        if _is_preserved(rel):
            continue
        hashes[rel] = _sha256_file(p)
    _save_manifest(version, hashes)
    print(f"seeded {MANIFEST_FILE.name} → {version} ({len(hashes)} files)")
    return 0


def update(
    version: str | None,
    dry_run: bool,
    yes_overwrite: bool,
    keep_conflicts: bool,
    no_sync: bool,
    assume_yes: bool,
) -> int:
    if yes_overwrite and keep_conflicts:
        sys.exit("--yes-overwrite and --keep-conflicts are mutually exclusive")
    print(f"anima_lora update — repo {REPO}")
    tag, tarball_url, body = _resolve_release(version)
    print(f"  target: {tag}")

    baseline_version, baseline_hashes = _load_baseline()
    if baseline_version:
        print(f"  current baseline: {baseline_version}")
    else:
        print("  no baseline manifest — first run; conflicts will be prompted")

    if version is None and baseline_version == tag:
        print(f"already on {tag}; nothing to do")
        return 0

    # Show release notes. Skip for branch tarballs (no body) and the GUI path
    # (--yes already showed them in the update dialog).
    if version != "main" and not assume_yes:
        _print_release_notes(tag, body)

    if not dry_run and not assume_yes and not _confirm_update(baseline_version, tag):
        return 0

    with tempfile.TemporaryDirectory(prefix="anima-update-") as td:
        tdir = Path(td)
        tar_path = tdir / "release.tar.gz"
        _download(tarball_url, tar_path)
        extracted_root = _extract_tarball(tar_path, tdir / "extracted")
        return _apply(
            extracted_root,
            tag,
            baseline_hashes,
            dry_run=dry_run,
            yes_overwrite=yes_overwrite,
            keep_conflicts=keep_conflicts,
            no_sync=no_sync,
        )


def _restart_daemon_if_idle() -> None:
    """Bring the training daemon onto the freshly-updated code.

    The daemon supervisor runs detached and has already imported the OLD
    ``anima_daemon/*`` into memory; swapping the files on disk doesn't touch
    it, so a new-code client would otherwise be talking to a stale-code daemon.
    Restarting fixes that — but training jobs run in their own detached
    processes (they imported everything at launch), so an *active* run is not
    disrupted by a supervisor restart and must not be killed:

    - idle → graceful shutdown (``kill_jobs=False``); the next ``ensure_daemon``
      relaunches it on the new code.
    - busy → leave it; warn that it stays on old code until the job finishes.

    Best-effort: any daemon hiccup is reported, never fatal to the update.
    """
    try:
        from anima_daemon import client as _client
    except Exception:  # noqa: BLE001 — daemon module optional / mid-swap
        return
    if not _client.is_running():
        return
    cl = _client.DaemonClient()
    active = (cl.health() or {}).get("active_job")
    if active:
        print(
            f"\n⚠ training daemon left running on the OLD code: job {active} is "
            "active.\n"
            "  It keeps using the pre-update daemon until that job finishes.\n"
            "  Restart when convenient — `make daemon-terminate` (it relaunches\n"
            "  on the new code the next time it's used)."
        )
        return
    try:
        cl.shutdown(kill_jobs=False)
        print(
            "\ntraining daemon stopped (was idle) — it will relaunch on the new "
            "code next time it's used."
        )
    except Exception as e:  # noqa: BLE001
        print(f"\ncould not stop the idle training daemon: {e} (restart manually)")


def _apply(
    new_root: Path,
    new_tag: str,
    baseline_hashes: dict[str, str],
    *,
    dry_run: bool,
    yes_overwrite: bool,
    keep_conflicts: bool,
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
            if yes_overwrite:
                action = "backup"
            elif keep_conflicts:
                action = "keep"
            else:
                action = _prompt_conflict(rel, dest, src)
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
        sync_command, backend = _uv_sync_command()
        print(f"\nrunning {' '.join(sync_command)} …")
        try:
            subprocess.run(sync_command, cwd=ROOT, check=True)
        except FileNotFoundError:
            print("  uv not found on PATH; skip or run `uv sync` manually")
        except subprocess.CalledProcessError as e:
            print(f"  uv sync failed (exit {e.returncode}); rerun manually")
            return e.returncode
        else:
            if backend is not None:
                _verify_windows_backend(backend)

    # Code on disk changed → the running daemon supervisor is now stale (skip
    # if nothing was written — a no-op update leaves the daemon correct).
    changed = (
        summary["wrote_new"]
        + summary["overwrote_unchanged"]
        + summary["code_backed_up"]
        + summary["config_overwrote"]
        + summary["config_backed_up"]
        + summary["deleted"]
    )
    if changed:
        _restart_daemon_if_idle()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Update anima_lora from GitHub release")
    ap.add_argument(
        "--version",
        help='Tag to install (e.g. "v1.0"). Default: latest release. Use "main" for the main branch tarball.',
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without touching files",
    )
    ap.add_argument(
        "--yes-overwrite",
        action="store_true",
        help="Non-interactive: on config conflicts, back up user file and overwrite",
    )
    ap.add_argument(
        "--keep-conflicts",
        action="store_true",
        help="Non-interactive: on config conflicts, keep the user's version",
    )
    ap.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the trailing `uv sync`",
    )
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the changelog confirmation prompt (e.g. when invoked from a GUI)",
    )
    ap.add_argument(
        "--seed-manifest",
        action="store_true",
        help="Write .anima_release.json for the current tree (no download) and exit; "
        "used by the bootstrap installer. Records --version, else resolves latest tag.",
    )
    args = ap.parse_args()
    if args.seed_manifest:
        return seed_manifest(args.version)
    return update(
        version=args.version,
        dry_run=args.dry_run,
        yes_overwrite=args.yes_overwrite,
        keep_conflicts=args.keep_conflicts,
        no_sync=args.no_sync,
        assume_yes=args.yes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
