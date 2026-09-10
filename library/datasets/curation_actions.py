"""Dataset curation helpers shared by GUI and preprocess.

Two decisions live here. **Preprocess decisions** (``use`` / ``skip`` /
``move``) are the GUI's per-image marks, saved to
``post_image_dataset/curation_decisions.json`` and turned into the resize
stage's ``skip`` list. **Exclusion** is ``anime_tools.exclude`` (torch-free)
pointed at the trainer's trees: an image's *workspace* artifacts — the resized
PNG, its caption sidecars, mask and OCR — move under
:data:`EXCLUDED_DIR` and its source rel goes into the ledger there; the source
image under ``image_dataset/`` stays where it is. The resize stage reads that
ledger as extra ``--skip`` (``ResizeRequest.excluded_dir``), which is the one
place an excluded image could come back, and every later stage walks the
resized tree — so the image never trains. :func:`restore_rels` reverses it.

Before anime_tools 0.6 the GUI's Delete gesture moved the *source* image to
``post_image_dataset/moved/`` and left the resized copy and its caches behind,
so a "moved" image could still be cached and trained. A ``moved/`` tree on
disk is inert and can be deleted or restored by hand.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from anime_tools.exclude import Entry, Result, Trees, exclude_one, restore_one
from anime_tools.exclude import excluded_rels as _excluded_rels
from anime_tools.exclude import read_entries as _read_entries
from anime_tools.exclude import rel_key as exclusion_rel_key

from library.env import resolve_under_home

SOURCE_DIR = "image_dataset"
RESIZED_DIR = "post_image_dataset/resized"
DEFAULT_MASK_DIR = "post_image_dataset/masks"
OCR_DIR = "post_image_dataset/ocr"
EXCLUDED_DIR = "post_image_dataset/_excluded"
"""The trainer's exclusion tree and ledger. ``post_image_dataset/`` is the
trainer's workspace (the package's own default is ``workspace/_excluded``,
which the anime_tools GUI uses; Export mirrors that tree's files under this
path too, so a curation done there also ends up beside — never inside — the
resized tree the trainer reads)."""

PACKAGE_WORKSPACE_EXCLUDED_DIR = "workspace/_excluded"
"""Where the anime_tools GUI keeps its ledger. Its rels are unioned into the
resize skip list so a trainer-side ``make preprocess-resize`` never re-creates
an image that was excluded in the package's own workspace."""


def rel_key(path: Path, root: Path) -> str:
    """Stable JSON key for an image path relative to a dataset root."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


# ---- exclusion ----------------------------------------------------------


def exclusion_trees(
    *,
    home: Path | str | None = None,
    mask_dir: Path | str | None = None,
) -> Trees:
    """The trainer's trees for ``anime_tools.exclude``, anchored on the repo
    home (or ``home``) so the GUI and a shell agree wherever they run from."""

    def under(p: Path | str) -> Path:
        return Path(home) / p if home is not None else resolve_under_home(p)

    return Trees(
        resized=under(RESIZED_DIR),
        masks=under(mask_dir or DEFAULT_MASK_DIR),
        ocr=under(OCR_DIR),
        excluded=under(EXCLUDED_DIR),
    )


def excluded_entries(*, home: Path | str | None = None) -> dict[str, Entry]:
    """The trainer ledger, keyed by source rel (empty when there is none)."""

    return _read_entries(exclusion_trees(home=home).excluded)


def excluded_rels(*, home: Path | str | None = None) -> tuple[str, ...]:
    return _excluded_rels(exclusion_trees(home=home).excluded)


def workspace_excluded_rels(*, home: Path | str | None = None) -> tuple[str, ...]:
    """Rels excluded in the anime_tools GUI's own workspace (its ledger lives
    under :data:`PACKAGE_WORKSPACE_EXCLUDED_DIR`); empty when it has none."""

    excluded = (
        Path(home) / PACKAGE_WORKSPACE_EXCLUDED_DIR
        if home is not None
        else resolve_under_home(PACKAGE_WORKSPACE_EXCLUDED_DIR)
    )
    return _excluded_rels(excluded)


def source_rel(path: Path, *, home: Path | str | None = None) -> str:
    """The ledger key for an image the GUI shows — its path relative to the
    source tree, extension kept, which is what ``resize --skip`` matches.

    A path under the resized tree maps back to the source image by directory
    + stem (resize re-encodes ``a.jpg`` → ``a.png``); when no source image is
    there any more the resized spelling is used as it is — resize cannot
    re-create what has no source, so the ledger only has to move the files.
    """

    from anime_tools._walk import IMAGE_EXTENSIONS

    src = (
        Path(home) / SOURCE_DIR if home is not None else resolve_under_home(SOURCE_DIR)
    )
    resized = (
        Path(home) / RESIZED_DIR
        if home is not None
        else resolve_under_home(RESIZED_DIR)
    )
    path = Path(path)
    if path.is_relative_to(src):
        return exclusion_rel_key(path.relative_to(src))
    if path.is_relative_to(resized):
        rel = path.relative_to(resized)
        for ext in IMAGE_EXTENSIONS:
            if (src / rel.parent / f"{rel.stem}{ext}").is_file():
                return exclusion_rel_key(rel.parent / f"{rel.stem}{ext}")
        return exclusion_rel_key(rel)
    raise ValueError(f"not under {SOURCE_DIR}/ or {RESIZED_DIR}/: {path}")


def exclude_images(
    paths: Iterable[Path],
    *,
    note: str = "",
    home: Path | str | None = None,
    mask_dir: Path | str | None = None,
) -> list[Result]:
    """Take these images out of the pipeline (see the module docstring).

    Each path is resolved to its source rel first; a path outside both trees
    raises ``ValueError`` before anything moves.
    """

    trees = exclusion_trees(home=home, mask_dir=mask_dir)
    rels = [source_rel(p, home=home) for p in paths]
    return [exclude_one(trees, rel, note=note) for rel in rels]


def restore_rels(
    rels: Iterable[str],
    *,
    home: Path | str | None = None,
    mask_dir: Path | str | None = None,
) -> list[Result]:
    """Put excluded images back and drop them from the ledger. A slot whose
    live path is occupied again stays under ``_excluded`` and is reported in
    ``Result.skipped`` — the package never overwrites a live file."""

    trees = exclusion_trees(home=home, mask_dir=mask_dir)
    return [restore_one(trees, rel) for rel in rels]


# ---- preprocess decisions ------------------------------------------------


def load_curation_decisions(
    path: Path | str | None,
    *,
    source_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load per-image preprocess decisions from JSON.

    The file is intentionally optional. Missing or malformed files behave as an
    empty decision set so normal CLI preprocess remains unchanged unless a GUI
    decision file is present.
    """

    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, dict):
        return {}
    strip_prefix = ""
    add_prefix = ""
    if source_dir is not None:
        saved_source = str(data.get("source_dir") or "").replace("\\", "/")
        source = Path(source_dir)
        if saved_source:
            try:
                cwd = Path.cwd().resolve()
                saved_abs = (cwd / saved_source).resolve()
                source_abs = (
                    source.resolve()
                    if source.is_absolute()
                    else (cwd / source).resolve()
                )
                if source_abs == saved_abs:
                    pass
                elif source_abs.is_relative_to(saved_abs):
                    strip_prefix = source_abs.relative_to(saved_abs).as_posix()
                elif saved_abs.is_relative_to(source_abs):
                    add_prefix = saved_abs.relative_to(source_abs).as_posix()
                else:
                    return {}
            except OSError:
                candidates = {str(source).replace("\\", "/"), source.as_posix()}
                if saved_source not in candidates:
                    return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in images.items():
        if isinstance(key, str) and isinstance(value, dict):
            norm_key = key.replace("\\", "/")
            if strip_prefix:
                prefix = strip_prefix.rstrip("/") + "/"
                if not norm_key.startswith(prefix):
                    continue
                norm_key = norm_key[len(prefix) :]
            elif add_prefix:
                norm_key = f"{add_prefix.rstrip('/')}/{norm_key}"
            out[norm_key] = dict(value)
    return out


def save_curation_decisions(
    path: Path,
    *,
    source_dir: str,
    images: dict[str, dict[str, Any]],
) -> None:
    """Write GUI decisions consumed by preprocess resize."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "source_dir": source_dir.replace("\\", "/"),
        "images": images,
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
