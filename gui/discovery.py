"""Filesystem discovery helpers for the dataset / adapter / image browsers.

Qt-free directory walks shared by the Image, Merge, and adapter tabs: list the
images under a tree, the safetensors in a dir, and the well-known
adapter/image roots that actually exist on disk.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools._walk import glob_images_pathlib

from gui._paths import ROOT


def _imgs(d: Path) -> list[Path]:
    """Return every image file under ``d`` (recursively).

    ``anime_tools._walk.glob_images_pathlib`` is the shared walker, so the
    browser sees exactly the files the curation stages and the trainer do —
    including the optional-plugin formats (avif / jxl) that a bare extension
    set here would miss.

    Walks subfolders so users who organize ``image_dataset/`` by character /
    series see the full pool in the browser. Cache filenames are stem-keyed
    and flat, so stems must stay unique across the tree — the trainer enforces
    this via ``_assert_unique_stems``; here we just sort and return.
    """
    if not d.exists():
        return []
    return glob_images_pathlib(d, True)


def _safetensors_in(d: Path) -> list[Path]:
    """Return .safetensors files in a directory, newest first."""
    if not d.exists():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix == ".safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _adapter_dirs() -> dict[str, Path]:
    """Directories likely to contain LoRA adapter checkpoints.

    Mirrors ``_image_dirs``: returns only paths that exist and actually have
    .safetensors files, keyed by a short display name.
    """
    dirs: dict[str, Path] = {}
    for name, path in [
        ("output/ckpt", ROOT / "output" / "ckpt"),
        ("output_temp", ROOT / "output_temp"),
        ("models/diffusion_models", ROOT / "models" / "diffusion_models"),
    ]:
        if path.exists() and any(path.glob("*.safetensors")):
            dirs[name] = path
    # Subdirs with .safetensors (iteration snapshots). Skip *-checkpoint-state dirs —
    # those are optimizer/state shards, not adapters.
    for parent, label in (
        (ROOT / "output" / "ckpt", "output/ckpt"),
        (ROOT / "output_temp", "output_temp"),
    ):
        if not parent.exists():
            continue
        for p in sorted(parent.iterdir()):
            if (
                p.is_dir()
                and not p.name.endswith("-checkpoint-state")
                and any(p.glob("*.safetensors"))
            ):
                dirs[f"{label}/{p.name}"] = p
    return dirs


def _image_dirs() -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for name, path in [
        ("image_dataset", ROOT / "image_dataset"),
        ("post_image_dataset/resized", ROOT / "post_image_dataset" / "resized"),
        ("ip-adapter-dataset", ROOT / "ip-adapter-dataset"),
        ("easycontrol-dataset", ROOT / "easycontrol-dataset"),
        ("output/tests", ROOT / "output" / "tests"),
        ("output/ckpt/sample", ROOT / "output" / "ckpt" / "sample"),
    ]:
        if path.exists():
            dirs[name] = path
    return dirs
