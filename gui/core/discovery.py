"""Filesystem discovery helpers for the dataset / adapter / image browsers.

Qt-free directory walks shared by the Merge and adapter tabs: list the
images under a tree, the safetensors in a dir, and the well-known adapter
roots that actually exist on disk.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools._walk import glob_images_pathlib

from gui.core.paths import ROOT


def _imgs(d: Path) -> list[Path]:
    """Every image file under ``d``, recursively.

    Uses the anime_tools walker so the browser sees the same files (avif / jxl
    included) as the curation stages and the trainer.
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

    Returns only paths that exist and actually have .safetensors files, keyed
    by a short display name.
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
