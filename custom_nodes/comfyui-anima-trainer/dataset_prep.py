"""Dataset preparation for the Anima LoRA trainer node.

Two modes:

- **Single-image** (a ComfyUI IMAGE tensor was connected): write every frame
  of the IMAGE batch as a PNG into a fresh temp dir, drop one `.txt` sidecar
  per image holding the supplied prompt.
- **Directory**: use the user-provided path as-is. Validate that at least
  one `.txt` caption sidecar exists (training without captions silently
  produces a useless LoRA).

Both modes return an absolute `(work_dir, dataset_config_path)` tuple. The
dataset_config.toml mirrors `configs/base.toml`'s `[general]` / `[[datasets]]`
blueprint with a single subset pointing at ``work_dir``.
"""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
from PIL import Image

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl")


def _tensor_to_pil(image_tensor) -> list[Image.Image]:
    """Convert a ComfyUI IMAGE tensor (`[B, H, W, C]`, float32 in [0,1]) to PILs."""
    # ComfyUI keeps IMAGE on CPU, but be defensive.
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected IMAGE of shape [B,H,W,C]; got {arr.shape}")
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return [Image.fromarray(frame) for frame in arr]


def _write_dataset_config(work_dir: str, output_path: str) -> None:
    cfg = (
        "[general]\n"
        "shuffle_caption = false\n"
        "caption_extension = '.txt'\n"
        "keep_tokens = 3\n"
        "\n"
        "[[datasets]]\n"
        "resolution = 1024\n"
        "batch_size = 1\n"
        "enable_bucket = true\n"
        "\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = '{work_dir.replace(chr(39), chr(39) + chr(39))}'\n"
        "  num_repeats = 1\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cfg)


def _count_sidecar_pairs(directory: str) -> int:
    n = 0
    for entry in os.listdir(directory):
        stem, ext = os.path.splitext(entry)
        if ext.lower() not in _IMAGE_EXTS:
            continue
        if os.path.exists(os.path.join(directory, stem + ".txt")):
            n += 1
    return n


def _count_images(directory: str) -> int:
    return sum(
        1
        for entry in os.listdir(directory)
        if os.path.splitext(entry)[1].lower() in _IMAGE_EXTS
    )


def prepare_dataset_dir(
    image,
    prompt: str,
    dataset_dir: str,
    tmp_root: str,
) -> Tuple[str, str, int]:
    """Return ``(work_dir, dataset_config_path, n_images)``.

    ``tmp_root`` is where single-image-mode temp dirs are created. Caller
    supplies it (usually ``anima_lora/output/tmp_trainer``); created on demand.
    """
    if image is not None:
        pils = _tensor_to_pil(image)
        if not pils:
            raise ValueError("IMAGE tensor has no frames.")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError(
                "Single-image mode requires a non-empty `prompt` widget."
            )
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(tmp_root, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix=f"{ts}_", dir=tmp_root)
        for i, pil in enumerate(pils):
            stem = f"img_{i:04d}"
            pil.save(os.path.join(work_dir, f"{stem}.png"), optimize=False)
            with open(
                os.path.join(work_dir, f"{stem}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(prompt)
        n_images = len(pils)
        dataset_cfg_path = os.path.join(work_dir, "dataset_config.toml")
        _write_dataset_config(work_dir, dataset_cfg_path)
        return work_dir, dataset_cfg_path, n_images

    if not dataset_dir:
        raise ValueError(
            "Neither an IMAGE input nor a `dataset_dir` was provided."
        )
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    n_images = _count_images(dataset_dir)
    if n_images == 0:
        raise ValueError(f"No images found in {dataset_dir}")
    if _count_sidecar_pairs(dataset_dir) == 0:
        raise ValueError(
            f"No .txt caption sidecars found in {dataset_dir}. Each image "
            "needs a same-stem .txt next to it."
        )
    # Write a fresh dataset_config under tmp_root pointing at the real dir.
    os.makedirs(tmp_root, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_cfg_path = os.path.join(tmp_root, f"dataset_config_{ts}.toml")
    _write_dataset_config(dataset_dir, dataset_cfg_path)
    return dataset_dir, dataset_cfg_path, n_images


def count_captioned_images(directory: str) -> int:
    """Public helper: number of images with .txt sidecars under ``directory``."""
    return _count_sidecar_pairs(directory)
