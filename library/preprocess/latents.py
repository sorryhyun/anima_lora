"""Cache VAE latents for a dataset directory.

Orchestration for ``preprocess/cache_latents.py``, which keeps only argparse +
VAE load; the walk → group-by-resolution → batched-encode → idempotent-save
loop lives here.

Idempotence note: a single ``{stem}_{WxH}_anima.npz`` can hold *multiple*
resolutions (one ``latents_{H}x{W}`` key each), so the skip is per-resolution
*inside* the encode loop rather than a whole-file existence check.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from library.io.cache import LATENT_CACHE_SUFFIX, resolve_cache_path
from library.io.cache_names import demoted_latents_key
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.preprocess._dataset import PreprocessStats, group_by_shape, walk_images
from library.preprocess._progress import ProgressFn


def get_latents_npz_path(
    image_path: Path,
    image_size: tuple[int, int],
    cache_dir: Path | None = None,
    image_dir: Path | None = None,
) -> Path:
    """Match ``AnimaLatentsCachingStrategy`` naming: ``{stem}_{WxH}_anima.npz``.

    With ``cache_dir`` the cache is redirected there (nested under the source
    subpath when ``image_dir`` is given); otherwise it lives next to the image.
    """
    suffix = f"_{image_size[0]:04d}x{image_size[1]:04d}{LATENT_CACHE_SUFFIX}"
    if cache_dir is None:
        return image_path.with_name(image_path.stem + suffix)
    return Path(
        resolve_cache_path(
            str(image_path),
            suffix,
            cache_dir=str(cache_dir),
            image_dir=str(image_dir) if image_dir is not None else None,
        )
    )


def _latent_cached(npz_path: Path, w: int, h: int) -> bool:
    """True iff ``npz_path`` already holds the ``(w, h)`` latent.

    A single NPZ can carry several resolutions (one ``latents_{H}x{W}`` key
    each), so the check is per-resolution, not whole-file existence. A
    truncated / unreadable NPZ counts as not-cached (it'll be re-encoded)."""
    if not npz_path.exists():
        return False
    key = f"latents_{h // 8}x{w // 8}"
    try:
        return key in np.load(npz_path)
    except Exception:
        return False


def count_pending_latents(
    data_dir: Path,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Return ``(pending, total)`` latent caches **without loading the VAE**.

    ``pending`` is the number of images whose ``(W, H)`` latent isn't already
    on disk; ``total`` is every enumerated image. Mirrors the per-resolution
    skip in :func:`cache_latents`, so the entry point can skip the (slow) VAE
    load entirely when ``pending == 0``. Reads only NPZ headers (no decode).
    With ``overwrite`` every enumerated image counts as pending."""
    image_files = walk_images(data_dir, recursive=recursive, pattern=path_pattern)
    if overwrite:
        return len(image_files), len(image_files)
    pending = 0
    for (w, h), paths in group_by_shape(image_files).items():
        for p in paths:
            npz_path = get_latents_npz_path(
                p, (w, h), cache_dir=cache_dir, image_dir=data_dir
            )
            if not _latent_cached(npz_path, w, h):
                pending += 1
    return pending, len(image_files)


def _decode_batch(
    batch_paths: list[Path],
    w: int,
    h: int,
    cache_dir: Path | None,
    data_dir: Path,
    overwrite: bool = False,
    image_transform: "Callable[[np.ndarray], np.ndarray] | None" = None,
) -> tuple[
    list[Path],
    list[tuple[Path, str]],
    list[tuple[Path, tuple[int, int]]],
    "torch.Tensor | None",
]:
    """CPU stage: per-resolution skip-probe + decode + transform a batch.

    Returns ``(skipped, failed, kept, img_batch)`` — already-cached paths,
    ``(path, reason)`` for images that wouldn't decode (truncated / corrupt),
    the ``(path, (w, h))`` survivors, and their stacked CPU tensor (``None`` if
    nothing survived). A bad file is isolated to its own entry rather than
    raising, so one corrupt staged PNG can't abort the whole run. Runs on a
    worker thread (pure PIL/numpy/torch-CPU), overlapping the previous GPU encode."""
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []
    kept: list[tuple[Path, tuple[int, int]]] = []
    tensors: list[torch.Tensor] = []
    for p in batch_paths:
        npz_path = get_latents_npz_path(
            p, (w, h), cache_dir=cache_dir, image_dir=data_dir
        )
        if not overwrite and _latent_cached(npz_path, w, h):
            skipped.append(p)
            continue
        try:
            img_np = np.array(Image.open(p).convert("RGB"))
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))
            continue
        if image_transform is not None:
            img_np = image_transform(img_np)
        tensors.append(IMAGE_TRANSFORMS(img_np))
        kept.append((p, (w, h)))
    img_batch = torch.stack(tensors, dim=0) if tensors else None
    return skipped, failed, kept, img_batch


def _save_batch(items: list[tuple[Path, np.ndarray, tuple[int, int]]]) -> None:
    """IO stage: write each ``(npz_path, latent, size)``. Preserves any
    other-resolution keys already in the file (read-modify-write). Each npz_path
    is written exactly once per ``cache_latents`` call, so threaded saves of a
    batch don't race. Runs on a worker thread, overlapping the next GPU encode."""
    for npz_path, lat_np, size in items:
        key_reso_suffix = f"_{lat_np.shape[-2]}x{lat_np.shape[-1]}"
        kwargs: dict = {}
        if npz_path.exists():
            npz = np.load(npz_path)
            for key in npz.files:
                kwargs[key] = npz[key]
        kwargs[f"latents{key_reso_suffix}"] = lat_np
        kwargs[f"original_size{key_reso_suffix}"] = np.array(list(size))
        kwargs[f"crop_ltrb{key_reso_suffix}"] = np.array([0, 0, size[0], size[1]])
        np.savez(npz_path, **kwargs)


def _demoted_cached(npz_path: Path, key: str) -> bool:
    """True iff the native npz already carries the σ-demote sibling ``key``.
    Missing/unreadable npz counts as not-cached (mirrors ``_latent_cached``)."""
    if not npz_path.exists():
        return False
    try:
        return key in np.load(npz_path)
    except Exception:
        return False


def _demote_jobs(
    data_dir: Path,
    *,
    native_edge: int,
    demote_edge: int,
    cache_dir: Path | None,
    recursive: bool,
    path_pattern: str | None,
) -> "list[tuple[tuple[int, int], tuple[int, int], list[Path]]]":
    """Enumerate σ-demote work: ``[(native_wh, demoted_wh, paths)]`` for every
    resized-image shape group on the demote route (off-route groups, e.g.
    native-896 images on 1024→896, are excluded entirely)."""
    from library.datasets.buckets import demote_bucket_for

    image_files = walk_images(data_dir, recursive=recursive, pattern=path_pattern)
    jobs = []
    for (w, h), paths in group_by_shape(image_files).items():
        bucket = demote_bucket_for(w, h, native_edge, demote_edge)
        if bucket is not None:
            jobs.append(((w, h), bucket, paths))
    return jobs


def count_pending_demoted(
    data_dir: Path,
    *,
    native_edge: int,
    demote_edge: int,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    overwrite: bool = False,
) -> tuple[int, int]:
    """``(pending, eligible)`` σ-demote sibling latents, without loading the VAE.

    ``eligible`` counts every image on the demote route (native-tier band);
    ``pending`` those whose native npz lacks the ``demoted_*`` key. Reads only
    NPZ headers. With ``overwrite`` every eligible image counts as pending."""
    pending = eligible = 0
    for (w, h), bucket, paths in _demote_jobs(
        data_dir,
        native_edge=native_edge,
        demote_edge=demote_edge,
        cache_dir=cache_dir,
        recursive=recursive,
        path_pattern=path_pattern,
    ):
        key = demoted_latents_key(*bucket)
        for p in paths:
            eligible += 1
            npz_path = get_latents_npz_path(
                p, (w, h), cache_dir=cache_dir, image_dir=data_dir
            )
            if overwrite or not _demoted_cached(npz_path, key):
                pending += 1
    return pending, eligible


def _save_demoted_batch(items: "list[tuple[Path, str, np.ndarray]]") -> None:
    """IO stage: append each ``(npz_path, key, latent)`` into the native npz
    (read-modify-write, all existing keys preserved — same discipline as
    ``_save_batch``)."""
    for npz_path, key, lat_np in items:
        npz = np.load(npz_path)
        kwargs = {k: npz[k] for k in npz.files}
        kwargs[key] = lat_np
        np.savez(npz_path, **kwargs)


def cache_demoted_latents(
    data_dir: Path,
    vae,
    *,
    native_edge: int,
    demote_edge: int,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    batch_size: int = 4,
    progress: ProgressFn | None = None,
    overwrite: bool = False,
) -> PreprocessStats:
    """Emit σ-demote sibling latents (sigma_lowres, 1024→896 route).

    For every resized image in ``native_edge``'s free-fit band: LANCZOS-downscale
    the resized PNG to its demote-tier free-fit bucket (``demote_bucket_for`` —
    the identical grid the trainer derives), VAE-encode, and store the latent as
    a ``demoted_{H}x{W}`` key INSIDE the image's existing native npz. No sibling
    files — bucket discovery, reconcile, and the ``{stem}_*_anima.npz`` glob
    consumers never see the emit. Pixel-space downscale → VAE re-encode is the
    probe's measured-safe arm (SwD "strategy B"; never latent-space downsample).

    Requires the native latent cache to exist first (``make preprocess`` /
    ``preprocess-vae``): an image without its native npz is counted in
    ``stats.failed`` and skipped. Idempotent per-key; ``overwrite`` re-encodes.
    """
    from library.preprocess.images import resize_to_bucket

    jobs = _demote_jobs(
        data_dir,
        native_edge=native_edge,
        demote_edge=demote_edge,
        cache_dir=cache_dir,
        recursive=recursive,
        path_pattern=path_pattern,
    )
    stats = PreprocessStats(seen=sum(len(paths) for _, _, paths in jobs))
    if progress is not None:
        progress(0, total=stats.seen)

    for (w, h), bucket, paths in jobs:
        key = demoted_latents_key(*bucket)
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            kept: list[Path] = []
            tensors: list[torch.Tensor] = []
            for p in chunk:
                npz_path = get_latents_npz_path(
                    p, (w, h), cache_dir=cache_dir, image_dir=data_dir
                )
                if not npz_path.exists():
                    stats.failed += 1
                    if progress is not None:
                        progress(1, detail=f"NO NATIVE NPZ {p.name}")
                    continue
                if not overwrite and _demoted_cached(npz_path, key):
                    stats.skipped += 1
                    if progress is not None:
                        progress(1, detail=f"skip {p.name}")
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    stats.failed += 1
                    if progress is not None:
                        progress(1, detail=f"FAILED {p.name}")
                    continue
                px = resize_to_bucket(img, bucket)
                tensors.append(IMAGE_TRANSFORMS(np.array(px)))
                kept.append(p)
            if not tensors:
                continue
            img_batch = torch.stack(tensors, dim=0).to(
                device=vae.device, dtype=vae.dtype
            )
            with torch.no_grad():
                latents = vae.encode_pixels_to_latents(img_batch).cpu()
            items = []
            for i, p in enumerate(kept):
                npz_path = get_latents_npz_path(
                    p, (w, h), cache_dir=cache_dir, image_dir=data_dir
                )
                items.append((npz_path, key, latents[i].float().numpy()))
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{p.name} → demote {bucket[0]}x{bucket[1]}")
            _save_demoted_batch(items)

    return stats


def cache_latents(
    data_dir: Path,
    vae,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    keep_stems: "set[str] | frozenset[str] | None" = None,
    image_transform: "Callable[[np.ndarray], np.ndarray] | None" = None,
    batch_size: int = 4,
    progress: ProgressFn | None = None,
    io_workers: int | None = None,
    overwrite: bool = False,
) -> PreprocessStats:
    """Encode every image under ``data_dir`` through ``vae`` → latent NPZs.

    ``vae`` is supplied loaded + on-device (``device``/``dtype`` are read off
    it). Returns counts; pass ``progress`` for a per-image bar. With
    ``overwrite`` the per-resolution skip is bypassed and every (W,H) latent is
    re-encoded (the matching ``latents_{H}x{W}`` key is replaced in place,
    other-resolution keys preserved).

    The VAE forward stays serial on the calling thread (single GPU stream); the
    per-batch disk decode + image transform and the npz read-modify-write are
    CPU/IO, so they're farmed to thread pools that overlap the GPU.
    ``io_workers`` sizes those pools (default
    ``min(8, cpu_count)``). Output is byte-identical to the serial path.

    ``keep_stems`` (when given) restricts the walk to images whose stem is in
    the set — used by cond≠target tasks to encode only the paired subset.
    ``image_transform`` (uint8 RGB (H,W,3) → same) is applied to each decoded
    image before VAE encoding — e.g. colorize's target white-balance. NB the
    per-resolution skip still keys on the *unmodified* cache name, so changing
    the transform requires ``overwrite`` (or a fresh ``cache_dir``)."""
    image_files = walk_images(data_dir, recursive=recursive, pattern=path_pattern)
    if keep_stems is not None:
        image_files = [p for p in image_files if p.stem in keep_stems]
    reso_groups = group_by_shape(image_files)
    stats = PreprocessStats(seen=len(image_files))

    if progress is not None:
        progress(0, total=len(image_files))

    batches: list[tuple[int, int, list[Path]]] = []
    for (w, h), paths in reso_groups.items():
        for s in range(0, len(paths), batch_size):
            batches.append((w, h, paths[s : s + batch_size]))

    workers = io_workers or min(8, (os.cpu_count() or 4))
    depth = max(2, workers // 2)  # decoded batches kept in flight (bounds host RAM)
    max_saves = max(2, workers)  # in-flight npz writes before backpressure
    it = iter(batches)
    decode_q: deque = deque()
    save_q: deque = deque()
    failed_all: list[tuple[Path, str]] = []

    def _submit_decode(decode_ex) -> bool:
        b = next(it, None)
        if b is None:
            return False
        w, h, bp = b
        decode_q.append(
            decode_ex.submit(
                _decode_batch, bp, w, h, cache_dir, data_dir, overwrite, image_transform
            )
        )
        return True

    with (
        ThreadPoolExecutor(max_workers=workers) as decode_ex,
        ThreadPoolExecutor(max_workers=workers) as save_ex,
    ):
        for _ in range(depth):  # prime the decode window
            if not _submit_decode(decode_ex):
                break
        while decode_q:
            skipped, failed, kept, img_batch = decode_q.popleft().result()
            _submit_decode(decode_ex)  # keep a decode in flight

            for p in skipped:
                stats.skipped += 1
                if progress is not None:
                    progress(1, detail=f"skip {p.name}")
            for p, reason in failed:  # isolate corrupt files; don't abort the run
                failed_all.append((p, reason))
                if progress is not None:
                    progress(1, detail=f"FAILED {p.name}")
            if img_batch is None:
                continue

            img_batch = img_batch.to(device=vae.device, dtype=vae.dtype)
            with torch.no_grad():
                latents = vae.encode_pixels_to_latents(img_batch).cpu()

            items: list[tuple[Path, np.ndarray, tuple[int, int]]] = []
            for i, (p, size) in enumerate(kept):
                npz_path = get_latents_npz_path(
                    p, size, cache_dir=cache_dir, image_dir=data_dir
                )
                items.append((npz_path, latents[i].float().numpy(), size))
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{p.name} → {size[0]}x{size[1]}")
            save_q.append(save_ex.submit(_save_batch, items))

            while len(save_q) >= max_saves:  # backpressure on the save side
                save_q.popleft().result()
        for f in save_q:  # surface any write error + drain remaining writes
            f.result()

    if failed_all:
        print(
            f"\n⚠ {len(failed_all)} image(s) could not be decoded and were skipped "
            f"(no latent cached — re-stage these, e.g. delete + re-run mangafy):"
        )
        for p, reason in failed_all:
            print(f"  {p}  ({reason})")

    return stats
