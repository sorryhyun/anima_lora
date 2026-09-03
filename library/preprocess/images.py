"""Resize a dataset directory into free-fit bucket resolutions.

The resize pass is ``anime_tools.stages.resize`` (the owner since the
API-first migration, 2026-09-03): ``make preprocess-resize`` runs it as a
``ResizeRequest``. What stays here is the programmatic wrapper embedders and
tests call — :func:`resize_to_buckets`, with the trainer's ``PreprocessStats``
shape, ``curation_decisions`` (translated into the stage's ``skip`` set) and
``ProgressFn`` — plus re-exports of the stage's pixel geometry
(:func:`resize_to_bucket`, :func:`process_image`, :class:`ResizeOptions`).
"""

from __future__ import annotations

from pathlib import Path

from anime_tools.stages.resize import (  # noqa: F401 — re-exports
    CAPTION_EXTENSIONS,
    ResizeOptions,
    ResizeStats,
    process_image,
    rel_key,
    resize_to_bucket,
    run_resize_images,
)
from library.datasets.buckets import DEFAULT_FREEFIT_MAX_RATIO, DEFAULT_TARGET_RES
from library.preprocess._dataset import PreprocessStats
from library.preprocess._progress import ProgressFn
from library.preprocess.resize_preview import (
    DEFAULT_FIT_MODE,
    DEFAULT_RESIZE_CROP_ANCHOR,
)


def curation_skips(decisions: dict[str, dict] | None) -> set[str]:
    """The ``skip`` set a GUI curation-decision map implies: every image whose
    ``action`` is ``skip`` or ``move`` (keys are paths relative to the source
    root, as ``load_curation_decisions`` returns them)."""
    if not decisions:
        return set()
    return {
        rel
        for rel, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("action") in {"skip", "move"}
    }


def resize_to_buckets(
    src: Path,
    dst: Path,
    *,
    resolution: int = 1024,
    min_bucket_reso: int = 512,
    max_bucket_reso: int = 2048,
    bucket_reso_steps: int = 64,
    target_res: list[int] | None = None,
    workers: int = 4,
    min_pixels: int = 500_000,
    copy_captions: bool = True,
    recursive: bool = False,
    path_pattern: str | None = None,
    verbose: bool = True,
    overwrite: bool = False,
    curation_decisions: dict[str, dict] | None = None,
    crop_anchor: str = DEFAULT_RESIZE_CROP_ANCHOR,
    bucket_resos=None,
    crop_margins=None,
    fit_mode: str = DEFAULT_FIT_MODE,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    progress: ProgressFn | None = None,
) -> tuple[PreprocessStats, dict[tuple[int, int], int]]:
    """Resize+crop every image under ``src`` into bucket resolutions under ``dst``.

    Mirrors the source subdir layout, copies caption sidecars, and skips images
    below ``min_pixels`` and those a curation decision marks ``skip`` /
    ``move``. Returns ``(stats, bucket_counts)`` where ``bucket_counts`` maps
    each ``(W, H)`` bucket to its image count (skipped + written) and
    ``stats.skipped`` counts every image not (re)written this run — too small,
    decided against, or already at its bucket. Pass ``progress`` for a
    per-image bar.

    ``resolution`` / ``min_bucket_reso`` / ``max_bucket_reso`` /
    ``bucket_reso_steps`` / ``bucket_resos`` / ``fit_mode`` are the snap-era
    knobs, accepted for signature stability and inert under free-fit.
    """
    options = ResizeOptions.build(
        target_res=target_res or list(DEFAULT_TARGET_RES),
        crop_anchor=crop_anchor,
        crop_margins=crop_margins,
        max_ratio=max_ratio,
    )
    skip = curation_skips(curation_decisions)
    if skip and verbose:
        print(f"Skipping {len(skip)} image(s) marked by curation decisions:")
        for rel in sorted(skip):
            print(f"  {rel}")

    if verbose:
        tiers = sorted(options.target_res)
        print(
            f"Resizing images under {src} to free-fit (tiers {tiers}, band, "
            f"max_ratio {options.max_ratio:g}) buckets"
        )
    total_hint = [0]
    if progress is not None:
        progress(0, total=0)

    def _progress(index: int, total: int, detail: str) -> None:
        if progress is None:
            return
        if total_hint[0] != total:
            total_hint[0] = total
            progress(0, total=total)
        progress(1, detail=detail)

    result: ResizeStats = run_resize_images(
        src=Path(src),
        dst=Path(dst),
        options=options,
        path_pattern=path_pattern,
        recursive=recursive,
        min_pixels=min_pixels,
        copy_captions=copy_captions,
        overwrite=overwrite,
        workers=workers,
        skip=skip,
        progress=_progress,
    )

    stats = PreprocessStats(
        seen=result.seen,
        written=result.written,
        skipped=result.skipped_small + result.skipped_excluded + result.skipped_current,
        failed=result.failed,
    )
    bucket_counts: dict[tuple[int, int], int] = {}
    for key, count in result.buckets.items():
        w, h = (int(v) for v in key.split("x"))
        bucket_counts[(w, h)] = count

    if verbose:
        if result.too_small:
            print(f"Skipped {result.skipped_small} images below {min_pixels:,} pixels:")
            for line in result.too_small:
                print(f"  {line}")
        for line in result.failures:
            print(f"  fail: {line}")
        if result.skipped_current:
            print(
                f"Skipped {result.skipped_current} image(s) already at their target "
                f"bucket (pass --overwrite to force re-resize); "
                f"{result.written} (re)written."
            )
        print("\nBucket distribution:")
        for reso in sorted(bucket_counts):
            tokens = (reso[0] // 16) * (reso[1] // 16)
            print(
                f"  {reso[0]:>4d}x{reso[1]:<4d}: {bucket_counts[reso]:>3d} "
                f"images  ({tokens} tokens)"
            )
    return stats, bucket_counts
