"""Resize-preview helpers shared by preprocess GUI surfaces.

The resize step cover-scales an image to its free-fit bucket and anchor-crops
to it. The geometry is ``anime_tools.stages.resize`` (the owner since the
API-first migration, 2026-09-03); this module re-exports it under the names the
GUI grew up with and adds the preview rectangle math, so a preview shows
exactly what the stage will keep without touching files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from anime_tools.stages.resize import (  # noqa: F401 — re-exports
    CROP_ANCHORS as RESIZE_CROP_ANCHORS,
)
from anime_tools.stages.resize import (  # noqa: F401 — re-exports
    DEFAULT_CROP_ANCHOR as DEFAULT_RESIZE_CROP_ANCHOR,
)
from anime_tools.stages.resize import (  # noqa: F401 — re-exports
    MARGIN_SIDES,
    normalize_crop_anchor,
    normalize_target_res,
    select_bucket,
)
from anime_tools.stages.resize import (
    normalize_crop_margins as _normalize_crop_margins_tuple,
)
from library.datasets.buckets import DEFAULT_FREEFIT_MAX_RATIO

# Free-fit is the only resize mode; "snap" (the old discrete constant-token bucket
# pool) was removed. FIT_MODES / DEFAULT_FIT_MODE are kept so existing call
# sites (GUI preview config tuples) stay stable.
FIT_MODES = ("freefit",)
DEFAULT_FIT_MODE = "freefit"


@dataclass(frozen=True)
class CropRect:
    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class ResizePreview:
    source_size: tuple[int, int]
    target_edge: int
    bucket_size: tuple[int, int]
    kept_rect: CropRect
    margin_rect: CropRect
    crop_anchor: str
    crop_margins: dict[str, float]


def normalize_crop_margins(raw) -> dict[str, float]:
    """Percent margins as a ``{top, right, bottom, left}`` dict (the GUI's
    shape) — the package's tuple normalizer, keyed."""
    values = _normalize_crop_margins_tuple(raw)
    return dict(zip(MARGIN_SIDES, values, strict=True))


def margin_crop_rect(width: int, height: int, crop_margins=None) -> CropRect:
    margins = normalize_crop_margins(crop_margins)
    left = width * margins["left"] / 100.0
    top = height * margins["top"] / 100.0
    right = width - width * margins["right"] / 100.0
    bottom = height - height * margins["bottom"] / 100.0
    return CropRect(
        left=left, top=top, width=max(1.0, right - left), height=max(1.0, bottom - top)
    )


def parse_bucket_resos(raw) -> list[tuple[int, int]]:
    """Normalize bucket filters from TOML/CLI values.

    Accepts ``["1008x1024"]``, ``["1008,1024"]``, ``[(1008, 1024)]``, or a
    comma-separated string. Empty input means "all supported buckets".
    """
    if raw is None:
        return []
    values = raw
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    out: list[tuple[int, int]] = []
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            width, height = int(item[0]), int(item[1])
        else:
            text = str(item).strip().lower().replace("×", "x")
            if "x" in text:
                left, right = text.split("x", 1)
            elif ":" in text:
                left, right = text.split(":", 1)
            else:
                continue
            width, height = int(left.strip()), int(right.strip())
        if width > 0 and height > 0:
            out.append((width, height))
    return sorted(set(out))


def format_bucket_resos(bucket_resos: Iterable[tuple[int, int]]) -> list[str]:
    return [f"{width}x{height}" for width, height in bucket_resos]


def normalize_fit_mode(fit_mode: str | None) -> str:
    value = str(fit_mode or DEFAULT_FIT_MODE).strip().lower()
    return value if value in FIT_MODES else DEFAULT_FIT_MODE


def select_resize_bucket(
    width: int,
    height: int,
    target_res: Iterable[int] | int | str | None = None,
    bucket_resos=None,
    *,
    fit_mode: str = DEFAULT_FIT_MODE,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
) -> tuple[int, tuple[int, int]]:
    """``(tier_edge, (W, H))`` for a source size — ``anime_tools.stages.resize.
    select_bucket``. ``fit_mode`` / ``bucket_resos`` are accepted for signature
    compatibility but no longer branch (free-fit is the only mode)."""
    return select_bucket(width, height, target_res, max_ratio=max_ratio)


def compute_resize_preview(
    width: int,
    height: int,
    target_res: Iterable[int] | int | str | None = None,
    *,
    crop_anchor: str | None = None,
    bucket_resos=None,
    crop_margins=None,
    fit_mode: str = DEFAULT_FIT_MODE,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
) -> ResizePreview:
    """Return the bucket and source-space crop rect used by preprocessing.

    ``fit_mode="freefit"`` runs the free-aspect token-band solver instead of
    snapping to a discrete bucket; the same ``select_resize_bucket`` feeds both
    this preview and ``process_image``, so the GUI/CLI preview is exact.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    anchor = normalize_crop_anchor(crop_anchor)
    anchor_x, anchor_y = RESIZE_CROP_ANCHORS[anchor]
    margins = normalize_crop_margins(crop_margins)
    margin_rect = margin_crop_rect(width, height, margins)
    work_w = max(1, round(margin_rect.width))
    work_h = max(1, round(margin_rect.height))
    edge, (bucket_w, bucket_h) = select_resize_bucket(
        work_w, work_h, target_res, bucket_resos, fit_mode=fit_mode, max_ratio=max_ratio
    )

    source_ar = work_w / work_h
    bucket_ar = bucket_w / bucket_h
    if source_ar > bucket_ar:
        kept_h = float(work_h)
        kept_w = kept_h * bucket_ar
        left = margin_rect.left + (work_w - kept_w) * anchor_x
        top = margin_rect.top
    else:
        kept_w = float(work_w)
        kept_h = kept_w / bucket_ar
        left = margin_rect.left
        top = margin_rect.top + (work_h - kept_h) * anchor_y

    return ResizePreview(
        source_size=(width, height),
        target_edge=edge,
        bucket_size=(bucket_w, bucket_h),
        kept_rect=CropRect(left=left, top=top, width=kept_w, height=kept_h),
        margin_rect=margin_rect,
        crop_anchor=anchor,
        crop_margins=margins,
    )
