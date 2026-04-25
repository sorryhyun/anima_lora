"""Aspect-preserving bucket spec for img2emb vision encoders.

Each bucket is an integer ``(h_patches, w_patches)`` pair — pixel size is
``(patch*h, patch*w)`` and patch-token count is ``h*w`` (plus 1 CLS when the
encoder prepends one). Buckets for an encoder are tuned so every entry lands
near the encoder's pretraining patch count, keeping the ViT's learned
positional embeddings close to their training distribution under interpolation.

Images are zero-padded to the encoder's ``t_max_tokens`` at preprocess time
so the cache stays a single ``(N, T_MAX_TOKENS, D)`` tensor and the resampler
doesn't need a padding mask.

**Train / test must use the same encoder + bucket spec.** Tokens produced by
one encoder have a different ``T`` and ``D`` than another; a resampler
trained on one will silently see a shifted KV length distribution at
inference time if the encoder flag diverges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketSpec:
    """Per-encoder bucket spec — patch size, CLS-prepended flag, bucket list."""
    encoder: str
    patch: int
    use_cls: bool
    buckets: tuple[tuple[int, int], ...]

    @property
    def t_max_patches(self) -> int:
        return max(h * w for h, w in self.buckets)

    @property
    def t_max_tokens(self) -> int:
        return self.t_max_patches + (1 if self.use_cls else 0)


# TIPSv2-L/14 — 448px native, 32x32=1024 patch tokens + 1 CLS. Buckets ~1014-1058.
TIPSV2_SPEC = BucketSpec(
    encoder="tipsv2",
    patch=14,
    use_cls=True,
    buckets=(
        (46, 23),  # 2:1 portrait,  1058 tokens, 644x322 px
        (39, 26),  # 3:2 portrait,  1014 tokens, 546x364 px
        (37, 28),  # ~4:3 portrait, 1036 tokens, 518x392 px
        (32, 32),  # 1:1,           1024 tokens, 448x448 px
        (28, 37),  # ~3:4 landscape,1036 tokens, 392x518 px
        (26, 39),  # 2:3 landscape, 1014 tokens, 364x546 px
        (23, 46),  # 1:2 landscape, 1058 tokens, 322x644 px
    ),
)

# PE-Core-L14-336 — 336px native, 24x24=576 patch tokens + 1 CLS (use_cls_token=True).
# Buckets sized so h*w ~ 576 (within ~2%). Patch=14, dimensions chosen as
# integer multiples of 14 so input H/W are divisible by patch_size.
PE_CORE_L14_336_SPEC = BucketSpec(
    encoder="pe",
    patch=14,
    use_cls=True,
    buckets=(
        (34, 17),  # 2.00 portrait, 578 tokens, 476x238 px
        (29, 20),  # 1.45 portrait, 580 tokens, 406x280 px
        (28, 21),  # 1.33 portrait, 588 tokens, 392x294 px
        (24, 24),  # 1.00 square,   576 tokens, 336x336 px (native)
        (21, 28),  # 0.75 landscape,588 tokens, 294x392 px
        (20, 29),  # 0.69 landscape,580 tokens, 280x406 px
        (17, 34),  # 0.50 landscape,578 tokens, 238x476 px
    ),
)


_SPECS: dict[str, BucketSpec] = {
    "tipsv2": TIPSV2_SPEC,
    "pe": PE_CORE_L14_336_SPEC,
}


def get_bucket_spec(encoder: str) -> BucketSpec:
    if encoder not in _SPECS:
        raise KeyError(
            f"Unknown encoder {encoder!r}; available: {sorted(_SPECS)}"
        )
    return _SPECS[encoder]


def pick_bucket(img_h: int, img_w: int, spec: BucketSpec) -> tuple[int, int]:
    """Return the bucket whose pixel aspect is closest to the image's in
    log-space (symmetric under aspect inversion).

    ``img_h`` / ``img_w`` are the source image's pixel dimensions. Bucket
    pixel size is ``(spec.patch*h_patches, spec.patch*w_patches)``.
    """
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"image dims must be positive, got h={img_h} w={img_w}")
    target = math.log(img_h / img_w)
    best = min(
        spec.buckets,
        key=lambda hw: abs(math.log(hw[0] / hw[1]) - target),
    )
    return best


def bucket_pixel_size(bucket: tuple[int, int], spec: BucketSpec) -> tuple[int, int]:
    """Return ``(H_pixels, W_pixels)`` for a bucket given the encoder's patch size."""
    h, w = bucket
    return h * spec.patch, w * spec.patch


def bucket_token_count(bucket: tuple[int, int], spec: BucketSpec) -> int:
    """Return token count for a bucket (``h*w`` patches, ``+1`` if CLS-prepended)."""
    h, w = bucket
    return h * w + (1 if spec.use_cls else 0)
