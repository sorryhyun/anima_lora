"""Aspect-preserving bucket spec for TIPSv2 (patch 14).

Each bucket is an integer ``(h_patches, w_patches)`` pair — pixel size is
``(14*h, 14*w)`` and token count is ``h*w`` (plus 1 CLS prepended by the
wrapper). The set is tuned so every bucket lands in a narrow ~1014-1058
patch-token range centered on TIPSv2-L/14's pretraining resolution
(``32x32 = 1024`` at 448x448), which keeps the ViT's learned positional
embeddings close to their training distribution under interpolation.

Images are zero-padded to ``T_MAX_TOKENS`` at preprocess time so the cache
stays a single ``(N, T_MAX_TOKENS, D)`` tensor and the resampler doesn't need
a padding mask. With all buckets within ~4% of ``T_MAX_TOKENS``, padded rows
are a small fraction; post-LayerNorm they're close to a constant bias
vector and the resampler learns to downweight them during cross-attention.

**Train / test must use the same bucket flag.** Tokens produced with
``--buckets`` have a different ``T`` dimension than the fixed 448² path; a
resampler trained on one will silently see a shifted KV length distribution
at inference time if the flag diverges.
"""

from __future__ import annotations

import math

# TIPSv2 patch size (fixed by the pretrained model).
PATCH = 14

# (h_patches, w_patches) — sorted by aspect (tall → wide). Each row's
# ``h*w`` is the bucket's patch-token count (add 1 for CLS).
TIPSV2_BUCKETS_1K: tuple[tuple[int, int], ...] = (
    (46, 23),  # 2:1 portrait,  1058 tokens, 644x322 px
    (39, 26),  # 3:2 portrait,  1014 tokens, 546x364 px
    (37, 28),  # ~4:3 portrait, 1036 tokens, 518x392 px
    (32, 32),  # 1:1,           1024 tokens, 448x448 px
    (28, 37),  # ~3:4 landscape,1036 tokens, 392x518 px
    (26, 39),  # 2:3 landscape, 1014 tokens, 364x546 px
    (23, 46),  # 1:2 landscape, 1058 tokens, 322x644 px
)

# Patch-token count budget after zero-padding. CLS adds 1 in the wrapper.
T_MAX_PATCHES = max(h * w for h, w in TIPSV2_BUCKETS_1K)  # 1058
T_MAX_TOKENS = T_MAX_PATCHES + 1                           # 1059 (incl. CLS)


def pick_bucket(img_h: int, img_w: int) -> tuple[int, int]:
    """Return the ``(h_patches, w_patches)`` bucket whose pixel aspect is
    closest to the image's in log-space (symmetric under aspect inversion).

    ``img_h`` and ``img_w`` are the source image's pixel dimensions. Bucket
    pixel size is ``(14*h_patches, 14*w_patches)``.
    """
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"image dims must be positive, got h={img_h} w={img_w}")
    target = math.log(img_h / img_w)
    best = min(
        TIPSV2_BUCKETS_1K,
        key=lambda hw: abs(math.log(hw[0] / hw[1]) - target),
    )
    return best


def bucket_pixel_size(bucket: tuple[int, int]) -> tuple[int, int]:
    """Return ``(H_pixels, W_pixels)`` for a ``(h_patches, w_patches)`` bucket."""
    h, w = bucket
    return h * PATCH, w * PATCH


def bucket_token_count(bucket: tuple[int, int], include_cls: bool = True) -> int:
    """Return token count for a bucket (``h*w`` patches, ``+1`` for CLS)."""
    h, w = bucket
    return h * w + (1 if include_cls else 0)
