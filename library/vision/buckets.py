"""Aspect-preserving bucket spec for img2emb vision encoders.

Each bucket is an integer ``(h_patches, w_patches)`` pair — pixel size is
``(patch*h, patch*w)`` and patch-token count is ``h*w`` (plus 1 CLS when the
encoder prepends one). Buckets are tuned so every entry lands near the
encoder's pretraining patch count, keeping the ViT's positional embeddings
close to their training distribution under interpolation.

Images are zero-padded to the encoder's ``t_max_tokens`` at preprocess time
so the cache stays a single ``(N, T_MAX_TOKENS, D)`` tensor and the resampler
doesn't need a padding mask.

**Train and test must use the same encoder + bucket spec**: a resampler
trained on one encoder's ``T``/``D`` silently sees a shifted KV length
otherwise.
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


# PE-Core-L14-336 — 336px native, 24x24=576 patch tokens + 1 CLS. Buckets sized
# so h*w ~ 576 (within ~2%); dims are multiples of patch=14.
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


# PE-Spatial-B16-512 — 512px native, 32x32=1024 patch tokens + 1 CLS. Buckets
# sized so h*w ~ 1024 (within ~3.5%); dims are multiples of patch=16.
# Aspects mirror PE-Core so a bucket choice maps 1:1 across both specs.
PE_SPATIAL_B16_512_SPEC = BucketSpec(
    encoder="pe_spatial",
    patch=16,
    use_cls=True,
    buckets=(
        (46, 23),  # 2.00 portrait, 1058 tokens, 736x368 px
        (39, 27),  # 1.44 portrait, 1053 tokens, 624x432 px
        (37, 28),  # 1.32 portrait, 1036 tokens, 592x448 px
        (32, 32),  # 1.00 square,   1024 tokens, 512x512 px (native)
        (28, 37),  # 0.76 landscape,1036 tokens, 448x592 px
        (27, 39),  # 0.69 landscape,1053 tokens, 432x624 px
        (23, 46),  # 0.50 landscape,1058 tokens, 368x736 px
    ),
)


# PE-Spatial-L14-448 — 448px native, 32x32=1024 patch tokens + 1 CLS. Same
# patch grid as B16-512 (identical bucket list), just patch=14 so pixel dims
# are 14/16 of the B16 ones (e.g. native 448x448).
PE_SPATIAL_L14_448_SPEC = BucketSpec(
    encoder="pe_spatial_l",
    patch=14,
    use_cls=True,
    buckets=PE_SPATIAL_B16_512_SPEC.buckets,
)


_SPECS: dict[str, BucketSpec] = {
    "pe": PE_CORE_L14_336_SPEC,
    "pe_spatial": PE_SPATIAL_B16_512_SPEC,
    "pe_spatial_l": PE_SPATIAL_L14_448_SPEC,
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
