"""Live vision-encoder wrapper for IP-Adapter / similar consumers.

Wraps PE-Core (or any encoder registered in library/vision/encoders.py) for
in-loop use during training and inference: accepts a tensor batch already
produced by the training dataset (in [-1, 1]) and re-buckets / resizes it
on-the-fly to the encoder's patch grid.

PE-Core uses Normalize(0.5, 0.5) i.e. maps [0, 1] -> [-1, 1]. Anima's training
``IMAGE_TRANSFORMS`` is also ToTensor + Normalize(0.5, 0.5), so a tensor read
from ``batch["images"]`` is already in PE's expected range and only needs to
be resized to the chosen bucket's pixel dims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from library.vision.buckets import BucketSpec, pick_bucket
from library.vision.encoders import EncoderInfo, get_encoder_info

logger = logging.getLogger(__name__)


# PE-Core normalization (also matches Anima's IMAGE_TRANSFORMS — Normalize(0.5, 0.5)).
PE_NORM_MEAN = (0.5, 0.5, 0.5)
PE_NORM_STD = (0.5, 0.5, 0.5)


@dataclass
class VisionEncoderBundle:
    """Loaded encoder + its registry info — what callers need to produce features."""

    name: str
    encoder: object  # _PEEncoder, callable on pixel_values
    info: EncoderInfo
    device: torch.device
    dtype: torch.dtype

    @property
    def bucket_spec(self) -> BucketSpec:
        return self.info.bucket_spec

    @property
    def d_enc(self) -> int:
        return self.info.d_enc


def load_pe_encoder(
    device: torch.device,
    *,
    name: str = "pe",
    model_id: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> VisionEncoderBundle:
    """Load PE-Core (or another registered encoder) for live use, in eval mode
    with ``requires_grad_(False)`` (set by the loaders).

    ``dtype`` is both the model compute dtype (cast once from the fp32
    checkpoint) and ``bundle.dtype``, which ``encode_pe_from_imageminus1to1``
    casts inputs to. Feature caching passes ``float32``.
    """
    info = get_encoder_info(name)
    resolved_id = model_id or info.default_model_id()
    encoder = info.loader(device, resolved_id, dtype=dtype)
    return VisionEncoderBundle(
        name=name,
        encoder=encoder,
        info=info,
        device=device,
        dtype=dtype,
    )


def pe_resize_for_image(
    image_minus1to1: torch.Tensor,
    spec: BucketSpec,
) -> torch.Tensor:
    """Not implemented: always raises. Bucket resize + encode is
    :func:`encode_pe_from_imageminus1to1`."""
    raise NotImplementedError(
        "Use encode_pe_from_imageminus1to1 for the full preprocess+encode path; "
        "this helper exists for callers that want to inspect bucket selection."
    )


def _bucket_pixels(spec: BucketSpec, h: int, w: int) -> tuple[int, int]:
    h_p, w_p = pick_bucket(h, w, spec)
    return h_p * spec.patch, w_p * spec.patch


def encode_pe_from_imageminus1to1(
    bundle: VisionEncoderBundle,
    images: torch.Tensor,
    *,
    same_bucket: bool = False,
) -> list[torch.Tensor]:
    """Encode a batch of [-1, 1] images via the PE encoder.

    PE-Core takes dynamic resolution, so each image is resized to its own
    aspect bucket and encoded alone. Returns one ``[T_i, D]`` tensor per sample
    (T differs across buckets). ``same_bucket=True``: the caller guarantees one
    shared aspect, and the batch runs as a single forward.
    """
    spec = bundle.bucket_spec
    encoder = bundle.encoder
    device = bundle.device
    dtype = bundle.dtype

    B = images.shape[0]
    images = images.to(device=device, dtype=dtype)

    if same_bucket:
        h, w = images.shape[-2:]
        target_h, target_w = _bucket_pixels(spec, h, w)
        if (h, w) != (target_h, target_w):
            images = F.interpolate(
                images, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        out = encoder(images)
        feats = out.last_hidden_state  # [B, T, D]
        return [feats[i] for i in range(B)]

    feats_list: list[torch.Tensor] = []
    for i in range(B):
        img_i = images[i : i + 1]
        h, w = img_i.shape[-2:]
        target_h, target_w = _bucket_pixels(spec, h, w)
        if (h, w) != (target_h, target_w):
            img_i = F.interpolate(
                img_i, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        out = encoder(img_i)
        feats_list.append(out.last_hidden_state[0])
    return feats_list
