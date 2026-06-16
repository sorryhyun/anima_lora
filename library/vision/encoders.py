"""Vision-encoder registry (originally for img2emb; reused live by IP-Adapter
and the Anima Tagger).

Two Meta Perception Encoder variants are registered:

* ``pe`` — PE-Core-L14-336. Global / CLIP-aligned features. Used by IP-Adapter,
  the DCW v4 fusion head's PE feature cache, and the Anima Tagger's primary
  encoder.
* ``pe_spatial`` — PE-Spatial-B16-512. Spatial-fine-tuned variant from the same
  paper. Patch=16, native 32×32 grid. No CLIP projection / no LN-post / no
  pool head — only the patch token sequence is meaningful. Used by the Anima
  Tagger as the auxiliary encoder for spatial detail / long-tail tags.

Both vendored at ``library/models/pe.py`` so we don't have to clone
perception_models or install xformers.

``encode(pixel_values)`` returns ``(last_hidden_state[B, T, D],
pooled[B, D_pool])``. ``T`` includes a CLS token at position 0 for both
variants. For ``pe_spatial`` the "pooled" output is just the token sequence
again (pool_type="none"); callers should consume ``last_hidden_state`` only.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from library.vision.buckets import BucketSpec, get_bucket_spec

logger = logging.getLogger(__name__)

# Repo root, used by the ``_default_*_model_id`` helpers to point at vendored
# checkpoints under ``models/``. encoders.py lives at
# ``library/vision/encoders.py`` so two ``parents`` jumps land on the repo
# root regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- shared output shim


class _EncoderOutput:
    """Minimal HF ``BaseModelOutput``-shaped container."""

    __slots__ = ("last_hidden_state", "pooler_output")

    def __init__(self, last_hidden_state: torch.Tensor, pooler_output: torch.Tensor):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


# --------------------------------------------------------------------------- PE family


def _default_pe_model_id() -> str:
    return str(REPO_ROOT / "models" / "pe" / "PE-Core-L14-336.pt")


def _default_pe_spatial_model_id() -> str:
    return str(REPO_ROOT / "models" / "pe" / "PE-Spatial-B16-512.pt")


class _PEProcessor:
    """PE's reference preprocessing: squash-resize + ToTensor + ``[0.5, 0.5,
    0.5]`` mean/std (i.e. map [0,1] to [-1, 1])."""

    _MEAN = (0.5, 0.5, 0.5)
    _STD = (0.5, 0.5, 0.5)

    def __init__(self, image_size):
        from torchvision import transforms

        size_hw = (image_size, image_size) if isinstance(image_size, int) else (
            int(image_size[0]), int(image_size[1])
        )
        self.image_size = size_hw
        self.transform = transforms.Compose(
            [
                transforms.Resize(size_hw, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ]
        )

    def __call__(self, images, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        if not isinstance(images, (list, tuple)):
            images = [images]
        return {
            "pixel_values": torch.stack([self.transform(img) for img in images], dim=0)
        }


class _PEEncoder:
    """Wraps the vendored PEVisionTransformer to produce
    ``(last_hidden_state, pooler_output)``."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, pixel_values: torch.Tensor) -> _EncoderOutput:
        feats, pooled = self.inner.encode(pixel_values)
        return _EncoderOutput(last_hidden_state=feats, pooler_output=pooled)


def _load_pe_variant(
    device: torch.device,
    model_id: str,
    *,
    config_name: str,
    repo_id: str,
    filename: str,
    download_make_target: str,
) -> _PEEncoder:
    """Build a vendored PE vision tower and load Meta's official ``.pt`` weights.

    ``config_name`` selects the ``PE_CONFIGS`` entry to instantiate;
    ``repo_id`` / ``filename`` parameterize the HF auto-download fallback.
    Used by both PE-Core and PE-Spatial registry entries — the only thing
    that differs between them is the build name and the download tuple.
    """
    from library.models.pe import build_pe_vision

    ckpt_path = Path(model_id)
    if not ckpt_path.is_file():
        try:
            from library.runtime.hf_download import hf_download
        except ImportError as e:
            raise FileNotFoundError(
                f"{config_name} checkpoint not found at {ckpt_path} and "
                f"huggingface_hub is not available for auto-download. "
                f"Run `make {download_make_target}` or install huggingface_hub."
            ) from e
        logger.info(
            f"{config_name} checkpoint missing at {ckpt_path} - fetching "
            f"{repo_id}/{filename} (one-time)."
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_download(
                what=f"{config_name} checkpoint",
                hint=f"make {download_make_target}",
                repo_id=repo_id,
                filename=filename,
                local_dir=str(ckpt_path.parent),
            )
        )
        if downloaded.resolve() != ckpt_path.resolve():
            shutil.move(str(downloaded), str(ckpt_path))
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"{config_name} checkpoint expected at {ckpt_path} after "
                f"download but missing. Check huggingface_hub or run "
                f"`make {download_make_target}`."
            )
    logger.info(f"Loading {config_name} from {ckpt_path}")
    model = build_pe_vision(config_name)
    model.load_pe_checkpoint(str(ckpt_path), verbose=True)
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    model.requires_grad_(False)
    return _PEEncoder(model)


def _load_pe_encoder(device: torch.device, model_id: str) -> _PEEncoder:
    return _load_pe_variant(
        device,
        model_id,
        config_name="PE-Core-L14-336",
        repo_id="facebook/PE-Core-L14-336",
        filename="PE-Core-L14-336.pt",
        download_make_target="download-pe",
    )


def _load_pe_spatial_encoder(device: torch.device, model_id: str) -> _PEEncoder:
    return _load_pe_variant(
        device,
        model_id,
        config_name="PE-Spatial-B16-512",
        repo_id="facebook/PE-Spatial-B16-512",
        filename="PE-Spatial-B16-512.pt",
        download_make_target="download-pe-spatial",
    )


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    bucket_spec: BucketSpec
    d_enc: int
    d_pool: int
    default_model_id: Callable[[], str]
    processor_factory: Callable[..., object]  # (image_size) -> processor
    loader: Callable[[torch.device, str], object]  # (device, model_id) -> encoder

    def t_max_tokens(self) -> int:
        return self.bucket_spec.t_max_tokens


_REGISTRY: dict[str, EncoderInfo] = {
    "pe": EncoderInfo(
        name="pe",
        bucket_spec=get_bucket_spec("pe"),
        d_enc=1024,
        d_pool=1024,
        default_model_id=_default_pe_model_id,
        processor_factory=_PEProcessor,
        loader=_load_pe_encoder,
    ),
    # PE-Spatial-B16-512 — spatial-fine-tuned PE, base width. d_pool=d_enc
    # because pool_type="none" (no separate pooled output exists; downstream
    # consumers use the token sequence directly).
    "pe_spatial": EncoderInfo(
        name="pe_spatial",
        bucket_spec=get_bucket_spec("pe_spatial"),
        d_enc=768,
        d_pool=768,
        default_model_id=_default_pe_spatial_model_id,
        processor_factory=_PEProcessor,
        loader=_load_pe_spatial_encoder,
    ),
}


def get_encoder_info(name: str) -> EncoderInfo:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown encoder {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)
