"""Vision-encoder registry (originally for img2emb; reused live by IP-Adapter).

Three encoders are wired in: TIPSv2-L/14 (Google, ``trust_remote_code``-loaded
HF custom model), PE-Core-L14-336, and PE-Core-G14-448 (both Meta Perception
Encoder, vendored at ``library/models/pe.py`` so we don't have to clone
perception_models or install xformers).

All expose the same shape for downstream code: ``encode(pixel_values)``
returns ``(last_hidden_state[B, T, D], pooled[B, D_pool])``. ``T`` includes
a CLS token at position 0 only when the encoder's ``BucketSpec.use_cls`` is
true (TIPSv2 and PE-Core-L14-336 do; PE-Core-G14-448 does not).
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


# --------------------------------------------------------------------------- shared output shim


class _EncoderOutput:
    """Minimal HF ``BaseModelOutput``-shaped container."""

    __slots__ = ("last_hidden_state", "pooler_output")

    def __init__(self, last_hidden_state: torch.Tensor, pooler_output: torch.Tensor):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


# --------------------------------------------------------------------------- TIPSv2-L/14


def _default_tipsv2_model_id() -> str:
    return str(REPO_ROOT / "models" / "tipsv2")


class _TIPSv2Processor:
    """TIPSv2's reference preprocessing: Resize + ToTensor in [0, 1]; no
    ImageNet mean/std. Accepts square ``int`` or ``(H, W)`` tuple."""

    def __init__(self, image_size):
        from torchvision import transforms

        size_hw = (image_size, image_size) if isinstance(image_size, int) else (
            int(image_size[0]), int(image_size[1])
        )
        self.image_size = size_hw
        self.transform = transforms.Compose(
            [transforms.Resize(size_hw), transforms.ToTensor()]
        )

    def __call__(self, images, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        if not isinstance(images, (list, tuple)):
            images = [images]
        return {
            "pixel_values": torch.stack([self.transform(img) for img in images], dim=0)
        }


class _TIPSv2Encoder:
    """Adapt TIPSv2's ``encode_image`` API to ``last_hidden_state`` /
    ``pooler_output``. ``last_hidden_state`` = CLS prepended to patch tokens."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, pixel_values: torch.Tensor) -> _EncoderOutput:
        out = self.inner.encode_image(pixel_values)
        if isinstance(out, (tuple, list)):
            cls, patches = out[0], out[1]
        elif isinstance(out, dict):
            cls = out.get("cls_token", out.get("cls"))
            patches = out.get("patch_tokens", out.get("patches"))
        else:
            cls = getattr(out, "cls_token", None)
            patches = getattr(out, "patch_tokens", None)
        if cls is None or patches is None:
            raise RuntimeError(
                f"TIPSv2 encode_image returned unexpected structure: {type(out)}"
            )
        if cls.dim() == 2:
            cls = cls.unsqueeze(1)
        last_hidden = torch.cat([cls, patches], dim=1)
        pooled = cls.squeeze(1)
        return _EncoderOutput(last_hidden_state=last_hidden, pooler_output=pooled)


def _ensure_tipsv2_siblings_cached(model_path: str) -> None:
    """TIPSv2's ``modeling_tips.py`` imports image_encoder.py / text_encoder.py
    as siblings at __init__ time. ``trust_remote_code`` only copies files
    listed in ``auto_map`` into the transformers_modules cache, so these
    siblings go missing; the fallback then calls hf_hub_download with the
    local path as repo_id and raises HFValidationError. Pre-copy them here."""
    src_dir = Path(model_path)
    if not src_dir.is_dir():
        return
    cache_dir = (
        Path.home() / ".cache/huggingface/modules/transformers_modules" / src_dir.name
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for sibling in ("image_encoder.py", "text_encoder.py"):
        src = src_dir / sibling
        if src.exists():
            shutil.copy2(src, cache_dir / sibling)


def _load_tipsv2_encoder(device: torch.device, model_id: str) -> _TIPSv2Encoder:
    from transformers import AutoModel

    logger.info(f"Loading tipsv2: {model_id}")
    _ensure_tipsv2_siblings_cached(model_id)
    inner = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    inner.eval().to(device).requires_grad_(False)
    return _TIPSv2Encoder(inner)


# --------------------------------------------------------------------------- PE-Core (L14-336 / G14-448)


def _default_pe_model_id() -> str:
    return str(REPO_ROOT / "models" / "pe" / "PE-Core-L14-336.pt")


def _default_pe_g_model_id() -> str:
    return str(REPO_ROOT / "models" / "pe" / "PE-Core-G14-448.pt")


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
    ``(last_hidden_state, pooler_output)`` matching TIPSv2's contract."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, pixel_values: torch.Tensor) -> _EncoderOutput:
        feats, pooled = self.inner.encode(pixel_values)
        return _EncoderOutput(last_hidden_state=feats, pooler_output=pooled)


def _make_pe_loader(
    config_name: str, download_target: str
) -> Callable[[torch.device, str], "_PEEncoder"]:
    """Return a loader closure that builds the vendored PE vision tower
    against the named config and loads Meta's official ``.pt`` checkpoint
    (CLIP-format) into it.

    ``model_id`` passed to the closure is a *local file path* to the
    ``.pt``, fetched via ``make download-pe`` / ``make download-pe-g``.
    """

    def _loader(device: torch.device, model_id: str) -> _PEEncoder:
        from library.models.pe import build_pe_vision

        ckpt_path = Path(model_id)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"PE checkpoint not found at {ckpt_path}. "
                f"Run `make {download_target}` to fetch facebook/{config_name}."
            )
        logger.info(f"Loading {config_name} from {ckpt_path}")
        model = build_pe_vision(config_name)
        model.load_pe_checkpoint(str(ckpt_path), verbose=True)
        model = model.to(dtype=torch.bfloat16, device=device).eval()
        model.requires_grad_(False)
        return _PEEncoder(model)

    return _loader


_load_pe_encoder = _make_pe_loader("PE-Core-L14-336", "download-pe")
_load_pe_g_encoder = _make_pe_loader("PE-Core-G14-448", "download-pe-g")


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
    "tipsv2": EncoderInfo(
        name="tipsv2",
        bucket_spec=get_bucket_spec("tipsv2"),
        d_enc=1024,
        d_pool=1024,
        default_model_id=_default_tipsv2_model_id,
        processor_factory=_TIPSv2Processor,
        loader=_load_tipsv2_encoder,
    ),
    "pe": EncoderInfo(
        name="pe",
        bucket_spec=get_bucket_spec("pe"),
        d_enc=1024,
        d_pool=1024,
        default_model_id=_default_pe_model_id,
        processor_factory=_PEProcessor,
        loader=_load_pe_encoder,
    ),
    # PE-Core-G14-448: width=1536 (un-projected feats), output_dim=1280
    # (projected pooled output). use_cls_token=False, so feats has no CLS row.
    "pe-g": EncoderInfo(
        name="pe-g",
        bucket_spec=get_bucket_spec("pe-g"),
        d_enc=1536,
        d_pool=1280,
        default_model_id=_default_pe_g_model_id,
        processor_factory=_PEProcessor,
        loader=_load_pe_g_encoder,
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
