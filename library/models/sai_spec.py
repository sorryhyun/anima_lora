# based on https://github.com/Stability-AI/ModelSpec
import argparse
import base64
import datetime
import logging
import mimetypes
import os
import subprocess
from dataclasses import dataclass, field
from typing import Union

from library.utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


ARCH_ANIMA_PREVIEW = "anima-preview"
ARCH_ANIMA_UNKNOWN = "anima-unknown"

ADAPTER_LORA = "lora"

IMPL_ANIMA = "https://huggingface.co/circlestone-labs/Anima"


@dataclass
class ModelSpecMetadata:
    """ModelSpec 1.0.1 compliant metadata for Anima safetensors models."""

    # === MUST ===
    architecture: str
    implementation: str
    title: str
    resolution: str
    sai_model_spec: str = "1.0.1"

    # === SHOULD ===
    description: str | None = None
    author: str | None = None
    date: str | None = None
    hash_sha256: str | None = None

    # === CAN ===
    implementation_version: str | None = None
    license: str | None = None
    usage_hint: str | None = None
    thumbnail: str | None = None
    tags: str | None = None
    merged_from: str | None = None
    trigger_phrase: str | None = None
    timestep_range: str | None = None

    additional_fields: dict[str, str] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, str]:
        metadata = {}
        for field_name, value in self.__dict__.items():
            if field_name == "additional_fields":
                for key, val in value.items():
                    key = key if key.startswith("modelspec.") else f"modelspec.{key}"
                    metadata[key] = val
            elif value is not None:
                metadata[f"modelspec.{field_name}"] = value
        return metadata


def _implementation_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            timeout=5,
        )
        if result.returncode == 0:
            return f"anima-lora/{result.stdout.strip()}"
        logger.warning("Failed to get git commit hash, using fallback")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"Could not determine git commit: {e}")
    return "anima-lora/unknown"


def _file_to_data_url(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as f:
        file_data = f.read()
    return f"data:{mime_type};base64,{base64.b64encode(file_data).decode('ascii')}"


def _resolution_string(reso: Union[int, tuple[int, int], str, None]) -> str:
    if reso is None:
        reso = (1024, 1024)
    elif isinstance(reso, str):
        reso = tuple(map(int, reso.split(",")))
    if isinstance(reso, int):
        reso = (reso, reso)
    if len(reso) == 1:
        reso = (reso[0], reso[0])
    return f"{reso[0]}x{reso[1]}"


def build_metadata_dataclass(
    lora: bool,
    timestamp: float,
    title: str | None = None,
    reso: int | tuple[int, int] | None = None,
    author: str | None = None,
    description: str | None = None,
    license: str | None = None,
    tags: str | None = None,
    merged_from: str | None = None,
    timesteps: tuple[int, int] | None = None,
    anima: str = "preview",
    optional_metadata: dict | None = None,
) -> ModelSpecMetadata:
    """Build ModelSpec 1.0.1 compliant metadata for an Anima checkpoint or LoRA."""

    architecture = ARCH_ANIMA_PREVIEW if anima == "preview" else ARCH_ANIMA_UNKNOWN
    if lora:
        architecture += f"/{ADAPTER_LORA}"

    if title is None:
        title = ("LoRA" if lora else "Checkpoint") + f"@{timestamp}"

    date = datetime.datetime.fromtimestamp(int(timestamp)).isoformat()
    resolution = _resolution_string(reso)

    timestep_range = None
    if timesteps is not None:
        if isinstance(timesteps, (int, str)):
            timesteps = (timesteps, timesteps)
        if len(timesteps) == 1:
            timesteps = (timesteps[0], timesteps[0])
        timestep_range = f"{timesteps[0]},{timesteps[1]}"

    additional = dict(optional_metadata or {})
    thumb = additional.get("thumbnail")
    if thumb and not thumb.startswith("data:"):
        try:
            additional["thumbnail"] = _file_to_data_url(thumb)
            logger.info(f"Converted thumbnail file {thumb} to data URL")
        except FileNotFoundError as e:
            logger.warning(f"Thumbnail file not found, skipping: {e}")
            additional.pop("thumbnail", None)
        except Exception as e:
            logger.warning(f"Failed to convert thumbnail to data URL: {e}")
            additional.pop("thumbnail", None)

    additional.setdefault("implementation_version", _implementation_version())

    return ModelSpecMetadata(
        architecture=architecture,
        implementation=IMPL_ANIMA,
        title=title,
        resolution=resolution,
        description=description,
        author=author,
        date=date,
        license=license,
        tags=tags,
        merged_from=merged_from,
        timestep_range=timestep_range,
        additional_fields=additional,
    )


def add_model_spec_arguments(parser: argparse.ArgumentParser):
    """Add ModelSpec metadata arguments to the parser."""
    parser.add_argument(
        "--metadata_title", type=str, default=None,
        help="title for model metadata (default is output_name)",
    )
    parser.add_argument(
        "--metadata_author", type=str, default=None,
        help="author name for model metadata",
    )
    parser.add_argument(
        "--metadata_description", type=str, default=None,
        help="description for model metadata",
    )
    parser.add_argument(
        "--metadata_license", type=str, default=None,
        help="license for model metadata",
    )
    parser.add_argument(
        "--metadata_tags", type=str, default=None,
        help="tags for model metadata, separated by comma",
    )
    parser.add_argument(
        "--metadata_usage_hint", type=str, default=None,
        help="usage hint for model metadata",
    )
    parser.add_argument(
        "--metadata_thumbnail", type=str, default=None,
        help="thumbnail image as data URL or file path (will be converted to data URL) for model metadata",
    )
    parser.add_argument(
        "--metadata_merged_from", type=str, default=None,
        help="source models for merged model metadata",
    )
    parser.add_argument(
        "--metadata_trigger_phrase", type=str, default=None,
        help="trigger phrase for model metadata",
    )
