"""Model and safetensors hash helpers used for training metadata.

These write-side helpers compute the hashes embedded in LoRA metadata
(`ss_sd_model_hash`, `sshs_model_hash`, `sshs_legacy_hash`, ...) so that the
resulting checkpoint can be indexed by sd-webui-additional-networks and the
wider stable-diffusion-webui ecosystem.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from io import BytesIO

import safetensors.torch


def model_hash(filename: str) -> str:
    """Old model hash used by stable-diffusion-webui."""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:
        return "IsADirectory"
    except PermissionError:
        return "IsADirectory"


def calculate_sha256(filename: str) -> str:
    """New model hash used by stable-diffusion-webui."""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:
        return "IsADirectory"
    except PermissionError:
        return "IsADirectory"


def addnet_hash_legacy(b: BytesIO) -> str:
    """Old model hash used by sd-webui-additional-networks for .safetensors files."""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b: BytesIO) -> str:
    """New model hash used by sd-webui-additional-networks for .safetensors files."""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata: dict) -> tuple[str, str]:
    """Precalculate the model hashes needed by sd-webui-additional-networks.

    Keeping the `ss_`-prefixed metadata keys only matches the original
    sd-scripts behavior — hashes include the metadata block, so callers that
    add non-`ss_` keys later do not invalidate the precomputed value.
    """
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes_ = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes_)

    mh = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return mh, legacy_hash


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "(unknown)"
