"""A run's files on disk: resumable latents, PNG paths, the manifest."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from library.moss.requests import GenerateRequest


def latent_path(out_dir: Path, rec: dict) -> Path:
    return (
        out_dir / "latents" / f"{rec['key']}_{rec['model']}_s{rec['seed']}.safetensors"
    )


def image_path(out_dir: Path, rec: dict) -> Path:
    return out_dir / f"{rec['key']}_{rec['model']}_s{rec['seed']}.png"


def save_latent(path: Path, latent: torch.Tensor, record: dict) -> None:
    """Final latent + its record in the metadata, written atomically."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(
        {"latent": latent.contiguous()},
        tmp,
        metadata={"record": json.dumps(record, ensure_ascii=False)},
    )
    tmp.replace(path)


def read_latent_record(path: Path) -> dict | None:
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return json.loads((f.metadata() or {})["record"])
    except Exception:
        return None


def load_latent(path: Path) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(path)["latent"]


def write_manifest(out_dir: Path, req: GenerateRequest, records: list[dict]) -> Path:
    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"request": req.to_argv(), "images": records}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return manifest
