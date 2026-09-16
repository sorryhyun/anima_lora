"""Shared argparse flag groups for tooling entry points.

``--dir`` / ``--cache_dir`` / ``--recursive`` (dataset IO) and ``--device`` /
``--dtype`` (compute) for ``preprocess/`` and ``bench/``. These helpers only
*add arguments* to a parser the caller owns (parsing/dispatch stay in the entry
point); every knob is opt-out or parameterized.

``--dtype`` strings round-trip through ``library.runtime.device.str_to_dtype``,
the same mapping ``library.runtime.harness.build_anima`` reads.
"""

from __future__ import annotations

import argparse

import torch

# All --dtype spellings accepted by library.runtime.device.str_to_dtype.
DTYPE_CHOICES: tuple[str, ...] = (
    "bf16",
    "bfloat16",
    "fp16",
    "float16",
    "fp32",
    "float32",
    "float",
)


def add_io_args(
    parser: argparse.ArgumentParser,
    *,
    dir_required: bool = True,
    dir_help: str = "Dataset directory.",
    cache_noun: str = "caches",
    include_cache_dir: bool = True,
    include_recursive: bool = True,
    include_batch_size: bool = False,
    batch_size_default: int = 8,
    include_num_workers: bool = False,
    num_workers_default: int = 4,
) -> argparse.ArgumentParser:
    """Inject the dataset-IO flag group shared by the preprocess cache scripts:
    ``--dir`` (``dir_required`` toggles required vs optional — PE's
    ``--centroid_only`` mode wants optional), ``--cache_dir``, ``--recursive``
    (mirrors source subdirs under ``--cache_dir``), and opt-in ``--batch_size``
    / ``--num_workers``. ``cache_noun`` is spliced into the ``--cache_dir``
    help (e.g. "latent caches", "PE caches").
    """
    parser.add_argument(
        "--dir",
        type=str,
        required=dir_required,
        default=None,
        help=dir_help,
    )
    if include_cache_dir:
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help=(
                f"Optional directory to write {cache_noun} into (created if "
                "needed). Defaults to writing alongside each source image."
            ),
        )
    if include_recursive:
        parser.add_argument(
            "--recursive",
            action="store_true",
            help=(
                "Walk subfolders under --dir. Caches mirror the source subdir "
                "structure under --cache_dir; stems must be unique within each "
                "subfolder but the same stem can repeat across folders."
            ),
        )
    if include_batch_size:
        parser.add_argument(
            "--batch_size",
            type=int,
            default=batch_size_default,
            help=f"Encoding batch size (default: {batch_size_default}).",
        )
    if include_num_workers:
        parser.add_argument(
            "--num_workers",
            type=int,
            default=num_workers_default,
            help=(
                "DataLoader workers for parallel PIL decode + transform. "
                f"0 = single-threaded. Default {num_workers_default}."
            ),
        )
    return parser


def add_device_args(
    parser: argparse.ArgumentParser,
    *,
    include_device: bool = True,
    include_dtype: bool = True,
    dtype_default: str = "bf16",
    dtype_choices: tuple[str, ...] = DTYPE_CHOICES,
) -> argparse.ArgumentParser:
    """Inject the compute flag group: ``--device`` and ``--dtype``.

    --device defaults to ``cuda`` when available, else ``cpu``.
    --dtype defaults to ``dtype_default`` (``bf16``, the production default);
    pass a narrower ``dtype_choices`` for scripts that only persist a subset.
    """
    if include_device:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        parser.add_argument(
            "--device",
            type=str,
            default=default_device,
            help="Compute device. Default: cuda if available, else cpu.",
        )
    if include_dtype:
        parser.add_argument(
            "--dtype",
            type=str,
            choices=list(dtype_choices),
            default=dtype_default,
            help=f"Model/storage dtype (default: {dtype_default}).",
        )
    return parser
