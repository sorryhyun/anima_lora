#!/usr/bin/env python3
"""Train the ×1 base: the regression head every later stage starts from.

Base decode points against the original image with the DINO term, the turbo point against the
VAE's decode of x̂0, manga pages in the mix, no GAN term. The GAN finish on the result
(the release) is ``train_x1.py``; ×2 is ``train_x2.py`` with ``init`` at this head's
file. Recipe and its records: ``project/pixel_decoder/plan.md``.

Every flag is an :class:`BaseX1Request` field; ``--config <file.toml>`` takes a flat TOML
of field names (field default → ``--config`` → CLI). Shipped config:
``configs/pixel_head/base_x1.toml``.

    make daemon-run ARGS="--queue --stall-timeout 900 scripts/pixel_head/base_x1.py \
        --config configs/pixel_head/base_x1.toml"

Writes ``<output_dir>/<output_name>/``: ``head.safetensors``, ``result.json`` (the
request, the eval history, per-image rows) and ``original | VAE decode | head`` sheets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from library.lines.requests import request_field as _f
from scripts.pixel_head._x1 import X1Request, run


@dataclass
class BaseX1Request(X1Request):
    """Train the ×1 base head (regression only)."""

    output_name: str = _f("base_x1", "the run's name")


def main() -> None:
    req = BaseX1Request.from_argv()
    # read when the CUDA allocator initializes: before torch is imported in run()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(req)


if __name__ == "__main__":
    main()
