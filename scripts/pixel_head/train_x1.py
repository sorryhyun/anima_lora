#!/usr/bin/env python3
"""Train the ×1 release: a short PE-GAN finish on the base head.

``base_x1.py``'s recipe on a new cosine cycle from the base, with the paired PE-Spatial
GAN term on the ``:real`` samples (the original image is its real; the turbo point keeps
its regression alone). Recipe and its records: ``project/pixel_decoder/plan.md``.

Every flag is an :class:`X1TrainRequest` field; ``--config <file.toml>`` takes a flat TOML
of field names (field default → ``--config`` → CLI). Shipped config:
``configs/pixel_head/x1.toml``. ``init`` is the base's file, written out: the catalog's
head is a release, not a base.

    make daemon-run ARGS="--queue --stall-timeout 900 scripts/pixel_head/train_x1.py \
        --config configs/pixel_head/x1.toml"

Writes the files ``base_x1.py`` writes, the GAN settings stamped in the head.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from library.lines.requests import request_field as _f
from scripts.pixel_head._x1 import X1Request, run

BASE = "output/pixel_head/x1ll_1122_1000/head.safetensors"


@dataclass
class X1TrainRequest(X1Request):
    """Train the ×1 release head (a PE-GAN finish on the base)."""

    init: str = _f(BASE, "the base head's file; empty = a fresh head")
    rollouts: int = _f(2000, "DiT forwards")
    eval_every: int = _f(500, "rollouts")
    pegan_ratio: float = _f(
        2.0, "the GAN term's gradient norm over the head / the regression's; 0 = off"
    )
    pegan_crop: int = _f(0, "paired crop, a multiple of 16; 0 = the whole image")
    pegan_lr: float = _f(1e-4, "discriminator LR")
    pegan_lambda_max: float = _f(1.0, "cap on the term's weight")


def main() -> None:
    req = X1TrainRequest.from_argv()
    # read when the CUDA allocator initializes: before torch is imported in run()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(req)


if __name__ == "__main__":
    main()
