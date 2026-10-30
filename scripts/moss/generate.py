#!/usr/bin/env python3
"""Render prompts with MOSS-Image — flags in ``library.moss.requests.GenerateRequest``,
work in ``library.moss.generate``."""

import os

from library.moss.requests import GenerateRequest


def main() -> None:
    req = GenerateRequest.from_argv()
    if req.blocks_to_swap:
        # Swapped blocks fragment the pool. Read when the CUDA allocator
        # initializes: before library.moss.generate.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from library.moss.generate import run_generate

    run_generate(req)


if __name__ == "__main__":
    main()
