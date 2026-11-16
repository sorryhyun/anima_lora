#!/usr/bin/env python3
"""Train a MOSS-Image LoRA on a precached folder — flags in
``library.moss.requests.TrainRequest``, work in ``library.moss.train``."""

import os

from library.runtime.proc import no_window_main
from library.moss.requests import TrainRequest


@no_window_main
def main() -> None:
    req = TrainRequest.from_argv()
    if req.expandable_segments:
        # Read when the CUDA allocator initializes: before library.moss.train.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from library.moss.train import run_train

    run_train(req)


if __name__ == "__main__":
    main()
