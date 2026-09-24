#!/usr/bin/env python3
"""Train a Qwen-Image-2.1 LoRA on a precached folder — flags in
``library.qwen21.requests.TrainRequest``, work in ``library.qwen21.train``."""

from library.qwen21.requests import TrainRequest


def main() -> None:
    req = TrainRequest.from_argv()
    from library.qwen21.train import run_train

    run_train(req)


if __name__ == "__main__":
    main()
