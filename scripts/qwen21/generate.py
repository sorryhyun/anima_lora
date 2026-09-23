#!/usr/bin/env python3
"""Render prompts with/without a Qwen-Image-2.1 LoRA — flags in
``library.qwen21.requests.GenerateRequest``, work in ``library.qwen21.generate``."""

from library.qwen21.requests import GenerateRequest


def main() -> None:
    req = GenerateRequest.from_argv()
    from library.qwen21.generate import run_generate

    run_generate(req)


if __name__ == "__main__":
    main()
