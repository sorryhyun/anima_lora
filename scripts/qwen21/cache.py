#!/usr/bin/env python3
"""Precache Qwen-Image-2.1 text embeddings + VAE latents — flags in
``library.qwen21.requests.CacheRequest``, work in ``library.qwen21.cache``."""

from library.qwen21.requests import CacheRequest


def main() -> None:
    req = CacheRequest.from_argv()
    from library.qwen21.cache import run_cache

    run_cache(req)


if __name__ == "__main__":
    main()
