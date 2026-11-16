#!/usr/bin/env python3
"""Precache MOSS-Image text embeddings + VAE latents — flags in
``library.moss.requests.CacheRequest``, work in ``library.moss.cache``."""

from library.runtime.proc import no_window_main
from library.moss.requests import CacheRequest


@no_window_main
def main() -> None:
    req = CacheRequest.from_argv()
    from library.moss.cache import run_cache

    run_cache(req)


if __name__ == "__main__":
    main()
