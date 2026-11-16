#!/usr/bin/env python3
"""Build a MOSS colorize cache (vision-tower page + VAE reference) — flags in
``library.moss.requests.ColorizeCacheRequest``, work in ``library.moss.colorize``."""

from library.runtime.proc import no_window_main
from library.moss.requests import ColorizeCacheRequest


@no_window_main
def main() -> None:
    req = ColorizeCacheRequest.from_argv()
    from library.moss.colorize import run_colorize_cache

    run_colorize_cache(req)


if __name__ == "__main__":
    main()
