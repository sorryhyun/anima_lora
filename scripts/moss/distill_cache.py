#!/usr/bin/env python3
"""Build the self-distillation teacher files of a MOSS training cache — flags in
``library.moss.requests.DistillCacheRequest``, work in ``library.moss.distill``."""

from library.moss.requests import DistillCacheRequest


def main() -> None:
    req = DistillCacheRequest.from_argv()
    from library.moss.distill import cache_teachers

    cache_teachers(req)


if __name__ == "__main__":
    main()
