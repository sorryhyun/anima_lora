#!/usr/bin/env python3
"""Build the MOSS-format captions of one image set — flags in
``library.moss.requests.CaptionsRequest``, work in ``library.moss.captions``."""

from library.moss.requests import CaptionsRequest


def main() -> None:
    req = CaptionsRequest.from_argv()
    from library.moss.captions import run_captions

    run_captions(req)


if __name__ == "__main__":
    main()
