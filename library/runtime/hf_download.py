"""Bounded, fail-fast HuggingFace downloads.

Auto-fetching a missing model (PE vision tower, Anima tagger, resume state)
must never wedge the job queue: a stalled connection inside ``hf_hub_download``
blocks the whole training/preprocess subprocess, which the daemon then can't
distinguish from real work — the "I clicked Train and it spins forever" hang.

Two guards:
  * pin an explicit socket timeout so a dead/stalled connection raises in
    seconds instead of hanging (a slow *trickle* is still caught by the daemon
    stall watchdog, the backstop for the residual case a socket timeout can't
    see);
  * translate network failures into a clear, actionable error that names the
    missing asset and the recovery command, instead of a raw urllib traceback.
"""

from __future__ import annotations

import os

# Per-request socket timeout (connect + read) for hub traffic, in seconds.
# Tunable via ANIMA_HF_TIMEOUT; bounds a fully stalled connection.
_DEFAULT_TIMEOUT = os.environ.get("ANIMA_HF_TIMEOUT", "30")


def ensure_hf_timeouts() -> None:
    """Pin huggingface_hub's socket timeouts unless the user set them.

    ``HF_HUB_DOWNLOAD_TIMEOUT`` bounds the streaming file-download read;
    ``HF_HUB_ETAG_TIMEOUT`` bounds the metadata HEAD/list call. Recent hub
    releases default both to 10s, but we set them explicitly so behavior is
    pinned regardless of the installed version and tunable in one place.
    """
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", _DEFAULT_TIMEOUT)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", _DEFAULT_TIMEOUT)


def _is_network_error(exc: BaseException) -> bool:
    """True only for *transport* failures (the ones that hang): connection and
    read timeouts, refused/reset connections.

    Deliberately excludes HTTP-status errors like ``EntryNotFoundError`` /
    ``RepositoryNotFoundError`` (HfHubHTTPError 404s) — those are *fast*
    responses, never a hang, and callers catch them specifically (e.g. the
    tagger's best-effort optional files), so they must propagate unchanged.
    """
    import socket

    net: list[type] = [socket.timeout, TimeoutError, ConnectionError]
    try:
        import requests  # huggingface_hub's transport

        net.append(requests.exceptions.ConnectionError)
        net.append(requests.exceptions.Timeout)
    except ImportError:
        pass
    return isinstance(exc, tuple(net))


def hf_download(*, what: str, hint: str = "make download-models", **kwargs):
    """``hf_hub_download`` with pinned timeouts and a fail-fast network error.

    ``what`` names the asset for the error message; ``hint`` is the suggested
    recovery command. Remaining kwargs pass straight through to
    ``hf_hub_download`` (``repo_id`` / ``filename`` / ``local_dir`` / ``token``
    / ``revision`` …). Non-network failures propagate unchanged.
    """
    from huggingface_hub import hf_hub_download

    ensure_hf_timeouts()
    try:
        return hf_hub_download(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_network_error(exc):
            raise FileNotFoundError(
                f"{what}: download from HuggingFace stalled or failed "
                f"({type(exc).__name__}: {exc}). Check connectivity (or set "
                f"HF_HUB_OFFLINE=1 if it is already cached locally), then "
                f"re-run `{hint}`."
            ) from exc
        raise
