"""Block compile for the pixel_readout scripts, under a per-signature compile cache.

The persistent inductor / AOT caches key on the FX graph, not on the ``mark_dynamic``
range, so a graph compiled by one script (e.g. head training, bands (2928, 3096) /
(4032, 4200)) replays its guards into another with a different band (the P3 prompts,
(3952, 4096)) and raises ``ConstraintViolationError`` on the first trace
(``t2a_read_1119_1337``, ``t2base_1119_1349``). ``isolate_compile_cache`` gives each
signature its own cache dir; see ``library/runtime/dynamo.py``.
"""

from __future__ import annotations


def compile_blocks_isolated(
    anima, seq_bands: list | None = None, tag: str = ""
) -> None:
    """``anima.compile_blocks(dynamic_seq=True, seq_bands=…)`` in an isolated cache.

    Call after ``stream_fp32_`` and any adapter, before the first forward. ``tag`` names
    a base the signature cannot see (``int8``)."""
    from library.runtime.dynamo import compile_signature, isolate_compile_cache

    isolate_compile_cache(
        compile_signature(
            n_token_families=None,
            seq_range=None,
            dynamic_seq=True,
            mode="",
            seq_bands=seq_bands,
        )
        + f";stream={getattr(anima, 'stream', 'bf16')}"
        + (f";{tag}" if tag else "")
    )
    anima.compile_blocks(dynamic_seq=True, seq_bands=seq_bands)
