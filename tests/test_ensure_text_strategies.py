"""Tests for ``ensure_text_strategies`` — the lazy installer for the global
tokenize/encode strategy singletons (suggestions.md #8).

The singletons are process-global with set-once semantics, so the test saves
and restores them around its mutation.

An installed strategy is never swapped: the CLI path built it with the vocab
pack the caches were encoded through.
"""

from __future__ import annotations

import pytest

from library.anima import text_strategies
from library.inference.text import ensure_text_strategies


@pytest.fixture(autouse=True)
def restore_singletons():
    """Snapshot and restore the process-global strategy singletons."""
    tok = text_strategies.TokenizeStrategy._strategy
    enc = text_strategies.TextEncodingStrategy._strategy
    try:
        yield
    finally:
        text_strategies.TokenizeStrategy._strategy = tok
        text_strategies.TextEncodingStrategy._strategy = enc


def test_noop_when_both_already_set():
    tok_sentinel = object()
    enc_sentinel = object()
    text_strategies.TokenizeStrategy._strategy = tok_sentinel
    text_strategies.TextEncodingStrategy._strategy = enc_sentinel

    # No path needed, must not raise, must not replace the installed strategies.
    ensure_text_strategies(None)

    assert text_strategies.TokenizeStrategy._strategy is tok_sentinel
    assert text_strategies.TextEncodingStrategy._strategy is enc_sentinel
