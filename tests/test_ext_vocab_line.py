"""Vocab-pack line block (``mapping["line"]``) — the gated line mode as rows.

Invariants:

* **Block = source rows + vec**, regenerated at load after ``iso``; a table
  already holding it is returned as is, a wrong-length table refuses.
* **Gate = an ext neighbour in the T5 id stream**: a lone ext id, stock ids
  and ids past the source rows never move; moved ids land at
  ``id + line_start``.
* **A pack without ``line`` encodes and digests as before**; adding one
  changes the digest.
* ``bake_vocab_pack.add_line`` writes the vector × row_scale × dose and
  places the block after the stored rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from library.anima import ext_vocab as ev
from library.anima.ext_vocab import T5_TABLE_SIZE as T

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "toolkits"))

import bake_vocab_pack as bvp  # noqa: E402


def _mapping(n=6, dim=4, start=None, vec=None):
    vec = vec if vec is not None else [1.0] * dim
    spec = ev.LineSpec(src_end=n, start=n if start is None else start, vec=tuple(vec))
    return {"rows": spec.end, "line": spec.to_json()}, spec


def test_line_spec_round_trip_and_materialize():
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    mapping, spec = _mapping()
    assert ev.LineSpec.from_mapping(mapping) == spec
    assert ev.LineSpec.from_mapping({"rows": 6}) is None
    full = ev.materialize(table, mapping)
    assert full.shape == (12, 4)
    assert torch.equal(full[:6], table)
    assert torch.equal(full[6:], table + 1.0)
    assert ev.materialize(full, mapping) is full
    with pytest.raises(ValueError):
        ev.materialize(torch.zeros(7, 4), mapping)


def test_line_gate():
    ids = [5, T + 1, 7, T + 2, T + 3, 9, T + 3, T + 1, T + 2, 1]
    assert ev.line_gate(ids) == [0, 0, 0, 1, 1, 0, 1, 1, 1, 0]
    assert ev.line_gate([]) == []
    assert ev.line_gate([T + 4]) == [False]


def test_apply_line_moves_gated_source_rows_only():
    enc = ev.HybridT5Encoder(
        t5_tok=None,
        qwen_tok=None,
        qwen_map={},
        char_map={},
        line_start=10,
        line_src_end=6,
    )
    # lone | run of 2 | run with a row past the source rows (an iso id, 8)
    ids = [3, T + 1, 4, T + 2, T + 5, 4, T + 3, T + 8, 1]
    assert enc.apply_line(ids) == [3, T + 1, 4, T + 12, T + 15, 4, T + 13, T + 8, 1]
    plain = ev.HybridT5Encoder(t5_tok=None, qwen_tok=None, qwen_map={}, char_map={})
    assert plain.apply_line(ids) is ids


def test_digest_covers_the_line_block():
    table = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    base = {"rows": 6, "qwen": {"100": 0}}
    with_line = {**base, **_mapping(vec=[0.5, 0.0, 0.0, 0.0])[0]}
    other = {**base, **_mapping(vec=[0.25, 0.0, 0.0, 0.0])[0]}
    d0 = ev.pack_digest(table, base)
    d1 = ev.pack_digest(table, with_line)
    assert d0 != d1 != ev.pack_digest(table, other)
    assert d1 == ev.pack_digest(ev.materialize(table, with_line), with_line)


def test_bake_add_line(tmp_path):
    table = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    mapping = {"rows": 6, "qwen": {}, "provenance": ["mapped"] * 6}
    src = tmp_path / "trained.pt"
    line = torch.tensor([1.0, -2.0, 0.0, 0.5])
    torch.save({"delta": {"line": line, "row_scale": 2.0}}, src)
    summary = bvp.add_line(table, mapping, src, dose=0.5)
    assert summary["rows"] == [6, 12] and mapping["rows"] == 12
    assert mapping["provenance"][6:] == ["line"] * 6
    full = ev.materialize(table, mapping)
    assert torch.allclose(full[6:], table + line * 2.0 * 0.5)
    with pytest.raises(ValueError):
        bvp.add_line(table, mapping, src, dose=0.5)  # already has one
