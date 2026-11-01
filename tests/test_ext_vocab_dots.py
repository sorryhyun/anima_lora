"""Dot runs (``mapping["dots"]``) and the ``…`` row.

Invariants (against the real tokenizers + pack, skipped when absent):

* **One row per ellipsis**: ``・・`` / ``・・・`` / ``‥`` / ``...`` / ``…`` beside
  a routed char encode as ``…``; 4+ dots as ``……``; a lone ``・`` as ``.``.
* **EN untouched**: a ``.`` / ``…`` run touching no routed char encodes as
  the stock T5 spelling, and a prompt with no other routed char never routes.
* The key reaches the digest only when set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from library.anima import ext_vocab as ev
from library.env import resolve_under_home

EXT_PREFIX = resolve_under_home("models/vocab_packs/anima_cjk_vocab_pack")
DOTS = {
    "run": "・･.．‥…",
    "count": {"‥": 2, "…": 3},
    "lone": {"・": ".", "･": "."},
    "anchored": ".…",
    "plain": {"…": "..."},
}
ELLIPSIS_QWEN = "1940"  # Qwen's `…` token


def _encs():
    if not EXT_PREFIX.with_suffix(".json").exists():
        pytest.skip("vocab pack not downloaded")
    from anima_lora import default_checkpoints
    from library.anima import strategy as strategy_anima

    ckpt = default_checkpoints()
    if not Path(ckpt.text_encoder).exists():
        pytest.skip("Qwen3 text encoder not downloaded")
    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    _, mapping = ev.load_ext_assets(EXT_PREFIX)
    row = int(mapping["rows"])  # the appended `…` row
    route = dict(mapping["route"])
    route["chars"] = route["chars"] + "…"
    dotted = {
        **mapping,
        "sym": {**mapping["sym"], ELLIPSIS_QWEN: row},
        "sym_char": {**mapping["sym_char"], "…": row},
        "route": route,
        "dots": DOTS,
    }
    t5, qw = tok.t5_tokenizer, tok.qwen3_tokenizer
    plain = ev.HybridT5Encoder.from_mapping(t5, qw, mapping, glyph_route=True)
    dots = ev.HybridT5Encoder.from_mapping(t5, qw, dotted, glyph_route=True)
    return plain, dots, ev.T5_TABLE_SIZE + row


@pytest.mark.parametrize(
    "typed, target",
    [
        ("はぁ・・・", "はぁ…"),
        ("はぁ‥", "はぁ…"),
        ("...はぁ", "…はぁ"),
        ("はぁ・・・・", "はぁ……"),
        ("ジョン・スミス", "ジョン.スミス"),
        ('Japanese text reads as "「...」".', 'Japanese text reads as "「…」".'),
    ],
)
def test_runs_encode_as_their_target(typed, target):
    _, dots, _ = _encs()
    assert dots.normalized(typed)[0] == target
    assert dots.encode(typed) == dots.encode(target)


def test_ellipsis_has_its_row():
    _, dots, row = _encs()
    ids, mask, _ = dots.encode_aligned("はぁ…", 16)
    assert ids[: sum(mask)].count(row) == 1
    ids, mask, _ = dots.encode_aligned("はぁ・・・・", 16)
    assert ids[: sum(mask)].count(row) == 2


def test_en_runs_stay_t5():
    plain, dots, row = _encs()
    for en in ("Wait... what…", "smiling, ellipsis…, 1girl", "ok."):
        assert not dots.routes(en)
    # beside routed text elsewhere, an untouched EN run is T5's spelling
    mixed = 'Wait… then she says "はい".'
    assert dots.encode(mixed) == plain.encode(mixed)
    assert row not in dots.encode(mixed)[0]


def test_digest():
    table = torch.zeros(4, 8)
    base = {"qwen": {}, "char": {}}
    assert ev.pack_digest(table, base) == ev.pack_digest(table, {**base, "dots": {}})
    assert ev.pack_digest(table, base) != ev.pack_digest(table, {**base, "dots": DOTS})
