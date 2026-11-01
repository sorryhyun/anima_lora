"""Encode fold (``mapping["fold"]``, plan_retrain § 2c).

Invariants (against the real tokenizers + pack, skipped when absent):

* **Folded = the target spelled**: a text with ``！ ~ 『』`` encodes exactly
  as the same text written ``! ～ 「」`` under the unfolded pack; ``routes()``
  answers for the folded text.
* The key reaches the digest only when set; a many-char entry is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from library.anima import ext_vocab as ev
from library.env import resolve_under_home

EXT_PREFIX = resolve_under_home("models/vocab_packs/anima_cjk_vocab_pack")
FOLD = {"！": "!", "？": "?", "~": "～", "『": "「", "【": "「", "』": "」", "】": "」"}


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
    mapping = {k: v for k, v in mapping.items() if k != "fold"}
    t5, qw = tok.t5_tokenizer, tok.qwen3_tokenizer
    return (
        ev.HybridT5Encoder.from_mapping(t5, qw, mapping, glyph_route=True),
        ev.HybridT5Encoder.from_mapping(
            t5, qw, {**mapping, "fold": FOLD}, glyph_route=True
        ),
    )


def test_folded_text_encodes_as_its_target():
    plain, fold = _encs()
    assert plain.fold is None
    for typed, target in (
        (
            'Japanese text reads as "『えっ！？』~".',
            'Japanese text reads as "「えっ!?」～".',
        ),
        ('Japanese SFX reads as "【ドン】".', 'Japanese SFX reads as "「ドン」".'),
        ("1girl, solo, smiling!", "1girl, solo, smiling!"),
    ):
        assert fold.encode(typed) == plain.encode(target)
        assert plain.encode(typed) == plain.encode(typed)  # no fold: untouched


def test_routes_reads_the_folded_text():
    plain, fold = _encs()
    assert plain.routes("wow！") and not fold.routes("wow！")
    assert fold.routes("~")  # → ～, a routed row


def test_digest_and_refusal():
    table = torch.zeros(4, 8)
    base = {"qwen": {}, "char": {}}
    assert ev.pack_digest(table, base) == ev.pack_digest(table, {**base, "fold": {}})
    assert ev.pack_digest(table, base) != ev.pack_digest(
        table, {**base, "fold": {"！": "!"}}
    )
    plain, _ = _encs()
    with pytest.raises(AssertionError):
        ev.HybridT5Encoder.from_mapping(
            plain.t5_tok, plain.qwen_tok, {**base, "fold": {"！？": "!"}}
        )
