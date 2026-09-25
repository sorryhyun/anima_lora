"""``row_text_map`` — the one function ``eval.cf_sense`` imports from the
source line's ``probe/merge_tables.py`` (a table-merge script, not vendored).
The function body is verbatim; ``cf_sense`` builds its ja pairs from it when a
table carries ext ids only (every table this line writes)."""

from __future__ import annotations


def row_text_map(ids: list[int]) -> dict:
    """ext id → piece text via the shipped pack. Reporting only — a failure
    here must not stop the merge."""
    try:
        from train.encoder import row_texts

        from library.anima.vocab_pack import load_vocab_pack
        from library.env import default_checkpoints
        from library.inference.text import ensure_text_strategies

        ck = default_checkpoints()
        tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
        return row_texts(tok, load_vocab_pack(ck.vocab_pack), ids)
    except Exception as e:  # noqa: BLE001 — provenance is best effort
        print(f"(row text decode unavailable: {e})", flush=True)
        return {}
