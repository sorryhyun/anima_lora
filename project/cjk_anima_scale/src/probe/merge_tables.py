"""``row_text_map`` — the one function ``eval.cf_sense`` imports from the
source line's ``probe/merge_tables.py`` (a table-merge script, not vendored),
and ``row_texts`` (from the source line's ``train/encoder.py``, the only part
of it still read). Both bodies are verbatim; ``cf_sense`` builds its ja pairs
from them when a table carries ext ids only (every table this line writes)."""

from __future__ import annotations


def row_texts(tok, pack, rows):
    """ext row → the piece text it stands for (Qwen piece, char row or symbol
    row); rows the pack cannot name are left out (zero delta)."""
    inv_q = {int(v): int(k) for k, v in pack.mapping["qwen"].items()}
    inv_c = {int(v): k for k, v in pack.mapping.get("char", {}).items()}
    inv_s = {int(v): k for k, v in pack.mapping.get("sym_char", {}).items()}
    qtok = tok.qwen3_tokenizer
    out = {}
    for r in rows:
        r = int(r)
        if r in inv_q:
            t = qtok.decode([inv_q[r]]).strip()
        elif r in inv_c:
            t = inv_c[r]
        elif r in inv_s:
            t = inv_s[r]
        else:
            continue
        if t:
            out[r] = t
    return out


def row_text_map(ids: list[int]) -> dict:
    """ext id → piece text via the shipped pack. Reporting only — a failure
    here must not stop the merge."""
    try:
        from library.anima.vocab_pack import load_vocab_pack
        from library.env import default_checkpoints
        from library.inference.text import ensure_text_strategies

        ck = default_checkpoints()
        tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
        return row_texts(tok, load_vocab_pack(ck.vocab_pack), ids)
    except Exception as e:  # noqa: BLE001 — provenance is best effort
        print(f"(row text decode unavailable: {e})", flush=True)
        return {}
