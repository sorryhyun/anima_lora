"""Hangul token splits for the Qwen3.5 tokenizer.

The default tokenizer writes about 41 % of Korean syllables inside multi-syllable tokens
(`` 이건``, ``인가요``) and splits some after a space into byte fragments. The TE is
frozen, so each of those forms is a separate vector to the DiT.

``syllable`` gives every Hangul syllable one form: the syllable tokenized alone, with
a space before it as its own token. 11,172 syllables exist and the vocab holds 976 as a
single token; the rest stay byte fragments, the same ones in every context. Text without
Hangul tokenizes as it does by default.

The mode is stamped in TE caches (``token_split``, only when it changed the tokens) and
in the LoRA; ``GenerateRequest.token_split`` ``auto`` follows the LoRA.
``docs/moss/findings/ko_token_split.md``.
"""

from __future__ import annotations

import re

_SYLLABLE = "가-힣"
_HANGUL = re.compile(f"[{_SYLLABLE}]")
_PIECES = re.compile(f"([{_SYLLABLE}]| (?=[{_SYLLABLE}]))")


def effective_split(mode: str, text: str) -> str:
    """The mode as ``text`` feels it: ``off`` when it has no Hangul to split."""
    return mode if _HANGUL.search(text) else "off"


def syllable_ids(tok, text: str) -> list[int]:
    """``text``'s token ids under the ``syllable`` split."""
    pieces = [p for p in _PIECES.split(text) if p]
    encoded = tok(pieces, add_special_tokens=False).input_ids
    return [i for ids in encoded for i in ids]


class SyllableTokenizer:
    """``tok`` under the ``syllable`` split, for the call ``loader.prompt_ids`` makes.
    Everything else (``apply_chat_template``, ``convert_tokens_to_ids``) is ``tok``'s;
    a call with other arguments, such as offset mapping, raises."""

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(
        self, text: str, *, return_tensors: str, truncation=False, max_length=None
    ):
        import torch
        from transformers import BatchEncoding

        if return_tensors != "pt":
            raise ValueError(f"return_tensors {return_tensors!r}, only 'pt' is split")
        ids = syllable_ids(self._tok, text)
        if truncation:
            ids = ids[:max_length]
        input_ids = torch.tensor([ids])
        return BatchEncoding(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        )


def resolve_token_split(req) -> str:
    """``GenerateRequest.token_split`` with ``auto`` read off the LoRA."""
    from safetensors import safe_open

    from library.env import resolve_under_home

    mode = req.token_split
    if mode == "auto":
        mode = "off"
        if req.lora:
            with safe_open(str(resolve_under_home(req.lora)), "pt") as f:
                mode = (f.metadata() or {}).get("token_split", "off")
    if mode != "off" and req.ref_vision:
        raise SystemExit("ref_vision encodes through the processor: token_split off")
    return mode


def with_token_split(tok, mode: str):
    """``tok`` as ``mode`` tokenizes (``off``: ``tok`` itself)."""
    if mode == "off":
        return tok
    if mode == "syllable":
        return SyllableTokenizer(tok)
    raise ValueError(f"token_split {mode!r}, not off or syllable")
