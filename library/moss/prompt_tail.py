"""An assistant-turn tail after the chat template's generation prompt.

The shipped prompt ends at ``<|im_start|>assistant\\n<think>\\n``. A tail closes the
empty think block and starts an answer (``sure``: ``\\n</think>\\n\\nSure,``); on Base
without a LoRA it sharpens hands and backgrounds (``docs/moss/findings/assistant_tail.md``).
The TE is causal, so the caption rows keep their states; the template's last ``\\n``
merges with the tail's first into one ``\\n\\n`` token, and ``sure`` adds 4 rows.

``sure`` is the cache default; the i2i configs pin ``off``. The mode is stamped in TE
caches (``prompt_tail``, when not ``off``) and in the LoRA; ``GenerateRequest.prompt_tail``
``auto`` follows the LoRA (unstamped: ``off``), and without a LoRA renders ``sure``. The
tail goes on cond and uncond alike.
"""

from __future__ import annotations

TAILS = {
    "off": "",
    "sure": "\n</think>\n\nSure,",
}


class TailTokenizer:
    """``tok`` whose ``apply_chat_template(tokenize=False)`` ends in ``tail``; every
    other call is ``tok``'s."""

    def __init__(self, tok, tail: str):
        self._tok = tok
        self._tail = tail

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

    def apply_chat_template(self, *args, tokenize=True, **kwargs):
        if tokenize:
            raise ValueError("the prompt tail is applied to the template text only")
        return (
            self._tok.apply_chat_template(*args, tokenize=False, **kwargs) + self._tail
        )


def with_prompt_tail(tok, mode: str):
    """``tok`` as ``mode`` templates (``off``: ``tok`` itself). Wrap before
    ``with_token_split``, which delegates the template to it."""
    if mode not in TAILS:
        raise ValueError(f"prompt_tail {mode!r}, not one of {tuple(TAILS)}")
    return tok if mode == "off" else TailTokenizer(tok, TAILS[mode])


def resolve_prompt_tail(req) -> str:
    """``GenerateRequest.prompt_tail`` with ``auto`` read off the LoRA; without one,
    ``sure`` (``off`` under ``ref_vision``)."""
    from safetensors import safe_open

    from library.env import resolve_under_home

    mode = req.prompt_tail
    if mode == "auto":
        mode = "off" if req.ref_vision else "sure"
        if req.lora:
            with safe_open(str(resolve_under_home(req.lora)), "pt") as f:
                mode = (f.metadata() or {}).get("prompt_tail", "off")
    if mode != "off" and req.ref_vision:
        raise SystemExit("ref_vision encodes through the processor: prompt_tail off")
    return mode
