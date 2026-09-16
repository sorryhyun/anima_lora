"""Slot-level surgery on the DirectEdit cross-attention conditioning tensor.

Given the two T5 token sequences from a (ψ_src, ψ_tar) edit pair, identify the
single contiguous slot range that differs (longest common prefix + longest
common suffix). Then build the edit conditioning by keeping ψ_src's slot
embeddings everywhere outside that range and transplanting ψ_tar's slot
embeddings only inside it.

Why this is OK to do on a cross-attention adapter:
  * T5 IDs map 1-to-1 to crossattn_emb slot indices — the LLM Adapter has 512
    target query slots fed by T5 input IDs.
  * The training-time invariant zeroes out crossattn_emb slots where
    ``t5_attn_mask == 0`` (see ``AnimaTextEncodingStrategy``), so the trailing
    padding region is well-defined as "zero" regardless of ψ.
  * Cross-attention drift outside the diff span: an open empirical question
    (the LLM Adapter cross-attends Qwen3's hidden states, which DO differ
    between ψ_src and ψ_tar everywhere).

Probe coverage: ``_archive/text_enc_probes/edit_slot_alignment.py`` reports
10/10 clean contiguous spans across replace/remove/add.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class T5DiffSpan:
    """Result of locating the contiguous differing token range.

    ``src_ids[start:src_end]`` ↔ ``tar_ids[start:tar_end]`` is the span that
    differs (either side may be empty for pure add/remove). Everything outside
    that range is bit-identical between the two trimmed T5 sequences.

    ``src_len`` / ``tar_len`` are the post-padding-trim lengths — the suffix
    region we copy from src runs ``[src_end:src_len]``.
    """

    start: int
    src_end: int
    tar_end: int
    src_len: int
    tar_len: int

    @property
    def src_span_len(self) -> int:
        return self.src_end - self.start

    @property
    def tar_span_len(self) -> int:
        return self.tar_end - self.start

    @property
    def suffix_len(self) -> int:
        return self.src_len - self.src_end


def _trim_pad_tail(ids: Sequence[int], pad_id: int) -> int:
    """Return the index just past the last non-pad token."""
    n = len(ids)
    while n > 0 and ids[n - 1] == pad_id:
        n -= 1
    return n


def find_t5_diff_span(
    src_ids: Sequence[int],
    tar_ids: Sequence[int],
    pad_id: int,
) -> T5DiffSpan:
    """Longest common prefix + longest common suffix on padding-trimmed IDs.

    Returns the diff endpoints and the trimmed lengths so callers can locate
    the suffix region without retrimming. Mirrors
    ``_archive/text_enc_probes/edit_slot_alignment.py::find_diff_span`` plus the trim
    step the probe does separately, so the regression set carries over.
    """
    src_len = _trim_pad_tail(src_ids, pad_id)
    tar_len = _trim_pad_tail(tar_ids, pad_id)

    start = 0
    upper = min(src_len, tar_len)
    while start < upper and src_ids[start] == tar_ids[start]:
        start += 1

    suf = 0
    upper = min(src_len - start, tar_len - start)
    while suf < upper and src_ids[src_len - 1 - suf] == tar_ids[tar_len - 1 - suf]:
        suf += 1

    return T5DiffSpan(
        start=start,
        src_end=src_len - suf,
        tar_end=tar_len - suf,
        src_len=src_len,
        tar_len=tar_len,
    )


def splice_crossattn_emb(
    *,
    crossattn_emb_src: torch.Tensor,
    crossattn_emb_tar: torch.Tensor,
    t5_ids_src: torch.Tensor,
    t5_ids_tar: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, T5DiffSpan]:
    """Build edit conditioning by keeping src slots outside the diff span and
    overwriting the span with tar slots. Re-pads to ``L`` (512) with zeros.

    Shapes:
      * ``crossattn_emb_src`` / ``crossattn_emb_tar``: ``(1, L, D)``.
      * ``t5_ids_src``       / ``t5_ids_tar``:        ``(1, L)``.

    Single-batch only — DirectEdit's invert/edit loops run B=1.

    Returns ``(spliced, span)`` so callers can log which token range moved.
    """
    if crossattn_emb_src.shape != crossattn_emb_tar.shape:
        raise ValueError(
            f"crossattn_emb shape mismatch: src={tuple(crossattn_emb_src.shape)} "
            f"vs tar={tuple(crossattn_emb_tar.shape)}"
        )
    B, L, _ = crossattn_emb_src.shape
    if B != 1:
        raise ValueError(f"splice expects B=1, got B={B}")
    if t5_ids_src.shape != (B, L) or t5_ids_tar.shape != (B, L):
        raise ValueError(
            f"t5_ids shape mismatch: src={tuple(t5_ids_src.shape)} "
            f"tar={tuple(t5_ids_tar.shape)} expected ({B}, {L})"
        )

    src_ids = t5_ids_src[0].tolist()
    tar_ids = t5_ids_tar[0].tolist()
    span = find_t5_diff_span(src_ids, tar_ids, pad_id)

    # Defensive: T5 IDs in the suffix region must match between src and tar.
    # Re-tokenisation in the splice contract guarantees this (it's how we
    # picked the suffix in find_t5_diff_span). If it ever fires, something
    # upstream tokenised the two captions inconsistently.
    suf_a = src_ids[span.src_end : span.src_len]
    suf_b = tar_ids[span.tar_end : span.tar_len]
    if suf_a != suf_b:
        raise RuntimeError(
            "T5 suffix mismatch after find_t5_diff_span — splice invariant "
            f"broken (src_suffix={suf_a!r} tar_suffix={suf_b!r})"
        )

    out = torch.zeros_like(crossattn_emb_src)
    cursor = 0
    if span.start > 0:
        out[0, cursor : cursor + span.start] = crossattn_emb_src[0, 0 : span.start]
        cursor += span.start
    tar_span_len = span.tar_span_len
    if tar_span_len > 0:
        out[0, cursor : cursor + tar_span_len] = crossattn_emb_tar[
            0, span.start : span.tar_end
        ]
        cursor += tar_span_len
    suffix_len = span.suffix_len
    if suffix_len > 0:
        out[0, cursor : cursor + suffix_len] = crossattn_emb_src[
            0, span.src_end : span.src_len
        ]
        cursor += suffix_len
    # Slots [cursor:L] stay zero — matches the AnimaTextEncodingStrategy
    # "zero padding tail" invariant the DiT trained against.
    return out, span
