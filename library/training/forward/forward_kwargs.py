"""Build the conditioning (positional cond + kwargs) for the DiT forward.

Two concerns sit here:

* **Postfix injection** — for networks with ``append_postfix``, splice the
  learned postfix vectors onto the cached T5 embedding and pool the *real*
  text BEFORE the splice so modulation guidance only sees real text.
* **Mode normalization** — ``build_forward_conditioning``
  collapses the in-model vs cached-crossattn text-conditioning split into ONE
  uniform ``(cond, kw)`` bundle so the trainer keeps a single forward call
  site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from library.training.forward.text_conds import PreparedTextConds


@dataclass(slots=True)
class ForwardKwargs:
    """Outputs of ``build_forward_kwargs``.

    ``crossattn_emb`` is the (possibly postfix-extended) embedding; ``kw``
    is the dict of additional kwargs to pass to the DiT call.
    """

    crossattn_emb: torch.Tensor
    kw: dict[str, Any]
    has_postfix: bool


def build_forward_kwargs(
    *,
    network: Any,
    crossattn_emb: torch.Tensor,
    t5_attn_mask: Optional[torch.Tensor],
    timesteps: torch.Tensor,
) -> ForwardKwargs:
    has_postfix = hasattr(network, "append_postfix")
    kw: dict[str, Any] = {}

    if has_postfix:
        # Pool text BEFORE injection so modulation guidance sees only real text.
        kw["pooled_text_override"] = crossattn_emb.max(dim=1).values
        seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
        crossattn_emb = network.append_postfix(
            crossattn_emb, seqlens, timesteps=timesteps
        )

    return ForwardKwargs(crossattn_emb=crossattn_emb, kw=kw, has_postfix=has_postfix)


@dataclass(slots=True)
class ForwardConditioning:
    """ONE uniform conditioning bundle for the DiT forward.

    ``cond`` is the positional conditioning passed to ``anima(...)``;
    ``crossattn_emb`` is the (possibly postfix-extended) cached T5-space
    embedding — None on the in-model text path; ``kw`` is the dict of extra
    kwargs for the DiT call, also threaded to the aux-loss producers and the
    method-adapter dispatch.
    """

    cond: torch.Tensor
    crossattn_emb: Optional[torch.Tensor]
    kw: dict[str, Any]
    has_postfix: bool


def build_forward_conditioning(
    *,
    network: Any,
    tc: PreparedTextConds,
    timesteps: torch.Tensor,
) -> ForwardConditioning:
    """Normalize ``PreparedTextConds`` into one ``(cond, kw)`` pair for both
    text-conditioning modes.

    The postfix splice runs learned modules, so call this inside the same
    autocast / grad scope as the primary forward.
    """
    if tc.crossattn_emb is None:
        # In-model text path (no cached LLM-adapter outputs, e.g.
        # EasyControl): the DiT encodes text itself from the Qwen3 embeds.
        return ForwardConditioning(
            cond=tc.prompt_embeds,
            crossattn_emb=None,
            kw=dict(
                target_input_ids=tc.t5_input_ids,
                target_attention_mask=tc.t5_attn_mask,
                source_attention_mask=tc.attn_mask,
            ),
            has_postfix=False,
        )
    # crossattn_emb is already in target (T5-compatible) space. Postfix
    # splice kwargs.
    fk = build_forward_kwargs(
        network=network,
        crossattn_emb=tc.crossattn_emb,
        t5_attn_mask=tc.t5_attn_mask,
        timesteps=timesteps,
    )
    return ForwardConditioning(
        cond=fk.crossattn_emb,
        crossattn_emb=fk.crossattn_emb,
        kw=fk.kw,
        has_postfix=fk.has_postfix,
    )
