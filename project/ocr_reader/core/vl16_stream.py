"""fp32 residual stream for PaddleOCR-VL-1.6 training (``finetune_vl16_lora.py --stream
fp32``): the ``stream`` arm of ``probes/precision_stream.py`` as the trainer runs it.

``install(base)`` takes the bf16 model before peft wraps it. Every parameter goes to fp32
except the ``nn.Linear`` weights inside a vision block or an LM decoder layer: those stay
bf16 and run as bf16 GEMMs, input rounded and output cast back to the stream's dtype. The
attention cores run on bf16 q / k / v: transformers' attention functions through the
wrapper registered here, ``vl16_compile``'s tower forwards through their own cast. The
residual stream, norms, RoPE, activations, the tower's embeddings and the projector are
fp32. ``embed_tokens`` and ``lm_head`` keep bf16 weights (in fp32 each weight, and its
gradient while the vocab rows train, is 404 MiB): the embedding's output is cast up, the
head runs as one more bf16 GEMM. The checkpoint the trainer saves is bf16.

``lean=True`` casts up only what is written to the stream: the linears whose output is
added to it (``out_proj`` / ``o_proj``, ``fc2`` / ``down_proj``) return fp32, so a LoRA
delta lands on the stream unrounded; q / k / v, ``fc1``, ``gate_proj`` and ``up_proj``
return the GEMM's bf16 output, and RoPE, the activation and the attention core's output
stay bf16 up to the next GEMM, which rounds its input in either form.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F

BLOCK = re.compile(r"\.(layers|blocks)\.\d+\.")
BF = torch.bfloat16
WRITERS = ("out_proj", "o_proj", "fc2", "down_proj")
"""The block linears whose output is added to the residual stream."""
_wrapped = False


def _bf16_linear(m: torch.nn.Linear, out: torch.dtype | None = None):
    """``out``: the output's dtype; None casts back to the input's."""

    def forward(x):
        return F.linear(x.to(BF), m.weight, m.bias).to(out or x.dtype)

    return forward


def _fp32_embedding(m: torch.nn.Embedding):
    def forward(ids):
        return F.embedding(ids, m.weight, m.padding_idx).float()

    return forward


def _bf16_attention(orig):
    def attn(module, query, key, value, attention_mask, *args, **kw):
        dt = query.dtype
        if dt == BF:
            return orig(module, query, key, value, attention_mask, *args, **kw)
        if attention_mask is not None and attention_mask.is_floating_point():
            attention_mask = attention_mask.clamp(min=torch.finfo(BF).min).to(BF)
        o, w = orig(
            module, query.to(BF), key.to(BF), value.to(BF), attention_mask, *args, **kw
        )
        return o.to(dt), w

    return attn


def install(base, lean: bool = False) -> int:
    """Put ``base`` on the fp32 stream in place; returns the number of bf16 linears."""
    global _wrapped
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    base.float()
    n = 0
    for name, m in base.named_modules():
        if isinstance(m, torch.nn.Linear) and BLOCK.search(name + "."):
            m.to(BF)
            writer = name.rsplit(".", 1)[-1] in WRITERS
            # a writer returns fp32 in both forms: the compiled tower hands out_proj
            # the attention core's bf16 output
            out = torch.float32 if writer else (BF if lean else None)
            m.forward = _bf16_linear(m, out)
            n += 1
    emb, head = base.get_input_embeddings(), base.get_output_embeddings()
    emb.to(BF).forward = _fp32_embedding(emb)
    head.to(BF).forward = _bf16_linear(head, torch.float32)
    if not _wrapped:
        for key in ("sdpa", "flash_attention_2"):
            ALL_ATTENTION_FUNCTIONS.register(
                key, _bf16_attention(ALL_ATTENTION_FUNCTIONS[key])
            )
        _wrapped = True
    return n
