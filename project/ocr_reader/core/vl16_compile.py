"""Block compile for PaddleOCR-VL-1.6 training (``finetune_vl16_lora.py --compile``).

Each tower encoder layer and LM decoder layer is ``torch.compile(dynamic=False)`` on its
``forward`` (an instance attribute, so the checkpoint wrapper stays outside the graph),
behind an eager prologue that marks the token axis on every call, as in
``library/moss/loader.py::compile_blocks`` and Anima's ``_make_dynamic_seq_forward``. The
marks go inside the callable so a checkpoint recompute re-marks.

* Tower: the packed token axis T (dim 0 of the hidden state and the rotary cos / sin) is
  marked strictly to ``tower_range``; ``cu_seqlens`` (images + 1) to ``[2, bs + 1]``. The
  stock SDPA path splits per image with ``lengths.tolist()``, which breaks the graph, so
  the attention core runs as the custom op ``ocr_vl16::tower_attn`` (per-image SDPA,
  opaque to dynamo; its backward recomputes each image's attention). The op reads
  ``cu_seqlens`` with ``tolist()``, so the trainer passes it on the CPU: on the card each
  of the 54 reads (27 layers, forward and backward) waits for the queued kernels.
  ``max_seqlen`` (a Python int, one graph per value under ``dynamic=False``) is dropped:
  the op does not use it; so are the precomputed ``interp_*`` tensors, which the encoder
  forwards to every layer and only the embeddings read.
* LM: the sequence axis S is soft-marked (``maybe_mark_dynamic``, no range). The layer
  runs SDPA with a padding mask, and Inductor's ``sdpa_constraint`` guards the
  backward's logsumexp stride on ``S % 8``, which a strict range contradicts (the same
  call as Anima under ``attn_mode="torch"``); the trainer pads S to a multiple of 8
  under ``--compile`` so that guard always holds. The crop count B (1 to ``--bs``
  under ``--token_budget``, a large crop trains alone) is marked strictly to
  ``[2, bs]`` on the hidden state, the 4D mask and the rotary cos / sin; B = 1 stays
  static (dynamo specialises sizes 0 / 1). The prologue also pins the intra-op thread
  count: the checkpoint recompute runs on the autograd device thread, and its
  differing ``num_threads`` doubles every LM graph through the GLOBAL_STATE guard.

A token count outside the range runs unmarked and specialises on its exact count, with
one warning per count. A call with grad off (generation, ``predict``) goes straight to
the layer's eager forward: no marks, no warning, no compiled code.

``attn="flash_attention_2"`` (the model loaded or switched to it; the fa2-sm120 wheel holds
the tower's head dim 72) takes ``flash_attn_varlen_func`` as the tower's core instead of
``tower_attn``: with one image ``max_seqlen`` is the token count, symbolic under the
mark; with several it is the encoder's int, one graph per value. The LM layers run
transformers' flash path unchanged.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


@torch.library.custom_op("ocr_vl16::tower_attn", mutates_args=())
def tower_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu: torch.Tensor, scale: float
) -> torch.Tensor:
    """Bidirectional attention within each image of a packed ``(T, H, D)`` batch."""
    b = cu.tolist()
    out = torch.empty_like(q)
    for s, e in zip(b[:-1], b[1:]):
        o = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1),
            k[s:e].transpose(0, 1),
            v[s:e].transpose(0, 1),
            scale=scale,
        )
        out[s:e] = o.transpose(0, 1)
    return out


@tower_attn.register_fake
def _(q, k, v, cu, scale):
    return torch.empty_like(q)


@torch.library.custom_op("ocr_vl16::tower_attn_bwd", mutates_args=())
def tower_attn_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = cu.tolist()
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def attn(qi, ki, vi):
        return F.scaled_dot_product_attention(qi, ki, vi, scale=scale)

    # autograd is off below the dispatcher inside a custom op; functorch's vjp is not
    for s, e in zip(b[:-1], b[1:]):
        _, vjp = torch.func.vjp(attn, *(t[s:e].transpose(0, 1) for t in (q, k, v)))
        gq, gk, gv = vjp(do[s:e].transpose(0, 1))
        dq[s:e], dk[s:e], dv[s:e] = (g.transpose(0, 1) for g in (gq, gk, gv))
    return dq, dk, dv


@tower_attn_bwd.register_fake
def _(do, q, k, v, cu, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, cu, scale = inputs
    ctx.save_for_backward(q, k, v, cu)
    ctx.scale = scale


def _backward(ctx, do):
    q, k, v, cu = ctx.saved_tensors
    dq, dk, dv = tower_attn_bwd(do.contiguous(), q, k, v, cu, ctx.scale)
    return dq, dk, dv, None, None


tower_attn.register_autograd(_backward, setup_context=_setup)


def _vision_attn_forward(
    self, hidden_states, cu_seqlens, position_embeddings, max_seqlen=None, **kwargs
):
    """``PaddleOCRVisionAttention.forward`` with the core as ``tower_attn``."""
    from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
        apply_rotary_pos_emb_vision,
    )

    n = hidden_states.shape[0]
    q = self.q_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.to(self.out_proj.weight.dtype) for t in (q, k, v))  # bf16 core
    o = tower_attn(
        q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens, self.scaling
    )
    return self.out_proj(o.reshape(n, -1)), None


def _vision_attn_forward_fa2(
    self, hidden_states, cu_seqlens, position_embeddings, max_seqlen=None, **kwargs
):
    """``PaddleOCRVisionAttention.forward`` with FlashAttention as the core: one image is
    ``flash_attn_func`` on ``(1, T, H, D)``; several are ``flash_attn_varlen_func``,
    k's bounds a copy of q's (dynamo breaks the graph on one tensor passed to two inputs
    of an ``autograd.Function``)."""
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
        apply_rotary_pos_emb_vision,
    )

    n = hidden_states.shape[0]
    q = self.q_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.to(self.out_proj.weight.dtype) for t in (q, k, v))  # bf16 core
    if cu_seqlens.shape[0] == 2:
        o = flash_attn_func(q[None], k[None], v[None], softmax_scale=self.scaling)[0]
    else:
        m = int(max_seqlen)
        o = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens.clone(), m, m, softmax_scale=self.scaling
        )
    return self.out_proj(o.reshape(n, -1)), None


def compile_layers(
    base, tower_range: tuple[int, int], max_images: int, attn: str = "sdpa"
) -> tuple[int, int]:
    """Patch the tower attention and compile every tower / LM layer in place. Run after
    peft wraps the linears and after the checkpoint flags are set. ``attn`` is the
    model's attention (``sdpa`` or ``flash_attention_2``). Returns the layer counts
    (tower, LM)."""
    if attn not in ("sdpa", "flash_attention_2"):
        raise ValueError(f"attn must be sdpa or flash_attention_2, got {attn!r}")
    flash = attn == "flash_attention_2"
    from library.runtime.dynamo import pin_dynamo_limit

    # one graph per crop count (LM) and the tower's first layer (its input has no grad)
    pin_dynamo_limit("recompile_limit", 32)
    torch._dynamo.config.specialize_float = True
    torch._logging.set_logs(recompiles=True, graph_breaks=True)  # each with its reason
    lo, hi = tower_range
    if not 1 < lo <= hi:
        raise ValueError(
            f"tower_range must be (lo, hi) with 1 < lo <= hi, got {tower_range}"
        )
    # Inductor's mix-order reduction guard sits at 4096: one dynamic range may not span
    # it, so a range past it is marked as two ([lo, 4095] and [4096, hi]), one graph each
    # (bs 8 at --token_budget 6144 packs ~5,000 tower tokens)
    spans = [(lo, min(hi, 4095))] + ([(max(lo, 4096), hi)] if hi >= 4096 else [])
    spans = [(a, b) for a, b in spans if a <= b]
    warned: set[int] = set()

    def tower(compiled, eager):
        def forward(hidden_states, cu_seqlens, position_embeddings, **kwargs):
            if not torch.is_grad_enabled():
                return eager(hidden_states, cu_seqlens, position_embeddings, **kwargs)
            for k in ("interp_indices", "interp_weights") + (
                () if flash else ("max_seqlen",)
            ):
                kwargs.pop(k, None)
            t = hidden_states.shape[0]
            span = next(((a, b) for a, b in spans if a <= t <= b), None)
            if span is not None:
                a, b = span
                torch._dynamo.mark_dynamic(hidden_states, 0, min=a, max=b)
                for pe in position_embeddings:
                    torch._dynamo.mark_dynamic(pe, 0, min=a, max=b)
                if max_images > 1:
                    torch._dynamo.mark_dynamic(cu_seqlens, 0, min=2, max=max_images + 1)
            elif t not in warned:
                warned.add(t)
                print(f"compile: {t} tower tokens outside {spans}", flush=True)
            return compiled(hidden_states, cu_seqlens, position_embeddings, **kwargs)

        return forward

    n_threads = torch.get_num_threads()

    def lm(compiled, eager):
        def forward(
            hidden_states, attention_mask=None, position_embeddings=None, **kwargs
        ):
            if not torch.is_grad_enabled():
                return eager(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            # the checkpoint recompute runs on the autograd device thread, whose
            # intra-op thread count is the process default, not the main thread's;
            # dynamo's GLOBAL_STATE guard then splits every graph in two
            if torch.get_num_threads() != n_threads:
                torch.set_num_threads(n_threads)
            b = hidden_states.shape[0]
            mark_b = 2 <= b <= max_images and max_images > 2
            if mark_b:
                torch._dynamo.mark_dynamic(hidden_states, 0, min=2, max=max_images)
            torch._dynamo.maybe_mark_dynamic(hidden_states, 1)
            if attention_mask is not None and attention_mask.ndim == 4:
                if mark_b and attention_mask.shape[0] == b:
                    torch._dynamo.mark_dynamic(attention_mask, 0, min=2, max=max_images)
                torch._dynamo.maybe_mark_dynamic(attention_mask, 2)
                torch._dynamo.maybe_mark_dynamic(attention_mask, 3)
            if position_embeddings is not None:
                for pe in position_embeddings:
                    if mark_b and pe.ndim >= 3 and pe.shape[pe.ndim - 3] == b:
                        torch._dynamo.mark_dynamic(
                            pe, pe.ndim - 3, min=2, max=max_images
                        )
                    torch._dynamo.maybe_mark_dynamic(pe, pe.ndim - 2)
            return compiled(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return forward

    n_t = n_l = 0
    for m in base.modules():
        name = type(m).__name__
        if name == "PaddleOCRVisionAttention":
            core = _vision_attn_forward_fa2 if flash else _vision_attn_forward
            m.forward = types.MethodType(core, m)
        elif name == "PaddleOCRVisionEncoderLayer":
            m.forward = tower(torch.compile(m.forward, dynamic=False), m.forward)
            n_t += 1
        elif name == "PaddleOCRDecoderLayer":
            m.forward = lm(torch.compile(m.forward, dynamic=False), m.forward)
            n_l += 1
    return n_t, n_l
