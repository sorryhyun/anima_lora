"""Attention backend selection and per-block ``torch.compile`` for Qwen-Image-2.1.

Two levers, both independent of the block swapper in ``blockswap.py``.

**Attention backend.** With ``use_kv_cache=True`` (the pipeline default) step 0
prefills and every later step decodes, so a per-processor backend covers
``steps - 1`` of the loop. There is little left to win there: SDPA already
dispatches the maskless decode attention to FlashAttention-2
(``pytorch_flash::flash_fwd_kernel``), so an explicit ``flash`` measures ±0. The
masked prefill segments are the attention left on a slow path.

**Block compilation.** The decode shape is compiled by default
(``segments is None``). The prefill branch runs a Python loop over per-segment
attention calls whose count and boundaries come from the prompt, so it would
specialize per prompt length for a single use per generation.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

# Backends that raise on a non-None `attn_mask` rather than honouring it.
_MASKLESS_BACKENDS = {"flash", "flash_hub", "_flash_3", "_flash_3_hub", "sage"}

# Their unpadding counterparts, for when the decode mask is not None.
_VARLEN_FALLBACK = {
    "flash": "flash_varlen",
    "flash_hub": "flash_varlen_hub",
    "_flash_3": "_flash_varlen_3",
    "_flash_3_hub": "_flash_3_varlen_hub",
    "sage": "sage_varlen",
}


def set_attention_backend(
    transformer,
    backend: str,
    *,
    padded_prompt: bool = False,
) -> str | None:
    """Point the transformer's attention processors at ``backend``.

    Per-processor, rather than ``ModelMixin.set_attention_backend``, which also
    sets the *globally* active backend, and ``QwenImage21AttnProcessor`` passes
    ``backend=None`` on each *prefill* segment, which falls through to exactly
    that global — handing flash-attn 2 the block-causal bool mask it rejects
    outright. Writing ``processor._attention_backend`` leaves the prefill on the
    global default.

    ``padded_prompt`` is whether the caller is passing a ``prompt_embeds_mask``;
    flash-attn 2 rejects that one too, so a mask-rejecting backend is swapped
    for its varlen sibling, which unpads instead. Returns the backend actually
    installed.
    """
    from diffusers.models.attention import AttentionModuleMixin
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _check_attention_backend_requirements,
        _maybe_download_kernel_for_backend,
    )

    if backend in (None, "", "default"):
        return None

    chosen = backend
    if padded_prompt and backend in _MASKLESS_BACKENDS:
        chosen = _VARLEN_FALLBACK[backend]
        print(
            f"attention: {backend} rejects an attn_mask and the prompt is padded "
            f"-> using {chosen}",
            flush=True,
        )

    name = AttentionBackendName(chosen)
    _check_attention_backend_requirements(name)
    _maybe_download_kernel_for_backend(name)

    count = 0
    for module in transformer.modules():
        if not isinstance(module, AttentionModuleMixin):
            continue
        processor = module.processor
        if processor is None or not hasattr(processor, "_attention_backend"):
            continue
        # `native` is spelled as an explicit backend rather than by clearing the
        # field: None would defer to the global active backend.
        processor._attention_backend = name
        count += 1
    print(
        f"attention: {chosen} on {count} processors' decode path "
        "(prefill segments stay on the global default)",
        flush=True,
    )
    return chosen


def _decode_dispatch(eager, compiled):
    """Route the decode shape to ``compiled`` and the prefill shape to ``eager``.

    ``segments`` is the prefill's per-segment boundary list, ``None`` on the
    decode path.
    """

    @functools.wraps(eager)
    def forward(*args, segments=None, **kwargs):
        target = eager if segments is not None else compiled
        return target(*args, segments=segments, **kwargs)

    return forward


# Block-forward inputs whose one varying axis is the joint sequence length, by
# argument name -> the dim that carries it. `hidden_states` is (B, T, D),
# `rotary_emb` (T, ...), `target_token_mask` (T,), `key_valid` (B, T), and the
# decode path's `attention_mask` (B, 1, 1, T).
_SEQ_AXES = {
    "hidden_states": 1,
    "rotary_emb": 0,
    "target_token_mask": 0,
    "key_valid": 1,
    "attention_mask": -1,
}


def _make_bounded_forward(block: nn.Module, compiled, lo: int, hi: int):
    """Wrap ``compiled`` in an eager ``mark_dynamic`` prologue over ``[lo, hi]``.

    The marks are applied on every call, not once: gradient checkpointing's
    recompute detaches the block inputs into fresh tensors that carry no mark,
    and a forward marked / recompute unmarked pair raises
    ``ConstraintViolationError``. Applying them here — inside what the
    checkpointed ``__call__`` reaches — keeps both passes in agreement.

    ``segments`` (Python ints: the text length and slot boundaries) cannot be
    marked; automatic dynamic shapes turns them symbolic on the first recompile
    after they change. A sequence outside ``[lo, hi]`` is left unmarked and
    specializes statically on that count.
    """
    import inspect

    signature = inspect.signature(block.forward)
    warned: set[int] = set()

    @functools.wraps(block.forward)
    def forward(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        arguments = bound.arguments
        seq = int(arguments["hidden_states"].shape[1])
        if lo <= seq <= hi:
            for name, dim in _SEQ_AXES.items():
                value = arguments.get(name)
                if isinstance(value, torch.Tensor) and value.shape[dim] == seq:
                    torch._dynamo.mark_dynamic(value, dim, min=lo, max=hi)
        elif seq not in warned:
            warned.add(seq)
            print(
                f"compile: {seq} joint tokens falls outside the bounded range "
                f"[{lo}, {hi}]; running an unmarked (static) specialization",
                flush=True,
            )
        return compiled(**arguments)

    return forward


def compile_blocks(
    blocks: nn.ModuleList,
    *,
    mode: str | None = None,
    backend: str = "inductor",
    dynamic: bool | None = False,
    decode_only: bool = True,
    seq_range: tuple[int, int] | None = None,
) -> int:
    """``torch.compile`` each block's ``forward``. Returns the block count.

    Compiles the bound method as an instance attribute, not ``block.compile()``:
    ``wait_for_block`` / ``submit_move_blocks`` run as forward hooks in
    ``nn.Module._call_impl``, and this form keeps them outside the traced
    region. Swapped weights are safe because ``wait_for_block`` puts a block's
    parameters on the device before its forward runs; the storage behind
    ``weight.data`` rotates under it.

    ``seq_range=(lo, hi)`` is the bounded mode: automatic dynamic shapes
    (``dynamic=None``) plus a ``mark_dynamic(min=lo, max=hi)`` prologue on the
    sequence axis of every block input (:func:`_make_bounded_forward`). Only
    the joint token count goes symbolic — the hidden and head dims stay static,
    which ``dynamic=True`` does not give — and dynamo knows the range up front.
    The training cache knows ``lo``/``hi`` before the model is placed.
    """
    if mode == "reduce-overhead":
        raise ValueError(
            "reduce-overhead records cuda graphs with static input addresses, "
            "which the block swapper rewrites every forward"
        )
    if seq_range is not None:
        lo, hi = int(seq_range[0]), int(seq_range[1])
        if not 1 < lo <= hi:
            raise ValueError(f"seq_range must satisfy 1 < lo <= hi, got {seq_range}")
        dynamic = None
        if lo < 4096 <= hi:
            # Inductor's mix-order reduction fusion derives a `seq >= 4096`
            # guard from the trace-time hint; under strict marks that guard is a
            # ConstraintViolationError at the first step. Same pin Anima makes.
            from library.runtime.dynamo import pin_inductor_flag

            pin_inductor_flag("triton.mix_order_reduction", False)
            print(
                "compile: inductor triton.mix_order_reduction pinned off "
                f"(bounded range [{lo}, {hi}] straddles 4096)",
                flush=True,
            )

    kwargs: dict[str, object] = {"backend": backend, "dynamic": dynamic}
    if mode is not None:
        kwargs["mode"] = mode

    # One graph per block per shape; decode and prefill are separate shapes and
    # the prompt length moves the prefix, so leave room above the default 8.
    limit = max(torch._dynamo.config.recompile_limit, 16)
    torch._dynamo.config.recompile_limit = limit

    for block in blocks:
        compiled = torch.compile(block.forward, **kwargs)
        if seq_range is not None:
            compiled = _make_bounded_forward(block, compiled, lo, hi)
        block.forward = (
            _decode_dispatch(block.forward, compiled) if decode_only else compiled
        )

    scope = "decode shape" if decode_only else "every shape"
    shapes = (
        f"bounded seq in [{lo}, {hi}] (automatic dynamic + mark_dynamic)"
        if seq_range is not None
        else f"dynamic={dynamic}"
    )
    print(
        f"compile: {len(blocks)} block.forward with backend={backend} mode={mode} "
        f"{shapes} ({scope}, recompile_limit={limit})",
        flush=True,
    )
    return len(blocks)


def recompile_report() -> str:
    """One line of dynamo's frame counters — a recompile storm shows up here."""
    from torch._dynamo.utils import counters

    stats = counters.get("frames", {})
    return f"dynamo frames: ok={stats.get('ok', 0)} total={stats.get('total', 0)}"
