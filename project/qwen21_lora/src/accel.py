"""Attention backend selection and per-block ``torch.compile`` for Qwen-Image-2.1.

Two levers, both independent of the block swapper in ``blockswap.py``:

**Attention backend.** ``QwenImage21AttnProcessor`` routes through diffusers'
``dispatch_attention_fn``, whose ``backend`` argument the processor fills from its
own ``_attention_backend`` on the *decode* call and leaves ``None`` on each
*prefill* segment. ``None`` does not mean native: it means the globally active
backend, which is what ``ModelMixin.set_attention_backend`` also installs. So
that method routes the prefill through flash-attn too, and the prefill hands it
the block-causal bool mask, which flash-attn 2 rejects outright.

:func:`set_attention_backend` here therefore writes ``processor._attention_backend``
and nothing else — the decode path takes the new kernel, the prefill keeps the
global default. With ``use_kv_cache=True`` (the pipeline default) that is step 0
prefilling and every later step decoding, so the backend covers ``steps - 1`` of
the loop.

The decode path has a mask of its own: the text-padding mask, which flash-attn 2
also rejects. ``encode_prompt`` returns ``prompt_embeds_mask=None`` when nothing
is padded — the single-prompt case — so ``flash`` applies there. A padded batch
needs ``flash_varlen``, which unpads instead; the function says which one it
picked and why.

**Block compilation.** The repo's own rule (``DiT.compile_blocks``) is to compile
the per-block inner forward and leave the module hooks eager around it, which is
exactly what the block swapper needs: ``wait_for_block`` / ``submit_move_blocks``
run as forward hooks in ``nn.Module._call_impl``, so compiling ``block.forward``
(the instance attribute, not ``block.compile()``) keeps the thread-pool waits out
of the traced region. ``block.compile()`` would wrap ``_call_impl`` itself and
trace the hooks.

Swapped weights are safe under compile because a block's parameters are always on
the device by the time its forward runs — ``wait_for_block`` is what guarantees
that — so the device never changes *at trace or call time*. Only the storage
behind ``weight.data`` rotates, and inlined nn.Module parameters are graph inputs.
Do not combine this with ``mode="reduce-overhead"``: cuda graphs record static
input addresses, which the swapper invalidates every block.

By default only the decode shape is compiled (``segments is None``). The prefill
branch runs a Python loop over per-segment attention calls whose count and
boundaries come from the prompt, so it would specialize per prompt length for a
single use per generation.
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

    Per-processor, never global — see the module docstring for why
    ``ModelMixin.set_attention_backend`` is the wrong call here.

    ``padded_prompt`` is whether the caller is passing a ``prompt_embeds_mask``;
    with one, a mask-rejecting backend is swapped for its varlen sibling.
    Returns the backend actually installed.
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
        # field: None would defer to the global active backend, which is what we
        # are deliberately not touching.
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

    ``segments`` is the prefill's per-segment boundary list and ``None`` on the
    decode path (and whenever the KV cache is off in the pipeline, which makes
    every step a prefill — then nothing here compiles, by design).
    """

    @functools.wraps(eager)
    def forward(*args, segments=None, **kwargs):
        target = eager if segments is not None else compiled
        return target(*args, segments=segments, **kwargs)

    return forward


def compile_blocks(
    blocks: nn.ModuleList,
    *,
    mode: str | None = None,
    backend: str = "inductor",
    dynamic: bool | None = False,
    decode_only: bool = True,
) -> int:
    """``torch.compile`` each block's ``forward``. Returns the block count.

    Compiles the bound method as an instance attribute so ``_call_impl`` still
    runs the block-swap hooks eagerly around it. Every block traces the same
    code, so only the first pays a full inductor compile and the rest hit the FX
    graph cache.
    """
    if mode == "reduce-overhead":
        raise ValueError(
            "reduce-overhead records cuda graphs with static input addresses, "
            "which the block swapper rewrites every forward"
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
        block.forward = (
            _decode_dispatch(block.forward, compiled) if decode_only else compiled
        )

    scope = "decode only" if decode_only else "every shape"
    print(
        f"compile: {len(blocks)} block.forward with backend={backend} mode={mode} "
        f"dynamic={dynamic} ({scope}, recompile_limit={limit})",
        flush=True,
    )
    return len(blocks)


def recompile_report() -> str:
    """One line of dynamo's frame counters — a recompile storm shows up here."""
    from torch._dynamo.utils import counters

    stats = counters.get("frames", {})
    return f"dynamo frames: ok={stats.get('ok', 0)} total={stats.get('total', 0)}"
