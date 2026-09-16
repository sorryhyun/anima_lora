"""Dynamo/inductor compile-config pinning helpers.

Set a ``torch._dynamo`` / ``torch._inductor`` config value so it survives into
the *backward* compile context (see ``pin_dynamo_limit``). Dependency-free
(torch + logging only) so any layer can import it without a cycle.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pin_dynamo_limit(name: str, value: int) -> int:
    """Raise a dynamo recompile budget so it holds in EVERY execution context.

    ``torch._dynamo.config.<name>`` is backed by a ``ContextVar`` (``user_override``),
    so a plain ``config.<name> = value`` assignment only takes effect in the thread
    /context that ran it. Dynamo compiles the grad-bearing block ``_forward`` in a
    *different* context (the AOTAutograd / backward compile path), where the override
    is absent and the read falls back to the config entry's ``default`` (8) — so the
    budget silently reverts and the loop spills to eager at the first grad forward,
    despite a correct setup-time raise. Pinning the canonical entry's ``.default``
    makes the raise context-independent. We set both: the override (same-context
    reads + log visibility) and the default (compile-/backward-thread reads).

    ``name`` may be an alias (e.g. ``cache_size_limit`` aliases ``recompile_limit``);
    the canonical entry is followed so the right ``.default`` is pinned. Returns the
    effective (max of current and requested) value.
    """
    import torch._dynamo as _dynamo

    cfg_mod = _dynamo.config
    target = max(getattr(cfg_mod, name), value)
    setattr(cfg_mod, name, target)  # context-local override (main thread + logs)
    try:
        entry = cfg_mod._config[name]
        # An alias (e.g. cache_size_limit) stores the canonical name fully
        # qualified; follow it to the real entry whose ``.default`` is the
        # cross-context fallback every thread reads.
        canon = (entry.alias or name).rsplit(".", 1)[-1]
        cfg_mod._config[canon].default = target  # global fallback (all contexts)
    except Exception as e:  # noqa: BLE001 - defensive against torch internals
        logger.warning(
            f"could not pin dynamo {name} default ({e}); budget may revert to 8 "
            "in the backward-compile context and spill to eager"
        )
    return target


def pin_inductor_flag(name: str, value: object) -> None:
    """Set a ``torch._inductor.config`` value so it holds in EVERY execution context.

    Same ContextVar trap as :func:`pin_dynamo_limit`, inductor edition: config
    ``user_override``s are thread-local, and the grad-enabled step-0 compile
    (grad-ckpt recompute / AOT backward path) runs its inductor scheduling in a
    context where a plain assignment's override is absent — the read falls back
    to the entry's ``.default``. For ``triton.mix_order_reduction`` that default
    is env-derived True, so the ``compile_blocks`` kill silently reverted and
    the fusion's hint-derived ``Ge(seq, 4096)`` guard broke strict dynamic-seq
    marks (ConstraintViolationError at the first training step under gradient
    checkpointing). Pin both: the
    override (same-context reads) and the default (compile-/backward-thread
    reads).

    ``name`` is the fully-qualified dotted key, e.g.
    ``"triton.mix_order_reduction"``. (No alias-following here — none of the
    pinned inductor keys alias; ``_config[name]`` IS the canonical entry.)
    """
    import torch._inductor.config as cfg_mod

    obj = cfg_mod
    *parents, leaf = name.split(".")
    for parent in parents:
        obj = getattr(obj, parent)
    setattr(obj, leaf, value)  # context-local override (main thread + logs)
    try:
        cfg_mod._config[name].default = value  # global fallback (all contexts)
    except Exception as e:  # noqa: BLE001 - defensive against torch internals
        logger.warning(
            f"could not pin inductor {name} default ({e}); the value may revert "
            "in the backward-compile context"
        )
