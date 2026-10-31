"""Dynamo/inductor compile-config pinning helpers.

Set a ``torch._dynamo`` / ``torch._inductor`` config value so it survives into
the *backward* compile context (see ``pin_dynamo_limit``). Dependency-free
(torch + logging only) so any layer can import it without a cycle.
"""

from __future__ import annotations

import logging
from typing import Optional

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


def pin_dynamo_flag(name: str, value: object) -> None:
    """:func:`pin_dynamo_limit` for a non-numeric ``torch._dynamo.config`` entry:
    the override plus the canonical entry's ``.default``, so the backward-compile
    context reads it too."""
    import torch._dynamo as _dynamo

    cfg_mod = _dynamo.config
    setattr(cfg_mod, name, value)
    try:
        entry = cfg_mod._config[name]
        canon = (entry.alias or name).rsplit(".", 1)[-1]
        cfg_mod._config[canon].default = value
    except Exception as e:  # noqa: BLE001 - defensive against torch internals
        logger.warning(
            f"could not pin dynamo {name} default ({e}); the value may revert in "
            "the backward-compile context"
        )


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


def pin_compiler_flag(name: str, value: object) -> None:
    """:func:`pin_inductor_flag` for ``torch.compiler.config`` (e.g.
    ``dynamic_sources``): the override plus the entry's ``.default``, so the
    checkpoint recompute's context reads it too."""
    import torch.compiler.config as cfg_mod

    setattr(cfg_mod, name, value)
    try:
        cfg_mod._config[name].default = value
    except Exception as e:  # noqa: BLE001 - defensive against torch internals
        logger.warning(
            f"could not pin torch.compiler {name} default ({e}); the value may "
            "revert in the backward-compile context"
        )


def compile_signature(
    *,
    n_token_families: Optional[int],
    seq_range: Optional[tuple],
    dynamic_seq: bool,
    backend: str = "inductor",
    mode: Optional[str] = None,
    seq_bands: Optional[list] = None,
) -> str:
    """Canonical signature string for ``maybe_clear_stale_compile_cache``.

    Every compile entry point must build the marker through this one
    formatter so equivalent configs serialize identically. ``mode`` is
    normalized so ``None`` and ``""`` ("inductor default") don't read as a
    signature change. ``seq_bands`` (per-band dynamic-seq) is appended only
    when set, so pre-existing union-range signatures — and their persistent
    cache dirs — stay byte-identical.
    """
    sig = (
        f"families={n_token_families};seq_range={seq_range};"
        f"dynamic_seq={dynamic_seq};backend={backend};mode={mode or None}"
    )
    if seq_bands:
        sig += f";seq_bands={sorted(tuple(b) for b in seq_bands)}"
    return sig


# captured on the first isolate_compile_cache call so later calls in the same
# process re-derive from the same root instead of nesting per-signature dirs
_compile_cache_base: Optional[str] = None


def isolate_compile_cache(signature: str) -> str:
    """Route this run's torch.compile caches to a per-signature directory.

    GOTCHA: the persistent compile caches key on the FX graph but NOT on the
    ``mark_dynamic`` value range, so processes compiled with different
    seq-range bounds poison each other through the shared default cache dir —
    e.g. an inference run's narrower stored guard gets replayed into a wider
    training-run mark and raises ``ConstraintViolationError`` instead of
    cleanly missing (hint-dependent, so it strikes "sometimes"). Wiping the
    shared dir doesn't fix it since inference keeps re-depositing; instead
    this points ``TORCHINDUCTOR_CACHE_DIR`` at a per-signature subdir so
    every entry inside was compiled under the same seq bounds.

    Must run BEFORE the first ``torch.compile`` trace in the process (torch
    reads the env var lazily per cache access). Build ``signature`` via
    ``compile_signature``. Returns the directory used.
    """
    global _compile_cache_base
    import hashlib
    import os

    if _compile_cache_base is None:
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not base:
            try:
                from torch._inductor.runtime.cache_dir_utils import default_cache_dir

                base = default_cache_dir()
            except Exception:  # noqa: BLE001 — torch internals move across versions
                import getpass
                import tempfile

                base = os.path.join(
                    tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
                )
        _compile_cache_base = base

    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    target = os.path.join(_compile_cache_base, f"anima-sig-{digest}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target
    logger.info(f"torch.compile cache isolated per compile signature: {target}")
    logger.info(f"compile signature: {signature}")
    return target
