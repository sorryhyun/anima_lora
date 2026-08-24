"""Shared run-shell for the bespoke distillation loops.

``scripts/distill_turbo`` / ``project/finished/mod_guidance`` keep their method math (DMD
student/fake/critic, GAD pooled-text objective) local and legitimately
divergent — but they repeat the same
orchestration *shell*: pick device/dtype, slice to a single overfit prompt, stand
up the TensorBoard writer, write the config snapshot, and apply the free-fit
dynamic-seq compile guard. Those are correctness policies, not method math, so a
change to compile/free-fit/snapshot semantics should land once here instead of
being missed in one of three near-identical copies.

These are deliberately small, stateless helpers — the loops call them and keep
their own dataloader/optimizer/inner-step logic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DTYPES: dict[str, "Any"] = {}


def resolve_device_dtype(dtype: str = "bf16", device: str = "cuda"):
    """``(torch.device, torch.dtype)`` for a training run.

    All three distill loops hardcode ``cuda`` + ``bfloat16``; this keeps that the
    default while leaving the string knobs overridable.
    """
    import torch

    if not _DTYPES:
        _DTYPES.update(
            bf16=torch.bfloat16,
            fp16=torch.float16,
            fp32=torch.float32,
        )
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
    return torch.device(device), _DTYPES[dtype]


def apply_single_prompt_slice(
    dataset: Any, index: int, *, logger: logging.Logger = log
) -> tuple:
    """Phase-0 overfit: pin ``dataset.samples`` to the single sample at ``index``.

    ``index`` is taken modulo the sample count, so it never indexes out of range;
    the dataloader then cycles the one (latent, text) pair. Returns the pinned
    sample tuple. The caller is responsible for the ``batch_size == 1`` precondition
    (a ``drop_last=True`` loader over one sample yields zero batches at B>1).
    """
    n = len(dataset.samples)
    pinned = index % n
    only = dataset.samples[pinned]
    dataset.samples = [only]
    logger.info(
        "single-prompt overfit mode: pinned idx=%d (post-slice len(dataset)=%d, latent=%s)",
        index,
        len(dataset),
        os.path.basename(only[0]),
    )
    return only


def create_tb_writer(
    log_dir: str,
    config_text: str,
    *,
    enabled: bool = True,
    logger: logging.Logger = log,
):
    """``(writer, run_log)`` — a timestamped TensorBoard run dir + config text.

    Returns ``(None, None)`` when ``enabled`` is False (``--no_log``). The run dir
    is returned so callers can drop sibling artifacts (e.g. a config snapshot)
    into the same timestamped folder.
    """
    if not enabled:
        return None, None
    from datetime import datetime

    from torch.utils.tensorboard import SummaryWriter

    run_log = Path(log_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_log))
    writer.add_text("config", config_text)
    logger.info("TensorBoard logs -> %s", run_log)
    return writer, run_log


def write_config_snapshot(
    path: Any, text: str, *, logger: logging.Logger = log
) -> bool:
    """Write a config snapshot to ``path``, logging success/failure. Never raises.

    Returns True on success. An ``OSError`` (read-only dir, etc.) is downgraded to
    a warning so a snapshot-write failure can't abort a long training run.
    """
    path = Path(path)
    try:
        path.write_text(text, encoding="utf-8")
        logger.info("Config snapshot written: %s", path)
        return True
    except OSError as e:
        logger.warning("Could not write config snapshot to %s: %s", path, e)
        return False


def ensure_dynamic_seq_for_freefit(
    token_counts, dynamic_seq: bool, *, logger: logging.Logger = log
) -> bool:
    """Force ``dynamic_seq`` on for the (free-fit) cached pool.

    Free-fit is the only resize mode now: a cached pool lands many distinct token
    counts inside one tier's band, so the static per-count compile cascade would
    explode and poison the compile cache. The bespoke distill loops never see
    train.py's auto-enable (project_daemon_wiring_pattern), so they call this.
    Callers gate it on ``torch_compile`` being set. Always returns ``True``
    (``token_counts`` is accepted for call-site compatibility, no longer read).
    """
    if not dynamic_seq:
        logger.info(
            "auto-enabling dynamic_seq — free-fit shapes need the single-graph "
            "dynamic-seq path (static compile would explode the graph cascade)"
        )
    return True
