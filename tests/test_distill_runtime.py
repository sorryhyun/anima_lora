"""library/training/distill_runtime.py — the shared distill run-shell helpers.

The free-fit dynamic-seq guard and the cosine floor are correctness policies the
bespoke loops share; pin them here so a change lands once.
"""

import logging

import pytest

from library.training.distill_runtime import (
    ensure_dynamic_seq_for_freefit,
)

log = logging.getLogger("test")


# Canonical 1024-tier token counts (4032/4200).
_ON_TABLE = {4032, 4200}


def test_freefit_guard_forces_dynamic_for_on_table_counts():
    # Free-fit is the only resize mode now, so the guard forces dynamic_seq
    # unconditionally — even when every count happens to land on the old table.
    assert ensure_dynamic_seq_for_freefit(_ON_TABLE, False, logger=log) is True


def test_make_scheduler_cosine_anneals_to_a_tenth_of_peak():
    """The turbo scheduler is warmup → cosine to 0.1·lr, and nothing else.

    The 'constant' shape was removed 2026-07-14 (superturbo_B2: never settles,
    rendered worse than its cosine twin) — pin that the anneal really lands at
    0.1·lr so a regression back to a flat tail can't slip in silently.
    """
    import torch

    from scripts.distill_turbo.primitives import make_scheduler

    total, lr = 1000, 3e-5
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    cosine = make_scheduler(opt, total, lr)
    for _ in range(total):
        opt.step()
        cosine.step()
    assert cosine.get_last_lr()[0] == pytest.approx(0.1 * lr, rel=1e-3)
