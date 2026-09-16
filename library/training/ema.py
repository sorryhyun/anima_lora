"""Model-EMA helpers for the bespoke distillation/finetune loops.

Loop-internal machinery: import by submodule path, not via the
``library.training`` facade.
"""

import copy

import torch


def make_ema(model):
    """Return a frozen eval-mode deep copy of ``model`` to track as its EMA."""
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema, model, decay):
    """One EMA step: params lerp toward ``model`` at rate ``1 - decay``, buffers copy."""
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)
