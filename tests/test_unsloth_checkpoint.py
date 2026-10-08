"""unsloth reentrant-checkpoint grad-flow invariant.

The reentrant unsloth checkpoint silently ZEROES grads that reach only
closed-over params (it broke BYG); a
grad-requiring tensor *input* resurrects them (the EasyControl precedent). Any
loop that checkpoints a forward whose only trainable weights are closed-over
LoRA params (turbo's GAN path via ``selective_block_grad_ckpt``, EasyControl)
relies on this — this test pins the mechanism so a checkpoint refactor can't
silently turn such a term into a no-op.
"""

from __future__ import annotations

import torch


def test_unsloth_ckpt_propagates_param_grads_via_input():
    import torch.nn as nn

    from library.anima.models import unsloth_checkpoint

    lin = nn.Linear(4, 4)  # closed-over param, the LoRA stand-in
    x = torch.randn(2, 4, requires_grad=True)  # grad-requiring input stand-in
    unsloth_checkpoint(lin, x).sum().backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.abs().sum() > 0
