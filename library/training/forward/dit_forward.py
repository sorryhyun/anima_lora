"""The frozen-DiT forward ritual shared by the distillation teacher/student passes.

``model.forward_mini_train_dit`` is the lightweight DiT forward the distillation
loops drive (teacher pred, student pred, validation, teacher-cache prefill). Each
call is wrapped in the same four-step ritual, copy-pasted across
``project/finished/mod_guidance`` (×4 sites) and the SPD / mod-guidance probes:

  1. stage block-swap if enabled (``prepare_block_swap_before_forward``),
  2. open a fresh CUDA-graph epoch (``cudagraph_mark_step_begin``) so a prior
     compiled output isn't overwritten before it's consumed,
  3. run under ``autocast("cuda", dtype)`` (and ``no_grad`` for frozen passes),
  4. optionally ``.clone()`` the result off the (possibly static / reused) output
     buffer so it survives the next compiled-fn call in the same step.

This is the model-touching half of the per-step prep (see
:mod:`library.training.forward.renoise` for the tensor-only half).
"""

from __future__ import annotations

import contextlib

import torch


def run_mini_train_forward(
    model,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    *,
    padding_mask: torch.Tensor,
    dtype: torch.dtype,
    no_grad: bool = False,
    clone: bool = False,
    **forward_kwargs,
) -> torch.Tensor:
    """Run ``model.forward_mini_train_dit`` under the standard distillation ritual.

    ``no_grad`` runs the forward under :func:`torch.no_grad` — use it for the
    frozen teacher / reference passes; leave it off for the grad-bearing student.
    ``clone=True`` lifts the result off a possibly-reused compiled output buffer,
    required when a teacher/reference pred must outlive a *later* forward in the
    same step; leave it ``False`` for the student pass (cloning a grad-bearing
    output only wastes memory). ``**forward_kwargs`` pass straight through —
    e.g. ``skip_pooled_text_proj=True`` (teacher) or
    ``pooled_text_override=...`` (student).
    """
    if getattr(model, "blocks_to_swap", 0):
        model.prepare_block_swap_before_forward()
    torch.compiler.cudagraph_mark_step_begin()
    grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with grad_ctx, torch.autocast("cuda", dtype=dtype):
        out = model.forward_mini_train_dit(
            noisy,
            timesteps,
            crossattn_emb,
            padding_mask=padding_mask,
            **forward_kwargs,
        )
    return out.clone() if clone else out
