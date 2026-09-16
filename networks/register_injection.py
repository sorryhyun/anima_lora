"""Shared mid-stack register-token injection for the Anima DiT (DSR mechanism).

Two owners drive this one implementation:

* the standalone register method (``--method register``,
  ``networks/methods/register.py``), and
* the LoRA family (``num_registers > 0`` in a lora-family TOML,
  ``networks/lora_anima/network.py``) — a full LoRA trained jointly with
  learnable registers.

Mechanism (DSR, arXiv:2605.05206 — "starting block" pattern):

* K non-decoded register tokens are concatenated onto the *self-attention*
  sequence at block ``insert_block`` (0 = stack entry, default 8 = DSR
  Tab. 9's sweet spot), carried through the remaining blocks as a
  residual-carrying scratchpad, and stripped before unpatchify.
* Rope-exempt: registers get identity cos/sin rows appended to the rope
  tuple, so no attention-path seq-axis slicing is needed.
* Mid-stack insertion rides ``forward_pre_hook``s on blocks ≥ ``insert_block``
  (concat at the insert block, rope swap on every later block — the rope tuple
  is re-passed per block by ``_run_blocks``), eager and compile-safe (hooks run
  at block ``__call__`` granularity, outside the compiled ``_forward``).
* Forces ``_native_flatten`` so the seq axis is dim 2 (``(B, 1, seq, 1, D)``)
  — bit-exact to the eager 5D path (see the ``bucketing`` skill).

The injector is owner-agnostic: the owner supplies ``get_scaled_tokens``, a
zero-arg callable returning the ``(K, D)`` register tensor already scaled by
the owner's live ``multiplier`` (so ``set_multiplier`` needs no injector
plumbing). Compile coupling: every forward past ``insert_block`` grows the seq
by a constant K, so owners must expose ``extra_seq_tokens`` for
``train.py``'s dynamic-seq MAX widening.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


class RegisterInjector:
    """Wraps ``anima._run_blocks`` to inject K register tokens mid-stack.

    Also computes the relocation-crossover readouts (``last_reg_ratio`` /
    ``last_patch_sink_ratio`` — training-time adoption gate, no-grad) each
    forward.
    """

    def __init__(
        self,
        *,
        num_registers: int,
        insert_block: int,
        get_scaled_tokens: Callable[[], torch.Tensor],
    ) -> None:
        self.K = int(num_registers)
        self.insert_block = int(insert_block)
        self._get_tokens = get_scaled_tokens
        self._applied = False
        self._anima = None
        self._orig_run_blocks = None
        self._orig_native_flatten = None
        self._hook_handles: list = []
        # per-forward (reg, rope_ext) handed to the insert-block pre-hooks
        self._inject = None
        # relocation-crossover readouts (sparse sink → top-0.2%/median outlier)
        self.last_reg_ratio: Optional[float] = None
        self.last_patch_sink_ratio: Optional[float] = None

    @property
    def extra_seq_tokens(self) -> int:
        return self.K

    def apply(self, anima) -> None:
        """Install the run-blocks wrapper + mid-stack pre-hooks on ``anima``."""
        if self._applied:
            return
        n_blocks = len(anima.blocks)
        if not (0 <= self.insert_block < n_blocks):
            raise ValueError(
                f"insert_block must be in [0, {n_blocks}), got {self.insert_block}"
            )
        self._anima = anima
        self._orig_native_flatten = anima._native_flatten
        anima._native_flatten = True  # eager native-flatten path

        inj = self
        self._orig_run_blocks = anima._run_blocks

        def wrapped_run_blocks(
            x_padded,
            t_embedding_B_T_D,
            crossattn_emb,
            attn_params,
            capture_blocks=None,
            feature_sink=None,
            stop_after_block=None,
            **block_kwargs,
        ):
            # x_padded: (B, 1, seq, 1, D) under native-flatten.
            B = x_padded.shape[0]
            seq = x_padded.shape[2]
            x_ext = x_padded
            if inj.K > 0:
                reg = inj._get_tokens().to(dtype=x_padded.dtype, device=x_padded.device)
                D = reg.shape[-1]
                reg = reg.view(1, 1, inj.K, 1, D).expand(B, -1, -1, -1, -1)
                rope_ext = None
                rope = block_kwargs.get("rope_cos_sin")
                if rope is not None:
                    cos, sin = rope  # each (seq, 1, 1, D_head)
                    pad_shape = (inj.K,) + tuple(cos.shape[1:])
                    rope_ext = (
                        torch.cat([cos, cos.new_ones(pad_shape)], dim=0),
                        torch.cat([sin, sin.new_zeros(pad_shape)], dim=0),
                    )
                if inj.insert_block == 0:
                    x_ext = torch.cat([x_padded, reg], dim=2)  # (B, 1, seq+K, 1, D)
                    if rope_ext is not None:
                        block_kwargs = {**block_kwargs, "rope_cos_sin": rope_ext}
                else:
                    # mid-stack insertion: the pre-hooks on blocks >= insert_block
                    # concat/rope-swap; blocks before it run at the bare seq.
                    inj._inject = (reg, rope_ext)

            try:
                out = inj._orig_run_blocks(
                    x_ext,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    attn_params,
                    capture_blocks=capture_blocks,
                    feature_sink=feature_sink,
                    stop_after_block=stop_after_block,
                    **block_kwargs,
                )
            finally:
                inj._inject = None
            # relocation crossover (sparse sink → track top-0.2%/median outlier).
            # Guarded on actual seq growth: an early-exit forward that stops
            # before insert_block never received the registers.
            with torch.no_grad():
                pt = out[:, :, :seq, :, :].float().norm(dim=-1).flatten()
                med = pt.median().clamp_min(1e-6)
                k = max(1, int(0.002 * pt.numel()))
                inj.last_patch_sink_ratio = (pt.topk(k).values.mean() / med).item()
                if inj.K > 0 and out.shape[2] > seq:
                    rt = out[:, :, seq:, :, :].float().norm(dim=-1).flatten()
                    inj.last_reg_ratio = (rt.max() / med).item()
            return out[:, :, :seq, :, :]  # strip registers before unpatchify

        anima._run_blocks = wrapped_run_blocks

        # Mid-stack insertion (DSR "starting block"): concat at insert_block's
        # entry, and swap in the extended rope on every block >= insert_block —
        # _run_blocks re-passes the ORIGINAL rope tuple to each block, so the
        # insert-block concat alone would leave later blocks with a seq-length
        # rope against seq+K tokens. Pre-hooks run eagerly at block __call__,
        # outside the compiled _forward, so this is compile-safe.
        if self.K > 0 and self.insert_block > 0:
            insert_at = self.insert_block

            def make_pre_hook(bi):
                def pre_hook(module, args, kwargs):
                    pending = inj._inject
                    if pending is None:
                        return None
                    reg, rope_ext = pending
                    x = args[0]
                    if bi == insert_at:
                        x = torch.cat([x, reg], dim=2)
                    if rope_ext is not None and kwargs.get("rope_cos_sin") is not None:
                        kwargs = {**kwargs, "rope_cos_sin": rope_ext}
                    return (x,) + tuple(args[1:]), kwargs

                return pre_hook

            for bi in range(insert_at, n_blocks):
                self._hook_handles.append(
                    anima.blocks[bi].register_forward_pre_hook(
                        make_pre_hook(bi), with_kwargs=True
                    )
                )

        self._applied = True

    def remove(self) -> None:
        """Restore the base forward bit-exactly."""
        if not self._applied:
            return
        anima = self._anima
        anima._run_blocks = self._orig_run_blocks
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        anima._native_flatten = self._orig_native_flatten
        self._anima = None
        self._applied = False
