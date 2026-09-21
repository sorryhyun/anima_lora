"""Attach the trainer's block swapper to a stock diffusers/transformers model.

``library.runtime.offloading.ModelOffloader`` is written to be driven from
inside a model's block loop — ``wait_for_block(i)`` before block *i* runs,
``submit_move_blocks(blocks, i)`` after (``library/anima/models.py::_run_blocks``).
Qwen-Image-2.1's blocks live in upstream diffusers and transformers, whose
forwards we do not own, so the same two calls are attached as ordinary module
hooks instead. The swapper itself is unmodified.

Both heavy modules are homogeneous ``nn.ModuleList``s, which is all it needs:

    QwenImage21Transformer2DModel.transformer_blocks   32 x ~0.44 GB
    Qwen3VLForConditionalGeneration
        .model.language_model.layers                   36 x ~0.39 GB

The swap *schedule* is ours rather than the offloader's (:func:`swap_schedule`):
``ModelOffloader.submit_move_blocks`` streams the whole model across PCIe every
forward no matter how few blocks are swapped, which at ``blocks_to_swap=4`` on
this model was 55 % of all CUDA time, more than the GEMMs. Driving
``_submit_move_blocks`` from an explicit schedule costs ``2 * blocks_to_swap``
moves per forward.
"""

from __future__ import annotations

import contextlib
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from library.runtime.device import weighs_to_device  # noqa: E402
from library.runtime.offloading import ModelOffloader  # noqa: E402


def find_blocks(model: nn.Module, path: str) -> nn.ModuleList:
    """``"model.language_model.layers"`` → that ``ModuleList``."""
    out: nn.Module = model
    for part in path.split("."):
        out = getattr(out, part)
    if not isinstance(out, nn.ModuleList):
        raise TypeError(f"{path} is {type(out).__name__}, not an nn.ModuleList")
    return out


def block_size_gb(blocks: nn.ModuleList) -> float:
    b = blocks[0]
    return sum(p.numel() * p.element_size() for p in b.parameters()) / 1024**3


def resident_size_gb(model: nn.Module, blocks: nn.ModuleList) -> float:
    """Bytes of everything that is *not* in ``blocks``."""
    swapped = {id(p) for b in blocks for p in b.parameters()}
    return (
        sum(
            p.numel() * p.element_size()
            for p in model.parameters()
            if id(p) not in swapped
        )
        / 1024**3
    )


def auto_blocks_to_swap(
    model: nn.Module,
    blocks: nn.ModuleList,
    free_gb: float,
    *,
    activation_reserve_gb: float = 2.5,
) -> int:
    """Fewest blocks to swap that still leaves ``activation_reserve_gb`` free.

    Swapping costs a PCIe round trip per block per forward, so the answer is
    the minimum that fits.
    """
    per = block_size_gb(blocks)
    budget = free_gb - resident_size_gb(model, blocks) - activation_reserve_gb
    can_keep = max(0, int(budget // per))
    # ModelOffloader needs at least two blocks left resident to swap against.
    return max(0, min(len(blocks) - 2, len(blocks) - can_keep))


def to_device_except_blocks(
    model: nn.Module, blocks: nn.ModuleList, device: torch.device
) -> None:
    """``model.to(device)`` with ``blocks`` left where they are.

    Everything outside the swapped list stays resident. The swapper places the
    blocks itself in ``prepare_block_devices_before_forward``; a plain ``.to()``
    would spike to the full bf16 footprint first, which is the OOM being
    avoided.
    """
    saved = []
    for name, child in list(model.named_children()):
        if child is blocks:
            saved.append((model, name))
    # The list may be nested — walk to its parent rather than assume a child.
    if not saved:
        for mod in model.modules():
            for name, child in list(mod.named_children()):
                if child is blocks:
                    saved.append((mod, name))
    for parent, name in saved:
        setattr(parent, name, None)
    try:
        model.to(device)
    finally:
        for parent, name in saved:
            setattr(parent, name, blocks)


def swap_schedule(
    num_blocks: int, blocks_to_swap: int, minimal: bool = True, restore: bool = True
) -> dict[int, tuple[int, int]]:
    """``{block index: (index to evict, index to load)}`` after that block runs.

    The forward starts with ``blocks[:N-S]`` on the device and ``blocks[N-S:]``
    on the CPU, and must end the same way; that restore is why a block outside the
    swapped tail is touched at all.

        idx in [0, S)      evict idx, load N-S+idx    make room for the tail
        idx in [S, N-S)    --                         resident, stays resident
        idx in [N-S, N)    evict idx, load idx-(N-S)  restore, during the tail

    The loads queued by the last ``S`` blocks are awaited by the *next*
    forward's ``wait_for_block``, which is what makes the restore free.
    Residency, presence-when-run and one submit per destination are replayed in
    ``tests/test_qwen21_swap_schedule.py``.

    The two ranges stay disjoint while ``2S <= N``; past that a block would
    need both hooks and the phases interleave, so ``minimal=False`` (and any
    ``S > N/2``) falls back to ``ModelOffloader``'s rolling ring, which is
    ``N`` moves — no worse than ``2S`` once ``S`` passes ``N/2``.

    ``restore=False`` drops the second range: training leaves the swapped tail
    on the device for the backward, which walks it back itself. What is left is
    the first range at any ``S``, which is ``ModelOffloader.submit_move_blocks``
    exactly (it gates on ``block_idx >= blocks_to_swap`` for the same reason),
    so the rolling-ring fallback does not apply.
    """
    if blocks_to_swap <= 0:
        return {}
    if not restore:
        return {
            idx: (idx, num_blocks - blocks_to_swap + idx)
            for idx in range(blocks_to_swap)
        }
    if not minimal or 2 * blocks_to_swap > num_blocks:
        return {
            idx: (idx, (num_blocks - blocks_to_swap + idx) % num_blocks)
            for idx in range(num_blocks)
        }
    schedule = {
        idx: (idx, num_blocks - blocks_to_swap + idx) for idx in range(blocks_to_swap)
    }
    schedule.update(
        {
            idx: (idx, idx - (num_blocks - blocks_to_swap))
            for idx in range(num_blocks - blocks_to_swap, num_blocks)
        }
    )
    return schedule


_recomputing = False


@contextlib.contextmanager
def _recompute_guard():
    global _recomputing
    _recomputing = True
    try:
        yield
    finally:
        _recomputing = False


def checkpoint_context_fn():
    """``context_fn`` for ``torch.utils.checkpoint`` — hooks off during recompute.

    Gradient checkpointing runs each block's forward a second time, inside the
    backward, and that second ``__call__`` fires the swap hooks again: every
    recomputed block would queue another move and the schedule would run one
    eviction ahead of the blocks still to be recomputed. Wrapping the recompute
    in this context makes both hooks no-ops, leaving the backward hooks
    ``ModelOffloader`` installs as the only driver of the backward direction.
    """
    return contextlib.nullcontext(), _recompute_guard()


class Attached:
    """A live block-swap attachment, and the way to take it back off."""

    def __init__(self, offloader: ModelOffloader, blocks: nn.ModuleList, model):
        self.offloader = offloader
        self.blocks = blocks
        self.model = model
        self.handles: list = []

    def prepare(self) -> None:
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def detach(self) -> None:
        """Remove the hooks, stop the mover, and bring every block back to CPU."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for idx in list(self.offloader.futures.keys()):
            self.offloader._wait_blocks_move(idx)
        self.offloader.futures.clear()
        self.offloader.thread_pool.shutdown(wait=True)
        if self.offloader.supports_backward:
            for handle in self.offloader.remove_handles:
                handle.remove()
            self.offloader.remove_handles.clear()
        cpu = torch.device("cpu")
        for block in self.blocks:
            # A swapped block's parameters and buffers sit on different devices.
            block.to(cpu)
            weighs_to_device(block, cpu)
        self.model.to(cpu)
        self.offloader = None
        self.blocks = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


def attach(
    model: nn.Module,
    blocks_path: str,
    blocks_to_swap: int,
    device: torch.device,
    *,
    supports_backward: bool,
    debug: bool = False,
    minimal_schedule: bool = True,
) -> tuple[Attached | None, nn.ModuleList]:
    """Hook ``ModelOffloader`` onto ``model``'s block list.

    Returns ``(attached, blocks)``; ``attached`` is None when
    ``blocks_to_swap == 0``, in which case the caller can move the model
    normally.

    ``minimal_schedule=False`` reproduces ``ModelOffloader``'s own rolling ring,
    for comparing against it. It is ignored under ``supports_backward``, where
    the forward only half-swaps (:func:`swap_schedule` ``restore=False``) and the
    offloader's own backward hooks walk the tail back.
    """
    blocks = find_blocks(model, blocks_path)
    if blocks_to_swap <= 0:
        return None, blocks

    offloader = ModelOffloader(
        blocks,
        blocks_to_swap,
        device,
        supports_backward=supports_backward,
        debug=debug,
    )
    attached = Attached(offloader, blocks, model)

    schedule = swap_schedule(
        len(blocks),
        blocks_to_swap,
        minimal=minimal_schedule,
        restore=not supports_backward,
    )
    loaded = {cuda_idx for _cpu, cuda_idx in schedule.values()}
    for idx, block in enumerate(blocks):
        if idx in loaded:
            attached.handles.append(
                block.register_forward_pre_hook(_make_wait_hook(offloader, idx))
            )
        if idx in schedule:
            attached.handles.append(
                block.register_forward_hook(
                    _make_submit_hook(offloader, blocks, *schedule[idx])
                )
            )
    if debug or minimal_schedule:
        print(
            f"blockswap: {len(schedule)} moves/forward for {blocks_to_swap} "
            f"swapped of {len(blocks)} blocks",
            flush=True,
        )

    return attached, blocks


def _make_wait_hook(offloader: ModelOffloader, idx: int):
    def hook(_module, _args):
        if _recomputing:
            return
        offloader.wait_for_block(idx)

    return hook


def _make_submit_hook(
    offloader: ModelOffloader, blocks: nn.ModuleList, cpu_idx: int, cuda_idx: int
):
    def hook(_module, _args, _output):
        if _recomputing:
            return
        offloader._submit_move_blocks(blocks, cpu_idx, cuda_idx)

    return hook
