"""Every block's ``wq_t`` on the host, streamed to the card for the backward."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

from library.runtime.offloading import HostMasters


class WqtStream:
    """Every block's ``wq_t`` in a page-locked host buffer, with two block-sized slots
    on the card: the block the backward is in and, loading on a worker thread, the one
    below it. The backward finds its copy through :meth:`get`, so the mover needs no
    hooks and the weights are not inputs of the compiled graph. A block reached out of
    order is copied in on the spot, at the cost of the wait.
    """

    def __init__(self, blocks: list[list[nn.Linear]]):
        shapes = [m.wq_t.shape for m in blocks[0]]
        device = blocks[0][0].wq_t.device
        offsets, total = [], 0
        for shape in shapes:
            total = -(-total // HostMasters._ALIGN) * HostMasters._ALIGN
            offsets.append(total)
            total += shape.numel()
        self.per_block = len(shapes)
        self.masters = HostMasters()
        self.host: list[torch.Tensor] = []
        for b, mods in enumerate(blocks):
            if [m.wq_t.shape for m in mods] != shapes:
                raise ValueError(f"block {b}: wq_t shapes differ from block 0's")
            flat = self.masters._alloc(total)
            for j, mod in enumerate(mods):
                self._view(flat, offsets[j], shapes[j]).copy_(mod.wq_t)
                del mod.held_wq_t
                mod.register_buffer(
                    "sid", torch.tensor([b * self.per_block + j]), persistent=False
                )
                mod.int8_mode = "stream"
            self.host.append(flat)
        self.slots = [
            torch.empty(total, dtype=torch.uint8, device=device) for _ in "ab"
        ]
        self.views = [
            [self._view(slot, o, sh) for o, sh in zip(offsets, shapes)]
            for slot in self.slots
        ]
        self.copy_stream = torch.Stream(device=device.type)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.loading = -1
        self.current = -1
        self.active = 0
        self.entered = 0
        self.cold = 0
        self.wait_s = 0.0

    @staticmethod
    def _view(flat: torch.Tensor, off: int, shape: torch.Size) -> torch.Tensor:
        return flat[off : off + shape.numel()].view(torch.int8).view(shape)

    def _copy(self, block: int, slot: int) -> None:
        # the slot's last user may still be queued on the default stream
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(self.copy_stream):
            self.slots[slot].copy_(self.host[block], non_blocking=True)
        self.copy_stream.synchronize()

    def _enter(self, block: int) -> None:
        start = time.perf_counter()
        self.entered += 1
        if self.pending is not None:
            self.pending.result()
            self.pending = None
        free = 1 - self.active
        if block != self.loading:
            self.cold += 1
            self._copy(block, free)
        self.wait_s += time.perf_counter() - start
        self.current, self.active, self.loading = block, free, -1
        if block > 0:
            self.loading = block - 1
            self.pending = self.pool.submit(self._copy, block - 1, 1 - free)

    def close(self) -> None:
        """Unregister the host buffers, or CUDA keeps their old sizes for the addresses."""
        self.pool.shutdown(wait=True)
        self.pending = None
        self.masters.release()

    def get(self, sid: int) -> torch.Tensor:
        block, j = divmod(sid, self.per_block)
        if block != self.current:
            self._enter(block)
        return self.views[self.active][j]
