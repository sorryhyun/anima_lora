"""Invariants of the Qwen-Image-2.1 probe's block-swap schedule.

``project/qwen21_lora/src/blockswap.py`` drives ``ModelOffloader._submit_move_blocks``
from its own schedule instead of ``submit_move_blocks``, to avoid streaming the
whole model across PCIe every forward. The schedule is only correct if it keeps
three promises, all checked here by replaying it against a residency set: every
block is on the device when it runs, residency never exceeds what the layout
starts with, and the end state equals the start state so the next forward can
run without re-preparing.

Pure Python — no torch import, no GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "project" / "qwen21_lora" / "src"


def _load_swap_schedule():
    """Import the one function without importing the module's torch deps."""
    spec = importlib.util.spec_from_file_location(
        "_qwen21_blockswap", _SRC / "blockswap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # torch / library not installed
        pytest.skip(f"qwen21 blockswap not importable: {exc}")
    return module.swap_schedule


swap_schedule = _load_swap_schedule()


def replay(num_blocks: int, blocks_to_swap: int, minimal: bool):
    """Run one forward under the schedule; return (end state, peak residency)."""
    schedule = swap_schedule(num_blocks, blocks_to_swap, minimal=minimal)
    resident = set(range(num_blocks - blocks_to_swap))
    start = set(resident)
    peak = len(resident)

    for idx in range(num_blocks):
        assert idx in resident, (
            f"block {idx} runs while off the device "
            f"(N={num_blocks}, S={blocks_to_swap}, minimal={minimal})"
        )
        if idx in schedule:
            to_cpu, to_cuda = schedule[idx]
            # The offloader swaps a pair through one device allocation, so the
            # eviction and the load are one step, not two.
            resident.discard(to_cpu)
            resident.add(to_cuda)
            peak = max(peak, len(resident))

    return resident, start, peak, schedule


SHAPES = [(32, 0), (32, 1), (32, 4), (32, 8), (32, 16), (36, 4), (36, 18), (40, 12)]


@pytest.mark.parametrize("minimal", [True, False])
@pytest.mark.parametrize("num_blocks,blocks_to_swap", SHAPES)
def test_schedule_round_trips(num_blocks, blocks_to_swap, minimal):
    end, start, peak, _ = replay(num_blocks, blocks_to_swap, minimal)
    assert end == start, "forward must end in the layout prepare() set up"
    assert peak <= num_blocks - blocks_to_swap, "residency budget exceeded"


@pytest.mark.parametrize("num_blocks,blocks_to_swap", SHAPES)
def test_each_destination_submitted_once(num_blocks, blocks_to_swap):
    """The offloader keys one pending future per destination index."""
    _, _, _, schedule = replay(num_blocks, blocks_to_swap, minimal=True)
    destinations = [cuda for _cpu, cuda in schedule.values()]
    assert len(destinations) == len(set(destinations))


def test_minimal_is_2s_moves_below_half():
    """Below N/2 the schedule is 2S moves, against the ring's N."""
    assert len(swap_schedule(32, 4, minimal=True)) == 8
    assert len(swap_schedule(32, 4, minimal=False)) == 32
    assert len(swap_schedule(32, 16, minimal=True)) == 32  # 2S == N, no saving


def test_falls_back_to_ring_past_half():
    """Past N/2 the two phases would overlap, so the ring is used instead."""
    assert swap_schedule(32, 20, minimal=True) == swap_schedule(32, 20, minimal=False)


def test_no_swap_is_no_hooks():
    assert swap_schedule(32, 0, minimal=True) == {}
