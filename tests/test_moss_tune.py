import pytest

from library.moss.requests import TuneSwapRequest
from library.moss.train import budget_for_free, step_reserve_gb
from library.moss.tune import flags, pick


def test_budget_for_free_inverts_reserve():
    joint = 4396
    for free in (5.0, 7.24, 9.64, 13.0):
        budget = budget_for_free(free, joint)
        assert step_reserve_gb(joint, grad_checkpointing=False, budget=budget) <= free
        if budget < 1.0:
            over = round(budget + 0.05, 6)
            assert step_reserve_gb(joint, grad_checkpointing=False, budget=over) > free


def test_budget_for_free_measured_points():
    # 2026-10-17 arms at 4096 + 300 tokens: swap 24 left 9.64 GB free before
    # training and ran budget 0.2 with 2.26 GB to spare.
    assert budget_for_free(9.64, 4396) == pytest.approx(0.2)
    assert budget_for_free(3.0, 4396) == 0.0
    assert budget_for_free(100.0, 4396) == 1.0


def test_pick_respects_margin():
    arms = [
        {
            "blocks_to_swap": 0,
            "grad_checkpointing": True,
            "budget": 1.0,
            "s_per_step": 2.09,
            "free_gb": 1.6,
        },
        {
            "blocks_to_swap": 24,
            "grad_checkpointing": False,
            "budget": 0.2,
            "s_per_step": 1.78,
            "free_gb": 2.26,
        },
        {
            "blocks_to_swap": 28,
            "grad_checkpointing": False,
            "budget": 0.25,
            "s_per_step": 1.73,
            "free_gb": 0.6,
        },
        {"blocks_to_swap": 4, "grad_checkpointing": False, "skipped": "does not fit"},
    ]
    chosen = pick(arms, margin_gb=1.0)
    assert chosen["blocks_to_swap"] == 24
    # 28 is faster but leaves too little; with room, 1.73 vs 1.78 is not a tie.
    arms[2]["free_gb"] = 2.0
    assert pick(arms, margin_gb=1.0)["blocks_to_swap"] == 28
    assert pick(arms, margin_gb=1.0, tolerance=0.05)["blocks_to_swap"] == 24
    arms[2]["free_gb"] = 0.6
    bf16 = "--int8 off --stream bf16 --no-ema_card"
    assert flags(chosen) == f"{bf16} --blocks_to_swap 24 --activation_memory_budget 0.2"
    assert flags(arms[0]) == f"{bf16} --grad_checkpointing --blocks_to_swap 0"
    assert pick(arms, margin_gb=5.0) is None


def test_request_round_trip():
    req = TuneSwapRequest(sizes="1024x1024:300", swap_grid="16,24", reference=False)
    assert TuneSwapRequest.from_argv(req.to_argv()) == req
