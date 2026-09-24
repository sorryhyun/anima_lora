"""GH #103: auto-resume skipped every epoch when gradient_accumulation_steps > 1.

The old code divided a ga-multiplied batch count by an optimizer-steps-per-epoch
denominator, over-skipping by ga². These pin the unit-correct split.
"""

import pytest

from library.training.checkpoints import resume_skip_plan


def test_issue_103_numbers():
    # 10 batches/epoch, ga=5 → 2 steps/epoch; 600 epochs done = step 1200.
    assert resume_skip_plan(1200, num_batches=10, gradient_accumulation_steps=5) == (
        600,
        0,
    )


def test_ga_one_is_unchanged():
    assert resume_skip_plan(1200, num_batches=10, gradient_accumulation_steps=1) == (
        120,
        0,
    )


def test_residual_inside_epoch_is_in_batches():
    # 20 batches/epoch, ga=4 → 5 steps/epoch; step 12 = 2 epochs + 2 steps = 8 batches.
    assert resume_skip_plan(12, num_batches=20, gradient_accumulation_steps=4) == (2, 8)


def test_partial_last_step_per_epoch():
    # 11 batches/epoch, ga=5 → accelerate syncs on the last batch → 3 steps/epoch.
    # 600 epochs = 1800 steps, no residual.
    assert resume_skip_plan(1800, num_batches=11, gradient_accumulation_steps=5) == (
        600,
        0,
    )
    # One full step into epoch 601 = 5 batches.
    assert resume_skip_plan(1801, num_batches=11, gradient_accumulation_steps=5) == (
        600,
        5,
    )


@pytest.mark.parametrize("ga", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("num_batches", [1, 7, 10, 11, 64])
def test_epoch_to_start_never_exceeds_steps_taken(ga, num_batches):
    import math

    steps_per_epoch = math.ceil(num_batches / ga)
    for resume_step in range(0, 5 * steps_per_epoch + 1):
        epoch, skip = resume_skip_plan(resume_step, num_batches, ga)
        assert epoch * steps_per_epoch + skip // ga == resume_step
        assert skip < num_batches
