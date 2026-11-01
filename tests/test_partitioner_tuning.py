"""Partitioner tuning + allocator default — the issue #58 VRAM levers.

The compile itself needs a GPU; what we pin here is the gating logic:
`_apply_partitioner_tuning` must set the functorch config flags exactly when
asked and never under gradient checkpointing (same repartitioning hazard as
`activation_memory_budget`, torch #166926), and `default_expandable_segments`
must never clobber a user-chosen PYTORCH_CUDA_ALLOC_CONF or ignore the
opt-out.
"""

import sys

import pytest

import torch._functorch.config as functorch_config

from library.runtime.allocator import default_expandable_segments
from library.runtime.harness import _apply_partitioner_tuning


@pytest.fixture
def restore_functorch_config():
    saved = (
        functorch_config.recompute_views,
        functorch_config.aggressive_recomputation,
    )
    yield
    functorch_config.recompute_views = saved[0]
    functorch_config.aggressive_recomputation = saved[1]


def test_tuning_sets_flags(restore_functorch_config):
    functorch_config.recompute_views = False
    functorch_config.aggressive_recomputation = False
    _apply_partitioner_tuning(
        recompute_views=True, aggressive_recomputation=True, grad_ckpt=False
    )
    assert functorch_config.recompute_views is True
    assert functorch_config.aggressive_recomputation is True


def test_tuning_skipped_under_grad_ckpt(restore_functorch_config):
    functorch_config.recompute_views = False
    functorch_config.aggressive_recomputation = False
    _apply_partitioner_tuning(
        recompute_views=True, aggressive_recomputation=True, grad_ckpt=True
    )
    assert functorch_config.recompute_views is False
    assert functorch_config.aggressive_recomputation is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only default")
def test_expandable_segments_respects_user_conf(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    monkeypatch.delenv("ANIMA_EXPANDABLE_SEGMENTS", raising=False)
    import os

    assert default_expandable_segments() is False
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
