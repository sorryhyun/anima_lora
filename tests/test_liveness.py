"""Liveness accounting (issues.md P1.1): configured-but-dead must be loud.

The inertness tests prove "flag off ⇒ identical training"; these prove the
counterpart — a skip-if-missing aux loss that is configured ON but never
consumes its aux input is counted by the composer's ``LivenessLedger`` and
flagged by ``audit()`` with the greppable ``LIVENESS:`` prefix. No forward
pass; the composer runs on tiny CPU tensors.
"""

from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace

import pytest
import torch

from library.training.losses import (
    _LIVENESS_PROBES,
    LOSS_REGISTRY,
    LivenessLedger,
    LossContext,
    build_loss_composer,
)
from library.training.progress import ProgressSink, run_scope


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        method="lora",
        loss_type="l2",
        masked_loss=False,
        multiscale_loss_weight=0.0,
        functional_loss_weight=0.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _net(**attrs) -> SimpleNamespace:
    defaults = {"_balance_loss_weight": 0.0}
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def _ctx(args, network, aux=None, is_train=True) -> LossContext:
    pred = torch.randn(2, 4, 8, 8)
    return LossContext(
        args=args,
        batch={},
        model_pred=pred,
        target=torch.randn_like(pred),
        timesteps=torch.rand(2),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(2),
        network=network,
        aux=aux or {},
        is_train=is_train,
    )


def test_every_probe_targets_a_registered_loss():
    assert set(_LIVENESS_PROBES) <= set(LOSS_REGISTRY)


def test_composer_records_consumed_and_skipped_batches():
    args = _make_args()
    net = _net(_repa_weight=0.05)
    ledger = LivenessLedger()
    composer = build_loss_composer(args, net, ledger=ledger)
    assert "repa" in composer.active_losses

    composer.compose(_ctx(args, net, aux={"repa": torch.tensor(0.3)}))
    composer.compose(_ctx(args, net, aux={}))  # dead-dispatch step

    assert ledger.seen["repa"] == 2
    assert ledger.live["repa"] == 1
    assert ledger.metrics(None) == {"liveness/repa": 0.5}


def test_dict_shaped_aux_requires_inner_tensor():
    # vr aux is a dict — the probe must mirror the handler's inner gate,
    # not just key presence.
    assert _LIVENESS_PROBES["flow_matching_vr"]({"vr": {"state": {}}}) is False
    assert _LIVENESS_PROBES["flow_matching_vr"]({"vr": {"z": torch.zeros(1)}}) is True


@pytest.mark.core
def test_audit_dead_feature_errors_with_greppable_prefix(caplog):
    ledger = LivenessLedger()
    for _ in range(10):
        ledger.record("repa", False)
    with caplog.at_level(logging.ERROR, logger="library.training.losses"):
        dead = ledger.audit(where="step 25 early check")
    assert dead == ["repa"]
    assert any(
        r.levelno == logging.ERROR and r.getMessage().startswith("LIVENESS:")
        for r in caplog.records
    )


def test_run_scope_extra_fields_failure_cannot_mask_exception(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=1, total_epochs=1, pid=1)

    def _boom() -> dict:
        raise ValueError("extra_fields bug")

    with pytest.raises(RuntimeError, match="real failure"):
        with run_scope(sink, final_step=lambda: 0, extra_fields=_boom):
            raise RuntimeError("real failure")
    with open(path, encoding="utf-8") as f:
        end = [json.loads(line) for line in f if line.strip()][-1]
    assert end["ev"] == "run_end" and end["status"] == "error"
