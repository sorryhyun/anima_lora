"""Invariants for the DMAD Phase −1 probe (docs/proposal/turbo_dmad.md § Phase −1).

* ``alignment_stats`` — cosine / agree-energy endpoints, the permutation null, and
  that it draws only from the generator it is handed (the probe must leave the
  training RNG stream untouched).
* config — ``dmad_probe`` default-off, TOML/CLI precedence, and its guards.
* ``bench/turbo/dmad_probe_read.py::read`` — the pre-registered verdict on
  synthetic rows.
"""

from __future__ import annotations

import random

import pytest
import torch

from bench.turbo.dmad_probe_read import read
from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.dmad_probe import alignment_stats


def _resolve(cli: list[str] | None = None, cfg: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, cfg or {})


# --- alignment_stats --------------------------------------------------------


def test_identical_is_fully_aligned():
    d = torch.randn(1, 16, 32, 32)
    s = alignment_stats(d.clone(), d, torch.Generator().manual_seed(0))
    assert s["cos"] == pytest.approx(1.0, abs=1e-6)
    assert s["agree"] == pytest.approx(1.0, abs=1e-6)


def test_negated_is_fully_opposed():
    d = torch.randn(1, 16, 32, 32)
    s = alignment_stats(-d, d, torch.Generator().manual_seed(0))
    assert s["cos"] == pytest.approx(-1.0, abs=1e-6)
    assert s["agree"] == pytest.approx(0.0, abs=1e-6)


def test_null_sits_at_chance():
    d = torch.randn(1, 16, 64, 64)
    s = alignment_stats(d.clone(), d, torch.Generator().manual_seed(0))
    assert abs(s["cos_null"]) < 0.02
    assert s["agree_null"] == pytest.approx(0.5, abs=0.02)


def test_uses_only_the_given_generator():
    d = torch.randn(1, 16, 16, 16)
    g = torch.randn(1, 16, 16, 16)
    before = torch.get_rng_state()
    alignment_stats(g, d, torch.Generator().manual_seed(1))
    assert torch.equal(before, torch.get_rng_state())


def test_null_is_reproducible_from_the_generator():
    d = torch.randn(1, 16, 16, 16)
    g = torch.randn(1, 16, 16, 16)
    a = alignment_stats(g, d, torch.Generator().manual_seed(3))
    b = alignment_stats(g, d, torch.Generator().manual_seed(3))
    assert a == b


# --- config -------------------------------------------------------------------


def test_off_by_default():
    assert _resolve().dmad_probe is False


def test_toml_and_cli_enable():
    assert _resolve(cfg={"dmad_probe": {"enabled": True}}).dmad_probe is True
    c = _resolve(["--dmad_probe", "--dmad_probe_lr", "1e-4"])
    assert c.dmad_probe is True and c.dmad_probe_lr == pytest.approx(1e-4)


def test_refuses_plain_dmd():
    with pytest.raises(ValueError, match="dpdmd"):
        _resolve(["--dmad_probe", "--base_loss", "dmd", "--student_steps", "4"])


def test_refuses_block_swap():
    with pytest.raises(ValueError, match="blocks_to_swap"):
        _resolve(["--dmad_probe", "--blocks_to_swap", "4"])


def test_refuses_unknown_head():
    with pytest.raises(ValueError, match="head"):
        _resolve(cfg={"dmad_probe": {"enabled": True, "head": "conv"}})


# --- pre-registered read ------------------------------------------------------


def _rows(n, *, cos, ceil, acc, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append(
            {
                "step": i + 1,
                "tau_dm": rng.random(),
                "grad_step": rng.randint(1, 3),
                "acc": acc,
                "margin": 1.0,
                "bce": 1.0,
                "cos": cos + rng.gauss(0, 0.01),
                "cos_null": rng.gauss(0, 0.002),
                "agree": 0.5 + cos / 2 + rng.gauss(0, 0.01),
                "agree_null": 0.5 + rng.gauss(0, 0.002),
                "ceil_cos": ceil + rng.gauss(0, 0.01),
                "ceil_agree": 0.6,
                "cos_dm2": cos * ceil,
            }
        )
    return rows


def test_read_pass():
    assert read(_rows(200, cos=0.3, ceil=0.4, acc=0.9))["verdict"] == "PASS"


def test_read_kill():
    assert read(_rows(200, cos=0.0, ceil=0.4, acc=0.9))["verdict"] == "KILL"


def test_read_weak():
    assert read(_rows(200, cos=0.05, ceil=0.4, acc=0.9))["verdict"] == "WEAK"


def test_read_unconverged_wins():
    assert read(_rows(200, cos=0.3, ceil=0.4, acc=0.6))["verdict"] == "UNCONVERGED"


def test_read_window_is_second_half():
    rows = _rows(100, cos=0.0, ceil=0.4, acc=0.5) + [
        {**r, "step": r["step"] + 100} for r in _rows(100, cos=0.3, ceil=0.4, acc=0.9)
    ]
    res = read(rows)
    assert res["window_steps"] == [101, 200]
    assert res["verdict"] == "PASS"
