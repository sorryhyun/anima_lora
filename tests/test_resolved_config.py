"""library/config/resolved.py — CLI/TOML/default precedence + snapshot rendering.

Pins the precedence rule the bespoke distill schemas share (a precedence bug is a
DX bug: ``ARGS=``/``PRESET=`` must mean the same thing across every loop) and the
provenance-snapshot rendering.
"""

from dataclasses import dataclass

import pytest

from library.config.resolved import (
    dataclass_snapshot_toml,
    pick,
)

CFG = {
    "data_dir": "from_toml",
    "optim": {"lr": 1e-4, "warmup": 0.02},
    "network": {"rank": 48},
}


@pytest.mark.parametrize("sentinel", [None, -1])
def test_pick_sentinel_falls_to_toml(sentinel):
    assert pick(sentinel, CFG, "optim.lr", 1.0) == 1e-4


def test_pick_missing_toml_key_uses_default():
    assert pick(None, CFG, "optim.nonexistent", 7) == 7
    assert pick(-1, CFG, "no.such.path", 3) == 3


def test_pick_real_zero_overrides_toml():
    # 0 / 0.0 are NOT sentinels — an explicit 0 must beat the TOML value.
    assert pick(0.0, CFG, "optim.lr", 1.0) == 0.0
    assert pick(0, CFG, "network.rank", 99) == 0


@dataclass(frozen=True)
class _Cfg:
    a: int
    b: str
    c: float
    opt: "str | None"


def test_snapshot_toml_reparses():
    # the rendered snapshot is valid TOML and recovers the (non-None) fields.
    import tomllib

    text = dataclass_snapshot_toml(_Cfg(1, "x", 2.5, "set"), title="t")
    assert tomllib.loads(text) == {"a": 1, "b": "x", "c": 2.5, "opt": "set"}
