"""Every kept run config loads, and every ``KEYS`` entry is used by one of
them (a key only archived configs used is dropped with its code)."""

import sys
import tomllib
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))


def _configs() -> list:
    return sorted((HOME / "configs").glob("*.toml"))


def test_configs_load():
    from reseed.config import load

    assert _configs()
    for f in _configs():
        run = load(f.stem)
        assert run.seed_rows().name == "trained.pt", f.stem


def test_keys_used():
    from reseed.config import KEYS

    used = set().union(
        *(tomllib.loads(f.read_text(encoding="utf-8")) for f in _configs())
    )
    assert set(KEYS) == used, f"unused {sorted(set(KEYS) - used)}"
