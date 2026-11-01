"""The shipped MOSS CNS γ table: every row on disk, cfg picked by nearest."""

import numpy as np
import pytest

from library.moss.cns import GAMMA_PATHS, auto_gamma_path


@pytest.mark.parametrize("key", sorted(GAMMA_PATHS))
def test_gamma_row_matches_its_key(key):
    variant, cfg = key
    d = np.load(auto_gamma_path(variant, cfg))
    assert float(d["cfg"]) == cfg
    assert d["gamma"].ndim in (2, 3)


@pytest.mark.parametrize(
    "variant,cfg,want",
    [("base", 3.6, 4.0), ("sft", 4.4, 4.0)],
)
def test_auto_picks_nearest_cfg(variant, cfg, want):
    assert auto_gamma_path(variant, cfg) == auto_gamma_path(variant, want)
