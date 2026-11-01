"""EasyControl dynamic-seq cond band — the cond stream is NOT in the target band.

Regression for the ConstraintViolationError raised at the first training step of a
``cond_res_scale < 1`` run under ``torch_compile``: the patched two-stream forward
marked BOTH seq axes with the target's token band, but Position-Aware Interpolation
downscales the cond grid, so its token count lands near ``scale²`` of the target's
(e.g. 1014 vs a (2940, 4200) band).
"""

import pytest

from networks.methods.easycontrol import create_network


def _make_network(**kwargs):
    return create_network(1.0, 32, 32, None, [], None, **kwargs)


@pytest.mark.parametrize("scale", [0.5])
def test_cond_band_tracks_cond_res_scale(scale):
    net = _make_network(cond_res_scale=scale)
    target = (2940, 4200)
    lo, hi = net._cond_seq_range(target)
    # The band must contain the whole s²-scaled target band...
    assert lo <= target[0] * scale * scale
    assert hi >= target[1] * scale * scale
    # ...and must not just be the target band back (that is the bug).
    assert lo < target[0]


def test_fit_widens_for_an_out_of_band_cond_shape():
    # cond≠target subsets pair a distinct image that can free-fit to another tier.
    net = _make_network(cond_res_scale=1.0)
    net._dynamic_seq_cond_range = (2940, 4200)
    net._fit_cond_seq_range(4032)  # in band -> untouched
    assert net._dynamic_seq_cond_range == (2940, 4200)
    net._fit_cond_seq_range(1008)  # 512-tier cond -> widen, no raise
    lo, hi = net._dynamic_seq_cond_range
    assert lo <= 1008 and hi >= 4200
