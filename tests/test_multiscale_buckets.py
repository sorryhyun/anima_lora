"""Multi-scale tiers (``--target_res``) under free-fit.

The discrete constant-token bucket pool was removed; each tier now survives only
as a token-count band (``EDGE_TOKEN_BANDS``). These lock the invariants the
preprocess + compile paths still rely on:
  * tier assignment (``choose_edge``) never upscales when a closer tier exists
    (and the 2.25MP → 1536 case the feature was built for holds),
  * compile_blocks' dynamo budget scales with the active tier count.
"""

import pytest

from library.datasets.buckets import (
    ALLOWED_TARGET_RES,
    EDGE_TOKEN_BANDS,
    choose_edge,
    token_count_families,
)


@pytest.mark.core
def test_1024_band_frozen():
    # The 1024 tier stays at its natural (4032, 4200) — the frozen top-5 aspect
    # set (DCW_ASPECT_BUCKETS, consumed by CNS/mod-distill) is drawn from it.
    assert EDGE_TOKEN_BANDS[1024] == (4032, 4200)


@pytest.mark.parametrize(
    "w,h,target_res,expected",
    [
        (1500, 1500, [512, 768, 1024, 1280, 1536], 1536),  # 2.25MP — the ask
        # ~0.95MP near-square: closer to 1024 (tiny upscale) than 768 (big
        # downscale) — the case the nearest metric exists for.
        (1000, 950, [768, 1024], 1024),
        (300, 300, [512, 768, 1024], 512),  # tiny → least-bad (smallest) tier
        (4000, 4000, [512, 1024], 1024),  # huge → least downscale = largest tier
    ],
)
def test_choose_edge_nearest(w, h, target_res, expected):
    assert choose_edge(w, h, target_res) == expected


def test_compile_blocks_budget_scales_with_tiers():
    import torch._dynamo as _dynamo

    from tests.test_native_flatten import _tiny_anima

    model = _tiny_anima()
    _dynamo.config.cache_size_limit = 1
    # full menu → 9 token-count families → 2*9 + 8 = 26.
    model.compile_blocks(
        backend="eager",
        n_token_families=token_count_families(list(ALLOWED_TARGET_RES)),
    )
    assert _dynamo.config.cache_size_limit >= 2 * 9 + 8
