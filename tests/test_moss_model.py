"""MOSS-Image vendored DiT + sampler invariants, on a tiny random model (CPU)."""

import pytest
import torch

from library.moss.generate import denoise
from library.moss.model import (
    MOSSImageScheduler,
    MOSSImageTransformer,
    RefCache,
    snapped_sigmas,
)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = MOSSImageTransformer(
        dim=96, ffn_dim=128, num_heads=4, num_layers=4, text_dim=32, latent_channels=8
    ).eval()
    # The zero-init-free random weights make every block contribute.
    return model


def _inputs(batch=2, s=7):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 8, 4, 6, generator=g)
    e = torch.randn(batch, s, 32, generator=g)
    mask = torch.ones(batch, s, dtype=torch.bool)
    mask[-1, 3:] = False
    return x, e, mask


@pytest.mark.core
@torch.no_grad()
def test_batched_matches_reference(tiny):
    x, e, mask = _inputs()
    ref = tiny(x, 500.0, e, mask).sample
    assert torch.allclose(tiny.forward_batched(x, 500.0, e, mask), ref, atol=1e-5)


@pytest.mark.core
@torch.no_grad()
def test_batched_per_sample_skip(tiny):
    """A batched CFG pair, uncond skipping a block, matches two reference forwards."""
    x, e, mask = _inputs()
    r0 = tiny(x[:1], 500.0, e[:1], mask[:1]).sample
    r1 = tiny(x[1:], 500.0, e[1:], mask[1:], skip_layers=[2]).sample
    out = tiny.forward_batched(x, 500.0, e, mask, skip_layers=[(), (2,)])
    assert torch.allclose(out, torch.cat([r0, r1]), atol=1e-5)


@torch.no_grad()
def test_features_then_head_is_forward(tiny):
    x, e, mask = _inputs()
    feats = tiny(x, 250.0, e, mask, return_features=True)
    _, head = tiny.time_conditioning(250.0, x.shape[0])
    out = torch.cat(
        [tiny.head(feats[b : b + 1], head[b : b + 1], 4, 6) for b in range(2)]
    )
    assert torch.equal(out, tiny(x, 250.0, e, mask).sample)


def _ref(batch=2):
    return torch.randn(batch, 8, 2, 3, generator=torch.Generator().manual_seed(2))


@pytest.fixture
def ref_t(tiny):
    """Sets ``tiny.ref_timestep`` for one test and restores the shared default."""

    def set_(value):
        tiny.ref_timestep = value

    yield set_
    tiny.ref_timestep = None


@torch.no_grad()
def test_ref_timestep_at_image_t_is_shared(tiny, ref_t):
    x, e, mask = _inputs()
    shared = tiny(x, 500.0, e, mask, ref=_ref()).sample
    ref_t(500.0)
    assert torch.allclose(tiny(x, 500.0, e, mask, ref=_ref()).sample, shared, atol=1e-6)


@torch.no_grad()
def test_ref_timestep_zero_batched_matches_reference(tiny, ref_t):
    x, e, mask = _inputs()
    shared = tiny(x, 500.0, e, mask, ref=_ref()).sample
    ref_t(0.0)
    single = tiny(x, 500.0, e, mask, ref=_ref()).sample
    assert not torch.allclose(single, shared, atol=1e-3)
    batched = tiny.forward_batched(x, 500.0, e, mask, ref=_ref())
    assert torch.allclose(batched, single, atol=1e-5)
    r1 = tiny(x[1:], 500.0, e[1:], mask[1:], skip_layers=[2], ref=_ref()[1:]).sample
    out = tiny.forward_batched(x, 500.0, e, mask, skip_layers=[(), (2,)], ref=_ref())
    assert torch.allclose(out, torch.cat([single[:1], r1]), atol=1e-5)


@pytest.fixture
def ref_iso(tiny):
    tiny.ref_isolate = True
    yield
    tiny.ref_isolate = False


@pytest.mark.core
@torch.no_grad()
def test_ref_isolate_batched_matches_reference(tiny, ref_t, ref_iso):
    x, e, mask = _inputs()
    ref_t(0.0)
    single = tiny(x, 500.0, e, mask, ref=_ref()).sample
    tiny.ref_isolate = False
    assert not torch.allclose(tiny(x, 500.0, e, mask, ref=_ref()).sample, single)
    tiny.ref_isolate = True
    batched = tiny.forward_batched(x, 500.0, e, mask, ref=_ref())
    assert torch.allclose(batched, single, atol=1e-5)
    r1 = tiny(x[1:], 500.0, e[1:], mask[1:], skip_layers=[2], ref=_ref()[1:]).sample
    out = tiny.forward_batched(x, 500.0, e, mask, skip_layers=[(), (2,)], ref=_ref())
    assert torch.allclose(out, torch.cat([single[:1], r1]), atol=1e-5)


@torch.no_grad()
def test_ref_isolate_skip_spares_the_reference(tiny, ref_t, ref_iso):
    """A skipped block still runs the isolated reference, so the skip-22 uncond
    reads the cond branch's reference; ``ref_follows_skip`` skips it too."""
    ref_t(0.0)
    x, e, _ = _inputs(batch=1)
    tokens, rope = tiny.embed_sample(x, e, _ref(1))
    mod, _ = tiny.time_conditioning(500.0, 1)
    mod, n_ref = tiny.ref_modulation(mod, (2, 3))

    def stream(skip):
        out = tiny.run_layers(tokens, rope, mod, skip, n_img=24, n_ref=n_ref)
        return out[:, -n_ref:]

    full = stream(frozenset())
    assert torch.allclose(stream(frozenset({2})), full, atol=1e-6)
    assert torch.allclose(stream([frozenset({2})]), full, atol=1e-6)
    tiny.ref_follows_skip = True
    try:
        assert not torch.allclose(stream(frozenset({2})), full, atol=1e-3)
    finally:
        tiny.ref_follows_skip = False


@pytest.mark.core
@torch.no_grad()
def test_ref_cache_matches_uncached(tiny, ref_t, ref_iso):
    """One reference build serves cond, the skip uncond and later steps, per-sample
    and batched, and gives what the full sequence gives."""
    ref_t(0.0)
    ref = _ref(1)
    calls = []
    for seed, t, s in ((1, 500.0, 7), (5, 900.0, 4)):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 8, 4, 6, generator=g)
        e = torch.randn(1, s, 32, generator=g)
        calls += [(x, t, e, None), (x, t, e, [2])]
    full = [tiny(x, t, e, skip_layers=k, ref=ref).sample for x, t, e, k in calls]
    tiny.ref_cache = cache = RefCache()
    try:
        cached = [tiny(x, t, e, skip_layers=k, ref=ref).sample for x, t, e, k in calls]
        x, t, e, _ = calls[0]
        batched = tiny.forward_batched(
            x.repeat(2, 1, 1, 1),
            t,
            e.repeat(2, 1, 1),
            skip_layers=[(), (2,)],
            ref=ref.repeat(2, 1, 1, 1),
        )
    finally:
        tiny.ref_cache = None
    for a, b in zip(cached, full):
        assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(batched, torch.cat(full[:2]), atol=1e-5)
    assert cache.builds == 2  # the per-sample reference, then the batch of two


@torch.no_grad()
def test_ref_timestep_ignored_without_ref(tiny, ref_t):
    x, e, mask = _inputs()
    plain = tiny(x, 500.0, e, mask).sample
    ref_t(0.0)
    assert torch.equal(tiny(x, 500.0, e, mask).sample, plain)


@pytest.mark.core
@pytest.mark.parametrize("steps", [1, 50])
def test_snapped_sigmas_match_scheduler(steps):
    sched = MOSSImageScheduler()
    sched.set_timesteps(steps)
    assert torch.equal(sched.sigmas[:-1], snapped_sigmas(steps).float())
    assert sched.sigmas[-1] == 0


@pytest.mark.core
@torch.no_grad()
@pytest.mark.parametrize("cfg,skip", [(3.0, (2,)), (3.0, ()), (1.0, ())])
def test_denoise_matches_scheduler_loop(tiny, cfg, skip):
    """Our loop vs ``MOSSImagePipeline.__call__``'s, written against the scheduler."""
    x, e, mask = _inputs(batch=1)
    _, ne, nmask = _inputs(batch=2, s=5)
    ne, nmask = ne[1:], nmask[1:]
    sched = MOSSImageScheduler()
    sched.set_timesteps(6)
    ref = x.clone()
    for t in sched.timesteps:
        pred = tiny(ref, t, e, mask).sample.float()
        if cfg != 1.0:
            un = tiny(ref, t, ne, nmask, skip_layers=skip).sample.float()
            pred = un + cfg * (pred - un)
        ref = sched.step(pred, t, ref, return_dict=False)[0]
    ours = denoise(
        tiny, x, e, mask, ne, nmask, snapped_sigmas(6), cfg=cfg, uncond_skip=skip
    )
    assert torch.equal(ours, ref)
    batched = denoise(
        tiny,
        x,
        e,
        mask,
        ne,
        nmask,
        snapped_sigmas(6),
        cfg=cfg,
        uncond_skip=skip,
        batched=True,
    )
    assert torch.allclose(batched, ref, atol=1e-4)


@torch.no_grad()
def test_denoise_spectrum_protocol(tiny):
    """A forecaster that replays the last observed features is called in order."""
    x, e, mask = _inputs(batch=1)
    calls = []

    class Hold:
        def begin(self, sigmas):
            calls.append(("begin", len(sigmas)))

        def is_full_step(self, i):
            return i % 2 == 0

        def observe(self, i, features):
            calls.append(("observe", i, tuple(features.shape)))
            self.last = features

        def forecast(self, i):
            calls.append(("forecast", i))
            return self.last

    denoise(
        tiny,
        x,
        e,
        mask,
        e,
        mask,
        snapped_sigmas(4),
        cfg=2.0,
        uncond_skip=(1,),
        spectrum=Hold(),
    )
    assert calls[0] == ("begin", 4)
    assert [c[0] for c in calls[1:]] == ["observe", "forecast", "observe", "forecast"]
    assert calls[1][2] == (2, 24, 96)


class _ExactX0:
    """Stub DiT whose velocity is the exact ``ε − x₀`` for a fixed ``x₀``."""

    def __init__(self, x0):
        self.x0 = x0

    def __call__(self, latents, t, e, m, skip_layers=(), return_features=True):
        return latents.clone()

    def time_conditioning(self, t, batch):
        return None, t / 1000

    def head(self, features, sigma, h, w):
        return (features - self.x0) / sigma.clamp_min(1e-4)


@pytest.mark.core
@torch.no_grad()
@pytest.mark.parametrize("sampler", ["euler", "er_sde"])
def test_denoise_samplers_land_on_exact_x0(sampler):
    """With the exact velocity, both samplers integrate to x₀ (er_sde's injected
    noise cancels against the drift on an exact model)."""
    g = torch.Generator().manual_seed(2)
    x0, noise = torch.randn(2, 1, 8, 4, 6, generator=g)
    e, mask = torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)
    z = denoise(
        _ExactX0(x0),
        noise,
        e,
        mask,
        e,
        mask,
        snapped_sigmas(20),
        cfg=3.0,
        sampler=sampler,
        seed=0,
    )
    assert torch.allclose(z, x0, atol=1e-5)
