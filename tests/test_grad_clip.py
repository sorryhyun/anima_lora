import torch

from library.training.grad_clip import GradClipper, MicroAccumulator


def _grads(norm: float, seed: int, shapes=((8, 4), (16,))):
    g = torch.Generator().manual_seed(seed)
    out = [torch.randn(s, generator=g) for s in shapes]
    total = torch.nn.utils.get_total_norm(out)
    return [t * (norm / total) for t in out]


def test_median_leaves_typical_norms_and_caps_spike():
    clip = GradClipper(mode="median", max_norm=1.0, k=2.0, window=50, warmup=10)
    gen = torch.Generator().manual_seed(0)
    typical = (1.0 + 0.2 * torch.rand(60, generator=gen)).tolist()
    for i, n in enumerate(typical):
        grads = _grads(n, i)
        before = [t.clone() for t in grads]
        clip_norm, cap = clip.clip_(grads)
        assert abs(clip_norm.item() - n) < 1e-4
        if i >= 10:  # past warmup: cap = 2 x median, nothing typical exceeds it
            assert all(torch.equal(a, b) for a, b in zip(before, grads))

    median = torch.tensor(typical[-50:]).median().item()
    spike = _grads(12.8, 999)
    direction = [t / 12.8 for t in spike]
    _, cap = clip.clip_(spike)
    assert abs(cap.item() - 2.0 * median) < 1e-5
    after = torch.nn.utils.get_total_norm(spike).item()
    assert abs(after - 2.0 * median) < 1e-4
    cos = sum((a * b).sum() for a, b in zip(spike, direction)) / after
    assert cos.item() > 0.9999


def test_median_warmup_uses_max_grad_norm():
    clip = GradClipper(mode="median", max_norm=1.0, k=2.0, warmup=10)
    grads = _grads(5.0, 1)
    _, cap = clip.clip_(grads)
    assert cap.item() == 1.0
    assert abs(torch.nn.utils.get_total_norm(grads).item() - 1.0) < 1e-5


def test_fixed_matches_torch_clip_bit_for_bit():
    for n in (0.3, 4.0):
        ours = _grads(n, 3)
        ref = [torch.nn.Parameter(t.clone()) for t in ours]
        for p, t in zip(ref, ours):
            p.grad = t.clone()
        GradClipper(mode="fixed", max_norm=1.0).clip_(ours)
        torch.nn.utils.clip_grad_norm_(ref, 1.0)
        assert all(torch.equal(a, p.grad) for a, p in zip(ours, ref))


def _params():
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(16))]


def _set_grads(params, grads):
    for p, g in zip(params, grads):
        p.grad = g.clone()


def test_per_micro_at_accum_1_matches_global_clip():
    for n in (0.4, 6.0):
        params = _params()
        grads = _grads(n, 5)
        _set_grads(params, grads)
        MicroAccumulator(GradClipper(mode="fixed", max_norm=1.0), 1).absorb(
            params, last=True
        )
        ref = [g.clone() for g in grads]
        GradClipper(mode="fixed", max_norm=1.0).clip_(ref)
        assert all(torch.equal(p.grad, r) for p, r in zip(params, ref))


def test_per_micro_bounds_a_spike_to_its_share():
    # accelerator.backward divides each micro loss by accum, so a micro's p.grad
    # is its own gradient / accum; norms are compared in full-update units.
    accum = 2
    params = _params()
    small = _grads(1.0, 6)
    spike = _grads(10.0, 7)
    acc = MicroAccumulator(GradClipper(mode="fixed", max_norm=1.0), accum)

    _set_grads(params, [g / accum for g in small])
    norm, _ = acc.absorb(params, last=False)
    assert abs(norm.item() - 1.0) < 1e-4
    assert all(p.grad is None for p in params)

    _set_grads(params, [g / accum for g in spike])
    norm, _ = acc.absorb(params, last=True)
    assert abs(norm.item() - 10.0) < 1e-3

    expect = [(s + t / 10.0) / accum for s, t in zip(small, spike)]
    for p, e in zip(params, expect):
        assert torch.allclose(p.grad, e, atol=1e-5)
    # global clip after the mean would have let the spike set the direction
    mean = [(s + t) / accum for s, t in zip(small, spike)]
    cos_spike = sum((m * t).sum() for m, t in zip(mean, spike)) / (
        torch.nn.utils.get_total_norm(mean) * 10.0
    )
    cos_ours = sum((p.grad * t).sum() for p, t in zip(params, spike)) / (
        torch.nn.utils.get_total_norm([p.grad for p in params]) * 10.0
    )
    assert cos_ours < cos_spike


def test_per_micro_with_real_backward():
    accum = 2
    params = _params()
    x = [torch.randn(4), torch.randn(4)]
    acc = MicroAccumulator(GradClipper(mode="fixed", max_norm=0.0), accum)
    for i, xi in enumerate(x):
        loss = ((params[0] @ xi).sum() + params[1].sum() * (i + 1)) / accum
        loss.backward()
        acc.absorb(params, last=i == accum - 1)

    ref = _params()
    loss = sum(
        ((ref[0] @ xi).sum() + ref[1].sum() * (i + 1)) / accum for i, xi in enumerate(x)
    )
    loss.backward()
    for p, r in zip(params, ref):
        assert torch.allclose(p.grad, r.grad, atol=1e-6)
