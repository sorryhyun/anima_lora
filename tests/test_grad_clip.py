import pytest
import torch

from library.training.grad_clip import GradClipper, GradSigmaLog, MicroAccumulator


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


def test_sigma_log_pairs_micros_with_their_norm(tmp_path):
    import json

    log = GradSigmaLog(tmp_path / "run.grad_sigma.jsonl", flush_every=2)
    log.add_micro(torch.tensor([0.3]), torch.tensor(0.1), ["a"])
    log.add_norm(1, torch.tensor(2.0), torch.tensor(3.0))
    log.add_micro(torch.tensor([0.9]), torch.tensor(0.2), ["b"])
    log.add_micro(torch.tensor([0.5]), torch.tensor(0.4), ["c"])
    log.add_norm(2, torch.tensor(5.0), None)
    rows = [
        json.loads(x)
        for x in (tmp_path / "run.grad_sigma.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["sigma"] == [0.30000001192092896] and rows[0]["keys"] == ["a"]
    assert rows[0]["cap"] == 3.0 and rows[1]["cap"] is None
    assert rows[1]["keys"] == ["b", "c"] and len(rows[1]["loss"]) == 2


def test_sigma_bands_cap_each_draw_against_its_band():
    # U-shaped norms: mid σ ~1, edge σ ~4. A global median clip would cut every
    # edge draw; per band, typical edge draws pass and an edge spike is capped
    # at k x the edge median.
    clip = GradClipper(mode="median", max_norm=1.0, k=2.0, sigma_edges=[0.2, 0.8])
    gen = torch.Generator().manual_seed(0)
    for i in range(40):
        edge = i % 4 == 0
        n = (4.0 if edge else 1.0) * (1.0 + 0.1 * torch.rand(1, generator=gen).item())
        sigma = torch.tensor([0.9 if edge else 0.5])
        grads = _grads(n, i)
        _, cap = clip.clip_(grads, sigma=sigma)
        if i >= 20:  # both bands past band_warmup
            assert torch.nn.utils.get_total_norm(grads).item() == pytest.approx(
                n, rel=1e-5
            )
            assert cap.item() > (6.0 if edge else 1.5)
    spike = _grads(40.0, 999)
    edge_med = clip._bands[2].nanmedian().item()
    _, cap = clip.clip_(spike, sigma=torch.tensor([0.95]))
    assert cap.item() == pytest.approx(2.0 * edge_med, rel=1e-5)
    assert torch.nn.utils.get_total_norm(spike).item() == pytest.approx(
        2.0 * edge_med, rel=1e-4
    )


def test_sigma_band_falls_back_to_global_cap_until_filled():
    clip = GradClipper(mode="median", max_norm=1.0, k=2.0, sigma_edges=[0.5])
    for i in range(12):
        clip.clip_(_grads(1.0, i), sigma=torch.tensor([0.3]))
    # high band empty: global cap 2 x median(1.0)
    _, cap = clip.clip_(_grads(5.0, 50), sigma=torch.tensor([0.7]))
    assert cap.item() == pytest.approx(2.0, rel=1e-5)
    assert int(clip._band_n[1]) == 1


def test_sigma_bands_off_matches_plain_median():
    a = GradClipper(mode="median", max_norm=1.0, k=2.0)
    b = GradClipper(mode="median", max_norm=1.0, k=2.0, sigma_edges=[0.3])
    for i in range(30):
        ga, gb = _grads(1.0 + i % 7, i), _grads(1.0 + i % 7, i)
        a.clip_(ga)
        b.clip_(gb)  # no σ given: global path
        assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_sigma_bands_need_median_mode():
    with pytest.raises(ValueError):
        GradClipper(mode="fixed", sigma_edges=[0.5])
    with pytest.raises(ValueError):
        GradClipper(mode="median", sigma_edges=[1.2])
