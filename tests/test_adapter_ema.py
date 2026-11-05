import torch

from library.training.ema import AdapterEMA, ema_decay


def _params():
    return [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2, 2))]


def test_decay_from_span():
    assert abs(ema_decay(0.1, 848) - (1 - 1 / 84.8)) < 1e-9
    assert ema_decay(0.1, 5) == 0.0


def test_starts_at_warmup_end_and_seeds():
    params = _params()
    ema = AdapterEMA(params, 0.5, "", updates=10, start=3, on_card=True)
    for v in (1.0, 2.0):
        with torch.no_grad():
            for p in params:
                p.fill_(v)
        ema.step()
    assert not ema.started
    with torch.no_grad():
        for p in params:
            p.fill_(3.0)
    ema.step()  # update 3 seeds
    assert ema.started and torch.equal(ema.shadows[0][0], torch.full((3,), 3.0))
    with torch.no_grad():
        for p in params:
            p.fill_(13.0)
    ema.step()  # decay 0.8 at span 0.5 of 10
    assert torch.allclose(ema.shadows[0][0], torch.full((3,), 5.0))


def test_swapped_in_restores_live_weights():
    params = _params()
    ema = AdapterEMA(params, 0.5, "0.2", updates=10, start=1, on_card=False)
    ema.step()
    with torch.no_grad():
        for p in params:
            p.fill_(10.0)
    ema.step()
    with ema.swapped_in(0) as on:
        assert on and torch.allclose(params[0], torch.full((3,), 2.0))
    assert torch.equal(params[0], torch.full((3,), 10.0))
    assert ema.main == 0 and ema.spans == [0.5, 0.2]


def test_extras_only_has_no_main():
    ema = AdapterEMA(_params(), 0.0, "0.05,0.3", updates=100, start=1)
    assert ema.main is None and ema.spans == [0.05, 0.3]
