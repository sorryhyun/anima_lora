"""MOSS Spectrum: window schedule and the forecaster on toy trajectories (CPU)."""

import torch

from library.moss.spectrum import MossSpectrum, window_schedule


def test_window_schedule_matches_anima_counts():
    full = window_schedule(50, flex_window=0.25)  # Anima's default
    assert all(full[:6]) and all(full[-3:])
    assert sum(full) == 22
    assert sum(window_schedule(30, flex_window=0.25)) == 17
    assert sum(window_schedule(20, flex_window=1.0)) == 12


def _run(sp, sigmas, traj):
    sp.begin(sigmas)
    errs = []
    for i in range(sp.num_steps):
        if sp.is_full_step(i):
            sp.observe(i, traj(i))
        else:
            errs.append((sp.forecast(i).float() - traj(i).float()).abs().max().item())
    return errs


def test_forecast_polynomial_trajectory():
    torch.manual_seed(0)
    a, b, c = (torch.randn(2, 16, 8) for _ in range(3))
    n = 30

    def traj(i):
        t = i / n
        return (a + b * t + c * t * t).to(torch.bfloat16)

    sp = MossSpectrum(w=1.0, lam=1e-4)
    errs = _run(sp, torch.linspace(1, 0, n + 1), traj)
    assert errs and sp.forecast_steps == len(errs)
    assert max(errs) < 0.05


def test_rows_are_independent():
    """Cond and uncond rows stacked in one predictor forecast as if alone."""
    torch.manual_seed(1)
    n = 20
    coef = torch.randn(3, 2, 4, 5)

    def traj(i, rows=slice(None)):
        t = i / n
        return (coef[0] + coef[1] * t + coef[2] * t**3)[rows].to(torch.bfloat16)

    sig = torch.linspace(1, 0, n + 1)
    both, solo = MossSpectrum(), MossSpectrum()
    both.begin(sig)
    solo.begin(sig)
    for i in range(n):
        if both.is_full_step(i):
            both.observe(i, traj(i))
            solo.observe(i, traj(i, slice(1, 2)))
        else:
            torch.testing.assert_close(both.forecast(i)[1:2], solo.forecast(i))


def test_matches_core_predictor():
    """Same estimator as networks.spectrum_forecast.SpectrumPredictor (which
    rounds τ and the coefficients to bf16, hence the loose tolerance)."""
    from library.moss.spectrum import _Forecaster
    from networks.spectrum_forecast import SpectrumPredictor

    torch.manual_seed(2)
    n = 50
    ts = [0, 1, 2, 3, 4, 5, 7, 10, 13, 17]
    hs = [torch.randn(3, 7).to(torch.bfloat16) * 0.1 + i / n for i in ts]
    for w in (0.3, 1.0):
        core = SpectrumPredictor(
            3, 0.1, w, torch.device("cpu"), hs[0].shape, total_steps=n
        )
        ours = _Forecaster(3, 0.1, w, n, 100)
        for t, h in zip(ts, hs):
            core.update(float(t), h)
            ours.update(float(t), h)
        for t_star in (18.0, 20.0):
            torch.testing.assert_close(
                ours.predict(t_star).float(),
                core.predict(t_star).float(),
                atol=0.03,
                rtol=0.03,
            )
