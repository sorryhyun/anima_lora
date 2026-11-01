"""Invariants for Foresight Guidance (FSG) pre-step latent calibration.

The load-bearing safety property (CONTRIBUTING Tier-2): FSG with K=0 or an
out-of-band σ is a bit-exact no-op vs the baseline trajectory. Also checks the
forward-backward operator's fixed-point property (a constant velocity field is a
fixed point) and the documented extra-forward cost (3·K per scheduled step).

These run on a fake DiT (a plain callable) — the hydra/FEI setters are pure
``getattr`` no-ops on a non-adapter model, so no checkpoint is needed.
"""

from __future__ import annotations

import torch

from library.inference.corrections.fsg import FSGCalibrator


class _ConstField:
    """Fake DiT returning a fixed velocity regardless of input. A constant
    field makes F(x)=x exactly, so calibration must be a numerical no-op."""

    def __init__(self, v: torch.Tensor):
        self.v = v
        self.calls = 0

    def __call__(self, x, t, embed, padding_mask=None, **kw):
        self.calls += 1
        return self.v.expand_as(x).clone()


def _inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, 1, 8, 8, generator=g)
    embed = torch.randn(1, 16, 32, generator=g)
    nembed = torch.randn(1, 16, 32, generator=g)
    pad = torch.zeros(1, 1, 8, 8)
    return x, embed, nembed, pad


def test_scheduled_gate():
    fsg = FSGCalibrator(band=(0.45, 0.85), k=3)
    assert fsg.scheduled(0.6)
    assert fsg.scheduled(0.45) and fsg.scheduled(0.85)  # inclusive
    assert not fsg.scheduled(0.94)  # above band (the diverging dead zone)
    assert not fsg.scheduled(0.2)  # below band
    # K=0 is globally inert regardless of σ.
    assert not FSGCalibrator(band=(0.45, 0.85), k=0).scheduled(0.6)


def test_out_of_band_is_identical_object():
    """Out-of-band σ returns the exact same tensor object (bit-exact baseline)."""
    x, embed, nembed, pad = _inputs()
    anima = _ConstField(torch.zeros(1, 4, 1, 8, 8))
    fsg = FSGCalibrator(band=(0.45, 0.85), k=3)
    out = fsg.calibrate(anima, x, 0.94, 0, embed, nembed, pad, 4.0)
    assert out is x
    assert anima.calls == 0  # no forwards spent off-band


def test_constant_field_is_fixed_point():
    """If v^c == v^u == const, every iteration is an exact round-trip (F(x)=x),
    so the calibrated latent equals the input within fp tolerance."""
    x, embed, nembed, pad = _inputs()
    anima = _ConstField(torch.full((1, 4, 1, 8, 8), 0.3))
    fsg = FSGCalibrator(band=(0.45, 0.85), k=4, d_sigma=0.1)
    out = fsg.calibrate(anima, x, 0.6, 0, embed, nembed, pad, 4.0)
    assert torch.allclose(out, x, atol=1e-5)
    assert anima.calls == 3 * 4  # 3 forwards (v^c, v^u, v^u_lo) per iteration


# --- FSG × Spectrum/SEA composition --------------------------------------
# FSG-scheduled steps are forced to actual forwards and carved out of the
# spectrum cache scheduler's decision domain (treated like warmup/tail). These
# pin the SEA-accounting invariant: forcing a step neither inflates nor deflates
# the window/SEA refresh fraction it is matched against.


def test_forced_step_does_not_advance_window():
    """Forcing every decision step actual leaves no decision steps, so the
    fraction is the empty-domain sentinel (0/ max(1,0)) — i.e. SEA schedules
    nothing and FSG+warmup+tail carry the whole trajectory."""
    from networks.spectrum import _window_decision_fraction

    all_decision = frozenset(range(6, 17))
    assert _window_decision_fraction(20, 6, 17, 2.0, 0.25, all_decision) == 0.0


# --- CFG++ substrate (the σ-scheduled guidance reweight) ------------------
# CFG++ is implemented as a pure reweight of the cond/uncond combine so it
# composes with any integrator (Euler, er_sde). These pin: (1) the reweight is
# algebraically identical to the "calibrate x̂ then Euler-step along v^u" form it
# replaced — so the validated Euler renders are unchanged; (2) the final-step
# boundary; (3) the weight is finite/positive across a real schedule.


def test_cfgpp_reweight_equals_calibrate_then_step():
    """w_eff form ≡ the old calibrate-then-step form, for the Euler integrator."""
    from library.inference.sampling import cfgpp_guidance_weight

    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, 1, 8, 8, generator=g)
    vc = torch.randn(1, 4, 1, 8, 8, generator=g)
    vu = torch.randn(1, 4, 1, 8, 8, generator=g)
    s_i, s_next, lam = 0.8, 0.72, 2.0
    step = s_i - s_next

    # Old form: calibrate x̂ = x − λ(1−σ')σ·Δv, then Euler-step along v^u.
    x_hat = x - lam * (1.0 - s_next) * s_i * (vc - vu)
    old = x_hat - step * vu

    # New form: reweight the combine, plain Euler step.
    w_eff = cfgpp_guidance_weight(s_i, s_next, lam)
    new = x - step * (vu + w_eff * (vc - vu))

    assert torch.allclose(old, new, atol=1e-6)


def test_cfgpp_weight_final_step_and_finite():
    from library.inference.sampling import (
        cfgpp_guidance_weight,
        get_timesteps_sigmas,
    )

    # Final step (σ_next → 0) collapses to λ.
    assert abs(cfgpp_guidance_weight(0.05, 0.0, 2.0) - 2.0) < 1e-9
    # Degenerate Δσ ≤ 0 falls back to λ rather than dividing by zero.
    assert cfgpp_guidance_weight(0.5, 0.5, 2.0) == 2.0
    # Across a real flow schedule: finite, positive, no blow-up.
    _, sigmas = get_timesteps_sigmas(20, 3.0, torch.device("cpu"))
    sig = [float(s) for s in sigmas]
    ws = [cfgpp_guidance_weight(sig[i], sig[i + 1], 2.0) for i in range(20)]
    assert all(0.0 < w < 50.0 for w in ws)


# --- CFG++ × Spectrum combine ---------------------------------------------
# The spectrum runner threads cfgpp_lambda as a side-channel and merges
# cond/uncond through `_combine_guided` in both its actual-forward and
# cached-prediction branches. These pin: (1) cfgpp off == plain CFG (the
# trajectory is bit-identical to a non-cfg++ spectrum run); (2) cfgpp on applies
# exactly the w_eff reweight; (3) the branch order is total (cfgpp wins over a
# stray smc state, matching generation.py's mutual-exclusion guard).


def test_spectrum_combine_off_is_plain_cfg():
    from networks.spectrum import _combine_guided

    g = torch.Generator().manual_seed(2)
    cond = torch.randn(1, 4, 1, 8, 8, generator=g)
    uncond = torch.randn(1, 4, 1, 8, 8, generator=g)
    gs = 4.0
    out = _combine_guided(cond, uncond, cfgpp_w_eff=None, guidance_scale=gs)
    assert torch.equal(out, uncond + gs * (cond - uncond))


def test_spectrum_combine_cfgpp_beats_smc():
    """cfgpp_w_eff set wins over a stray smc_cfg — matches generation.py's
    refusal to run both combines at once."""
    from networks.spectrum import _combine_guided

    class _Boom:
        def combine(self, *a):  # pragma: no cover - must never be called
            raise AssertionError("smc_cfg.combine called despite cfg++ active")

    g = torch.Generator().manual_seed(4)
    cond = torch.randn(1, 4, 1, 8, 8, generator=g)
    uncond = torch.randn(1, 4, 1, 8, 8, generator=g)
    out = _combine_guided(
        cond, uncond, cfgpp_w_eff=2.0, smc_cfg=_Boom(), guidance_scale=4.0
    )
    assert torch.equal(out, uncond + 2.0 * (cond - uncond))
