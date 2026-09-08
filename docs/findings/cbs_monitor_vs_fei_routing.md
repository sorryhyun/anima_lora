# CBS complexity and FEI routing peak at opposite ends of the σ trajectory

Tested whether the Complexity-Balanced Splitting criterion (CBS; Issachar,
Lischinski & Fattal, arXiv:2606.06477) could re-place the boundaries our
ChimeraHydra FreqRouter routes on. The question: CBS argues for dedicating
sub-networks to timestep intervals chosen by equidistributing a complexity
monitor, and we already split on FEI — so does CBS's principled boundary agree
with where FEI puts its routing signal?

**It does not — they are anti-aligned.** CBS's path-acceleration monitor
concentrates modeling complexity at the noise end (t ≈ 0.95–1.0); FEI's
discriminative signal lives at the clean end (t ≈ 0.1). The two answer
different questions, and the data says borrowing CBS knots to curate FreqRouter
would push adapter capacity to exactly the region where FEI has nothing to route
on.

![CBS path-acceleration monitor m(t) with equidistribution knots (right panel)
vs the FEI e_low trajectory and inter-artist std (left, center). The green m(t)
spikes at t→1; the CBS knots sit at t≈0.84–0.95; the FEI std-peak (orange) is at
t≈0.10.](assets/cbs_monitor_vs_fei.png)

## The monitor, for free, on the real trajectory

CBS's monitor is `m(t) = ‖d²x_t/dt²‖` (their Eq. 13/14). Because `dx/dt =
v_t(x_t)`, this is just the second difference of the realized latent
sequence the FEI probe already captures at every sampler step — no velocity
hook, and no auxiliary proxy model (their §4.5 robustness argument is moot when
you have the real base model). `bench/fera_artist/probe_fei_trajectory.py` now
forms a non-uniform central second difference of `x_t`, RMS-normalized per
element so it averages fairly across native buckets, then equidistributes
`∫m(t)dt` to derive the paper's time-split knots for N=2 and N=3.

## Evidence — the two signals are at opposite ends

From `20260605-1804-cbs_cfg4` (20 prompts / 20 artists, 1 seed, native buckets,
28 steps, flow_shift=3, CFG=4, FEI div=16). The t-axis runs 1.0 (noise) →
0.10 (near-clean).

| t (sampler σ) | step | `m(t)` path accel | `e_low(z_t)` | std(e_low) across artists |
|---|---|---|---|---|
| 0.988 | 1  | 76.5 | 0.0021 | 0.0003 |
| 0.961 | 3  | 46.1 | 0.0027 | 0.0005 |
| 0.934 | 5  | 18.3 | 0.0040 | 0.0009 |
| 0.863 | 9  | 9.2  | 0.0100 | 0.0029 |
| 0.691 | 16 | 4.4  | 0.0566 | 0.0177 |
| 0.500 | 21 | 1.79 | 0.2072 | 0.0559 |
| 0.449 | 22 | 1.70 | 0.2687 | 0.0676 |
| 0.334 | 24 | 1.40 | 0.4359 | 0.0891 |
| 0.100 | 27 | (NaN, endpoint) | 0.6881 | 0.1006 |

`m(t)` decays ~50× from a spike at the noise end to a tight ~1.3 plateau
(std ±0.2) by mid-trajectory. `std(e_low)` — FEI's inter-artist routing signal —
does the opposite, climbing monotonically to its max at the clean end.

Derived boundaries (`cbs` block in `teacher_curve.json`):

| Boundary | t |
|---|---|
| CBS N=2 knot (equidistribution) | 0.909 |
| CBS N=3 knots | 0.838, 0.951 |
| FEI std(e_low) peak (max routing signal) | 0.100 |
| FEI e_low slope-max (regime boundary) | 0.364 |

Half of all path-acceleration complexity falls in t ∈ [0.91, 1.0] — the first
~3 denoising steps off pure noise. `|CBS_N2 − slope-max| = 0.54`: not a
near-miss, opposite ends.

## Why they diverge — and why that's the correct answer

The two quantities measure different things:

- Path acceleration = base-flow approximation difficulty. From isotropic
  noise the ODE bends hard in the first few steps as it commits to a mode, then
  runs nearly straight (consistent with rectified-flow's "trajectories are
  mostly straight after commitment"; the plateau is ~50× below the spike). CBS
  allocates base-model capacity to that early curvature.
- FEI = where a content/style adapter can discriminate and act. At the noise
  end every latent is ~pure high-freq (`e_low ≈ 0.002`, between-artist std
  ≈ 0.0003 — nothing to route on). Low-freq structure and its between-artist
  variance resolve toward the clean end, matching
  [`sigma_signal_where_anima_resolves.md`] (x0 resolves by σ≈0.45, detail in the
  σ<0.45 tail).

So CBS answers "where does the *base* network struggle to fit the velocity
field," while FreqRouter answers "where can a *LoRA adapter* most effectively
bend the output toward the trained concept." Adapter leverage ≠ base-flow
curvature; there is no reason they would align, and they don't.

## Implications

1. Do not curate FreqRouter / Chimera band boundaries with CBS. Borrowing
   the knots would shove adapter capacity to t≈0.91, where FEI has zero
   discriminative signal and the adapter has nothing to act on — it would fight
   what FEI correctly does. The "complexity-prior-on-the-router" idea is
   refuted, with a mechanism, not just a null result.

2. CBS's criterion belongs on the base model / distillation, not routing. A
   2–4 step turbo student (DP-DMD, `student_steps=2`) should concentrate its
   scarce steps and capacity at the high-σ end where the curvature budget sits.
   That is the apt home for this monitor; Chimera routing is not.

3. FEI is vindicated for adapter routing. Its signal sits exactly where the
   content resolves; CBS measuring something orthogonal is consistent with FEI
   being the right input for adapter gating specifically.

## Caveats

- Single seed, 20 prompts. Early-step `m` is noisy (step 1: 76.5 ± 25.7),
  but the monotone decay holds across all 27 steps and the plateau is tight, so
  the N=2 knot stays firmly at the noise end regardless — the *direction* is
  robust; the exact 0.909 is not.
- FEI ran at div=16, not Chimera's default div=4. That shifts the `e_low`
  curve and the slope-max number, but not the headline: discrimination resolves
  at the clean end is structural, not a div artifact. `m(t)` is computed on the
  raw latent and is div-independent.
- CFG=4 only. CFG injects spatial gradients, so
  the front-loading magnitude could differ at CFG=1. A CFG=1 run would settle
  whether the early-curvature spike is intrinsic or guidance-amplified; it will
  not move the noise-end-vs-clean-end conclusion.
- t-axis stops at ≈0.10. Anima's 28-step Euler with flow_shift=3 doesn't
  reach t=0, so the FEI-discrimination peak may sit slightly past the measured
  endpoint — only reinforcing "clean end."

## Reproduce

```bash
uv run python bench/fera_artist/probe_fei_trajectory.py \
  --k_per_artist 1 --infer_steps 28 --guidance_scale 4.0 --label cbs_cfg4
# CFG=1 / div=4 variant to nail the caveats:
uv run python bench/fera_artist/probe_fei_trajectory.py \
  --k_per_artist 1 --infer_steps 28 --guidance_scale 1.0 \
  --fei_sigma_low_div 4 --label cbs_cfg1_div4
```

The probe writes per-step `m_accel` + a `cbs` block (`knots`, `total_monitor`,
`fei_std_peak_t`, `elow_slopemax_t`) into `teacher_curve.json`, and the
third plot panel overlays the monitor, cumulative ∫m, and all four boundaries.

Date: 2026-06-05. Run: `bench/fera_artist/results/20260605-1804-cbs_cfg4/`
(git bc49aae). Paper: `2606.06477v1.pdf` (CBS).
