# i2i tone drift as accumulated x_t trust

Status: Phase 0 measured 2026-11-12; the reading does not explain the 6× (see "Phase 0
result"). Phase 1 not started.

## The reading

Two facts from the tone line:

- The sanitize LoRAs carry a small tone error in the single-step x̂₀ with a correct x_t,
  on training pairs as on held-out pages (e8 EMA: Δb −0.09 / −0.10 at σ 0.1 / 0.3, 0 at
  σ 0.6).
- The render carries about 6× that (Δb −0.63), and 5–8× for L.

A sampler step mixes x_t and x̂₀ with weights that sum to one, so a per-step error that
did not feed back would average, not grow. Growth needs the model to carry the tone of
x_t into x̂₀.

That is what training teaches. The mean of a channel over a few thousand tokens is nearly
noise-free in x_t at any σ below ~0.95, and in training it is always the target's mean. So
the cheapest correct rule is x̂₀ tone = x_t tone / (1−σ), plus the LoRA's own error δ. In a
render x_t holds the earlier steps' errors, and nothing pulls them back. In i2i the
reference holds the right tone on every unedited token, but training never rewards reading
it from there, because x_t is equally right.

Write m = (tone of x_t) / (1−σ), the page tone the state implies. With trust k (the share
of m that x̂₀ keeps; the rest comes from the reference) and the er_sde first-order update
`x' = r_α·r·x + α_t·(1−r)·x̂₀`:

```
x̂₀ tone = k·m + (1−k)·ref + δ
m'      = m + (1−r)·((1−k)·(ref − m) + δ)
```

- k = 1: m' = m + (1−r)·δ. Every step adds its error; the last step adds δ whole.
- k < 1: the drift settles at δ / (1−k). A 6× render needs k ≈ 0.83 or more.

Σ(1−r) on the 50-step shift-5 grid, for δ constant over a σ range and k = 1:

| δ holds on | Euler | er_sde |
|---|---|---|
| σ ≤ 0.3 | 2.6 | 2.6 |
| σ ≤ 0.45 | 3.2 | 3.3 |
| σ ≤ 0.6 | 3.9 | 4.0 |
| all σ | 9.5 | 11.2 |

With δ ≈ −0.1 below σ 0.45–0.6 this predicts a render Δb of −0.3 to −0.4 against the
measured −0.63: the right order, about half unexplained. δ was measured on a correct x_t
only; its size on a drifted x_t is unknown.

The table also agrees with the settled result that the sampler is not the cause: Euler and
er_sde differ by under 3 % where δ is non-zero, and CNS only recolors injected noise whose
channel mean is 1/√N already.

Why it is worth a run although sanitize v3 passes the mean gates: v3 depends on a late tone
term whose weight does not carry across run lengths (`tone_weight 0.001` passes at 600
steps and fails at 2,216), and Δb still grows with training. Lowering k divides every tone
error by the same factor, whatever its sign, without a tuned weight, and applies to
swap_text unchanged.

## Phase 0: measure k (probe, minutes)

Extend `project/finished/moss_lora/src/tone_probe.py`. For each pair and σ, run the forward
twice with the same noise: on x_t, and on x_t + (1−σ)·c·d over the whole page, with d a
tone direction from `library/moss/calibration/tone_dirs.safetensors` and c in Lab units.

```
k(σ) = ⟨x̂₀(shifted) − x̂₀(plain), d⟩ over the rest tokens / c
```

- Directions L and b; c = ±1, ±2 (linearity check); σ = 0.1, 0.3, 0.45, 0.6, 0.8.
- LoRAs: base, 200p (no drift), `idt_cs0_600` EMA (drifts most), the shipped v3.
- 24 pairs of the identity tree plus the five held-out pages, as before.

Read, fixed now:

- k ≥ 0.8 at σ ≤ 0.45 on the drifting LoRAs: the reading holds, go to Phase 1.
- k ≤ 0.3: the model already re-anchors on the reference; the drift is not accumulated
  trust and this proposal closes.
- 200p decides between two stories for its clean renders: low k, or k near 1 with δ ≈ 0.

If k is high, Σ(1−r)·δ(σ) with the probe's δ at five σ replaces the table above.

## Phase 0 result (2026-11-12)

`project/finished/moss_lora/src/tone_trust_probe.py` (job `20261112-175629-b13298`, output
`project/finished/moss_lora/space/tone_trust_probe.json`), on the render base (int8 GPTQ, fp32
stream). The 200p and `idt_cs0_600` files no longer exist; the LoRAs are base and the shipped v3
(`moss_i2i_sanitize_idt_w0003_late_lr5e5_e8`) EMA, its e4 and raw. Six identity-tree pairs plus
the five pages (the per-sample spread was ±0.03 in a smoke run); shifts L +2, b ±2 (±1 and ±2
gave the same k in the smoke run). A v̂ that ignores the shift gives k = 1−σ.

k on the rest tokens of the held-out pages (training pairs within 0.05):

| σ | 1−σ | base L / b | v3 L / b | v3 e4 L / b | v3 raw L / b |
|---|---|---|---|---|---|
| 0.1 | 0.90 | 0.93 / 0.89 | 0.78 / 0.66 | 0.83 / 0.72 | 0.78 / 0.65 |
| 0.3 | 0.70 | 0.77 / 0.68 | 0.52 / 0.36 | 0.58 / 0.42 | 0.52 / 0.36 |
| 0.45 | 0.55 | 0.66 / 0.58 | 0.37 / 0.23 | 0.43 / 0.28 | 0.37 / 0.23 |
| 0.6 | 0.40 | 0.56 / 0.48 | 0.26 / 0.14 | 0.29 / 0.18 | 0.25 / 0.13 |
| 0.8 | 0.20 | 0.39 / 0.32 | 0.11 / 0.06 | 0.14 / 0.07 | 0.11 / 0.05 |

- The base sits at the no-response line. The trained LoRAs are well below it: they already
  pull x̂₀'s tone back toward the reference, b more than L. e4 is closer to 1−σ than e8, so
  training lowers k further while Δb grows. k does not track the drift.
- The response is linear in c and has no L↔b cross-talk (≤ 0.02). The part outside the L / a / b
  directions grows with σ (residual 0.13 at σ 0.1, 0.74 at 0.8), so k above σ 0.6 is the
  weakest number.
- The gate's two lines (k ≥ 0.8 / ≤ 0.3 at σ ≤ 0.45) bracket the result: 0.65–0.83 at σ 0.1,
  0.23–0.43 at 0.45. Put through the recurrence on the 50-step shift-5 grid (Euler, k(σ)
  interpolated), a constant δ on σ ≤ 0.3 / 0.45 / 0.6 / all σ ends at 1.5 / 1.8 / 1.9 / 2.1 ×δ
  for v3 b (k = 1: 1.9 / 2.7 / 3.2 / 8.8). The 6× needs k ≥ 0.83; the measured k gives
  at most ~2×. Accumulated trust is not what amplifies the drift, and Phase 1 targets
  the wrong quantity.
- What remains open is δ on the render's own x_t (only measured on a correct x_t).

**Trajectory probe (2026-11-12).** `project/finished/moss_lora/src/tone_traj_probe.py` (job
`20261112-182012-d9b47b`, `space/tone_traj_probe.json`, renders
`output/tests/moss_sanitize_traj/`): the five pages × seeds 42 / 43 / 44 through the sanitize
generate config (CFG 3 + skip-22, ER-SDE, Spectrum, int8, fp32 stream), per step the rest
tokens' m and guided x̂₀ fitted on the tone directions. Mean over 15 renders, b:

| σ | v3 m / x̂₀ | v3 raw m / x̂₀ |
|---|---|---|
| 0.80 | −0.80 / −0.18 | −0.38 / −0.02 |
| 0.60 | −0.38 / −0.31 | −0.18 / −0.24 |
| 0.46 | −0.37 / −0.40 | −0.19 / −0.22 |
| 0.32 | −0.46 / −0.42 | −0.27 / +0.03 |
| 0.19 | −0.48 / −0.46 | −0.10 / +0.25 |
| 0.12 | −0.45 / −0.43 | +0.14 / +0.39 |
| final | −0.44 (decoded −0.27) | +0.40 (decoded +0.55) |

- No late build-up: v3's m is flat from σ 0.6 on, as the low k predicts.
- v3 EMA: on a correct x_t its δb is ≈ 0 at σ ≤ 0.45 (Phase 0: +0.02 / −0.04 / +0.05), on its
  own render the guided x̂₀ sits at −0.3 to −0.46 from σ 0.6 down. The render's cool b comes
  from x̂₀ on the trajectory, not from δ on a correct x_t. The probe does not separate the
  render's state from guidance (Phase 0 is the cond branch alone).
- v3 raw: the warm b enters at σ ≤ 0.3, the size of its δ on a correct x_t there (+0.34 /
  +0.41), not amplified.
- The latent fit is a rough proxy: b keeps its sign against the decoded PNG (−0.44 / −0.27,
  +0.40 / +0.55), L does not (+0.58 vs −0.05 for v3), and per page the two differ by up to 0.7.
  The mask is from the seed-0 render, so seeds 42–44 may edit some "rest" pixels.

## Phase 1: train with a tone-shifted x_t

Per step, with probability p, on the whole page:

```
x_t'   = x_t + (1−σ)·(u_L·d_L + u_b·d_b)      u ~ U(−A, A) Lab units
target = (x_t' − x₀) / σ                       # the v that still lands on x₀
```

The reference and x₀ are untouched, so the only place the right tone can be read is the
reference. This is the opposite of the reverted tone jitter, which moved the target and
the reference together and left x_t's tone correct.

- A = 2 (the renders drift up to ~2 in L, ~1 in b), p = 0.5. The added target term scales
  as (1−σ)/σ; check its size against the v-MSE at the lowest trained σ before the first run
  and clamp A·(1−σ)/σ if it dominates.
- Micro arm first: the `idt_cs0_600` recipe (Muon lr 1e-4, 600 steps, identity tree, no
  tone term), control vs shifted, same σ sequence. Then the v3 recipe at full length, with
  and without its tone term.

Gates, fixed now (renders as in the tone line: five pages × seeds 42 / 43 / 44, shared mask,
`bench/moss_i2i_eval/tone_map.py`):

- Mechanism: k of the shifted arm at σ ≤ 0.45 is below the control's by at least 0.3.
- Tone: mean |ΔL| ≤ 0.5 and |Δb| ≤ 0.3, sat× in 0.97–1.03, no page past |ΔL| 1.0 —
  without the tone term in the micro arm.
- Edits no worse than the control by eye on hard_to_sanitize1 (「無断転載禁止」, the
  handle) and hard_to_sanitize2 (red caption boxes). The old jitter failed here.

## Not in this proposal

- A render-time re-anchor (set the rest tokens' x̂₀ tone to the reference's each step). It
  would confirm accumulation without training, and stays the fallback for LoRAs already
  shipped.
- Colorize, which changes tone on purpose.
- The CNS paper (arXiv 2605.30332) prompted the question but has no part in the mechanism:
  it does not treat the channel mean, and its App. A.3 mismatch argument compares γ of x̂₀
  with the 1−t signal share of x_t.
