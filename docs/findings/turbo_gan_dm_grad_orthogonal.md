# Turbo GAN generator gradient is elementwise DM-orthogonal — CLOSED

Status: **CLOSED 2026-07-20.** The OPD²-style sign gate on the adversarial
gradient (`_archive/proposals/turbo_gan_dm_sign_gate.md`) died at Phase 0: the GAN
generator gradient carries **no structured sign-disagreeing component against
the DM signal** — agree-energy is indistinguishable from a permutation null in
the aggregate and in every τ-bin. There is nothing for a sign (or PCGrad-style
projection) gate to veto. **Do not re-propose gradient surgery between the GAN
and DM terms on the DP-DMD loop.**

## Premise tested

OPD² ("On-Policy Delta Distillation", NAVER AI Lab, arXiv:2607.15161, Eq. 8)
applies a noisy auxiliary gradient only where it sign-agrees with a trusted
primary. The mapping to DP-DMD: primary = the detached DM surrogate
`grad_signal` at `x_pred`, auxiliary = the GAN generator gradient riding the
combined backward. The gate would be the selective version of the
`gan_delay_steps` escape window — veto only the anti-DM components, forever,
instead of holding the whole GAN at 0 for N steps.

The premise to measure (pre-registered, with the verdict thresholds fixed
before looking): does the GAN generator gradient carry a substantial (>~25%
energy), τ- or phase-localized, sign-disagreeing component against DM?

## What was measured

Measure-only telemetry (`[gan] dm_gate_telemetry`, shipped): inside
`gan_generator_term`, `g_gan = autograd.grad(gan_gen_loss, x_pred)` — read,
never accumulated, so training numerics are byte-identical — compared
elementwise against the applied DM signal, τ-binned (8 uniform bins), with a
permutation null computed on the same tensors from a dedicated generator.

Run: 95 steps riding a 3k-warm resume of `anima_turbo_cdm` (GAN-on the whole
run, `weight_gen 0.03`, `delay_steps 0`; run `anima_turbo_cdm_dmgate`, logs
`output/logs/turbo/20260720-194645`).

| metric | observed | null |
|---|---|---|
| agree-energy (aggregate) | 0.489 ± 0.032 | 0.488 ± 0.011 |
| agree-energy (per τ-bin) | 0.45–0.53, no bin beyond noise | 0.48–0.49 |
| agree-rate | 0.496 | ~0.5 orthogonal baseline |
| cos(g_gan, grad_signal) | 0.006 ± 0.038 | — |
| applied-magnitude ratio | ~6× median, heavy-tailed 0.5–50× | — |

The pre-registered Phase-1 trigger needed a >~25-point localized excess over
null; observed deviation is <1 point with 1–3 points of noise. Judgeable at
n=19 flush points precisely because the required effect size is enormous
relative to the noise floor.

## What it means

- **The GAN is not fighting DM** — it pushes along elementwise-orthogonal
  directions. A sign gate would randomly delete ~half the GAN's energy, which
  is exactly "a weaker GAN overall" (the confound the proposal's
  matched-magnitude control existed to expose), not selectivity.
- **The GAN push is NOT negligible**: ~6× the applied DM per-element magnitude
  (heavy-tailed). Substantial pressure, zero DM-alignment — mechanistically
  consistent with the `anima_turbo_R` "GAN spent" plateau verdict: realism
  pressure orthogonal to distribution matching buys texture, not diversity.
- This kills the *class* of GAN↔DM gradient-surgery ideas (sign veto, PCGrad
  projection, conflict-aware reweighting), not just Eq. 8 — they all
  presuppose a structured conflicting component that does not exist.

## Reusable traps (methodology)

1. **~0.5 agree-rate is the orthogonal baseline, not conflict.** Elementwise
   sign agreement between two high-dimensional latent fields sits at ~50% for
   *independent* signals. Judge on agree-**energy vs a permutation null**
   computed on the same tensors — never on the rate.
2. **Compare applied gradients, not raw tensors.** `loss_dmd` is a `.mean()`,
   so the DM push at `x_pred` is `grad_signal/numel`, while the GAN backward
   injects `gan_w·g_gan` unscaled — a raw-tensor magnitude ratio under-reports
   the GAN by ~numel (≈10⁵) and would have mis-closed this line as "GAN
   contributes nothing".
3. The telemetry is generic two-signal conflict tooling: `dm_gate_stats`
   (`scripts/distill_turbo/metrics.py`) + `DmGateTelemetry` work for any pair
   of gradients at `x_pred` (div vs DM, CDM vs DM, softrank vs DM), with the
   RNG-neutrality contract (dedicated generator) already handled. The
   dedicated gate tests it would have carried were never written — the line
   closed at Phase 0.

## Caveat

Late-phase warm regime only (3k-warm, near the LR floor). The early/cold-start
delay-window regime was never measured — the flag rides any future GAN-on run
for free if that read is ever wanted, but the token-disc-head λ=0 ramp control
(froze pose identically with zero GAN generator gradient) already deflated the
"GAN gradient causes early collapse" story independently.

## Pointers

- Proposal + full Phase-0 design/verdict: `_archive/proposals/turbo_gan_dm_sign_gate.md`
- Flag: `[gan] dm_gate_telemetry` / `--gan_dm_gate_telemetry`
  (`configs/methods/turbo.toml`); provenance `ss_turbo_gan_dm_gate_telemetry`
- TB tags: `train/gan_dm_{agree_rate,agree_energy,agree_energy_null}_tau{0..7}`
  + aggregates `train/gan_dm_{agree_rate,agree_energy,agree_energy_null,cos,mag_ratio}`
- OPD² — arXiv:2607.15161 (Eq. 6–9)
