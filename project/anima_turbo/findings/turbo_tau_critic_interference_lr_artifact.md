# Turbo τ-split critic — the cross-band interference signal is an LR artifact

**Verdict: closed 2026-07-20.** The DP-DMD fake/critic LoRA shows no
harvestable cross-band (τ) gradient interference at the LR it actually trains
at. The τ-split line is closed; `fake_tau_banks` ships inert at `1`.

Proposal (archived): `_archive/proposals/turbo_tau_split_critic.md`.
Probe: `bench/turbo/tau_critic_probe.py`.

## The two reads

Same bundle (`anima_superturbo_B` step 3500, `fake_rank=96`), same seed, same
400 fake-only updates, same probe set. Only the LR differs.

| | tail LR 4.04e-6 | peak LR 3e-5 |
|---|---|---|
| result dir | `bench/turbo/results/20260714-2032-p0b-superturbo-B/` | `bench/turbo/results/20260714-2109-p0b-superturbo-B-peaklr/` |
| improvement lo-band | 2.68e-5 (floor 1.55e-6) → ~17× | 3.83e-4 (floor 3.33e-5) → passes |
| improvement hi-band | 5.27e-5 (floor 2.71e-6) → ~19× | −6.15e-5 (floor 3.54e-4) |
| `degradation_lo_arm_on_hi_band` | positive (forgetting) | −6.99e-5 (lo arm *improved* hi band) |
| G1 / G2 | true / true | false / false |
| verdict | `FIRE — Phase 1 unlocked` | `G1_FAIL — close the line` |

The pre-registered rule (proposal §"If G1 fails") is close the line.

## Why the first read was misleading

At 4e-6 over 400 updates the critic barely moves — every clone stays in the
linear regime around its init, so the arms differ by a nearly-deterministic
first-order term while `ctl`/`ctl2` (same LR, different seed only) barely
separate. That collapses the noise floor to ~1e-6 and inflates the ratio.
The 17–19× measured magnitude of specialization was real but tiny in
absolute terms; the ratio was carried by the denominator.

At the real LR the arms enter the nonlinear regime, seed variance grows two
orders of magnitude (hi-band floor 3.54e-4), and the band-specialization
signal does not grow with it. G2 inverts outright: the lo-restricted clone
*improved* on the hi band, i.e. the bands share one solution — exactly the
"split is pointless" condition G2 was written to catch.

Reusable lesson: a probe fine-tune must run at the LR the target actually
trains at. Reading a gate off an annealed-tail LR measures the linearized
neighborhood of the init, not the training dynamics, and systematically
inflates any signal-to-noise ratio. Pre-register the peak-LR arm as the
primary, not the confirmation.

## Honest caveat

The peak-LR hi-band noise floor (3.54e-4) exceeds the *entire* signal of the
tail-LR run, so that arm is arguably underpowered rather than cleanly
negative — it bounds the effect as "smaller than seed variance at 400
updates", not "absent". Nobody re-ran it with more updates. If this line is
ever reopened, the only defensible door is re-running P0b at peak LR with
enough updates to pull the floor down — not a Phase-1 A/B, which was never
justified.

## What was kept

- P0a telemetry — `TauBinCriticLoss`, τ-binned critic loss, rides every
  turbo run regardless. This was always scoped to survive the verdict.
- `fake_tau_banks` / `fake_tau_boundary` (`configs/methods/turbo.toml
  [network]`) ship at `1` / `0.5`. `banks=1` is byte-identical to the
  pre-change loop (same RNG stream, second stack never constructed).
  Invariant tests: `tests/test_turbo_tau_critic.py`.
- The probe itself (`bench/turbo/tau_critic_probe.py`) is reusable for any
  "does capacity specialization by τ buy anything" question.

## What was never run

Phase 1's four gates — rendered grids at student NFE, within-run CMMD,
profile-flattening mechanism check, diversity grids — were never
evaluated. Phase 1 was wired (commit `9d3a8438`) before the peak-LR probe
landed, which is why the tree briefly looked like an open line. The only
long `banks=2` run is `anima_superturbo_C` (`output/logs/turbo/20260715-081251`),
which is the null NFE=2 arm and confounded three ways (banks=2 + `fake_rank=48`
+ init change, no single-bank control). Every turbo run since 2026-07-16 is
`fake_tau_banks = 1`.

Sits next to the student-side "rank ≠ lever" finding: capacity is not the
binding constraint on either half of the DP-DMD loop.
