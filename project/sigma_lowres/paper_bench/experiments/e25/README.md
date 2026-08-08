# E25 — σ-local angle lever (population-level exploitation of the cancellation axis)

| | |
|---|---|
| **Status** | **SKETCH 2026-08-08 — NOT pre-registered.** Constants, arms, and gates are deliberately unfrozen; freezing requires the E25.0 read below plus an explicit E20.4-adjacency justification. Nothing here licenses a run. |
| **Question** | The line now knows *where* the residual gap lives: the cancellation is locally enforced (E21), its axis is one direction per σ across routes/stores (E24 STRUCTURED), and the residual is **angle-borne** — closing ρ → −1 removes 70–100 % of it while amplitude matching removes ≤ 45 % (E24.2). Can a **population-level** lever exploit this — either by filtering the residual direction out of demoted-sample gradients, or by making the implicit resolution dimension explicit — without touching anything the line has closed? |
| **Licensed by** | E24 STRUCTURED clause (this is the named follow-up; the lever **must be σ-local** — a single fixed subspace is refuted), E24.2 knob read (target = angle/residual direction, never amplitude rebalancing), E21 (adaln carries 86–87 % of the phase-response amplitude — the natural injection/damping site), pooled evidence only. |
| **Explicitly NOT licensed** | Anything per-sample (gated on E22 → 22.4 → E23a, unchanged); any PI-RoPE revival (G11); and — the sharpest adjacency — anything that re-derives a ledger term as an objective correction (E20.4 closed that at the estimand level). A frozen E25 must argue explicitly why its lever is an in-expectation filter / conditioning change and not the closed derived-data-term family. |

## Candidate arms (sketch only)

- **E25a — σ-local residual projection guard**: during demoted training
  steps, damp/project the component of the LoRA gradient along the
  σ-binned pooled residual direction (B+C)̂(σ), read from a probe-built
  lookup (per σ bin; co-rotating with the axis field per E24). Optimizer-
  side, in expectation, no per-sample estimate anywhere.
- **E25b — explicit resolution conditioning**: inject a resolution/scale
  embedding into the adaln pathway during LoRA training (micro-
  conditioning precedent), making the dimension the network currently
  absorbs implicitly (E21: adaln amplitude concentration) explicit.
  Prediction to gate on: ρ deepens / h(B+C) shrinks at the verdict σ,
  measured by the existing σ probe at the same operating-point protocol.

Both are Tier 1.5 if pursued: bench + invariant test required; quality
gate = no CMMD regression on the E4-style yardstick, pitch stays
wall-clock-at-fixed-steps (autoscale lesson).

## E25.0 — prerequisite read (cheap, CPU, must precede any freeze)

Two facts the sketch depends on that are **not yet measured**:

1. **Residual-direction reliability**: is the pooled (B+C)̂ direction
   reproducible across draw sets per (route, σ)? (B and C are each
   reliable, but their sum is the small difference of large legs — its
   direction may not be.) Free read on the three existing arm_sums
   stores with the E24 machinery. If unreliable pooled, E25a is dead
   before it starts.
2. **Frame-relative stationarity**: is the axis stationary in the ĝ(σ)
   frame (E24's descriptive co-rotation, promoted to a pre-registered
   estimand)? Decides whether E25a's lookup can be one direction in a
   normalized frame or must interpolate per σ bin.

## Kill criteria (sketch level)

- E25.0 (1) fails → E25a dead; E25b unaffected.
- Any formulation that requires a per-sample quantity → out of scope,
  full stop (that is E23a's gated territory).
- If the frozen version cannot distinguish itself from E20.4's closed
  family in one paragraph, it does not run.
