# E19 appendix — the B/C geometry, per σ, in one picture

![B/C legs planted on the sigma axis, route 1024→768](bc_comb_768.png)

`fig_bc_comb.py` → `bc_comb_768.png`. Figure assembly only — every
number is read from the committed E14 probe-matched ledger
(`runs/20260801-2304-e14-ledger-probematched/ledger.json`, reenc ref);
nothing is refit. Companion to the paper's Fig. `ledgergeom`, which
draws one σ bin per route; this draws one route (1024→768) across all
15 bins.

## Conventions

- At each bin slot: **B⊥** (data leg, blue) points straight up at its
  true debiased length √(2S)·‖ḡ‖; **C⊥** (graph leg, red) folds back
  from B's tip at the bin's angle arccos ρ (debiased ρ, clamped to
  [−1, 1] for drawing); **B+C** (black) runs base → C-tip and is the
  realized gap vector. One shared scale (bar at right, units of
  ‖ḡ_src‖); the dashed envelope traces |B⊥|(σ).
- Bin slots are **equally spaced** (the E14 grid is dense at both ends;
  planting at true σ would overlap the low-σ triangles). σ values label
  the slots; the E9 reduction-failure window (σ 0.56–0.81) is shaded.
- Unit-honesty carries over: the σ=1 slot is a non-verdict cell
  (ratio-of-small-numbers; its ρ = −1.02 is out of the estimator's
  domain and drawn clamped), and leg *magnitudes* are licensed via
  h(·) in the ledger — the drawing uses √(2S), √(2F) for shape only.

## The three reads the format makes visible

1. **The near-cancellation, at a glance.** At σ = 0.3 both legs are
   ~1.6–1.9 ‖ḡ‖ — each intervention alone is enormous — while the black
   resultant is ~20% of either leg. The gap the training run actually
   feels is the small residue of two large opposed responses.
2. **Shape-invariance vs scale collapse.** The triangle's *shape*
   (angle ρ ≈ −0.8…−0.93, ratio 0.62–0.99) barely changes from
   σ = 0.0125 to 0.9875; what changes is overall scale — legs shrink
   ~40× from the σ = 0.3 peak to the endpoint, with the sharp drop at
   σ 0.3 → 0.43 just before the window. This is the visual form of the
   E19 verdict chain: the angle is a global, σ-stable property (19.3:
   also depth- and type-uniform), not a feature of any particular
   regime.
3. **Why 768 never fully heals.** |B⊥| < |C⊥| at every bin (no
   crossing, 19.0 item 4), so a same-direction residue of C always
   survives — the black arrows all point the same way. On 896 the
   analogous picture would show the blue leg overtaking the red near
   σ ≈ 0.5 (the crossing = the cliff mechanism), killing the resultant
   inside the window.

## One caution for readers of the paper's depth table

The floor's depth profile (early blocks 3–8 carry ~3× the late-block
gap at σ = 1 — Q2, the paper's `tab:depth`) and the cancellation
angle's depth profile (19.3: uniform, no special band, magnitude ∝
branch energy) are **different quantities in different σ regimes**.
Both are true; conflating them would suggest the anti-alignment lives
early-block, which 19.3 refuted.

## Regenerate

```bash
uv run python project/sigma_lowres/paper_bench/experiments/e19/fig_bc_comb.py
```

Variants worth drawing on demand: the same comb for 896/512 (three
stacked rows makes the crossing/no-crossing contrast per route
visible), and — once 19.4's ledger lands — a C_pi third leg overlaid at
the four 19.4 bins.
