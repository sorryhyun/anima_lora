# E9 — interventional B/C ledger

| | |
|---|---|
| **Status** | **DONE 2026-07-31** (launched 2026-07-30; wall 3.8 h) |
| **Verdict** | **Branch (i): negative interference.** I_768(σ) < 0 at every bin, with B and C near anti-parallel. Branch (ii) (F collapse) is dead. |
| **Consumed by** | This run spawned the cancellation line: [E14](../e14/) (probe-matched ledger, low σ) → [E17](../e17/)/[E19](../e19/) (closure + origin) → [E21](../e21/) (LOCAL) → [E22](../e22/) (per-sample, 22.4 HOLDS) → [E24](../e24/) (axis geometry) → [E25](../e25/) (lever sketch). |
| **Runs** | `bench/results/20260731-0721/` + its `ledger.json` (vector stores stay under the gitignored `bench/results/`) |
| **Instrument** | `bench/run_sigma_probe.py --repromote --keep_arm_sums --self_floor`, routes 896/768/512, σ ∈ [0.5,1.0] 4 bins + endpoint, D=8, N=24, deterministic |
| **Analysis** | `../../vector_ledger.py` (shared with [E10](../e10/)) |
| **Origin** | `paper/review/response.md` — the pre-registered I_768 scalar probe was found confounded (it estimates I(σ) + [F(σ) − F(1)], not I(σ)); pre-registered branches in `paper/action.md` |
| **In the paper** | §4.6 (the 768 mid-σ dip as the interaction signature); closes the §4.5 reenc-proxy `[pending]` |

**Design.** B = ḡ_rp − ḡ₀ (data branch, native graph); C = ḡ_dem − ḡ_rp
(graph branch, demoted data); cross-set-debiased S/F/I per (route, bin).
Reads F(σ) directly (no σ-flat assumption), settles the 768 window two
ways (I < 0 vs F collapse), localizes the Goldilocks prediction (window
center at |B⊥| ≈ |C⊥|), and closes the §4.5 reenc-proxy `[pending]`
(B vs native vs B vs reenc).

## Results — branch (i), negative interference

- **I_768(σ) < 0 at every bin** (−0.31 → −0.014 window→endpoint) with
  B, C near anti-parallel (ρ ≈ −0.93 in-window; sign robust to
  cross-set debiasing, same-set −0.37 vs cross −0.31 at bin 0).
- **Amplitude matching localizes the 768 window center at σ ≈ 0.69**
  (|B⊥|/|C⊥| = 0.98). (Did not reproduce probe-matched: E19 19.0 on
  E14's grid found **no in-window 768 crossing** — ratio peaking ≈ 0.99
  at σ = 0.0875, 0.76–0.88 through this window; the crossing↔window
  localization holds on 896 only, ≈ 0.5.)
- **Branch (ii) dead:** F_768 falls monotonically to its endpoint
  (0.200 → 0.0036), never below it in-window — Q1's σ=1 endpoint
  reduction stands (though F(σ) is strongly σ-dependent for every
  route, confirming the retracted σ-flat extension stays retracted;
  endpoint scope per E13 H1: the σ = 1 bin is its own estimand, not
  the σ→1⁻ limit).
- The cancellation itself is **universal** (ρ −0.62…−1.24 on all
  routes); routes differ by amplitude matching: 896 matched but small
  (net S+F+I ≤ +0.012), 768 matched mid-window (≤ +0.029), 512
  amplitude-mismatched at low σ (|C⊥|/|B⊥| ≈ 1.8 at bin 0, net up to
  +0.197). (The line then resolved the structure of "universal": ρ is
  route-uniform while amplitudes break — E19; the cancellation is
  LOCAL, pointwise per depth×type cell — E21; one shared axis per σ —
  E24; and per-sample at σ = 0.7 — E22.4.)
- **Reenc-proxy closed:** |B⊥| against reenc is within ~4% of |B⊥|
  against native at the signal-carrying low-σ bins (worst ±23% at bins
  where B is already small) — the shared down+up+encode pipeline cost
  is a minor share of the data intervention.

## How much of the gap the interference erases

Quantified 2026-07-31 from `ledger.json`; applied to §4.6 as its own
paragraph. The actual demote gap is exactly h(B+C) (ḡ_dem = ḡ₀+B+C by
construction), so each account's predicted/actual ratio is directly
computable per (route, bin):

- **No-interference scalar account (S+F)**: overpredicts the in-window
  gap **2.4–3.8× at 768** (2.4/3.3/3.8 across the three in-window
  bins), 1.6–5.8× at 896, 1.2–3.0× at 512.
- **Fully additive counterfactuals (h_B + h_C)**: overpredicts 3.5–8×
  — the realized gap is only **~20–30% of the additive sum** in-window;
  i.e. ~70–80% of the additively-predicted gap is erased by the
  interference. That is the quantitative content of branch (i).
- **Vector account (B+C)**: exact by construction — including I is what
  removes the overprediction.

**Unit-honesty caveat (load-bearing for the manuscript):** the
quadratic scalar ledger S+F+I itself *under*predicts in-window
magnitudes ~3–4× (median ratio 0.24×) because |B⊥|, |C⊥| ≈ 0.5–1.0 G
there — the small-perturbation truncation is out of its domain. So:
use S/F/I for **sign, decomposition, and window localization**; quote
**magnitudes via the exact h(·) counterfactuals** (h_B, h_C, h(B+C) are
all in `ledger.json`). Do not cite S/F/I values as gap magnitudes.

## Manuscript status

APPLIED 2026-07-31 — ledger + branch-(i) resolution written into §4.6
(two new paragraphs after the fork text), the full ledger + instrument
details added as an appendix table (`tab:e9ledger`), the Goldilocks
localization scored per AQ4 in `paper/review/additional_question.md`,
and the §4.6 + §4.5 `[pending]` markers stripped.
