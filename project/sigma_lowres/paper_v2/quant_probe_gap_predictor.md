# Quant reference perturbations + the learned gap certifier (proposal)

Status: **DRAFT 2026-08-13 — nothing frozen, nothing runnable licensed.**
Per the roadmap rule, every runnable stage below needs its own
pre-registration under `../paper_bench/experiments/` before GPU or
verdict; this doc orders the candidates and records the rationale before
those registrations exist. Origin: user direction 2026-08-13, growing
out of the E25-family closure (the conditioning *lever* is dead for
ship; the surviving use of that line's harness is *prediction*, offline).

One sentence: **use fake-quantization as a family of controlled
reference perturbations to (a) make the two-term gap model's fitted
quantities measurable, and (b) feed a small physics-structured predictor
that certifies routing windows for new setups without an arm campaign.**

## Premise sources (committed facts this builds on)

- **Two-term reduction** (`sec_theory.tex` Eq. twoterm): the shipped
  effective law is gap ≈ ½·a_e²·(‖Δr̄(σ)‖/‖ḡ_src(σ)‖)² + floor_e, with
  a_e and the floor **fitted per route**, and the floor's two-part story
  (RoPE_e + Resid_e) *written but not separately measured*. The spectral
  rival account carries one free parameter (tolerance δ) and an
  uncalibrated Jacobian gain ("carrying it into gradient units costs one
  calibrated gain, about which a spectral model of the data says
  nothing").
- **B/C identity + near-cancellation** (E19/E24, `sec_theory.tex`
  Eq. branches): both legs are measurable arm differences; measured gap
  is the residual of a deep anti-alignment.
- **E25b/E25e closure** (`../paper_bench/experiments/e25e/`): the
  conditioning lever absorbs the per-step graph-leg substitution at the
  gradient level (25b Stage-1 IMPROVES) but the endpoint diverges under
  *any* live delta on demote steps — 25e removed the common-mode channel
  structurally and the collapse persisted (ΔW cos 0.44/0.52 vs combo
  0.74/0.77, both corpora). Training-time correction is closed;
  **offline prediction is the surviving use.** What carries over is the
  harness (per-step (route, σ) instrumentation), NOT the trained
  weights: the learned differential was ~90 % common-mode in 25b and
  seed-variable (0.003–0.077) in 25e — not a usable backbone.
- **Environment wall** (roadmap §3 T0): cross-boot *direction* reads are
  dead (0.32–0.47); h-unit / ρ scalars are boot-portable
  (Δρ ≈ 0.03–0.05). Any predictor target must be a **scalar**.
- **ΔW closeness ≠ render closeness, measured three times** (25b, 25e
  in both directions): a per-step gap predictor certifies *per-step
  substitution*, never ship quality. The claim boundary is recorded here
  once and inherited by every stage.
- **Free labels already paid for**: the committed scalar tables —
  E24 verdict tables (e193/e194), `e26_grid_across_sigma.json` scalars,
  `e28_read768.json`, `e28f1_read.json` — survive store reclamation and
  are boot-portable. A first predictor fit costs zero GPU.
- **E20.4 known-input kill** (line-wide guard): measured/estimated
  quantities must not become training-time embedding inputs. The
  predictor is therefore an **offline, run-start certifier** — its
  output picks routing windows *before* the run; it never feeds back
  into a live training step.

## Object 1 — fake-quant as a reference perturbation family

All probes are **fake-quant** (quantize→dequantize values, full-precision
arithmetic). Real low-precision kernels would re-couple the value change
with the accumulation change and destroy the decomposition the probes
exist to read. Four instruments, in dependency order:

### Q-data — input-latent fake-quant at native grid

Data-side information deletion along the **amplitude** axis (downscale
deletes along the frequency axis). Estimand: cos(B̂_e, Q̂_data) per
verdict bin, debiased, within-run. Reading:

- **High** ⇒ the adapter's gradient response to data-side information
  deletion is intervention-generic (funnels through a common direction
  with amplitude set by ‖Δr̄‖). Assumption (iv)'s route amplitude a_e is
  then promoted from per-route fit toward an **intervention-independent
  constant** — the predictor gains transfer across interventions,
  including precision routing.
- **Low** ⇒ the data branch is what-was-deleted-specific; the spectral
  account's band-specific narrative gains, and the predictor keeps
  per-route a_e. Either answer is informative; the read is
  prediction-symmetric.

### Q-graph — activation fake-quant at fixed grid and content

A graph-fidelity perturbation that carries **zero** positional /
token-count content by construction. Projecting Ĉ_e onto Q̂_graph splits
the floor into a generic-computation-fidelity component (Resid_e-like)
and a grid-specific remainder (RoPE_e / token-count) — the **first
independent handle** on the floor's written two-part decomposition.
Upgrades the floor from one fitted constant per route toward a measured
sum, partially predictable for unseen routes.

### Bits-dial — amplitude response

Sweep quantization strength (int8 → fp4-class) to inject a **known
residual-level amplitude family** and read the gradient-level response:
an empirical effective Jacobian gain per (σ, route). This is exactly the
quantity the spectral account needs as a hand-calibrated gain, and it
validates the exact-link (Eq. exact) saturation on the amplitude axis.
Doubles as an independent stress test of assumption (iii)
(graph-relative stationarity).

### δ-from-dtype — the parameter-free rival baseline

Tie the spectral account's tolerance δ to the quantization noise floor
the training already tolerates (bf16 ⇒ relative rounding ≈ 2⁻⁸–2⁻⁹).
This upgrades the **rival**, not our model — recorded as such: it makes
the head-to-head scoring parameter-free (its structural failures — total
order, ratio-only, no floor — are then unambiguously formal, not fitting
slack), and it supplies a zero-parameter baseline the learned predictor
must beat to justify existing. Falsifiable side prediction, owned by a
*future* registration, not this doc: fp8 training should **widen** the
certifiable demote window (lower fidelity bar ⇒ earlier safe
substitution).

### Probe caveats (recorded once, inherited by every stage)

- Perturbations must clear finite-draw noise: bf16-scale fake-quant is
  likely invisible at D = 12; expect int8-and-coarser, which puts the
  amplitudes **outside the quadratic domain** — read through h(·)/the
  exact link per the paper's standing unit convention, and always
  through the debiased instrument (Eq. debias).
- All direction reads are **within-run or same-boot** (T0); scalar
  outputs are the deliverable wherever possible.
- Quant noise is signal-dependent; the probes measure the response to
  *this* perturbation family, and the transfer claim (Q-data high-cos ⇒
  intervention-generic) is exactly what the fingerprint is designed to
  test, not assume.

## Object 2 — the learned gap certifier

A small predictor, physics-structured, never a curve-fit black box:

- **Target**: debiased h-unit per-(route, σ-bin) excess-gap scalars.
  Not directions (environment wall), not renders (ΔW ≠ render ×3).
- **Structure**: predict the two-term reduction's coefficients — a_e
  and floor_e — from cheap features, then compose through the exact
  link. Coefficient-level learning is deliberate: the committed label
  set is small (few routes × 5 bins × few adapters/corpora), which a
  two-coefficient model matches and a deep model would overfit.
- **Features** (all cheap, no arm campaign):
  ‖Δr̄(σ)‖/‖ḡ‖ (forward-only — both arms' mean residuals need no
  backward pass), bits-dial gain(σ, route), the Q-graph floor
  component, spectral t_ω(route) at δ_dtype.
- **Labels**: the committed scalar tables (zero GPU); new arm probes
  are spent on **validation only**.
- **Baselines**: must beat δ-from-dtype (parameter-free) and be
  compared against the fitted two-term reduction (the ceiling — it sees
  the labels the predictor must predict).
- **Held-out validation with a pre-registered prediction**: the E28
  **896 cell is still undecided** — the predictor publishes its 896
  numbers *before* any 896 probe runs; a new-corpus cell is the second
  candidate. Prediction-before-measurement is the whole point; a
  post-hoc fit is worthless here.
- **Use**: run-start certification of routing windows for a new
  corpus/adapter — and once-per-**base** if E31 lands base-carried
  (E30/E31 sets the generalization scope; NOT-EXPRESSED shrinks the
  predictor to per-adapter, recorded, not fatal).

## Staging (cost sketch — order, not registration)

| stage | what | cost class |
|---|---|---|
| **Q0** | fake-quant hooks in `bench/sigma_probe/kernel.py` (input-latent + activation sites), reusing the debias/rel machinery; invariant tests; single-bin smoke | CPU + minutes GPU |
| **Q1** | Q-data / Q-graph fingerprints at the 5 verdict bins, 768 + native, one boot, within-run reads | ≈ 3–4 GPU-h |
| **Q2** | bits-dial amplitude response (rides the Q1 store family; more draws if the rel gate demands) | +1–2 GPU-h |
| **P0** | predictor fit on committed scalars + Q1/Q2 features; scored vs both baselines | zero GPU |
| **P1** | held-out validation: pre-registered 896 (or new-corpus) prediction, then the probe | ≈ 3–4 GPU-h |
| **δ-paper** | δ-from-dtype instantiation in the appendix scoring pipeline | CPU only |

Q0/Q1 gate everything: if both fingerprints land ≈ 0 within noise, the
predictor loses its structured features and P0 proceeds on
‖Δr̄‖ + t_ω alone, reported as the weaker form.

## Explicitly NOT this

- **No training-time lever revival** — 25e closed the family for ship;
  E20.4's kill stands: no measured quantity becomes a training input,
  and the certifier's output never touches a live step.
- **No cross-boot direction estimands** (T0 is the single home of the
  rule).
- **No ship/render claims** — certification-level only, stated in every
  eventual registration.
- **No real low-precision kernels inside probes** — fake-quant only.
  Actual fp8 *training* (the δ-window prediction) is a separate future
  line with its own Tier 1.5 obligations.
- **No sweep creep** — bins, routes, and bit-widths freeze at each
  stage's registration with its own multiplicity accounting.

## Kill switches

- Q1 double-null (both fingerprints ≈ 0) ⇒ structured-feature claim
  dies; P0 degrades gracefully and says so.
- P0 fails to beat the parameter-free baseline on held-out tables ⇒ the
  learned model dies; δ-from-dtype stands alone as baseline hardening
  for the paper's scoring section.
- P1 prediction misses ⇒ recorded as the verdict; no re-fit-and-re-predict
  on the same held-out cell.
- E31 NOT-EXPRESSED ⇒ scope shrinks to per-adapter; the proposal
  survives with the narrower claim.
