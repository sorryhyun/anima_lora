# E15 — two-level unbiased demotion: MLMC roulette with a model-priced coin

| | |
|---|---|
| **Status** | **15.0 DONE 2026-07-31 — GATE FAILED, line priced out.** 15.1/15.2 not run (pre-registered kill). Run: `runs/20260731-2114-e14-pricing/`. Renumbered from **E14** 2026-08-01 (the new E14 is the low-σ vector ledger); the run dir and its envelope keep the historical `e14` label |
| **Verdict** | The per-sample correction second moment is ~**half the intrinsic per-step variance** (E2 ≈ V_total/2 per bin, lower bound), so the 1/q roulette is unaffordable: A needs q̄ ≈ 0.45 at the 1.5× inflation cap → net **+2.6–4.1%** vs the ≥20% gate. In-window safety of static demotion is an **aggregate-coherence** phenomenon — per-sample differences are O(‖g‖) and cancel across images/steps; MLMC pays the per-sample second moment to buy back a tiny aggregate first moment. Redesign lever recorded below (anchored control variate), not pursued under this record. |
| **Question** | The map certifies one route in one window by *measurement*. Can a randomized two-level correction make demotion unbiased **by construction** at every σ — turning the gap map from a gatekeeper into a *price list* — and buy the routes the map rejects (768; all-σ) at their measured E4 discount? |
| **Depends on** | [E9](../e9/) (arm sums → ‖Δḡ‖ per bin; the I-switch-off mechanism), [E5](../e5/) (the X-form gap model that prices the coin), [E4](../e4/) (cost calibration, the already-measured static-768 control, the seed-noise yardstick, `claim_accumulated_bias.md`), [E1](../e1/) (paired debiased object, kernel-path rule). [E13](../e13/) (DONE 2026-08-01 — resolved the curve ends: the mid-σ peak is a real plateau, 896's σ→1 approach an instrument limit) did not block. |
| **Instrument** | trainer delta in `train.py::_maybe_sigma_demote` (below) + `bench/compare_ckpt_dw.py` (15.1) + the E4 harness (`e4_flops.py`, `e4_seed_yardstick.py`, `e4_render_eval.py`) (15.2) |
| **In the paper** | §5 discussion paragraph at minimum ("the gap map doubles as the variance/price map of an unbiased estimator"); the method itself is line work beyond the current manuscript |

## The estimator

On an eligible step, demote always; with probability q also compute the
native gradient and add the inflated correction:

> ĝ = g_dem + (𝟙[U < q]/q) · (g_nat − g_dem),  E[ĝ] = g_nat exactly, at every σ.

Rhee–Glynn / Russian-roulette debiasing — the two-level case of unbiased
MLMC. Safety stops being a measured property of a (route, window) cell
and becomes algebra; the map's job shifts to *pricing*: the variance
added is (1/q − 1)·E‖g_nat − g_dem‖², and ‖Δg(σ)‖ is exactly what the
probe line has been measuring all along.

Two structural free lunches:

- **CRN pairing is automatic.** On a correction step the two gradients
  share batch, ε, and σ — only the latent grid is swapped. The
  difference is the *paired* object of the whole probe program, i.e.
  the small one (E9: the realized in-window difference is 20–30% of
  the additive |B|+|C| sum, because I < 0 cancels it).
- **Peak VRAM = native peak.** Implement as two sequential backwards
  with loss weights (1 − 1/q)·L_dem then (1/q)·L_nat (grads accumulate
  in `.grad`; the negative first weight is fine for autograd). Never
  hold both graphs.

## What it buys — priced from E4's measured FLOPs

Per-step relative cost derived from E4 (`e4_flops.py`, n=3):
c_896 = 1 − 0.308 = **0.692** (896only, 100% demote), c_768 = 1 − 0.522
= **0.478** (unsafe768). In-window mass at this recipe m = 244/480 =
0.508. A correction step adds one native fwd+bwd (relative cost 1), so
net FLOPs saving = (demoted mass)·(1 − c_e) − (mass)·q̄.

| candidate | route / gate | gross | net @ q̄=0.1 | net @ q̄=0.2 | static status today |
|---|---|---|---|---|---|
| **A** | 768 @ σ>0.5 + MLMC | −26.5% | **−21.4%** | −16.4% | `sigma768` measured, **below the yardstick on ≥1 corpus** (E4 5-arm) — the rescue target |
| **B** | 896 all-σ + MLMC | −30.8% | −20.8% | −10.8% | `896only` measured, below yardstick on ≥1 corpus; dissolves E4's open σ-gate tension into allocation |
| **C** | 768 all-σ + MLMC | −52.2% | (needs q̄(σ) ≥ ~0.3 at low σ) | — | `unsafe768` = the review's negative control |
| — | shipped `sigma896` | −15.1% | — | — | the bar to beat |

Break-even vs shipped for A at q̄ ≈ 0.22. B and C spend corrections in
the low-σ region priced by the mid-σ peak (E13 subsequently confirmed
the peak as a real plateau in raw paired units — not
estimator-manufactured); 15.0 scores all three, prior on A.

## The coin — three tiers

**q0 — constant.** Baseline; the variance-optimal σ-blind choice.

**q1(σ) — model-priced.** The variance-under-cost-constraint optimum is
Neyman-style: q\*(σ) ∝ √E[‖Δg(σ)‖²] / √(correction cost). E5's
headline X form supplies precisely the needed quantity — its empirical
ingredient **is** the linear mismatch loading ‖δg⊥‖ = c·m(σ) (held-out
RMSE ~0.09) — and the *posterior* second moment √(μ̂²(σ) + s²(σ))
replaces the point estimate, with s(σ) from the fit covariance plus a
structural inflation where the curve was then unresolved (E13 has since
resolved both spots: the mid-σ flank is a real plateau, and 896's σ→1
approach is certified an instrument limit — H3). So "sample more where the model is
uncertain" is not a heuristic bolt-on; it falls out of the optimal
allocation. Mechanism agreement worth pre-registering: E9's I_768 goes
−0.31 → −0.014 window→endpoint, so q1 should *rise toward σ=1* — the
corrections concentrate exactly where the interference protection
switches off. (E13 H1 later falsified the premise: the measured curve
is flat inside the dense-high window — the predicted monotone rise is
absent, and the endpoint gap is a separate estimand.) Frozen as
`q_table.json` in the manifest; deterministic
and CRN-compatible across arms.

**q2(σ, optimizer state) — drift-priced.** Gradient-space unbiasedness
is not update-space unbiasedness: AdamW preconditions by 1/(√v̂+ε), so
a raw-space bias component landing on low-v̂ coordinates moves weights
disproportionately — and *accumulated drift in update space* is
exactly the object `claim_accumulated_bias.md` says the per-step
verdict does not certify. The drift-priced coin scales q1 by
ρ_P = ‖P_t Δg‖ / ‖Δg‖ (norms relative to their preconditioned/raw
gradient scales), estimated online for free: every coin-hit step
yields an actual Δg sample, so an EMA of ρ_P costs nothing extra.
**Scoped as telemetry-first**: a trajectory-dependent coin breaks CRN
pairing across arms and muddies pre-registration, so 15.1 only *logs*
ρ_P(σ); q2 is promoted to a 15.2 arm iff ρ_P is structured in σ
(decision rule below), else the finding is "raw-space pricing
suffices" and q2 dies without an arm.

## Groundings

| # | design choice | grounded by | pointer |
|---|---|---|---|
| g1 | the correction is cheap in-window (variance ∝ realized ‖B+C‖, not additive) | I < 0 cancels 70–80% of the additive gap in-window; realized = h(B+C) exactly | E9 `ledger.json`, `bench/results/20260731-0721/` |
| g2 | corrections should concentrate σ→1 | I_768 −0.31 → −0.014 window→endpoint; endpoint gaps real (896 gap_∞ +0.019, 768 +0.092±0.012 debiased) — the implied σ→1 *rise* later falsified (E13 H1: high-σ flat) | E9; E1a/E1b; E13 free reanalysis table |
| g3 | q1's magnitude model exists and predicts held-out | X form, ‖δg⊥‖ = c·m(σ); RMSE ~0.09; floor law 2-for-2 on unseen floors | E5 `runs/20260729-1130-e5-holdout/`, `-refit/` |
| g4 | the static candidates' costs and failures are measured | −30.8% (896only), −52.2% (unsafe768), sigma768 −26.5% gross; sigma768/896only below yardstick on ≥1 corpus; sigma896 inside | E4 FLOPs table + `runs/20260730-e4-yardstick-5arm/` |
| g5 | the drift-risk object is already named | in-band per-step verdicts do not certify accumulated update-space drift | `../e4/claim_accumulated_bias.md` |
| g6 | wiring is a delta, not new plumbing | per-batch σ-first gate, sibling `demoted_{H}x{W}` keys, `--sigma_lowres_route` (768 arms already ran), token-budget union, seed-keyed σ stream (identical demote set across arms) | `methods.md` Phase 1b; E4 commit `5b63ebb9`, in-vivo CRN check |
| g7 | paired deterministic ΔW comparison has no chaos floor | twin runs bit-identical over 1200 compiled steps with `--deterministic`; nondeterministic floor 0.413 | `methods.md`; `bench/compare_ckpt_dw.py` |
| g8 | clipping would re-bias | `max_grad_norm` defaults 1.0, applied to the *combined* grad | `library/config/cli_args.py:86`, `library/training/loop.py:481` |
| g9 | mean-difference norms are recoverable without GPU; per-draw second moments are NOT | E9 kept arm sums, not per-draw vectors | E9 `--keep_arm_sums`; honesty item h4 |

## Phases

### 15.0 — pricing, free (no GPU)

From E9 arm sums: ‖B+C‖ per (route, bin) → predicted variance inflation
V(q)/V₀ and expected cost per candidate × coin tier; E5-X fit +
covariance → the q1 table. Emit `q_table.json` + a candidate-ranking
`result.json` into `runs/`.

- **Gate to 15.1**: some candidate has predicted net ≥ **−20% FLOPs at
  ≤ 1.5× variance inflation**. Else the line closes as "priced out" —
  itself a publishable one-liner (the map prices the estimator out).
- Stated limit (h4): arm sums give the *mean*-difference norm; per-draw
  dispersion around it is not in the stored vectors, so 15.0 variance
  predictions are lower bounds. 15.1 measures the realized number.

### 15.1 — in-vivo estimator validation (~40 min GPU)

Three deterministic paired twins at E4 tenth scale (480 steps, bs 1,
`--deterministic --paired_step_rng`, stock lora recipe, hews):
**native / static-768@σ>0.5 / MLMC-768@σ>0.5 (q1)**. Coin on its own
seed-keyed RNG stream (σ-stream precedent) so the hit set is
arm-reproducible. All arms `--max_grad_norm 0` (h1).

Pre-registered:

- **H1 (rescue, primary):** cos(ΔW_mlmc, ΔW_native) **>**
  cos(ΔW_static768, ΔW_native) via `compare_ckpt_dw.py` (global + depth
  profile). *Falsifies*: ordering flat or reversed ⇒ per-step
  unbiasedness does not survive the optimizer at this q — jump to h2
  diagnostics before burning 15.2.
- **H2 (variance):** realized correction-term second moment within
  ~2× of 15.0's lower bound. *Falsifies*: ≫2× ⇒ per-draw dispersion
  dominates the mean difference; reprice (q up, net savings down)
  and re-run the 15.0 gate.
- **H3 (drift telemetry, decides q2):** log ρ_P(σ) on every coin hit.
  *Promote q2* iff ρ_P varies ≥ 2× across σ bins with a stable shape
  across the run; else q2 dies here.
- Free telemetry: fraction-of-steps-corrected vs q_table prediction;
  `norm/avg_grad_norm` native clip-vs-noclip check (h1's side
  condition: confirm clipping was inert for the native arm at this
  recipe, so `--max_grad_norm 0` is not itself a confound).

### 15.2 — the E4-protocol A/B (~2–3 h GPU + evals)

Winner candidate only. Arms: **native / sigma896 (shipped) /
MLMC-winner (q1)** [+ q2 iff H3 promoted] × 3 seeds × 2 artists (E4
grid: hews 60-stem/8ep, channel 15-stem/32ep, 480 steps). Frozen
manifest first (`e4_manifest.py` pattern), q_table + coin seeds inside.

- **H4 (headline):** the MLMC arm lands **at-or-inside the seed-noise
  yardstick on both corpora** — the bar its static sibling measurably
  failed — at net wall ≤ −20%. *Confirms*: routes off the map are
  purchasable at list price. *Falsifies*: below yardstick where
  static also fell ⇒ record "estimator-level unbiasedness does not
  transfer through AdamW at practical q" — a real finding; the coin
  tiers (q floor, q2) are the only retry levers, one retry max.
- **H5 (no optimizer pathology):** no loss excursions time-locked to
  coin hits beyond the paired native band; `norm/avg_grad_norm` tail
  bounded by 1/q · native tail.
- **Kill switch:** if H4 fails with H1 having passed, the gap is
  between single-tenth ΔW fidelity and full-recipe rendering — write
  the negative result into `claim_accumulated_bias.md`'s ledger and
  stop; do not tune q by render iteration (that reintroduces the
  seed-lottery reading E4 exists to prevent).

## 15.0 results (2026-07-31, `runs/20260731-2114-e14-pricing/`, `e15_price.py`)

**Gate FAILED under both accountings, which agree.** Two independent
reads, ambiguities resolved in opposite directions:

- *E1b conservative* (per-image relative units, within-image-only
  baseline, N=40, full σ range): A best net **+4.1% @ q̄ 0.44**,
  B −25.2% @ 0.56, C infeasible at any q̄ ≤ 0.6.
- *E9 one-process optimistic* (absolute units vs ‖μ̄‖², **full**
  baseline = between-image + within-image, lower-bound numerator —
  every ambiguity favors the estimator): A best net **+2.6% @ q̄ 0.47**.

| bin σ | ‖μ̄‖ | between | within | V_total | E2(768), LB |
|---|---|---|---|---|---|
| 0.5625 | 0.061 | 120 | 211 | 330 | 189 |
| 0.6875 | 0.096 | 242 | 187 | 430 | 254 |
| 0.8125 | 0.198 | 101 | 90 | 191 | 89 |
| 0.9375 | 1.078 | 101 | 46 | 147 | 43 |

**The mechanism (the real yield of 15.0).** The map's in-window
"gap ≈ 0" is a *cosine-of-aggregates* statement. At the per-sample
level the difference is O(‖g‖): per-image rel-diff ‖Δᵢ‖/‖gᵢ‖ ≈
0.7–1.6 in-window (E1b law-of-cosines), vs aggregate ‖Δḡ‖/‖ḡ‖ ≈
0.15–0.35 (E9 exact vectors) — a ~3× coherence gap, so per-sample
differences mostly cancel across images and steps. Static demotion is
safe *because training averages them* (an **amplitude** statement:
E22.4 later showed the B/C cancellation *angle* holds per-sample at
σ = 0.7) — and that is precisely why
roulette debiasing is priced out: the correction term carries the
per-sample second moment (≈ V_total/2, before per-draw dispersion up
to 8× worse) while the bias it removes is the tiny aggregate first
moment. E[ĝ] = g_nat is bought at a variance price the map's cosine
units never showed. The estimator isn't wrong; it's expensive in
exactly the regime where it's unnecessary, and unaffordable (512-like
rel-diffs 2–5) where it would matter.

**Pre-registered consequence honored:** 15.1/15.2 do not run; no q
tuning beyond the frozen sweep. Deviation recorded in `result.json`:
q1 was data-priced from E1b/E9 vectors rather than model-priced from
E5-X (the full σ range turned out to be measured; E5-X stays the
extrapolation tool for unmeasured tiers).

**Redesign lever (out of scope here, needs its own record):** debias
the *mean*, not each step — an anchored control variate (SVRG-flavor):
estimate μ_Δ(σ) on occasional paired steps *without* 1/q inflation,
add μ_Δ deterministically on every demoted step. Deterministic
additive term ⇒ near-zero added variance; the price moves from
variance to *staleness bias* of the anchor, which is where the q2
drift-pricing idea (AdamW-preconditioned norm) naturally re-enters as
the refresh trigger. The 15.0 tables above are exactly the inputs that
would price it.

## Hazards, pre-registered

- **h1 — clipping re-biases.** `clip_grad_norm_` on the combined grad
  fires preferentially on 1/q-inflated coin steps ⇒ E[α_t·ĝ] ≠ g_nat.
  All arms run `--max_grad_norm 0`, with the native clip-inertness
  check above. If clipping is *not* inert at this recipe, redesign
  (clip-aware coin) before any arm burns.
- **h2 — Adam nonlinearity.** Unbiasedness is gradient-space; v̂
  spikes from rare 1/q hits transiently deflate effective LR on hit
  coordinates. Floor q ≥ 0.05; H3/H5 telemetry is the watch.
- **h3 — estimand drift vs probe data.** Trainer g_dem includes
  yarnsig rope; E9's Δ data is plain-demote. 15.0 prices are
  approximate; 15.1 measures the wired estimand.
- **h4 — variance honesty.** See 15.0 stated limit.
- **h5 — kernel-path chaos.** Every paired read in-process (E1 rule);
  the 15.1 twins are one process per arm with `--deterministic`, so
  cross-arm ΔW comparison is chaos-floor-free (g7).
- **h6 — loss-log pollution.** progress.jsonl gets raw L_dem; the
  weighted correction magnitude logs to its own tag
  (`sigma_lowres/mlmc_corr`), else `run-status` losses look insane on
  coin steps.
- **h7 — bs>1 gate conservatism** unchanged (every-sample-σ gate);
  exact at bs 1, which is what 15.1/15.2 run.

## Implementation deltas

1. `_maybe_sigma_demote`: coin draw from a dedicated seed-keyed
   generator; on hit, flag the step for the second (native) pass.
2. Loop: two-backward combination with weights (1−1/q, 1/q) on hit
   steps; grad-accum interaction = none new (weights are per-loss).
3. `--sigma_lowres_mlmc q|path/to/q_table.json` (scalar = q0; table =
   q1/q2), `--sigma_lowres_route` reused for 768 (exists, g6).
4. Token budget already unions demoted counts; a hit step runs both
   token counts — both already inside the dynamic-seq range.
5. 768 sibling keys: `SIGMA_DEMOTE="1024:768"` emit pass (already
   exercised by E4's 768 arms).
6. ρ_P telemetry hook reading Adam `exp_avg_sq` on hit steps (q2 EMA
   lives here later; 15.1 logs only).

## Run commands (sketch — pin at manifest freeze)

```bash
# 15.0 — free
uv run python project/sigma_lowres/paper_bench/experiments/e15/e15_price.py \
  --ledger bench/results/20260731-0721/ledger.json \
  --e5_fit project/sigma_lowres/paper_bench/runs/20260729-1322-e5-refit/ \
  --results_root project/sigma_lowres/paper_bench/runs

# 15.1 — three twins through the daemon (GPU work never in background bash)
make lora --queue ARGS="--sigma_lowres --sigma_lowres_route 1024:768 \
  --sigma_lowres_mlmc runs/<15.0>/q_table.json \
  --deterministic --paired_step_rng --max_grad_norm 0 ..."   # + native & static twins
```

## What lands in the paper

The kill branch fired, and it earns *more* than the planned paragraph:

- §5 discussion: the map prices an unbiased two-level estimator — and
  prices it **out** (A: net +2.6% at the inflation cap vs −15.1%
  shipped). Measurement as pricing, with a number.
- The aggregate-coherence finding is a genuine sharpening of the
  paper's account: the certified safety is a property of the
  σ-averaged aggregate gradient, not of per-step gradients (per-sample
  ‖Δ‖ ≈ 0.7–1.6‖g‖ in-window vs aggregate 0.15–0.35; amplitude only —
  the cancellation angle itself is per-sample, E22.4). This gives
  `claim_accumulated_bias.md` its cleanest statement of *why* the
  per-step-vs-accumulated distinction matters, and independently
  motivates why the E4 A/B — not any per-step read — was the right
  certification instrument.
