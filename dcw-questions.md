# DCW learnable calibrator — open questions

Things to dive into to decide whether v4 (and any v5) is feasible, and where to spend the next polish budget. Excludes data-crawling — that's a matter of time. Ordered roughly by leverage.

Conventions: "the pool" = the 525-row clean pool (5 aspects × 35 prompts × 3 seeds). r_α = CV r between α̂ and per-row integrated tail residual. r_σ = CV r between σ̂ and per-prompt seed std.

---

## 1. Is r_α actually a quality proxy?

The whole research program rates v4 by per-row integrated tail residual r — but FM val loss already failed as a quality proxy on Anima (memory `project_fm_val_loss_uninformative`), and DCW's own scalar ablation showed band-mask choice could move ∫|gap| by 30% without an obvious perceptual win on Anima samples. r_α improving from 0.474 → 0.518 (`7:` → `7:14`) is meaningless if a 9% relative r-lift translates into ~0% perceptual lift.

**Status update 2026-05-05** — first perceptual A/B has been run (one cell only: 832×1248, n_images=8, n_seeds=2, baseline vs λ=+0.01 LL one_minus_sigma at CFG=4). Positive λ recovered missing details perceptually, confirming that integrated-|gap| does track quality at this cell (memory `project_dcw_perceptual_cfg4_confirmed`). So the methodology question is closed (`scripts/dcw/measure_bias.py --save_images` works); the data-collection question — does this hold across LoRAs / aspects / styles? — is now the bottleneck for Gate C.

- **Experiment**: 12-prompt × 4-LoRA × 2-aspect side-by-side: scalar (-0.015, LL) vs v4-prod (`7:`) vs v4-tw7-14 vs no-DCW. Score per-prompt only after the panel is shuffled. If v4 ≤ scalar perceptually, the entire calibration superstructure is overhead with no payoff. The shipped scalar `−0.015` is now known to be wrong-direction at CFG=4 non-square — include `λ = +bucket_prior` (positive per-aspect) as an additional baseline.
- **Sub-question**: is there a *band-of-r_α* below which the lift is invisible? If perceptual delta only shows up past r_α ≈ 0.7, none of the current architectures clear the bar.

## 2. Is 35 × 3 really small? — *not for α̂, but devastating for σ̂²*

The user's framing. Decompose:

- **For α̂ (per-row mean prediction)**: 525 rows on 285k params is heavy overparam (param:row ≈ 540:1) but CV r=0.55 holds out, and the prototype passed at 176 rows. Mean prediction *is* learnable at this scale. The ceiling is feature/architecture, not n.
- **For σ̂² (per-prompt variance prediction)**: per-prompt label is `Var_seed(r_p,s)` from **n=3 seeds**. Sample-variance standard error scales as √(2/(n−1)) ≈ 1.0 at n=3 — the *target itself is 100% noise* before any feature work. r_σ ceiling at n=3 is bounded by `σ_signal / √(σ_signal² + σ_label²)` which for the observed 13% seed share is structurally < 0.5 even with a perfect head. **Gate B's 0.4 threshold is approximately *at* the n=3 ceiling.**
- **Concrete experiment**: simulate the σ̂² CV r ceiling as a function of seeds-per-prompt holding total budget fixed. Compare 35 prompts × 3 seeds (current) vs 21 prompts × 5 seeds vs 12 prompts × 9 seeds at the same 105 rows/aspect. If r_σ ceiling rises monotonically with seeds-per-prompt, the σ̂² channel needs *fewer prompts but more seeds*, not the proposal's 200×3.

## 3. Is g_obs the head, or is the head a g_obs detector?

`--no_g_obs` ablation drops r_α 44–49% across every window. r_σ collapses to ~0. **Most of v4's signal is the network re-discovering "trajectory commits early"**, not learning a prompt → calibration map. This may be fine — we still get a usable predictor — but it raises:

- Is c_pool / aspect contributing anything *orthogonal* to g_obs, or just compensating for noise in g_obs? Run the symmetric ablation: **g_obs-only head** (drop c_pool + aspect + aux). If r_α stays at 0.50, the prompt channel is dead and the architecture should be reframed as "online AR(1) extrapolator with prompt as a regularizer".
- If g_obs alone gets us 80% of the way, can we replace the 285k-param MLP with a closed-form AR(1) extrapolator + a small residual head? That saves the inference parameter count and makes calibration much cheaper.

## 4. Supervise on `:4` / `:7`, apply to whole trajectory — does end-to-end Gate C beat tail-supervision?

Reframed: the r_α "hot zone" at `7:14` / `14:21` / `21:` (r ≈ 0.52–0.55) is a **metric artifact** — late-trajectory tails are smoother, easier to predict, but predicting them well doesn't make a better corrector. The calibrator's actual job is to learn the *early-commitment* signal (where seed-vs-prompt is decided) and emit a single λ that's then applied across the whole trajectory; AR(1) propagation handles the rest. Both α̂ and σ̂² should be supervised on `:4` or `:7`.

- This collapses the previous "split-window" question (different windows for α̂ vs σ̂²) — both want the early window. r_σ already peaks at `:7` (+0.44); r_α is *lower* there (+0.31) but that's the right thing to predict, not a defect.
- **Experiment**: train the head with `--target_window :7` (and `:4`), then run end-to-end perceptual A/B against the current `7:`-supervised production head. Decision gate is Gate C, not r_α. Decoupling supervision window from inference application window may also be needed (currently coupled via artifact metadata) — check `lambda_for_step` reads `target_start`/`target_end` independently from how it should *apply* λ.
- If `:7` supervision underperforms `7:` on Gate C, it means AR(1) propagation isn't doing the work we thought, and the v4 mental model (early observation → tail correction) needs revisiting. If it matches or wins, drop the mid/late framing entirely.

## 5. The "13% seed-noise floor" — is it a hard r_α ceiling or a budget question?

From step ≥6 onwards the per-step seed variance share stabilises at 13–14%; integrated tail target inherits the same 13.5%. The naive r upper bound is `√(1 − 0.135) ≈ 0.93`, but realised CV r maxes out at 0.546 — a factor of 1.7× gap.

- Is the gap **feature-side** (c_pool can't represent enough to close it), **supervision-side** (single-seed labels still inject within-cell noise even at 13% floor), or **architecture-side** (one shared MLP)? The g_obs ablation says feature/supervision dominate, but doesn't separate them.
- Concrete diagnostic: **train on seed-mean labels at 12 prompts × 9 seeds** (same row count). If r jumps from 0.55 → 0.75+, the ceiling was supervision noise, not feature ceiling. If it stays at 0.55, the features are spent.

## 6. Is c_pool ever load-bearing, or does aspect + g_obs subsume it?

The c_pool feasibility study at n=16 found *no* simple scalar > r=0.30 against per-prompt amplitude, with caption length the strongest at +0.26. At n=525 we never re-ran the same diagnostic — we just trained the full MLP. Possible state of the world: **the head is learning caption-length and centroid-cosine through c_pool**, and the rest of the 1024-dim is dead weight that the L2 reg holds near zero.

- **Experiment**: feature-importance on the trained head. (a) Train on c_pool zero'd. (b) Train on c_pool replaced by `[caption_length, cos(c_pool, μ_centroid), token_l2_std]` only. If (b) ≥ (a), the 1024-dim `c_pool` is doing nothing the 3 aux scalars don't already do. That collapses the head to ~5k params and lets us train it on n=200.
- This question is the cheap version of #5 — answers "is the feature space actually rich enough to scale signal with data".

## 7. Does v4 produce sign-correct α̂ on flat-style prompts?

The flat-style failure (channel/caststation over-sharpening under scalar `λ = -0.015`) was previously framed as "c_pool can't see style". Reframed: cross-attention embeddings demonstrably represent artist style (otherwise generation couldn't render it), so the failure is **scalar-uniformity**, not feature-representation. At CFG=4 production the optimal λ varies in sign across (CFG × aspect × prompt) cells (Q1 of `dcw-research.md`); a constant negative scalar is wrong-direction on cells where the gap is positive. v4's per-prompt α̂ should fix this by construction.

- **Validation, not research**: 8 known-flat prompts × 3 seeds through the v4 controller. Read α̂ from the tqdm trace. If it's small/correct-sign on flat prompts and standard-correction on detail-dense ones, the failure is closed and the caption-length backstop (`tau_short`) becomes optional.
- If α̂ still comes out wrong-sign on flat prompts despite cross-attention representing style, *then* the question reopens — but as a "head capacity / loss" issue, not a feature-representation one.

## 8. The bucket-prior sign-flip is structural — does v4's α̂·μ_g/Σμ_g distribution inherit it?

A2 found `S_pop` sign-flips mid → late within every aspect at CFG=4: a single λ_scalar can either zero ∫g OR minimise per-step |g|, not both. v4 distributes a *scalar* α̂ across tail steps weighted by μ_g, with no S_pop dependency since `9f02612`. Same monotone-in-i shape — so v4 inherits the same structural limit on per-step gap reduction even with a perfectly calibrated α̂.

- Have we measured **per-step |gap| under v4 with the new λ formula** (CFG=4, n=24)? If late-step |gap| comes down but mid-step |gap| goes up, we've moved error around, not removed it.
- Does the right answer require a **per-step head** (output (α̂_0, …, α̂_{N-1}) instead of scalar α̂), even though PC2 was only 0.95% of variance? PC2 share is averaged across prompts; for the *sign-flipping subset* of trajectories it could be much higher.

## 9. NLL is a wrong objective when σ̂² is unreliable

Every target window in the k-sup sweep has net-negative NLL — composite shrinkage hurts. We've responded by disabling shrinkage at inference, but **we're still training the head with Gaussian NLL loss**. That loss couples α̂ to log σ̂², and a miscalibrated σ̂² can pull α̂ away from MSE-optimal.

- **Experiment**: train identical architecture with plain MSE on α̂ (single-output head, no σ̂²). Compare CV r_α at every window. If MSE matches NLL, σ̂² is purely deadweight in the current regime — drop it from the architecture entirely until A3-scale data, instead of training-then-disabling.
- Companion: re-test the per-prompt aggregate σ_aux loss (already in the trainer) at varying `lambda_sigma_aux` weights. The current value is implicit; sweeping it might actually let σ̂² learn instead of training-then-clamping.

## 10. What's the realistic perceptual cost-benefit?

10–15% wall-time overhead is paid every inference call once v4 ships. The break-even depends on the perceptual lift over scalar DCW (which is itself ~free). Even if Gates A/B/C all pass, what's the *minimum perceptual lift* worth the +12% cost?

- Decide the threshold *before* running Gate C. If the team's threshold is ≥ 60% prefer-v4, we should know the answer before investing more in v5 architecture (split-window heads, multi-band g_obs).
- **Cheaper alternative to consider explicitly**: per-aspect *scalar* λ (just three numbers, baked at calibration time, zero overhead). The v4 pitch over scalar is per-prompt steering. If g_obs is doing all the heavy lifting (#3), per-prompt steering may not be load-bearing — and per-aspect scalar may already be ~80% of v4's perceptual win at 0% cost.

---

## Suggested order of attack

1. **#1** (Gate C, ~1 day): is anything we're doing perceptually visible at all?
2. **#2 sample-size simulation** + **#5 seed-mean CV** (~1 day, no new data): caps r_σ and r_α ceilings on existing data — tells us if v5 should buy more seeds or more prompts.
3. **#3 g_obs-only ablation** + **#6 caption-length-only head** (~half a day): tells us whether c_pool's 1024 dims are doing real work.
4. **#9 MSE-only head** (~half a day): is NLL even the right loss at current scale?
5. **#4 split-window** + **#8 per-step head** (~1 day each): the two architecture moves likeliest to break the 0.55 r_α ceiling without changing data scale.
6. **#7 flat-style sanity check** (~half a day): just a v4 α̂ readout on 8 flat-style prompts. Confirms the scalar-uniformity fix actually fixes the user-visible failure.
7. **#10 cost-benefit threshold** is the gate-keeping question for #5–#6 effort.

Anything <0.55 r_α at this point is fighting noise. The upside of these questions is usually "stop doing the wrong thing", not "find a new feature".
