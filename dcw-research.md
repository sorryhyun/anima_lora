# DCW learnable calibrator — research questions

External-audience version. We're looking for prior work, theoretical bounds, or community experience on the open problems below — *not* for someone to run our experiments. The companion doc [`dcw-questions.md`](./dcw-questions.md) covers internal ablations.

## Context (one paragraph)

We're working on a DiT + flow-matching anime/manga model (Anima). DCW (Yu et al., [arXiv:2604.16044](https://arxiv.org/abs/2604.16044)) is a training-free, sampler-level correction that mixes each Euler step's `prev_sample` toward (or away from) the model's `x0_pred` to close the SNR-t bias. Scalar form (one global λ) ships and helps on detail-dense targets. v4 is a learnable per-trajectory calibrator: a small MLP that takes (aspect prior, mean-pooled cross-attn text embedding `c_pool`, observed LL-band Haar gap over the first k=7 steps) and predicts heteroscedastic `(α̂, log σ̂²)` for the remaining steps, distributed across the tail proportional to a per-aspect μ_g profile. Architecture and gates are in [`docs/proposal/dcw-learnable-calibrator-v4.md`](./docs/proposal/dcw-learnable-calibrator-v4.md); 2026-05-05 follow-ups in [`docs/proposal/dcw-v4-target-window-and-sigma-drop.md`](./docs/proposal/dcw-v4-target-window-and-sigma-drop.md).

Current pool: 525 rows = 5 aspects × 35 prompts × 3 seeds. CV r between α̂ and per-row integrated tail residual (the v4 supervision target) plateaus at ~0.55. r between predicted σ̂ and per-prompt seed std is ~0–0.3 depending on target window. Gate B (NLL improvement) fails. Gate C (perceptual A/B) has not been run yet.

We would value input on the following.

---

## 1. Bias-direction (CFG × aspect) interaction — what governs the sign?

Yu et al.'s Key Finding 2 (`||v_θ(x̂_t)|| > ||v_θ(x_t_fwd)||` at late steps) **does reproduce on Anima at production CFG=4 on non-square aspects, and fails on square + on CFG=1**. Integrated signed LL gap on 28-step baselines:

| Run | Setting | ∫ gap_LL | Direction |
|---|---|---:|---|
| `archive/dcw/results/20260503-1720` | CFG=1, no LoRA, no mod-guidance | **−406** | paper-opposite |
| `bench/dcw/results/20260504-1648` | CFG=4, 1024² | **−188** | paper-opposite |
| `output/dcw/20260505-0130` | CFG=4, 832×1248 (HD portrait) | **+89** | paper-direction |
| `output/dcw/20260505-0612` | CFG=4, 1248×832 (inv-HD landscape) | **+205** | paper-direction |

The shape on the CFG=4 non-square plots matches the paper's Fig 1c almost exactly: brief negative dip at steps 1–3, then `reverse > forward` from step ~6 to end, gap rising to +20 in the LL band by σ→0. The CFG=1 bench (`gap_curves.png` in archive) reverses every late step. The 1024² CFG=4 result sits between — net signed gap negative but per-step shape mixed.

The implication for any sampler-level correction is that **the optimal sign of λ is not a fixed property of the architecture; it depends on guidance strength and on patch geometry**. Our scalar default `λ = -0.015` was tuned on the CFG=1 / no-LoRA bench and shipped before A2 measured the per-aspect CFG=4 baseline. The v4 per-aspect bucket priors flip back to *positive* λ_scalar for non-square (`+0.0059` and `+0.0127`); 1024² stays slightly positive at `+0.0046`.

What we'd appreciate input on:

- Has anyone characterised this **(CFG × aspect)** interaction on a flow-matching DiT? CFG amplifies cross-attention's contribution to `v_θ`, and the *sign* of the resulting velocity-norm gap depending on token-count / patch-count would be a clean theoretical result.
- Is there a published result on how the velocity-norm bias direction depends on the **conditioning-information density** (text-tokens-per-image-patch ratio differs across aspects since images are bucketed to constant 4096 tokens)?
- Three speculative mechanisms in `archive/dcw/README.md` (manifold-mismatch readout, max-padded cross-attention sink, mod-guidance interaction). None tightly explain the (CFG × aspect) interaction. Is there a community consensus on what governs the sign of the SNR-t bias?
- Cleanest empirical test would be reproducing on a non-Anima DiT under matched CFG/aspect settings — but that's out of scope for us. If anyone has a flow-matching DiT with mod-guidance and would dump `gap = ||v_rev|| − ||v_fwd||` per step at CFG ∈ {1, 4} × aspect ∈ {square, non-square}, the cross-architecture point would settle whether this is universal or Anima-specific.
- 2026-05-05 perceptual confirmation (one cell, n_images=8, n_seeds=2): paper-direction positive λ=+0.01 LL one_minus_sigma at CFG=4 832×1248 inv-HD-portrait recovered missing details vs baseline — matches the +89 → −16% integrated-|gap| numerical signal. So the (CFG × aspect) interaction is **perceptually load-bearing**, not just a numerical artifact. The shipped scalar `−0.015` (CFG=1-tuned) pushes the wrong direction at production CFG=4 non-square.

## 2. Heteroscedastic regression when the variance label has SE > signal

The σ̂² head is supervised by per-prompt seed variance from n=3 seeds. Sample-variance standard error scales as √(2/(n−1)) ≈ 1.0 at n=3 — the *target itself is ~100% noise* before any modeling. The Kendall & Gal heteroscedastic NLL ([NeurIPS 2017](https://arxiv.org/abs/1703.04977)) assumes the per-sample label is roughly trustworthy. Empirically our σ̂² channel doesn't learn (r_σ ≈ 0; NLL net-negative at every target window).

- Is there literature on heteroscedastic regression where the variance label itself is variance-estimator noise? "Variance of variance" corrections, debiasing, or hierarchical priors that pool across prompts before fitting?
- Closely-related: bayesian deep learning with extremely noisy aleatoric labels — is there a recommended loss family that doesn't collapse to mean prediction at this regime?
- Aronow & Lee 2013 / Efron 2014 on shrinkage estimators of variance feel related; we haven't found a deep-learning analog.

## 3. Theoretical r ceiling for AR(1)-like prefix → tail extrapolation

Empirically: lag-1 autocorr of LL gap within a trajectory ≈ 0.55; corr(early-dev k=7, late-dev) ≈ 0.89 seed-conditional. Per-step seed share = 73% at step 0, drops to 13–14% by step 6 and stays flat. Production CV r_α plateaus at 0.55 against a naive √(1 − 0.135) ≈ 0.93 noise-floor upper bound — a 1.7× gap.

- For an AR(p) (or general Gaussian-process) trajectory with known autocorrelation and known per-step heteroscedastic noise, what is the **information-theoretic limit on R²** for predicting an integrated tail functional from a fixed-length prefix? Closed-form Kalman / Wiener-filter analog?
- More specifically: under a state-space model where the "trajectory commitment" happens at low-SNR steps (0–6) and is partially observable through the noise, is there a known predictability ceiling that depends on the overlap between the noise-dominated and the state-revealing windows? Our window plot ([`output/dcw/seed_band_curves.png`](./output/dcw/seed_band_curves.png) reproducible via `bench/dcw/plot_seed_band.py`) shows seed share collapsing exactly where prompt cross-attention starts biting.

## 4. Validation metrics for sampler-level corrections

FM val loss does not track perceived sample quality on Anima (lower val loss has not predicted better samples across our runs). r_α on integrated gap is our internal metric, but we don't know whether r_α=0.55 is perceptually meaningful or invisible. FID/KID/CLIPScore are population metrics, slow per-prompt, and don't capture per-trajectory wins. Perceptual A/B is the gold standard but doesn't scale to a 28-step × 8-window × 5-aspect sweep.

- What's the field's current best-practice proxy metric for **sampler-level** modifications (not training-time)? Has anyone published a perceptual-correlated proxy for guidance-style ablations that we could reuse?
- Specifically for diffusion *guidance* literature (CFG schedules, dynamic thresholding, oscillation guidance, CADS) — what metric do reviewers/authors use to argue the change helps? We've found these papers leaning heavily on FID + cherry-picked panels; we'd love a mid-cost reproducible proxy.

## 5. Calibration sample-size scaling laws

For a per-prompt heteroscedastic head on a conditional generative model, are there known empirical scaling laws for (n_prompts, n_seeds_per_prompt)? Our intuition:
- α̂ (mean prediction): more prompts > more seeds, because mean cancels seed noise via averaging.
- σ̂² (variance prediction): more seeds > more prompts at fixed budget, because the per-prompt variance label is the bottleneck.

We have no prior reference for *where the crossover sits* or whether it's even monotone. Has anyone characterised this for diffusion calibrator-style heads, or for analogous heteroscedastic problems (Bayesian neural network calibration on simulator outputs, ensemble distillation with finite-sample variance targets)?

## 6. "Trajectory commits early" — formal characterisation in diffusion

Two empirical findings:
- Step 0 has 73% seed-share; by step 6 it's stable at 13–14%.
- Late-step DCW gap is highly predictable from the first k=7 LL norms (seed-conditional r=0.89).

Combined: the *seed* is doing most of its damage in the very early high-noise steps, and "which mode the trajectory committed to" is mostly settled by step 6. This feels related to:
- DPM-Solver / DDIM observations that early steps determine global structure.
- Karras et al. ([2206.00364](https://arxiv.org/abs/2206.00364)) on noise schedule shape.
- Mode-collapse literature in score-based models.

Is there a theoretical or empirical paper that formalises "commitment time" in score-based / flow-matching samplers? A reduced-dimensionality probe of when seed entropy collapses into a mode, beyond the IS-IS divergence frame?

## 7. Per-step gap derivative sign-flip — known structural property?

We measured the per-step derivative `S_pop[i] := ∂gap[i]/∂λ` under a `λ × (1−σ_i)` correction (anchored at 2 anchors per aspect, n=12×2 each). It **sign-flips mid → late within every aspect**:

```
                 early(0–9)   mid(10–19)   late(20–27)
1024²              +35           +117         −235
832×1248           +43           +208         −150
1248×832           +36           +155         −266
```

Consequence: a single monotone λ schedule cannot simultaneously zero ∫g and minimise per-step |g|. Increasing |λ| at late steps over-corrects mid steps and vice versa.

- Is this a known structural property of flow-matching samplers under CFG, or specific to our band-mask / Haar-LL correction site?
- Does it appear in the analogous literature on dynamic CFG schedules (CFG++ — [Chung et al. 2024](https://arxiv.org/abs/2406.08070)), oscillation guidance, or guidance-interval recent work? They all face the question "does a single monotone schedule suffice?" but rarely measure per-step derivatives at this granularity.

## 8. When is post-hoc correction non-Pareto-improving?

A constant scalar λ is by construction *not* per-prompt — it applies uniformly across cells where the optimal correction varies in sign and magnitude (this connects directly to Q1's CFG × aspect interaction). v4's per-prompt α̂ moves this from "method limitation" to "calibrator quality" question, but only if α̂ predicts the right *sign* on the long tail of out-of-distribution prompts. We don't have a principled treatment of when sampler-level corrections are dominated by no-correction on a subgroup.

- Group fairness literature treats subgroup-dominated estimators but feels mismatched here (intent isn't a protected attribute, and the dominator changes per-cell).
- CFG itself has this shape (high CFG saturates flat prompts, low CFG produces bland detail) — what's the standard answer beyond "tune per-prompt"? Dynamic-CFG and oscillation-guidance papers all face this implicitly; we haven't found one that treats it as the central question.
- For a per-prompt calibrator like v4: are there published guarantees on "applies zero correction at the right rate on null-correction-needed prompts"? Equivalent to selective-prediction / abstention literature; specific application to per-step diffusion correction would be useful.

## 9. LL-as-causal-lever — does single-band correction propagate through the DiT's nonlinear forward to tighten detail-band magnitudes?

The paper (Yu et al., §5.3) motivates the wavelet decomposition because *DPMs reconstruct low-frequency contours first, then high-frequency details* (ref [61]). It then applies **separate** corrections per subband (own λ_l for LL, own λ_h for HH, etc.). Our bench observation is structurally different and stronger: applying correction to **LL only** propagates through the DiT's nonlinear forward and tightens **all four** bands' magnitudes downstream, even when the LL band's own integrated signed gap doesn't shrink.

Triangulated across three benches:

1. **2026-05-03 band-mask sweep** (CFG=1, square): `λ × LL-only` strictly better than `λ × all-bands` and `λ × HH-only` on every metric checked — late-half |gap|, no sign flips, all four per-band gaps improved vs baseline, perceptually equivalent or better. HH-only correction sign-flipped HH visually; all-bands correction worsened detail bands.
2. **2026-05-04 A2 sweep** (CFG=4, 832×1248, λ=+0.01 LL one_minus_sigma vs baseline): LL signed gap **widened** (+89 → +102), but per-band magnitudes of detail bands collapsed dramatically: LH 33 → 22 (−33%), HL 39 → 12 (−69%), HH 47 → 5 (−89%). Integrated-|gap| total fell 350 → 294 (−16%).
3. **2026-05-05 perceptual A/B** (same configuration as #2): user confirms positive λ recovered missing details vs baseline. Perceptual signal tracks the integrated-|gap| (across all 4 bands), not LL-signed-gap alone.

So the picture: LL-only correction is a **causal lever**, detail bands are the **readout**. The mechanism is the DiT's nonlinear forward — applying LL-only correction at step `i` shifts `x̂_{i+1}`, which then drives all four bands' velocity outputs at step `i+1` and after. This is consistent with the paper's "low-freq first" motivation but goes further: **you don't need to correct every band; correcting LL is sufficient because the nonlinear forward couples them downstream**.

What we'd appreciate input on:

- Is there literature on **single-band sampler correction propagating across bands through a denoiser's nonlinear forward**? Closest analog we've found is "frequency-aware guidance" papers that apply per-band weights to CFG, but those weight all bands simultaneously. We're claiming a *cross-band coupling* property that should be testable as: train any latent diffusion model, apply LL-only correction at one step, measure downstream detail-band magnitudes — they should tighten if the property holds. Universal across DPMs or specific to certain architectures (DiT vs UNet, flow-matching vs ε-pred)?
- Is this **a property of the DiT specifically** (the global self-attention couples all spatial frequencies in one operation) **or of latent diffusion in general** (the VAE's spatial-channel mixing couples them)? Cleanest test: apply LL-only correction on a pixel-space DDPM and see whether detail bands still tighten downstream. The paper's CIFAR experiments are pixel-space — if their λ_h ablation rows can be re-read as "what happens when only λ_l fires", that data may already exist.
- Theoretical question: for a Lipschitz-bounded denoiser on a multi-band signal, is there a known result that per-band correction at step `i` produces bounded perturbations on every band at step `i+1`? Some kind of "frequency-domain stability" result for diffusion sampling we should know about?
- Practical implication if the property generalizes: any sampler-level frequency intervention can be reduced to a *single-band* lever, drastically cutting the parameter space (one λ instead of four) without losing per-band control. This would also re-frame how multi-band controllers like the paper's Eq 20–21 should be designed — they may be over-parameterized.

---

## What's least useful to send our way

- "Run a bigger calibration sample" — covered, not the bottleneck for σ̂².
- "Try a deeper MLP" — capacity sweeps haven't moved CV.
- Recommendations to switch off flow-matching or off cross-attention text conditioning — not workable for this codebase.

## What's most useful

- Pointers to specific papers / preprints / threads that have asked the same question and have a result (positive or negative).
- Theoretical bounds (information-theoretic, Kalman-style, calibration scaling law) we can use as a sanity check on the empirical 0.55 r_α ceiling.
- Anecdotes from groups doing similar sampler-level corrections on diffusion / flow-matching, especially anything on the sign-flip (#1) or per-step S_pop sign-flip (#7) — these are the two findings most likely to be reproducible-or-not on other backbones, and we genuinely don't know.

## Where to read deeper

- Method overview: [`docs/methods/dcw.md`](./docs/methods/dcw.md)
- Full v4 derivation, gates, fallback ladder: [`docs/proposal/dcw-learnable-calibrator-v4.md`](./docs/proposal/dcw-learnable-calibrator-v4.md)
- 2026-05-05 σ̂² drop + target-window finding: [`docs/proposal/dcw-v4-target-window-and-sigma-drop.md`](./docs/proposal/dcw-v4-target-window-and-sigma-drop.md)
- Bench scripts: [`bench/dcw/k_supervision_sweep.py`](./bench/dcw/k_supervision_sweep.py), [`bench/dcw/plot_seed_band.py`](./bench/dcw/plot_seed_band.py)
- Trainer: [`scripts/dcw/train_fusion_head.py`](./scripts/dcw/train_fusion_head.py)
- Anima form details + sign-flip discussion: [`archive/dcw/README.md`](./archive/dcw/README.md), `archive/dcw/findings.md`
