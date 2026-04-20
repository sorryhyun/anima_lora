# Proposal: Reframing Anima's Variants Through the SNR-t Bias Lens

*Companion to `plan.md` (DCW integration). Where `plan.md` answers "how do we add DCW," this answers the more useful question: "what do Anima's existing knobs actually do to the SNR-t trajectory, and where do they leave room?"*

**Reference paper:** Yu et al., *Elucidating the SNR-t Bias of Diffusion Probabilistic Models* (CVPR 2026, arXiv:2604.16044).

---

## 1. The central claim, compressed

Training pins a deterministic mapping `t → SNR(x_t)`. Inference breaks it: solver discretisation error + network prediction error compound, and the paper's Fig. 1c shows empirically that **reverse-process samples at timestep `t` have systematically lower SNR than the matching forward-process samples at the same `t`**. The network, conditioned on `t`, is trained for the wrong operating point at every step — and because its response is SNR-monotone (Fig. 1b: lower-SNR input → larger-norm noise prediction → more aggressive "denoising"), the drift is not self-correcting. It's a slow, accumulating bias, not noise that washes out.

Three takeaways that matter for how we read our own variants:

1. **"Inference error" factors cleanly into two independent sources** — solver error and network-prediction error — and interventions should be labelled by which one they touch. Most of what we've built so far touches only one of the two.
2. **The bias lives in the sampling trajectory**, not in any one model forward. Training-time tricks can reduce prediction error but cannot, by construction, reduce solver error.
3. **The reconstruction sample `x^0_pred` is cheaply available at every step and contains directional information pointing back toward the ideal forward distribution.** The paper's entire method is this single observation plus a schedule. It's shallow in a good way — this is a lever we haven't used.

---

## 2. Classifying our existing interventions

| Variant | Stage | What it actually does | Relation to SNR-t bias |
|---|---|---|---|
| **LoRA (classic)** | train | Fits velocity error on data | Neutral — doesn't change trajectory geometry |
| **OrthoLoRA** | train | Low-rank w/ orthogonal parameterisation | Neutral on SNR-t. Regularises ΔW spectrum |
| **T-LoRA** | train | Rank varies with `t` — low-rank at high noise, full-rank at low noise | **Indirectly SNR-t-aware**: acknowledges different `t`-regimes need different capacity. But still trains pointwise — no trajectory-level signal |
| **HydraLoRA** | train | MoE router picks expert per-layer | Neutral |
| **ReFT** | train | Residual-stream intervention on DiT blocks | Neutral |
| **APEX** | train | Self-adversarial distillation to 1–4 NFE | **Masks the bias** — at 1–4 NFE the *concept* of trajectory drift still applies, but APEX trains the model to be robust to it by construction. Does not "fix" SNR-t bias; it trains around it |
| **Modulation guidance (distill-mod)** | train + inf | Learned `pooled_text_proj` MLP feeds AdaLN | Neutral — operates on text conditioning, orthogonal axis |
| **Postfix / prefix / postfix-σ** | train + inf | Learned vectors spliced into crossattn | `postfix-σ` is σ-conditional → *mildly* SNR-t-aware (different prompt enhancement per timestep), but only for the conditioning signal, not the sample itself |
| **Embedding inversion** | inf-time opt | Optimises text embedding | Orthogonal |
| **Spectrum** | inf | Skips transformer blocks on predicted steps | **Adds** a third error source (forecasting error); does not address SNR-t |
| **ER-SDE sampler (stage-3)** | inf | Higher-order stochastic solver | **Reduces solver-discretisation error**, touches one of the two error sources |
| **DCW (proposed)** | inf | Post-step correction along `prev - x^0_pred` | **Reduces the integral effect of both error sources** at the output |

Reading this top-down: **of twelve knobs, exactly two (ER-SDE, DCW) operate at the trajectory level. Everything else either trains the network or modifies conditioning.** That's the gap.

---

## 3. Per-variant analysis

### 3.1 T-LoRA — partial, uncoordinated SNR-t awareness

T-LoRA's schedule `r(t) = floor((1 - t)^α · (R_max - R_min)) + R_min` encodes the prior that different `t` deserve different capacity. This is a **training-only** awareness of SNR-t: the mask is disabled at inference by design (`docs/methods/timestep_mask.md:59`).

The DCW paper formalises a related fact that T-LoRA's schedule is *groping toward*: the network's error profile is not uniform in `t`. But T-LoRA's choice is driven by a different intuition ("high-noise needs more capacity") rather than measurement of where bias accumulates.

**Reframe:** T-LoRA is a capacity schedule; DCW is an error-correction schedule. They're complementary — T-LoRA reduces pointwise prediction error at each `t` during training; DCW corrects the *integrated* effect of remaining prediction+solver error during inference. No reason to expect them to overlap, and empirically we should see T-LoRA + DCW > either.

**Potential follow-up:** if the DCW paper's Key Finding 1 (network-output norm monotone in input SNR mismatch) is reproducible on Anima, we could inform T-LoRA's α. Specifically: measure `||v_θ(x̂_t, t)|| / ||v_θ(x_t, t)||` across steps; the ratio's shape tells us where prediction error is worst, which is where T-LoRA should concentrate rank. Right now α is a hyperparam; this would make it measurable.

### 3.2 APEX — trains around the bias, doesn't fix it

APEX collapses 28-step sampling to 1–4 NFE by condition-shift distillation. At the limit of 1 NFE, the concept of trajectory drift dissolves because there's no trajectory. But 2–4 NFE still has per-step drift, and it's *extreme* per step because `σ` jumps by huge amounts.

Three angles:

1. **APEX's 3-forward train pattern (`real / fake@real_xt sg / fake@fake_xt`) is arguably a crude version of the "align reverse with forward" intuition.** The fake branch generates a fake trajectory point; the consistency loss ties the real branch to it. This is implicitly pulling the real output toward a distribution that behaves like the forward process. Not the same as DCW, but in the same neighbourhood of reasoning.
2. **APEX + DCW might fight or compound.** If APEX has learned to overshoot in a way that compensates for undershoot in the vanilla trajectory, DCW would double-correct. Worth explicit ablation.
3. **The more interesting move:** could a DCW-style post-step correction be *folded into APEX training* as a regulariser on the fake-branch target? I.e., instead of the current `T_mix`, target `T_mix + λ·(T_mix - x^0_fake)`. This would teach the student that the correct inference-time output is *already* the differentially-corrected one, so post-hoc DCW becomes unnecessary at inference. This is speculative but cheap to try after plain DCW lands.

### 3.3 Spectrum — composition caveat

Spectrum skips block forwards on cached steps and forecasts features. This is a third error source orthogonal to the two in the DCW paper. On a cached step:

```
feat_cached = polynomial_fit(feat_history) + residual_bias_correction
noise_pred  = final_layer(feat_cached)
x0_pred     = latents - σ_i · noise_pred
```

`x0_pred` here inherits the forecasting error. Running DCW on top corrects `prev_sample` toward this biased `x0_pred`. The paper's theoretical guarantee (Theorem 5.1) assumes `x^0_pred = γ_t x_0 + ϕ_t ε` — a clean signal+noise decomposition. Forecasting error breaks that assumption.

**Two plausible outcomes:**

- DCW is *more* valuable on Spectrum runs because there's *more* bias to correct. The direction `prev - x0` still points toward higher-SNR states regardless of where the bias came from.
- DCW *amplifies* forecasting error because `x0_pred` is now genuinely wrong, not just biased.

This is empirically decidable and worth an ablation row. Spectrum already has `spectrum_calibration` for residual bias correction — DCW might make that knob redundant or complementary.

### 3.4 ER-SDE sampler — partial overlap with DCW

Our ER-SDE stage-2/3 solver uses Taylor-expansion terms to reduce per-step discretisation error (`library/inference/sampling.py:40`). This directly addresses *one* of the two error sources in the DCW paper.

**Reframe:** ER-SDE attacks solver error. DCW attacks the integrated-over-time effect of remaining prediction-plus-solver error at the output. They compose, but ER-SDE partially pre-empties what DCW would otherwise correct. Expected headroom from DCW on top of ER-SDE is smaller than on top of plain Euler — but still nonzero because prediction error is untouched by the solver.

**Practical:** record DCW gains on both `--sampler euler` and `--sampler er_sde`; expect a larger delta on Euler.

### 3.5 Postfix-σ — σ-aware conditioning

Postfix-σ (`configs/methods/postfix.toml`, `docs/methods/postfix-sigma.md`) conditions the injected postfix vectors on the sampling σ. This is a form of SNR-t awareness — but on the *conditioning path*, not the sample path.

**Reframe:** postfix-σ gives the network a slightly different prompt at each timestep, anticipating that the network's optimal prompt varies with `t`. DCW gives the *sample* a slightly different target at each timestep, anticipating that the sample's optimal value varies with `t`. Same underlying premise (SNR-t-dependent behaviour), applied to orthogonal tensors.

Both should work. The σ-conditional postfix memory note (`project_postfix_slot_collapse.md`) reveals that the original postfix *effectively* collapsed to K=1 — σ-conditioning is how we're trying to re-extract actual variation. It's a feature worth keeping, and it doesn't interact with DCW.

### 3.6 Modulation guidance — orthogonal

Learned `pooled_text_proj` MLP feeds AdaLN. Operates on text → modulation path, entirely separate from the sampler / latent path. No interaction with DCW expected.

### 3.7 OrthoLoRA — orthogonal, but suggestive

OrthoLoRA regularises ΔW to have orthonormal rows/columns. Unrelated to SNR-t, but the DCW paper's Assumption 5.1 (`x^0_θ = γ_t x_0 + ϕ_t ε`, with `γ_t < 1` representing "energy loss") echoes a similar intuition about bounded operator norms. Probably coincidence.

---

## 4. What the DCW paper reveals about things we *could* build

### 4.1 A measurement we haven't made

Before we commit to any correction method, **we should reproduce the paper's Fig. 1c experiment on Anima**. The plot is:

- `||v_θ(x_t, t)||_2` for forward-noised `x_t` (green line)
- `||v_θ(x̂_t, t)||_2` for reverse-sampled `x̂_t` at the same timestep (red line)

If Anima shows the same gap (reverse > forward), the bias is real on our model and DCW has a mechanism to bite. If not, DCW's premise doesn't hold on flow-matching DiTs at our scale and we need a different intervention. This is maybe 50 lines of diagnostic code and one afternoon of GPU time. It's a precondition that should come *before* the hyperparam sweep in `plan.md` §3.

### 4.2 A loss-weighting we haven't tried

The DCW paper's Key Finding 1 says `||v_θ(x_t, s)||` scales monotonically with SNR mismatch. During *training*, the batch distribution over `t` and the `σ_t` schedule together imply that some timesteps see bigger mismatches than others (at inference). A principled training-time intervention would be **loss-reweighting inversely proportional to measured inference-time SNR drift at each `t`**. Timesteps where the inference trajectory drifts more get heavier training weight, because the network needs to be more robust at those operating points.

This is speculative. It requires a chicken-and-egg solve (measure drift on a model that isn't yet trained with this weighting). But it's a natural continuation of the DCW paper's framing that isn't in the paper.

### 4.3 A principled T-LoRA schedule

Per §3.1: measure the Fig. 1b prediction-error curve on Anima, use it to set T-LoRA's α directly instead of hand-tuning. One-time calibration, would make T-LoRA empirically grounded rather than heuristic.

### 4.4 A DCW-style inner loop for embedding inversion

Embedding inversion (`make invert`) optimises text embeddings through the frozen DiT using flow-matching loss. The optimisation trajectory has its own SNR-t-like bias: the optimiser sees gradients through sampling steps that themselves drift. A DCW-style correction inside the inversion loop might stabilise it. Speculative — worth a quick try after plain DCW lands.

---

## 5. Recommended ordering

1. **Diagnostic first** (§4.1): reproduce Fig. 1c on Anima. Small script, ~1 afternoon. If the gap is weak or absent on flow-matching DiTs at our scale, the rest of this plan needs reconsidering.
2. **Plain DCW** per `plan.md` — pixel-mode first, because it's trivial and tests the mechanism without DWT confounds.
3. **DCW ablations:** pixel vs low vs dual; `--sampler euler` vs `er_sde`; with and without `--spectrum`; with and without APEX. This is the matrix that tells us where DCW is actually doing work vs where it's already pre-empted.
4. **Decide on follow-ups** (§4.2, §4.3, §4.4) based on ablation results.

---

## 6. Summary

The DCW paper doesn't just propose a method — it proposes a framing. That framing exposes that **almost everything we've built is training-side or conditioning-side**, and the single sampler-side intervention we have (ER-SDE) addresses only one of the two inference error sources. DCW fills the remaining quadrant cheaply.

Beyond the immediate integration, the framing gives us a language for measuring and comparing future proposals: every new knob can be asked (a) which error source does it target? (b) train-time or inference-time? (c) conditioning or sample path? The map in §2 was easy to fill in *after* reading the paper; it was not obvious beforehand. That's the real deliverable.
