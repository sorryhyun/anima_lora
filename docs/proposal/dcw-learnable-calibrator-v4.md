# DCW Calibrator v4 (online observation + prompt + aspect fusion)

**Status:** proposal · supersedes [v3](../../archive/dcw-learnable-calibrator/proposals/dcw-learnable-calibrator-v3.md) (which becomes the second fallback above [v2](../../archive/dcw-learnable-calibrator/proposals/dcw-learnable-calibrator-v2.md)) · **Effort:** ~4-6 days · **Cost at inference:** bucket lookup + DWT during k-step warmup + 1 MLP forward at step k + per-step cap check; ~10-15% wall-time overhead · **Reference papers:** Yu et al. arXiv:2604.16044 (DCW); Kendall & Gal NeurIPS 2017 (heteroscedastic aleatoric uncertainty)

## Theoretical motivation

DCW's own theorem (Yu et al. §5.1, Eq 12) gives the analytical SNR of the biased reverse sample:

```
SNR(x̂_t) = γ̂_t² · ᾱ_t / (1 - ᾱ_t + (√ᾱ_t · β_{t+1} / (1 - ᾱ_{t+1}))² · φ_{t+1}²)
```

The pair `(γ̂_t, φ_t)` comes from the reconstruction model `x⁰_θ(x̂_{t+1}, t+1)` (Assumption 5.1), so it is **not fixed by the timestep**. It is a function of what the network actually predicts on this trajectory — which depends jointly on prompt, aspect (patch geometry / token count constrains what `x⁰_θ` can fit), and the realized noise sequence (seed). The DCW paper's correction `λ_t · (x̂_{t-1} − x⁰_θ(x̂_t, t))` (Eq 17) uses a **single offline-tuned scalar λ averaged over all three sources of variation**. That is, by the paper's own theory, a marginal approximation to the per-trajectory optimum.

v4's three input channels are exactly the three sources of variation in `(γ̂_t, φ_t)`:

| v4 channel | What it estimates in DCW theory |
|---|---|
| Aspect prior `μ_g[aspect]` (Layer 1) | `E[gap_t \| aspect]` — patch geometry / token-count effect on what `x⁰_θ` can fit |
| `c_pool` + aux scalars | `E[gap_t \| aspect, prompt]` — text-conditional shape of `x⁰`, hence `φ` |
| `g_obs[0:k]` (NEW in v4) | Sufficient statistic of the realized `(γ̂_τ, φ_τ)` for τ < k under *this* seed's trajectory |

**Why few-step observation is theoretically justified, not just empirical.** Eq 13 is a recursion: `x̂_{t-1}` is linear in `x̂_t` plus structured noise, so the gap process is AR(1)-like with smooth structure. Empirically this shows up as `lag-1 ≈ 0.55` in the 1543 sweep — consistent with the recursion. Once you observe a prefix of an AR-like walk driven by a fixed `ε` realization, the late-step `(γ̂, φ)` trajectory is largely predictable. v4's `r²(seed-dev) = 0.79` at `k=7` is a direct measurement of that AR predictability.

**Why per-prompt seed std ≈ between-prompt std (the thing that killed v3).** `φ_t` compounds through Eq 13, and `γ̂_t < 1` damps amplitude, so the seed-driven branch of the recursion can dominate at low-SNR steps. The observed 0.94 within/between LL ratio is consistent with the recursion's structural variance budget — not measurement noise, but a real load-bearing source of variance. Prompt-only conditioning (v3) provably cannot reduce it; observation (v4) can, because it samples the realized branch directly.

**Heteroscedastic head ↔ residual recursion variance.** Kendall & Gal `(α̂, log σ̂²)` maps cleanly onto this picture: `σ̂²` estimates the residual `Var(φ_τ | aspect, prompt, g_obs[0:k])` for τ > k. The shrinkage `σ²_prior / (σ²_prior + σ̂²)` is then a Bayes-optimal posterior weight on the observation channel under that variance.

**One-line summary.** *DCW Eq 12 says optimal `λ_t` is a function of `(γ̂_t, φ_t)`, which are realization-dependent through Eq 13's recursion; v4 conditions `λ` on the three measurable axes of that realization (aspect, prompt, k-step prefix), with heteroscedastic `σ̂²` capturing residual recursion variance.*

## Why v4 (over v3)

The 1543 5-seed sweep (`bench/dcw/results/20260504-1543-small_invhd`, 8 prompts × 5 seeds × 28 steps at production CFG=4 inverse-HD) established two facts that change the architecture:

**1. Per-prompt seed variance dominates the signal v3 was trying to predict.**

| band | mean(\|gap\|/seed) | within-prompt seed std | between-prompt std | within/between |
|------|---|---|---|---|
| LL | 380 | 349 | 370 | **0.94** |
| LH | 35 | 28 | 34 | 0.80 |
| HL | 41 | 28 | 46 | 0.61 |
| HH | 32 | 24 | 18 | **1.33** |

6/8 prompts sign-flip across 5 seeds on LL integrated gap. v3's prompt-only head can at best predict the seed-mean per prompt; the heteroscedastic shrinkage then attenuates that prediction toward zero on the (large) noise-dominated subset. Net: v3's head can't break through the seed-noise ceiling.

**2. But within a started trajectory, the seed commits early.**

Non-overlapping early-vs-late predictability of LL gap, seed-conditional (subtracting per-prompt mean trajectory):

| observed prefix k | sigma_k | corr(early-dev, late-dev) | r² | sign agreement |
|---|---|---|---|---|
| 4 | 0.86 | 0.79 | 0.62 | 82.5% |
| 7 | 0.75 | **0.89** | **0.79** | **90.0%** |
| 10 | 0.64 | 0.93 | 0.87 | 90.0% |
| 14 | 0.50 | 0.96 | 0.92 | 92.5% |

Predicting the **dominant final-step gap** seed-deviation: r=0.78 from k=4, r=0.91 from k=10, r=0.93 from k=14.

The seed-noise that confounds prompt-only prediction becomes **observable** once denoising starts. v4 reframes the problem from "predict an unobservable RV from text" to "extrapolate an observed walk from a few measurements" — a much easier signal.

**v4 fuses three input channels** that are individually weak but jointly strong:
- **Aspect** — sets the per-bucket prior `μ_g[aspect, i]` (= v2 Layer 1)
- **Prompt** — `c_pool` carries the prompt's expected amplitude/sign tendency
- **Observation** — first k=7 measured LL gaps disambiguate which seed-realization is happening

## Architecture

Three inputs, one fusion head, online during inference.

### Input 1 — aspect prior (= v2/v3 Layer 1)
Per-aspect bucket profile `μ_g[aspect, i]`, `S_pop[aspect, i]`, `λ_scalar[aspect]`. Resolved at sampler init from `(H, W) → aspect_id`. Same artifact as v2/v3.

### Input 2 — prompt embedding
Mean-pooled `crossattn_emb_v0` → `c_pool` (1024-dim) + auxiliary scalars `[caption_length, cos(c_pool, μ_centroid), token_l2_std]` (3-dim). Same as v3.

### Input 3 — observed prefix gap (NEW)
First k=7 LL-band gap measurements `g_obs[0:k]`. Computed at runtime using the same single-level orthonormal 2D Haar DWT path as `bench/dcw/measure_bias.py`. The bench's `v_fwd / v_rev` machinery is extracted into a runtime module (`library/inference/dcw_observation.py`).

### Fusion head
Single shared MLP `h(c_pool, aspect_emb, g_obs[0:k], aux) → (α̂, log σ̂²)`:
- **Input**: c_pool (1024) + learned aspect_emb (16) + g_obs (7) + aux (3) = 1050-dim
- **Architecture**: 3-layer MLP, 1050 → 256 → 128 → 2, GELU, layer norm, dropout 0.1
- **Params**: ~285k total (single shared head, not per-bucket — aspect is an input)
- **Output**: scalar `(α̂, log σ̂²)` for the *remaining* (N − k) steps
- **Targets**: per-seed integrated residual on the tail, `r_p,s = Σ_{i=k}^{N} (g_p,s[i] − μ_g[aspect, i])`
- **Loss**: Gaussian NLL (Kendall & Gal):
  ```
  L = (1/N) Σ_{p,s} [ (r_p,s − α̂_p)² / (2 σ̂²_p)  +  ½ log σ̂²_p ]
  ```
  + L2 weight decay (1e-4). Single shared head, all aspects together.

Single shared head (not per-bucket) is cheap because aspect is now an input and the calibration set can be aspect-balanced rather than aspect-stratified — n=200 total instead of n=200×3.

### Why warmup is left unmodified by the head

During steps 0..k the controller applies only the bucket-prior correction (= v2's per-step lookup). The head fires once at step k with `g_obs[0:k]` collected from those uncorrected-by-head observations. Training data is collected the same way (baseline sampling with bucket-prior correction during warmup), so the train/inference observation distribution matches.

Alternative considered and rejected: head fires at every step k..N with running observation. Adds compute, gives marginal lift (the trajectory is already largely committed by k=7 — see lag-10 autocorrelation = 0.44).

### Inference flow
```python
# Setup (once per inference)
aspect_id          = resolve_aspect(H, W)
mu_g, S_pop, lam0  = bucket_profile[aspect_id]
head, aspect_emb   = fusion_head_artifacts
c_pool, aux        = pool_text_embed(crossattn_emb)
k                  = warmup_steps  # default 7

# Warmup steps 0..k: bucket-prior DCW only, capture observations
g_obs = []
for i in range(k):
    base_lambda  = lam0 * (1 - sigma[i])
    correction   = mu_g[i] * lam0 / S_pop[i]                # v2 lookup
    lambda_i     = clamp_overshoot(base_lambda + correction)
    v_fwd, v_rev = step_with_dcw_and_observe(lambda_i)      # extra DWT pass
    g_obs.append(LL_gap(v_fwd, v_rev))

# Step k: head prediction (one MLP forward)
alpha_hat, log_sigma2 = head(
    c_pool, aspect_emb[aspect_id], torch.tensor(g_obs), aux
)
sigma2     = exp(log_sigma2)
shrinkage  = sigma2_prior[aspect_id] / (sigma2_prior[aspect_id] + sigma2)
alpha_eff  = alpha_hat * shrinkage
if aux["caption_length"] < tau_short[aspect_id]:
    alpha_eff = 0.0                                          # v3 backstop kept

# Remaining steps k..N: bucket prior + head correction
tail_norm = sum(mu_g[k:])
for i in range(k, N):
    base_lambda = lam0 * (1 - sigma[i])
    bucket_corr = mu_g[i] * lam0 / S_pop[i]
    head_corr   = alpha_eff * mu_g[i] / tail_norm
    lambda_i    = clamp_overshoot(base_lambda + bucket_corr + head_corr)
    # ... DiT step (no extra DWT pass needed)
```

`σ²_prior[aspect]` is the average `σ̂²` over stable training prompts in that aspect (calibrated, not learned).

### Reference artifact

`<base_dit_name>_dcw_v4.safetensors`:
- (all of v2's per-aspect profile)
- `fusion_head`: shared MLP weights (~285k params, ~1.1 MB)
- `aspect_emb`: 3 × 16 learned embedding (192 floats)
- `tau_short[aspect_id]`: caption-length threshold per bucket (= v3)
- `sigma2_prior[aspect_id]`: average head σ̂² on stable prompts per bucket
- `k`: warmup-step count (default 7)
- metadata: head architecture, training hyperparams, calibration n × seeds × buckets, validation NLL/r per bucket, k-sweep result, git SHA

## Quality gates

### Gate A — overshoot guard (= v2/v3)
Held-out 32 × 2 prompts × 3 buckets at CFG=4. Per-bucket overshoot fraction ≤ 2% with controller on. Unchanged.

### Gate B — head calibration (NEW, blocking)
Held-out 48 prompts × 3 seeds × 3 buckets not in training set:

| Metric | Threshold | Action |
|---|---|---|
| Pearson r(α̂_p, mean_s r_remaining) | ≥ 0.6 (was 0.5 in v3) | Head amplitude tracks prompt-mean tail residual |
| Pearson r(α̂_p,s, r_p,s) **seed-conditional** | ≥ 0.7 (NEW) | Observation channel is doing real work |
| Pearson r(σ̂_p, std_s(r_p,s)) | ≥ 0.4 | σ̂ tracks ground-truth seed noise (uses k≥3 seeds) |
| Held-out NLL improvement vs N(0, σ²_pop) baseline | ≥ 15% (was 5% in v3) | Higher bar because observation is supposed to add a lot |
| Calibration: \|r − α̂\|² vs σ̂² (binned) | within 2× across all bins | Confidence is well-calibrated |

If `r(α̂_p,s, r_p,s) seed-conditional < 0.7` → **observation channel is not adding over prompt-only**. Fall back to v3 (or v2 if v3 also fails).

If `r(α̂_p, mean_s) < 0.6` on any bucket → fall back per-bucket to v3.

### Gate C — perceptual side-by-side (= v3 Gate C, expanded)
4 LoRAs (flat / painterly / detail-dense / base) × 12 prompts × {square, inverse-HD} × {v4, v3, v2, scalar}. Four-way comparison.

| Metric | Threshold |
|---|---|
| v4 vs v3 overall preference | ≥ 55% prefer v4 |
| v4 vs v2 overall preference | ≥ 60% prefer v4 |
| Inverse-HD subset, v4 vs v3 | ≥ 60% prefer v4 (this is where observation is supposed to add over v3) |
| Flat LoRA × short prompts (caption-length backstop active) | v4 ≥ scalar (don't regress) |
| No (LoRA × bucket) cell where v4 loses ≤ 30% | else fall back to v3 for that cell |

If v4 ≤ v3 perceptually overall → ship v3 if its gates pass, else v2.

## Analysis phase (~2-3 days)

| # | Run | Output | Why |
|---|---|---|---|
| **A1** | Per-aspect baseline at production env, n=48 × 2 seeds × 3 buckets (= v2 §A1). **Independent sampling per bucket** — bucket prior `μ_g[aspect, i]` is a marginal expectation; no within-prompt cross-bucket pairing required. Source images in `post_image_dataset/lora/` are cached at a fixed resolution per stem, so cross-bucket pairing would require re-resize + re-cache per (image × aspect). | per-aspect `gaps_per_sample.npz` + `μ_g`, `S_pop` | Bucket prior. Reuse if v2/v3 done. |
| **A2** | λ-sweep per aspect, n=12 × 2 (= v2 §A2) | `S_pop`, `λ_scalar` per bucket | Layer 1 sensitivity. Reuse if done. |
| **A3** | **Calibration: n=200 × 3 seeds aspect-balanced (each prompt at its native cached aspect), production env, baseline-only with bucket-prior correction during warmup**. Single shared head learns the aspect contribution via `aspect_emb`; pairing same-prompt across aspects is **not required** for training. (Optional anchor-prompt augmentation: defer until A7's first training run shows whether `aspect_emb` gradients are noisy; if so, +30 anchor prompts × 3 aspects × 3 seeds = ~270 extra samples, +60 re-caches, +~2.5h sampling.) | per-(prompt,seed) **full per-step LL gap trajectory** | Training data for fusion head. ~9-12 hours wall on a 5060 Ti total (not per bucket — single shared head). |
| **A4** | k-sweep on A3 data (k ∈ {4, 7, 10, 14}) | optimal warmup-step count | Pick lowest k that gets `r²(seed-dev) ≥ 0.8` on held-out. |
| **A5** | Multi-band ablation on A3 data | does adding LH/HL/HH to g_obs help over LL-only? | Default LL-only; bump only if r² jumps ≥ 0.05. |
| **A6** | Caption-length threshold sweep | `τ_short[aspect]` per bucket | = v3 §A4. |
| **A7** | Head training: train+validate on A3 with 80/20 prompt split, single shared head | head weights, validation NLL / r(α̂, mean) / r(α̂, per-seed) / r(σ̂, σ_seed) | Gate B inputs. |
| **A8** | Per-LoRA cross-check: rerun A3 mini (n=64 × 3) on 1 painterly + 1 flat LoRA, single aspect (~2 hours each) | per-LoRA fusion-head divergence | Closes `project_dcw_when_to_use`. |

**Decision points after analysis:**

| Outcome | Action |
|---|---|
| A7 r(α̂,mean) ≥ 0.6 AND r(α̂,per-seed) ≥ 0.7 AND r(σ̂,σ_seed) ≥ 0.4 | Proceed to v4 implementation full ship |
| A7 r(α̂,per-seed) ≥ 0.7 but r(σ̂) < 0.4 | Ship `α̂` with caption-length backstop only, no shrinkage; risky on unstable prompts. Recommended fallback: v3 if v3's prompt-only head gates pass. |
| A7 r(α̂,per-seed) < 0.7 (observation channel weak) | **Fall back to v3.** Observation didn't add over prompt-only. |
| A7 r(α̂,mean) < 0.6 on inverse-HD only | Ship v4 on square buckets, v3 on inverse-HD (mixed) |
| A4 cannot find k with r²(seed-dev) ≥ 0.8 | Investigate: probably LL-only insufficient → A5 multi-band; if still fails, fall back to v3 |
| A8 finds per-LoRA divergence > 50% on any LoRA × bucket | Per-LoRA head override path required; +1 day implementation |

## Implementation phase (~3 days)

| # | Deliverable | Files | Effort |
|---|---|---|---|
| **I1** | A3 calibration runner | `bench/dcw/calibrate_v4.py` (extends `measure_bias.py` to iterate over aspect-balanced n=200 cached samples × 3 seeds; dumps per-step LL trajectories with bucket-prior warmup applied) | 4h |
| **I2** | Head training script | `bench/dcw/train_fusion_head.py` (loads A3 npz + text-emb caches + bucket profiles; trains single shared MLP with NLL; dumps `fusion_head.safetensors`) | 4h |
| **I3** | Reference artifact aggregator | extend v2/v3's exporter to bundle bucket profiles + fusion head + aspect_emb + tau_short + sigma2_prior + k into one `.safetensors` | 2h |
| **I4** | Online observation module | new `library/inference/dcw_observation.py` — extracts the bench's `v_fwd / v_rev / DWT-LL` path into a runtime hook callable from inside the sampler step. **Critical**: must reuse the existing CFG cond-branch forward where possible to avoid 2× warmup cost. | 5h |
| **I5** | Inference module | new `library/inference/dcw_online.py` — `OnlineFusionDCWController` with warmup-loop observation, fusion at step k, head correction for tail. Integrate with `dcw_calibration.py` resolution chain (v4 → v3 → v2). | 5h |
| **I6** | Inference flags | `inference.py` — `--dcw_v4_disable` (forces v3/v2), `--dcw_v4_warmup_k <int>` (override default 7), `--dcw_v4_shrinkage_disable`, `--dcw_v4_backstop_disable` | 1h |
| **I7** | Validation harness | `bench/dcw/calibrate_per_lora.py` extension — Gates A, B, C with the 4-way side-by-side grid generator | 4h |
| **I8** | Gate runs | A: 32×2×3 buckets; B: 48×3×3 buckets held-out; C: perceptual | 1.5 days, dominated by C |
| **I9** | Doc updates | If pass: `docs/methods/dcw-calibrator.md` with v4 architecture, gate results, fallback policy. If fail: archive v4 with negative result; v3 (if it passes) or v2 ships. | 2h |

## Fallback ladder

Three tiers, resolved at controller init:

1. **Full v4 ship** if all gates pass.
2. **Mixed v4/v3** if v4 Gate B passes on a subset of buckets. Resolution chain prefers v4 head where available, else v3 prompt-only head, else v2 bucket profile.
3. **v4 → v3 fallback** if v4 Gate B fails entirely → fall back to v3 (if its gates pass).
4. **v4 → v2 fallback** if both v4 and v3 fail → ship v2 from existing v2 doc.

## Risks

1. **Online gap measurement cost.** Each warmup step requires both a fwd-noise and rev-cond DiT pass + DWT. For k=7 of 28 steps:
   - **Naive**: 2× per-step cost during warmup → 25% × 100% extra = 25% wall-time overhead
   - **CFG reuse**: under CFG > 1 inference, the cond branch is already computed; only the fwd-noise pass is extra → ~12% overhead
   - **Mitigation**: `dcw_observation.py` must reuse CFG cond-branch outputs. If CFG=1 (no uncond), accept the 25% warmup cost.

2. **DWT integration.** The bench script's single-level orthonormal 2D Haar DWT needs to be extracted into a runtime module. Same wavelet basis, same level, same band selection. Low risk — bench code is already factored.

3. **k choice trade-off.** Smaller k = less overhead but worse prediction. A4 picks lowest k with `r²(seed-dev) ≥ 0.8`. From the 1543 data, k=7 should pass; k=4 marginal.

4. **Head overfit at n=200 × 3 seeds = 600 samples.** ~285k params on 600 samples. Mitigated by L2 + dropout (0.1) + early stopping on 20% prompt-held-out. Single shared head (not per-bucket) means the parameter-to-data ratio is 3× better than v3.

5. **σ̂ collapse.** Same v3 risk; same mitigations: ε floor on σ̂², init `log σ̂² ≈ log(σ²_pop)`, monitor histogram during training, abort if collapsed.

6. **Per-LoRA divergence.** v2/v3 risk; v4 inherits. **But v4 has a structural advantage**: the observation channel naturally captures the actual LoRA behavior at runtime — flat-style LoRAs that diverge from the bucket profile will produce diverging g_obs, which the head can respond to. May reduce or eliminate per-LoRA-override need. A8 quantifies.

7. **Stacked LoRA composition.** v4 advantage (continued from #6): observation captures the actual composed behavior at runtime, not the assumed pick-largest-multiplier behavior. Bucket profile fallback uses pick-largest as in v2/v3.

8. **CFG drift.** v4 head calibrated at CFG=4. CFG=2/6 → fall back to v3 prompt-only head (if its gates pass) or v2 bucket profile. Recalibrating per CFG would require k=7 observation under each CFG, so the artifact size grows with CFG bins.

9. **Distribution shift in g_obs between train and inference.** Critical correctness: training A3 must apply the bucket-prior correction during warmup so g_obs distribution matches inference. If A3 collects raw-baseline g_obs but inference applies bucket-prior, the head sees a different input distribution at runtime. **Mitigation enforced in I1.**

10. **Mod-guidance interaction.** A3 runs with mod-guidance on (production env). v4 head implicitly conditions on mod-guidance being present in g_obs. New `pooled_text_proj` checkpoints may shift g_obs distribution → head may need recalibration. Same risk as v3.

## What v4 is NOT

- **Not a per-step head.** Head outputs scalar `(α̂, log σ̂²)` for the tail, distributed across remaining steps proportional to the bucket profile. Per-step heads were a v1 upgrade path; PC2 is only 0.95% of variance.
- **Not a multi-band controller.** LL-only by default. A5 ablates whether g_obs should include LH/HL/HH; if r² lift < 0.05, stays LL-only.
- **Not a probe-based controller in the v1 sense.** v1's K=4 probe fit a closed-form α from the early-step gaps and used it as the controller for the rest. v4's k=7 observation feeds a learned head that *also* has prompt and aspect — observation is one of three inputs, not the sole signal.
- **Not a CFG-aware controller.** Single CFG profile + single head at CFG=4.
- **Not a training loss change.** LoRA training untouched.

## Open questions

- **Multi-band g_obs (A5).** LL is the dominant band; LH/HL/HH may add prediction lift on aspect-dependent residuals. If A5 finds ≥ 0.05 r² lift, expand g_obs to 28 features (k=7 × 4 bands).
- **Optimal k.** A4 sweeps {4, 7, 10, 14}. Default recommendation k=7 based on the 1543 r² curve, but actual optimum is data-dependent.
- **Warmup correction policy.** Currently: bucket-prior correction during warmup. Alternative: zero correction during warmup, head predicts based on raw g_obs. Pro of zero-warmup: cleaner observation signal. Con: warmup steps are uncorrected, which may degrade quality of those steps. Pick whichever produces better Gate C.
- **σ²_prior choice.** Same v3 question — average σ̂² on stable training prompts, or `Var(r_p,s) / k_seeds` directly. Pick less-aggressive on Gate B.
- **Caption-length backstop necessity.** If observation channel works as well as the 1543 data suggests, the caption-length backstop may be redundant — flat-style failures will be caught by g_obs naturally. A6 + Gate C will tell.
- **Does v4 generalize across pooled_text_proj checkpoints?** Same v3 concern. Mitigated somewhat by observation channel — if mod-guidance shifts the trajectory, g_obs sees the shift.
- **Online vs once-at-step-k head firing.** Currently fires once at step k. Could fire at every step k..N with a running observation; expected marginal gain is small (lag-10 autocorr 0.44, lag-20 0.14). Defer until v4-base ships.

## Concrete deliverables checklist

- [ ] A1 — per-aspect baseline (3 buckets × n=48) [reuse v2 §A1 if done]
- [ ] A2 — per-aspect λ-sweep [reuse v2 §A2 if done]
- [ ] A3 — n=200 × 3 seeds aspect-balanced calibration (full per-step LL trajectories with bucket-prior warmup)
- [ ] A4 — k-sweep ({4, 7, 10, 14})
- [ ] A5 — multi-band ablation
- [ ] A6 — caption-length threshold sweep
- [ ] A7 — fusion head training + validation
- [ ] A8 — per-LoRA cross-check (1 painterly + 1 flat)
- [ ] **Decision point: full v4 / mixed / fall back to v3 / fall back to v2**
- [ ] I1 — calibration runner
- [ ] I2 — head training script
- [ ] I3 — reference artifact aggregator
- [ ] I4 — online observation module (CFG cond-branch reuse critical)
- [ ] I5 — fusion DCW controller
- [ ] I6 — inference flags
- [ ] I7 — validation harness
- [ ] I8 — Gate A / B / C runs
- [ ] I9 — docs

Total **~4-6 days** from start to perceptual-gate result. v3/v2 fallback ladder adds zero days because both prior tiers are already analysis-phase-complete; if v4's gates fail at the decision point, the chosen fallback ships from its existing doc with no extra implementation work.

## Appendix — evidence underlying v4

| Finding | Source | Implication for v4 |
|---|---|---|
| Per-prompt seed std ≈ between-prompt std (LL: 349 vs 370) | `bench/dcw/results/20260504-1543-small_invhd` | Prompt-only head (v3) hits a ceiling; observation channel needed |
| 6/8 prompts sign-flip across 5 seeds (LL integrated gap) | same | Per-seed targets needed; k≥3 seeds for σ̂ validation |
| HH band: within-prompt std > between-prompt std (ratio 1.33) | same | Don't try to predict per-prompt HH; LL is the only band where signal > noise per-prompt |
| Lag-1 autocorr of LL gap within trajectory = +0.55 | same | Trajectory is smooth enough for short-prefix extrapolation |
| corr(early-dev k=7, late-dev) = 0.89 seed-conditional | same | k=7 observation predicts late-trajectory with r²=0.79 |
| corr(early-dev k=7, final-step seed-dev) = 0.91 | same | Final-step gap (the dominant integrated contributor) is recoverable from k=10 observation |
| Sign agreement early-dev vs late-dev = 90% at k=7 | same | Sign of the tail residual is recoverable, not just amplitude |
| ICC=0.704 on integrated δ at production env | `bench/dcw/stability_predictor_check.py` (1406) | ~30% of per-prompt variance is between-seed (consistent with the 1543 decomposition) |
| Aspect bucket dominates over prompt for bucket prior | v2 §"Why v2 (revised)" | Aspect is a real input, not an afterthought |
| Caption length is strongest stability proxy at scalar level (n=16) | `project_dcw_cpool_feature_feasibility` | Aux scalars worth keeping in head input despite being weak alone |
| DCW gap sign-flips by aspect ratio under CFG=4 | `project_dcw_cfg_aspect_signflip` | Aspect prior must be per-bucket (= v2/v3 Layer 1, retained) |
| FM val loss doesn't track quality | `project_fm_val_loss_uninformative` | Gate C (perceptual) is the only ground truth; head NLL is necessary but not sufficient |
