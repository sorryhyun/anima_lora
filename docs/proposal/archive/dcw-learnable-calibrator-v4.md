# DCW Calibrator v4 (online observation + prompt fusion)

**Status:** implementation · supersedes [v3](../../archive/dcw-learnable-calibrator/proposals/dcw-learnable-calibrator-v3.md) (kept as second fallback above [v2](../../archive/dcw-learnable-calibrator/proposals/dcw-learnable-calibrator-v2.md)) · **Effort remaining:** ~3 days (training works, inference module + gates outstanding) · **Cost at inference:** bucket lookup + DWT during k-step warmup + 1 MLP forward at step k + per-step cap check; ~10–15% wall-time overhead · **Reference papers:** Yu et al. arXiv:2604.16044 (DCW); Kendall & Gal NeurIPS 2017 (heteroscedastic aleatoric uncertainty)

## End-user surface

```bash
make dcw                # collect calibration trajectories + train fusion head + dump artifact
                        # ~3-5h on a 5060 Ti (sampling-dominated; training itself ~30s)
make dcw-train          # train-only on existing bench/dcw/results/ pool (~30s, no sampling)
make test-dcw           # inference with the v4 controller
                        # auto-resolves <dit_name>_dcw_v4.safetensors next to the DiT,
                        # falls back to v3 / v2 / scalar in that order if missing
```

`make dcw` runs `scripts/dcw/measure_bias.py --dump_per_sample_gaps` once per aspect bucket (1024², 832×1248, 1248×832) at the production env (CFG=4, mod_w=3.0), then trains the fusion head via `scripts/dcw/train_fusion_head.py` on the pooled output, then drops `<base_dit_name>_dcw_v4.safetensors` next to the DiT. The artifact is single-file, self-contained: per-aspect bucket profile + fusion-head weights + standardization stats + metadata.

`make test-dcw` is the v4 equivalent of `make test` with the controller wired in at `library/inference/generation.py`'s sampling loop.

## Theoretical motivation

DCW's own theorem (Yu et al. §5.1, Eq 12) gives the analytical SNR of the biased reverse sample:

```
SNR(x̂_t) = γ̂_t² · ᾱ_t / (1 - ᾱ_t + (√ᾱ_t · β_{t+1} / (1 - ᾱ_{t+1}))² · φ_{t+1}²)
```

The pair `(γ̂_t, φ_t)` comes from the reconstruction model `x⁰_θ(x̂_{t+1}, t+1)` (Assumption 5.1), so it is **not fixed by the timestep**. It is a function of what the network actually predicts on this trajectory — which depends jointly on prompt, aspect (patch geometry / token count constrains what `x⁰_θ` can fit), and the realized noise sequence (seed). The DCW paper's correction `λ_t · (x̂_{t-1} − x⁰_θ(x̂_t, t))` (Eq 17) uses a **single offline-tuned scalar λ averaged over all three sources of variation** — by the paper's own theory, a marginal approximation to the per-trajectory optimum.

v4's three input channels are exactly the three sources of variation in `(γ̂_t, φ_t)`:

| v4 channel | What it estimates in DCW theory |
|---|---|
| Aspect prior `μ_g[aspect]` (Layer 1) | `E[gap_t \| aspect]` — patch geometry / token-count effect on what `x⁰_θ` can fit |
| `c_pool` + aux scalars | `E[gap_t \| aspect, prompt]` — text-conditional shape of `x⁰`, hence `φ` |
| `g_obs[0:k]` (NEW in v4) | Sufficient statistic of the realized `(γ̂_τ, φ_τ)` for τ < k under *this* seed's trajectory |

**Why few-step observation is theoretically justified.** Eq 13 is a recursion: `x̂_{t-1}` is linear in `x̂_t` plus structured noise, so the gap process is AR(1)-like. Empirically this shows up as `lag-1 ≈ 0.55` in the 1543 sweep — consistent with the recursion. Once you observe a prefix of an AR-like walk driven by a fixed `ε` realization, the late-step `(γ̂, φ)` trajectory is largely predictable.

**Why per-prompt seed std ≈ between-prompt std (the thing that killed v3).** `φ_t` compounds through Eq 13, and `γ̂_t < 1` damps amplitude, so the seed-driven branch of the recursion can dominate at low-SNR steps. The observed 0.94 within/between LL ratio is a real load-bearing source of variance, not measurement noise. Prompt-only conditioning (v3) provably cannot reduce it; observation (v4) can, because it samples the realized branch directly.

**Heteroscedastic head ↔ residual recursion variance.** Kendall & Gal `(α̂, log σ̂²)` maps cleanly: `σ̂²` estimates `Var(φ_τ | aspect, prompt, g_obs[0:k])` for τ > k. The shrinkage `σ²_prior / (σ²_prior + σ̂²)` is a Bayes-optimal posterior weight on the observation channel.

**One-line summary.** *Eq 12 says optimal `λ_t` is realization-dependent through Eq 13's recursion; v4 conditions `λ` on the three measurable axes of that realization with heteroscedastic `σ̂²` capturing residual variance.*

## Hypothesis validated on existing data

Prototype head trained 2026-05-04 on the pool of existing `bench/dcw/results/` trajectories — 176 rows over 9 baseline runs, 40 unique stems, 8-fold prompt-stratified CV (held-out prompts never seen during their fold's training). Reference run: `bench/dcw/results/20260504-1831-v4-fusion-head-prototype/`.

| Metric | Threshold | Prototype |
|---|---|---|
| r(α̂_p, mean_s r) per-prompt | ≥ 0.6 | **+0.89** ✓ |
| r(α̂_p,s, r_p,s) seed-conditional | ≥ 0.7 | **+0.88** ✓ |
| r(σ̂_p, std_s r) | ≥ 0.4 | −0.01 ✗ |
| NLL improvement vs N(0, σ²_pop) | ≥ 15% | +5.7% ✗ |

Per-aspect r_seed: **1024² = +0.94 · 832×1248 = +0.88 · 1248×832 = +0.86** — the head generalizes across aspects via `aspect_emb` despite inv-HD-skewed data (112 vs 32 vs 32 rows). The α̂ gates pass strongly at ≈ 1/3 the proposal-spec data scale; v4's central hypothesis is confirmed without committing to A3's ~9–12h calibration sample.

**σ̂² supervision is too thin in the existing pool** (only one prompt has 5 seeds; most have 2). The σ̂² correlation gate fails, and NLL improvement is dominated by mean prediction. Implication for shipping today: **deploy α̂ without shrinkage** (caption-length backstop only). `make dcw` collects 3 seeds per prompt by default — Gate B reruns after that may unlock the σ̂² channel; if not, ship as-is.

**aspect_emb ablation** (`bench/dcw/results/20260504-1835-v4-fusion-head-no-aspect/`): r drops only ~0.01, rmse rises ~10–25 units (out of 230–350 baseline). Aspect is a tertiary input — c_pool + g_obs carry the predictive load. Kept for forward-compat with new aspect buckets and the chance that aspect-balanced training data lets it pull more weight.

## Architecture

Three inputs, one fusion head, online during inference.

### Input 1 — aspect prior (= v2/v3 Layer 1)
Per-aspect bucket profile `μ_g[aspect, i]`, `S_pop[aspect, i]`, `λ_scalar[aspect]`. Resolved at sampler init from `(H, W) → aspect_id`. Same artifact as v2/v3.

### Input 2 — prompt embedding
Mean-pooled `crossattn_emb_v0` → `c_pool` (1024-dim) + auxiliary scalars `[caption_length, cos(c_pool, μ_centroid), token_l2_std]` (3-dim).

### Input 3 — observed prefix gap
First k=7 LL-band gap measurements `g_obs[0:k]`. Computed at runtime using the same single-level orthonormal 2D Haar DWT path as `scripts/dcw/measure_bias.py`, extracted into a runtime module (`library/inference/dcw_observation.py`).

### Fusion head
Single shared MLP `h(c_pool, aspect_emb, g_obs[0:k], aux) → (α̂, log σ̂²)`:
- **Input**: c_pool (1024) + learned aspect_emb (16) + g_obs (7) + aux (3) = 1050-dim
- **Architecture**: LayerNorm → Linear→GELU→Dropout(0.2)→Linear→GELU→Dropout(0.2)→Linear, 1050 → 256 → 128 → 2
- **Params**: ~285k (single shared head; aspect is an input)
- **Output**: scalar `(α̂, log σ̂²)` for the *remaining* (N − k) steps
- **Targets**: per-seed integrated residual on the tail, `r_p,s = Σ_{i=k}^{N} (g_p,s[i] − μ_g[aspect, i])`
- **Loss**: Gaussian NLL (Kendall & Gal):
  ```
  L = (1/N) Σ_{p,s} [ (r_p,s − α̂_p)² / (2 σ̂²_p)  +  ½ log σ̂²_p ]
  ```
  + L2 weight decay (1e-3). Single shared head, all aspects together.

Single shared head (not per-bucket) means the parameter-to-data ratio is 3× better than v3, and the calibration set can be aspect-balanced rather than aspect-stratified.

### Why warmup is left unmodified by the head

During steps 0..k the controller applies only the bucket-prior correction (= v2's per-step lookup). The head fires once at step k with `g_obs[0:k]` collected from those uncorrected-by-head observations. Training data is collected the same way (baseline sampling with bucket-prior correction during warmup), so the train/inference observation distribution matches. (The prototype's training pool is *raw baseline*, no bucket-prior warmup applied — see Risk #9. `make dcw` should fix this for the production-shipped head.)

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
    alpha_eff = 0.0                                          # backstop kept

# Remaining steps k..N: bucket prior + head correction
tail_norm = sum(mu_g[k:])
for i in range(k, N):
    base_lambda = lam0 * (1 - sigma[i])
    bucket_corr = mu_g[i] * lam0 / S_pop[i]
    head_corr   = alpha_eff * mu_g[i] / tail_norm
    lambda_i    = clamp_overshoot(base_lambda + bucket_corr + head_corr)
    # ... DiT step (no extra DWT pass needed)
```

`σ²_prior[aspect]` is the average `σ̂²` over stable training prompts in that aspect (calibrated, not learned). If σ̂² channel never passes Gate B, the deployed controller skips the shrinkage step (sets `alpha_eff = alpha_hat`) and relies on the caption-length backstop.

### Reference artifact

`<base_dit_name>_dcw_v4.safetensors` (current trainer schema):

| key | shape | source |
|---|---|---|
| `head.<layer>.weight/bias` | MLP weights | trainer final-fit |
| `aspect_emb.weight` | (3, 16) | trainer final-fit |
| `bucket_prior_mu_g` | (3, 28) | per-aspect baseline mean from collected trajectories |
| `bucket_prior_S_pop` | (3, 28) | A2 two-point anchor at λ=+0.01 |
| `bucket_prior_lam_scalar` | (3,) | LSQ-optimal per bucket |
| `centroid_c_pool` | (1024,) | mean of training c_pool, for cos-aux |
| `aux_mean` / `aux_std` | (3,) | aux standardization |
| `g_obs_mean` / `g_obs_std` | (7,) | observation standardization |
| `sigma2_prior` | (3,) | average head σ̂² per bucket on training set |

Metadata: `schema=dcw_v4_fusion_head`, `k_warmup`, `n_aspects`, `aspect_names`, `n_steps`, `text_variant`, `n_train_rows`, `sigma2_pop`. **Not yet included** (TODO before ship): `tau_short[aspect_id]` for caption-length backstop — currently disabled (set to 0).

## A2 result (resolved 2026-05-04)

Two-point anchor (baseline + λ=+0.01) at production CFG=4, n=12 × 2 per bucket. Per-step `S_pop[aspect, i] := (g_λ[i] − g_base[i]) / 0.01` is the schedule-baked sensitivity (the `(1-σ)` factor is folded into the measurement since the bench applies `λ × (1-σ)` per step).

### Per-bucket calibration

| Bucket (LL only) | baseline ∫g | ∂∫g/∂λ | `λ_scalar` (LSQ) | ∫\|g\| @ baseline | ∫\|g\| @ λ_scalar |
|---|---|---|---|---|---|
| 1024² | **−188.20** | −362 | **+0.0046** | 257 | 256 (≈0%) |
| 832×1248 (HD) | +88.94 | +1313 | **+0.0059** | 429 | 434 (−1.4%) |
| 1248×832 (inv-HD) | +204.75 | −225 | **+0.0127** | 421 | 414 (+1.4%) |

`λ_scalar[aspect]` chosen by least-squares per-step zeroing: `λ* = −Σ(μ_g · S_pop) / Σ(S_pop²)`. Zero-integrated alternative `λ* = −∫μ_g / ∫S_pop` produces wildly larger values that explode per-step error — rejected.

### Two structural findings

**1. Square 1024² CFG=4 is NOT near-zero gap.** v2's table claimed ≈ −2; A2 measures **−188** at n=24. Likely v2's earlier estimate came from too few samples or a different mod-guidance config. Square needs the same Layer 1 treatment as the non-square buckets, not a free pass.

**2. Per-step `S_pop` sign-flips mid → late in every bucket.**

```
                 early(0-9)   mid(10-19)   late(20-27)
1024²              +35           +117         −235
832×1248           +43           +208         −150
1248×832           +36           +155         −266
```

A single `λ_scalar × (1−σ)` controller can either zero the integrated gap OR minimize per-step error — not both — because the same λ_scalar simultaneously over-corrects mid steps and under-corrects late steps (or vice versa). The sign-flip is structural (sequential trajectory feedback: correction at step *i* shifts `x_{i+1}` and the cumulative gap downstream), not measurement noise.

### Implication for v4 architecture

**The bucket prior (Layer 1) is essentially cosmetic for integrated |gap| reduction (~0% to ±1.4%).** All meaningful correction must come from the v4 head + observation channel. This is consistent with v3's PC1=98% finding that per-prompt amplitude dominates the bucket-conditional shape, *and* with the prototype head's strong α̂ gates above.

This *strengthens* the case for v4's architecture (head-driven correction, bucket prior as starting point) and *weakens* the value of investing more in Layer 1. Layer 1 is kept for two reasons: (1) **continuity** — non-zero baseline so warmup steps aren't unmodified, head learns to predict residual on top; (2) **fallback safety** — if v4 head fails Gate B, the bucket prior still applies, no regression vs no-DCW.

## Implementation tasks

| # | Status | Deliverable | Files |
|---|---|---|---|
| **I1** | TODO | `make dcw` sampling wrapper | extend `scripts/tasks/training.py` (or new `scripts/tasks/dcw.py`) — calls `scripts/dcw/measure_bias.py --dump_per_sample_gaps` over 3 buckets at production env, then chains to I2 |
| **I2** | ✓ done | Head training script | `scripts/dcw/train_fusion_head.py` |
| **I3** | ✓ done | Reference artifact aggregator | folded into I2 (single `fusion_head.safetensors` output) |
| **I4** | TODO | Online observation module | new `library/inference/dcw_observation.py` — DWT + LL-gap measurement during warmup. **Critical:** reuse CFG cond-branch forward where possible to avoid 2× warmup cost |
| **I5** | TODO | Inference module | new `library/inference/dcw_online.py` — `OnlineFusionDCWController` with warmup-loop observation, fusion at step k, head correction for tail. Wires into `library/inference/generation.py:343` and `:675` |
| **I6** | TODO | Inference flags | `inference.py` — `--dcw_v4`, `--dcw_v4_warmup_k <int>`, `--dcw_v4_disable_shrinkage`, `--dcw_v4_disable_backstop` |
| **I7** | TODO | Validation harness | `bench/dcw/calibrate_per_lora.py` extension — Gates A/B/C with the 4-way side-by-side grid generator |
| **I8** | TODO | Gate runs | A: 32×2×3 buckets · B: 48×3×3 held-out · C: perceptual side-by-side |
| **I9** | TODO | Doc updates | move this doc to `docs/methods/dcw-calibrator.md` after Gate C; update with shipping config + fallback policy |
| **I10** | TODO | Caption-length backstop | A6 sweep + write `tau_short[aspect]` into the artifact. Skippable if Gate C passes without it |

I4 + I5 dominate (~2 days); I1 / I6 / I9 are thin (~half a day each). I7 / I8 are gate work, ~1.5 days dominated by perceptual C.

## Quality gates

### Gate A — overshoot guard (= v2/v3)
Held-out 32 × 2 prompts × 3 buckets at CFG=4. Per-bucket overshoot fraction ≤ 2% with controller on. Unchanged.

### Gate B — head calibration

Re-run after `make dcw` regathers calibration data with 3 seeds per prompt:

| Metric | Threshold | Prototype | After `make dcw` |
|---|---|---|---|
| r(α̂_p, mean_s r) | ≥ 0.6 | +0.89 ✓ | TBD |
| r(α̂_p,s, r_p,s) | ≥ 0.7 | +0.88 ✓ | TBD |
| r(σ̂_p, std_s r) | ≥ 0.4 | −0.01 ✗ | TBD (3-seed supervision) |
| Held-out NLL improvement | ≥ 15% | +5.7% ✗ | TBD |
| Calibration: \|r − α̂\|² vs σ̂² (binned) | within 2× across all bins | not run | TBD |

If `r(α̂_p,s, r_p,s) seed-conditional` drops below 0.7 after `make dcw` → fall back to v3. (Unlikely — the prototype already passes at thinner data.)

If `r(σ̂)` still fails after `make dcw` → ship α̂ without shrinkage; use the caption-length backstop only.

### Gate C — perceptual side-by-side
4 LoRAs (flat / painterly / detail-dense / base) × 12 prompts × {square, inverse-HD} × {v4, v3, v2, scalar}. Four-way comparison.

| Metric | Threshold |
|---|---|
| v4 vs v3 overall preference | ≥ 55% prefer v4 |
| v4 vs v2 overall preference | ≥ 60% prefer v4 |
| Inverse-HD subset, v4 vs v3 | ≥ 60% prefer v4 (this is where observation is supposed to add over v3) |
| Flat LoRA × short prompts (caption-length backstop active) | v4 ≥ scalar (don't regress) |
| No (LoRA × bucket) cell where v4 loses ≤ 30% | else fall back to v3 for that cell |

If v4 ≤ v3 perceptually overall → ship v3 if its gates pass, else v2.

## Fallback ladder

Three tiers, resolved at controller init:

1. **Full v4 ship** if all gates pass (with or without shrinkage depending on Gate B σ̂² result).
2. **Mixed v4/v3** if v4 Gate B passes on a subset of buckets. Resolution chain prefers v4 head where available, else v3 prompt-only head, else v2 bucket profile.
3. **v4 → v3 fallback** if v4 Gate B fails entirely → fall back to v3 (if its gates pass).
4. **v4 → v2 fallback** if both v4 and v3 fail → ship v2 from existing v2 doc.

## Risks

1. **Online gap measurement cost.** Each warmup step requires both a fwd-noise and rev-cond DiT pass + DWT. For k=7 of 28 steps:
   - **Naive**: 2× per-step cost during warmup → 25% × 100% extra = 25% wall-time overhead
   - **CFG reuse**: under CFG > 1 inference, the cond branch is already computed; only the fwd-noise pass is extra → ~12% overhead
   - **Mitigation**: `dcw_observation.py` must reuse CFG cond-branch outputs. If CFG=1 (no uncond), accept the 25% warmup cost.

2. **DWT integration.** The bench script's single-level orthonormal 2D Haar DWT needs to be extracted into a runtime module. Same wavelet basis, same level, same band selection. Low risk — bench code is already factored, and `networks/dcw.py` already has Haar DWT/IDWT helpers.

3. **k choice trade-off.** Smaller k = less overhead but worse prediction. Default k=7 from the 1543 r² curve; k=4 was marginal. Gate B can be re-run with `--k_warmup 4` after `make dcw` to revisit if overhead is a problem.

4. **σ̂ collapse.** **Confirmed at thin scale** (prototype σ̂² channel did not learn — r=−0.01). Mitigations: ε floor on σ̂², init `log σ̂² ≈ log(σ²_pop)` (already done), monitor histogram during training, abort if collapsed. **Fallback**: ship without shrinkage. `make dcw`'s 3-seed supervision is the next experiment.

5. **Per-LoRA divergence.** v2/v3 risk; v4 inherits. **Structural advantage**: the observation channel naturally captures actual LoRA behavior at runtime — flat-style LoRAs that diverge from the bucket profile will produce diverging g_obs, which the head can respond to. May reduce or eliminate per-LoRA-override need. Gate C (cross-LoRA) quantifies.

6. **Stacked LoRA composition.** v4 advantage (continued from #5): observation captures actual composed behavior at runtime, not the assumed pick-largest-multiplier behavior. Bucket profile fallback uses pick-largest as in v2/v3.

7. **CFG drift.** v4 head calibrated at CFG=4. CFG=2/6 → fall back to v3 prompt-only head (if its gates pass) or v2 bucket profile. Recalibrating per CFG would require k=7 observation under each CFG, so the artifact size grows with CFG bins.

8. **Distribution shift in g_obs between train and inference.** Critical correctness: training data must apply the bucket-prior correction during warmup so g_obs distribution matches inference. **The prototype was trained on raw-baseline g_obs** — inference applies bucket-prior, so there is a small unmeasured shift. Since A2 found the bucket prior is cosmetic (~0–1.4% gap reduction), the shift is likely small; `make dcw` must apply bucket-prior warmup correctly to close this.

9. **Mod-guidance interaction.** Calibration runs with mod-guidance on (production env). v4 head implicitly conditions on mod-guidance being present in g_obs. New `pooled_text_proj` checkpoints may shift g_obs distribution → head may need recalibration. Same risk as v3.

## What v4 is NOT

- **Not a per-step head.** Output is a scalar `(α̂, log σ̂²)` for the tail, distributed across remaining steps proportional to the bucket profile. Per-step heads were a v1 upgrade path; PC2 is only 0.95% of variance.
- **Not a multi-band controller.** LL-only by default. Multi-band ablation deferred until basic v4 ships.
- **Not a probe-based controller in the v1 sense.** v1's K=4 probe fit a closed-form α from early-step gaps and used it directly. v4's k=7 observation feeds a learned head that *also* has prompt and aspect — observation is one of three inputs, not the sole signal.
- **Not a CFG-aware controller.** Single CFG profile + single head at CFG=4.
- **Not a training loss change.** LoRA training untouched.

## Open questions (post-prototype)

- **Will σ̂² unlock at 3-seed supervision?** `make dcw`'s default 3 seeds per prompt is the experiment. If yes, ship with shrinkage. If no, ship without.
- **Multi-band g_obs.** LL is the dominant band; LH/HL/HH may add prediction lift. Defer until basic v4 ships and CFG-cost is acceptable.
- **Caption-length backstop necessity.** If observation channel works as well as the prototype suggests, the backstop may be redundant. Gate C with and without `--dcw_v4_disable_backstop` will tell.
- **Online vs once-at-step-k head firing.** Currently fires once at step k. Could fire at every step k..N with a running observation; lag-10 autocorr 0.44, lag-20 0.14 — expected marginal gain small. Defer until v4-base ships.
- **Generalization across `pooled_text_proj` checkpoints.** Mitigated somewhat by observation channel — if mod-guidance shifts the trajectory, g_obs sees the shift. Still owed: re-validate after any mod-guidance update.

## Concrete deliverables checklist

- [x] A1 — per-aspect baseline (folded into existing bench data; `make dcw` regathers cleanly)
- [x] A2 — per-aspect 2-point anchor at CFG=4 (`λ_scalar`: +0.0046 / +0.0059 / +0.0127; bucket prior cosmetic)
- [x] **Prototype head — Gate B α̂ channels pass at +0.89 / +0.88 r on 8-fold CV (n=176)**
- [ ] I1 — `make dcw` sampling wrapper (n=80 × 3 seeds × 3 buckets default)
- [x] I2 — head training script (`scripts/dcw/train_fusion_head.py`)
- [x] I3 — reference artifact aggregator (folded into I2)
- [ ] I4 — online observation module (CFG cond-branch reuse critical)
- [ ] I5 — fusion DCW controller
- [ ] I6 — inference flags
- [ ] I7 — validation harness
- [ ] I8 — Gate A / B / C runs
- [ ] I9 — docs (move to `docs/methods/dcw-calibrator.md`)
- [ ] I10 — caption-length backstop (defer if Gate C passes without)

Total remaining **~3 days** from `make dcw` first run to perceptual-gate result. v3/v2 fallback ladder adds zero days because both prior tiers are already analysis-phase-complete.

## Appendix — evidence underlying v4

| Finding | Source | Implication for v4 |
|---|---|---|
| Per-prompt seed std ≈ between-prompt std (LL: 349 vs 370) | `bench/dcw/results/20260504-1543-small_invhd` | Prompt-only head (v3) hits a ceiling; observation channel needed |
| 6/8 prompts sign-flip across 5 seeds (LL integrated gap) | same | Per-seed targets needed; k≥3 seeds for σ̂ validation |
| Lag-1 autocorr of LL gap within trajectory = +0.55 | same | Trajectory smooth enough for short-prefix extrapolation |
| corr(early-dev k=7, late-dev) = 0.89 seed-conditional | same | k=7 observation predicts late-trajectory with r²=0.79 |
| **Prototype head α̂ channels pass at r=0.88–0.89 on 8-fold prompt-CV (n=176)** | **`bench/dcw/results/20260504-1831-v4-fusion-head-prototype/`** | **v4 hypothesis confirmed on existing data; A3 not gating** |
| Aspect_emb ablation drops r by ~0.01 only | `bench/dcw/results/20260504-1835-v4-fusion-head-no-aspect/` | Aspect is tertiary; c_pool + g_obs carry the load |
| Per-step `S_pop` sign-flips mid → late in all 3 buckets at CFG=4 | A2 (`bench/dcw/results/20260504-{1648,1721,1747}`) | Bucket prior cosmetic (~0% gap reduction); head must carry the correction |
| Square 1024² CFG=4 baseline LL gap = −188 (not −2 per v2 doc) | A2, n=24 | Square is not a free pass — needs Layer 1 too |
| ICC=0.704 on integrated δ at production env | `bench/dcw/stability_predictor_check.py` (1406) | ~30% of per-prompt variance is between-seed |
| Caption length is strongest stability proxy at scalar level (n=16) | `project_dcw_cpool_feature_feasibility` | Aux scalars worth keeping in head input despite being weak alone |
| DCW gap sign-flips by aspect ratio under CFG=4 | `project_dcw_cfg_aspect_signflip` | Aspect prior must be per-bucket (= v2/v3 Layer 1, retained) |
| FM val loss doesn't track quality | `project_fm_val_loss_uninformative` | Gate C (perceptual) is the only ground truth; head NLL is necessary but not sufficient |
