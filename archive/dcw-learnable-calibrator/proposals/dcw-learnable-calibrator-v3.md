# DCW Calibrator v3 (per-aspect bucket + heteroscedastic prompt head)

**Status:** **superseded by [v4](../../../docs/proposal/dcw-learnable-calibrator-v4.md)** (online observation + prompt + aspect fusion). v3 retained as a fallback above v2 if v4's online observation channel is unworkable. The 5-seed sweep (`bench/dcw/results/20260504-1543-small_invhd`) showed per-prompt seed std ≈ between-prompt std on LL — v3's prompt-only head has no input to disambiguate seed-realization, and the heteroscedastic shrinkage will collapse it toward zero on the seed-noise-dominated half of prompts. v4 adds observed early-step gap measurements to break the symmetry.

**Original status:** proposal · supersedes [v2](./dcw-learnable-calibrator-v2.md) (which remains the **fallback floor implementation** if v3 analysis gates fail) · **Effort:** ~5-7 days · **Cost at inference:** bucket lookup + 1 MLP forward (head) + per-step cap check · **Reference papers:** Yu et al. arXiv:2604.16044 (DCW); Kendall & Gal NeurIPS 2017 (heteroscedastic aleatoric uncertainty)

## Why v3 (over v2)

v2 ships per-aspect-bucket profiles and stops there. The 1406 stability analysis (`bench/dcw/stability_predictor_check.py`) showed that bucket-level profiles leave **~70% of per-prompt variance on the table** at inverse-HD CFG=4 (ICC=0.704 on integrated δ). v3 recovers most of that with a prompt-conditioned head while keeping v2 as the safe fallback.

The two big constraints v3 has to respect, established by analysis:

1. **Per-prompt seed-stability is bimodal**: 50% of prompts have SNR ≥ 1.5 (clear signal); 50% have SNR < 1 (seed-noise-dominated). A head that fits all prompts equally will be polluted by the noisy half.

2. **Simple c_pool scalars do NOT predict per-prompt amplitude** at n=16 (best r=+0.26). The head must use the full c_pool vector via a real MLP, requiring ≥200 calibration prompts. Scalars do *weakly* predict seed-variance (cos(centroid)/token-spread r≈0.45, p=0.08; caption length p=0.10), so they belong as auxiliary inputs, not as the main feature.

The architecture that addresses both: **heteroscedastic regression** (Kendall & Gal). The head outputs `(α̂_p, log σ̂²_p)` and is trained with Gaussian NLL using per-seed targets. The `1/σ̂²` term in the loss naturally penalises overconfident fits on noisy prompts; at inference, the prediction is shrunk toward the bucket prior by `σ²_prior / (σ²_prior + σ̂²)`. Noisy prompts naturally contribute zero correction without an explicit stability gate.

## Architecture

Three layers, conditional on the previous's evidence holding.

### Layer 1 — per-aspect bucket profile (= v2)
Resolved at sampler init from `(H, W) → aspect_id`. Provides `μ_g[aspect, i]`, `S_pop[aspect, i]`, `λ_scalar[aspect]`. Same artifact as v2; v3 reuses it unchanged.

### Layer 2 — heteroscedastic prompt head (new)
MLP `h(c_pool, aux) → (α̂_p, log σ̂²_p)`:
- **Input**: mean-pooled `crossattn_emb_v0` (1024-dim) + auxiliary scalars `[caption_length, cos(c_pool, μ_centroid), token_l2_std]` (3-dim, derived from same embed).
- **Architecture**: 3-layer MLP, 1024+3 → 256 → 128 → 2, GELU, layer norm. ~280k params.
- **Targets**: per-seed integrated residual `r_p,s = Σ_i (g_p,s[i] - μ_g[aspect, i])`.
- **Loss**: Gaussian NLL with per-seed targets:
  ```
  L = (1/N) Σ_p Σ_s [ (r_p,s − α̂_p)² / (2 σ̂²_p)  +  ½ log σ̂²_p ]
  ```
  Plus a small L2 weight decay (1e-4). Train per aspect bucket separately (head per bucket).

### Layer 3 — caption-length backstop (cheap pre-shrinkage)
If `nonpad_tokens(c_pool) < τ_short`, multiply `α̂_p` by `0` (or attenuate by 0.5; see A4). Independent of head confidence. Catches the user-visible flat-style failure (`project_dcw_when_to_use`) which we know correlates with short prompts.

### Inference flow
```python
# Setup (once per inference)
aspect_id            = resolve_aspect(H, W)
mu_g, S_pop, lam0    = bucket_profile[aspect_id]
head                 = prompt_head[aspect_id]            # may be None → v2 fallback
c_pool, aux          = pool_text_embed(crossattn_emb)
nonpad               = aux["caption_length"]

# Per-prompt prediction
if head is None:
    alpha_eff = 0.0                                       # v2 path
else:
    alpha_hat, log_sigma2 = head(c_pool, aux)
    sigma2 = exp(log_sigma2)
    shrinkage = sigma2_prior[aspect_id] / (sigma2_prior[aspect_id] + sigma2)
    alpha_eff = alpha_hat * shrinkage
    if nonpad < tau_short[aspect_id]:
        alpha_eff = 0.0                                   # backstop

# Per-step (= v2 + α_eff term, overshoot guard unchanged)
for i in range(n_steps):
    base_lambda = lam0 * (1 - sigma[i])
    correction  = alpha_eff * mu_g[i] / sum(mu_g)         # distribute α across steps
    lambda_proposed = base_lambda + correction / S_pop[i]
    # ... overshoot cap as in v2
```

`σ²_prior[aspect]` is set to the average `σ̂²` over stable training prompts in that bucket (calibrated, not learned).

### Reference artifacts (extends v2's)

`<base_dit_name>_dcw_v3.safetensors`:
- (all of v2's per-aspect profile)
- `prompt_head[aspect_id]`: MLP weights (~280k params × 3 buckets ≈ 3 MB total)
- `tau_short[aspect_id]`: caption-length threshold per bucket (calibrated from A4)
- `sigma2_prior[aspect_id]`: average head σ̂² on stable prompts per bucket
- metadata: head architecture, training hyperparams, calibration n per bucket, validation NLL/MSE per bucket, git SHA

## Quality gates

### Gate A — overshoot guard (= v2 Gate A)
Unchanged. Held-out 32 × 2 prompts × 3 buckets at CFG=4. Per-bucket overshoot fraction ≤ 2% with controller on.

### Gate B — head calibration (NEW, blocking)
Held-out 48 prompts × 2 seeds × {3 buckets} not in training set:
| Metric | Threshold | Action |
|---|---|---|
| Pearson r(α̂_p, mean_s r_p,·) per bucket | ≥ 0.5 | Head amplitude tracks ground truth |
| Pearson r(σ̂_p, |r_p,1 − r_p,2|/√2) per bucket | ≥ 0.4 | σ̂ tracks ground-truth seed noise |
| Mean Gaussian NLL on held-out vs N(0, σ²_pop) baseline | NLL(head) < NLL(pop) by ≥ 5% | Head better than constant predictor |
| Calibration: empirical |r_p − α̂_p|² vs predicted σ̂² (binned) | within 2× across all bins | Confidence is well-calibrated |

If r(α̂_p, ·) < 0.5 on any bucket → head architecture is failing on that bucket → **fall back to v2 for that bucket only**, ship per-bucket-mixed.

If r(σ̂_p, ·) < 0.4 globally → uncertainty channel is uninformative → **fall back to v2 entirely** (no shrinkage means raw `α̂_p` is dangerous on unstable prompts).

### Gate C — perceptual side-by-side (= v2 Gate B, expanded)
4 LoRAs (flat / painterly / detail-dense / base) × 12 prompts × {square, inverse-HD} × {v3, v2, scalar}. Three-way comparison.

| Metric | Threshold |
|---|---|
| v3 vs v2 overall preference | ≥ 55% prefer v3 |
| v3 vs scalar overall | ≥ 60% prefer v3 |
| Inverse-HD subset, v3 vs v2 | ≥ 60% prefer v3 (this is where the head is supposed to add over v2) |
| Flat LoRA × short prompts (caption-length backstop active) | v3 ≥ scalar (don't regress) |
| No (LoRA × bucket) cell where v3 loses ≤ 30% | else fall back to v2 for that cell |

If v3 ≤ v2 perceptually overall → ship v2 as planned, archive v3.

## Analysis phase (~2 days)

| # | Run | Output | Why |
|---|---|---|---|
| **A1** | Per-aspect baseline at production env, n=48 × 2 seeds × 3 buckets (= v2 §A1) | per-aspect `gaps_per_sample.npz` + `μ_g`, `S_pop` | Bucket profile (Layer 1). Reuse v2's run if already done. |
| **A2** | λ-sweep per aspect, n=12 × 2 (= v2 §A2) | `S_pop`, `λ_scalar` per bucket | Layer 1 sensitivity. |
| **A3** | **Calibration run (NEW): n≥200 × 2 seeds × 3 buckets at production env, baseline-only (no DCW)** | `gaps_per_sample.npz` per bucket → per-seed integrated residuals `r_p,s` | Training data for head. ~2 hours wall on a 5060 Ti per bucket. |
| **A4** | Caption-length threshold sweep on A3 data | `τ_short[aspect]` per bucket | Pick τ that maximises `mean SNR (kept) − mean SNR (dropped)`. |
| **A5** | Head training: train+validate on A3 with 80/20 prompt split, per bucket | head weights, per-bucket validation NLL/r(α̂,r)/r(σ̂,σ_seed) | Gate B inputs. |
| **A6** | Per-LoRA cross-check: rerun A3 mini (n=64 × 2) on 1 painterly + 1 flat LoRA, per aspect (~1.5 hours each) | per-LoRA per-aspect head divergence | Closes v2 §A3 / `project_dcw_when_to_use`. |

**Decision points after analysis:**

| Outcome | Action |
|---|---|
| A5 r(α̂_p, r_p) ≥ 0.5 and r(σ̂_p, σ_seed) ≥ 0.4 on all buckets | Proceed to v3 implementation full ship |
| A5 r(α̂) ≥ 0.5 but r(σ̂) < 0.4 | Ship `α̂` without shrinkage but with caption-length backstop only; risky on unstable prompts. **Recommended:** fall back to v2 |
| A5 r(α̂) < 0.5 on inverse-HD only | Ship v3 on square buckets, v2 on inverse-HD (mixed) |
| A5 r(α̂) < 0.5 on all buckets | **Fall back to v2.** v3 architecture doesn't transfer; reconsider |
| A6 finds per-LoRA profile divergence > 50% on any LoRA × bucket | Per-LoRA head override path required; +1 day implementation |

## Implementation phase (~3 days)

| # | Deliverable | Files | Effort |
|---|---|---|---|
| **I1** | A3 calibration runner | `bench/dcw/calibrate_v3.py` (extends `measure_bias.py` with prompt-iteration over n=200 cached samples; reuses CSV/npz writers) | 3h |
| **I2** | Head training script | `bench/dcw/train_prompt_head.py` (loads A3 npz + text-emb caches, trains per-bucket MLPs with NLL, dumps `prompt_head_<bucket>.safetensors`) | 4h |
| **I3** | Reference artifact aggregator | extend v2's exporter to bundle bucket profiles + heads + tau_short + sigma2_prior into one `.safetensors` | 2h |
| **I4** | Inference module | new `library/inference/dcw_prompt_head.py` — `PromptHeadController(aspect_id, head_state_dict, ...)` with `predict(c_pool, aux) → (alpha_eff, info)`; integrate with `library/inference/dcw_calibration.py` resolution chain | 4h |
| **I5** | Inference flags | `inference.py` — `--dcw_v3_disable` (forces v2), `--dcw_v3_shrinkage_disable` (raw α̂), `--dcw_v3_backstop_disable` | 1h |
| **I6** | Validation harness | `bench/dcw/calibrate_per_lora.py` — Gates A, B, C with the 3-way side-by-side grid generator | 4h |
| **I7** | Gate runs | A: 32×2×3 buckets; B: 48×2×3 buckets held-out; C: perceptual | 1.5 days, dominated by C |
| **I8** | Doc updates | If pass: `docs/methods/dcw-calibrator.md` with v3 architecture, gate results, fallback policy. If fail: archive v3 with negative result; v2 ships. | 2h |

## Fallback to v2

The mixed ship is real and should be planned for. Three flavours:

1. **Full v2 ship** if Gate A or Gate B fails entirely. v2 doc remains the source of truth; v3 archived.
2. **Mixed v2/v3** if Gate B passes on a subset of buckets. Resolution chain prefers v3 head where available, else v2 bucket profile.
3. **v3 with shrinkage disabled** if r(σ̂) fails but r(α̂) passes. Caption-length backstop becomes the only protection against noisy prompts. **Not recommended** — flat-style failures will still leak through; cleaner to fall back fully.

## Risks

1. **Calibration cost.** A3 is ~6 hours on one GPU (3 buckets × 2h). Not free; not a casual experiment. v2 ships in ~2 days; v3 in 5-7. Mitigated by v2 fallback being the default if any gate fails.

2. **Head overfit at n=200.** ~280k params on 200 × 2 = 400 samples. Mitigated by L2 + dropout (0.1) + early stopping on the held-out 20%. Auxiliary scalars (caption length, centroid-cos) provide regularisation by exposing low-dim structure.

3. **σ̂ collapse.** Without enough variation in `r_p,s` across seeds, NLL training can collapse `σ̂² → 0` and degenerate to pure MSE. Mitigated by:
   - Floor: `σ̂² = exp(log σ̂²) + ε`, ε=1e-3
   - Initialise `log σ̂²` near `log(σ²_pop)` so the model starts at the constant-noise baseline
   - Monitor σ̂ histogram during training; abort if collapsed

4. **Per-LoRA divergence.** v2 §A3 / `project_dcw_when_to_use` failure mode. v3 same risk: flat-style LoRAs may need their own head. A6 closes; per-LoRA override path is +1 day.

5. **Inter-aspect head transfer untested.** v3 trains a head per bucket. If the heads share structure (same `α̂` ↔ same prompts across aspects), one shared head + aspect-id input would simplify. Untested because no inter-aspect prompt overlap in cache (`project_dcw_cpool_feature_feasibility` and the existing-data check). Open question, not blocking — per-bucket head is the safe default.

6. **CFG drift.** v3 head is calibrated at CFG=4. CFG=2/6 → fall back to v2 bucket profile or scalar.

7. **Stacked LoRA composition.** Pick-largest-multiplier for the bucket profile (v2 unchanged). For the head: pass the user prompt's c_pool regardless of LoRA stack — head is text-conditioned, not LoRA-conditioned.

## What this is not

- **Not a per-step head.** Head outputs scalar `(α̂, log σ̂²)`, distributed across steps proportional to the bucket profile. Per-step heads were the v1-shape upgrade path; not adopted because PC2 (the per-step shape variation) is only 0.95% of variance, not worth the parameter cost.
- **Not a CFG-aware controller.** Single CFG profile + single head per bucket at CFG=4. CFG sweep would multiply artifact size without clear evidence it helps.
- **Not a probe-based controller.** No K-step probe at inference. Stability is fully delegated to `σ̂` from the head + caption-length backstop.
- **Not a multi-band head.** LL-only.
- **Not a training loss change.** LoRA training untouched.

## Open questions

- **Does shared-head + aspect-id input outperform per-bucket heads?** Untested — would need inter-aspect prompt overlap (currently zero in cache). Optional: re-preprocess 24 source images to multiple aspects, retrain shared head, compare.
- **Caption-length threshold sensitivity.** A4 picks one τ per bucket from the 200-prompt training set. Cross-validation on held-out prompts only. Real perceptual sensitivity is captured by Gate C.
- **σ²_prior choice.** "Average σ̂² on stable training prompts" is one option; another is `Var(r_p,s) / 2` from training data directly. Pick whichever produces less aggressive shrinkage on validation Gate B.
- **Does the head generalise across pooled_text_proj checkpoints?** Mod-guidance is on by default in measure_bias; the head will be calibrated with the production 0429 pooled_text_proj. New mod-guidance versions may shift the c_pool distribution.

## Concrete deliverables checklist

- [ ] A1 — per-aspect baseline (3 buckets × n=48) [reuse v2 §A1 if done]
- [ ] A2 — per-aspect λ-sweep [reuse v2 §A2 if done]
- [ ] A3 — n=200 × 2 calibration per bucket
- [ ] A4 — caption-length threshold sweep
- [ ] A5 — head training + validation NLL / r(α̂) / r(σ̂)
- [ ] A6 — per-LoRA cross-check (1 painterly + 1 flat)
- [ ] **Decision point: full v3 / mixed / fall back to v2**
- [ ] I1 — calibration runner
- [ ] I2 — head training script
- [ ] I3 — reference artifact aggregator
- [ ] I4 — inference module
- [ ] I5 — inference flags
- [ ] I6 — validation harness
- [ ] I7 — Gate A / B / C runs
- [ ] I8 — docs (v3 ship docs OR v2 fallback docs + archived v3 negative result)

Total **~5-7 days** from start to perceptual-gate result. v2 fallback adds zero days because v2 is already analysis-phase-complete after A1+A2; if v3's gates fail at the decision point, v2 ships from the existing v2 doc with no extra implementation work.
