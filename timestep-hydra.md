# σ-conditional HydraLoRA router

Extend HydraLoRA's layer-local router to take σ as an explicit input, so the expert
mixture becomes σ-dependent. Attacks internal representation specialization on the
B-side of each adapted Linear. Orthogonal to postfix-sigma (`cond-timestep` postfix
mode), which attacks K/V-input specialization.

Companion: `docs/methods/hydra-lora.md` (base architecture), `docs/methods/postfix-sigma.md`
(σ-conditional postfix — sibling track, already implemented).

---

## TL;DR

Add a σ-embedding branch to each HydraLoRAModule's router. The branch is a small
Linear that biases the expert logits. Zero-init so training starts identical to
base HydraLoRA; σ-dependence only emerges if gradients push it.

**Parameter cost**: one `Linear(sigma_feature_dim, num_experts)` per router. Few KB total.

**Pre-analysis result**: already run. Case A confirmed — explicit σ conditioning is
a free gain. See "Pre-analysis" below.

---

## Motivation

Current HydraLoRA router reads `pooled_x` (mean-pooled layer input) per layer per
sample (`networks/lora_modules.py:_compute_gate`). Layer input statistics vary
strongly with σ (high σ → noisy, low σ → clean), so the router *could* be implicitly
σ-aware — but the pre-analysis shows it effectively isn't.

TimeStep Master (Xu et al., ICML 2025) established that noise-level-dependent expert
specialization improves generation across UNet/DiT/MM-DiT; Diff-MoE, MoDE, EC-DiT,
and eDiff-I all converge on timestep-expert routing as useful. Track B is the
minimum viable version on top of existing HydraLoRA infrastructure — preserve
expert identity (each expert still means "this artist/style specialist"), only
modulate *when* each expert activates.

---

## Prior art

- **TimeStep Master (TSM)** (Xu et al., ICML 2025) — asymmetric mixture of timestep
  LoRA experts with multi-scale intervals and core-context gating; directly relevant.
- **Diff-MoE** (Cheng et al., ICML 2025) — DiT with expert-specific timestep conditioning.
- **MoDE** (Reuss et al., 2024) — noise-conditioned router on top of DiT.
- **EC-DiT** (Apple, ICLR 2025) — expert-choice routing adaptive across depth and timestep.
- **eDiff-I** (Balaji et al., 2022) — original timestep-expert paper.

---

## Three-option comparison

### Option 1 — Timestep-specific `lora_down` (A-side) — *rejected*

`lora_down: (K_time, rank, in_dim)` with σ selecting a bucket.

- **T-LoRA conflict**: T-LoRA already does timestep rank masking on the A-side.
  Stacking K per-bucket A matrices on top creates dual-layered timestep semantics.
- **Expert identity drift**: `B_e` must decode from K different A bases. "Expert 0
  always means artist 0" becomes ill-defined across timesteps.
- **Parameter blowup**: K × A + E × B per module.

### Option 2 — σ-conditional router (B-side) — *preferred*

Router input extended: `router(pooled_x, σ_emb) → gate_logits`:

```python
# Existing (networks/lora_modules.py:_compute_gate)
logits = self.router(pooled_x)                      # (B, num_experts)

# Proposed
sigma_feat = sinusoidal(sigma)                       # (B, sigma_feature_dim)
sigma_bias = self.sigma_mlp(sigma_feat)              # Linear → E logit biases, zero-init
logits = self.router(pooled_x) + sigma_bias
gate = softmax(logits, dim=-1)
```

Properties:

- **Orthogonal to T-LoRA**: T-LoRA acts on A-side rank (per-timestep); σ-conditional
  router acts on B-side mixture.
- **Expert identity preserved**: artist N expert stays artist N at all σ; σ only
  varies *when* and *how much* it activates.
- **Parameter cost trivial**: few KB per router.
- **Spectrum-compatible**: router not called on cached steps (all blocks skipped).
- **Zero-init diagnostic**: `|sigma_mlp.weight|` at convergence directly measures
  how much σ-conditioning was actually used.

### Option 3 — Full 2D (K × E grid) — *deferred*

`lora_up: (K, E, out, rank)` with σ selecting a bucket, router selecting expert
within it.

- **Parameter cost K×**: for K=3, E=4, 3× current HydraLoRA up-weights.
- **Dual-axis load balancing**: per-bucket expert balance + per-timestep bucket
  balance. Training-timestep sampling is non-uniform (logit-normal), which skews
  bucket balance.
- **Gradient dilution per slot**: each (k, e) slot sees ~1/(K×E) of effective
  batch signal. Expert-collapse risk amplified.

Revisit only if Option 2 shows large per-bucket gate-distribution divergence
after training.

---

## Pre-analysis — ran (Case A confirmed)

**Script**: `bench/hydralora/analyze_router_sigma_correlation.py`.

**Procedure**: load trained HydraLoRA checkpoint → wrap each HydraLoRAModule's
`_compute_gate` → run forward over N samples × several σ → bucket σ equal-frequency
→ compute per-bucket mean gate distribution per module → pairwise JS divergence
across buckets → decision tree.

**Result on `anima-hydra-0417-324.safetensors`** (196 modules, 8 samples × 5 σ):

| Metric | Value |
|---|---|
| median max-pairwise JS | **0.00001** |
| mean | 0.00037 |
| p90 | 0.00101 |
| max (single worst module) | 0.01006 |
| **Case** | **A** (≪ 0.05 cutoff) |

Decision: **implement Option 2**. Current router is effectively σ-blind in per-bucket
mean gate — explicit σ conditioning is free gain.

**Secondary findings** (worth knowing, not load-bearing):
- The top-10 most-σ-varying modules are all `cross_attn.q_proj` (7/10) or deep-block
  `self_attn.qkv_proj`; MLP layers are totally flat (median 0).
- Mid-block peak at blk14–17 for cross-attn.q — matches Phase 0's "middle-block
  functional gap" depth range. Weak but consistent signal that σ-routing pressure
  exists exactly where the K/V gap is largest.
- Recommended re-run before committing: `--num_samples 32 --sigmas "0.05,0.15,0.25,0.35,0.5,0.65,0.8,0.95"`
  to rule out noise at the JS ≈ 0.01 tail.

Full JSON: `bench/hydralora/results/sigma_correlation_0417-324.json`.

---

## Option 2 — implementation design

**Module change** (`networks/lora_modules.py:HydraLoRAModule.__init__`):

```python
self.sigma_feature_dim = sigma_feature_dim  # new ctor arg, default 128
self.sigma_mlp = nn.Linear(sigma_feature_dim, num_experts, bias=False)
nn.init.zeros_(self.sigma_mlp.weight)       # zero-init → identity to base router
```

**Gate computation**:

```python
def _compute_gate(self, x_lora, sigma):
    pooled = _mean_pool(x_lora)
    logits = self.router(pooled)
    if sigma is not None:
        sigma_feat = _sinusoidal(sigma, self.sigma_feature_dim)
        logits = logits + self.sigma_mlp(sigma_feat)
    return torch.softmax(logits, dim=-1)
```

Sinusoidal σ features match the DiT t_embedder functional form (same approach as
postfix-sigma — inline, not a dit coupling). σ in [0, 1].

**Plumbing σ to the module**: the `forward(x)` signature doesn't carry σ today.
Options:
- (a) Register a `set_sigma(σ)` method on the network that stashes σ on every
  HydraLoRAModule, called once per training step from `train.py:527` alongside
  `set_timestep_mask`. Mirrors the existing T-LoRA pattern.
- (b) Thread σ through `org_module.forward` via `transformer_options` (more
  invasive, no precedent in this repo).

Prefer (a) for consistency with T-LoRA. One call site to update.

**Save/load format**: add `sigma_mlp.weight` per module to the saved state_dict.
Old `*_moe.safetensors` files lack this key → falls back to zero-init (recovers
base HydraLoRA gate). Backward-compatible read, forward-incompatible write.

---

## Load balancing modification

Current Switch Transformer load-balancing loss (`networks/lora_anima.py:get_balance_loss`)
aggregates gates globally over the batch. With σ-conditional routing, global balance
can hide per-bucket collapse (e.g., expert 0 only at high σ, expert 1 only at low σ —
globally balanced, per-bucket one-hot).

**Proposed**: add per-σ-bucket balance term:

```python
L_global = switch_loss(all_gates)                    # existing
L_per_bucket = mean(switch_loss(gates_in_bucket))    # new
L_total = L_global + alpha_bucket * L_per_bucket
```

Start with `alpha_bucket ≈ 0.3 × alpha_global`. Higher suppresses σ-specialization
(undesired); lower allows per-bucket collapse. Tune empirically based on observed
bucket-wise gate entropy.

Bucket assignment at training time: σ is already in `[0, 1]`; bucket by quantile
(e.g., 3 equal-frequency bins computed on a rolling window of recent step σ).

---

## Compatibility analysis

| Component | Compat | Notes |
|---|---|---|
| T-LoRA | ✅ orthogonal axes | T-LoRA: A-side rank × σ. Track B: B-side expert × σ. Ablation needed for marginal contribution of each. |
| Spectrum | ✅ | Cached steps skip all hydra modules (no blocks run). Router not called. No interaction. |
| Modulation guidance | ✅ orthogonal | Modulation = adaLN. Hydra = internal Linear adapters. |
| Existing `*_moe.safetensors` | ⚠ new field | `sigma_mlp.weight` must be added to saved format. Old files fall back to σ=0 equivalent (zero-init). |
| ComfyUI hydralora node | ⚠ degraded | Node pins gates uniformly. σ-conditional gates can't be expressed in its current interface. Options: (a) pin σ to a user-chosen value, (b) document limitation, (c) extend the node. Default: (b). |
| postfix-sigma (cond-timestep) | ✅ orthogonal paths | Postfix: K/V input. Hydra: internal projections. Stackable. |

---

## Training recipe

- **Base**: existing HydraLoRA pipeline (`networks/lora_modules.py:HydraLoRAModule`,
  `configs/methods/hydralora.toml`). Cache LLM adapter outputs required as today.
- **Mods**:
  - `HydraLoRAModule.__init__` adds `sigma_mlp`.
  - `_compute_gate` gains a `sigma` argument (via `set_sigma` path).
  - `get_balance_loss` gains a per-bucket term.
- **Warmup**: because `sigma_mlp.weight` is zero-init, early training is identical
  to base HydraLoRA. No special warmup needed.
- **σ sampling**: same as base (logit-normal for rectified flow). Per-σ-bucket
  balance term ensures even the under-sampled tails learn gate distributions.

---

## Evaluation

- **Primary**: per-artist generation quality vs. base HydraLoRA on held-out prompts.
  Expected signal: cleaner style separation at late σ (detail), tighter layout at
  early σ.
- **Diagnostic**: `|sigma_mlp.weight|` per module at convergence. Near-zero →
  Track B didn't learn σ-dependence (either the pre-analysis was misread or σ
  genuinely has no signal here).
- **Gate visualization**: per-artist, per-σ-bucket mean gate distribution. Story:
  different artists should show different σ-specialization patterns.
- **Ablation**: T-LoRA on/off × Track B on/off (4-cell grid). Identifies marginal
  contribution of each timestep-signal path.
- **Re-run pre-analysis after training**: compute the same JS-divergence on the
  Track-B-trained checkpoint. If median JS is still small, Track B didn't use its
  capacity — consider Option 3 or drop.

---

## Risks and mitigations

1. **Per-bucket expert collapse** — balance loss as above; monitor bucket-wise gate
   entropy during training.
2. **σ-conditioning redundant with implicit `x`-based routing** — pre-analysis
   already run (Case A); redundancy unlikely, but confirm by comparing Track B
   on/off ablation.
3. **T-LoRA + Track B double-counting σ** — four-cell ablation. Drop whichever
   contributes less.
4. **ComfyUI node incompatibility** — document limitation; CLI inference uses
   the σ-conditional path correctly. Node extension is a separable follow-up.

---

## Phases

### Phase B0 — pre-analysis — ✅ done

Case A confirmed on `anima-hydra-0417-324.safetensors`. Proceed to B1.

Recommended sanity re-run before B1: `--num_samples 32` with denser σ.

### Phase B1 — Option 2 implementation (~3 days)

- Add `sigma_mlp` + `set_sigma` to `HydraLoRAModule` and the network wrapper.
- Wire `set_sigma` call from `train.py` next to `set_timestep_mask`.
- Add per-σ-bucket balance loss to `get_balance_loss`.
- Extend save/load so moe safetensors gains `sigma_mlp.weight` per module.
- Config: `configs/methods/hydralora.toml` — new `hydralora_sigma` block.
- Train on existing multi-artist dataset.

### Phase B2 — evaluation + ablations (~2 days)

- Primary generation eval + gate visualization.
- Post-training JS-divergence re-analysis.
- T-LoRA × Track B four-cell ablation.
- Decide whether σ-conditioning is worth keeping as default.

---

## Open questions

1. **Is σ-correlation block-depth-dependent?** The B0 result shows mid-block
   (blk14–17) and late-block (blk23–27) carry the most σ-correlation signal in
   the *current* σ-blind router. If σ-pressure is concentrated there, Track B
   could apply `sigma_mlp` only to those blocks instead of all 28.
2. **How does the trained router's σ-specialization correlate with postfix-sigma's
   σ-residual?** If the two learned σ-specializations are highly correlated, one
   path is redundant. Measure after both land.
3. **Artist × σ interactions.** If artist specialization is concentrated at
   specific σ (e.g., artist A's expert dominates mid-σ, B's dominates early σ),
   gate visualizations should expose it. Useful for understanding *what* Track B
   learned.
