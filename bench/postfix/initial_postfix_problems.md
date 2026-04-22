# Postfix σ-residual analysis — why the current checkpoint is a domain prior, not a prompt enhancer

Empirical findings from `bench/postfix/analyze_sigma_tokens.py` on `output/anima_postfix.safetensors`:

- **All K=32 slots are literally identical** — effective K=1. Zero-init of the final MLP layer (`cond_mlp[-1]`, `sigma_mlp[-1]`) combined with positional-symmetric splice (all K slots land in the zero-padding region) keeps permutation symmetry unbroken during training: identical slots → identical gradients → identical slots forever.
- **σ-residual is a gain knob, not a schedule**: `cos(residual(σ=0), residual(σ=1)) = +0.986`, SVD effective DoF = 1. Magnitude grows monotonically 19.2 → 20.85 (~8%) across σ. No rotation between noise/detail regimes.
- **Residual dominates base**: `‖sigma_residual‖ / ‖base_postfix‖ ≈ 2.76`. The caption-conditional `cond_mlp` branch is washed out by the caption-independent σ branch.
- **No textual anchor**: nearest T5 tokens to the postfix are generic (`▁your`, `▁post`, `▁D`, `à`, `▁werden`) at cosine ~0.17. Top-k is nearly invariant in σ — the ~8% magnitude sweep doesn't carve between different textual directions.

**Interpretation (user's reading is correct)**. The current trained postfix is a **domain prior**: one fixed direction in T5-compatible space, applied as a σ-modulated attention sink, irrespective of the prompt. It pulls generation toward the training distribution's style. Useful, but:

- Not prompt-conditional — same direction for every caption, so it's not "adding something after reading the prompt".
- Not a prompt enhancer in the textual-inversion sense — no "masterpiece"-like concept injected that a user could reproduce by adding a word.
- Not a σ-scheduler — one magnitude knob isn't a schedule.

## Why plain flow-matching MSE produces this failure mode

The loss on `v_θ(x_t, t, c_embed)` has no term that rewards prompt-conditional use of the postfix *specifically*, or σ-scheduled direction change. Three structural gaps:

1. **Slot permutation invariance.** All K slots sit at equivalent zero-padding positions and are interchangeable in the cross-attention softmax. The loss is invariant to permuting them. Zero-init starts them identical; the loss gradient respects the symmetry; they stay identical. Capacity collapses to K=1.
2. **No pressure against the σ-branch swamping the base.** The σ-branch can efficiently encode a prompt-agnostic "average correction" that helps every training sample — and that's the lowest-loss path, so gradient descent finds it. Nothing penalizes washing out caption conditioning.
3. **MSE prefers the smoothest σ-dependence that works.** A single direction with σ-varying magnitude is the simplest fit; rotation between directions across σ is strictly harder to find without an explicit curvature signal.

The loss rewards *any* constant-direction residual that helps the dataset on average. One-vector-one-knob is the maximum-exploit solution of that reward.

## How to force "adds something helpful after reading the prompt, not textually expressible"

The goal is (a) prompt-conditional and (b) super-textual — i.e. the postfix should encode image-relevant information that no text substitution can reproduce. "Textually interpretable" is an anti-goal; the postfix should be free to live *off* the T5 token manifold if that's where the signal is.

### A. Break slot symmetry (prerequisite to any capacity gain)

Without this, K-effective=1 no matter what loss you use. Two minimal options, either alone suffices:

- Per-slot identity embedding: `postfix[k] = mlp_out[k] + slot_embed[k]` with `slot_embed: nn.Parameter(K, D)` random-init (~0.02 std).
- Drop zero-init on `cond_mlp[-1]` / `sigma_mlp[-1]` — use small random init. Loses the "starts identical to baseline" property but gains ~K× capacity immediately.

### B. Budget the σ-branch so the base survives

Prevent the σ-residual from dominating:

- Soft L2 on the σ-residual output: `λ_σ · ‖sigma_residual‖²`.
- Multiplicative coupling: `postfix = base · (1 + tanh(σ_gain))` — the σ-branch can only rescale the base, not bypass it.
- Hard clip: `‖sigma_residual‖ ≤ α · ‖base‖`, `α < 1`.

Any of these forces the caption to matter.

### C. Directly require prompt-conditional variance

Auxiliary objectives that penalize caption-agnostic postfix outputs:

- Contrastive: for (caption_i, caption_j) pairs in a batch, minimize `cos(postfix(c_i), postfix(c_j))` between different captions. (Cheap — 1-liner.)
- InfoNCE between pooled postfix and pooled caption embedding, with a small critic head.

These are signal-only losses; they don't care whether the postfix is textual or not — only that it varies with the prompt.

### D. Distill from embedding inversion — the "super-textual target" objective ⭐

Cleanest fit for "helpful AFTER reading the prompt, not textually expressible". The plumbing already exists in this repo (`scripts/inversion/invert_embedding.py`, the `postfix_func` variant in `configs/methods/postfix.toml`).

Embedding inversion produces `e*` — a post-adapter embedding that the frozen DiT uses to reconstruct a specific image with lower flow-matching loss than any text prompt achieves. By construction, `e* - t5_adapter(caption)` is a vector that *cannot* be obtained by rewriting the caption. That delta is exactly what you want the postfix to learn.

Pipeline:

1. For a set of (image, caption) pairs, run `invert_embedding.py` to get `e*(image)`.
2. Precompute `δ = e* − t5_adapter(caption)` — the "super-textual residual" per sample.
3. Train postfix with loss `MSE(postfix_network(caption), δ_spliced)` in place of (or added to) flow-matching MSE.

Advantages:

- Target is per-image, so postfix is forced to read the prompt to approximate it (no prompt-agnostic solution can fit).
- Target is provably super-textual (inversion found it because text couldn't).
- `postfix_func` in the TOML already wires a functional-MSE-vs-inversions loss; re-reading that path and generalizing it to direct-embedding-delta supervision is the minimum implementation change.

### E. σ-scheduling (defer)

Only meaningful after A–D — once the base postfix is genuinely prompt-conditional, `cond-timestep` can become a real schedule (different super-textual correction at high vs low σ) instead of a gain knob. Encourage K>1 effective σ-DoF by a finite-difference smoothness *penalty inversion* — penalize `||residual(σ_i+Δ) − residual(σ_i)||` being too small, not too large.

## Recommendation

**Minimum viable change: A + D.** Break slot symmetry (per-slot identity embedding, ~1 line) and retarget the loss to inversion deltas (generalize the existing `postfix_func` variant). That addresses both failure modes with minimum scope — B/C/E are refinements on top.

Once that's trained, re-run `analyze_sigma_tokens.py`:

- Slot-symmetry check should report non-zero inter-slot differences.
- Residual/base ratio should drop (σ-branch weaker relative to caption).
- Top-k nearest T5 tokens should show **low** cosine (≤0.1) *and* vary per caption — that's the "off-manifold super-textual" signature you want. (Low cosine to every T5 token = the postfix is pointing outside the text subspace.)

---

## Follow-up: second checkpoint (`output_temp/anima_postfix_exp.safetensors`, cond-mode, no σ)

Before committing to A+D, we inspected the earlier cond-mode (no σ-branch) checkpoint via a sibling script `analyze_cond_postfix.py`. Result: **the cond-mode checkpoint exhibits the same collapse, plus a new one that σ-absence rules out.**

| diagnostic | result | meaning |
|---|---|---|
| slot-symmetry max inter-slot \|diff\| | **0.00e+00** | all K=64 slots identical ⇒ effective K=1 (same as σ checkpoint) |
| pairwise cos across 128 captions | min **+0.903**, median **+0.995** | every caption emits the same direction |
| SVD across captions | top σ = 751, next = 71 (10× gap), **DoF@90% = 1** | one fixed direction, caption-modulated only in gain |
| ‖P − P_mean‖ / ‖P‖ | **0.126** | ~13% of output varies with caption; 87% is a prompt-agnostic bias |
| per-caption T5 NN top-5 | near-identical across probed captions | `cond_mlp` does not actually read the prompt |

**Implication.** The σ-branch is *not* the cause of collapse — `cond_mlp` itself collapses. Even without any σ path, flow-matching MSE finds a degenerate solution: a single fixed direction, produced regardless of caption, modulated only in magnitude. The σ-branch in the sibling checkpoint is just a second, larger gain knob layered on top.

This shifts the binding constraints:
1. **A (slot symmetry)** — still necessary; without it K-eff stays 1 under any loss.
2. **A is not sufficient on its own.** Flow-MSE doesn't pressure `cond_mlp` to produce caption-varying outputs; the cheapest-gradient-descent solution is a constant. Something must explicitly reward prompt-conditional variance.

So the updated MVP is **A + C**, not A + D:

- **C (contrastive) is cheap, in-repo, no data dependency.** It directly attacks the `cos(postfix(c_i), postfix(c_j)) ≈ 0.995` finding. A loss term that penalizes inter-caption cosine gives `cond_mlp` no way to be constant without paying an explicit price.
- **D (inversion deltas) remains blocked on data.** `inversions/results/` has one file. D can't start until a `make invert` sweep has been run over a meaningful chunk of the dataset. It's still the strongest long-term target (per-image super-textual supervision), but it's not the fastest path to "stop collapsing."

## Patches applied (A + C)

Code changes landed as the first pass. Sequencing: land A+C, retrain, re-run the analyzer; if the post-retrain checkpoint still feels shallow, escalate to D once the inversion sweep is done.

### A — per-slot identity embedding
`networks/postfix_anima.py`

- Added `self.slot_embed: nn.Parameter(K, D)` in the cond / cond-timestep branch. Random-init via new kwarg `slot_embed_init_std` (default **0.02** at construction time, **0.0** when loading a legacy checkpoint so old files round-trip unchanged).
- `append_postfix` now does `postfix = cond_mlp(pooled) + slot_embed.unsqueeze(0)` — small per-slot bias that breaks the permutation symmetry so gradients differ across K from step 1.
- `slot_embed` persisted as a top-level `slot_embed` key in the saved safetensors; `load_weights` warns when the key is missing (legacy checkpoint → keeps init value).
- Added to `_cond_param_list` so the optimizer actually trains it.

At init: `max inter-slot |diff| = 0.118` on a fresh module (effective K immediately > 1).

### C — inter-caption contrastive loss
`networks/postfix_anima.py`, `library/training/losses.py`, `library/training/metrics.py`

- New kwargs `contrastive_weight` (default 0.0 = off) and `contrastive_buffer_size` (default 32).
- `PostfixNetwork.get_contrastive_loss()` — mean off-diagonal cosine between the current postfix and a **detached MoCo-style memory queue** of recent postfixes. The queue is necessary because `base.toml` has `batch_size=1`; within-batch contrastive would always be zero. The current postfix stays in the compute graph; buffered entries are detached.
- Registered as `postfix_contrastive` in `LOSS_REGISTRY` / `_STAGE_SCALAR_BROADCAST`. Activation rule in `build_loss_composer`: active iff `network.contrastive_weight > 0`. No `train.py` edits — the composer wires it automatically.
- Scalar cached on `network._last_contrastive_value` so `reg/postfix_contrastive` and `reg/postfix_contrastive_weighted` surface in W&B / TensorBoard without recomputing (which would also double-update the queue).

### Config
`configs/methods/postfix.toml` — active postfix_sigma block now carries:

```toml
network_args = [
    "mode=cond-timestep",
    "splice_position=end_of_sequence",
    "cond_hidden_dim=256",
    "sigma_feature_dim=128",
    "sigma_hidden_dim=256",
    "slot_embed_init_std=0.02",
    "contrastive_weight=0.1",
    "contrastive_buffer_size=32",
]
```

### Validation criteria for the retrained checkpoint

After `make postfix` with these patches, re-run **both** analyzers. Pass criteria:

- `analyze_sigma_tokens.py`
  - `slot-symmetry max |diff|` **> 0** (A worked)
  - `SVD effective DoF @ 90% energy` **≥ 2** (slots carrying distinct directions)
  - `residual/base ratio` **< 1.5** (σ-branch no longer dominates — partial, since we didn't add §B)
- `analyze_cond_postfix.py` on the same checkpoint
  - `pairwise cos across captions` — median **< ~0.7** (was 0.995)
  - `SVD DoF @ 90% energy` across captions **> 1** (was 1)
  - `deviation / total` **> 0.3** (was 0.126)
  - Per-caption T5 NN top-5 visibly varies across probed captions (was near-identical)
- Training log: `reg/postfix_contrastive` should start near 1.0 and drop monotonically toward ~0.2–0.4 within an epoch. If it stays at 1.0, the caption signal isn't reaching `cond_mlp` — check the pooling path.

If all pass, the postfix has stopped being a pure domain prior and is genuinely reading the prompt. Only then is D (inversion-delta distillation) worth the data cost to push from "reads the prompt" to "encodes super-textual information the prompt cannot express."
