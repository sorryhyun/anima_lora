# Soft Tokens — per-layer × per-t learnable text tokens (SoftREPA)

Per-layer, time-indexed soft tokens in T5-compatible space. DiT is frozen. ~1M trainable params at default config (n_layers=10, K=4, D=1024, n_t_buckets=100). Each of the first `n_layers` DiT blocks gets its own learned (K, D) token bank plus a per-(t-bucket, layer) D-vector offset, spliced into `crossattn_emb` for that block alone. Trained with plain FM.

Reference: Lee et al., Aligning Text to Image in Diffusion Models is Easier Than You Think (arXiv:2503.08250, NeurIPS 2025) — "SoftREPA". The base recipe adopts only the parameterization (per-layer × per-t soft tokens), trained under plain FM; the paper's InfoNCE contrastive objective was originally skipped because at Anima's training batch size (B=1) there are no in-batch negatives, and the paper itself reported SD3 FID regression at paper-strength contrastive. An optional, B=1-adapted contrastive objective is now available (off by default) — it builds negatives by swapping a cached text embedding off disk instead of using batch peers. See §"Contrastive objective" below and `_archive/proposals/soft_tokens_contrastive.md`.

## Quick start

```bash
make exp-soft-tokens                    # default preset
python tasks.py exp-soft-tokens         # cross-platform
```

Inference is supported. `create_network_from_weights` loads the checkpoint and the denoising loop fires the per-step splice for you: `library/inference/generation.py` and `networks/spectrum.py` call `soft_tokens_net.append_postfix(embed, seqlens, timesteps=t)` once per CFG branch before each forward (cond + uncond, including the tiled path), mirroring the training-side trainer hook. On Spectrum cached steps the blocks don't fire, so soft tokens silently no-op for those steps — it composes freely with `--spectrum`.

## What it is

For each block `k ∈ [0, n_layers)`, the cross-attention input is replaced by a layer-specific variant:

```
s^(k, t)         = tokens[k] + t_offsets[bucket(t), k]      # shape (K, D)
crossattn_emb_k  = splice(crossattn_emb, s^(k, t))
block_k(x, ..., crossattn_emb_k)                            # original block, modified text input
```

`tokens ∈ ℝ^(n_layers × K × D)` is the base bank; `t_offsets ∈ ℝ^(n_t_buckets × n_layers × D)` is a per-(bucket, layer) D-vector broadcast across the K-token axis. Zero-init on `t_offsets` means at step 0 the layer banks reduce to their base values — no time conditioning until gradients learn it.

```
                       Soft Tokens
              ┌─────────────────────────────┐
              │  DiT Block 0                │
              │  ┌─────────────┐            │
crossattn ────┼─►│ +s^(0,t)    │  cross    │
   (B,S,D)    │  │   spliced   │  attn ──► │ ──► x'
              │  └─────────────┘            │
              ├─────────────────────────────┤
              │  DiT Block 1                │
              │  ┌─────────────┐            │
crossattn ────┼─►│ +s^(1,t)    │  cross    │
   (B,S,D)    │  │   spliced   │  attn ──► │ ──► x''
              │  └─────────────┘            │
              ├─────────────────────────────┤
              │  ...                        │
              ├─────────────────────────────┤
              │  DiT Block (n_layers..N-1)  │  no splice — block sees
              │  cross-attn (unmodified)    │  the original crossattn
              └─────────────────────────────┘
```

The crossattn passed in is unchanged across blocks (Anima is not joint-stream MM-DiT — text features don't evolve through blocks). Each block independently sees a different splice; no strip/re-prepend dance.

## Parameter count

```
n_layers · K · D       (base tokens)
+ n_t_buckets · n_layers · D   (t-offsets, broadcast across K)
```

Defaults: 10 · 4 · 1024 + 100 · 10 · 1024 ≈ 41k + 1.05M ≈ 1.05M params. 30–60× lighter than a typical LoRA.

## Implementation map

| File | Role |
|------|------|
| `networks/methods/soft_tokens.py` | `SoftTokensNetwork` — per-(layer, t) token bank, splice hook, save/load. |
| `apply_to(text_encoders, unet)` | Walks `unet.blocks[:n_layers]`, replaces each `block.forward` with a wrapper that splices `s^(k, t)` into `crossattn_emb` before calling the original (monkey-patch). |
| `append_postfix(crossattn_emb, seqlens, timesteps)` | Receives `timesteps` from `train.py`'s existing per-step hook; computes `(n_layers, B, K, D)` step-scoped tokens and caches them on the network. Returns `crossattn_emb` unchanged — splicing happens inside the block hooks. |
| `_make_block_hook(layer_idx, org_forward)` | Closure that reads the cached step tokens at `layer_idx`, splices into `crossattn_emb`, calls the original block forward. |
| `SoftTokensMethodAdapter` (same file) | Contrastive extra-forward driver: stashes `neg_crossattn_emb` in `prime_for_forward`, runs the negative forwards + the active objective in `extra_forwards`, replays the deferred ∂L/∂v_neg in `after_backward`, surfaces metrics. Auto-resolved by `resolve_adapters` when `_contrastive_target_weight > 0`. |
| `contrastive_loss(...)` / `step_contrastive_warmup(...)` | InfoNCE over the negatives (with optional jaccard penalty) + the warmup gate. |
| `softrank_loss(...)` / `softrank_diagnostics(...)` / `_soft_rank(...)` | Soft-rank objective: differentiable listwise rank of the matched caption among candidates via `softtorch` (`contrastive_objective=softrank`). Dual ψ⁺/ψ⁻ banks ride a `branch` selector through `_set_step_tokens`/`_bank_forward`. See `_archive/proposals/soft_tokens_softrank.md`. |
| `library/datasets/base.py` | `setup_contrastive_negatives` / `_load_te_for_stem` — negative TE sourcing + `neg_crossattn_emb` / `neg_jaccard` on the example. |
| `library/datasets/identity_pairs.py` | `IdentityPairSampler.hard_negative` / `shuffled` / `tag_jaccard` — negative policy. |
| `library/training/losses.py::_soft_tokens_contrastive_loss` | Applies the warmup-gated `λ_con` to the adapter's InfoNCE scalar. |
| `configs/methods/soft_tokens.toml` | Default config (splice_position=front_of_padding, lr=1e-3, 4 epochs; plain FM, contrastive off). |
| `configs/gui-methods/soft_tokens.toml` | Sibling for `make lora-gui GUI_PRESETS=soft_tokens`. |
| `scripts/experimental_tasks/training.py::cmd_soft_tokens` | Task entry-point. |
| `tasks.py` `exp-soft-tokens` | Make/CLI registration. |

## Splice position

Two options, mirroring postfix:

| Mode | Where | Trade-off |
|---|---|---|
| `front_of_padding` (shipped config default) | place K tokens at `[seqlens[i], seqlens[i]+K)` per sample (`scatter`) | Algorithm-faithful — SoftREPA concatenates the soft tokens onto the text sequence, so the position immediately after the real caption is the direct analogue. Caption-position-aware. Displaces the strongest sinks. Per-sample variable indices via the cached `crossattn_seqlens`. |
| `end_of_sequence` (code-level fallback default) | overwrite the K tail slots `[S-K, S)` of the zero-padding region | Anima-specific shortcut: static splice index → maximally compile-friendly, and gets attention mass only because Anima's deep-padding slots are attention sinks. Caption-position-agnostic. Preserves the strongest front-of-padding sinks intact. Not what the paper does. |

Both `configs/methods/soft_tokens.toml` and `configs/gui-methods/soft_tokens.toml` ship `splice_position=front_of_padding`. The `SoftTokensNetwork` constructor default is `end_of_sequence` (the compile-friendly fallback used when no config overrides it). The choice is metadata-tagged (`ss_splice_position`) so checkpoints round-trip with the right splice mode; toggle via `network_args = ["splice_position=end_of_sequence"]`.

Not benched. The fop-vs-eos choice is principled (paper-faithfulness) but never measured on Anima — see the open A/B item in §"What this does not do / open knobs" below. `end_of_sequence` could plausibly compete: fop displaces the strongest sinks (perturbing what the frozen model relies on), eos rides the tail sinks while leaving them intact.

Anima's text-encoder padding invariant (zero-padded positions act as cross-attention sinks) means writing into the padded tail is *not* a no-op — those slots receive attention mass and the soft tokens get exposure to every spatial query. See the "Text encoder padding" note in the root CLAUDE.md.

## Why a separate module from `postfix.py`

Postfix splices once at the cached adapter output (training-time and inference-time, in `train.py:762` and `library/inference/generation.py`). Soft tokens splice per-block via a monkey-patched `Block.forward`. Different surface entirely — keeping them separate avoids muddying the postfix abstraction. Both modules expose `append_postfix(...)` so `train.py`'s existing per-step trainer hook routes timesteps to either family without code changes.

## Why no slot-collapse

The existing postfix module logs an aggressive guard against K-slot permutation symmetry collapse (`anima_postfix.safetensors` was effectively K=1 due to zero-init + symmetric splice — see the postfix module docstring and the `slot_embed_init_std` knob). Soft tokens structurally avoid this: tokens at different `(k, t)` pairs are consumed at different positions in the network and gradients differ from step 1, so no symmetry to break.

> Removed: bank-axis dispersive regularizer (2026-05-22). Earlier versions
> shipped an optional parameter-space dispersive regularizer (Wang & He,
> Diffuse and Disperse, arXiv:2506.09027) over the bank's `K` and
> `n_t_buckets` axes, meant to guard against slot collapse and under-sampled
> bucket degeneracy. It was removed after it showed no effect worth keeping —
> soft tokens already structurally avoid slot collapse (see "Why no
> slot-collapse" above: different `(k, t)` pairs are consumed at different
> positions, so gradients differ from step 1 and there's no symmetry to break).
> The repr-space variant was separately probed and found redundant
> ([[project_soft_tokens_contrastive_phase0]]). Plain FM is now the baseline;
> the only optional add-on is the contrastive objective below.

## Contrastive objective (optional, B=1-adapted SoftREPA InfoNCE)

A revival of SoftREPA's contrastive objective, off by default. It is data-conditioned and needs negatives — it sharpens prompt-following by making the matched text explain the anchor's latent better than mismatched text does. Full design + phasing: `_archive/proposals/soft_tokens_contrastive.md`.

The B=1 trick (no batch peers): a negative is a different stem's cached text embedding (`{stem}_anima_te.safetensors`, the post-LLM-adapter `crossattn_emb`) swapped off disk — the same cached-feature-swap precedent the IP-Adapter identity pairs use, but swapping the TE feature instead of the PE feature. Each step runs the primary forward (matched text = the positive) plus `k` extra DiT forwards with the negative text spliced through the same soft tokens; the logit of a forward is its negative flow-matching error against the shared velocity target:

```
ℓ_*           = -‖v_* − v_target‖² / τ      (mean over C·H·W; logit = neg FM error)
L_contrastive = -log( exp(ℓ_pos) / Σ_{pos, neg_1..k} exp(ℓ_*) )
L_total       = L_FM + λ_con · L_contrastive                         (post-warmup)
```

Only `crossattn_emb` differs across the forwards, so the gradient isolates text-conditioning. Each negative is one extra full DiT forward — `k=4` ≈ 5× step time — so keep `k ∈ {1, 2}`. To further amortize the cost, `contrastive_every_n` runs the negatives only every Nth optimizer step (the term is a small-weight auxiliary regularizer; the warmup window already proves the bank trains fine with it fully off for a stretch). It's a manual frequency knob, not auto-scaled: effective strength ≈ `weight × 1/N`, so bump `contrastive_weight` if you want to hold the average pull constant. Firing-step peak memory is unchanged and off-steps are cheaper, so it's a free throughput lever with no OOM risk.

> Why not a fused single-backward instead? A tempting alternative is to run the `k` negatives with grad and do one combined backward (cutting `2k`→`k` DiT forwards). At the shipped default preset (no gradient checkpointing, `blocks_to_swap=0`) that holds a full second forward's activation graph co-resident — ~+6 GB on a ~13 GB run, OOM-risking a 16 GB card — which is exactly why `extra_forwards` keeps the `no_grad` value pass + `after_backward` grad-cache replay split ([[project_blockswap_extra_forwards_gradcache]]). `contrastive_every_n` gets the throughput win without the memory hit.

Negative modes (`contrastive_negative_mode`):

| Mode | Sourcing | Notes |
|---|---|---|
| `shuffled` (default) | an unrelated image (no character/copyright overlap) | The Phase-1 go/no-go negative. |
| `jaccard` | shuffled sourcing + per-negative logit down-weight `ℓ_neg −= α·s` | `s` = caption tag-overlap (character ∪ copyright ∪ artist) Jaccard; a near-miss negative pulls less gradient. Cheap middle path — no new sampler. `α = contrastive_jaccard_alpha`. |
| `hard` | a same-artist / different-character sibling (style-matched, content-different) | Cancels style-induced velocity similarity so the only axis left to win on is content. Falls back to `shuffled` for orphan/untagged artists — on the current dataset Phase 0 measured the strict pool at ~29% coverage, so ~71% of steps degrade to shuffled. |

Negative grouping comes from the shared caption index (`make caption-index` → `post_image_dataset/captions/caption_index.json`), reusing `IdentityPairSampler` (`hard_negative` / `shuffled` / `tag_jaccard`). The index path is not a user knob.

Wiring. Negatives are sourced in `library/datasets/base.py::setup_contrastive_negatives` / `_load_te_for_stem` (surfaced as `neg_crossattn_emb` `(B, k, S, D)` on train steps only — validation FM-MSE stays a clean baseline). The `k` extra forwards + InfoNCE live in `SoftTokensMethodAdapter.extra_forwards` + `SoftTokensNetwork.contrastive_loss`; the warmup-gated weight is composed in `library/training/losses.py::_soft_tokens_contrastive_loss` (active iff `_contrastive_target_weight > 0`). `step_contrastive_warmup` holds `λ_con` at 0 for the first `warmup_ratio` of steps. The objective leaves no learned parameters — a trained checkpoint is bit-identical whether or not contrastive was on, and inference ignores it entirely.

Config knobs (`network_args`, all off-by-default-safe):

| Knob | Default | Meaning |
|---|---|---|
| `contrastive_weight` | `0.0` | λ_con; `0` = bit-identical to plain FM (dataset stops producing negatives → no extra forwards). |
| `contrastive_k` | `1` | negatives per step → `(k+1)×` forward cost. |
| `contrastive_every_n` | `1` | run the negatives only every Nth optimizer step (gated on `global_step // accum` so an accumulation window fires uniformly). Manual knob, not auto-scaled → effective strength ≈ `weight × 1/N`. No extra memory. |
| `contrastive_negative_mode` | `shuffled` | `shuffled` \| `jaccard` \| `hard`. |
| `contrastive_jaccard_alpha` | `1.0` | logit penalty for `jaccard` (sweep 0.5–2.0). |
| `contrastive_tau` | `0.5` | InfoNCE temperature. |
| `contrastive_warmup_ratio` | `0.1` | hold λ_con at 0 for the first 10% of steps. |

TensorBoard signals: `reg/soft_tokens_contrastive` (raw InfoNCE), `_weighted`, `_lambda_live` (warmup gate), `soft_tokens/contrastive_acc` (positive beats every negative) and `soft_tokens/contrastive_logit_gap`.

### Soft-rank objective (differentiable listwise rank, default)

A second objective on the same extra-forward plumbing (negatives, warmup,
`contrastive_every_n`, compose seam, `after_backward` grad-cache), selected with
`contrastive_objective=softrank` (the shipped config default). Full design:
`_archive/proposals/soft_tokens_softrank.md`. Where InfoNCE's negative branch is
*unbounded* (maximizing negative error has no fixed point — the SoftREPA
degrade-while-loss-drops failure), soft-rank replaces the softmax with a
differentiable listwise rank of the matched caption among the candidates. With
per-candidate reward `r_j = −‖v_j − v_target‖²` over `j ∈ {matched, neg₁…neg_k}`:

```
L_softrank = softtorch.rank(r, method)[matched] − 1      (rank ∈ [1, m]; 1 = matched wins)
```

`softtorch.rank` (SoftSort / NeuralSort relaxation, Prillo & Eisenschlos 2020)
standardizes the candidate axis and returns a smooth, 1-indexed rank. Gradient
flows through the ordering (no detach on the ranking — `v_target` is the only
constant), so it pushes the bank to make the matched caption win outright, and it
stays bounded by the SoftSort relaxation (`L → 0` as `rank → 1`, self-annealing
at the win). No EMA shadow, no external scorer — it costs `k` extra forwards per
firing step, the same as InfoNCE.

This is the chosen objective. AGSM (the paper-faithful bounded target-shift with a
PL-weighted EMA bank) was the third option and was removed 2026-05-30: in the
live A/B its `w_matched` stayed pinned at chance (`1/(k+1)` for every epoch — "Δ
rotates but w stays at chance"), while soft-rank's `matched_rank` descended
monotonically toward 1 and won on eyeballed image quality. This matches the
offline Tier-A gradient probe ([[project_softrank_agsm_gradient_probe]]): soft-rank's
`cos(∂L/∂V, ∂margin/∂V)` ≈ 0.86 vs AGSM's ≈ 0.25, at the same boundedness. The same
chance-pin sank the mod-guidance AGSM probe, which was removed with it.

Dual banks ψ⁺/ψ⁻ (opt-in `dual_bank=true`, default in the shipped config). A
branch axis on the bank: `tokens (2,n_layers,K,D)` + bank-major `t_offsets`. ψ⁺ is
spliced on the anchor + positive passes, ψ⁻ on the negative value/replay passes —
the negative push refines ψ⁻ without spending generative fidelity on the bank kept
at inference. Inference uses ψ⁺ only (injecting ψ⁻ in the uncond branch
over-suppresses detail). The load side keys off the on-disk tensor rank (4D ⇒
dual), so `load_weights` slices the ψ⁺ branch when an inference net reads a dual
file; single-bank checkpoints stay loadable and `dual_bank=false` is bit-identical
to the single-bank path. The flag accepts the old name `agsm_dual_bank` as a
deprecated alias. (Dual-bank is orthogonal to the objective — it works under
InfoNCE too; the live softrank run shows ψ⁻ training via `tokens_neg_mean_norm`.)

Gradient flow mirrors InfoNCE exactly — `L` rides the anchor's FM backward (grad
via the live `v_pos`, negatives detached), and `∂L/∂v_neg` is deferred to
`after_backward` (the block-swap-safe grad-cache split,
[[project_blockswap_extra_forwards_gradcache]]). The objective leaves no learned
parameters beyond the bank(s) — inference ignores it entirely, same as InfoNCE.

| Knob | Default | Meaning |
|---|---|---|
| `contrastive_objective` | `infonce` | `infonce` \| `softrank`. |
| `softrank_method` | `neuralsort` | SoftSort relaxation: `neuralsort` (smooth gradient through ties — the near-miss regime) or `softsort` (faster, flat at exact ties). |
| `softrank_softness` | `0.1` | SoftSort softness (its temperature). `standardize=True` puts the candidate axis at unit scale, so ~0.1 is right; lower = sharper rank / stronger pull (sweep 0.05–0.3). Ignored by InfoNCE. |
| `dual_bank` | `false` | dual ψ⁺/ψ⁻ banks; inference keeps ψ⁺ only. Accepts `agsm_dual_bank` as an alias. |

Requires `contrastive_k ≥ 2` — `softtorch` standardizes the candidate axis,
which is degenerate (gap-independent) with a single negative. Needs the `softtorch`
dependency (pot/torchopt/numba). Does not use `contrastive_tau`.

TensorBoard signals (soft-rank): `reg/soft_tokens_contrastive` (= the rank loss),
`_weighted`, `_lambda_live`, `soft_tokens/softrank_matched_rank` (the headline:
→ 1 as the matched caption wins outright, → m when it loses; it descended
monotonically in the live run), plus `soft_tokens/contrastive_acc` /
`contrastive_logit_gap` (reward-gap mirrors of the InfoNCE diagnostics for
cross-objective comparability) and, under dual bank,
`soft_tokens/tokens_neg_mean_norm` (ψ⁻ magnitude).

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `train.py` already passes `timesteps=...` into `append_postfix` (legacy `cond-timestep` postfix mode); soft tokens piggyback on the same hook. |
| Standard inference | ✅ | `create_network_from_weights` loads the bank (contrastive forced off — it leaves no params); `library/inference/generation.py` fires `append_postfix(..., timesteps=t)` per CFG branch each step, including the tiled path. |
| Spectrum inference | ✅ | `networks/spectrum.py` fires the same per-step splice on *actual* steps; cached steps skip all blocks so soft tokens no-op there (composes with `--spectrum`). |
| `torch.compile` (`_run_blocks`) | ✅ | `end_of_sequence` keeps `crossattn_emb` shape static; the cached `_step_layer_tokens` is read as a runtime tensor with static shape. `front_of_padding` uses `scatter` with dynamic per-sample indices but static buffer shape — also compile-clean. |
| `blocks_to_swap` | ❌ method-forced 0 | The hook captures each `Block` by reference at `apply_to()` time; a swapped block is a different object instance, so the hook would fire on the wrong tensor. |
| `gradient_checkpointing` | ✅ | The hook is the outermost wrapper; the original `forward` (which itself runs `checkpoint(_forward, ...)`) is called underneath, and the spliced `crossattn_emb` is part of the saved input graph. |
| Modulation guidance | ✅ orthogonal | Modulation = per-block AdaLN path; soft tokens = K/V input path per block. |
| T-LoRA / OrthoLoRA | n/a | Soft tokens freeze the DiT; LoRA-family methods are not stacked in this config. |

## Evaluation

What to measure to know if this is doing anything:

1. `|t_offsets|` at convergence as a function of bucket: flat/near-zero → time conditioning collapsed (the per-layer base tokens absorbed everything; SoftREPA's `use_dc_t=True` won't be load-bearing). Curve should grow away from zero, ideally with structure across t.
2. Per-layer token norm: `‖tokens[k]‖` should differ across `k`. If they converge to a single shared bank, we're effectively running a single-layer postfix and the per-layer parameterization is dead weight.
3. **Held-out prompt-following**: this is the load-bearing question — soft tokens target text-image alignment / prompt-following. If they move those metrics, they're earning their parameters; if not, they're overhead.
4. Anatomy / style breakdown: REPA helped anatomy on Anima but broke anime style (vision-encoder photo-prior leak). Soft tokens have no external visual prior, so the failure mode shouldn't recur — but they also can't reproduce the anatomy gain. The plausible win is text alignment, not structural quality. If anatomy also improves, that's a surprise worth tracking.
5. Splice position A/B: `front_of_padding` (shipped default, algorithm-faithful) vs `end_of_sequence` (compile-friendly Anima shortcut). Front-of-padding displaces the strongest sinks and might give the tokens more attention mass at the cost of disturbing what the pretrained model relies on; end-of-sequence leaves the front sinks intact and rides the tail sinks. Faithfulness picked the default — still worth a short bench to confirm it on Anima samples.

## Hyperparameters worth sweeping

| Knob | Default | Range to try | Why |
|---|---|---|---|
| `n_layers` | 10 | 5, 10, 14, 28 | SoftREPA used 5/24 layers on SD3. 10/28 is proportional. Going to 28 (all blocks) doubles params and tests whether deep-block tokens do anything. |
| `network_dim` (K) | 4 | 1, 4, 8, 16 | SoftREPA used m=4 on SD3. K=1 collapses to "per-layer prefix vector" — clean ablation. |
| `n_t_buckets` | 100 | 0 (disable t-cond), 20, 100 | Setting `t_offsets.weight.requires_grad_(False)` is a clean ablation for whether time conditioning is load-bearing. |
| `init_std` | 0.02 | 0.0, 0.02, 0.1 | Zero-init = strict identity at step 0 (block sees zeroed padding tail). 0.02 = small perturbation. 0.1 = aggressive. |
| `splice_position` | `front_of_padding` (shipped) | both | Config ships fop (algorithm-faithful); constructor fallback is eos. See §"Splice position" above. |
| `learning_rate` | 1e-3 | 1e-4 to 5e-3 | Soft tokens are tiny + zero-inited offsets; high LR is fine. |
