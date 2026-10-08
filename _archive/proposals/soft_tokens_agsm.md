# Soft Tokens — AGSM: bounded dual-token alignment guidance

Status: **ARCHIVED 2026-05-29 — LANDED.** The bounded AGSM target shipped on the
soft-tokens path (`contrastive_objective=agsm`, paper-faithful Plackett–Luce Δ +
γ⁺/γ⁻ self-anneal, dual-bank ψ⁺/ψ⁻ behind `agsm_dual_bank`) and an A/B showed it
**helps prompt-following and quality**. Phase 0 reward premise held; the implementation and PL correction are
recorded in [[project_soft_tokens_agsm_pl_correction]]. Open items below are
faithfulness refinements, not blockers: **3b renoise = DEFER** (probe MATTERS by
the literal gate but is weak in practice), **3c time-shaping = PARKED**, and the
**dual-bank ship-vs-revert A/B** (3a) remains the one outstanding decision. The
method now lives in the code + method doc (`docs/experimental/soft_tokens.md`); the
original proposal text is preserved below for the phasing/gating rationale.

# Original proposal

Status: **proposal** (2026-05-22). Builds on `docs/experimental/soft_tokens.md`
and supersedes the InfoNCE direction in `docs/proposal/soft_tokens_contrastive.md`
*conditionally* — see "Relationship to the contrastive proposal". Reuses the
cached-TE negative plumbing already shipped for that proposal
(`library/datasets/base.py::setup_contrastive_negatives`, `IdentityPairSampler`).

Reference: Lee, Hong, Kwon, Ye, *Alignment-Guided Score Matching for Text-to-Image
Alignment in Diffusion Models* (ICML 2026 Spotlight; code "coming soon"). This is
the **direct sequel to SoftREPA** (Lee et al., arXiv:2503.08250) — the same paper
our soft-tokens parameterization is adapted from, same first author.

## TL;DR

SoftREPA's contrastive objective is unstable: its validation ImageReward degrades
even as training loss drops (the paper's own stability plot; mirrored by the SD3
FID regression we flag in `soft_tokens.md`). AGSM diagnoses the cause as
**unbounded contrastive divergence** — pushing negative pairs to *maximize* their
denoising error has no fixed point — and fixes it three ways:

1. **Shift the score-matching target instead of maximizing negative error.**
   Positives regress toward `v_t + γ⁺·Ã(t)·Δ`, negatives toward `v_t − γ⁻·Ã(t)·Δ`.
   Bounded by construction.
2. **Dual token banks ψ⁺ / ψ⁻** so positive and negative guidance don't fight
   over one shared token space.
3. **Reward-free Plackett–Luke normalization** of an *intrinsic* alignment reward
   read off the model's own denoising likelihood — **no external reward model**.

The headline for us: this is **implementable entirely from existing pieces**. The
extra-forward loop, cached-TE negatives, splice, and loss-compose site are already
in the tree from the contrastive proposal. The only genuinely new internal
machinery is a second token bank and an EMA shadow of predictions. Nothing
external — no ImageReward, no CLIP scorer, no teacher checkpoint.

## Why this is in-house (the "no external model" answer)

AGSM is *reward-free* by design. The alignment reward is

```
r(x_t, c) = −‖v_θ^ψ(x_t, c) − v_target‖²            # denoising likelihood proxy
p(z | x_t, c) = softmax_i r(x_t, c_i)                # Plackett–Luce over candidates
```

i.e. it asks "which candidate caption explains this noised latent best", scored by
*our own* flow-matching error. The guidance direction Δ is built from an **EMA of
our own soft-token-conditioned predictions** — a self-distillation target, not an
external model. So every term is computed from the frozen Anima DiT + the trainable
banks we already have. Concretely:

| AGSM ingredient | Where it comes from in our tree |
|---|---|
| ε_θ⁺ / ε_θ⁻ (pos/neg conditioned preds) | `SoftTokensMethodAdapter.extra_forwards` — already runs pos + k neg forwards |
| negative captions (D⁻) | `setup_contrastive_negatives` / `_load_te_for_stem` (cached-TE swap) |
| v_target | `v_target = noise − latents` (`soft_tokens.py:670`) — already computed |
| reward r(x_t,c) | the per-forward `−‖v−v_target‖²` already computed for the InfoNCE logit |
| ψ⁺ / ψ⁻ banks | **new**: second `SoftTokensNetwork`-style bank (or doubled `num_tokens` split in half) |
| Δ from EMA preds (ε̂⁺, ε̂⁻) | **new**: EMA shadow of the pos/neg velocity predictions |

## The flow-matching mapping (why ε→v is free here)

AGSM is written in ε-prediction. Anima is velocity flow-matching with
`v = ε − x₀` and a fixed data `x₀`, so a shift of the ε-target by `δ` is *exactly*
a shift of the v-target by `δ` (the `x₀` term is constant). The shipped FM target
is already `v_target = primary.noise − primary.latents`. Therefore AGSM's target
becomes, with no reparameterization:

```
positives (D⁺):   L⁺ = ‖ v_θ^{ψ⁺} − ( v_target + γ⁺·Ã(t)·Δ ) ‖²
negatives (D⁻):   L⁻ = ‖ v_θ^{ψ⁻} − ( v_target − γ⁻·Ã(t)·Δ ) ‖²
L_AGSM = E_{D⁺}[L⁺] + E_{D⁻}[L⁻]
```

Open derivation items (flagged, not free):

- **Ã(t)** — AGSM's bounded time-weight. SoftREPA/AGSM use an ε-noise schedule;
  ours is the FM σ∈[0,1] schedule already bucketed by `n_t_buckets`. Start with
  `Ã(t)=1` (constant, bounded) and only add t-shaping if a t-bucket sweep shows
  it matters. Do **not** import the ε-schedule weighting blindly.
- **Δ** — the guidance direction. AGSM: implicit-reward-weighted EMA of `(ε̂⁺,ε̂⁻)`.
  Velocity form (paper Algorithm 1 lines 10–11 / Eq. 17, **now implemented**):
  the Plackett–Luce weighting is *not* optional — it is the load-bearing
  self-annealing that bounds the correction (§3.3). Per candidate `j ∈ {matched,
  neg₁…neg_k}`: `w_j = softmax_j(−‖v̂_ema_j − v_target‖²/τ)`, baseline
  `Σ_k w_k v̂_ema_k`, `Δ_j = v̂_ema_j − baseline`. Needs an EMA of the predictions
  (see below). Reuses `contrastive_tau` as the PL temperature.

## New machinery (the only two non-existing pieces)

### 1. Second token bank ψ⁻

Today `SoftTokensNetwork` holds one bank (`tokens` + `t_offsets`). AGSM needs two
guidance regions. Cheapest: keep one `nn.Parameter` of shape
`(2, n_layers, K, D)` (or two banks) and splice ψ⁺ on positive forwards, ψ⁻ on
negative forwards, via the existing `apply_to` / splice path. Param count doubles
the bank term only (still ~1–2M total at our scale). The ablation in the paper
(dual > positive-only > shared at equal budget) is the thing to reproduce in
Phase 2.

### 2. EMA shadow of predictions

A bounded target needs a slow reference. Maintain an EMA of the pos/neg velocity
predictions (or of the bank weights, then forward through it — cheaper to EMA the
*outputs* per step). This is a standard `decay·ema + (1−decay)·current` update;
no external model. Lives next to the adapter, updated in the trainer step after
the optimizer step.

Everything else — extra forwards, negative sourcing, warmup gate, the
`aux["soft_tokens_contrastive"]` → `_soft_tokens_contrastive_loss` compose path —
is reused. The handler in `losses.py:318` already applies a warmup-gated weight to
an aux scalar; AGSM just puts a different scalar (`L_AGSM`) in that slot.

## Relationship to the contrastive proposal

`soft_tokens_contrastive.md` Phase 1 (plain InfoNCE, `shuffled` negatives) is
**already implemented but not yet benched**. That is the cheap probe that tells us
whether we even *have* the SoftREPA instability on Anima:

- **If Phase 1 A/B is stable and helps** → InfoNCE is fine here; AGSM's bounded
  reformulation buys little; shelve this proposal as "fallback if instability
  appears later."
- **If Phase 1 shows val-reward-degrades-while-loss-drops** (the SoftREPA pattern)
  → that is the empirical trigger to adopt AGSM. Don't tune τ/k; switch the
  objective in the *same* `extra_forwards`/compose seam.

So this proposal is **gated behind the Phase 1 result of the contrastive proposal,
not run in parallel.** They share all infrastructure; only the loss math differs.

## The premise risk to falsify first (cheap, no training)

AGSM's reward is the **denoising likelihood**, and we have a hard, repeatedly
confirmed finding that **FM-MSE does not track quality on Anima**
([[project_fm_val_loss_uninformative]]; why we moved to CMMD,
[[project_cmmd_val_signal]]). AGSM's entire alignment signal is built from exactly
that quantity.

The saving grace is that Plackett–Luce uses it as a **relative ranking across
candidate captions for the same latent**, not as an absolute quality score —
relative ordering can survive when absolute MSE is uninformative. But this must be
checked before any training:

- **Probe (Phase 0). ✅ BUILT + RAN 2026-05-29 — PASS.**
  `bench/soft_tokens_contrastive/reward_premise_probe.py` is the no-training ranking
  test: for n anchors it scores the matched caption + k cached-TE negatives by
  `−‖v−v_target‖²` at the anchor's own `(x_t, ε, t)` (only `crossattn_emb` differs —
  the `extra_forwards` contract), LoRA-off and with a trained bank, across a σ grid,
  seed-averaged. Reports rank@1 (matched beats all k) + margin vs `shuffled` (the
  kill-gate) and `hard` (same-artist/diff-character) negatives. **Result** (24
  anchors, k=2, run `results/20260529-1157-phase0-agsm/`): LoRA-off **shuffled
  rank@1 = 0.993**, **hard rank@1 = 0.958**, both with positive margin, vs chance
  0.333 → **PASS**. The reward premise holds: matched text explains the anchor's
  latent better than mismatched, *even though* absolute FM-MSE is uninformative — relative ranking survives. Two notes:
  (1) the margin **grows monotonically with σ** (perfect rank@1 by σ≥0.45, near-chance
  for hard at σ=0.15) — caption-conditioning is most discriminable when `x_t` is
  mostly noise and the model must guess `x0` from text, weakest near the clean latent;
  this argues any `Ã(t)` (Phase 3c) should *up*-weight high σ, not import the
  ε-schedule blindly. (2) The trained `tenth` bank (objective=agsm, 1/10 data) does
  **not** beat the frozen base on ranking (shuffled identical; hard slightly worse,
  0.927 vs 0.958) — the frozen cross-attention already carries the discriminative
  signal; the bank isn't (yet) sharpening the reward axis.

## Phasing — gates, cheapest-first

- **Phase 0 — reward-premise probe (no training). ✅ DONE 2026-05-29 — PASS.**
  The ranking test above (`reward_premise_probe.py`). Gate cleared: LoRA-off
  shuffled rank@1 0.993 / hard 0.958, positive margin, vs chance 0.333. The reward
  premise is not dead on arrival → Phase 1/2 are unblocked. (PASS ≠ AGSM beats
  plain-FM — that is Phase 2's CMMD A/B.)
- **Phase 1 — (deferred to the contrastive proposal).** Its plain-InfoNCE A/B is
  the instability detector. AGSM is only justified if that A/B exhibits the
  SoftREPA degrade-while-loss-drops pattern.
- **Phase 2 — AGSM target-shift, single bank first. ✅ IMPLEMENTED (2026-05-22;
  corrected to the paper 2026-05-29, not yet benched).** The bounded target
  (`v_target + γ⁺·Ã(t)·Δ⁺` / `v_target − γ⁻·Ã(t)·Δ⁻_j` with `Ã=1`) lives in
  `SoftTokensMethodAdapter.extra_forwards`, selected by the network arg
  `contrastive_objective=agsm` and reusing the InfoNCE compose seam +
  `after_backward` grad-cache verbatim (only the loss math differs). Single bank
  (ψ⁺ = ψ⁻), constant `Ã(t)=1`. **Correction (2026-05-29, paper now available):**
  the initial reconstruction used a uniform-mean, matched-excluded, single shared
  `Δ = v̂⁺ − mean_j(v̂⁻_j)`; the paper's Algorithm 1 uses a **Plackett–Luce-weighted,
  per-candidate** `Δ_j = v̂_ema_j − Σ_k w_k v̂_ema_k` (`w` = softmax over **all**
  candidates incl. matched). That PL weighting is the self-annealing that bounds
  the negative branch (§3.3) — `agsm_targets`/`agsm_losses` now compute it, with
  separate `γ⁺/γ⁻` (paper SD3/flow used `(1, 0.1)`). Knobs: `agsm_gamma` (γ⁺,
  default 1.0), `agsm_gamma_neg` (γ⁻, default = γ⁺; toml sets 0.1),
  `agsm_ema_decay` (default 0.99), PL temperature = `contrastive_tau`. The EMA
  shadow is a plain tensor attribute
  (never saved; refreshed once per optimizer step in `after_backward` on
  `sync_gradients`). **Cost note:** the EMA value passes make this ~`(2k+1)` extra
  forwards/firing-step, above the proposal's headline `(k+1)×` — Δ is read off the
  shadow bank's *own* predictions (the load-bearing self-distillation decoupling),
  which costs the extra matched + mismatched EMA forwards; a cheaper live-Δ
  approximation was rejected because it reintroduces the moving-target dynamic AGSM
  exists to remove. Still to do: A/B vs (a) plain FM and (b) Phase-1 InfoNCE on the
  prompt-following / CMMD axis (FM-MSE val deltas are uninformative —
  `project_fm_val_loss_uninformative`). Keep `k ∈ {1,2}`.
- **Phase 3 — paper-faithful upgrades, gated on a positive Phase-2 bench.** Three
  independent sub-steps, each its own A/B against the shipped Phase-2 single-bank /
  `x_t` / `Ã=1` baseline. They are faithfulness refinements, not the thing that
  makes or breaks the method. As of 2026-05-29 **3a is built** (behind a flag) so
  its open work is a *falsification* A/B; **3b** is reduced to a single
  integrate-or-skip decision driven by an offline probe; **3c** stays parked.

  - **3a — dual bank ψ⁺/ψ⁻ (§3.3, the paper's headline structural contribution).
    ✅ IMPLEMENTED 2026-05-29 — behind `agsm_dual_bank` (default off). The open
    question is the A/B that could *falsify* it.**

    What got built: a branch axis on the bank — `tokens (2,n_layers,K,D)`, a
    bank-major `t_offsets (n_t_buckets, 2·n_layers·D)`, and a doubled EMA shadow.
    The adapter routes every negative value/EMA/replay forward through ψ⁻
    (`neg_branch = 1 if n_banks>1 else 0`); the anchor + matched-EMA stay on ψ⁺
    (branch 0). PL weights / `agsm_targets` / grad-cache are untouched — the
    per-candidate EMA preds come off ψ̂⁺ for the matched and ψ̂⁻ for the mismatched
    (Algorithm 1 line 8). Single bank (`n_banks=1`) is branch-0 throughout and
    **bit-identical to Phase 2** (24+7 tests in
    `tests/test_soft_tokens_contrastive.py`). Inference loads **only ψ⁺** for both
    CFG branches (Appendix H: injecting ψ⁻ into the uncond branch over-suppresses
    unmentioned detail → lower human-preference, so ψ⁻ is training-only); the
    checkpoint stamps `ss_n_banks`, single-bank checkpoints stay loadable, and
    `load_weights` slices the ψ⁺ branch when an inference net reads a dual file.

    **How to falsify it (the A/B that decides ship-vs-revert).** Run dual vs the
    shipped single-bank Phase-2 baseline at **equal ψ⁺ budget** (K=4 either way;
    dual spends 4⁺+4⁻, the extra ψ⁻ is free at inference) — flip
    `agsm_dual_bank=true` in `soft_tokens.toml`, same everything else, CMMD +
    prompt-following A/B (FM-MSE val is uninformative, `project_fm_val_loss_uninformative`).
    - **KILL → revert to single bank:** dual ≤ single on CMMD / prompt-following.
      The decoupling bought nothing — the negative push didn't need its own token
      space — and the doubled training cost + on-disk branch dim aren't worth it.
      Keep the flag for the record but default off.
    - **KEEP → make dual the default:** dual > single, reproducing the paper's
      Table 12 ordering (dual > positive-only > shared). Then the ψ⁻ decoupling is
      load-bearing and the inference cost is still zero.
    - **Diagnostic to watch:** `soft_tokens/tokens_neg_mean_norm` (ψ⁻ magnitude).
      If it stays ≈ ψ⁺'s norm and tracks it, ψ⁻ never differentiated → expect KILL.

  - **3b — Eq. 51 renoise: integrate or skip, decided by an offline probe (no
    training).** Algorithm 1 (and our Phase 2) evaluate the EMA guidance preds at
    the anchor's own `(x_t, t)`. The flow derivation (Appendix D, Eq. 47/51)
    evaluates them at a **renoised** `(x_{t+Δ}, t+Δ)` — one step toward noise —
    because the local Gaussian transition `p_θ(x_t | x_{t+Δ})` is what defines the
    reward. For Anima the renoise is closed-form and cheap (no extra noise draw):

    ```
    x_{t+Δ} = (1−t−Δ)x₀ + (t+Δ)ε = x_t + Δ·v_target          (clamp t+Δ ≤ 1)
    ```

    **The decision is whether this eval-point move changes anything the loss sees.**
    `bench/soft_tokens_contrastive/renoise_probe.py` answers it with no training:
    it recomputes the per-candidate guidance `Δ_j` direction and the matched PL
    weight `w_matched` at `x_t` vs `x_{t+Δ}` on cached anchors, across a σ grid and
    a Δ sweep, and reports `cos(Δ⁺)` + `|Δw_matched|`.
    - **NO-OP → skip 3b:** `Δ⁺` cosine ≈ 1 and `|Δw_matched|` ≈ 0 (in the
      informative mid/high-σ band). The Algorithm-1 `x_t` collapse loses nothing,
      so the wiring (clamp + bucketize `t+Δ` + an extra EMA eval-point arg) buys
      nothing — don't integrate.
    - **MATTERS → integrate + A/B:** the renoise moves the guidance direction or
      re-ranks the matched weight. Then wire it (only the EMA value passes move to
      `(x_{t+Δ}, t+Δ)`; live `v_pos` + the regression target stay at `x_t`;
      `_bank_forward(use_ema=True)` already takes the noised tensor + timesteps, so
      it's: pass the renoised tensor + `t+Δ`), test renoise **alone with the scale
      pinned at 1** to isolate the eval-point effect, A/B vs the `x_t` baseline.
      Δ ∈ {one t-bucket width `1/n_t_buckets`, fixed `0.02–0.05`}; sweep.

    > **Probe verdict → DEFER 3b (MATTERS by the literal gate, weak in practice).**
    > Base DiT (`results/20260529-1241-phase3b-gate/`, k=2, 24 anchors, 3 seeds,
    > shuffled): Δ⁺ cos (σ×Δ mean) = **0.718** (≪ 0.98), `|Δw_matched|` =
    > **0.0016** (≈ 0). Bank-on, trained `_tenth` ψ⁺ spliced
    > (`results/20260529-1246-phase3b-bank-light/`, n=4 light): Δ⁺ cos **0.710**,
    > `|Δw|` **0.0013** — **statistically identical to base, within noise.** So the
    > renoise barely re-ranks captions but rotates the guidance direction; that
    > rotation is a property of the **frozen DiT velocity field**, not the bank.
    > Three reasons it is a weak lever, not a clear win:
    > 1. **Δ-controlled, not intrinsic.** At small `Δ=0.02` in the informative
    >    σ≥0.45 band the renoise is nearly inert (cos 0.86–0.93, `|Δw|`≈0); the low
    >    aggregate is driven by `Δ=0.0714` (a full t-bucket) and the σ-extreme rows.
    >    The paper folds `λ_t` to keep the effective step ≈1 (*small*), i.e. the
    >    high-cos near-no-op corner is the regime 3b would actually run in.
    > 2. **Small-vector confound.** `w_matched` ≈ 0.333 (chance for k=2) in **both**
    >    arms → the trained bank never sharpens the PL weight off chance (matches
    >    Phase 0, [[project_agsm_reward_premise_holds]]), so `Δ_j` is only the
    >    *small* caption-difference vector, whose direction is perturbation-
    >    sensitive by construction. Low cos ⇏ "Eq. 51 carries better guidance."
    > 3. **No PL-weight movement** (`|Δw|` ≈ 0 everywhere): the reward ranking — the
    >    thing AGSM is built on — is eval-point-invariant; only the small Δ vector
    >    moves.
    >
    > **Decision:** do **not** wire 3b now. It sits behind the two things that
    > actually gate the method — the (still-informal) Phase-2 CMMD A/B and the
    > built-but-unbenched **Phase-3a** dual-bank A/B. 3b becomes worth revisiting
    > only if a better-trained bank first pushes `w_matched` **above chance** (the
    > current 1/10 bank does not); at that point re-run this probe and, if cos is
    > still low at small Δ, fold the renoise into a 3a/3c run rather than as its
    > own A/B. Probe: `bench/soft_tokens_contrastive/renoise_probe.py`.

  - **3c — Ã(t)/B(t) time-shaping (parked).** Only if a t-bucket sweep on the
    shipped `Ã=1` (resp. pinned `B=1`) shows signal left on the table — and the
    Phase-0 probe's finding that the caption-margin **grows with σ** is the hint
    that any `Ã(t)` should up-weight high σ, not import the ε-schedule blindly.

## Costs to keep honest

- **Extra forwards.** Same `(k+1)×` step cost as the contrastive path; AGSM does
  not reduce it. `k ∈ {1,2}` only.
- **EMA memory.** One shadow of the bank (small) or of per-step predictions
  (transient). Negligible vs the frozen DiT.
- **Reconstruction risk.** The full paper is now in the tree (arXiv:2605.30038,
  Algorithm 1 + Appendix D flow derivation + Table 11 recipe), so the Δ
  reward-weighting is no longer guessed — it was corrected to the paper's PL form
  on 2026-05-29 (see Phase 2). What remains a reconstruction is the **B=1
  adaptation** (paper trains batch-16, 1:3 in-batch negatives; we use cached-TE
  negatives at effective batch 1) and the **`x_t` collapse** of the Eq. 51 renoise
  (Phase 3b). The flow-matching `v = ε − x₀` mapping (and its closed-form renoise
  `x_{t+Δ} = x_t + Δ·v_target`) is the load-bearing assumption.
- **Goal-mismatch.** AGSM optimizes COCO-style prompt *alignment* (counting, no
  repeated objects) on a general backbone. Our soft tokens train on a
  character/style dataset where the objective is closer to identity/style fidelity.
  Decide whether "alignment reward" points where we want before Phase 2 — the
  Phase 0 ranking probe partly answers this (does matched-caption preference even
  exist on our data).

## What this does NOT do

- Does not add any external reward model, scorer, or teacher checkpoint — every
  term is computed from the frozen Anima DiT + trainable banks (reward-free).
- Does not change `batch_size`, the dataloader batching, or the splice/block hook —
  negatives are cached-TE swaps (B=1-safe), same as the contrastive proposal.
- Does not run in parallel with the contrastive Phase 1 — it is gated on that
  result and reuses its plumbing.
- Does not claim parity with the paper's SD1.5/SDXL/SD3 numbers — it is a
  flow-matching, B=1-adapted reconstruction, gated on the Phase 0 reward probe.

## Reference points

- Module + extra-forward loop: `networks/methods/soft_tokens.py`
  (`SoftTokensMethodAdapter.extra_forwards` `:653`, `contrastive_loss` `:545`,
  `step_contrastive_warmup` `:524`, `v_target` `:670`)
- Loss compose site: `library/training/losses.py`
  (`_soft_tokens_contrastive_loss` `:318`, registered key `soft_tokens_contrastive`)
- Negative sourcing (reuse): `library/datasets/base.py::setup_contrastive_negatives`
  / `_load_te_for_stem`, `library/datasets/identity_pairs.py::IdentityPairSampler`
- Reward-premise probe (Phase 0, DONE): `bench/soft_tokens_contrastive/reward_premise_probe.py`
  (structural sibling: `negative_audit.py` — proves a hard negative *exists*; this one
  proves the FM reward *ranks* it)
- Renoise probe (Phase 3b gate): `bench/soft_tokens_contrastive/renoise_probe.py`
  (the no-training `x_t` vs `x_{t+Δ}` Δ-direction / `w_matched` comparison that
  decides whether to integrate 3b)
- Sibling proposal (gates this one): `docs/proposal/soft_tokens_contrastive.md`
- Method doc: `docs/experimental/soft_tokens.md`
- Quality-signal context: [[project_fm_val_loss_uninformative]],
- Papers: AGSM (ICML 2026, https://jaayeon.github.io/AGSM/); SoftREPA
  (arXiv:2503.08250, NeurIPS 2025)
