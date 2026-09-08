# AGSM reward premise holds on Anima — relative FM-ranking survives where absolute FM-MSE doesn't

> **STATUS (AGSM landed 2026-05-29).** This Phase-0 gate passed and the method
> shipped: bounded AGSM trained on the soft-tokens path
> (`contrastive_objective=agsm`, paper-faithful Plackett–Luce Δ + dual-bank
> ψ⁺/ψ⁻), and an A/B showed it helps prompt-following and quality. So the
> "Phase 2 make-or-break" framing below is settled (positive), not open work. The
> proposal is archived as LANDED at `_archive/proposals/soft_tokens_agsm.md`. This
> note remains the standing Phase-0 premise record.
> See [[project_soft_tokens_agsm_pl_correction]].

AGSM (`_archive/proposals/soft_tokens_agsm.md`) builds its entire alignment signal from
the model's own denoising likelihood — the per-candidate FM error
`r(x_t, c) = −‖v_θ(x_t, c) − v_target‖²` that the Plackett–Luce weighting ranks
across candidate captions. That collides head-on with a hard, repeatedly confirmed
Anima finding: FM-MSE does not track quality ([[project_fm_val_loss_uninformative]],
why we moved to CMMD). If the reward is noise, AGSM is dead on arrival — the
proposal flagged this as "the single most important early kill-check."

Phase 0 ran it (`bench/soft_tokens_contrastive/reward_premise_probe.py`, no
training). The premise holds. This note records why that is not a
contradiction, plus two non-obvious results that fall out of the same run.

## The probe

For `n` cached anchors, at each σ on a grid (seed-averaged), build the anchor's
own `x_t = (1−σ)x0 + σε` and FM target `v_target = ε − x0`, then run a bare-DiT
forward per candidate caption sharing that exact `(x_t, ε, t)` — only
`crossattn_emb` differs (the InfoNCE/AGSM `extra_forwards` contract). Reward is the
InfoNCE logit's own quantity, `−‖v − v_target‖²` (τ dropped — irrelevant to
ranking). Candidates = matched caption + `k` negatives, two pools:

- shuffled — random other stems (the proposal's literal kill-gate).
- hard — same-artist / different-character siblings (`negative_audit.py`'s
  68.7%-strict pool; style fixed, content differs).

Run `results/20260529-1157-phase0-agsm/` — 24 anchors, k=2, 2 seeds, bank =
`anima_soft_tokens_tenth` (objective=agsm, ⅒ data, K=4, n_layers=6,
front_of_padding). Chance rank@1 = 1/(k+1) = 0.333.

## Headline: PASS

| arm | pool | rank@1 (σ-mean) | margin vs mean neg |
|---|---|---|---|
| base (LoRA-off) | shuffled | 0.993 | +0.0072 |
| base (LoRA-off) | hard | 0.958 | +0.0060 |
| bank (agsm ⅒) | shuffled | 0.993 | +0.0072 |
| bank (agsm ⅒) | hard | 0.927 | +0.0060 |

Matched text explains the anchor's latent better than mismatched, near-always for
random negatives and robustly even for same-artist/different-character siblings.

### Why this does NOT contradict "FM-MSE is uninformative"

The two statements are about different quantities, and the distinction is the whole
reason AGSM can work here:

- [[project_fm_val_loss_uninformative]] is about the absolute FM-MSE of one
  model across training — lower val FM-MSE has not tracked better samples.
- AGSM (and this probe) uses FM error as a relative ranking across candidate
  captions for the same fixed `(x_t, ε, t)`. Everything that makes absolute MSE
  a bad quality proxy (seed variance, σ-dependent scale, the metric's blindness to
  perceptual quality) is held constant across the candidates and cancels in the
  ranking. What's left is purely "which caption steers the velocity toward this
  latent's content" — and that signal is intact.

So the right mental model: absolute FM-MSE is a bad *altimeter*, but the FM error is
still a usable *compass* when you only compare directions at one point.

## Non-obvious result 1: discriminability GROWS with σ

The margin is not flat across noise level — it climbs monotonically, and rank@1 is
*weakest at low σ*:

| σ | base hard rank@1 | base shuffled margin (vs mean) |
|---|---|---|
| 0.15 | 0.812 | +0.0003 |
| 0.30 | 0.938 | +0.0008 |
| 0.45 | 1.000 | +0.0017 |
| 0.60 | 1.000 | +0.0034 |
| 0.75 | 1.000 | +0.0083 |
| 0.90 | 1.000 | +0.0287 |

This inverts the naive "σ→1 is pure noise so no caption can explain it" intuition
(which the probe's own first-draft docstring assumed — now corrected). The
mechanism:

- Low σ: `x_t` is almost the clean latent. The velocity is dominated by the
  (small) residual toward `x0`, which the near-complete latent already determines —
  so every caption produces nearly the same low error. Captions are barely
  distinguishable; ranking is noisy.
- High σ: `x_t` is mostly noise, so the model must reconstruct `x0` largely
  from the text. The matched caption points at the right content; a mismatched one
  points at the wrong content → large, reliable error gap.

This dovetails with [[project_sigma_signal_resolves_by_045]] (x0 resolves by
σ≈0.45) from the other side: σ≥0.45 is exactly where caption-ranking becomes
perfect. Implication for AGSM: the PL reward is most trustworthy at mid/high σ.
If a `Ã(t)` time-shaping is ever added (proposal Phase 3c), it should up-weight
high σ — do not import the ε-noise-schedule weighting blindly (it generally
does the opposite).

## Non-obvious result 2: the trained bank adds ~nothing to ranking

The `tenth` bank was trained with the AGSM objective (weight 0.15), yet spliced in
it does not beat the frozen base on this ranking test — shuffled is identical to 4
decimals, and hard is slightly worse (0.927 vs 0.958). Reading:

- The frozen cross-attention already carries the discriminative signal the
  reward reads off. The premise's PASS is a property of the base model, not of the
  bank.
- The bank only nudges K=4 spliced tokens; at ⅒ data it hasn't (yet) sharpened the
  caption-vs-latent matching, and on the hard axis it's a hair noisier. This is a
  Phase-0 observation, not a verdict on AGSM — the bank's job is to improve
  generation (Phase 2 CMMD), not to win this ranking probe.

## What this does and doesn't license

- Does: clear the Phase 0 gate. The reward premise is not dead on arrival;
  Phase 1 (InfoNCE instability detector) / Phase 2 (AGSM target-shift) are unblocked.
- Doesn't: promise AGSM beats plain-FM or InfoNCE on quality. This is the
  ranking property AGSM needs, measured by the same FM error that is uninformative
  in the absolute — a strong necessary condition, not a sufficient one. The
  make-or-break is Phase 2's CMMD / prompt-following A/B.

## Reproduce

```
uv run python -m bench.soft_tokens_contrastive.reward_premise_probe \
    --num_samples 24 --num_seeds 2 --contrastive_k 2 \
    --adapter output/ckpt/anima_soft_tokens_tenth.safetensors --label phase0-agsm
```

Probe: `bench/soft_tokens_contrastive/reward_premise_probe.py`. Structural sibling
(does a hard negative exist): `negative_audit.py`. Proposal (archived as LANDED):
`_archive/proposals/soft_tokens_agsm.md` §"premise risk to falsify first" + Phase 0.
