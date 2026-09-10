# Memorization report v2 — caption-independent uncond-Δ statistic

Status: Phase 0 RUN 2026-07-08 — G1/G2 PASS, G3 split (see results
section). The mechanism claim is verified: the member signal survives
caption removal (uncond-Δ AUC within 0.008/0.034 of caption-Δ on the two
overfit checkpoints, clean control at chance). But the single
eyeball-confirmed xerox row is NOT top-ranked by uncond-Δ (72/167) —
weight-side and generation-side rankings disagree at the item level, so
Phase 1 should ship the `both` decomposition, not replace the ranking
with uncond-z alone. Successor instrument to the closed low-σ reweight
line (`_archive/proposals/memorization_lowsigma_reweight.md`). Pure metric upgrade for the
measure-only xerox report — no training intervention anywhere in this
line. Phase 0 needed no retraining (scored existing checkpoints).
Phase 2 (proposed) promotes uncond-Δ into a per-epoch memorization eval
running at the CMMD cadence — CMMD tracks quality on the val split and is
structurally blind to member memorization, so this is a complementary gauge,
not a replacement (see Phase 2 section).

## Where this comes from — the reweight line in three sentences

The reweight line (Phase 0–0.6, all 2026-07-07) validated that memorization
is per-step-gradient-local at low σ, built and validated an online per-item
Δ-gap estimator (multi-draw grid measurement, Spearman ρ 0.47 vs the offline
MIA probe), and then closed its intervention arm: per-item low-σ loss
downweighting cannot un-memorize in-band even when correctly targeted (three
runs triangulated a −0.05…−0.09 AUC ceiling vs a −0.19 gate; in-band AUC
unmoved with reliable targeting). The eyeball postscript found the pivotal
fact: the memorized residual is caption-independent — the xeroxed
attributes are untagged, so they live in the style/unconditional pathway
(`project_lora_crossattn_learns_labeled_only`), which is why per-item loss
pressure failed (shared-pathway residual, not item-gradient-exclusive).
What survives and ships: the measure-only report
(`--mem_reweight_mode measure --mem_extra_sigmas 0.5 0.7` →
`<ckpt>_memgap.json`).

| closed-line run | AUC (gate was ≤ 0.65) | sanity ρ |
|---|---|---|
| measure control | 0.838 | 0.281 |
| reweight z=1.5, noisy estimator | 0.749 | 0.247 |
| reweight z=0.75, noisy estimator | 0.852 (null) | 0.196 |
| reweight z=0.75, multi-draw estimator | 0.789 | 0.471 |

## The problem with the shipping statistic

Both the offline probe (`bench/memorization/loss_gap.py`) and the online
tracker (`library/training/mem_reweight.py`) measure the Δ-gap with the
item's real caption. That conflates two things the report exists to
separate:

- benign tag/style fit — the adapter got better at rendering this
  artist's tagged content. This is what the LoRA is for; holdouts show it
  too (healthy holdout Δ > 0 is documented probe behavior).
- memorization — the adapter reproduces this frame's untagged
  attribute bundle regardless of caption (the eyeball finding).

A well-captioned, on-style member scores a high caption-Δ without being
xeroxed — a false positive for the "which images is this LoRA xeroxing"
report.

## The idea

Measure the Δ-gap with the caption replaced by the null embedding —
the T5("") sidecar that caption dropout already stages
(`library/preprocess/uncond.py::ensure_uncond_crossattn`, the same tensor
the CFG-uncond branch sees at inference). Everything else identical: same
paired base-vs-adapted forwards, same σ grid, same antithetic noise.

- `uncond-Δ` = caption-independent confidence gain on this item ≈ the
  memorized residual. This is the ranking statistic the report should ship.
- `caption-Δ − uncond-Δ` = the caption-coupled remainder ≈ legitimate
  tag/style fit. Free per-item decomposition, useful report column on its
  own ("is this item memorized, or just well-fit?").

This doubles as the verification of the eyeball mechanism: the visual
finding shows the reproduced attributes aren't in the caption; only a
measurement can show the residual is reachable without one. If uncond-Δ
separates members from holdouts as well as caption-Δ does, the residual is
genuinely unconditional — which is also the premise the recaption-and-retrain
successor (route untagged attributes to cross-attn by labeling them) stands
on. Run this BEFORE building that.

## Phase 0 — offline, existing checkpoints, no training

Extend `loss_gap.py` with `--null_captions` (member/holdout scoring uses the
uncond embedding instead of each item's TE cache; the flag lands in the
result envelope). Score three existing checkpoints, both modes each:

- `bench_sincos_half_rw_measure` — the known overfit (caption-Δ AUC 0.838).
- `bench_sincos_half_rw_reweight_z075_grid` — second overfit sample
  (0.789), checks the result isn't checkpoint-specific.
- the banked clean reference config (`plain_tenth`, caption-Δ AUC 0.54) —
  false-positive control.

Gates:

- G1 (mechanism): on the overfit checkpoints, uncond-Δ member-vs-holdout
  AUC ≥ caption-Δ AUC − 0.05. The member signal must survive caption
  removal; that is the claim.
- G2 (clean control): on the clean adapter, uncond-Δ AUC ≤ 0.60 — the
  new statistic must not invent signal where the caption one finds none.
- G3 (specificity, the point of the exercise): holdout-mean uncond-Δ <
  holdout-mean caption-Δ on the overfit checkpoints — the style-fit
  component (which holdouts share) must drop out of the new statistic. The
  per-item eyeball xerox row (`sincos/10562024`) must rank in the uncond-Δ
  member top-8.

Cost: four probe passes (~144 items × 2 models × σ grid each), no training.

## Phase 0 results (2026-07-08)

`--null_captions` landed in `loss_gap.py` (swaps every item's caption for
the T5("") sidecar; same paired forwards, same per-item probe noise).
Caption-Δ baselines reused from the banked seed-0 runs (identical 144-item
selection, verified stem-for-stem). Runs:
`results/20260708-0012-uncond_rw_measure`, `-0034-uncond_rw_z075_grid`,
`-0055-uncond_plain_tenth`, `-0124-uncond_rw_measure_allmem`.

| checkpoint | caption-Δ AUC | uncond-Δ AUC | holdout mean capΔ → uncΔ |
|---|---|---|---|
| rw_measure (overfit) | 0.838 | 0.830 (p 1e-4) | −0.0113 → −0.0056 |
| z075_grid (overfit) | 0.789 | 0.755 (p 1e-4) | −0.0112 → −0.0050 |
| plain_tenth (clean) | 0.540 | 0.510 (p 0.32) | −0.0018 → −0.0018 |

- G1 PASS (both ≥ caption − 0.05): the member signal survives caption
  removal essentially intact; per-σ shape matches (peak σ 0.5–0.7). The
  memorized residual is genuinely reachable without a caption — the
  recaption-and-retrain successor's premise holds.
- G2 PASS: clean control at chance (0.510 ≤ 0.60).
- G3 SPLIT. First half passes in the intended (magnitude) reading:
  holdout-mean |uncond-Δ| is half of |caption-Δ| on both overfit
  checkpoints — the caption-coupled style-fit component does drop out.
  (The literal `<` inequality assumed positive holdout style-fit; on
  these checkpoints it is negative, so the sign convention inverts.)
  Second half FAILS: `sincos/10562024` ranks 72/167 among members by
  uncond-Δ (mid-pack; also absent from the caption-Δ 48-member draw, so
  no evidence caption-Δ ranks it higher either). Weight-side confidence
  gain and generation-side visible xeroxing order items differently —
  consistent with the probe's own caveat ("a loss gap can exist without
  extractable copying", and vice versa).
- Not redundant (kill-criterion ρ > 0.95 not hit): member-level
  Spearman(caption-Δ, uncond-Δ) = 0.61 / 0.66 — the decomposition
  carries real per-item information.

**Verdict: instrument validated, ranking caution.** uncond-Δ is the
better statistic (caption-independent, style-fit-free, no invented
signal on clean adapters) and Phase 1 should proceed — but ship
`--mem_measure_captions both` as the default report surface (uncond-z
ranking + caption-Δ column) rather than uncond-only, and keep
generation-side eyeball/probe.py as the per-item conviction step; the
top-8 gate shows weight-side rank alone does not nominate the
eyeball-confirmed xerox.

## Phase 0.5 — detector verification: FP-breadth zoo + merge interference (2026-07-12)

Step 2 of the "is this measure an effective overfitting detector" ladder
(endpoint discrimination and dose ordering were already banked; the missing
claims were false-positive breadth on presumed-healthy checkpoints, and
whether the measure works on merged artifacts — the `ckpt/soup` /
`scripts/toolkits/merge_loras.py` use case). One daemon command job, 6 probe runs,
`results/20260712-18*/-19*`. All runs used `--sigmas 0.5 0.7`: the probe
draws the same per-item seeded noise for every σ, so per-σ AUCs are exactly
comparable to the banked 4-σ runs' `per_sigma` entries (K=4 and 48/48 item
counts kept for the same reason). Compare per-σ, not overall.

Zoo (uncond-Δ, instance gate) — healthy-population calibration:

| checkpoint (recipe) | AUC (σ0.5 / σ0.7) | perm p | verdict |
|---|---|---|---|
| plain_quarter (25%, 2e-5, 4ep, r32) | 0.665 (0.676 / 0.622) | 0.005 | flagged |
| autorandom_matched_tenth (10%, 5e-5, 4ep, r16) | 0.504 (0.540 / 0.513) | 0.20 | clean |
| plain_tenth (banked anchor) | 0.510 | 0.32 | clean |

The healthy population is NOT uniformly at chance: `plain_quarter` sits at
the 0.65 flag line with a real p. Either 25%-sample × 4ep genuinely carries
mild member signal (more repeats/image than the tenth runs — the dose story
is consistent) or the 0.65 threshold is tight for production recipes.
Unresolved pending a render/probe.py check of its top-flagged items —
the scalar-vs-nomination caveat (G3b) applies. Caveats: its cross-artist
holdout is only n=17 (a 25% draw of the whole pool leaves few artist dirs
untouched; the 48/48 member-vs-same gate is what fired), and the intended
high-epoch cell (`anima_tenth16`) has no surviving checkpoint — the negative
class is still thin (n=3).

Merge B — memorization survives merging (uncond-Δ, sincos perspective;
`bench_sincos_half_rw_measure ⊕ anima_ama2`, raw exact sum, rw snapshot
copied beside the merged file so membership replays identically):

| | σ0.5 AUC | σ0.7 AUC |
|---|---|---|
| solo rw_measure (banked 20260708-0012) | 0.777 | 0.735 |
| merged with ama2 | 0.733 | 0.714 |

Member signal essentially intact (−0.02…−0.04, p=0.0001) after merging with
an unrelated artist LoRA. Merging does not launder memorization — a
merged/souped artifact inherits its worst ingredient (consistent with the
banked soup AUC ≈ max finding and ΔW near-orthogonality,
`project_artist_lora_merge_interference_phase0`) — and the probe audits
merged checkpoints as-is (copy any ingredient's snapshot beside the file to
probe from that ingredient's perspective).

Merge A — the measure detects merge interference (caption-Δ, artist1
perspective, style-fit mode; June visual ground truth: raw 3-way sum =
grain overdrive, norm-controlled = healthy):

| | member Δ | cross-artist Δ | member−cross |
|---|---|---|---|
| solo artist1 | +0.011 | −0.022 | +0.033 |
| artist123 raw sum | −0.157 | −0.205 | +0.048 |
| artist123 norm | −0.008 | −0.027 | +0.018 |

The raw merge flips member Δ hard negative — the merged model fits artist1's
own training distribution *worse than base* (15× the solo spread; the
loss-side signature of magnitude overdrive, global not member-specific: the
member−cross spread survives). The norm merge restores member Δ to ≈0 with a
mildly compressed spread — exactly the known strength↔noise tradeoff of
routing-free merges. Verdict: caption-Δ member-level is a working
loss-side merge-interference gauge, validated against visual ground truth;
`Δ_member(merged) − Δ_member(solo)` is the statistic (strongly negative =
overdriven/bad merge; ≈0 = healthy; spread compression = dilution).
Shipping proposal: `merge_interference_probe.md` (calibration Phase 0 →
`scripts/check_merge.py`).

Ladder position after this: endpoint ✓, dose ✓, FP breadth ~ (3 negatives,
one borderline pending render adjudication), merge auditing ✓. Remaining
before shipping the Phase-2 per-epoch gauge: the trajectory/early-warning
test (per-epoch checkpoints of the overfit + clean recipes, offline scoring;
the curve must rise before the post-hoc verdict, and stay flat on clean).

## Phase 1 — online tracker + report surface (only if Phase 0 passes)

- `--mem_measure_captions caption|uncond|both` on the tracker
  (`measure_grid_delta` takes the embedding as an argument already — the
  knob only selects what the producer passes). `both` doubles measurement
  forwards; default `uncond` once validated.
- `_memgap.json` items grow `ema_uncond` / `z_uncond` (and the decomposition
  when `both`); `flagged_stems` ranks on uncond-z.
- Docs page for the measure-only mode; per-subset opt-in knob. This is the
  Phase-1 ship the reweight line deferred, landing with the better
  statistic.

## Phase 2 (proposed) — epoch-level memorization eval, CMMD's blind sibling

Motivation — CMMD is blind to memorization. CMMD (the live per-epoch
quality signal, `project_cmmd_val_signal`) is a paired PE-Core MMD² between
generated and reference images on the held-out val split. It measures
*distributional* match, so it is structurally blind to member memorization:
a run can xerox its training frames and still post a clean CMMD — the val
split is not the memorized set, and a memorizer still matches (often flatters)
the val distribution. Quality and memorization are different failure modes
and need different gauges. Nothing currently surfaces memorization during
a run: `loss_gap.py` is post-hoc, and the Phase-1 online tracker is
piggyback-on-train-step and emits a per-item report, not a per-epoch scalar.

Proposal — promote uncond-Δ into a per-epoch measurement pass, run at the
CMMD cadence and reusing its split. One dedicated pass (decoupled from the
train batch — repeat/batch-stable, unlike the piggyback tracker), scheduled
next to CMMD:

- members = trained items (`path_pattern`); holdouts = the CMMD val
  split (already unseen). Compute member-vs-holdout uncond-Δ AUC per
  epoch — the validated statistic (Phase 0 G1: uncond-Δ AUC 0.830 ≈ caption-Δ
  0.838 on the overfit ckpt; G2: 0.510 on the clean control). A rising AUC
  across epochs = memorization onset — the early-warning curve CMMD cannot
  give. The holdout set already exists (no new data), so this is free relative
  to the CMMD wiring.
- Cost sits below CMMD per item: `N × 8` grid forwards
  (2σ × 2 antithetic noise × {base, adapted}) vs CMMD's `N × 20` sequential
  denoise forwards — 8 < 20/item, and run at CMMD's cadence it is one more
  eval pass, not a per-step tax (contrast the closed reweight line's K=1
  every-step config). See the closed line's Phase 0.6 for the ρ 0.47
  validation of the grid estimator.
- The per-item report rides along for free from the same pass
  (`_memgap.json`, "which members is this LoRA xeroxing") — but carry the
  Phase 0 G3 ranking caveat: weight-side uncond-Δ ranked the one
  eyeball-confirmed xerox 72/167, so the trustworthy output is the scalar
  member-vs-holdout AUC (memorization onset/severity); per-item nomination
  still wants the generation-side render / `probe.py` check as conviction.

This is eval, not intervention — the pass touches no loss and no gradient
(the "No training intervention" non-goal below still holds); it only reads the
model. Guards inherit the closed line: plain-LoRA stacks only (grid forwards
reuse the step's adapter conditioning), and hard-require `blocks_to_swap=0`
(the second-forward-under-block-swap hazard,
`project_blockswap_extra_forwards_gradcache` — raise, not warn).

Gate for shipping it: the per-epoch uncond-Δ AUC curve must (a) stay ≈ 0.5 on
a known-clean run (`plain_tenth`) for its whole trajectory and (b) rise
monotonically on the known-overfit config (sincos-half → 0.82) — i.e. the
curve must *track* the post-hoc verdict it is meant to pre-empt, not just hit
the right endpoint.

## Non-goals / guards

- No training intervention. Loss-side suppression is closed with the
  reweight line — this proposal must not reopen it (see that doc's kill
  note; Arms A/B/C all closed).
- Not the recaption-and-retrain proposal — that is a separate successor
  whose Phase 0 consumes this instrument; don't build it first.
- Frozen everywhere else: same recipe, same probe machinery, same gates
  style. The only new thing anywhere is which embedding the measurement
  forwards see.

## Kill criteria

- G1 fails (uncond-Δ AUC collapses toward 0.5 while caption-Δ stays
  high) → the residual is caption-coupled after all; the eyeball read was
  wrong or attribute-level only. Bank it, keep the caption-Δ report,
  and the recaption successor loses its premise before costing a retrain.
- G1 and G3 pass but uncond-Δ ≈ caption-Δ per-item (ρ > ~0.95) → the
  decomposition adds nothing over the simpler statistic; keep caption-Δ,
  note the equivalence, close.
