# Tag read-back reward — one verified judge for selection, and the RWR artist LoRA rebuilt on it

Status: **Phase 0a PASS (tagger judge, content-tag axis).** The scoring primitive
(`anime_tools.tagger.readback::TagReadback`), the validity harness
(`bench/readback/run_bench.py`), and the turbo render driver
(`bench/readback/render_turbo.py`) are built and run. Verdict: the tagger read-back
is a valid per-image caption-*adherence* instrument, on both real and generated
images; it is (by design) orthogonal to the turbo text/pose teacher-gap, which is a
scope boundary, not a failure. Results:

- Real-data controls (val N=756, cached PE, zero renders): shuffled-caption drop
  0.991 (gate ii ≥ 0.90); true-vs-random-caption AUROC 0.98 (≥ 0.80).
  logsigmoid ≈ calibrated-recall; general-only tags (identity stripped) still 0.97+,
  refuting the PE-within-family-blindness risk on the adherence axis.
- Generated-image cross-prompt discrimination (the in-axis render test, 5 arms ×
  16 prompts × 2 seeds): true-prompt vs other-prompt AUROC 0.98–1.00,
  win(true>random) 1.000 on teacher AND every 4-step turbo student, recall@1 over
  16 prompts 0.75–0.97. The instrument transfers cleanly from training images to
  generated images — the deployment surface the real-data control alone can't cover.
- Turbo teacher>student pair set (gate i as literally specified): chance-level
  AUROC 0.52–0.56 — EXPECTED. Teacher carries a small mean adherence edge (−1.086
  vs student −1.18…−1.29 logsigmoid) but it is swamped by same-prompt seed jitter.
  The turbo teacher-gap (glyph text lost, pose collapse — see `turbo.md`,
  `project_turbo_teacher_gap_2026_06_29`, glyph probe in `bench/turbo`) lives in the
  text/pose axis, which content-tag read-back cannot see (tagger spatial-floor closed).
  Read-back correctly does not punish turbo's caption-following degradation — the
  property the proposal demanded. This is where the MLLM escalation arm binds if a
  consumer needs text/pose-sensitive scoring; it stays a stub until then.
- Turbo checkpoint spread (gate iii): weak — between-checkpoint std 0.055 vs
  same-prompt seed jitter 0.20 (ratio 0.28); `_500` is a clear early outlier, `_1k`→
  `_3500` converged. Content adherence is largely saturated by the warm-started
  student, consistent with distillation refining non-content axes.

Net: ship the primitive for content-adherence selection consumers
(`dave_mod_bestofn` `q_tag`, soup ingredient gating, seed selection, RWR
self-captioning — all content-adherence use cases). Phase 0b (style verifier) and the
RWR phases may proceed on this instrument. Supersedes PR #67
(`reward_weighted_artist_lora.md`, unmerged): the RWR estimator and phase discipline
are inherited from it, the reward stack is rebuilt.

Runs: `bench/readback/results/20260722-1105-v3-refit/` (real-data),
`bench/readback/results/20260722-1153-turbo/` (+ generated-image + turbo pair/spread);
renders at `bench/readback/renders/20260722-1117-turbo/`.

- Planned bench: `bench/readback/run_bench.py` (new; Phase-0 judge validation on
  pairs with known ground-truth ranking — no training, no new models).
- Planned bench: `bench/rwr/verifier_probe.py` (from PR #67, revised: style
  verifier validated content-matched, see Phase 0b).
- Planned script: `scripts/rwr/` (from PR #67; ReST grow/improve over the existing
  CFM inner loop via `library/runtime/harness.py::build_anima` — no `train.py` fork).
- Premise sources:
  - Read It Back (arXiv 2607.11886) — image-conditioned prompt log-likelihood as a
    training-free T2I reward; group-relative use cancels the language prior; their
    ablation: scalar 1–5 judging degrades, VQA-decomposition is mid, likelihood
    wins — and reward-policy alignment rivals reward-model scale.
  - `library/captioning/anima_tagger_model.py` — the in-house judge: multi-label
    tag logits + 3-class rating + people-count off frozen PE, dual-encoder
    hard-routed (PE-Core → identity/artist/character sub-head; PE-Spatial →
    localized tags), per-tag calibrated thresholds.
  - `bench/tagger_eval/run_bench.py` — per-tag / per-KB-slice / per-frequency-tier
    calibration measurement already exists; the judge is verified, not assumed.
  - `docs/proposal/dave_mod_bestofn.md` § Phase 0 — already wants a per-image
    `q_tag` (tagger recall vs prompt tags). This proposal promotes that one-off
    scorer to the shared primitive.
  - `docs/methods/turbo.md` — the <1 s 4-step grow engine; its caption-following
    degradation (teacher-gap: text lost, pose collapse) is a *known confound* the
    reward design must not punish blindly.
- Metric-trap priors this design must obey:
  - CMMD is distribution-level, fragile at n≈24–96, misranks within-family at
    cfg-1 — it cannot be the per-image judge and must not gate any A/B here.
  - `_archive/proposals/paired_gram_eval.md` Phase 0 failed: token-Gram style scores
    are content-dominated. Any style verifier must prove it is not a content
    detector (Phase 0b is content-matched for exactly this reason).
  - PE manifold is collapsed on our data (IP-Adapter diagnostics, archived:
    participation ratio ~6, cross-pair cosine 0.69) and PE-Spatial's final layer
    was blind to the model axis (paired-eval probe, archived). Within-family
    discrimination is an empirical question, not a given — hence Phase 0a.
  - No quality reward exists (the Null-TTA guard). Tag read-back measures
    adherence, not aesthetics; likelihood rewards are explicitly biased against
    text-independent aesthetic detail (Read It Back § related work). Quality stays
    on eyeball grids in every gate below.
  - `docs/findings/spectral_fraction_metric_inverts.md` — a score that cannot
    reproduce the eyeball on known cases predicts nothing; validate before use.

## The claim being made precise

Two lines currently point at the same missing primitive:

1. Selection consumers with no judge. Seed lottery, soup ingredient averaging
   (uniform, no gate — a bad draw is averaged in, not dropped), turbo checkpoint
   ranking (currently manual eyeballing of 4-step renders), best-of-N candidate
   pools (`dave_mod_bestofn.md`). All need a cheap, per-image, caption-aware
   score. CMMD can't do per-image; FM val loss tracks nothing.
2. RWR artist LoRA (PR #67) with a reward hole. Its composite reward
   (`style + λ_q·quality − λ_n·novelty`) has no prompt-alignment term, its grow
   phase rolls candidates under loose/dropout prompts and then FM-regresses them
   against captions they were never verified to match, and its `λ_q` term has no
   verified instantiation in this repo.

The claim: the Anima Tagger, read backwards, is that primitive — and it is the
strong instantiation of the Read It Back idea for this repo, not the weak one.
In the paper's taxonomy, tag read-back sits in the "VQA decomposition" cell; their
criticisms of that cell (decomposer lossiness, judge calibration) are both void
here because Anima captions already are an atomic decomposition (a tag set —
no decomposer), and the judge is a purpose-trained, threshold-calibrated,
domain-verified classifier with a live eval bench — not a zero-shot generalist
asked for yes-tokens.

```
readback(x, caption) := mean over content tags t ∈ caption of  log σ(tag_logit_t(x))
```

- Content tags only. Artist / quality / meta tags are masked: artist tags are
  the thing RWR is trying to learn (scoring them is circular), and quality tags
  are prior-driven noise (`docs/findings/mod_guidance_quality_tag_axis.md`).
  Rating and people-count heads score as two extra atomic checks.
- Group-relative only. Same caption, N images → compare within the group.
  This is the paper's language-prior cancellation, transposed: per-tag base rates
  and tag-order arbitrariness cancel exactly. Absolute readback values are never
  compared across prompts.
- Both judges over the same interface. The escalation judge — an external MLLM
  (Qwen3-VL-8B) teacher-forcing the tag string, i.e. SpectraReward proper — is a
  second implementation behind the same `score(images, caption) → [N]` call.
  Phase 0a benches both; the tagger is expected to win on cost and domain, the
  MLLM exists for what a set-based classifier structurally cannot see
  (composition, counting-beyond-people, spatial relations — the tagger
  spatial-floor line is closed, that ceiling is real). Note: Anima's own text
  encoder (Qwen3-0.6B base + T5-bridge) is not a candidate self-reward
  branch — the paper's backbone sweep puts even 4B judges at marginal-to-negative,
  and the Self-Spectra alignment result presumes a competent pretrained
  understanding branch, which 0.6B + a from-scratch projector is not.

### What this is not

Not a quality reward (guard above). Not an RL proposal — no policy gradient, no
backprop through sampling, nothing optimizes against the judge with gradients
anywhere in this document. Every use is selection/weighting (reward-hacking
pressure ≈ 0 for ranking; bounded for RAFT selection by the real-image anchor).
Not a tagger change — the judge is frozen throughout.

## Phase 0a — read-back validity (gate; no training, no new models)

`bench/readback/run_bench.py`. Score pairs where the correct ranking is already
known, with both judges:

| pair set | known ranking | source |
|---|---|---|
| turbo student vs teacher renders, same prompt/seed | teacher > student on text/pose adherence | `bench/turbo/` probes, teacher-gap grids |
| mod-guidance on/off arms | per the archived eyeball verdicts | `docs/findings/mod_guidance_quality_tag_axis.md` grids |
| xerox pairs (memorized render vs training image vs healthy render) | healthy ≈ memorized on readback of the memorized caption; memorized ≫ healthy on caption-shuffled readback delta | `bench/memorization/probe.py` outputs |
| shuffled-caption negative control | true caption > shuffled caption, per image | any render set + `caption_index.json` |

Gate (pre-registered): group-relative readback must (i) rank every known pair
correctly with AUC ≥ 0.8 per pair set, (ii) drop under the shuffled-caption
control for ≥ 90% of images, and (iii) show non-trivial within-family spread —
turbo checkpoints `_1000/_2000/_4500` must not score as noise (this is the
PE-blindness risk made falsifiable). Tagger judge failing while MLLM passes →
proceed on MLLM, note the cost. Both failing → stop; there is no instrument,
and the RWR phases below do not start.

Deliverable on pass: `library/` scoring helper + the pair-set harness, immediately
reusable by `dave_mod_bestofn.md` Phase 0 (its `q_tag`), soup ingredient gating,
and seed selection — the consumers exist whether or not RWR proceeds.

## Phase 0b — style verifier, content-matched (gate; inherited from PR #67, hardened)

`bench/rwr/verifier_probe.py`, with two changes forced by repo history:

1. Content-matched comparisons. Held-out same-artist vs other-artist negatives
   are compared within matched content-tag strata (via `caption_index.json`),
   or with tag-explained variance regressed out. An unmatched probe that passes is
   uninformative — paired-Gram "passed" unmatched framings and was a content
   detector. The gate must show style separation at fixed content.
2. Two verifier candidates, benched head-to-head:
   - PE-Core centroid cosine (PR #67's choice), content-matched as above.
   - The tagger's own artist sub-head logit — for in-vocab artists this is a
     trained style discriminator (identity head, PE-Core trunk), already
     calibrated by `tagger_eval`. For out-of-vocab target artists it degrades to
     the centroid path; both must be reported.

Near-duplicate control reuses `bench/memorization/probe.py` (the xerox detector)
instead of a new feature-space dedupe — the "style detector, not memorization
detector" check in PR #67's gate is precisely what that probe already measures.

Gate: content-matched AUROC well above chance for at least one candidate, and
score not dominated by nearest-train-image distance. Fails → RWR stops here
(Phase 0a deliverables stand on their own).

## Phase 1 — RWR single round, with the pairing fixed

As PR #67 Phase 1 (grow with `turbo + base` → score → top-k → CFM inner loop,
gradient never through rollouts), with three revisions:

- Self-caption the grow pool. Every selected candidate is re-tagged
  (`anime_tools.tagger.cli.main` path) and FM-trains against the caption read
  back from it, not the prompt that rolled it. This closes the pairing hole:
  dropout-prompt candidates get a caption at all, loose-prompt candidates stop
  training caption↔image pairs that never matched, and turbo's degraded
  caption-following (teacher gap) stops mattering — the pairing is corrected
  after the fact rather than penalized. It also removes the incentive structure
  under which ReST rounds erode prompt-following (autophagy on the alignment
  axis), complementing PR #67's accumulate-don't-replace real-image anchor.
- Selection score: `style_adherence − λ_n·novelty_penalty`, from Phase 0b and
  the memorization probe. If an alignment term is added to selection (optional —
  self-captioning already carries alignment), it must be group-relative within
  generator type (turbo candidates vs turbo candidates, full-step vs full-step),
  or the known turbo adherence deficit silently starves the 80% turbo pool and
  breaks the cost model.
- `λ_q·generic_quality` is dropped. No verified instantiation exists (guard);
  quality lives on the blind grid in the gate, as it already does everywhere else
  in this repo.

Gate: unchanged from PR #67 — 28-step SDE samples beat a matched MLE-on-images
LoRA on the Phase-0b verifier and on a blind grid for style adherence without
visible memorization of training compositions — plus one addition: readback of
the self-captions on final 28-step samples must not regress vs the MLE baseline
(the alignment-erosion canary, measured with the Phase-0a instrument).

## Phase 2 — multi-round ReST

As PR #67 Phase 2 verbatim (full-step candidate mix ~20%, real-image anchor,
novelty term, regeneration-frequency as the one knob, eyeball diversity gate),
with the Phase-1 readback canary carried across rounds: alignment drift across
outer rounds is the specific autophagy mode self-captioning is meant to prevent,
so measure it every round, not once.

## Escalation ladder (explicit, to prevent scope creep)

1. Tagger read-back, selection-only (this proposal).
2. External MLLM likelihood judge — only where Phase 0a shows the tagger ceiling
   binding on a real consumer (compositional prompts, open-vocab content).
3. LoRA-adapting the MLLM judge on `image_dataset/` (domain + tag format; the
   paper's pretrain-stage-beats-instruct result says captioning-style adaptation,
   not instruction tuning) — only after 2 shows validated signal with domain misses.
4. Policy-gradient RL against any of these — out of scope; re-propose
   separately with the reward-hacking analysis this document deliberately avoids
   needing.

## Cost accounting

Phase 0a is cached-render scoring: one tagger forward (PE encoders + heads) per
image over existing grids — minutes, one GPU, mostly reusing cached PE features
where they exist. The MLLM arm adds one Qwen3-VL-8B teacher-forced forward per
image (bf16 fits a 16 GB card; batch of tag-string scorings is cheap — no
generation, prompt-length forward only). RWR phases inherit PR #67's accounting
unchanged (grow amortized, improve = normal FM step); self-captioning adds one
tagger forward per selected candidate — noise.

## Open questions

- Readback aggregation. Mean log-confidence vs calibrated-threshold recall/F1
  (the `dave_mod_bestofn.md` `q_tag` shape): log-confidence is denser, F1 is
  threshold-honest. Phase 0a reports both; pick by gate AUC.
- Tail tags. Per-tag calibration is weakest on tail-frequency tags
  (`tagger_eval` per-tier readout); should readback weight tags by calibration
  quality, or mask below a frequency floor? Phase 0a's per-pair-set breakdown
  decides.
- Self-caption drift. Self-captioning trains the LoRA toward *tagger-visible*
  content — a subtle bias toward the tagger's vocabulary. Bounded by the
  real-image anchor (real captions stay in every round), but worth one grid: do
  round-2+ samples lose consistently-untaggable content? (Prior:
  `project_lora_crossattn_learns_labeled_only` says untagged content routes to
  style/uncond anyway, so the marginal harm should be small.)
- In-vocab vs OOV artists. If the artist sub-head wins Phase 0b decisively but
  the target artist is OOV, is a few-shot head refit (frozen trunks, one linear
  row) worth it? Cheap to test inside Phase 0b.

## References

- Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image
  Generation (arXiv 2607.11886) — SpectraReward / Self-SpectraReward; reward
  taxonomy (scalar / VQA / likelihood); backbone-scale non-monotonicity;
  reward-policy alignment result.
- PR #67 `docs/proposal/reward_weighted_artist_lora.md` (unmerged, superseded) —
  the RWR/ReST estimator rationale, turbo-grow safety argument (FM training is
  sampler-agnostic), cost model, and phase discipline inherited here.
- `library/captioning/anima_tagger_model.py`, `bench/tagger_eval/run_bench.py` —
  the judge and its calibration bench.
- `bench/memorization/probe.py` — xerox / near-duplicate control, novelty term.
- `docs/proposal/dave_mod_bestofn.md` — `q_tag` consumer; best-of-N order
  statistics for candidate pools.
- `_archive/proposals/paired_gram_eval.md` (Phase 0 failed) — the content-dominance
  trap Phase 0b is designed against.
- `_archive/proposals/seed_lottery_noise_floor.md`, `docs/experimental/soup.md` —
  selection consumers.
- `docs/methods/turbo.md` — grow engine; teacher-gap adherence confound.
