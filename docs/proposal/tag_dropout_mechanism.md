# Tag dropout / randomize mechanism — does caption-variant training buy sparse-prompt richness, and what does it cost?

Status: **Phase 0/1 RUN 2026-07-22 — near-miss, N≥64 rerun owed** (artist mignon, 79 imgs; harness `bench/tag_dropout/`, results `bench/tag_dropout/results/20260722-1957-phase1`). Phase 0: instrument valid (shuffle discrimination 0.958) but richness>base failed on a soup shard (weak single-artist LoRA, not a bug); the characteristic set must be TF-IDF-distinctive (`--min_distinct_ratio`, default 2.0), not raw artist-frequency. Phase 1 (A=shuffle_only vs B=dropout_0.1, 20 ep): sparse-richness win-rate B>A = 0.571 vs gate ≥0.6 at N=28 — every richness axis agrees, adherence non-regressed (0.562); underpowered, so the pre-registered N≥64 run (10 seeds) is the honest resolution — record neither "inert" nor "works" yet. Gotcha: a `{stem}.variants.txt` sidecar next to the resized image OVERRIDES the CLI `--caption_tag_dropout_rate`; move sidecars aside to A/B rates. Instrument dependency satisfied: the
tag-readback judge passed its validity gate
(`tag_readback_reward.md` Phase 0a, `bench/readback/`).

## What is actually being tested

The caption-variant system has shipped for a while without a mechanism story
or an outcome measurement. What it is (grounded in code, not lore):

- Cache-time, not train-time. `make preprocess` materializes
  `{stem}.variants.txt` sidecars via
  `anime_tools.captions.variants::generate_caption_variants`:
  v0 = pristine caption, v1..v3 smart-shuffled (`@artist` prefix and section
  anchors protected), then per-tag dropout at
  `caption_tag_dropout_rate = 0.1` (live default, `configs/preprocess.toml`).
  The TE step encodes all variants; training samples among them
  (`use_shuffled_caption_variants = true`, `configs/methods/lora.toml`) at
  20% pristine v0 / 80% uniform over v1..v3
  (`library/anima/strategy.py:381`), so the effective per-draw tag-absence
  probability is 0.8 × 0.1 = 0.08. A `--use_shuffled_caption_variants_only`
  flag (never draw v0 → absence 0.1) already exists — a free higher-exposure
  arm that needs no re-caching.
- Three separable axes, only two of them on: order (shuffle),
  presence (dropout 0.1), identity (`caption_tag_randomize_rate`,
  currently 0.0 — replaces a tag with a fresh dual-single vocab token while
  keeping its slot).
- The dataset-side `caption_tag_dropout_rate` in `library/datasets/subsets.py`
  is inert for us (captions are pre-encoded); the variants are the live
  mechanism. Arms in this proposal are therefore preprocess-time arms.

## Priors that make this analyzable now

1. The extreme case is settled: LoRA cross-attn learns *labeled* tags
   only; content that never carries a tag routes to style/uncond
   (prior finding, `project_lora_crossattn_learns_labeled_only`). Tag dropout
   at rate p is a dose knob on that axis — each tag's content is trained
   as "unlabeled" p of the time. The user-facing hope (rich renders from
   sparse captions) is exactly this migration; the same mechanism predicts
   the unmeasured cost: migrated content becomes non-negatable (appears
   unprompted; removing the tag from the prompt no longer removes the
   content).
2. The null-attention result (`bench/null_key/report.md`): the 512-pad
   tail is a pure per-query denominator constant, so caption *length* sets a
   global cross-attn damping factor. Dropout shortens captions → the dropout
   arm trains under systematically stronger null-gating. That is a genuine
   confound between "content unlabeled" and "caption shorter" — and
   tag-randomize is its natural control: identity erased, slot count
   unchanged. The two quirks form the axis-separating pair.
3. The trigger-binding subtlety: `@artist` survives every variant
   (prefix-protected), so dropped-tag content has two places to go —
   the uncond/style pathway or the always-present trigger token. For artist
   LoRAs the trigger is the *desired* sink. The design below distinguishes
   them (render with vs without the trigger).
4. Instrument: the tag-readback judge is validated for per-image caption
   adherence on real AND generated images (AUROC 0.98–1.00 cross-prompt,
   shuffled-caption drop 0.991). Matched-seed paired renders across arms
   remove the seed-jitter that made the turbo checkpoint-spread read weak
   (between-ckpt std 0.28× seed jitter — the lesson: never compare arms
   unpaired).

## Metric-trap priors this design obeys

- No CMMD anywhere (fragile at n≈24–96, misranks within-family at cfg-1).
- Readback measures adherence/coverage, not quality; quality stays on
  blind eyeball grids in every gate (no-quality-reward guard).
- Tagger ceiling: composition/spatial relations are invisible to a set-based
  classifier (spatial-floor line closed). Richness here means attribute
  coverage, and the claim is scoped to that.
- Per-tag calibration is weakest on tail tags — richness/leakage sets are
  restricted to tags above a frequency floor with healthy `tagger_eval`
  calibration.

## Hypotheses (pre-registered)

- H1 (richness — the user's hope): sparse-prompt renders from the
  dropout-trained arm show higher characteristic-content coverage than the
  no-dropout arm.
- H2 (cost — leakage): dropout-trained content intrudes on prompts that
  omit it; per-prompt controllability decreases with p.
- H3 (adherence): prompted-tag adherence does not regress at p = 0.1.
- H4 (locus): the migration lands on the trigger token (richness requires
  `@artist` in the prompt) rather than the global uncond — the benign
  outcome for multi-LoRA composition.
- H5 (weight-side): the dropout arm shifts adaptation mass out of
  cross-attn modules toward self-attn/MLP (per-module ΔW share).

## Phase 0 — prompt sets + instrument wiring (no training)

Build the evaluation harness against existing checkpoints (any current
artist LoRA; all are dropout-0.1-trained, so Phase 0 cannot attribute — it
only proves the harness):

- Characteristic tag set per artist from `caption_index.json`
  (`make caption-index`): content tags above the frequency floor, present in
  ≥k% of the artist's images, minus identity/artist/quality tags (same
  masking as the readback primitive).
- Prompt sets: (a) sparse — `@artist, 1girl, solo` style stubs;
  (b) detailed — held-out val captions; (c) omission — detailed captions
  with the top-m characteristic tags textually removed (the H2 probe);
  (d) no-trigger — set (a) without `@artist` (the H4 probe).
- Scores (all group-relative, matched seed across arms):
  `richness = readback(render, characteristic_set ∖ prompted_set)`,
  `adherence = readback(render, prompted_set)`,
  `leakage = readback(render, omitted_set)` on omission prompts.
- Gate: harness sanity only — richness/adherence/leakage must reproduce
  known orderings on existing models (e.g. adherence(true caption) ≫
  adherence(shuffled caption); richness(artist LoRA) > richness(base DiT) on
  sparse prompts). Fails → fix instrument before spending any training.

Deliverable: `bench/tag_dropout/run_eval.py` (render + score, reusable for
every later arm), prompt-set builder off `caption_index.json`.

## Phase 1 — the attribution A/B (2 trains)

Two arms, identical everything except the variant knob, on one mid-size
artist:

- A `shuffle_only`: variants on, `caption_tag_dropout_rate = 0` —
  isolates the dropout axis (shuffle stays in both arms; its own effect is
  Phase 2 material, not confounded here).
- B `dropout_0.1`: the live default.

Operational discipline:

- Per-arm TE caches via the subset `cache_dir` / text-cache redirect (the
  colorize pattern — `project_text_cache_dir_te_redirect`) so both variant
  families coexist on disk; VAE/PE caches shared. Verify arm identity from
  the run's `.snapshot.toml` + cache paths before scoring (the
  misplaced-key-silent-default lesson from turbo).
- Same seed, same steps/LR (the settled lora.toml stack), `--queue`d behind
  whatever is running.
- Renders: prompt sets (a)–(d) × ≥4 seeds × both arms, matched noise;
  `make gen` batch.

Gate (helps): paired per-(prompt, seed) richness win-rate B > A ≥ 0.6
with N ≥ 64 pairs on sparse prompts, and adherence non-regression
(paired win-rate ≥ 0.45 on detailed prompts), and a blind eyeball grid
with no quality regression. Leakage (H2) and locus (H4) are measurements,
not gates — they price the mechanism rather than pass/fail it.

Weight-side read (H5, free): per-module-family ΔW Frobenius share
(cross-attn q/kv/proj vs self-attn vs MLP) for both checkpoints — the
migration story predicts B's cross-attn share is lower. Corroboration only;
never a gate (channel scaling redistributes gradients identically in both
arms, but the mapping from ΔW share to behavior is not calibrated).

Outcome tree:

- H1 pass, H2 small → dropout is validated as shipped; document as a
  finding, consider whether 0.1 is on the dose plateau (Phase 2a).
- H1 pass, H2 large → dropout trades controllability for richness; surface
  the knob in docs/GUI guidance (community-facing — sparse-prompt users want
  it, composition-heavy users don't).
- H1 fail, H2 fail → 0.075 effective dose is inert; either close (it's
  cheap insurance) or run one high-dose arm (0.3) to find where the
  mechanism turns on before concluding anything.
- H1 fail, H2 pass (leak without richness) → dropout is a net cost at this
  dose; propose lowering the default.

## Phase 2 — axis separation (conditional, ≤2 trains)

Only the axis Phase 1 implicates:

- 2a dose–response: p ∈ {0.3} (and 0.05 if the budget allows) — locate
  the knee on whichever of H1/H2 moved. Cheapest first step:
  `--use_shuffled_caption_variants_only` lifts effective absence 0.08 → 0.10
  on arm B's existing cache (train-flag-only arm, no re-preprocess).
- 2b length-vs-identity: `randomize_r` arm at r matched to B's effective
  absence rate (erasure pool, r-family). Randomize keeps slot count —
  if 2b reproduces B's effect, the mechanism is identity-unbinding; if not,
  it's the caption-length/null-gate regime (connects directly to
  `bench/null_key`). Also the first live test of the shipped-but-off
  randomize knob.
- 2c (optional) pristine arm (variants off entirely) to price the
  shuffle axis itself — only if someone needs it; shuffle is not currently
  in question.

## Cost accounting

- Phase 0: prompt-set construction + one render grid on an existing
  checkpoint (~50–100 images, `make gen`) + tagger scoring (one PE+heads
  forward per image) — about an hour of GPU, mostly renders.
- Phase 1: TE-cache regen ×2 (minutes; text-only, VAE/PE untouched),
  2 LoRA trains (the dominant cost, hours each, daemon-queued),
  ~2×4 prompt-set×seed render grids (~200–300 images total), scoring
  minutes.
- Phase 2: +1–2 trains, same eval harness.

## Open questions

- Aggregation for richness: mean logsigmoid over the characteristic set vs
  calibrated-recall count — Phase 0 reports both, pick by which reproduces
  the known orderings more cleanly (same policy as readback Phase 0a).
- Artist choice: needs enough images that the characteristic set is stable
  (≥40 images above the tag-frequency floor) but small enough to train
  fast. Default to one of the soup-pipeline artists so configs exist.
- ~~Does variant sampling weight v0 vs v1..v3 uniformly?~~ Resolved at
  proposal time: 20/80 weighted (`library/anima/strategy.py:381`), effective
  absence rate 0.08. Legacy caches without the `v0_intact` marker sample
  uniformly — Phase 1 must confirm both arms' caches are modern.
- Multi-LoRA composition interaction (H4's payoff): if migration is
  trigger-bound, dropout-trained LoRAs should compose better than
  uncond-bound ones — out of scope here, but worth one line in the Phase 1
  report if H4 resolves.

## References

- `anime_tools.captions.variants` — the mechanism under test.
- `configs/preprocess.toml` (`caption_shuffle_variants=4`,
  `caption_tag_dropout_rate=0.1`, `caption_tag_randomize_rate=0.0`).
- `docs/proposal/tag_readback_reward.md` + `bench/readback/` — the validated
  judge (Phase 0a PASS) and its scoping (content-adherence axis only).
- `bench/null_key/report.md` — pad tail = null attention; caption length as
  global cross-attn damping (the Phase 2b rationale).
- Prior findings: LoRA cross-attn learns labeled tags only; turbo
  checkpoint-spread seed-jitter lesson (pair everything); text-cache
  redirect pattern (colorize).
- arXiv 2607.19139 — the register/readback paper whose sink analysis
  prompted the null-key bench; motivating context only, its mechanism does
  not transfer to cross-attn DiTs.
