# CJK-aware Anima — plan (Phase 2 onward)

*Line home: [`motivation.md`](motivation.md) (why) ·
[`done.md`](done.md) (what is measured and built) ·
[`report_0816_phase2.md`](report_0816_phase2.md) (Phase 2 measured verdicts —
2b unit gates + 2c first pass).*

Status: **Phase 2b CLOSED, Phase 2c first pass MEASURED** (2026-08-16). Two
packs trained at the settled design (`param=global`, `loss=span`) saturate the
span loss and reach ~87% of the addressable teacher signal in the readout space
on the corpus holdout — and the 20-prompt render grid splits exactly along
supervision density: high-visit tag content transfers, thin/zero-visit content
(fantasy vocab, character names, prose function words) collapses, while the
teacher renders at EN parity everywhere. G5 additionally demoted the flat
`cos_vs_en ≥ 0.6` gate: the exact argmin of the span objective scores 0.13 on
it, so it was blind to the objective, not the student to the gate. All of this
is measured in [`report_0816_phase2.md`](report_0816_phase2.md#phase-2c--first-pass-2026-08-16).
Consequence: **the tag register is coverage-bound and unblocked** (D1 widening
+ name register, CPU-only); the prose registers stay **objective-bound** (no
sequence-level term, no gradient). Scope is **Japanese only**; zh/ko are
"rerun the same pipeline with a different corpus" once ja passes.

## Phase 2 — distillation design (settled)

Student and teacher share the Qwen3 hidden states (same native-JA source
text); only the T5-side ids differ:

```
teacher:  adapter(qwen_hidden(ja), t5_ids(en_translation))     # no grad
student:  adapter(qwen_hidden(ja), t5ext_ids(ja))              # grad → ext rows only
loss:     match(student_out, teacher_out)
```

**Settled at 2-i-a** (G0 + G2; confirmed at 2c scale): a shared `global`
correction over the ext rows — low-rank + per-dim diagonal + scalar gain, no
per-row freedom — under `loss=span` (`global_row`'s 1,892 extra rows buy
nothing measurable end-to-end). The measurement story — why flat cosine is a
control and not a gate, the readout space, the real-query bank, centered
readouts — is settled and lives in the report ([G2 defects](report_0816_phase2.md#the-three-instrumentation-defects),
[G5](report_0816_phase2.md#g5--the-flat-gate-is-measured-blind-2026-08-16));
`build_query_bank.py`'s real image-token queries and `fit_centers`' centered
readout stay load-bearing. Escalation rungs stay defined but **unused**, each
needing fresh justification:

- **2-ii: + low-rank adapter on `q_proj`/`in_proj` of the adapter blocks.**
  Only if 2-i plateaus below target *after* the coverage work — nothing in the
  first pass indicates a capacity limit (G0 passed; the failures are
  unsupervised rows, which 2-ii cannot fix either). Breaks the EN-bit-exactness
  guarantee, so it must clear an EN-regression gate and ship as a separate
  optional sidecar.
- **2-iii: full adapter finetune** — not proposed.

### Training loop — built

`scripts/distill_cjk/` (see [`done.md`](done.md)); one-off gate drivers in
[`gates/`](gates/) (`g2.py`, `g34.py`, `g5.py`, and `coverage.py` — the
CPU-only per-prompt span-visit diagnostic that sizes the corpus work). Two
properties are contracts: the **padding invariant** (max-pad 512, no
`crossattn_seqlens` masking) holds on both arms, and **GPU work goes through
the daemon** (`make daemon-run` / `--queue`). The Phase-1 cost model
(~10⁵–10⁶ pairs, "hours not days") stays recorded as wrong: the surface
saturates in ~20 GPU-minutes and the binding constraint is span-carrying ext
rows, not compute.

## Corpus — where the next win is

Visit distribution was assumed to be the whole game for the tag register, and
D1-wide has now tested that: 5.4× the captions moved the 500+ band 381 → 756
and cleared most of the 1–4 band, **and moved no `v=0` token off zero**
([report](report_0816_phase2.md#d1-wide--the-gelcrawl-widening-measured-2026-08-16)).
So the diagnosis splits, and only the second half is still open:

- **1–49 bands = coverage-bound.** Closed by D1-wide. More captions here have
  diminishing value; they multiply the same glossary.
- **`v=0` = wording-bound**, not coverage-bound. `armor` occurs 39× and `鎧`
  still has 0 visits, because the glossary bound the tag to `アーマー`. That is
  the D1-words row above and it is a review task, not a corpus task.
- **`v=0` on non-tag vocabulary** (`京都` `暖簾` `騎士` `畑`) is unreachable
  from tag captions at any pool size — D2/D3 register work or D5 names.

`gates/coverage.py` prints the per-prompt gap; it is now the instrument for
the *wording* pass, and its `v=0` column is the review queue.

| ID | Source | Status / next |
|---|---|---|
| D1-wide | **`~/gelcrawl/retrieved/` EN tag captions** — 16,053 vs the 3,008 `image_dataset/` captions D1 was built on | **DONE and MEASURED** ([report](report_0816_phase2.md#d1-wide--the-gelcrawl-widening-measured-2026-08-16)) — 3,008 → 16,128 captions, corpus 18,990 → 45,230 pairs, 500+ visit band **381 → 756**, 4.3× visits. But it buys **visits, not vocabulary** (rows visited flat at ~6,400), so it improves the 1–49 bands and **leaves every v=0 token at 0**. Not a further lever: more captions multiply the same glossary. |
| D1-pairs | **`p1atdev/danbooru-ja-tag-pair-20241015`** (CC0) — the danbooru wiki's own `other_names` (Oct-2024) + `calm3-22b-chat` fill; 151,431 rows (93,393 character / 22,330 copyright / 35,708 general) | **Tail fill DONE** (`datasets/tag_pairs.py`, CPU): 5,248 previously-unresolved tags filled, unmapped segments **42,530 → 13,714**. **Item 2 DONE 2026-08-17** ([measured](report_0816_phase2.md#d1-pairs-item-2--the-arbiter-re-selection-measured-2026-08-17)): 1,538 tags re-selected to `tagpair_verified`, 4,438 wordings moved in total; the polysemy class (`bow`) is *unwinnable by F1* and lands in the review file instead — the residual lever on this source is the human sign-off, not more arbitration. Original rationale: this source knows the *booru sense* that MT structurally cannot, because MT translates the English string. It agrees with only 35% of our 5,435 MT-derived wordings, and on the high-traffic disagreements it is usually right — `bow` → 蝶結び (ribbon) not our お辞儀 (*bowing*, 1,582 occ), `on back` → 仰向け (supine) not our 後ろ姿 (*from behind*, 2,032 occ), `clothes lift` → たくし上げ not 服が持ち上がる, `multiple girls` → 群像 not 複数の女の子1人. Lever: feed its names as candidates into the existing back-translation arbiter (`--mt`, GPU) so they can win on evidence, which also turns `tag_glossary_review.md` from an eyeball pass into a sourced diff. It likewise carries the native-kanji candidates D1-words is about — **626 tags / 55,821 occ (8.6% of tag mass)** where our choice is a katakana loanword and it offers a kanji (`penis`→陰茎, `short hair`→短髪, `nude`→裸). **Not** a drop-in: unfiltered by its own admission (Chinese leaks `skirt`→裙, `pleated skirt`→百褶裙; its own MT errs, `after anal`→後背位 = *doggy style*), so it is a candidate source and the arbiter stays load-bearing. |
| D1-words | **Glossary wording** — the real v=0 cause | **The unblocked lever now.** `armor`→`アーマー` not `鎧`, `from above`→`上から` not `俯瞰`, `close-up`→`クローズアップ` not `接写`: the kana-proves-Japanese rule picks katakana loanwords over native kanji *and* excludes the kanji from `alt_pool`, so it is reachable from neither register. 119 tags / 1,735 occ; **half are Chinese and correctly rejected** (`bed`→`床` is *floor* in JA), so it is a **human review axis**, not an automatic fix — and a new one the current review file does not surface. Lever: a second review batch → `tag_overrides.json` (CPU, free). |
| D5-names | Wikidata lexicon → **name register** | Compose name-bearing captions directly (thousands of paired names already resolved). Risk 4 is now measured in renders — MT transliterates names, so their rows never accumulate visits through D1 alone. |
| D2 | Danbooru artist commentary, 73,015 native-JA records, 9,068 paired | **In the mix but inert under `loss=span`** (prose carries no spans — measured, [report](report_0816_phase2.md#d2--what-the-commentary-corpus-buys-2026-08-16)). MT pass resumable at 5,721/69,668. Blocked on the sequence-term decision; do not grow it before that. |
| D3 | STAIR Captions (~820k JA on COCO) + YJ Captions | Pre-aligned sentence register, zero MT. Same block as D2. Verify license (CC BY 4.0 last checked). |
| D4 | JESC (~2.8M subtitle pairs) | Casual dialogue / speech-bubble register. Same block; subsample if ever mixed. |

Ja-only scoping: the ext table's zh/ko rows stay physically present (inert
without zh/ko input). The sidecar ships per-row provenance
(`trained` / `zero-shot`); never-visited rows stay at zero-shot init and get
flagged — if eval shows they poison nearby generations, demote them to the
`<unk>` fallback in the encoder mapping (a JSON edit).

## Phases & gates

- **Phase 2a — corpus. BUILT** (`post_image_dataset/cjk_distill/`, five
  registers; builders in [`done.md`](done.md) / [`datasets/README.md`](datasets/README.md)).
  Two items outlive it:
  - **OPEN — user sign-off on the tag glossary. Hard ship blocker.** 36.6% of
    span tokens are `mt_unverified` (`colored inner hair` → 色付きの陰毛), and
    a bad wording trains that tag's rows toward the wrong meaning. G4b measured
    the trust-policy hedge as a non-lever (dropping unverified spans is
    *worse*), which makes the human review the only instrument.
    `assets/tag_glossary_review.md` (200 rows ⇒ ~64% of occurrences) +
    `spotcheck.md`; fixes → `datasets/tag_overrides.json`. **Measured: D1-wide
    grows this surface far less than feared** — the widened corpus's 7,242 new
    tags carry only 3.2% of its occurrence mass, and the current glossary
    already covers 97.3% of it. Batch them into the same file. The **second**
    review axis (D1-words: katakana primary vs native-kanji candidate, 119
    tags) is new and must be batched in alongside.
  - **D6 stays eval-only**; the owed instrument (same template, different
    quoted strings) is still owed, and it will not be a cosine in this space
    (G3: ~0.02 readout headroom). Glyph identity is Phase 4's job.
- **Phase 2b — loop + unit gates. CLOSED 2026-08-16.** G0b/G0/G1/G2/G3/G4
  green; verdicts + the three instrumentation defects in the
  [report](report_0816_phase2.md). Settled: `param=global`, `loss=span`,
  `--trust provenance`, `--train_registers tags,tags_alt`.
- **Phase 2c — train + eval. FIRST PASS MEASURED 2026-08-16** — packs, G5, the
  flat probes, the render grid, and the coverage diagnosis are in the
  [report](report_0816_phase2.md#phase-2c--first-pass-2026-08-16). Remaining 2c
  work, in order:
  1. **Coverage pass**: D1-wide **DONE** (corpus is built; 45,230 pairs);
     D1-pairs **tail fill DONE** (unmapped segments 42,530 → 13,714);
     D1-pairs **item 2 DONE 2026-08-17** (arbiter re-selection + the owed
     widened `--mt` rebuild in one job: 4,438 wordings moved, unmapped
     segments → **878**, grid moves on the coverage-bound prompts —
     [report](report_0816_phase2.md#d1-pairs-item-2--the-arbiter-re-selection-measured-2026-08-17)).
     Remaining, re-ordered by what the item-2 measurement showed binds:
     **(a) human review sign-off** — the arbiter cannot win the polysemy class
     (`bow`→蝶結び back-translates to the *sense*, not the string) nor the
     katakana/kanji class (`bed`→床), both now sourced in
     `tag_glossary_review.md` (400 rows) → `tag_overrides.json`;
     **(b) name register** (D5) — n1/n2 unchanged, 博:10 麗:31;
     **(c) targeted caption widening** for tokens under the O(100+) floor
     (`鎧`:23, `照明`:3) — crawl caption-only by EN tag from the coverage
     under-floor list. Then retrain (~20 GPU-min; pass
     `--loss span --steps 8000 --batch_size 32` explicitly — the distill.py
     defaults are a different experiment) and re-render the grid.
  2. **Sequence-term decision** for the span-less registers (prose/quotes) —
     the flat probes rule `flat` out as the extra term; `attn` is the measured
     candidate (+13% commentary in its own register, tags unharmed). Decide,
     then size D2/D3/D4 to it.
  3. Iterate 1–2 until the grid passes acceptance.
  - **Acceptance (re-based by G5 — the flat 0.6 gate is retired to a
    control):**
    - **rendered same-seed grid** (`assets/ja_eval_prompts.json`, 20 prompts ×
      `en / ja_t5en / ja_ext`): user judges `ja_ext` ≈ `ja_t5en` on prompt
      adherence — this is the binding gate;
    - per-register `cos_student_vs_en_attn` against the teacher's, fixed
      holdout mix (G3: `recovery_attn` is a mix statistic; first pass: 0.80 vs
      0.875 on-distribution);
    - `gates/coverage.py`: no user-facing content token under the visit floor;
    - discrimination **far** ≤ 0.2 flat (pathology guard — the probes show
      what its violation looks like: `flat`-trained near-disc 0.914);
    - **native-register readout (D7)**: student-vs-teacher cos on held-out
      LoveHina lines, reported separately from the MT-derived holdout
      (register-drift signal, not a hard gate);
    - EN regression: G1 stays green post-training (trivially true for 2-i);
    - *controls, not gates*: flat `cos_vs_en` and student-vs-teacher flat cos
      (report direction of movement only — G5 measured their ceiling under
      this objective at 0.13–0.31).
- **Phase 3 — ship.** Promote encoder + strategy shim out of `bench/` (auto
  route CJK spans through the ext encoder when the sidecar is present; flag to
  disable), extend the adapter embed at DiT load, sidecar as a release asset
  (pattern: the CNS γ npz), ComfyUI loader-node touchpoint (see Deployment).
  TE-cache regeneration note applies (padding-invariant §: any tokenizer change
  invalidates cached `.npz`/TE sidecars for JA captions — EN caches unaffected
  by construction).
- **Phase 4 (deferred, separate proposal when reached) — OCR captions + glyph
  rendering.** OCR in-image JA via manga-image-translator (the `make mask` MIT
  backend), append verbatim strings to captions, train image-level so the DiT
  binds ext-token ids to glyphs. **Explicitly ordered**: keep text-masking ON
  until this phase — unmasking makes the loss pay for glyph pixels the model
  can't yet produce.

## Deployment — a "vocab pack", not a LoRA

The Phase 2 artifact **cannot be a LoRA**: LoRA expresses a low-rank delta on
an *existing* matrix, while this artifact is (a) new rows appended to the
adapter's `embed [32128, 1024]` table (a shape change) and (b) a tokenizer
mapping (segmentation rules + char/token → row-id JSON), which is behavior,
not weights. It ships as the two files Phase 1 already emits —
`ext_embed.safetensors` + `ext_embed.json` — as a release asset (pattern:
CNS γ npz). Per surface:

- **In-repo**: strategy shim routes CJK spans through `HybridT5Encoder`;
  `load_dit_model` appends the rows to `llm_adapter.embed`. Auto-discovery
  of the sidecar (flag to disable). Composes with any checkpoint and any DiT
  LoRA by construction (disjoint parameters). Preprocess/TE-caching uses the
  same strategy, so JA captions cache with ext ids transparently.
- **ComfyUI** (verified against core, 2026-08-15): tokenization lives in the
  CLIP object (`comfy/text_encoders/anima.py::AnimaTokenizer`, stock
  `T5TokenizerFast`; ids ride conditioning extras as `t5xxl_ids`), and the
  MODEL holds the adapter with a **hardcoded** `Embedding(32128, …)`
  (`comfy/ldm/anima/model.py:159`). One custom node — "Anima JA Vocab":
  `(MODEL, CLIP, vocab_pack) → (MODEL, CLIP)` — wraps the CLIP's t5xxl
  tokenize path (pure Python) and object-patches the adapter embed with the
  extended table (ModelPatcher object patch, per the Adapter Loader repo's
  forward_hook-not-override invariant — don't clobber the class). Home:
  `ComfyUI-Anima_lora-Adapter`. Endgame: upstream to ComfyUI core (core
  already owns Anima natively) → node-free.
- **Bake-in rejected**: expanding `embed.weight` inside a forked DiT
  checkpoint forks the base model, *breaks* stock ComfyUI load (32128 is
  hardcoded in core), and still doesn't carry the tokenizer — the
  irreducible piece is code, which a weights file can't ship.
- The optional 2-ii fragment (low-rank `q_proj`/`in_proj` deltas) *is* a
  genuine LoRA and rides the existing Anima Adapter Loader — but always
  alongside the vocab pack, never instead of it.
- Phase 4's eventual glyph-rendering training, by contrast, outputs an
  **ordinary DiT LoRA** on every existing loader — it merely *depends* on
  the vocab pack being loaded, like any LoRA depends on its base.

## Risks / open questions

1. ~~**Position misalignment**~~ — RESOLVED: bounded by G4a (JA/EN token ratio
   0.96–1.20), disqualified as an objective by G2, and disqualified as a *gate*
   by G5. Flat metrics are controls.
2. **Teacher ceiling**: distilling to t5en caps at t5en. Accepted for this
   phase (the first-pass renders show the teacher ceiling is comfortably high);
   image-level training (Phase 4) is the lever past it.
3. **Register mismatch**: if D1's MT-JA dominates, the student may handle
   translated-sounding JA better than native phrasing — D7 is the instrument
   (held out entirely; student-vs-teacher cos reported separately). A gap
   between the two *is* register drift. Widening D1 5× raises this risk's
   exposure; keep the D7 readout in every retrain.
4. **Proper-noun distribution gap** — now MEASURED, promoted from risk to work
   item: the n1/n2 renders lose character identity and the coverage diagnostic
   shows why (name-token rows at 0–37 visits). Lever: the D5 name register
   (Corpus table). Track proper-noun coverage separately in the spot-check.
5. **Row-norm / manifold**: zero-shot rows needed a ×1.66 norm rescale;
   training may drift norms — log row-norm distribution vs the original
   table as a diagnostic.
6. **Discrimination as a trap** — now MEASURED: the pure-`flat` probe is the
   worked example (near-disc 0.914, readout alignment negative). The far ≤ 0.2
   gate and the near-vs-1.0 residue reading stay as specified in the
   acceptance list.
