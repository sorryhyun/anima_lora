# CJK-aware Anima — plan

*What remains. Verdicts: [`findings.md`](findings.md) · what exists:
[`deliverables.md`](deliverables.md). Rewritten 2026-08-30 after the plan3
closure; the earlier plan/plan2/plan3 files are folded in (plan3's measured
part lives on as [`reports/0830_adapter_lora.md`](reports/0830_adapter_lora.md),
the glyph-line proposal at `_archive/cjk_aware_anima/plan2_glyph_line.md`).*

Status: **research levers on the encoder side are exhausted except one; v1
is shippable with `cjk_vocab_pack_synthja_v2` (glossary signed off 2026-08-30; corpus rebuilt + retrained 2026-08-31, [`reports/0831_axis_joiner_rebuild.md`](reports/0831_axis_joiner_rebuild.md)).**
Scope stays Japanese-only for v1; Korean is planned in [`plan_ko.md`](plan_ko.md) (corpus job + joint retrain, encoder unchanged); zh after that.

## Phase 3 — ship v1 (the vocab pack)

Ordered; nothing here needs a new experiment.

1. ~~**Glossary sign-off**~~ DONE 2026-08-30, no overrides changed → no re-cache. (Was: review
   `assets/tag_glossary_review.md`, write fixes to `datasets/tag_overrides.json`,
   `tag_glossary.py --reselect`, rebuild pairs. Re-cache is only needed for
   rows whose wording changed (the stager is pair-keyed) — in practice one
   `cache_synth2` refresh.
   **2026-08-30 addendum — two corpus bugs fixed before the retrain** (see
   `datasets/README.md`): (a) character tags outside `image_dataset` fell to
   `general` and were MT-rendered as words (`ame (mignon)` → 雨（可愛い）);
   axis now falls back to the wiki category / artist-OC form. (b) the student
   joiner `、` was itself an ext row; pairs now join with `", "` (80 %) /
   `、` (20 %), recorded per pair. Both change wordings corpus-wide →
   `cache_synth2` restaged from scratch; retrained as
   `cjk_vocab_pack_synthja_v2` (label `2c-synthja-v2`).
2. ~~**Retrain `synthja`**~~ DONE 2026-08-31 as `synthja_v2` — acceptance met
   where v1 promised it (tags clean, mixed names improved, far-disc 0.089,
   coverage unchanged, G1 green); n1/n2/r3 full-JA names remain the expected
   fails. Recipe for the record: (~20 GPU-min; pass
   `--loss span --steps 12000 --batch_size 32 --param global --trust provenance`
   + the register sampling explicitly — `distill.py` defaults are a different
   experiment) and re-render both grids. Acceptance = the 2c surface:
   rendered same-seed grid `ja_ext ≈ ja_t5en` on the tag / quote / mixed-name
   prompts (n1/n2/r3 full-JA names are **expected fails**, out of v1);
   per-register `cos_student_vs_en_attn` at the `synthja` band; `coverage.py`
   no user-facing tag token under floor; far-disc ≤ 0.2; D7 LoveHina readout
   reported; G1 green.
2b. ~~**Allowed-kanji filter**~~ DONE 2026-08-31 — JA wordings gated on
   joyo+jinmeiyo+reviewed whitelist (3,072 chars, veto-only; zh leakage the
   kana/Shift-JIS guards missed is gone: 崩坏→崩壊, 结月ゆかり dropped,
   僵尸→キョンシー). 22-wording glossary diff, 0 emoticon/general regressions,
   corpus census clean; retrained as `cjk_vocab_pack_synthja_v4`
   (label `2c-synthja-v4-kanjifilter`). See
   [`reports/0831_kanji_filter.md`](reports/0831_kanji_filter.md).

3. **Promote the encoder** — `HybridT5Encoder` + strategy shim out of
   `bench/cjk_adapter/` into `library/anima/`, sidecar auto-discovery (flag to
   disable), `load_dit_model` row append, TE-cache path through the shim.
   Unit test: EN bit-exact with and without the sidecar.
   *2026-09-01: module promotion DONE — `library/anima/ext_vocab.py` is the
   canonical home, `bench/cjk_adapter/ext_vocab.py` is a re-export shim, all
   56 cjk tests green. The strategy shim / `load_dit_model` row append /
   TE-cache path remain open (not needed for the external test deploy).*
4. **Release asset + docs** — pack files on the release tag, a
   a `cjk_vocab_pack.md` under `docs/methods/` (user-facing: what works, what doesn't,
   the JA TE-cache regeneration note), i18n of the GUI/guidebook line if a
   field is exposed.
   *2026-09-01: test release live at
   https://huggingface.co/sorryhyun/anima-vocab-pack-ja — `synthja_v4`
   metadata-stamped as `anima_ja_vocab_pack.{safetensors,json}` + the Qwen3
   tokenizer files (self-contained). Repo docs still owed for the full ship.*
5. **ComfyUI node** in `ComfyUI-Anima_lora-Adapter` (tokenize wrap + embed
   object patch). Ship after the in-repo path is verified, not before.
   *2026-09-01: `AnimaVocabPackLoader` landed (node v3.9.0): CLIP-clone
   tokenizer wrap (EN bit-exact verified against comfy's native
   `AnimaTokenizer`; comfy's bundled qwen25 fast tokenizer is vocab-identical
   to qwen3_06b, encoder output verified id-for-id vs the repo-side encoder)
   + clamp-pre-hook/embed-forward-hook pair for the 32128 hardcode.
   `ext_vocab.py` added to the node's `_vendor` surface. Owed before public
   node publish: a rendered same-seed grid through the ComfyUI path.*

Kill/rollback: if the retrained pack regresses any tag prompt vs the current
`synthja` grid, ship the current pack and treat the offending override as a
review bug.

## Phase 5 — corpus extension (the one open research lever)

Everything on the encoder side has been falsified for rare kanji names
*except the target itself*: every arm distilled toward the frozen adapter's
output on captions where the name was **swapped into someone else's
attributes**. The untried hypothesis is that the teacher's composition of a
name is only learnable from pairs where the name **co-occurs with its own
attributes** in the caption — real captions, not templates.

Two sub-items, cheapest first; both are text-only builds off the existing
`build_pairs.compose` path and train with the settled recipe.

### 5a. Under-floor general tags (cheap, ships in the pack)

**Run 2026-08-31** (`datasets/synth_tags.py`, register `tags_synth_ja`,
pack `synthja_v3` — report 0831 §6): the mechanism works — every 0-visit
target moved (c1/c2/c3/t2/t6) with no name regression — **but the t3 armor
gate did not clear at floor 300** (`銀の鎧` 193→300 was only 107 pairs).
Remaining §5a item: one rerun at a higher floor/per-target for the armor
family (+ `--extra-terms`) before calling t3 phrase-binding-limited.

### 5b. Real co-occurrence corpus for names (research)

- **Source**: caption-only crawl by character tag from the gelcrawl route
  (the D1-wide crawler already fetches text-only), for the grid's characters
  plus the wiki `post_count ≥ 3000` set — every caption then carries the
  name *with* its canonical attributes (`hakurei reimu, black hair, red bow,
  detached sleeves, hair tubes, …`). Compose through the glossary, name pinned
  (the `names_synth_ja` path), register `names_real_ja`.
- **Pre-check before training** (CPU + one daemon job): the residual probe on
  the *teacher* side already separates the three grid characters at margin
  0.78, so the target exists. Run `coverage.py` to confirm the new corpus
  puts `博`/`麗`/`霊`/`夢` co-visited with their attribute rows, not just
  above floor.
- **Arm**: retrain from the signed-off `synthja` with `names_real_ja` in the
  mix (sampling ~0.2, as `names_synth_ja`). Rows only — no adapter LoRA
  (plan3 showed no instrument can rank LoRA arms).
- **Gate** (eyeball, this arm vs `synthja`): n1/r3 full-JA Reimu renders
  black hair + red bow + miko; r1 keeps its gain; t1/t2/m1 clean; residual
  margin on Reimu leaves the ≈0 floor (`residual_probe.py` — floor gate, not
  selector).
- **Kill criterion**: if margin stays ≈0 with co-occurrence coverage complete,
  the encoder side is closed for rare names at any corpus, and the only
  remaining route is a DiT-side target (image-level training with JA
  captions = the glyph line's prerequisite, below). Do not follow with more
  pairs, JESC, STAIR, or D2 — recorded as inert.

### Instrument owed alongside 5b

A **DiT-side render scorer** (turbo 4-step student + Anima Tagger: does the
render carry the character's tags?) so arm selection stops depending on the
eyeball grid. Build it before 5b's second arm if the first is ambiguous;
findings §5 says nothing in adapter space can rank arms.

## Phase 4 (deferred) — the glyph line

Promoted from the old plan.md Phase 4 as plan2 (2026-08-17), now parked
behind Phase 3; full proposal in `_archive/cjk_aware_anima/plan2_glyph_line.md`.
Essentials so it is not re-designed from scratch:

- Goal: the DiT **renders** quoted JA text — OCR in-image JA into captions
  (`「営業中」と書かれた看板`), then train image-level with those pixels **in**
  the loss. Output is an ordinary DiT LoRA that *depends on* the vocab pack.
- Prerequisite = Phase 3 step 3 (TE caching through the ext shim).
- Phase 0 ceilings resolved (findings §6); OCR backend is swappable
  (manga-ocr + logprob gate ≈ −0.3 default); masks reused as localisation.
- **Core safety rule**: selective unmask — captioned text regions trainable,
  uncaptioned text regions stay masked; text-free images are the negative
  class. Keep text masking ON everywhere until this phase.
- Gates: render→OCR CER on held-out strings (the owed D6 instrument, never a
  cosine), no text spam on the 2c grid, G1, tag-register readout unchanged,
  and an unmask-everything control arm.
- Escalation only on measured CER failure: font-rendered glyph strip as an
  **EasyControl** condition (synthetic composites train it without OCR).

## Not planned

- 2-iii full adapter finetune; any further adapter-LoRA arm (rank / MLP /
  token gate / attn weight) — plan3 closed, no ranking instrument.
- More pairs of the synthetic kind, JESC / STAIR / D2 growth, mT5.
- ko/zh before ja ships.

## Risks carried forward

1. **Register drift** — D1's MT-JA dominates; D7 (LoveHina) readout in every
   retrain is the instrument; a gap vs the MT holdout is the signal.
2. **Teacher ceiling** (t5en 0.69–0.76 flat / 0.823 readout) — accepted for
   v1; only image-level training passes it.
3. **Zero-shot zh/ko rows poisoning neighbours** — provenance-flagged; demote
   to `<unk>` in the mapping JSON if the grid shows it.
4. **Ship surface** — ComfyUI core hardcodes 32128; the node must object-
   patch, and the pack is a release asset that has to be on the tag.
