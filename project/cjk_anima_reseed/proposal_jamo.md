# proposal_jamo — Hangul rows from jamo identity, then warm (2026-10-09)

**Status (10-11): phase 1 done, G1 not passed; next D128, then J256.** The jamo factors
carry glyph form and jamo positions to held-out syllables, not syllable
identity. The runs and their reads:

| run | report | on held-out H (32): v4 free exact · jamo F1 (chance ~0.13) |
|---|---|---|
| J64 / F64 / J96 / J128, R3 | `reports/jamo_phase1_2026_10_10.md` (floor skipped, user 10-10) | J64 1 · 0.25, J96 2 · 0.25, J128 3 · 0.27; F64-reg 0 · 0.15 |
| the jamo curriculum (`jamo_lone` → `jamo_j64_lone`) | `reports/jamo_curriculum_2026_10_11.md` | 1 · 0.23: a jamo's standalone row is not its block component |
| J64 at J128's steps (`jamo_j64_112`) | `reports/jamo_vl_reads_2026_10_11.md` § 4 | 0 · 0.17: more steps on the same 64 cost the transfer |

Settled on the way: R3 keeps `C × cls` over plain `C`; F64's free rows hold
no additive jamo structure; the KO reader is v4 (`probes/jamo_vl.py`, 71 %
on font-drawn H); `jamo_vl_sheet.py` scores glyph / jamo P / R / F1.
Hangul per-glyph routing (`glyph_route_ko`) and `factor = "jamo"` are in
(`reseed/jamo.py`, `reseed/trainer.py`).

## 1. Why

- **Every syllable already has a row; the seed trains none** (kozh16,
  10-09, trained 7 syllables + ㄹ cold beside it). The punct pack
  holds all 11 172 syllables: 2 512 are single Qwen tokens (`qwen` map;
  뷁, 똠, 갃 among them), 8 596 are two byte fragments and 64 are three
  (U+C000–C03F, 쀀–쀿; 쀼 the only one in KS X 1001). The fragment ones are
  regrouped per char onto `char` rows (8 660). One T5-side row per syllable,
  routed or not (`ext_encoder`, checked 10-09).
- **kozh16's Hangul misses are jamo misses** (`reports/kozh16_2026_10_09.md`,
  sheet `results/20261009-1414-kozh16-render/`, read by eye 10-09). Of 7
  syllables + ㄹ, cold at 225 steps each, 가 and ㄹ are exact. The other 6
  are one jamo off: 힝→항, 몹→믑, 없→잆, 양→앙, 한→환 (the vowel), 감→갈
  (the final). **The initial is right in all 8.** The rows learned "a
  syllable starting with ㅎ / ㅁ / ㅇ / ㄱ" and not the rest of the block.
  The hanzi miss the same way: 说→祱 keeps 兑 and swaps the radical. Geometry:
  the Hangul rows grow a stick of their own (~57° off both seed sticks), and
  their spikes sit outside both seed balls (0.086 / 0.11 of the energy in
  the kana / kanji top-40, against 0.28 / 0.30 for held-out seed rows).
- **Per-syllable cold training does not scale.** At the cold recipe (225
  steps a row) and kozh16's rate (3 600 steps / 25.8 min), KS X 1001's 2 350
  rows are ≈ 529 k steps ≈ 63 h; all 11 172 ≈ 300 h. A dialogue corpus
  reaches the rare syllables a handful of times at best.
- **Hangul is compositional.** syllable = (initial, vowel, final), index
  `(cho·21 + jung)·28 + jong`, 19 × 21 × 28. The glyph's layout is set by
  the vowel's class (vertical ㅏㅐㅑㅒㅓㅔㅕㅖㅣ: the initial on the left;
  horizontal ㅗㅛㅜㅠㅡ: on top; compound ㅘㅙㅚㅝㅞㅟㅢ: both) and by whether
  there is a final, so 6 layouts. A row built from jamo parameters
  trains every syllable that shares a jamo.
- **Not through routing.** A byte-level T5 stream (2–3 rows per syllable)
  has nothing to carry identity: a lead fragment is shared by up to 96
  syllables, UTF-8's 4 + 6 + 6 bit split does not follow jamo boundaries,
  and the frozen adapter never learned to compose a glyph from a sequence
  (the fragment-mean init already showed it: 60 % of random char-row pairs
  above cos 0.5, `ext_vocab.build_ext_table`). A jamo T5 stream (3 rows) has
  the same composition problem. The factorisation lives in the **row
  parameters**; the T5 stream keeps one row per syllable.
- **What a jamo stream would look like** (checked 10-09). Compatibility
  jamo already route cleanly, one row each: `ㅇㅓㅄ` → ext 29 191 / 30 012 /
  30 052, `ㅇㅓㅂㅅ` → … / 29 608 / 29 491. Conjoining jamo (NFD
  U+110B U+1165 U+11B9) are re-composed by Qwen's tokenizer to `없` → its
  syllable row 25 998. So a T5-side fold (없 → ㅇㅓㅄ, Qwen text kept) is
  mechanically possible. But those rows already mean the **standalone**
  jamo, as typed in ㅋㅋ / ㅠㅠ: full-size glyphs drawn side by side, not
  positioned parts of a block. Folding onto them would give one row two
  meanings.

## 2. The row model

    Δ(s) = b + C[cho, cls(jung)] + V[jung] + F[jong]        (+ r(s) in phase 2)

Delta units (`raw × row_scale`, over the pack row, as every reseed run).
`cls` ∈ {vertical, horizontal, compound}; `F[0]` (no final) = 0, `b`
absorbs it. 1 + 57 + 21 + 27 = 106 vectors of 1 024. `C × cls` because an
initial's place and shape follow the vowel (ㄱ in 가 vs 고); the final
always sits at the bottom. The plain additive model (`C[cho]`, 68 vectors)
is the fallback; R3 (phase 1) kept `cls`.

Compound jamo are one index each in the syllable code: 11 compound finals
(ㄳ ㄵ ㄶ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅄ) and 7 compound vowels (ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ
ㅢ). A further variant writes them as sums, e.g. `F[ㅄ] = F₁[ㅂ] + F₂[ㅅ]`
and `V[ㅘ] = V[ㅗ] + V[ㅏ]` (+ a small own term). That moves the rare
compound finals onto the common simple ones (ㅄ: 없 / 값; ㄳ: 몫 / 넋). It
is an open question (§ 6).

`r(s)` is a per-syllable free residual, 0 at phase 2's start. It is the
part of a glyph's identity that the jamo do not explain.

## 3. Next — D128, then J256

**Why the set's design matters** (design matrix: a syllable → its b / C[cho,
cls] / V / F indicators, 106 columns; CPU, 10-11). A *memorizable* row is
one outside the span of the set's other rows: the factors can fit it
without touching what the others share. H is *determined* when its row lies
in the set's row space, so its composition follows from the trained rows
and not from the init.

| set | rank | memorizable | H determined | distinct pairs CV / VF / CF |
|---|---|---|---|---|
| J64 | 64 | **64 / 64** | 0 / 32 | 64 / 56 / 58 |
| J96 | 95 | 80 / 96 | 17 / 32 | 88 / 76 / 80 |
| J128 | 102 | **2 / 128** | 32 / 32 | 109 / 83 / 84 |
| D128 (top 1000) | 94 | 6 / 128 | 32 / 32 | **125 / 108 / 113** |

Every J64 row is memorizable, which reads J64-112's drop: more steps fit
each of the 64 on its own. J128 covers nearly every cell twice and leaves 2.

**D128** (user 10-11): 128 syllables whose jamo combinations differ as much
as possible, from common syllables only. Greedy over the 1 000 most
frequent KS X 1001 syllables (the Qwen merge-rank proxy of
`probes/jamo_sets.py`), H kept out: most cells seen 0 times, then most seen
once (each cell twice, as J's order), then most new (cho, jung) / (jung,
jong) / (cho, jong) pairs, then frequency. Against J128 it keeps the
memorizable count low (6 vs 2) and draws more pairs (VF 108 vs 83, CF 113
vs 84), with no rare blocks (its rarest: 뉘 뷔 뚫 흙 몫 얘; J128's: 챦 퐈 곬
퀭 뾔 쟬). The cost: 8 cells no common syllable holds are left out —
ㅃ / ㅆ / ㅉ / ㅍ before a compound vowel, the finals ㄽ ㄾ ㄿ ㅋ — so rank
94 vs 102; ㄳ, ㄵ and ㅒ are seen once.

- **Arm**: `jamo_d128`, `factor = "jamo"` at 56 steps a row (7 168 = J128's
  total, ~50 min). The set into `assets/jamo_sets.json` as `D128`; its own
  data build (Hangul only, as `jamo_j64`).
- **Read**: H and the words, v4 free (`jamo_read.py render --held`,
  `jamo_vl.py score`, `jamo_vl_sheet.py`), against J128 at the same
  budget: exact, glyph R, jamo P / R / F1 over chance. Separates the set's
  design from its size.
- **Then J256**, 56 a row (14 336 steps, ~1.7 h): D128 extended by the same
  rule if D128 ≥ J128 on H, else J128 + the next 128 by frequency; H kept
  out either way.
- **Gate G1 → phase 2**: H identity well above J128 (exact and jamo P, not
  recall alone). If J256 moves only recall, as J64 → J128 did, the factor
  model's ceiling is form, not identity: stop at phase 1 and take
  per-syllable cold on a frequency-ranked subset, or a glyph-image init in
  place of jamo.

## 4. Phase 2 — warm on Korean dialogue (if G1 passes)

- **Init**: every syllable's row = the phase-1 composition. All 11 172 get
  one for free; KS X 1001 is what the fonts train. `r(s) = 0`, factors
  and residuals both trainable (factor lr below the residuals': open, § 6).
- **Norm pull on `r(s)` only**, so a rare row rests at its composition and
  not at the pack row (`sent_kanji`: under AdamW the pull walks a rare warm
  row back to the pack).
- **Data**: a KO dialogue corpus for the `sent` tier and windows. Candidates
  (`status.md` § KO / ZH): songys Chatbot_data (MIT, 11.8 k pairs),
  SmileStyle (CC BY-NC), OpenSubtitles ko (unclear). Lettering:
  **horizontal first**, since KO manga / webtoon dialogue is mostly
  horizontal. `HORIZONTAL_FRAC = 0.3` and the `tategaki` windows are
  JA-tuned, so a KO item draws horizontal by default and vertical as the
  minority. Routing is in; the builder must write `build.json`
  `glyph_route_ko` (no builder writes it yet).
- **Standalone jamo** (ㅋㅋ, ㅠㅠ, ㄹㅇ, common in dialogue) are their own
  glyphs: free rows outside the factor model.
- **Reads**: KO strings held out of the lines by 5-gram (the ruler's rule),
  v4, per glyph. Spot-check the JA ruler on seed_1008
  (the JA rows are frozen, so only shared-caption leakage can move it).

## 5. Budget

| step | cost |
|---|---|
| D128 / J256 sets + data builds | CPU |
| D128 train | ~50 min |
| J256 train | ~1.7 h |
| H render + v4 read + sheet | GPU, ~5 min |
| phase 2 | set by the corpus; for comparison, per-syllable cold on KS X 1001 ≈ 63 h |

## 6. Open questions

- **Delta or effective factorisation.** Default: the delta, with the pack
  row kept. On seed_1008 the trained delta's per-row component runs against
  the pack's (same-row centred cos −0.28, kana and kanji, 10-09), so the
  pack's per-syllable part may fight the composed one. Fallback arm:
  replace each Hangul pack row with the Hangul pack mean, so the
  composition carries the whole identity.
- Compound jamo as sums of simple ones, or one vector each (§ 2).
- Phase 2: factors trainable at a lower lr, or frozen.
- KO corpus and its licence.
