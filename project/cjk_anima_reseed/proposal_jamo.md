# proposal_jamo — Hangul rows from jamo identity, then warm (2026-10-09)

**Status: proposal.** Runs after `proposal_refactor.md` lands (the trainer,
renderers and KO fonts still live in `../cjk_anima_scale`). Nothing here is
built yet.

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
is the fallback; R3 (§ 4) decides between them.

Compound jamo are one index each in the syllable code: 11 compound finals
(ㄳ ㄵ ㄶ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅄ) and 7 compound vowels (ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ
ㅢ). A further variant writes them as sums, e.g. `F[ㅄ] = F₁[ㅂ] + F₂[ㅅ]`
and `V[ㅘ] = V[ㅗ] + V[ㅏ]` (+ a small own term). That moves the rare
compound finals onto the common simple ones (ㅄ: 없 / 값; ㄳ: 몫 / 넋). It
is an R3 read on F64's rows before it becomes an arm.

`r(s)` is a per-syllable free residual, 0 at phase 2's start. It is the
part of a glyph's identity that the jamo do not explain.

## 3. Prerequisites (CPU unless noted)

1. **Hangul per-glyph routing.** `glyph_route` splits JA tokens only
   (`ext_vocab.is_ja_glyph`). The pack has 755 multi-syllable Hangul Qwen
   tokens with rows of their own (회사, 처럼, 다시, 여기 …; `하세요` → ext
   1 208, routed or not). Phase 1 draws single glyphs and does not need
   this; phase 2's dialogue lines do. Extend the glyph test to Hangul
   syllables, space-prefixed forms included, behind the pack's mapping so
   that existing packs and pure-JA prompts encode bit-identically.
2. **Reader.** The ruler reads with `sfx` (JA SFX reader) and `vl` (stock
   PaddleOCR-VL 1.6, multilingual). KO reads use `vl` alone. Before
   trusting a miss, calibrate it per syllable on font-drawn Hangul through
   the same crop path (GPU, minutes).
3. **Floor.** No Hangul string has a cached floor. The floor is the pack's
   untrained Hangul rows on seed_1008, rendered on the read sets. **Adding
   this floor key needs the user's OK.**
4. **Fonts.** The six KO faces in `kozh/` (Nanum Gothic, Do Hyeon, Jua,
   Black Han Sans, Nanum Pen Script, Nanum Myeongjo) cover KS X 1001 only,
   so trained and held sets stay inside it. LXGW WenKai covers all 11 172.
5. **Trainer mode** (`cjk_scale.train`, or its successor after the
   refactor): `factor = "jamo"`. The optimizer holds the factor tensors
   (and in phase 2 the residuals). Each step composes Δ for the live
   Hangul rows into `rows.delta.raw` before the forward, so gradients reach
   the factors through the composition. It sits beside `stick_only` /
   `ball_on`, which already restructure the update. The norm pull applies
   to the composed rows as it does to free rows (open: § 7).
6. **Config.** `factor = "jamo"` plus the sets from 4 as `rows` / `held`
   specs. `lang = { korean = … }` as kozh16 has it (KO faces, `korean
   text` / `Korean text reads as` captions).

## 4. Phase 1 — jamo identity (cold, micro arms)

**Sets** (`probes/jamo_sets.py`, CPU). Every jamo occurs in KS X 1001
(19 / 21 / 28; layouts VF 1 069, HF 585, CF 347, V 149, C 109, H 91). Cells
to cover: 19 initials × 3 classes + 21 vowels + 28 finals + 6 layouts =
112. A coverage-greedy pick covers all 112 cells with 96 syllables and 109
with 64 (checked 10-09). A pure-coverage pick lands on rare syllables
(찮 쉽 쮸 흗 …); break ties by frequency once a KO frequency list exists.
**Held-out H**: 32 common syllables (가 나 다 … 요 했 습 …) whose cells
the trained set covers. No arm trains H.

**Arms**, ~25 min each (3 600 steps):
- **J64**: `factor = "jamo"`, the 64 trained syllables.
- **F64**: free cold rows (kozh16's recipe), the same 64 syllables, data
  and steps (56 steps a row, under the 225 recipe: an equal budget is the
  comparison).

Data: lone / bubble1 / grid tiers, **Hangul-only grids**. kozh16 put
Hangul and hanzi in the same grids (887 of 1 600 items), which confounds
any "Hangul sits near hanzi" geometry read. Every other row stays frozen at
seed_1008.

**Reads** (per glyph, sheets viewed):
- **R1, trained 64**: J64 vs F64 vs the floor. Does the factorisation cost
  identity on the glyphs it trained? Score each miss **per jamo position**
  (initial / vowel / final right or wrong), not only per syllable: kozh16
  got every initial right and missed mostly vowels.
- **R2, zero-shot on H** (the main read): J64's composed rows vs the floor
  vs F64-regressed (the jamo model fitted to F64's free rows by least
  squares, composed on H). Does jamo identity reach syllables never drawn?
- **R3, CPU**: how much of F64's free-row energy the jamo model explains
  (fit R², leave-one-out cos), with and without `cls`. Does free training
  already learn the composition, and does `cls` earn its 38 extra vectors?

**Gate G1 → phase 2**: J64 reads H well above the floor, glyph by glyph,
and stays level with F64 on its trained 64. If H transfers but trained
identity drops, phase 2's residual is the fix, so go on. If H does not
transfer, stop here: either per-syllable cold on a frequency-ranked subset,
or a glyph-image init in place of jamo.

Scaling, if G1 passes: J96 (all 112 cells), then J256 on a frequency
ranking, before fixing phase 2's factor table.

## 5. Phase 2 — warm on Korean dialogue

- **Init**: every syllable's row = the phase-1 composition. All 11 172 get
  one for free; KS X 1001 is what the fonts train. `r(s) = 0`, factors
  and residuals both trainable (factor lr below the residuals': open, § 7).
- **Norm pull on `r(s)` only**, so a rare row rests at its composition and
  not at the pack row (`sent_kanji`: under AdamW the pull walks a rare warm
  row back to the pack).
- **Data**: a KO dialogue corpus for the `sent` tier and windows. Candidates
  (`task_report.md` § 4): songys Chatbot_data (MIT, 11.8 k pairs),
  SmileStyle (CC BY-NC), OpenSubtitles ko (unclear). Lettering:
  **horizontal first**, since KO manga / webtoon dialogue is mostly
  horizontal. `HORIZONTAL_FRAC = 0.3` and the `tategaki` windows are
  JA-tuned, so a KO item draws horizontal by default and vertical as the
  minority. Needs prerequisite 1.
- **Standalone jamo** (ㅋㅋ, ㅠㅠ, ㄹㅇ, common in dialogue) are their own
  glyphs: free rows outside the factor model.
- **Reads**: KO strings held out of the lines by 5-gram (the ruler's rule),
  `vl`, per glyph, against the floor. Spot-check the JA ruler on seed_1008
  (the JA rows are frozen, so only shared-caption leakage can move it).

## 6. Budget

| step | cost |
|---|---|
| sets, routing, trainer mode, R3 | CPU |
| reader calibration | GPU, minutes |
| floor render (needs OK) | GPU, one pass over H + 64 + 96 |
| J64, F64 | 2 × ~25 min |
| J96 / J256 | ~40 min / ~1.5 h (if G1 passes) |
| phase 2 | set by the corpus; for comparison, per-syllable cold on KS X 1001 ≈ 63 h |

## 7. Open questions

- **Delta or effective factorisation.** Default: the delta, with the pack
  row kept. On seed_1008 the trained delta's per-row component runs against
  the pack's (same-row centred cos −0.28, kana and kanji, 10-09), so the
  pack's per-syllable part may fight the composed one. Fallback arm:
  replace each Hangul pack row with the Hangul pack mean, so the
  composition carries the whole identity.
- `C × cls` or plain `C`: R3 decides, or run both as J64 variants.
- Compound jamo as sums of simple ones, or one vector each (§ 2).
- Phase 2: factors trainable at a lower lr, or frozen.
- KO corpus and its licence.
- `vl`'s Korean accuracy on these renders (prerequisite 2).
- The floor key (prerequisite 3).
