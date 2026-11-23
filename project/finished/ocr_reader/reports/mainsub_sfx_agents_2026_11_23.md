# main / sub: SFX-only crops checked and SFX crops with speech partly inside labelled by agents (2026-11-23)

Definitions and the loader rules these feed: [`main-sub-approach.md`](../main-sub-approach.md).

## SFX-only crops: Facts

- Set: train crops of `m109_mainsub.jsonl` whose box holds one SFX main and no other annotated
  text, whole or partly visible (`crop_dataset.sfx_only`): 20,090 (20,085 after the loader's
  other filters).
- 200 drawn at random (seed 7), the 4 px training crop upscaled to a short side of 300 px, read by
  4 general-purpose agents × 50 (local: scratchpad `sfxonly/result{0..3}.tsv`).
- none 190, other 2, unsure 8. The 2: `MomoyamaHaikagura_042_25` (a neighbouring hand-lettered
  column cut by the edge), `UltraEleven_102_09` (floor numbers in the background); both read on the
  crop by Claude. The 8: strokes cut by the edge that the agents could not tell from art.
- The user (2026-11-23): empty subs go in under the negative cap, 1 : 1 with `swaprelax_v5`;
  "그림인지 글자인지 모르면 출력 안하는게 맞는듯" (unsure marks are not typed).

## SFX-only crops: Interpretation

- At 2 of 200 with text, the empty sub is right on about 99 % of the set under the user's rule
  for unsure marks; the misses are text the annotation does not cover, which no filter on the
  annotation can find.

## SFX crops with speech partly inside: Facts

- Set: train crops whose main is an SFX, with a speech item 0.05 – 0.8 inside and no speech
  whole inside: 636. Before this pass they trained `OCR:` only.
- 5 Sonnet agents × 130 (the last 116), the crop upscaled as above, with the main and the full
  annotation text of each partly visible item given as a hint. Rules given: type what the crop
  shows, a glyph when about half or more shows, one item per line in reading order; check the
  main (local: scratchpad `sfxpart/result{0..4}.tsv`).
- typed 231, empty 380, unsure 25; main wrong 4.
- Claude's read, labels visible, of 12 random typed rows: 12 right in content and column order.
  Of the flagged rows: three typed a glyph whose dakuten or radical is cut, from the hint text
  (`AkkeraKanjinchou_044_27` `だ` with `た` showing, `YamatoNoHane_093_13` `ぜ` / `せ`,
  `UchiNoNyan'sDiary_001_29` `狩` with `守` showing). `HarukaRefrain_041_09` left out the
  numbers of a clock face.
- The user (2026-11-23): "시계도 일단 sub로 넣자"; the main fixes applied; the labels train as
  proposed (typed → sub, empty and unsure → SFX-only negatives).
- Written to `assets/m109_agent_subs.tsv` (local; pass `sfxpart`), with the clock numbers added to
  `HarukaRefrain_041_09` and the main fixes `PrismHeart_075_34` `わぁっ` → `わあっ`,
  `HanzaiKousyouninMinegishiEitarou_090_27` `ゴ\nゴ` → `ゴ\nウ・・・\nゴ`,
  `HeiseiJimen_054_15` `ファン\nファン\nファ` → `ファン` × 3. `AosugiruHaru_028_21`'s main covers
  part of the SFX and has no fix (`mainsub_plan.md`). The two SFX-only crops with text are in
  the same file as `other` (pass `sfxonly_check`).
- Loader after the change: `OCR2:` sub 26,677 → 26,907 (230 from the agents), sub_neg 6,669 →
  6,726 (SFX-only 3,240, `swaprelax_v5` 3,240, Manga109 reviewed 246).

## SFX crops with speech partly inside: Interpretation

- Sub positives on SFX-main crops go from 163 to 393: the "SFX main, speech sub" case the split
  was meant for now has a few hundred rows, still under 1.5 % of `OCR2:`.
- The empty two thirds are crops with a balloon edge and no readable glyph, which makes them
  harder negatives than the plain SFX-only ones.
- The hint text lets an agent complete a glyph the crop cuts; the three found are small next to
  231, and they are left as typed.
