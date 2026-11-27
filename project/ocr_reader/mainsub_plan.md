# ocr_reader — main / sub: open questions and pending work

Definitions and the dataset: [`main-sub-approach.md`](main-sub-approach.md). A finished item
leaves this file for a report.

## Open, for the user

- "Largest" when a dense block meets a few large glyphs. `GarakutayaManta_006_34` is reviewed
  with the English paragraph as main and `ひ～` as sub (the main is over the cap, so only the sub
  trains); `AosugiruHaru_031_20` is the same case unreviewed (proposed main `ググッ`, the ranking
  list as sub).
- When a sub counts as complete. Most `OCR2:` positives are unreviewed ruby reads, about half
  right in a 14-row read ([report](reports/mainsub_data_2026_11_21.md)).
- How a main / sub read is scored. The v7 gates read per kind (COO SFX / speech, doujin SFX, K3);
  the val subs are unreviewed ruby reads.
- How the `OCR2:` sub text enters a caption. `anime_tools` v0.14.0 reads `OCR:` only (the user,
  2026-11-24: "일단은 ocr만 쓰는식으로 하자").
- The line-break pass on val outside the speech manifest: proposed, not decided.

## Work

- `TARGET_NORM = 13` (2026-11-27, the user): outside a Hangul string a run of `ー` / `―` /
  hyphens (any mix, any count, one included) → one `ー`, Chinese included; `〰` → `~` and a `~`
  run → one `~`. The hyphen stays, as one `-`, in a run of hyphens alone beside an ASCII letter
  or digit (`747-400D`) and in a Hangul string, as norm 12 had it (dashes and `ー` / `-` runs
  → one `-`, a lone `ー` kept). In the target and the scoring keys (`core/textnorm.py` fold 7).
  No run on it yet.
  - The user's decisions: "영어나 한글 사이에 있을때만 하이픈 보존"; `〰` → `~`, a run one; on
    keeping `-` in Chinese against one spelling in kanji-only Japanese: "일본어 장음으로 하자.
    왜냐하면 다른 많은 만화 대사에서 일본어 장음을 쓰고 있고, 그걸 '늘임'으로 학습하고 있으니까.
    중국어에서의 호환성 문제는 있지만 적어도 대시 하이픈 장음이 들쭉날쭉 나오는거보단 나을거라
    생각해".
  - Facts, the stored reads rescored on the new key (`mainsub_1124_1923` / `mainsub_1126_2213`
    / `mainsub_1127_0028`; `output/ocr/eval/*.jsonl`, `val_ep*.jsonl`, local): val main
    11,086 / 11,086 / 11,083 of 11,634 (95.3 % each), sub 641 / 642 / 647 of 1,168; Manga109
    test SFX 2,098 / 2,061 / 2,083 of 2,558 (82.0 / 80.6 / 81.4 %), speech 2,449 / 2,457 /
    2,453 of 2,559 (95.7 / 96.0 / 95.9 %); sincos SFX 351 / 320 / 312 of 593; K3 KO 32 / 30 /
    30, ZH 36 / 35 / 35. Paired `1923` → `0028`: val main +134 / −137, speech +25 / −21,
    Manga109 SFX +100 / −115 (p 0.34), sincos +28 / −67.
  - Facts, the runs it rewrites in `load_mainsub` train + val labels: kana strings `―` runs
    1,394 and `ー` runs of two or more 1,366, hyphens not beside a letter or digit 18; `〰` in
    30 labels; strings of ideographs alone, Manga109 132 dash or hyphen runs and 79 `ー` runs,
    `gelnote_zh` 15 dash or hyphen runs (`——全部剪掉吧` trains as `ー全部剪掉吧`). Hyphens
    kept beside a letter or digit: 66.
  - Facts, the variant not taken (`-` kept in a string of ideographs with no kana and no `ー`):
    the Manga109 kanji-only strings split 132 `-` against 139 `ー`; rescored, val main 11,062 /
    11,061 / 11,082 and speech 2,441 / 2,447 / 2,451 before the `〰` fold.
  - `anime_tools`: the same fold as `ocr.reader.fold_drawn_lines`, applied by `guard` after
    its runaway and length tests, in the sibling checkout's working tree (not committed, no
    release, pin unchanged). The local HF build of `load_mainsub` predates norm 13.
- `TARGET_NORM = 12` (2026-11-27, the user; replaced by 13 the same day): every `―` → `-`, a `ー` / `-` run → one `-`, a
  `~` run → one `~`, a lone `ー` kept (`ケーキ`; `core/textnorm.py` fold 7), in the target and
  the scoring keys. On `mainsub_1126_2213`'s val reads 168 of 1,466 misses differed only in the
  count of `ー` / `~` / `―` (140 short) and 198 read `ー` for a labelled `―`. The stored reads
  rescored under the new keys (val main / Manga109 test SFX / speech / sincos SFX):
  `mainsub_1124_1923` 92.9 % / 82.0 % / 92.1 % / 351, `mainsub_1126_2213` 93.0 % / 80.5 % /
  91.8 % / 320. `cjk_rl`'s `sent_strings` takes the fold through `normalize_target`.
  First run: `mainsub_1127_0028` (local, 4 × 4, 3 epochs, otherwise `mainsub_1126_2213`'s
  recipe): val main 92.6 %, sub 55.4 % (ep3); Manga109 test SFX 81.4 %, speech 91.8 %; sincos SFX
  312 (heart-blind 397 against 442); K3 KO 30, ZH 35. Its ep2 reads sincos SFX 285, Manga109
  SFX 78.6 %. The sincos losses are hearts read as a glyph (`びく♥` → `びくん` / `びくっ`,
  `ガク♥` → `ガクル`): 121 rows against 51 on `mainsub_1126_2213`. (Until 2026-11-27 this
  entry gave val main 93.3 % and speech 93.0 %; the run's `summary.md` and
  `reports/eval/ocr_eval_mainsub_1127_0028_cap4_ocr.md` hold 92.6 % and 91.8 %.)
  - Facts, the stored reads (`output/ocr/eval/*.jsonl`, `val_ep*.jsonl`, local) paired row by
    row on the norm 12 key, sign test; run-to-run spread unknown (no two runs of one recipe).
    `mainsub_1124_1923` → `mainsub_1127_0028`: val main 10,806 → 10,768 of 11,634 (+164 / −202,
    p 0.05); Manga109 test SFX 2,097 → 2,083 of 2,558 (+101 / −115, p 0.38), speech 2,357 →
    2,348 of 2,559 (+36 / −45, p 0.37); sincos SFX 351 → 312 of 593 (+28 / −67): the 497 rows
    with a heart 293 → 255, the 96 without 58 → 57; K3 KO +1 / −3, ZH +2 / −3.
    `mainsub_1126_2213` on the same rows: val main 10,810, Manga109 SFX 2,060, speech 2,348,
    sincos 320 (heart rows 263). Heart dropped or read as a glyph, of 497: 156 / 184 / 209
    (`1923` / `2213` / `0028`).
  - Facts, with every `ー` / `-` / `―` run scored as one mark: val main 11,065 / 11,065 /
    11,062, speech 2,450 / 2,458 / 2,454. The 425 val mains whose keyed target holds `-`
    (typed `―` 249, a `ー` run 169, `-` 7): exact 149 / 152 / 114, a read differing only by a
    lone `ー` there 238 / 236 / 276. `0028`'s val reads hold `-` 293 times and no `ーー`;
    the other two hold `ーー` 144 / 147 times and `-` twice.
  - Facts, labels: no val main or Manga109 test label changed in content between the two
    runs' evals (868 val targets differ by the folds only). Review exports after `review_1124k`
    edit 655 `swaprelax_v5` rows of 20,327 (383 punctuation or dots, which the key folds) and
    326 `mainsub_det` rows; no export edits a ZH training row. Claude's read of 36 sampled
    crops among the 398 val mains all three runs miss: about 10 are the label's (`あつ` for
    `あっ`, `カ` for `力`, `「」` for `『』`, `!` counts), about 4 a small-kana or `〰` call.
  - Interpretation: the val main gap is the lone-`ー` against run / dash choice the fold
    leaves in one token; the sincos loss sits on heart rows and is already in
    `mainsub_1126_2213`, so it is not the third epoch; Manga109 SFX, speech and K3 are inside
    the paired noise. Batch recipe and epochs are not separated from each other.
- `TARGET_NORM = 11` (2026-11-26): a `・` run is `…` per three dots, at least one, in the
  training target (`・・・・・` → `……`, the user; `.` runs and
  `…` as labelled; `core/textnorm.py` fold 6). Before it, `load_mainsub` main targets held
  2,343 train / 294 val labels with a `・` run against 21,818 / 2,146 with `…`. No run on it
  yet.
- Suspect subs (2026-11-27): the 903 flagged `OCR2:` rows that train, val included, fixed by
  agents, pass `subq1127` ([report](reports/mainsub_subq_agents_2026_11_27.md)); the ruby quarter
  no longer moves with the seed. Then all 5,138 unchecked ruby subs in the quarter, pass
  `subr1127` (45.5 % content wrong, [report](reports/mainsub_subr_agents_2026_11_27.md)). Not
  run: the ruby subs outside the quarter (they do not train), the 3,429 non-ruby unreviewed subs,
  a rule for furigana group spacing. No run on the new labels yet; the local HF build of
  `load_mainsub` predates them.
- Anima low-confidence reads (2026-11-27, the user): `load_mainsub` trains
  `assets/anima_mainsub.tsv`'s 326 applied rows (`ok` 168, `agent` 158): `OCR:` main 309, main
  `""` 17; `OCR2:` sub 36, sub `""` 290 (they go in with the reviewed negatives, ahead of the
  SFX-only and KO / ZH draws). No image shared with the K3 or doujin gates. No run on it yet.
- No-label boxes (`neg_other` 22,718, `drop` 2,678 train,
  [report](reports/hayai_first_pass_2026_11_24.md)): off the review page since 2026-11-27 (the
  user: the page lists only sources that train; none had been applied). Their crops stay cut
  under `crops_sfxp/det_nolabel/` (local).
- `hayai disagrees` after tier 1 ([report](reports/hayai_agent_tier1_2026_11_24.md)): tier 2
  (distance ≤ 0.4, dakuten / longer / shorter read, about 620) and tier 3 (same length, other
  glyphs, about 2,900) not run. `BEMADER_P_047_27` and `ToutaMairimasu_037_11` (furigana untyped)
  wait for the user.
- Manga109 test SFX rows v7 misses: 156 rows only `sfxpc_1121_0117` reads right against 108 the
  other way, and the 11 non-empty `OCR2:` answers on the 292 crops
  ([report](reports/mainsub_1124_0151.md)). Not read.
- Mains out of train `OCR:` until reviewed: 8,412 waiting on a line-break proposal (the user,
  2026-11-22: "b, c는 일단 학습에서 빠져있으니 괜찮을듯"), and 162 in `assets/m109_break_held.tsv`
  (local; [report](reports/mainsub_1124_0138.md)). On 11-24 agents fixed the top 1,559 rows of
  the page's `linebreak suspicious` sort, breaks and ruby subs
  ([report](reports/mainsub_lb_suspicious_agents_2026_11_24.md)); 6,833 `mainsub_det` rows are
  left on the sort, 41 of them held from the passes (their ruby unread).
- KO label errors that recur in the `swaprelax_v5` reviews (`review_1124b`, `review_1124c`): hearts
  missing (101 of 273 reviewed edits), y-vowel and look-alike jamo misreads, a vertical column
  split per glyph, `…` written for `..` / `...` / `……`. v7 reads no KO heart and splits 223 of the
  248 per-glyph columns as the labels do.
  - Hearts: the 891 rows where hayai v2.7 read a heart the label lacked were reviewed on 11-24
    on a page sort since removed (`review_1124g` / `h` / `i`: 866 hearts added, ♥ 481, ♡ 385; a
    few to the sub). Labels with a heart: 385 → 1,263 of 20,327.
  - Columns: 250 labels split per glyph joined into one line on 11-24 (51 by the user in
    `review_1124d`, 199 in bulk in `review_1124e`) and their spacing reviewed (`review_1124f`,
    which also put back 6 breaks that were printed rows).
- Label fixes not made:
  - 19 mains of 9–14 characters print a mark their label lacks
    (`output/ocr/linebreak/agent/speech9/missing_marks.tsv`, local).
  - 3 labels type quote marks as `゛`, which `normalize_target` folds onto the kana
    ([report](reports/mainsub_linebreak_agents_2026_11_22.md)).
  - The proposal's main-in-sub repeat check does not fold full-width digits
    (`HarukaRefrain_002_11`: main `８年`, sub `8年` / `ひっ`).
  - `AosugiruHaru_028_21`: the main `アハハハ` covers part of the SFX (a `キ…` column beside it);
    trains as proposed.
- Shelved (the user, 2026-11-23: furigana and line breaks first): finding the crops with text
  among the ~20k SFX-only `OCR2:` negatives ([report](reports/mainsub_sfx_agents_2026_11_23.md)).

## Reports

- [`mainsub_data_2026_11_21.md`](reports/mainsub_data_2026_11_21.md): the detector crops, the
  proposals, the line-break and furigana passes, the agent reviews.
- [`mainsub_1121_1806.md`](reports/mainsub_1121_1806.md): the first run (val main 86.4 %, sub
  49.2 %) and its v7 gates.
- [`mainsub_labels_2026_11_22.md`](reports/mainsub_labels_2026_11_22.md): the loader changes
  of 11-22 and the count of missing line breaks.
- [`mainsub_linebreak_agents_2026_11_22.md`](reports/mainsub_linebreak_agents_2026_11_22.md)
  and [`mainsub_linebreak_agents_9to14_2026_11_22.md`](reports/mainsub_linebreak_agents_9to14_2026_11_22.md):
  the agents' breaks on 15+ and 9–14 character speech.
- [`mainsub_1122_1446.md`](reports/mainsub_1122_1446.md): the sync-free training forward.
- [`mainsub_1123_1328.md`](reports/mainsub_1123_1328.md): `OCR2:` negatives from swap_relax and
  reviewed JA rows (main 85.4 %, sub 51.0 %, empty on the 292 JA crops 72.6 %).
- [`mainsub_sfx_agents_2026_11_23.md`](reports/mainsub_sfx_agents_2026_11_23.md): the SFX-only
  check (190 / 2 / 8 of 200) and the agent subs of the 636 SFX crops with speech partly inside.
- [`mainsub_1124_0138.md`](reports/mainsub_1124_0138.md): the 1k run on the loader of `74042c06`
  (main 85.4 %, sub 50.8 %, empty on the 292 JA crops 99.0 %).
- [`mainsub_1124_0151.md`](reports/mainsub_1124_0151.md): the 2-epoch run (whole val main
  91.8 %, sub 54.7 %; doujin SFX 334 / 593, Manga109 test SFX 80.8 %, speech 89.0 %, K3 KO 30,
  ZH 33) against `sfxpc_1121_0117` and v6; shipped as Hub `v7/` (`59768fc6`).
- [`mainsub_1124_1503.md`](reports/mainsub_1124_1503.md): token-mean against per-crop-mean loss
  at 1k on the labels of 11-24 (`--loss_mean`; Manga109 test SFX 65.4 % against 66.4 %, 107 / 133
  paired, p 0.11; speech 86.2 % against 86.4 %); no 2-epoch run on it.
- [`hayai_first_pass_2026_11_24.md`](reports/hayai_first_pass_2026_11_24.md): hayai v2.7 over
  all 164,090 detector boxes; 12,947 of the 122,853 main / sub rows differ from the main; the
  25,396 no-label rows on the page.
- [`hayai_agent_tier1_2026_11_24.md`](reports/hayai_agent_tier1_2026_11_24.md): the user's 70-row
  review of the hayai sort and 22 Sonnet agents over the next 861 rows (734 applied); trim gate
  0.5 → 0.3.
- [`mainsub_lb_suspicious_agents_2026_11_24.md`](reports/mainsub_lb_suspicious_agents_2026_11_24.md):
  20 Sonnet agents over the top 1,600 rows of the `linebreak suspicious` sort; 1,559 applied
  (539 with a sub), 41 held.
- [`mainsub_1124_1923.md`](reports/mainsub_1124_1923.md): v7's recipe on the 11-24 labels, 8 × 2,
  2 epochs on RunPod (val main 91.9 %, sub 54.9 %; vs the first v7: doujin SFX 334 → 350, Manga109
  test SFX 80.8 → 81.9 %, K3 KO 29 → 32, ZH 36); shipped as Hub `v7/` in place (`169eacbc`),
  `anime_tools` v0.15.0.
- [`mainsub_1127_0706.md`](reports/mainsub_1127_0706.md): LM full FT against LM LoRA r64 at 1k
  (Manga109 test SFX 1,677 against 1,709, p 0.05; doujin SFX 176 against 155, p 0.02, speech 109
  against 96, p 0.01; K3 level; 12.6 against 9.2 GB); the DoRA arms cancelled (peft DoRA ran an
  extra fp32 matmul per Linear: 7.3 crops/s).
- [`mainsub_subq_agents_2026_11_27.md`](reports/mainsub_subq_agents_2026_11_27.md): the training
  set holds next to no duplicate crops (37 identical, 784 tight / loose pairs); 903 flagged subs
  that train (val 79 included) through 16 Sonnet agents (ok 257, fixed 477, empty 142, unsure
  27; 4 main fixes), pass `subq1127`; the ruby quarter fixed across seeds.
- [`mainsub_subr_agents_2026_11_27.md`](reports/mainsub_subr_agents_2026_11_27.md): all 5,138
  unchecked ruby subs of the training quarter through 103 Sonnet agents (ok 2,394; content wrong
  2,340, 45.5 %; spacing only 369), pass `subr1127`.
- [`blank_tokens_2026_11_27.md`](reports/blank_tokens_2026_11_27.md): the reader's bf16
  gradient error by token (blank patches: 37 % of the block-0 gradient, 93 % of its error; 65 %
  off in bf16, 46 % with `fp32_lean`, the other tokens 13 % and 8.6 %); reads under noise on the
  blank patches and with the blank tokens replaced at the tower's output (exact level with
  clean); `mainsub_1127_1418` (no gradient through blank tokens) against `mainsub_1127_1419` at
  1k: Manga109 test SFX 1,681 against 1,675 (level), speech 2,389 against 2,406 (p 0.04). The
  probe arms and the trainer flag are not in the tree.
