# sent_kanji: the kanji rows trained with the kana (2026-10-05 – 10-06)

From the user (10-05, after `sent_ball_2026_10_05.md` § 7): sent_whole's run
with preview51's 1 185 kanji rows free too. `configs/sent_kanji.toml`,
`_archive/configs/sent_kanji_f0.toml`.

- **The trainer's norm pull walked the rare kanji home** (§ 2). With μ 0
  every touched row carries `FREE_RESIDUAL` 1e-3 · ‖f‖²; under AdamW a row
  absent from the batch long enough for its v to cool takes that pull alone,
  ~lr toward 0 a step. sent_kanji's kanji ended at row cos 0.118, length
  0.069 — the rarest quartile at 0.008. The kana (in nearly every batch) did
  not see it.
- **With the pull off (`sent_kanji_f0`) the kanji rows hold** (cos 0.954,
  length 1.08, 1 185 / 1 185 nearest their start) and the target kanji show
  up on the page more often: in any read 41 vs preview51's 26 (strings more /
  less 18 / 5, p 0.011), vs sent_whole's 25 (18 / 7, p 0.043). No more
  kanji strings read whole (exact 0 vs 1).
- **The short kana strings fall as every warm pass's do**: exact on the 46
  kana-only strings 8 (preview51) / 5 (sent_whole) / 3 (f0); cer +0.036 vs
  preview51 (p 0.46), +0.075 vs sent_whole (p 0.25). Over all 96 the text
  ties both (cer +0.009 / +0.026, p ≥ 0.8); the page ties preview51.

## 1. The data

`lines = "m109_pack"`: `~/manga109s/derived/dialogue_pack.tsv`
(`make_dialogue_pack.py` there), every Manga109-s text on the pack's rows,
line breaks joined — 82 971 lines (dialogue_2_10: 42 974, its charset the
old cjk_renderable one, no 応 / 転 / 壊; ~1 000 of its lines are in no
current annotation XML). Sent lines (8–14 cells) 19 133 → 28 126; kanji in
≥ 1 line 1 048 → 1 130, in ≥ 10 lines 647 → 797. Windows 171 k → 1.30 M; 杷
has none (lone only).

sent_whole's table at 1 348 rows: 134 800 items (bubble1 5 %, windows 25 %,
dialogue 70 %), every tier filled, 41 min (10 workers). The kanji are 22.5 %
of the trained glyph occurrences. 37 GB img + 24 GB TE + 33 GB latents.

## 2. sent_kanji (the pull on)

53 920 steps (40 / row) × batch 4, lr 2e-4, the kana step × 0.12 (`row_lr`:
sent_whole's summed lr over 8.3× the steps), 391 min.

| family | row cos | \|end\| / \|start\| | nearest own start |
|---|---|---|---|
| kana 163 | 0.989 (0.961–0.995) | 1.005 | 160 / 163 |
| kanji 1 185 | 0.118 (−0.01–0.90) | 0.069 | 523 / 1 185 |

Kanji by occurrence quartile (44 / 76 / 127 / 323 in `train.jsonl`): row
cos 0.03 / 0.05 / 0.45 / 0.89, length 0.008 / 0.03 / 0.18 / 0.91. warm_cos
0.98 at step 2 000, 0.58 at 12 000, flat at 0.42 from 30 000.

Why: the pull's gradient per element is 2μ·f / N ≈ 5e-8 (f ≈ 0.034 raw,
N 1 348); AdamW steps m / (√v + ε). In the batch, the data gradient sits in
v and the pull is lost under it; absent, v decays 0.99 a step (√v halves in
~140), and once the data's trace is gone the pull alone steps ≈ 0.8 lr toward
0. A Q1 kanji is drawn every ~1 200 steps, a Q4 one every ~170. The scale
line's cold kanji runs trained ~300 rows over ~90 k steps (every row drawn
often), and from 0 the pull is the norm guard it was put in for.

Five ruler strings by eye (`ruler.py render --only`): 痙攣しとる → kana
only; 妹属性 → 腿の絆る惟; 櫻木真乃's name gone.

## 3. sent_kanji_f0 (the pull off)

`free_residual = 0` (new run key; `cjk_scale.train(free_residual=)`, the
constant stays the default), sent_kanji's data dir, rows, `row_lr`, lr and
seed — the batches pair step for step.

| family | row cos | \|end\| / \|start\| | stick | nearest own start |
|---|---|---|---|---|
| kana 163 | 0.990 (0.967–0.997) | 1.015 | 119.4 → 121.3, cos 0.992 | 163 / 163 |
| kanji 1 185 | 0.954 (0.909–0.978) | 1.081 | 114.2 → 126.8, cos 0.986 | 1 185 / 1 185 |

Kanji by quartile: row cos 0.976 / 0.961 / 0.946 / 0.918, length 1.05 /
1.07 / 1.09 / 1.13 — now the frequent rows move most. Kanji spike cos 0.942
(the kana's 0.989).

Loss, paired windows of 5 400 steps (f0 − sent_kanji): fm −0.0007 → −0.0020,
in_box −0.0023 → −0.0059 (5 %), out_box equal to the fourth digit. f0's
in_box passed sent_kanji's final window (0.1148) by step ~20 000.

## 4. The ruler

`ruler.py run --pack punct --arms seed_fixed_1005_stick080@punct,sent_whole,sent_kanji_f0`
→ `results/20261006-1454-ruler-sensitive-sent_kanji_f0/`.

| strings | arm | exact | contained | ≤ 2 edits | cer | target kanji in best / any read |
|---|---|---|---|---|---|---|
| kanji 50 | preview51 | 1 | 1 | 3 | 0.774 | 12 / 26 of 120 |
| | sent_whole | 1 | 1 | 3 | 0.777 | 15 / 25 |
| | sent_kanji_f0 | 0 | 0 | 2 | 0.757 | **26 / 41** |
| kana-only 46 | preview51 | 8 | 11 | 11 | 0.579 | |
| | sent_whole | 5 | 11 | 13 | 0.540 | |
| | sent_kanji_f0 | **3** | 6 | 11 | 0.616 | |

Paired, target kanji in the best read: f0 vs preview51 16 / 6 (p 0.052), vs
sent_whole 15 / 5 (p 0.041). cer on the kanji strings −0.017 vs preview51
(21 / 15, p 0.41). By bin vs sent_whole: short +0.034, mid +0.040, long
+0.003 (p ≥ 0.66). Page vs preview51: en_cls −0.000, en_match −0.0006,
en_tok_out −0.006 (p ≥ 0.48).

Closer, not whole: 田中さま → 田中ささま (preview51 十甲さ兎ま);
舌出してください → 舌出してくさい; 何度も何度も → 何度もも; 変わってくれ、
おっさん → 変わてさん. preview51's 8 short exact: f0 keeps はい… only
(sent_whole はい ほー んー フー さわって).

## 5. What the page draws (`score_page`)

`g_f1` is the headline text score from here on (user, 10-06: less junk
lettered and the string's glyphs right; `criteria.md`).

From the user (10-06): precision / recall over what is drawn, not exact /
≤ 2 edits. Every text box read (both readers, averaged); kana, ー and kanji as
a bag (order and box free): `g_p` = the string's letters among all drawn,
`g_r` = the string's letters drawn, `g_f1`, `g_r_kanji` / `g_r_kana`,
`drawn` = letters drawn. Regions: a box is on the string at half its letters
the string's; `a_p` = the on boxes' share of the text area; `iou_en` = the
text area's IoU with the EN ref's (layout only: the EN page letters its other
bubbles too). Re-read, no render → `results/20261006-1502-ruler-sensitive-glyph_f1/`.

| arm | g_p | g_r | g_f1 | g_r_kanji | drawn | a_p | en_match |
|---|---|---|---|---|---|---|---|
| preview51 | 0.170 | 0.635 | 0.241 | 0.193 | 45.6 | 0.227 | 0.436 |
| sent_whole | 0.193 | 0.647 | 0.260 | 0.167 | 39.8 | 0.266 | 0.437 |
| sent_kanji_f0 | 0.202 | 0.670 | 0.271 | 0.336 | 37.3 | 0.318 | 0.436 |

Paired, f0 vs preview51 (mean Δ, better / worse, p): g_p +0.032 (63 / 31,
0.0013), g_f1 +0.030 (60 / 34, 0.0095), g_r_kanji +0.143 (19 / 6, 0.015),
drawn −8.3 (0.011), a_p +0.091 (44 / 26, 0.041); g_r +0.035 (0.91),
en_match −0.001 (0.66). By bin: long g_p +0.082 (26 / 6, 0.0005), drawn
−19.9 (4 / 26, 6e-5), g_r −0.016 (0.2); mid g_r +0.123 (20 / 8, 0.036),
g_r_kanji 7 / 0 (0.016), a_p +0.129 (0.011); short nothing (drawn +2.2,
0.061).

- **Fewer letters, more of them right — the dialogue data's.** sent_whole
  vs preview51 shows the long-page part too (g_p +0.072, 0.02; drawn −17.5,
  2e-5) and loses short F1 (−0.018, 0.029). The long pages that "letter less"
  (`sent_ball_2026_10_05.md` § 7) letter less of what is not the string.
- **The kanji add kanji recall**: f0 vs sent_whole g_r_kanji +0.169 (18 / 6,
  0.023), the rest tied (g_f1 +0.011, 0.059).
- The bag credits a common kana read in junk text, alike for every arm.

## Open

- The short kana loss: f0's kana rows moved less than sent_whole's (0.990
  vs 0.981), so not their turn alone — the kana stick grew (119.4 → 121.3,
  sent_whole's shrank) and the kanji moved under the same lines. A run with
  the kana held at preview51 and the kanji free splits it.
- Recall on long does not move for any arm (g_r 0.56–0.59): the pages
  letter less junk, not more of the string.
