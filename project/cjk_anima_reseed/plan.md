# plan — 225 new kanji rows (2026-10-06)

The user (10-06): cover the frequent kanji preview51 has no trained row for —
**joyo at ≥ 30 occurrences + jinmeiyo at ≥ 10** in Manga109-s — about 220
rows. Which recipe trains them, the scale line's `retrain_kanji_b*` or this
line's, is open (§ 3).

## 1. The rows

Coverage census (10-06): preview51's trained kanji are the 1 185 of
`anima_cjk_vocab_pack_preview51_trained.json`; frequency is every text of
the Manga109-s annotations (123 k texts, 281 k kanji occurrences, unfiltered
— `dialogue_pack.tsv` is filtered to the pack's charset and would hide the
gap); joyo / jinmeiyo from `scripts/distill_cjk/corpus/kanji_allow.py`.

| group | in the set | trained in preview51 | gap | share of kanji occurrences | covered |
|---|---|---|---|---|---|
| joyo | 2 138 | 1 113 | 1 025 | 98.4 % | 93.7 % |
| jinmeiyo (not joyo) | 814 | 47 | 767 | 1.2 % | 43.6 % |
| neither | — | 25 | 261 seen | 0.4 % | 28.7 % |

All told 92.8 % of kanji occurrences have a trained row; 12.4 % of the texts
hold at least one kanji without.

**The 225** (162 joyo ≥ 30, 63 jinmeiyo ≥ 10, by frequency) — coverage
92.8 → 96.7 %:

```
joyo:     剣頼薬瀬絵蔵帯攻戸鉄啓岡闘観騒捜辺労査絹仮弾値総権渉敷迎里織検示幻暁設港鶴菊字巻黙価援割呪荘争駄郷軽掘銃継幽亜幕獣恵区賊系済摩収害換囲斬費各倍訓象州虎聴努隣規防奪闇丁竜展率梨氷蛮免酔尽駆競洋藩砲駅祖豆杯飾米染挙障臓牙皇県涙統拓豪伯窓埋詠億狂肩迫審誌雄善凍響慎縁恩植威那旦臣嵐栄及斎瑠侵仏財邦刺粉薄史慶詳招岸稼営至鉛伏謎歴訪略
jinmeiyo: 喰児雛云嘩砦萩朋栗湘緋貰昴祐辻橘叶眞枇也癌鷹喋灼翔勿燈馴嶋篠紗柏筈竿淋智薩駿鳳揃帖厨丞兜莉冴駕彗柴棲杖兒槍詫弘儲蘇阿舵梢琵綺琶
```

- The scraped lists are slightly off: 児 is joyo (filed jinmeiyo here — it
  is in either way); 麦 (14) and 舎 are joyo and missing from the list (舎
  already has a row); 辨 / 瓣 sit in the joyo list. Re-check the 225 by eye
  before the vocabs file is written.
- 港 県 財 have rows in seed_fixed_1005 that are not in the trained list
  (with 筒 某: leftovers of an earlier line, untrained by f0). They are in the
  225 and train cold like the rest.
- The trained list holds 22 rows Manga109 never draws (牬 镬 悅 滿 樂 …). Left
  as they are.

## 2. The data the 225 can get

Measured 10-06:

- **The scale line's windows** come from `dialogue_2_10.tsv` (+ the
  training set's JA text). dialogue_2_10 is filtered to the old
  cjk_renderable charset: a median **2** lines per new kanji, ~80 of the 225
  with **none** (頼 帯 蔵 騒 呪 闇 謎 …). b1–b4 as built cannot draw them in
  words.
- **Manga109 lines of 8–14 cells** (the `sent` tiers), charset = the 1 185 +
  the 225: a median 7 lines per new kanji, p10 2; 70 under 5 lines. The new
  rows cannot live on `sent` lines alone.
- Windows (2–6 glyphs) from every Manga109 text on that charset: every
  candidate has ≥ 10 occurrences (the threshold), so ≥ 10 windows' worth of
  contexts each.
- **The ruler holds none of the 225** (0 of 96 strings): a read needs its
  own strings (§ 4).

`make_dialogue_pack.py` builds from preview51's trained list; a pack with
the 225 needs it re-run on the new charset (a new tsv, not an overwrite —
sent_kanji's data dir was built from the current one).

## 3. Which recipe

| | `retrain_kanji_b*` (scale line) | this line (`sent_kanji_f0`) |
|---|---|---|
| rows | cold, the batch only; every other row frozen at the context | warm, every row free |
| tiers | lone / grid (to 235 / 100 px for kanji) + bubble1 + bubbleN, the band law's three bands incl. 0.7–0.9 | bubble1 / bubbleN + `sent` 70 %, px ≤ 52, upper edge ≤ 0.8 |
| budget | 225 / 338 steps a row by ink, lr 1e-3 | 40 steps a row, lr 2e-4 |
| norm pull | on (the norm guard of a cold row) | off |
| what it is for | identity from nothing | dialogue fit on rows that already read |

**Read of the record:**

- **For cold rows the scale recipe has the evidence.** retrain_kana (scale
  table) and the reseed's cold tables (kana / kana_mix / kana_up / kana_big)
  trained the same kana cold at the same 135 steps / row and lr; every
  reseed table read under retrain_kana on the banner grid (words official
  33 vs 11–14 of 104) and none beat it on the dialogue ruler
  (`reports/ruler_2026_10_05.md` § 3–4). What the reseed tables lacked is the
  large glyph trained high (`_archive/reports/grid_64_2026_10_03.md`: ≥ 64 px
  0.8 % vs 19.7 %). Kanji need it more: more strokes, and `sent`'s 22–33 px
  is under where a dense kanji resolves.
- **The cold kanji budget was measured** (C3, `../cjk_anima_scale/_archive/plan_retrain.md`
  § 1): 225 / row gave new kanji identity and halved the seed's dense ones;
  450 brought them back; 225 / 338 by ink is the split of record.
- **The reseed recipe cannot make a cold row.** 40 steps at 2e-4 is a
  fine-tune: a cold row would get ~1 / 35 of the ink budget's lr × steps. And
  a cold row in a run with the norm pull off has no norm guard.
- **The reseed recipe is what moved dialogue** (F1, kanji recall: f0 vs
  preview51, `progress.md` § 4) — on rows that already had identity.
- **Not measured**: either recipe on kanji against the other. b1–b3 were
  never read alone; b4 was read only as the whole seed (`floor_score.md`).

**So: two stages, the scale recipe first.**

- **A — identity (`retrain_kanji_b5`, scale line).** The 225 cold, ink
  budget (≈ 63 k steps; b4's rate 76.7 k in 524 min → ≈ 7 h local), the
  scale table as b4 built it, context = `seed_fixed_1005_stick080`. Two
  changes before it runs: the windows' line file is a per-run key (a
  Manga109 file on the new charset in place of dialogue_2_10, which starves
  80 of the rows); and the read is § 4's.
- **B — dialogue (this line), every row free, the 225 taken out (user,
  10-08).** `sent_kanji_225`: `rows_from = "retrain_kanji_b5"` (A's merged
  rows: stick080 + the 225), the 1 185 + 163 kana + the 225 all in `rows`
  and trained (kana row_lr 0.12), so L_pres's page demand spreads over every
  row as in pres. sent_kanji_pres's recipe (free_residual 0, lr 2e-4; λ 10 ·
  L_pres at σ 0.8–0.9 every 2nd step); the step count set by the 225 at
  40 / row (≈ 9 k steps, ≈ 1.8 h at pres's 0.72 s / step — a config key or
  `--max_steps`, not 40 × every free row). Data: a `sent` build on the new
  charset's lines, items carrying ≥ 1 of the 225, the kana : kanji glyph
  share as `sent_kanji`'s build; `b5_held.tsv` out by 5-gram.
- **Then the 225 alone go onto `sent_kanji_pres`'s rows** (no training);
  its 1 185 + 163 stay as measured. Before the read, `probe_stick_move`'s
  read of the 225's mean against pres's kanji stick (pres's sits 11.5° off
  stick080's, |stick| 114 → 125): if it differs materially, decide then —
  the ready option adds pres's kanji Δstick (pres − start, its shared
  vector; one direction for both families, cos +0.89) to the 225, read
  beside the plain transplant.
- **Watch in B:** the 225 adapt to a context that moved ≈ 9 k steps from
  stick080, where pres's moved 54 k. B's own Δstick (kana / the 1 185)
  against pres's — size and cos — says how close that context is to the one
  the 225 land in.
- In place of the first B (f0's recipe over every row, 1 185 + 163 + 225
  free, ≈ 63 k steps): pres's rows keep what the ruler measured, B's other
  rows are trained for their gradient and dropped, and B costs a seventh.

**A's prep (10-08, done):** `configs/runs/retrain_kanji_b5.toml` on
context `seed_fixed_1005_stick080` (rows only; its singles = preview51's
trained list in its `data/vocabs.json`); the 225 regenerated from the rule
and matched to § 1's list, ink pinned (158 at 337.5 / row, 67 at 225 →
≈ 68.4 k steps); windows cut `$MANGA109S/derived/dialogue_pack_b5.tsv`
(`make_dialogue_pack.py --extra`, 89 390 lines) and hold out
`b5_held.tsv` — § 4's strings: 160 lines of 5–14 chars covering 211 of the
225, a line dropped when it would leave a new kanji under 60 % of its free
windows (median 311 → 268); 藩 鉛 緋 昴 眞 燈 柏 薩 帖 駕 彗 兒 杖 詫 are
read by the single ruler only. Pack: `anima_cjk_vocab_pack_punct`. Stick
scale: A trains at full length; the × 0.8 decision is after A.

**Decide before A:**

- **Stick scale.** preview51's kanji rows carry their family mean × 0.8; A
  trains the 225 cold beside them at full length. Either scale the 225's mean
  × 0.8 after A (as preview5 was made), or let B absorb it. B is warm and
  free, so its stick is trained; but B at 2e-4 moved the kanji stick by only
  ~11 %. (10-08: settled at the transplant — the 225's mean read against
  pres's kanji stick, B above.)
- **A micro arm first?** (`feedback_micro_arms`.) The top 36 of the 225
  (C3's size) both ways — scale table at 225 / 338, and this line's table
  with the grid / lone tiers on at the same budget and lr 1e-3 — read on
  § 4's strings: ~1.5 h an arm. It settles the open row of the table above
  for kanji, and the full A then runs the winner.

## 4. The read

The dialogue ruler has none of the 225 and its strings come from the training
set's captions. Two parts:

- **The ruler as is**, every arm against f0 and preview51: the 225 must not
  cost what f0 won (g_f1, g_r_kanji, en_match).
- **A new-kanji string set**: Manga109 lines (held out of A and B's windows
  and lines by 5-gram, as the ruler is) of 5–14 glyphs, each with ≥ 1 of the
  225, one per kanji where possible (~100); the ruler's prompts reused,
  each string on one. Scored as the ruler: `g_r` on the 225 (per kanji,
  hit if any read holds it), g_f1, and the page. The floor is f0 / preview51
  on the same strings, rendered once.

## Run

```bash
# 0. the vocabs file and the line file (CPU)
# A
ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack_punct .venv/bin/python project/cjk_anima_scale/scale.py retrain_kanji_b5 data
ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack_punct .venv/bin/python project/cjk_anima_scale/scale.py retrain_kanji_b5 train --submit --queue
# B
.venv/bin/python project/cjk_anima_reseed/run.py sent_kanji_225 data
make daemon-run ARGS="project/cjk_anima_reseed/run.py sent_kanji_225 train"
# the 225 onto sent_kanji_pres's rows (script to write), then the stick read
.venv/bin/python project/cjk_anima_reseed/probes/probe_stick_move.py
```
