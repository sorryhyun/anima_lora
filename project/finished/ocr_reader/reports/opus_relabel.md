# Opus relabel of the low-confidence tail — hito + gelnote (2026-10-14)

Three rounds of Opus hand labels on low-logprob hito and gelnote crops, plus the confident half of v4's hito self-labels as a second manifest.

## Round 1

### Facts

Opus agents hand-labelled 1,200 crops: 1,000 from hito and 200 from gelnote. A
second Opus pass then checked every label it would keep against its own crop.
469 labels are usable. They are in `manifest_pseudo_opus_relabel.parquet`
(`kind = sfx`, `source = opus`); train with `--extra_manifest pseudo_opus_relabel`.
`hito/opus_manifest.py` rebuilds the manifest.

#### Which crops

- hito (1,000): the reads in `prelabel_v4lp.jsonl` with `n_tokens ≥ 4`, lowest
  logprob first (cut at −0.544; 738 KO-edition crops, 262 JA-edition). We did not
  cut by logprob alone. Its bottom 1,000 are 99 % 1–2-token reads (`…`, `!`, a
  single kana), and a 16-crop sample of them was 16 / 16 with no text: misregistered
  diff noise. The guards already drop those.
- gelnote (200): 100 KO + 100 ZH. They are drawn at random (seed 0) from the
  rows that the stock × hayai25 filter dropped and that have `n_tokens ≥ 4` and no
  runaway. The gelnote sweeps store no logprob, so disagreement stands in for low
  confidence.

The labelling agents never saw the reader's output. Each crop got one status
(`ok / no_text / multi / cut / illegible`) and a K3-style label: hearts as ♡, `…`
for exactly 3 dots, KO lines joined with a space. Every agent refused crops showing
sexual content with characters who look like minors. Those galleries are all
marked `excluded`, and the list lives outside the repo (`hito_ocr/FLAGGED.md`).

#### Result

| pool | n | usable | excluded | no_text | multi | cut | illegible | dropped by check |
|---|---|---|---|---|---|---|---|---|
| hito KO-edition | 738 | 289 | 285 | 53 | 33 | 53 | 23 | 2 |
| hito JA-edition | 262 | 7 | 172 | 59 | 7 | 11 | 6 | 0 |
| gelnote KO | 100 | 80 | 1 | 0 | 4 | 10 | 3 | 2 |
| gelnote ZH | 100 | 93 | 0 | 0 | 0 | 7 | 0 | 0 |

- hito KO = Korean hand-lettered SFX (출렁, 퓨얏, 띠옹 띠옹 띠옹, 꿈이에혀 …).
  v4's read matched the Opus label on 0 / 345 first-pass `ok` rows. v4 reads
  this lettering as katakana (`キイロゥー`, `ムイラロ`, `カイロッ`).
- hito JA: 7 / 262 are usable.
- gelnote: the stock read already matched on 54 / 175 first-pass `ok` rows.
- Check pass (473 `ok` rows outside excluded galleries): 454 match, 15 fixed
  (dot counts, a final consonant, a heart count), 4 dropped (text clipped or two
  balloons), 0 labelled from the wrong image. The wrong-image check was needed
  because labelling agents had shared one zoom folder.

Files: `hito_ocr/ko/prelabel_opus.jsonl` and `gelnote_crops/relabel_opus.jsonl`
hold all 1,200 rows. Each row has `status`, `lang`, `label` and `note`, plus the
old v4 / stock read.

### Interpretation

- Because the labelling agents never saw the reader's output, the old read could
  not anchor them.
- The hito KO rows are the rows v4 cannot produce on its own.
- The low tail on the hito JA side is almost all junk.
- On gelnote, agreement threw the 54 stock-correct rows away because the voter
  (hayai25) misread them, not the teacher.

## Round 2 — ranks 1 001–2 000 (2026-10-14)

### Facts

The next 1 000 hito reads by logprob (`n_tokens ≥ 4`, −0.544 … −0.268; 731 KO-edition,
269 JA-edition, 118 galleries). `hito/opus_batch.py` built the 15 batches (small crops
pre-enlarged into a per-batch `zoom/` folder, so no agent writes images), 15 Opus
labellers worked them, 6 Opus checkers verified every `ok` row, and
`opus_batch.py merge` wrote `ko/prelabel_opus_b2.jsonl`. Prompts: `hito/opus_prompts/{label,verify}.md`.

What changed from round 1: labellers transcribe every crop and carry a separate
`flag` field for pages the owner should review; a crop an agent will not read gets
`status = declined` on that row alone, and nothing is excluded at merge. Round 1 had
zeroed 457 / 1 000 rows by excluding whole galleries after a few refusals; 397 of this
round's crops sit in those same galleries and are now labelled.

| | n | usable | multi | cut | illegible | no_text | declined | dropped by check |
|---|---|---|---|---|---|---|---|---|
| hito KO-edition | 731 | 393 | 180 | 75 | 41 | 12 | 1 | 19 |
| hito JA-edition | 269 | 64 | 132 | 38 | 32 | 4 | 2 | 7 |

- 457 usable (419 `ok` + 38 `ok_fixed`); v4's read matched on 21 of them. `multi` is
  the big loss (312): the diff cut a whole panel or page with several balloons.
- Check pass (483 `ok` rows): 419 match, 38 fixed, 26 dropped (clipped text or two
  blocks), 0 wrong-image.
- Flags: 69 rows in 11 galleries, 7 of them already on round 1's list and 4 new
  (`hito_ocr/FLAGGED.md`, `ko/flagged_galleries_opus_b2.json`). 50 usable rows sit in
  galleries flagged this round and 101 in the union of both rounds' lists;
  `opus_manifest.py --drop_flagged` leaves them out, the default keeps them.

`manifest_pseudo_opus_relabel` now holds 926 rows (hito 753: KO 681 / JA 69 / mixed
3; gelnote 173). Rebuild: `hito/opus_manifest.py`.

### Interpretation

With no agent writing images, the round-1 wrong-image mix-up cannot recur.

## Round 3 — round 1's excluded crops + ranks 2 001–2 500 (2026-10-14)

### Facts

`opus_batch.py select --redo_excluded --n 957`: the 457 crops round 1 had zeroed (they are
the lowest logprobs of all, −2.03 … −0.54) plus the next 500 (−0.268 … −0.194). Same
prompts, 15 labellers, 4 checkers, `ko/prelabel_opus_b3.jsonl`.

| | n | usable | multi | cut | illegible | no_text | declined | dropped by check |
|---|---|---|---|---|---|---|---|---|
| redo of round 1's excluded | 457 | 61 | 183 | 37 | 25 | 150 | 0 | 1 |
| ranks 2 001–2 500 | 500 | 237 | 146 | 76 | 27 | 2 | 1 | 11 |

- 298 usable (273 `ok` + 25 `ok_fixed`; KO 224 / JA 69 / mixed 4 / en 1); v4 matched
  28. In the redo set a third is `no_text` (misregistered diff noise) and 40 % is
  whole-page `multi`.
- Check pass (310 `ok`): 273 match, 25 fixed, 12 dropped, 0 wrong-image.
- Flags: 115 rows in 16 galleries, 3 new (`1060384`, `2172149`, `2661490`). 125 usable
  rows sit in a gallery on any round's list.

`manifest_pseudo_opus_relabel` → 1 224 rows (hito 1 051: KO 905 / JA 138; gelnote 173);
`--drop_flagged` → 988. `manifest_pseudo_hito_v4` → 15 830 (the 2 500 hand-labelled uids
are out of its pool).

### Interpretation

The redo set was mostly junk, so round 1's gallery exclusions cost ~60 labels, not 457.

## Confident half of the v4 self-labels

### Facts

`hito/v4_manifest.py` → `manifest_pseudo_hito_v4` (15 830 rows: JA 8 926, KO 6 904).
Per side: `pseudo_label.guard` ok, script matches the side, not hand-labelled, logprob
at or above the side's median (JA ≥ −0.0057, KO ≥ −0.0086). K3 put the top half at
~85 % exact on gelnote KO and the sincos calibration at ~81 % for JA; on hito itself
this is unmeasured (the owed KO tier check), which is the accepted risk of using them
unchecked.

Training arm (P3 ep2 recipe + both manifests):

```
ANIMA_MANGA109S_ROOT=/media/sorryhyun/new/dataset/manga109s/Manga109s_released_2026_05_21 \
make daemon-run ARGS="--stall-timeout 900 \
  project/finished/ocr_reader/ocr/finetune_vl16_lora.py --run vl16_p2b_kozh_hito --epochs 2 --bs 8 \
  --grad_accum 2 --lr 1e-4 --tower_rank 64 --lm_full --lm_lr 1e-5 --skip_stock_val \
  --extra_manifest pseudo_gelnote_ko --extra_manifest pseudo_gelnote_zh \
  --extra_manifest pseudo_opus_relabel --extra_manifest pseudo_hito_v4 --speech_ratio 0.909525"
```

### Interpretation

The script filter drops KO reads without Hangul because those are v4's katakana
misreads of Korean lettering.
