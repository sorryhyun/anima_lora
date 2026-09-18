# synth_pair Δ1 — the full-inventory table under the paired loss (2026-09-18)

> `plan_synth2.md`'s Δ1: the 53 k seed run rebuilt with ΔFM (`--pair_loss 1
> --pair_ref en`, flat 0, uniform weights), 356 rows at ≈ 596 draws each.
> **The gate is missed on two of its three clauses and passed on the third.**
> Singles 12/36 against plain FM's 13/36 and the joint native 29/128 against
> 36/128 — but the tail is 15/128 against 24/128, and the whole joint loss is
> in the `en` clause (14 vs 24) while `swap` improves (15 vs 12). Under the
> sub-exact ruler Δ1 is *above* plain FM on the same multi-glyph groups
> (pooled lift +0.178 vs +0.098, P = 0.016). ΔFM bought the scene and did not
> buy the read: it is not a drop-in replacement for the seed table, and the
> `en`-clause collapse is a new question, not a budget shortfall.

## What ran

| stage | job | wall |
|---|---|---|
| `scenes s1w` (pool rebuild) | `20260917-234153-209e43` | 116.5 m |
| data + train + eval | `20260918-015823-5a6c3f` | 565.2 m |
| native | `20260918-112958-62cbff` | 9.0 m |

An earlier launch of the same argv, `20260918-013849-26448e`, was stopped at
17.2 m and carries no result.

Arm dir `output/wake_probe/rows_synth_d1_d1_s53k`, data
`output/wake_probe/data_synth_d1`. Argv (train job):

```
src/wake_probe.py --stage data train eval --arm rows
  --scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic
  --single_scenes s1,s1w --single_max_ar 2 --data_tag synth_d1
  --units kana --units kana_ext*1 --units kanji:200*1
  --units list:、,。,・,ー,～,〜,！,？,「,」,！！,・・・,・・・・*1
  --scene_mix single=1.0 --n_items 10000 --scene_frac 1.0 --natural_frac 0
  --strings_frac 0 --flat_bubble 1.0 --scene_fill 0.7 --scene_min_glyph 28
  --scene_max_lines 1 --scene_vertical 1 --pair_ref en --pair_ref_pool 4
  --shapes 448,512:2,448x512,512x448 --train_steps 53000 --batch 4
  --t_min 0.7 --t_max 0.9 --compile 1 --grad_ckpt 0 --aggressive_recompute 0
  --lr_rows 2e-3 --lr_decay cosine --free_residual 1e-3 --box_weight 4
  --seeds 2 --no_floor --c_flat 0 --pair_loss 1 --arm_tag d1_s53k
```

**Build.** 356 ext rows (92 kana + `kana_ext` + 200 corpus kanji + 13
punctuation), every source at weight 1. 10 000 composites over 651/1004
scenes, **0 flat items**, every item with a Latin sibling (2 220 reference
strings, 2 289 distinct reference captions). One-column, vertical, ≥ 28 px,
fill 0.7, quota cap 5 per glyph; 9 904 of 10 000 items are single-glyph.
`ja_comic` kept 178/292 scenes with one anchor bubble. 53 000 steps × batch 4
= 212 000 draws over 356 rows ≈ **596 draws/row** — inside the 848 (53 k over
250 rows) the plan priced, because the inventory grew to 356.

**Training was clean.** Loss 0.01677 → 0.01466, `fm_plain` flat at ≈ 0.099,
`pres` ≈ 6e-4 throughout, 1.69 it/s stable. `delta_norm_mean` 161 → 144 (25 k)
→ 79 (51 k) and `rel` 0.69 → 0.34: under cosine decay the table *contracts*
over the back half. Nothing diverged and no guard fired; the Δ0 guard on
`delta_norm_mean` per draw at 356 rows (a row is in ≈ 1.1 % of batches) held.

## The gate, as `plan_synth2.md` wrote it

> beaten on singles and on the **joint** native number with the tail under
> plain FM's

where joint = `both ∧ en cos ≥ 0.85` and tail = the count with en cos < 0.85,
both off `native_reads.json`. Baseline is `rows_synth_full_fm10k_full_s53k_qoff`
(plain FM, 53 k, 434 rows, 490 draws/row).

| | plain FM `src53k` | ΔFM `d1_s53k` | gate |
|---|---|---|---|
| `single` | 13/36 | **12/36** | ✗ |
| native joint | **36**/128 | 29/128 | ✗ |
| native tail | 24/128 | **15**/128 | ✓ |

`single_ext` 18 → 17, `single_kanji` 18 → 18, `en` 24/24 both — the EN side is
untouched, as on every arm in this line. (`plan_synth3.md`'s where-it-stands
table prints `src53k` ext as 20/36; the arm's own `report.md` and the Δ1 gate
in `plan_synth2.md` both say 18/36. 18 is used here.)

## Native, by clause

| clause | arm | both | joint | tail | en cos |
|---|---|---|---|---|---|
| en | plain | 36 | 24 | 16 | 0.882 |
| en | **ΔFM** | 22 | 14 | 11 | **0.915** |
| swap | plain | 18 | 12 | 8 | 0.932 |
| swap | **ΔFM** | **19** | **15** | **4** | **0.956** |

**The whole loss is the `en` clause.** `swap` improves on every column at
once — more hits, more joint, half the tail. `en` loses 14 both and 10 joint.
Averaged rulers hide this: `en cos` is *up* in both clauses and `box IoU`
rises 0.13 → 0.27 (en) and 0.32 → 0.48 (swap), so on the scene rulers ΔFM
wins everywhere. Reading the two together, "ΔFM is more conservative and
therefore writes less" does **not** survive — a conservative table would lose
both clauses, and `swap` gains.

Per glyph, `both` of 16 (`prev` = plain, `curr` = ΔFM):

| glyph | en | swap |
|---|---|---|
| あ | 11 → 9 | 7 → 7 |
| か | 10 → **4** | 0 → 2 |
| す | 10 → **3** | 1 → 3 |
| 日 | 5 → 6 | 10 → 7 |

か and す carry −13 of the −14 on `en`, and both of them *gain* on `swap`.
日 — the kanji the plan flagged for flat-0 drift (drawn as Latin "a" under
plain FM, and Δ0's 12 rows were all kana) — is the one glyph that does not
drop on `en`; it is also the lowest `en cos out` (0.866/0.905) and lowest box
IoU (0.18/0.40) of the four, i.e. it moves the scene most and sits least in
the EN word's box.

**Sheets.** `src/probe/cross_sheet.py` (written for this read, free — it reads
the `native_reads.json` both arms already wrote) puts the two arms on one
page, one row per arm per prompt, each row keeping the native sheet's own
`enref s0 / trained s0 / enref s1 / trained s1` layout. The `enref` cells come
from the shared `output/wake_probe/native_enref/512_28_4` cache, so both rows
carry the *same* reference image and each arm's drift from it is the vertical
comparison:

    python src/probe/cross_sheet.py \
      output/wake_probe/rows_synth_full_fm10k_full_s53k_qoff/native \
      output/wake_probe/rows_synth_d1_d1_s53k/native

→ `output/wake_probe/rows_synth_d1_d1_s53k/native/cross/x_<glyph>_<clause>.png`,
labels `p## <arm> s# HIT e<en cos>`. `x_か_en.png` and `x_す_en.png` are where
the −13 lives; `x_日_en.png` is the flat-0 drift check.

## Sub-exact: the ruler that orders them the other way

Exact match is floor-saturated on every multi-glyph group of both arms, so
`src/probe/sub_exact.py` over the three groups they share (n = 88 each):

| group | plain FM | **ΔFM** |
|---|---|---|
| `line` | +0.122 | **+0.234** |
| `combo` | +0.026 | **+0.149** |
| `corpus` | **+0.190** | +0.140 |
| pooled | +0.098 | **+0.178** |

Pooled lift difference 0.080, 95 % CI [0.006, 0.152], P = 0.016. (Pooling
*all* groups each arm has gives +0.067 vs +0.167, but the group sets differ —
Δ1's data carries no `word` rows and its `phrase_held` is n = 8 against 32 —
so the three-group comparison above is the one to quote.)

So the two rulers disagree by construction: ΔFM writes the target glyph
*less often exactly* and puts *more of the right glyphs* on the page. That is
the same finding `plan_synth3.md` built its S2 gate on, now measured on a
full-inventory table rather than a micro arm.

## The uniform-exposure read (what only Δ1 could give)

The 53 k plain run drew `kanji` and `kana_ext` at weight 2 against kana's 1,
so its kanji 18 vs kana 13 was never evidence about strokes. Δ1 forces every
source to `*1` at 596 draws/row:

| unit kind | `exact (sfx)` |
|---|---|
| kana | 12/36 |
| `kana_ext` (voiced / small) | 17/36 |
| kanji (corpus top-200) | 18/36 |

**Per draw, kanji are not harder than kana — they are easier**, and the
earlier 2× exposure was not what produced the gap. This is the number K's
weights were to be set from; it says the kanji budget does not need extra
draws per row relative to kana, and that plain kana is the weak kind.

## Owed / not done

- **`single_extra` is unreadable from this run.** The 13 punctuation rows are
  in the inventory (`extra units: 13 singles 、 。 ・ ー ～ 〜 ！ ？ 「 」 ！！ ・・・ ・・・・`)
  and 26 renders sit in `eval_reads.json`, but `single_extra` is missing from
  `EVAL_GROUPS` (`src/common/prompts.py`), which filters both the report table
  and the sheets (`src/eval/stage.py`). `src/data/stage.py` says to read the
  group off the sheets — the sheet is never written. One line plus a re-run of
  the eval stage.
- **K0** (`merge_tables.py` Δ1 ⊕ punct-only, then eval the merged table on
  both blocks) has not run.
- The `en`-clause collapse is unexplained. It is not a budget shortfall
  (`swap` gained at the same budget) and not general conservatism. The
  standing geometry — `c_flat` ⟂ the EN quoted-frame direction Q
  (`wake_rows_geometry`) — is the first place to look, but `--pair_ref en`
  means every sibling in this run carried the EN frame, which the Δ0 arms
  (12 rows) could not have exposed.
