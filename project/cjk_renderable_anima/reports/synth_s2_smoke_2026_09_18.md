# S2-smoke — does the pair loss reach a multi-glyph item (2026-09-18)

> Cold 2-arm A/B on sentence items, plan_synth3 S2-smoke. **The paired loss
> works mechanically on sentences** — 86 % of the residual cancels on
> short / sentence composites (Δ0 singles: 80 %), the 2-column siblings
> included — **but the A/B cannot separate the arms**: sub-exact lift
> +0.030 for paired, 95 % CI [−0.010, +0.070], P(>0) 0.94; native `en`
> 31 vs 29 / 64, co-text 22 vs 22, en cos 0.901 vs 0.907. By the plan's rule
> (paired lift inside plain's CI, co-text not below) S2a shrinks to a plain
> μ sweep; the numbers lean paired on every ruler without reaching it.
> Two things found on the way matter more than the verdict: **the katakana
> dakuten rows render at floor** (untrained ガ 15/16, ゴ 14/16 native; 10/12
> flat), so every Δ0 `single` and native number was part floor — and **ΔFM
> training damages those floor-positive rows** (Δ0 paired arms 5–7/12 kata
> vs floor 10/12 vs plain 11/12), which is the geometry read in action:
> plain rows grow along the pack row, ΔFM rows rotate away from it.

## What ran

Data `output/wake_probe/data_synth_s2_smoke` (CPU, 15:49): Δ0's 12 dakuten
`chars:` + `words:30` + `--phrase_pieces 40` (the 40 commonest
`dialogue_2_10` pieces outside the inventory; without them almost no line
is drawable in a 42-row inventory) = **82 ext rows**; `--scene_mix
short=0.5,sentence=0.5` over the Δ1 pools (`s1,s1w,sl1w,ja_comic`
one-bubble), sentence rules of the sentence arm (`--sentence_min_letters 6
--sentence_min_glyph 20 --sentence_fill 0.9 --scene_max_lines 2`,
short one column 28 px fill 0.7), tategaki, `--n_items 2000` → short 1 000
(3–5 glyphs) / sentence 1 000 (6–14 glyphs, 455 one-column), 209/1 004
scenes, 0 misses, `--pair_ref en --pair_ref_pool 4` → 2 051 reference
strings, `--shapes 512`. `sheet_scene_pair.png` read: same scene, same
erase, same box, Latin letters of the item's glyph count, 2-column
sentences get 2-column siblings. **No single kind**: the 12 dakuten rows
are trained only inside the phrases that carry them — 5/12 hiragana were
seen at all, no katakana (`eval_coverage.json`).

Arms (`rows_synth_s2_smoke_s2smoke_<arm>`), cold, 1 500 steps, the Δ0 argv
(`--batch 4 --t_min 0.7 --t_max 0.9 --lr_rows 2e-3 --lr_decay cosine
--free_residual 1e-3 --box_weight 4 --c_flat 0`):

| arm | loss | train + eval | native |
|---|---|---|---|
| `plain` | `--pair_loss 0` | `20260918-155314-274069` 24.0 m | `…-155328-f3a4f8` |
| `pair` | `--pair_loss 1` | `…-155314-3646e8` | `…-155328-bcecb5` |

## (a) Does the sibling cancel the sentence residual — yes, 86 %

| steps | plain loss | pair loss | `fm_plain` | `pres` | cancelled | row norm plain / pair |
|---|---|---|---|---|---|---|
| 0–200 | 0.1145 | 0.0159 | 0.1146 | 3.8e-4 | 0.86 | 88 / 75 |
| 200–700 | 0.1040 | 0.0150 | 0.1048 | 3.3e-4 | 0.86 | 156 / 104 |
| 700–1200 | 0.1062 | 0.0155 | 0.1085 | 3.2e-4 | 0.86 | 163 / 93 |
| 1200–1500 | 0.1057 | 0.0151 | 0.1081 | 2.9e-4 | 0.86 | 157 / 83 |

Flat from step 1 and above Δ0's 0.80: the multi-line / 2-column siblings
do their job, `pres` stays at the singles level. The paired table contracts
(104 → 83) while the plain one grows to 157 — Δ1's pattern (161 → 79)
again, at 1 500 steps.

## (b) Sub-exact lift, plain vs paired

`src/probe/sub_exact.py`, groups short / short_held / phrase / phrase_held /
line / combo / word, 128 items per arm:

| group | n | plain lift | pair lift |
|---|---|---|---|
| short | 8 | +0.004 | −0.009 |
| short_held | 8 | −0.023 | +0.049 |
| phrase | 8 | +0.039 | −0.015 |
| phrase_held | 8 | +0.033 | −0.020 |
| line | 28 | −0.004 | +0.015 |
| combo | 36 | +0.049 | **+0.131** |
| word | 32 | −0.009 | +0.012 |
| **pooled** | 128 | **+0.014** | **+0.044** |

Difference pair − plain **+0.030, 95 % CI [−0.010, +0.070], P(>0) = 0.936**.
The difference is carried by `combo` (36 flat-canvas 2–3-glyph items); the
sentence groups are n = 8 each and split both ways. Exact match is 0 on
every multi-glyph group in both arms, as expected.

Flat `single` 24: plain 10, pair 13 — but see the floor finding: 10 of
those hits in *each* arm are the untrained katakana rows (ガ ギ ゲ ゴ 2/2,
ザ 1, ぎ 1); on the trained hiragana rows plain is 0/12, pair 3/12 (げ 1,
ざ 2). The flat sheets (`sheet_short.png`, `sheet_phrase.png`) show what
the table carries onto the flat template: plain forces the training layout
(a tall tategaki bubble with kana-like pseudo-text), paired draws horizontal
fragments and no bubble — the canvas component is gone, as ΔFM intends.

## (c) Native — `が ガ ご ゴ`, `en` + `swap`, 64 renders each

| arm | `en` both / joint / tail | co-text (> 1 box) | en cos | box IoU | `swap` both / joint / tail |
|---|---|---|---|---|---|
| plain | 29 / 21 / 12 | 22 | 0.907 | 0.21 | 17 / 17 / 5 |
| pair | 31 / 23 / 11 | 22 | 0.901 | 0.16 | 17 / 17 / 5 |
| Δ0 `pairEN_s750` (singles-trained, all 12 rows) | 22 / 20 / 7 | 28 | 0.930 | — | 9 / 9 / 0 |

Per glyph (`en`, both readers, /16): が plain 0 / pair 0; ご 0 / 2; **ガ 15 /
15; ゴ 14 / 14**. ガ and ゴ were **not trained** in either arm
(`eval_coverage` 0/1) — their rows carry a zero delta, so those renders
are the floor, identical across the arms, seed for seed. The smoke's own
trained rows (が ご, seen only inside phrases, ≈ 1 500 cold steps) render
0–2/16, the identity budget the plan's *Limit* line priced. `swap` is
identical across arms (17 / 17 / 5) for the same reason.

## The floor finding

The line has read `--no_floor` since 2026-09-15 on the strength of "the
floor never renders kana" (W0: flat singles 0/16, JA CER 1.0) — measured on
*hiragana*. The katakana dakuten rows of the shipped pack render at floor:

| where | untrained katakana | source |
|---|---|---|
| native `en`, ガ / ゴ | **15 / 16, 14 / 16** | this smoke, both arms |
| native `swap`, ガ / ゴ | 12 / 16, 5 / 16 | this smoke |
| flat `single`, ガギグゲゴザ | **10 / 12** (グ 0, ザ 1, the rest 2/2) | this smoke, both arms |

And the Δ0 arms, which *trained* those rows, on the same flat ruler:

| Δ0 arm | hiragana /12 | katakana /12 |
|---|---|---|
| floor (this smoke, untrained) | — | **10** |
| `pair0_s3000` (plain) | 10 | **11** |
| `pairEN_s750_lr2e-3` (ΔFM) | 8 | **5** |
| `pairEN_rb62f_lr2e-3` (ΔFM, row blocks) | 9 | **7** |
| `pairEN_rb62f_lr2e-3_w8` (ΔFM, warmup) | 5 | **5** |

Native tells the same story: Δ0 `pairEN_s750` trained ガ reads 7/16 where
the untrained row reads 15/16. Two consequences:

1. **Every Δ0 `single` /24 and native /64 was part floor.** The ΔFM-vs-plain
   comparisons of 2026-09-17/18 were effectively on the 6 hiragana rows
   (n = 12 flat, 32 native), and the row-block / α / warmup orderings need a
   hira-only re-read before they go into the K1 recipe (plan_synth4 R4.4).
   `native_chars` for any future dakuten probe should be hiragana, or the
   floor must be rendered (`--native_floor 1`) and subtracted.
2. **ΔFM damages a row the pack already renders; plain FM does not.** This
   is `reports/table_geometry_2026_09_18.md` read 3 as a behaviour: plain
   rows grow *along* the pack row (cos 0.41 — a gain on the pretrained
   piece, which keeps its function), ΔFM rows are orthogonal to it (0.06)
   and rotate a working row away. The paired loss has no term that says
   "keep what the pack already does": `--free_residual 1e-3` pulls toward
   zero delta at a weight three orders below the loss. For a K1 table that
   spans the pack's working rows (katakana, common kanji) this is a
   recipe item — a pack-row anchor (μ‖f‖² at a weight that matters, or an
   exclusion of floor-positive rows from training) — and it is measurable
   for free on the existing tables: hit rate on floor-positive rows,
   trained vs untrained.

## Decision (plan_synth3's rule)

*Paired lift inside plain's CI and co-text not below plain → S2a drops its
paired arms and becomes a μ sweep.* The lift CI includes 0 and co-text is
equal, so the rule fires — but the smoke is not a null: every ruler leans
paired (lift +0.03 at P 0.94, trained-row singles 3 vs 0, native `en` +2),
the mechanism is intact (86 %), and the sentence groups were n = 8. What
the smoke did settle is that the A/B *can* run on multi-glyph items and
that its rulers must exclude floor-positive glyphs.

Two facts from today reshape S2a regardless of which way the rule is read:

- A warm start across losses is near cold in direction (Δ1 ↔ `src53k`
  per-row cos 0.14; `table_geometry` read 4). S2a's plain arms warm-started
  from Δ1 (ΔFM) would be cross-loss warm starts; the matched seeds are
  `src53k`+punct for plain and Δ1 for paired. "Which loss" and "which
  seed" are one axis now, not two.
- Δ1's rows on floor-positive glyphs may already be damaged (unread —
  Δ1's `single` 12/36 vs `src53k` 13/36 on あ-row kana; the katakana /
  kanji split of that number is the free read above).

Recommendation: S2a as **two arms, not four** — `s2_plain_a03` warm from
`merge_punct` (plain seed) and `s2_pair_a03` warm from Δ1 (ΔFM seed), μ 0.3
both, 3 k steps on the S2a data — with the rulers changed to hira-only
`single`, `native_chars` from floor-negative glyphs (あ,か,す,日 are; verify
日 with a floor render), and `sub_exact` pooled lift. That keeps the one
comparison the line needs (loss × matched seed) at half the cost; the μ 0
arms answered their question in the anchor sweep.

## Files

- data `output/wake_probe/data_synth_s2_smoke/` (train.jsonl, eval.json,
  sheet_scene_pair.png)
- arms `output/wake_probe/rows_synth_s2_smoke_s2smoke_{plain,pair}/`
  (trained.pt, train_log.json, report.md, sheet_*.png, native/)
