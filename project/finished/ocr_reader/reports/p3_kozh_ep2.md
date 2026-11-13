# P3 × 2 epochs (`vl16_p2b_kozh_ep2`, 2026-10-12)

The P3 recipe ([`p3_kozh.md`](p3_kozh.md)) with `--epochs 2` and nothing else
changed: same 81 002 rows (42 420 SFX incl. 3 838 gelnote KO / ZH pseudo,
38 582 speech), same seed 0. The LR schedule is linear decay over the whole run,
so this is a fresh run, not P3 continued: its `ep1/` is a mid-schedule
checkpoint (lr still 5.2e-5), not comparable to P3's annealed ep1. Warmup is
3 % of the run, so 303 steps instead of 151. 10 124 steps, ~228 min of
training, peak 12.2 GB. One seed.

```
ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 900 \
  project/finished/ocr_reader/ocr/finetune_vl16_lora.py --run vl16_p2b_kozh_ep2 --epochs 2 --bs 8 \
  --grad_accum 2 --lr 1e-4 --tower_rank 64 --lm_full --lm_lr 1e-5 --skip_stock_val \
  --extra_manifest pseudo_gelnote_ko --extra_manifest pseudo_gelnote_zh --speech_ratio 0.909525"
```

Checkpoint: `output/ocr/vl16_p2b_kozh_ep2/ep2/`. Chosen to replace P3 ep1 as Hub `v4/`; the
anime_tools pin (`OCR_READER_REVISION`) still points at the 1-epoch commit `9fe3d41e`.

## Train loss

P3 ep1 was still falling when its LR reached 0 (0.17 → 0.14 → 0.12 over the
last 1 500 steps). That was the reason to try a second epoch. In this run the
loss drops from 0.168 to 0.116 at the epoch boundary. That step is the second
pass over the same rows, so the ep2 loss (0.084 at the end) says nothing about
the gates.

| in-domain val | P3 ep1 | this run ep1 (mid-schedule) | **this run ep2** |
|---|---|---|---|
| SFX exact | 86.0 % | 82.2 % | **87.9 %** |
| speech exact | 90.6 % | 89.0 % | **91.7 %** |
| runaway SFX / speech | 13 / 12 | 12 / 9 | 12 / 13 |

## Gates (paired against P3 ep1)

| | v3 | P3 ep1 | **ep2** | ep2 wins / losses vs P3 ep1 |
|---|---|---|---|---|
| sincos SFX strict (n = 597) | 289 | 316 | **346** | 59 / 29, z +3.2 |
| sincos SFX ♡-blind | 336 | 379 | 387 | 48 / 40, z +0.9 |
| sincos speech strict (n = 272) | 121 | 113 | 123 | 18 / 8, z +2.0 |
| sincos speech ♡-blind | 151 | 147 | 151 | 15 / 11 |
| COO SFX (n = 2 558) | 2 147 | 2 174 | **2 223 (86.9 %)** | 116 / 67, z +3.6 |
| COO speech (n = 2 559) | 2 254 | 2 265 | 2 273 (88.8 %) | 46 / 38, z +0.9 |
| COO runaway SFX / speech | 20 / 11 | 19 / 14 | 19 / 9 | — |
| K3 KO exact (n = 48) | 1 | 26 | 26 | vs stock 6 / 1, z +1.9 (same as ep1) |
| K3 ZH exact (n = 48) | 16 | 29 | 30 | vs stock 3 / 4, z −0.4 |
| K3 KO / ZH cer | 1.664 / 0.545 | 0.244 / 0.223 | 0.190 / 0.143 | — |

COO is rescored on the current `exact_key` (`rescore_eval.rescore`).

- **Most of the sincos strict gain is the heart.** Strict rises by 30 and
  ♡-blind by only 8 (n.s.). 21 of the 59 wins are lines that ep1 already read
  right apart from the heart: `へっ` → `へっ♥`, `ぱんぱん♥` → `ぱん♥ぱん♥`.
  The heart is part of the label, so the gain is real. The glyph reading
  itself barely moved.
- **This is not the B′ pattern.** In the finished line, ×3 epochs lifted COO
  and left sincos flat or worse. Here COO SFX rises (z +3.6) and so does
  sincos strict.
- **KO / ZH stay at the stock level.** Seeing the pseudo rows a second time
  cuts the cer on both languages, but exact does not move. Getting past stock
  still needs more KO / ZH data.
- One seed per side; the second-seed debt carries over from P3.
