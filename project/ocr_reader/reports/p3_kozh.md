# P3 — P2b + gelnote KO/ZH pseudo rows (`vl16_p2b_kozh`, 2026-10-11)

One arm, one seed. The P2b recipe (tower LoRA r64 / α128 + LM decoder full FT
lr 1e-5, embed / norm / lm_head frozen, 1 epoch, bs 8 × 2, seed 0) plus the P1
gelnote pseudo rows (KO 1 706 + ZH 2 132, `kind = sfx`). Speech is held at v3's
38 582 rows by `--speech_ratio 0.909525`. Train set: 81 002 rows (42 420 SFX
incl. 3 838 pseudo, 38 582 speech). 117 min, peak 12.2 GB.

```
ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 900 \
  project/ocr_reader/ocr/finetune_vl16_lora.py --run vl16_p2b_kozh --epochs 1 --bs 8 \
  --grad_accum 2 --lr 1e-4 --tower_rank 64 --lm_full --lm_lr 1e-5 --skip_stock_val \
  --extra_manifest pseudo_gelnote_ko --extra_manifest pseudo_gelnote_zh --speech_ratio 0.909525"
```

Checkpoint: `output/ocr/vl16_p2b_kozh/ep1/` (`adapter_model.safetensors` +
`lm.safetensors`). Hub: `sorryhyun/paddleocr-vl-1.6-manga-lora`, folder `v4/`.

## K3 — KO / ZH (48 + 48 scored rows; **Opus draft labels, user review owed**)

`ocr/eval_k3.py --compare stock v3 pl_kozh p2b p2b_kozh`, paired against stock.

| run | KO exact | KO spaced | KO kana leak | ZH exact | ZH kana leak | paired vs stock (KO / ZH) |
|---|---|---|---|---|---|---|
| stock VL-1.6 | 21 | 15 | 6 | 31 | 0 | — |
| v3 `vl16_b2_norm4` | 1 | 0 | 41 | 16 | 9 | 0/20 z −4.5 · 2/17 z −3.4 |
| `vl16_pl_kozh` (NC pseudo, can't ship) | 23 | 21 | 2 | 28 | 2 | 3/1 z +1.0 · 2/5 z −1.1 |
| P2b `vl16_p2b_r64_lmfull` | 7 | 6 | 36 | 19 | 10 | 0/14 z −3.7 · 3/15 z −2.8 |
| **P2b + KO/ZH** | **26** | 21 | 4 | **29** | 0 | 6/1 z +1.9 · 3/5 z −0.7 |

Against v3: KO 25 / 0 (z +5.0), ZH 16 / 3 (z +3.0).

- **The JA-only fine-tunes lost KO / ZH.** v3 reads 41 of 48 KO crops as kana
  and gets 1 right. P2b (same JA-only data) does the same. This is the first
  number behind the README's "KO / ZH got worse".
- **3 838 pseudo rows restore it to the stock level, not beyond it.** KO is
  +5 over stock (z +1.9), ZH −2 (z −0.7). Beating stock clearly needs more
  KO / ZH data.
- n = 48 per language puts the binomial SE near 7 points. Treat the paired
  counts as the evidence, not the totals.

## JA gates (paired against P2b, same seed and schedule)

| | v3 | P2b | **P2b + KO/ZH** | paired vs P2b |
|---|---|---|---|---|
| sincos SFX strict (n = 597) | 289 | 320 | 316 | 43 / 47, z −0.4 |
| sincos SFX ♡-blind | 336 | 379 | 379 | 45 / 45 |
| sincos speech ♡-blind (n = 272) | 151 | 145 | 147 | 16 / 14 |
| COO SFX (n = 2 558) | 2 147 (83.9 %) | 2 143 (83.8 %) | **2 174 (85.0 %)** | 120 / 89, z +2.1 |
| COO speech (n = 2 559) | 2 254 (88.1 %) | 2 271 (88.7 %) | 2 265 (88.5 %) | 38 / 44, z −0.7 |
| COO runaway SFX / speech | 20 / 11 | 17 / 10 | 19 / 14 | — |
| in-domain val SFX / speech | 86.8 / 91.1 % | 86.6 / 90.5 % | 86.0 / 90.6 % | — |

COO rows are rescored on the current `exact_key`. One COO SE is about 18 rows.

- **Adding the KO / ZH rows costs JA nothing.** sincos is flat, and COO
  speech sits inside 1 SE of P2b and above v3.
- The COO SFX +31 (z +2.1) is single-seed. The re-drawn speech subsample also
  changes the data order, so don't credit it to KO / ZH yet.

## P3 gate (roadmap)

| clause | result |
|---|---|
| K3 KO and ZH both above v3 | yes (z +5.0 / +3.0), on draft labels |
| sincos within 1 SE of v3 | yes (316 vs 289) |
| COO SFX / speech within 1 SE of v3 | yes (both above) |

Owed before the line can call this v4 final: the user's K3 label review
(re-score with `--compare`, no model re-run), a second seed, and more
licence-clean KO / ZH data to clear stock.
