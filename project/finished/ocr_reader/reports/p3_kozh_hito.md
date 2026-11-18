# P3 + hito pseudo rows (`vl16_p2b_kozh_hito`, 2026-10-15)

The P3 ×2 recipe with the two hito manifests appended, paired against Hub v4.

## Facts

### Setup

The P3 ×2 recipe ([`p3_kozh_ep2.md`](p3_kozh_ep2.md)) with the two hito
manifests from [`opus_relabel.md`](opus_relabel.md) appended:
`pseudo_opus_relabel` (1 224 rows, flagged kept) and `pseudo_hito_v4`
(15 830 confident v4 self-labels). Nothing else changed: same seed 0,
2 epochs, same `--speech_ratio`. 98 108 train rows (P3: 81 002), 12 262
steps.

```
ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 900 \
  project/finished/ocr_reader/ocr/finetune_vl16_lora.py --run vl16_p2b_kozh_hito --epochs 2 --bs 8 \
  --grad_accum 2 --lr 1e-4 --tower_rank 64 --lm_full --lm_lr 1e-5 --skip_stock_val \
  --extra_manifest pseudo_gelnote_ko --extra_manifest pseudo_gelnote_zh \
  --extra_manifest pseudo_opus_relabel --extra_manifest pseudo_hito_v4 --speech_ratio 0.909525"
```

Checkpoint: `output/ocr/vl16_p2b_kozh_hito/ep2/`. **Not a v4 replacement.**
Hub `v4/` stays `vl16_p2b_kozh_ep2`.

### In-domain val

| | P3 ×2 ep2 | this run ep2 |
|---|---|---|
| SFX exact | 87.9 % | 88.5 % |
| speech exact | 91.7 % | 91.4 % |
| runaway SFX / speech | 12 / 13 | 11 / 11 |

### Gates (paired against P3 ×2 ep2 = Hub v4)

| | v4 | hito ep2 | hito wins / losses | z |
|---|---|---|---|---|
| sincos SFX strict (n = 597) | 346 | 329 | 24 / 41 | −2.1 |
| sincos SFX ♡-blind | 387 | 383 | 32 / 36 | −0.5 |
| sincos speech strict (n = 272) | 123 | 134 | 24 / 13 | +1.8 |
| sincos speech ♡-blind | 151 | 160 | 19 / 10 | +1.7 |
| COO SFX (n = 2 558) | 2 223 | 2 210 | 80 / 93 | −1.0 |
| COO speech (n = 2 559) | 2 273 | 2 274 | 39 / 38 | +0.1 |
| COO runaway SFX / speech | 19 / 9 | 16 / 9 | — | — |
| K3 KO exact (n = 48) | 26 | 25 | vs stock 6 / 2, z +1.4 | — |
| K3 ZH exact (n = 48) | 30 | 31 | vs stock 5 / 5, z 0.0 | — |
| K3 KO kana leak | 3 | 0 | — | — |
| K3 KO / ZH cer | 0.190 / 0.143 | 0.201 / 0.132 | — | — |

COO is scored on the current `exact_key`. The prediction files are
`output/ocr/eval/{sfx_p2b_kozh_hito_597,vl16_p2b_kozh_hito_test,k3_p2b_kozh_hito}.jsonl`.

- Sincos SFX: ♡-blind is −4 (n.s.). In 12 of the 41 losses only the heart is
  wrong. Typical losses: `ぱん♥` → `ぱん`, `ブルン♥` → `ブルン★`, `びく♥` →
  `びくん`. P3 ep1 → ep2 had gained 59 wins, 21 of them heart-only.
- Heart rates: every hito row goes in as `kind = sfx`; `pseudo_hito_v4` 6.1 %,
  `pseudo_opus_relabel` 12.0 %, Manga109 train SFX 0.2 %, gelnote KO / ZH
  1.7 % / 0.1 %; sincos SFX labels 81 %.
- KO: the additions are 912 Opus rows plus 6 904 confident v4 KO self-labels.
  K3 KO exact 26 → 25; K3 KO kana leak 3 → 0. K3 KO is a mix (Opus notes: 19
  hand, 16 typeset, the rest outlined / brush / other).
- Sincos speech +11 (z +1.8). COO SFX z −1.0, COO speech z +0.1.
- One seed per side.

## Interpretation

- The sincos SFX strict loss is mostly the heart. This run gives back most of
  the heart gain that P3 ep1 → ep2 had made. The cause is open. A label-mix
  dilution explanation does not fit: the added rows carry more hearts than the
  base SFX pool.
- KO / ZH are still at stock. The KO additions did not move K3 KO exact; the
  kana leak going to 0 is the only KO change. The hand-lettered half of K3 KO
  is where hito KO SFX should have shown, and it did not.
- Sincos speech +11 is borderline on one seed. COO does not move on either kind.
