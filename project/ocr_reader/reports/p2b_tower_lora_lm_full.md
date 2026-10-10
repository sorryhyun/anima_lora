# P2b — tower LoRA r64 + LM decoder full FT vs v3 (2026-10-10)

One arm, one seed. Same data and schedule as v3 (`vl16_b2_norm4`, Hub v3):
the `manifest.parquet` train split (38 582 SFX + 38 582 speech, grey only),
1 epoch, bs 8 × 2, seed 0, `--skip_stock_val`. Only the parametrization
changes.

| | v3 | P2b `vl16_p2b_r64_lmfull` |
|---|---|---|
| tower + projector | full FT, fp32 master, lr 1e-5 | LoRA r64 / α 128 on 164 Linears (35.7 M), lr 1e-4 |
| LM | LoRA r16 / α 32 on 126 projections (6.0 M), lr 1e-4 | 18 decoder layers full FT (254.8 M), fp32 master, lr 1e-5 |
| embed_tokens / final norm / lm_head | frozen | frozen |
| peak VRAM | 12.1 GB | **7.0 GB** |
| throughput | — | 12.1 crops/s, about 106 min / epoch |

Run:

```
ANIMA_MANGA109S_ROOT=~/manga109s/Manga109s_released_2026_05_21 make daemon-run ARGS="--stall-timeout 900 \
  project/finished/cjk_aware_anima_dit/ocr/finetune_vl16_lora.py --run vl16_p2b_r64_lmfull \
  --epochs 1 --bs 8 --grad_accum 2 --lr 1e-4 --tower_rank 64 --lm_full --lm_lr 1e-5 --skip_stock_val"
```

Checkpoint: `output/ocr/vl16_p2b_r64_lmfull/ep1/`
(`adapter_model.safetensors` + `lm.safetensors`; `Vl16Reader` merges the
adapter and then loads `lm.safetensors`).

## Results

| | v3 | P2b | paired (P2b wins / losses, McNemar z) |
|---|---|---|---|
| **sincos SFX strict** (n = 597) | 289 | **320** (+31) | 66 / 35, z = 3.1 |
| **sincos SFX ♡-blind** | 336 | **379** (+43) | 76 / 33, z = 4.1 |
| sincos speech ♡-blind (n = 272) | 151 | 145 | 11 / 17, z = −1.1 |
| sincos user-checked SFX strict (n = 86) | 43 | 50 | — |
| COO SFX (n = 2 558) | 2 147 (83.9 %) | 2 143 (83.8 %) | — |
| COO speech (n = 2 559) | 2 254 (88.1 %) | 2 271 (88.7 %) | — |
| COO runaway SFX / speech | 20 / 11 | 17 / 10 | — |
| in-domain val SFX / speech exact | 86.8 % / 91.1 % | 86.6 % / 90.5 % | — |

COO rows are rescored on the current `exact_key` (`rescore_eval.rescore`).
The sincos rows come from `eval_sfx.py` (strict = `exact`, ♡-blind =
`exact_noheart`).

- **The doujin gate moves; in-domain does not.** In the finished line, five
  arms in a row moved COO and left sincos flat. This arm does the opposite:
  COO is tied and sincos rises, with a paired z above 3. Single-seed caveat
  below.
- **The wins are mostly hearted / doubled SFX.** v3 garbles the stroke or
  drops the heart, P2b reads both: `びくっ♡` (v3 `ぐくっ♥`), `びく♡` (v3
  `じく♥`), `ぬぎ♡ ぬぎ♡` (v3 `ぬぎぐぬぎ`), `じゅぽ じゅぽ` (v3 `ぐゅぽ`).
  That fits the finished line's verdict that the headroom is on the label /
  LM side.
- **No sign the LM full FT hurt the language prior.** COO speech is +17
  (inside 1 SE) and in-domain val speech is −0.6 pt. Sincos speech is −6 and
  not significant.

## Basis caveat — sincos is 597, not 617

Eight of the 163 label pages are missing from
`post_image_dataset/resized/sincos/`. The originals are still under
`retrieved/sincos/`, but label boxes are in resized-page coordinates, so the
originals cannot stand in. The directory mtime is 2026-10-10. Both models
were re-scored on the 155 pages that remain, using a filtered label file
(975 → 933 rows; scored SFX 617 → 597). **Compare these numbers only with
each other, never with the `/ 617` rows in `finished/…/eval.md`.**

Cross-check: v3 on 597 gives ♡-blind 336 / strict 289. Its recorded 617-row
figures are 354 / 305, i.e. 59.3 % / 49.4 % then against 56.3 % / 48.4 % now.
The drop is consistent with losing 20 rows, so the remaining pages do not
appear to have been re-cut. Prediction files:
`output/ocr/eval/sfx_{v3,p2b}_present.jsonl`,
`output/ocr/eval/vl16_p2b_r64_lmfull_test.jsonl`.

## Not yet known

- **Seed.** One seed per side. A second P2b seed is owed before this counts.
- **Attribution.** Two changes landed together, so the gain could come from
  the tower LoRA, the LM full FT, or both. The cheapest split is a tower LoRA
  r64 + LM LoRA r16 arm (same memory class as P2b).
- **KO / ZH.** Untested; the data is JA-only, like v3.

## Environment note

`pandas` is not in the default dependency groups (only the opt-in `sr`
group), so `pandas==2.3.3` was installed into `.venv` by hand. The next `uv
sync` removes it. Don't let uv resolve it unpinned: it picked
`3.1.0rc0` once.
