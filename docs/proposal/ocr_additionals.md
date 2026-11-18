# OCR reader: additional proposals beside the two-prompt line

Status: 2026-11-18. Against the two-prompt reader ([`ocr_sfx_prefix.md`](ocr_sfx_prefix.md), Hub
`v7_preview/` = `sfxpc_1118_0412`). The 1k control is `sfxpc_1118_0031`.

Closed and removed from this list: the 4 px augmentation and fp32 gradient accumulation (in code
since 3c2401a6, first run in 1 below); memory without gradient checkpointing (`--ckpt_lm_only` +
`--compile`, 12.0 GB at 20 crops/s; 709cc1c4, 43ee0daa); the int8 ConvRot tower bench and FA2 at
the tower's head dim 72 (both measured in 709cc1c4); the smaller items.

## 1. TARGET_NORM 7 and the 1k on the 11-18 changes (`sfxpc_1118_2037`)

### Facts

- `sfxpc_1118_2037` against `sfxpc_1118_0031`: same seed (0), data recipe, 1,000 steps, bs 4 × 4,
  token budget 3,072, tower LoRA r 64, LM full FT. It differs in:
  - `TARGET_NORM 7`: dot runs are no longer folded in the training target (`...`, `・・・`, `…`
    trained as labelled; `‥` → `..`). Scoring keys still fold every dot run to one `…`.
  - the 4 px augmentation (`Augment(tight=True)`, no `pad_jitter`) and fp32 gradient accumulation
    (3c2401a6);
  - `--ckpt_lm_only --compile`, with B marked and S padded to a multiple of 8 (709cc1c4,
    43ee0daa);
  - 38 swaprelax_v5 labels edited (line breaks; KO review export `ko_review_1118a`).
- Evals (reports in `project/finished/ocr_reader/reports/`, rows in `output/ocr/eval/` (local)):

| eval | n | `0031` | `2037` |
|---|---|---|---|
| COO SFX exact, `SFX:` | 2,558 | 66.6 % | 67.0 % |
| COO speech exact, `OCR:` | 2,559 | 86.8 % | 86.2 % |
| COO speech runaway, `OCR:` | 2,559 | 11 | 94 |
| doujin SFX (sincos) strict / ♡-blind | 597 | 173 / 253 | 110 / 211 |
| doujin SFX preds holding a heart (GT: 484) | 597 | 183 | 102 |
| K3 KO / ZH exact | 48 / 48 | 23 / 29 | 26 / 27 |
| val SFX exact (in-training val) | | 74.8 % | 73.0 % |

- One run per arm; spread unknown.
- COO speech runaways at `2037`: 83 of 94 hold `・・・・` or longer, 33 end in six or more `・`;
  the longest runs to the 96-character decode cap. COO speech reads holding `・・・・` or longer: 0 at `0031`, 342 at `2037`;
  the GT holds 319. Manga109 speech labels spell a pause as `・・` 6,209 times (`manifest.parquet`).
- Doujin SFX rows `0031` read exactly and `2037` did not: 78 (15 the other way). Most lose the heart
  (`びく♡` → `びく`, `パンッ♡ パンッ♡` → `パンッ\nパンッ`). `textnorm.training_target` keeps the heart
  (`♥`) under norm 7.
- The user read the runaways as clearly norm 7.

### Interpretation

- With the fold gone, a Manga109 pause is a run of repeated `・` tokens instead of one `…`, and the
  reader loses where the run ends. The runaway rise lines up with that; the other changes do not
  touch dots.
- The heart loss is not explained. Five changes ride together; norm 7 does not touch hearts in the
  target, so the augmentation, the accumulation and the compile path stay candidates.
- Arms, 1k each against `2037`:
  - A: dot fold back (norm 6), or `・` runs only folded to `…` with `.` / `...` / `…` kept apart;
  - B: eager (no `--ckpt_lm_only --compile`).
  - If neither brings the hearts back, a switch for the old `pad_jitter` augmentation.

## 2. SFX share of the training signal

### Facts

| | v6 (`vl16_p2b_kozh_nl_swaprelax`), one epoch | v7 (`load_sfxp`), one epoch |
|---|---|---|
| SFX positive rows | 38,582 COO polygon crops | 21,332 `SFX:` detector crops |
| speech rows (with KO / ZH) | about 62.8 k | 63,015 |
| negatives | none | 15,753 `OCR:`, 5,332 `SFX:` |

Mean target length: SFX 3.5 characters, speech 13.5. The loss is a token mean over the batch.

### Interpretation

A crop's weight scales with its label length, so an SFX crop weighs about 0.3 of a speech crop.
With fewer SFX rows on top, the SFX share of the gradient is roughly half v6's (estimate: about
16 % → 9 %). COO SFX sits 7 points under v6 while COO speech has reached it. Arms:

- per-crop loss mean (every crop weighs the same; `target_loss` change);
- `SFX:` positives oversampled × 1.5–2, or the speech rows thinned.

## 3. Warm start from v6

v7 trains from stock for 13,286 steps. v6 already reads COO SFX at 85.2 % and doujin ♡-blind 423;
v7 has to add the prompt split and the `""` answers. Start from v6 merged (tower LoRA merged,
`lm.safetensors` loaded), then a fresh tower LoRA + LM full FT under both prompts, 1k–3k steps.

- Risk: v6 learnt `OCR:` = read everything, so the `OCR:` leak on SFX crops (8.4 % at `0412`) may
  hold longer.
- Code: `--init_adapter` refuses `--lm_full` / `--tower_rank`; a load path for v6's merged weights
  is needed.
