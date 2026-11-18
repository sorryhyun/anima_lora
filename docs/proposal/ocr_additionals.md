# OCR reader: additional proposals beside the two-prompt line

Status: open list, 2026-11-18. Each item stands alone against the two-prompt reader
([`ocr_sfx_prefix.md`](ocr_sfx_prefix.md), Hub `v7_preview/` = `sfxpc_1118_0412`). No item has run.
The 1k control for a micro arm is `sfxpc_1118_0031` (same data and recipe, old augmentation, bf16
accumulation).

## 1. Applied in code, not yet run

### Facts

- **Augmentation on the 4 px crops.** `augment.pad_jitter` re-pads from an assumed 12 % pad
  (`base=0.12`), but the `--sfx_prefix` crops carry min(0.12 · long, 0.3 · short, 4 px). Simulated
  on the 105,432 train rows of `load_sfxp("train")` (one draw each, real w / h): 9.7 % of the crops
  are cut past the margin into the glyphs (median 3 px, p90 10 px beyond it) with the full label
  kept; 52 % are extended with flat border colour by a median 7 px. Every metric group is hit alike
  (9.6–10.9 %). Rotation (±8°, fixed canvas, border replicate) also pushes corners out of a 4 px
  margin (not measured).
- Change: `Augment(tight=True)` skips `pad_jitter` and rotates on a canvas expanded to the rotated
  bounding box (border colour). `CropDataset(tight=)` passes it; `finetune_vl16_lora.py` sets it
  with `--sfx_prefix`. On 400 train crops the long side grows by p50 1.000, p99 1.105, max 1.125
  (rotation only). Colour, contrast, invert, JPEG and scale draws are unchanged. The v6 path and
  `finetune_ovis3_lora.py` are unchanged.
- fp32 gradient accumulation. With `--lm_full` the 255 M LM decoder parameters summed their
  `--grad_accum` micro-step gradients in bf16 and were cast to fp32 at the step. Now a
  `register_post_accumulate_grad_hook` per parameter folds each bf16 gradient into the fp32
  master's `.grad` as backward produces it and frees it. CPU check, 4 micro-steps of differing
  scale: the hook path equals the fp32 sum (max error 0); a bf16 sum is off by up to 1.65e-2
  relative. The LoRA adapters are fp32 already (peft 0.20 `autocast_adapter_dtype=True`).
- Memory cost of the hook: the resident accumulator is 1.0 GB fp32 instead of 0.5 GB bf16 through
  micro-steps 2–4; the 1.5 GB transient at the optimizer step is gone. `0412` peaked at 14.3–14.6 GB
  without `--grad_ckpt` against ~15.2 GB free beside Xorg, and at 7.0 GB with it.

### Interpretation

- Of the ep1 errors against v6 on COO SFX (`sfxp_1117_1816`: 107 trailing extras, 14 cut short),
  some may come from clipped crops trained on full labels. Not measured.
- A 1k arm against `0031` reads both changes together; one 1k arm each separates them.
- Without `--grad_ckpt`, measure the peak first (`--largest_first --steps 30`).

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
- `SFX:` positives oversampled × 1.5–2, or the speech rows thinned (which also shortens the epoch).

## 3. Warm start from v6

v7 trains from stock for 13,286 steps. v6 already reads COO SFX at 85.2 % and doujin ♡-blind 423;
v7 has to add the prompt split and the `""` answers. Start from v6 merged (tower LoRA merged,
`lm.safetensors` loaded), then a fresh tower LoRA + LM full FT under both prompts, 1k–3k steps.

- Risk: v6 learnt `OCR:` = read everything, so the `OCR:` leak on SFX crops (8.4 % at `0412`) may
  hold longer.
- Code: `--init_adapter` refuses `--lm_full` / `--tower_rank`; a load path for v6's merged weights
  is needed.

## 4. Memory without gradient checkpointing

`0412` needed `--grad_ckpt` (12.0 crops/s, 298 min) where the 1k runs ran without it
(17.8 crops/s). The proposal reads the peak as long-label batches making the LM side dear, which
`--largest_first` (vision cost) does not reach.

- Count label tokens in the token budget, so the rare long-label batch is cut.
- Checkpoint only the batches above a cost threshold (HF modules toggle per step).
- torchao `AdamW8bit` for the LM masters' optimizer state (no bitsandbytes on cu132; about
  1.5 GB less).

## 5. int8 ConvRot on the tower

### Facts

- Linear MACs per crop, estimated from `models/paddleocr_vl_1.6/config.json`: tower 27 layers,
  hidden 1152, MLP 4304, about 650 patch tokens → about 267 GMAC; LM 18 layers, hidden 1024, MLP
  3072, about 180 tokens (merged image tokens + text) → about 46 GMAC. The tower holds about 85 %.
  Tower attention: about 26 GMAC.
- The LM's K (1024, 2048, 3072) divides by 256, but the LM is trained in full; `library/int8`
  converts frozen base linears only (the base weight gradient is never computed).
- The tower is frozen under a LoRA (r 64), the pattern `library/int8` serves on MOSS (training
  1.59 → 1.05 s/step against bf16, `docs/moss/optimization/int8.md`).
- comfy_kitchen (installed copy): `_build_hadamard` takes only powers of 4 (16, 64, 256); group
  128 raises. The CUDA `int8_linear` uses its fused rotate + quantize kernels only at group 256
  (`_fused_convrot_ok`, `_should_use_convrot_fused_kernel`). Other groups take the staged path:
  a separate Hadamard matmul over the activation (`_rotate_activation`), then
  `quantize_int8_rowwise`, then the same int8 GEMM. `input_act` fusion is also group-256 only.
- The 11-13 reward-reader microbench (`project/cjk_rl/reports/reward_reader_2026_11_13.md`)
  measured decode shapes only (M = 4: bf16 14 µs, ConvRot W8A8 72 µs).

| tower linear | K | group | path |
|---|---|---|---|
| q / k / v / out | 1152 | 64 (18 × 64) | staged |
| fc1 | 1152 | 64 | staged |
| fc2 | 4304 → 4352 | 256 (17 × 256) | fused |

fc2's K reaches 4352 when fc1's output gets 48 zero rows: gelu(0) = 0, so fc2 receives the padded
input with no copy per call. In `full` mode fc1's backward (K = N = 4352) is also group 256.
`library/int8/{convrot,linear}.py` hard-code `GROUP = 256` and would take a group argument.

### Interpretation

- fc2 (the fused path, about a third of the tower's linear MACs) is the safe first piece. The
  staged group-64 linears pay an extra activation round trip on a short K (1152); whether they
  beat bf16 is open.
- Group 64 spreads outliers over fewer columns than 256; small strokes (dakuten, handakuten) are
  the worry. GPTQ calibration as on MOSS, and the reader then runs on the same int8 tower at
  inference.
- Tower prefill at inference (M in the hundreds per crop) has not been measured; the 11-13 verdict
  covers decode only.
- First step: a GEMM microbench at the tower's shapes (M ≈ 2,600 for 4 crops) — bf16, group-64
  staged, fc2 group-256 fused. A few minutes on the card.

## 6. FlashAttention head dim for the tower

### Facts

- The tower's head dim is 72 (1152 / 16 heads). The repo's `fa2-sm120` build has head dims 64
  and 128 only (`docs/moss/optimization/fa2_sm120.md`), so `finetune_vl16_lora.py --attn
  flash_attention_2` needs stock flash-attn (its `--attn` help). Training runs on SDPA.
- The LM's head dim is 128 with GQA (16 query heads, 2 KV heads). The build's fp16-accumulate
  backward needs `num_heads_k == num_heads`, so the LM would take the stock backward.
- The `sfxp_1117_1816` notes: bs 8 without checkpointing ran out of memory at 6k tokens per batch
  with SDPA and with flash-attn 2 alike.

### Interpretation

- Stock FA2 runs a head dim of 72 on its 96 kernel (rounded up to a multiple of 32); a build for the
  tower adds the hdim-96 instantiation. To check against the build's dispatch.
- Tower attention is about 9 % of the tower's MACs by the estimate in 5, so the gain is bounded
  unless SDPA falls back to a masked or math kernel on the packed NaViT batch. Profile which SDPA
  backend the tower takes before building.

## 7. Smaller items

- Epoch 2 sampled by epoch-1 per-row loss: speech is near v6 already.
- One image prefix for both prompts: the task text follows the image in the template, so the
  tower and the LM's image positions are the same under `OCR:` and `SFX:`. At inference every box is
  read twice; sharing the image KV cuts most of the second read. In training it applies only to boxes
  that carry both rows.
- LM EMA or an average of the last checkpoints, kept on the host (the card has no room for it).
