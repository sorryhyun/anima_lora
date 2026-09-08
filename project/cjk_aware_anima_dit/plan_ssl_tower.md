# plan_ssl_tower — label-free domain adaptation of the VL-1.6 vision tower (2026-09-08, rev. 1)

*Side line of [`plan_ocr.md`](plan_ocr.md), under the O2b reader (B′ =
`vl16_tower_lr1e-5`, 312–316 / 617 on the sincos gate). One question — **Q-S:
does adapting the vision tower on unlabelled in-domain text crops, with no
transcription, lift the doujin gate after the same COO SFT?** The LM, the
labels and the SFT recipe are not touched; only the tower's starting point is.*

## Why this line exists (findings § tower, 2026-09-08)

| fact | number / pointer |
|---|---|
| B (LoRA on the LM, tower frozen) → B′ (tower unfrozen, same COO crops) | sincos exact **106 → 304**, garbage misses (sim < 0.5) **185 → 43**, mid 130 → 110 — perception was the bottleneck, and *grey Manga109* supervision transferred to colour doujin |
| what B′ still misses (313 rows) | 160 near-misses (sim ≥ 0.8) of which **55 are ♡-only** and 61 ♡/〜/ー-only; 110 mid; 43 garbage. ♡ appears in 497 / 617 gate rows and 337 / 77k COO train rows → an **LM-side label gap**, not a tower problem — out of scope here (needs ♡-bearing labels: synthetic or pseudo) |
| context is not the lever | `reports/0908_context_margin_sweep.md`: wider crops / red-box marker / oracle page text never beat the 12 % crop on any reader (B′ 316 → 294 at pad 0.35, 45 at 1.5) — a PE-Core/PE-Spatial side channel is closed as a direction |
| the official recipe | ERNIEKit SFT (`paddleocr_vl_sft.md` @ release/v1.4) = *Full* FT at lr 5e-6 on ~30k labelled samples; offers nothing for the no-label case; the 16 GB card cannot hold fp32 AdamW states for 800 M params (bnb 8-bit has no cu132 binary) — full FT dropped 2026-09-08 |
| unlabelled corpus | `deepghs/AnimeText` **test** split on the volume (73,725 images, 8.4 GB parquet; CC-BY-NC-SA → research build). `ocr/animetext_crops.py` cuts the text boxes at 12 % pad: 7.6 boxes / image (block + line hierarchy), ≈ 560k crops for the split; the sample is mostly manga bubbles + hand-lettered SFX, closer to doujin than "anime screenshots" suggested |
| **pixel-SimMIM smoke** (`ssl_tower_simmim.py --smoke`, 30 steps, lr 2e-5, linear head → 588 pixels, L1) | mechanics fine (22.5 crops/s at bs 16, 8.6 GB); masked L1 2.15 → 0.37. **But** median relative ΔW 4e-4 (max 9e-3) collapsed the tower output: cosine to stock features **0.38**, feature norm **×0.22**, the stock LM reads `""` on every crop through it, and a 30-step SFT from it scores **0 %** where the stock-tower SFT smoke scores 48 %. Bisect: any 9-layer block of encoder weights alone does it (layers 9–17 → norm ×0.33); embeddings / post-LN are innocent |

Reading of the smoke: a pixel target on the *final* features pulls the whole
representation toward pixel space in one coordinated direction per Adam step —
tiny weights, huge features. Lowering the lr slows that, it does not change
the destination. The objective has to live in the space the projector reads.

## Decisions

1. **Target = the frozen stock tower's features, not pixels.** Masked-token
   features of the student are regressed to the stock tower's `last_hidden_state`
   on the *unmasked* crop (data2vec / BEiT-v2 with the base model as its own
   tokenizer). The target space is the projector's input by construction, so
   the SFT stage re-aligns from a nearby point instead of from a collapsed one.
   A feature-distillation term on the **unmasked** tokens (`--kd_unmasked`,
   default 0.1) anchors the rest. Pixel-SimMIM stays in the script as
   `--target pixel` for the record; it is not an arm.
2. **The read-through check is the Phase-0 gate, before any SFT.** After the
   SSL epoch, the stock LM must still read: cosine to stock features ≥ 0.9 on
   held-out crops and the 6-crop CPU probe non-empty (`probes`, below). A tower
   that fails this is discarded without spending the 1.5 h SFT.
3. **One epoch, then the same SFT.** 150k crops (a 20k-image draw of the test
   split; the full 560k is a follow-up if S2 lifts), **1 epoch** at lr 1e-5
   cosine, 5 % warmup, bs 32, mask 60 % of 2×2 blocks ≈ 2.5 h with the teacher
   forward. Then `finetune_vl16_lora.py --train_tower --init_tower <ep1> …`
   with the exact B′ flags (1 epoch, tower 1e-5, LoRA 1e-4 — ep1 was B′'s best;
   ep3 did not improve), ≈ 1.5 h. Total ≈ 4 h GPU + evals.
4. **Gate = the 617 sincos SFX rows, B′ re-run on the same day as control.**
   Call it a lift at **≥ +15 exact over B′** (the B′ variant spread
   `lr1e-5` / `col100` / `ep3` is 297–316, so ±10 is seed noise); report the
   miss-class split (garbage / mid / near) beside it — the hypothesis is that
   SSL moves the garbage + mid rows, not the ♡ rows.
5. **Kill rule.** If S2 ≤ B′ + 10 with the Phase-0 gate passed, the line closes:
   label-free tower adaptation is not a lever for this reader and the remaining
   headroom is on the label side (♡ / small-kana data). No lr / mask-ratio
   sweep before that — one arm, one verdict.

## Steps

| step | what | gate / output | cost |
|---|---|---|---|
| **S0** | `ssl_tower_simmim.py --target feat --smoke` (30 steps) + the bisect probe on its `ep1/tower.safetensors` | cosine ≥ 0.9, LM reads non-empty, SFT smoke ≈ 48 % (stock-tower smoke level) | 10 min |
| **S1** | `animetext_crops.py` on a 20k-image draw → 150k crops; SSL 1 epoch (`--run simmim_feat_e1`) | Phase-0 read-through gate on `ep1`; recon sheet + held-out feature loss in `history.jsonl` | 2.5 h |
| **S2** | `finetune_vl16_lora.py --train_tower --init_tower … --run vl16_tower_ssl` (B′ flags); `eval_sfx.py --reader vl16 --ckpt … ` + `eval_manga109.py` | sincos exact vs B′ (same-day control), COO test unchanged (≥ 80 %) | 1.5 h + 15 min |
| S3 (only if S2 lifts) | full 560k crops × 1 epoch; ep2 of SSL; the in-domain source (our doujin pages via `AnimeText_yolo`, sincos excluded) as a second corpus | same gate | 6–8 h |

## Probes

- `ocr/ssl_tower_simmim.py` — the trainer (`--target feat|pixel`).
- Read-through probe (the S0/S1 gate): load the base VL-1.6, swap in
  `tower.safetensors`, compare `visual(...).last_hidden_state` to stock on
  held-out crops (cosine, norm ratio) and greedy-read 6 COO val crops with the
  stock LM — the bisect script of 2026-09-08 (`findings.md` § tower); to be
  filed as `ocr/tower_readthrough.py` at S0.

## Out of scope

♡ / small-kana omissions (label-side), PE-Core/PE-Spatial context channels
(closed by the margin sweep), DINO / EMA-teacher objectives (a teacher we
already have for free is the frozen base), lr / ratio sweeps before the S2
verdict.
