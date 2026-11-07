# P1: the copy / fill block C, micro arm

Date: 2026-11-07. Phase P1 of [proposal.md](../proposal.md), step 2 of [plan.md](../plan.md).

Result in one line: **400 steps on clean illustrations with random text-shaped holes give a
block that, under the `cut` gate on AnimeText boxes, removes the stamps and the watermark of
the sanitize eval pages on every seed, with no sanitize pair seen.** It builds the copy head
by step 200, earlier than sanitize's own pairs do.

## Setup

- **Gate in training** (`TrainRequest.ref_gate = "cut"`, new): the image queries inside the
  hole lose the reference keys inside the hole and read the rest. `model._attention` attends
  the hole rows a second time over a gathered key set (FA2, gradients, no mask);
  `tests/test_moss_ref_gate.py` checks it against a dense masked SDPA. Not the proposal's
  `diag`: under `diag` a hole query still reads the true content two cells away inside the
  hole, which teaches a shifted copy.
- **Holes** (`library/moss/training/holes.py`): another page's OCR text layout (1,564 layouts,
  19,524 boxes, the manga pages of colorize v5's cache), flipped and shifted at random, half
  the draws adding 1–3 random boxes. Median cover 14 % of the page; 355 of 400 steps had a
  hole (`gate_rate 0.9`), mean cover 0.146.
- **Data** (`project/lora_blocks/src/build_c_data.py`): identity pairs over colorize v5's
  1,667 colour illustration targets, the latent as its own `ref_scale 1` reference
  (symlinks). The empty prompt on every step (`caption_dropout 1`).
- **Run**: [`configs/c_fill_s400.toml`](../configs/c_fill_s400.toml), the sanitize_copy_s300
  recipe (muon, lr 5e-5, rank 32, int8 plan, EMA off), 400 steps, a save every 100. 2.36 s/step,
  16.4 min, peak 10.55 GB. `output/ckpt_moss_i2i/c_fill_s400[/…_sN].safetensors`.
- **Eval**: `project/moss_te_sae/src/ref_gate.py --part fill`
  (`output/tests/ref_gate/20261107-1911-fill`): s400 on `usual_case1`, `usual_case2`,
  `hard_to_sanitize1`, seeds 0–2, the sanitize generate settings on the empty prompt, the P0
  AnimeText masks; `ctrl` (C, no gate) against `cut`. Copy head:
  `copy_vs_step.py --extra sanitize=c100…c400` (`output/tests/copy_vs_step/20261107-1916`,
  `usual_case1`, seed 0, the sanitize prompt, no gate).

## Removal, against sanitize v3

v3 columns are P0's runs (`20261107-1735` / `-1751-sanitize`, same masks and seeds).

| page | v3 alone | v3 + diag1 | C + cut |
|---|---|---|---|
| usual_case1 (watermark, top left) | kept, 3 / 3 seeds | removed, 3 / 3 | **removed, 3 / 3** (white corner, as the page around it) |
| hard_to_sanitize1 (two stamps) | clean | clean; a small mark on seed 2 | **both removed, 3 / 3**; hair runs on under the top stamp, sweater ribs under the second |
| usual_case2 (faint URL, mostly not boxed) | clean | faint red mark on seed 1 | the two boxed chalkboard bits redrawn; the unboxed watermark stays, as the mask leaves it |

C without the gate copies the page, stamps and watermark included (identity is what it
trained on).

| page | arm | ΔL out | Δb out | L1 out | L1 out vs own ctrl |
|---|---|---|---|---|---|
| usual_case1 | v3 alone | −0.29 … −0.12 | −0.64 … −0.59 | 5.55–5.60 | — |
| | v3 + diag1 | −0.29 … −0.06 | −0.68 … −0.60 | 5.58–5.65 | 0.42–0.47 |
| | C | −0.79 … −0.63 | −0.25 … −0.18 | 5.90–6.16 | — |
| | C + cut | −0.76 … −0.58 | −0.25 … −0.20 | 5.86–6.14 | 0.37–0.43 |
| usual_case2 | v3 alone | −0.01 … +0.72 | −0.33 … −0.28 | 6.17–6.59 | — |
| | C + cut | +0.08 … +0.23 | −0.41 … −0.37 | 6.33–6.37 | 0.39–0.49 |
| hard_to_sanitize1 | v3 alone | −0.17 … −0.05 | +0.05 … +0.09 | 6.16–6.19 | — |
| | v3 + diag1 | −0.26 … −0.23 | +0.03 … +0.15 | 6.18–6.26 | 0.76–0.86 |
| | C + cut | −0.03 … +0.07 | −0.04 … +0.03 | 6.11–6.17 | 0.69–0.72 |

## Copy head, C's saves

| step | h19 copy s0 / late | h19 ref mass s0 | top-8 | heads > 0.1 | L1 vs ref | largest copy heads |
|---|---|---|---|---|---|---|
| 100 | 0.000 / 0.002 | 0.00 | 1.5 | 12 | 7.45 | b5 h3 .32, b8 h14 .21, b14 h15 .19 |
| 200 | 0.63 / 0.59 | 0.82 | 3.3 | 49 | 5.92 | b39 h19 .63, b37 h20 .47, b40 h17 .40 |
| 300 | 0.76 / 0.71 | 0.87 | 5.0 | 59 | 6.07 | b39 h19 .76, b36 h21 .70, b40 h20 .69 |
| 400 | 0.77 / 0.77 | 0.89 | 5.4 | 67 | 6.22 | b40 h20 .78, b39 h19 .77, b36 h21 .75 |
| sanitize micro, 200 / 300 | 0.42 / 0.68 | 0.54 / 0.82 | 2.2 / 4.2 | 31 / 44 | 7.8 / 6.3 | ([copy_vs_step.md](copy_vs_step.md)) |
| sanitize v3, 1108 / 2216 | 0.85 / 0.88 | 0.98 | 6.0 / 6.2 | 61 / 80 | 6.3 / 5.8 | |

## Claims

1. **C + cut is a sanitize model with no sanitize pairs.** On the two pages with boxed marks
   it removes them on every seed, and fills the hole from the picture around it (hair, ribbed
   knit, white ground). On `usual_case1` it does what v3 alone misses and v3 + `diag1` fixes.
   The removal comes from the gate; the block contributes copying outside and filling inside.
2. **Outside the hole C is close to v3, not at it.** L1 outside is at or below v3's on
   `hard_to_sanitize1` (6.11–6.17 against 6.16–6.19) and inside v3's seed spread on
   `usual_case2`, and 0.3–0.5 above it on `usual_case1` (5.86–6.14 against 5.55–5.60), where C
   darkens the page (ΔL −0.6 … −0.8 against v3's −0.1 … −0.3). C has no tone term and 18 %
   of v3's steps.
3. **Holes do not slow the copy; they speed it.** h19 is at 0.63 by step 200 against 0.42 on
   sanitize's own pairs, 59 copy heads at 300 against 44, 67 at 400 against v3's 61 at step
   1108. The phase change sits in the same window (after step 100, by 200).
4. P1's pass, as written: L1 outside at or below v3's at ≤ 1/4 of its steps holds on two pages
   of three; the fill reads as background on the pages with something to remove.

## What it means

- C is worth a file of its own: the copy is the same protocol every task builds, and the hole
  training adds a fill that `cut` can call where a mask says. Sanitize-by-mask is reached
  without the curated pairs.
- P2 changes shape. The zero-shot half is done here; what remains is whether a short
  fine-tune on 50 sanitize pairs from C (with `cut` on the pairs' diff region) brings the
  tone and the unboxed marks, against the same from kaiming.
- The darkening on `usual_case1` is the copy's tone, not the fill's: it is there without the
  gate too. More steps, or a tone term, are the levers; v3 needed one.

## Not tested

- One run, one seed of the hole draw; s100–s300 not rendered under `cut`.
- Held-out clean illustrations with random holes (how good the fill is where there is ground
  truth); text holes on B&W manga pages (C saw colour illustrations only).
- `cut` on the other tasks' LoRAs; C stacked under swap_text or a Hangul block.
- The sanitize prompt on C's copy-head read (C trained on the empty prompt); `cfg 3` on an
  empty prompt (cond and uncond differ only by the skip-22 block).
