# What the i2i LoRAs share: a reference copy head the LoRA builds

Scripts: `src/i2i_common.py` (weights, CPU; `output/tests/i2i_common/20261105-0905/`),
`src/copy_head.py` (attention + cuts, `output/tests/copy_head/20261105-run1/`),
`src/copy_bias.py` (logit bias on colorize's copy heads,
`output/tests/copy_bias/20261105-run1/`).

LoRAs: swap_text `…_v3_en400_ko1709_ksplit_rw2_mod16_ema5e5_int8_e2`, colorize
`moss_colorize_v5`, sanitize `moss_i2i_sanitize_idt_w0003_late_lr5e5_e8`, uncensor
`…_combined_muon_lr5e5_rw10to1_mod16` (weights only). Twins for a reproducibility
ceiling: swap_text en400 e2, colorize `3k_textref1_iso_muon_ema`, sanitize
`idt_mod16_notone`, uncensor `combined_muon_lr5e5_rw10`. t2i control:
`ama_mitsuki_muon_ema_fp32s` (+ `ama_mitsuki_muon_ema`). All rank 32, alpha 128, Muon,
seed 0.

## Claims

1. **Only the late blocks carry a reproducible delta.** Twins share 0.06–0.18 of their
   output subspace on blocks 0–25 and 0.61–0.83 on 36–41, t2i included: under Muon the
   early and middle ΔW is mostly run-specific.
2. **swap_text, sanitize and uncensor share a late-block component; colorize and t2i do
   not.** Full output-subspace overlap on blocks 36–41: 0.19–0.31 among the three,
   0.06–0.08 against colorize, 0.03–0.06 against t2i.
3. **Every i2i LoRA puts its largest per-head change on b39 h19, and turns it into a
   copy head that Base does not have.** 17–61 % of each i2i LoRA's v / o energy in
   block 39 sits on head 19 (uniform 4 %, t2i 1–4 %). With the LoRA, its image queries
   attend 0.85–0.99 to the reference token at their own cell; on Base 0.000–0.006.
   Within the head, q / k overlap across all four tasks; v / o overlap among the three
   local-edit tasks and not with colorize (route shared, copied value not).
4. **The copy gate is spatial and set at step 0.** swap_text: 0.98–0.99 outside the text
   boxes, 0.36–0.49 inside, the same at step 0 (pure noise) and over all steps.
   sanitize, with no boxes in its prompt, drops copy on the watermarks and text it
   removes, read from the reference. Only colorize relaxes it over time (0.55–0.65 →
   0.37–0.43 in the last third).
5. **Cutting the head degrades the copy but does not break it.** Cutting b39 h19's
   reference keys moves the render 1.8–8.7 luma L1 from the LoRA render, 3–8× the
   control cut on b39 h5 (reads the reference 0.67, copy 0.007). Cutting the top eight
   copy heads moves it 3.2–10.6; Base is 15–29 from the reference. The copy is spread
   over several heads.
6. **swap_text alone retunes Base's t2i string heads.** It puts 24–31 % of its q / k
   energy on b19 h6, b21 h20, b24 h11, b26 h12 (string_read.md's heads); the other i2i
   LoRAs 1–5 %. Weight evidence for string_read's open question: the LoRA reuses Base's
   string read rather than building one elsewhere (not causal).
7. **A negative bias on colorize's copy heads adds colour where the LoRA already puts
   some, at almost no cost in line alignment.** Bias −4 on the eight heads: chroma
   0.58 → 1.34, luma r 0.9962 → 0.9938 on one wild2 page; the other page stays grey at
   every bias. It removes a brake; it does not create colour intent.

## 1. Weights

**Shared init.** `down` starts at the same seed-0 kaiming draw in every run (cos to the
init 0.68–0.99 per module after training), so ΔW input subspaces align by
construction. `i2i_common.py` rebuilds the init (`LoRAAdapter` in LoRANetwork's order,
cos 0.974 against a trained `down`), compares `down − init` on the input side, and
reads cross-task agreement from the output side (up, zero init).

**Energy** follows the step count (‖ΔW‖ 24 / 41 / 56 / 85 for 660 / 2216 / 4218 / 10000
steps) and is flat over blocks for every task, ~52 % on `ffn_up_gate`: under Muon
energy says nothing about where a task acts. Top-8 singular share 0.36–0.43 (flat 0.25).

**Output-subspace overlap**, energy-weighted, full 32 directions (random 0.010), mean
over proj:

| pair | 0–7 | 8–18 | 19–25 | 26–35 | 36–41 |
|---|---|---|---|---|---|
| swap–sanit | 0.053 | 0.050 | 0.042 | 0.097 | 0.258 |
| swap–uncen | 0.053 | 0.053 | 0.045 | 0.084 | 0.194 |
| sanit–uncen | 0.095 | 0.079 | 0.072 | 0.148 | 0.312 |
| colorize–others | 0.041–0.054 | 0.038–0.050 | 0.035–0.048 | 0.036–0.055 | 0.057–0.080 |
| t2i–i2i | 0.032–0.039 | 0.035–0.039 | 0.032–0.040 | 0.027–0.047 | 0.031–0.060 |
| twins | 0.063–0.139 | 0.056–0.123 | 0.071–0.178 | 0.326–0.573 | 0.611–0.825 |

Cross-task overlap is larger on q / k than on v / o (swap–sanit 0.11 / 0.11 against
0.085 / 0.089).

**Per-head energy share** (uniform 0.042), q / k / v / o:

| head | swap | colorize | sanitize | uncensor | t2i |
|---|---|---|---|---|---|
| b39 h19 | 0.22 0.22 0.41 0.31 | 0.35 0.30 0.61 0.35 | 0.25 0.27 0.27 0.17 | 0.21 0.23 0.19 0.07 | 0.03 0.03 0.01 0.04 |
| b19 h6 | 0.31 0.23 0.16 0.05 | 0.05 0.06 0.07 0.04 | 0.03 0.03 0.06 0.04 | 0.09 0.08 0.09 0.04 | 0.27 0.23 0.14 0.04 |
| b21 h20 | 0.29 0.24 0.14 0.06 | 0.01 0.01 0.02 0.04 | 0.01 0.01 0.02 0.04 | 0.01 0.01 0.01 0.04 | 0.05 0.04 0.03 0.04 |
| b24 h11 | 0.27 0.24 0.16 0.05 | 0.04 0.03 0.05 0.04 | 0.03 0.02 0.04 0.04 | 0.04 0.04 0.05 0.04 | 0.08 0.06 0.07 0.04 |
| b26 h12 | 0.24 0.20 0.13 0.05 | 0.04 0.04 0.03 0.04 | 0.05 0.04 0.04 0.04 | 0.04 0.03 0.03 0.04 | 0.06 0.06 0.03 0.04 |

b19 h6 is also tuned by the t2i LoRA: a general caption-read head.

**Inside b39 h19** (128-dim head side of q / k / v, residual write directions of o;
both init-free; random 0.06 / 0.003):

| | q head side | k head side | v head side | o write dirs |
|---|---|---|---|---|
| swap / sanit / uncen pairs | 0.35–0.47 | 0.33–0.49 | 0.25–0.35 | 0.11–0.23 |
| colorize against them | 0.34–0.42 | 0.28–0.36 | 0.08–0.10 | 0.01 |
| twins | 0.75–0.91 | 0.73–0.89 | 0.64–0.96 | 0.67–0.91 |

**mod16** (`shared_block_modulation`): 63–74 % of its energy on the `shift_f` chunk in
every mod16 LoRA, t2i included, but the direction does not reproduce (swap twin cos
0.13): a generic shift, not an i2i component.

## 2. Attention: Base and LoRA on the same input

`copy_head.py`: per task two held-out inputs (swap_text `1483610_0007_en`,
`589883_0004_ko`; sanitize `hard_to_sanitize1`, `usual_case1`; colorize the first two of
`wild2`), the shipped `configs/moss/i2i/<task>/generate.toml`, seed 0. `base` is the
LoRA at multiplier 0 (its ref flags still apply). Cond branch, 1024 sampled image
queries; step 0 sees the same x_T, prompt and reference in both arms.

b39 h19, step 0 / first / middle / last third:

| | ref | copy (own cell) |
|---|---|---|
| swap_text in0 | 0.998 / 0.998 / 0.997 / 0.997 | 0.984 / 0.986 / 0.989 / 0.989 |
| swap_text in1 | 0.982 / 0.983 / 0.983 / 0.979 | 0.961 / 0.965 / 0.968 / 0.963 |
| sanitize in0 | 0.978 / 0.991 / 0.991 / 0.954 | 0.946 / 0.960 / 0.960 / 0.910 |
| sanitize in1 | 0.975 / 0.970 / 0.959 / 0.927 | 0.877 / 0.881 / 0.873 / 0.850 |
| colorize in0 | 0.608 / 0.649 / 0.652 / 0.435 | 0.554 / 0.596 / 0.597 / 0.368 |
| colorize in1 | 0.704 / 0.716 / 0.692 / 0.493 | 0.647 / 0.664 / 0.644 / 0.431 |
| Base, all six | 0.000–0.006 | 0.000–0.001 |

- On Base the head reads image and template keys only. Base's strongest own-cell
  readers sit early (b5 h3 0.22–0.33, b2 h7, b14 h15, b4 h0, b1 h7 at step 0).
- colorize spreads its copy over b34 h6, b35 h3, b37 h20, b39 h19 (0.55–0.71 each at
  step 0); the swap_text top eight (below) miss b34 h6.
- Map (`copy_head/…/h19_step0_maps.png`, all image queries at step 0): swap_text in-box
  0.488 / 0.364, out-box 0.987 / 0.980, and the same over all steps (0.468 / 0.368,
  0.991 / 0.984). Not every box drops; the low cells sit on some balloons. sanitize's
  low cells are the `@…` handle, `無断転載禁止`, the `X GreatoDoggo` mark, plus a
  speckle on flat background in in1. In the earlier swap_text dump
  (`swap_text_attn/20261029-2324`) the in-box queries' mass goes to the template
  (0.56).

## 3. Cuts

Image queries of the listed heads cannot attend to reference keys (cond and uncond, all
steps). `cut_top8` = the swap_text run's top own-cell heads: b39 h19, b41 h2, b40 h20,
b36 h3, b37 h20, b35 h3, b36 h21, b39 h11. Luma L1 (0–255) against the `lora` render /
against the reference, in0 | in1:

| arm | swap_text | sanitize | colorize |
|---|---|---|---|
| lora | 0 / 6.0 \| 0 / 3.2 | 0 / 7.3 \| 0 / 5.7 | 0 / 6.8 \| 0 / 9.4 |
| cut_h19 | 8.7 / 10.5 \| 2.8 / 4.8 | 3.9 / 8.6 \| 2.0 / 6.2 | 1.8 / 7.5 \| 3.3 / 10.2 |
| cut_ctrl (b39 h5) | 1.1 / 6.0 \| 0.6 / 3.3 | 0.7 / 7.3 \| 0.4 / 5.8 | 0.7 / 6.8 \| 1.5 / 9.1 |
| cut_top8 | 10.6 / 12.4 \| 4.5 / 6.7 | 5.5 / 9.7 \| 3.2 / 6.6 | 3.2 / 8.2 \| 5.7 / 12.8 |
| base | – / 29.3 \| – / 25.5 | – / 25.4 \| – / 15.5 | – / 16.6 \| – / 24.4 |

swap_text outside the text boxes reads within 0.5 of the whole-page numbers. At sheet
size every cut render looks like the LoRA render; swap_text in0 `cut_top8` letters
`大丈夫?` as `エ丈夫?` (one seed). colorize in1 `cut_top8` came out with more colour
(pink sofa, blue balloons), which led to section 4.

## 4. A logit bias on colorize's copy heads

`copy_bias.py`: colorize's top eight own-cell heads from section 2 (b35 h3, b34 h6,
b39 h19, b37 h20, b34 h9, b35 h13, b40 h20, b33 h20); image queries add `bias` to every
reference key's logit, cond and uncond, all steps. Same two wild2 pages and config.
Reads as `bench/moss_i2i_eval/colorize_wild.py`:

| bias | in0 luma r / chroma | in1 luma r / chroma |
|---|---|---|
| +1 | 0.9949 / 0.26 | 0.9964 / 0.56 |
| 0 | 0.9943 / 0.27 | 0.9962 / 0.58 |
| −1 | 0.9936 / 0.28 | 0.9959 / 0.77 |
| −2 | 0.9929 / 0.26 | 0.9952 / 1.04 |
| −4 | 0.9911 / 0.26 | 0.9938 / 1.34 |

On in1 the colour grows where v5 already tinted (sofa pink, hair orange, the middle
panel's figures pink); lines stay. in0 stays grey: the bias releases the reference's
hold, the colour intent comes from elsewhere (caption `character` colour items,
project_colorize_v5). +1 does nothing: the copy is saturated.

## Limits

- One seed, two inputs per task; attention mass is not influence; weight energy is not
  influence.
- No t2i or Base copy-head *cut* (Base has no copy there to cut).
- The cut and bias arms recompute the cut heads through SDPA, the rest through FA2.
- Chroma on wild2 stays low in absolute terms (1.34) and has no target: whether the
  added colour is right is untested.

## Open

- colorize holdout with targets (`colorize_eval.py` chroma error) at bias 0 / −2 / −4:
  decides whether the bias is worth a generate / ComfyUI knob.
- The bias combined with `character` colour items on wild2.
- Region bias for swap_text / sanitize: raise copy outside the edit region against
  tone drift (`docs/moss/methods/tone_loss.md`).
- Why b39 h19: whether Base's RoPE-frequency structure in that head makes a same-cell
  match cheapest (weights only).
