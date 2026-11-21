# LPIPS through the VAE window on an Anima LoRA: ama mitsuki, two ratios (2026-11-21)

`lpctl_1121_1900` / `lpr03_1121_1900` / `lpr10_1121_1900` and their renders `lpab_1121_1900`: step
2 after `lpips_grad_2026_11_21.md`. The user's blind read: ratio 0.3 ahead of the control 11 – 5,
ratio 1.0 behind it 5 – 11, one training seed. Design: `plan.md`, "Pixel losses through the head,
into the DiT".

## Facts

- The user (2026-11-21): "그래 차라리 그게 낫겠다. ama mitsuki 로 해보자"; on the read: "아니 블라인드
  레포로 올리자구".
- Term (`library/training/pixel_lpips.py`): per train sample at σ ≤ 0.7, one random window of
  24 + 2 × 4 tokens; LPIPS-VGG of the bf16 VAE decode of x_t − σ·v against the same decode of x0,
  margins left out; its gradient into v scaled per sample to `ratio` × ‖∂L_FM/∂v‖.
- Arms: `configs/methods/lora.toml` as of this date (rank 32, alpha 128, LR 5e-5, GA 2, 8 epochs),
  ama mitsuki (106 images, 424 updates), seed 42, `paired_step_rng`, `gradient_checkpointing`;
  control = the term measured and not added, `lpr03` = ratio 0.3, `lpr10` = ratio 1.0. Configs
  `configs/lp*_1121_1900.toml`. One training seed.
- Memory and time: at the default preset the first VAE decode ran out of memory with 14.34 GB
  allocated (smoke job 20261121-185603), so every arm runs gradient checkpointing; each run took
  12.5 min, the control included (it runs the same VAE window).
- The term was live on 81 % of micro-batches (smoke, 26 / 32). Mean λ at the logged steps: 0.016
  at ratio 0.3, 0.058 at ratio 1.0.
- Median ‖(alpha / rank)·up @ down‖ of the final EMA files: control 0.4893, `lpr03` 0.4918,
  `lpr10` 0.4925 (`lpab_1121_1900`).
- Paired training log, `pixel_lpips/lpips` at the logged steps where the term was live in both
  arms at the same σ (167 of 212; same image, σ, noise and window by `paired_step_rng`). Mean, arm −
  control, standard error of the paired difference, steps where the arm is lower:

  | steps | control | `lpr03` − control | `lpr10` − control |
  |---|---|---|---|
  | epochs 1 – 2 (n 43) | 0.1595 | −0.0007 ± 0.0006, 24 / 43 | −0.0031 ± 0.0009, 36 / 43 |
  | epochs 3 – 6 (n 86) | 0.1529 | −0.0003 ± 0.0004, 45 / 86 | −0.0026 ± 0.0004, 68 / 86 |
  | epochs 7 – 8 (n 38) | 0.1534 | −0.0004 ± 0.0008, 22 / 38 | −0.0043 ± 0.0007, 35 / 38 |

  `lpr10` in the second half by σ: −0.0024 at σ ≤ 0.2 (n 10, control 0.065), −0.0038 at 0.2 – 0.4
  (n 28, control 0.116), −0.0032 at 0.4 – 0.7 (n 45, control 0.195). These are training images.
- Renders (`bench/pixel_readout/lora_ab.py --label lpab_1121_1900`, results (local)
  `bench/pixel_readout/results/20261121-1936-lpab_1121_1900/`): the lora_recipe prompts p0 – p3 ×
  render seeds 42 – 45 at the `make test` recipe; labelled pair sheets in `labelled/<arm>/`; blind
  sets `s45_LPIPS03_ama` and `s45_LPIPS10_ama` (16 pairs each) pushed to `anima-blind-pairs`
  (bf4ee36), keys in `output/blind/<set>/key.json` (local).
- The user's blind verdicts (2026-11-21, given in chat, pairs p01 – p16 in order):
  `s45_LPIPS03_ama` "abababaaaabababb", `s45_LPIPS10_ama` "babaaaaaabbabbbb"; written into the
  sets' `verdicts.tsv`. Scored against the keys, control – arm:

  | set | p0 classroom | p1 hug | p2 maid | p3 beach | total |
  |---|---|---|---|---|---|
  | `lpr03` (ratio 0.3) | 1 – 3 | 3 – 1 | 1 – 3 | 0 – 4 | 5 – 11 |
  | `lpr10` (ratio 1.0) | 2 – 2 | 4 – 0 | 4 – 0 | 1 – 3 | 11 – 5 |

  By render seed 42 / 43 / 44 / 45: `lpr03` 2 – 2, 1 – 3, 0 – 4, 2 – 2; `lpr10` 3 – 1, 2 – 2,
  2 – 2, 4 – 0. Binomial over 16 pairs, two-sided: 0.21 for each set. Each set is one (artist,
  training seed) cluster, so the lora_recipe sign-flip test has nothing to flip. Side A was picked
  9 / 16 and 8 / 16 times; the arm sat on side A in 6 and 7 pairs.
- The two sets hold the same 16 (prompt, render seed) cells and the same control renders. Per cell:
  the arm picked in both sets 5, the control in both 5, `lpr03` picked and `lpr10` not 6, the
  reverse 0.
- Spread: `docs/anima/findings/lora_recipe.md`, "Single (artist, seed) clusters swing 6–2 either
  way in every A/B" (8 pairs per cluster there, 16 here; no seed twin of these runs).
- Not run: a seed twin, held-out images for the LPIPS read, a fixed λ, another σ gate, the default
  preset without the term as a second control.

## Interpretation

- At ratio 1.0 the term lowers its own objective on the training draws by 2 – 3 %, on nine steps
  in ten; at ratio 0.3 it does not separate from the control. The control's LPIPS itself falls by
  about 4 % over the run, so the term adds roughly as much again as FM training already gives.
- ‖dW‖ is within 0.7 % across the arms, so the sheets compare direction without a rescale.
- The blind counts are 11 of 16 on opposite sides. The lora_recipe single-cluster swing is 6 of 8,
  so neither count is an effect by that record, and the two sets share their control renders, so
  they are not two independent reads.
- The read does not follow the training log: `lpr03`, whose logged LPIPS is level with the
  control's, takes the picks, and `lpr10`, which lowers it, loses them. A lower one-step LPIPS on
  training draws does not predict the render read here.
- Two patterns hold across both sets and are worth a look on the labelled sheets: p1 (two
  characters) goes to the control 7 – 1, p3 (beach) to the arm 1 – 7; and no cell prefers `lpr10`
  where it does not prefer `lpr03`, which fits an order `lpr03` ≥ control ≥ `lpr10` on the six
  split cells.
- By the plan's rule ratio 1.0 is the "worse" side and is not carried on. Ratio 0.3 is neither a
  shown gain nor level: a seed twin (control + ratio 0.3 at seed 43) is what would settle it; two
  clusters put the sign-flip floor at 0.25.
