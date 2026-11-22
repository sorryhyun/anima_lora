# ×2 output — plan (2026-11-19, updated 2026-11-22)

The P5 topic "×2 / ×4 output" of `plan.md`: decode and 2× upscale in one readout of the DiT's
last features, so a 1024-tier render leaves the head at 2048. This file holds only what is
ahead. Records in use: `reports/`. Finished ones: `archive/` (local). Each has an index in its `README.md`.

Code: `scripts/pixel_head/train_x2.py` on `library/anima/pixel_head_train.py`. Outputs:
`output/pixel_head/<run>/` (local).

## State

- The form: a ×1 head plus a second 2× stage, 32×32 output pixels per token. A ×2 file holds
  the ×2 output only.
- The file to replace: Hub `anima_pixel_head_x2.safetensors` = `x2bgan2_1121_1340`. The user
  reported on it (2026-11-21) a systematic tiled pattern on fine texture and a blurred region
  of fine strokes.
- The ×1 base to start from: `x1ll_1122_1000`,
  `output/pixel_head/x1ll_1122_1000/head.safetensors` (local).

## The sequence after the ×1 head is fixed (2026-11-22)

Two runs in order, then the read. Neither is queued and neither config is written.

### Facts

The user (2026-11-22): "x1 base 에서 init해서 x2 b only 로 12k with low lr (the same with x1 low
lr) 그다음 gan finetune 2k 하는걸로 x2 plan 업데이트 하자".

| | 1. B only | 2. GAN finetune |
|---|---|---|
| starts from | `init` = the base's path, with a new second stage | `resume` = step 1's `head_x2.safetensors` |
| `rollouts` | 12000 | 2000 |
| updates (16 per rollout) | 192,000 | 32,000, all with the term |
| `lr` | 5e-5, one cosine | 5e-5, a new cosine |
| `pegan_ratio` | 0 | 2 |
| `pegan_start` | unused | 0 |
| `eval_every` | 1000 | 500 |
| seed | 0 | 0, and a twin at 1 |
| cost, estimated | 2.8 – 3.4 h + evals | 43 – 45 min + evals, each |

- Every other key is `configs/pixel_head/x2.toml`'s: arm B (the real 2× image as the target, σ
  0.0577 – 0.1364, LPIPS 0.1, dc 1.0). That file's own `lr`, `rollouts`, `pegan_ratio` and
  `pegan_start` are the shipped file's recipe, so both run configs set them.
- `init` is written out: empty, it loads the catalog's head, which is not the base.
- Where the values come from: step 1's LR, length and eval interval are `x1ll_1122_1000`'s.
  Step 2's length is the user's. Its LR and eval interval are the ×1 finish's
  (`x1fgan_1122_1247`), its ratio and start `x2gft_1121_2225`'s. The twin is carried over from
  this plan's previous version.
- Costs come from 0.84 (`x2b4k_1121_0950`) to 1.02 s per rollout (`x2move_1121_1350` at rollout
  150) without the term, and 1.29 – 1.35 s with it (`x2gftl_1121_2300`, `x2gft_1121_2225`).
- Both load paths were checked on the CPU (2026-11-22): the base into a new ×2 head, and a file
  saved by `train_x2.py` as a resume.

Records the steps rest on, test / text against the real 2× image:

| run | updates | LR | summed LR | dB | LPIPS |
|---|---|---|---|---|---|
| `x2b4k_1121_0950`, B only from the preview | 64,000 | 2e-4 | 6.4 | 30.45 / 25.92 | 0.141 / 0.137 |
| step 1 | 192,000 | 5e-5 | 4.8 | | |
| `x2gft_1121_2225`, finetune on that head | 16,000 | 2e-4 | 1.6 | 29.74 / 25.40 | 0.161 / 0.149 |
| `x2gftl_1121_2300`, the same, longer | 48,000 | 2e-4 | 4.8 | 29.86 / 25.48 | 0.152 / 0.135 |
| step 2 | 32,000 | 5e-5 | 0.8 | | |

- Summed LR is LR × updates / 2 (cosine), warmup left out.
- On `x2gftl` against the shorter finetune (`reports/x2_tile_gft_2026_11_22.md`):
  - The user's read: "확실히 길게 학습하는게 좋네".
  - Claude's read, labelled: parallel stripes at 48,000 updates, none at 16,000.

Read:

1. Step 1: its evals beside `x2b4k`'s row.
2. Step 2: its evals beside step 1's end and the `x2gft` row.
3. Renders, the new finetune beside the shipped file, one pair per file: the `text_04` mesh, the
   monochrome render (seed 42), the fine-stroke renders (seeds 42, 43). The user's read decides
   the Hub file.

Bars: spread unknown at these settings. `seed` also moves the held-out noise in `train_x2.py`
(`DiTRollout`), so the twin's gap is a spread only after both heads are read on one noise.

### Interpretation

- Question: does a converged low-LR B head from the ×1 base, with a 2k finetune, give sharp
  lines without the regular pattern? Yes on the user's read → it replaces the Hub file. A
  pattern → the finetune's length or ratio is redone from step 1's head. No pattern and soft
  lines → the finetune is too weak at this LR.
- Not separated against `x2b4k`: the init, the LR and the length change together.
- Not measured on ×2: LR 5e-5. On ×1 every layer started trained. Here the second stage starts
  new, and step 1's summed LR is 0.75 of `x2b4k`'s, which itself continued a trained stage. A
  step 1 that ends behind `x2b4k` and still falling is short, and is read so before step 2.
- Step 2 has twice `x2gft`'s updates at half its summed LR, with the discriminator's LR
  unchanged at 1e-4. Whether the stripes follow the update count or the head's movement is not
  known.

## Open

- The turbo point at ×2: postponed by the user (2026-11-20).
- ×4: not started.
