# Handoff probe: which student step loses the teacher's layout (2026-11-12)

Arm 0 (`arm0.md`) left one gap that both seeds share: the students draw a frontal standing figure
where the teacher draws the jump from below and the tumbling group. The student is role-separated
(step 0 regresses on the teacher anchor, steps 1–3 take DM / GAN / CDM), so the probe hands the
trajectory over at a student step boundary, in both directions, to see which step loses the layout.
Forward only, no training.

Script: `bench/turbo/probe_handoff.py`. Run: `bench/turbo/results/20261112-1033-handoff/` (job
`20261112-102845-97e863`, 6.5 min). One grid per student and prompt (`<student>_p{0,1,2}.jpg`, rows
are seeds, columns are arms), full-size images under `png/`.

**The render read below is Claude's, on downscaled grids. The user's read is pending.**

## Setup

- Base: merged14 on int8 (its own GPTQ file), fp32 stream, compiled blocks, as the judge renders.
- Prompts: `bench/turbo/judge_hard.txt` (five-girl group, café, guitarist), seeds 1–4, 768 × 1344.
- Students: the twin's EMA 250 (`1112_0834_250_ema`) and the control's EMA 125
  (`1112_0833_125_ema`). They differ in seed and in checkpoint.
- Teacher: 28 Euler steps, cfg 4, shift 3. The student's grid (σ 1, 0.9, 0.75, 0.5, 0) is a subset of
  the teacher's, so a handoff lands on a grid point of both.
- Noise: one draw per (prompt, seed), shared by every arm. It is drawn on the CPU generator; the
  judge renders in `output/tests/turbo4/` came from `inference.py --from_file`, which draws on CUDA,
  so the images here do not pair with those.

| Arm | Who draws what |
|---|---|
| `T` | teacher alone |
| `T>S@b` | teacher down to the student's boundary b (σ 0.9 / 0.75 / 0.5 for b = 1 / 2 / 3), student finishes |
| `S` | student alone, 4 steps, cfg 1 |
| `S>T@b` | student runs steps 0..b−1, teacher finishes from that σ |

## Paired number: low-passed cosine to the teacher's image

Cosine between the final latent of an arm and of `T` for the same noise, after an 8× average pool
and with the per-channel mean removed. Mean over 12 images; the three per-prompt means in brackets
(group, café, guitarist).

| Arm | twin EMA 250 | control EMA 125 |
|---|---|---|
| `T>S@3` (teacher to σ 0.5) | 0.947 (0.937 / 0.955 / 0.949) | 0.980 (0.977 / 0.984 / 0.979) |
| `T>S@2` (teacher to σ 0.75) | 0.858 (0.834 / 0.875 / 0.864) | 0.933 (0.922 / 0.947 / 0.929) |
| `T>S@1` (teacher to σ 0.9) | 0.695 (0.684 / 0.740 / 0.662) | 0.803 (0.783 / 0.843 / 0.784) |
| `S` | 0.455 (0.524 / 0.512 / 0.328) | 0.537 (0.618 / 0.537 / 0.456) |
| `S>T@1` (student to σ 0.9) | 0.641 (0.686 / 0.627 / 0.610) | 0.672 (0.754 / 0.661 / 0.601) |
| `S>T@2` (student to σ 0.75) | 0.574 (0.661 / 0.603 / 0.458) | 0.615 (0.709 / 0.605 / 0.531) |

Loss per student step, read down the `T>S` column (each row hands one more step to the student):

| Step the student takes over | σ range | twin EMA 250 | control EMA 125 |
|---|---|---|---|
| 3 | 0.5 → 0 | 0.05 | 0.02 |
| 2 | 0.75 → 0.5 | 0.09 | 0.05 |
| 1 | 0.9 → 0.75 | 0.16 | 0.13 |
| 0 | 1 → 0.9 | 0.24 | 0.27 |

The cosine measures how close an arm stays to the teacher's image for that noise. It is not a
quality score: a different layout of equal quality reads low.

## Renders

- **Teacher's step 0, student's steps 1–3 (`T>S@1`): the layout survives.** On the control's EMA 125
  all four guitarist seeds keep the jump from below, the small full-body figure, the tilted stage
  edge and the crowd line; the group prompt keeps the lying and leaning figures. The twin's EMA 250
  keeps the figure's scale and placement and loses part of the camera tilt (seed 2 turns upright).
  Both render cleanly from a σ 0.9 state they were not trained on.
- **Student alone (`S`): a larger, centred, frontal figure.** The pose still follows the noise (the
  same arm raised, the same guitar direction as the teacher's image of that seed), and on seeds 1 and
  2 the control's EMA 125 still bends the knees as in a jump. What is gone is the camera: no view from below, no tilt,
  the figure fills the frame. On the group prompt the figures stand or sit upright in a packed
  group photo.
- **`T>S@2` and `T>S@3` are the teacher's image** with small changes of detail.
- **Student's step 0, teacher's finish (`S>T@1`): hazy and desaturated on every image.** The teacher
  does not receive a state on its own path. On the control's EMA 125 the frontal centred layout
  stays. On the twin's EMA 250 the teacher redraws a dynamic pose on three of four guitarist seeds
  (a hand reaching to the camera, a jump).
- **`S>T@2`: the student's layout, hazy.** After the student's first two steps the teacher changes
  the rendering and not the framing.
- **Café:** `T>S@1` keeps the teacher's apron colours and counter; `S` changes the outfits (frilled
  aprons, black sleeves) and keeps two standing figures. This prompt has little layout to lose.

## Reads

- **Step 0 is the largest single loss.** Handing only σ 1 → 0.9 to the teacher takes the student
  from 0.46 / 0.54 to 0.70 / 0.80, and the renders agree. No arm in `plan.md` acted on step 0: MFD,
  the static grid and the VA-TDM fake all change the DM term, which reaches steps 1–3 only
  (`dm_first_step = false`, `detach_after_first`).
- **Steps 1–3 mostly keep what they are handed.** The loss shrinks with each later handoff. Step 1
  is second (0.13–0.16) and is where the twin's EMA 250 flattens the camera.
- **The reverse direction is a partial cross-check.** `S>T@1` agrees on the control (the layout is
  the student's once it has drawn step 0) and not on the twin. The haze says the student's σ 0.9
  state is off the teacher's path. Whether that is anchor misfit or the chord the anchor target
  defines (the mean velocity to σ 0.75, applied over 1 → 0.9) is not separated here.
- Step 0's training signal is weak in the recipe: `div_weight 0.1`, backwarded into the same
  gradient as DM / CDM / GAN and clipped with them. `train/div_loss` fell from about 0.20 to 0.16 over
  250 iterations in both arm 0 runs (quintile means 0.218 / 0.197 / 0.171 / 0.170 control,
  0.191 / 0.174 / 0.164 twin, outside the bursts). Its floor for a rank-64 LoRA is unknown.

## Limits

- The two students differ in seed and in checkpoint (125 against 250), so the larger step-1 loss of
  the twin cannot be assigned to either.
- 3 prompts × 4 seeds, one noise draw each.
- `T>S` hands the student a state from the teacher's path, not from its own step 0. A student
  trained with a better step 0 would see states of that kind in training; these did not.
- The read is on 320 px cells. The café grids were opened for the twin only.

## Consequence

The next arms act on step 0 (`plan.md`, "Step-0 arms"), before MFD. The probe's `S` row is the
paired number for them: a better step 0 should move it toward the `T>S@1` row.
