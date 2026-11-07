# The reference copy forms in a sharp window, after the pixels already copy

Date: 2026-11-07. Script: `project/moss_te_sae/src/copy_vs_step.py` (b39 h19 own-cell
copy mass at step 0 and in the last third of the sampler, the copy mass summed over the
eight largest (block, head) pairs, the number of pairs over 0.1, luma L1 against the
reference; `copy_head.Probe` underneath). Runs: `output/tests/copy_vs_step/20261107-1643`
(epoch saves of three i2i runs, Base per task, two inputs for sanitize and swap_text, one
for uncensor) and `…/20261107-1708` (a step-saving micro arm on `usual_case1`). Each
folder has `summary.json` and `sheet_<task>.png` (columns: ref, then the checkpoints in
order). Seed 0, the tasks' shipped generate settings (int8 DiT, uncensor with its cache
caption and `prompt_tail off`).

Micro arm: `configs/sanitize_copy_s300.toml` — the sanitize v3 recipe (277 pairs, muon,
lr 5e-5, rank 32, `ref_timestep 0`, `ref_isolate`) on the int8 plan, 300 steps, EMA off,
a save every 25 steps (`save_every_steps`, added to `TrainRequest` for this). 2.27 s/step,
11.2 min. Checkpoints `output/ckpt_moss_i2i/probe_sanitize_copy_s300[/…_sN].safetensors`.

Question: how much of an i2i LoRA's training goes into the copy? The lora_blocks proposal
assumes a reusable copy block saves part of every task run; the answer bounds that part
([opinion.md](../opinion.md)).

## Epoch saves

| task | step | h19 copy s0 / late | top-8 | heads > 0.1 | L1 vs ref |
|---|---|---|---|---|---|
| sanitize | Base | 0.000 / 0.000 | 1.5 | 10 | 15.4 |
| | 1108 (e4) | 0.85 / 0.81 | 6.0 | 61 | 6.3 |
| | 2216 (final) | 0.88 / 0.85 | 6.2 | 80 | 5.8 |
| swap_text | Base | 0 / 0 | 1.4 | 10 | 29.3 |
| | 2109 (e1) | 0.98 / 0.99 | 7.5 | 68 | 6.3 (outside text 6.0) |
| | 4218 (final) | 0.98 / 0.99 | 7.7 | 79 | 6.0 (outside text 5.7) |
| uncensor | Base | 0 / 0 | 1.5 | 11 | 29.6 |
| | 330 (e5) | 0.55 / 0.29 | 2.6 | 30 | 10.0 |
| | 660 (e10) | 0.93 / 0.89 | 5.9 | 60 | 6.9 |
| | 990 (e15) | 0.94 / 0.94 | 6.4 | 62 | 6.4 |
| | 1320 (final) | 0.96 / 0.94 | 6.9 | 68 | 6.2 |

Sanitize's second input (`hard_to_sanitize1`) and swap_text's (`589883_0004_ko`) read the
same: h19 0.97 / 0.95 and 0.96 / 0.96 at the two saves. Sanitize v3's saves are EMA
(`ema_window 0.1`), uncensor's too; swap_text's are the EMA of the int8 run.

## Micro arm, sanitize, steps 25–300

| step | h19 copy s0 / late | h19 ref mass s0 | top-8 | heads > 0.1 | L1 vs ref | largest copy heads at step 0 |
|---|---|---|---|---|---|---|
| 25 | 0.000 / 0.000 | 0.00 | 1.5 | 10 | 13.6 | b5 h3 .33, b14 h15 .21, b8 h14 .19 |
| 50 | 0.000 / 0.000 | 0.00 | 1.5 | 10 | 11.1 | same |
| 75 | 0.000 / 0.000 | 0.00 | 1.5 | 11 | 9.6 | same |
| 100 | 0.000 / 0.001 | 0.00 | 1.5 | 11 | 8.0 | same |
| 125 | 0.000 / 0.002 | 0.00 | 1.5 | 13 | 8.4 | same |
| 150 | 0.002 / 0.007 | 0.01 | 1.5 | 17 | 7.8 | same |
| 175 | 0.063 / 0.051 | 0.11 | 1.8 | 26 | 8.5 | b5 h3 .32, b37 h20 .26, b8 h14 .23 |
| 200 | 0.418 / 0.326 | 0.54 | 2.2 | 31 | 7.8 | **b39 h19 .42**, b5 h3 .32, b37 h20 .27 |
| 225 | 0.599 / 0.566 | 0.74 | 3.0 | 40 | 7.7 | b39 h19 .60, b37 h20 .43, b36 h21 .40 |
| 250 | 0.571 / 0.548 | 0.67 | 3.1 | 38 | 8.4 | b39 h19 .57, b36 h21 .42, b37 h20 .42 |
| 275 | 0.648 / 0.612 | 0.80 | 3.8 | 45 | 7.0 | b39 h19 .65, b36 h21 .57, b40 h20 .54 |
| 300 | 0.678 / 0.648 | 0.82 | 4.2 | 44 | 6.3 | b39 h19 .68, b36 h21 .60, b40 h20 .60 |

## Claims

1. **The copy head is a phase change.** b39 h19's own-cell mass is zero through step 150,
   crosses 0.06 → 0.42 → 0.60 between steps 175 and 225, then climbs slowly: 0.68 at 300,
   0.85 at 1108, 0.88 at 2216. The other late copy heads (b37 h20, b36 h21, b40 h20) turn
   on in the same window, in that order. Uncensor's e5 (step 330, 66 pairs an epoch, seen
   through the EMA) at 0.55 / 0.29 is the same transition on another task.
2. **Most of the pixel copy arrives before the head.** L1 against the reference falls
   15.4 → 8.0 over steps 0–100 with own-cell copy mass at zero and the over-0.1 head
   count unchanged (10 → 11). The sheet shows what that is: Base re-draws the subject;
   the step-25 render already has the reference's composition, pose and palette, and
   steps 25–150 sharpen it. Base holds own-cell heads in the *early* blocks (b5 h3 0.33,
   b14 h15 0.21, b8 h14 0.19), and training leaves them where they are: a RoPE position
   match, not a content copy. The first 100 steps learn to use what those heads deliver;
   the late content-copy circuit comes after, and is what makes small detail exact (the
   marks above the character's head appear only from step 200).
3. **After the head forms, what keeps moving is breadth.** From step 300 to 2216 h19
   gains 0.2; the heads over 0.1 go 44 → 61 → 80, and L1 6.3 → 5.8. swap_text (68 → 79)
   and uncensor (60 → 68) show the same with h19 flat. This is the protocol spreading
   over more heads, consistent with
   [`copy_protocol.md`](../../moss_te_sae/reports/copy_protocol.md)'s "no slice carries
   the copy".
4. **Identity pairs are not needed for the head.** The micro arm trained on sanitize's
   own pairs (277, 39 of them identity) and built the head in the same window; the copy
   is learned from the unedited part of any pair.

## What it means for lora_blocks

- The copy costs about 200–300 steps on any pair set: the first 10–15 % of a sanitize
  run, 5 % of a swap_text run. A reusable copy block does not shorten a long run by much;
  what it changes is **where a task fine-tune starts** (past the phase change, with its
  gradient free for the edit from step 1) and **what the pairs have to teach** (nothing
  about copying).
- The sharp window is the interesting part. A task trained from a copy block skips it, and
  the "where" and "what" a task needs are learned on a network that already reads the
  reference exactly — the same reason instruction-edit models pretrain on fill.
- P1 is cheaper than the proposal plans: 300 steps of plain pairs on the int8 plan already
  build the head. The hole with the `diag` gate is what turns a copy block into a fill
  block; that is the part still to test, and it is the part that decides whether C is
  worth a file of its own.
- Base's early positional own-cell heads and the learned late content head are the two
  halves of an induction-style circuit (Olsson et al., 2022): the position match exists
  without training, the content copy forms in a window. The upstream ablation in
  [opinion.md](../opinion.md) (which block ranges' deltas h19 needs) is the next read on
  the mechanism.

## Not tested

- One seed, one input on the micro arm; two inputs on the epoch saves.
- The window's position against learning rate, rank and pair count (uncensor at 66 pairs
  an epoch lands in the same place in steps, not in epochs).
- Whether the step-100 render's copy comes through the early positional heads
  (ablate b5 h3 / b8 h14 / b14 h15 on the step-100 weights).
- The same curve on the bf16 plan (v3 ran bf16; the micro arm ran int8).
