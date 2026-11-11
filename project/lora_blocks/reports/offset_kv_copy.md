# The offset copy: no positional half, a late lookup, and a sanitize fine-tune that cannot move it

Date: 2026-11-11. Copy block under `ref_pos offset` + `ref_adapters kv`
(`c_fill_bw10_mod16_off_kv_lr2e4_s4800`, Muon lr 2e-4, 4,800 steps) against the aligned copy block
(`c_fill_bw10_mod16_s800`, lr 5e-5, 800 steps), and P2's sanitize fine-tune from each.

## Runs and reads

- **Sanitize ft from offset C**: [`_archive/configs/lora_blocks/sanitize_ft_off_kv_s300.toml`](../configs/sanitize_ft_off_kv_s300.toml),
  P2's recipe (277 pairs, lr 5e-5, EMA off, tone 0.003 from step 150) with offset + kv. 1.7 s/step,
  peak 12.3 GB. `output/ckpt_moss_i2i/sanitize_ft_off_kv_s300`. Its aligned twin is
  `sanitize_ft_cbw10_s300` ([c_fill_mod16_bw10.md](c_fill_mod16_bw10.md)).
- **Eval**: `ref_gate.py --part sanitize --arms ctrl,cut`, five pages, seeds 0–2:
  `output/tests/ref_gate/20261107-2202-p2/offkvft` beside `cbw10ft`.
- **Copy probe**: `copy_vs_step.py` (now also writes copy / reference mass per block and the
  late-step top heads) over both lines' saves, `usual_case1` (in0) and `hard_to_sanitize1` (in1),
  seed 0, the sanitize prompt: `output/tests/copy_vs_step/20261111-offset/{aligned,offset}`.
  `base` is the line's last file at multiplier 0, so it carries that line's reference placement.
- **Weights**: ‖ΔW‖_F per module and block range, from the low-rank factors.

## Sanitize eval, `ctrl`

| page | offset ft | aligned ft (cbw10-ft) |
|---|---|---|
| usual_case1 | watermark gone; s0 reframed (zoom, horns lost), a garbled mark on s1 | gone 3 / 3, copied |
| usual_case2 | washed out and slightly zoomed 3 / 3 | copied |
| hard_to_sanitize1 | handle and anger bubble gone; **`無断…` stamp back on s1 / s2** | all gone 3 / 3 |
| hard_to_sanitize2 | text gone; red balloons stay red blocks, white ones flat patches | text gone 3 / 3 |
| hard_to_sanitize3 | black blob and redrawn background; lettering back on s0 / s2 | clean 3 / 3 |

Outside the mask, mean of seeds 0–2, L1 offset / aligned: 14.8 / 6.2, 22.1 / 7.0, 11.5 / 10.8,
**15.3** / 17.7, 8.1 / 6.2. The training loss was below the aligned run to step 100 and level with it after.

## Copy probe

Own-cell copy mass at step 0 (image query → the reference token at its own cell), heads over 0.1
at step 0 / in the last third, L1 against the reference:

| line | save | top heads, step 0 | heads > 0.1 s0 / late (in0; in1) | copy mass blocks 0–13 / 28–41 (in1) | ref mass (in1) | L1 in0 / in1 |
|---|---|---|---|---|---|---|
| aligned | Base | b5 h3 .33, b14 h15 .21, b8 h14 .19 | 10 / 17 | 3.9 / 0.4 (in0) | 143 (in0) | 15.4 / — |
| | C s200 | b39 h19 .39, b38 h22, b37 h20 | 25 / 36; 33 / 44 | 3.8 / 7.2 | 227 | 7.1 / 9.1 |
| | C s400 | b39 h19 .97, b40 h20 .95 | 70 / 87; 72 / 98 | 4.2 / 23.5 | 319 | 5.8 / 6.8 |
| | C s800 | b39 h19 .99, b40 h20 .98 | 80 / 106; 79 / 120 | 4.5 / 31.7 | 348 | 5.7 / 6.2 |
| | ft s300 | b39 h19 .90 | 63 / 81; 65 / 90 | 3.8 / 22.2 | 274 | 6.7 / 11.7 |
| offset | Base | — (max .02) | 0 / 0 | 0.07 / 0.0 (in0) | 90 (in0) | 88.5 / — |
| | C s800 | b40 h2 .06 | 0 / 0; 0 / 0 | 0.04 / 0.15 | 80 | 52.9 / 61.6 |
| | C s1600 | b40 h2 .24, b41 h6 .17 | 5 / 3; 6 / 37 | 0.03 / 2.0 | 80 | 24.3 / 24.1 |
| | C s2400 | b41 h6 .32, b39 h19 .30 | 21 / 2; 22 / 45 | 0.03 / 5.3 | 79 | 24.3 / 21.3 |
| | C s4800 | b41 h6 .87, b40 h20 .82, b39 h19 .82 | 37 / 48; 41 / 50 | 0.01 / 15.1 | 68 | 9.5 / 9.9 |
| | ft s100 | b41 h6 .83 | 34 / **1**; 38 / 46 | 0.01 / 13.2 | 66 | 26.1 / 17.4 |
| | ft s300 | b41 h6 .89 | 34 / **4**; 36 / 46 | 0.01 / 11.7 | 68 | 23.1 / 10.5 |

## Weights

| ‖ΔW‖_F, q, blocks 0–13 (the other modules and ranges read the same) | aligned | offset |
|---|---|---|
| copy block | 4.2 | 49.5 |
| ft − copy block, relative | 0.63 | **0.10** |

Aligned C's delta is flat over depth (q+k+v+o per block 2.2–3.6, rising at 32–41). Offset C's main
bank is largest at blocks 4–19 (~25) and falls with depth; its kv export adapters rise to the end
(b36 28 → b41 72). The ft moved the export adapters 5–10 %, least at 36–41.

## Claims

1. **Offset takes away the copy's free half.** Base's early own-cell heads (b5 h3, b8 h14,
   b14 h15) are a RoPE position match; with the reference one image width to the right, there is
   no same-position key, and Base's own-cell mass is 0.000 everywhere (aligned Base: 10 heads,
   early-block mass 3.9). The offset copy block never builds an early stage of its own (blocks
   0–13 stay at 0.01–0.04 through 4,800 steps). Its copy is the late content lookup alone.
2. **The late lookup is the same circuit.** b39 h19, b40 h20 and b36 h3 / h21 are top heads in
   both lines; offset adds b41 h6 and b40 h11 on top. The ten-fold larger main bank in blocks 4–19
   is consistent with the offset run having to make the image queries' content matchable without
   the positional stage. That reading is not tested.
3. **It forms about 10× later and stays narrower.** The aligned phase change is at ~200 steps at
   lr 5e-5. The offset one sits between steps 800 and 1,600 at 4× the lr, and at 4,800 it has half the heads
   (37–41 against 79–80), a lower reference mass than Base (68 against 90), and L1 9.5–9.9 against
   5.7–6.2. The pixels copy at s1600 (sheet) while own-cell mass is still small, together with the
   few-cell shift seen in the fill eval, so part of the early offset copy may land off the own cell.
   This probe does not measure that.
4. **The sanitize ft from offset C is mostly offset C.** At equal lr and steps the fine-tune moves
   both lines by a similar absolute amount, but the offset init is ten times larger, so the ft
   changes it by 10 % against 63 %. The init's failure modes stay: the masked stamp and
   lettering return, and pages flatten or zoom.
5. **On usual_case1 the offset ft loses the copy during denoising.** Step-0 copy is intact (34
   heads), but in the last third only 1–4 heads stay over 0.1 (offset C: 48; aligned ft: 81), and
   the render is reframed (L1 23–26). With no positional anchor, nothing holds the frame once the
   late lookup weakens. Whether the lost copy causes the reframe or follows it, one seed cannot
   say.

## What it means

- An offset copy block is not a drop-in replacement for the aligned one. It costs ~10× the
  steps to form and copies less exactly, because the positional half Base gives aligned for free is gone.
- A fine-tune from it needs a learning rate scaled to the init: at lr 5e-5 for 300 steps the
  task gets 10 % of the weights' say.

## Canvas coordinates (2026-11-11)

[`_archive/configs/lora_blocks/1111_1232.toml`](../configs/1111_1232.toml): the lr 2e-4 s1600 offset + kv run with
`ref_pos = "canvas"` (the reference at r = 0 as the right panel, image and reference centred on
the two-panel canvas) and nothing else changed. Stopped at s800. For a reference of the image's
size the two placements differ in the r value (0 against 1) and in the image sitting half a width
left of the text tokens; the image-to-reference displacement is +W in both.

At matched steps it is the offset run:

| | canvas | offset twin | aligned C bw10 mod16 |
|---|---|---|---|
| loss, steps 300–400 | 0.379 | 0.387 | 0.078 |
| loss, steps 700–800 | 0.151 | 0.162 | 0.049 |
| copy-probe luma L1, s600 (in0 / in1) | 40.5 / 46.1 | 38.5 / 48.8 | – |
| copy-probe luma L1, s800 | 32.7 / 35.8 | 32.5 / 37.4 | 5.7 / 6.2 |
| top-8 copy mass at step 0, s800 | 0.64 / 0.71 | 0.80 / 0.85 | 5.9 / 7.3 |
| heads over 0.1 at step 0, s800 | 1 / 2 | 2 / 5 | 80 / 79 |

- The loss breaks at steps 400–500 in both. The late heads are the same (b39 h19, b40 h20,
  b41 h6, b37 h9); the early own-cell heads do not come back.
- Fill eval at s800 (`output/tests/ref_gate/20261111-1111_1232_at800`): out-L1 25–45 on `ctrl`
  (offset s1600 15–40, aligned 5.5–12.8). The character, palette and rough composition carry
  over and the picture is redrawn; usual_case1 grows a second head behind the first on seeds 1
  and 2. `cut` clears the small watermark box on usual_case1 and draws a second figure on
  hard_to_sanitize3 on all three seeds (out-L1 43–52 against `ctrl` 26–36).
- It agrees with the teacher-forced table of `docs/moss/proposal/ref_canvas.md`: under
  `ref_isolate` with a clean reference, canvas and offset coordinates read alike.
- Probe outputs: `output/tests/copy_vs_step/20261111-canvas/{canvas,offset_twin}`. The offset
  rows of `20261111-offset/offset` at s800 / s1600 (L1 53–62 / 24) are the s4800 run's saves,
  whose LR warmup is 240 steps against 80 here; they are not the matched comparison.

## Not tested

- The offset ft at lr 2e-4 (the init's lr), or the offset C trained shorter at lr 2e-4 so that its
  norm is closer to the aligned one.
- Copy mass at the displaced cells (an off-by-k own-cell read) to measure the s1600 shift.
- Whether an early positional stage can be given back: aligned RoPE for the reference keys in
  the early blocks only, offset after.
- One seed and two inputs on the probe; three seeds on the eval.
