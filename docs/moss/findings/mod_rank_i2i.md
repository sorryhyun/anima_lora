# `mod_rank` 16 on i2i: better preservation outside the edit, edit strength unchanged

Measured 2026-10-28 on the uncensor i2i task, MOSS-Image Base. Follows the t2i check in
`docs/moss/README.md` (`--mod_rank`: loss unchanged at r16 on caststation and ama_mitsuki).
Runs: `output/ckpt_moss_i2i/moss_i2i_uncensor_combined_muon_lr5e5_rw10to1{,_mod16}`;
renders: `output/tests/moss_i2i_uncensor/ab_mod16/{rw10to1,rw10to1_mod16}`, sheet
`compare_input_base_mod16.jpg` beside them.

## Setting

66 pairs (`post_image_dataset/moss/i2i_uncensor_combined`), fixed caption, full-size
isolated reference at `ref_timestep 0`, rank 32 / alpha 128, Muon lr 5e-5, 10 epochs = 660
updates, `region_weight` 10 → 1, `ema_window` 0.1, seed 0, swap 38 / budget 0.175. The arm
adds `--mod_rank 16` (`mod_alpha` default 90.5) and changes nothing else; the σ draws match
on all 660 steps, so the losses pair step by step. Cost: 3.59 → 3.78 s/step, +0.46 % file
size (~344k parameters), peak allocated 12.96 → 13.11 GB.

## Measured

Paired training loss, mod16 against the baseline (share of steps where mod16 is lower):

| steps | loss | edit region | rest (copied) |
|---|---|---|---|
| 1–110 | −1.5 % (87 %) | −0.7 % (71 %) | −2.7 % (86 %) |
| 111–220 | −6.8 % (95 %) | −4.1 % (84 %) | −17.1 % (99 %) |
| 221–330 | −3.5 % (95 %) | −2.2 % (85 %) | −9.1 % (97 %) |
| 331–440 | −2.4 % (94 %) | −1.4 % (75 %) | −8.2 % (98 %) |
| 441–550 | −1.5 % (86 %) | −0.0 % (70 %) | −7.7 % (95 %) |
| 551–660 | −2.6 % (89 %) | −0.8 % (71 %) | −5.5 % (95 %) |

Unlike t2i, the loss drops, and almost all of it on the copied tokens.

The modulation adapter: entropy rank 15.3 of 16, ‖ΔW‖ 2.6 % of the base
`shared_block_modulation` weight (block 40's q adapter: 1.05 %). Per chunk: FFN shift
12.9 %, attention gate 4.5 %, the other four 0.6–1.9 %. The block adapters come out as in
the baseline (mean ‖ΔW‖ 1.378 in both).

## Images

Six held-out pages, seed 0, ER-SDE 50 steps, CFG 3 (`output/tests/moss_i2i_uncensor/inputs`).

- **Preservation outside the edit: mod16 is visibly better, text most of all** (read by eye
  on the sheet).
- **Edit: a tie.** The edited area (ΔE > 8 against the input) is 0–1.8 % of the page in
  both arms, and inside it the arms differ by median ΔE 0.5–1.7.
- Page-level numbers miss the preservation gain: PSNR outside the edit moves ±0.5 dB either
  way per page, and ΔL / Δb outside it follow the same pattern in both arms. Small glyphs are
  too few pixels to move a page mean.

## Hypothesis (untested)

With `ref_timestep 0` the reference tokens take their modulation from the same
`shared_block_modulation` at t = 0 (`MOSSImageTransformer.ref_modulation`), so the adapter
can set the FFN shift for clean reference tokens apart from the noisy ones. That is a
handle the block adapters do not have, and it fits a gain that lands on the copied tokens.
An arm with `ref_timestep` equal to the image's would separate the two.

## Sanitize, quarter data (2026-10-29)

Runs: `output/ckpt_moss_i2i/moss_i2i_sanitize_q1_e4{,_mod16}`; renders
`output/tests/moss_i2i_sanitize/ab_q1_mod16/{base,mod16}/s{42,43,44}` (sheets
`compare_<page>.jpg`, tone `tone_s{42,43,44}.{txt,json}` beside them); probes
`bench/moss_bbox_grounding/results/20261029-1040-sanitize_q1_base` and
`20261029-1110-sanitize_q1_mod16`.

The v3 recipe (`configs/moss/i2i/sanitize/train.toml`: rank 32, Muon lr 5e-5, isolated
reference at `ref_timestep 0`, tone 0.003 from half-way, `ema_window` 0.1, seed 0) on 69 of
the 277 pairs (`post_image_dataset/moss/i2i_sanitize_idt_q1`: symlinks, seed-0 draw listed
in `subset.json`), 4 epochs = 276 steps. The arm adds `--mod_rank 16`; the σ draws match on
every step. Cost: 3.90 → 4.02 s/step, peak allocated 12.94 → 13.09 GB.

**Loss.** Flow-matching loss (the logged loss minus 0.003 · tone from step 139), mod16
against the baseline (share of steps where mod16 is lower):

| steps | 1–46 | 47–92 | 93–138 | 139–184 | 185–230 | 231–276 |
|---|---|---|---|---|---|---|
| FM | −0.7 % (80 %) | −3.7 % (89 %) | −4.9 % (98 %) | −13.0 % (98 %) | −7.2 % (98 %) | −4.6 % (96 %) |

The tone term itself does not move (second half −1.9 %, lower on 48 % of steps).
Teacher-forced on the probe's 21 pages (16 training pairs, 5 held-out pages rendered by each
arm), the gain sits mostly on the copied region `K`, as on uncensor: σ 0.9 `E` −2.6 % / `K`
−7.0 %, σ 0.6 −1.9 / −6.0 %, σ 0.3 −1.3 / −3.8 %.

**Images** (the five eval pages × seeds 42 / 43 / 44, `generate.toml`): **a tie.** Each
arm fails where the other does: the stamped `無断転載禁止` stays and the `@…` handle leaves a
ghost (hard_to_sanitize1), red balloons stay as blank red shapes and white ones smear
(hard_to_sanitize2), red smears replace the English on the dark ground (hard_to_sanitize3).
Both clean `usual_case1`'s corner watermark at all three seeds and usual_case2's faint one.

**Tone: mod16 renders warmer.** `bench/moss_i2i_eval/tone_map.py` per seed, Lab on the
pixels neither arm changed, 15 page × seed images:

| arm | ΔL | \|ΔL\| | Δa | Δb | median ΔE | images past \|ΔL\| 1 |
|---|---|---|---|---|---|---|
| base | −0.50 | 0.64 | +0.70 | +0.29 | 1.74 | 3 |
| mod16 | −0.45 | 0.48 | +0.78 | **+0.87** | 1.81 | 2 |

Δb is higher in mod16 on 15 of 15 images and has one sign over 89–100 % of the 128 px cells:
a page-wide offset, not a local stain. L is a little closer to the input. Neither arm
reaches v3 (ΔL −0.03, Δb −0.23, ΔE 1.07 at 2,216 steps); the tone term had 138 steps here.

**The modulation adapter**: ‖ΔW‖ 1.56 % of `shared_block_modulation`; per chunk FFN shift
7.3 %, attention gate 2.8 %, FFN scale 1.8 %, the other three 0.3–1.0 %. The same ranking as
uncensor (recomputed: 12.9 / 4.5 / 1.9 %).

**Its change is largest at t = 0, the reference's timestep.** Δmod(t) = ΔW · time(t), per
chunk, against the Base modulation at the same t (the image rows trained at σ ≥ 0.17,
2.2 % of steps below 0.3, median 0.78):

| | shift_a | scale_a | gate_a | shift_f | scale_f | gate_f |
|---|---|---|---|---|---|---|
| ‖Δmod(0)‖ / ‖Δmod(900)‖ | 1.7× | 4.0× | 3.0× | 2.0× | 2.0× | 1.7× |
| cos(Δmod(0), Δmod(900)) | 0.34 | −0.08 | **−0.55** | 0.24 | −0.09 | 0.49 |
| cos(Δmod(0), Δmod(300)) | 0.85 | 0.87 | 0.91 | 0.88 | 0.89 | 0.90 |

At t = 0 the delta is 0.21 % of the Base modulation (FFN shift 2.66 %), at t 300–1000
0.07–0.09 %, and it points elsewhere: the attention gate's delta reverses between the
clean reference and a σ 0.9 image. Δmod is near-orthogonal to the Base modulation at both
ends (|cos| ≤ 0.20), so it reshapes channels rather than scaling a branch. Uncensor (no tone
term, 660 steps) shows the same: ‖Δmod(0)‖ above ‖Δmod(300)‖ and ‖Δmod(900)‖ in every chunk
(attention gate barely: 0.178 vs 0.170), attention gate cos −0.48 against t 900. The image rows almost never visit t ≈ 0, so this end of Δmod is a
reference-only parameter, shared by all 42 blocks; the block adapters apply one ΔW to
reference and image rows alike. That fits the mark appearing from block 0. Whether the mark
is used, rather than being a content-dependent response to the shift, is not measured
(`mod` off costs `E` nothing at σ 0.9).

**Blocks** (`sanitize_probe.py`; `mod` = only the modulation adapter off). Share of the LoRA's
gain over Base lost, `E` / `K`:

| | base | mod16 | v3 (`bbox_grounding.md`) |
|---|---|---|---|
| blocks 26–35 off, σ 0.9 | 0.36 / 0.41 | 0.36 / 0.32 | 0.79 / 0.34 |
| blocks 14–41 off, σ 0.9 | 0.97 / 0.84 | 1.15 / 0.73 | 1.08 / 0.77 |
| `mod` off, σ 0.9 / 0.6 / 0.3 | — | 0.03 / 0.05, −0.03 / 0.04, **−0.14** / 0.01 | — |
| reference-row mark AUC, blocks 0 / 13 / 25 / 35 | 0.33 / 0.36 / 0.36 / 0.68 | 0.60 / 0.59 / 0.61 / 0.66 | 0.51 / 0.71 / 0.80 / 0.71 |
| top caption head (E − K, log mass) | b39 h19 2.9 | b39 h19 3.8 | b40 h20 5.3 |

- At 276 steps neither arm has v3's structure: blocks 26–35 are not edit-specific yet.
- mod16 marks the text in the reference stream from block 0 (AUC ~0.6), where the baseline's
  change is larger off the text (AUC < 0.5 up to block 30). Its caption heads are stronger.
- The modulation adapter alone carries a small share: 4–5 % of the gain on `K` at σ 0.9 / 0.6
  (bootstrap CI clear of 0), and **costs** `E` at σ 0.3 (−0.14, CI [−0.20, −0.08]).
- It is fitted to the block adapters: with them off in 14–41 and the modulation adapter on,
  `E` ends up worse than Base (1.15).

### Hypothesis (untested)

(Superseded by § mod16 without the tone term: the term shrinks the FFN shift rather than
adding to it.) The FFN shift adds one per-channel offset to every token at a given t: a
page-wide colour handle. The tone term, on from step 139, can pull b through it and overshoot warm, as
`tone_b_weight 3` did (Δb +1.3, `docs/moss/methods/tone_loss.md`); without the term sanitize drifts
cool (Δb −1.10). mod16 trained with `tone_weight 0` separates this from the adapter holding
tone by itself.

### mod16 without the tone term

`moss_i2i_sanitize_q1_e4_mod16_notone` (`--tone_weight 0`, otherwise the mod16 arm; renders
`…/ab_q1_mod16/mod16_notone/s{42,43,44}`, three-arm tone `tone3_s*.{txt,json}`, sheets
`compare3_<page>.jpg`). Three-arm mask, 15 images:

| arm | ΔL | \|ΔL\| | Δa | Δb | sat× | median ΔE | past \|ΔL\| 1 |
|---|---|---|---|---|---|---|---|
| base (tone 0.003) | −0.50 | 0.64 | +0.70 | +0.29 | 1.046 | 1.73 | 3 |
| mod16 (tone 0.003) | −0.45 | 0.49 | +0.78 | +0.87 | 1.048 | 1.81 | 2 |
| mod16, no tone term | −0.18 | 0.44 | +0.50 | **−0.13** | 1.007 | **1.32** | 1 |

Its Δb is the lowest of the three on 15 of 15 images. The flow-matching loss matches mod16 to
step 138 and is 2.6 / 6.0 / 6.8 % lower over the last three 46-step chunks (the tone term's
cost). Edits tie by eye, with one miss: hard_to_sanitize1 s44 keeps the anger mark.

Not settled: without a baseline at `tone_weight 0` on the same 276 steps this cannot tell
"mod16 holds tone by itself" from "the drift has not built up at 276 steps and the term
overshoots warm in both arms". The −1.10 cool drift is from another recipe
(`idt_cs0_600`: lr 1e-4, 600 steps).

What the tone term changed (the two runs are identical to step 138):

- **In the modulation adapter it acts on the FFN shift, and shrinks it.** Of ‖Δmod_tone −
  Δmod_notone‖², 74–75 % is the FFN shift at t 0, 300 and 900 (attention gate 7–9 %,
  attention shift 7–10 %). With the term every chunk's delta is smaller: FFN shift 2.66 vs
  3.63 % of the Base modulation at t 0, 1.45 vs 2.28 % at t 900; cos tone vs no-tone 0.77–0.83
  for the FFN shift, 0.83–0.94 for the gates and scales.
- **It does not move the shared i2i part.** The reference-only Δmod's cosine with uncensor
  and swap_text is the same with and without the term (attention gate 0.91 / 0.93 against
  uncensor, 0.88 / 0.89 against swap_text).
- **The block adapters cannot place it.** Blocks 26–41 agree (cos 0.81 / 0.93); blocks 0–25
  diverge (cos 0.18–0.26), but they diverge as much between mod16 and the no-mod baseline
  (0.20–0.25): those adapters carry ≤ 0.06 of the gain, and any change to the run
  decorrelates them.

So the warm cast is not the term adding an offset through the adapter: the term pulls the
FFN shift down and turns it, and the run whose FFN shift it did not touch holds tone best.
That fits the adapter holding tone by itself, with the single-step tone term (on x̂₀ at the
training σ) pulling against what the 50-step sample needs; the no-mod control at
`tone_weight 0` would still be needed to rule out the short run.

## Shared across i2i tasks (2026-10-29)

Cosine between adapters of different runs. Every run below is seed 0, rank 32, Muon,
kaiming: their `down` matrices start the same (cos 0.99–1.00 after training), so the two t2i
mod16 runs (`ama_mitsuki_muon_ema_mod16`, `caststation_ema097_mod16`) are the init-only
control. Computed from the low-rank factors (`tr(u₁ᵀu₂ · d₂d₁ᵀ)`), never the full ΔW.

The modulation adapter's reference-only part, Δmod(0) − Δmod(900), per chunk:

| pair | shift_a | scale_a | gate_a | shift_f | scale_f | gate_f |
|---|---|---|---|---|---|---|
| sanitize – uncensor | +0.48 | +0.78 | +0.91 | +0.67 | +0.70 | +0.57 |
| sanitize – swap_text | +0.23 | +0.83 | +0.88 | +0.37 | +0.26 | +0.43 |
| uncensor – swap_text | +0.16 | +0.76 | +0.92 | +0.39 | +0.41 | +0.90 |
| any i2i – t2i, t2i – t2i | −0.41 … +0.18 in every chunk | | | | | |

(swap_text: `moss_i2i_swap_text_1300_full_rw2_mod16_ema5e5`.) At t 900 (the image side)
sanitize and uncensor still agree; swap_text's FFN gate turns the other way (−0.45 / −0.48).

The block adapters, mean cos over every target in a block range:

| pair | 0–13 | 14–25 | 26–35 | 36–41 |
|---|---|---|---|---|
| swap_text with / without mod16 (same task) | 0.07 | 0.08 | 0.52 | 0.82 |
| sanitize – uncensor | 0.08 | 0.08 | 0.23 | 0.37 |
| sanitize or uncensor – swap_text | 0.04 | 0.04 | 0.14 | 0.20–0.30 |
| any of those – colorize (`moss_colorize_3k_textref1_iso_muon_ema`) | 0.01–0.03 | 0.01–0.02 | 0.04–0.05 | 0.06–0.08 |
| any i2i – t2i, t2i – t2i | ≤ 0.003 | ≤ 0.002 | ≤ 0.015 | ≤ 0.013 |

- The i2i runs share a direction the t2i runs do not: in the reference-only modulation
  (attention gate 0.88–0.92 across all three tasks) and in blocks 26–41.
- In the blocks the overlap follows how much of the page is copied: two removal tasks share
  most, swap_text (erase and letter) less, colorize (repaints every pixel, keeps the
  structure) least, though colorize uses the same isolated `ref_timestep 0` reference.
- Similar directions do not show the part is reusable: no init or transplant test yet.

## Open

- A preservation metric that sees text: ΔE or OCR on text crops outside the edit, not a
  page mean.
- The same A/B on swap_text, where the copied area holds most of the page's text.
- Sanitize at full length (2,216 steps), where v3's block structure exists.
- Sanitize q1 baseline with `tone_weight 0`: the control for the `_notone` arm above
- Whether the shared i2i part transfers: `docs/moss/proposal/moss_i2i_ambitious.md`.
