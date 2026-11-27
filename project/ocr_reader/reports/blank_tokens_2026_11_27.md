# VL16 reader: the bf16 gradient error sits on the blank patches' tokens, the reads do not depend on them, and cutting their gradient at 1k steps brought no gain (2026-11-27)

Paths under `output/` are local. Reader in every probe: `output/ocr/mainsub_1127_0706/best`
(1k steps), merged as `core.readers.Vl16Reader` loads it.

**The probe arms and the trainer flag named below are not in the tree.** They ran from the
working tree on 2026-11-27 and were reverted on the user's word ("어차피 안쓸거니까 코드쪽은
되돌리고 리포트만"). Each section says what its arm did.

A patch is "blank" when the standard deviation of its grey levels (0..255, at the size the
tower sees) is under 4.

## 1. Where the error is: Facts

- `probes/precision_stream.py --tok_grad` on 32 random val crops (`--grad_seed 1`), the CE
  gradient of the vision tokens leaving each tower block, relative distance from the fp32 arm
  (`precl_vl16_ckpt_1127_1205`, local):

  | block output | bf16 | `stream_lean` | `stream` |
  |---|--:|--:|--:|
  | 0 | 40.6 % | 28.7 % | 28.8 % |
  | 1 | 30.8 % | 22.2 % | 22.3 % |
  | 13 | 14.9 % | 9.0 % | 8.9 % |
  | 26 | 6.1 % | 2.8 % | 2.7 % |

- The fp32 gradient's norm over the same tokens: 0.108 at block 0, 0.047 at block 13, 0.005 at
  block 26. fp32 run twice: distance 0.
- Arm `stream_block<k>` (the stream arm's bf16 GEMMs and attention core in vision block k only,
  everything else fp32), `precb_vl16_ckpt_1127_1320`:

  | bf16 block | error at block 0 output | at block 1 output |
  |---|--:|--:|
  | 0 | 12.1 % | 9.4 % |
  | 1 | 22.1 % | 17.5 % |
  | 2 | 14.6 % | 11.0 % |
  | 3 | 9.6 % | 7.4 % |
  | 6 | 7.4 % | 5.9 % |
  | 13 | 1.7 % | 1.4 % |
  | 20 | 2.3 % | 1.9 % |
  | 26 | 0.9 % | 0.8 % |
  | all 27 | 28.9 % | 22.3 % |

  With block 1 alone in bf16 the gradient at block 13's output is 8.3 % off.
- Arm `stream_skip<k>` (the stream arm with the first k vision blocks fp32),
  `precs_vl16_ckpt_1127_1400`:

  | fp32 blocks | error at block 0 output | blank tokens | other tokens | tower weight gradient (median) |
  |--:|--:|--:|--:|--:|
  | 0 | 28.8 % | 46 % | 8.5 % | 6.8 % |
  | 1 | 20.7 % | 34 % | 4.7 % | 3.8 % |
  | 2 | 22.4 % | 36 % | 7.5 % | 5.7 % |
  | 4 | 19.0 % | 30 % | 7.0 % | 6.1 % |
  | 8 | 7.1 % | 11 % | 3.4 % | 3.1 % |
  | 27 | 0.7 % | 0.8 % | 0.6 % | 0.9 % |

- At block 0's output, the 1 % of tokens with the largest squared error
  (`precb_vl16_ckpt_1127_1340`):

  | arm | error | their part of the error | their part of the fp32 gradient | error over the other 99 % |
  |---|--:|--:|--:|--:|
  | bf16 | 40.5 % | 86 % | 40 % | 19.5 % |
  | `stream_lean` | 28.7 % | 92 % | 22 % | 8.9 % |
  | `stream` (tower) | 28.9 % | 93 % | 20 % | 8.8 % |
  | `stream_block1` | 22.1 % | 96 % | 11 % | 4.5 % |

  The fp32 gradient's own top 1 % of tokens hold 82 % of its squared norm at block 0 and 92 %
  at block 26.

## 1. Where the error is: Interpretation

- The error at block 0 is not a sum over 27 backward steps. Rounding in the first few blocks
  moves the stream, and every gradient behind it is taken at the moved state: one bf16 block
  early in the tower puts error on gradients that never pass back through it.
- The one-block contributions do not add (blocks 0 – 3 alone square-sum to 30.6 %, and with
  those four in fp32 19 % is left), and `stream_skip1` reads under `stream_skip2`: a single
  draw moves by 2 – 4 points.

## 2. Which tokens: Facts

- `--tok_dump` wrote, per vision token and block, the gradient's and the output's squared norm
  and squared distance from fp32, four attention statistics and the crop's patches
  (`prect_vl16_ckpt_1127_1350`): 19,896 tokens in 32 crops, 50.3 % blank, all of them white.
- At block 0's output:

  | | blank tokens | other tokens |
  |---|--:|--:|
  | part of the fp32 gradient's squared norm | 37 % | 63 % |
  | part of the error, `stream_lean` / bf16 | 94 % / 93 % | 6 % / 7 % |
  | relative error, `stream_lean` | 46 % | 8.6 % |
  | relative error, bf16 | 65 % | 13 % |

  Median over the crops of the blank tokens' part of the gradient: 26 %. Blank tokens are the
  worse side in 32 of 32 crops (`stream_lean`) and 31 of 32 (bf16).
- The tokens of the top 1 % sit at the 93rd percentile of their crop's gradient norm and the
  25th of its patch contrast; 3 – 8 % of them are border patches (16 % of all tokens are).
- How far a token's output moves from fp32, 99th percentile of the relative distance: at block
  3 under 1.2 % for both kinds in both arms; at block 26 blank 21 % (`stream_lean`) and 55 %
  (bf16), other tokens 8 % and 18 %.
- Squared norm of the blank tokens' output over the other tokens', medians: 1.6 – 1.8 at
  blocks 0 – 13, 1.2 at block 20, 18.9 at block 26.
- The three tokens of a crop that take the most attention in block 1 (median 18 % of a query's
  attention each; 0.3 % for a token at large) are blank in 77 % of cases and hold 0.0 – 0.4 %
  of the gradient and 0.1 % of the error.
- Per crop the error differs by arm: crop 21 is 106 % off under `stream_lean` and 32 % under
  bf16, crop 31 321 % under bf16 and 70 % under `stream_lean`. The top-1 % sets of two arms
  share 26 – 60 % of their tokens.
- Claude's read of three overlays (crop, fp32 gradient, error, labels visible; `OCR2:` crops):
  the gradient sits on the furigana strokes and on one or two blank patches, the error on the
  blank patches.

## 2. Which tokens: Interpretation

- Blank patches enter the tower identical, so what a blank token holds is set by position and
  attention alone, and a small shift early on grows with depth. The picture matches the
  register tokens reported for ViTs; the comparison is from memory, not checked against the
  paper.
- Which crop goes wrong changes with the arm, so the error behaves as heavy-tailed noise drawn
  per crop and rounding, not as a fixed bias.

## 3. Do the reads depend on the blank tokens: Facts

- `probes/blank_noise.py`: the reads of `eval_manga109.py --manifest sfxp_gt` (Manga109 test,
  `OCR:`, SFX 2,558 and speech 2,559, greedy, same batches), with a change on the way into
  `generate`. Pixel arms add Gaussian noise of sigma grey levels to the processor's
  `pixel_values` on the blank (`blank`) or the other (`ink`) patches, rounded to a grey level
  and clamped; `blankshift1` darkens every blank-patch pixel by one level. 44.6 % of the
  patches are blank. Run `blank_0706_1127_1346`:

  | arm | pixel values changed | SFX reads changed | SFX only clean / only arm right | SFX exact | speech reads changed | speech exact |
  |---|--:|--:|---|--:|--:|--:|
  | `clean2` (read again) | 0 | 0 | 0 / 0 | 1,677 | 0 | 2,402 |
  | `blank1` seed 0 | 61 % | 39 | 5 / 9 | 1,681 | 12 | 2,402 |
  | `blank1` seed 1 | 61 % | 31 | 5 / 7 | 1,679 | 15 | 2,402 |
  | `ink1` | 91 % | 30 | 4 / 9 | 1,682 | 13 | 2,399 |
  | `blank4` | 70 % | 56 | 8 / 8 | 1,677 | 15 | 2,399 |
  | `ink4` | 96 % | 93 | 22 / 16 | 1,671 | 25 | 2,401 |
  | `blankshift1` | 99 % | 39 | 6 / 8 | 1,679 | 8 | 2,403 |

  The SFX crops whose read changed hold fewer blank patches (0.27 – 0.31 of the crop) than the
  rest (0.35). Of `blank1` seed 0's 39 changed SFX reads, 25 are wrong before and after.
- Two arms leave the pixels alone and replace the blank tokens' states where the tower hands
  them to the projector: `outmean` by the mean over the crop's own blank tokens, `outother` by
  that mean of the next crop of the batch. Run `blankout_0706_1127_1349`:

  | arm | SFX reads changed | SFX only clean / only arm | SFX exact | speech reads changed | of them spacing or line breaks only | speech only clean / only arm | speech exact |
  |---|--:|---|--:|--:|--:|---|--:|
  | `outmean` | 135 | 24 / 23 | 1,676 | 65 | 34 | 11 / 6 (p 0.33) | 2,397 |
  | `outother` | 139 | 28 / 21 (p 0.39) | 1,670 | 74 | 43 | 12 / 6 (p 0.24) | 2,396 |

- The read rule, set before the results: both arms level with clean = the LM does not use the
  blank tokens' content; `outother` alone worse = crop-wide content is kept there; `outmean`
  already worse = they carry position.
- Speech exact of this checkpoint is 2,402 here and 2,294 in `mainsub_1127_0706.md`: the
  scoring key changed with `TARGET_NORM = 13` (`1f6c6d80`) in between. SFX exact is 1,677 in
  both.

## 3. Do the reads depend on the blank tokens: Interpretation

- Noise on blank patches changes no more reads than the same noise on the other patches, and
  fewer at sigma 4. The blank tokens' unstable states do not reach the read.
- Another crop's blank mean does what the crop's own does, so the LM takes no crop-wide
  content from the blank tokens. What moves is where a speech read breaks its lines.
- The replacement is at the tower's output only. Whether the other tokens read the blank ones
  inside the tower is not covered.

## 4. Cutting the blank tokens' gradient: Facts

- `core/vl16_blank.py` detached the blank tokens' states with the forward unchanged: `out` at
  the projector's input, `all` at every vision block's output as well. Probe
  `precision_stream.py --blank_detach none,out,all`, each mode against its own fp32 arm
  (`precd_vl16_ckpt_1127_1420`):

  | cut | tower weight gradient error, `stream_lean` / bf16 | block 0's linears | blank tokens' part of the block-0 gradient | fp32 weight gradient against the uncut one: cosine / norm |
  |---|--:|--:|--:|---|
  | none | 6.9 % / 11.8 % | 12.2 % / 19.2 % | 37 % | |
  | `out` | 7.0 % / 13.6 % | 10.1 % / 23.8 % | 39 % | 0.87 – 0.94 / 0.72 – 0.77 |
  | `all` | 4.0 % / 8.6 % | 4.2 % / 10.9 % | 0 % | 0.83 – 0.94 / 0.68 – 0.76 |

- Training, 1k steps, seed 0, the command of `mainsub_1127_1212`
  (`finetune_vl16_lora.py --mainsub --steps 1000 --val_limit 500 --tower_rank 64 --ckpt_lm_only
  --compile --lm_full --stream fp32_lean`): `mainsub_1127_1418` with `--blank_detach all`,
  `mainsub_1127_1419` without. Both 22.3 crops/s, 12.6 GB. `mainsub_1127_1212` is the same
  command on `TARGET_NORM = 12`.
- The read, set before the results: val main / sub (500 each); Manga109 test paired between
  `1418` and `1419`; the doujin gate (`eval_sfx.py --pad cap4`) as numbers; spread unknown (no
  seed twin), with the `1419` / `1212` gap as a rough bound. A gap inside it is level.

  | read | n | `1418` cut | `1419` control | only `1418` / only `1419` | `1212` |
  |---|--:|--:|--:|---|--:|
  | val main exact | 500 | 451 | 459 | 5 / 13 (p 0.10) | |
  | val sub exact | 500 | 266 | 272 | 8 / 14 (p 0.29) | |
  | Manga109 test SFX exact | 2,558 | 1,681 | 1,675 | 137 / 131 (p 0.76) | 1,683 |
  | Manga109 test speech exact | 2,559 | 2,389 | 2,406 | 22 / 39 (p 0.04) | 2,393 |
  | Manga109 test SFX runaway | 2,558 | 15 | 31 | | 25 |
  | doujin SFX exact | 593 | 195 | 200 | | 191 |
  | doujin speech exact | 272 | 103 | 102 | | 94 |

  `1419` against `1212`: SFX only-one-right 116 / 124 (p 0.65), speech 34 / 21 (p 0.10).
- Training loss, mean over logged steps 775 – 1000: `1418` 0.1295, `1212` 0.1242.
- Jobs: `make daemon-jobs RUN=mainsub_1127_1418` / `_1419` (training),
  `RUN=mainsub_1127_<run>_cap4_ocr` (Manga109 test, doujin gate).

## 4. Cutting the blank tokens' gradient: Interpretation

- The blank tokens' gradient at the early blocks does not come down from the tower's output.
  It arrives through attention from the other tokens, so the cut at the projector leaves it
  (and its error) in place; the cut at every block removes it.
- SFX is level: `1418` − `1419` is +6, inside the −8 between `1419` and `1212`. Speech is 17
  under the control, a little past that bound, and 4 under `1212`; val main leans the same way.
  No read favours the cut except the SFX runaway count, which does not show in exact.
- With the 1k `--stream fp32` run reading level (`09651914`), the gradient noise on the blank
  tokens is not what holds the reader back at this scale.
- `project/cjk_rl` backpropagates this reader's CE to the pixels in bf16. The numbers here are
  on real crops; on a render a blank patch can be where a stroke is missing, so its gradient
  may be signal there. Not measured.
