# Paired feature matching and the paired GAN on the σ 0.75 state (2026-11-17 – 11-18)

The anchor student's step 0 with terms on the teacher's features of the anchor end state: feature
matching (`1123`, `1403`, `1442`), then a paired GAN on top of `1442` (`0953`, `1028`), its dose
arm (`1122`), the gap-scale probe, the 2×2 pooled disc head (`1215`), FastGen's discriminators
against the port and FastGen's conv head on three taps (`1302`). The floor probe and the 0.01 arm (`1105`) are in
[step0_depth.md](step0_depth.md).

## Setup

### Facts

- Real: the cached target's state, `x_t = ε − 0.25 · v_target`, the teacher's own state at σ 0.75.
  Student: `x_s = ε − 0.25 · v_student`, with gradient. Paired by ε.
- Feature matching: the relative L2 between the frozen teacher's features of `x_s` and of `x_t` at
  σ 0.75, added to the anchor MSE. `dpdmd.fm_weight` is the weight on the feature term's gradient
  into `v`; `dpdmd.fm_block_idx` the tap (-1 = middle); `train/fm_grad_ratio` logs the rms of the
  weighted feature gradient into `v` over the anchor MSE's.
- Paired GAN: `gan.weight_gen > 0` under `dpdmd.anchor_only`. A token-head disc reads the taps of the
  FM forward, real = `x_t`, fake = `x_s`, so it runs no teacher forward of its own
  (`steps.anchor_fm_loss` / `paired_disc_update`). `gan.paired_ratio` rescales the generator
  gradient each step to that multiple of the FM gradient's rms on the teacher features;
  `train/gan_grad_share` logs the GAN's share of the feature-path gradient.
- Tests: `tests/test_turbo_anchor_fm.py`.
- Every arm: `0823` recipe, rank 64, `student_lr` 5e-5, `m15_k14of28_p1024` (1024 prompts × 2
  draws, one draw per epoch, so 2048 distinct pairs each seen once), 2048 iterations,
  `grad_accum 2`.

### Interpretation

- The regression weighs every element of the velocity alike; what breaks the image is the part of
  the residual that steps 1–3 read. Paired by ε, nothing pulls a seed toward another seed's layout.
- The paired GAN asks for a state that looks like a teacher state while the anchor MSE holds the
  pairing. Teacher samples as the real set is LADD's arrangement. The loop's GAN, on step 0,
  separates a blurred x̂₀ from a sharp image and pulls toward the dataset rather than the soup
  teacher (`findings/turbo_gan_dm_grad_orthogonal.md`).

## Feature matching: `1123`, `1403`, `1442`

### Facts

- 0.1 (`turbo_1117_1123`) was stopped at step 130 (feature gradient 8–14x the MSE's, anchor
  0.077 → 0.086 over steps 50–100).
- 0.05 (`turbo_1117_1403`): not passed. S cos 0.776 (1020: 0.785), T>S@1 0.944 (0.944),
  sameness S 0.296 (0.285), held-out anchor MSE 0.0768 (0.0741, above the 0.002 spread). The
  feature gradient ran 6–10x the MSE's, but the feature loss did not fall below 0.01's (0.655
  against 0.640 over steps 100–730, same seed). Renders: the group lower halves on seeds 3 and 4
  hold together a little better than 1020's, while the guitarist seeds 1, 2 and 4 still break, in
  different places (`bench/turbo/results/20261117-1429-handoff_fm005_1403`). The user reads
  1403's composition as better than 1020's and the term as promising, so the line goes on.
- Tap depth (`probe_fm_floor`, v1.1 student, 16 pairs; student residual over an equal-norm
  random one, median): block 3 0.6, 7 1.1, 14 10.3, 20 9.3, 27 28.0 (min 19). Blocks 14 and 20
  sit at a relative distance of 0.5–0.7, block 27 at 0.010 against a jitter of 0.0004.
- A 120-step smoke at tap 27, weight 0.05: `fm_grad_ratio` 0.02, and the feature loss falls with
  the anchor regression alone (0.0099 → 0.0083), which tap 14 never did; 1.08 it/s against 1.5
  (no early exit).
- Tap 27 at 15 (`turbo_1117_1442`), about 6x at the start, level with 1403's share: not passed
  on the probe. S cos 0.781, T>S@1 0.944, sameness S 0.291, held-out anchor MSE 0.0772 (above
  the spread, like 1403's 0.0768). The ratio held at 5–6x, the feature loss sat flat at 0.009 (the
  smoke's regression alone took it to 0.0083), and the training anchor ran 3.5% over 1403's.
  Renders against 1403: the guitarist's seeds 2 and 4 hold a whole guitar and seed 1 grows 1020's
  large hand again; the group's seed 3 doubles the grey-haired girl and seed 4 merges the lower
  faces (`bench/turbo/results/20261117-1517-handoff_fmtap27_1442`).
- The user read `1442`'s renders as fine (2026-11-18) and started the paired GAN on it.

### Interpretation

- At 0.05 the term perturbs the regression without closing the feature gap.
- The last block reads the student's residual and ignores a random one, which is why the tap moved
  to 27 with `fm_weight` set from a smoke's `fm_grad_ratio`.

## Paired GAN: smoke, `0953`, `1028`

### Facts

- Smoke (`smoke_pgan`, 80 it, λ 1.0, `disc_lr` 1e-5, tap 27): GAN share 0.88–0.93; disc margin
  0.05 at step 80.
- `turbo_1118_0953`: `1442` + paired GAN at block 27, λ 0.5, `disc_lr` 5e-5, generator held off
  200 steps then a 200-step ramp. Log, steps 1600–2048: training anchor MSE 0.131 against `1442`'s
  0.070, FM loss 0.0104 against 0.0081, student grad norm 12.5 against 3.0. The GAN share is 0.99
  after the ramp; disc margin 0.49 → 0.11. Renders not made.
- `turbo_1118_1028`: `0953` with `gan.paired_ratio` 0.5 and `weight_gen` 1.0 (ramp only).

  | steps | 0–200 | 200–400 | 400–800 | 800–1200 | 1200–1600 | 1600–2048 |
  |---|---|---|---|---|---|---|
  | anchor MSE, `1442` | 0.0950 | 0.0893 | 0.0858 | 0.0807 | 0.0715 | 0.0705 |
  | anchor MSE, `1028` | 0.0940 | 0.0885 | 0.0879 | 0.0832 | 0.0735 | 0.0705 |
  | FM loss, `1442` / `1028` | 0.0089 / 0.0088 | 0.0092 / 0.0091 | 0.0090 / 0.0090 | 0.0085 / 0.0086 | 0.0082 / 0.0082 | 0.0081 / 0.0080 |
  | GAN share | 0 | 0.23 | 0.43 | 0.43 | 0.43 | 0.43 |
  | gen loss | 0.85 | 1.21 | 1.50 | 1.77 | 1.82 | 1.91 |
  | disc loss | 1.23 | 0.98 | 0.89 | 0.80 | 0.78 | 0.79 |
  | disc margin | 0.44 | 1.41 | 2.15 | 2.83 | 2.98 | 3.06 |
  | student grad norm, `1442` / `1028` | 3.9 / 4.5 | 3.0 / 3.5 | 2.9 / 3.1 | 3.1 / 3.2 | 2.7 / 2.9 | 3.0 / 3.2 |

- `1028` renders: `bench/turbo/results/20261118-1105-handoff_pgan_1028` (same probe and settings as
  `1442`'s); student-only sheets against `1442`, seeds 1–4: `~/S_1442_vs_1028_seed{1..4}.jpg`.
- The user's read (2026-11-18): the GAN has a clear effect; with the margin issue fixed it should
  be usable. The line goes on with the GAN.
- Claude's read of the student sheets (S column, `1442` against `1028`):
  - Every seed keeps `1442`'s layout in all three prompts.
  - Guitarist: `1442`'s seed 2 guitar is broken (body and neck do not join); `1028` draws a whole
    guitar on seeds 1–4.
  - Group: `1442`'s seed 3 draws two grey-haired girls, `1028` one; `1442`'s seed 4 merges the two
    lower faces, `1028` draws them apart.
  - Café: level; the sign and menu text is garbled in both.

### Interpretation

- `0953`: λ was set against the smoke's weak disc (1e-5); a disc at 5e-5 makes the generator
  gradient about 10× larger, and the GAN then drives the student off the anchor target.
- `1028`: with the share held at 0.43 the GAN term leaves the anchor regression where `1442` has
  it, and the change it makes is local structure (whole objects, one figure per character) rather
  than layout.
- The margin rises through the run with no step at epoch 2 (new ε, same prompts), which reads more
  as student and teacher states staying separable than as prompt memorisation. The generator
  signal then comes from a disc near saturation, where its gradient says less about how far a
  state is from the teacher's.

## Generator dose: `1122` (`paired_ratio` 1.0)

### Facts

- `turbo_1118_1122`: `1028` with `gan.paired_ratio` 0.5 → 1.0, nothing else changed. Config header:
  1.0 becomes the parent if the anchor fit holds and the renders improve; otherwise 0.5 stays.
- Training log (`train/*`, means over steps a ≤ s < b; both runs binned the same way):

  | steps | 0–200 | 200–400 | 400–800 | 800–1200 | 1200–1600 | 1600–2048 |
  |---|---|---|---|---|---|---|
  | anchor MSE, `1028` / `1122` | 0.0938 / 0.0931 | 0.0890 / 0.0910 | 0.0870 / 0.0889 | 0.0844 / 0.0847 | 0.0736 / 0.0747 | 0.0703 / 0.0732 |
  | FM loss, `1028` / `1122` | 0.0088 / 0.0089 | 0.0092 / 0.0093 | 0.0090 / 0.0090 | 0.0086 / 0.0086 | 0.0082 / 0.0082 | 0.0080 / 0.0081 |
  | GAN share, `1028` / `1122` | 0 / 0 | 0.22 / 0.38 | 0.43 / 0.67 | 0.43 / 0.67 | 0.43 / 0.68 | 0.43 / 0.68 |
  | disc margin, `1028` / `1122` | 0.42 / 0.44 | 1.39 / 1.34 | 2.14 / 1.83 | 2.82 / 2.37 | 2.98 / 2.43 | 3.05 / 2.57 |
  | disc loss, `1028` / `1122` | 1.23 / 1.22 | 0.98 / 1.01 | 0.89 / 0.96 | 0.80 / 0.88 | 0.78 / 0.87 | 0.79 / 0.86 |
  | `fm_grad_ratio`, `1028` / `1122` | 6.76 / 5.77 | 5.86 / 6.07 | 6.91 / 7.06 | 6.57 / 7.25 | 6.53 / 6.32 | 6.31 / 6.73 |

- Probe (`probe_handoff --anchor_k 14 --anchor_steps 28`, `judge_hard` 3 prompts × seeds 1–4, EMA
  2048; `bench/turbo/results/20261118-1157-handoff_pgan_1122`, local), `1028` / `1122`: held-out
  anchor MSE 0.0739 / 0.0748, `S` cos 0.774 / 0.779, `T>S@1` cos 0.943 / 0.942, sameness `S`
  0.284 / 0.279. Spread unknown (`1028` has no seed twin).
- Read, blind: student sheets from `bench/turbo/blind_sheets.py`, two columns shuffled per row, 12
  rows (`output/blind/turbo/1122_vs_1028`, local). The user's picks, `1010 ---- 0111`: `1028` 6
  (group seeds 1, 3, 4; guitarist seeds 1, 2, 4), `1122` 2 (group seed 2, guitarist seed 3), no
  pick on the 4 café rows.
- The user's read (2026-11-18): it is already visible; the GAN does not try to break things, but
  the perspective of objects is ambiguous and it is comparatively monotonous.
- Verdict from the header: the renders did not improve, 0.5 stays the parent.
- Claude did not read these sheets.

### Interpretation

- Doubling the generator's push lowers the disc margin (3.05 → 2.57) and the user picks against it.
  One arm, but it goes against reading a lower margin as the goal of the next arm.
- The dose that helps is at or below 0.5. Past it the term costs depth and variety within an
  image while local structure stays whole.
- The token head scores each token of block 27 by itself and the loss does not use the pairing, so
  the generator is pushed toward what a teacher token typically looks like. Flatter, more uniform
  rendering is what that pressure would produce; not measured.

## Gap scale and disc logits: `probe_gap_scale`

### Facts

- `bench/turbo/probe_gap_scale.py --config <arm> --resume auto`, 16 cached training pairs, block
  27 at σ 0.75. The resume bundles hold step 1024, so every number is the online student (and
  disc) at step 1024 of 2048. Results `bench/turbo/results/20261118-12{08,09,10}-gap_scale_*`
  (local). Means over the 16 pairs unless marked.
- `keep_k`: the share of the gap's energy, `‖f_s − f_t‖²`, left after a k×k token mean pool (no
  spatial correlation: 1/k²). `jitter`: `v_target` plus Gaussian noise of the student residual's
  rms. `own`: the teacher features about their per-channel spatial mean.

  | | `1442` (no GAN) | `1028` (ratio 0.5) | `1122` (ratio 1.0) |
  |---|---|---|---|
  | keep 2×2 / 4×4, student gap | 0.612 / 0.400 | 0.608 / 0.389 | 0.602 / 0.378 |
  | keep 2×2 / 4×4, jitter | 0.289 / 0.091 | 0.289 / 0.091 | 0.289 / 0.091 |
  | keep 2×2 / 4×4, own | 0.394 / 0.201 | 0.394 / 0.201 | not run in this form |
  | relative feature L2, student, k = 1 / 2 / 4 | 0.0092 / 0.0058 / 0.0038 | 0.0093 / 0.0058 / 0.0038 | 0.0097 / 0.0060 / 0.0039 |
  | student over jitter, median, k = 1 / 2 / 4 | 32 / 68 / 142 | 32 / 65 / 130 | 29 / 60 / 116 |

- Token logits of the step-1024 disc (mean over tokens, then over pairs):

  | | `1028` | `1122` |
  |---|---|---|
  | teacher state / student state / jitter state | 1.00 / −2.17 / 1.00 | 0.40 / −2.03 / 0.39 |
  | std over tokens, teacher / student | 1.34 / 1.17 | 1.23 / 1.13 |
  | correlation over tokens, teacher against student at the same position | 0.12 | 0.18 |
  | teacher tokens below 0 / student tokens above 0 | 28% / 3.9% | 45% / 5.4% |

- No twin or second draw of pairs: spread unknown for the gaps between the three arms.

### Interpretation

- The gap is spatially coherent: 2×2 pooling keeps 60% of it and 4×4 keeps 38–40%, twice and four
  times what an equal-norm random residual keeps, and more than the teacher's own features keep
  (39%, 20%). A disc on pooled tokens still has a gap to read, and it reads the student-shaped
  residual more selectively (student over jitter 32 → 68 → 142).
- The disc keys on the student's residual and not on a perturbation of that size: the jitter
  state scores as the teacher state does.
- The logit of a token is nearly unrelated between the teacher and the student state at the same
  position (0.12, 0.18). A relativistic paired loss would cancel a shared per-position offset, and
  there is little of one, so that arm is dropped.
- The token head spends part of its signal on the 40% of the gap below the 2×2 scale. A 2×2 pooled
  head removes that part and keeps the coherent part; whether the token-scale part is what flattens
  the rendering is what a pooled-head arm at ratio 1.0 against `1122` would show.

## Pooled disc head: `1215` (`disc_pool` 2)

### Facts

- `gan.disc_pool` k (new knob, paired GAN on the token head): the tokens are folded to the patch
  grid and mean-pooled k×k before the head's MLP, one logit per cell. Tests in
  `tests/test_turbo_anchor_fm.py`.
- Smoke (60 it, no delay or ramp, ratio 1.0, `disc_pool` 2): GAN share 0.67–0.71, disc margin
  0.00 → 0.2, disc loss 1.40 → 1.29, 0.98 it/s.
- `turbo_1118_1215`: `1122` with `gan.disc_pool` 1 → 2, nothing else changed (ratio 1.0).
- Training log (`train/*`, means over steps a ≤ s < b), `1122` / `1215`:

  | steps | 0–200 | 200–400 | 400–800 | 800–1200 | 1200–1600 | 1600–2048 |
  |---|---|---|---|---|---|---|
  | anchor MSE | 0.0931 / 0.0931 | 0.0910 / 0.0904 | 0.0889 / 0.0885 | 0.0847 / 0.0860 | 0.0747 / 0.0740 | 0.0732 / 0.0729 |
  | FM loss | 0.0089 / 0.0088 | 0.0093 / 0.0092 | 0.0090 / 0.0090 | 0.0086 / 0.0087 | 0.0082 / 0.0081 | 0.0081 / 0.0081 |
  | GAN share | 0 / 0 | 0.38 / 0.38 | 0.67 / 0.67 | 0.67 / 0.67 | 0.68 / 0.68 | 0.68 / 0.68 |
  | disc margin | 0.44 / 0.42 | 1.34 / 1.37 | 1.83 / 2.03 | 2.37 / 2.54 | 2.43 / 2.62 | 2.57 / 2.77 |
  | disc loss | 1.22 / 1.23 | 1.01 / 1.00 | 0.96 / 0.93 | 0.88 / 0.86 | 0.87 / 0.85 | 0.86 / 0.84 |
  | logit std | 0.19 / 0.17 | 0.56 / 0.55 | 0.83 / 0.88 | 1.03 / 1.09 | 1.12 / 1.17 | 1.17 / 1.22 |

- Probe (`probe_handoff --anchor_k 14 --anchor_steps 28`, `judge_hard` 3 prompts × seeds 1–4, EMA
  2048; `bench/turbo/results/20261118-1250-handoff_pgan_pool2_1215`, local), `1122` / `1215`:
  held-out anchor MSE 0.0748 / 0.0758, `S` cos 0.779 / 0.771, `T>S@1` cos 0.942 / 0.942, sameness
  `S` 0.279 / 0.311, low-frequency share of the anchor error 0.447 / 0.444. Spread unknown (`1122`
  has no seed twin).
- Read, blind: student sheets `output/blind/turbo/1215_vs_1122` (local), 12 rows, at the probe
  PNGs' resolution (one file per row after the first send). The user's picks, `1010` on the 4
  group rows: `1215` 2 (seeds 1, 3), `1122` 2 (seeds 2, 4); no pick on the café and guitarist rows.
- The user's read (2026-11-18): it does not look meaningful. Verdict: level; the token head stays.
- The user's call after it: the GAN should be a proper implementation before arms are compared
  (not DMD2's), so the next arm is FastGen's 4×4 stride-2 conv head on taps 13, 20, 27.
- Claude did not read these sheets.

### Interpretation

- With a quarter of the logits the pooled disc separates the two states a little more than the
  token disc does (margin 2.77 against 2.57), in line with `probe_gap_scale`: the 2×2 mean keeps
  the coherent part of the gap and drops a random residual.
- The anchor fit and the FM loss sit where `1122` has them, so on the log the pooled head is a
  change in what the GAN pushes on, not in how hard it pulls the regression.
- Sameness `S` is the one probe column that moves (0.279 → 0.311); without a spread it is a number
  to put beside the user's read of variety, not a finding.

## FastGen's discriminators against the port (2026-11-18)

### Facts

- NVlabs FastGen at commit `898e757` (2026-11-10), `fastgen/networks/discriminators.py`,
  `fastgen/methods/distribution_matching/dmd2.py`, `fastgen/configs/`.
- Every FastGen head ends in a global pool: logits are `[B, num_taps]`, one per image and tap.
- Image DiT head (QwenImage, Flux): Conv2d 4×4 stride 2 (D → D/2), GroupNorm, LeakyReLU 0.2, 1×1
  conv, global average. About 8·D² parameters.
- Cosmos Predict2.5-2B recipe (28 blocks, width 2048): head `multiscale_down_mlp_large` (branches
  at scales 1, 2, 4, 8, 16, each 1×1 conv → GroupNorm → LeakyReLU → average pool → 3×3 conv →
  GroupNorm → LeakyReLU → global average; concatenated into an MLP of width 1024 with dropout
  0.1), taps 13, 20, 27, `gan_loss_weight_gen` 0.03, `gan_use_same_t_noise`, all learning rates
  1e-5, batch size 1 per GPU.
- The port (`_archive/proposals/turbo_gan.md`): a mean-pooled-token MLP as v0 ("only go spatial
  if v0 underperforms"), then the per-token head; LayerNorm → Linear → LeakyReLU → Linear, about
  D²/2 parameters, one middle tap. Kept from FastGen: the softplus losses, same τ / ε, 0.03, 1e-5,
  the approximate R1, frozen-teacher features with early exit.
- FastGen has no per-token head, and the port has no spatial operation in the head before
  `gan.disc_pool`.

### Interpretation

- `disc_pool` 2 is close to the scale-2 branch of the Cosmos head without its 3×3 conv and with a
  logit per cell. If the user's read favours `1215`, the arms with a source are FastGen's own:
  the taps 13 / 20 / 27 and the multi-scale head, one at a time.

## FastGen's head and taps: `1302` (conv head, taps 13 / 20 / 27)

### Facts

- New in the loop: `gan.disc_head = "conv"` (FastGen's `Discriminator_ImageDiT`: per tap a 4×4
  stride-2 conv → GroupNorm → LeakyReLU 0.2 → 1×1 conv on the patch grid, averaged to one logit)
  and `gan.feature_blocks` (a list of taps, one head each). The feature-path gradient is each
  term's direct gradient on its taps, pulled back through the teacher together, so nested taps
  count each path once; `gan.paired_ratio` sets rms(GAN, over its taps) against rms(FM, on its
  tap). Tests in `tests/test_turbo_anchor_fm.py` (12).
- Width: at FastGen's hidden D / 2 = 1024 the three heads hold 100.7M parameters and the run went
  out of memory at the first teacher pullback (14.15 GiB allocated on the 15.5 GiB card). The arm
  runs hidden 256: 25.2M parameters, peak 13.43 GB.
- Smoke (60 it, no delay or ramp): 0.97 it/s, GAN share 0.865, `fm_grad_ratio` mean 5.3, disc
  margin 0.05 → 1.0, disc loss 1.53 → 0.99.
- `turbo_1118_1302`: `1122` with the conv head, taps 13 / 20 / 27 and `disc_hidden` 256; ratio
  1.0, `disc_lr` 5e-5, delay and ramp as `1122`. Head and taps change together.
- Training log (`train/*`, means over steps a ≤ s < b), `1122` / `1302`:

  | steps | 0–200 | 200–400 | 400–800 | 800–1200 | 1200–1600 | 1600–2048 |
  |---|---|---|---|---|---|---|
  | anchor MSE | 0.0931 / 0.0919 | 0.0910 / 0.0897 | 0.0889 / 0.0846 | 0.0847 / 0.0822 | 0.0747 / 0.0699 | 0.0732 / 0.0697 |
  | FM loss | 0.0089 / 0.0089 | 0.0093 / 0.0093 | 0.0090 / 0.0090 | 0.0086 / 0.0087 | 0.0082 / 0.0081 | 0.0081 / 0.0081 |
  | GAN share | 0 / 0 | 0.38 / 0.56 | 0.67 / 0.86 | 0.67 / 0.86 | 0.68 / 0.87 | 0.68 / 0.87 |
  | `fm_grad_ratio` | 5.77 / 5.85 | 6.07 / 5.88 | 7.06 / 6.18 | 7.25 / 6.05 | 6.32 / 6.23 | 6.73 / 5.98 |
  | disc margin | 0.44 / 1.68 | 1.34 / 4.35 | 1.83 / 7.03 | 2.37 / 9.22 | 2.43 / 10.45 | 2.57 / 11.15 |
  | disc loss | 1.22 / 0.94 | 1.01 / 0.60 | 0.96 / 0.35 | 0.88 / 0.24 | 0.87 / 0.17 | 0.86 / 0.15 |
  | student grad norm | 3.99 / 4.13 | 3.23 / 2.96 | 3.43 / 2.87 | 3.72 / 3.03 | 3.20 / 2.70 | 3.61 / 2.89 |

- For reference, training anchor MSE over steps 1600–2048: `1442` (no GAN) 0.0705, `1028` (token
  head, ratio 0.5) 0.0703.
- Probe (`probe_handoff --anchor_k 14 --anchor_steps 28`, `judge_hard` 3 prompts × seeds 1–4, EMA
  2048; `bench/turbo/results/20261118-1338-handoff_pgan_conv3_1302`, local), `1122` / `1302`:
  held-out anchor MSE 0.0748 / 0.0764, `S` cos 0.779 / 0.777, `T>S@1` cos 0.942 / 0.943, sameness
  `S` 0.279 / 0.296, low-frequency share of the anchor error 0.447 / 0.458. Spread unknown.
- Read, blind: `output/blind/turbo/1302_vs_1122` (local), one file per row, 12 rows. The user's
  picks: pending.

### Interpretation

- This disc wins outright: margin 11 and disc loss 0.15, against 2.6 and 0.86 for the token head.
  One averaged logit per tap from a conv over the grid separates the two states far more easily
  than per-token logits do, and nothing in the recipe holds it back (`disc_lr` 5e-5, no R1, no
  renoise; FastGen runs this head at 1e-5 with renoised inputs and a weight of 0.03).
- With the generator gradient normalised, the saturated disc still gives a direction, but the
  student's side of the log looks more like the no-GAN run than like `1122`: training anchor MSE
  0.0697 (`1442`: 0.0705), a feature-path gradient into `v` below `1122`'s, and a lower student
  gradient norm. The GAN's pull on `v` in this arm may be weaker than the token head's at the
  same feature-space ratio; the renders decide whether it does anything visible.
