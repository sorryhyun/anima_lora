# Paired feature matching and the paired GAN on the σ 0.75 state (2026-11-17 – 11-18)

The anchor student's step 0 with terms on the teacher's features of the anchor end state: feature
matching (`1123`, `1403`, `1442`), then a paired GAN on top of `1442` (`0953`, `1028`). The floor
probe and the 0.01 arm (`1105`) are in [step0_depth.md](step0_depth.md).

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
