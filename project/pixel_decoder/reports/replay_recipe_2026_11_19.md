# Decode cost, rollout profile and the replay recipe (2026-11-19)

Covers `e2e_1119_1209` (the head against the VAE on the `make test SPECTRUM=1` sampler), the
rollout profile (`prof_bf16_1119_1217`, `prof_fp32_1119_1217`) and the training arm that followed
(`replay_1119_1225`). Code and per-run records: `bench/pixel_readout/` (`e2e_speed.py`,
`rollout_profile.py`, `convnext_head.py`) and its `README.md`; results under
`bench/pixel_readout/results/` (local). One seed everywhere; spread unknown.

## e2e_1119_1209 — decode cost on the shipped sampler

### Facts

anima-base, Euler 28 steps, shift 3, CFG 4, the Spectrum flags of `scripts/tasks/inference.py`
(16 actual forwards, the last at σ 0.10), compiled blocks, the 16 P3 prompts at ~1 MP, seed 0.
Head `convnext_1119_1106` (24.9M parameters, 95 MB). Each final latent decoded four ways; times
with CUDA sync and pixels on the CPU; peak VRAM above the resident DiT. The DiT ran on the bf16
stream (`bench/_anima.build_anima` does not apply `stream_fp32_`).

| per image, median of 16 | seconds | peak above resident DiT (MB) |
|---|---|---|
| DiT sampling | 6.58 | 1349 |
| head | 0.007 | 244 |
| VAE resident | 0.141 | 1298 |
| VAE moved CPU ↔ GPU per image | 0.178 | 1384 |
| VAE loaded from disk per image | 0.828 | 1384 |

16 images: DiT 105.6 s; decode adds 0.2 s (head), 2.8 s (swap), 12.9 s (disk).
Head vs VAE(x_final): 27.09 mean, colour (13) 28.52, mono (3) 20.89. P3 (`0938` head, σ 0.0577,
50 steps, CFG 3.5) read 30.90 / 22.51. The user (2026-11-19): the σ 0.10 regression was
expected; P4 carries it.

### Interpretation

On this sampler the decode is 2–11 % of an image, so the head's speed matters little here; its
use is per-step previews (P4), pixel losses beside the DiT, a turbo sampler where the DiT share
shrinks (measured since: VAE swap 15 % on turbo, `release_2026_11_19.md`) and MOSS, where the fp32 VAE decode does not fit
beside the DiT. The decode-time VRAM drop (1.38 → 0.24 GB) does not lower the overall peak at
1 MP, which DiT sampling sets.

## rollout_profile — where a training rollout's time goes

### Facts

12 train images (4,032–4,200 tokens), one warm pass then one timed pass, medians.

| phase | bf16 stream | fp32 stream |
|---|---|---|
| load (latent + TE) | 1.8 ms | 1.7 ms |
| DiT forward at σ_last | 203 ms | 207 ms |
| one VAE decode | 141 ms | 141 ms |
| 4 head updates, eager | 73 ms | 73 ms |
| 4 head updates, `torch.compile(dynamic=True)` | 62 ms | 62 ms |

Every head up to `convnext_1119_1106` was trained and scored on bf16-stream features;
`inference.py` defaults to `--stream fp32`.

### Interpretation

DiT and VAE are 82 % of a rollout, and each rollout fed 4 updates. Reusing rollouts is the lever;
head compile is worth 2.5 % of a rollout. The fp32 stream costs 2 % of the DiT forward and puts
training on the features the shipped sampler produces.

## replay_1119_1225 — fp32 stream, head compile and a replay buffer

### Facts

Parent `convnext_1119_1106`. Changed together (the user's arm): `--stream fp32`, compiled head
training forward, `--rollouts 900 --opt_steps 16 --replay 64` (each update draws one of the last
64 rollouts; 14,400 updates as the parent, a quarter of the DiT forwards and VAE decodes).
11.3 min against 25.7 (0.76 against 0.43 s / rollout, compile and evals included).

| updates 2400 / 4800 / 7200 / 9600 / 12000 / 14400 | test (30) | text (8) |
|---|---|---|
| `1106` | 26.94 / 30.57 / 31.73 / 32.89 / 33.44 / 33.64 | 23.47 / 26.01 / 26.93 / 27.62 / 27.96 / 28.04 |
| `replay_1119_1225` | 29.03 / 31.30 / 31.97 / 32.63 / 33.21 / 33.35 | 24.45 / 26.01 / 26.72 / 27.20 / 27.54 / 27.68 |

- Per image, replay − `1106`: test −0.47 … −0.13 (0 / 30 up), text −0.41 … −0.28 (0 / 8 up).
- Against VAE(x̂0): test 34.93, text 28.77.
- P1's linear (fit on bf16 features) reads 25.21 on the fp32-stream test features against 26.28
  on bf16.
- A bf16-stream twin (`replaybf16_1119_1231`) was queued and cancelled by the user: the fp32
  stream is settled on other evidence, so a drop reads as replay's.
- Claude's read (labelled, 2× crops of text 00 / 02 and sheet 08): no visible difference between
  the two heads; both lose the small furigana and the smallest glyph strokes against the VAE.
- The user's read (2026-11-19, native-resolution crops of the 8 text windows and sheet 08): no
  difference, or replay slightly better, which the user attributes to the fp32 stream.
- Adopted into the recipe under the user's rule set before the run: no large drop → replay goes
  in, and the saved time goes to more data.

### Interpretation

Replay led to 7,200 updates and fell behind from 9,600 on; the 900 distinct rollouts against
3,600 are the likely limit late in the run, which more rollouts at the same reuse address. The
P1 drop shows the two streams' features differ by more than rounding noise, so heads trained on
the bf16 stream read the shipped sampler's features off their training distribution.
**A head on the replay recipe costs 44 % of the wall time for a −0.3 dB gap that neither read
could see.**
