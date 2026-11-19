# Plan (2026-11-19)

Phases are the proposal's (`proposal.md`); this file holds their order, what is settled and what
is open. Each run's header and numbers live in `bench/pixel_readout/README.md`.

## Done

### P1 — linear readout

`readout_1118_2219`: test 26.28. The user's read: "진짜 잘나오네".

### P2 — ConvNeXt head, reconstruction only

`convnext_1119_0938` (no skip): test 32.93, text
27.45, against 26.50 / 21.58 for the head with a P1 skip (`reports/p2_skip_2026_11_19.md`).
The user's verdict (2026-11-19): "대 합격". Claude's labelled read has the grid and grain gone
and the head a little softer than the VAE; on manga pages (`sheets_1119_1032`) the smallest
text and screentone dots stay soft. The training method is settled (the user, 2026-11-19);
see Recipe.

### P3a — generated images and adapter transfer

`p3_1119_1056` (`reports/p3_2026_11_19.md`): 16 prompts × 2 seeds, standard sampler, target
VAE(x_final). Colour 30.90 on base, 31.24 with `anima_merged15`; mono 22.51 / 22.85; the
CFG-combined feature reads level with the cond feature (+0.03 dB). Real manga pages
(`manga_1119_1052`): the VAE alone is 21.5–39.8 dB from the original, VAE(x̂0) 24.3–36.3 from
VAE(x0).

Claude's call (2026-11-19, sheets sent to the user, standing unless flagged): the head
generalizes to generated images and to the LoRA; what is left is dense fine strokes (screentone,
hatching, fur lines), on generated and real images alike. The reasoning is in the report's
interpretation. P3 continues as the training strategy for manga and greyscale (In progress).

### Recipe change — fp32 stream, head compile, replay buffer

`replay_1119_1225` (`reports/replay_recipe_2026_11_19.md`): test 33.35 / text 27.68 against
`1106`'s 33.64 / 28.04 in 11.3 min against 25.7. The user's read: no difference, or replay
slightly better (attributed to the fp32 stream). Adopted 2026-11-19. Same report: decode cost
on the Spectrum sampler (`e2e_1119_1209`) and the rollout profile.

### P3b — manga pages in the mix, target VAE(x̂0)

`pagemix_1119_1242` (`reports/pagemix_2026_11_19.md`): the replay recipe with `--target xhat0`,
pages at share 0.33, 2,000 rollouts. Test 34.83 / text 29.65 / manga 28.15 against VAE(x0)
(replay 33.35 / 27.68; `mangaup` manga 26.64); 30 / 30 test and 8 / 8 text images up; tone level
with the VAE. Target, rollout count and pages changed together. The user's verdict
(2026-11-19): "거의 구분 안되는 정도고 그나마 강조선이 살짝 흐린정도네. p3은 통과라고 봐도
될거같아." P3 passed.

## Recipe (settled unless a run says otherwise)

- fp32 DiT stream (`--stream fp32`), head training forward compiled under `--compile`, replay
  buffer `--replay 64 --opt_steps 16` (`replay_1119_1225`; earlier heads: bf16 stream, 4 updates
  on the newest rollout). Time saved goes to more rollouts and data (the user, 2026-11-19).
- No skip, out at default init, layer scale γ 1 (`--no_skip --ls_init 1.0`). The three were
  changed together and are not separated; no run is planned to separate them.
- Loss 0.5 L1 + 5 MSE + 0.05 L_grad, AdamW, cosine, clip 1.0, one DiT forward and one VAE decode
  per rollout.
- LR 2e-4 at batch 1 (`0938`). `convnext_1119_0950` (gradient over 4
  rollouts, LR 1e-3, the user's arm) ended level with it: test 32.96 against 32.93, text 27.46
  against 27.45, after a faster start (27.57 against 25.69 at rollout 600).
- One 2× upsample stage before the third block (`--up 1`, the user's arm): `convnext_1119_1106`
  test 33.64 / text 28.04 against `0938`'s 32.93 / 27.45, 38 / 38 images up, same speed, 24.9M
  parameters; spread unknown. Adopted 2026-11-19 (Claude's call, sheets sent, standing unless
  flagged).
- Target VAE(x̂0) (x̂0 = x_t − σ·v from the rollout's own forward) and manga pages at share 0.33,
  2,000 rollouts (`pagemix_1119_1242`, P3 passed on the user's read). Pages:
  `post_image_dataset/pixel_pages/` (local; `build_pages.py`, 3,000 train pages, 1,500
  Manga109-s + 1,500 hitocrawl JA, seed 0, ~4,100 tokens, one fixed tag caption `monochrome,
  greyscale, comic, speech bubble, japanese text`; 10 eval pages with original pixels).
- Current head: `bench/pixel_readout/results/20261119-1304-pagemix_1119_1242/head.safetensors`
  (local).

## Next

### 1. P4 — σ-conditioned head

Needed by cjk_rl arm D and turbo: σ over [0, 1], target
VAE(x̂0) at that σ, read at σ ∈ {0.06, 0.5, 0.9}.

Designed around the Spectrum schedule (the user, 2026-11-19). `make test SPECTRUM=1` (28 steps,
shift 3, warmup 7, window 2 + 0.25, stop-caching 27) runs 16 actual forwards, at σ 1.0 … 0.917
(steps 0–6), 0.882, 0.844, 0.80, 0.75, 0.66, 0.545, 0.395, 0.187, 0.10. The 12 cached steps
(σ 0.90 … 0.265) never run the blocks: `final_layer` reads a Chebyshev-forecast feature, so the
head's tap exists there too, as a forecast. Consequences for the design:

- The read set includes the shipped grid's last σ, 0.10. The P2 head is trained at 0.0577 only;
  on the Spectrum sampler it reads colour 28.52 / mono 20.89 against P3a's 30.90 / 22.51
  (`e2e_1119_1209`), a regression the user called expected.
- Read on actual-forward features and, separately, on forecast features at the cached steps; a
  head that holds on forecasts gives a preview at every step for the cost of the head alone.
- Whether to train on forecast features is decided by that read, not up front.

### 2. P5 — pixel losses through the head

GAN on the head, arm D's reader NLL through the head,
or a ResShift refiner. Separate proposals once P4 reads out.

## Open

- Speed lines (강조선) slightly soft, footnote-size glyphs blur, under zoom only (the user's read
  and Claude's on `pagemix_1119_1242`).
- The P3a generations (6 mono, 26 colour) not re-run on the pagemix head (`p3_generated.py`).
- Training on sampled final latents: in reserve (P3a did not break on generated images).
- The DiT GEMMs run under bf16 autocast (the stream is fp32 since `replay_1119_1225`); whether
  that rounding costs readout PSNR is unchecked.
- No spread for any trained head: one seed per run. A seed twin of the recipe is owed before a
  gap under about 1 dB is read as a gain. P3's seed-to-seed gap of one prompt (mean 1.98 dB) is
  content variation between images, not a run spread.
- Grey tint on mono generations (Claude's read on `p3_1119_1056`): on teacher-forced greyscale
  pages the pagemix head adds about 0.5 level of chroma over the VAE (`reports/pagemix_2026_11_19.md`);
  generated mono images are not re-checked.
- MOSS: nothing until the Anima phases read out.
- RGBA (the user, 2026-11-19: an open item only). An alpha channel on the head's output is
  cheap to add, and labels can come from real RGBA cutouts composited onto random backgrounds
  (exact alpha) or a matting model on generated images. What the head alone cannot give: the DiT
  renders a composite, so semi-transparent regions (hair tips, glass, smoke) carry no
  background-free foreground colour; LayerDiffuse tunes the DiT for that. The baseline to beat is
  post-hoc matting (BiRefNet, anime-segmentation) on hair-edge sheets.
