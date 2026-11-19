# Plan (2026-11-19)

P1 – P4 are done (the user, 2026-11-19: "일단은 p4까지 완료인걸로"). Their record and the settled
training recipe are in `reports/phases_p1_p4_2026_11_19.md`; each run's design and numbers are in
`reports/`. The release head is `mixr_1119_1455`
(`bench/pixel_readout/results/20261119-1527-mixr_1119_1455/head.safetensors`, local). This file
holds what is next and what is open.

## Next

Order (the user, 2026-11-19): "배포되면 프로젝트 모델 코어는 라이브러리에 편입하는 쪽으로
계획을 잡자. p5는 아마 픽셀로스 등 integration일텐데 배포 및 승격을 선행하게 될듯". Release,
then promotion of the model core into `library/`, then the P5 topics.

### 1. Release

- Done 2026-11-19: `mixr_1119_1455` uploaded to the private repo `sorryhyun/anima-pixel-head`
  (the user: private first, they edit the card later). Card source `deploy/README.md`; decode
  cost on turbo and Spectrum and the upload record in `reports/release_2026_11_19.md` (turbo:
  VAE swap 15 % of DiT + decode, head 7 ms).
- Left to the user: the card text, making the repo public.
- Open: the head's turbo point was trained with the Hub turbo repo's
  `old/anima_turbo_4step.safetensors`; v3 and v4 are not read. A `p3_generated.py` run with
  `--adapter` on v3 / v4 answers it (minutes).

### 2. Promotion — the model core into `library/`

After the release. The steps below are Claude's draft, to be settled when the item opens.

- What moves: `PixelHead` / `ConvNeXtBlock`, `load_head`, `lat_patches`, `from_patches` and
  `PATCH`, the tap on `final_layer.linear`'s input, the CFG feature combine
  f_u + g·(f_c − f_u), and one decode call (features + x_t → pixels). Destination
  `library/anima/pixel_head.py`: the tap, the 16-px patch and the 64-value latent patch are
  Anima's. `bench/pixel_readout/` imports from it and keeps training, evals, sheets and probes.
- A model-catalog row for the released weight (`library/downloads.py`); the loader's default
  path comes from it.
- An `inference.py` decode switch (head or VAE) and its `test-*` lever. The head is read only
  where it was trained: shift-3 Euler at 20 … 50 steps on the base DiT, and the 4-step turbo
  grid. Outside that the head still decodes and the CLI logs a warning; no refusal, no fallback
  (the user, 2026-11-19: "거부까진 안해도 될듯. 경고 cli 로그만 출력하면 될거같고").
- `anima_lora` façade re-export for embedders.
- `CONTRIBUTING.md` Tier 2: `e2e_speed.py` as the bench, an invariant test that runs without
  the card (a saved head loads from its metadata; decode output shape per token grid), and a
  `docs/anima/` page.
- ComfyUI node (the user plans one, 2026-11-19), after the library lands. Its own standalone
  repo symlinked into `../comfy/custom_nodes/`, the head code a `_vendor/` copy of
  `library/anima/pixel_head.py` (`make vendor-sync`). Two parts: a MODEL patch that keeps the
  last forward's pre-head features and x_t (cond and uncond, for the CFG combine), and a decode
  node that turns them into an IMAGE in place of VAE Decode; the same out-of-range warning as
  the CLI. The tap sits on the DiT (`final_layer.linear`'s input), not on a sampler, so the
  stock KSampler and the Spectrum KSampler need no change: Spectrum's cached steps still run
  `final_layer` on the forecast feature (`ComfyUI-Spectrum-KSampler/spectrum.py`,
  `_spectrum_fast_forward`), and its tail steps are actual forwards. x_t and σ come from a
  `CALC_COND_BATCH` wrapper, which runs above Spectrum's model wrapper on every step.

## P5 topics

After the release and the promotion, on the library's head (the user, 2026-11-19: "기존 plan은
이제 p5에서 unnest해서 여러 주제를 적자. x2 or x4, rgba, pixel loss 등"). Each topic is its own
line of work with its own proposal; order not set. The first steps are Claude's drafts.

### ×2 / ×4 output

The head writes 16×16 px per token after one internal 2× stage; ×2 output is 32×32 per token,
decode and upscale in one readout. Under L1 / MSE a 1024-tier feature has no 2048 detail to
regress, so a ×2 head trained on the recipe's loss alone gives a blurred mean
(`proposal.md`); it needs a pixel loss (below) or a refiner. What exists: the SR sidecar's
1-step ResShift ×2 / ×4 students (`project/finished/sr/`).

- First steps: the chain swap with no training (head output in place of the VAE decode ahead
  of ResShift ×2, minutes); a count of training images whose master in `image_dataset/` is at
  least 2× the resized side, the supply of real ×2 targets.

### RGBA

An alpha channel on the head's output is cheap to add, and labels can come from real RGBA
cutouts composited onto random backgrounds (exact alpha) or a matting model on generated images.
What the head alone cannot give: the DiT renders a composite, so semi-transparent regions (hair
tips, glass, smoke) carry no background-free foreground colour; LayerDiffuse tunes the DiT for
that. The baseline to beat is post-hoc matting (BiRefNet, anime-segmentation) on hair-edge
sheets.

### Pixel losses on the head

What is left against the VAE on `mixr_1119_1455`, on the user's read: screentone dots smoothed
into soft shading; earlier, speed lines and footnote-size glyphs soft under zoom.
Measured beside it: textured regions 1 – 2.5 levels darker than the VAE in linear light
(`tone_probe.py`), the meter for any arm here.

- GAN on the head: FastFrames uses DINOv3 ViT-S/16 heads at layers 4 / 8 / 12 with a
  relativistic loss and R1 / R2; the turbo line's disc code is in git history.
- A paired real-pixel target: head training is teacher-forced from cached latents, so each pair
  has its real image (`post_image_dataset/resized/`), a target past the VAE without a GAN.
- A ResShift 1-step refiner from the head's output, read against the GAN arm.

### Pixel losses through the head, into the DiT

The head as the differentiable decode for losses that train the DiT or a LoRA: cjk_rl arm D
(reader NLL through the head, its D0 cost probe rerun against the VAE-window path), a GAN or
refiner term for the turbo student.

### A head at every step

Spectrum's cached steps (σ 0.90 … 0.265) read a Chebyshev-forecast feature at the tap, and
cjk_rl arm D reads x̂0 at steps 0 – 1; both want a head over σ in [0, 1]. `mixr_1119_1455`
covers σ 0.058 … 0.136 on base and 0.5 on turbo. Use: a preview at every step.

### MOSS

A head on MOSS-Image (different DiT, 128-channel latents at /16, the same 16-px token); decode
memory is the motive there (the fp32 VAE decode does not fit beside the DiT). Nothing run.

## Open

- Screentone: small dots smoothed into soft shading, a systematic difference from the VAE (the
  user's read on `mixr_1119_1455`, good enough to release). Taken up under P5, pixel losses on
  the head.
- Tone: the head is darker than the VAE on textured regions (the user's read; `tone_probe.py`,
  `reports/p4_mix_2026_11_19.md`): about 2.5 levels in linear light on screentone-like blocks of
  mono renders on the base DiT, 1 on turbo, under 1 level over a whole image; flat regions level.
  The loss has no term on local tone. Same P5 topic.
- Spectrum renders of `mixr_1119_1455` have numbers only (34.37 / 26.00); no sheet was made.
- The head's flat grey on screentone carries a faint regular weave at the patch scale (Claude's
  read, labelled, base 20 / 50 `manga_page` crops); not measured.
- Speed lines (강조선) slightly soft, footnote-size glyphs blur, under zoom only (the user's read
  and Claude's on `pagemix_1119_1242`).
- The P3a generations (6 mono, 26 colour) not re-run on the pagemix head (`p3_generated.py`).
- Training on sampled final latents: in reserve (P3a did not break on generated images).
- The DiT GEMMs run under bf16 autocast (the stream is fp32 since `replay_1119_1225`); whether
  that rounding costs readout PSNR is unchecked.
- The compiled head (`torch.compile(head, dynamic=True)`) hits `recompile_limit` 12 on image
  shape and runs eager after that, in every run of the replay recipe
  (`reports/p4_mix_2026_11_19.md`); speed only.
- No spread for any trained head: one seed per run. A seed twin of the recipe is owed before a
  gap under about 1 dB is read as a gain. P3's seed-to-seed gap of one prompt (mean 1.98 dB) is
  content variation between images, not a run spread.
- Grey tint on mono generations (Claude's read on `p3_1119_1056`): on teacher-forced greyscale
  pages the pagemix head adds about 0.5 level of chroma over the VAE (`reports/pagemix_2026_11_19.md`);
  generated mono images are not re-checked.
