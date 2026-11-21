# Plan (2026-11-19)

P1 – P4 are complete, with the release, the promotion and the ComfyUI node below (the user,
2026-11-19: "일단은 p4까지 완료인걸로", then "p4는 완료된걸로 갱신하고 커밋하자"). Their record and the settled
training recipe are in `reports/phases_p1_p4_2026_11_19.md`; each run's design and numbers are in
`reports/`. The release head is `mixr_1119_1455`
(`bench/pixel_readout/results/20261119-1527-mixr_1119_1455/head.safetensors`, local). This file
holds the P4 wrap-up, the P5 topics and what is open.

## P4 wrap-up (done 2026-11-19)

Order (the user, 2026-11-19): "배포되면 프로젝트 모델 코어는 라이브러리에 편입하는 쪽으로
계획을 잡자. p5는 아마 픽셀로스 등 integration일텐데 배포 및 승격을 선행하게 될듯". Release,
then promotion of the model core into `library/`, then the P5 topics.

### 1. Release

- Done 2026-11-19: `mixr_1119_1455` uploaded to the private repo `sorryhyun/anima-pixel-decoder`
  (the user: private first, they edit the card later). Card source `deploy/README.md`; decode
  cost on turbo and Spectrum and the upload record in `reports/release_2026_11_19.md` (turbo:
  VAE swap 15 % of DiT + decode, head 7 ms).
- Left to the user: the card text, making the repo public.
- Open: the head's turbo point was trained with the Hub turbo repo's
  `old/anima_turbo_4step.safetensors`; on v3 / v4 it reads 31.50 / 31.40 against 34.61
  (`reports/release_2026_11_19.md`). A turbo point on v3 or v4 (`--points … :adapter`,
  ~7 min per 500 rollouts) is the fix; not run.

### 2. Promotion — the model core into `library/`

Done 2026-11-19 (the user: "승격 ㄱㄱ"):

- `library/anima/pixel_head.py`: `PixelHead`, `ConvNeXtBlock`, `load_pixel_head` (shape and
  trained σ ranges from the metadata), `lat_patches`, `from_patches`, `PATCH`, `PixelTap`
  (hooks on the DiT forward and `final_layer.linear`, CFG combine, decode), `out_of_range`.
  `bench/pixel_readout/` imports the model code from it.
- Catalog row `anima_pixel_head`, pack `anima-pixel-head` (`models/vae/`, fetched on first use;
  GUI pack strings in four languages).
- `inference.py --decoder pixel_head [--pixel_head <file>]` (`library/inference/pixel_decoder.py`)
  on the single, interactive and batch paths; `PIXEL=1` / `PIXEL_HEAD=` on `test-*` and
  `test-turbo`; `GenerationRequest(decoder=…, pixel_head=…)`. Warns outside the trained σ, on a
  Spectrum cached last step and under `--cfgpp` / `--smc_cfg`; refuses `--tiled_diffusion`;
  `--latent_path` falls back to the VAE with a warning.
- Façade: `anima_lora.models.{PixelHead, load_pixel_head, PixelTap}`,
  `anima_lora.inference.{ensure_pixel_decoder, pixel_decode, save_decoded}`.
- Tier 2: bench `e2e_speed.py`, test `tests/test_pixel_head.py` (CPU), doc
  `docs/anima/inference/pixel_head.md`.
- Check through the daemon (one seed, one prompt, head PNG vs VAE decode of the saved latent):
  base 28 steps CFG 4 46.2 dB, turbo 38.2 dB; no warnings.

- ComfyUI node: built and published 2026-11-19, GitHub `sorryhyun/ComfyUI-Anima-PixelDecoder`,
  registry `comfyui-anima-pixel-decoder` 0.1.0, linked from the card with colour / mono samples. `AnimaPixelHeadTap` + `AnimaPixelDecode`,
  `_vendor/pixel_head.py` via `make vendor-sync`. Turbo in ComfyUI: total 1.127 s (VAE Decode)
  vs 1.005 s (head), 36.91 dB (`reports/comfy_node_2026_11_19.md`). Not run in ComfyUI: the
  Spectrum KSampler, the base DiT at CFG 4.

## P5 topics

After the release and the promotion, on the library's head (the user, 2026-11-19: "기존 plan은
이제 p5에서 unnest해서 여러 주제를 적자. x2 or x4, rgba, pixel loss 등"). Each topic is its own
line of work with its own proposal; order not set. The first steps are Claude's drafts.

### ×2 / ×4 output

Its own plan: `x2_plan.md` (finished arms as pointers into `reports/`, the settled recipe and its
full run, the deployment plan, next arms). In short: the head
writes 32×32 px per token through a second 2× stage; arm A (`x2a_1119_1802`, regression onto the
released ResShift ×2 student's output) came out soft, 1.7 – 3.0 dB under the teacher run on the
release head's ×1 output; LPIPS and a tone term (`x2lt_1119_2013`) and the SR line's DMD / GAN
critic (`reports/x2_critic_2026_11_20.md`) did not close it. A ×2 file shipped as a test file
(`reports/x2_preview_release_2026_11_19.md`). Next, on the user's word (2026-11-20): arm B, the
real 2× image as the target at the base points, the turbo point left out of ×2 for now.

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
Idea and first probe (PixelGen-style LPIPS through the head; DWD, arXiv:2611.22287, the same family as our head): `idea.md`.

Step 1, the gradient-agreement probe (the user, 2026-11-21: "let's go ahead with item 1 and if
any, proceed to 2"; item 2 = a head over a wider σ range).

Facts (the design, set before the run):

- `bench/pixel_readout/lpips_grad_probe.py --label lgrad_1121_1845`: forward + backward, no step.
  16 of P1's held-out test images, base DiT on the fp32 stream, teacher-forced at σ 0.0577 / 0.1 /
  0.1364 (the heads' range) and 0.3 (outside it), two noise draws each, one 24-token window with a
  4-token margin per image, target VAE(x0), both decodes clamped.
- Arms: a = ∂LPIPS/∂v through the fp32 VAE decode of the window; a16 = the same through the bf16
  VAE; b = a head's gradient as v receives it under the row-space projection,
  (W Wᵀ)⁻¹ W ∂L/∂f; c = the head's raw ∂L/∂f against Wᵀ g_a. Heads: `mixr_1119_1455` (target
  VAE(x̂0)) and `realr_1119_2200` (target the original image + LPIPS at the base points).
- Read: cos(b, a) per head and σ, beside its two in-run references: cos(a16, a) and the floor
  cos(b on one draw, a on the other); cos(a, a) across the draws is what a perfect decoder reads on
  that floor. Also the norm ratio, the row-space share of c, PSNR of the head against VAE(x̂0), and
  seconds and peak GB of each backward. Added after the first smoke, before the full run: σ 0.5
  and 0.7, and a step check (each gradient's LPIPS decrease through the fp32 VAE at three step
  sizes, with the head's gradient from the other draw as its null).
- Bars: spread unknown, no earlier run on this quantity.
- Cost: minutes.
- Result (`reports/lpips_grad_2026_11_21.md`): `mixr` cos(b, a) 0.115 / 0.154 / 0.166 at σ 0.0577 /
  0.1 / 0.1364, floor 0.060 / 0.078 / 0.081, cos(a16, a) 0.511 / 0.495 / 0.434; `realr` level with
  it; row-space share of the head's gradient 0.032 – 0.038. By the reading rule below: not passed.
  The wider-σ head (item 2) was not trained.
- One 512² window backward: fp32 VAE 0.154 s / 4.33 GB, bf16 VAE 0.087 s / 2.46 GB, head 0.019 s /
  0.70 GB.

Interpretation (the reasoning for the arm):

- Question: does a head send LPIPS into v the way the VAE decode does? Yes → the head is the
  decode of the LoRA arm and the wider-σ head is trained with target VAE(x̂0). No → the wider head is
  not trained for this purpose; the windowed VAE decode (its cost is in this run) or a latent-only
  decoder carries the term.
- cos(b, a) in v space is the first-order share of the VAE path's LPIPS decrease that a step along
  the head's gradient buys, so it reads without a bar. Claude's reading rule, fixed before the
  result: "yes" if `mixr`'s cos(b, a) at the three in-range σ sits nearer cos(a16, a) than the
  floor; the numbers go to the user either way.
- After the result: the decode of an LPIPS term on a LoRA is the windowed VAE (bf16), at any σ, or
  a decoder whose only inputs are x_t and v. Next arm, proposed and not run: FM against FM +
  λ·LPIPS through the bf16 VAE window on one LoRA, λ from the measured gradient-norm ratio.

Step 2, the LoRA arm on ama mitsuki (the user, 2026-11-21: "그래 차라리 그게 낫겠다. ama mitsuki 로
해보자").

Facts (the design, set before the runs):

- Code: `library/training/pixel_lpips.py` (`--pixel_lpips_ratio` / `--pixel_lpips_weight` /
  `--pixel_lpips_measure`, σ gate 0.7, window 24 + 2 × 4 tokens), test `tests/test_pixel_lpips.py`.
  Per sample at σ ≤ 0.7: LPIPS-VGG of the bf16 VAE decode of x_t − σ·v on one random window against
  the same decode of x0; the term's gradient into v is scaled per sample to `ratio` × the FM
  gradient's norm.
- Arms, configs in `configs/` (linked into `configs/gui-methods/custom/`), seed 42, ama mitsuki
  (106 images, 424 updates), `configs/methods/lora.toml` as it stands plus `paired_step_rng` and
  `gradient_checkpointing`: `lpctl_1121_1900` (the term measured, not added), `lpr03_1121_1900`
  (ratio 0.3), `lpr10_1121_1900` (ratio 1.0).
- Smokes: at the default preset the first VAE decode ran out of memory with 14.34 GB allocated
  (job 20261121-185603); with gradient checkpointing 16 updates ran at ~0.86 s per micro-step, the
  term on 26 / 32 batches, λ 0.045 at ratio 1.0 (job 20261121-185722).
- Read: (1) the paired training logs, `pixel_lpips/lpips` at the logged steps of the last two
  epochs; (2) `bench/pixel_readout/lora_ab.py --label lpab_1121_1900`: the lora_recipe prompts
  p0 – p3 × render seeds 42 / 43 at the `make test` recipe, pair sheets control | arm, blind sets
  with the sides shuffled; the user's read decides. ‖dW‖ reported, not matched.
- Bars: spread unknown, one training seed.
- Result (`reports/lpips_lora_2026_11_21.md`): paired training-log LPIPS against the control,
  ratio 0.3 −0.0003 … −0.0007, ratio 1.0 −0.0026 … −0.0043 (control 0.153); ‖dW‖ within 0.7 %;
  render seeds 42 – 45, blind sets `s45_LPIPS03_ama` / `s45_LPIPS10_ama` pushed to
  `anima-blind-pairs` on the user's word. The user's blind read (2026-11-21), control – arm:
  ratio 0.3 5 – 11, ratio 1.0 11 – 5 (16 pairs each, one training seed; p1 7 – 1 to the control
  and p3 1 – 7 to the arm over both sets). Ratio 1.0 is not carried on; ratio 0.3 waits on a seed
  twin (control + ratio 0.3 at seed 43), proposed, not queued.

Interpretation (the reasoning for the arms):

- The ratio has no source: PixelGen's λ 0.1 is on a pixel x-prediction and does not translate, so
  0.3 and 1.0 bracket it. The per-sample ratio rule is Claude's choice over a fixed λ, because the
  term's gradient scales with σ and the FM gradient's norm moves with σ too.
- A visible gain on the user's read → a second training seed, then prompts outside the working set.
  Level or worse at both ratios → the term is parked; the ratio and the σ gate are the knobs left.

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
