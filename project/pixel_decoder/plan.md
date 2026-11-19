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

### P3 — generated images and adapter transfer

`p3_1119_1056` (`reports/p3_2026_11_19.md`): 16 prompts × 2 seeds, standard sampler, target
VAE(x_final). Colour 30.90 on base, 31.24 with `anima_merged15`; mono 22.51 / 22.85; the
CFG-combined feature reads level with the cond feature (+0.03 dB). Real manga pages
(`manga_1119_1052`): the VAE alone is 21.5–39.8 dB from the original, VAE(x̂0) 24.3–36.3 from
VAE(x0).

Claude's call (2026-11-19, sheets sent to the user, standing unless flagged): the head
generalizes to generated images and to the LoRA; what is left is dense fine strokes (screentone,
hatching, fur lines), on generated and real images alike. The reasoning is in the report's
interpretation. P3 continues as the training strategy for manga and greyscale below.

## Recipe (settled unless a run says otherwise)

- No skip, out at default init, layer scale γ 1 (`--no_skip --ls_init 1.0`). The three were
  changed together and are not separated; no run is planned to separate them.
- Loss 0.5 L1 + 5 MSE + 0.05 L_grad, AdamW, cosine, clip 1.0, one DiT forward and one VAE decode
  per rollout.
- LR 2e-4 at batch 1, 4 updates per rollout (`0938`). `convnext_1119_0950` (gradient over 4
  rollouts, LR 1e-3, the user's arm) ended level with it: test 32.96 against 32.93, text 27.46
  against 27.45, after a faster start (27.57 against 25.69 at rollout 600).
- One 2× upsample stage before the third block (`--up 1`, the user's arm): `convnext_1119_1106`
  test 33.64 / text 28.04 against `0938`'s 32.93 / 27.45, 38 / 38 images up, same speed, 24.9M
  parameters; spread unknown. Adopted 2026-11-19 (Claude's call, sheets sent, standing unless
  flagged). Current head: `results/20261119-1151-convnext_1119_1106/head.safetensors` (local).

## Next

### 1. P3 — training strategy for manga and greyscale

Each step is one ≈ 27-min arm on the settled recipe, parent `convnext_1119_1106`.
Order, cheapest first:

- Target VAE(x̂0) in place of VAE(x0). x̂0 = x_t − σ·v from the rollout's own forward, so the VAE
  decode count stays one. On manga VAE(x0) is out of reach from σ_last features (VAE(x̂0) vs
  VAE(x0) 24.3–36.3 on the 10 pages), and when sampling the target is VAE(x_final), which is
  VAE(x̂0). Read: real held-out test / text against both targets, the P3 mono prompts.
- Manga and greyscale pages in the training mix. Pools: Manga109-s (87 titles, 8,519 images,
  `/media/sorryhyun/new/dataset/manga109s`), hitocrawl JA pages (13,275, `…/hitocrawl/retrieved`),
  the 68 mono-captioned images already in the anima-base set. Held out: the 4 Manga109 titles and
  6 hitocrawl artists in `manga_1119_1052`. Pages resized to ~4100 tokens, latents precached once,
  one fixed tag caption (`monochrome, greyscale, comic, speech bubble, japanese text`) to start.
  A share of about one rollout in three is a starting point, not a measured value.
- Eval set (fixed before the first arm): the 6 mono generations of `p3_1119_1056` (base) and the
  26 colour generations (regression), the 10 real pages of `manga_1119_1052` against VAE(x̂0) and
  the original, the P2 test / text sets. Sheets labelled; the user's read decides.

Built and checked 2026-11-19: `build_pages.py` → `post_image_dataset/pixel_pages/` (local;
3,000 train pages, 1,500 Manga109-s + 1,500 hitocrawl JA, seed 0, plus the 10 eval pages with
original pixels; 1.6 GB, 471 s). `smoke_pages` (20 rollouts) ran the VAE(x̂0) target, the page mix
(9 / 20 page rollouts at share 0.33) and the manga eval; its VAE vs original and VAE(x̂0) on the
eval pages match `manga_1119_1052` within 0.1 dB. Arms queued by the user in a later session:

```bash
# arm 1: target VAE(x̂0), parent convnext_1119_1106, pages scored but not trained on
make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/convnext_head.py --compile \
  --no_skip --ls_init 1.0 --up 1 --target xhat0 --pages post_image_dataset/pixel_pages \
  --page_share 0 --label convnext_mmdd_hhmm"
# arm 2: arm 1 + pages in the mix
#   … same flags with --page_share 0.33
```

The sampled-latent arm from the old P3 entry (train on sampled final latents) stays in reserve:
P3 did not break on generated images.

### 2. P4 — σ-conditioned head

Needed by cjk_rl arm D and turbo: σ over [0, 1], target
VAE(x̂0) at that σ, read at σ ∈ {0.06, 0.5, 0.9}.

### 3. P5 — pixel losses through the head

GAN on the head, arm D's reader NLL through the head,
or a ResShift refiner. Separate proposals once P3's manga arms and P4 read out.

## Open

- A longer run on the settled recipe: `0938` was still rising when its schedule ended
  (+0.29 dB on test over the last 600 rollouts). Folded into the manga arms, which can run
  longer once the mix is fixed.
- The features are taken under bf16 autocast; whether their rounding costs readout PSNR is
  unchecked.
- No spread for any trained head: one seed per run. A seed twin of the recipe is owed before a
  gap under about 1 dB is read as a gain. P3's seed-to-seed gap of one prompt (mean 1.98 dB) is
  content variation between images, not a run spread.
- Grey tint on mono generations (Claude's read on `p3_1119_1056`): check whether the manga mix
  removes it before treating it on its own.
- MOSS: nothing until the Anima phases read out.
