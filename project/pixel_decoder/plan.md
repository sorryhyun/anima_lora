# Plan (2026-11-19, updated 2026-11-22)

What is ahead on the pixel decoder: the ×1 recipe as confirmed, the GAN finish that is to become
the ×1 release, the P5 topics and what is open. What ran is in `reports/` (index:
`reports/README.md`):

- P1 – P4 and their recipe `reports/phases_p1_p4_2026_11_19.md`; the P4 wrap-up (release,
  promotion, node) `reports/p4_wrapup_2026_11_19.md`;
- the release head `realr_1119_2200` `reports/real_target_2026_11_19.md`;
- the ×1 batch `1121_2230` and `x1ll_1122_1000` `reports/x1_recipe_2026_11_22.md`;
- pixel losses into v `reports/lpips_grad_2026_11_21.md`, `reports/lpips_lora_2026_11_21.md`;
- ×2: `x2_plan.md`.

## State

### Facts

- The user (2026-11-22): "x2 하듯이 마지막에 gan 마무리 하는걸 배포본으로, 그리고 x1ll 은 일종의 base
  느낌으로, x2나 추후 rgba 용도 등으로 사용하면 어때?"
- `x1ll_1122_1000` is done, training and reads, and is the base (the user, 2026-11-22: "ㅇㅋ 이거 베이스로
  확정하자"): `output/pixel_head/x1ll_1122_1000/head.safetensors` (local).
- Two ×1 files follow from that:

  | role | head | used for | state |
  |---|---|---|---|
  | base | `x1ll_1122_1000` | the init of the GAN finish, of ×2 and of RGBA | done, fixed as the base; local only |
  | release | the base + a GAN finish | Hub `anima_pixel_head.safetensors`, the catalog, the node | not run |

- Until the user picks a finish, the Hub ×1 file stays `realr_1119_2200` and the Hub ×2 file
  `x2bgan2_1121_1340`.

### Interpretation

- The split fits what each file is for. The stages built on the head (×2, alpha) regress or bring
  their own GAN finish, so they want the converged regression head with no painted texture in its
  trunk. The decoder that people look at can trade some dB for texture.
- A finish is not a fix of the fine strokes: on ×2 it painted texture over the merged area and
  did not recover the strokes (`reports/x2_tile_gft_2026_11_22.md`). That is the form probe's
  question (Open).

## The ×1 base recipe

### Facts

`bench/pixel_readout/convnext_head.py`, the argv of `x1ll_1122_1000`:

```bash
make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/convnext_head.py --compile \
  --no_skip --ls_init 1.0 --up 1 \
  --init mixr_1119_1455 --points 0.0577-0.1364:real 0.5:adapter --target xhat0 \
  --adapter output/ckpt/release/anima_turbo_4step_v4_longer.safetensors \
  --lambda_lpips 0.05 --pages post_image_dataset/pixel_pages --page_share 0.33 \
  --replay 64 --opt_steps 16 --lr 5e-5 --rollouts 12000 \
  --eval_points 0.03 0.25 --eval_every 1000 --no_start_eval --n_sheets 4 --label <name>"
```

| part | value | record |
|---|---|---|
| form | width 1024, 2 ConvNeXt at token spacing, a 2× pixel shuffle, 1 ConvNeXt, no skip | `upsample_2026_11_19.md`, `p2_skip_2026_11_19.md` |
| init | `mixr_1119_1455` (target VAE(x̂0)) | `p4_mix_2026_11_19.md` |
| base points | σ 0.0577 – 0.1364, target the original image, LPIPS 0.05 | `real_target_2026_11_19.md` |
| turbo point | σ 0.5 with v4_longer, target VAE(x̂0), in turn with the base point | `real_target_2026_11_19.md` |
| pages | share 0.33 | `pagemix_2026_11_19.md`; 0.1 was run and not taken up, `x1_recipe_2026_11_22.md` |
| schedule | LR 5e-5, one cosine over 12,000 rollouts, 16 updates each from a 64-rollout replay | `x1_recipe_2026_11_22.md` |
| σ range | not widened | `x1_recipe_2026_11_22.md` |
| stream | fp32, DiT and head compiled | `replay_recipe_2026_11_19.md` |

At σ 0.0577 against the original image, dB / LPIPS (spread: the `realr` / `x1twin` gap of two
3,000-rollout runs, test 0.04 / 0.0013; unknown at 12,000):

| head | test 30 | text 8 | manga 10 |
|---|---|---|---|
| `realr_1119_2200` (Hub) | 33.75 / 0.0807 | 28.61 / 0.0671 | 26.02 / 0.1020 |
| `x1long_1121_2230` (LR 2e-4) | 34.13 / 0.0741 | 29.06 / 0.0596 | 26.32 / 0.0962 |
| `x1ll_1122_1000` (base) | 34.25 / 0.0725 | 29.20 / 0.0580 | 26.42 / 0.0959 |
| VAE(x̂0) | 35.11 / 0.0560 | 30.29 / 0.0375 | 26.95 / 0.0618 |

### Interpretation

- Length and the lower LR add up. The VAE path stays 0.86 dB and 0.0165 LPIPS ahead on test;
  texture, the LPIPS side of that gap, is what the finish is for.

## The GAN finish on the ×1 base

Not queued, and the code for step 1 is not written.

### Facts

Records it rests on:

- The form, from ×2 (`reports/x2_tile_gft_2026_11_22.md`): the PE-Spatial term at gradient ratio 2
  on a converged head, from rollout 0 of a new cosine, 1,000 rollouts = 16,000 GAN updates. No
  pinwheel on two seeds. At 48,000 updates (`x2gftl_1121_2300`) parallel stripes came back while
  LPIPS improved. With the term from rollout 1000 on an unconverged head (`x2bgan2_1121_1340`):
  pinwheel tiles.
- Its cost on ×2: `x2gft_1121_2225` against `x2b4k_1121_0950` 30.45 → 29.74 dB, LPIPS
  0.141 → 0.161 on test; 1.35 s per rollout against the control's 0.83, peak 9.9 GB.
- A re-warmed cycle without the term does not return to its start in 1,000 rollouts at LR 2e-4
  (`x2gctl_1121_1120`), so an arm is read against a control of the same cycle.
- On ×1 a warm start at LR 5e-5 dips a quarter as deep as at 2e-4
  (`reports/x1_recipe_2026_11_22.md`).

Step 1, code and smoke:

- `convnext_head.py` has no GAN term. Add `--pegan_ratio` (default 0), `--pegan_crop` 512,
  `--pegan_lr` 1e-4, on `PEDisc` and the weight rule of `library/anima/pixel_head_train.py`:
  - the term only on `:real` samples (real = the original image, pages included); turbo samples
    keep their regression alone;
  - the weight set once per rollout, on a `:real` replay sample, against rec + 0.05 · LPIPS;
  - a paired crop of min(512, side) pixels; the ratio and the source head stamped in the file.
- Smoke, 100 rollouts from the base. It answers the preconditions that no record covers:
  - peak GB: the ×1 loop keeps 64 whole images on the card where ×2 keeps windows;
  - the weight stays under its cap of 1.0 (×2: 0.041 → 0.006);
  - the discriminator's margin at rollout 50 (×2: 3.6). Its LR 1e-4 was measured beside a head
    at 2e-4, here the head is at 5e-5;
  - seconds per rollout.

Step 2, three runs, configs or argv recorded here before they are queued:

| run | parent | change |
|---|---|---|
| `x1fctl_<mmdd_hhmm>` | `x1ll_1122_1000` | `--init` → `output/pixel_head/x1ll_1122_1000`, `--rollouts` 12000 → 2000, `--eval_every` 1000 → 500 |
| `x1fgan_<mmdd_hhmm>` | `x1fctl` | `--pegan_ratio` 0 → 2, nothing else changed |
| `x1fganw_<mmdd_hhmm>` | `x1fgan` | `--seed` 0 → 1 |

- 2,000 rollouts because the points alternate: 1,000 of them are base rollouts, and about 16,000
  of the 32,000 updates carry the term, the count of `x2gft`.
- Changed together against `x2gft` and not separated: the head LR (5e-5), the crop's reach (512 px
  at ×1 covers twice the image side it covers at ×2), pages in the mix, the turbo point in turn.

Step 3, the read:

1. Blind pairs, one (prompt, seed) per file, on the line's working set (P3's 16 prompts × seeds
   0, 1 at base 28 steps CFG 4 and at turbo v4_longer 4 steps; the monochrome and fine-stroke
   renders): control against finish (does the term add anything), and Hub `realr` against finish
   (does it ship).
2. Labelled crops for regular patterns (pinwheel, stripes, weave): the monochrome render's dot
   screen, the held-out manga pages, flat colour areas.
3. Numbers beside them, with no bar: held-out dB and LPIPS at every point, head against VAE on
   the renders, high-pass amplitude and correlation as in the report's corrections, chroma on
   greyscale pages. Spread: the `x1fgan` / `x1fganw` gap.
4. The ship read, on a set no arm has been read on, seeds 7 and 8, base and turbo:
   - `safe, 1girl, solo, knitted sweater, plaid skirt, autumn park, fallen leaves`
   - `safe, no humans, old bookstore, wooden shelves, books, window light`
   - `safe, 1boy, solo, detailed fur collar, snow, night, city lights`
   - `safe, no humans, meadow, tall grass, wildflowers, cloudy sky`
   - `safe, monochrome, comic, screentone, 1boy, solo, school hallway`
   - `safe, monochrome, greyscale, hatching, old man, beard, portrait`
   - `safe, monochrome, comic, speed lines, 1boy, running, motion blur`
   - `safe, monochrome, comic, screentone, no humans, street, power lines, japanese text`

The user's read of 1 and 4 decides.

Step 4, after the user picks a finish:

- The finish replaces Hub `anima_pixel_head.safetensors`; card, docs
  (`docs/anima/inference/pixel_head.md`) and node card follow. A copy already fetched keeps the
  same file name, so the local one is replaced by hand and the release note says so.
- The base does not go under the catalog's name. `scripts/pixel_head/train_x2.py` with an empty
  `init` loads the catalog's head, which would then be the GAN file: every stage trained on the
  base names its path in `init`.

Cost, estimated from the records: smoke about 5 minutes; control about 25; each GAN run 30 – 40;
reads about 25. About two hours, queued on the user's word.

### Interpretation

Question: does a short GAN finish on the converged base give texture that reads better than both
the control and the Hub file, without a regular pattern and without hurting turbo?

- Yes → the finish is the release (step 4).
- A regular pattern → ratio 1 at the same length. Not a longer run: on ×2 length is what brought
  the stripes.
- No pattern, level with the control → the term adds nothing at this form. The user picks the
  base or `realr` for the Hub, and the form probe is next.
- Better on base, worse on turbo → the shared trunk carries the term into the turbo point. The
  next arm puts the term on that point too, with VAE(x̂0) as its real.

## P5 topics

After the release and the promotion, on the library's head (the user, 2026-11-19: "기존 plan은
이제 p5에서 unnest해서 여러 주제를 적자. x2 or x4, rgba, pixel loss 등"). Each topic is its own
line of work with its own proposal; order not set. The first steps are Claude's drafts.

### ×2 / ×4 output

Its own plan: `x2_plan.md`. Next there: B only from the ×1 base, longer, then a short GAN finish.

### RGBA

Init: the ×1 base, not the release. An alpha channel on the head's output is cheap to add, and
labels can come from real RGBA cutouts composited onto random backgrounds (exact alpha) or a
matting model on generated images. What the head alone cannot give: the DiT renders a composite,
so semi-transparent regions (hair tips, glass, smoke) carry no background-free foreground colour;
LayerDiffuse tunes the DiT for that. The baseline to beat is post-hoc matting (BiRefNet,
anime-segmentation) on hair-edge sheets.

### Pixel losses on the head

The GAN finish above is this topic's current arm. Behind it: a ResShift 1-step refiner from the
head's output, read against the GAN arm.

### Pixel losses through the head, into the DiT

The head as the decode for losses that train the DiT or a LoRA did not pass its gradient probe
(`reports/lpips_grad_2026_11_21.md`); such a term decodes through the windowed bf16 VAE
(`library/training/pixel_lpips.py`). On the LoRA, the user's blind read put ratio 0.3 ahead and
ratio 1.0 behind (`reports/lpips_lora_2026_11_21.md`). Left: a seed twin of control + ratio 0.3
at seed 43, proposed, not queued. Other uses named for it: cjk_rl arm D (its D0 cost probe
rerun against the VAE-window path), a term for the turbo
student. Idea: `idea.md`.

### A head at every step

Spectrum's cached steps (σ 0.90 … 0.265) read a Chebyshev-forecast feature at the tap, and
cjk_rl arm D reads x̂0 at steps 0 – 1; both want a head over σ in [0, 1]. `mixr_1119_1455`
covers σ 0.058 … 0.136 on base and 0.5 on turbo. Use: a preview at every step.

### MOSS

A head on MOSS-Image (different DiT, 128-channel latents at /16, the same 16-px token); decode
memory is the motive there (the fp32 VAE decode does not fit beside the DiT). Nothing run.

## Open

- The form probe, not queued: can the ×1 form fit fine strokes at all? A 16-window overfit on
  fine-stroke crops with the VAE decode as target, the ×1 form beside one with a pixel-shuffle
  stage (`bench/pixel_readout/x2_overfit.py` is the ×2 version; minutes). It was planned before
  the ×1 head is fixed. If another form wins, the base is retrained and the finish redone, so it
  is cheaper before the finish than after.
- The base is local only, `output/pixel_head/x1ll_1122_1000/`. The bench's `--init` and `--head_run` take that
  folder's path: the bare run name resolves to the results folder, which no longer holds the head.
- Screentone: small dots smoothed into soft shading, a systematic difference from the VAE (the
  user's read on `mixr_1119_1455`, good enough to release). Read again on the finish.
- Tone: the head is darker than the VAE on textured regions (the user's read; `tone_probe.py`,
  `reports/p4_mix_2026_11_19.md`): about 2.5 levels in linear light on screentone-like blocks of
  mono renders on the base DiT, 1 on turbo, under 1 level over a whole image; flat regions level.
  The loss has no term on local tone.
- The turbo point is trained on v4_longer; other turbo files are outside it
  (`reports/release_2026_11_19.md`).
- Spectrum renders of `mixr_1119_1455` have numbers only (34.37 / 26.00); no sheet was made.
- Not run in ComfyUI: the Spectrum KSampler, the base DiT at CFG 4.
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
- Spread: one twin pair at 3,000 rollouts (`realr` / `x1twin`, which also differ in the held-out
  noise); none at 12,000.
- Grey tint on mono generations (Claude's read on `p3_1119_1056`): on teacher-forced greyscale
  pages the pagemix head adds about 0.5 level of chroma over the VAE (`reports/pagemix_2026_11_19.md`);
  generated mono images are not re-checked.
