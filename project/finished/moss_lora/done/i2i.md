# MOSS-Image i2i — a VAE reference on RoPE's `r` axis

Status: settled 2026-10-25 — the design record and probes behind the shipped i2i surface
(`GenerateRequest.ref_image`, `configs/moss/i2i/`; user doc `docs/moss/methods/i2i.md`). Proposed
2026-10-19, four probes measured. The colorize sub-line's kept run and OCR-caption arm:
`project/finished/moss_lora/colorize_ocr.md`; v3 recipe: `project/finished/moss_lora/README.md`.

## Premise

MOSS-Image has no image conditioning. Its 4D RoPE `(r, f, h, w)` keeps an `r` axis that
training always sets to 0. Qwen3.5's vision tower conditions MOSS zero-shot but, at 32 px
per token, brings back content, not line placement
(`docs/moss/findings/vision_tower_reference.md`). So a clean VAE latent of the condition
image goes into the single-stream sequence on `r = 1`, and a plain LoRA learns to read it.

## Mechanism

- **Sequence** `[image | ref | text]`. Ref tokens go through the image's `video_embedder`,
  get no noise, and take the t = 0 block modulation (`--ref_timestep`, default 0; `-1` =
  the step's, the pre-probe-4 behaviour).
- **RoPE**: ref at `r = 1`, the image's `f` (= text length), `h, w` stretched onto the
  target grid, `(i + 0.5)·H/h_ref − 0.5`, so a half-size ref lines up position for position.
- **Head and loss** read image tokens only; text RoPE and the key mask shift by the ref
  token count. At inference the ref goes into every CFG branch.
- **Off path**: `ref=None` matches the old forward, single and batched (tiny CPU model);
  with a ref, `forward` and `forward_batched` agree to 2.4e-7.
- **Files**: `library/moss/model.py` (`_RoPE.forward(ref_hw=)`, `embed_sample` /
  `forward` / `forward_batched(ref=)`), `generate.py` (`denoise(ref=)`), `cache.py` /
  `scan.py` (`*.moss_ref.safetensors`), `train.py`, `tune.py`; probe scripts
  `bench/moss_i2i_eval/colorize_{eval,wild}.py` (`_render.py`); the cache and the
  vision-tower TE call shipped as `library/moss/colorize.py` / `vision_te.py`.

## Probe 1: EasyControl colorize, 1-hour budget

1000 pairs (synthetic mangafied page → colour original), 8 held out. Both runs: TE input =
the page through the vision tower + the colour caption (`filter_to_colors_and_protected`);
LoRA rank 32 on q/k/v/o/ffn, lr 1e-4, 1600 steps, `--plain_caption_rate 0.3`. The ref run
adds the page as a half-size VAE latent (720–1080 tokens). v1 58.6 min at 2.16 s/step;
ref 71.3 min at 2.67 s/step (swap 36, budget 0.2). Eval: Base + LoRA, ER-SDE + Spectrum,
30 steps, CFG 3, seed 1; ref to the LoRA arms only.

| Arm | 64 px luma PSNR | luma r (128 px) | mean chroma | chroma error vs target |
|---|---|---|---|---|
| target | — | 1 | 16.1 | 0 |
| B&W page, copied as is | — | 0.994 | 0.0 | 16.1 |
| base, page + colour caption | 9.96 | 0.25 | — | — |
| v1 (vision tower only) + caption | 11.28 | 0.42 | 15.3 | 16.4 |
| v1 page only | 11.30 | 0.40 | — | — |
| **ref + caption** | **25.62** | **0.97** | 17.2 | **13.6** |
| **ref, page only** | **24.77** | **0.96** | 17.2 | 18.9 |

Chroma = mean CbCr distance from grey at 1/8 size; chroma error = mean CbCr distance to
the target.

- **Alignment comes from the ref**: every held-out page 22.6–28.3 dB (v1 8.6–14.1); line
  work, faces, hands, lettering and background match the page; v1 redrew the layout.
- **It colours, it does not copy**: with the caption the chroma error beats grey (13.6 vs
  16.1); without it the colours are plausible but not the target's (18.9).
- Limits: one seed, 8 pages; Base gets no ref, so the fair comparison is v1 vs ref.

## Probe 2: colour manga pages, lr 5e-5

The same 1000 pairs + 500 pages of EasyControl's `manga_color` slice (hitocrawl, JA / EN /
KO editions; `colorize_cache.py --manga 500`); holdout the same 8 + 4 manga works, no page
of a held-out work trains in any edition. Ref half size, lr 5e-5, 2 epochs = 3000 steps,
137 min at 2.75 s/step (swap 28, budget 0.15 from `tune_swap`). Epoch means 0.309 / 0.278
(ref_v1 0.302 / 0.276; datasets differ, shapes only). Eval as probe 1, 12 held out, by eye
vs ref_v1:

| | manga run vs ref_v1 |
|---|---|
| Lettering on the manga pages | Better. `889354_en`: "YOU ARE SO CUTE, NANOHA." and "OOH YOU SQUIRTED." read right; ref_v1 gives "YO U ARE", "DON YOU SQUIRTES". Small balloons garble in both. Red balloon lettering comes out black in both |
| Hair colour on the manga pages | Closer on 2 of 3 works (orange `889354`, blonde `403801`); ref_v1 is closer on `623751` |
| The 8 illustrations, with caption | Even. One clear miss: `eight_tohyama` gets two green-haired characters where the target has one purple, one green |
| Page only | More conservative: near the page's grey where hue is ambiguous, lower saturation; ref_v1 picks a vivid, often wrong, hue |

Limits: one seed, 12 pages, by eye. The half-size ref is the likely limit on small text
(untested).

## Probe 3: real B&W pages, no target

8 pages from `comfy/input/to_colorize{,2}` never through Mangafy (4 Korean manga pages, 4
grey-shaded illustrations), page only, ref_v1 and the manga run, seed 1
(`colorize_wild.py`). Luma r against the page; chroma as above (colorize targets average 16).

| | luma r, manga / illus. | chroma, manga / illus. |
|---|---|---|
| ref_v1 | 0.90 / 0.94 | 3.1 / 12.4 |
| manga run | 0.91 / 0.97 | 3.0 / 7.7 |

- **Alignment holds off-distribution** (panels, black fills, screentone, balloon
  placement); no degradation augmentation needed. On the Korean pages the manga run is
  clearly ahead (lettering, faces, no background wash).
- **The manga pages barely colour** in both: real pages are sparser than Mangafy's output
  and nothing fixes hue without a caption (a colour caption is the untested lever).
- On the illustrations ref_v1 colours vividly but invents (cat ears, a teal tail, a
  starred background); the manga run stays faithful but desaturated.

## Probe 4: the reference at t = 0

Ref tokens shared the image's block modulation, so a clean latent was modulated as noised.
`--ref_timestep 0` gives them the t = 0 modulation (shift, scale, both gates) in every
block; image and text keep the step's. Arm: `ref_v1`'s request (`colorize_ref`, 1000
pairs, 1600 steps, lr 1e-4, rank 32, ref 1/2) with only `--ref_timestep 0`, vs
`moss_colorize_ref_v1` (not retrained). 71.9 min at 2.70 s/step; epoch means 0.290 /
0.270 (ref_v1 0.302 / 0.276; epoch 1 order differs, read epoch 2). 12-page holdout, seed 1.

| Arm | 64 px luma PSNR, 8 illus. / 4 manga | luma r | chroma error, 8 illus. |
|---|---|---|---|
| ref_v1 + caption | 25.62 / 23.73 | 0.966 | 13.6 |
| **t0 + caption** | **27.27 / 26.10** | **0.975** | **13.3** |
| ref_v1, page only | 24.77 / 23.34 | 0.964 | 18.9 |
| **t0, page only** | **26.58 / 25.96** | **0.974** | **16.7** |
| manga run + caption (more data, other lr) | 27.49 / 26.74 | 0.975 | 13.0 |

- **t = 0 wins**: ahead on 11 of 12 pages with the caption, within 0.2 dB of the manga run
  on the illustrations at 1000 pairs / 1600 steps. `yomu` gains 8 dB (pink background
  where ref_v1 painted blue); without it +0.7 dB on the illustrations, +2.4 on manga.
- By eye: `eight_tohyama` keeps one purple- and one green-haired character (the manga run
  made both green); `ru_zhai`'s background turns warm as in the target; `ama_mitsuki`'s
  hoodie turns blue where the target's is beige.
- Limits: one seed, 12 pages.
- **Now the default**: `--ref_timestep` 0; LoRAs from before the flag carry no stamp and
  load with the shared timestep.
- **Not measured**: `--ref_isolate` (off by default) makes ref tokens attend only to
  themselves (image and text still attend to them). With t = 0 the ref's per-block K/V
  then no longer depends on the step, so it could be computed once per image and reused
  across steps and CFG branches. Stamped in the LoRA metadata like `ref_timestep`.

## Settled

- The vision tower stays in the TE input; no text-only-TE arm.
- Ref size is capped at 1/2 per side; no size arms.

## Open questions, in order

1. **Ref guidance**: arms with the ref dropped from uncond (image CFG) and a separate ref
   guidance scale (InstructPix2Pix-style); training needs ref dropout.
2. **Other tasks on the same path**: lineart / sketch → colour, edits (needs pairs), 2× SR
   (a half-size ref is an SR condition; check against the finished ResShift line in
   `project/finished/`).
3. **Eval protocol**: more seeds, a larger holdout, a safe-only holdout for anything
   published, real B&W pages (`colorize_wild.py`).
4. **Ship surface** (after 1): `GenerateRequest(ref_image=, ref_scale=)` with VAE-encode in
   the request path; a ref input on the ComfyUI MOSS node; `reference_latents` / ref scale
   stamped in the LoRA metadata (train.py already writes `reference_latents`) so inference
   can refuse a mismatch; a cache stage for refs outside the probe script.

**Gate before the ship surface**: alignment holds on a second task (lineart or SR).
