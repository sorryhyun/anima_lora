# Qwen3.5's vision tower conditions MOSS-Image zero-shot

Measured 2026-10-19 on MOSS-Image Base. Probe: `_archive/bench/moss_vision_reference/vision_recon.py`
(TE call now in `library/moss/vision_te.py`); grid and per-cell numbers land in
`output/tests/moss_vision_recon/` (`grid.png`, `index.json`).

## Setting

MOSS's `text_encoder/` is the whole `Qwen3_5Model`, vision tower included: `visual.*`
is a 27-block ViT (width 1152) whose merger outputs 4096, the LM's hidden size. Patch 16
and a 2×2 merge give one LM token per 32×32 px, so a 1 MP image is ~1000 tokens. The
T2I pipeline never calls the tower, and `library/moss/loader.py` leaves it on the CPU.
MOSS ships no processor config. The probe uses Qwen3.5-9B's own
(`Qwen2VLImageProcessor`: patch 16, temporal patch 2, merge 2, mean/std 0.5) through
`Qwen3VLProcessor`, which expands `<|image_pad|>` and returns the `mm_token_type_ids`
that M-RoPE needs.

The MOSS report has no reference-image conditioning or editing
(`project/finished/moss_lora/report_notes.md`). The DiT's RoPE has an `r` axis reserved for
references, always 0 (`project/finished/moss_lora/README.md`). So nothing in training showed the DiT
an image-derived text stream.

The TE call is the pipeline's (chat template, `add_generation_prompt`,
`enable_thinking`, `last_hidden_state`, one prompt per forward, no padding), with the
image item first in the user turn:

```
<|im_start|>user\n<|vision_start|>[image tokens]<|vision_end|>{text}<|im_end|>\n<|im_start|>assistant\n<think>\n
```

## Run

Four safe-rated images from the resized tree (704×1488, 752×1376, 768×1360, 1440×720),
rendered at their own size. Base, no LoRA, ER-SDE + Spectrum, 50 steps, compiled,
CFG 3, uncond `''` skipping block 22, seed 1. Captions are layout JSON. Arms:

| Arm | TE input | TE rows |
|---|---|---|
| `cap` | caption only (the T2I baseline) | 108–199 |
| `img` | image only, at the output's pixel count | 1001–1024 (989–1012 image) |
| `img_lo` | image only, at 384² | 140–148 (128–136 image) |
| `img_tmpl` | `img` with the image-token rows dropped before the DiT | 12 |
| `img_cap` | image, then caption | 1099–1191 |
| `img_desc` | image, then "Describe this image in detail." | 1007–1030 |

## Observations

- **The image alone gets the content back.** With no caption (`img`), all four images
  come back with the same characters, outfits and hair colours and roughly the same
  scene, down to who stands on which side and held objects.
- **It is not pixel-aligned.** Positions, proportions and line work drift. Skin went
  much darker on the kneeling figure. A 64 px luma PSNR against the source sits at 7.7–12.0
  on every arm, and the image arms do not beat the caption baseline consistently:

  | Image | `cap` | `img` | `img_lo` | `img_tmpl` | `img_cap` | `img_desc` |
  |---|---|---|---|---|---|---|
  | dan_6768037 | 11.88 | 10.99 | 10.27 | 11.45 | 12.03 | 10.79 |
  | 13133868 | 11.46 | 11.51 | 10.61 | 9.70 | 11.75 | 11.71 |
  | 9859367 | 8.02 | 9.01 | 8.69 | 8.59 | 10.76 | 9.13 |
  | 11115380 | 7.67 | 10.05 | 8.50 | 8.95 | 9.79 | 10.25 |

  At 64 px this measures colour layout more than content; by eye the image arms are much
  closer to the source than `cap`.
- **The DiT reads the image-token rows.** Without them (`img_tmpl`), all four come out
  off-topic (a poster, a game UI screen, a different character): the 12 template rows saw
  the image through the LM's attention but carry no summary the DiT can use.
- **Resolution matters.** At ~130 image tokens (`img_lo`) the content gets looser
  than at ~1000: the kneeling figure comes back pale and in a different crouch, and the cake
  scene garbles the second character.
- `img_cap` is the closest by eye overall. Adding text after the image (`img_desc`)
  changes little.
- Cost: ~1000 extra text rows at 1 MP, so the joint sequence grows by about a quarter.
  The TE pass with the tower resident took about 0.4 s per prompt.

## Hypotheses (untested)

- Qwen is causal, so the image rows see only the tokens before them (`user\n`). Text
  after the image changes the trailing rows only, which may be why `img_desc` ≈ `img`.
  An instruction or colour spec placed *before* the image would shape the rows the DiT
  actually reads.
- At 32 px per token the stream carries content and coarse layout, not line placement, so
  a structure-preserving task (colorize, edit) needs a LoRA that learns alignment from these
  tokens, or a VAE reference in the sequence (the free `r` RoPE axis is the slot for one).
- The probe is one seed on Base. SFT was not run.

## Follow-up (measured 2026-10-19)

On colorize, a LoRA trained on these tokens alone reached luma r 0.42; adding a half-size
VAE reference on `r = 1` reached 0.97 (`project/finished/moss_lora/done/i2i.md`).

## Follow-up: the image rows as a detector for sanitize (measured 2026-11-14)

`bench/moss_i2i_eval/vision_detect_probe.py`, run `output/tests/vision_detect_probe/20261114-1411`:
189 staged sanitize pairs, INT8 TE, a linear probe on the image rows of the texted page for
"the clean twin differs in this 32 px cell", 5 folds over artists. Held-out AP / recall of
faint cells at 1 % false positives:

| features | rows added per page | AP | faint recall |
|---|---|---|---|
| image rows, image alone | 1,007 | 0.669 | 0.094 |
| image rows, sanitize caption before the image | 1,007 | 0.665 | 0.090 |
| image rows, neutral caption before the image | 1,007 | 0.661 | 0.085 |
| image rows at a quarter of the pixels | 242 | 0.496 | 0.027 |
| the cached reference latent, 2×2 cells, read linearly | 0 | 0.708 | 0.229 |

The order of caption and image and the caption's content change nothing a location probe
reads. The rows are behind a linear read of the latent the DiT already holds, on faint cells in
all five folds. On the eval pages the probe fires on `usual_case1`'s handle (which v3 leaves)
among 13–23 % of the page's cells, and not on `hard_to_sanitize3`'s signature. No LoRA was
trained with `ref_vision` on sanitize.
