# caststation casbox: held objects, OCR boxes, and self-distillation on top

Closed 2026-11-15 on channel (caststation), 30 images. Two caption changes on top of the thin
MOSS-format captions (`casthin_1115_1150`, the control) and one recipe change on top of those:

1. **`mask_dilate 16`**: the position stage's crop tagger sees each subject mask grown by 16 px,
   so an object in hand reaches its holder's clause.
2. **OCR boxes on text lines** (`text_boxes`): each OCR'd line of a text clause carries its box.
3. **Self-distillation** (balanced hybrid95, `configs/moss/selfdistill.toml`) over captions 1 + 2.

Captions 1 + 2 trained on flow matching alone (`casbox_1115_1226`) are the line. Self-distillation
over them (`casboxsd_1115_1230`) softens the renders. The probe shows why: the flow-matching
student is already below the Base teacher's flow loss at σ 0.95.

## 1. Held objects: `mask_dilate`

The position stage tags each subject on a crop with every pixel outside its SAM3 person mask
blanked white, so a neighbour's hair cannot leak into the clause. An object in hand lies outside
the person mask and was blanked with the background: channel6's `drinking straw` scored 0.85 on
the full image and ≤ 0.001 on every crop, `holding phone` 0.93 against 0.36 on the best crop.
Those tags stayed in the shared bag.

`--mask_dilate N` (anime_tools 0.10.0) grows the mask by N px with an elliptical kernel before
blanking, never into another subject's mask, and the crop box follows the grown mask. At 0, 16,
32 and 64 px on the stored label maps, 16 px already brought the phone back on 11429918 and 32 px
on 9146144. Wider radii mainly let in background and unmasked bits of neighbours (a halo, a
neighbour's held item). A phone held above the head was still out of reach at 64 px.

Position dry run at 16 px, against the same run without it (`thin_all`):

| | without | 16 px |
|---|---|---|
| tags moved out of the bag | 773 | 804 |
| clause tags | 1299 | 1356 |
| novel tags | 0 | 0 |

Every held-object tag in the set moved into a clause:

| Image | Without | 16 px |
|---|---|---|
| channel6 | `holding phone` → middle, `drinking straw` in the bag | `holding phone` → left, `drinking straw` → middle |
| 12248609 | `holding phone`, `holding` in the bag | both → left |
| 11429918 | in the bag | bottom middle |
| 8562493 | in the bag | left and right |
| 9146144 | `holding phone` in the bag | left |

All 27 multi-subject captions changed somewhere (+152 / −129 tags by scope). The losses include
tags such as `arm tattoo` and `upper body`, whose scores move once the crop holds some background.

Applied as `CaptionsRequest.mask_dilate` (default 16, `configs/moss/captions.toml`).

## 2. OCR boxes

MOSS reads text boxes as `abs_centralized` `[y0, x0, y1, x1]`: pixels with the origin at the image
centre. Before this, only a `{stem}.layout.txt` published by anime_tools' Export carried them; the
cache's own conversion wrote bare text lines in both formats.

`CacheRequest.text_boxes` (on by default) reads each image's sidecar from the OCR tree
(`post_image_dataset/ocr`, which mirrors the resized tree; a symlinked image is followed). It
scales the box from the image the OCR read to the canvas the cache encodes (cover scale, centre
crop), then writes it:

- `tags`: after the quote, `Japanese text reads as "こいつ…" [202,126,280,154].`
- `layout_json`: as the text item's `bbox`, with the top-level `bbox_coord_space` / `bbox_format`.

On caststation, 12 of 30 captions have OCR. 24 of their 25 text lines got a box. The remaining one
(12248609, `ミン`) has no sidecar entry. 12248742's boxes match the example in anime_tools'
`layout_json` docs, which Export produced from the same sidecar.

## Runs

Both runs use `casthin_1115_1150`'s recipe: 480 steps, rank 32, Muon, int8 stream, the same
latents (copied), `caption_format tags`, `prompt_tail sure`. Configs are in
`project/pe_lora/configs/`.

| Run | Captions | Target | Last-step loss |
|---|---|---|---|
| `casthin_1115_1150` | thin | flow | 0.270 |
| `casbox_1115_1226` | thin + `mask_dilate 16` + OCR boxes | flow | 0.244 |
| `casboxsd_1115_1230` | same as casbox | balanced hybrid95 (Base teacher, σ ≥ 0.95, replace) | 0.219 |

A single step's loss does not rank runs, and casboxsd's is partly a different loss.

## Renders

`output/tests/caststation_thin/<run>_{anima,clause,text}`: four prompts × two seeds per set.
`anima` and `clause` are the casthin evaluation sets. `text` (`prompts_text.txt`) writes
boxed text lines: the same Miku prompt with the box at the top and then at the bottom, a vertical
note beside Neru, and two name labels over a height chart.

- **Text placement:** all three LoRAs put the text inside the requested boxes, casthin included,
  though it never trained on a boxed caption. So the inline form is readable to MOSS as written.
- **Sharpness** (24 renders paired by prompt and seed):

| Pair | Laplacian variance, median ratio | High-frequency share (> ¼ Nyquist), median ratio | Lower in |
|---|---|---|---|
| casboxsd / casbox | 0.60 | 0.71 | 22 / 24 (20 / 24) |
| casbox / casthin | 0.99 | 0.99 | 15 / 24 |

The user read casboxsd as blurrier than casbox. The caption changes left sharpness where it was.

## Probe: is the teacher still useful on these captions?

`bench/moss_self_distill/teacher_probe.py`, results in `results/20261115-1303-capgap_casbox`.
It is forward-only and runs 30 images at σ 0.9 / 0.95 / 0.98 with two noise draws each. Its `tag`
group is the casthin cache and its `lay` group the casbox cache (both tag captions), with teacher
embeds for both.

**Headroom** is `fm(caption) − fm(image + caption)`: how much flow loss the image in the TE saves
over the caption alone, i.e. what the teacher can give.

The criteria were fixed before reading:

- the new captions shrink Base's headroom at both σ 0.95 and 0.98 → the captions carry what the
  teacher used to add;
- the flow-matching casbox LoRA's loss is at or below the Base teacher's → the teacher has nothing
  to pull it toward.

| | σ 0.95 | σ 0.98 |
|---|---|---|
| Base headroom, casthin captions | +0.0194 | +0.0219 |
| Base headroom, casbox captions | +0.0192 | +0.0200 |
| casbox LoRA (caption) − Base teacher (casbox captions) | −0.0104, eff −0.29 [−0.47, −0.11] | −0.0071, eff −0.27 [−0.51, 0.01] |
| casbox LoRA − Base, both on caption only | −0.0296 | −0.0270 |

- **First criterion fails.** The new captions lower Base's caption-only loss and the teacher's loss
  by about the same amount, so at σ 0.95 the teacher has as much to give Base as before.
- **Second criterion holds.** The flow-matching student sits 0.010 below the Base teacher at σ
  0.95, with a confidence interval clear of zero. Replacing its target there pulls it up toward
  Base.

On 2026-11-14, on the earlier rating-free Anima captions, the gap between the student and the
teacher was smaller: the flow-matching LoRA was about 0.024 below Base against a teacher headroom
of 0.018. There, balanced hybrid95 made no consistent visible difference. Here it softens the
renders. A probe of casthin's LoRA, to place the old gap on the same day, was dropped: casbox is
the captions going forward.

Base-only renders of the same sets, to check that casboxsd sits between Base and casbox in
sharpness, were not run.

## Outcome

- caststation trains on casbox captions with flow matching alone. A well-captioned set gets no
  balanced hybrid95 by default; on such a set the student passes the Base teacher within 480
  steps.
- `CaptionsRequest.mask_dilate` 16 and `CacheRequest.text_boxes` on are the shipped defaults.
- Boxed text in a `tags` prompt places text where it asks, so render text the way the cache
  writes it.
