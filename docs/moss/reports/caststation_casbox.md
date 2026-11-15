# caststation casbox: held objects, OCR boxes, and self-distillation on top

Closed 2026-11-15 on channel (caststation), 30 images. Two caption changes on top of the thin
MOSS-format captions (`casthin_1115_1150`, the control) and one recipe change on top of those:

1. **`mask_dilate 16`**: the position stage's crop tagger sees each subject mask grown by 16 px,
   so an object in hand reaches its holder's clause.
2. **OCR boxes on text lines** (`text_boxes`): each OCR'd line of a text clause carries its box.
3. **Self-distillation** (balanced hybrid95, `configs/moss/selfdistill.toml` at the time) over captions 1 + 2.

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

## Follow-up, 2026-11-15: a gated teacher, lr 2e-4, a band-limited teacher

Four more arms on the casbox cache, all against `casbox_1115_1226`. None beat it.

| Run | lr | Teacher steps (σ ≥ 0.95) | Laplacian variance against casbox (24 pairs) |
|---|---|---|---|
| `casboxsd_1115_1230` | 1e-4 | balanced hybrid95, 45 of 45 | 0.60, lower in 22 |
| `casgate_1115_1409` | 1e-4 | `distill_gate`, 38 of 45 open | 0.83, lower in 18 |
| `casgatelr_1115_1409` | 2e-4 | `distill_gate`, 42 of 45 open | 0.90, lower in 18 |
| `casgateadd_1115_1532` | 1e-4 | `distill_gate` on `distill_add 1`, 37 of 45 open | 0.84, lower in 19 |
| `casband_1115_1507` | 1e-4 | teacher in the band 0 < f ≤ ⅛ Nyquist only | 0.96, lower in 16 |
| `casboxlr_1115_1336` | 2e-4 | none (flow only) | 1.08, lower in 9 |

Sheets: `output/tests/caststation_thin/casgate_sheets/`.

**`distill_gate`** keeps the teacher target on a teacher step only when the teacher's own flow
MSE is below the student's on that sample (`fm_t < fm_s`), and trains the flow target otherwise.
A teacher step that passes has a positive inner product with the flow step on the prediction:
`(v_s − u)·(v_s − v_t) = ½(fm_s − fm_t + ‖v_t − v_s‖²)`.

- **The gate barely closes.** The saved (EMA) flow-only LoRA is below the teacher at σ 0.95, but
  the raw weights during training are not. On the 45 teacher steps, which share σ and noise
  across runs, raw student − Base teacher:

| Steps | casbox 1e-4 | casboxlr 2e-4 | casboxsd 1e-4 | casgate 1e-4 | casgatelr 2e-4 |
|---|---|---|---|---|---|
| 1–120 | +0.024 | +0.027 | +0.021 | +0.021 | +0.024 |
| 121–240 | +0.010 | +0.033 | +0.016 | +0.015 | +0.030 |
| 241–360 | +0.012 | +0.038 | +0.028 | +0.028 | +0.051 |
| 361–480 | −0.012 | −0.006 | +0.005 | +0.004 | +0.021 |
| steps above the teacher | 29 | 37 | 39 | 38 | 42 |

- **Replaced steps hold the student at the teacher's level**, which keeps the gate open. Saved
  weights, same probe as above (`results/20261115-{1355-capgap_casboxlr,1437-capgap_casgatelr,1443-capgap_casgate}`):

| LoRA | LoRA − Base teacher, σ 0.95 | σ 0.98 | LoRA − Base (caption), σ 0.98 |
|---|---|---|---|
| casbox | −0.0104 | −0.0071 | −0.027 |
| casboxlr | −0.0084 | −0.0046 | −0.025 |
| casgate | −0.0023 | +0.0076 | −0.012 |
| casgatelr | +0.0034 | +0.0136 | −0.006 |

- **The user's read:** a trade-off. The gated teacher adds some detail and depth and pulls the
  style toward Base (hair gloss, saturated cyan and pink, Neru's outfit drifting); hands are not
  reliably better, and flow only is sometimes better on them. On these captions at lr 1e-4,
  following the style is preferred. The gate read better than the band-limited teacher, so the
  flag stays.
- A stricter threshold (`eff > τ` on the same quantities) would open 26 / 19 / 11 of 45 steps at
  τ 0.5 / 0.75 / 1 on casgate. It moves along the same casbox–casboxsd line; not run.

**Gate on add mode** (`casgateadd`): the flow term stays on every teacher step and the gated
teacher term is added at its gradient norm. Two reads were fixed before the run and both failed.
The gate opened on 37 of 45 steps against replace's 38, not the flow-only trajectory's 29. The
saved LoRA's gain over Base at σ 0.98 came back to −0.018, between replace's −0.012 and flow
only's −0.027; what is missing sits in the 0–⅛ Nyquist band (LoRA − teacher +0.0054 there,
−0.0007 on casbox). So the teacher term itself, not only the missing flow step, keeps the raw
weights above the teacher at high σ. Against replace, add is the better of the two: the fit is
closer to flow only at both σ (σ 0.95: −0.025 against −0.022), its renders sit nearer casbox's
(mean pixel distance lower in 16 of 24 pairs), Neru's outfit and the boxed-text compositions follow
casbox in cells where `casgate` changed them, and the user read it as better than replace. The
stray shape in one two-girl cell is in both.

**lr 2e-4** does not reopen room for the teacher in the saved weights (table above), and the
teacher does not repair what it breaks (`findings/t2i_lora_recipe.md` §5).

**Band-limited teacher.** `teacher_probe.py --bands` splits each flow loss by radial band on the
32 ch latent grid (`results/20261115-1449-bands_casbox`, casbox LoRA):

| σ 0.95 | DC | 0–⅛ | ⅛–¼ | ¼–½ | > ½ Nyquist |
|---|---|---|---|---|---|
| Base caption − Base teacher | +0.0003 | +0.0108 | +0.0051 | +0.0025 | +0.0005 |
| casbox LoRA − Base teacher | −0.0014 | −0.0021 | −0.0030 | −0.0031 | −0.0010 |
| eff, LoRA → teacher | −0.80 | −0.14 | −0.31 | −0.45 | −0.45 |

| σ 0.98 | DC | 0–⅛ | ⅛–¼ | ¼–½ | > ½ Nyquist |
|---|---|---|---|---|---|
| Base caption − Base teacher | +0.0018 | +0.0160 | +0.0016 | +0.0004 | +0.0001 |
| casbox LoRA − Base teacher | −0.0022 | −0.0007 | −0.0030 | −0.0011 | −0.0001 |
| eff, LoRA → teacher | −0.71 | −0.04 | −0.75 | −0.76 | −0.46 |

The teacher's headroom sits in 0–⅛ Nyquist (56 % at σ 0.95, 80 % at 0.98), the one band where
the trained LoRA only ties it; at σ 0.9 the LoRA is below the teacher in every band (eff −0.6 or
lower). `casband` trained the teacher's velocity in that band and the flow target in the rest of
the spectrum, the band's term scaled to the flow residual's norm there. It kept sharpness and
broke composition: stray large shapes and stacked heads in the two-girl cells, outfits off the
prompt. The user called it a clear failure. A tie in flow loss on the training images did not
mean the band was free to pull: it is the band that sets the large shapes. The flag was removed.

## Outcome

- caststation trains on casbox captions with flow matching alone. The gated teacher, lr 2e-4 and
  the band-limited teacher did not change that. A well-captioned set gets no
  self-distillation by default; on such a set the student passes the Base teacher within 480
  steps.
- The self-distill recipe (`configs/moss/selfdistill.toml`) is now the gated add form, the better
  of the teacher forms here, in place of balanced hybrid95.
- `CaptionsRequest.mask_dilate` 16 and `CacheRequest.text_boxes` on are the shipped defaults.
- Boxed text in a `tags` prompt places text where it asks, so render text the way the cache
  writes it.
