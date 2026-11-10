# moss_ref_canvas — does MOSS-Image Base read a reference panel beside the image?

The premise of `docs/moss/proposal/ref_canvas.md`, on Base with no LoRA and no training.

## `probe.py`: teacher-forced loss on character_ref pairs

Pairs of `post_image_dataset/moss/i2i_character_ref` whose reference has the target's grid
(124 of 1,000), one per artist per round, fixed noise per pair, flow-matching MSE over the
image panel at σ 0.9 / 0.6 / 0.3. Each reference condition also runs with another artist's
reference of the same grid (`_shuf`); the read is own − shuf.

| condition | reference | coordinates | reference rows |
|---|---|---|---|
| `none` | — | Base | — |
| `none_capshuf` | — | Base, the other pair's caption | — |
| `none_canvas` | — | the image at its left-panel coordinates | — |
| `joint` | noised at the step's σ | r = 0, beside, re-centred | the step's timestep, attention open |
| `open` | clean | as `joint` | `ref_timestep 0`, attention open |
| `iso` | clean | as `joint` | `ref_timestep 0`, `ref_isolate` |
| `offset` | clean | r = 1, `ref_pos offset` | `ref_timestep 0`, `ref_isolate` |

`none_canvas_box`, `joint_box`, `iso_box`: the same under a caption with one `panel` item per
panel, each with its canvas box. `joint` is Base on one 2W-wide picture
(`canvas_rope` equals the stock RoPE of that picture); `iso` is the i2i recipe on canvas
coordinates; `offset` is what `moss_i2i_character_ref_off_e1` trained on.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_ref_canvas/probe.py"

## `panels.toml`: two-panel renders

Four prompts on a 1728×1216 canvas, seeds 0–2 (`panels_prompts.txt`: two-panel layout boxes,
the proposal's target + `reference` panel caption, a character-sheet tag prompt, a plain
sentence). No prompt names hair, eyes or outfit, so a match across the panels comes from the
panels.

    make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --config bench/moss_ref_canvas/panels.toml"

## Run 20261110-2151-base_int8_n24 (Base, int8 DiT, 24 pairs / 24 artists, ~10 min)

own − shuf, flow-matching MSE over the image panel (bootstrap 95 % CI):

| condition | σ 0.9 | σ 0.6 | σ 0.3 |
|---|---|---|---|
| `joint` | **−0.0153** [−0.0221, −0.0098] | **−0.0112** [−0.0174, −0.0068] | **−0.0136** [−0.0221, −0.0075] |
| `open` | −0.0031 [−0.0065, +0.0005] | −0.0034 [−0.0054, −0.0015] | −0.0014 [−0.0046, +0.0018] |
| `iso` | −0.0012 [−0.0045, +0.0023] | −0.0013 [−0.0031, +0.0005] | −0.0017 [−0.0041, +0.0005] |
| `offset` | −0.0009 [−0.0041, +0.0026] | −0.0011 [−0.0026, +0.0003] | −0.0015 [−0.0034, +0.0004] |

Scale: the other pair's caption (`none_capshuf` − `none`) costs +0.0078 at σ 0.9, +0.0003 at
0.6, 0 at 0.3.

- Base reads the matching reference when the panel is noised with the image: twice what the
  right caption is worth at σ 0.9, and at every σ. The shuffled reference is another
  artist's, so the gain is character and style together.
- Clean and isolated, on canvas or `offset` coordinates, it reads a tenth of that; the two
  coordinate systems do not differ. Opening the attention on the clean reference (`open`)
  recovers little, and so does matching the noise under isolation (next run): Base needs both.
- The panel-box caption changes nothing in any condition (`joint_box`, `iso_box`,
  `none_canvas_box` within 0.001 of their plain rows).
- Every reference condition sits above `none` (`joint` +0.005 / +0.014 / +0.026, `iso` +0.026 /
  +0.017 / +0.027, `open` +0.042 / +0.031 / +0.045), the clean ones most. The re-centre alone
  costs +0.001 / +0.004 / +0.008. Why a panel raises the loss is not checked (a guess: two
  full-bleed illustrations side by side are not a page Base draws; its own two-panel renders
  have a gutter).

## Run 20261110-2201-base_int8_n24_refsigma (same pairs and noise, ~5 min)

`--conditions none,iso_s3,iso_s6,iso_s9`: the reference isolated on canvas coordinates at a
fixed σ with `ref_timestep` to match, the configurations `RefCache` can hold. own − shuf:

| reference σ | image σ 0.9 | image σ 0.6 | image σ 0.3 |
|---|---|---|---|
| 0 (`iso`, run above) | −0.0012 [−0.0045, +0.0023] | −0.0013 [−0.0031, +0.0005] | −0.0017 [−0.0041, +0.0005] |
| 0.3 | −0.0036 [−0.0062, −0.0011] | −0.0019 [−0.0033, −0.0007] | −0.0025 [−0.0043, −0.0009] |
| 0.6 | −0.0026 [−0.0054, −0.0001] | −0.0019 [−0.0034, −0.0005] | −0.0021 [−0.0037, −0.0006] |
| 0.9 | −0.0016 [−0.0032, −0.0001] | −0.0009 [−0.0017, −0.0001] | −0.0006 [−0.0015, +0.0002] |
| the step's, not isolated (`joint`) | −0.0153 | −0.0112 | −0.0136 |

- Isolation alone removes 82–90 % of `joint`'s gain: on the diagonal (reference σ = image σ,
  `joint` plus `ref_isolate`) −0.0016 / −0.0019 / −0.0025 remain. Base's reading across
  panels needs the reference rows to read the image and the text.
- No configuration `RefCache` can hold reads more than a quarter of `joint`. The best is the
  reference at σ 0.3 with `ref_timestep 300`: two to three times the clean reference, CI off
  zero at every image σ, and a smaller rise over `none` (+0.015 / +0.014 / +0.021 against
  +0.026 / +0.017 / +0.027).

`panels.toml`, same day (`output/tests/moss_ref_canvas_panels`): the box prompt and the
sentence prompt hold hair, outfit and style across the two panels on all 6 renders; the
`reference image of the character` panel caption repeats the whole scene, pose included; the
character-sheet tags give sheets whose views change outfit on 2 of 3 seeds.
