# MOSS t2i self-distillation: bounding the teacher on long runs

Status: proposal (2026-11-14). The recipe **add90 ema97** shipped 2026-11-09
(`configs/moss/selfdistill.toml`, user doc
[`experimental/self_distill.md`](../experimental/self_distill.md)); its arms, probe and verdict
are in [`reports/self_distill_arms.md`](../reports/self_distill_arms.md). On a 2,832-step run it
leaves the LoRA further from its training images than Base at σ ≥ 0.95, which the training loss
does not show. Proposed: a Base teacher at σ ≥ 0.95 (hybrid95) as the safe form, and the EMA
teacher under a teacher-step budget. No run of either yet.

Source: D-OPSD, arXiv 2605.05204, off-policy arm (§ 3.3).

## The objective

```
x_t   = (1 − σ)·x₀ + σ·ε
u_s   = v_θ (x_t, t, TE(caption))                  student, grad
u_t   = v_θ̄ (x_t, t, TE(image, caption))           teacher, no grad
loss  = FM + w·‖u_s − sg(u_t)‖²    on steps with σ ≥ 0.9     (add90)
```

θ̄ is Base (`distill base`) or an EMA 0.97 copy of the LoRA (`distill ema`). The teacher's TE
call has the training image (~1,036 vision rows at 1 MP) ahead of the caption, and the DiT gets
all of those rows. `w` scales the teacher term to the flow term's gradient norm on the
prediction (`distill_add 1`).

## Evidence, 2026-11-14

Probe: `bench/moss_self_distill/teacher_probe.py`, forward-only on cached student and teacher
embeds, int8 Base, fp32 stream, 2 noise draws per image and σ, intervals by bootstrap over
images. Runs under `bench/moss_self_distill/results/20261114-*`.

For a student and a teacher on the same `x_t`:

```
eff = (FM_student − FM_teacher) / ‖u_t − u_s‖²
```

1 means the teacher's whole shift points at the image, 0 that it buys no flow loss, below 0
that it costs.

### The Base teacher's shift is signal at σ ≥ 0.95, in every caption format

Sincos, 24 of the 354 images, Base student and Base teacher:

| caption | eff, σ 0.9 | eff, σ 0.95 | eff, σ 0.98 | ‖u_t − u_s‖², σ 0.95 |
|---|---|---|---|---|
| tag | 0.70 [0.57, 0.81] | 0.97 [0.87, 1.06] | 1.14 [0.92, 1.34] | 0.0187 |
| tag, shuffled / dropped variant | 0.70 [0.56, 0.82] | 0.96 [0.85, 1.06] | 1.16 [0.93, 1.36] | 0.0189 |
| layout JSON | 0.70 [0.57, 0.82] | 0.94 [0.83, 1.03] | 1.12 [0.94, 1.29] | 0.0192 |

- The teacher's flow loss is below the student's on 92 % of images at σ 0.9 and on all of them
  at σ 0.95 and 0.98.
- The caption format moves the teacher by 0.0015 at σ 0.95 (tag teacher against layout-JSON
  teacher), 8 % of its distance from the student. A tag student against the layout-JSON teacher
  reads 0.94 [0.84, 1.03]. Giving the teacher another caption than the student's changes
  nothing.
- So the defect on tag captions
  ([tag_captions.md](tag_captions.md#confound-fm-only-vs-add90-ema97)) is not a worse teacher
  under tags, and not the caption variants.

**A trained student needs the gate at 0.95.** Base teacher against a LoRA student:

| student | σ 0.9 | σ 0.95 | σ 0.98 |
|---|---|---|---|
| tag LoRA, FM only | −0.38 [−0.51, −0.27] | 0.28 [0.08, 0.46] | 0.62 [0.38, 0.83] |
| tag LoRA, add90 ema97 | 0.00 [−0.13, 0.12] | 0.83 [0.72, 0.93] | 1.12 [1.00, 1.23] |

At σ 0.9 the Base teacher knows nothing the LoRA lacks: its shift there is the pull to Base's
style seen on 2026-11-09.

### After 2,832 steps of add90 ema97 the LoRA is above Base at σ ≥ 0.95

Flow loss on training images, LoRA minus Base on the same `x_t` (sincos, 24 images; images
above 0 in brackets at σ 0.95):

| LoRA | σ 0.9 | σ 0.95 | σ 0.98 |
|---|---|---|---|
| tag, FM only (`sincostagfm_1113_0912`) | −0.0124 | −0.0108 [−0.0143, −0.0067] (4) | −0.0095 |
| tag, add90 ema97 (`sincostag_1113_0717`) | −0.0056 | **+0.0049** [+0.0017, +0.0084] (15) | **+0.0054** |
| layout JSON, add90 ema97 (`1111_2235`) | −0.0066 | **+0.0091** [+0.0062, +0.0118] (22) | **+0.0153** |

The teacher moved with it. Flow loss of the LoRA teacher (image in context) minus the Base
teacher's:

| LoRA | σ 0.9 | σ 0.95 | σ 0.98 |
|---|---|---|---|
| tag, FM only | −0.0118 | −0.0106 | −0.0068 |
| tag, add90 ema97 | −0.0034 | +0.0114 [+0.0081, +0.0146] | +0.0175 |
| layout JSON, add90 ema97 | −0.0023 | +0.0182 [+0.0138, +0.0228] | +0.0286 |

- Both caption formats, the layout-JSON LoRA more. The layout-JSON LoRA's teacher at σ 0.95
  (0.5339) is where the Base student without the image is (0.5336).
- Each LoRA's teacher is still ahead of its own student at σ 0.95 (tag: eff 0.93
  [0.65, 1.15], lower on 96 % of images; layout JSON: 0.70 [0.34, 0.97], 71 %), so the term
  keeps pulling.

### It accumulates, and the teacher–student distance does not show it

Training logs of the tag runs, same step = same image, σ and noise. Flow MSE of the add90 ema97
run minus the FM-only run's, on the 612 teacher steps:

| epoch (354 steps) | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| σ 0.9–0.95 | +0.0020 | +0.0046 | +0.0062 | +0.0067 | +0.0080 |
| σ ≥ 0.95 | +0.0025 | +0.0072 | +0.0077 | +0.0125 | +0.0138 |

The σ ≥ 0.95 band is still rising at epoch 8. The logged teacher term over the same run:

| steps | tag | layout JSON | head + prose |
|---|---|---|---|
| 1–354 | 0.0128 | 0.0136 | 0.0085 |
| 2,125–2,832 | 0.0116 | 0.0105 | 0.0041 |

The student closes 10 % (tag) and 23 % (layout JSON) of its distance to the teacher; the rest
is what a caption cannot say. With `distill_add 1` that term's gradient is scaled about 6.5×
(`sqrt(0.5 / 0.012)`) to the flow term's norm on every teacher step.

### At 480 steps neither teacher has drifted

Caststation, 30 images, the 480-step LoRAs of 2026-11-09, LoRA minus Base as above:

| LoRA | σ 0.9 | σ 0.95 | σ 0.98 |
|---|---|---|---|
| SFT | −0.0221 | −0.0258 | −0.0251 |
| hybrid90 (Base teacher, replaces FM) | −0.0178 | −0.0121 | −0.0098 |
| add90 ema97 | −0.0144 | −0.0147 | −0.0115 |

- Both teacher arms fit the training images about half as far as SFT at σ ≥ 0.95 and stay
  below Base. This is the regime the recipe was judged in: less of the training compositions
  memorised at the σ that sets composition.
- The EMA teacher's own flow loss is below the Base teacher's here (−0.0090 at σ 0.95).
- 101 teacher steps here against 612 on sincos; per image the sincos run had fewer (1.7 against
  3.4). What grew is the number of teacher updates on the weights.
- Not split: whether a Base teacher drifts the same way at 2,832 steps. No Base-teacher LoRA
  of that length exists.

### The caption rows alone carry about a tenth of the image

`cut`: the teacher's embeds with the 1,036 image rows and their two markers dropped before the
DiT, so the sequence has the student's shape (the caption rows saw the image in the TE). Sincos tag, Base:

| teacher | FM gain over the student, σ 0.95 | ‖u_t − u_s‖² | eff | images with the teacher lower |
|---|---|---|---|---|
| image + caption | 0.0181 | 0.0187 | 0.97 [0.87, 1.06] | 100 % |
| caption rows only | 0.0016 | 0.0034 | 0.47 [0.05, 0.86] | 58 % |

The DiT takes the image from the image rows. The `cut` arm also differs from the student in the
prompt tail (off against `sure`), which is not separated here. Renders of both arms on Base and
on the FM-only LoRA: `results/20261114-1648-sincos_cut/teacher_renders.png`.

### Teacher renders

`results/20261114-1615-sincos_tag/teacher_renders.png`: four training images, image + caption
through Base and through the add90 ema97 tag LoRA, CFG 3, 50 steps, one seed. Read by the user:

- The LoRA teacher carries more detail than the Base teacher.
- On a multi-view page with SFX (10657487) it adds a lot of stray text. 164 of the 354 captions
  carry a text or SFX clause.
- Against the sources, which separate a blurred background from the figure, the renders draw
  the background too sharp and the lines too thick.

## Reading

- The objective holds for a fixed teacher: at σ ≥ 0.95 nearly all of what the image-conditioned
  Base adds over the caption is the image.
- The teacher term trades training-image fit at high σ for the model's own rendition. Over 100
  teacher steps that is the steadiness on general prompts; over 600 it passes Base and keeps
  going.
- The EMA teacher has no fixed point by construction: it is the student 33 updates ago plus
  the image, the LoRA runs on a sequence it never trained on (1,036 image rows ahead of the
  caption), and the pair moves together, so their distance (`loss_t`) stays near 0.012 while
  both leave the data.
- The extra hand under tag captions is one face of this, not a caption-format effect: the
  layout-JSON LoRA drifted further by flow loss and rendered clean on the images read.

## Proposal

Both arms on the sincos tag cache, the `sincostag_1113_0717` config otherwise (2,832 steps,
lr 5e-5, `caption_variant_rate 0.75`), about 50 minutes each.

### A. hybrid95: Base teacher, σ ≥ 0.95, in place of FM on those steps

`distill = "base"`, `distill_sigma_min = 0.95`, `distill_add = 0`. No code. The teacher is
fixed, and the gate sits where its shift is still signal for a trained student. Known cost: the
pull to Base's palette grows with dose (2026-11-09); σ ≥ 0.95 is 8 % of steps against 21 % at
0.9. If it shows hybrid90's 480-step faults (a broken pose, a stray object), the variant is the
same teacher with FM kept (`distill_add = 1`).

### B. EMA teacher under a teacher-step budget

New field `distill_max_steps N` (0 = no cap): a gated step takes the teacher with probability
`N / expected teacher steps`, from its own RNG stream, so the σ and noise sequence stay those of
an uncapped run. `N = 120`: caststation's 480-step run took 101 and did not drift. Keeps the
EMA teacher's style and spreads its updates over the run.

- A clamp on the balance scale is this cap in another form: `loss_t` stays near 0.012, so the
  scale is ~6.5 throughout and a clamp below it equals a lower `distill_add`.
- If the budget holds the drift but loses the general-prompt gain, the second form is an
  anchored teacher: the EMA copy at multiplier `m < 1` in the teacher forward, which runs from
  hybrid (`m = 0`) to today's teacher (`m = 1`).

### Logging

`loss_fm_t`, the teacher's own flow MSE on teacher steps (its output and the flow target are
both there). Against the FM-only run of the same seed it is the number that moved in the tables
above.

### Read, fixed before the runs

1. **Drift**: `teacher_probe.py` on the 24 images, LoRA student minus Base at σ 0.95 and 0.98.
   Pass: neither interval lies above 0.
2. **Anatomy**: the 6 tag and 6 default prompts of
   [tag_captions.md](tag_captions.md#confound-fm-only-vs-add90-ema97), same seeds, same rule
   (a visible hand without five fingers, or a wrong limb count, is broken). Pass: 0 broken, as
   FM only.
3. **Worth it**: Space 004 and the general prompts against `sincostagfm_1113_0912`, read by the
   user. On sincos the teacher term has not yet shown anything FM only lacks; an arm that passes
   1 and 2 and reads like FM only is not worth its teacher cache.

## Open questions

- **Base teacher at length.** Arm A's drift read answers whether the accumulation is the EMA
  loop's or the teacher term's.
- **Transfer.** Other artists, a character set (ama mitsuki), i2i. For glyph tasks the tower is
  32 px per token, so the teacher would carry layout, not strokes.
- **Other contexts for the teacher.** A sparse student caption against image + full caption;
  an instruction placed before the image (the TE is causal, so it shapes the image rows); the
  image at another resolution or a crop.
- **What the teacher adds, by content.** Its renders sharpen backgrounds and thicken lines
  against the source. Whether the high-σ velocity carries that, or only composition, is not
  measured.
- **On-policy at high σ.** Rolling the student out over the first sampler steps and distilling
  on its own states, after a bounded form exists. Under shift 5, σ ≥ 0.9 is ~18 of 50 steps.

## Related work

Cited from memory, not read for this proposal. Read the PDFs before relying on any number.

- **Privileged-information distillation.** LUPI (Vapnik 2009); Generalized Distillation
  (Lopez-Paz et al. 2016). This is the general form: a teacher with train-only information
  teaches a student without it.
- **Context distillation.** Askell et al. 2021; Snell et al. 2022, "Learning by Distilling
  Context". The LLM form of this idea: the same model with a prompt context is the teacher.
- **Guidance distillation.** Meng et al., CVPR 2023, "On Distillation of Guided Diffusion
  Models". The same structure on a diffusion model: the student regresses on the same model's
  stronger-conditioned (CFG) velocity. Here the stronger condition is the image context.
- **Stable Target Field.** Xu et al., ICLR 2023. Replaces the one-sample `ε − x₀` with a
  reference-batch posterior average at high noise to cut target variance. This is the argument
  for teacher targets at high σ.
- **DreamBooth prior preservation.** Ruiz et al., CVPR 2023. FM mixed with a base-model
  self-target; `preserve_loss` is of this kind.
- **On-policy line.** DAgger (Ross et al. 2011); GKD (Agarwal et al., ICLR 2024); MiniLLM
  (Gu et al.); Self Forcing (Huang et al. 2025, rollout training for autoregressive video
  diffusion).
- **Contrast: step distillation.** Reflow (Liu et al. 2022) and DMD / DMD2 (Yin et al.).
  Matching the teacher's instantaneous velocity on rollouts does not make a few-step model; the
  Anima turbo is DMD.
