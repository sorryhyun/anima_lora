# Turbo ReFL — teaching the base model's text rows through its distilled student

A note on the method the glyph stage runs on (written 2026-11-25): what it is, what has been
measured, the conditions it seems to need, and what is still open. How the stage runs in
detail is in [README.md](README.md); the runs are in [plan.md](plan.md) and
[reports/](reports/). The name is the user's.

## The method

A reward model reads the image, and its loss is carried back by gradient to the thing being
taught. Three choices make it this method and not plain reward fine-tuning:

- The image comes from the distilled few-step student (the 4-step turbo), not from the base
  model. Three of its steps run with no graph; only the last is differentiated. With shift 3
  the student's steps sit at σ 1 → 0.9 → 0.75 → 0.5 → 0, so the one differentiated step is
  the whole second half of the trajectory, and its output is the image itself.
- What is trained sits on the text side: one row of the embedding table per glyph. The
  denoiser, the decoder and the reward models are frozen.
- The reward is a small OCR reader's cross-entropy of the asked glyph on the crop where the
  glyph was drawn (a second reader's at a low weight; the text detector's score when nothing
  was drawn).

The student and the base model read the same table, so a row taught on the student is used on
the base at its ordinary sampler. The student is a cheap differentiable stand-in for the
base; the knowledge lands in the input space the two share. This is the reverse of the usual
flow, where the teacher's knowledge goes to the student.

In the literature's terms it is direct reward backpropagation truncated to the last sampling
step (the DRaFT-1 / ReFL shape), applied to token embeddings as textual inversion is, with an
OCR loss as the reward.

## What has been measured

Facts, each from the named report:

- 16 rows trained this way read alone on 59 of 256 held-out pages before, and on 136, 175
  and 199 after 8, 16 and 24 steps a row ([g1long](reports/g1long_2026_11_25.md)).
- The same reader used as a scalar reward under GRPO, on sentence captions with every row
  trainable, changed nothing on the ruler in 300 groups ([p1](reports/p1_2026_11_13.md)).
  The two setups differ in more than the gradient.
- A trained row keeps its length and its neighbours in the table and turns about 15°. Part of
  the turn is common to all rows: the other rows' mean displacement alone takes a row that
  spent no render from 59 to 112, where a drawn direction of the same length stays at 59
  ([row_line](reports/row_line_2026_11_25.md)). In the run in progress
  (`g2a_1125_1642`, not finished) that displacement, measured on 8 hiragana and 8 kanji,
  takes the 164 kana rows from 743 to 1 322 of 2 624 pages before any of them is trained.
- A row that already reads can be pushed past its glyph by further steps (あ to あら,
  [g1shared](reports/g1shared_2026_11_25.md)).
- On the base model (anima-base-v1.0, 28 steps, cfg 4, VAE decode, a sign prompt the rows
  never saw) the 24-step rows of 4 glyphs letter one clean glyph on all 8 renders, by
  Claude's labelled read, where the start rows repeat the glyph or add others
  ([g1long](reports/g1long_2026_11_25.md), second part). Eight renders; model, sampler,
  decoder, frame and size all differ from training at once.
- Cost on a 16 GB card: about 3 s for a group of 4 renders with their backward passes.

## Conditions it seems to rest on

The first four are the user's (2026-11-25). The notes under each are how the glyph stage
meets it.

1. The task is built from data the model has already learned. The scenes are training
   captions, and the rows were first seeded by ordinary training, so the model already draws
   the glyph on some renders. The reward selects and sharpens something inside the model's
   range. A single differentiated step has no direction toward a thing the frozen denoiser
   cannot draw at all.
2. A precise, small OCR model. It has to be right on a lone glyph, calibrated enough that a
   confidence floor separates a drawn glyph from noise, and small enough to run and
   differentiate once per render. Its blind spots become the method's: it cannot tell a small
   kana from the full-size one drawn alone, or ニ from 二.
3. The trained region is limited. One row a glyph, a fixed short step, everything else
   frozen. A row can change what its token asks for and nothing about how the model draws,
   which bounds how far the reward can be gamed and is also why the result moves to the base
   model.
4. The area to draw in and the area to keep are clearly defined. The caption fixes a scene
   and a speech bubble and asks for the glyph in it; the loss is taken on the detected crop
   only, and the rest of the page is asked for nothing.

## Further notes (Claude's)

Opinions and open points, none of them measured unless a report is named.

- A fifth condition: the student's last step has to span the range where the target is
  decided. Glyph strokes are still open at σ 0.5, so one step of gradient can change which
  glyph is drawn. The same truncation on the 28-step base would reach texture only. The
  corollary matters for the sentence stage: what is decided earlier (how many bubbles, where,
  the length and order of a string) is not reached by this gradient, and may need a second
  differentiated step or a different signal.
- A sixth: the student and the base have to share the pathway that is trained. Here the
  student is a delta on the same denoiser and the text path is untouched. The transfer is the
  claim the whole method stands on and it is the least measured: 8 renders, crossing the
  merge and the turbo delta together.
- Condition 4 is met by construction, not by a term. No loss protects the rest of the page;
  it is protected only because the trainable part speaks through one token. That has been
  looked at and not counted.
- The reward has no finished state. The cross-entropy keeps falling after the glyph reads,
  the step has a fixed length, and a row is then pushed into drawing more than its glyph. A
  stop per row (by its own training-side count) is parked in the plan; without one the number
  of steps is a setting that the slowest and the fastest rows both pay for.
- Much of what a row learns is not about its glyph. The shared displacement is something
  like "letter one glyph, plainly, alone", and it is learned again by every row at full
  price, since a render teaches only the row in its caption. Learning that part once is the
  obvious saving, and it also raises the question of where it belongs: it may be a property
  of the frame and not of any row.
- The same shared part may be a drift toward what the reader finds easiest: the trained
  rows' signs are bold, clean and centred. For dialogue that is the aim. For SFX and
  hand-lettered text a reader-shaped style would be a loss, and nothing here measures the
  variety of lettering before and after.
- A lone glyph is the unit where the reader is least ambiguous, and still not every glyph is
  decidable alone (small kana, the shapes the two scripts share, katakana against kanji).
  Those need a word around them, so some of the table can only be taught at the word stage.
- Two readers at once (the row gradients' cosine is 0.21,
  [hayai_g0](reports/hayai_g0_2026_11_25.md)) is cheap insurance against one reader's
  quirks. It does not help where both share a blind spot.
- Checks that would say the most for their cost: the permuted-label control (how much of the
  gain needs the right label), the cosine between the row gradient through the last step and
  through the step before it (whether the wide last step or the reader is doing the work),
  and the transfer read on the base on a set no row was tuned on.
- Where else the shape might fit: any token whose meaning a small differentiable judge can
  read off a localized part of the image, on a model that has a distilled student sharing its
  text path.

## Related work

From memory, not checked against the papers:

- Reward backpropagation through sampling: DRaFT (truncation to the last K steps), AlignProp,
  ReFL (one late step, on a predicted clean image). They train the denoiser, usually as LoRA.
- Few-step models as cheap differentiable generators: ReNO optimises the initial noise of a
  one-step model against rewards, per image.
- Text-side training: textual inversion (embeddings on the denoising loss), TextCraftor (the
  text encoder on rewards), Glyph-ByT5 (glyph knowledge placed in the text encoder).
- OCR as a training signal: AnyText's text perceptual loss.

No paper known to the writer puts the reward through a distilled student to train rows that
are then used on the teacher. A search has not been run.
