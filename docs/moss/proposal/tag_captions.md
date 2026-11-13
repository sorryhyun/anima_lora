# Tag captions for MOSS LoRAs: what breaks, and repairing the caption in tag form

Status: proposal (2026-11-13). The FM-only ablation is read: on sincos the tag LoRA trained
with plain FM draws clean hands and legs, the one trained with add90 ema97 does not
([confound](#confound-fm-only-vs-add90-ema97)). Steps 2–4 not started.

## The question

A MOSS style LoRA is used by people who type Booru tags. The wanted outcome is a LoRA trained
on tag captions that renders as cleanly as one trained on prose. Three caption formats have
been trained so far:

- **tag**: the Anima tag caption verbatim (flat bag + position clauses), `caption_format text`.
- **head**: a head tag line (rating, character, series, @artist, count) + `\n` + image-grounded
  PE prose (`project/pe_lora/src/rewrite_set.py --vision on`, `head_captions.py`).
- **layout_json**: the repo default, tags re-bucketed into MOSS's layout JSON, with
  `plain_caption_rate 0.2`.

On sincos (354 images, 2,832 steps, add90 ema97) the tag LoRA draws extra fingers and a third
leg, the head LoRA leaves Base's composition under tag prompts, and the layout_json LoRA does
neither. The same tag captions under plain FM give clean anatomy. The loss curves do not show
any of this.

## Evidence

### Loss separates head from tag at high σ, and nothing else

Runs of one dataset share the seed, so step *n* has the same image, σ and noise in every arm.
The numbers are paired per-step differences relative to the tag arm, with a 95 % bootstrap
interval.

Caststation (30 images, 480 steps, plain FM; `castag_1112_2220`, `casnl_1112_2220`,
`cashead_1112_2310`):

| σ | n | prose | head |
|---|---|---|---|
| < 0.6 | 90 | 0 | 0 |
| 0.6–0.85 | 216 | −0.15 % [−0.26, −0.05] | −0.26 % [−0.38, −0.15] |
| ≥ 0.85 | 174 | −1.16 % [−1.65, −0.69] | −1.81 % [−2.29, −1.34] |

Head is below prose at σ ≥ 0.85 by 0.65 % [0.39, 0.91]. The gap is there in the first 60 steps,
with the LoRA near zero (head −1.53 %, prose −0.94 %), and is −2.20 % / −1.24 % in the last 160.
Most of it is Base predicting these images better from that caption, not the LoRA learning
more.

Sincos (2,832 steps, add90 ema97; `sincostag_1113_0717`, `sincoshead_1113_0008`, `1111_2235`):

| vs tag | σ < 0.6 | σ 0.6–0.9 | σ ≥ 0.9 |
|---|---|---|---|
| head | −0.03 % | −0.03 % | −4.53 % [−4.96, −4.12] |
| layout_json | −0.03 % | −0.02 % | −0.24 % [−0.49, +0.01] |

At σ ≥ 0.9 the logged loss of a distill run is FM plus the teacher term, so the 4.5 % is not a
pure FM gap. Tag and layout_json are indistinguishable at every σ, and their renders differ.

### Renders: caststation shows no format effect, sincos does

Caststation, train format × prompt format, 4 prompts × 2 seeds
(`output/tests/pe_tag_vs_nl/caststation_eval/render/matrix_p*.png`): every LoRA carries the
style under every prompt format. Composition follows the prompt format and the seed; the three
LoRAs draw nearly the same picture in a column.

Sincos, tag prompts, one seed per prompt
(`output/tests/sincos_distill_ab/tagfmt/sheet_tagfmt_4arm.png`):

- **tag**: keeps Base's composition, but draws a second hand on both Miku prompts where Base
  and the layout_json LoRA draw one. Six fingers on miku s4, three legs on sailor s6; counts in
  the [confound](#confound-fm-only-vs-add90-ema97) table.
- **head**: leaves Base's composition. Two girls on a `1girl` prompt (sailor s6), cat ears and
  off-model outfit on neru s2, a small full-body figure on miku s1.
- **layout_json**: keeps Base's composition, hands and legs clean on the four images read.

The head LoRA held up under tag prompts at 30 images × 480 steps and does not at 354 × 2,832.

### Base reads dense tags

Earlier tag prompts had about 13 tags; the training captions have 56 on average. With
hand-written 40-tag prompts (`prompts_booru.txt`, sheets `booru_p*.png`) Base and all three
caststation LoRAs bind per-figure attributes from position clauses in 8 of 8 renders of a
two-character prompt, and hold the outfit that the 13-tag prompts let drift. Three dynamic
scenes written as sparse tags / dense tags / head + prose (`prompts_dense3.json`, sheets under
`render/dense3/`) show no broken dense-tag cell; prose fixes the camera more tightly than the
same content as tags. These prompts are single-figure, have no redundant tags and were written
by hand, so they are not what a training caption looks like.

### What the training tag captions look like

Caststation, 30 images: 21 are `multiple views`, 26 carry position clauses, 1,690 tags in all
(1,304 in the flat bags, median 43). The flat bags hold 604 distinct tags, 388 of them on one
image only. Read against the image, four safe-rated captions show three defects:

1. **One object, many tags.** `11429918` spends 11 tags on one bowtie (`bow`, `bowtie`,
   `striped bow`, `diagonal-striped bowtie`, `striped neckerchief`, …). `channel6` has 7 for the
   bags, 5 for a bandaid, 4 for a phone. Counting only tags whose words are a subset of another
   tag's, 12 % of flat-bag tags are such repeats; synonyms (`cellphone` / `smartphone`) are not
   counted.
2. **No owner.** `channel6` (Neru, Miku, Teto) lists `aqua hair, blonde hair, red hair, bandaid
   on knee, sweater around waist, bike shorts, cardigan` in the flat bag; the right clause is
   `kasane teto` and nothing else. In `11429918` the one panel showing Una has the clause `upper
   body, holding, bowtie`: her hair colour and beret are in no tag, and the bag says `black
   hair`.
3. **Composition gets one tag, accessories get dozens.** `channel4` is Miku crouched on another
   girl's back at night. That is `on person` plus a clause; there is no background tag and no
   pose tag, against 6 tags for the sleeves and collar. Framing / camera tags: median 1 per
   caption.

The PE prose for the same images states the background, pose and who is where, and is longer
(median 190 words against 110). It also invents: "five girls" for `11429918` (two characters,
four figures), a "maid-like headband" for the beret.

### Raising the clause cap does not fix ownership

`anime_tools` position stage on the flattened captions, dry run, 27 images with more than one
subject (`post_image_dataset/pe_ab/caspos/cap*/report.json`):

| setting | tags moved to a clause | left in the flat bag | of those, in no clause | tags per image (median) |
|---|---|---|---|---|
| cap 8 (shipped) | 302 | 1,249 | 807 | 61 |
| cap 16 | 390 | 1,161 | 715 | |
| cap 32 | 402 | 1,149 | 703 | 62 |
| cap 32, `--keep_shared_tags --bind_view_traits` | 429 | 1,122 | 468 | 82 |

344 of the flat tags are names, counts, rating, series and scene tags, which belong there. What
keeps the rest:

- **Multi-view gates (about 235 tags).** On a `multiple views` sheet the character's hair, eyes
  and body are shared by every panel and stay flat by design. They are common, not unowned.
- **Crop-tagger recall (468 tags, with the gates off).** The crop tagger keeps the general tag
  and misses the specific one: `choker` moves to Teto's clause and `black choker` stays flat;
  likewise `bike shorts` / `bike shorts under skirt`, `bandaid on leg` / `bandaid on knee`.
  Props and compound tags (`bubble tea`, `spring onion`, `headset`, `aqua necktie`, `holding
  phone`) mostly reach no clause.
- **Identity gate.** A clause takes a hair or eye colour only if the bag names it, so Una's
  clause stays without one.

Dropping every unowned subject tag from the flat bag would delete 703 tags with the gates on
(the character's hair on every multi-view sheet) or 468 with them off, real details among them.

## Readings

- **R1, unweighted density.** A Booru caption gives every tag the same weight, says little
  about composition and does not say whose an attribute is. The LoRA is trained to express
  dozens of small attributes it cannot place, and learns them as loose parts: hands, legs,
  accessories without a body to belong to. Prose and layout JSON both carry structure the flat
  bag lacks (sentences; per-item captions). Fits the loss gap sitting only at high σ, and the
  layout_json arm being clean.
- **R2, tag captions under add90 ema97.** The teacher term acts at σ ≥ 0.9, where the tag
  caption explains least. layout_json under the same recipe is clean, so the teacher alone is
  not enough. **Supported by the confound read**: the same captions under plain FM are clean.
- **R3, the caption variants.** The tag arm draws a shuffled, tag-dropped variant on 75 % of
  steps; the layout_json and head arms draw none (`caption_variant_rate` 0 in their metadata).
  Under plain FM the variants do no harm. Whether the teacher term breaks on the tag form or on
  the variants is open: the teacher is cached per caption variant, so a dropped tag moves the
  student's caption and the teacher's together.

## Confound: FM only vs add90 ema97

`sincostagfm_1113_0912` (`project/pe_lora/configs/sincostagfm_1113_0912.toml`): the tag arm
with `distill = "off"`, everything else equal. Renders: the 6 tag prompts
(`tagfmt/tag_fm`), the 6 default prompts (`tag_fm`), Space 004 (`space004/tag_fm`).

Read, fixed before the renders exist: every image at full size, same prompt and seed side by
side with `tag_add90_ema97`. An image counts as broken when a visible hand does not have five
fingers or a figure has the wrong number of limbs.

- FM only is broken on about as many images: the distill term is not the cause; R1 and R3
  remain.
- FM only is clean: the defect needs the teacher term on tag captions (R2).

**Result (2026-11-13): FM only is clean, so the defect needs the teacher term on tag
captions.** Pair sheets: `output/tests/sincos_distill_ab/fm_vs_distill/`.

| arm | tag prompts (6) | default prompts (6) |
|---|---|---|
| tag add90 ema97 | 2 broken, 2 unclear | 1 broken, 1 unclear |
| tag FM only | 0 broken, 0 unclear | 0 broken, 0 unclear |

- Broken, add90 ema97: six fingers on the open hand (tag miku s4), three legs (tag sailor s6),
  both hands malformed (default miku s1).
- Unclear, add90 ema97: the lower hand of tag miku s1; a dark leg-like shape under the skirt
  beside a bare leg (tag sailor s3); one bare leg and one in a white stocking (default sailor
  s3).
- The add90 ema97 LoRA draws two hands on all four Miku renders; FM only draws one, as Base and
  the layout_json LoRA do on the tag prompts. FM only also crops sailor s6 at the waist in both prompt sets. Part
  of the gap is therefore fewer hands and legs in frame, and the extra hand is itself what the
  teacher term adds on tag captions.
- Space 004 (a crowd scene) cannot be counted for fingers at render size. The add90 ema97 render
  has a stray striped object in the centre; the FM-only render has none and keeps the packed
  layout.
- One seed per prompt, read by one reader.

Pure-FM loss, tag FM vs tag add90 ema97, first 488 steps: −0.21 % [−0.26, −0.17] at σ 0.6–0.9,
0 below. σ ≥ 0.9 is not comparable (the distill run's loss includes the teacher term).

The head-side counterpart, `sincosheadfm_1113_0720` (head captions, FM only), was cancelled
before it started. It answers a different question: whether the head LoRA's drift under tag
prompts comes from the teacher term.

## Proposed work

Each step is one sincos run (about 45 minutes) or less, and the next one depends on the
previous read.

1. **Fill the confound** (done): the teacher term on tag captions.
2. **Split the variants from the tag form** (R3): the tag arm on add90 ema97 with
   `caption_variant_rate 0`. Same read as step 1. Clean means the variants were the trigger and
   tag captions can keep add90 ema97 without them; broken means tag captions train on plain FM.
3. **Repair the caption, keeping the tag form.** Output stays in the position-clause grammar,
   so the parser, the shuffle and the layout-JSON mapping are unchanged.
   - *Collapse repeats.* Keep the most specific tag of a chain. This needs a tag implication
     table (the Danbooru implication export); `docs/position_captions.md` in `anime_tools`
     rules out a heuristic version, and the `choker` / `black choker` split above is the
     same problem from the other side.
   - *Assign the unclaimed tags with the TE's vision tower.* The 468 tags no crop claimed are
     about 17 per image. The TE (PE LoRA off) gets the image, the figures' positions and the
     tag list, and answers per tag: which figure, all figures, the scene, or a repeat of
     another tag. It selects from the given tags and writes none, so every output tag can be
     checked against the input.
   - *Add composition tags* from a closed list (framing, camera, background), chosen the same
     way.
   - Check before any training: output tags ⊆ input tags ∪ the closed list, and 10 captions
     read against their images.
4. **Train on the repaired captions** against the current tag captions on sincos, with the
   recipe steps 1–2 leave standing. Read as in step 1, plus style against the layout_json arm.

R1 alone is not what breaks anatomy: the unrepaired tag captions are clean under plain FM. Step 3
is then a quality question (ownership, composition) and not a fix for the defect above, and is
worth building only if a plain-FM tag LoRA still shows a caption-shaped fault.

## Not in this proposal

- **T-LoRA.** A σ-dependent rank mask regularises high σ, the range add90 ema97 already
  covers. It would be an arm against add90 ema97, not an answer to the caption question, and
  MOSS has no implementation.
- **Tag order.** Anima's order and the USERGUIDE's (quality → meta → character → series →
  artist → general) differ in where the count tag sits; no arm.
- **Teaching Base to read tags better** (a broad-data tag LoRA, or a teacher that sees prose
  while the student sees tags). Base already binds a clean dense tag prompt; the defect found
  here is in the captions.
