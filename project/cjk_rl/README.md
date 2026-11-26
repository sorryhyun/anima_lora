# cjk_rl — OCR-reward training of the JA vocab-pack rows

Opened 2026-11-13 out of the reseed line (`../cjk_anima_reseed/`). The line trains the JA rows
of the vocab pack (jp_v1's rows) on a loss read off the model's own renders by the OCR reader,
with the 4-step turbo stack as the generator. The target is sentence fidelity: the asked
string, in order, at its length, with no invented dialogue around it.

The method (the reader's loss carried by gradient through the turbo's last steps to the rows,
which the base model then reads) is written up in [turbo_refl.md](turbo_refl.md). The next step
and the runs done are in [plan.md](plan.md).

## Stages

| stage | dates | what trains | where it stands |
|---|---|---|---|
| GRPO (arm R) | 11-13 | jp_v1's rows by GRPO on sentence captions | failed: 300 groups change nothing on the ruler ([p1](archive/reports/p1_2026_11_13.md)) |
| glyph | 11-25 – 26 | one glyph a caption, the 164 kana | done: solo of 2 624 held-out pages 743 → 1 716 (g2b) / 1 785 (g2a); the kana rows `kmix_1126_1158` |
| sentence | 11-26 – | kana strings from the reader's labels, from kmix's rows | the window loss's micro arm passed; the run at scale `sents_1126_1856` submitted 11-26 |

The GRPO stage's proposal (arms A, D, R-credit, the SFX track) is in
[archive/proposal.md](archive/proposal.md); its arm D became the gradient path below (G-D).
[archive/proposal_glyph.md](archive/proposal_glyph.md) planned the glyph stage past its first
16 rows and [archive/sent_rl.md](archive/sent_rl.md) the sentence stage up to its run at scale;
both are folded into this file, `plan.md` and the reports.

## How a run goes

Code: `src/gd.py` (the path), `src/g1.py` (the glyph loop and its read), `src/s1.py` (the
sentence loop and its read), `src/loc.py` (a glyph's place from attention).

### Common to both stages

- A row is one row of the T5-side embedding table that the vocab pack adds to
  `llm_adapter.embed`. As the model sees it, a row is its pack row
  (`anima_cjk_vocab_pack_punct`) plus a delta; the delta starts at jp_v1's (`seed_1008`) or at
  an earlier run's rows, and only the deltas of the rows in the run move. Lengths are in row
  norms, the mean norm of a pack row (206.26).
- Frozen: the DiT (`m15v11`: merged15 plus the turbo v1.1 delta), the pixel head, both
  readers and the detector.
- A render: the bubble frame, `<scene>. Text reads as "<text>".`, the scene with
  `speech bubble` in it and no text tag; 4 Euler steps (σ 1 → 0.9 → 0.75 → 0.5 → 0), cfg 1,
  shift 3, 768², decoded by the pixel head. Renders go 4 images a forward.
- The page is read as any page is: AnimeText finds the boxes, the tuned reader reads each,
  16 pages a call. One box is picked (`gd.pick`): among boxes read at confidence 0.6 or over,
  the one closest to the asked text; with none at 0.6, the best of the rest.
- The loss on a page with a box is the tuned reader's cross-entropy on the picked crop plus
  0.2 times hayai v2.7's on the same crop. A page with no box takes −log of the detector's top
  text score. There is no page term.
- The gradient goes from the loss to the image, through the differentiated steps to the text
  embedding (the earlier steps run with no graph), and through the LLM adapter to the rows.
- A row's step: a render's row gradient is clipped at 0.12; after 16 renders that hold the row
  (`accum`) their mean joins the row's momentum (0.7), and the row moves 0.03 row norms
  against the momentum's direction, whatever the gradient's size. With `max_steps` a row at
  that many steps is passed over.
- Guards, each off when absent from a config (first set in `g2b_1126_0058`; the defaults from
  item 7 of the plan on). A step's renders are drawn at the row as it stood before the step,
  so their count reads that row.
  - `near_score`: a read of a look-alike of the glyph (the same kana with another voiced mark,
    and へ / ヘ with their voiced forms across the scripts) counts that much of an exact read
    in the step's score. The loss is unchanged.
  - `taper`: the step's length is 0.03 row norms times min(1, 2 (1 − score / renders)): full
    while half the renders or fewer read, none when all of them do.
  - `[back]`: a row whose score over its last 2 steps falls more than `drop` under its best
    2-step score returns to the row it stood at then, with its momentum cleared. With `over`
    a 2-step score is a best only above that share of its renders.
- The tuned reader runs on FlashAttention 2 (`reader_attn`, the default from 2026-11-25;
  earlier runs used SDPA), its read and its loss alike, and with `reader_compile` compiled.
  hayai stays on SDPA in fp32: its masks are not FA2's, and FA2 takes no fp32.
- Scenes: 25 text-free scene prompts picked by render (`assets/glyph_scenes.json`), split by
  artist folder into 16 training scenes and 4 read scenes (6, 12, 45, 15) from folders no
  training scene comes from.
- A run reads the start rows and its saves on the read scenes, with blind sheets (the arms
  as rows A / B in a drawn order). Where an earlier run read the start rows on the same
  pages, the config names it (`start_from`) and they are not rendered again.

### The glyph stage (`src/g1.py`)

- The quote holds one trained row. A group is one glyph on one training scene at 4 seeds; a
  pass is every (glyph, training scene) once, shuffled, each pass on new seeds.
- Only the last step (σ 0.5 → 0) is differentiated, once a render.
- A render teaches the one row in its caption: 3.05 s a group, 12 s a row step, 14.2 GiB
  peak on the 16 GB card.
- `[start_shift]`: a row begins at its start plus the mean displacement of an earlier run's
  rows (g1long's 16 rows, its own left out; 0.18 – 0.20 row norms).
- `skip_lone_read`: a training page with one box is not read (the box is the pick either
  way); such a page is `lone` in the log, its `exact` from the loss forward's argmax. No run of
  record has used it.
- The read: every glyph on the read scenes at seeds 0 – 3, 16 pages a glyph and arm. A page
  counts as `solo` when a box reads exactly the glyph, `on_page` when some box holds it, and
  carries `extra`, the letters read beyond the glyph's. The next glyph's `solo` on the same
  pages is the null. hayai reads the tuned reader's boxes (`hy_solo`).

### The sentence stage (`src/s1.py`)

- Rows: the 164 kana, `ー`, `。`, `、`, `～`, the dot row, `♥`, `「`, `」`, from where kmix
  (`kmix_1126_1158`) or jp_v1 has them. `♡`'s row is not trained. A small kana is trained as
  itself (`fold_small` off); kmix's 20 small rows were trained with the full-size glyph.
- Strings: the OCR reader's training labels as its rule normalises them
  (`project/ocr_reader/core/crop_dataset.py::normalize_target`), punctuation in, `♡` written
  as `♥` and a run of two or more `・` as one `…` (in the quote and in the target), lengths
  2 – 12, the ruler's 5-grams and `$CHARACTER_REF_EXCLUDE` held out
  (`probes/sent_strings.py` → a local file the config names). `[draw]` leaves out strings
  heavy in `ー` / `~`.
- The quote holds the string with no line break. The target takes its breaks from the page's
  own read, aligned to the string, so where a render wraps costs nothing.
- A group is one string on one training scene at `train_seeds` seeds; a string is drawn once
  and the scenes go round. A render's gradient goes to every row in its quote; a row steps
  when 16 renders that hold it have come in, whatever strings they were.
- The backward covers the last two steps (`back_steps = 2`, σ 0.75 → 0.5 → 0): the image's
  cotangent goes back through the last step to its input and from there through the step
  before; the two steps' embedding cotangents are summed. One step's graph is alive at a
  time.
- The loss (`loss = "window"`, `window_rows = "each"`): on a page with a box a row takes its
  own glyph's cross-entropy on the window its quote token's cross-attention points at
  (`src/loc.py`: blocks 8 and 13 – 15 by a hook outside the compiled blocks), a backward a
  row. With `skip_hit` a position the page's read already hits takes no window and no
  backward. The detector's term on a page with no box stays.
- A row's score for the guards: over the renders that hold it, the positions of its glyph
  the aligned read hits; a look-alike counts 0.5.
- The read, at each save, against kmix's rows: val strings in no train label by length
  (2 – 4, 5 – 8, 9 – 12), with and without punctuation (`textnorm.exact_key` / the line's
  `norm`): the whole string alone, positions hit, glyphs drawn over asked, under both
  readers; a sample of train strings not trained on; the 164 kana alone. Blind sheets, a
  string a file.

## What the reports say

The glyph stage:

- 16 rows read alone on 59 of 256 held-out pages before and on 136, 175 and 199 after 8, 16
  and 24 steps ([g1long](archive/reports/g1long_2026_11_25.md)). On the base model in a sign prompt
  (8 renders) the trained rows letter one clean glyph.
- A step's direction is mostly noise, and the noise costs little; twice the step loses. A
  trained row keeps its length and its neighbours and turns about 15°; part of the turn is
  common to all rows, and that part alone lifts a row that spent no render
  ([row_line](archive/reports/row_line_2026_11_25.md)).
- From the shared start a row needs 4 – 6 steps fewer, and a row that already reads can be
  pushed past its glyph by the fixed step ([g1shared](archive/reports/g1shared_2026_11_25.md)).
- The 164 kana rows read alone on 0.28 of held-out pages before, 0.50 at the shared start and
  0.68 after 6 steps ([g2a](archive/reports/g2a_2026_11_25.md)); from jp_v1's rows with no shared
  start, 0.65 after 16 guarded steps, and a few (え, っ, の) do not start
  ([g2b](archive/reports/g2b_2026_11_26.md)).
- Without the `speech bubble` tag the rows read at about EN's level, and the model letters
  EN on a fifth of plain pages ([plain](archive/reports/plain_2026_11_25.md),
  [pscreen](archive/reports/pscreen_2026_11_25.md), [mix](archive/reports/mix_2026_11_26.md)).

The sentence stage:

- kmix's rows already carry into short strings, and the read falls with length; most of what
  a page letters is set in the step σ 0.75 → 0.5 ([sent_prep](reports/sent_prep_2026_11_26.md)).
- A clause of two strings draws one bubble with the two strings' kana mixed, in EN as well
  ([multi_read](reports/multi_read_2026_11_26.md)).
- The string's cross-entropy through two steps moves nothing; string pages come out wide where
  the labels stand in columns ([sentm](reports/sentm_2026_11_26.md)).
- A glyph's cross-entropy on its own attention window moves the read of strings
  ([sentw](reports/sentw_2026_11_26.md)).
- A line break in the quote does not move the wrap, and the reader's line break gives a break
  row no shared direction ([sent_scale](reports/sent_scale_2026_11_26.md)).

## Rules

Every read is the tuned PaddleOCR-VL-1.6 reader (`anime_tools.ocr.reader.OcrReader`); stock
VL16 is not a reader.

The line (the user's calls, 2026-11-13):

- The goal is the intended sentence (order, length, no invented text), not crisper glyphs.
  Rewards and reads are sequence-level. SFX is in scope and needs a style guard.
- Prompt pool: every caption with a JA text / SFX clause, only the quoted strings swapped. It
  holds out the ruler's images and 5-grams and the `$CHARACTER_REF_EXCLUDE` tags. No rating
  filter.
- Rows can learn order: order reaches the DiT through the adapter's RoPE over the T5 stream.
- For GRPO (archived): SDE a = 0.7; arm A starts with the `src/grpo.py` fixes (summed log π,
  G 16 – 24) and is a sample-count fallback, not an order fix.

The glyph stage (the user's calls, 2026-11-25 – 26):

- G-D is the arm; GRPO on the glyph reward is out.
- Scenes come from artists whose captions hold little text, or from captions that had no OCR
  clause, picked by the relaxed rule of
  [glyph_scenes](archive/reports/glyph_scenes_2026_11_25.md). Training and read scenes are split by
  artist folder.
- A box read under a confidence of about 0.6 counts as nothing drawn. A page with no box
  takes the detector's own score as its signal to draw.
- No page term in the loss. No x_3 cache.
- hayai v2.7's cross-entropy on the same crop at weight 0.2 (two readers for robustness).
- The shifted start is not read as an arm of its own (`read_shift = true` reads it).
- A row is guarded by its own training-side count: `taper`, `[back]` with `over` 0.5,
  `near_score` 0.5.
- Words are real words only, no random strings.

The sentence stage (the user's calls, 2026-11-26):

- Kana words and sentences come before the kanji.
- The read's strings come from the reader's val split and are in no train label.
- The backward covers the last two steps.
- Every page is read (`skip_lone_read` does not apply: the breaks and the score need it).

## Files

- [plan.md](plan.md) — the current run, what is open, what is parked, the runs done.
- [issues.md](issues.md) — what was found and is not fixed or planned yet.
- [turbo_refl.md](turbo_refl.md) — the method as a note: what it is, what has been measured,
  the conditions it seems to need, open points.
- `src/` — the loops and the path (below).
- `probes/`, `configs/`, `reports/` — the sentence stage's (from `kmix_1126_1158` on), and the
  probes the live code imports or the open items use.
- `archive/` — the GRPO and glyph stages' records: the superseded proposals and the sentence
  stage's planning file, `archive/reports/`, `archive/configs/` (every config before
  `kmix_1126_1158`, and the `time*` timings) and `archive/probes/` (probes no live code
  imports). Archived probes are not kept runnable where they sit; `git mv` one back to run it.
  `src/g1.py`'s `start_from` looks in `archive/configs/` when `configs/` has no such run.
- `assets/` — scene prompts (`glyph_scenes.json`, `plain_scenes.json`, gitignored) and small
  run inputs.
- `results/` (gitignored) — one directory a probe run; renders under `output/cjk_rl/`.

### reports/ (the sentence stage)

| report | runs | what |
|---|---|---|
| [sent_prep](reports/sent_prep_2026_11_26.md) | `wread_1126_1207`, `tswap_1126_1209` | kana strings on kmix's rows; a render's text swapped at a step; the quote's characters and the labels |
| [multi_read](reports/multi_read_2026_11_26.md) | `wmulti_1126_1252`, `enmulti_1126_1258`, `tread_1126_1306` | a clause of two or three short strings: no kana pair page of 288 with both read |
| [sentm](reports/sentm_2026_11_26.md) | `sentm_1126_1331` | the sentence micro arm on the string's cross-entropy: val positions hit 0.345 → 0.331; pages wide where labels stand in columns |
| [sentw](reports/sentw_2026_11_26.md) | `sgrad_*`, `aloc_1126_1501`, `sentw_1126_1528` | the window loss and its micro arm: val positions hit 0.345 → 0.422; passed |
| [sent_scale](reports/sent_scale_2026_11_26.md) | `qbreak_1126_1751`, `bgrad*_1126_18*`, `bplace_1126_1840` | before the run at scale: line breaks in the quote, a break row's gradient, a page's own wrap as label (17 pages of 480) |

### archive/reports/ (the GRPO and glyph stages)

| report | runs | what |
|---|---|---|
| [ruler_turbo](archive/reports/ruler_turbo_2026_11_13.md) | – | the 64-string ruler on base / merged14 / merged14 + v3: the soup loses the string, the turbo is a fair proxy |
| [p0](archive/reports/p0_2026_11_13.md) | – | GRPO's P0: sampler (SDE a = 0.7), group spread, cost of one update |
| [p1](archive/reports/p1_2026_11_13.md) | – | P1: 300 GRPO groups on the rows change nothing on the ruler |
| [reward_reader](archive/reports/reward_reader_2026_11_13.md) | – | the reader beside the update: int8 / fp8 dropped, group-batched read, tuned reader only |
| [position_read](archive/reports/position_read_2026_11_25.md) | – | the ruler's reads aligned per bubble: hits by position, substitutions, the EN references |
| [turbo_read](archive/reports/turbo_read_2026_11_25.md) | – | fresh renders on `m15v11`: the ruler at four seeds, a kanji grid |
| [glyph_g0](archive/reports/glyph_g0_2026_11_25.md) | – | the tagless frame with and without `speech bubble`, the terms' spread, the pixel head against the VAE |
| [glyph_scenes](archive/reports/glyph_scenes_2026_11_25.md) | – | the scene pool picked by render (25 of 128), every glyph on four scenes |
| [gd_g0d](archive/reports/gd_g0d_2026_11_25.md) | `g0d_1125_1119` | G-D's path built and checked: memory, seconds, the row gradient's size and direction |
| [g1micro](archive/reports/g1micro_2026_11_25.md) | `g1micro_1125_1148` | 16 rows, 8 steps: solo 59 → 132 of 256 |
| [reader_pair](archive/reports/reader_pair_2026_11_25.md) | `hayai_1125_1234` | hayai v2.7 beside the tuned reader on g1micro's pages: solo 60 → 125 |
| [hayai_g0](archive/reports/hayai_g0_2026_11_25.md) | `g0h_1125_1243` | hayai's cross-entropy on G1's path: the two row gradients' sizes and cosine |
| [g1long](archive/reports/g1long_2026_11_25.md) | `g1long_1125_1335`, `sign_1125_1504` | 16 rows, 24 steps: solo 59 → 136 → 175 → 199; the rows on the base model in a sign prompt |
| [reader_room](archive/reports/reader_room_2026_11_25.md) | `room_*`, `lone_1125_2200`, `time{a,b,c}_*` | the reader side of a group timed; `skip_lone_read`; `reader_compile`: a group 3.25 → 2.90 s |
| [row_line](archive/reports/row_line_2026_11_25.md) | `line_1125_1517`, `line_1125_1527` | a step against a walk on noise; the rows at twice their displacement (122) and at the others' mean displacement (112) |
| [g1shared](archive/reports/g1shared_2026_11_25.md) | `g1shared_1125_1538` | from the shared start: solo 112 → 148 → 156 at 0, 4, 8 steps |
| [g2a](archive/reports/g2a_2026_11_25.md) | `g2a_1125_1642` | the 164 kana, 6 steps from the shared start: solo 743 → 1 322 → 1 686 → 1 785 of 2 624 |
| [plain](archive/reports/plain_2026_11_25.md) | `plain_1125_2249`, `en{plain,bubble}_1125_2258` | g2a's rows with no bubble tag: 22 → 93 of 256; EN words 71 (bubble 104) |
| [pscreen](archive/reports/pscreen_2026_11_25.md) | `pscreen_1125_2306` | 160 scenes by their EN count with no bubble tag: 0.205 of renders |
| [mix](archive/reports/mix_2026_11_26.md) | `mix_1125_2338` | 16 rows trained in both frames: under g2a's in both (plain 202 / 238, bubble 297 / 329 of 512) |
| [g2b](archive/reports/g2b_2026_11_26.md) | `g2b_1126_0058`, `split_1126_0910` | the kana from jp_v1's rows, 16 guarded steps: solo 743 → 1 716; 2 002 crops the two readers split on |

### src/ and probes/

| file | what |
|---|---|
| `src/pool.py` | P1's prompt pool: training captions with each quoted OCR string swapped for another of its length → `output/cjk_rl/pool/pool.json` |
| `src/reward.py` | GRPO's read terms: r_seq over every asked string (one bubble each), g_p over all of them |
| `src/grpo.py` | the GRPO loop (arm R) |
| `src/gd.py` | G-D's path: the policy with a step-by-step backward to the rows; the reader's cross-entropy and the detector's score on a page tensor |
| `src/g1.py` | the glyph loop and its read |
| `src/s1.py` | the sentence loop and its read |
| `src/loc.py` | a glyph's place on a generated page from the quote tokens' cross-attention |
| `probes/turbo_read.py`, `position_read.py` (CPU) | the render and read every loop and probe imports; the ruler and a kanji grid on `m15v11` (the transfer read before the v2 preview); the ruler's reads per position |
| `probes/glyph_g0.py`, `glyph_scenes.py` | the frame, captions and scenes the loops import |
| `probes/grpo_p0.py` | GRPO's P0; `turbo_read.py` takes its SDE sampler |
| `probes/plain_scenes.py` | scenes screened by their EN count (the screen a multi-string scene pool would reuse) |
| `probes/base_sign.py` | a save's rows on another model of the ruler's table, in a prompt of its config (the transfer read) |
| `probes/row_line.py`, `g2_rows.py` (CPU) | a run's saved rows read with no training (`wread_1126_1207` ran on it); a run's rows sorted into the plan's groups with labelled sheets |
| `probes/word_read.py`, `multi_read.py` (CPU), `templ_read.py` | string pages per position; clauses of several strings; pool templates of two dialogue strings |
| `probes/sent_strings.py` (CPU), `orient_read.py`, `text_swap.py` | the sentence loop's strings from the reader's labels; which way string pages are lettered; a render's text swapped at a step |
| `probes/sent_grad.py`, `attn_loc.py` | a row's gradient a render by step and by loss; whether a quote token's attention window holds its glyph |
| `probes/quote_break.py`, `break_grad.py`, `break_place.py` | whether a page wraps where the quote breaks; a break row's gradient from the reader's line break; the denoising error at a page's own wrap |
| `archive/probes/` | `gd_g0d.py` (G-D's row gradient), `hayai_g0.py` (hayai's loss on G1's path), `reader_pair.py` / `reader_split.py` (both readers on the same crops; the crops they split on), `reader_room.py` (the reader side of a group timed), `reader_batch.py` (GRPO's reward reader), `row_walk.py` / `row_geom.py` (CPU; how far rows moved, where they sit in the table), `rows_mix.py` (CPU; kmix's row mix) |
