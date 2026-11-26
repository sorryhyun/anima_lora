# Sentence RL — kana sentences from kmix (rough plan, opened 2026-11-26)

> Archived 2026-11-26: superseded; the line's live state is in `../README.md` and `../plan.md`.

What the sentence run trains, on what strings, and how it is read. The run's settings,
question and gate go in its config header when it is submitted; this file is the plan before
that and names the records it rests on. Where it stands in the line: [plan.md](../plan.md) item 7.

## State (2026-11-26)

The micro arm on the string's cross-entropy ran (`sentm_1126_1331`, 400 groups): the read
of strings did not move ([reports/sentm_2026_11_26.md](../reports/sentm_2026_11_26.md)). The
user took the line up again the same day, and a second micro arm on the window loss
(`sentw_1126_1528`, the same 400 strings) moved it: the val strings' positions hit
0.345 → 0.422 where `sentm` went to 0.331, the kana alone 351 → 345 of 656
([reports/sentw_2026_11_26.md](../reports/sentw_2026_11_26.md), with the three probes that
led to the loss). The user passed it (11-26: "블라인드 결과는 그냥 안봐도 합격이라 괜찮을듯",
one of the 36 sheets read). What runs next is the run at scale; its settings are under
"The run at scale" below.

The run at scale is submitted (`sents_1126_1856`, 11-26): the user, after `bplace_1126_1840`, "그래
학습 올리자 가보자구". The quote holds no break; the break's place as a page's own wrap
is read again on this run's rows.

## Decided (the user's calls, 2026-11-26)

- Start: the kana rows `kmix_1126_1158` (g2b's 16-step rows with ten of g2a's).
- Strings: the OCR reader's training labels, as the reader's rule normalises them
  (`project/ocr_reader/core/crop_dataset.py::normalize_target`, `TARGET_NORM = 10`).
  Punctuation stays in. A blank heart is written as the filled one (`♡` → `♥`) in the quote
  and in the loss's target: the reader in use does not read `♡` yet.
- From the draw after `sent_strings_s1126` (the micro arm's), a run of two or more `・` is
  written as one `…`, in the quote and in the loss's target (`probes/sent_strings.py`). The
  labels hold such a run in 2 343 of 115 529 train mains and 294 of 11 634 val mains
  (`load_mainsub`, counted 11-26); `sent_strings_s1126` holds it in 1 494 of its 24 891
  train strings and 131 of its 1 844 val strings.
- The read's strings come from the reader's val split and are in no train label.
- A small kana is trained as itself, its target and its read the small glyph (`fold_small`
  off). kmix's 20 small rows were trained with the full-size glyph as their target.
- Guards: `taper`, `near_score` 0.5, `[back]` with `over` 0.5.
- Ship: the rows this run trains from kmix, if they read well, go out as a v2 preview with
  the kana alone. The kanji are taken as a matter of scale and come after.
- The backward covers the last two steps, σ 0.75 → 0.5 → 0 (on `tswap_1126_1209`'s table).
- The tuned reader runs on FlashAttention 2 with `reader_compile` on (on `timeb_1126_1238`'s
  seconds, `reports/reader_room_2026_11_25.md`, third part). `skip_lone_read` does not apply:
  the run needs every page's read for the breaks and the score.

## Facts it rests on

- `wread_1126_1207` (forward only, 48 pool strings, 96 pages a length, jp_v1's rows → kmix's
  rows): the whole string alone in a box 19 → 52 at 2 glyphs, 2 → 21 at 3, 2 → 13 at 4,
  0 → 0 at 5 – 8; share of positions hit 0.41 → 0.76, 0.28 → 0.62, 0.27 → 0.53, 0.20 → 0.35;
  kmix's hit share by position 0.66, 0.52, 0.47, 0.38, 0.26; glyphs drawn over glyphs asked
  0.84 at 5 – 8; the first glyph alone on 11 of 384 pages. Spread unknown.
  `results/20261126-1215-row_line-wread_1126_1207/` (local).
- `tswap_1126_1209` (forward only, kmix's rows: a render's first k steps on string A, the
  rest on string B; k = 0 is B's own render, k = 2 gives B to σ 0.75 → 0.5 → 0, k = 3 to
  σ 0.5 → 0 alone; small kana folded; 128 pages a cell for a glyph, 96 for a length).
  Spread unknown. `results/20261126-1227-text_swap-tswap_1126_1209/` (local).

  | | B alone, k 0 / 2 / 3 | A alone, k 0 / 2 / 3 | B's positions hit, k 0 / 2 / 3 | A's positions hit, k 0 / 2 / 3 |
  |---|---|---|---|---|
  | a glyph | 80 / 71 / 14 | 0 / 7 / 49 | 0.70 / 0.63 / 0.18 | 0.00 / 0.06 / 0.41 |
  | 2 glyphs | 63 / 46 / 10 | 0 / 1 / 19 | 0.82 / 0.72 / 0.28 | 0.05 / 0.06 / 0.37 |
  | 3 | 22 / 20 / 5 | 0 / 0 / 4 | 0.64 / 0.57 / 0.28 | 0.12 / 0.12 / 0.31 |
  | 4 | 13 / 7 / 1 | 0 / 0 / 1 | 0.55 / 0.50 / 0.27 | 0.04 / 0.05 / 0.20 |
  | 5 – 8 | 0 / 0 / 0 | 0 / 0 / 0 | 0.36 / 0.35 / 0.22 | 0.12 / 0.11 / 0.18 |
- `wmulti_1126_1252`, `enmulti_1126_1258` (forward only; a clause of two or three short
  strings, `Text reads as "A", "B".`, on kmix's rows and with EN words;
  [reports/multi_read_2026_11_26.md](../reports/multi_read_2026_11_26.md)): no page of 288 kana
  pairs or 128 EN pairs with both strings each read by a box; 1.24 and 1.02 boxes a page; a
  kana string read exactly by a box on 86 of 288 pages alone, 25 and 15 of 288 as a pair's
  first and second. Spread unknown.
- `tread_1126_1306` (forward only; 24 pool templates of two dialogue strings, their strings
  swapped for the same pairs; the same report, second part): 2.27 boxes a kana pair page and
  2.05 with one string asked; a kana string read exactly by a box on 8 of 96 pages alone, 1
  and 1 of 96 in a pair. Spread unknown.
- `sentm_1126_1331` (the micro arm; [reports/sentm_2026_11_26.md](../reports/sentm_2026_11_26.md)):
  the val strings' positions hit 0.345 → 0.331 over 288 pages, the whole string 10 → 6 of
  96 at 2 – 4 glyphs and 0 at 5 – 12, the kana alone 351 → 354 of 656; 401 row steps on
  103 rows; 4.26 s a group, peak 14.12 GiB. Spread unknown.
- Which way a page is lettered (the same report): the draw's source crops are vertical
  on 0.83 – 0.90 of them; of the micro arm's 384 string pages at the start rows 73 are
  tall and 250 wide, and the tall ones are hit at 0.23 of their positions against 0.40.
- Where a quote's characters go on the T5 side (the active pack with `glyph_route`, jp_v1's
  delta in row norms): a kana its own row (1.2 – 1.5); `ー` its own row (1.52; `―` folds to
  it); `「` `」` 0.9; `。` 0.48, `、` 0.35; `〜` and `~` one row, 0.42; `♥` 0.46, `♡` 0.48, a row
  each; a dot run inside a string one row, 0.38; `！` `？` fold to ASCII and are T5's own
  tokens, with no row to train.
- `sgrad_1126_1436`, `aloc_1126_1501`, `sgrad_1126_1511`
  ([reports/sentw_2026_11_26.md](../reports/sentw_2026_11_26.md)): a row's renders agree on
  a direction at 0.032 on its glyph alone and 0.003 on a string's cross-entropy, with the
  one-step backward as with two; a quote token's cross-attention (blocks 13 – 15, σ 0.75
  and 0.5) points at its glyph, the window about it read as that glyph on 0.767 of the
  positions the page's read hit; the glyph's cross-entropy on that window gives its row
  0.016 through two steps, and the windows of a whole string summed give it nothing.
  Spread unknown.
- `sentw_1126_1528` (the same report): positions hit on the val strings 0.345 → 0.361 →
  0.422 at 200 and 400 groups, the whole string 10 → 14 of 288 pages, train labels not
  trained on 0.402 → 0.496, the kana alone 351 → 345; a row's successive gradients at
  0.041 against `sentm`'s 0.008; 12.9 s a group. Spread unknown.
- The reader's main labels (`load_mainsub`, `OCR:` rows), distinct targets made of kana and
  those marks: train 28 975 (2 – 4 glyphs 9 552, 5 – 8 11 739, 9 – 12 4 570, 13 – 20 2 565,
  over 20 460), val 3 517, of them 2 396 in no train label. 12 547 of the train targets hold
  a line break; `…` is in 5 738, `ー` in 5 628, `!` in 4 913, `?` in 2 827, `♡` in 244.
- The current pool (`output/cjk_rl/pool/pool.json`) holds 355 kana-only strings.
- `reports/gd_g0d_2026_11_25.md` (single glyphs): a step's share of the four-step ce gradient
  on a row, steps 0 – 3 (σ 1 → 0.9 → 0.75 → 0.5 → 0): 0.48 / 0.40 / 0.33 / 0.08; one step's
  backward 0.95 s for a group's 4 renders (a group 3.0 – 3.2 s with the last step's alone).

## The run

What stays as the glyph stage has it (README, "The glyph stage, as it runs"): the bubble
frame, 4 Euler steps at cfg 1 and shift 3, the pixel head, the box picked by its sequence
score to the string, the tuned reader's cross-entropy of the string on the picked crop plus
0.2 of hayai's, the detector's term on a page with no box, no page term, the fixed-length
step along a row's momentum.

What changes:

- The quote holds a string, and a render's gradient goes to every row in it. A row steps when
  16 renders that hold it have come in, whatever strings they were.
- Rows trained: the 164 kana, `ー`, `。`, `、`, `～`, the dot row, `♥`, `「`, `」`, each from
  where kmix or jp_v1 has it. `♡`'s row is not in the run.
- A group is one string on one training scene at 4 seeds. A string is drawn once; the scenes
  go round the 16 training scenes.
- Strings: drawn from the train labels over lengths 2 – 12 (longer ones wait), with the
  ruler's 5-grams and `$CHARACTER_REF_EXCLUDE` held out as the pool's rule has it. The draw
  is a local file (Manga109-s is not redistributed); a config names the file and its seed.
- Line breaks: the quote holds the string with no break. The loss's target takes its breaks
  from the page's own read, aligned to the string, so where a render wraps costs nothing.
- A row's score for the guards: over the renders that hold it, the positions of its glyph
  that the aligned read hits; a look-alike counts 0.5.
- The loss, from `sentw_1126_1528` on (`loss = "window"`, `window_rows = "each"`): on a
  page with a box a row takes the gradient of its own glyph read alone on the window its
  token's cross-attention gives (`src/loc.py`), a backward a row, in place of its part of
  the string's cross-entropy on the box. The box, the row's score and the detector's term
  stay.
- The backward: the last two steps (σ 0.75 → 0.5 → 0), the user's call on
  `tswap_1126_1209`'s table. The image's cotangent goes back through the last step to its
  input, and from there through the step before; the two steps' cotangents of the embedding
  are summed before the rows. One step's graph is alive at a time. A one-step arm on the
  same draw of strings is not planned.

The read, at each save, against kmix's rows:

- Strings of the val split that are in no train label, by length (2 – 4, 5 – 8, 9 – 12), on
  the read scenes and seeds: the whole string alone, positions hit, glyphs drawn over asked,
  under the tuned reader and hayai. Two counts side by side: with punctuation
  (`textnorm.exact_key`) and without it (the line's `norm`, as every read so far).
- A sample of train-label strings on the same scenes, for the tuned reader's count over
  hayai's on strings it was trained to write against strings it was not.
- The 164 kana alone, for what the run costs them.
- Blind sheets, a string a file, kmix's rows against the run's. The user's read decides.

Cost, to be set in the config: a group is about 4.0 s with the two-step backward (3.05 s
with the last step's alone, and 0.95 s a step's backward), so 1 000 groups are 67 min. A group of a 7-glyph string is 28
row-renders, against 4 in the glyph stage.

## The run at scale — settings (2026-11-26, submitted as `sents_1126_1856`)

`sentw_1126_1528`'s settings with the lines below changed. Records:
[reports/sentw_2026_11_26.md](../reports/sentw_2026_11_26.md) (sixth part) and
[reports/sent_scale_2026_11_26.md](../reports/sent_scale_2026_11_26.md).

### Facts

| setting | value | set by |
|---|---|---|
| a position the page's read already hits | no window and no backward (`skip_hit = true`) | the user: "이미 읽힌 위치의 bwd 빼는건 찬성" |
| a position whose window is its neighbour's | stays in | the user: "겹치는건 포함하는걸로 하고" |
| seeds a string | 1 (`train_seeds = 1`) | the user: "본 학습에서는 seed 1로 하는게 좋을듯" |
| a row's budget | 64 renders that backed it (`accum = 16`, `max_steps = 4`) | the user: "row당 예산은 일단 64로 생각하고 있어" |
| strings heavy in `ー` / `~` | left out of the draw: a run of three or more, or over half of the string (`[draw] glyphs = "ー~"`, `run_max = 2`, `share_max = 0.5`); 478 of 24 891 | the user asked for the thinning ("적당히 샘플링 시 필터링하면 좋을듯"); the thresholds are Claude's pick |
| the rows the draw cannot fill (ヂ, ヅ, ヲ …) | left as kmix has them, no glyph groups | the user: "di du yo는 보류하지 뭐 실제로 쓸거같진 않아" |
| size | an estimated 3 h | the user: "일단 예산범위는 추정 3h로 두자" |
| strings and saves | `groups = 8000` of the thinned draw in drawn order, `save_groups = [4000, 8000]` | Claude's fit to the 3 h |
| the mix of lengths, glyph groups for the kana alone, the start rows | the labels' own mix, none, kmix: as `sentw_1126_1528` | not set by the user |
| line breaks | none in the quote | the user: "줄바꿈 어떻게 할지는 일단 placeholder로", then, on the proposal to run without breaks and read `break_place.py` again on the run's rows: "그래 학습 올리자 가보자구" |

- The fit to 3 h, a simulation over the thinned draw at a hit share of 0.36 (the start
  rows' on string pages) with 0.85 s a render and 0.58 s a row-render: 8 000 strings are
  5 311 renders and 8 128 row-renders, 2.6 h, and 85 of the 171 rows reach 64; with the
  start read and two saves' reads at about 10 min each, 3.1 h. 12 000 strings: 3.0 h of
  training, 111 rows. With the neighbour's-window positions left out, 8 000 strings would
  bring 52 rows to 64.
- `src/s1.py` holds `skip_hit`, `train_seeds` and `[draw]`; `configs/smoke_gate.toml` is
  the smoke of these settings.
- Line breaks, as the run stands: the quote holds none. `qbreak_1126_1751`: a line feed
  or the literal backslash-n in the quote does not move where the page wraps.
  `bgrad_1126_1806`, `bgradw_1126_1815`, `bgradt_1126_1815`: a break row's gradient from
  the reader's line-break token, from three starts. `bplace_1126_1840`: a page's own
  wrap as the label of a break's place, 17 pages kept of 480
  ([reports/sent_scale_2026_11_26.md](../reports/sent_scale_2026_11_26.md), last parts).

### Interpretation

- The neighbour's-window positions are where a dropped glyph gets its only signal, and
  the passed micro arm trained with them in. What to watch for with them in: the glyphs
  drawn over asked at 5 – 8 glyphs (`sentw`: 0.94 → 1.17) and repeats on the sheets.
- The budget spreads the steps over the rows where `sentw` put most of them on a few
  (8 rows took 10 steps or more, 65 none): 4 steps a row is at most 0.12 row norms.

## Interpretation

- The rows trained alone already carry into short strings, so the run starts from a working
  state, and what it has to mend is glyphs dropped from a string and the fall along its
  length.
- A row is one vector at every position, so order and count over a long string may be out
  of its reach; the read by length shows where that begins.
- The swap puts most of what a page letters in the step σ 0.75 → 0.5: with B's text on the
  last two steps the page is close to B's own render, and with it on the last step alone
  the page mostly keeps A, for a single glyph as for a string. The last step still moves
  the read part of the way, and its gradient alone took the single glyphs from 0.28 to 0.68
  of their pages, since a row moved by it acts at every step. So the swap does not say the
  one-step backward fails on strings; it says the two-step backward reaches the step where
  the string is set.
- Reusing the reader's labels as prompts does not touch the reader. What it can do is flatter
  the read: a reader writes a sentence it was trained on more readily from poor glyphs. The
  val-only strings, hayai beside the tuned reader and the user's sheets are the checks.
- Frequent rows (い, っ, ん, ー) will take many more steps than rare ones. The guards shorten
  a row's step as it reads; whether a cap on a row's steps is needed as well is open.

## Open

- A cap on a row's steps beside the guards.
- The mix of lengths in the draw, and when strings over 12 glyphs come in.
- `♡` at ship: its row is left as jp_v1 has it; a fold of `♡` to `♥` in the pack would send
  both to the trained row.
- The kana that few label strings hold (ヂ, ヅ, ゔ, ヲ): whether single-glyph groups are mixed
  in for them.

- Captions of several short strings in one clause (the user's suggestion, 11-26: "짧은 단어
  여러개 들어가는 케이스도 있으면 어떨까 해"). Not in the run as planned above. On the read
  scenes the model draws one bubble for two strings, in EN as well, and the kana in it are a
  mix of the two strings
  ([reports/multi_read_2026_11_26.md](../reports/multi_read_2026_11_26.md)). What it needs
  before it can go into the draw, each the user's call:
  - scenes that draw a bubble a string. The pool's templates that held two dialogue strings
    were read on the user's call (`tread_1126_1306`): they draw two boxes whether one string
    or two are asked, and the asked string is read on 8 of 96 pages alone. Left: new
    text-free scenes screened by EN pairs under `Text reads as`, as `pscreen_1125_2306`
    screened the plain frame;
  - the loss on such a page: each string on a box of its own (`reward.match` assigns them),
    and what a string with no box left takes;
  - its share of the groups.

- Vertical lettering (the user, 11-26): strings of three glyphs or more that are not
  SFX-like should come out in columns, with the prompt as it is. A string's label gives
  the way it stood (`orient`). Levers named and not tried: a render weighted by whether
  its box stands that way, `back_steps = 4`, scenes with tall bubbles. The tall pages'
  lower count is the model's lettering, not the reader's read (the user, 11-26: the reader
  is trained mostly on columns), so no reader-side fix comes before these.
- Line breaks: not in the run at scale. A page's own wrap as the label of a break's place
  (`bplace_1126_1840`, 17 pages kept of 480 on kmix's rows) is read again on the run's
  rows; if it gives the break row a direction, a micro arm on that loss with the glyph
  rows held is proposed, and how the break's caption is encoded after the read (the
  text encoder kept on the GPU, or no line feed on the Qwen side) is settled there.
- Half the backward by reading the loss off the σ 0.75 step's own x̂₀
  ([reports/sentw_2026_11_26.md](../reports/sentw_2026_11_26.md), last part). Not looked at.

## Order of work

1. `tswap_1126_1209`'s read.
2. Done: the loop is `src/s1.py` beside `src/g1.py` (a quote of several rows, a row's
   renders counted over strings, the score per position, the target's breaks from the
   read), the two-step backward `back_steps` with `gd.Policy.inputs`, `last(wrt_x)` and
   `vjp`.
3. Done: `probes/sent_strings.py` → `output/cjk_rl/pool/sent_strings_s1126.json` (local),
   24 891 train strings and 1 844 val strings in no train label.
4. Done to the micro arm: the smoke and `sentm_1126_1331`.
5. Done: the gradient and attention probes, the window loss in `src/s1.py` with
   `src/loc.py`, and its micro arm `sentw_1126_1528`, passed by the user.
6. The run at scale: `sents_1126_1856`, submitted 11-26.
7. `probes/break_place.py` on that run's rows.
