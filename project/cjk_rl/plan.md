# Plan

The current run, what is open, what is parked, and the runs done. How a run goes is in
[README.md](README.md), what the runs measured in [reports/](reports). A run's settings,
question and gate go in its config header (`configs/<handle>_mmdd_hhmm.toml`) when it is
submitted, and this file then names the run.

## Now: the sentence run at scale (`sents_1126_1856`, run 2026-11-26)

### Facts

- Done: [sents](reports/sents_2026_11_26.md). Val positions hit 0.345 → 0.385 → 0.391 at
  the start, 4 000 groups and the end (`sentw_1126_1528` at 400 groups: 0.422), whole val
  strings 10 → 14 of 288 pages, the kana alone 351 → 361 of 656. The run walked its 8 000
  strings in 5 232 groups (2 768 strings held only rows at `max_steps`); 88 of the 172
  rows reached 4 steps.
- The blind sheets (36 val strings, start against end) are with the user (11-26); no read
  recorded.
- The end rows are the pack `anima_cjk_vocab_pack_jp_v1_sents`, baked and linked into
  ComfyUI on the user's word (11-26: "해당 학습된거 comfy cjk vocab pack loader로 사용할 수
  있게 해줘 여기서 써보게").
- `bplace_1126_2131` (the break-place probe on the end rows) was stopped at 36 of 120
  strings to free the GPU for the user; no result.
- The user (11-26), on the run's summary: "일단은 ocr이나 다른 부수요소 (이슈?) 를 손보고
  다시 도전해볼 생각이야". The line waits on the reader and the side issues
  ([issues.md](issues.md)) before another run.

- Submitted on the user's word after `bplace_1126_1840`: "그래 학습 올리자 가보자구". Its
  question, gate branches, read and cost are in the config header.
- It is `sentw_1126_1528` with the settings below changed; the micro arm was passed by the
  user (11-26: "블라인드 결과는 그냥 안봐도 합격이라 괜찮을듯", one of 36 sheets read).

| setting | value | set by |
|---|---|---|
| a position the page's read already hits | no window and no backward (`skip_hit = true`) | the user: "이미 읽힌 위치의 bwd 빼는건 찬성" |
| a position whose window is its neighbour's | stays in | the user: "겹치는건 포함하는걸로 하고" |
| seeds a string | 1 (`train_seeds = 1`) | the user: "본 학습에서는 seed 1로 하는게 좋을듯" |
| a row's budget | 64 renders that backed it (`accum = 16`, `max_steps = 4`) | the user: "row당 예산은 일단 64로 생각하고 있어" |
| strings heavy in `ー` / `~` | left out: a run of three or more, or over half the string (`[draw]`, `run_max = 2`, `share_max = 0.5`); 478 of 24 891 | the user asked for the thinning ("적당히 샘플링 시 필터링하면 좋을듯"); the thresholds are Claude's pick |
| the rows the draw cannot fill (ヂ, ヅ, ヲ …) | left as kmix has them, no glyph groups | the user: "di du yo는 보류하지 뭐 실제로 쓸거같진 않아" |
| size | an estimated 3 h | the user: "일단 예산범위는 추정 3h로 두자" |
| strings and saves | `groups = 8000` in drawn order, `save_groups = [4000, 8000]` | Claude's fit to the 3 h |
| the mix of lengths, glyph groups, the start rows | the labels' own mix, none, kmix: as `sentw_1126_1528` | not set by the user |
| line breaks | none in the quote | the user: "줄바꿈 어떻게 할지는 일단 placeholder로", then the go above |

- The fit to 3 h, a simulation over the thinned draw at a hit share of 0.36 (the start rows'
  on string pages) with 0.85 s a render and 0.58 s a row-render: 8 000 strings are 5 311
  renders and 8 128 row-renders, 2.6 h, and 85 of the 171 rows reach 64; with three reads of
  about 10 min, 3.1 h. 12 000 strings: 3.0 h of training, 111 rows. With the neighbour's-window
  positions left out, 8 000 strings would bring 52 rows to 64.
- `configs/smoke_gate.toml` is the smoke of these settings.

### Interpretation

- The neighbour's-window positions are where a dropped glyph gets its only signal, and the
  passed micro arm trained with them in. What to watch with them in: glyphs drawn over asked
  at 5 – 8 glyphs (`sentw`: 0.94 → 1.17) and repeats on the sheets.
- The budget spreads the steps over the rows where `sentw` put most of them on a few (8 rows
  took 10 steps or more, 65 none): 4 steps a row is at most 0.12 row norms.
- The run's read is the header's second branch (the read stays at `sentw`'s), and the
  budget is what ended it: the five rows most strings hold were at 4 steps by group 587.
  The report's Interpretation has the rest.

### Next

1. The reader and the side issues first (the user, 11-26); the next sentence run comes
   after. Proposed for then, not queued: the same recipe again from these rows, or a larger
   `max_steps` (about this run's 2.2 h of training and its reads; `[start]` in `s1.py`
   takes the kana alone and would have to take the marks' rows too). The blind sheets and
   the user's own renders on the pack have no read recorded.
2. When the line is taken up again: `probes/break_place.py` on its rows (the config of
   `bplace_1126_2131` under a new name; `bplace_1126_1840` kept 17 pages of 480 on
   kmix's), and `configs/smoke_gate.toml` under `--smoke` for `s1.py`'s end save, which no
   run has gone through yet.
   If the page's own wrap gives the break row a direction, a micro arm on that loss with the
   glyph rows held is proposed, and how the break's caption is encoded (the text encoder kept
   on the GPU, or no line feed on the Qwen side) is settled there.
3. Ship (the user, 11-26): the rows trained from kmix, if they read well, go out as a v2
   preview with the kana alone; kmix itself is not the release. The kanji come after, as a
   matter of scale, and do not hold the preview.

## Open (the user's calls)

The sentence stage:

- A cap on a row's steps beside the guards (frequent rows such as い, っ, ん, ー take many
  more steps than rare ones).
- The mix of lengths in the draw, and when strings over 12 glyphs come in.
- `♡` at ship: its row is left as jp_v1 has it; a fold of `♡` to `♥` in the pack would send
  both to the trained row.
- Captions of several short strings in one clause (the user, 11-26: "짧은 단어 여러개
  들어가는 케이스도 있으면 어떨까 해"). On the read scenes the model draws one bubble for
  two strings, and the pool's two-string templates draw two boxes whether one string or two
  are asked ([multi_read](reports/multi_read_2026_11_26.md)). Before it can go into the draw:
  scenes that draw a bubble a string (new text-free scenes screened by EN pairs, as
  `pscreen_1125_2306` screened the plain frame); the loss on such a page (each string on a box
  of its own by `reward.match`, and what a string with no box takes); its share of the groups.
- Vertical lettering (the user, 11-26): strings of three glyphs or more that are not SFX-like
  should come out in columns, with the prompt as it is; a label gives the way it stood
  (`orient`). Levers named and not tried: a render weighted by whether its box stands that
  way, `back_steps = 4`, scenes with tall bubbles. The tall pages' lower count is the model's
  lettering, not the reader's (the user, 11-26), so no reader-side fix comes first.
- A layout bank for the loop ([idea.md](idea.md); the user, 11-27: "아이디어 자체는
  좋은거같아"): the gradient taken only on (scene, seed) pairs with room for the string. A
  pair holds the bubble's place across strings and reads long strings at 0.12 – 0.55 of
  their positions; the bubble is set by σ 0.75
  ([layout_hold](reports/layout_hold_2026_11_27.md)). Its steps (the screen, then a micro
  arm against unscreened seeds) are in the idea file; none is queued, and they come after
  the reader's retrain.
- Half the backward by reading the loss off the σ 0.75 step's own x̂₀
  ([sentw](reports/sentw_2026_11_26.md), last part). Not looked at.

Before the v2 preview ships (from the archived glyph proposal's G3, not run): the rows
against jp_v1's in the ruler's frame on `m15v11` (`turbo_read.py`'s ruler part), on the base
and merged15 at 28 steps (rows that help the stack and hurt the base ship as a pack for that
stack), and the ruler's short bin for composition. The user's read of the sheets decides
each.

The kanji (after the kana sentences):

- 1 410 frequent kanji, 6 steps each from a start shifted by the mean displacement of kmix's
  rows (`[start_shift]`, `save = "mix"`): about 29 h of training, and a read of all of them
  22 560 renders an arm, about 3.3 h. At 4 steps (most of the kana's gain), or on the cheaper
  step under Parked, is the user's call. `skip_lone_read` takes about 0.17 s off a 3.04 s
  group; with `reader_compile` a group came to 2.90 s (`timeb_1126_1238`).

The glyph stage's leftovers:

- The shapes the two scripts share (へ べ ぺ / ヘ ベ ペ) folded as the small kana are, in the
  read and the loss. The read can be recounted from `records.json` with no render.
- The detector's misses on large one- and two-stroke glyphs (く ノ ヘ ロ ン), and how such
  pages are counted.
- The plain frame (the user, 11-25: no lettering without `speech bubble` on ComfyUI renders
  of the g2a pack): leave the kana on the bubble frame's recipe, or a plain-frame arm of its
  own scenes and share of renders. g2a's rows carry to plain captions at about EN's level;
  `mix_1125_2338` read under g2a's rows in both frames.

## Parked

- A step that shrinks with the mean gradient's norm (the user left it out, 11-25); the step
  tapers by the row's training-side count instead.
- A term for text outside the picked box, for rows that double their glyph. It is a page term,
  which the rules leave out; offered on 11-26, not taken up.
- The no-box term off for a row once its glyph reads; offered on 11-26, not taken up.
- The shared part kept as its own vector that every row's renders train: the cheaper step the
  kanji may need.
- G1's permuted-label control: not run.
- The x_3 cache: set for the runs after the micro arm and dropped the same day (the user,
  11-25). Its cost is not measured.
- The GRPO stage's other arms (A: rows plus an llm_adapter LoRA; R-credit; the SFX track):
  [archive/proposal.md](archive/proposal.md). Its arm D is what the line runs now.

## Notes kept from planning

- The return replayed on logged scores (the rows then move as logged): drop 4 with no floor,
  with `over` 0.5, and drop 6 with `over` 0.5 give 21, 18 and 11 returns on
  `g1long_1125_1335`'s 384 steps, 5, 3 and 3 on `g1shared_1125_1538`'s 128 (あ at step 8
  under each).
- Two things were filed as "past the glyph". A row that reads is moved a full step by the few
  renders that fail (ぬ in `g1long_1125_1335`: 14 – 16 of 16 at steps 12 – 16, 7 – 11 at
  steps 21 – 24). A row that doubles its glyph (タ, ミ, こ in `g2a_1125_1642`) never read over
  9, 6 and 4 of 16 on the training side.
- `reader_attn = "sdpa"` was added to `g2a_1125_1642`'s config on 11-26: it ran before
  FlashAttention 2 became the default, and `start_from` compares the key.
- g2a's 6-step rows as a pack (the user, 11-25): `anima_cjk_vocab_pack_jp_v1_g2a`
  (`models/vocab_packs/`, local; linked into ComfyUI's `models/vocab_packs` for
  `AnimaVocabPackLoader`), baked from the run's `rows.pt` onto `anima_cjk_vocab_pack_punct`
  with `scripts/toolkits/bake_vocab_pack.py --glyph_route`. It is `jp_v1` with the 164 kana
  rows replaced (no other row and no routing map differs).
- `sents_1126_1856`'s end rows as a pack (the user, 11-26):
  `anima_cjk_vocab_pack_jp_v1_sents`, baked and linked the same way from the run's
  `rows.pt`. It is `jp_v1` with 170 rows replaced: the 164 kana and `ー 。 、 ~ … ♥`.

## Done

Items 1 – 6 and their reports and configs are in `archive/`. Solo is of 256 held-out pages
(16 glyphs, 4 read scenes, 4 seeds), the start rows first;
items 4 and 6 are of 2 624 (164 glyphs). The GRPO stage's runs (11-13) are in the README's
report table.

| item | what | run | report | number | verdict |
|---|---|---|---|---|---|
| – | G0a – c: the tagless frame, the terms' spread, the pixel head against the VAE | `probes/glyph_g0.py` | [glyph_g0](archive/reports/glyph_g0_2026_11_25.md) | pixel head against VAE: sign test p ≥ 0.19 on every term | – |
| – | the scene pool and the crossed grid | `probes/glyph_scenes.py` | [glyph_scenes](archive/reports/glyph_scenes_2026_11_25.md) | 25 of 128 candidates selected; solo EN 0.57, hiragana 0.27, kanji 0.05 | – |
| 1 | G-D's path, and G0d | `g0d_1125_1119` | [gd_g0d](archive/reports/gd_g0d_2026_11_25.md) | the last step's ce gradient five decades over Adam's ε (bar: two) | passed |
| 2 | G1's micro arm: 16 rows, 8 steps | `g1micro_1125_1148` | [g1micro](archive/reports/g1micro_2026_11_25.md) | solo 59 → 132 | adopted on the user's read |
| – | hayai v2.7 beside the tuned reader | `hayai_1125_1234`, `g0h_1125_1243` | [reader_pair](archive/reports/reader_pair_2026_11_25.md), [hayai_g0](archive/reports/hayai_g0_2026_11_25.md) | hayai counts the micro arm 60 → 125; the two row gradients' cosine 0.21 | – |
| – | the reader side of a group, timed | `room_1125_1324` | [reader_room](archive/reports/reader_room_2026_11_25.md) | 0.75 s a group | – |
| – | `reader_compile` + `skip_lone_read` in the loop, the read's ce a group and its saves on threads | `timea_1126_1240`, `timeb_1126_1238`, `timec_1126_1247` | [reader_room](archive/reports/reader_room_2026_11_25.md), third part | a training group 3.25 → 2.90 s, a read group 2.20 → 1.99 s | – |
| – | a one-box training page left unread (`skip_lone_read`) | `room_1125_2155`, `lone_1125_2200` (smoke) | [reader_room](archive/reports/reader_room_2026_11_25.md), second part | read 0.317 → 0.148 s a group; same box and image gradient on 256 of 256 pages | – |
| 3 | G1 long: 16 rows, 24 steps, hayai's term | `g1long_1125_1335` | [g1long](archive/reports/g1long_2026_11_25.md) | solo 59 → 136 → 175 → 199 at 8, 16, 24 steps | adopted on the user's read |
| – | the 24-step rows on the base model, sign prompt | `sign_1125_1504` | [g1long](archive/reports/g1long_2026_11_25.md), second part | 4 glyphs, 2 seeds | – |
| 3b | the row line: the rows' walk, their place in the table, forward reads | `line_1125_1517`, `line_1125_1527` | [row_line](archive/reports/row_line_2026_11_25.md) | solo at twice the 8-step displacement 122, at the other rows' mean displacement 112, at a drawn direction 59 | – |
| 3c | G1 from the shared start: 16 rows, 8 steps | `g1shared_1125_1538` | [g1shared](archive/reports/g1shared_2026_11_25.md) | solo 112 → 148 → 156 at 0, 4, 8 steps (hayai 104 → 147 → 167) | adopted on the user's read |
| 4 | G2a: the 81 hiragana and the 83 katakana, 6 steps from the shared start, small kana folded | `g2a_1125_1642` | [g2a](archive/reports/g2a_2026_11_25.md) | solo 743 → 1 322 → 1 686 → 1 785 at the start, the shifted start, 4, 6 steps (hayai 835 → 1 420 → 1 762 → 1 840) | – |
| – | g2a's rows read with no bubble tag | `plain_1125_2249` | [plain](archive/reports/plain_2026_11_25.md) | 16 rows, solo of 256 plain pages 22 → 93; on scene 35 0 → 4 of 64 | – |
| – | EN words on the same scenes, both frames | `enplain_1125_2258`, `enbubble_1125_2258` | [plain](archive/reports/plain_2026_11_25.md) | 16 words, solo of 256: plain 71, bubble 104 | – |
| – | the plain screen: new scenes by their EN count with no bubble tag | `pscreen_1125_2306` | [pscreen](archive/reports/pscreen_2026_11_25.md) | 160 scenes, EN solo 0.205; 24 at 8 of 16 or more | – |
| – | 16 rows in the plain and bubble frames on 12 of the plain screen's scenes, 6 steps | `mix_1125_2338` | [mix](archive/reports/mix_2026_11_26.md) | solo of 512 a frame, this run against g2a's rows: plain 202 against 238, bubble 297 against 329 | – |
| 6 | G2b: the kana from jp_v1's rows, 16 steps, with `taper`, `[back]`, `near_score` | `g2b_1126_0058` | [g2b](archive/reports/g2b_2026_11_26.md) | solo 743 → 1 118 → 1 393 → 1 566 → 1 716 at 4, 8, 12, 16 steps (g2a 1 686 and 1 785 at 4 and 6); rows against g2a's 6-step rows 64 up, 76 down | – |
| – | the crops the two readers split on | `split_1126_0910` | [g2b](archive/reports/g2b_2026_11_26.md), second part | 2 002 crops: hayai only 1 308, tuned only 694 | – |
| 7 | the kana rows: g2b's 16-step rows with ten of g2a's (っ え グ ざ ヂ の な ゥ や at 6 steps, ヒ at 4) | `kmix_1126_1158` (`archive/probes/rows_mix.py`) | – | ten rows where g2b reads 5 or more of 16 under g2a | adopted on the user's read |
| 7 | kana strings on jp_v1's and kmix's rows, forward only | `wread_1126_1207` | [sent_prep](reports/sent_prep_2026_11_26.md) | the whole string alone 23 → 86 of 384 pages, none at 5 – 8 glyphs | – |
| 7 | a render's text swapped at a step | `tswap_1126_1209` | [sent_prep](reports/sent_prep_2026_11_26.md) | B alone at k 0 / 2 / 3: a glyph 80 / 71 / 14, 2 glyphs 63 / 46 / 10 | – |
| 7 | a clause of two or three short strings, on the read scenes and on the pool's two-string templates | `wmulti_1126_1252`, `enmulti_1126_1258`, `tread_1126_1306` | [multi_read](reports/multi_read_2026_11_26.md) | kana pairs with both strings read: 0 of 288 pages on the read scenes, 0 of 96 on the templates | – |
| 7 | the sentence micro arm: 400 groups of kana strings from kmix's rows, the backward over two steps | `sentm_1126_1331` | [sentm](reports/sentm_2026_11_26.md) | val strings' positions hit 0.345 → 0.331 of 288 pages; kana alone 351 → 354 of 656 | – |
| 7 | which way the micro arm's string pages are lettered | `probes/orient_read.py` on `sentm_1126_1331` | [sentm](reports/sentm_2026_11_26.md), second part | start rows: 73 tall, 250 wide of 384 | – |
| 7 | a row's gradient a render: its glyph alone, a string, its own positions' tokens | `sgrad_1126_1436` | [sentw](reports/sentw_2026_11_26.md), second part | renders of a row agree at 0.032 / 0.003 / 0.016 (last step) | – |
| 7 | a quote token's cross-attention as its glyph's place | `aloc_1126_1501` | [sentw](reports/sentw_2026_11_26.md), third part | a token's window read holds its glyph on 0.335 of 1 744 tokens, another token's 0.073 | – |
| 7 | the gradient of a glyph read on its attention window | `sgrad_1126_1511` | [sentw](reports/sentw_2026_11_26.md), fourth part | 0.016 through two steps; every glyph's window summed −0.001 | – |
| 7 | the sentence micro arm on the window loss, a backward a row | `sentw_1126_1528` | [sentw](reports/sentw_2026_11_26.md), fifth part | val strings' positions hit 0.345 → 0.422 of 288 pages (sentm 0.331); kana alone 351 → 345 of 656 | passed |
| 7 | a line break in the quote: none, a line feed, the literal backslash-n | `qbreak_1126_1751` | [sent_scale](reports/sent_scale_2026_11_26.md), second part | the read breaks after the asked glyph on 0.024 / 0.024 / 0.063 of the pages with a box | – |
| 7 | the reader's line break as a break row's gradient, through four steps | `bgrad_1126_1806` | [sent_scale](reports/sent_scale_2026_11_26.md), fifth part | renders of different strings agree at −0.003 ± 0.003 (the last two steps), 0.004 ± 0.006 (all four) | – |
| 7 | the same from a space's row: the pack's U+3000, T5's `▁` | `bgradw_1126_1815`, `bgradt_1126_1815` | [sent_scale](reports/sent_scale_2026_11_26.md), last part | all four steps: 0.000 ± 0.003, −0.002 ± 0.003; the read breaks after the asked glyph on 0 and 2 of 96 | – |
| 7 | a page's own wrap as the label of a break's place: the denoising error by place, and its gradient on a break row | `bplace_1126_1840` | [sent_scale](reports/sent_scale_2026_11_26.md), last part | 17 pages kept of 480; the error at the true place under another place's on 0.560 / 0.577 / 0.578 of the pairs (a row, a line feed, the literal spelling; null 0.513 / 0.496 / 0.481); the row's gradient across strings −0.012 ± 0.014 | – |
| 7 | the sentence run at scale: 8 000 strings at a seed, 64 renders a row, the window loss | `sents_1126_1856` | [sents](reports/sents_2026_11_26.md) | val strings' positions hit 0.345 → 0.385 → 0.391 of 288 pages at 4 000 groups and the end (sentw 0.422); kana alone 351 → 361 of 656 | – |
| – | what fixes a page's bubble: the (scene, seed), the string, the steps before σ 0.75 | `lhold_1127_1851` | [layout_hold](reports/layout_hold_2026_11_27.md) | own pages' picked boxes within 0.1 of the side: 0.80 – 0.83 at one (scene, seed), 0.18 – 0.20 at another seed; a k = 2 swap page's bubble IoU 0.97 to A's, 0.64 – 0.68 to B's | – |
| 7 | the break-place probe on the run at scale's rows | `bplace_1126_2131` | [sents](reports/sents_2026_11_26.md), last part | 5 pages kept of 144 when stopped | stopped at 36 of 120 strings |

- The verdicts on items 2, 3 and 3c are the user's, in each report's Facts. Item 3c's is the
  call for item 4: the shared start at 6 steps.
- `sentw_1126_1528`'s verdict is the user's (11-26), given on the counts with one of the 36
  blind sheets read (4 pages to 4); in the report's Facts.
- kmix is a data build, no arm: the ten rows where g2b's 16-step solo is under g2a's 6-step
  solo by 5 or more of 16 under both readers, taken from g2a on the user's read of their ten
  blind sheets (11-26: "다 그대로 가져와도 되겠다", "히는 4스텝"); the config header holds
  the counts.
- hayai's term at weight 0.2 is the user's call (11-25), made on the two hayai reports.
- Item 4 ran on the user's calls (11-25): 6 steps a row from the shared start, the two
  scripts in one run, and the small kana folded onto their full-size glyphs (`fold_small`)
  after a first submit (`g2a_1125_1630`) was stopped 6 min in. Its 28 rows in a group (17 not
  learned, 9 past their glyph, 12 a reader's problem) are in the report.
- Item 6's full sheets (g2a's 6-step rows against g2b's 16-step rows, 164 files, sent 11-26):
  no read recorded beyond kmix's ten.
