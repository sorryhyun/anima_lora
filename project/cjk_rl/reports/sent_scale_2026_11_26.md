# Before the run at scale — line breaks, what a hit goes with, the draw (2026-11-26)

Reads asked by the user after `sentw_1126_1528` passed: the saved records of
`aloc_1126_1501` and `sentw_1126_1528` read again (no GPU), a forward-only probe of a
line break in the quote (`qbreak_1126_1751`), the train strings counted for `ー`, `~` and
quote marks, gradient probes of a break row from three starts (`bgrad_1126_1806`,
`bgradw_1126_1815`, `bgradt_1126_1815`), and a probe of a page's own wrap as the label
of a break's place (`bplace_1126_1840`, the last part). A probe's settings,
question and gate are in its config header. The record reads were one-off scripts and are not kept as probes.
The smoke of the settings the user fixed is the part before those.

# A line break and a glyph's window

The user (11-26): "줄바꿈의 경우 window가 어떻게 되나 모르겠네 이를테면 글자 다음이
줄바꿈이거나, 줄바꿈 다음 글자라면?", and on the answer that the quote holds no break:
"캡션에서 줄바꿈이 없다고?? ocr 레이블에는 줄바꿈 있지 않았어? (아니마가 줄바꿈을 인식하는진
모르겠다만...)".

## Facts

- 12 547 of the 28 975 train targets hold a line break (`../archive/sent_rl.md`; [sent_prep](sent_prep_2026_11_26.md)). The draw
  (`probes/sent_strings.py`) takes them out, so a quote has none, and the window loss's
  target is a glyph alone (`src/s1.py` drops `score`'s target with breaks on that branch).
- A window is a square of 1.3 glyph sides about the token's attention peak, a glyph side
  being the root of the box's area over the string's glyphs; it is cut at the page's
  edge, not at the box or a line (`src/loc.py`).
- `aloc_1126_1501`'s records, maps `b13_15/raw`, 275 pages with a box: the page's read
  has two lines or more on 114 (2 – 4 / 5 – 8 / 9 – 12 glyphs: 12 of 88, 41 of 92, 61 of
  95). A break's place is known only where the read hit the glyph before it.

  | position, hit by the page's read | n | its window's read holds its glyph | holds it alone |
  |---|---|---|---|
  | a read of one line | 279 | 0.781 | 0.362 |
  | inside a line, a read of several | 215 | 0.772 | 0.205 |
  | the last of a line | 54 | 0.704 | 0.407 |
  | the first of a later line | 36 | 0.722 | 0.167 |

  | adjacent tokens | n | centres apart, glyph sides, median | under 0.5 |
  |---|---|---|---|
  | one line, both hit | 200 | 0.95 | 0.080 |
  | across a break, both hit | 13 | 1.50 | 0.077 |
  | across a break, the next one missed | 31 | 1.08 | 0.129 |
  | a miss in the pair, all | 1 209 | 0.65 | 0.374 |

- Spread unknown. A share on 36 – 54 positions has a standard error near 0.07.

## Interpretation

- The glyphs beside a break are found about as often as the others; the gap is inside
  the standard error. Two tokens across a break do not share a window. A shared window
  comes with a glyph that is not drawn.
- On a read of several lines the window more often holds a second glyph, which reads as
  the square taking in the next line. What that does to the gradient is not measured.

# `qbreak_1126_1751` — a line break in the quote

## Facts

- The tokenizer, on CPU: a line feed in the quote (what `process_escape` makes of a
  prompt's `\n`) leaves the T5 side's ids as they are and adds one Qwen token (198). The
  literal backslash and `n` add a Qwen token (1699) and, on the T5 side, an ext row and
  T5's own `▁` `n`.
- Dataset captions spell a break the literal way
  (`anime_tools.captions.position_clauses.LINE_BREAK`): 574 of the 2 097 captions with a
  text clause under `post_image_dataset/resized`. No training path calls
  `process_escape` (`rg` over `library`, `scripts`, `train.py`, `networks`); every
  inference path does.
- `probes/quote_break.py`, forward only, 3.1 min: the 24 val strings of `aloc`'s 5 – 8
  and 9 – 12 bins on scenes 6 and 45 at seeds 0 and 1, kmix's rows, the break after glyph
  n // 2. `results/20261126-1755-quote_break-qbreak_1126_1751/`, pages
  `output/cjk_rl/quote_break/qbreak_1126_1751/` (local).

  | quote | pages | with a box | the read breaks after the asked glyph | after it or a neighbour | the read has a break | positions hit | reads with a backslash or a Latin n |
  |---|---|---|---|---|---|---|---|
  | no break | 96 | 85 | 0.024 | 0.106 | 0.518 | 0.310 | 0 |
  | a line feed | 96 | 83 | 0.024 | 0.108 | 0.506 | 0.342 | 1 |
  | literal backslash-n | 96 | 79 | 0.063 | 0.152 | 0.494 | 0.324 | 4 |

- An arm's first column minus the no-break arm's, ± the standard error over the 24
  strings: a line feed 0.000 ± 0.021, the literal spelling 0.031 ± 0.027.
- The no-break arm's reads are `aloc_1126_1501`'s on 96 of 96 pages.
- The pages were not viewed.

## Interpretation

- The header's second branch: the 4-step student on kmix's rows does not wrap where the
  quote breaks, in either spelling. The quote stays without breaks.
- A training caption and an inference prompt spell a break differently on both
  tokenizers. Whether the base model at 28 steps follows either spelling is not read
  here; it is outside this line.

# What a hit goes with

The user (11-26): "내가 알기로 2 gram 기준으로 성공하는게 중간중간 생겨서 그럼 학습하면서 점점
해당 2gram은 학습대상에서 빼는식으로 하면 되지 않나? 정확히 2gram식인진 모르겠다만".

## Facts

- `sentw_1126_1528`'s read at 400 groups, the val and seen strings (384 pages, a string
  on 4 scenes at 2 seeds, 307 positions): a position is hit on 0.561 of the pages where
  the one before it is hit and on 0.313 where it is missed. Adjacent hit pairs inside a
  string over the count its hits placed at random would give (strings of 5 glyphs or
  more): 0.97 at the start rows, 1.03 at 200 and at 400 groups.
- A position's hits over its string's 8 pages, against a binomial at the mean (0.438):

  | hits of 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|---|---|
  | positions | 35 | 42 | 44 | 39 | 40 | 35 | 28 | 27 | 17 |
  | binomial | 3 | 19 | 52 | 81 | 79 | 49 | 19 | 4 | 0 |

- One glyph over its positions, hits of 8: `っ` 0 – 8 (13 positions), `そ` 0 – 7 (11),
  `あ` 0 – 8 (10), `ー` 0 – 7 (15, median 2).
- Two positions of one glyph, the difference of their hit shares: 0.25 where the glyph
  before is the same (82 pairs), 0.29 where it differs (935); first glyphs left out,
  0.255 (50) and 0.274 (746).
- By place in the string: the first glyph 0.58, the second 0.54, the last 0.66, the ones
  between 0.31 (166 positions).
- `aloc_1126_1501`'s 1 744 tokens under two gates: the window already reads its glyph
  0.335; of the rest, the centre is within half a glyph side of a neighbour's 0.344; left
  0.321.
- The train strings hold 9 071 distinct pairs of adjacent glyphs (the string's start
  counted as one); in a draw of 5 588 strings, 898 pairs come 8 times or more and are
  0.70 of the pairs drawn.
- The user (11-26), on a position the page's read already hits: "이미 읽힌 위치의 bwd 빼는건
  찬성"; on the seeds: "본 학습에서는 seed 1로 하는게 좋을듯".

## Interpretation

- A hit does not go by pairs. The dependence on the glyph before is between pages (a
  page reads most of its string or little of it); inside a string the hits are placed as
  chance would.
- What holds across seeds and scenes is a position of a string. A glyph does not (the
  same row reads on every page of one string and on none of another), and the glyph
  before explains little of it, so a tally a pair would leave out positions it has no
  word on.
- The gate on the render's own read is the exact form of the user's idea and costs
  nothing, since the read comes before the backward. `sgrad_1126_1511` has the reason:
  renders agree at 0.008 on positions the read hit and 0.034 on those it missed.

# The draw: `ー`, `~` and quote marks

The user (11-26): "데이터셋에서 ー 나 ~ 가 좀 많이 들어가는데, 적당히 샘플링 시 필터링하면
좋을듯? \" 나 ' normalize도 한번만 체크해주고".

## Facts

- The 24 891 train strings, 142 043 glyphs (keys of `s1.key`, so `―` counts as `ー`):

  | glyph | strings holding it | glyphs | once / twice / three times / four or more a string |
  |---|---|---|---|
  | `ー` | 4 849 (0.195) | 6 610 (0.047) | 3 597 / 893 / 252 / 107 |
  | `~` | 1 147 (0.046) | 2 012 (0.014) | 630 / 303 / 134 / 80 |
  | `…` | 6 391 (0.257) | 7 098 (0.050) | 5 727 / 622 / 41 / 1 |
  | `っ` | 5 457 (0.219) | 6 234 (0.044) | 4 802 / 558 / 79 / 18 |

- The longest run of `ー` / `~` in a string: none 18 937, one 4 806, two 707, three 305,
  four or more 136. Strings where they are over half of the glyphs or run three or
  longer: 478 (2 – 4 / 5 – 8 / 9 – 12 glyphs: 55 / 283 / 140), such as `お~~~~っ`,
  `へえーーーっ`, `うーー`, `ドールガぁ―――ン!!!`. Left out, `ー` goes 6 610 → 5 716 glyphs
  and `~` 2 012 → 1 260; strings holding `ー` 4 849 → 4 580, `~` 1 147 → 931.
- `src/s1.py` takes the rule as `[draw]` (`glyphs`, `run_max`, `share_max`); the read's
  strings are not thinned.
- Quote marks in the 115 529 train labels (after the reader's fold, `textnorm`, which
  sends `“` `”` `＂` to `"` and `＇` to `'`): `"` in 290, `'` 3, `’` 3, `‘` 1, `〝` 15,
  `〟` 14, `′` 2. `"` an odd number of times in 16, among them a voiced mark written as a
  quote (`あ"~~…`, `え"`, `……う"っ`).
- The draw allows no quote mark, so the pool holds none (0 of 24 891), and `s1.key`
  folds only what NFKC does (`＂`, `＇`).
- The clause's own quotes on the T5 side: `"あ"` gives the quote token 96 on both sides
  of the row; `“あ”`, `‘あ’`, `'あ'`, `＂あ＂` and `〝あ〟` each give other tokens.

## Interpretation

- `ー` is no more frequent than `…` or `っ` a glyph. What sets it and `~` apart is the
  run, which reads as one stroke whose count a read cannot give (not measured). `ー` is
  the row `sentw` moved furthest (0.32 row norms) and the one read least of the four
  most frequent. The rule leaves out 1.9 % of the strings and keeps a kana with one `ー`
  (`へー`). The thresholds are Claude's pick on the user's "적당히".
- For this line the quote marks need nothing: no string holds one. The reader's labels
  leave `〝` `〟` and the single curly quotes unfolded on about 35 labels, and a prompt
  that wraps the clause in curly quotes does not reach the tokens the clause was trained
  with. Both belong to other lines (`project/ocr_reader`, the prompt path).

# The smoke of `skip_hit` and `train_seeds`

## Facts

- `src/s1.py` with `configs/smoke_gate.toml` under `--smoke` (`sentw_1126_1528`'s
  settings with `skip_hit = true` and `train_seeds = 1`; 24 groups of one render, `accum`
  4), the job 2.1 min: 93 row-renders backed of the 131 a render held a row in
  (163 glyphs asked, 0.368 of the positions hit), 3 row steps, peak 13.30 GiB, 3.3 s a
  group with the compiles inside it. `results/20261126-1757-s1-smoke_smoke_gate/` (local).
- A render that backs nothing of a row is in the row's score and not in its `accum`; a
  row steps on `accum` renders that backed it (`src/s1.py`'s docstring).
- The smoke ran before `[draw]` was added; `thinned` was checked on the pool on CPU
  (24 891 → 24 413 strings).

## Interpretation

- The loop runs with both settings. The seconds a group hold the recompiles of a batch
  of one and set no cost for the run at scale.

# `bgrad_1126_1806` — the reader's line break as a break row's gradient

The user (11-26), on `qbreak_1126_1751`: "흠... 줄바꿈을 학습시킬수는 없나? 내생각에 줄바꿈만큼
확실한 glyph signal row는 없는데".

## Facts

- The trainable table (3 174 ext ids) holds no row for a line feed or for `↵`; the pack
  has a row for `↵` (ext 63360, norm 221.2, the pack's mean 206.3) and a `fold` table
  (one character to one, on the T5 side only) with 10 entries.
- `probes/break_grad.py`, no training, 4.7 min: 24 train strings of 5 – 12 glyphs past
  index 400, each on a training scene at 4 seeds, the break after glyph n // 2. The Qwen
  stream holds a line feed; the T5 ids hold the table's `〆` row there, set to the pack's
  `↵` row. Two losses on the picked box with the target broken at the asked place, each
  backed through the four steps. `across`: the mean cosine of the break row's gradient
  between renders of different strings, ± a jackknife over strings.
  `results/20261126-1812-break_grad-bgrad_1126_1806/` (local).

  | loss | step σ 0.5 → 0 | 0.75 → 0.5 | 0.9 → 0.75 | 1 → 0.9 | the last two | all four |
  |---|---|---|---|---|---|---|
  | the break token's cross-entropy, across | 0.001 ± 0.004 | −0.004 ± 0.005 | 0.011 ± 0.009 | −0.002 ± 0.004 | −0.003 ± 0.003 | 0.004 ± 0.006 |
  | the same, norm, median | 0.083 | 0.117 | 0.269 | 0.280 | 0.165 | 0.668 |
  | the target's mean cross-entropy, across | −0.002 ± 0.003 | −0.004 ± 0.003 | 0.000 ± 0.004 | −0.001 ± 0.005 | −0.004 ± 0.002 | 0.001 ± 0.003 |
  | the same, norm, median | 0.033 | 0.047 | 0.134 | 0.149 | 0.077 | 0.328 |

- 95 of the 96 renders backed (79 with a box over `conf_min`); the read breaks after the
  asked glyph on 1, and has two lines or more on 30. The break token's cross-entropy,
  median: 5.35. The two losses' gradients on a render: cosine 0.33 (the last two steps)
  and 0.30 (all four).
- Two seeds of one string agree at 0.026 and 0.059 through the last and the step before
  on the break token's loss (no interval computed).
- `sgrad_1126_1511`, the size a row trained on: 0.032 ± 0.009 on its glyph alone, 0.016
  ± 0.008 on its windows.

## Interpretation

- The header's second branch: no part of the backward gives the break row a direction
  that renders of different strings share, at an untrained symbol row and a break in
  the middle of the string. The gradient is three times larger through the first two
  steps than the last two, which is where a page's lines would be set, and it has no
  more agreement there.
- A step of the break row would average renders of different strings, so the reader's
  gradient is not a way to train it as the probe stands. What is not tried: the reward
  form on that one row (the read breaking after the asked glyph, 1 of 96 at the start,
  so the reward would need a graded form such as the distance of the nearest break),
  breaks at the labels' own places, and a row that starts elsewhere. `src/grpo.py` is on
  the older policy path and would need porting; none of it is queued.

# `bgradw_1126_1815`, `bgradt_1126_1815` — the break row from a space's row

The user (11-26), on `bgrad_1126_1806`: "쩝... 차라리 스페이스나 전각스페이스같이 비슷한 문자로
init하는건?".

## Facts

- The tokenizer, on CPU: U+3000 in a quote is one ext row on the T5 side (ext 58880,
  norm 198.4), which the table does not hold; an ASCII space, a tab, U+00A0 and U+2003
  between routed glyphs leave no token there.
- `bgrad_1126_1806` with the row's start changed and nothing else, 4.7 min each: the
  pack's row of U+3000 (`bgradw`), and the adapter's own T5 row of the space piece `▁`
  (id 3, norm 132.4; `bgradt`). `results/20261126-18*-break_grad-bgrad{w,t}_1126_1815/`
  (local).

  | the row's start | the read breaks after the asked glyph | two lines or more | the break token's cross-entropy, median | across, the last two steps | across, all four | norm, all four, median |
  |---|---|---|---|---|---|---|
  | the pack's `↵` (`bgrad`) | 1 of 96 | 30 | 5.35 | −0.003 ± 0.003 | 0.004 ± 0.006 | 0.668 |
  | the pack's U+3000 (`bgradw`) | 0 of 96 | 35 | 5.41 | 0.002 ± 0.003 | 0.000 ± 0.003 | 1.389 |
  | T5's `▁` (`bgradt`) | 2 of 96 | 28 | 4.58 | −0.002 ± 0.002 | −0.002 ± 0.003 | 0.807 |

- No single step's `across` is 2 standard errors over 0 on either start (the largest:
  U+3000 at σ 0.75 → 0.5, 0.004 ± 0.005; `▁` at σ 1 → 0.9, 0.003 ± 0.003).

## Interpretation

- The header's third branch on both: a space's row does not make the page wrap at the
  asked place, and the reader's gradient gives it no direction that renders share.
  Three starts agree, so the 0 is not the start's.
- How line breaks are handled is a placeholder in `../archive/sent_rl.md` (the user, 11-26).

# `bplace_1126_1840` — a page's own wrap as the label of a break's place

The user (11-26), after the three gradient probes: "내생각에 sent rl에서 줄바꿈은 (나름)
중대한 이슈인거같은데... 어떻게 학습시킬 방법이 없을까? 뭔가 이거만한 위치 정보를 주면서
유저들도 잘 활용할 수 있는 토큰이 없는데".

## Facts

- `sentw_1126_1528`'s read records, counted on CPU (the pages of every arm whose best
  box hits half the string or more, strings of 3 glyphs or more): 244 of 565 read as
  two lines or more; a line ends after a hit kana on 102 of 1 249, after `…` on 16 of
  120, `っ` 8 of 72, `ー` 6 of 35, `!` 1 of 44.
- `probes/break_place.py`, no training, the job 10.1 min: 120 train strings of 5 – 12
  glyphs past index 400, each on a training scene at 4 seeds with no break in the
  quote, kmix's rows. `results/20261126-1850-break_place-bplace_1126_1840/`, pages
  `output/cjk_rl/break_place/bplace_1126_1840/` (local).
- Of the 480 renders, 403 have a box and 152 read as two lines or more. 17 are kept (16
  strings): every line end after a hit glyph, half the string or more hit. Of the
  others with two lines, a break is not placed on 116 and under half of the string is
  hit on 105 (both on 87). The kept pages' hit shares: 13 between 0.50 and 0.62, 4 at
  0.83 or over. Their boxes: 4 tall, 13 wide. The read's glyph count is the string's on
  20 of the 126 boxed pages with two lines.
- The forward score: `auc` is the share of (true place, other place) pairs where the
  break at the true place has the lower error, `null` the same with another string's
  places standing in, ± a jackknife over strings; `gap` is the other places' mean error
  minus the true places', over the error with no break.

  | spelling | step (σ) | auc | null | auc − null | gap |
  |---|---|---|---|---|---|
  | a row at the pack's `↵`, a line feed on the Qwen side | 0 (1.00) | 0.558 | 0.502 | 0.056 ± 0.090 | 0.003 |
  | | 1 (0.90) | 0.564 | 0.543 | 0.021 ± 0.094 | 0.008 |
  | | 2 (0.75) | 0.559 | 0.495 | 0.064 ± 0.094 | 0.005 |
  | | the three, mean | 0.560 | 0.513 | 0.047 ± 0.056 | 0.005 |
  | the line feed alone | 0 (1.00) | 0.552 | 0.479 | 0.073 ± 0.084 | 0.003 |
  | | 1 (0.90) | 0.647 | 0.478 | 0.168 ± 0.082 | 0.004 |
  | | 2 (0.75) | 0.532 | 0.530 | 0.002 ± 0.100 | 0.002 |
  | | the three, mean | 0.577 | 0.496 | 0.081 ± 0.056 | 0.003 |
  | the literal backslash-n | 0 (1.00) | 0.498 | 0.462 | 0.036 ± 0.075 | 0.002 |
  | | 1 (0.90) | 0.568 | 0.497 | 0.071 ± 0.086 | 0.005 |
  | | 2 (0.75) | 0.668 | 0.483 | 0.185 ± 0.073 | 0.024 |
  | | the three, mean | 0.578 | 0.481 | 0.097 ± 0.051 | 0.010 |

- The break row's gradient, the mean cosine between pages of different strings, 17
  pages:

  | vector | step 0 | step 1 | step 2 | the three summed |
  |---|---|---|---|---|
  | true place − another place | 0.014 ± 0.024 | 0.001 ± 0.018 | 0.035 ± 0.040 | −0.012 ± 0.014 |
  | another − another | −0.013 ± 0.011 | 0.039 ± 0.049 | −0.004 ± 0.028 | 0.009 ± 0.029 |
  | the true place alone | 0.140 ± 0.060 | −0.010 ± 0.019 | 0.019 ± 0.046 | 0.034 ± 0.032 |

- Spread unknown. The pages were not viewed.

## Interpretation

- None of the header's three branches is reached at 17 pages. The gradient's standard
  errors (0.014 – 0.040) are the size of what is looked for (`sgrad`'s 0.016 and 0.032),
  so its 0 says nothing either way.
- The score is over its null in all nine cells and by 2 standard errors in two of them
  (the line feed at σ 0.9, the literal spelling at σ 0.75), the largest gap on the
  spelling the dataset captions use. That leans toward the DiT already reading a break's
  place a little in that spelling. Nine cells on the same 17 pages do not make it a
  result.
- What limits the probe is the label, not the loss: on kmix's rows a page of 5 – 12
  glyphs seldom reads well enough to place its breaks (17 of 480, 4 of them at 0.83 of
  the string or over), and a read that hits half a garbled string places a break
  loosely. The label gets better as the strings read better, so this loss fits after the
  run at scale or on its rows, or with a label that does not need the glyphs read (the
  lines' glyph counts, the tokens' attention centres).
- A 2 × 2 or 3 × 3 block read the other way gives a wrong place (the user, 11-26:
  "ㅁㅁ\nㅁㅁ 이면 줄바꿈이 애매하려나? 세로쓰기랑 가로쓰기 어느쪽인지도 헷갈릴거같고").
  The rule that every line end follows a hit glyph drops a read whose order is broken;
  a rule on both glyphs beside the break would be tighter.

