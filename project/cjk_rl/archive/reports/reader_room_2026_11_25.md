# Reader room — the seconds of G1's reader side by part, three changes timed, a one-box page left unread (2026-11-25), and the switches timed in G1's loop (2026-11-26)

Covers two runs of `../probes/reader_room.py`, one smoke of G1's loop and three timing runs
of the loop, a section each.

## The parts and three changes (`room_1125_1324`)

### Facts

`room_1125_1324` (`../configs/room_1125_1324.toml`, daemon job `20261125-132734-99a91e`;
`../results/20261125-1331-reader_room-room_1125_1324/result.json`, local).

Readers only (the detector, the tuned reader, hayai v2.7; no DiT), on 128 pages
`g1micro_1125_1148` wrote for its read: 16 glyphs × the start and the trained rows, one scene a
glyph, seeds 0 – 3, a group being the 4 seeds. Loss as item 3 will run it (`hayai_weight` 0.2).
Cases under the serial order: box 78, low 28, none 22. Peak allocated 8.1 GiB.

Seconds a group (mean over 32 groups), G1's order (one page a call):

| part | s a group |
|---|---|
| read | 0.346 |
| – tensor → BGR | 0.013 |
| – detector: prepare / forward / boxes | 0.014 / 0.080 / 0.001 |
| – reader: processor | 0.006 |
| – reader: `generate` (its forwards: 0.211, 3.6 a page, 15 ms each) | 0.234 |
| loss | 0.407 |
| – tuned reader's ce (processor 0.005, forward 0.104) | 0.112 |
| – hayai's ce | 0.068 |
| – detector's score, no-box pages | 0.018 |
| – backward to the image | 0.207 |

- `g1micro_1125_1148`'s loop read 0.41 s a group for the read and 0.27 for the loss without
  hayai's term, with the DiT resident and one step's graph alive.
- The GPU's kernel seconds against the wall seconds, first 8 groups (`torch.profiler` on a
  second pass): read 0.86 of 2.56 (33 %), loss 1.32 of 3.33 (40 %).
- A page's read: median 0.072 s, 90th percentile 0.154, max 0.314. A `generate` call holds 1
  crop at the median (max 2) and runs 2 decode steps at the median, 4.3 at the mean, 10 at the
  90th percentile, 25 at most; 3 of 106 calls run 16 or more.

The changes, against the serial order on the same pages:

| variant | read | loss | sum | case / box / text equal (of 128) | image-gradient cosine, median (mean) |
|---|---|---|---|---|---|
| serial | 0.346 | 0.407 | 0.753 | – | – |
| `max_new_tokens` 16 | 0.341 | 0.404 | 0.745 | 127 / 128 / 127 | 1.000 (1.000) |
| 4 pages a call | 0.215 | 0.295 | 0.510 | 127 / 127 / 123 | 0.989 (0.969) |
| 4 pages a call, cap 16 | 0.205 | 0.291 | 0.496 | 126 / 127 / 122 | 0.989 (0.969) |

- The 4-pages-a-call order needs the group's 4 images before any graph is held: a fourth
  no-graph forward, 0.40 s a group (`gd_g0d_2026_11_25.md`: 1.62 s for the 4 steps).
- The user's call (11-25): item 3 runs without the x_3 cache.

### Interpretation

- The reader side is not held up by image preparation: the tensor copy, the detector's
  letterbox and the two processor calls sum to 0.04 s a group. Two thirds of its wall time is
  the GPU waiting between the small kernels of a small crop's forward (15 ms a decode forward,
  one crop), which is Python dispatch on one thread. More cores or a worker pool do not reach it.
- Nothing inside G1's order comes off. The decode cap saves 0.008 s a group, since a call runs
  2 steps at the median and three calls in 106 reach 16.
- 4 pages a call saves 0.24 s a group on the reader side and costs the 0.40 s forward, a net
  loss of 0.16 s, and it moves the image gradient (cosine 0.99 at the median, 0.97 at the mean)
  and 5 picks' text. G1's order stays.
- What is left is the reader's forward itself. `generate` exists in training only to pick the
  crop and to log `exact` and box / low; on a page with one box (88 of the 106 pages with a
  box) the pick needs no read, and skipping it there would take about 0.19 s off a 3.1 s group
  at the price of redefining those log fields. Compiling the reader would cut dispatch on
  every forward, with shapes that change by crop. Neither is worth building for item 3 (about
  5 of its 80 minutes); item 4 can take the first if its hours matter.

## A one-box page left unread (`room_1125_2155`, `lone_1125_2200`)

The user's call (11-25: "생략 구현 해보자") on the last point above. Built as
`gd.Reader.read(skip_lone=True)` and G1's config key `skip_lone_read` (absent: off): on a
training page where the detector finds one box, `generate` does not run, the box is the pick
(case `lone` in the log), and `exact` on that page is the loss forward's argmax being the
target at every token and the EOS (`gd.Reader.ce`'s `hit`). The read on held-out scenes is
unchanged.

### Facts

`room_1125_2155` (`../configs/room_1125_2155.toml`, daemon job `20261125-215135-a961b5`;
`../results/20261125-2153-reader_room-room_1125_2155/result.json`, local): readers only, the
serial order against the same order with the skip, on 256 pages `g2a_1125_1642` wrote for its
read (16 hiragana and 16 katakana, arms `s00` and `s06`, one scene a glyph, seeds 0 – 3).
Cases under the serial order: box 203, low 30, none 23.

| order | read, s a group | loss, s a group | a page's read, median (90th percentile) |
|---|---|---|---|
| serial | 0.317 | 0.438 | 0.072 (0.096) |
| one-box pages unread | 0.148 | 0.439 | 0.027 (0.082) |

- 201 of the 256 pages were left unread (of the 233 with a box); serial's case on them: box
  173, low 28. `generate` calls: 233 → 32.
- The picked box is serial's on 256 of 256 pages, and the image gradient's cosine with
  serial's is 1.000 on every page.
- On the 201 unread pages, the forward's argmax being the glyph against serial's read being
  the glyph: both 139, the read only 4, the argmax only 3, neither 55.

`lone_1125_2200` run with `--smoke` (`../configs/lone_1125_2200.toml`, daemon job
`20261125-215135-0222b4`; `../results/20261125-2154-g1-smoke_lone_1125_2200/`, local): G1's
loop with the switch on, 2 small kana, 6 groups. The log's cases over its 24 renders: lone 21,
box 1, none 2. Seconds a group: read 0.130 (`g2a_1125_1642`: 0.368), loss 0.469 (0.451).
The saves and the four reads ran as before.

### Interpretation

- The skip leaves the loss as it was: the same box and the same gradient on every page. It
  takes 0.17 s off a group, about 5.6 % of `g2a_1125_1642`'s 3.04 s, which on the 1 410
  kanji at 6 steps (about 29 h) is about 1.6 h.
- The argmax stands in for the read on 194 of 201 pages. The 7 that differ are small beside a
  step's 16 renders, but the two counts are not the same quantity: the argmax runs on the
  tensor path's pixels and ignores the reader's guard, and with `fold_small` it counts the
  loss's target only, not the pair's other glyph (no small kana was among this run's 32
  glyphs, so that part is not measured here).
- A log from a run with the switch on has no box / low split on one-box pages. The smoke's
  6 groups are too few, and include the warm-up, to read its seconds as more than a check
  that the loop's `read` falls as the probe's did.

## The switches in G1's loop, and the read's ce and saves (`timea_1126_1240`, `timeb_1126_1238`, `timec_1126_1247`)

The user's call (11-26: "webp저장 스레드 및 ce 한번에 ... 그거랑 스위치 켜기. 한번 시간 재볼까").
`src/g1.py`'s read changed between arms A and B: a group's picked crops go through one ce
forward (a crop a forward before), and the webp pages are written by a pool of 4 threads.

### Facts

Three runs of G1's loop (`../configs/timea_1126_1240.toml`, `timeb_1126_1238.toml`,
`timec_1126_1247.toml`; daemon jobs `20261126-123739-3a8bfc`, `20261126-123837-e99b8d`,
`20261126-124452-db17bb`; `../results/20261126-124*-g1-time*/`, local): 4 glyphs, 8 training
scenes, 1 pass, 32 groups from g2a's shared start, and two reads of 64 pages. A: the tuned
reader on FlashAttention 2, no switch, the read's code as it stood. B: A with
`reader_compile` and `skip_lone_read` on, and the read's new code. C: B on SDPA.

Seconds a training group, the 24 groups after the first 8 (`s_per_group_past_warm`):

| arm | head | last_fwd | read | loss | last_bwd | sum |
|---|---|---|---|---|---|---|
| `g2b_1126_0058` (SDPA, no switch; 10 496 groups, 164 kana) | 1.252 | 0.451 | 0.361 | 0.444 | 0.578 | 3.086 |
| A | 1.251 | 0.450 | 0.493 | 0.484 | 0.575 | 3.253 |
| B | 1.252 | 0.451 | 0.236 | 0.381 | 0.578 | 2.898 |
| C | 1.253 | 0.451 | 0.190 | 0.396 | 0.578 | 2.868 |

- Cases over the 128 training renders: A box 105, low 16, none 7; B lone 102, box 18, low 1,
  none 7; C lone 105, box 13, low 2, none 8. `exact`: 72, 72, 69.
- Peak allocated 14.07 GiB in all three.
- Recompiles in the log (switched on by `vl16_compile.compile_layers`, so A logs none): the
  DiT block's graph twice in B and C, at the read's first forward with no graph and at the
  first training group's forward with one; in C also one of the reader's LM layer, on the
  `S % 8` guard of SDPA. None after the first training groups.

Seconds a read group (4 pages), the median over 16 groups, the start arm / the save:

| arm | render | read | hayai | ce | save | group |
|---|---|---|---|---|---|---|
| A | 1.67 / 1.67 | 0.229 / 0.256 | 0.043 / 0.025 | 0.126 / 0.129 | 0.143 / 0.145 | 2.20 / 2.25 |
| B | 1.67 / 1.68 | 0.228 / 0.190 | 0.044 / 0.024 | 0.070 / 0.071 | 0.000 | 1.99 / 1.96 |
| C | 1.66 / 1.68 | 0.201 / 0.179 | 0.043 / 0.022 | 0.069 / 0.070 | 0.000 | 1.97 / 1.95 |

- The start arm's 64 pages, A against B: the boxes' reads equal on 64 of 64; the recorded
  `ce` of the picked crop differs by 0.019 at the mean and 0.083 at most (52 pages with a
  box). A webp encode of a 768² page: 0.034 s on one thread, 0.009 s a page on 4 (CPU, 16
  pages).
- Spread unknown: 24 training groups and 16 read groups an arm, no twin.
- The user's call on these (11-26): flash_attention_2 stays ("그냥 fa2 쓰는걸로 하자"), and the
  sentence run takes FA2 with `reader_compile`.

### Interpretation

- The two switches take about 0.36 s off A's group (11 %) and about 0.19 s off g2b's (6 %);
  the DiT's three phases do not move. The read's change takes 0.2 – 0.3 s off a read group,
  about 10 minutes of a read of g2b's size (2 624 groups).
- A's read sits 0.13 s over g2b's with no switch on either. The runs do not say whether that
  is FlashAttention 2's `generate` or these 4 glyphs' pages; C against B puts the attention's
  share of the whole group at 0.03 s once the switches are on.
- The recompiles are one graph a grad mode of the DiT's block and cost nothing past the warm
  groups: the DiT phases match g2b's long means.
