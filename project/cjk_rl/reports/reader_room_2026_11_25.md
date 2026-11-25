# Reader room — the seconds of G1's reader side by part, and three changes timed (2026-11-25)

Covers run `room_1125_1324` (`../configs/room_1125_1324.toml`, `../probes/reader_room.py`,
daemon job `20261125-132734-99a91e`). Result:
`../results/20261125-1331-reader_room-room_1125_1324/result.json` (local).

## Facts

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

## Interpretation

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
