# probe_accum: per-row accumulation against the trainer's AdamW (2026-10-06)

From the user (10-06, after `probe_scene_2026_10_06.md`): a draw's gradient
on a row is ~97 % its own — does stepping a row only once it has summed N
draws beat stepping on every draw? To first order the two move a row by the
same signal and noise at a matched step size, so the pair isolates drift:
the noise walk moving the row and the later draws read at the moved row.

`probe_accum.py`: 32 kanji of f0's at mid frequency (80–130 items each in
`sent_kanji`'s data: 例飼曜舞福越希修北徹契号職荷左線館鳴静増街壊鳥状側源治閉悲波城勉),
the 3 131 items holding one, 15 % of the multi-glyph texts held out (415
items, 233 texts). The rows at `seed_fixed_1005_stick080` (f0's start), only
the 32 live, every other row frozen there. Two arms on the same batches, σ
and ε (SEED 0 per arm), 1 500 steps × batch 4, constant lr, compiled:

- **plain** — `torch.optim.AdamW` (the trainer's betas, wd 0), lr 2e-4;
- **accum** — per row the gradient summed until 16 items have held the glyph,
  then one AdamW step on that row alone (its own m / v / t; between events
  it does not move), the remainder in one last step.

Per row 190 draws (median), 12.5 events. Read: the held-out items at start /
plain / accum under the same σ / ε per item (2 draws each, 800 item-draws),
the box-share loss's in-box term; per row over the items holding it. Jobs
`20261006-183753-86706e` (a16, 23.9 min) and `20261006-190314-620e89` (a16m,
accum only, plain reused, 11.8 min) → `output/cjk_anima_reseed/probe_accum/`.

| arm | lr | displacement (median) | cos with f0's move | held-out in-box Δ vs start |
|---|---|---|---|---|
| plain | 2e-4 | 58.1 | 0.35 | −0.00203 ± 0.00036 |
| accum (a16) | 8e-4 (lr √N) | 26.5 | 0.29 | −0.00153 |
| accum (a16m) | 1.76e-3 | 51.3 | 0.31 | −0.00182 |

accum − plain on the in-box term: a16 +0.00049 ± 0.00024, plain lower on 23
of 32 rows (sign p 0.02); a16m +0.00020 ± 0.00016, 21 of 32 (p 0.11). The
total loss reads the same way.

- **Accumulation does not beat stepping on every draw.** At a matched
  distance (a16m, 0.88× plain's) it is a tie leaning plain, and the lean is
  about what the remaining 12 % of distance buys (a16: 0.46× the distance,
  0.76× the loss drop). Nothing here says the noise walk costs the rows
  anything at 190 draws; the lever is closed at this scale.
- **lr × √N did not match the distance: AdamW steps a sparse row larger
  than its lr.** A row is in ~1 batch of 8; its v averages in the absent
  steps' zeros (v ≈ draw rate · E[g²]), and m keeps stepping it while it is
  absent, so each plain draw moves the row ~1 / √(draw rate) more than a
  dense row's — a16 measured 2.2× (√8 ≈ 2.8). Event-only v has no such
  inflation. In the trainer, a row's effective step grows as it gets rarer.
- cos with f0's move favours plain in both reads, but f0's move is itself a
  plain-AdamW walk; it is not a neutral signal direction.

Not read: the render (a held-out loss change is not a glyph-F1 verdict), the
1 348-row run's interference, warmup / cosine, rows rarer than ~160 draws per
1 500 steps (f0's rare kanji see ~6 per 900).

## A look (user, 10-06): four held-out lines, two arms

`probe_accum.py look --label a16m` (job `20261006-191857-d13f08`, 1.4 min):
four held-out `sent_34` lines on their own captions and canvases, plain's
and accum's (a16m) rows, the ruler's sampler on the punct pack, seeds 0 / 1
→ `output/cjk_anima_reseed/probe_accum/a16m/look/` (`sheet.png`,
`look.json`). Glyph F1 over the 8 renders: plain 0.39, accum 0.32; exact 0
for both.

| target | plain s0 / s1 | accum s0 / s1 |
|---|---|---|
| 北 (北村くん彼女いるの？) | 北 / 北 | 水ヒ (北 split) / 北 |
| 街 (街の声を聞いてみた) | 衙-like / 街 | 街-like / 街 |
| 曜 (土曜に服買いに行くよ) | missed / missed | missed / missed |
| 勉 (勉強ダメだから……) | missed / missed | missed / missed |

Page and layout are the same per seed; the arms differ in the glyphs only.
The one visible split is 北 at seed 0, plain's way; 曜 and 勉 fail under
both. Eight renders: consistent with the held-out loss's lean, not a read.
