# seed synthesis — the seed's own renders as the canvas (2026-10-01, draft; arm read 10-02)

Takes over `proposal_length.md`: its span lever ran (`experiments/span_reband`)
and closed, and the reads it left behind say where the leftover slot is
decided and what a row can do about it. The open question is unchanged — the
repeats are the base's text region minus the word (`こんにちちは`, dup
100 / 184 on `sent`).

## What the three reads settled (10-01)

**1. `span_reband` — the seed's word-length windows at 0.85–0.95.** b0507
`scene_window` items (≈ 34 px, the word filling 0.7–1.0 of its bubble along
the reading axis), filtered to a bubble that holds ≤ 2 columns / lines
(2 589 of 4 060, data check of 10-01), moved from 0.5–0.7 to 0.85–0.95 and
repeated × 4 (41 % of the records; the rest of the seed's mix at its own
bands), warm μ 0.02, 174 kana rows × 23 steps. Job `20261001-200229-76d777`,
`experiments/span_reband/results/20261001-2002-r0/`.

| `sent`, 184 | official | ≤ 1 edit | dup | box | box_h | flat_white | IoU vs EN |
|---|---|---|---|---|---|---|---|
| floor | 29 | 92 | 100 | 0.156 | 0.180 | 0.132 | 0.198 |
| span_reband | 16 (+6 / −19, p 0.015) | 70 (+9 / −31, p 7e-4) | **113** (+29 / −16, p 0.07) | 0.132 | 0.211 | 0.168 | 0.228 |

The banner stays where it is and the glyphs in it get smaller and more
(`ここんにちはは。`, `ここんにちは は涼きです??`, a `。`-ended dialogue line);
some renders take the items' layout outright (はい in a bubble, a lone big
はい). Gains: おしい 1 → 3, たすけて 1 → 2, ちょっと来い 2 → 3; losses are the
large-glyph words (やったネ 5 → 1, パソコン 3 → 0, こんにちは 2 → 0). The same
picture as `delta_scale`'s Δ 0.9 (smaller glyphs in the same banner, dup
100 → 124) and `b0305_reband` (smaller still → columns).

**2. `shared_dir` on it — which side of 0.8 the loss sits.** Seed 0, 92
keys. `full_s` = the arm's rows above 0.8 and the seed's below, `s_full` the
reverse. Job `20261001-205634-54280f`.

| rows (above 0.8 / below) | official | ≤ 1 edit | dup | box |
|---|---|---|---|---|
| floor (seed / seed) | 15 | 45 | 40 | 0.149 |
| `full` (arm / arm) | 9 | 38 | 50 | 0.128 |
| `full_s` (arm / seed) | 9 | 42 | 51 (+18 / −7, p 0.04) | 0.127 |
| `s_full` (seed / arm) | 13 | 46 | 45 | 0.143 |

Everything the arm does, it does above 0.8; below it its rows are the
seed's for every count. A σ gate at inference has nothing to separate here.

**3. `inject_count` — where the slot count is committed on the trajectory.**
No training. 60 floor renders with a leftover slot (main-box read ≤ 2 edits
once doubles collapse); **A** = the banner erased and the word redrawn
filling the same extent with n glyphs (× 6/5, the banner's own ink, a bold
face — the canvases read 59 / 60 official); **B** = the floor render itself.
At σ_s the sampler's x_t is replaced by `(1 − σ_s)·z + σ_s·ε` (same ε for A
and B), the rest of the trajectory as the floor's (seed rows, caption,
seed). Job `20261001-211227-d4453b`,
`experiments/inject_count/results/20261001-2113-ic0/`, sheet
`output/cjk_anima_scale/experiments/inject_count/sheet.png`.

| injected at σ | A official | A dup | B official | B dup | A vs B official (p) |
|---|---|---|---|---|---|
| (floor, this subset) | 0 | 60 | | | |
| 0.95 (0.947) | 13 | 39 | 10 | 43 | 8 / 5, 0.58 |
| 0.9 | **29** | **21** | 5 | 48 | 26 / 2, 3e-6 |
| 0.85 (0.844) | 36 | 18 | 3 | 50 | 33 / 0, 2e-10 |
| 0.8 | 39 | 15 | 2 | 51 | 38 / 1, 2e-10 |

- At 0.95 the injected state is overwritten: A and B come out the same and
  the scene itself re-rolls (both arms are new compositions). The count is
  decided between 0.95 and 0.9.
- At 0.9 the 5-slot banner holds in half the renders, at 0.85 in 60 %, at
  0.8 in 65 % — a gradual commit, not a switch; a third of A's banners get
  their sixth slot back below 0.8 (`おしい` p1 s1: canvas `おしい` → A@0.9
  `おおししい`).
- The ceiling of the lever: whatever puts the trajectory on the n-slot
  banner by σ 0.9 turns 0 / 60 official into 29 / 60 and dup 60 → 21.

**Reading.** Across `b0305_reband`, `span_reband`, `delta_scale` and this
read: **the rows decide what fills the region — glyph size and slot count —
and the base decides the region** (its extent is the `japanese text` prior,
set by 0.95, `findings.md` § 1 / § 3). A row acting at 0.85–0.95 is at the
right σ for the count (H1 + read 3); the two arms taught the wrong count
because their items' glyphs were small for their region. The span lever
(a shorter region through the rows) is closed; the count lever (fewer,
larger slots in the base's region) is open, and `cf_sense`'s teacher-forced
`move` was not needed to open it.

## The objection: an absolute size in the item is a bias in every caption

Under H1 a row carries its items' layout to every caption. The seed's
banner is already that: singles trained at 0.7–0.9 on 50–400 px glyphs
(`b0709`), and the band law puts large px at high σ — **whatever trains at
0.85–0.95 with one glyph size teaches that size.** An item set of "big
glyphs filling a banner" would lock the banner in further, against the page
axis (`product_criteria.md`: text where EN sits; floor IoU 0.198). The
`span_reband` items had the opposite absolute size (34 px) and taught it.

The way out is an item set with **no common size**: the region in each
canvas is whatever the base drew — a banner, a bubble, a subtitle line, the
short line of an untagged caption (§ 3) — and the word fills *that* region
with n slots. Across such items the only shared direction is "n slots fill
the region you are given"; the size cancels. The precedent is the singles:
`grid_single` taught fill-of-cell from 1 × 1 (150–400 px) to 3 × 3
(51–136 px) and renders per cell (§ 7). Whether a σ-less row carries the
same relative rule for a word is the one thing no read covers; it is the
arm's question.

## Proposal: self-generated canvases, the word filling the base's region

Items are the seed's own renders with the string replaced — `garble_replace`'s
construction with the premise met this time: the canvas is rendered by
the rows that will train on it, so the FM residual outside the glyphs is
≈ 0 at every σ, and the region's size distribution is the base's, not a
pool's.

**Canvas.** The current seed rows (routed), the `sent` ruler's settings
(512², 28 steps, cfg 4), caption `{p}, japanese text. Japanese text reads as
"X".` — the ruler's own caption shape, no tag change. `p` from the scene
prompt stream **outside the `sent` grid's 4 prompts**; `X` a word of 2–6
glyphs from the windowed word pool over the seed's trained singles
(`polish_seed`'s line pool), **never a `sent` / `target` string nor a
trigram of one** (the line's hold rule).

**Selection.** Read every render (sfx + VL). Keep a render when its main
text box (the largest CJK box; no other CJK box > 1.5 × its area) reads
within 2 edits of X once doubled glyphs are collapsed — the leftover-slot
renders and the near-misses. Exact reads are not items (their redraw is a
no-op: zero residual everywhere). The 10-01 floor keeps 63 / 184 (34 %).

**Redraw** = `inject_count.redraw`: erase the box (ring-median fill, padded
to the new glyph height), draw X in one line (or one column when the box is
taller than wide) with n glyphs contiguous, `fs = min(box length / (n ·
1.05), 1.5 × box height)`, centred on the box, the box's own ink colour, a
face from the render set minus the light ones. The canvas is read again;
a redraw the readers do not read ≤ 1 edit is dropped. Loss box = the drawn
glyph box (`layout: grid` + `boxes`, as `garble_replace`).

**Training.** `span_reband`'s shape with these items in place of the
rebanded windows: the seed's mix at its own bands + the items at
**0.85–0.95** repeated to ≈ 40 % of the records; warm μ 0.02 from the seed,
174 kana rows (the kana half first; kanji words carry kanji rows frozen) ×
23 steps = 4 002 steps, lr 1e-3 cosine, batch 4, routed. Plain FM — at
σ ≥ 0.85 the CF correction is a fraction of plain FM on A
(`proposal_length.md` § Count's caveat), so the pair buys nothing there.

**Read.** `sent` 184 paired against the floor: dup (the target), official /
≤ 1 edit (must hold: 29 / 92), **and the page axis must not move toward
larger text** — box, box_h, IoU vs EN ref at the floor's or better; a
box_h rise with dup falling is the size bias and fails the arm. Diagnostics:
`shared_dir --arms full,full_s,s_full` (which side of 0.8 moved) and the
`inject_count` ceiling (A@0.9: official 29 / 60 on the dup subset — the arm
cannot beat the trajectory put on the n-slot banner by hand).

## Steps

**Step 1 — canvases (GPU, ≈ 2 h).** 3 000 renders (≈ 1.7 s each) + reads →
≈ 1 000 items at the floor's keep rate; redraw and re-read on CPU. New
renders, not the floor cache: these are training data, and the floor's
184 are the read set. Sheet of 24 items before step 2.

**Step 2 — the arm (GPU, ≈ 1 h).** As above; read `sent` + `shared_dir`.

**If dup falls with the strings held and the placement at the floor's**:
the relative rule is carried; scale to the kanji rows and the full word pool,
and fold the recipe into `builder.TABLE` as a tier whose canvases come from
the seed (a generation step inside `data`).
**If dup falls and box_h rises**: the row took a size after all — the item
set had a common size (check the region size histogram against the
floor's; widen with untagged-caption and bubble-prompt canvases).
**If dup does not move**: the count is not a σ-less row's to carry; the
remaining route is σ-gated rows (a layout delta applied above 0.8 only), a
pack-format change, to be weighed against living with the slot.

## Result — the arm at 208 canvases × 10 words (2026-10-02)

Step 1 at 660 renders: 208 canvases (371 near-misses; 155 have no bubble, 8
do not fit), and each canvas drawn 9 more times with another pool word of
the same glyph count (user, 10-01: the same render, bubble and redraw, the
caption's quote replaced — the string alone changes, the render was not
selected on the swapped word, and the rows with an item go 150 → 163 of
174) → 2 073 items, px 32 / 43 / 55, the region's long side 134 / 188 / 276.
Step 2 as written: the seed's kana mix + the items at 0.85–0.95 × 6 (42 % of
29 838 records), warm μ 0.02, 174 rows × 23 steps, warm_cos 0.980. Jobs
`grow1`, `swap_r0`; `experiments/seed_synth/results/20261001-2355-swap_r0/`,
sheets `output/cjk_anima_scale/experiments/seed_synth_warm_swap/sheets_sent/`.

| `sent`, 184 | official | ≤ 1 edit | dup | box | box_h | flat_white | IoU vs EN |
|---|---|---|---|---|---|---|---|
| floor | 29 | 92 | 100 | 0.156 | 0.180 | 0.132 | 0.198 |
| seed_synth swap | 19 (+9 / −19, p 0.09) | 71 (+16 / −37, p 0.006) | **116** (+35 / −19, p 0.04) | 0.170 | 0.186 | 0.153 | 0.170 |

| main box | long side (quartiles) | glyph px | glyphs read − glyphs asked |
|---|---|---|---|
| floor | 354 / 401 / 457 | 62 / 78 / 93 | 2.37 |
| seed_synth swap | 371 / 427 / 467 | 65 / 78 / 92 | 2.65 |
| span_reband | 356 / 404 / 455 | 56 / 68 / 84 | 2.83 |

- The banner keeps its glyph size and gets longer, with more glyphs in it:
  `たすけて` → `たたすすけけて`, `ちょっと来い` → `ちちょっと来来いい`. The
  extra glyphs land on the long words (5 glyphs 1.69 → 2.88, 6 glyphs
  1.25 → 2.75 over the string; 2–3 glyphs unchanged).
- Per string, ≤ 1 edit of 8: ちょっと来い 7 → 1, やったネ 7 → 3, たすけて
  5 → 2, おしい 7 → 5; カメラ official 2 → 4, こうえん / かんがえ lose their
  last reads.
- The item set's size did not transfer (px 43 in the items, 78 in the
  banner before and after — `span_reband`'s 34 px items shrank it to 68),
  so the "no common size" answer held for size. The count did not come
  with it: three item sets at 0.75–0.95 (`b0305_reband`, `span_reband`,
  this one) each put more glyphs in the base's region. This is the
  proposal's third branch — the count is not a σ-less row's to carry from
  items trained above 0.85.
- Not run: `shared_dir` on this arm, the 3 000-render step 1.

## Closed by these reads

- The span lever (`proposal_length.md` § Span): rows at 0.85–0.95 do not
  shorten the base's region; they refill it.
- `cf_sense`-style count reads: teacher-forced, single-step; the trajectory
  read above answers the σ question directly.
- A σ gate at inference for `span_reband`-type arms: the loss is above 0.8.
- The counterfactual pair at 0.8–0.9: ≈ plain FM on A there.

Code: `experiments/span_reband/run_exp.py`, `experiments/shared_dir/run_exp.py`
(`full_s`), `experiments/inject_count/run_exp.py` (`redraw`, `Injector`).
