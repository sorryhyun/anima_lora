# experiments — idea validation for the scale line, bench-style

One directory per experiment; each validates one idea before any
production code changes. This is the line's
`bench/`: same envelope, same discipline, but scoped to the line and free
to import `cjk_scale/` and the line's `src/` primitives. The pre-retrain
experiments (influence, 300f pieces, line mode, stage_i, plan_2900's micros,
the hypothesis probes) moved to `../_archive/experiments/` on 2026-09-28;
their reads are in `../_archive/reports/` and `../hypothesis.md`.

The entries below name item pools as their data of record does — band group
+ recipe (`b0507` `scene_window`, `g0305` `grid_single`). Since 2026-10-02
the code names them by form and glyph px (`bubbleN_34`, `grid_16`); the
table is `../README.md` § Item pools, and `builder.tier_of` reads both.

## Contract

- `<exp>/run_exp.py` — the entry point, a thin argparse script. `--dry_run`
  must plan (sample, count, print) without touching a model. An experiment
  that builds on another's loads it with `cjk_scale.paths.load_experiment`
  (fresh each call — its import-time `pin_old_seed()` runs again); scoring
  comes from `cjk_scale.reads`, not from another experiment.
- Results: `<exp>/results/<YYYYMMDD-HHMM>[-<label>]/` with the standard
  `result.json` envelope (`bench/_common.py::write_result`) + `report.md`.
  Always pass `--label` — same-minute runs overwrite the dir
  (`project_bench_run_dir_collision`).
- Heavy artifacts (gradient tensors, latent caches) go under
  `output/cjk_anima_scale/<exp>_<label>/`, referenced from `result.json`,
  never into this tree (it is committed). Row arms (a `trained.pt` the
  eval renders, one dir per arm: `tl_*`, `tp_*`) go under
  `output/cjk_anima_scale/experiments/<arm>/`, with the stage's
  `--arm_path` pointed there.
- GPU work goes through the daemon
  (`make daemon-run ARGS="project/cjk_anima_scale/experiments/<exp>/run_exp.py …"`)
  and every launch names the pack
  (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`).
- Never write into a `data_*` dir: sample records, keep sub-set
  latent/TE caches in the experiment's own output dir.
- A verdict that closes (or opens) an idea is written into
  `../reports/` like any other read; this tree holds
  the machinery and the raw envelopes, not the line's memory.

## Experiments

- `stage_b/` — kept as the module `p1_cap` / `p2_route` / `c3_kanji` import
  (its item builders, donor sets and native reads); its per-render scoring
  moved to `cjk_scale/reads.py` (2026-09-30), which every experiment reads
  arms with. The proposal's Stage B: 36 donor kana trained on 568
  manga109s lines (`scene_spelled`, glyph-balanced draw, no repeated glyph)
  plus a count tier (`scene_single_small`: one glyph at 24–40 px in a bubble
  it fills 0.2–0.4 of, 0.3 of b0507). `u_S` = the donors' mean tangential Δ,
  added at one coefficient to the seed rows of ひ ま わ り さ く ら み ど も
  (arms `tb_*`) plus a random ⟂ control, read on five spelled words made of
  them and the ten alone. The held-out keys' floor was rendered once into
  `native_spell/`. **Ran 2026-09-26** → composition transfers (≤ 1 edit
  11 → 66 / 160, random 9) and doubling with it (repeats 25 → 53 / 320)
  (`../_archive/reports/stage_b_2026_09_26.md`).
- `p2_route/` — retrain_experiments § 4 C0 / C2: per-glyph routing
  (`ANIMA_VOCAB_GLYPH_ROUTE`, set in-process) rendered against the spelled
  and the unrouted (piece-row) caption of the same word, same prompts ×
  seeds; routed renders only in each dir's `native_route/`. **Ran
  2026-09-28 (c0)** → `p1_mix` routed ≤ 1 edit 14 vs spelled 11 / 16, piece
  row 0 (`../retrain_experiments.md` § 4). **c2**: 8 held-in donor words on floor /
  Stage B / `p1_cold` / `p1_mix` / `p1_lone` → routed = spelled on every
  arm, `p1_mix` ≤ 1 edit 80 / 128 (floor 1, `p1_lone` 9).
- `c3_kanji/` — retrain_experiments § 4 C3: 36 kanji, cold, on `p1_mix`'s rows as
  context; windows of dialogue lines (2–4 glyphs, kanji-first draw) as the
  in-word tier, routed captions, `p1_mix`'s table, 225 steps / row.
  **Ran 2026-09-28** → words ≤ 1 edit 3 → 36 / 96; new kanji official
  0 → 51 / 192, the seed's dense kanji 65 → 31 (`../retrain_experiments.md` § 4).
- `retrain_read/` — retrain_experiments § 5: a retrain run's routed read on a
  smaller grid (4 prompts × 2 seeds = 8 renders / key), the floor and
  `p1_mix` from their caches of record (8 × 2 restricted to prompts < 4):
  the run's `read` words (en; C2's eight pair with the cached floor /
  `p1_mix` `native_route/`) + 8 hiragana singles (swap, floor from
  `native_spell/`) + 6 katakana (swap, floor rendered once into
  `native_r4_swap/`); ≈ 260 renders. **Ran 2026-09-28** (`kana`) → C2's
  words ≤ 1 edit 29 / 64 (`p1_mix` 34, floor 1), katakana words 20 / 32,
  singles at the floor, `dup` 42 vs `p1_mix`'s 31 (`../retrain_experiments.md` § 5).
- `row_geometry/` — the retrain rows in row space (CPU, no render):
  `retrain_kana` + `c3_kanji` 225 / 450 vs the seed, the raw pack, T5 and
  `u_S` — shared direction, glyph-pair structure, PR, Procrustes, row vs
  read. **Ran 2026-09-28** (`rg1`) → one shared direction (⟂ the pack,
  common across runs, cos 0.83), centered PR = the seed's, no rotation, no
  read predictor (`../reports/row_geometry_2026_09_28.md`).
- `real_kana/` — `retrain_kana`'s 174 kana rows warm-trained on the
  training set's kanji-free OCR images (captions verbatim, loss box = the
  quoted lines' union); `--native` (own size, batch 1, dynamic-seq compile),
  `--full_sigma` (the LoRA trainer's sigmoid σ, no band law), `--plain_mse`.
  **Ran 2026-09-28** (`native_sig12`, 291 images, 2 088 steps) → every read
  ≈ 0 vs `retrain_kana` (words official 16 → 0 / 104, singles contained
  73 → 14 / 112); plain MSE drifted further and was stopped (`../future.md` § 1).
- `target4k/` — the seed's `target` ruler at 768×1344 (the user's ComfyUI shape,
  4 032 tokens), beside the 512² cache. **Ran 2026-09-30** → 8 / 14 vs 5 / 14 at
  512², but every render is a small scene on a black canvas with the string as a
  subtitle under it. `--raw` (Δ 0) and `--base` (no pack: Japanese → `<unk>`)
  draw the same composition, 0 / 14. The letterbox is the base's, and with no pack
  the text sits in the bubbles (`../reports/polish_seed_2026_09_30.md`).
- `polish_seed/` — `polish_b1` (loaded as a module, constants patched) on
  `seed_retrain_0930`'s 1 362 singles, 4 steps / row, μ 0.1; `--read` = `sent`
  (4 prompts × 2 seeds) + `target` 512² / 4 k against the seed's routed floor;
  `--color` drops monochrome / line-art scenes. **Ran 2026-09-30** → sent official
  29 → 11 / 184, ≤ 1 edit 95 → 60, targets held (`../reports/polish_seed_2026_09_30.md`).
- `sigma_split/` — idea.md § 2b check 1 / § 3, no training: the seed's rows
  gated by σ through the sampler's `context_alt` + `tag_drop_sigma` (Δ scale
  1 vs 0 at encode; arms `lo` / `hi` / `garble`), read on the `sent` grid
  against the routed floor cache; `--traj` decodes x̂0 per σ.
  **Ran 2026-09-30** → below σ 0.5 nothing moves the text (`lo` 0 / 184,
  `hi` = floor), text is decided at 0.9–0.7; switched at 0.8, `garble` keeps
  the base's bubbles and reads 16 / 29 official, 61 / 92 ≤ 1 edit
  (`../reports/sigma_split_2026_09_30.md`).
- `delta_scale/` — why `garble_replace`'s strings repeat, no training: `probe`
  (CPU) splits each word token's adapter output and DiT cross-attn keys into
  identity / slot / residual over all 24 slot orders of a 4-glyph word, per
  arm × Δ scale; `rows` + `read` render the seed rows with Δ × s on the `sent`
  ruler (`garble_replace`'s read leg). **Ran 2026-10-01** → the slot share falls
  with the reads across arms (cold 0.009), but shrinking the seed's Δ restores
  it and the reads fall anyway (Δ 0.9 ≤ 1 edit 92 → 48, dup 100 → 124;
  Δ 0.75 → 5): the rows shorten the base's sentence-length line, and the
  leftover slots are the repeats (`../reports/delta_scale_2026_10_01.md`).
- `b0305_reband/` — the seed's own b0305 kana items (`retrain_kana`'s 5 800
  `scene_window` items, 12–24 px) moved to σ 0.75–0.93 and trained warm from
  the seed (μ 0.02, 23 steps × 174 rows). **Ran 2026-10-01** → the floor's
  banner turns into the items' layout (small columns, sentence-length lines)
  on every kana caption at an ordinary drift (warm_cos 0.959); the read was
  stopped at 158 / 184 renders (`../hypothesis.md`).
- `shared_dir/` — `hypothesis.md` prediction 0, no training: an arm's update
  split into the rows' mean and the per-row residual, each rendered on the
  `sent` ruler at every σ or only below 0.8 (the seed rows above). **Ran
  2026-10-01** on `b0305_reband` (seed 0, 92 keys) → the layout break is in
  the residual and acts above 0.8; the mean keeps the layout and adds
  glyphs (dup 40 → 54) (`../hypothesis.md` § Result). `full_s` (the arm's
  rows above the switch, the seed's below) added 10-01 for `span_reband`:
  = `full`, the arm's effect is all above 0.8.
- `span_reband/` — `proposal_length.md` step 1: the seed's b0507
  `scene_window` items filtered to a bubble of ≤ 2 columns / lines, no sign
  / open region / second bubble (2 589 of 4 060), moved to σ 0.85–0.95 and
  repeated × 4 inside the seed's mix (41 % of 25 167 records), warm μ 0.02,
  174 kana rows × 23 steps. **Ran 2026-10-01** → the banner stays, its
  glyphs get smaller and more: dup 100 → 113, official 29 → 16, ≤ 1 edit
  92 → 70; box 0.156 → 0.132, box_h 0.180 → 0.211 (`../proposal_seed_synthesis.md`).
- `inject_count/` — no training: 60 floor `sent` renders with a leftover
  slot, the banner redrawn with the word filling its extent (`A`) or the
  render itself (`B`), put on the sampler as `(1 − σ)·z + σ·ε` at
  σ 0.95 / 0.9 / 0.85 / 0.8 and run out under the floor's rows and caption.
  **Ran 2026-10-01** → at 0.95 A = B (the state is overwritten, the scene
  re-rolls); at 0.9 A keeps the n-slot banner in half (official 29 / 60 vs
  B 5, dup 21 vs 48), 0.85 → 36, 0.8 → 39: the count commits between 0.95
  and 0.9, gradually (`../proposal_seed_synthesis.md` § 3).
- `seed_synth/` — `proposal_seed_synthesis.md` step 1 at 300 renders, no
  training: the seed rows on the scene pools' prompt stream with its text
  frame kept (`s1w`'s bubble / saying / sign frames and six canvases, the
  seed's training caption), a held-out kana window of 2–6 glyphs per render;
  every render read, the near-misses redrawn in their own bubble by the
  line's scene renderer (one line, or one column — two unequal top-aligned
  columns from 5 glyphs; the ink kept inside the outline, the rest of the
  bubble wiped) and read again. **Ran 2026-10-01** (render job
  `20261001-220655-a4042f`, redraw `20261001-225714-ab075e`) → exact 97 /
  300 (27 27 23 14 6 of 60 at 2–6 glyphs), near 175, a doubled glyph in the
  main box 76; the text extent spreads (long side 127 / 185 / 283 px, 30 %
  columns) where the `sent` floor's is one banner (354 / 401 / 457); 98
  canvases, all read ≤ 1 edit (74 near-misses have no bubble, 3 do not fit).
  The first look (the ruler's bare caption at 512²) drew the word over the
  scene — `seed_synth_nobubble_partial/`.
  `--grow 360 --swap 9` (10-01, job `20261001-2318-grow1`): 660 renders →
  208 canvases (the first 98 byte-identical), each drawn 9 more times with
  another pool word of the same glyph count, the caption's quote replaced
  (`garble_replace`'s `VARIANTS`; a random draw unless a candidate carries a
  glyph under 6 draws) → 1 865 kept of 1 872 dealt, 2 073 items; the kana
  rows with an item 150 → 163 of 174 (the rest are punctuation, ヂ, ヵ).
  **The arm** (`--tag swap --legs data train read`, job `swap_r0`, result
  `20261001-2355-swap_r0`): the seed's kana mix + the items at 0.85–0.95
  × 6 (42 % of 29 838 records), warm μ 0.02, 174 rows × 23 steps, warm_cos
  0.980 → `sent` 184: **dup 100 → 116** (+35 / −19, p 0.04), official
  29 → 19, ≤ 1 edit 92 → 71 (p 0.006); the banner is longer (box 0.156 →
  0.170, long side 401 → 427 px) at the floor's glyph size (78 px), IoU vs
  EN 0.198 → 0.170 (`../proposal_seed_synthesis.md` § Result).
- `kana_reband/` — `retrain_kana` with the band alone changed: its items, every
  one at σ 0.75–0.93, cold on the old seed, the kana run's trainer and 135 /
  row. `--rows hira`: the 81 hiragana rows on the 6 521 all-hiragana items
  (42 of the 2 314 multi-cell grids survive the filter), 10 935 steps. Read on
  `retrain_read`'s grid against `retrain_kana`'s reads of record. **Ran
  2026-10-02** (job `20261002-081557-f90bb5`, `results/20261002-0815-hira_r0/`)
  → nothing reads: 9 words `en` official 10 → 0 / 72, ≤ 1 edit 33 → 0, ≤ 2
  edits 53 → 0; 8 singles `swap` official 17 → 0 / 64, contained 44 → 1. The
  renders take the items' layout (small columns in bubbles and white boxes,
  sentence-length) with no identity — `garble_replace` cold's picture, and
  `windows.py`'s "0.8–0.95 dead at 48 px" for the singles. Row norm 204 at
  the end (the kana run ≈ 250). Sheets `…/kana_reband_cold_hira/sheets_r4/`.
- `sigma_split --b0305`, the mirror arms (10-02): the base above σ 0.8 (its
  garble, or the caption with no pack), the seed rows below, on § 8's 16
  renders → official 1 and 0 of 16 against the seed's 7; the rows write the
  word into the base's slots with repeats (`../findings.md` § 8).
- `grid_small/` — does a grid teach identity if its glyphs are small: the 81
  hiragana rows cold, 135 / row, no lone glyph above 40 px — `grid_single`
  2×2 – 3×3 at 24–36 px (σ 0.5–0.7) and 12–23 px (0.3–0.5), 60 % of 8 100
  items, + `builder.TABLE`'s `b0507` / `b0305` word groups; a cell's bubble
  sized to its glyph and the glyph at the cell's centre ± 8 %
  (`render_grid(bubble_fit=, cell_jitter=)`). Read as `kana_reband`'s, with
  the `p1_cold` / `p1_mix` caches on C2's words. **Training 2026-10-02** (job
  `20261002-112155-8e8225`); the day's reads are
  `../reports/band_size_2026_10_02.md`.
- `grid_44/` — `grid_lone` with a 44 px tier: half of `grid_29` moved to
  `grid_44` (0.7–0.9, the law's band for a single ≥ 40 px) with a `lone_44`
  twin, the lone share split over three sizes; `recap` legs as `grid_lone`'s
  (plain captions, `--band lo hi` to put every item at one σ band). Arm of
  record `recap_b7593` (every item at 0.75–0.93), trained and read 10-02;
  the read beside `grid_small` / `grid_lone` is
  `../../cjk_anima_reseed/reports/grid_small_lone_2026_10_02.md`.
- `reseed_recap/` — `grid_44`'s recap at the kana run's rows and budget: the
  81 hiragana + 85 katakana rows (the kana run's less its punctuation) cold,
  135 / row = 22 410 steps, 16 600 items; `grid_44`'s table with half of
  `grid_44` given to `builder.TABLE`'s `bubble1_52` (7.5 % each, `lone_44`
  5 %), plain grid captions, `--bands hp` (default: `grid_lone`'s `recap_hp`
  bands + the gradient read's for the 44–50 px tiers, untrained before) or
  `law`. Read on the kana run's 13 words + 14 singles (`en` / `swap` against
  its reads of record, and the plain clause against it and `grid_lone`'s
  recap arms). **Ran 2026-10-03** (`hp`, job `20261003-020144-3cd123`,
  193.7 min, `results/20261003-0201-hp/`) → under `retrain_kana` on the plain
  read — words official 15 vs 33 of 104, ≤ 1 edit 42 vs 73, dup 76 vs 52;
  singles official 30 vs 47 of 112, contained 90 vs 91, repeats 23 vs 7 —
  and over `grid_lone`'s `recap_hp` (60 / row) on the hiragana keys (words
  ≤ 2 edits 48 vs 33 of 72, singles official 22 vs 12 of 64). On the sheets
  both draw a banner; this arm's is longer, with kanji-like glyphs in the
  extra slots, and its lone single comes out small or not at all.
- `reseed_anchor/` — `reseed_recap` (rows, budget, table, plain captions,
  `hp` bands) with the columns lettered as Japanese and the base's own text
  beside the rows. Lettering (both variants): the `bubbleN` tiers drawn
  `tategaki` + `vert_forms` (the font's vertical ー and small kana), no window
  opening on a small kana or ー (15 408 of 187 218 dropped). `--variant
  anchor` adds 30 % of the window draws closed by a corpus ！ / ？ (10 930
  marked windows, 158 glyphs; the fold sends the mark to the base's `!` /
  `?`, no ext row) and an EN word (`sigma_split`'s 36-word pool) in one
  random cell of half the multi-cell grids (3 067 grids, outside the item's
  kind and px); `fix` is the lettering alone. Items, bands and px as
  `reseed_recap`'s (16 600). `build(prepare=)` carries the window rules.
  Data `results/20261003-0734-anchor_data/`, `…-fix_data/`. **`anchor` ran
  2026-10-03** (job `20261003-074735-8ffb80`, train 153 min,
  `results/20261003-0747-anchor/`) → no lift over `reseed_recap hp` on the
  plain read, paired: words official 11 vs 15 of 104 (7 / 11, p 0.48),
  ≤ 1 edit 40 vs 42, ≤ 2 73 vs 73, dup 75 vs 76; singles official 21 vs 30 of
  112 (7 / 16, p 0.09; hiragana 4 / 13, p 0.05), contained 87 vs 90,
  repeats 32 vs 23 (p 0.15). `en` / `swap`: words official 2 vs 4, ≤ 2 41 vs
  35; singles official 8 vs 14. Both far under `retrain_kana` (plain words
  official 33, singles 47). On the sheets the two arms draw the same scenes
  and banners. `fix` not trained.
- `stick_scale/` — the trained rows' shared mean ("stick") rescaled per
  family, the per-row residuals kept, no training: `run_exp.py --rows seed`
  (0930 seed, kana + kanji sticks, s 0.75 … 0 on `sent` seed 0 against the
  routed floor) and `--rows anchor` (`reseed_anchor`'s 166 kana rows, s 0.9 /
  1.0 / 1.1 on the 16 kana keys, paired against s = 1.0), on `shared_dir`'s
  `Rows`; `geometry.py` (CPU) reads the cold kana rows in row space (cos to
  the seeds, exposure / σ per row, T5, stick and spikes). Ran 2026-10-03
  (`results/20261003-1252-st0/`, `…-1312-sta0/`, `…-1315-geo/`) → a shorter
  stick costs identity and keeps the repeats (0930: ≤ 1 edit 45 → 25 → 11 →
  2 → 0 of 92, dup 40 → 44–53); ±10 % on the anchor rows is noise. Reads:
  `../../cjk_anima_reseed/reports/stick_2026_10_03.md`.
- `grad_identity/` — the band law on the training gradient, no training
  (`conflict`'s plumbing): the grid_44 items, the 81 hiragana rows cold at
  the pack rows, σ swept 0.2–0.95 with one ε per item × σ. Pass 1 swaps the
  caption's slot glyph (true row vs other rows — null: different rows'
  gradients are near-orthogonal at every σ); pass 2 (`--render`) re-draws
  the item with the slot's glyph swapped and reads one row's gradient on
  both — f = 1 − cos is the glyph-dependent share. `--render_tiers
  bubble1_52 bubble1_32 bubbleN_34 bubbleN_18` runs pass 2 on the bubble
  tiers (`render_into_scene(ref_text=…)`; `bubble1_52` from `retrain_kana`'s
  records) and reads a window's other rows from the same backward (f_cross).
  `--analyze <dir>` re-reads a results dir on CPU. Reads 10-02:
  `../reports/grad_identity_2026_10_02.md`.
