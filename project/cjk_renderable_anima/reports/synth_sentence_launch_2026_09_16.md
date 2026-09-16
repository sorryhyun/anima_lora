# S line through the sentence-arm launch — status, recipe of record, frame-mix 2×2, next-steps record (2026-09-16)

> Moved out of `plan_synth.md` on 2026-09-16 evening, text unchanged: the
> status notes (09-15 night → 09-16 evening), the data mix and recipe of
> record with the measured exposure curve, the frame-mix 2×2 and micro-loop
> tables, and the three-branch next-steps record — branch C done, B's scene
> pool / wrapping / tall-bubble / punctuation / sentence-arm launches. Closing
> section: the sentence arm was stopped. `plan_synth.md` now carries only
> what is still to run.
>
> Index: [`README.md`](README.md). Backticked paths (`src/…`, `plan_synth.md`,
> `output/…`) are relative to the line root `project/cjk_renderable_anima/` or
> the repo root. Relative doc links below (`findings_seed.md`, `synth.md`,
> `reports/…`) were written from the line root.

## Status notes (as they stood in `plan_synth.md`)

> **Status 2026-09-16 (afternoon):** branch C is **done** — the 53k table
> was read in row space, at the adapter output, by transplant and by a
> pinned-trigger arm ([`findings_seed.md`](findings_seed.md) is the
> one-place summary). Verdicts: one shared trigger direction (18 % of the
> energy, ⟂ the pretrained quote direction Q under every frame incl. `She
> is saying`) + near-orthogonal residuals, no manifold to amortise;
> flat-trained identity does not transfer onto the composite trigger
> (0/64 for two flat donors), composite-trained identity does (23/64);
> inheriting the trigger buys **no steps** (`--pin_dir`: 54 = 54 at 2 k,
> 41 vs 39 at 500) — **composite exposure per row is the cost, and the
> levers left are per item** (rows per composite, glyph size). Katakana's
> miss is not row-space interference. **Decision (user, 09-16): next is
> branch B as originally planned** — finish the `scenes_sl1` pool
> (resumed at 1 208 / 3 400), render sentences / meaningful multi-token
> strings into the scenes, and **continue training warm-started from the
> 53k table** (`--init_rows`; it holds hiragana / kanji identity and the
> trigger, lacks words / katakana / small kana / sentences — all
> exposure). A rides inside B (punctuation rows through the phrases). The
> 92-kana interference arm is demoted to *only if katakana still fails
> after sentence exposure*.
>
> **Status 2026-09-16 (morning):** the micro loop is **closed** — the
> frame-mix 2×2 on 6 rows (table in *Frame-mix 2×2*) settled the last
> two mechanism questions: **frames are the lever** (swap hits 23 → 46
> with JA hits 60 → 54 and flat singles held) and **Q is inert** on
> frame-mix data (on vs off within 2 hits / 0.003 cos) — the recipe of
> record is frame-mix composite 0.9, Q off, no `c_flat`. The 12-row
> frame-mix arm's 日 collapse was **per-row exposure, not frames**
> (12/16 at ≈ 1 330 draws per row, 2/0 at ≈ 670), which the first
> full-inventory run then confirmed at scale: **433 rows × 53 000 steps
> (≈ 490 draws per row) fails the flat gates** — singles 13/36, ext
> 18/36, kanji 18/36, **words 0/32** (the item sampler gave the 92 words
> ≈ 8 items each), katakana + small kana the failure family, hiragana
> mostly holding, kanji missing to near-shape neighbours. Exposure
> curve on record: 1 330 → 100 %, 670 → 75 %, 490 → 36 % singles;
> **≈ 1 000 draws per row** is the planning number, and words / pieces
> need a pinned share. Two instrument facts from the same day: the probe
> holds its text cache in RAM (≈ 1.3 MB per caption; 30 k items blew the
> 46 GB box) — **≈ 10 k items per run**; native sheets now carry the EN
> `hi` reference as the leading column. **Sentence line opened** (user):
> a sentence-anchored scene pool `scenes_sl1` is rendering (55 EN
> sentence anchors, 3 400 prompts, smoke 30 % kept, bubbles wrap the
> sentence), and **Manga109 dialogue** is wired in as the phrase source
> (`--phrase_file`, 40 446 lines of 3–10 pieces, held by book;
> `--phrase_pieces` adds the file's uncovered pieces as rows — the top
> ones are the **untrained punctuation rows** ー ？ ！ 。 、 ・・・ ～ 「」).
> Next steps are the three branches in *Next steps (2026-09-16)*.
>
> Earlier status (2026-09-15 night, kept for the chronology): the
> full-scale cap-only isolation was killed at 15 min and the S line
> moved to a micro loop — 6 rows (あかす出人日), 2 000 steps, ≈ 25 min
> per arm incl. native. Five arms + three re-renders settled that day
> (`reports/synth_micro_loop_2026_09_15.md`): the `c_flat` cap is not a
> lever (cap 0.75 = removed), composite share is (0.4 → 0.9: hit & kept
> 23 → 41, en cos 0.797 → 0.860), the rows are bound to the JA clause
> frame (か 0/16 as Latin strokes under `swap`, あ 11/16), and Q fixed
> on in training looked glyph-dependent at 6 rows. Rulers: **`en cos` /
> `box IoU` against a shared EN reference** (`English text reads as
> "hi"`, same prompt and seed) replace the floor-based kept margin;
> floor renders are off by default. Flat 0 measured and closed — worse
> on every ruler, wipes unchanged; the flat items hold identity, the
> bubble is the composites' own canvas.
>
> What the S line is, how the instrument works, the S0 recipe / result and
> the measured budgets moved to [`synth.md`](synth.md); chronology is
> [`reports/`](reports/README.md). The P-line record (`plan.md`) stays the
> flat-only control; no P-line weights are used anywhere in the S line.
> Target artefact, kill criteria and the P2–P4 phase content in `plan.md`
> stand, re-based on the S table.

## Data mix (recipe of record; verdicts on `data_synth_micro6_fm` = frame-mix on the 6 micro rows)

The S0b mix (40 flat / 40 composite / 20 flat phrases, one flat layout,
`c_flat` cap 1.5) is the flat-only prior measured as the problem: `f`
learns the flat canvas first and 40 % composites do not undo it. The
micro loop moved the share and the number moved with it — this table is
what stands. The shares are settled (c9, flat 0); the **frame / font /
ink / tilt axes** were added 2026-09-15 night and are what the running
arm measures.

| share | source | as built | why |
|---|---|---|---|
| **90 %** | scene composites: one unit in the scene's own text slot — a bubble, or a held sign (singles only at the micro scale; phrases return at full scale where the region holds them) | `--scene_frac 0.9`, `--scene_fill 0.7`, `erase_miss` gate; pool `--scenes s0,s1` = **443** scenes (s0 174 `reads_as` + s1 269 over four frames; the open-fill / seam rules of 2026-09-15 in) | **the measured lever**: 0.4 → 0.9 lifts hit & kept 23 → 41, en cos 0.797 → 0.860, with flat singles 12/12 and EN 24/24 held (on s0 alone) |
| render | glyph font / colour / tilt | `pick_font` over 16 faces (`assets/fonts/FONTS.md`: 源暎アンチック, 源柔 / 源真ゴシック, コーポレート・ロゴ, たぬき油性マジック, 破線G, こよみゆる, Noto Serif CJK; Noto Sans out), cmap-checked per string; ink = the anchor's own lettering colour (`anchor_ink`) when it contrasts; 30 % tilted ±7° | every constant of the render (one gothic face, black ink, dead-level) is one more thing a row can absorb — flat 0 showed the rows take whatever is constant. Untested as a lever; rides in the frame-mix arm |
| **10 %** | flat singles in the font bubble | `--flat_bubble 1.0` (one flat layout) | **identity and frame-independence exposure, measured**: flat 0 lost 日 to Latin "a", JA hits 60 → 46, swap 23 → 5, and wiped exactly as often — the flat items are not the wipe source. Keep; the share above 10 % is untested at 0.9 |
| 0 % | natural phrases on flat canvases | `--natural_frac 0` | a flat item is what the rows over-learn; if phrases come back at full scale they ride *inside composites*, not on flat canvases |
| 0 % | random-order strings | `--strings_frac 0` (S1's question; the lever stays) | |
| 0 % | real corpus crops | out (two in three labels wrong, `datacheck.md`) | |
| **caption frame** | s0 composites are all the JA `reads as` clause on a bubble | **built 2026-09-15 (`--scene_frames`)**: the scene prompt draws a frame — `reads_as` / `bubble_reads` (`There is a speech bubble that reads "…"`) / `saying` (`She is saying "…"`) / `sign` (`He is holding a sign that reads "…"`) — and the composite caption swaps the JA text into the *same* frame. In the live build: reads_as 1 472 / bubble_reads 781 / sign 379 / saying 248 composites | rows are frame-bound (か 0/16 as Latin strokes under `swap`); the frame now comes from the image, no caption-only `--frame_mix` needed. s1: 269/1000 kept (27 %), `bubble_reads` 36 %, **`sign` 34 % and the first non-bubble placement (99 px boards)**, `saying` 21 % (still a bubble), `reads_as` 17 %. **`sfx` set aside** (user, 23:00): 102 kept but the base draws the word on a title bar / banner, not as SFX — pool `s1sfx` exists, not in the mix |

Builds on record: `data_synth_micro6_fm` (6 rows, 1 600 items over the
443 s0 + s1 scenes — the 2×2's data), `data_synth_micro12_fm` (12 rows,
3 200 items), `data_synth_full_fm10k` (433 rows, 10 000 items — the seed
run's data; `data_synth_full_fm` at 30 000 items is on disk and unusable:
RAM). The 6-row `data_synth_micro6_c9` (s0 only, Noto Sans, black ink, no
tilt) is the single-frame control.

**Exposure, measured (the number the full-scale run failed on):** draws
per row = steps × batch / rows, and the flat singles follow it — 1 330 →
12/12 (m6fm), 670 → 18/24 with 日 lost (m12fm), 490 → 13/36 with katakana
and small kana gone (full 53k). Steeper than linear at scale. Two
consequences for the full-scale data: (1) plan ≈ 1 000 draws per row and
size steps from the row count, (2) **pin the share of every kind that
must be learned** — the composite sampler draws by item, so at 433 rows
the 92 words got 754 of 10 000 items (≈ 8 each, 33 word rows at norm 0)
and scored 0/32. Sentence pieces (`--phrase_pieces`) are in the same
position: they are trained only through the phrases that carry them, so
the phrase share *is* their exposure.

## Recipe of record (micro, rows arm, from scratch; full-scale run owed)

    Δ_r = f_r        (row-norm units; no shared vector)

- **No `c_flat`** (`--c_flat 0`): cap 0.75 ≡ no `c_flat` on every native
  number and on the sheets (23 vs 24 hit & kept); the switch is dropped,
  not tuned. No encoder, no warm start (`f_r` from zero), **no Q at train
  time** — closed 2026-09-16: on frame-mix data Q on vs off is within 2
  hits and 0.003 cos on both clauses (the 6-row "Q helps か" reading was
  a single-frame artefact). Rows-only is the S line.
- **Frame mix** (`--scenes s0,s1`, four prompt frames from the image):
  the measured lever for frame independence — swap hits 23 → 46 of 64 on
  the same 6 rows, JA hits 60 → 54, flat singles 12/12, 日 12/16 on both
  clauses. `s1sfx` (banner placement) and `sl1` (sentence bubbles) are
  pools not yet in any arm.
- σ band 0.7–0.9; rows lr 1e-3, cosine; `μ‖f‖²` pull 1e-3
  (`--free_residual`); rectified flow on the band, `--box_weight 4` inside
  the swapped box.
- Batch 4, compile, no grad-ckpt, pool `448,512:2,448x512,512x448`.
- **Micro scale (what every verdict above is on):** 6 rows
  (`--units chars:あかす出人日` — 3 kana + 3 corpus kanji, one Qwen piece
  each), 2 000 steps ≈ 13 min at 2.5 it/s, + native on `en` and `swap`
  ≈ 25 min per arm. Recipe arm `rows_synth_micro6_fm_m6fm_s2k_qoff` (jobs
  `d7a880` / `78c719`); single-frame control `…micro6_c9_m6c9_s2k_qoff`;
  12-row `…micro12_fm_m12fm_s2k_qoff` (≈ 670 draws per row).
- **Full scale, first run (2026-09-16, `rows_synth_full_fm10k_full_s53k_qoff`,
  jobs `9393f3` / `106e0e`):** 92 kana + 68 ext + 200 corpus kanji + 100
  words (8 held), 10 000 items, 53 000 steps = 342 min at 2.59 it/s, ≈
  490 draws per row. **Failed the flat gates** (singles 13/36, ext 18/36,
  kanji 18/36, word 0/32, combo 0/36; native en 36/64, swap 18/64 with か
  0/16). Not the seed. Training itself was clean (loss flat 0.09–0.11,
  mean row norm 0.58, one shared row at 2.2 — **ext row 58974, the `~`
  symbol row, trained through the `kaguya-sama … ~tensai …~` copyright tag
  in 81 composites; a tag leak, not a clause piece — drop it at bake;
  `reports/krzh16_2026_09_16.md`). What it measured: the exposure curve above, and
  that 433 rows at composite 0.9 do not interfere *visibly* beyond
  exposure (hiragana held, katakana did not — an exposure-or-interference
  question the 92-kana arm below answers).
- **Full scale, the gate run (owed):** same inventory, **≈ 1 000 draws
  per row** — 433 rows → ≈ 108 000 steps ≈ 12 h — with the word share
  pinned (≥ 25 % of items); or warm-started from the 53k table (P0b showed
  warm starts keep identity; `--init_rows`) for the remaining budget.
  RAM rule: ≈ 10 000 items per data build.
- Eval: `stage native` renders trained conds only (`--native_floor 0`),
  both clauses (`--native_clauses en,swap`), scored against the shared
  `English text reads as "hi"` refs (`--stage enref` once, arm-independent).
- Controls: 0.4-share arms (`…micro6_m6_s2k_{nocflat,cap075}`), S0b
  (`rows_synth_s0b_s24k_S0b`, full scale, cap 1.5) and P0b
  (`encoder_wdsek_w120_s24k_p0b`, flat-only).

Exact train argv: `output/daemon/jobs/20260915-180203-54e3e5/job.json`
(`--stage train eval --arm rows --data_tag synth_micro6_c9 --train_steps
2000 --batch 4 --t_min 0.7 --t_max 0.9 --compile 1 --grad_ckpt 0
--aggressive_recompute 0 --lr_rows 1e-3 --lr_decay cosine --free_residual
1e-3 --box_weight 4 --seeds 2 --no_floor --c_flat 0`). The S0b argv stays
in `reports/synth_s0_s0b_2026_09_15.md` "S0b build + launch" as the
full-scale template (swap its data flags for the table above).

## Frame-mix 2×2 (2026-09-16, settled at 6 rows)

Prediction on record (09-15 night) was: frame mix lifts `swap` on the
か-type rows with JA hits ≥ 50 and en cos ≥ 0.86, and Q adds nothing.
Measured (native あかす日 × 8 prompts × 2 seeds, hits = both readers;
chronology `reports/synth_micro_loop_2026_09_15.md`, last sections):

| arm | rows | frames | Q | JA hit / en cos / IoU | swap hit / en cos / IoU | singles |
|---|---|---|---|---|---|---|
| 0.9 Q off (c9) | 6 | 1 | off | 60 / 0.860 / 0.07 | 23 / 0.903 / 0.09 | 12/12 |
| 0.9 Q on | 6 | 1 | on | 48 / 0.838 / 0.09 | 23 / 0.885 / 0.13 | 10/12 |
| m12fm | 12 | 4 | off | 44 / 0.891 / 0.18 | 26 / 0.926 / 0.24 | 18/24 |
| m6fm Q on | 6 | 4 | on | 54 / 0.870 / 0.12 | 44 / 0.902 / 0.14 | 11/12 |
| **m6fm Q off** | 6 | 4 | off | 54 / 0.868 / 0.12 | **46** / 0.905 / 0.17 | 12/12 |

The prediction held on both counts. Frames are the lever (swap 23 → 46
at a cost of 6 JA hits, flat eval untouched); Q is inert on frame-mix
data and closed; 日 at 12/16 on both clauses at ≈ 1 330 draws per row
says the m12fm collapse (2 / 0) was exposure. Placement (IoU 0.12–0.17
vs the floor's 0.36) is still the unpassed ruler — m12fm's higher IoU is
the weak-delta pattern (rows that draw little sit at the base's
placement), not a placement gain.

## Micro loop (2026-09-15 evening) — what is settled at 6 rows

Data: `synth_micro6` (60 / 40 flat / composite, 1 600 items) and
`synth_micro6_c9` (10 / 90). Native あかす日 × 8 prompts × 2 seeds; hits =
both readers.

| arm | JA clause hit / en cos | swap clause hit / en cos | note |
|---|---|---|---|
| 0.4, cap 0.75 | 56 / 0.817 | – | = no `c_flat` on every number |
| 0.4, no `c_flat` | 57 / 0.797 | 10 / 0.830 | |
| 0.4, Q fixed | 52 / 0.829 | – | |
| **0.9, Q off** | **60 / 0.860** | 23 / 0.903 | か 0/16 under swap (Latin strokes), あ 11/16 |
| 0.9, Q on | 48 / 0.838 | 23 / 0.885 | か 7/16 under swap, 日 → 目 (JA 3/16) |

Settled: cap ≠ lever; composite share = lever; rows are frame-bound;
Q = frame-independence for some glyphs at an identity pull for others.
Micro verdicts are on mechanism; 246-row interference is untested
(W2's 24-kana collapse) — the winner needs one full-scale run.

## Next steps (2026-09-16) — three branches, one arm at a time

The mechanism is settled and the recipe is fixed; what is open is
**capacity** (rows × exposure) and **what the rows are for** (units vs
sentences). The user named three directions (09-16 morning); they are
not exclusive, and each has a cheap first arm.

**A. Train the untrained tokens** — the vocab pack's symbol block is
routed but untrained, and the sentence line needs it: fullwidth
punctuation (ー ？ ！ 。 、 ・・・ ～ 「」) are ext rows, ASCII `!?,.…` are
pretrained pieces, and the OCR line (`anime_tools.ocr._text.drop_symbols`)
keeps punctuation as read — no fullwidth → ASCII fold — so captions at
inference carry the ext rows. Decision (user, 09-16): **follow the OCR
line**, do not normalise to ASCII, and train these rows *through the
phrases* (`--phrase_pieces`; they are its top pieces), never as singles.
Owed on the data side: apply `drop_symbols` to the phrase file so the
training character set equals the OCR caption set. The probe's `norm()`
strips punctuation before matching, so punctuation rendering is read on
the sheets, not in the hit counts — a punctuation-aware read (CER on the
unstripped string) is a small eval change if it becomes a gate.

**B. Sentence training as is** — the phrase composites on the sentence
pool. Pieces in place: `scenes_sl1` (rendering; 55 sentence anchors, 30 %
kept, bubbles wrap the sentence, regions 60–200 px), `--phrase_file` on
Manga109 dialogue (40 446 lines 3–10 pieces, 87 books, 6 held out whole →
`phrase_held`; `phrase` = 16 trained lines for the memorisation /
generalisation split), coverage 11 255 lines with the 433-row inventory
+ 100 phrase pieces (3 458 without them). Two blockers before an arm:
(1) **the composite draw is single-line** — the region's long side / 32
px caps a phrase at 3–6 glyphs, and 108 of 180 phrase-kind draws in the
smoke fell back to singles; 2–3-line horizontal wrapping in
`render_into_scene` / `region_capacity` is the fix (sl1's wider bubbles
help but do not remove it); (2) **phrase share must be pinned** (the
words lesson) — `--natural_frac` sets the flat-phrase share and the
composites mirror the flat kinds, so the share of items that are phrases
is the exposure of every phrase piece. First arm: 6 rows' worth of
exposure on ≈ 100 pieces is not cheap; the honest micro version is the
**12 micro rows + their phrase pieces** on `sl1` + s1, phrase share 0.5,
≈ 1 000 draws per row, reading `phrase` vs `phrase_held` and the
punctuation on the sheets.

**C. Analyse the trained tokens** — the 53k table is 434 rows of
measured signal, cheaper to read than to retrain. Questions it can answer
without a GPU: which rows hit and which did not (per family: hiragana
mostly yes, katakana no, small kana no, kanji to near-shape neighbours),
whether hit correlates with row norm / items-per-text / font coverage /
the row's pretraining frequency (the pack's `stats`), whether the
katakana failure is exposure (draws per row equal to hiragana's, so
**no** — it is either interference with hiragana or a render-side
cause: fonts, the JA readers' katakana bias) — and the geometry of the
learned rows (cos between hit and miss rows, hiragana vs katakana
clusters; the wake geometry tools in `wake_geometry.py`). **Geometry
read 2026-09-16** (`reports/rows_manifold_2026_09_16.md`,
`src/bench/rows_manifold.py`): one shared direction (18 % of the
energy, the same one S0/S0b rows shared, ⟂ Q at the adapter output) plus
near-orthogonal residuals; ば↔ぱ-type small-mark locality only; hit
tracks along-m̂ (+0.32) and exposure (+0.24), not crowding (−0.09);
hira↔kata pairs are random-distance — **katakana's miss is not
row-space interference**; word rows are untrained (norm 0.12); no
frame — `She is saying "…"` included — shares a representation with the
delta beyond the frozen adapter's frame shift. **Transplant + pin
(same day, `reports/transplant_2026_09_16.md`)**: flat-trained residuals
render 0/64 on the 53k m̂, composite-trained ones transfer (23/64); a
frozen inherited m̂ buys no steps (`en` 54 = 54 at 2 k, 41 vs 39 at 500)
and costs `swap` at convergence — exposure per row on composites is the
budget, and the levers left are per-item (rows per composite, glyph
size). One GPU arm
belongs here: **92 basic kana at 23 000 steps** (≈ 1 000 draws per row,
≈ 2.5 h) — if katakana holds there, the 53k katakana loss was
interference at 433 rows; if not, it is render-side.

Order of record (revised 2026-09-16 afternoon, user): C is done
(`findings_seed.md`); **B is next** — (1) the scene pool: `scenes_sl1`
(job `20260916-114945-8f389e`) was stopped at 2 125/3 400 renders with no
kept/rejected pass; **`scenes_sl1w`** replaced it (wide shapes
576×448 … 640×384, 1 000 renders, **276 kept, 28 %**, usable-region short
side median 83 px) and is the sentence pool of record; (2) **wrapping —
done 2026-09-16 evening**: `render_into_scene` / `fit_text` /
`region_capacity` take `max_lines` (`--scene_max_lines`, default 3) and
`cuts` (the caller's Qwen piece boundaries, so a row's unit is never
split across lines); layout is **vertical first** (columns right-to-left,
ー〜 rotated, 、。 top-right of the cell; horizontal lines only when the
text cannot fit that way), more columns win only at 1.4× the glyph,
kinsoku-nudged cuts. The wrap alone did not move it — the floor did:
`--scene_min_glyph` **40 → 28** (default). Phrase fit share on sl1w × Manga109
lines 4–12 glyphs (828 draws): 40 px 1 line 3 % → 3 cols 15 %; 28 px 1 line
16 % → 2 cols 43 % → **3 cols 52 % (34 % vertical)**; 3 cols over 2 buys
~9 points. 28 px at 512 is ≈ 3.5 latent tokens a glyph — this is the
small-glyph risk below taken on purpose. **Tall bubbles (user, 2026-09-16
evening: tategaki pool first, regenerate when short).** The wrap fit is
bounded by bubble *height*, and the base draws EN horizontally, so its
bubbles are wide: sl1w 116/276 taller than wide (61 at AR ≥ 1.3, height
median 136 px); anchor length, frame and canvas do not move it (short
anchors just make small bubbles). The lever is asking for *Japanese*
text: `ja_reads_as / ja_bubble_reads / ja_saying` frames (`japanese text`
tag, 24 built-in short manga anchors, `--scene_ja_anchors`), the letters
garbled and erased anyway, the judge taking every detector box as an
anchor (no read match). Smokes, 96 renders, tall canvases
`384x640,448x640,448x576`: plain **20 kept (21 %), tall 17/20, AR ≥ 1.3
15, height 121 px**; with `--scene_extra_tags monochrome,screentone`
**30 kept (31 %), tall 24/30, AR ≥ 1.3 24, height 134 px** — rejects are
`open_bubble` ~45 % (the base letters JA columns with no bubble) and
`small_box` ~28 % (tategaki columns are narrow; `--scene_min_box` 56 → 40
for a full run). **Not run at scale** (user, 09-16 evening): the intent is
the rendered EN pool sl1w, tall-centred, with the JA text swapped in —
the full ja_manga job (`20260916-154856-a14ef5`, 2 400 renders) was killed
at launch; the JA-frame recipe stays on record as the lever if the tall
pool runs short. `--scene_tall_ar 1.0` on the data stage keeps only
tall-region scenes of every listed pool, so the sentence pool = the tall
subsets of sl1w (116) + s1 (90) + s0 (74). `comic` / `2koma` /
`greyscale` stay out of the prompts (native held-out tokens).
(3) **punctuation first, as singles** (user, 09-16 evening): the fullwidth
marks `、。・ー〜～！？「」`, `！！ ・・・ ・・・・` and small `っ ッ` are ext rows
the 53k table never touched (rows come from the captions, and no phrase
was in); they are the top pieces of the Manga109 dialogue outside the
inventory (ー 6 712, っ 4 573, ？ 4 473, ！ 4 424 …). They are
single-letter addresses, so they train like kana singles on the s0 / s1
pools, not through phrases: `--units list:…` (each unit one Qwen piece with
an ext row; `！？` is two pieces and stays out) adds them to the singles
pool at 2× and forms eval group `single_extra` — read on the sheets,
since the readers' `norm()` strips punctuation before matching. Arm
**`rows_synth_punct_punct_s4k`** (job `20260916-155710-01337f`): the 53k
recipe (`--scenes s0,s1`, 92 kana + ext + 200 kanji + 100 words, composite
0.9, min glyph 32) + 15 extra units, 6 000 items, **`--init_rows` the 53k
table**, 4 000 steps. `drop_symbols` on the phrase file changes 0 of
40 446 lines (already clean). (4) **the sentence arm, budget 24 000
steps** (user): the rendered EN pool sl1w, tall-centred
(`--scene_tall_ar 1.0`; + s1 / s0 tall subsets if the pool runs short),
JA phrases swapped in — `--phrase_file … --phrase_pieces 40`, 10 000
items, `scene_frac 0.8 / natural_frac 0.1` (phrase share 0.5 among
composites), min glyph 28, 3 columns — warm-started from the punctuation
table (`--init_rows` takes a comma list since 09-16 — the 53k table + the
punctuation table, later overriding by ext id — if the two are kept
apart), read on `phrase` vs `phrase_held`, `word`, `swap`, and the flat
singles as the regression guard. Queued behind the punctuation arm
(user, 09-16 16:00: cancel only if the punctuation arm looks wrong):
**`rows_synth_sent_tall_sent_s24k`**, jobs `20260916-160004-6b244f`
(data train eval; pool = tall subsets of sl1w + s1 + s0 ≈ 280 scenes;
the 15 extra units stay in the singles pool) and `…-e3ec6e` (native, en +
swap). The 92-kana arm runs only if
katakana still fails after sentence exposure; the 108k gate run is
re-based on the sentence table.

Flat 0 is measured and closed (2026-09-15 20:50,
`rows_synth_micro6_c10_m6c10_s2k_flat0`): seed-0 wipes unchanged (11/32),
JA hits 60 → 46, swap hits 23 → 5, 日 drifts to Latin "a"; with no flat
items the rows learn the *bubble* as their canvas. Flat 10 % stays. The
wipe is the delta norm, not the mix; the placement lever (text on a
subtitle bar / on the scene, as the EN refs place "hi") is scene-stage
work — `s1sfx` (banner) is the pool that has it and is in no arm yet.

## Scene yield (open-risk note as of 09-16 morning)

- **Scene yield.** s0 17 % kept, s1 27 %, sl1 smoke 30 % (sentences
  read back better than single words); 443 scenes serve the unit arms,
  ≈ 1 000 sentence scenes are rendering. Each scene serves many swaps;
  distinct-items-per-row is bounded by `--n_items`, not by scenes.

## Sentence arm stopped (2026-09-16 evening, user): the data has no sentences

`rows_synth_sent_tall_sent_s24k` ran as job `20260916-180408-b915a8` (the
queued `20260916-160004-6b244f` spec, warm-started from
`output/wake_probe/rows_synth_full_fm10k_merge_punct/trained.pt` — the 53k
table with the `punct_only_s3k` rows merged in, `merge.json`). Data stage:
10 000 train items `{'font': 1000, 'phrase': 1000, 'scene': 8000}`, eval
181 prompts. Stopped at step ≈ 10 300 / 24 000 (2.5 it/s, loss 0.06–0.13)
on the user's read: **`data_synth_sent_tall` holds no sentences**. Nothing
from this arm is a result; the data build is what is owed
(`plan_synth.md`, step 1). Punctuation arms on disk from the same
afternoon, unread here: `rows_synth_punct_punct_s4k`,
`rows_synth_punct_only_punct_only_s3k` (each with `report.md`).


## Relaunch (2026-09-16 20:13): `rows_synth_sent_q_sent_s24k` — quotas, tategaki only, sl1w

Jobs `20260916-201323-226f69` (data train eval) and `…-a06a68` (native, en +
swap), argv in `README.md` *How to run*. Three earlier submissions the same
evening (19:48, 20:03, 20:11) were killed during their data stage as the
user read the smoke sheets — one column for the short kind, two columns at
most for sentences, 6-glyph sentences as one column, two scenes out. What
changed against the stopped arm, all in `src/data/synth.py` / `src/common/render/scene.py`
/ `src/cli/data.py`:

- **Hard kind quotas** (`--scene_mix single=0.1,short=0.5,sentence=0.4`):
  the composites' kinds are counts, shuffled. **Text first, then a scene
  that holds it**: a length uniformly among the kind's lengths that at
  least 10 scenes hold, a text of that length, then a scene among those
  whose capacity holds it, weighted `1 / (1 + uses)` (scene-first with a
  text that fits piled the short kind at 2 glyphs; plain uniform-among-
  fitting put 124 of 400 items on one scene). A text that renders nowhere
  is a recorded miss, never another kind.
- **Columns**: tategaki only (`--scene_vertical 1`); `single` and `short`
  one column (`--short_max_lines 1`); sentences at most two
  (`--scene_max_lines 2`; three read wrong on the sheet), and one whenever
  ≥ 10 scenes hold the text in one column (`--scene_fewest_lines 1`,
  `caps1`). Sentences draw at **20 px / fill 0.9** (`--sentence_min_glyph`,
  `--sentence_fill`; at 28 px / 0.7 two of 276 sl1w bubbles hold a 6-glyph
  column, at 20 px / 0.9 about half) — the small-glyph risk taken on
  purpose for the sentence kind only; short / single stay 28 px / 0.7.
- **Sentence floor**: ≥ 6 kana + kanji letters (punctuation / digits / ー
  not counted) *and* ≥ 4 distinct — the length floor alone let
  ハハハハハハ / おやおやおや through.
- **Short kind = 2–5 pieces** from `dialogue_2_10.tsv` (= `dialogue_3_10.tsv`
  + 2 528 two-piece lines from the same XMLs under the same character-set
  rule; 87 books unchanged, so the held books are the same six), below the
  sentence floor, ≥ 2 distinct letters, and (`--short_lexical 1`, user:
  combined glyphs must make words) carrying a word piece — a multi-glyph
  kana Qwen piece or a kanji. Keeps 8 740 of 11 611 such lines; the drops
  are mostly interjections / SFX (あっ, ぎゃああ, せーの) plus words the
  tokenizer splits into single glyphs (きつね, ほんと？).
- **Scenes**: all of sl1w minus `--scene_drop sl1w:332,957` (bubble-less
  tall regions — a hooded sketch's body, a box beside a figure — that the
  quota reused 12–13× per 400 items); `--scene_min_tokens 900` (the 512²
  family only, user: no 448² — sl1w is 960–1 120 tokens throughout, s1 /
  s0 would lose their 448² and 448×512 scenes); `--shapes 512` for flat
  items (none: `--scene_frac 1.0`, flat singles stay the eval guard).
  Warm start `rows_synth_full_fm10k_merge_punct/trained.pt`, 24 000 steps.
- Eval groups added: `short` / `short_held` (16 each); `phrase` /
  `phrase_held` now sample sentences only (213 eval prompts, was 181).

Last smoke before the launch (400 items, CPU, before the two-scene drop):
planned 40 / 200 / 160, drawn 40 / 200 / 160, **missed 0**, 138/276
scenes used, busiest 13; short 2–4 glyphs (64 / 58 / 78), one column
200/200; sentences 6–16 glyphs flat, **6–8 one column (40/40), 9–16 two
columns**. Training pools: sentence 3 606 (held 311), short 2 652 (held
217), 4 248 lines in no kind. The 1-column short cap is 4 glyphs (≥ 10
scenes at 28 px) — 5-piece lines longer than that do not draw; the tall
pool (plan step 2) is the lever for longer one-column text.

Deleted after the launch (plan step 1; 17 GB): the smoke data builds
(`data_nokana_flat_smoke`, `data_nokana_smoke`, `data_phrase_smoke`,
`data_synth_sent_smoke`), aborted builds (`data_synth_punct_tall`,
`data_synth_punct_cold`), the punctuation arms' data (`data_synth_punct`,
`data_synth_punct_only` — the arm dirs with `report.md` stay), the flat-0
data (`data_synth_micro6_c10`; its arm dir stays), the unusable 30k build
`data_synth_full_fm` and its RAM-killed arm
`rows_synth_full_fm_full_s53k_qoff`, the sentence-less
`data_synth_sent_tall` + stopped `rows_synth_sent_tall_sent_s24k`, and the
superseded scene runs `scenes_sl1` (stopped, no judge pass), `scenes_ja_manga`
(killed at launch), `scenes_sl1w_smoke`, `scenes_sl1smoke`. Kept: every
arm dir a table or report cites, `data_synth_full_fm10k` (the 53k seed's
data), the micro data builds the 2×2 / micro-loop tables are on, and the
two JA-frame smokes (`scenes_ja_manga_smoke`, `scenes_ja_tall_smoke`) —
the measured evidence behind step 2.
