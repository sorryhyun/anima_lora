# plan_synth — the S line, live plan (rows on self-generated scene composites)

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

## Gates (S0 gates stand for the full-scale run; rulers re-based 2026-09-15)

Scene ruler is now **`en cos`** (PE-Spatial cos to the EN reference of
the same prompt/seed; floor ≈ 0.93–0.97 is the ceiling), placement ruler
**`box IoU`** (glyph box vs the "hi" box; floor 0.36–0.51, every trained
cond so far 0.05–0.25 — read with the sheets, it is harsh on small
boxes). `stage native` renders trained conds only (`--native_floor 1`
restores the old kept margin). Native runs on **both** clauses — `en`
(the trained JA frame) and `swap` (the EN ref's caption, word swapped) —
and a row counts as a word token only when it hits under `swap`.

- **native (EN clause, 8 held-out prompts × 4 kana × 2 seeds): hit & kept ≥ 24/64**, from
  P0b's measured baseline of **2/64** (its 32/64 hits are canvas wipes on
  the margin ruler). Scene-kept alone ≥ 56/64 (the delta must stop
  overriding the scene).
- singles ≥ 30/36 (P0b 36; reader noise), `single_ext` non-small ≥ 18/28,
  `single_kanji` ≥ 22/36, word ≥ 8/32, EN 24/24 — nothing P0b holds is
  lost.
- `phrase_held` > 0/16; `flip` order statistic ≥ 24/48 if strings are in.

Also read, not gated: row norm at the end of training (every arm drives
it to ≈ 125–130; the placement / identity trade-off is the delta norm),
per-char hits under `swap` (frame independence is per glyph, not per
arm — か vs あ), and the `native_swap` sheets before trusting a box IoU on
a small box.

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
`probes/rows_manifold_probe.py`): one shared direction (18 % of the
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

## Open risks

- **Small glyphs inside composites.** A kana in a 64–100 px bubble at 512
  is 4–6 latent tokens a side; the box-weighted loss and the 32 px floor
  are the mitigation. `--scene_fill 0.7` made this slightly worse (median
  51 px) in exchange for a realistic layout; if singles or dakuten fail,
  raise the bubble bar, not the loss.
- **Erase artefacts as a cue.** Ring-median fill inside a shaded bubble
  can leave a patch the row latches onto. `erase_miss` catches the
  wrong-blob case, not the patch; the sheets are the check.
- **The residual wipe is the delta, not the mix.** Seed 0 wipes 4/8
  prompts at flat 10 % and at flat 0 alike (was 7/8 at 60 %); every arm
  drives the row norm to ≈ 125, and that is what overrides the scene.
  The wipes are seed-shaped, so per-char n = 16 hides differences under
  ≈ 8. No data-mix arm is expected to move this further.
- **The bubble is a canvas.** Every composite puts the glyph inside a
  round white bubble; with flat items scarce the rows learn the bubble as
  their unit (flat 0: white disc on black). Placement diversity in the
  scene stage (subtitle bar / on-scene text) is the lever, not more
  bubbles.
- **Capacity at 0.9 — measured 2026-09-16.** 433 rows at ≈ 490 draws
  per row (10 k items, 53 k steps) fell to 13/36 singles; the exposure
  curve (1 330 / 670 / 490 → 100 / 75 / 36 %) is the risk realised. The
  levers are steps (≈ 1 000 draws per row → ≈ 108 k steps) and pinned
  shares per kind, not the composite share. Whether katakana's loss is
  interference on top of exposure is the 92-kana arm's question.
- **RAM.** The probe keeps every caption's text embedding in RAM (≈ 1.3
  MB each): 30 k items = 33 GB before latents and the DiT load, on a 46
  GB box. ≈ 10 k items per data build until the cache is paged.
- **Scene yield.** s0 17 % kept, s1 27 %, sl1 smoke 30 % (sentences
  read back better than single words); 443 scenes serve the unit arms,
  ≈ 1 000 sentence scenes are rendering. Each scene serves many swaps;
  distinct-items-per-row is bounded by `--n_items`, not by scenes.
- **Bubble capacity for sentences.** Single-line draw: long side / 32 px
  glyphs — the sentence line's blocker (branch B); wrapping is the fix,
  raising `--scene_min_box` is not (it is the short side).

## Not this plan

- Regulariser strength / `out_scale` sweeps (scale probe: direction, not
  magnitude).
- Zeroing or shrinking `c` at inference (table-parts probe: every part
  alone is 0/16). Warm-starting from P0b is now *measured*, not just
  argued (2026-09-15, `--init_rows`): identity survives, the trigger never
  grows — not a shortcut. The flag stays for seeding from S0's `f` later.
- Inference-time guidance away from a `c`-only branch: `f + g` without `c`
  was 0/16, so that direction removes the glyph before the canvas. A
  64-render curiosity at most.
- The glyph encoder in any form (see *Recipe*); the S line is rows-only.
- Contrastive terms on text-free native images.
- Pasting onto the dataset's real images (off-manifold paste, caption
  style mismatch, nsfw/artist tags) — the self-generated scene replaces it.
- `c_flat` in any form: the micro loop measured cap 0.75 ≡ removed
  (2026-09-15); the S recipe drops the switch.
- Q (the quoted-EN adapter-output direction) as an inference-time
  replacement for a trained `c` — measured, halves exact hits. Q fixed on
  in training — **closed 2026-09-16**: inert on frame-mix data (2×2).
- Normalising fullwidth punctuation to ASCII in the phrase data — the OCR
  line does not, so inference captions carry the ext rows; train the rows
  (branch A), do not fold them.
- More items per data build past ≈ 10 k without paging the text cache.
- The floor-based kept margin as a gate — replaced by `en cos` / `box
  IoU`; a bare ground with a bubble scored as kept, and the base itself
  wipes `portrait, simple background` for EN.
- Restarting a running arm for a monitor value (leak, `‖c_flat‖`): the
  gates read at the end, and a mid-run change loses attribution.
