# cjk_anima_reseed

The JA vocab pack's rows re-seeded cold, then trained warm on dialogue lines. Where the line stands — methods, data mix, the ruler, progress over preview51: **`progress.md`**. What an arm is judged on: `criteria.md`. Next: `plan.md` (225 new kanji rows), then `release_plan.md` (`anima-jp-extended` on HF). How to read a trained row: `structure_candidate.md`. The live reads: `reports/` (the ruler, sent_ball / sent_whole / sent_stick, sent_kanji). Why the line started: `_archive/motivation2.md`.

`_archive/` (10-06): the banner-grid era's reports and the configs of the arms that lost (cold kana tables, stick / ball re-fits, sent_ball / lr2 / stick), and the done `sent_plan.md`. A config there still runs by path: `run.py project/cjk_anima_reseed/_archive/configs/<run>.toml …`.

## Code

| file | what |
|---|---|
| `run.py` | front door: `run.py <run> data [--frac f] \| train \| read` (`read`: the plain read against `READ_AGAINST` → `results/`) |
| `probes/probe_split.py` | rows swapped at render, no training: two runs' rows split at a σ switch, one run's rows as stick / spikes, x̂0 per σ (`_archive/reports/probe_split_2026_10_04.md`); a cold arm's ball on retrain_kana's stick and back (`_archive/reports/ball_2026_10_04.md`); the ball runs as trained (`_archive/reports/ball_rk_2026_10_04.md`) |
| `stick_fit.py` | CPU: the stick runs' sticks, the kana / kanji burr, the band arms' sticks, the hiragana rows as stick + ball and the swap arms' EN-ref cos (`--legs ball`), scene numbers and sheets on cached renders (`_archive/reports/{ball,stick_fit,stick_scene,stick_rk,stick_rk_jt50}_2026_10_04.md`) |
| `ruler.py` | the dialogue ruler (`criteria.md`): `build` (CPU) draws 96 bubble-dialogue strings from the training set's captions, each with its own image's prompt and an EN reference line → `output/cjk_anima_reseed/ruler/ruler.json`; `run [--arms a,b] [--label l]` (GPU) renders what is missing — the floor (EN refs, retrain_kana, seed_retrain_0930) once — and reads every arm against it → `results/<ts>-ruler-<label>/` (`reports/ruler_2026_10_05.md`) |
| `transplant.py` | CPU: a focus run's rows onto another run's (`FROM` / `ONTO` / `NAME` at its top; 10-08: `sent_kanji_225`'s 225 onto `sent_kanji_pres` → `seed_1008`). `stick` reads the focus rows' mean against ONTO's kanji stick and both runs' Δstick; `write` → `output/cjk_anima_reseed/<NAME>/trained.pt` and the baked pack (`plan.md` § 3) |
| `punct_pack.py` | CPU: the punct base pack (10-05, the green leaf's fix) → `models/vocab_packs/anima_cjk_vocab_pack_punct`: the raw pack plus wider folds, dot runs and a `…` row (ext 69 558, at T5's `...`); the rules in its docstring |
| `probes/probe_geom.py` | rows held, no step: f0's first batches' in-box / out-box gradients on the rows in `structure_candidate.md`'s terms, split-half signal, an AdamW replay against SGD, and the draws by σ / tier (`reports/probe_geom_2026_10_06.md`) |
| `probes/probe_cf.py` | rows held, no step: counterfactual-input FM against plain FM per draw on f0's start rows — A′ / B sibling pairs re-lettered on the `sent` items' own scenes (`swap` / `dup`), σ 0.5–0.7, split-half signal and the CF leverage λ (`reports/probe_cf_2026_10_06.md`) |
| `probes/probe_scene.py` | rows held, no step: a crossed lines × scenes × noise-draw block per row, the gradient's variance split by a three-way ANOVA — how much the scene sets, under the box-share loss, each of its terms and plain MSE (`reports/probe_scene_2026_10_06.md`) |
| `probes/probe_accum.py` | GPU smoke: 32 mid-frequency kanji live on f0's data, plain AdamW against per-row accumulation (step a row once N items have held it) on the same draws, held-out in-box loss per row (`reports/probe_accum_2026_10_06.md`) |
| `probes/probe_pres.py` | rows held, no step: Axis 2 as a loss — the student's prediction under the JA caption against the frozen base's under the EN-swapped caption, outside the dilated text box, same x_σ; its gradient on f0's start rows beside the data term's in-box / out-box, σ 0.5–0.95, and its value by σ × length (`reports/probe_pres_2026_10_06.md`) |
| `probes/probe_pres_train.py` | GPU smoke: L_pres trained on f0's hiragana (live; `h16` 16 rows λ 5, `h32` 32 rows λ 10 on its own 7.5 k build via `data`), box-share FM + λ · L_pres at σ 0.8–0.95 against a λ 0 arm (`plain`), held-out band / high-σ evals and the rows' geometry; its `rows.pt` renders on the ruler via `ruler.py --rows_pt` |
| `probes/probe_jl.py` | rows held, no step, **fp32** (bf16's row gradient is at cos 0.3–0.6 to it): `idea2.md`'s Jacobian lens — M_in / M_out = E[JᵀJ] from box / off-box output probes per σ band × family, two fits, f0's start rows; A / B stability, the trace ratio by σ, the cross-fit text-only λ, the trained moves through it (`--weight pair` takes the pairs' size tail off), `--drift` refits at other rows; `split` (σ inside a band) / `check` (the lens against the ruler's reads) on CPU (`reports/probe_jl_2026_10_07.md`) |
| `probes/probe_grad.py` | the scale line's `grad_identity` pass 2 on tiers drawn here (`_archive/reports/grid_64_2026_10_03.md`) |
| `configs/<run>.toml` | the run; keys and their meaning in `reseed/config.py`'s docstring: `rows`, `read`, `seed`, `steps_per_row`; optional `shares`, `upper_shift` (`kana_up`), `stick_from` (`stick_*`), `rows_from` (`stick_rk_*`, `sent_stick`; alone, a plain warm run: `sent_whole`), `drop_tiers`, `band`, `tag_drop` (`stick_rk_fb_jt50`), `ball_on` (`ball_rk*`), `warm` (`sent_ball`), `data_from` (`sent_ball_lr2`, `sent_whole`), `lr` (`sent_ball_lr2`), `pack` (`punct`), `lines` and `row_lr` (`sent_kanji`), `free_residual` (`sent_kanji_f0`), `pres` (`sent_kanji_pres`), `focus` and `held` (`sent_kanji_225`) |
| `reseed/table.py` | **the table**: one row per tier — recipe, share, σ band, glyph px, `px_keep` — and the scene knobs |
| `reseed/recipes.py` | `bubble1` / `bubbleN` / `sent` / `grid` |
| `reseed/pools.py` | rows, scenes (+ the `s1s` pool, mono weighting), the windowed word pool, the dialogue lines (`sent`) |
| `reseed/builder.py` | one pass over the table → `output/cjk_anima_reseed/<run>/data` |

```bash
.venv/bin/python project/cjk_anima_reseed/run.py sent_kanji data --frac 0.03   # sizes: sheet_<tier>.png
.venv/bin/python project/cjk_anima_reseed/run.py sent_kanji data
make daemon-run ARGS="project/cjk_anima_reseed/run.py sent_kanji_f0 train"
make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run --pack punct --arms seed_fixed_1005_stick080@punct,sent_kanji_f0 --label sent_kanji_f0"
.venv/bin/python -m pytest project/cjk_anima_reseed/tests
```

Renderers, scene pools, fonts and the trainer are `../cjk_anima_scale`'s
(`src/`, `cjk_scale.train`). Its `cjk_scale.builder` / `recipes` rebuild the
seed of record and are not imported here (`tests/test_boundary.py`).

## The table

`reseed_anchor --variant fit` (2026-10-03) flattened: `grid_44`'s grid / lone
sizes, `reseed_recap`'s shares and `hp` bands, `fit`'s bubble sizing
(`bubble1_32` at 0.5–0.8 of its bubble, `bubbleN_18` on `s1s` with
`cross_min`), mono scenes at 10 %, columns lettered (`tategaki` +
`vert_forms`), no window opening on `scene.NO_HEAD` or a small kana (`V_SMALL` — anchor's rule let ぁぃぅぇぉゎ through), plain grid captions. A
3 % build lands every tier's median px on `run1003_reseed_anchor_fit`'s.

Whole bubbles (10-03, after that build): the erase spares the bubble outline
(`render_into_scene(keep_outline=True)` — the interior mask holds the
outline, and the erase rectangle painted it away at its sides on 186 of
2 690 scenes), and `pools.whole_bubbles` drops the scenes whose anchor
bubble sits under `BUBBLE_EDGE_MIN` px from the canvas edge (500 of 2 480:
the edge cuts the outline) or whose erase would leave the letters (15).

Left out against the scale builder: the ！ / ？ marks and EN cells (the
`anchor` read: singles lower, words tied), the band law's gate and its groups
(each tier carries its band; `px_keep` is the size cut the gate made), the
rebuild passes (`derive` / `reband`), the per-row draw weights (one
`steps_per_row`), the piece / line tiers.

`grid_64` (10-03): 65 px grids at σ 0.65–0.8, the band read off the
gradient (`_archive/reports/grid_64_2026_10_03.md`), in the table at share 0 and
last — a run's `shares` turns it on (`kana_big`); a share-0 tier is not
drawn, so the runs before it build what they built.

`sent_34` / `sent_22` (10-05, share 0 — a run's `shares` turns them on): a
Manga109 dialogue line (8–10 / 8–14 cells) lettered in 2–3 columns of a
bubble at font px 26–36 / 16–22 (the ink px counts the column gaps; 20–28 /
13–19 read small). The
line: every ellipsis drawn `…` under 4 dots, `……` at 4 or more (T5 reads it
as `...`, no pack row); every other char routed to its own single row, as the
windows are; the `read` strings held by trigram, the dialogue ruler's 5+
glyph strings by 5-gram. The layout: columns cut at Qwen pieces, never on a
`…`; one column that holds the line, a block under 1.5× as tall as wide
(3×3, 3×4, 2×3), a region over 1.3× as wide as tall, a fill under 0.65 and a
sign frame are all re-drawn (`table.SENT_*`).
