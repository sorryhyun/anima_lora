# cjk_anima_reseed

The JA vocab pack's rows re-seeded cold, then trained warm on dialogue lines;
shipped `seed_1008` = jp_v1. Where the line stands — recipe of record, the
ruler's table, what is settled, what is open: **`status.md`**. What an arm is
judged on: `criteria.md`. Next: `proposal_refactor.md` (in progress), then
`proposal_jamo.md`. The live reads: `reports/` (the ruler, sent_kanji,
probe_pres, sent_kanji_pres, kozh16).

`_archive/` (force-tracked, skipped by search): the docs folded into
`status.md` (progress, plan, idea / idea2 / idea3, structure_candidate,
task_report), the banner-era and closed probes with their reports, the
configs of the arms that lost, `stick_fit.py`, and the motivation docs. It
runs from the freeze-point commit named in `status.md`, not from this tree.

## Code

| file | what |
|---|---|
| `run.py` | front door: `run.py <run> data [--frac f] \| train` |
| `ruler.py` | the dialogue ruler's CLI (`criteria.md`; the code is `src/eval/ruler/`: `build` / `arms` / `render` / `score` / `stats` / `read`): `build` (CPU) draws 96 bubble-dialogue strings from the training set's captions, each with its own image's prompt and an EN reference line → `output/cjk_anima_reseed/ruler/ruler.json`; `run [--arms a,b] [--label l]` (GPU) renders what is missing — the floor (EN refs, retrain_kana, seed_retrain_0930) once — and reads every arm against it → `results/<ts>-ruler-<label>/` (`reports/ruler_2026_10_05.md`) |
| `transplant.py` | CPU: a focus run's rows onto another run's (`FROM` / `ONTO` / `NAME` at its top; 10-08: `sent_kanji_225`'s 225 onto `sent_kanji_pres` → `seed_1008`). `stick` reads the focus rows' mean against ONTO's kanji stick and both runs' Δstick; `write` → `output/cjk_anima_reseed/<NAME>/trained.pt` and the baked pack |
| `punct_pack.py` | CPU: the punct base pack (10-05, the green leaf's fix) → `models/vocab_packs/anima_cjk_vocab_pack_punct`: the raw pack plus wider folds, dot runs and a `…` row (ext 69 558, at T5's `...`); the rules in its docstring |
| `probes/kozh_geometry.py` | CPU: a `lang` run's KO / ZH rows against its seed's kana / kanji balls — sticks, row norms, cos and stick components, the groups' spikes in either ball's top-40 beside 8 held-out seed rows, nearest seed rows (`reports/kozh16_2026_10_09.md`) |
| `probes/kozh_render.py` | GPU: a `lang` run's rows drawn against its seed through the ruler's `Renderer` at 512², seed 0 — each row alone in a bubble, a few words in a bubble / on a sign / plain, captioned in the row's language → `results/<ts>-<run>-render/sheet.png` |
| `configs/<run>.toml` | the run; keys and their meaning in `reseed/config.py`'s docstring. Kept: `punct` (the mark rows → seed_fixed_1005), `sent_kanji` (the data of f0 / pres), `sent_kanji_pres` (the rows of record), `sent_kanji_225` (the 225 of record), `kozh16` (KO / ZH rows on seed `1008`) |
| `reseed/table.py` | **the table**: one row per tier — recipe, share, σ band, glyph px, `px_keep` — and the scene knobs |
| `reseed/recipes.py` | `bubble1` / `bubbleN` / `sent` / `grid` |
| `reseed/pools.py` | rows, scenes (+ the `s1s` pool, mono weighting), the windowed word pool, the dialogue lines (`sent`), the KO / ZH faces and captions (`lang`) |
| `reseed/builder.py` | one pass over the table → `output/cjk_anima_reseed/<run>/data` |
| `reseed/trainer.py` | the trainer (ported from scale's `cjk_scale/train.py`, numerics unchanged): the run's rows cold or warm on `context`, every other row frozen there, box-share FM, σ per item in its band, optional L_pres → `<run>/trained.pt` (the whole merged rows), `train_log.json`, `train_record.json`; `rows.py` (the `ExtDelta` rows) and `loss.py` (box-share FM, L_pres) beside it |
| `reseed/__init__.py` | the paths (`HOME`, `OUT`, `SCALE_OUT`, the seed rows), the pack digests, `bootstrap()` |
| `src/` | vendored from scale's `src/` under its top-level names (`common`, `data`, `train`, `eval`): renderers, scene pools, Qwen inventory, readers, TE / latent caches; byte-faithful to the reads of record — no cleanup. `eval/ruler/` is this line's own: the dialogue ruler |
| `assets/fonts/` | the JA faces + `kozh/` (KO / ZH; FONTS.md, licences; the binaries are gitignored — copy them) |

```bash
.venv/bin/python project/cjk_anima_reseed/run.py sent_kanji data --frac 0.03   # sizes: sheet_<tier>.png
.venv/bin/python project/cjk_anima_reseed/run.py sent_kanji data
make daemon-run ARGS="project/cjk_anima_reseed/run.py sent_kanji_pres train"
make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run --pack punct --arms seed_fixed_1005_stick080@punct,sent_kanji_pres --label sent_kanji_pres"
.venv/bin/python -m pytest project/cjk_anima_reseed/tests
```

The line stands alone (`tests/test_boundary.py`): no live file imports the
scale line's `cjk_scale` (`../finished/cjk_anima_scale/`). It reads its outputs
(`output/cjk_anima_scale`: the seed rows, the arms of record, the scene pools,
the EN refs) and appends to its two scene caches.

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
last — a run's `shares` turns it on (`_archive/configs/kana_big.toml`); a share-0 tier is not
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
