# plan — one run = the rows + what to read (2026-09-25, rough)

The line has grown a stage chain, a joint stage, a run file, a units file,
a budget table, an eval block, a context switch and ~25 CLI overrides to
say one thing: **train these rows, on data drawn for them, and read these
strings.** Today that took two config files, a units file, a code fix and
a killed job to get right (`next.md` § 4). This plan collapses it.

## 1. A run is one file

```toml
# runs/run0925_300f.toml
vocabs = "ja_pieces_0925_300.txt"    # what trains — one vocab per line; everything else is frozen at the seed
                                     # (terminology fixed 2026-09-25: vocab = token string, idx = ext id, row = trained weight)
read = ["はい", "おしい", "やったネ", "ちょっと来い", "こんにちは"]   # native_sent + target; the exact word/en rulers are automatic
```

That is the whole surface. No `units` / `pieces` split, no `n_items`, no
`seed_table`, no `context`, no `[budget]`, no `[eval]`, no stage name, no
`--tag`, no `--init_anchor` / `--lr_rows` / `--steps_per_row`. A second
run is a second file.

## 2. Everything else is a rule, not a knob

Fixed in code, changed by editing code with a report beside it — never
per run:

- **σ per item** = the band law (`windows.py`): the item's kind × px picks
  its band. Single 0.7–0.9, piece 0.5–0.7 in scenes / 0.3–0.5 small, sentence
  0.3–0.5. Drawn per item in one run; there is no stage and no chain.
- **Data per row kind** = a fixed recipe table: a single row gets
  `scene_single` + `grid_single`; a piece row gets `scene_piece` (two px
  tiers) + `grid_string` + `scene_short` + `scene_sentence` (corpus lines
  holding it). Which recipes run is decided by which kinds are in `rows`;
  no shares to set, no `missing_source` renormalisation.
- **Volume** = `items_per_row` × rows (one constant, ~70; 300 rows → ~20 k).
- **Trainer** = μ 0, lr 1e-3, batch 4, cosine, warmup 10 %, `grid_box` on,
  90 steps/row. The reads that set each are in `reports/`; a different
  trainer is a code change with a new report, not a flag.
- **Seed** = `rows_step1_0921_merged` (one constant). Rows in the file train
  from their seed value; rows outside it ride frozen at the seed. A row the
  seed lacks starts cold — the only case that prints.
- **Read** = `read` strings on `native_sent` + `target`, plus the automatic
  `word` (18 of the rows) and `en` (24) rulers, plus あ い as the
  frozen-row control. Nothing to name.

## 3. The pipeline

```
scale.py run0925_300f data     # CPU: rows → items (kind-typed recipes, σ band stamped per item)
scale.py run0925_300f train    # daemon: the rows file, everything else frozen
scale.py run0925_300f eval     # daemon: read + word + en + control → one contact sheet
```

`data` writes one dir per run (`output/cjk_anima_scale/<run>/data/`),
`train` one table (`<run>/trained.pt`), `eval` one sheet
(`<run>/sheet.png`) + `reads.json`. No `data_<stage>_<tag>` /
`rows_<stage>_<tag>` layout, no joint merge, no symlinked arm dirs.

## 4. What this deletes

- `configs/stage*.toml`, `configs/joint*.toml`, `cjk_scale/joint.py`, the
  `[budget]` / `[eval]` / `context` / `pieces` / `n_items` run keys, the
  `warm_from` chain and `--tag` arms, `train.py::noisy_by_band`'s
  per-stage envelope (the band is per item, full stop).
- `scale.py` overrides: `--init_anchor --lr_rows --steps_per_row
  --train_steps --n_items --seed --batch --compile --eval_only --seed_only
  --conflict_*` (the conflict probe becomes `scale.py <run> conflict`,
  same rows, no flags).
- `inventory_ext`'s `ev_ext` path — the rows file is the inventory, the
  tokenizer maps it, done.
- `recipes.missing_source` share renormalisation — the recipe table is by
  kind, so a recipe with no source never enters.

## 5. What it keeps

`windows.py` (the law, with provenance), `recipes.py` drawers, `loss.py`
(box share, grid box), `rows.py` (the table, frozen rows), `eval.py` /
`bake.py`, the scene pools, the seed table, `reports/`.

## 6. Order

1. Let `run0925_300f` (job `20260925-144054-bcf21e`) finish and read it on
   the current code — do not refactor under a running job. **Done** (read in
   `reports/piece_2026_09_25.md` + `floor_score.md`).
2. New `RunConfig` = `{vocabs, read}`; recipe table by kind; per-item band
   at build time; one output dir per run. Old configs move to `_archive/`.
   **Done 2026-09-25** — `scale.py <run> data|train|eval|conflict`,
   `builder.TABLE`, `<run>/{data,trained.pt,sheet.png,reads.json}`; the line's
   stage code vendored into `src/` (no probe-line reference left).
3. Re-run `run0925_300f` as the first run of the new shape; its numbers
   must match the old run's (same rows, same data seed) — that is the
   refactor's test.
4. A singles run (`rows = kana + kanji:200`) is the same file shape; the
   kanji band question (`next.md`) is then just another run.

Open: whether two px tiers of `scene_piece` (36 px and 19 px) stay one
recipe with a drawn px or two rows in the table — decide from
`reports/grid_box_2026_09_25.md` § 2 (the small tier priced ≥ the large).
