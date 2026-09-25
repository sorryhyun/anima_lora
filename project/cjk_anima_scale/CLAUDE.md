# cjk_anima_scale

Production line for the JA vocab pack: a run is its vocabs + what to read;
one loss, one trainer, σ per item from the band law (`plan.md`). `README.md`
= state, `design.md` = the design, `band_experiment_results.md` = the vocab
band law. The line is self-contained: its stage code is its own `src/`, and
nothing here imports, paths into or configures from `../cjk_renderable_anima/`
(the independent research line) — `tests/test_line.py` asserts it.

## Layout

- `scale.py` front door: `scale.py <run> data | train | eval | conflict
  [--submit [--queue]] [--workers N]` (`--workers`: render processes for
  `data`, default cpu − 2); `scale.py windows | runs | ledger`. No other flag.
- `configs/runs/<run>.toml` — **the run, the whole surface**: `vocabs` (a units
  file under `assets/units/`, one vocab per line — or a list of `data.units`
  specs) and `read` (the `native_sent` strings). Everything else is a rule in
  code: the recipe table by kind + volume (`builder.TABLE`, `ITEMS_PER_VOCAB`),
  the trainer constants (`train.py`, each naming the report that set it), the
  seed table (`paths.SEED_TABLE`), the rulers (`eval.py`). Changing a rule is a
  code change with a report beside it.
- `cjk_scale/` the line's code (`windows` = the law, `config` = the run file +
  data pools, `recipes` + `builder` = data, `rows` + `train`, `eval`,
  `conflict`, `bake`, `ledger`; `legacy` reads the archived stage configs for
  `experiments/`).
- `src/` — **the stage packages, vendored 2026-09-25** (top-level `common` /
  `data` / `train` / `eval` / `scenes` / `probe`, `cli`, `stages`,
  `run_stage.py`): only the import closure the line uses. The renderers, readers,
  scoring and sheets are **byte-faithful** to the reads of record — never clean
  them up; the only edits are path plumbing (`common/paths.py` `OUT` +
  `--data_path` / `--arm_path`, `common/prompts.py` `TARGET_PROMPTS`, the
  trimmed `cli/__init__.py` / `stages.py`, `probe/merge_tables.py` = just
  `row_text_map`). `src/` sits at the same depth as the source did, so
  `parents[N]` still lands on the repo root. `paths.bootstrap()` puts it first
  on `sys.path` (its `train` must win over the repo's `train.py`).
- `assets/` — what `src/` reads: `fonts/` (the render set; the font files are
  gitignored, `FONTS.md` says where they come from), `units/` (vocab files),
  `target_prompts.txt` (the `target` ruler).
- `_archive/` — the retired stage surface (`configs/stage*|joint*.toml`, the
  stage-shaped run files, `joint.py`, `boxprobe.py`); gitignored by the
  repo-wide `_archive/` rule, see its README.
- Outputs: `output/cjk_anima_scale/<run>/` — `data/` (items, `vocabs.json`,
  `build.json`, caches), `trained.pt` (the vocabs' rows only), the eval arms
  `ctx/` (`overwrite(seed, trained)`) and `floor/` (`load(seed)`), `sheet.png` +
  `reads.json` (floor and trained on every ruler), `conflict/`. Beside the runs:
  the scene pools `scenes_<tag>/`, the EN reference cache `native_enref/`, the
  seed table `rows_step1_0921_merged/` — seed-side reads of record (`cf_sense_*`,
  the floor of `floor_score.md`) live flat inside it — and the pre-collapse
  stage records `{data,rows}_<stage>_<tag>/` (read-only, `paths.legacy_*`).

## Invariants

- **Terminology (fixed 2026-09-25): `vocab` = the token string** (kinds: single /
  piece / multi), **`idx` = its ext id, `row` = its trained weight.** New and touched
  code uses these names (`row_text` → a `vocab` map, "inventory" → the run's vocabs);
  the on-disk `trained.pt` keys (`ext_ids`, `raw`) stay — every stage reader opens
  them. Eval tables are built from exactly two operations (`cjk_scale/eval.py`):
  `load` (trained.pt → idx → row) and `overwrite` (top rows over base) — the ctx arm
  is `overwrite(seed, trained)`, the floor arm `load(seed)` (the whole seed table,
  never inventory-filtered).

- A run's vocabs train from their seed row (`paths.SEED_TABLE`); every other row a
  caption touches rides frozen at the seed and is stripped from `trained.pt`. A
  vocab the seed lacks starts cold — the only case that prints. `vocabs.json` in
  the data dir is the run's manifest (a flat list; the stages' `words.json` /
  `small.json` are not written).
- `cjk_scale/windows.py` is a **row table with provenance**, not a formula. A new
  read changes a row and its source string; the tests hold the reads.
- Three kinds, by Qwen tokens then glyphs (`windows.unit_kind`): **single** = one
  token, one glyph (あ); **piece** = one token, 2+ glyphs (って — one ext row
  carries the string); **multi** = 2+ tokens (a line, a small-kana digraph あっ =
  host + small row). An item takes its heaviest unit's kind. The research line's
  `t_band_multi` / `_remap_band` "multi" meant ≥ 2 glyphs (piece + multi here) —
  do not reuse that word for it. `px` is √(box area / glyphs), the ink-stat px of
  the reports.
- Orientation is a draw, not a fit fallback: `horizontal_frac` (0.3) of multi-glyph
  scene items / grid cells are left-to-right lines (scene items on the
  `horizontal_scenes` pools only — `sl1w`), marked in the caption
  (`horizontal Japanese text reads as` / `, written horizontally.`); a single glyph
  has none. Not a window axis.
- σ is per item: the builder stamps every item with its band (`windows.window`
  of its kind × px × layout) and the trainer draws σ inside it. A band group's
  gate keeps an item iff the group band is inside its window (or 0.8 of it);
  the ±20 % `px_target` gate reads the tier's **drawn** px, before the band gate
  truncates it. Each band group restarts from the pools' post-build rng state,
  so it draws the item stream its old stage build drew (verified 2026-09-25:
  records and pixels identical, workers 1) — do not reorder rng consumption
  in `recipes.py` / `build_pools`.
- Every launch names the pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`)
  and GPU verbs go through the daemon (`--submit`). The piece tiers read
  `MANGA109S` from the repo's `.env` (`config.phrase_file` calls `load_dotenv`,
  so a daemon child finds it too); the path never enters the repo.
- Tests: `.venv/bin/python -m pytest project/cjk_anima_scale/tests` (line-local,
  not part of the repo suite).
