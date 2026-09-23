# cjk_anima_scale

Production line for the JA vocab pack: one loss, one trainer, a σ-band
schedule whose stages differ only in their data. `README.md` = state,
`design.md` = the design, `band_experiment_results.md` = the vocab band law.
The research surface stays in `../cjk_renderable_anima/` (reports, probes).

## Layout

- `scale.py` front door: `--run <r> --stage <s> --steps data train eval bake [--submit [--queue]]`
  (`--tag <t>` instead of `--run` for a run-less smoke); `scale.py stages | runs | windows | ledger`.
- `cjk_scale/` the code. **Not `src/`**: the probe's `src/` exposes top-level
  `common` / `data` / `train` / `eval`; this package imports them and must never
  shadow them (`tests/test_line.py` pins the direction).
- `configs/stage*.toml` one file per stage — the **band recipe**: `band`, `gate`,
  `warm_from` (the previous stage; `""` = the run's seed table), `[data]` pools,
  `[[data.mix]]` recipes + shares, `[train]` (the trainer surface, warmup as
  `lr_warmup_ratio` of the stage's steps), `[eval]`. **Never which rows.**
- `configs/runs/<run>.toml` — the **run**: `units` / `pieces` / `phrase_file` /
  `n_items`, `seed_table`, `seed`, `[budget]` (steps per row per stage), `[eval]`
  rows. `config.load(stage, run)` merges run over stage; `run_full` is production,
  `run0923_micro` the 24-row chain read.
- Outputs: `output/cjk_anima_scale/` — stage dirs `{data,rows}_scale_<stage>_<tag>/`
  (the probe's layout with a `scale_` prefix), the scene pools `scenes_<tag>/`,
  the EN reference cache `native_enref/` and the seed table
  `rows_step1_0921_merged/`. `paths.bootstrap()` points the probe's
  `common.paths.OUT` at this root **before** any probe module loads, so its
  eval / native / cf_sense and `probe/*.py` open a stage table unchanged. The
  probe line's `output/wake_probe/` keeps symlinks to the shared pools; nothing
  under it is read by this line.

## Invariants

- `--run` names a chain (its name is the tag); `warm_from = "<stage>"` resolves to
  that stage's `trained.pt` under the same tag. No cold row inside a stage except
  a new one.
- `cjk_scale/windows.py` is a **row table with provenance**, not a formula. A new
  read changes a row and its source string; the tests hold the reads.
- Three kinds, by Qwen tokens then glyphs (`windows.unit_kind`): **single** = one
  token, one glyph (あ); **piece** = one token, 2+ glyphs (って — one ext row
  carries the string); **multi** = 2+ tokens (a line, a small-kana digraph あっ =
  host + small row). An item takes its heaviest unit's kind. The probe's
  `t_band_multi` / `_remap_band` "multi" meant ≥ 2 glyphs (piece + multi here) —
  do not reuse that word for it. `px` is √(box area / glyphs), the ink-stat px of
  the reports.
- Orientation is a draw, not a fit fallback: `horizontal_frac` (0.3) of multi-glyph
  scene items / grid cells are left-to-right lines, marked in the caption
  (`horizontal Japanese text reads as` / `, written horizontally.`); a single glyph
  has none. Not a window axis.
- The band gate keeps an item iff the stage band is inside its window (or
  `min_overlap` of it); the ±20 % `px_target` gate reads the recipe's **drawn**
  px, before the band gate truncates it.
- Every launch names the pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`)
  and GPU steps go through the daemon (`--submit`). The phrase stages read
  `MANGA109S` from the repo's `.env` (`config.load` calls `load_dotenv`, so a
  daemon child finds it too); the path never enters the repo.
- Tests: `.venv/bin/python -m pytest project/cjk_anima_scale/tests` (line-local,
  not part of the repo suite).
