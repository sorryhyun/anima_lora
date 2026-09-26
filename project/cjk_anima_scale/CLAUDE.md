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
- `configs/runs/<run>.toml` — **the run, the whole live surface**: `vocabs` (a
  vocabs file under `assets/vocabs/`, one vocab per line — or a list of
  `data.vocabs` specs) and `read` (the `native_sent` strings). Everything else
  is a rule in code: the recipe table by kind + volume (`builder.TABLE`,
  `ITEMS_PER_VOCAB`), the trainer constants (`train.py`, each naming the
  report that set it), the seed rows (`paths.SEED_ROWS`), the rulers
  (`eval.py`). Changing a rule is a code change with a report beside it.
  The pre-collapse stage-shaped run files stay in `configs/runs/` as records
  (`scale.py runs` lists them as such; `load_run` refuses them).
- `configs/data_build/` + `configs/train/` — the pre-collapse stage / joint
  configs, **split** (2026-09-25) into their data-generation half (band +
  pools + `[[mix]]` recipes) and their trainer half (`warm_from` + the train
  values), `[eval]` blocks dropped. Records, restored so they stay tracked;
  read only by `cjk_scale/legacy.py` for `experiments/`.
- `cjk_scale/` the line's code (`windows` = the law, `config` = the run file +
  data pools, `recipes` + `builder` = data, `rows` + `train`, `eval`,
  `conflict`, `ledger`; `legacy` reads the split pre-collapse configs for
  `experiments/`). Baking a run's rows into a pack is one command, not a
  module:
  `.venv/bin/python scripts/toolkits/bake_vocab_pack.py output/cjk_anima_scale/<run> --out models/vocab_packs/anima_cjk_vocab_pack_<run>`.
- `src/` — **the stage packages, vendored 2026-09-25** (top-level `common` /
  `data` / `train` / `eval` / `scenes` / `probe`, `cli`, `stages`,
  `run_stage.py`) and **pruned the same day to the code the line runs**: the
  six `stages.STAGES` entries plus what `cjk_scale/` imports. Dead stages
  (`stage_data`, `stage_train`, `enref`, `native_rescore`, `classify`) and
  their levers are gone — `eval/classify.py` (its `_en_word_pairs` lives in
  `eval/cf_sense.py`), `train/trainables.py`, `train/encoder.py`,
  `data/pair.py`; `data/vocabs.py` is the vocab-spec parser (was `units.py`).
  The renderers, readers, scoring and sheets are **byte-faithful** to the
  reads of record — never clean them up; the only edits are path plumbing
  (`common/paths.py` `OUT` + `data_dir` / `arm_dir` = `--data_path` /
  `--arm_path`, no tag fallback; `common/prompts.py` `TARGET_PROMPTS`), the
  trimmed `cli/` / `stages.py`, `probe/merge_tables.py` = just `row_text_map`
  + `row_texts`, and the prune. `src/` sits at the same depth as the source
  did, so `parents[N]` still lands on the repo root. `paths.bootstrap()` puts
  it first on `sys.path` (its `train` must win over the repo's `train.py`).
- `assets/` — what `src/` reads: `fonts/` (the render set; the font files are
  gitignored, `FONTS.md` says where they come from), `vocabs/` (vocab files),
  `target_prompts.txt` (the `target` ruler).
- `_archive/` — the retired stage code (`joint.py`, `boxprobe.py`); gitignored
  by the repo-wide `_archive/` rule, see its README. Its configs moved back to
  `configs/`.
- Outputs: `output/cjk_anima_scale/<run>/` — `data/` (items, `vocabs.json`,
  `build.json`, caches), `trained.pt` (**the whole merged rows**: the seed's
  rows, rescaled into the run's `row_scale`, with the run's vocabs' rows on
  top — `seed_merged` marks the format), the trained side's ruler outputs
  at the run root (no `ctx/`, no `floor/` since 2026-09-26), `sheet.png` + `reads.json` (floor and trained on every ruler),
  `conflict/`. Beside the runs: the scene pools `scenes_<tag>/`, the EN
  reference cache `native_enref/`, the seed rows `rows_step1_0921_merged/` —
  seed-side reads of record (`cf_sense_*`, the floor of `floor_score.md`)
  live flat inside it, and it is **every run's floor arm**: one read cache,
  a run renders only the keys it lacks (`eval.ensure_floor`) — and the pre-collapse stage records
  `{data,rows}_<stage>_<tag>/` (read-only, `paths.legacy_*`).

## Invariants

- **Terminology (fixed 2026-09-25): `vocab` = the token string** (kinds: single /
  piece / multi), **`idx` = its ext id, `row` = its trained weight.** New and touched
  code uses these names (`row_text` → a `vocab` map, "inventory" → the run's vocabs);
  the on-disk `trained.pt` keys (`ext_ids`, `raw`) stay — every stage reader opens
  them — and so does the item records' `units` key (`train.jsonl`: the item's
  vocabs; `_ink_stats` and the builder's counts read it). **"table" and "ctx"
  are retired terms** (2026-09-25): `trained.pt` is *the rows*, and the merge
  that the old `ctx/` sidecar performed (`overwrite(seed, trained)`, with the
  row_scale rescale) happens once, at save (`rows.Rows.state_dict`) — the
  trained eval arm is the run dir itself, the floor arm the seed rows' dir
  (the whole seed rows, never inventory-filtered; `paths.floor_dir()`). Eval refuses
  a pre-merge vocabs-only `trained.pt` (no `seed_merged` key) — retrain.

- A run's vocabs train from their seed row (`paths.SEED_ROWS`); every other row a
  caption touches rides frozen at the seed; `trained.pt` carries them all (the
  merged save). A vocab the seed lacks starts cold — the only case that prints. `vocabs.json` in
  the data dir is the run's manifest (a flat list; the stages' `words.json` /
  `small.json` are not written).
- `cjk_scale/windows.py` is a **band law written as rows with provenance**, not a formula. A new
  read changes a row and its source string; the tests hold the reads.
- Three kinds, by Qwen tokens then glyphs (`windows.vocab_kind`): **single** = one
  token, one glyph (あ); **piece** = one token, 2+ glyphs (って — one ext row
  carries the string); **multi** = 2+ tokens (a line, a small-kana digraph あっ =
  host + small row). An item takes its heaviest vocab's kind. The research line's
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
