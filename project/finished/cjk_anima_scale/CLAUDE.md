# cjk_anima_scale

**Finished 2026-10-09** (`README.md` § Finished): frozen, runnable by path;
live work is in `../../cjk_anima_reseed/`.

Production line for the JA vocab pack: a run is its vocabs + what to read;
one loss, one trainer, σ per item from the band law. `README.md`
= state, `proposal_seed_synthesis.md` = the open question, `band_experiment_results.md` = the
vocab band law; the finished plans (`plan.md` = the collapse spec the code
implements, `plan_retrain.md`, …) are in `_archive/`. The line is self-contained: its stage code is its own `src/`, and
nothing here imports, paths into or configures from `../cjk_renderable_anima/`
(the independent research line) — `tests/test_line.py` asserts it.

## Layout

- `scale.py` front door: `scale.py <run> data | train | eval | conflict
  [--submit [--queue]] [--workers N]` (`--workers`: render processes for
  `data`, default cpu − 2); `scale.py <out> merge <run> <run>…` (CPU: disjoint
  runs from the seed → one merged rows file); `scale.py windows | runs |
  ledger`. No other flag.
- `configs/runs/<run>.toml` — **the run, the whole live surface**: `vocabs` (a
  vocabs file under `assets/vocabs/`, one vocab per line — or a list of
  `data.vocabs` specs), `read` (the `native_sent` strings) and optional
  `context` (a run whose merged rows replace the seed rows for this run —
  warm-from, frozen context, merge base, and the singles its windows may
  carry; `retrain_kanji_b*` chain on it; a rows dir with no run config, as
  b5's `seed_fixed_1005_stick080`, ends the chain and names its singles in
  its `data/vocabs.json`), `phrases` (the dialogue lines the windows cut;
  default `config.PHRASE_FILE`) and `held` (a corpus file of strings held
  out of the windows as `read` is, never read by eval — both `$MANGA109S`
  paths, so no corpus text enters the repo). Everything else
  is a rule in code: the recipe table by kind + volume (`builder.TABLE`,
  `ITEMS_PER_VOCAB`), the trainer constants (`train.py`, each naming the
  report that set it), the seed rows (`paths.SEED_ROWS`), the rulers
  (`eval.py`). Changing a rule is a code change with a report beside it.
  `run0923_micro` (the pre-collapse stage shape `load_run` refuses) and
  `run0925_300f` stay in `configs/runs/` because the tests read them.
- `configs/data_build/` + `configs/train/` — the pre-collapse stage / joint
  configs, **split** (2026-09-25) into their data-generation half (band +
  pools + `[[mix]]` recipes) and their trainer half (`warm_from` + the train
  values), `[eval]` blocks dropped. Records, restored so they stay tracked;
  nothing reads them (their reader `cjk_scale/legacy.py`, for the archived
  influence experiments, was removed 2026-09-30).
- `cjk_scale/` the line's code (`windows` = the law, `config` = the run file +
  data pools, `recipes` + `builder` = data, `rows` + `train`, `eval`,
  `conflict`, `ledger`, `budget` = steps / items per vocab by kind × glyphs ×
  warm, `merge`; `reads` = the experiments' per-render scoring and McNemar
  pairing, `paths.load_experiment` = one experiment importing another's
  `run_exp.py`). Baking a run's rows into a pack is one command, not a
  module:
  `.venv/bin/python scripts/toolkits/bake_vocab_pack.py output/cjk_anima_scale/<run> --out models/vocab_packs/anima_cjk_vocab_pack_<run>` (`--glyph_route` ships routing on, as a routed run's rows need).
  **`builder` / `recipes` are frozen** (2026-10-03): they rebuild the seed of
  record; new data recipes went to `../../cjk_anima_reseed/reseed/`, which
  since 2026-10-09 carries its own copy of `src/` and the trainer.
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
  `--arm_path`, no tag fallback; `common/prompts.py` `TARGET_PROMPTS`), `data/inventory.py`'s opt-in
  `qwen_pieces(char_rows=True)` (byte-split glyphs → their `char` rows, and the symbol block's Qwen rows (`sym`, 10-05); every
  `cjk_scale` lookup passes it, the records ran without it),
  `common/render/scene.py`'s opt-in `render_into_scene(tategaki=True)`
  (columns top-aligned; a turned ー 〜 … placed by its ink on the column
  axis — off, it sits up to 0.2 em left of it, as every data dir of record
  was drawn) and `render_into_scene(vert_forms=True)` (a column's ー 〜 …,
  small kana, 、。 and brackets as the font's `vert` alternates through
  libraqm — off, a small kana in a column is the horizontal glyph;
  `experiments/reseed_anchor`), `data/grid.py`'s opt-in `render_grid(bubble_fit=(lo, hi), cell_jitter=x)`
  (a cell's ellipse sized to its ink instead of the cell; the ink at the cell's
  centre ± a share of it instead of anywhere in it — `experiments/grid_small`)
  and `render_grid(line_cells=…)` (cells always a line, unmarked: an EN word),
  `train/stage.py`'s latent cache name hashed past 200 chars (a native-size
  dir has one shape per image), the
  trimmed `cli/` / `stages.py`, `probe/merge_tables.py` = just `row_text_map`
  + `row_texts`, and the prune. `src/` sits at the same depth as the source
  did, so `parents[N]` still lands on the repo root. `paths.bootstrap()` puts
  it first on `sys.path` (its `train` must win over the repo's `train.py`).
- `assets/` — what `src/` reads: `fonts/` (the render set; the font files are
  gitignored, `FONTS.md` says where they come from), `vocabs/` (vocab files),
  `target_prompts.txt` (the `target` ruler).
- `_archive/` — gitignored by the repo-wide `_archive/` rule, see its README:
  the retired stage code (`joint.py`, `boxprobe.py`), and (2026-09-28) the
  pre-retrain docs, reports, experiments, run files and vocab lists the
  retrain superseded. Archived experiments no longer run from there.
- Outputs: `output/cjk_anima_scale/<run>/` — `data/` (items, `vocabs.json`,
  `build.json`, caches), `trained.pt` (**the whole merged rows**: the seed's
  rows, rescaled into the run's `row_scale`, with the run's vocabs' rows on
  top — `seed_merged` marks the format), the trained side's ruler outputs
  at the run root (no `ctx/`, no `floor/` since 2026-09-26), `sheet.png` + `reads.json` (floor and trained on every ruler),
  `conflict/`. Beside the runs: the scene pools `scenes_<tag>/`, the EN
  reference cache `native_enref/`, the experiments' row arms
  `experiments/<arm>/` (`tl_*`, `tp_*`), the seed rows `rows_step1_0921_merged/` —
  seed-side reads of record (`cf_sense_*`, the floor of `floor_score.md`)
  live flat inside it, and it is **every run's floor arm**: one read cache,
  a run renders only the keys it lacks (`eval.ensure_floor`) — and the pre-collapse stage records
  `{data,rows}_<stage>_<tag>/` (read-only).

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

- A run's vocabs train from their seed row (`paths.SEED_ROWS`, or the run's
  `context` rows); every other row a caption touches rides frozen at the seed; `trained.pt` carries them all (the
  merged save). A vocab the seed lacks starts cold — the only case that prints.
  **Singles always start cold** (`budget.COLD_KINDS`, retrain_experiments): their
  in-word groups draw routed windows, and a data dir built with them
  (`build.json` `glyph_route`) is trained with `ANIMA_VOCAB_GLYPH_ROUTE=1`
  set in-process — never in the submit shell; `eval` reads such a run the
  same way, against the routed floor cache `<seed rows>/routed/`
  (`eval.floor_arm_dir`), never the unrouted cache of record. `vocabs.json` in
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
- **Item pools are tiers named `<form>_<px>`** (2026-10-02): `lone` / `grid` /
  `bubble1` / `bubbleN` (+ `piece_*` / `line_*`) and the median ink px the
  tier was built at on the kana run — `bubbleN_34`, `grid_82`
  (`builder.TIER_PX`; the table is README § Item pools). The record key is
  `tier`. Data dirs, reports and results before that date say
  `b0507/scene_window` (band group / recipe); read a record's tier with
  `builder.tier_of`, never by its `group`. A group — the tiers of one kind
  drawn at one band from one rng restart — has no name.
- σ is per item: the builder stamps every item with its band (`windows.window`
  of its kind × px × layout) and the trainer draws σ inside it. A group's
  gate keeps an item iff the group band is inside its window (or 0.8 of it)
  — a tier with `gate = "group"` skips it and takes the group's band (a lone
  glyph's px is its own ink box, so the gate drops ー and the small kana
  from a small lone tier: `experiments/grid_lone`);
  the ±20 % `px_target` gate reads the tier's **drawn** px, before the band gate
  truncates it. Each group restarts from the pools' post-build rng state,
  so it draws the item stream its old stage build drew (verified 2026-09-25:
  records and pixels identical, workers 1) — do not reorder rng consumption
  in `recipes.py` / `build_pools`.
- Every launch names the pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`)
  and GPU verbs go through the daemon (`--submit`). The piece tiers read
  `MANGA109S` from the repo's `.env` (`config.phrase_file` calls `load_dotenv`,
  so a daemon child finds it too); the path never enters the repo.
- Tests: `.venv/bin/python -m pytest project/finished/cjk_anima_scale/tests` (line-local,
  not part of the repo suite).
