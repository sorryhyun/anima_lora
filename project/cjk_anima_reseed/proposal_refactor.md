# proposal_refactor — reseed torn down and rebuilt, scale archived (2026-10-09)

**Status: proposal.** Nothing has been moved yet. The user (10-09): before the
next experiment, prune `cjk_anima_reseed` (too many stale experiments and idea
docs) and archive `cjk_anima_scale`. Work happens on `cjk-reseed`; `main` holds a
stale copy of both lines.

The first line of work the rebuilt tree has to host is
[`proposal_jamo.md`](proposal_jamo.md) (Hangul rows from jamo factors; being
written separately). It needs the trainer (a new `factor = "jamo"` mode beside
the update modes), the renderers with the `kozh/` faces, the lone / bubble1 /
grid tiers, the `lang` and `held` keys, the ruler's readers and a row-geometry
read. Everything below keeps those.

## 1. Current state

### 1.1 reseed (`cjk-reseed` @ `8d2cd03c`)

86 tracked files:

| where | count | notes |
|---|---|---|
| root docs | 10 | README, progress, criteria, plan, idea, idea2, idea3, structure_candidate, release_plan, task_report (1 433 lines); `proposal_jamo.md` untracked, in progress |
| root scripts | 5 | run.py, ruler.py, stick_fit.py, transplant.py, punct_pack.py (2 262 lines) |
| `reseed/` | 6 | config, table, recipes, pools, builder, `__init__` (2 315 lines) |
| `probes/` | 13 | 7 071 lines |
| `reports/` | 12 | |
| `configs/` | 7 | kozh16, punct, sent_kanji, sent_kanji_f0, sent_kanji_pres, sent_kanji_225, sent_whole |
| `tests/` | 1 | `test_boundary.py`, 2 tests |
| `_archive/` | 34 | 16 configs, 15 reports, 3 md. Under the repo-wide `_archive/` ignore rule but force-tracked, so ripgrep already skips it |

Untracked: `results/` (44 run dirs, 1.2 GB, gitignored), `models/pe/PE-Spatial-B16-512.pt`
(330 MB, gitignored; written 10-05 19:58, probably a download from
`anime_tools.vision.pe` run with this dir as the cwd).

Line state at writing: `seed_1008` (= jp_v1, shipped) is the seed of record.
`kozh16` has data, and **its `trained.pt` landed at 14:10 today**. A daemon job
rendering it (`kozh_render`) was running at 14:12. No `reports/kozh16_*.md`
exists yet, though README names one.

Baseline (10-09, run once each): `pytest project/cjk_anima_reseed/tests` 2
passed; `pytest project/cjk_anima_scale/tests` 86 passed.

### 1.2 scale

168 tracked files: `cjk_scale/` 15 modules, `src/` 38 files, `experiments/` 30
dirs, `configs/` 20, `assets/` 34 (fonts' FONTS.md + licences, vocabs,
`glyph_ink.json`, `target_prompts.txt`), `reports/` 12, 12 root docs, `tests/`
3, `runs/` 2, `scale.py`. Untracked: `_archive/` (3.1 MB), font binaries (135 MB,
including `kozh/`'s 12 faces and `_dl/`).

**Branches.** `main` has not touched `project/cjk_anima_scale` since the merge
base `e8e78686` (`git diff cjk-reseed...main -- project/` shows only
`gui_curation_handoff/plan.md`). `cjk-reseed` changes 27 scale files (+1 940 /
−51): the trainer's reseed hooks (`pres`, `steps`, `free_residual`,
`row_step_scale`), `loss.pres_loss` / `en_caption`, `retrain_kanji_b5` (vocabs,
run config, `glyph_ink.json` +225), the kozh FONTS.md section and 12 OFL
licences, and `data/stage.py` / `inventory.py` edits. The archive therefore
happens on the branch, and main's copy is superseded when the branch merges.

### 1.3 What reseed takes from scale

Two mechanisms. `reseed/__init__.py::bootstrap()` puts `project/cjk_anima_scale`
on `sys.path` and calls `cjk_scale.paths.bootstrap()`, which puts scale's `src/`
first, so the top-level names `common` / `data` / `train` / `eval` / `probe`
resolve there. Separately, `cjk_scale.paths.OUT` (`output/cjk_anima_scale`) is
read for seed rows, scene pools and floor arms.

Static closure over the live entry points (AST walk, function-level imports
included; `load_experiment("…")` calls followed):

| entry | scale modules reached |
|---|---|
| `reseed/config.py` | `cjk_scale.{config, paths}`: `RunConfig`, `phrase_file`, `paths.OUT`, `SEED_ROWS`, `SEED_ROWS_0921` |
| `reseed/pools.py` | `cjk_scale.config.{DATA, dataset_ja_lines}`, `cjk_scale.windows.glyph_count`, `paths.OUT`; `common.{bubble, models, render.flat, render.scene}`, `data.{inventory, stage, synth}` |
| `reseed/recipes.py`, `builder.py` | `common.{prompts, readers, render.*}`, `data.{grid, inventory, stage, synth}` |
| `run.py train` | `cjk_scale.train` → `rows`, `loss`, `config`, `paths`, **`budget`** → **`builder`** → **`recipes`**, `merge`, `eval.cf_sense` → `probe.merge_tables`; `train.stage`, `common.{models, hooks, shapes}` |
| `run.py read` | `paths.load_experiment("grid_lone")` → `experiments/{grid_lone, grid_small, kana_reband, retrain_read, p2_route, stage_b}`, `cjk_scale.{builder, recipes, eval, reads}`, `src/{cli, stages, eval.*}` |
| `ruler.py` | `cjk_scale.config.is_ja_text`, `paths.OUT`; `common.{hooks, models, readers, text}`, `eval.enref.EnRef`; `gs_rkstick` → `probes/probe_split` → `cjk_scale.reads`, `experiments/sigma_split` |
| `transplant.py` | `cjk_scale.train.vocab_idx`, `data.inventory.qwen_pieces` |
| probes (live and dead) | `cjk_scale.{train, rows, loss}`, `common.*`, `data.*`, `train.stage`; `probe_split` / `stick_fit` add `cjk_scale.reads`, `eval.enref`, `probe.merge_tables`, `experiments/{row_geometry, sigma_split}` |

Findings:

- **`tests/test_boundary.py` checks only direct imports.** At runtime the
  trainer reaches the frozen builder: `train.plan()` always calls
  `budget.mix_factor`, which imports `builder.TABLE` → `recipes`. Reseed
  overrides every number that path produces (it always passes `steps_per_row`
  or `steps`, and `cold`), so the budget only fills `train_record.json`'s
  `budget_factor` / `mix_factor`. `run.py read` reaches six scale experiments.
- **No live reseed code uses the scale experiments.**
  `experiments/stick_scale/geometry.py` is not imported anywhere in reseed;
  `stick_fit.py` loads `experiments/row_geometry`; `run.py read` loads
  `grid_lone`; `probe_split` loads `sigma_split`. All of these are banner-era
  code.
- **Fonts.** `common/paths.FONT_DIR` = `<src>/../assets/fonts`; `pools.lang_fonts`
  globs its `kozh/`. The binaries are gitignored, so `git mv` will not carry them.
- **Outputs.** Reseed reads `output/cjk_anima_scale/{seed_retrain_0930,
  rows_step1_0921_merged, retrain_kana, seed_fixed_1005_stick080,
  retrain_kanji_b5_stick080, scenes_*, native_enref,
  experiments/scene_{bubble_check,colorful}.json}`. These are run outputs and
  stay where they are.

The minimal set the live reseed needs (≈ 4.7 k lines):

- `src/common/` whole (13 files): `__init__`, paths, text, prompts, shapes,
  models, hooks, readers, bubble, `render/{__init__, flat, ink, scene}`.
- `src/data/{__init__, inventory, stage, synth, grid, vocabs}`.
- `src/train/{__init__, stage}`.
- `src/eval/enref.py` plus a docstring-only `eval/__init__`.
- `cjk_scale/{train, rows, loss}.py`.
- From `cjk_scale/config.py`: `DATA`, `PHRASE_FILE`, `_expand`, `is_ja_text`,
  `dataset_ja_lines`.
- From `cjk_scale/paths.py`: `OUT`, `SEED_ROWS`, `RAW_PACK_SHA`, `PUNCT_PACK_SHA`,
  and `bootstrap`'s `sys.path` order.
- `windows.glyph_count`, 2 lines. It differs from `loss.glyph_count`, which has
  `max(1, …)`, so the pools keep their own copy.
- `assets/fonts/`: FONTS.md, `licenses/`, `.gitignore`, the binaries and `kozh/`.

**Dead weight for reseed** (stays with scale as record): `cjk_scale/{builder,
recipes, budget, merge, eval, conflict, ledger, reads, windows (bar
glyph_count)}`, `scale.py`, `src/{cli, scenes, probe, stages.py, run_stage.py,
eval/{stage, native, cf_sense, summary}}`, `experiments/` (30), `configs/`,
`runs/`, `assets/{vocabs, glyph_ink.json, target_prompts.txt}`, `fonts/_dl/`,
every doc and report.

### 1.4 Outside readers of scale

| path | reads | action |
|---|---|---|
| `tests/test_bake_vocab_pack.py:28` | `sys.path` → `project/cjk_anima_scale/src`, imports `common.hooks.ExtDelta` (`test_bake_equals_hook`; repo suite) | repoint to reseed's `src/` |
| `library/anima/ext_vocab.py:48,54` | docstring cites `retrain_experiments.md`, `plan_retrain.md` (already in scale's `_archive/`) | path → `project/finished/cjk_anima_scale/…` |
| `scripts/toolkits/bake_vocab_pack.py:5` | docstring | same |
| `docs/methods/cjk_vocab_pack.md:130–147`, `docs/experimental/anima_cjk_vocab_ext.md` (7 places) | doc links to scale files / modules | same; module names `cjk_scale/{windows,budget,loss,train}.py` stay valid under finished/ |
| `.env.example:36` | comment | "the reseed line's dialogue line files" |
| `project/README.md`, `project/finished/README.md`, `finished/{cjk_aware_anima, cjk_aware_anima_dit, cjk_renderable_anima}/README.md` | links `../cjk_anima_scale/` | relink; scale off the active list, reseed on it (reseed is not listed there today) |
| `.claude/agents/archive-explorer.md:57` | "production continued in the live `project/cjk_anima_scale/`" | user-owned config: the user edits it |
| `~/ComfyUI-Anima_lora-Adapter/_vendor/library/anima/ext_vocab.py` | the vendored docstring | `make vendor-sync` at the next node publish |

Nothing in `bench/` (incl. `bench/cjk_adapter`, `bench/cjk_distill`),
`scripts/` (bar the docstring), `scripts/tasks/`, `tasks.py`, `custom_nodes/`,
`gui/`, `anima_lora/` or `library/` (bar the docstring) imports scale or
reseed. `~/manga109s/derived/*.py` does not either. `punct_pack.py` imports
`library` only.

## 2. Target layout

```
project/cjk_anima_reseed/
  README.md            home: code table, run lines, the table (rewritten, shorter)
  status.md            where the line stands (new, § 2.1)
  criteria.md          the ruler's definition (kept)
  release_plan.md      until anima-jp-extended is public, then _archive/
  proposal_jamo.md     next line of work
  run.py               data | train  (the `read` verb goes)
  ruler.py             the dialogue ruler (gs_rkstick / rand_turn derived arms go)
  transplant.py        how seed_1008 = jp_v1 was made
  punct_pack.py        how the punct base pack was made (every live config sits on it)
  reseed/              __init__ (paths: HOME, OUT, SCALE_OUT, seed rows, pack shas, bootstrap)
                       config  table  recipes  pools  builder
                       trainer.py  rows.py  loss.py      ← cjk_scale/{train,rows,loss}.py
  src/                 ← scale's src subset, byte-identical except path plumbing
    common/  data/  train/  eval/{__init__,enref}.py
  assets/fonts/        FONTS.md  licenses/  .gitignore  (+ binaries, kozh/; untracked)
  configs/             punct  sent_kanji  sent_kanji_pres  sent_kanji_225  kozh16
  probes/              kozh_geometry  kozh_render
  reports/             ruler_2026_10_05  sent_kanji_2026_10_06  probe_pres_2026_10_06
                       sent_kanji_pres_2026_10_07
  tests/               test_boundary.py (rewritten)  test_config.py  test_src.py
  _archive/            configs/ reports/ probes/ docs  stick_fit.py  (git add -f)
  results/             unchanged, gitignored
project/finished/cjk_anima_scale/   the whole scale tree, + .ignore
```

`src/` keeps scale's top-level names, and `bootstrap()` keeps putting it first
(its `train` must win over the repo's `train.py`). It sits at the same depth as
scale's did, so `parents[4]` still lands on the repo root. Path-plumbing edits:

- `common/paths.OUT` stays `output/cjk_anima_scale`, because the scene pools
  and the EN-ref cache live there. It is named as read-only.
- `FONT_DIR` resolves to reseed's `assets/fonts`.
- `prompts.TARGET_PROMPTS` is dropped (its only reader, `eval/native`, is not
  vendored).

The scale CLAUDE.md rule carries over: renderers, readers and scoring are
byte-faithful to the reads of record, so no cleanup.

The trainer port edits:

- `plan()` drops `budget`. The caller always gives the steps and `cold`, and
  the record loses `budget_factor` / `mix_factor`.
- `rc: RunConfig` becomes the reseed `Run`. The trainer reads its name, path
  and rows plus `context`.
- `data_dir` / `run_dir` defaults go; `data` and `out` are required.

Numerics stay unchanged; § 4 step 3 checks this.

### 2.1 `status.md` — what survives

One short doc. It replaces progress.md and absorbs plan.md, idea*.md,
structure_candidate.md and task_report.md § 4:

- **Recipe of record and lineage.** punct pack (`punct_pack.py`) → `punct` marks
  → `seed_fixed_1005` → × 0.8 stick = `seed_fixed_1005_stick080` (preview51) →
  `sent_kanji_pres` (whole warm, free_residual 0, lr 2e-4, λ 10 · L_pres at σ
  0.8–0.9 every 2nd step, data = `sent_kanji`'s build) + the 225 (scale's
  `retrain_kanji_b5` cold → `sent_kanji_225` warm, kept rows only) →
  `transplant.py` → `seed_1008` = jp_v1 (sha `ce0ec15f7168…`). With the
  output dirs that hold each.
- **The ruler and the standing table.** progress § 4: f0 / pres / seed_1008
  against preview51 and the old floor, the paired reads, and what is not won
  (cer vs the old floor, short-kana exact falling with each warm pass, long
  recall flat).
- **Modes settled** (progress § 1, one line each): cold never beat the old
  seed; stick re-fits lose the word fit; any warm ball turn costs short strings
  as a random turn does (`rand_turn`); whole-warm is the mode; any warm run
  with rare rows needs `free_residual = 0`.
- **Banner-era verdicts** (progress § 5): five lines, pointers into
  `_archive/reports/`.
- **Probes that closed with no lever** (one line each): probe_geom (box
  weighting inert on row geometry), probe_cf (CF leaves the ∂v/∂e scatter),
  probe_scene (scene diversity is not the scatter), probe_accum (accumulation ≈
  AdamW), idea2 / probe_jl and its PE lens (stopped: no text-only subspace at
  σ 0.8–0.9), idea3 / probe_twin (stopped: identity has no directions of its
  own; the neighbours' rows turn 0.6–0.9 as much).
- **Not tried** (idea.md): per-occurrence credit, an OCR / spelling loss, a
  quote-scoped sequence adapter, the new-kanji string set (plan § 4, user
  10-08: by eye).
- **Reading a row** (structure_candidate): offset = `raw × row_scale`, row =
  pack row + offset; the `m_pack + s + (1 − α)·q + e` table. The jamo
  composition will be read in these terms.
- **KO / ZH notes** (task_report § 4, § 2):
  - TanukiMagic maps 你 to an empty outline.
  - A window needs every glyph to be a row, so 8 Hangul make no window.
  - KO / ZH corpus candidates and licences.
  - `korean text` / `Korean text reads as` captions.
  - 个 is in Shift-JIS, so the language is named per row.
  - Bake with `--base …_punct`.
  - The 1008 kana / kanji ball table.
  - kozh16's own reads once its report exists.
- **Open**: progress § Open (short kana loss, long strings, horizontal 6.8 %,
  the ruler strings not held out of the windows) and the release remainder.

## 3. Move list

| path | destination / action | why |
|---|---|---|
| `README.md` | keep, rewrite | home; code table shrinks to the live files |
| `progress.md` | fold → `status.md`, then `_archive/` | the standing; last touched 10-07, before seed_1008 |
| `criteria.md` | keep | the ruler's definition; `ruler.py` cites it |
| `plan.md` | fold → `status.md` (lineage, two-stage finding, unread new-kanji set), `_archive/` | done 10-08: A, B, transplant, jp_v1 |
| `idea.md` | fold ("not tried"), `_archive/` | 10-05 review; direction 2 (dialogue-length data) done as `sent` tiers |
| `idea2.md` | `_archive/` (one line in status) | both probes stopped (10-07, 10-08) |
| `idea3.md` | `_archive/` (one line in status) | stopped 10-08 |
| `structure_candidate.md` | fold (convention + decomposition table), `_archive/` | reference; the jamo reads need its terms |
| `release_plan.md` | keep until public, then `_archive/` | § Order: 1.3 (LoRA on jp_v1) and the public flip left |
| `task_report.md` | fold § 2 / § 4, `_archive/` | discarded run; its redo notes survive |
| `proposal_jamo.md` | keep (not touched here) | next line |
| `proposal_refactor.md` | `_archive/` once done | this |
| `run.py` | keep; drop `read` + `READ_AGAINST` | `read` is the banner grid via `experiments/grid_lone` |
| `ruler.py` | keep; drop `gs_rkstick`, `rand_turn`, the `probes/` path insert | derived arms of lost arms; renders cached; `gs_rkstick` is the only `probe_split` caller |
| `transplant.py` | keep | provenance of seed_1008 / jp_v1 |
| `punct_pack.py` | keep | provenance of `anima_cjk_vocab_pack_punct` |
| `stick_fit.py` | `_archive/` | banner era (10-04); needs `cjk_scale.reads`, `experiments/row_geometry` |
| `reseed/*` | keep; port § 2 | |
| `configs/punct.toml` | keep | lineage (marks → seed_fixed_1005) |
| `configs/sent_kanji.toml` | keep | data recipe of f0 / pres (`data_from = "sent_kanji"`) |
| `configs/sent_kanji_pres.toml` | keep | rows of record |
| `configs/sent_kanji_225.toml` | keep | the 225 of record |
| `configs/kozh16.toml` | keep | live (trained 10-09) |
| `configs/sent_kanji_f0.toml` | `_archive/configs/` | pres = f0 + `pres`; its numbers live in status |
| `configs/sent_whole.toml` | `_archive/configs/` | tied preview51; data from archived `sent_ball` |
| `probes/kozh_geometry.py` | keep; inline `probe_jl._offsets` (12 lines) | KO / ZH geometry, jamo R3 |
| `probes/kozh_render.py` | keep | renders a `lang` run |
| `probes/probe_split.py`, `probe_grad.py` | `_archive/probes/` | banner era |
| `probes/probe_geom.py`, `probe_cf.py`, `probe_scene.py`, `probe_accum.py` | `_archive/probes/` | 10-06, closed with no lever |
| `probes/probe_pres.py`, `probe_pres_train.py` | `_archive/probes/` | L_pres is the trainer's `pres` now |
| `probes/probe_jl.py`, `probe_twin.py` | `_archive/probes/` | stopped |
| `probes/probe_stick_move.py` | `_archive/probes/` | read done 10-08 (plan § 3 table) |
| `reports/ruler_2026_10_05.md` | keep | the ruler as built, the floor |
| `reports/sent_kanji_2026_10_06.md` | keep | why `free_residual = 0` |
| `reports/probe_pres_2026_10_06.md` | keep | L_pres's definition |
| `reports/sent_kanji_pres_2026_10_07.md` | keep | the rows of record's read |
| `reports/sent_ball_2026_10_05.md` | `_archive/reports/` | ball verdict, folded |
| `reports/probe_{accum,cf,geom,scene}_2026_10_06.md`, `probe_jl_2026_10_07.md`, `probe_jl_pe_2026_10_08.md`, `probe_twin_2026_10_08.md` | `_archive/reports/` | closed / stopped |
| `_archive/*` (existing 34) | stay | |
| `tests/test_boundary.py` | rewrite (§ 4 step 3) | |
| `results/` | stay | gitignored; ruler / probe envelopes |
| `models/pe/` | delete (user OK) | stray 330 MB duplicate of `models/pe` |
| scale `src/` subset (§ 1.3) | copy → `cjk_anima_reseed/src/` | live renderers, readers, train plumbing |
| scale `cjk_scale/{train,rows,loss}.py` | copy → `reseed/{trainer,rows,loss}.py` | the trainer |
| scale `assets/fonts/` (tracked + binaries, `kozh/`) | copy → `cjk_anima_reseed/assets/fonts/` | `FONT_DIR`, `lang_fonts` |
| `project/cjk_anima_scale/` | physical `mv` → `project/finished/cjk_anima_scale/`, `git add -A` both paths | finished line (§ 5) |
| `output/cjk_anima_scale/`, `output/cjk_anima_reseed/` | **stay** | not in git; absolute paths in every record |

### 3.1 Config keys

Keys in `reseed/config.py` KEYS, by the configs that use them:

| key | used by kept configs | only by archived |
|---|---|---|
| rows, seed, steps_per_row, pack | all | |
| shares | punct, sent_kanji, pres, 225 | |
| read, row_lr, rows_from, lr | sent_kanji, pres, 225 | |
| lines | sent_kanji (`m109_pack`), 225 (`m109_b5`) | |
| data_from | pres (`"sent_kanji"`) | `/`-form: ball_rk (scale data dir) |
| free_residual, pres | pres, 225 | |
| focus, held | 225 | |
| lang | kozh16 | |
| **upper_shift** | | kana_up |
| **stick_from** | | stick_*, sent_stick |
| **drop_tiers**, **band** | | stick_* |
| **tag_drop** | | stick_rk_fb_jt50 |
| **ball_on**, **warm** | | ball_rk*, sent_ball* |

Changes:

- Drop the 7 bold keys, their `load()` asserts and `Run` fields, and the
  trainer arguments behind them: `stick_only`, `band`, `tag_drop`, `ball_on`,
  `drop_tiers`, and `row_cap`, which reseed never passes.
- Drop `data_from`'s `/` form (a scale data dir) and `SEEDS` `"0921"`, which
  only archived configs use, together with `SEED_ROWS_0921`.
- `seed` becomes optional when `rows_from` is set: three kept configs carry
  `seed = "0930"  # unused`.
- The `lines` default (scale's `PHRASE_FILE`, dialogue_2_10, used by `punct`
  and `kozh16`) becomes a `LINES` entry.
- The `_archive/configs/` stop loading, which ends README's "still runs by
  path". The tag in step 0 is how to re-run them.

## 4. Steps

Each step is one commit and leaves `pytest project/cjk_anima_reseed/tests` green
(and scale's while it is at its old path). **Before any step:** the daemon queue
holds no job running a reseed or scale script. The code imports lazily inside
functions, so a running job that loses a file mid-run fails.

0. **Freeze point.** Tag `cjk-reseed-pre-refactor` at the branch head (user
   OK). Build a smoke data dir there: `configs/_smoke.toml` = kozh16 under
   its own name, `run.py _smoke data --frac 0.05` (CPU; never into `kozh16/`,
   whose data is live). Keep it as the byte-reference for step 3. Baseline
   tests: 2 / 86 (§ 1.1).
1. **Prune reseed, no code moves.** `git mv` per § 3 into `_archive/`
   (`git add -f` for anything new there). Code:
   - `run.py`: drop `read`.
   - `ruler.py`: drop `gs_rkstick` / `rand_turn` and the `probes/` insert.
   - `kozh_geometry`: inline `_offsets`.

   Write `status.md` and rewrite README. Tests: reseed 2 (its scan covers
   `run.py` + `reseed/`), scale 86 untouched.
2. **Vendor `src/` and fonts.**
   - Copy the § 1.3 subset to `cjk_anima_reseed/src/` and the fonts (tracked
     + `cp -a` the binaries incl. `kozh/`).
   - Apply the path plumbing of § 2.
   - `bootstrap()` puts reseed's `src/` first and scale's line dir (not its
     `src/`) after it, so `cjk_scale.*` still imports but resolves `common`
     / `data` / `train` to reseed's copy. It no longer calls
     `cjk_scale.paths.bootstrap()`.
   - Add `tests/test_src.py`:
     - every vendored file is byte-equal to scale's, bar the listed plumbing
       files (a one-time check, deleted at step 5);
     - `train` resolves under reseed's `src/`;
     - `common.paths.OUT` is `output/cjk_anima_scale`;
     - `FONT_DIR` has faces and `kozh/`.
3. **Port the trainer.**
   - `cjk_scale/{train,rows,loss}` → `reseed/{trainer,rows,loss}` with the
     § 2 edits.
   - `DATA` / `is_ja_text` / `dataset_ja_lines` / `PHRASE_FILE` / `_expand`
     → `reseed/{table,pools,config}`.
   - Paths and pack shas → `reseed/__init__`, with
     `SCALE_OUT = REPO/"output"/"cjk_anima_scale"` named read-only.
   - Scale leaves `sys.path`.
   - `test_boundary.py` asserts that no live file (root scripts, `reseed/`,
     `probes/`, `src/`) imports `cjk_scale`, and that none names
     `cjk_anima_scale` except as `output/cjk_anima_scale`.

   Verify:
   - (a) CPU: `run.py _smoke data --frac 0.05` → `train.jsonl`,
     `build.json` and image hashes equal step 0's.
   - (b) GPU through the daemon: `run.py kozh16 train --out <scratch>
     --max_steps 50` → per-step loss equal to `kozh16/train_log.json`'s
     first 50 (seed 0, same data).
   - (c) `ruler.py run --only 0,1 --label smoke` renders bit-equal to the
     cached renders, allowing for the session noise plan.md § 3 measured
     (mean |Δ| ~4 / 255). Equal by eye is the bar.
4. **Drop the dead keys** (§ 3.1) in reseed's copy. Add
   `tests/test_config.py`: every `configs/*.toml` loads, and every KEYS entry
   is used by at least one of them. Re-run step 3 (b).
5. **Archive scale.**
   - Repoint `tests/test_bake_vocab_pack.py` to reseed's `src/` and run that
     file once.
   - Fix the § 1.4 doc links.
   - `mv project/cjk_anima_scale project/finished/cjk_anima_scale`. This
     carries the gitignored `_archive/`, the font binaries and `_dl/`.
   - Bump its depth so it stays runnable by path, as `cjk_renderable_anima`
     did: `src/common/{__init__,paths}.py` `parents[4]` → `[5]`,
     `cjk_scale/paths.REPO = LINE.parents[2]`, `prompts.TARGET_PROMPTS`.
   - Add `.ignore` (`*`).
   - Add an entry in `finished/README.md`; drop scale from `project/README.md`'s
     active list and add reseed.
   - Remove the byte-equality check from `test_src.py`.
   - Run scale's tests once at the new path (86 expected).
6. **Close out.** Write kozh16's report (README already names it). Move this
   proposal to `_archive/`. The user edits `archive-explorer.md`.

## 5. Where scale goes

Recommend `project/finished/cjk_anima_scale/` (tracked).

- **It ran to a conclusion.** It shipped the preview3–preview51 seeds and
  `retrain_kanji_b5` (the 225 in jp_v1).
- **Its band law is cited by live material:** `docs/methods/cjk_vocab_pack.md`,
  `docs/experimental/anima_cjk_vocab_ext.md`, the `ext_vocab.py` docstring and
  `project/README.md`.
- **That is `finished/`'s case** in `project/README.md`: "the verdicts stay
  visible in the repo". `_archive/` (gitignored, local plus the private mirror)
  is for killed or superseded lines.

`.ignore`: `*`, as `cjk_renderable_anima` has it. After step 5 no live code
reads a scale path. The one live reader today (`test_bake_vocab_pack.py`) is
repointed first. If the user wants `band_experiment_results.md` to stay
searchable, the rule becomes `/*` plus `!/band_experiment_results.md`.

Staying runnable by path (the depth bump) keeps `scale.py <run> data | train`
for a future cold kanji batch (stage A) and keeps its 86 tests running. The
cost is two copies of `src/` that may drift: scale's is frozen, reseed's is
live.

## 6. Risks

- **Outputs do not move.** `output/cjk_anima_scale` (40 G) and
  `output/cjk_anima_reseed` (63 G) are not in git. Records hold absolute paths:
  - `trained.pt` `args.{context, init_rows, data}`
    (`sent_kanji_pres` → `output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt`);
  - `train_record.json`;
  - `build.json`;
  - result envelopes.

  `stick_fit` / `probe_split` read `args.context` back (archived). Renaming
  `output/cjk_anima_scale` would break data builds: the scene pools, the EN
  refs, `rows_from`, and the ruler's floor arms (`retrain_kana`,
  `seed_retrain_0930`, `seed_fixed_1005_stick080`).
- **Stale `run_config` paths.** `args.run_config` in records points at
  `project/cjk_anima_reseed/configs/<run>.toml` and
  `project/cjk_anima_scale/configs/runs/<run>.toml`. It goes stale for
  archived configs and for scale after the move. Only writers use it today.
- **Daemon.** Queued jobs name script paths at submit. Lazy imports mean a
  running job can fail after a move. Check the queue before each step;
  `kozh_render` was running at writing.
- **`$MANGA109S`.** `LINES`, `held` and the `PHRASE_FILE` default stay
  `$MANGA109S/derived/*.tsv`, expanded by `_expand` after `load_dotenv()`. The
  port must keep that call (a daemon child has no shell env). No corpus file
  moves.
- **Fonts are gitignored.** `git mv` leaves the binaries behind;
  `pools.lang_fonts` asserts on an empty `kozh/`. Copy by hand (step 2) and
  check in `test_src.py`.
- **Byte-faithfulness.** Any edit to the vendored renderers changes builds of
  record. Step 2's equality test and step 3 (a) catch it.
- **The trainer port.** `plan()` without the budget changes
  `train_record.json` fields only. Step 3 (b) is the check.
- **`_archive/` hides from search.** New files there need `git add -f`. That
  is the line's existing convention, and ripgrep then skips them, as intended.
- **`proposal_jamo.md`** names `cjk_scale.train` and "renderers and KO fonts
  still in `../cjk_anima_scale`". Its author updates those names after step 3.
- **Merge to main.** Main's scale is the merge-base copy and main has no
  `project/cjk_anima_*` commits since, so the moves should merge cleanly.

## 7. Open questions

1. Scale → `project/finished/` (recommended) or `_archive/`? Should it stay
   runnable by path (the depth bump), or freeze as a record?
2. Will a future pack need another cold stage A (scale's builder and
   `retrain_kanji_b*` recipe, e.g. more kanji for a jp_v2)? If yes, scale must
   stay runnable (Q1), or its builder is ported, which this proposal does not
   do.
3. Tag `cjk-reseed-pre-refactor` and push it? After step 4 the archived
   configs and probes run only from that tag.
4. `release_plan.md`: is 1.3 (LoRA on jp_v1) closed, and is `anima-jp-extended`
   public? If both, it goes to `_archive/` at step 1.
5. `sent_kanji_f0.toml`: archive (proposed; pres = f0 + `pres`), or keep as the
   text arm of record?
6. `structure_candidate.md`: fold into status (proposed), or keep whole as the
   reference for the jamo reads?
7. Shared place: vendored `src/` under reseed (proposed), or promote the
   renderers / readers to an installed package (`library/…`)? The latter is a
   larger change with its own tests.
8. Delete the stray `project/cjk_anima_reseed/models/pe/` (330 MB)?
9. Merge `cjk-reseed` into `main` after the refactor, or keep the line on the
   branch?
