# proposal_refactor — reseed torn down and rebuilt, scale archived (2026-10-09)

**Status: steps 0–3 done (10-09); 4–6 left.** Work happens on `cjk-reseed`;
`main` holds a stale copy of both lines.

| step | commit | what |
|---|---|---|
| 0–1 | `0a23be0c` | freeze point `b3dee68e` (in `status.md`); the prune; `status.md`; README |
| 2 | `9a6f6bf2` | scale's `src/` subset and the fonts vendored into reseed |
| 3 | `19aac524` | `cjk_scale/{train,rows,loss}` → `reseed/{trainer,rows,loss}`; no live file imports `cjk_scale` |

The verification of each is in its commit message.

**Decided (user, 10-09):** scale → `project/finished/cjk_anima_scale/`;
reseed is the home for cold training (a cold-batch capability reseed lacks is
ported into its table / recipes when found, not run from `finished/`);
scale's depth bump (step 5) stays as a cheap fallback that keeps it runnable
by path; no tag (archived configs and probes re-run from the freeze-point
commit).

**References the remaining steps use:**
- Smoke data: `output/cjk_anima_reseed/_smoke.toml` (kozh16 under its own
  name, kept out of `configs/`) → `_smoke/data`; file hashes in
  `_smoke_ref.sha256`, `build.json` in `_smoke_ref_build.json`.
- The training check (step 3 (b)):
  `make daemon-run ARGS="project/cjk_anima_reseed/run.py kozh16 train --out output/cjk_anima_reseed/<scratch> --max_steps 50"`.
  Step-1 loss was 0.15419110655784607 for the port and the freeze-point code
  in one session; steps 25 / 50 vary by ~2e-4 run to run.

## 1. Config keys (step 4)

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
  `reseed/trainer.py` arguments behind them: `stick_only`, `band`,
  `tag_drop`, `ball_on`, `drop_tiers`, and `row_cap`, which reseed never
  passes.
- Drop `data_from`'s `/` form (a scale data dir) and `SEEDS` `"0921"`, which
  only archived configs use, together with `SEED_ROWS_0921`.
- `seed` becomes optional when `rows_from` is set: three kept configs carry
  `seed = "0930"  # unused`.
- The `lines` default (`config.PHRASE_FILE`, dialogue_2_10, used by `punct`
  and `kozh16`) becomes a `LINES` entry. `_expand` keeps its `load_dotenv()`
  call (a daemon child has no shell env).

## 2. Outside readers of scale (step 5)

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

Nothing in `bench/`, `scripts/` (bar the docstring), `scripts/tasks/`,
`tasks.py`, `custom_nodes/`, `gui/`, `anima_lora/` or `library/` (bar the
docstring) imports scale or reseed.

`.ignore` for `finished/cjk_anima_scale/`: `*`, as `cjk_renderable_anima`
has it. If its `band_experiment_results.md` should stay searchable, the rule
becomes `/*` plus `!/band_experiment_results.md`. The cost of the depth bump
is two copies of `src/` that may drift: scale's is frozen, reseed's is live.

## 3. Steps

Each step is one commit and leaves `pytest project/cjk_anima_reseed/tests`
green (and scale's while it is at its old path). **Before any step:** the
daemon queue holds no job running a reseed or scale script (lazy imports: a
running job that loses a file mid-run fails).

4. **Drop the dead keys** (§ 1). Add `tests/test_config.py`: every
   `configs/*.toml` loads, and every KEYS entry is used by at least one of
   them. Re-run the training check.
5. **Archive scale.**
   - Repoint `tests/test_bake_vocab_pack.py` to reseed's `src/` and run that
     file once.
   - Fix the § 2 doc links.
   - `mv project/cjk_anima_scale project/finished/cjk_anima_scale`. This
     carries the gitignored `_archive/`, the font binaries and `_dl/`;
     `git add -A` both paths.
   - Bump its depth so it stays runnable by path, as `cjk_renderable_anima`
     did: `src/common/{__init__,paths}.py` `parents[4]` → `[5]`,
     `cjk_scale/paths.REPO = LINE.parents[2]`, `prompts.TARGET_PROMPTS`.
   - Add `.ignore` (`*`).
   - Add an entry in `finished/README.md`; drop scale from `project/README.md`'s
     active list and add reseed.
   - Remove the byte-equality check (and `SCALE_SRC`) from `tests/test_src.py`.
   - Run scale's tests once at the new path (86 expected).
6. **Close out.** Move this proposal to `_archive/`. The user edits
   `archive-explorer.md` and deletes the scratch outputs
   (`output/cjk_anima_reseed/{_smoke*, _ruler_*}`).

## 4. Risks

- **Outputs do not move.** `output/cjk_anima_scale` (40 G) and
  `output/cjk_anima_reseed` (63 G) are not in git, and records hold absolute
  paths (`trained.pt` `args`, `train_record.json`, `build.json`, result
  envelopes). Renaming `output/cjk_anima_scale` would break data builds (the
  scene pools, the EN refs, `rows_from`) and the ruler's floor arms.
- **Stale `run_config` paths.** `args.run_config` in records points at
  `project/cjk_anima_reseed/configs/<run>.toml` and
  `project/cjk_anima_scale/configs/runs/<run>.toml`; it goes stale for
  archived configs and for scale after the move. Only writers use it.
- **`_archive/` hides from search.** New files there need `git add -f`.
- **Merge to main.** Main's scale is the merge-base copy and main has no
  `project/cjk_anima_*` commits since, so the moves should merge cleanly.

## 5. Open questions

- Delete the stray `project/cjk_anima_reseed/models/pe/` (330 MB, byte-equal
  to the repo's `models/pe/PE-Spatial-B16-512.pt`)? The auto-mode check
  blocked the agent; the user runs `rm -rf project/cjk_anima_reseed/models`.
- Merge `cjk-reseed` into `main` after the refactor, or keep the line on the
  branch?
