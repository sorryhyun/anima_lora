# refactor.md — `probes/` → `src/`, and the two grab-bag packages

Written 2026-09-16, **not executed**. Scope: this line's code tree only
(`project/cjk_renderable_anima/`). No behaviour change, no flag change, no
output-path change — the CLI surface and the data dirs must come out byte for
byte identical. Review this, then I run it.

## What is wrong today

| | |
|---|---|
| `probes/` is not probes | It holds the whole instrument — the trainer, the data builder, the eval, the scene compositor. Three of the 26 files are actual probes. |
| `wake/` mixes two kinds of thing | Genuinely shared plumbing (`models` `hooks` `readers` `render` `bubble`) sits next to code exactly one stage reads: `enref` (eval only), `trainables` + `encoder` (train only), `units` + `inventory` (data only). |
| Two grab-bag files | `common.py` is five concerns in one (repo paths, kana inventories, prompt templates, CER, canvas shapes). `cli.py` is 766 lines of argparse in one module. |
| Two overloaded stage files | `stage/eval.py` (672) carries **four** stages — `eval`, `native`, `enref`, `native_rescore`. `stage/scenes.py` (874) carries generation *and* the keep/reject judge. |
| `probes/` vs `bench/` never separated | `wake_geometry.py` and `rows_manifold_probe.py` never run the sampler — they are rulers over a finished arm, and read as probes only because of where they live. |

## Target layout (agreed: by role — a stage owns the plumbing only it reads)

```
src/
  wake_probe.py          entry point, unchanged CLI          (was probes/wake_probe.py)
  stages.py              the --stage registry                (was probes/stage/__init__.py)
  common/                what 3+ stages share
    __init__.py          docstring map + the sys.path bootstrap (see trap 2)
    paths.py             REPO OUT CORPUS_* FONT_DIR data_dir arm_dir
    text.py              kana inventories, regexes, norm/lev/cer
    prompts.py           TPL_* EVAL_GROUPS NATIVE_CLAUSES NATIVE_PROMPTS EN_WORDS
    shapes.py            wh parse_shape parse_shapes
    models.py hooks.py readers.py bubble.py      (moved verbatim)
    render/flat.py       font layouts: find_fonts pick_font sample_layout render_string crop_bubble
    render/scene.py      the S-line compositor: split_lines fit_text region_capacity erase_* anchor_* render_into_scene
  data/    stage.py  synth.py  units.py  inventory.py
  train/   stage.py  trainables.py  encoder.py
  eval/    stage.py  native.py  enref.py  classify.py  salad.py
  scenes/  stage.py  judge.py           (874 → ~450 + ~420: generation vs keep/reject)
  cli/     __init__.py (build_parser)  run.py  data.py  train.py  eval.py  scenes.py
  probe/   order_probe.py  transplant_table.py  merge_tables.py  quote_dir.py
  bench/   wake_geometry.py  rows_manifold.py
tests/     (new — see below)
```

`probe/` = runs the model to ask a question. `bench/` = reads a finished arm
and scores it, no sampler. Neither emits the repo-root `bench/` `result.json`
envelope, so the name is local to this line; noted in the README.

Only the entry-point path changes for a daemon recipe:
`…/probes/wake_probe.py` → `…/src/wake_probe.py`. Every flag, default and
output dir stays.

## Traps found while planning (the reason this is not a blind `git mv`)

1. **Path-depth constants move with the file.** `common.py` computes
   `REPO = parents[4]`, `render.py` computes `FONT_DIR = parents[2]`,
   and `wake_geometry` / `transplant_table` / `rows_manifold` use `parents[3]`.
   Under the new tree `common/paths.py` keeps `parents[4]`, but everything in
   `src/probe/` and `src/bench/` goes to `parents[4]`, and `FONT_DIR` moves into
   `paths.py` (`parents[2]`) instead of being recomputed in the renderer.
2. **The `sys.path` bootstrap is load-bearing and implicit.** Today
   `wake/common.py` puts the repo root *and* the frozen line's `ocr/` dir on
   `sys.path` as an import side effect — `models.py` (`library.*`) and
   `readers.py` (`pseudo_label`) only work because everyone imports `common`
   first. Splitting `common.py` into four modules breaks that by accident, so
   the bootstrap moves to `common/__init__.py`, where any `from common.x import y`
   is guaranteed to trigger it.
3. **`cli` is a real package in site-packages.** I hit
   `ImportError: cannot import name 'build_parser' from 'cli'
   (.venv/lib/python3.13/site-packages/cli/__init__.py)` while writing the
   baseline dumper. With `src/` at `sys.path[0]` ours wins, but `cli` `data`
   `train` `eval` `bench` `common` are all generic top-level names once `src/`
   is on the path, and `bench` also collides with the repo's own `bench`
   package. Nothing in this tree imports either, so it is safe — but it is the
   reason the entry scripts must insert `src/` at **position 0**, not append.
4. **`render.py` splits cleanly, `stage/eval.py` does not.** The renderer has a
   hard seam at its "S line" comment (line 265) with no call crossing it. The
   eval module shares `_sheet_row` and `_blank_cell` between the `eval` and
   `native` halves — those two go to `eval/stage.py` and `native.py` imports
   them, rather than being duplicated.
5. **`--init_rows` is now a comma list** (53k table + punctuation table). Any
   test fixture that warm-starts must exercise the list form, not just one path.

## How the refactor gets verified (no repo test suite is run)

- **Golden CLI dump.** Serialise every action's `option_strings` / `dest` /
  `default` / `choices` / `nargs` / `type` / `help` plus group membership to
  JSON before the move, re-dump after, `diff` must be empty. (The dumper is
  written; it is what caught trap 3.)
- **Byte-identical data dir.** Build one small CPU-only data dir before and
  after (`--stage data`, fixed seed, tiny `--n_single`) and compare file
  hashes — the tree already promises a recipe rebuilds its data dir byte for
  byte, so this is the real regression test.
- **`tests/`** (new, run only as `.venv/bin/python -m pytest
  project/cjk_renderable_anima/tests` — never the repo suite):
  - every module imports, and every `STAGES` entry resolves to a real callable
  - the CLI golden dump, checked in as a fixture
  - `parse_units` — canonical source order, so typed order cannot change data
  - `parse_shape` / `parse_shapes` incl. the multiple-of-16 assert
  - `norm` / `cer` on the punctuation-stripping cases
  - `split_lines` kinsoku (no NO_HEAD char starts a line, no NO_TAIL ends one)

## Docs that reference the old paths (updated in the same commit)

`README.md` (11 refs), `plan.md` (7), `plan_synth.md`, `findings.md`,
`findings_seed.md`, `datacheck.md`, `freetext.md`, three files under
`reports/`, and `../cjk_aware_anima_dit/README.md` (points at
`../cjk_renderable_anima/probes/`). Also `.gitignore`-adjacent hygiene: the
tracked tree has `probes/__pycache__/` on disk — dropped, not moved.

## Deliberately not doing

- Not renaming `wake_probe.py` — every daemon recipe, report and memory names it.
- Not touching `output/wake_probe/` or any arm-dir naming; `data_dir()` /
  `arm_dir()` semantics are unchanged.
- Not merging `data/stage.py` (562) with `data/synth.py` (333); they are one
  stage but two distinct data regimes and both are already coherent.
- Not adding a `result.json` envelope to `bench/` — these rulers are not repo
  Tier-1.5 benches and inventing the envelope would imply a contract they do
  not keep.
