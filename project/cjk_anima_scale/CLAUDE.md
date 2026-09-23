# cjk_anima_scale

Production line for the JA vocab pack: one loss, one trainer, a σ-band
schedule whose stages differ only in their data. `README.md` = state,
`design.md` = the design, `band_experiment_results.md` = the vocab band law.
The research surface stays in `../cjk_renderable_anima/` (reports, probes).

## Layout

- `scale.py` front door: `--stage <s> --tag <t> --steps data train eval bake [--submit [--queue]]`;
  `scale.py stages | windows | ledger`.
- `cjk_scale/` the code. **Not `src/`**: the probe's `src/` exposes top-level
  `common` / `data` / `train` / `eval`; this package imports them and must never
  shadow them (`tests/test_line.py` pins the direction).
- `configs/stage*.toml` one file per stage: `band`, `gate`, `warm_from`, `[data]`
  pools, `[[data.mix]]` recipes + shares, `[train]` (the whole trainer surface),
  `[eval]`.
- Outputs: `output/wake_probe/{data,rows}_scale_<stage>_<tag>/` — the probe's
  layout with a `scale_` prefix, so its eval / native / cf_sense and `probe/*.py`
  open a stage table unchanged.

## Invariants

- `--tag` names a chain; `warm_from = "<stage>"` resolves to that stage's
  `trained.pt` under the same tag. No cold row inside a stage except a new one.
- `cjk_scale/windows.py` is a **row table with provenance**, not a formula. A new
  read changes a row and its source string; the tests hold the reads.
- `kind` is keyed on the **glyphs the item draws**: one → `single`, else `multi`.
  A piece (って: one token, 2 glyphs) and a small-kana digraph (あっ: two tokens,
  2 glyphs) are both `multi`. `px` is √(box area / glyphs), the ink-stat px of
  the reports.
- The band gate keeps an item iff the stage band is inside its window (or
  `min_overlap` of it); the ±20 % `px_target` gate reads the recipe's **drawn**
  px, before the band gate truncates it.
- Every launch names the pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`)
  and GPU steps go through the daemon (`--submit`); `MANGA109S=~/manga109s` for
  the phrase stages. The path never enters the repo.
- Tests: `.venv/bin/python -m pytest project/cjk_anima_scale/tests` (line-local,
  not part of the repo suite).
