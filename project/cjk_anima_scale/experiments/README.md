# experiments — idea validation for the scale line, bench-style

One directory per experiment; each validates one idea (usually from
`../idea.md`) before any production code changes. This is the line's
`bench/`: same envelope, same discipline, but scoped to the line and free
to import `cjk_scale/` and the line's `src/` primitives. The two influence
experiments read the old stage-layout dirs (`data_<stage>_<tag>`) through
`cjk_scale/legacy.py` (the archived stage configs).

## Contract

- `<exp>/run_exp.py` — the entry point, a thin argparse script. `--dry_run`
  must plan (sample, count, print) without touching a model.
- Results: `<exp>/results/<YYYYMMDD-HHMM>[-<label>]/` with the standard
  `result.json` envelope (`bench/_common.py::write_result`) + `report.md`.
  Always pass `--label` — same-minute runs overwrite the dir
  (`project_bench_run_dir_collision`).
- Heavy artifacts (gradient tensors, latent caches) go under
  `output/cjk_anima_scale/<exp>_<label>/`, referenced from `result.json`,
  never into this tree (it is committed).
- GPU work goes through the daemon
  (`make daemon-run ARGS="project/cjk_anima_scale/experiments/<exp>/run_exp.py …"`)
  and every launch names the pack
  (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`).
- Never write into a `data_*` dir: sample records, keep sub-set
  latent/TE caches in the experiment's own output dir.
- A verdict that closes (or opens) an idea is written into
  `../reports/` and the wake roll-up like any other read; this tree holds
  the machinery and the raw envelopes, not the line's memory.

## Experiments

- `influence_smoke/` — first contact for `idea.md`'s gradient bank +
  validation influence: does `I[c, s] = v_sᵀ ḡ_c` rank the recipes the way
  the rulers did, and does the linearized prediction match the measured
  dev-loss change across the run0925_300f delta? **Ran 2026-09-25** →
  verdict in `../reports/influence_smoke_2026_09_25.md` (bank on hold:
  estimator needs row-conditioned sampling; the dev-loss target can't see
  the acceptance axis).
- `influence_target/` — step 1 after the smoke: does the dev target see what
  piece saw? Value-only (no bank): in-box FM loss on matched
  correct / doubled renders of the 8 piece pieces at the seed, step-5 000
  and final 300f tables. T1 = per-piece identity gain vs the piece gains
  (2-glyph bought, 3+ not); T2 = the doubled-vs-correct margin, the
  acceptance-axis read. **Ran 2026-09-25** (`--label t1`) → verdict in
  `../reports/influence_target_2026_09_25.md`: **both fail** — the loss gain
  ranks the unrendered long pieces first and すごい at zero, the doubling
  margin moves in no pattern. Loss-target influence closed; bank v2 not built.
- `parity_300f/` — plan.md § 6-3 without the retrain: replays run0925_300f
  through the one-file code (`plan` CPU = `--dry_run`: vocabs, trainer
  record, save-time merge; `steps`: the first N steps vs the old log; `eval`:
  the old rows under the new eval vs the old reads). **Ran 2026-09-26** →
  parity holds (`../reports/piece_only_2026_09_26.md` § 1).
- `piece_only/` — next.md § 4a (2): the 300 pieces on `scene_piece` items
  only (`build(…, table=)` with the piece groups cut to that tier), data →
  train → eval as `run0926_300f_sp`. **Ran 2026-09-26** → not the doubling
  lever (`../reports/piece_only_2026_09_26.md` § 2).
- `spell_b/` — the five singles あ り が と う trained on in-line real words
  (`scene_spelled`: unspaced image, spaced caption → each glyph its single
  row) at the piece bands, ありがとう held out; read spelled + alone.
  **Ran 2026-09-26** → composition bought, count lost
  (`../reports/spell_2026_09_26.md` § 4).
- `transplant_line/` — the shared "line" Δ, training-free: `transplant`
  (300f_sp's piece direction added to the seed singles), `strip` (spell_b's
  rows minus their shared component), `shared` (the seed plus only it); read
  on keys the floor cache holds — the eval refuses a missing floor key.
  **Ran 2026-09-26** → `../reports/transplant_line_2026_09_26.md`.
- `canvas/` — `plan_canvas.md` (the plan and its verdict, beside the
  script): does the base spell on a 512-token canvas? Training-free on the
  seed rows: per `--canvas` WxH, the EN control, native あ / い and the
  `cf_sense ja` identity peak (`d0`), or A.1's EN per-px ceiling (`d1`),
  each beside the 512² reads the seed dir already holds. **Ran 2026-09-26
  (d0)** → 256×512 / 512×256 pass (EN 23/24, identity kept, peak 0.7 ≈ 0.8;
  the tall canvas doubles); taken into `recipes.GRIDS` as `1x2` / `2x1`.
