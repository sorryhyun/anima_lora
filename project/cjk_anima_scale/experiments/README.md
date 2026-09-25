# experiments — idea validation for the scale line, bench-style

One directory per experiment; each validates one idea (usually from
`../idea.md`) before any production code changes. This is the line's
`bench/`: same envelope, same discipline, but scoped to the line and free
to import `cjk_scale/` and the probe's `src/` primitives.

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
- Never write into a `data_scale_*` dir: sample records, keep sub-set
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
  piecenat saw? Value-only (no bank): in-box FM loss on matched
  correct / doubled renders of the 8 piecenat pieces at the seed, step-5 000
  and final 300f tables. T1 = per-piece identity gain vs the piecenat gains
  (2-glyph bought, 3+ not); T2 = the doubled-vs-correct margin, the
  acceptance-axis read. **Ran 2026-09-25** (`--label t1`) → verdict in
  `../reports/influence_target_2026_09_25.md`: **both fail** — the loss gain
  ranks the unrendered long pieces first and すごい at zero, the doubling
  margin moves in no pattern. Loss-target influence closed; bank v2 not built.
