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
  never into this tree (it is committed). Row arms (a `trained.pt` the
  eval renders, one dir per arm: `tl_*`, `tp_*`) go under
  `output/cjk_anima_scale/experiments/<arm>/`, with the stage's
  `--arm_path` pointed there.
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
- `piece_only/` — reports/next_2026_09_25.md § 4a (2): the 300 pieces on `scene_piece` items
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
- `transplant_piece/` — `../proposal.md` Stage A, training-free and out of
  sample: `u_P` = the mean tangential Δ of 292 of 300f_sp's pieces, added at
  one coefficient (their mean projection, 94.1) × α to the seed rows of the
  8 piece-ruler pieces, plus a random ⟂ control; read on the floor cache's
  `native_piece/` keys. 300f_sp is `scene_piece`-only, so no item carries a
  held-out piece beside a donor (leak 0). **Ran 2026-09-26** → transfers:
  contained 27 → 52 / 256 (300f_sp 76), random control 11
  (`../reports/transplant_piece_2026_09_26.md`).
- `stage_b/` — `../proposal.md` Stage B: 36 donor kana trained on 568
  manga109s lines (`scene_spelled`, glyph-balanced draw, no repeated glyph)
  plus a count tier (`scene_single_small`: one glyph at 24–40 px in a bubble
  it fills 0.2–0.4 of, 0.3 of b0507). `u_S` = the donors' mean tangential Δ,
  added at one coefficient to the seed rows of ひ ま わ り さ く ら み ど も
  (arms `tb_*`) plus a random ⟂ control, read on five spelled words made of
  them and the ten alone. The held-out keys' floor was rendered once into
  `native_spell/`. **Ran 2026-09-26** → composition transfers (≤ 1 edit
  11 → 66 / 160, random 9) and doubling with it (repeats 25 → 53 / 320)
  (`../reports/stage_b_2026_09_26.md`).
- `f0_interaction/` — `../proposal_factorizedrows.md` § 2 (F0), no DiT: the
  Qwen + `llm_adapter` forward at a glyph's position, alone vs in a spelled
  word, with only its own row changed (`self`) or every glyph's (`all`), on
  Stage B's u1 / random ⟂ / donor rows. **Ran 2026-09-26** → the adapter
  passes a row's change through blind to its neighbours (out cos
  0.95–0.97, gain 0.97; donor ≈ random), so the gate lives at the hook
  (`../reports/f0_interaction_2026_09_26.md`).
