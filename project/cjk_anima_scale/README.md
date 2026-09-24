# cjk_anima_scale — the JA vocab pack at scale

Opened 2026-09-23 out of [`../cjk_renderable_anima/`](../cjk_renderable_anima/).
That line is the research surface (probe code, `reports/`, `findings.md`);
this one is the production line that builds the pack on what it settled:
one loss, one trainer, and a σ-band schedule whose stages differ only in
their data. Nothing here has trained yet — the plans are plans.

## The vocab band law

[`band_experiment_results.md`](band_experiment_results.md) is the **vocab
band law**: which σ band a vocab-pack row trains in, keyed on what. It is not
a theory — every line of it is a measured read (EN ceiling on the base model,
JA training arms with two seeds), and it holds only over the sizes, layouts
and units those reads covered. As it stands:

- **The band is keyed on the row's glyph count.** Single-glyph rows train at
  0.7–0.9; multi-glyph one-token rows at 0.5–0.7. The two are two runs (or
  two per-item bands), never one band.
- **Rendered px sets the floor the band may reach**, not the band itself
  (§ 2, the per-px window table: 12–16 px text lives at 0.2–0.6, 48 px at
  0.5–0.7, 128 px at 0.8). A grid cell sits one step higher.
- **Nothing above 0.9** — 0.8–0.95 is dead at 48 px for kana and kanji alike.
- **Ink, stroke density and the bubble ellipse move no band.** Kanji take the
  kana band; density is an exposure / px question (§ 6 item 3, open).
- **16 px glyphs carry more caption leverage than 24 px**, lower in σ — small
  text is a band question, not a capability question.

The plans that produced it (`plan_band.md`, `plan_kanji.md`) closed on
2026-09-23 and were deleted; they are in git history at `f5cd4c0c`. The
probe line's step 1a / 1b / merge / step 2 recipe (`recipe.md`) was retired
the same day — `configs/stage*.toml` carry its settings, the band law § 3–4
its reads (git `ff2f70f9` has the last copy). The
reads themselves are the dated reports under `../cjk_renderable_anima/reports/`
(`cf_rebin_gate0`, `cf_band_a1`, `band_b1`, `cf_kanji_c1`, `band_c2_kanji`).
A new read that changes a row of the law goes into
`band_experiment_results.md`, with its report there.

## Files

| file | what |
|---|---|
| [`band_experiment_results.md`](band_experiment_results.md) | **the vocab band law** — the verdict, the per-px window table, the training reads, what is left unrun |
| [`design.md`](design.md) | the scale pipeline: stage schedule, data builder, thin trainer, open questions (§ 6) |
| [`plan_canvas.md`](plan_canvas.md) | plan only — does the law hold on a ~500-token canvas (2× throughput) |
| [`micro_chain_result.md`](micro_chain_result.md) | **the first run of the chain** — the three band stages at 30 / 30 / 30 on 24 warm rows (8 kana + 8 kanji + 8 pieces), read rule by rule; the seed-baseline and table-membership tool changes it forced |
| `configs/stage*.toml` | the four stages — the band recipes: band, gate, warm chain, recipe mix, trainer surface, eval; never which rows |
| `configs/runs/*.toml` | the runs — which rows, the seed table, steps per row per stage: `run_full` (production), `run0923_micro` (the 24-row chain read) |
| `cjk_scale/` | the code (`windows` = the law, `recipes` + `builder` = data, `rows` + `train`, `eval`, `bake`, `ledger`); `scale.py` is the front door |
| `runs/` | `ledger.jsonl` — every submitted job |

## Where it stands (2026-09-24)

The law is written and encoded (`cjk_scale/windows.py`); the four stage
configs, the data builder with the band gate, the thin trainer, the eval
delegation and the front door exist. **The chain has run once, on 24 warm
rows** ([`micro_chain_result.md`](micro_chain_result.md)): at μ = 0.1 on
every stage it passes the chain-end rule (singles 24 → 26 / 32, pieces
2 → 5 / 16, EN held), which set `init_anchor = 0.1` in the stage files; at
μ ≤ 0.01 stage0507 undid stage0709's singles. The piece read is a budget
statement (the next cell is 60 / 60 / 60), and the piece `cf_sense` ruler
renders too large to read the 0.3–0.5 band. **No stage has trained at
scale.** Seed table for the chain:
`output/cjk_anima_scale/rows_step1_0921_merged` (2 274 rows, the probe line's
`step1_0921` + `step1_0921z` merge). Every launch states
the raw pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`,
sha `7b9fce0b…`) and goes through the daemon.

## Running a stage

```bash
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack   # MANGA109S comes from .env
.venv/bin/python project/cjk_anima_scale/scale.py --run run_full --stage stage0709 --steps data   # CPU
.venv/bin/python project/cjk_anima_scale/scale.py --run run_full --stage stage0709 --steps train eval --submit --queue
.venv/bin/python project/cjk_anima_scale/scale.py --run run_full --stage stage0507 --steps data train eval --submit
.venv/bin/python project/cjk_anima_scale/scale.py stages | runs | windows | ledger
```

`--run` names the chain (`configs/runs/<run>.toml`: the rows, the seed
table, steps per row per stage); its name is the tag every stage dir of the
chain carries, and `warm_from = "<stage>"` in a stage file resolves to that
stage's table under it. `--tag` alone runs a stage without a run file (smoke
builds on the default inventory). Everything lands under `output/cjk_anima_scale/` — the stage dirs
`{data,rows}_scale_<stage>_<tag>/`, the scene pools `scenes_<tag>/`, the EN
reference cache and the seed table — and `paths.bootstrap()` points the
probe's output root there, so its `eval` / `native` / `cf_sense` and every
`probe/*.py` reader open them unchanged (the probe's own `output/wake_probe/`
holds symlinks to the shared pools). `--submit` records the job in
`runs/ledger.jsonl`.
The code is the `cjk_scale/` package (not `src/`: the probe's `src/` exposes
top-level `common` / `data` / `train` / `eval`, which a second `src/` would
shadow); tests: `.venv/bin/python -m pytest project/cjk_anima_scale/tests`.
