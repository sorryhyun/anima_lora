# Where MOSS-Image resolves the picture, and where Spectrum spends its full steps

Measured 2026-10-21 on MOSS-Image Base and SFT. Probe, run and full tables:
`_archive/bench/moss_sigma_staircase/` (`probe_x0.py`, `README.md`, run
`results/20261021-2204-base-sft`). Re-measures the Anima finding
`docs/anima/findings/sigma_signal_where_anima_resolves.md` on MOSS
(`_archive/proposals/moss_anima_negatives_revisit.md` Tier 3 §3).

## Setting

On Anima, one bare forward's `x0_pred = x_σ − σ·v` is a recognisable picture at σ ≈ 0.75
and essentially final at σ ≈ 0.45. Several MOSS levers were set against that shape. MOSS
differs in three places that could move it:

- Training σ is noise-biased (median 0.78; 34 % of draws above 0.85).
- The sampler runs shift 5 snapped to the training grid (`model.snapped_sigmas`).
- The DiT is single-stream: the text rows run all 42 blocks on every step.

The probe noises 16 cached ~1 MP latents at each of the 50 σ of the default sampler grid
and runs one forward per σ (true caption length, CFG 1, no LoRA). Error is latent-MSE
against the clean latent, normalised by its value at σ 0.952 (Anima's normalisation), on
the full latent and on a 4×4 average-pooled latent (layout and colour blocks, no detail).

## Measured

| level (normalised error) | Anima σ | MOSS full latent σ (steps above / 50) | MOSS 4×4-pooled σ (steps above) |
|---|---|---|---|
| 0.331, Anima's "recognisable" | 0.75 | 0.737 (32) | 0.870 (22) |
| 0.20 | 0.63 | 0.572 (39) | 0.802 (28) |
| 0.092, Anima's "final" | 0.45 | 0.337 (45) | 0.634 (37) |

Base; SFT's knees are within 0.012 σ of these. The Anima finding's text puts its 20 %
crossover at σ ≈ 0.55, but its own table (0.142 at 0.55, 0.214 at 0.65) puts it at
σ ≈ 0.63; this table uses the latter.

- **The full-latent curve matches Anima's down to σ ≈ 0.75** (0.341 at σ 0.746 against
  0.331 at 0.75). Below that MOSS converges more slowly: 0.163 at σ 0.5 against ~0.117, and
  "final" arrives at σ 0.34, not 0.45.
- **Layout resolves well above that.** On decoded grids σ 0.95 is still a blur,
  composition and pose are fixed by σ 0.90, and σ 0.80 matches the final image except for
  detail (eyes, accessories, line weight).
- **SFT resolves where Base does**: error 1–3.5 % higher at every σ, knees unmoved.
- **The samples agree**: std of the error at σ 0.5 is 0.037 against a mean of 0.106.

Where the steps go, for the same σ boundaries:

| | above 0.87 | above 0.80 | above 0.75 | above 0.45 |
|---|---|---|---|---|
| MOSS 50-step shift-5 grid | 22 steps | 28 | 31 | 43 |
| Anima 28-step shift-3 grid | — | — | 14 | 22 |
| MOSS training σ draws (`TrainRequest` defaults) | 29 % | 45 % | 56 % | 92 % |

## Reading

The σ knee carries over from Anima; the step index does not (σ 0.75 is step 31 of 50 on
MOSS, step 14 of 28 on Anima). On MOSS the layout forms within the first 22–28 steps and
everything after step ~39 is the detail tail (< 0.2 normalised error). Training spends 45 %
of its draws above σ 0.80 and 8 % below σ 0.45.

For the levers set against Anima's shape:

- **DAVE τ** and **Spectrum `flex_window`** are defined in steps, not σ, so no Anima σ
  number sits in them. The shipped DAVE τ 0.10 covers steps 0–4 (σ 1.0 → 0.983), where the
  pooled error (`rel_low`) goes from 0.65 to 0.58; the layout knee at σ 0.87 is step 22.
- **The REPA σ 0.85 taper** (`_archive/moss_lora/repa.md`) fits MOSS's shape: at σ 0.85,
  `x0_pred` is still at 0.52 of the σ 0.95 error. A taper there cuts ~33 % of training
  draws.
- **Tier 3 §4 (text drive across σ)** should read its front-loaded window over σ ≥ 0.80
  (steps 0–28), not Anima's σ ≥ 0.85.

## Hypotheses (not tested)

- **A lower sampler shift would move steps from layout formation to the detail tail.**
  Shift 5 spends 22 of 50 steps above σ 0.87. On Anima, reshaping the σ schedule at fixed
  NFE was refuted (`_archive/bench/sigma_reshape/`), so expectations are low. Shift is a
  `GenerateRequest` knob: a `make gen`-scale A/B.
- **A detail- or style-oriented LoRA may be under-served by the default training σ.** Only
  8 % of draws fall below σ 0.45, where MOSS is still converging. The earlier MOSS LoRA round
  found style at high σ, which cuts the other way; an arm with a less noise-heavy
  `sigma_mu` would test it.
- **The knee is a property of the denoiser, not the sampler.** A schedule change moves which
  steps land on each side of it. Only training can move it, and since the knee is largely
  the SNR at which `x0` shows through `x_σ`, a σ reweighting would likely move it little.

## Spectrum's full-step schedule against the staircase

Given the layout knee at σ 0.87–0.80 (steps 22–28 of 50), are Spectrum's full forwards spent
in the right places? Same day, Base. Probe, runs and grids: `_archive/bench/moss_spectrum/`
(`probe_skip_cost.py`, `run_bench.py` with `full=` / `--example`; runs
`20261021-2300-skipcost-base5`, `20261021-2321-sched19`, `20261021-2329-space_ba`).

**Verdict: the shipped growing window stays.**

### Setting

The shipped schedule (ws 2, fw 0.5, warmup 6, tail 3) runs 19 full forwards in 50
steps: steps 0–5, then 7, 9, 12, 15, 19, 23, 28, 33, 39, 45, then 47–49. Six full
forwards land at σ 1.0–0.978, where the pooled `x0_pred` is still a blur, and four land
in the layout band σ 0.95–0.80.

### Observations

**Open loop points the wrong way.** `probe_skip_cost.py` replays 250 schedules on
recorded p50 trajectories (5 prompts) and scores `Σ |Δσ_i|·‖v̂_i − v_i‖`.

- The per-step error concentrates below σ 0.5, because the shift-5 grid's Δσ is large
  there.
- Spacing the full forwards evenly in log-SNR halves the score at NFE 19 (0.093 against
  the shipped 0.121).
- NFE 16 matches the shipped NFE 19 on this score.
- Within the window family the ranking agrees with closed loop (fw 0.25 < 0.5 < 1.0).
  Across families it does not.

**Closed loop: the shipped window wins.** Five schedules were run as Euler-50 arms.

| arm | NFE | 5 prompts, 1024² rel-L2 | Space 004 / 017 rel-L2 | grids |
|---|---|---|---|---|
| shipped | 19 | **0.401** | **0.619 / 0.441** | closest to p50 everywhere |
| warmup 4, two forwards moved to steps 11/13 | 19 | — | 0.627 / 0.446 | same images as shipped |
| log-SNR spacing | 19 | 0.455 | 0.654 / 0.450 | close; manga bubble and a hand change |
| step-index spacing | 19 | 0.587 | 0.748 / 0.582 | poses and compositions change (fridge, sunset, 017 right character) |
| log-SNR spacing | 16 | 0.578 | 0.789 / 0.530 | fridge layout, 004 bottom centre and 017 Hina move |

The 5-prompt set is `run_bench.PROMPTS` at seed 42 and cfg 3. The Space set is
`bench/moss_ab/space/` 004 and 017 at their own size, cfg and seed. The numbers are
mean latent rel-L2 against p50, which is a drift number. The grids are the verdict.

### Reading

- An error made above σ ≈ 0.87 grows into a composition change. A larger error made below
  σ 0.5 is absorbed as detail.
- The open-loop score assumes displacement errors carry through unchanged, so it
  weights the late steps and misses this.
- The shipped window is dense early and sparse late, which matches the staircase.
- Moving forwards out of warmup into the layout band changes nothing. The early
  forwards cost about the same wherever they fall before σ 0.92.
- Lowering the NFE to 16 costs layout. The speed-up stays at 2.6×.

### Not tested

- Reweighting the open-loop score by a measured per-σ propagation gain, from perturb
  and rollout runs.
- A capped forecaster history (`history`/K). At 2 MP the history costs about 1.7 GiB
  over p50 (12.5 GiB against 10.8 GiB), which fits 16 GB as it is.
