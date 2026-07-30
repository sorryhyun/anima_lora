# E4 exercise launch — 2026-07-29 (single-seed, 2 artists × 3 arms)

Scope decision (user): exercise pass with **seed 1001 only** → 6 checkpoints
(full manifest still specs 3 seeds / 18 ckpts).

## Prereqs closed today

- `--sigma_lowres_route NATIVE:DEMOTE` landed (cli_args.py + train.py
  `_sigma_route`; default 1024:896 unchanged; pinned by
  `tests/test_sigma_lowres.py::TestSigmaRoute`). RNG-free — CRN pairing holds.
- 1024:768 demote siblings emitted for both artists (daemon job
  `20260729-210046-237f39`, 151 images, verified on all 75 train stems).
- 1024:896 siblings pre-existed for all 75 train stems (verified).
- NEW: `token_step_hist` — train.py accumulates a realized patch-token
  histogram per train step (post demote swap), merged into the `run_end`
  progress event → per-arm FLOPs = Σ hist[t]·FLOPs(t) with FLOPs(t) measured
  post-hoc (FlopCounterMode) per distinct token count.

## Recipe

`--method lora --preset default` (the tenth4s in-vivo recipe family), bs 1,
`--deterministic --paired_step_rng --seed 1001`. Steps equalized across
artists at 480: hews 8 epochs × 60 imgs, channel 32 epochs × 15 imgs.
Caption-variant sampling stays ON (shipped recipe); the draws ride the
seeded global stream and no arm-conditional code touches Python RNG, so
variants are identical across arms (shared nuisance, not a confound).

## Jobs (daemon, queued serially 2026-07-29 21:02)

| job | run |
|---|---|
| 20260729-210235-c8bc73 | e4_hews_native_s1001 |
| 20260729-210235-d7f628 | e4_hews_sigma896_s1001 (`--sigma_lowres --sigma_lowres_yarnsig`) |
| 20260729-210235-e07fff | e4_hews_unsafe768_s1001 (`--sigma_lowres --sigma_lowres_threshold 0.0 --sigma_lowres_route 1024:768`) |
| 20260729-210245-e89bbe | e4_channel_native_s1001 |
| 20260729-210245-593636 | e4_channel_sigma896_s1001 |
| 20260729-210245-d16ebb | e4_channel_unsafe768_s1001 |

## Eval (after training)

Per manifest `eval_protocol`: cfg 1.0 / 20 steps (the trainer CMMD
convention — keep for scored metrics), one render per (ckpt, prompt) at the
prompt's own bucket + gen_seed → 3×24 + 3×15 = 117 images. CMMD holdout /
member + noise floor + paired PE-Core cosine across arms. Optional add-on
discussed: small cfg≈4.5 eyeball grid (pre-register before rendering if
used). FLOPs from `token_step_hist` in each run's `run_end` event.

## Amendments (2026-07-29 evening)

- **SFW prompts**: eval prompts restricted to safe/sensitive bands
  (`e4_prompts_sfw.json`; hews 9, channel 12). NB this mismatches the
  all-ratings reference pools — for the paper, score on the FULL frozen
  prompt set and use the SFW subset for displayed figures only.
- s1001 cfg1/20 eval: `runs/20260729-2148-e4-eval-sfw-s1001/` (+ a cfg4/28
  arm-vs-arm-only pass in `...-sfw-cfg4-s1001/`). Render PNGs gitignored.
- **Grid completed**: s1002 + s1003 × 3 arms × 2 artists queued (jobs
  20260729-2209*), then cfg1/20 evals into `...-sfw-s1002/` `...-sfw-s1003/`,
  then `e4_seed_yardstick.py` → `runs/20260729-2148-e4-yardstick/` — tests
  whether cos(native~sigma896 | same seed) ≥ cos(native~native | cross seed)
  (demote footprint inside the seed lottery ⇒ sample-level differences are
  recipe noise).
- FLOPs (fixed, all 6 s1001 runs): native 8.66/8.62 PFLOPs fwd, sigma896
  7.29 (−15.8%), unsafe768 4.15/4.10 (−52%); wall tracks FLOPs ~1:1.
- **4th arm added — `896only`** (unconditional demotion on the SAFE route:
  `--sigma_lowres --sigma_lowres_threshold 0.0`, default 1024:896; the
  tenth4s sweep's 896only analogue): 3 seeds × 2 artists (jobs
  20260729-2229*). Isolates the σ-gate's contribution — sigma896 vs 896only.
  The queued 3-arm evals/yardstick were killed and requeued as 4-arm
  versions (evals resume-reuse existing renders; result.json rescored over
  all 4 arms). Final FLOPs pass over all 24 runs → yardstick dir.

## Amendments (2026-07-30)

- **5th arm added — `sigma768`** (σ>0.5 gate + yarnsig on the 1024→768
  route: `--sigma_lowres --sigma_lowres_route 1024:768
  --sigma_lowres_yarnsig`, threshold default 0.5): 3 seeds × 2 artists
  (jobs 20260730-0835*), recipe otherwise verbatim from sigma896.
  Rationale: unsafe768 (threshold 0, no yarnsig) differs from sigma896 in
  route AND gate AND yarnsig — sigma768 keeps the paper's gate + yarnsig so
  sigma896 vs sigma768 is a clean **route-only** comparison (and sigma768
  vs unsafe768 reads the gate+yarnsig contribution on the unsafe route).
  NB the sigma896-vs-896only "σ-gate isolation" comparison is still
  two-knob confounded (gate AND yarnsig) — 896only has no yarnsig.
