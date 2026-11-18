# DP-DMD (Turbo Anima) — diversity-preserved few-step distillation

Distills the CFG=4 Anima teacher into a few-step LoRA student via
Diversity-Preserved Distribution Matching Distillation (Wu, Li, Zhang, Ma —
arXiv:2602.03139). The output is a plain LoRA: load it through the normal LoRA path
and run `--infer_steps` matched to `student_steps` (currently 4) with `--cfg 1.0`
(CFG is baked in).

Structure and math: `docs/anima/structure/turbo.md`.

- Training: `scripts/distill_turbo/distill.py` — bespoke single-GPU loop
  (bypasses `train.py`/accelerate).
- Harness: `networks/methods/turbo_dmd.py::TurboDMDNetwork` — two `LoRANetwork`
  stacks (student + fake) view-toggled on one frozen DiT.
- Config: `configs/methods/turbo.toml` — bespoke sectioned schema read only
  by the script. Don't `print-config METHOD=turbo` (the flat method+preset merge
  doesn't apply here). CLI flags override TOML values.

## Quick start

```bash
make turbo                                       # configs/methods/turbo.toml defaults
make turbo ARGS="--int8 off --stream bf16"       # bf16 base
make turbo ARGS="--student_rank 128 --iterations 5000"
make turbo ARGS="--single_prompt_idx 0"          # single-prompt overfit
make turbo PRESET=graft                           # a preset's blocks_to_swap / sample_ratio / gradient_checkpointing carry over

make test-turbo                                   # infer latest student LoRA @ 4 steps, cfg=1.0
```

A ready-made 4-step student is published at
[huggingface.co/sorryhyun/anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step);
point `--lora_weight` (or `make test-turbo`'s `output/ckpt/`) at its `.safetensors`.
The published student was trained with terms the loop no longer has (the GAN, L_CDM, MFD, the
transition-aware fake and the dynamic schedule); `make turbo` now runs the base DP-DMD loop only.
A TOML that sets a removed key to a non-default value is refused (`scripts/distill_turbo/config.py::_REMOVED_KEYS`).

`make turbo` honors `PRESET` (translates `blocks_to_swap` /
`gradient_checkpointing` / `sample_ratio` from `configs/presets.toml` into CLI
flags) and appends `ARGS` last so user overrides win. The output is
`output/ckpt/<output_name>.safetensors` plus the standard `.snapshot.toml` (and a
per-run resolved-config snapshot in the TB log dir). The student composes with
concept LoRAs linearly (ranks add), like LCM-LoRA with a style LoRA.

## Per training step

Three views share one frozen DiT, toggled per forward by `TurboDMDNetwork.set_view`:
teacher (both LoRA stacks off), student (student on), fake (fake on).

1. Teacher K-step CFG anchor (no grad): `k_anchor` Euler steps on the
   `teacher_anchor_steps` grid from a shared `ε` give `v_target = (ε − z_tk) / (1 − t_k)`.
2. Student rollout. Step 0 (from `ε`, σ = 1) regresses on `v_target`; that term
   is backwarded at once and the graph severed. The loop draws a grad step `g ~ U{1..N−1}`, runs steps `1..g−1` under
   no_grad, and grads only step `g`'s x0-prediction `x_g − σ_g·v_g`.
3. DMD on `x_θ`: CFG-guided teacher against the cond-only fake at `τ_DM ~ U[0,1]`,
   applied as `loss = (grad_signal · x_θ).mean()`. `grad_clip` runs once on the
   accumulated student grad.
4. Fake update: `fake_steps_per_student_step` flow-matching steps on `x_θ.detach()`,
   each on fresh (τ, ε).

Derivation, sign convention and the view chain: `docs/anima/structure/turbo.md`.

The teacher uncond is the T5("") sidecar (`library/anima/uncond.py`), not a zero
tensor: a zero crossattn is out of distribution, and its `v_real_uncond` amplified at
(α−1)=3× drives the student off-manifold (saturated white output). Staged by
`make preprocess-te`; shared with the mod-guidance distill.

A fake (critic) head-start runs `fake_warmup_steps` fake-only updates before the main
loop, so the critic is calibrated before the student LR warmup ends; it removes the
early `grad_signal_rms` spike (~step 50). The student is untouched during it.

## Warm start from an extracted delta (optional)

The shipped recipe cold-starts from `weight_svd`. `[network] student_init_weights` /
`fake_init_weights` instead seed the stacks from a plain LoRA checkpoint (Hub v3 was
trained this way; on merged14 a warm start from Hub v3 regressed under the v1 recipe,
`_archive/anima_turbo/reports/warm_v3.md`, local). `warm_start_plain_lora`
(`networks/methods/turbo_dmd.py`) reconstructs the file's exact ΔW per runtime module
(re-fusing defused q/k/v), SVD-truncates it to the stack's rank, and writes
`lora_down`/`lora_up`. File rank may differ from the network's in either direction.
Modules absent from the file keep their constructed init.

The usual seed is the official turbo v1.1 release delta (adaln included), extracted
against the base DiT:

```bash
python scripts/toolkits/extract_delta_lora.py \
    --tuned models/diffusion_models/anima-turbo-v1.1.safetensors \
    --rank 96 --act_scales models/extracted/act_scales_base_4step.safetensors \
    --include_adaln \
    --out models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors
```

Prefer the `_asvd` file (activation-whitened extraction, `--act_scales`): same rank and
size as the plain SVD extraction, and a rank-96 ASVD delta reconstructs about as well as
a plain rank-128 one. The official turbo is a working few-step map but mode-collapsed;
DP-DMD's anchor has to re-expand the modes.

`fake_init_weights` normally points at the same file, so the critic starts calibrated to
the warm-started student.

Constraints. Warm start is mutually exclusive with `*_down_init = "weight_svd"`
(config validation rejects it; set `student_down_init = "kaiming"`). It runs after
`.to(device)` (SVD on-device) and before `compile_dit_blocks` traces the forwards. A
MoE / step-expert checkpoint is refused. Rank warm-started checkpoints by rendered
4-step grids, not `fm_mse`.

## Config surface (`configs/methods/turbo.toml`)

Sectioned, bespoke. Every key is also a CLI flag named after its `TurboConfig` field
(`--student_steps` for `dmd.student_steps`; `scripts/distill_turbo/config.py`), and
the flag wins over the TOML. The shipped defaults:

| Section | Key | Default | Notes |
|---|---|---|---|
| top | `output_name` | `anima_turbo` | output stem under `output/ckpt/` |
| top | `iterations` | `1000` | the v1 length (`_archive/anima_turbo/1112_v1.md`, local); Hub v3 ran 10k from a warm start |
| top | `dit_path` / `dit_nonquant_path` | int8 GPTQ file / bf16 DiT | forward copies and bf16 weights (`docs/anima/optimizations/int8.md` § Forward copies) |
| top | `int8` / `stream` | `stream` / `fp32` | teacher, student and fake forwards on the int8 base and the fp32 residual stream (`docs/anima/optimizations/int8.md`, `fp32_stream.md`). The output is a normal LoRA either way. Code defaults are `off` / `fp32` |
| `[network]` | `student_rank` / `fake_rank` | `64` / `64` (alpha `180` / `180`) | `fake_rank ≥ student_rank` (fake is a score tracker, capacity ceiling on DM strength); the warm start SVD-truncates the r96 delta to this rank |
| `[network]` | `train_adaln` / `adaln_rank` / `adaln_alpha` | `true` / `16` / `90` | adaln LoRA on student + fake (`docs/anima/methods/adaln.md` §Turbo) |
| `[network]` | `student_down_init` / `fake_down_init` | `weight_svd` / `weight_svd` | cold start: LoRA down from the base weight's top singular directions |
| `[network]` | `student_init_weights` / `fake_init_weights` | `""` (off) | warm start from an extracted delta (below); excludes `weight_svd` |
| `[dmd]` | `student_steps` (N) | `4` | Euler steps the student rolls; inference matches (`--infer_steps 4`) |
| `[dmd]` | `teacher_cfg` (α) | `4` | CFG scale baked into the teacher anchor + DMD real score (Anima prod CFG=4) |
| `[dmd]` | `dm_x0_norm` | `true` | per-sample x0-space magnitude normalization of the DM grad |
| `[dmd]` | `norm_floor` | `0.05` | clamp_min for the `dm_x0_norm` denominator (latent scale) |
| `[dpdmd]` | `k_anchor` (K) | `7` | teacher steps rolled to the diversity anchor; must end on the student's σ₁ (below) |
| `[dpdmd]` | `teacher_anchor_steps` | `28` | teacher σ-grid the K is counted against |
| `[dpdmd]` | `div_weight` (λ) | `0.1` | weight on the first-step diversity MSE; `0` skips the teacher anchor (its `2·k_anchor` teacher forwards per step) and the diversity term |
| `[dpdmd]` | `anchor_only` | `false` | train step 0 on the anchor regression alone: no rollout, no DM, no fake update or warmup |
| `[dpdmd]` | `anchor_cache` | `""` | directory of cached anchor targets (below); the run then trains on the cached samples only |
| `[optim]` | `student_lr` / `fake_lr` | `5e-5` / `5e-5` | under the cosine (to 0.1×) from the `weight_svd` cold start; 2e-4 rendered worse (`_archive/anima_turbo/reports/runs.md`, `1112_1110`, local). `fm_mse` is anti-correlated with quality; rank by rendered 4-step samples |
| `[optim]` | `fake_steps_per_student_step` | `4` | keep the fake ahead of the moving x_θ |
| `[optim]` | `fake_warmup_steps` | `200` | fake (critic) head-start before the main loop — kills the early grad_signal_rms spike (~step 50); `0` = off |
| `[optim]` | `grad_clip` | `1.0` | grad-norm cap (both nets) |
| `[optim]` | `lr_schedule` / `grad_accum` | `cosine` / `1` | shape after the 2 % linear warmup (student, fake); iterations per student optimizer step |
| `[ema]` | `window` | `0.2` (code fallback `0.0`) | student EMA span as a fraction of the run (0 = off); the EMA is the checkpoint that ships. Under `lr_schedule = "constant"` it is a fraction of the updates; under `cosine`, of the summed LR (each update decays the shadow by `decay ** (lr / mean lr)`; 0.2 reaches back about 45 % of the updates) |
| `[sampling]` | `t_distribution` | `uniform` | τ sampling for the fake update + warmup (or `sigmoid`) |
| `[sampling]` | `flow_shift` | `3.0` | σ-schedule shift for the student/teacher Euler grids (matches inference; the `k_anchor` σ is shift-coupled, so change both together) |

Validation enforces `student_steps ≥ 2` and `1 ≤ k_anchor < teacher_anchor_steps`.
`div_weight = 0` leaves step 0 with no loss (a warning, not
a refusal: step 0 then stays at its init).

`k_anchor / teacher_anchor_steps` sets the σ the diversity anchor lands at; the step
count sets only how finely the teacher integrates to it, and its cost. The anchor target
is the teacher's mean velocity over σ [σ_K, 1], and step 0 applies it over [σ_1, 1], so K
has to end on the student's first boundary σ_1: at 4 steps and shift 3 the grid is σ 1 → 0.9
→ 0.75 → 0.5 → 0, so 7 of 28 lands on σ 0.9 (the earlier 6 of 12 and 14 of 28 both ended at
σ 0.75 while step 0 covered only 1 → 0.9). `setup.build_run` logs a warning when σ_K ≠ σ_1. `div_loss` measures how well the student hits the anchor, not
how diverse the anchor is — pose diversity shows only in sample grids (the
PE-pooled metric is blind to pose).

### Anchor target cache

The anchor target depends only on the prompt, the noise and the frozen teacher, so it can be
rolled once. `python -m scripts.distill_turbo.anchor_cache --out <dir> --prompts N --draws K
[--k_anchor 7 --teacher_anchor_steps 28 …]` (a daemon job, resumable) writes K targets for each of
N samples of `data_dir`; the noise of a pair is regenerated from a seed hashed from (seed, sample
key, draw), so only the target is stored, and a cache built at another `k_anchor` with the same
seed uses the same noise. A run reads it with `dpdmd.anchor_cache = "<dir>"`; the manifest's
teacher fields (`k_anchor`, `teacher_anchor_steps`, `flow_shift`, `teacher_cfg`, the DiT) have to
match the run's. With `anchor_only` an iteration is then one student forward and backward.

## Inference: step count

The student is trained at `student_steps` (currently 4), so
`--infer_steps 4 --cfg 1.0` is the matched schedule. An under-trained
student behaves like a continuous velocity field (the DMD loss is trained at random
τ; only step 0 is grid-anchored), so it can integrate better at more Euler steps
than it was trained for. At 2 steps a single `0.75→0` Euler jump crosses the entire
detail-forming band below σ≈0.5
(`docs/anima/findings/sigma_signal_where_anima_resolves.md`); 4 steps get function
evaluations at σ 0.75 and σ 0.5. If a
checkpoint looks better at more steps than its trained grid, that's the tell that
distillation hasn't reached a true N-step map yet — train longer or raise
`student_steps`. Keep `--cfg 1.0` at any step count (CFG is baked in).

## Reading the metrics

Trigger fake interventions on `dm_rel_gap` ↑ / `dm_cos` ↓, not on `fake_loss` ↑ (a
rising fake loss against a sharpening student is the expected equilibrium). The DMD
surrogate loss value is a gradient vehicle with a random sign, not a loss to read. The
live TB scalars:

| TB scalar | Read |
|---|---|
| `div_loss` | `‖v_first − v_target‖²` — first-step diversity MSE (pre-weight). Falling = step-1 velocity converging on the diverse teacher anchor. |
| `dm_rel_gap` | `rms(τ·Δ_dm)/rms(τ·v_real_dm)` — fraction of the teacher score the gap still is. ↑ = fake lagging. |
| `dm_cos` | `cos(v_fake_dm, v_real_dm)` — →1 healthy; ↓ = fake pointing the wrong way (worse than a magnitude miss). |
| `dm_mag_ratio` | `rms(v_fake)/rms(v_real)` — ≈1 healthy. |
| `x_pred_std` / `v_student_rms` | collapse → 0 or runaway up = student exploding (`v_student_rms` leads). |

### Where to read them from

- `make run-status` — `step N/total`, it/s, ETA, last losses, last checkpoint,
  and whether the run is `RUNNING` / `OK` / `ERROR` / `DEAD` (no `run_end` and the
  pid is gone). Defaults to the newest run; `RUN=<output_name|job id>` picks one,
  `ARGS="--list"` reports all, `ARGS="--json"` for a machine-readable digest. The
  stream is `output/daemon/jobs/<id>/progress.jsonl` for a daemon job (`make turbo
  --queue`, `make daemon-run`) and `output/logs/<output_name>.progress.jsonl` for an
  inline run (`--no_log` disables it).
- The console — the tqdm postfix on a TTY; redirected or daemon-headless, the
  same numbers come out as a greppable `step N/total … it/s … ETA` line at
  `log_interval` cadence (`grep '^.*step ' run.log`).
- `make export-logs RUN=output/logs/<run> SUMMARY=1` — max step + last value
  per TB tag, when you want every scalar rather than the loop's headline set.
- `grad_steps.jsonl` in the TB run dir — one row per iteration: pre-clip grad norms
  (student, fake) beside that iteration's DM query σ_a, grad-step σ and `g`. The TB `grad_norm_*` scalars are interval means and cannot be
  paired with the draw.

## Loop gotchas

- **A key in the wrong TOML section silently defaults.** `toml_get(cfg, "dmd.student_steps", …)`
  ignores a misplaced key with no warning. Verify every arm via the `ss_turbo_*` safetensors metadata or the `.snapshot.toml`, and
  read the snapshot's `# source config:` line — GUI/daemon runs read
  `configs/gui-methods/custom/turbo.toml`, not `configs/methods/turbo.toml`.
- **Diversity can plateau with every DM scalar healthy** — pose diversity shows only in grids.

## Limitations & composition

- Plain-LoRA bake is the hard constraint. Anything needing a step-size or
  per-t input at inference (Shortcut / MeanFlow Δt-conditioning, timestep-conditioned
  T-LoRA — its mask is training-only)
  gives nothing after the bake.
- Spectrum: incompatible by construction — Spectrum's Chebyshev cache assumes
  ≥16 steps. Don't stack.
- Mod guidance: not characterized on turbo students (they also move the adaln
  modulation pathway the guidance steers through).
- Block swap. The student rolls `N` forwards and the teacher `K` anchor
  forwards per step, with graphs alive across them. Each forward calls
  `rewind_block_swap()`, which undoes the last forward's moves on the same device
  slots, and a backward first seeks to where it starts
  (`library/runtime/offloading.py::ModelOffloader`, `rewindable`). The default
  config keeps swap off: it costs about 1.5x step time for 1 GB of peak. No full
  student has trained on it.

## References

- Wu, Li, Zhang, Ma — arXiv:2602.03139, *Diversity-Preserved Distribution Matching
  Distillation*. Reference impl: `dpdmd/train_sd35_dpdmd.py` (SD3.5-M, flow-matching).
- `_archive/proposals/dpdmd.md` — the DP-DMD proposal.
- `docs/anima/structure/turbo.md` — structural walkthrough: the diversity-anchor / DMD
  gradient split, the flow-matching velocity↔x0 conversion, and the sign convention.
- `_archive/proposals/dmd2_decoupled_improvements.md` — CA-era decision log.
