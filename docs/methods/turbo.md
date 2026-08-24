# DP-DMD (Turbo Anima) — diversity-preserved few-step distillation

Distills the CFG=4 Anima teacher into a **few-step LoRA student** via
**Diversity-Preserved Distribution Matching Distillation** (Wu, Li, Zhang, Ma —
arXiv:2602.03139). The output is a **plain standard LoRA** — there is no
inference-side turbo code; you load it through the normal LoRA path and run
`--infer_steps` matched to `student_steps` (currently 4) with `--cfg 1.0` (CFG is
baked into the student during distillation).

> **History.** This replaced the CA-decoupled DMD2 ("CFG-as-Spear, Distribution-
> Matching-as-Shield", Liu et al. arXiv:2511.22677) objective on **2026-05-30**.
> The whole turbo program had been spent managing the CA branch's standing CFG
> bias (it never reaches a fixed point — see [[project_turbo_alpha4_overdistill]]),
> and every CA-side lever came back inert or harmful
> ([[project_turbo_fei_gap_phase0]], `ca_band`). DP-DMD removes the CA branch
> entirely. The structural walkthrough (diversity-anchor / DMD gradient split,
> flow-matching velocity↔x0 math, the per-step schedule) lives at
> `docs/structure/turbo.md`; the CA-era decision log survives at
> `_archive/proposals/dmd2_decoupled_improvements.md`. The original migration proposal
> is archived at `_archive/proposals/dpdmd.md`.

- **Training:** `scripts/distill_turbo/distill.py` — bespoke single-GPU loop
  (bypasses `train.py`/accelerate).
- **Harness:** `networks/methods/turbo_dmd.py::TurboDMDNetwork` — two `LoRANetwork`
  stacks (student + fake) view-toggled on one frozen DiT.
- **Config:** `configs/methods/turbo.toml` — **bespoke sectioned schema** read only
  by the script. Don't `print-config METHOD=turbo` (the flat method+preset merge
  doesn't apply here). CLI flags override TOML values.

## Quick start

```bash
make turbo                                       # configs/methods/turbo.toml defaults
make turbo ARGS="--student_rank 128 --iterations 5000"
make turbo ARGS="--single_prompt_idx 0"          # Phase 0 single-prompt overfit
make turbo PRESET=low_vram                        # grad ckpt + offload + sample_ratio

make test-turbo                                   # infer latest student LoRA @ 4 steps, cfg=1.0
```

A ready-made 4-step student is published at
**[huggingface.co/sorryhyun/anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step)**
— download the `.safetensors` and point `--lora_weight` (or `make test-turbo`'s
`output/ckpt/`) at it to run few-step inference without distilling your own.

`make turbo` honors `PRESET` (translates `blocks_to_swap` /
`gradient_checkpointing` / `sample_ratio` from `configs/presets.toml` into CLI
flags) and appends `ARGS` last so user overrides win. The output is
`output/ckpt/<output_name>.safetensors` — a normal LoRA — plus the standard
`.snapshot.toml` (and a per-run resolved-config snapshot in the TB log dir).

At inference the student LoRA loads through the existing LoRA adapter path; the
caller just sets the step count and CFG=1. It composes with concept LoRAs the way
LCM-LoRA composes with style LoRAs (linear LoRA composition, ranks add).

## How it works (one screen)

The student is a **genuine N-step Euler rollout**, role-separated by step. Linear
flow path `z_t = (1−t)·x + t·ε`, velocity `v = ε − x` — Anima's native schedule.
Three roles share one frozen DiT; the LoRA stacks toggle per forward via
`TurboDMDNetwork.set_view` (each `LoRAModule` short-circuits on `not self.enabled`,
so a view switch is an O(num_modules) flag flip, negligible vs a DiT forward):

```
teacher view  — both LoRA stacks off  → base velocity (CFG'd at α=teacher_cfg)
student view  — student on, fake off  → v_student (the rollout)
fake   view   — fake on, student off  → v_fake_cond_dm (the score tracker)
```

Per training step:

1. **Teacher K-step CFG anchor (no-grad).** From a shared noise `ε`, roll the
   *teacher* (CFG-guided, `v_u + α·(v_c − v_u)`) for `k_anchor` Euler steps on the
   `teacher_anchor_steps` grid to an intermediate latent `z_tk` at continuous time
   `t_k`. The first-step diversity target is `v_target = (ε − z_tk) / (1 − t_k)`.
   `t_k` is read from the **teacher** grid, not the student grid — a σ mismatch
   silently mis-scales `v_target`. This anchor is what de-collapses pose/composition
   diversity (the DMD mode-seeking collapse the old `dm_x0_norm` band-aid was
   fighting at the *symptom*).
2. **Student N-step rollout.** Step 0 (from `ε`, t=1) is **diversity-supervised**:
   `div_loss = ‖v_first − v_target‖²`. Under `detach_after_first` (load-bearing)
   the diversity term is backwarded immediately and the step-0 graph is severed —
   the DMD reverse-KL from later steps must **not** flow back into the diversity
   mapping (their Fig 5: preference rises while diversity falls without it). Steps
   2..N then carry the DMD-refine grad, routed by `grad_step` (also honored under
   the anchor — `[dmd].grad_step`): **`all`** rolls 2..N with grad (BPTT, holds the
   N-graph) onto the true endpoint `x_θ`; **`last`** (default) backward-simulates
   2..N−1 under no_grad and grads only the cleanest-σ final step onto `x_θ`
   (memory-flat, but the noisy refinement steps train only indirectly — and under
   `per_step_expert` only head N−1 trains); **`random`** samples one refinement step
   `g~U{1..N−1}`, backward-simulates the `1..g−1` prefix under no_grad from the
   post-anchor latent, and grads only step `g`'s **one-step x0-prediction**
   `x_g − σ_g·v_g` (memory-flat; supervises every refinement grid point + trains
   every head, at the cost of spreading the mode-seeking DMD grad across all
   refinement σ rather than concentrating it on the tail — A/B vs `last` for pose
   diversity; CMMD is blind to it).
3. **DMD on `x_θ` (no-grad teacher + no-grad fake, τ_DM ∈ [0,1]).** The real score
   is **CFG-guided** (`v_u + α·(v_c − v_u)`) — *not* cond-only. This is the one
   un-decoupling vs the CA-era code: the old DM branch was deliberately unguided
   because CFG lived in the separate CA branch; with CA gone, guidance has to ride
   the single DMD real score (matches the reference `compute_dmd_loss`). Without it
   `v_real ≈ v_fake` (`dm_cos ≈ 0.9999`) and the quality gradient is noise. The
   fake stays cond-only. `Δ_dm = v_real_cond_dm − v_fake_cond_dm`; the x0-space
   grad is `τ_dm·Δ_dm` (optionally per-sample x0-norm), applied via the DMD2 grad
   trick `loss = (grad_signal · x_θ).mean()`.
4. **Assemble + backward.** `loss = loss_dmd (+ div_weight·div_loss if not
   detached)`. `grad_clip` runs once on the accumulated
   student grad (diversity + DMD) either way.
5. **Fake update.** `fake_steps_per_student_step` plain flow-matching MSE steps on
   the student's `x_θ.detach()` distribution (resampling τ_fake, ε_fake each) —
   keeps the fake score tracker ahead of the moving x_θ.

The teacher uncond is the **T5("") sidecar** (`library/anima/uncond.py`), *not*
a zero tensor — a zero crossattn is fed-out-of-distribution and the resulting
`v_real_uncond` amplified at (α−1)=3× drives the student off-manifold (saturated
white output). Staged by `make preprocess-te`; shared with the
mod-guidance distill.

A **fake (critic) head-start** runs `fake_warmup_steps` fake-only updates before the
main loop, calibrating the fake against the student's `x_θ` distribution so the
critic is ready before the student LR warmup ramps — this kills the early
`grad_signal_rms` spike (~step 50). The student is untouched during it.

## Warm start from an extracted delta — the standard path

Both LoRA stacks are **seeded from a plain LoRA checkpoint** rather than cold-started
(`[network] student_init_weights` / `fake_init_weights`, both shipped on by default):
`warm_start_plain_lora` (`networks/methods/turbo_dmd.py`) reconstructs the file's exact
ΔW per runtime module (re-fusing defused q/k/v), SVD-truncates it to the stack's rank,
and writes `lora_down`/`lora_up`. File rank may differ from the network's in either
direction. Modules absent from the file keep their constructed init.

The shipped seed is the **official `anima_turboV10` release delta**, extracted against
the base DiT:

```bash
python scripts/toolkits/extract_delta_lora.py \
    --tuned models/diffusion_models/anima_turboV10.net.safetensors \
    --rank 96 --act_scales models/extracted/act_scales_base_4step.safetensors \
    --out models/extracted/anima_turboV10_delta_r96_asvd.safetensors
```

The `_asvd` file is the **activation-whitened** extraction (`--act_scales`; ASVD-style
functional truncation) — same rank and size as the plain SVD extraction, strictly
better capture (a rank-96 ASVD delta reconstructs about as well as a plain rank-128
one). Prefer it.

Why this and not a cold start: the official turbo is polished but **mode-collapsed**
([[project_official_turbo_v10_eval]]) — starting there hands the student the
few-step map for free and leaves DP-DMD's diversity anchor with the one job it is
actually good at, re-expanding the modes. It also sidesteps the OrthoInit /
plain-LoRA cold-start tangent problem entirely.

`fake_init_weights` normally points at the **same file**: at init the student *is*
the warm-start distribution, so a matched critic starts calibrated instead of
chasing it from zero.

**Constraints.** Warm start needs plain single-head Linear LoRA modules, so it is
mutually exclusive with `per_step_expert` and
`*_down_init = "weight_svd"` (config validation rejects the combination — set
`student_down_init = "kaiming"` when enabling). It runs after `.to(device)` (SVD
on-device) and **before** `compile_dit_blocks` traces the forwards. A MoE /
step-expert checkpoint is refused outright.

**It changes the length of a run.** A warm-started student is already a working
few-step map at step 0, so distillation is fine-tuning, not construction: the
shipped default is now **`iterations = 750`** (was 2000+ cold), and the 750-step
student renders excellently at `--infer_steps 4 --cfg 1.0`. Rank checkpoints by
rendered 4-step grids, not `fm_mse` ([[project_turbo_lr_instability_threshold]]).

## Config surface (`configs/methods/turbo.toml`)

Sectioned, bespoke. Every key has a matching CLI override flag (see
`scripts/distill_turbo/config.py` argparse). The shipped defaults:

| Section | Key | Default | Notes |
|---|---|---|---|
| top | `output_name` | `anima_turbo_T750` | output stem under `output/ckpt/` |
| top | `iterations` | `750` | short because the student is warm-started (see above) |
| top | `use_masked_loss` | `true` | **student-only** mask on the DMD grad; fake/critic stays full-frame |
| `[network]` | `student_rank` / `fake_rank` | `96` / `96` | `fake_rank ≥ student_rank` (fake is a score *tracker*, capacity ceiling on DM strength); matches the extracted delta's rank |
| `[network]` | `student_init_weights` / `fake_init_weights` | `…/anima_turboV10_delta_r96_asvd.safetensors` | **warm start, standard** — both stacks seeded from the extracted official delta |
| `[network]` | `fake_tau_banks` / `fake_tau_boundary` | `1` / `0.5` | τ-split critic — **line CLOSED 2026-07-20, leave at `1`** (`docs/findings/turbo_tau_critic_interference_lr_artifact.md`; the Phase-0 gate fired at tail LR then failed `G1` at peak LR, and Phase 1 was never gated). `2` = dual fake banks, bank 0 owning `τ < boundary`; updates AND DMD queries route by drawn τ (matched compute by construction). Requires `batch_size=1`; `1` = byte-identical shipped loop. Memory +1 fake LoRA (params/grads/Adam); consider `fake_warmup_steps` ×2 |
| `[dmd]` | `student_steps` (N) | `4` | Euler steps the student rolls; inference matches (`--infer_steps 4`) |
| `[dmd]` | `teacher_cfg` (α) | `4` | CFG scale baked into the teacher anchor + DMD real score (Anima prod CFG=4) |
| `[dmd]` | `grad_step` | `random` | which refinement step(s) carry the DMD grad: `all` (BPTT) / `last` (tail-only, memory-flat) / `random` (one-step x0-pred at `g~U{1..N−1}`, memory-flat, trains every head). Honored under **both** `base_loss`. |
| `[dmd]` | `dynamic_schedule` | `true` | CDM-style continuous schedule (arXiv:2605.06376 §3.2; line plan): per-iteration random rollout grid — length `N~U{2..student_steps}` (dpdmd; `{1..}` plain dmd), interior anchors continuous-uniform, `t₁=1` pinned so the DP anchor composes unchanged. Trains v_θ over continuous t instead of the 4 fixed inference points. `false` = bit-identical legacy fixed-grid loop; validation renders + inference always stay on the static `flow_shift` grid. Incompatible with `per_step_expert` (heads keyed to fixed grid steps). |
| `[dmd]` | `dm_x0_norm` | `true` | per-sample x0-space magnitude normalization of the DM grad ([[project_turbo_dmd_x0_norm_wins]]) |
| `[dmd]` | `norm_floor` | `0.05` | clamp_min for the `dm_x0_norm` denominator (latent scale) |
| `[dpdmd]` | `k_anchor` (K) | `6` | teacher steps rolled to the diversity anchor |
| `[dpdmd]` | `teacher_anchor_steps` | `12` | teacher σ-grid the K is counted against |
| `[dpdmd]` | `div_weight` (λ) | `0.05` | weight on the first-step diversity MSE |
| `[dpdmd]` | `detach_after_first` | `true` | **load-bearing** stop-grad after step 1; keep True (A/B only) |
| `[optim]` | `student_lr` / `fake_lr` | `1e-5` / `2e-5` | fake runs hotter; **do not raise the student to 2e-5** — adversarial instability ([[project_turbo_lr_instability_threshold]]) |
| `[optim]` | `fake_steps_per_student_step` | `4` | keep the fake ahead of the moving x_θ |
| `[optim]` | `fake_warmup_steps` | `50` | fake (critic) head-start before the main loop — kills the early grad_signal_rms spike (~step 50); `0` = off |
| `[optim]` | `grad_clip` | `1.0` | grad-norm cap (both nets) |
| `[sampling]` | `t_distribution` | `uniform` | τ sampling for the fake update + warmup (or `sigmoid`) |
| `[sampling]` | `flow_shift` | `2.0` | σ-schedule shift for the student/teacher Euler grids (matches inference) |
| `[gan]` | `weight_gen` | `0.03` (**on**) | teacher-feature GAN generator term — see below |
| `[gan]` | `delay_steps` / `warmup_steps` | `0` / `0` | generator-side λ ramp: hold at 0 for `delay_steps` (disc still trains), then linear 0 → `weight_gen` over `warmup_steps`. Mandatory for `disc_head="token"` on a collapsed warm start — unramped dense logits froze pose at the init's mode (2026-07-18) |

Validation enforces `student_steps ≥ 2` (step 1 is diversity-supervised + detached,
so at least one further step must carry the DMD loss) and
`1 ≤ k_anchor < teacher_anchor_steps`.

**Anchor fidelity — why the defaults dropped `14/28 → 4/8` (2026-06-01).** The shipped
`k_anchor`/`teacher_anchor_steps` have since settled at `6/12` — the same 0.5 σ-fraction,
bought back a little integration fidelity. The A/B that motivated the drop: both
ratios anchor at the *same* σ-fraction (`14/28 = 4/8 = 0.5`), so the diversity
anchor lands at the same continuous time; the only change is how coarsely the
teacher integrates to it (4 Euler forwards vs 14 — the anchor rollout gets ~3.5×
cheaper). A/B'd at 500 steps, sigmoid τ, `div_weight=0.05`, every other knob
identical (logs `20260531-144835` k14/t28 vs `20260601-121104` k4/t8): training
metrics are a **wash**. `dm_cos` (~0.979), `dm_mag_ratio` (~0.99), and `dm_rel_gap`
(~0.18–0.19) are flat within run-to-run noise; `div_loss` is equal-to-marginally
*lower* under k4/t8 (tailμ 0.093 vs 0.095); no instability spike. The only
systematic difference is `v_student_rms` / `x_pred_std` sitting ~1–2% higher in the
low-k run — the variance-inflation / over-bake lean ([[project_turbo_alpha4_overdistill]],
[[project_turbo_dmd_x0_norm_wins]]) — but well inside noise at this length.

Caveat before reading this as "lower K is free": **`div_loss` measures how well the
student *hits* the anchor, not how *diverse* the anchor is.** A k4 anchor is a
coarser, smoother target, so equal-or-lower `div_loss` does not prove the diversity
injection survived — a less-faithfully-integrated anchor can land off the teacher's
true trajectory, which these scalars can't see. The anchor's whole job is pose
de-collapse on real captions, and that only shows in sample grids (the PE-pooled
metric is blind to pose — [[project_dpdmd_pivot_phase0]]). Verdict on the lowered
defaults is therefore metrics-green / grid-pending: A/B `anima_turbo_J500_500`
(k4/t8) vs `anima_turbo_I_sigmoid_500` (k14/t28) at `--infer_steps 2 --cfg 1.0` and
read pose diversity + saturation, not the scalars.

## Inference: step count

The student is trained at `student_steps` (currently 4 — was 2 until `5ef128d`), so
`--infer_steps 4 --cfg 1.0` is the matched schedule. **However**, an under-trained /
lightly-distilled student behaves like a continuous velocity field (the DMD quality
loss is trained at *random* τ, not on the N-step grid; only step 0 is
grid-anchored), so it can integrate **better** at more Euler steps than it was
trained for — the 2-step era made this concrete: a single `0.75→0` Euler jump
crosses the entire detail-forming band below σ≈0.5
([[project_sigma_signal_resolves_by_045]]), while 4 steps get a function evaluation
at σ=0.5 *and* preserve the σ=0.75 anchor (one motivation for the 2→4 move). If a
checkpoint looks better at more steps than its trained grid, that's the tell that
distillation hasn't reached a true N-step map yet — train longer or raise
`student_steps`. Always keep `--cfg 1.0` regardless of step count (CFG is baked;
don't double-guide).

## Per-step expert (`per_step_expert`, default off)

One rank-`student_rank` LoRA normally absorbs two conflicting gradients: the
**diversity** loss on step 0 (`div_loss = MSE(v_first, v_target)`, then a detach)
and the **DMD** reverse-KL on steps 1..N. The detach already severs the two
backward graphs, so `per_step_expert=true` splits the student into one **shared
`lora_down`** plus **K = `student_steps` up-heads** (`StepExpertLoRAModule`),
selecting head `k` for denoise step `k` by the step counter — no router (the step
index is known at call time, unlike FeRA's FEI/σ case). Head 0 then sees only the
diversity gradient, head k only step-k's DMD gradient; only the shared down-proj is
trained by both. Per-step inference compute is unchanged (one head active per step).

Turn it on in `[network]` (`per_step_expert = true`) or `--per_step_expert`. Treat
it as a **hypothesis test vs the single-head student**, not a presumed win: if the
shared LoRA was never capacity/interference-bound it buys a heavier checkpoint +
inference plumbing for nothing. Promote only if it beats baseline on the CMMD val
signal ([[project_cmmd_val_signal]]) with visibly preserved step-0 diversity.

### What it costs — the plain-LoRA property is gone

This is the load-bearing trade. The shipped single-head turbo is a **normal LoRA**:
it merges into the DiT (`make merge`), loads through any stock LoRA path, and that
simplicity *is* the headline. A per-step-expert student is **not**:

- **`make merge` refuses it** — K per-step heads can't fold into one static DiT
  weight (it would need K baked copies). It's caught by the `.lora_ups.` non-bakeable
  marker, same as Hydra moe.
- **Kept-live only.** Inference rebuilds a router-free `StepExpertLoRAModule` network
  on the (fused-qkv) DiT and selects the head per step — CLI via
  `set_step_index(i)` in the denoise loop (the loader keys off the
  `ss_turbo_per_step_expert` metadata stamp), ComfyUI via the dedicated
  `AnimaTurboPerStepExpertLoader` node (stock LoRA / `AnimaAdapterLoader` raise,
  since they can't drive step-indexed head selection).
- **`make test-turbo` pins `--infer_steps` to the trained head count K** (read from
  metadata); head k binds to step k, so `infer_steps` must equal K. Overshoot repeats
  the last (quality) head; undershoot skips it. Keep `--cfg 1.0`.

Escape hatch if the shared down-proj becomes a compromise between the two
objectives: per-head down (doubles params, removes sharing) — documented, not v0.

## Reading the metrics

Trigger fake interventions on **`dm_rel_gap` ↑ / `dm_cos` ↓**, *not* on `fake_loss`
↑ (a rising fake loss against a moving, sharpening student is expected
equilibrium). Watch `div_loss` fall as the student's step-1 velocity converges on
the teacher anchor. The live TB scalars:

| TB scalar | Read |
|---|---|
| `div_loss` | `‖v_first − v_target‖²` — first-step diversity MSE (pre-weight). Falling = step-1 velocity converging on the diverse teacher anchor. |
| `dm_rel_gap` | `rms(τ·Δ_dm)/rms(τ·v_real_dm)` — fraction of the teacher score the gap still is. ↑ = fake lagging. |
| `dm_cos` | `cos(v_fake_dm, v_real_dm)` — →1 healthy; ↓ = fake pointing the wrong way (worse than a magnitude miss). |
| `dm_mag_ratio` | `rms(v_fake)/rms(v_real)` — ≈1 healthy. |
| `x_pred_std` / `v_student_rms` | collapse → 0 or runaway up = student exploding (`v_student_rms` leads). |
| `gan_gen_loss` / `gan_disc_loss` | softplus-hinge generator / discriminator losses (pre-weight); 0 when the GAN is off. |

### Where to read them from

Three surfaces, cheapest first — none of them need a TensorBoard export:

- **`make run-status`** — `step N/total`, it/s, ETA, last losses, last checkpoint,
  and whether the run is `RUNNING` / `OK` / `ERROR` / `DEAD` (no `run_end` and the
  pid is gone). Defaults to the newest run; `RUN=<output_name>` picks one,
  `ARGS="--list"` reports all, `ARGS="--json"` for a machine-readable digest. It
  reads `output/logs/<output_name>.progress.jsonl` — the same structured stream
  the GUI and daemon tail (`--no_log` disables it).
- **The console** — the tqdm postfix on a TTY; redirected or daemon-headless, the
  same numbers come out as a greppable `step N/total … it/s … ETA` line at
  `log_interval` cadence (`grep '^.*step ' run.log`).
- **`make export-logs RUN=output/logs/<run> SUMMARY=1`** — max step + last value
  per TB tag, when you want every scalar rather than the loop's headline set.

## GAN + f-distill (FastGen levers)

DP-DMD is structurally **DMD2 with the GAN amputated**. Two levers port the missing
adversarial machinery from NVlabs FastGen (`_archive/proposals/turbo_gan.md`) — the
GAN now ships **on** at the FastGen `weight_gen = 0.03`; f-distill stays off:

- **Teacher-feature GAN** (`[gan] weight_gen > 0`, FastGen idea 1). A tiny
  discriminator (`networks/methods/turbo_dmd.py::TeacherFeatureDiscriminator`,
  ~2M params) reads the **frozen teacher DiT's** mid-block activations — captured
  with a compile-safe forward hook on `blocks[feature_block_idx]` (default middle).
  Two head granularities (`disc_head`): `"pooled"` (v0 default) mean-pools each
  tap's tokens to one logit per tap; `"token"` (LADD-style) applies the **same**
  MLP to every token for dense per-patch logits — identical parameters, so resume
  bundles load across a switch; a pooled global logit only constrains global
  statistics and saturates easily, while the dense head penalizes local defects
  (glyphs, hands, texture). The generator term `softplus(−disc(feat))` is added to
  the student loss; the disc trains on the fake/critic cadence with its own AdamW
  (`disc_lr`, betas (0, 0.99)), optional approximate-R1 (`r1_weight`; per-token
  MSE under `"token"`). The student output stays a **plain LoRA** (the disc is
  discarded at save, like the fake). FastGen QwenImage recipe: `weight_gen=0.03`,
  `use_same_t_noise=true`, middle block, `disc_lr=1e-5`.

  **Token-head caveat (2026-07-18):** at full λ from step 0 on a collapsed warm
  start (turboV10 init), the dense per-token gradient pins the student's
  pose/layout at the init's modal composition — the collapsed image is already
  locally "real" (zero GAN grad) while any pose excursion is penalized at every
  token position, so the DM + div terms recover appearance axes but never the
  layout. Use `delay_steps`/`warmup_steps` to hold the generator-side λ at 0
  through the escape window (~500 steps) and ramp in after. **Observability:**
  the hinge losses are blind to this (both arms sit at equilibrium ≈0.69/1.39) —
  read `train/gan_disc_margin` (mean real − fake logit) and
  `train/gan_logit_spread` (per-token logit std) instead, plus the cross-seed
  `ac_sim` diversity validation (set `validate_every_n_steps` ≤ 250 for short
  smokes — at the default 1000 it never fires in a 500-step run);
  `train/gan_weight_gen_eff` traces the ramp itself.
- **f-distill reweighting** (`[f_distill] f_div != "rkl"`, FastGen idea 2). Scales
  the DMD signal per-sample by `h = f'(r)`, `r = exp(disc logits)` (free from the
  GAN head). Requires `weight_gen > 0`. `"rkl"` ≡ uniform h ≡ plain DMD2 (no-op).
  Targets mode-collapse — **bench against the diversity anchor; they may not be
  additive** (decision gate 2).

**Cost (honest).** Without the idea-3.1 feature-tap API there is no early-exit, so
the GAN adds **+1 grad-bearing teacher forward** in the student step (the generator
term must flow grad through the teacher into `x_pred`) and **+2 no_grad teacher
forwards** per disc step. Consider `--grad_ckpt` when the GAN is on. `weight_gen=0`
keeps the entire path off → byte-identical DP-DMD (no disc, no hooks, no extra
forwards). **Decision gate 1:** A/B `weight_gen` 0 vs 0.03 at fixed seed/data/steps,
2-step `--cfg 1.0`, ship only on a CMMD/A-B win without diversity collapse (reuse
`diversity.py`).

## Limitations & composition

- **Plain-LoRA bake is the hard constraint.** Anything needing a step-size or
  per-t input at inference (Shortcut / MeanFlow Δt-conditioning, timestep-conditioned
  T-LoRA — its mask is training-only, see [[project_tlora_inference_full_rank]])
  gives nothing after the bake.
- **Spectrum:** incompatible by construction — Spectrum's Chebyshev cache assumes
  ≥16 steps. Don't stack.
- **Mod guidance:** tunable — the distilled `pooled_text_proj` may still help, but a
  turbo student may have re-learned the modulation pathway implicitly. Test, don't
  assume.
- **Block swap.** The student rolls `N` forwards and the teacher `K` anchor
  forwards per step (multi-forward); the offloader desyncs on a 2nd DiT forward
  ([[project_blockswap_extra_forwards_gradcache]]). The loop calls
  `prepare_block_swap_before_forward(free_cache=False)` before each forward, but the
  default path keeps `blocks_to_swap=0` (activation-dtype LoRA GEMMs keep
  activation memory low enough to run full-res on 16 GB without swap). Audit the
  multi-forward offloader path before turning swap on.

## References

- Wu, Li, Zhang, Ma — arXiv:2602.03139, *Diversity-Preserved Distribution Matching
  Distillation*. Reference impl: `dpdmd/train_sd35_dpdmd.py` (SD3.5-M, flow-matching).
- `_archive/proposals/dpdmd.md` — the migration proposal (Phase 0 GO, the
  pose-vs-pooled-cosine metric caveat, the depth-m fallback).
- `docs/structure/turbo.md` — structural walkthrough: the diversity-anchor / DMD
  gradient split, the flow-matching velocity↔x0 conversion, and the sign convention.
- `_archive/proposals/dmd2_decoupled_improvements.md` — CA-era decision log; the record
  of why the CA branch was abandoned.
- `docs/findings/asymflow_parameterization.md` — Anima's `u = ε − x0` velocity path
  (the conversion the renoise/grad-assembly relies on).
