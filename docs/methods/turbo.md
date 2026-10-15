# DP-DMD (Turbo Anima) — diversity-preserved few-step distillation

Distills the CFG=4 Anima teacher into a few-step LoRA student via
Diversity-Preserved Distribution Matching Distillation (Wu, Li, Zhang, Ma —
arXiv:2602.03139). The output is a plain standard LoRA — there is no
inference-side turbo code; you load it through the normal LoRA path and run
`--infer_steps` matched to `student_steps` (currently 4) with `--cfg 1.0` (CFG is
baked into the student during distillation).

Structural walkthrough: `docs/structure/turbo.md`.

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
make turbo ARGS="--student_rank 128 --iterations 5000"
make turbo ARGS="--single_prompt_idx 0"          # single-prompt overfit
make turbo PRESET=low_vram                        # grad ckpt + offload + sample_ratio

make test-turbo                                   # infer latest student LoRA @ 4 steps, cfg=1.0
```

A ready-made 4-step student is published at
[huggingface.co/sorryhyun/anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step)
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

The student is a genuine N-step Euler rollout, role-separated by step. Linear
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

1. Teacher K-step CFG anchor (no-grad). From a shared noise `ε`, roll the
   teacher (CFG-guided, `v_u + α·(v_c − v_u)`) for `k_anchor` Euler steps on the
   `teacher_anchor_steps` grid to an intermediate latent `z_tk` at continuous time
   `t_k`. The first-step diversity target is `v_target = (ε − z_tk) / (1 − t_k)`.
   `t_k` is read from the teacher grid, not the student grid — a σ mismatch
   silently mis-scales `v_target`. This anchor is what de-collapses pose/composition
   diversity (the DMD mode-seeking collapse).
2. Student N-step rollout. Step 0 (from `ε`, t=1) is diversity-supervised:
   `div_loss = ‖v_first − v_target‖²`. Under `detach_after_first` (load-bearing)
   the diversity term is backwarded immediately and the step-0 graph is severed —
   the DMD reverse-KL from later steps must not flow back into the diversity
   mapping (their Fig 5: preference rises while diversity falls without it). Steps
   2..N then carry the DMD-refine grad, routed by `grad_step` (also honored under
   the anchor — `[dmd].grad_step`): `all` rolls 2..N with grad (BPTT, holds the
   N-graph) onto the true endpoint `x_θ`; `last` (default) backward-simulates
   2..N−1 under no_grad and grads only the cleanest-σ final step onto `x_θ`
   (memory-flat, but the noisy refinement steps train only indirectly);
   `random` samples one refinement step
   `g~U{1..N−1}`, backward-simulates the `1..g−1` prefix under no_grad from the
   post-anchor latent, and grads only step `g`'s one-step x0-prediction
   `x_g − σ_g·v_g` (memory-flat; supervises every refinement grid point, at the cost of spreading the mode-seeking DMD grad across all
   refinement σ rather than concentrating it on the tail).
3. DMD on `x_θ` (no-grad teacher + no-grad fake, τ_DM ∈ [0,1]). The real score
   is CFG-guided (`v_u + α·(v_c − v_u)`) — not cond-only — since the DMD real
   score is the only place guidance enters (matches the reference
   `compute_dmd_loss`). Without it
   `v_real ≈ v_fake` (`dm_cos ≈ 0.9999`) and the quality gradient is noise. The
   fake stays cond-only. `Δ_dm = v_real_cond_dm − v_fake_cond_dm`; the x0-space
   grad is `τ_dm·Δ_dm` (optionally per-sample x0-norm), applied via the DMD2 grad
   trick `loss = (grad_signal · x_θ).mean()`.
4. Assemble + backward. `loss = loss_dmd (+ div_weight·div_loss if not
   detached)`. `grad_clip` runs once on the accumulated
   student grad (diversity + DMD) either way.
5. Fake update. `fake_steps_per_student_step` plain flow-matching MSE steps on
   the student's `x_θ.detach()` distribution (resampling τ_fake, ε_fake each) —
   keeps the fake score tracker ahead of the moving x_θ.

The teacher uncond is the T5("") sidecar (`library/anima/uncond.py`), not
a zero tensor — a zero crossattn is fed-out-of-distribution and the resulting
`v_real_uncond` amplified at (α−1)=3× drives the student off-manifold (saturated
white output). Staged by `make preprocess-te`; shared with the
mod-guidance distill.

A fake (critic) head-start runs `fake_warmup_steps` fake-only updates before the
main loop, calibrating the fake against the student's `x_θ` distribution so the
critic is ready before the student LR warmup ramps — this kills the early
`grad_signal_rms` spike (~step 50). The student is untouched during it.

## Warm start from an extracted delta — the standard path

Both LoRA stacks are seeded from a plain LoRA checkpoint rather than cold-started
(`[network] student_init_weights` / `fake_init_weights`, both shipped on by default):
`warm_start_plain_lora` (`networks/methods/turbo_dmd.py`) reconstructs the file's exact
ΔW per runtime module (re-fusing defused q/k/v), SVD-truncates it to the stack's rank,
and writes `lora_down`/`lora_up`. File rank may differ from the network's in either
direction. Modules absent from the file keep their constructed init.

The default seed is the official turbo v1.1 release delta (adaln included),
extracted against the base DiT:

```bash
python scripts/toolkits/extract_delta_lora.py \
    --tuned models/diffusion_models/anima-turbo-v1.1.safetensors \
    --rank 96 --act_scales models/extracted/act_scales_base_4step.safetensors \
    --include_adaln \
    --out models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors
```

The `_asvd` file is the activation-whitened extraction (`--act_scales`; ASVD-style
functional truncation) — same rank and size as the plain SVD extraction, strictly
better capture (a rank-96 ASVD delta reconstructs about as well as a plain rank-128
one). Prefer it.

The official turbo is a working few-step map but mode-collapsed; starting there
leaves DP-DMD's diversity anchor to re-expand the modes, and avoids the
plain-LoRA cold-start tangent.

`fake_init_weights` normally points at the same file: at init the student *is*
the warm-start distribution, so a matched critic starts calibrated instead of
chasing it from zero.

Constraints. Warm start needs plain single-head Linear LoRA modules, so it is
mutually exclusive with `*_down_init = "weight_svd"` (config validation rejects the combination — set
`student_down_init = "kaiming"` when enabling). It runs after `.to(device)` (SVD
on-device) and before `compile_dit_blocks` traces the forwards. A MoE
checkpoint is refused outright.

A warm-started student is already a few-step map at step 0, so distillation is
fine-tuning rather than construction. Rank checkpoints by rendered 4-step grids,
not `fm_mse`.

## Config surface (`configs/methods/turbo.toml`)

Sectioned, bespoke. Every key has a matching CLI override flag (see
`scripts/distill_turbo/config.py` argparse). The shipped defaults:

| Section | Key | Default | Notes |
|---|---|---|---|
| top | `output_name` | `anima_turbo_v3_longer` | output stem under `output/ckpt/` |
| top | `iterations` | `10000` | |
| top | `use_masked_loss` | `false` | student-only mask on the DMD grad; fake/critic stays full-frame. Needs masks (`make mask`) |
| `[network]` | `student_rank` / `fake_rank` | `64` / `64` (alpha `180` / `180`) | `fake_rank ≥ student_rank` (fake is a score tracker, capacity ceiling on DM strength); the warm start SVD-truncates the r96 delta to this rank |
| `[network]` | `train_adaln` / `adaln_rank` / `adaln_alpha` | `true` / `16` / `90` | adaln LoRA on student + fake (`docs/methods/adaln.md` §Turbo) |
| `[network]` | `student_init_weights` / `fake_init_weights` | `…/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors` | both stacks seeded from the extracted official delta |
| `[network]` | `fake_tau_banks` / `fake_tau_boundary` | `1` / `0.5` | τ-split critic. `1` = single fake. `2` = dual fake banks, bank 0 owning `τ < boundary`; updates AND DMD queries route by drawn τ (matched compute by construction). Requires `batch_size=1`; memory +1 fake LoRA (params/grads/Adam). Showed no gain over `1` (`docs/findings/turbo_tau_critic_interference_lr_artifact.md`) |
| `[dmd]` | `student_steps` (N) | `4` | Euler steps the student rolls; inference matches (`--infer_steps 4`) |
| `[dmd]` | `teacher_cfg` (α) | `4` | CFG scale baked into the teacher anchor + DMD real score (Anima prod CFG=4) |
| `[dmd]` | `grad_step` | `random` | which refinement step(s) carry the DMD grad: `all` (BPTT) / `last` (tail-only, memory-flat) / `random` (one-step x0-pred at `g~U{1..N−1}`, memory-flat). Honored under both `base_loss`. |
| `[dmd]` | `dynamic_schedule` | `true` | CDM-style continuous schedule (arXiv:2605.06376 §3.2): per-iteration random rollout grid — length `N~U{2..student_steps}` (dpdmd; `{1..}` plain dmd), interior anchors continuous-uniform, `t₁=1` pinned so the DP anchor composes unchanged. Trains v_θ over continuous t instead of the 4 fixed inference points. `false` = bit-identical legacy fixed-grid loop; validation renders + inference always stay on the static `flow_shift` grid. |
| `[dmd]` | `dm_x0_norm` | `true` | per-sample x0-space magnitude normalization of the DM grad |
| `[dmd]` | `norm_floor` | `0.05` | clamp_min for the `dm_x0_norm` denominator (latent scale) |
| `[dpdmd]` | `k_anchor` (K) | `6` | teacher steps rolled to the diversity anchor |
| `[dpdmd]` | `teacher_anchor_steps` | `12` | teacher σ-grid the K is counted against |
| `[dpdmd]` | `div_weight` (λ) | `0.1` | weight on the first-step diversity MSE |
| `[dpdmd]` | `detach_after_first` | `true` | **load-bearing** stop-grad after step 1 |
| `[optim]` | `student_lr` / `fake_lr` | `5e-5` / `5e-5` | stable on the warm start but near the edge — expect an excursion around ~1k and keepers from 2k+. A cold start is unstable with the student above 1e-5. `fm_mse` is anti-correlated with quality; rank by rendered 4-step samples |
| `[optim]` | `fake_steps_per_student_step` | `4` | keep the fake ahead of the moving x_θ |
| `[optim]` | `fake_warmup_steps` | `200` | fake (critic) head-start before the main loop — kills the early grad_signal_rms spike (~step 50); `0` = off |
| `[optim]` | `grad_clip` | `1.0` | grad-norm cap (both nets) |
| `[sampling]` | `t_distribution` | `uniform` | τ sampling for the fake update + warmup (or `sigmoid`) |
| `[sampling]` | `flow_shift` | `3.0` | σ-schedule shift for the student/teacher Euler grids (matches inference; `k_anchor` σ is shift-coupled, so change both together) |
| `[gan]` | `weight_gen` | `0.03` (on) | teacher-feature GAN generator term — see below |
| `[gan]` | `disc_head` | `token` | `pooled` (code default) / `token` — see below |
| `[gan]` | `delay_steps` / `warmup_steps` | `0` / `0` | generator-side λ ramp: hold at 0 for `delay_steps` (disc still trains), then linear 0 → `weight_gen` over `warmup_steps` |

Validation enforces `student_steps ≥ 2` (step 1 is diversity-supervised + detached,
so at least one further step must carry the DMD loss) and
`1 ≤ k_anchor < teacher_anchor_steps`.

`k_anchor / teacher_anchor_steps` sets the σ-fraction the diversity anchor lands
at (`6/12 = 0.5`); the step count sets only how finely the teacher integrates to
it, and its cost. `div_loss` measures how well the student hits the anchor, not
how diverse the anchor is — pose diversity shows only in sample grids (the
PE-pooled metric is blind to pose).

## Inference: step count

The student is trained at `student_steps` (currently 4), so
`--infer_steps 4 --cfg 1.0` is the matched schedule. An under-trained /
lightly-distilled student behaves like a continuous velocity field (the DMD quality
loss is trained at random τ, not on the N-step grid; only step 0 is
grid-anchored), so it can integrate better at more Euler steps than it was
trained for. At 2 steps a single `0.75→0` Euler jump crosses the entire
detail-forming band below σ≈0.5
(`docs/findings/sigma_signal_where_anima_resolves.md`); 4 steps get a function
evaluation at σ=0.5 and keep the σ=0.75 anchor. If a
checkpoint looks better at more steps than its trained grid, that's the tell that
distillation hasn't reached a true N-step map yet — train longer or raise
`student_steps`. Always keep `--cfg 1.0` regardless of step count (CFG is baked;
don't double-guide).

## Reading the metrics

Trigger fake interventions on `dm_rel_gap` ↑ / `dm_cos` ↓, *not* on `fake_loss`
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

Three surfaces, cheapest first:

- `make run-status` — `step N/total`, it/s, ETA, last losses, last checkpoint,
  and whether the run is `RUNNING` / `OK` / `ERROR` / `DEAD` (no `run_end` and the
  pid is gone). Defaults to the newest run; `RUN=<output_name>` picks one,
  `ARGS="--list"` reports all, `ARGS="--json"` for a machine-readable digest. It
  reads `output/logs/<output_name>.progress.jsonl` — the same structured stream
  the GUI and daemon tail (`--no_log` disables it).
- The console — the tqdm postfix on a TTY; redirected or daemon-headless, the
  same numbers come out as a greppable `step N/total … it/s … ETA` line at
  `log_interval` cadence (`grep '^.*step ' run.log`).
- `make export-logs RUN=output/logs/<run> SUMMARY=1` — max step + last value
  per TB tag, when you want every scalar rather than the loop's headline set.

## GAN + f-distill (FastGen levers)

DP-DMD is structurally DMD2 without the GAN. Two levers port the adversarial
machinery from NVlabs FastGen (`_archive/proposals/turbo_gan.md`); the GAN is on
by default at the FastGen `weight_gen = 0.03`, f-distill is off:

- Teacher-feature GAN (`[gan] weight_gen > 0`, FastGen idea 1). A tiny
  discriminator (`networks/methods/turbo_dmd.py::TeacherFeatureDiscriminator`,
  ~2M params) reads the frozen teacher DiT's mid-block activations — captured
  with a compile-safe forward hook on `blocks[feature_block_idx]` (default middle).
  Two head granularities (`disc_head`): `"pooled"` (code default) mean-pools each
  tap's tokens to one logit per tap; `"token"` (LADD-style) applies the same
  MLP to every token for dense per-patch logits — identical parameters, so resume
  bundles load across a switch; a pooled global logit only constrains global
  statistics and saturates easily, while the dense head penalizes local defects
  (glyphs, hands, texture). The generator term `softplus(−disc(feat))` is added to
  the student loss; the disc trains on the fake/critic cadence with its own AdamW
  (`disc_lr`, betas (0, 0.99)), optional approximate-R1 (`r1_weight`; per-token
  MSE under `"token"`). The student output stays a plain LoRA (the disc is
  discarded at save, like the fake). FastGen QwenImage recipe: `weight_gen=0.03`,
  `use_same_t_noise=true`, middle block, `disc_lr=1e-5`.

  The token head contributes no disc→student gradient early in a warm-started
  run, so it does not cause a pose freeze; `fake_adaln = true` does. The hinge
  losses sit at equilibrium (≈0.69/1.39) either way — read
  `train/gan_disc_margin` (mean real − fake logit) and `train/gan_logit_spread`
  (per-token logit std), plus the cross-seed `ac_sim` diversity validation
  (`validate_every_n_steps` defaults to 1000, so it never fires in a 500-step
  run); `train/gan_weight_gen_eff` traces the ramp.
- f-distill reweighting (`[f_distill] f_div != "rkl"`, FastGen idea 2). Scales
  the DMD signal per-sample by `h = f'(r)`, `r = exp(disc logits)` (free from the
  GAN head). Requires `weight_gen > 0`. `"rkl"` ≡ uniform h ≡ plain DMD2 (no-op).
  Targets mode collapse.

Cost. There is no early exit after the tapped block, so the GAN adds +1 grad-bearing teacher forward in the student step (the generator
term must flow grad through the teacher into `x_pred`) and +2 no_grad teacher
forwards per disc step. `--grad_ckpt` is a valid VRAM lever here, with one loop invariant: `set_view` is global state read at forward time and checkpoint recompute defers to `.backward()`, so every checkpointed forward must be backwarded while its own view is still live (the loop enforces this in `steps.py`; nothing in the metrics catches a violation — the recompute silently runs under the wrong view). `weight_gen=0`
keeps the entire path off → byte-identical DP-DMD (no disc, no hooks, no extra
forwards).

## Soft-rank caption auxiliary (`[softrank]`, default off)

The 4-step student follows prompts worse than the teacher
(`bench/turbo/results/20260611-2020-phase0-turboN1250/`). `[softrank]` adds a
differentiable ranking term: every `every_n` steps the student also
denoises the same noise under `k` caption-negatives (no-grad), and a pairwise-sigmoid
soft-rank loss (`scripts/distill_turbo/softrank.py`) pushes the matched caption to
rank first. It joins the split-backward diversity pass, so `weight=0` is
byte-identical DP-DMD.

| Key (`[softrank]` / CLI) | Default | Meaning |
|---|---|---|
| `weight` / `--softrank_weight` | 0 (off) | Loss weight |
| `k` / `--softrank_k` | 2 | Caption negatives per step (≥2; each is one extra student forward) |
| `every_n` / `--softrank_every_n` | 4 | Apply on every n-th step |
| `softness` / `--softrank_softness` | 0.1 | Sigmoid temperature (→0 = hard rank) |
| `pool_size` / `--softrank_pool_size` | 2 | Negative-caption pool |

Regression probe: `bench/turbo/caption_ranking_probe.py --adapter <ckpt>`. Tests:
`tests/test_turbo_softrank.py`. Design: `_archive/proposals/turbo_caption_ranking.md`.

## Loop gotchas no metric catches

- **A key in the wrong TOML section silently defaults.** `toml_get(cfg, "dmd.dynamic_schedule", …)`
  ignores a misplaced key with no warning. Verify every arm via the `ss_turbo_*` safetensors metadata or the `.snapshot.toml`, and
  read the snapshot's `# source config:` line — GUI/daemon runs read
  `configs/gui-methods/custom/turbo.toml`, not `configs/methods/turbo.toml`.
- **NFE=2 GAN runaway.** At `student_steps=2` the token disc can run away (`gan_margin`
  0.22→6.94, `gan_spread`→0.99) and collapse the student to noise with every DM term healthy;
  the same config at NFE=4 stays flat. Signature: at NFE=4 the token head has an equilibrium
  plateau of 0.27–0.30 by step 2–3k, so a rising spread alone is not the tell — runaway = sustained
  bin-median >0.40 **plus** `gan_disc_margin` >0.5 **plus** `gan_disc_loss` below ~1.1 (healthy
  1.33–1.36). Damp with `gan_r1_weight ≈ 100` and/or `gan_disc_lr` 1e-5→3e-6; do not cut
  `fake_steps_per_student_step` (disc cadence is coupled to the fake inner loop).
- **`gan_logit_spread` and `cdm_grad_rms` are hard-zero under `gan_disc_head="pooled"`** — never
  baseline a token-head run against a pooled one.
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
  forwards per step (multi-forward); the offloader desyncs on a 2nd DiT forward
  (`library/runtime/offloading.py::ModelOffloader`). The loop calls
  `prepare_block_swap_before_forward(free_cache=False)` before each forward, but the
  default path keeps `blocks_to_swap=0` (activation-dtype LoRA GEMMs keep
  activation memory low enough to run full-res on 16 GB without swap). The
  multi-forward offloader path is not exercised by the default config.

## References

- Wu, Li, Zhang, Ma — arXiv:2602.03139, *Diversity-Preserved Distribution Matching
  Distillation*. Reference impl: `dpdmd/train_sd35_dpdmd.py` (SD3.5-M, flow-matching).
- `_archive/proposals/dpdmd.md` — the DP-DMD proposal.
- `docs/structure/turbo.md` — structural walkthrough: the diversity-anchor / DMD
  gradient split, the flow-matching velocity↔x0 conversion, and the sign convention.
- `_archive/proposals/dmd2_decoupled_improvements.md` — CA-era decision log; the record
  of why the CA branch was abandoned.
- `docs/findings/asymflow_parameterization.md` — Anima's `u = ε − x0` velocity path
  (the conversion the renoise/grad-assembly relies on).
