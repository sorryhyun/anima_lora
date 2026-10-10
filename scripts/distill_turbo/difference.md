# DMAD: official code vs `[dmad]` here

How the `[dmad]` path in this package (`dmad.py`, `steps.py::dmad_signal`,
`distill.py`) differs from the official DMAD training code
(<https://github.com/Yzmblog/DMAD>, read at `9a9f664`, 2026-10-09). The setup
history and results are in `docs/proposal/turbo_dmad.md`.

The reference is the **MiniMax-H3 recipe** (`train/h3/configs/dmad_minimax_h3.yaml`,
`train/h3/lightx2v_train/trainers/dmd/minimax_h3_dmad_trainer.py`,
`minimax_h3_trainer.py`). It is the paper's checkpoint and the closest match to
our setup: both student and critic are LoRAs. The Wan2.1
(`train/wan/rcm/configs/experiments/dmad/wan2pt1_t2v.py`) and SDXL
(`train/image/main/sd_guidance.py`) recipes train full models. They are cited
only where they add something H3 lacks. The "ours" column is the Phase 0c run
(`output/ckpt/anima_turbo_p0c_dmad_raw_cdm.snapshot.toml`).

## Student

| | Official (H3) | Ours (p0c) |
|---|---|---|
| Adapter | LoRA r128 / α128 on attn q/k/v/out + FF | LoRA r64 / α180 (adaLN modules r16). Initialized from the v1.1 DP-DMD student delta, SVD-projected from r96 |
| Optimizer | AdamW lr 4e-5, β (0, 0.99), wd 0.01, eps 1e-8 | AdamW lr 5e-5, β (0.9, 0.999) (default), wd 0 |
| LR schedule | constant, no warmup | 2 % warmup, then cosine to 0.1× (`primitives.make_scheduler`) |
| Grad clip | none (`max_grad_norm: 0`) | 1.0 |
| Rollout | DMD2 re-noise rule: each step's x0 estimate is re-noised with **fresh noise** to the next level. Intermediate levels are random draws from the critic's σ distribution, kept as a running min | Deterministic Euler ODE on a CDM dynamic grid (`sample_dynamic_sigmas`: N ~ U{n_min..4}, sorted-uniform interior, t₁ = 1). No re-noising |
| Step that gets the GAN signal | end step ~ U{0..N−1}. Only the last evaluation has grad; its x0 estimate is G | g ~ U{1..N−1} (`dmd_grad_step = "random"`). One-step x0 prediction `x_g − σ_g·v_g`. **Step 0 never gets the critic signal** (it is the diversity anchor) |
| Prompts | student step: a random pool caption | the batch caption |
| EMA | two power EMAs (σ_rel ≈ 0.10 / 0.05). The released student is the EMA | none; renders use live weights |
| Iterations / batch | 800 (LoRA critic), 8 GPUs × batch 1 | 1500, 1 GPU × batch 1 |

## Generator loss

| | Official (H3) | Ours (p0c) |
|---|---|---|
| Form | `L = −d_real(G) − (w_gap / E_w)·d_teacher(G)`, backpropagated through the frozen critic into the student. Weights 1 | Surrogate `(g · x_pred).mean()` with `g = ∂(−λ_T h_T − λ_R h_R)/∂x_pred`, λ_T = λ_R = 150 (`DmadDisc.student_signal`, under block checkpointing). `signal_rms = 0` leaves `g` raw (> 0 RMS-normalizes it per sample) |
| Scale | ∂L/∂G = −∂d/∂G / B | ∂L/∂x_pred = λ·(−∂h/∂x) / (B·C·H·W): `.mean()` over every latent element |
| Critic noise at the student query | the same σ distribution as the critic update (below) | τ ~ U(0, 1), fresh ε (`steps.dmad_signal`) |
| Gap routing | on: 10 CDF-decile bands of the σ draw. `w = sigmoid((median − gap_band) / 2.0)` with `gap = d_real(T) − d_real(Q)`, EMA 0.99. Starts once 5 bands have ≥ 10 updates | none (`gap_r` logged only) |
| Other student terms | none. GAN-only; the teacher is dropped from memory after setup | DP-DMD step-0 diversity anchor (`div_weight` 0.1, stop-grad after step 0) and **L_CDM** (`cdm_weight` 1.0). L_CDM takes the same disc input gradient at an off-trajectory extrapolation (`steps.cdm_off_trajectory_loss`) |

## Critic

| | Official (H3) | Ours (p0c) |
|---|---|---|
| Backbone | a second LoRA (r128 / α128) on a copy of the transformer. Blocks after the feature block are skipped | zero-init LoRA stack r64 / α180 (`make_aux_stack`) on the frozen teacher view |
| Feature tap | **last block** (49) | **middle block** (`feature_block_idx = −1` → 14 of 28) |
| Heads | per-token `RMSNorm(non-affine) → Linear(D, D) → SiLU → Linear(D, 1)`, one per head. Token logits mean-pooled (audio and video pooled separately, then averaged) | per-token `LayerNorm → Linear(D, D/2) → LeakyReLU(0.2) → Linear(D/2, 1)` (`TeacherFeatureDiscriminator`, `granularity="token"`), mean-pooled |
| Optimizer | critic LoRA: AdamW 4e-5, β (0, 0.99), wd 0.01. Heads: a separate AdamW with the same settings. Constant LR, no clip | one AdamW over stack + heads: 4e-5, β (0, 0.99), wd 0. Constant LR, no clip (`dmad_grad_clip = 0`) |
| Loss | `softplus(d_t(G)) + softplus(d_r(G)) + softplus(−d_t(Q)) + softplus(−d_r(T))` | the same four terms (`DmadDisc.accumulate`) |
| Samples per update | **its own fresh no-grad rollout** each critic step. A real clip is drawn first and its caption conditions the rollout | **the x_pred the student just trained on** (detached), plus a **replay window of the 4 newest pairs** re-scored every update |
| Teacher samples Q | precomputed offline: one 31-step teacher sample per caption (`data_process/gen_teacher_data.py`), independent of G's noise | online: the step-0 anchor rollout (12-step CFG grid from `k_anchor` 6) finished to σ = 0 (`finish_anchor_rollout`). **Shares the student's initial ε** |
| Real samples T | a real clip with the same caption as G and Q | the batch's training latent (same caption) |
| Critic σ | one draw `shift(U[0.02, 0.98], 12)` (video; audio shift 2), shared by G / T / Q. Independent ε per sample | τ ~ U(0, 1), **unshifted**, one (τ, ε) per pair. **The same ε** for the target and the student branch |
| Update ratio | 1 student : 1 critic (`fake_update_ratio: 1`) | 1 : 1 |
| Spectral norm | none in H3. Wan: frozen-gain SN on the critic's block + head Linears and on the generator's first half of blocks. SDXL: SN on the critic encoder and the generator encoders | none |
| Warmup | none | none in p0c (`dmad_disc_warmup_steps` 0; the knob exists) |

## Which H3 choices are H3-specific

Checked against the SDXL recipe (`train/image/experiments/sdxl/train_common.sh`,
`dmad_sdxl_4step.sh`, `main/train_sd.py`), an image model like ours.

**H3/video-specific (SDXL does otherwise):**

- **Critic σ shift 12.** This is H3's own video flow shift. SDXL draws
  timesteps uniform in [0, 1000) (`--critic_max_timestep 1000`), which is
  close to our uniform τ. Anima's equivalent of H3's choice would be shift 3
  (`flow_shift`), not 12.
- **Fresh rollout per critic step.** This comes from the LightX2V loop. SDXL
  trains the critic on the same detached generator output as the generator
  step (`train_sd.py::train_one_step`), as we do.
- **Caption-paired real and teacher samples.** SDXL draws real, teacher and
  prompt batches from independent loaders.
- **Separate audio/video token pooling and RMSNorm before the heads.** The
  RMSNorm is there for H3's ~1e4 residual stream.
- **`gap_tau` 2.0.** SDXL uses 0.5.

**Shared by H3, Wan and SDXL, and absent here:**

- Optimizer β1 = 0 on both sides (SDXL: generator 5e-7, critic 5e-7, β (0, 0.99)).
- Gap routing.
- Generator EMA.
- Offline teacher samples, independent of G's noise.
- DMD2 backward simulation with re-noising, where the trained step can be step 0
  (SDXL `--backward_simulation`: step ~ U{0..N−1} on the fixed grid).
- Critic features from the deepest stage: SDXL encoder + mid block, Wan and
  H3 last block.

**Spectral norm splits by setup.** SDXL and Wan (full-model generator and
critic) use it on both. H3 (LoRA generator and critic) does not.

**Ours only, in no official recipe:** the replay window, shared ε between Q
and G and within a critic pair, the middle-block tap, step 0 excluded from
the critic signal, a deterministic ODE rollout, β1 0.9 with cosine decay on
the student, and the λ / numel surrogate scale.

## Untested candidates for the p0c saturation collapse

These are differences that could bear on the critic saturating at ~300 and
staying saturated until ~980. None is measured. They are ordered by whether
any official recipe shares our choice.

- **Replay window** (ours only). The critic re-trains on the newest 4 pairs
  every step, all from the student it is scoring. Every official recipe uses
  only the current sample.
- **Shared noise** (ours only). Q and G start from the same ε, and the critic
  scores each pair at one shared ε. Official samples have independent noise.
- **Rollout and step 0** (ours only). Ours is a deterministic ODE rollout,
  and step 0 is held only by the diversity anchor. Official recipes re-noise
  between steps and put the critic signal on step 0 too.
- **Feature tap** (ours only). Middle block vs deepest stage.
- **Optimizer and schedule.** The student has β1 0.9 and cosine decay vs
  β1 0 everywhere officially. With the critic's gradient near zero, Adam
  keeps taking ~lr-sized student steps. Weight decay (H3, Wan 14B 0.01) would
  pull both LoRAs toward zero; ours has none.
- **Critic σ distribution.** Ours is uniform, as in SDXL. It is unlikely to
  matter alone; H3's shift 12 is a video-schedule choice.
- **Spectral norm.** Ours and H3 both lack it, so it does not separate this
  run from the paper's LoRA recipe. SDXL and Wan use it.
