# RSD reconstruction — design & spec

Reconstruction of RSD (One-Step Residual Shifting Diffusion for Image SR via
Distillation, ICML 2026, paper `22490`). Their repo is a "Coming soon" placeholder,
so this is built from the paper (Algorithm 1 App. B; hyperparams App. C; schedule
App. J) on top of the vendored ResShift source (`sr/resshift/`). Proposal context:
`_archive/proposals/resshift_sr_sidecar.md` §3 Phase 2 (retired — shipped).

RSD = DMD2 distribution-matching distillation ported to ResShift. Same machinery
as our `turbo` DP-DMD loop (`docs/methods/turbo.md`, `networks/methods/turbo_dmd.py`)
— fake-critic + GAN + two-timescale alternation — over ResShift's residual-shift
forward (not flow-matching), in the VQ-f4 latent.

## Three networks (all UNetModelSwin, 174M)

| net | init | trainable | role |
|---|---|---|---|
| teacher `f*` | v2 s15 ckpt | frozen | 15-step ResShift, x0-pred |
| student `G_θ` | teacher weights + zero-init noise conv | yes | 1-step generator `(z_t, y0, t, ε)→ẑ0` |
| fake `f_φ` | teacher weights + zero-init noise conv | yes | critic: ResShift trained on student outputs (Eq 7); hosts GAN head |
| disc `D_ψ` | random | yes | small head off `f_φ` encoder bottleneck ([B,640,8,8]) |

Noise injection (App. C): ResShift's UNet is deterministic. To make `G_θ`/`f_φ`
stochastic one-step generators, add a separate zero-initialized conv on ε whose
output is added to `input_blocks[0]`'s output. Zero-init ⇒ at step 0 the student is
bit-identical to the teacher; ε-dependence is learned. (We subclass UNetModelSwin and
override forward; see `models.py::StochasticUNet`.)

## Teacher decision (2026-06-29)

Paper distills v1 (`resshift_realsrx4_s15_v1.pth`, the SinSR teacher). We use v2
(better-trained, same arch/schedule) by user choice → final student is stronger, but
RealSR numbers won't match the paper's Table 1 exactly. Reconstruction-correctness
signal becomes "1-step student beats the v2 teacher perceptually" (the paper's
central claim: RSD LPIPS 0.273 < teacher 0.360; MUSIQ 65.9 > 59.9 on RealSR) rather
than hitting the exact published digits.

## Official release reconciliation (2026-07-01)

The RSD authors released code (ICML 2026). It ships the orchestration only —
`trainer.py` / `sampler.py` / `main_distill.py` / config / data pipeline — and omits the
entire `models/` package (the DMD-core `training_student_losses` / `training_fake_losses`,
`UNetModelSwinGanWrapper`, and the noise-consuming UNet `forward`). SinSR (the base repo)
supplies only `UNetModelSwin` + `create_gaussian_diffusion`, not those RSD additions — so the
generator-gradient form and exact node timesteps remain unverifiable from public code. What
*is* observable (config + `trainer.py`) confirmed most of this reconstruction; two mechanisms
diverged and are now selectable (both default to backward-compatible behavior except SiD):

- SiD normalization (`train.py --sid_denom`): official (`trainer.py:1836`) divides the
  per-sample DMD loss by `|teacher_x0 − student_x0| + 1e-8`. Our original divided by
  `|z_t − teacher_x0|` floored 0.05. Default is now `student` (official); `input` keeps the old form.
- Noise injection (`--noise_mode`): official (`sampler.py:79`) widens the first input conv
  by `noise_channels=1` and concats eps; the original reconstruction added a separate 3-ch
  `noise_proj` after block0. **Default is now `concat`** (the release form) — it won a 500-iter A/B
  (`--weights student`, 30-img eval set): MUSIQ 65.8 (concat) vs 64.0 (add), both on the new
  `student` denom. `add` stays available for the original arch. Both are zero-init ⇒ identical to
  teacher at step 0. Back-compat is inference-side: `infer.py` rebuilds arch from ckpt meta
  (`noise_mode`, `noise_channels`), so pre-concat checkpoints (no meta) still load as `add`.

Confirmed matching (config): K=5, lr 5e-5, AdamW(0.9,0.95), wd 0, EMA 0.999, 3000 iters,
λ_gan 3e-3, λ_lpips 2.0, κ=2.0, exp-0.3/15-step/xstart, image-space LPIPS, w_t weighting off
(`normalize_generator_loss_by_t_power_ten: False`). Deliberate deltas retained: v2 teacher
(vs paper's v1) and art + light degradation (vs ImageNet + Real-ESRGAN). Our `dc_loss`
color term is an addition (official `mse_loss` is off).

## Forward process (ResShift, from `gaussian_diffusion.py`, verified)

- `q_sample(x0, y0, t)`: `x_t = η_t·(y0 − x0) + x0 + κ·√η_t·ε`  (Eq 4; `e0=y0−x0`).
- Schedule: `exponential`, power 0.3, `steps=15`, κ=2.0, min_noise 0.04, etas_end
  0.99, `predict_type='xstart'` (from `sr/resshift/configs/realsr_swinunet_realesrgan256.yaml`).
- Chain ends at `z_T ~ N(z_y, κ²η_T·I)`, η_T≈1 → `N(z_y, κ²I)` (Gaussian centered on
  the LR latent, not pure noise — the ResShift property).
- x0 prediction: `p_mean_variance(model, x_t, y0, t)['pred_xstart']`.
- VQGAN `VQModelTorch`: `encode(x)→[B,3,64,64]`, `decode(z, force_not_quantize=True)`,
  `scale_factor=1.0`. LR is bicubic-upsampled to HR size then encoded: `z_y=Enc(up(y0))`.

## Algorithm 1 (per iteration), two-timescale K=5

```
# K fake/critic updates per 1 generator update
for k in 1..K:
    sample (x0,y0); z0=Enc(x0); z_y=Enc(up(y0))
    t_n ~ U{t_1..t_N}; eps~N(0,I); z_{t_n} ~ q(z0, z_y, t_n)
    ẑ0_tn = G_θ(z_{t_n}, z_y, t_n, eps)            # stop-grad on θ here
    t ~ U{1..T}; z_t ~ q(ẑ0_tn, z_y, t)
    L_fake = ||f_φ(z_t, z_y, t)_x0pred − ẑ0_tn||²   # Eq 7 on generator data
    L_GAN_D = D-step on  D_ψ(f_φ.enc(ẑ0_tn,z_y,0)) [fake] vs D_ψ(f_φ.enc(z0,z_y,0)) [real]
    step φ on (L_fake + λ2·L_GAN_D);  step ψ

# 1 generator update
sample batch; ẑ0_tn = G_θ(z_{t_n}, z_y, t_n, eps); t~U{1..T}; z_t~q(ẑ0_tn,z_y,t)
L_θ = DMD2 grad:  push (f_φ(z_t)_x0 − f*(z_t)_x0) onto ẑ0_tn   # Eq 10, stop-grad form
z_T ~ N(z_y,κ²I); ẑ0 = G_θ(z_T, z_y, T, eps); L_LPIPS = LPIPS(x0, Dec(ẑ0))   # pixel space
L_GAN_G = −D_ψ(f_φ.enc(ẑ0_tn,z_y,0))
step θ on (L_θ + λ1·L_LPIPS + λ2·L_GAN_G)
```

Two timesteps: `t_n` (one of N nodes, drives student) and independent `t~U{1..T}`
(evaluates teacher+fake on the re-noised student output). DMD grad uses x0-predictions
of teacher vs fake at the same `z_t`. w_t weighting omitted (=1, DDPM-aligned, Eq 46).

DMD2 generator gradient (mirror `turbo_dmd`): with `g = (f_φ_x0 − f*_x0).detach()`
normalized per-sample by the teacher-derived magnitude `‖z_t − f*_x0‖` (mean-abs,
floored at 0.05 — SiD/DMD form, App C; not by `g`'s own norm, which would flatten
the distribution-matching signal and amplify critic noise once `f_φ≈f*`), apply via
`L_θ = 0.5·MSE(ẑ0_tn, (ẑ0_tn − g).detach())` so `dL/dẑ0 = g`.

## Hyperparameters (App. C)

λ1(LPIPS)=2, λ2(GAN)=3e-3, lr=5e-5 constant (student/fake/disc), AdamW
betas (0.9,0.95), K=5, N=4 multistep (evenly placed in [1,15] ending at 15;
exact set not enumerated → use {4,8,12,15} or {1,5,10,15} — pick & log), T=15,
~3000 generator iters (≈15k fake), no warmup/scheduler. Batch/EMA "same as SinSR"
(source from SinSR; default EMA 0.999). Inference always single-step from t=T.

## Data / validation (DECIDED: distill on our art directly)

- Train: HR 256² random crops from our art (`image_dataset/`), LR via bicubic ×4
  + light JPEG/blur (art-appropriate per proposal §1b; teacher stays in-distribution —
  Phase 0 showed v2 handles clean/bicubic LR on art well). `z_y = Enc(up_bicubic(LR))`.
  ResShift's full Real-ESRGAN GPU pipeline (`trainer.py:549`) is a later swap-in if the
  light degradation under-covers. User chose art over DIV2K/ImageNet (2026-06-29) →
  faster to a domain-relevant model; we lose the clean "loop vs domain" separation.
- Validation: reuse Phase-0 frozen eval set (`sr/data/{hr,lr}_eval`) → student 1-step vs
  v2 teacher (15-step) vs bicubic, PSNR/SSIM/LPIPS/CLIPIQA/MUSIQ. Gate = student beats
  teacher perceptually (LPIPS/MUSIQ), the paper's central claim.
- Eval: RealSR / RealSet65 (small standard sets) → PSNR/SSIM/LPIPS/CLIPIQA/MUSIQ (pyiqa).
- Repro targets (paper, v1 teacher): RealSR 1-step LPIPS 0.273 / CLIPIQA 0.706 / MUSIQ
  65.9; ImageNet-Test LPIPS 0.193. With v2 teacher, expect student to beat teacher.
- N ablation knee at N=4 (Table 5). Inference cost: NFE=1, 0.06s, 539 MB (paper).

## Feasibility — VRAM dry-run PASSED (2026-06-29)

`dry_run.py` builds all 5 nets + VQGAN + LPIPS-VGG and runs a real fake-update +
generator-update (backward + AdamW step). Peak VRAM on the 16 GB 5070 Ti:
bs=1 → 6.57 GB, bs=2 → 10.21 GB, bs=4 → OOM (fp32). So fp32 microbatch ≤2;
bf16 amp + grad-accum reaches the paper's effective batch. RSD fits. Note: the
realsr arch is 119M params/net (not the paper's quoted 174M) — v2 ckpt loads
strict=True into it, so the arch is correct; the 174M figure counts differently.

## Files

`models.py` (builders + StochasticUNet + disc head) · `rsd_diffusion.py` (q_sample /
sampling wrappers) · `losses.py` (L_θ/L_fake/L_GAN/L_LPIPS) · `data.py` (ResShift
degradation pipeline) · `train.py` (two-timescale loop, EMA, ckpt) · `dry_run.py`
(instantiate all + 1 step + VRAM gate). Make targets `sr-rsd-*`.

## Open / not-in-paper (must choose)

κ & η table → from upstream config (κ=2.0 ✓). Disc head arch → mirror DMD2 (small
conv head). Exact N=4 timesteps → choose+log. Batch/EMA → from SinSR.

## Throughput (2026-07-02 efficiency pass — measured, 16 GB 5070 Ti)

The loop was bs=2-eager-fp32-bound (GPU util 40–78%). Fixes shipped in `train.py`:
bf16 default (native-bf16 VQGAN, mirroring `infer.py` — the old `--amp` never
covered the 12 fp32 VQGAN encodes/step), fused 2B VQGAN encode (gt+lq), fused
`d_real`/`d_fake` GAN pass (was 2×5 serial `encode_features`/step), fused
node-path + LPIPS-path student forward (one 2B fwd+bwd), and class-level
block-compile (`rsd_models.compile_swin_blocks`). Marginal gen-steps/s × bs
(steps 5→25, 26-iter runs, `rsd_hr_cap4096`):

| config | it/s | rel. samples/s | peak VRAM |
|---|---|---|---|
| OLD loop, bs2+ckpt (3k-iter run log) | 0.586 | 1.17 | — |
| new, bs2+ckpt | 0.653 | 1.31 | 7.9 GB |
| new, bs4 no-ckpt (default) | 0.618 | 2.47 | 12.8 GB |
| new, bs6+ckpt | 0.419 | 2.51 | 13.8 GB |
| new, bs6 no-ckpt `--compile` | ≥0.499 | ≥2.99 | 14.8 GB |
| new, bs8 (any) | OOM | — | >15.5 GB |

≈2.1× the old loop at the bs4 default; ≈2.6× with `--bs 6 --compile` (thin headroom
— don't run alongside a desktop-heavy session). Key negative results: whole-graph
`torch.compile` is strictly worse than eager (Swin `window_partition` graph breaks;
tens of minutes of warmup since ckpt regions trace as higher-order ops), and
grad-ckpt costs ~25% at these tiny 256² activations — it's a fit-bigger-bs lever
only. Block-compile's win is the ~1.2 GB VRAM save (unlocks bs6-no-ckpt), not
per-step speed. NOT re-benched: dataloader headroom at the new rates (full ~4096px
PNG decode per 256² crop; raise `--num_workers` or pre-tile if starved). Batch
reuse across the K critic updates (5× fewer encodes) is designed but NOT built —
needs a 500-iter quality A/B first.
