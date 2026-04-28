# APEX — Self-Adversarial One-Step Distillation

Turn a pretrained Anima LoRA into a 1-NFE (or 2–4 NFE) generator by querying the same DiT under a shifted text condition as an endogenous adversarial reference. No discriminator, no external teacher, no architectural change to the DiT.

Paper: Liu et al., *Self-Adversarial One Step Generation via Condition Shifting* (arXiv:2604.12322, Apr 2026). We implement the paper's method, **not** the upstream `LINs-lab/APEX` code — the released code uses TwinFlow-style time-sign shifting which requires an intrusive DiT change. Condition-space shifting is a clean fit for Anima's existing `crossattn_emb` hook point.

## Quick start

```bash
make apex                           # default preset
python tasks.py apex                # cross-platform
```

The default config warm-starts from a prior T-LoRA checkpoint (see the note on warm-start requirement below) and trains for 2 epochs with `apex_warmup_ratio=0.20`, `apex_rampup_ratio=0.10`. Block swap is forced off: APEX runs 3 DiT forwards per step and block swap is incompatible with the multi-forward pattern (see §"Memory and block swap").

## What APEX actually is

Standard flow-matching training uses a single supervised term

$$\mathcal{L}_{\text{sup}} = \|\mathbf{F}_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\mathbf{z} - \mathbf{x})\|^2$$

against the OT velocity target. APEX adds a second branch that queries the **same network** with an affine-shifted condition

$$\mathbf{c}_{\text{fake}} = \mathbf{A}\mathbf{c} + \mathbf{b}$$

and uses it as an adversarial reference. The fake-branch query at `(x_t, t)` is stop-gradiented and becomes the target for a **mixed consistency** loss on the real branch (Eq. 23–24):

$$\mathbf{T}_{\text{mix}} = (1-\lambda)\mathbf{v}_{\text{data}} + \lambda \cdot \text{sg}(\mathbf{F}_\theta(\mathbf{x}_t, t, \mathbf{c}_{\text{fake}}))$$

$$\mathcal{L}_{\text{mix}} = \|\mathbf{F}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{T}_{\text{mix}}\|^2$$

Separately, the fake branch is trained to track the model's *current* one-step fake distribution via an endpoint predictor $\mathbf{x}^{\text{fake}} = \mathbf{x}_t - t\mathbf{F}_\theta$ (Eq. 11) and a fresh OT interpolation (Eq. 12):

$$\mathcal{L}_{\text{fake}} = \|\mathbf{F}_\theta(\mathbf{x}^{\text{fake}}_t, t_{\text{fake}}, \mathbf{c}_{\text{fake}}) - (\mathbf{z}_{\text{fake}} - \mathbf{x}^{\text{fake}})\|^2$$

Total:

$$\mathcal{L}_{\text{APEX}} = \lambda_p \mathcal{L}_{\text{sup}} + \lambda_c \mathcal{L}_{\text{mix}} + \lambda_f \mathcal{L}_{\text{fake}}$$

The only new trainable parameters are the 2 scalars `(a, b)` in `ConditionShift` (mode=`scalar`, default). The rest of the gradient flows into the existing LoRA delta. Proposition 5 of the paper proves this gradient is proportional to the Fisher divergence $D_F(p_\theta \| p_{\text{mix}})$ with **constant time weighting** $w \equiv 1$ — the GAN-aligned update structure without a discriminator's sample-dependent weight.

## Implementation map

| File | Role |
|------|------|
| `networks/methods/apex.py` | `ConditionShift(dim, mode, init_a, init_b)` — 3 parameterizations: `scalar` (2 params), `diag` (2D), `full` (D²+D). Runtime-dtype-safe via `.to(c.dtype)` inside forward. |
| `library/training/apex_loss.py` | `apex_schedule_weights(step, warmup, rampup, ...)` scheduler + `ApexAux` dataclass stashing `F_real / T_mix_v / F_fake_on_fake_xt / target_fake` between the two halves of the loss. |
| `library/anima/training.py` | CLI args (`--apex_*`) and the `apex_omega` weighting scheme ($\omega(t) = t(1-t)$, the Eq. 24 weight after Prop. 3 endpoint→velocity conversion). |
| `networks/lora_anima/` | `_maybe_attach_apex_shift(network, kwargs)` — called from both `create_network` and `create_network_from_weights` so warm-start via `dim_from_weights=true` still attaches the shift. Also registers a 0.1× LR param group for `(a, b)`. |
| `train.py` | 3-forward APEX branch inside `get_noise_pred_and_target` (real → fake-sg → fake-on-fake), L_mix/L_fake aggregation inside `_process_batch_inner`, cold-start guard after `network.load_weights`, `_apex_step` counter in `on_step_start`. |
| `configs/methods/apex.toml` | Method config — see "Hyperparameters" below. |

## Training loop structure

Per step, inside `get_noise_pred_and_target`:

1. Real-branch forward `F_real = anima(x_t, t, c)` — grad-enabled, same as standard LoRA training.
2. Build `x_fake = x_t - t·sg(F_real)` (Eq. 11, endpoint predictor) and a fresh OT trajectory `x_fake_t = t_fake·z_fake + (1-t_fake)·x_fake`.
3. Compute `c_fake = ConditionShift(c)`. Grad flows into `(a, b)` via L_fake.
4. `v_fake_sg = anima(x_t, t, c_fake)` **under `torch.no_grad()`** — this is the stop-gradiented target for L_mix.
5. `F_fake_on_fake_xt = anima(x_fake_t, t_fake, c_fake)` — grad-enabled. Trains the fake branch to fit the current one-step fake distribution.

Then inside `_process_batch_inner`:

6. L_sup = standard FM loss against `noise - latents`, scaled by `lambda_p`.
7. L_mix = MSE(`F_real`, `T_mix_v`), scaled by `lambda_c × lam_c_eff`.
8. L_fake = MSE(`F_fake_on_fake_xt`, `target_fake`), scaled by `lambda_f × lam_f_eff`.
9. `loss = L_sup + L_mix + L_fake`. Single backward.

`lam_c_eff` and `lam_f_eff` come from the warmup/rampup schedule (see below).

## Hyperparameters

All fields in `configs/methods/apex.toml`:

| Key | Default | Meaning |
|-----|---------|---------|
| `weighting_scheme` | `"apex_omega"` | Eq. 24 time weight after endpoint→velocity conversion ($\omega = t(1-t)$). |
| `apex_lambda` | `1.0` | T_mix mixing coefficient (Eq. 23). `0.0` → pure supervised, `1.0` → pure fake-branch target. |
| `apex_lambda_p` | `1.0` | Weight on L_sup. |
| `apex_lambda_c` | `1.0` | Weight on L_mix. Gated by rampup schedule. |
| `apex_lambda_f` | `1.0` | Weight on L_fake. Gated by rampup schedule. |
| `apex_loss_form` | `"mix"` | Primary form from Phase 0 §7.2. `"gapex"` available as a debug fallback. |
| `apex_warmup_ratio` | `0.20` | Fraction of `max_train_steps` with `lam_c = lam_f = 0` (pure L_sup). |
| `apex_rampup_ratio` | `0.10` | Fraction of `max_train_steps` over which `lam_c, lam_f` linearly rise 0 → target. |
| `apex_warmup_steps` | `0` | Absolute override; ignored if `0`. |
| `apex_rampup_steps` | `0` | Absolute override; ignored if `0`. |
| `apex_condition_shift_mode` | `"scalar"` | `scalar` (2 params), `diag` (2D), or `full` (D²+D). Start scalar; the other modes are ablation territory. |
| `apex_condition_shift_init_a` | `-1.0` | Table 7 optimum. |
| `apex_condition_shift_init_b` | `0.5` | Table 7 stable range center. |
| `apex_shift_lr_scale` | `0.1` | LR multiplier on `(a, b)` vs. LoRA params. Keeps the shift off the unstable Table 7 corner. |
| `blocks_to_swap` | **`0` (forced)** | Multi-forward pattern breaks block swap — see below. |

## Warm-start is mandatory

Phase 0 observed **−48% NFE=1 W1** on cold start vs. cold-start plain FM. The failure mode is specific: at init, `F_θ` is random, so `x_fake = x_t - t·F_θ` is noise, and `L_fake` trains the fake branch to fit random trajectories. Through shared weights this contaminates the real branch before it learns anything coherent.

The training loop refuses to launch `--method apex` without either

1. a nonzero warmup (`apex_warmup_ratio > 0` or `apex_warmup_steps > 0`), or
2. a `--network_weights <path>` warm-start.

The shipped `apex.toml` has both (belt + suspenders). The warm-start path points at a prior T-LoRA checkpoint with `dim_from_weights=true` — `save_weights` in `networks/lora_anima/` already converts OrthoLoRA deltas to plain LoRA at save time via a thin SVD, so the T-LoRA checkpoint loads directly into the plain-LoRA path APEX uses. No converter script needed.

## Memory and block swap

APEX runs **3 DiT forwards per step** (real + fake-sg + fake-on-fake). Block swap moves weights between CPU and GPU during a forward and relies on the matching backward to move them back. After the first forward in a multi-forward step, blocks `0..blocks_to_swap-1` are on CPU; the next forward then hits `RuntimeError: Unhandled FakeTensor Device Propagation ... found two different devices cuda:0, cpu` from `torch.compile`. This is the same constraint that postfix variants already work around by forcing `blocks_to_swap = 0` — APEX takes the same route.

If the 3× forward exceeds VRAM, the levers in order of preference are:

1. Drop `--train_batch_size` in the preset.
2. Lower image resolution / constant-token bucket size.
3. Reduce the rank of the warm-start LoRA (re-train a smaller base first).

Do **not** re-enable `blocks_to_swap` for APEX — it will crash as above. A proper fix would require resetting offloader state between every forward and validating that the backward hooks still bring the right blocks back for the oldest graph, which is a non-trivial piece of work and not currently planned.

## Gradient flow

- **L_sup** → LoRA delta only (through `F_real`).
- **L_mix** → LoRA delta only. `T_mix_v` is detached; `v_fake_sg` is computed inside `torch.no_grad()` explicitly to prevent graph attachment.
- **L_fake** → LoRA delta **and** `ConditionShift.(a, b)`. This is the only path by which `(a, b)` learn. (L_mix also uses `c_fake` but it's detached before being fed to the no-grad fake branch.)

The `ConditionShift` parameters live on `network.apex_condition_shift` and are exposed through `prepare_optimizer_params_with_multiple_te_lrs` as a separate param group with LR = `unet_lr × apex_shift_lr_scale`.

## Saving and loading

`save_weights` in `networks/lora_anima/` emits the LoRA state dict as normal. `ConditionShift` parameters are stored alongside under `apex.condition_shift.*` keys (same pattern as HydraLoRA's router keys) so a single `.safetensors` file round-trips through `load_weights`. A non-APEX consumer that loads an APEX checkpoint will see those extra keys and should either ignore them or apply a matching `c → a·c + b` transform before DiT forward — the LoRA delta alone is **not** a drop-in replacement for the APEX LoRA, because it was trained against a shifted condition.

Currently the custom ComfyUI loader does **not** apply ConditionShift — that's still a pending item. Test generations for now run inside the Anima `inference.py` path which loads `network.apex_condition_shift` when present.

## Debug knobs

- `--apex_loss_form gapex` — switches from $\mathcal{L}_{\text{mix}}$ (Eq. 24) to the split $\mathcal{G}_{\text{APEX}} = (1-\lambda)\mathcal{L}_{\text{sup}} + \lambda\mathcal{L}_{\text{cons}}$ form (Eq. 22). Phase 0 §7.2 verified per-sample gradient equivalence to fp32 noise, so this is for debugging only — expect identical training dynamics at higher compute cost.
- `--apex_lambda 0.0` — disables the fake-branch contribution to `T_mix`, reducing L_mix to a consistency loss against pure $\mathbf{v}_{\text{data}}$. Useful as a sanity check that the fake branch is actually doing something when you see no improvement over plain FM.
- Set `apex_warmup_ratio = 1.0` to run pure L_sup for debugging the base training loop without APEX terms ever activating. (The schedule clamps to 0 before `warmup_steps`; with ratio=1.0 you never leave warmup.)

## References

- Liu et al., *Self-Adversarial One Step Generation via Condition Shifting*, arXiv:2604.12322v1, 2026.
- `archive/bench/apex_phase0.py` — standalone 2D toy that validates Theorem 1 (gradient equivalence, max rel err 1.7e-7) and the cold-start failure mode. Run time ~45s on CPU.
- `docs/guidelines/graft-guideline.md` — the target consumer for 1-NFE LoRAs.
- `docs/methods/invert.md` — the target consumer for fast inversion.
