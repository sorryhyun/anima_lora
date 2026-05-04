# σ-conditional postfix (`cond-timestep` mode)

Extension of the `cond` postfix mode (caption-conditional) that adds a zero-init σ-conditional residual. Designed so training starts identical to `cond`, and σ-dependence only emerges if gradients push it — `|sigma_residual|` at convergence is a direct "did σ-conditioning help" diagnostic.

Companion: `prefix-tuning.md` (static postfix/prefix baseline), plus the σ-conditional HydraLoRA router (B-side analogue) in `hydra-lora.md`.

---

## Design

```
base_postfix     = cond_mlp(mean_pool(t5_emb, mask))   # same as cond mode
sigma_residual   = sigma_mlp(sinusoidal(σ))            # new, zero-init final layer
postfix(σ)       = base_postfix + sigma_residual       # additive
```

Both `cond_mlp` and `sigma_mlp` are 2-layer MLPs with zero-init final projections. The sinusoidal σ features match the DiT `t_embedder` functional form (inlined in `networks/methods/postfix.py` rather than coupling to the DiT).

**Motivation:**

1. High σ: text conditioning dominates (layout/semantics). Low σ: self-consistency dominates (refinement). A single static postfix can't serve both optimally.
2. Phase 0 found the T5/inversion functional gap in middle blocks (8–20) varies implicitly with σ. Static postfix averages this away.

**Parameter cost:** `sigma_feature_dim × sigma_hidden + sigma_hidden × K × D`. At the default K=64, D=1024, `sigma_feature_dim=128`, `sigma_hidden=256`: ~16M total params for postfix + σ branch combined (same order as `cond`).

---

## Usage

Two ways to run it:

```bash
# Clean per-variant path:
make lora-gui GUI_PRESETS=postfix_sigma

# Toggle-block path: activate the cond-timestep block in configs/methods/postfix.toml, then:
make exp-postfix
```

The toggle block looks like:

```toml
network_dim = 64
network_args = [
    "mode=cond-timestep",
    "splice_position=end_of_sequence",
    "cond_hidden_dim=256",
    "sigma_feature_dim=128",
    "sigma_hidden_dim=256",
]
output_name = "anima_postfix_sigma"
max_train_epochs = 2
```

Test: `python inference.py --postfix_weight output/anima_postfix_sigma.safetensors ...`, or `make exp-test-postfix` against the latest postfix output.

---

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `train.py` passes `timesteps` into `append_postfix`. |
| Standard inference | ✅ | Non-tiled `generate_body` recomputes `postfix(σ_i)` per denoising step against cached base embeds. |
| Spectrum inference | ✅ with smoothness caveat | Actual-step forwards recompute `postfix(σ)`; cached steps skip blocks so cross-attn never runs. Chebyshev fit assumes smooth `F(σ)` — sharp σ-transitions will be washed out. |
| Tiled diffusion | ⚠ not supported | Raises `NotImplementedError`. Separate refactor needed. |
| Modulation guidance | ✅ orthogonal | Modulation = per-block AdaLN path; postfix = K/V input path. Ablation needed to verify non-redundant contribution. |
| T-LoRA | ⚠ mild overlap | Both carry σ signal. Not conflicting but ablation recommended. |
| HydraLoRA + σ-conditional router | ✅ orthogonal paths | Postfix: K/V input. Hydra: internal projections. Stackable. |

---

## Evaluation

- **Primary diagnostic:** `|sigma_residual(σ)|` as a function of σ at convergence. Flat/near-zero → `cond-timestep` collapsed to `cond` (no σ-dependence learned).
- **Functional MSE** at blocks 8, 12, 16, 20 (Phase 0 depth range) on a held-out prompt set. Expected win: closes the T5/inversion gap more than static `cond` in a σ-selective way.
- **Spectrum A/B:** quality drop with Spectrum vs. without should be comparable to the baseline (non-postfix) Spectrum quality drop. Larger drop implies σ-residual is too spiky — either add a smoothness regularizer (L2 on finite differences of `sigma_residual` across adjacent training timesteps) or use `--spectrum_calibration 0.5` at inference to absorb narrow σ features.

---

## Files

- `networks/methods/postfix.py` — mode implementation, save/load, metadata.
- `library/inference/generation.py` — per-step postfix in non-tiled and Spectrum paths.
- `networks/spectrum.py` — actual-step σ-conditional postfix application.
- `configs/methods/postfix.toml` — commented `cond-timestep` block.
- `configs/gui-methods/postfix_sigma.toml` — clean per-variant config.
