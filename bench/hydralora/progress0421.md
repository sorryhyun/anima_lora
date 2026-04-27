# HydraLoRA router: unsticking — 2026-04-21

Day-long bisection of why the router had never actually trained on any
HydraLoRA checkpoint. Pre-morning state (moe from 2026-04-20 23:44): router
weights unchanged from init (std≈0.01, bias std≈0), gate marginal entropy
1.0000 across every module, max-pairwise JS across σ = 0. The network was
paying for 4 expert heads but running as a single averaged head.

## Sequence of checkpoints

All runs use `configs/methods/lora.toml` → merged preset `default`.
Benchmarks: `bench/hydralora/analyze_router_sigma_correlation.py` on the
resulting `*_moe.safetensors`. All numbers taken with 32 samples × 7 σ =
224 forwards.

| ckpt (mtime)      | use_ortho | balance | expert_init_std | router_lr_scale | epochs | router.w std | router.b std | expert cos (mean) | norm_H (mean / min) | top1 dom | dead / total | max JS |
|-------------------|:---------:|--------:|----------------:|----------------:|-------:|-------------:|-------------:|------------------:|--------------------:|---------:|-------------:|-------:|
| 0421-0011         |     –     | 1e-5    | 1e-4            |     1 (init)    |   1    | 0.0097       | 0.0003       | 1.0000            | 1.0000 / 1.0000     |  0       | 0 / 112      | 0      |
| 0421-0051 (10×)   |     –     | 1e-5    | 1e-4            |      10         |   1    | 0.0097       | 0.0003       | 1.0000            | 1.0000 / 0.9998     |  0       | 0 / 112      | 0      |
| 0421-0120 (std↑)  |     –     | 1e-5    | **1e-2**        |      10         |   1    | 0.0094       | 0.0003       | 0.988             | 1.0000 / 1.0000     |  0       | 0 / 112      | 0      |
| 0421-0149         |     –     | 1e-5    | **1e-1**        |      50         |   1    | 0.0097       | 0.0003       | **0.384**         | 1.0000 / 1.0000     |  0       | 0 / 112      | 0      |
| 0421-0810 (8ep)   |     –     | 1e-5    | 5e-2            |      20         |   8    | 0.0088       | 0.0115       | 0.600             | 0.9999 / 0.996      |  ~0      | 0 / 112      | ~0     |
| 0421-1436 (bal 0) |     –     | **0**   | **0**           |      20         |   1    | **0.0770**   | **0.0869**   | ~0.000            | **0.681 / 0.188**   | **0.80** | **94 / 224** | 4.4e-3 |
| 0421-2125 (ortho) | **true**  | 1e-3    | 1e-4            |      20         |   4    | 0.0670       | 0.0760       | **0.0000** (exact)| **0.718 / 0.079**   | 0.611    | 88 / 224     | **0.024** |

`expert cos` = mean pairwise cosine of `lora_ups.{i..j}.weight` across
`i<j∈{0..3}`, averaged over modules. Init=0 produces `1.0` (all zero
vectors collapse to cos=1 under `cosine_similarity`; for nonzero init this
reflects real angular spread). `norm_H` = marginal-gate normalized entropy
(1.0 = uniform, 0.0 = one-hot).

## What moved what

1. **Router LR knob added.** `network_router_lr_scale` (`networks/__init__.py`
   allowlist, `networks/lora_anima.py:prepare_optimizer_params_*`). Pulls
   `router.*` + `sigma_mlp.*` params into their own param group with
   `lr = base_lr × scale`. Bench 0421-0051 / 0120 / 0149 showed **no effect**.

2. **Name-classification bug in the knob.** `_is_router_param` used
   `".router." in name`, but `named_parameters()` on a `HydraLoRAModule`
   yields leading-less names (`router.weight`, `sigma_mlp.0.weight`). So
   router params were silently landing in the plain `lora` group at 1×
   unet_lr; the 10× / 50× numbers above were applied to an empty group.
   Fixed to `name.startswith("router.") or name.startswith("sigma_mlp.")`
   in `networks/lora_anima.py:assemble_params`. Verified by instantiating a
   `HydraLoRAModule` and dumping its param names.

3. **`expert_init_std` was too small to break expert symmetry.** Even
   after the LR fix, at `init_std=1e-4` all 4 `lora_ups` had pairwise cos
   ≈ 1.0 through training (run 0421-0051: cos 1.0000, identical to init).
   Bumping to 1e-2 → 0.988, 1e-1 → 0.384. But the router still sat uniform
   — evidence that expert differentiation isn't sufficient for routing
   differentiation.

4. **Balance loss was the active force pinning gates uniform.**
   `balance_loss_weight=1e-5` → `0` with `expert_init_std=0`
   (`2026-04-21 14:36`): **router broke out** — weight std 0.077 (7.7×
   init), bias std 0.087, 80% samples dominantly route, 42% dead experts,
   min norm_H = 0.19. Confirmed Switch-Transformer balance loss is the
   load-bearing attractor; the earlier "router never trains" regime was
   the balance loss winning against a tiny router gradient, not the
   gradient being absent.

5. **Zero-init experts leave routing-starved experts permanently dead.**
   In the balance-off/init-zero run, the 42% dead experts had near-zero
   norms — the router picked favorites on step 1 and the ignored ones
   never received gradient. Fine for quality if you only need 2-3 experts
   per module, but wastes the `num_experts=4` capacity.

6. **OrthoHydra fixes the cold-start deadlock structurally.** Switching to
   `OrthoHydraLoRAExpModule` (`use_ortho=true + use_hydra=true`) partitions
   the top-`E·r` SVD columns of the pretrained weight into `E` disjoint
   slices, giving each expert its own orthonormal output basis. `P_eff[i]^T
   P_eff[j] = 0` for `i ≠ j` is structurally guaranteed, so the router's
   per-expert scores differ meaningfully at step 0 *before* any expert has
   trained. 4ep run (0421-2125): all 4 experts have meaningful norm
   (median 0.405, min 0.18), pairwise cos = 0.0000 exactly, router hits
   healthy norm_H=0.72 / top1-dom=0.61, σ-correlation starts appearing
   (max JS 0.024). Documented in
   `docs/methods/hydra-lora.md#orthogonalized-experts`.

## Exit criteria vs current state

From `docs/methods/hydra-lora.md` "Fixes" section targets, measured on
0421-2125:

| criterion                           | target            | current       |
|-------------------------------------|-------------------|---------------|
| `‖router.weight‖` at final step     | > 1.5× init       | ~6.7× ✓       |
| median norm_entropy                 | ∈ [0.6, 0.95]     | 0.77 ✓        |
| mean dominant-top1                  | > 0.2             | 0.61 ✓        |
| zero dead experts                   | 0                 | 88/224 ✗      |
| `make test-hydra` quality           | ≥ LoRA baseline   | not evaluated |

"Dead experts" here means router-ignored, not parameter-zero — OrthoHydra
ensures the weights exist. Whether to actively recover them (increase
`balance_loss_weight` slightly, enable `expert_warmup_ratio` more
aggressively, or raise `num_experts` to 2 with the expectation that each
is used) is the next decision.

## Config landing point

```toml
# configs/methods/lora.toml (as of 2026-04-21 21:25)
use_ortho = true
use_hydra = true
num_experts = 4
balance_loss_weight = 1e-3
expert_init_std = 1e-4
network_router_lr_scale = 20
expert_warmup_ratio = 0.1
hydra_router_layers = ".*(mlp\\.layer[12])$"
use_sigma_router = true
sigma_feature_dim = 16
sigma_router_layers = ".*(mlp\\.layer[12])$"
```

Router scope narrowed to MLP layers (`mlp.layer[12]`) — cross-attn got
dropped mid-day ("I dropped most routers") on the theory that per-token
style routing belongs in the MLP, not the attention projections. Worth a
follow-up JS comparison on cross-attn vs MLP routers once we have more
data.

## Open questions

- **σ-signal.** mean-max-JS=1.2e-3 (p90 2.5e-3, max 2.4e-2) — signal
  exists but most routers are still σ-agnostic. Worth checking after a
  longer run whether the concat-σ design (new: sigma_feature_dim=16
  concatenated into router input) keeps growing or plateaus.
- **Dead-expert recovery.** Current 88/224 is router preference. Does
  raising `balance_loss_weight` to ~5e-3 recover them without
  re-introducing the uniform attractor? Or does it just reintroduce it?
- **Quality gate.** All the above is routing geometry. `make test-hydra`
  vs non-hydra baseline hasn't been rerun since the unstick. Needs doing
  before declaring the routing healthy.
