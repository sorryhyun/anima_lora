# HydraLoRA Router Balancing Proposal

This note is a proposal for the next HydraLoRA experiments in Anima. The short
version is: the HydraLoRA concept is still plausible, but the current evidence
does not show a solved routing system. The problem is not only "early timestep"
or "lower layer" difficulty. The observed pattern is mixed:

- Early blocks are often close to uniform, which means the router may be inert.
- Middle and deep blocks show hard collapse in several modules.
- The latest blocks show the clearest sigma-dependent routing signal.
- A single global balance weight is too blunt for this pattern.

The next work should separate three questions:

1. Is inference actually exercising the trained sigma router?
2. Are experts being used in a useful, non-collapsed way?
3. Which blocks/layers have enough signal to justify Hydra routing?

## Current State

Current Hydra settings in [`configs/methods/lora.toml`](../../configs/methods/lora.toml):

```toml
use_hydra = true
num_experts = 4
balance_loss_weight = 2e-6
balance_loss_warmup_ratio = 0.3
network_router_lr_scale = 20
hydra_router_layers = ".*(mlp\\.layer[12])$"
use_sigma_router = true
sigma_feature_dim = 16
sigma_router_layers = ".*(mlp\\.layer[12])$"
per_bucket_balance_weight = 0.3
num_sigma_buckets = 3
```

Latest diagnostic file:
[`results/sigma_correlation_0424-484.json`](results/sigma_correlation_0424-484.json)

Overall router/sigma result:

| Metric | Value |
| --- | ---: |
| mean max JS across sigma buckets | 0.00184 |
| median max JS across sigma buckets | 0.00031 |
| max max JS across sigma buckets | 0.03389 |
| diagnostic case | A |
| mean normalized entropy | 0.6649 |
| median normalized entropy | 0.6873 |
| p10 normalized entropy | 0.2838 |
| total dead experts | 97 / 224 |
| collapsed modules | 18 |
| balanced modules | 23 |
| mean dominant top1 fraction | 0.6168 |
| verdict | PARTIAL |

Block-bin aggregation from the same run:

| Blocks | mean H | dead experts/module | dominant top1 frac | mean max JS |
| --- | ---: | ---: | ---: | ---: |
| 0-3 | 0.900 | 0.38 | 0.125 | 0.00028 |
| 4-7 | 0.919 | 0.38 | 0.125 | 0.00039 |
| 8-11 | 0.702 | 1.12 | 0.733 | 0.00062 |
| 12-15 | 0.605 | 2.75 | 0.893 | 0.00145 |
| 16-19 | 0.499 | 2.75 | 1.000 | 0.00045 |
| 20-23 | 0.398 | 2.88 | 0.877 | 0.00096 |
| 24-27 | 0.631 | 1.88 | 0.565 | 0.00871 |

Interpretation:

- The median sigma JS is near zero, so most routers are not using sigma
  strongly even when explicit sigma features are enabled.
- The late blocks, especially 24-27, carry the strongest sigma signal.
- Early blocks are not the main collapse site. They are mostly high-entropy and
  weakly specialized.
- Mid/deep blocks are where many routers collapse to one expert.
- Current balancing is helping somewhat, but it is not enough and it is not
  selective enough.

## Immediate Correctness Gate

Before comparing image quality from Hydra checkpoints, fix and test sigma
propagation in inference.

The bench script already does this correctly. In
[`analyze_router_sigma_correlation.py`](analyze_router_sigma_correlation.py),
the diagnostic loop calls `network.set_sigma(sigma_tensor)` before each DiT
forward.

Training also calls `set_sigma(timesteps)`.

The risky path is CLI/Spectrum inference: if the attached Hydra network does
not receive the current denoising timestep before each DiT forward, the
sigma-router columns are effectively zero-padded at inference. That makes
visual tests of sigma-router checkpoints misleading.

Required fix:

- Call `network.set_sigma(t_expand)` or equivalent before every actual DiT
  forward in normal generation.
- Do the same in Spectrum's real-forward path.
- Clear sigma after generation if the network exposes `clear_sigma()`.
- Add a regression test or a cheap instrumentation test proving that a
  sigma-router Hydra module receives nonzero/current sigma during inference.

Do this before treating `make test-hydra` images as evidence for or against
sigma routing.

## Hypotheses

### H1: Global balance is the wrong control surface

The existing load-balance loss is already computed per module and then averaged.
So "per-router balancing" should not mean merely moving the same loss inside a
loop. It should mean per-router coefficients, per-router state, and per-router
decisions.

Current behavior needs different pressure in different places:

- Uniform/inert routers should not receive stronger uniform pressure.
- Collapsed routers need anti-collapse pressure or exploration.
- Healthy routers should be mostly left alone.

### H2: Some routers are signal-poor

High entropy is not automatically good. A router with normalized entropy near
1.0 and no sigma/style separation may just be doing dense averaging across
experts. In those layers, Hydra can add parameters and optimizer noise without
giving useful conditional capacity.

### H3: Sigma routing is depth-dependent

The 0424 run suggests sigma signal is strongest in the last blocks. Applying
sigma routing to all MLP layers may dilute the experiment. A late-only or
mid-late-only Hydra scope may be a cleaner test.

### H4: Layer-local RMS may miss semantic/style routing signal

Hydra currently routes from layer-local low-rank activations. That is a good
default because it keeps routing local to the adapted linear layer. But if the
task needs style/identity/semantic specialization, the local signal may be too
weak. A small global conditioning feature may be needed as a bias, not as a full
replacement for layer-local routing.

## Proposed Tracks

### Track A: Adaptive per-router balancing

Replace the single global balance pressure with a local coefficient per Hydra
module. Keep the base Switch-style balance loss, but weight it based on recent
router health.

Suggested state per router:

```text
ema_usage[E]
ema_entropy
ema_dead_count
ema_dominant_top1_fraction
local_balance_weight
```

Suggested control rule:

```text
if ema_entropy < low_entropy or ema_dead_count > dead_threshold:
    increase local_balance_weight
elif ema_entropy > high_entropy and sigma/style JS is still tiny:
    decrease local_balance_weight or mark router as inert
else:
    keep local_balance_weight near base
```

Use a target band, not exact uniformity. For 4 experts:

- `H < 0.4`: likely collapse.
- `H ~= 0.55-0.9`: useful operating band.
- `H ~= 1.0` with low JS/top1: probably inert dense averaging.

Optional extension: keep usage EMA per sigma bucket. This matters if a router is
balanced globally but collapsed separately inside each sigma region.

Implementation cautions:

- Do not force every router to be uniform. Some layers may genuinely need only
  one or two experts.
- If using loss-free balancing bias, either anneal it to zero before saving or
  serialize and apply it during inference. Otherwise training and inference see
  different routers.
- Log local weights so bad routers can be identified after a run.

Bench:

```bash
python bench/hydralora/analyze_router_sigma_correlation.py \
  --lora_weight output/ckpt/anima_hydra-<tag>_moe.safetensors \
  --dataset_dir post_image_dataset \
  --num_samples 32 \
  --out_json bench/hydralora/results/sigma_correlation_<tag>.json
```

Compare:

- Current baseline: `balance_loss_weight = 2e-6`
- Static sweep: `1e-5`, `5e-5`, `1e-4`, `5e-4`
- Adaptive per-router: base `2e-6` or `1e-5`, local multiplier capped at
  something like `0x..100x`

Primary readout:

- Fewer dead experts without pushing every router to entropy 1.0.
- Lower number of collapsed modules.
- No image-quality regression.

### Track B: Gate floor during training

Use a small exploration floor during training:

```python
soft_gate = softmax(logits)
gate = (1.0 - eps) * soft_gate + eps / num_experts
```

Suggested schedule:

- Start `eps = 0.05`.
- Anneal to `0.0` by 30-50 percent of training.
- Use normal softmax at inference.

Reason:

- Prevents early expert starvation.
- Keeps dead experts trainable long enough for routing gradients to find them.
- Avoids a permanent inference-time mismatch if annealed to zero.

Bench:

- Same router diagnostic as Track A.
- Also inspect per-expert LoRA-up weight norms. A router can look balanced while
  some expert heads remain undertrained.

Pass condition:

- Fewer dead experts than baseline.
- Similar or better validation loss.
- No persistent need for the gate floor at inference.

### Track C: Dominant-expert dropout

During router warmup, occasionally mask the currently dominant expert for a
module or sample. This is stronger than entropy loss, so it should be temporary
and low probability.

Suggested schedule:

- Enable only during the first 10-30 percent of training.
- Apply with small probability, for example `p = 0.05..0.15`.
- Do not use at inference.

Bench:

- Compare against Track B. Do not combine both at first.
- Watch validation loss closely; dropout can harm if specialization is real.

Pass condition:

- It revives dead experts better than gate floor without worsening images.

### Track D: Layer-scoped Hydra placement

Do not assume every MLP layer should be Hydra. The current block-bin result
suggests at least three alternatives:

1. Late-only MLP Hydra: blocks 20-27 or 24-27.
2. Mid-late MLP Hydra: blocks 12-27 or 16-27.
3. Mixed target: late MLP plus selected attention projections.

Initial candidates:

```toml
# Fresh training regexes match original module names like
# `blocks.24.mlp.layer1`, not saved LoRA keys like
# `lora_unet_blocks_24_mlp_layer1`.
hydra_router_layers = "blocks\\.(2[0-7])\\.mlp\\.layer[12]$"
sigma_router_layers = "blocks\\.(2[0-7])\\.mlp\\.layer[12]$"
```

If the current selector cannot express block scopes cleanly, add a separate
block filter rather than making regexes harder to maintain.

Bench:

- Keep total rank/parameter budget visible.
- Compare against full-MLP Hydra at the same training budget.
- Measure both router health and image quality.

Pass condition:

- Equal or better image quality with fewer collapsed routers and lower memory.
- Better late-block sigma JS if sigma routing is the intended effect.

### Track E: Hybrid conditioning router

If Tracks A-D still leave routing inert or collapsed, add a small global
conditioning feature to the router:

```text
router_input = [
  layer_local_rank_signal,
  sigma_features,
  pooled_text_or_style_feature
]
```

This should be a bias into the layer-local router, not a return to a purely
global router. Keep the layer-local path as the main input.

Implementation cautions:

- Zero-init the new conditioning projection so step 0 matches current Hydra.
- Make it optional and restricted to Hydra routers.
- Keep the feature small. A projected 16-64 dim pooled conditioning vector is
  enough for a first test.
- Cache compatibility matters because Hydra requires cached text outputs.

Bench:

- Compare against the best Track A-D result, not only against the current
  baseline.
- Inspect whether experts correlate with prompt/style groups, not just sigma.
- Use fixed prompt seeds for visual comparisons.

Pass condition:

- Router specialization aligns with prompt/style changes.
- Visual identity/style improves without worsening base quality.

## Bench Matrix

Use fixed data, seed, prompt set, training length, and checkpoint step wherever
possible.

| Run | Change | Purpose |
| --- | --- | --- |
| B0 | current 0424 checkpoint | reference |
| B1 | static balance `1e-5` | check if current weight is too low |
| B2 | static balance `5e-5` | medium pressure |
| B3 | static balance `1e-4` | strong pressure |
| B4 | static balance `5e-4` | likely too strong, useful boundary |
| A1 | adaptive per-router balance | targeted anti-collapse |
| Bf1 | gate floor `eps=0.05 -> 0` | anti-starvation |
| D1 | late MLP only | test layer-scope hypothesis |
| D2 | mid-late MLP only | test broader layer-scope hypothesis |
| E1 | hybrid pooled conditioning | test missing semantic signal |

Recommended diagnostic command:

```bash
python bench/hydralora/analyze_router_sigma_correlation.py \
  --lora_weight output/ckpt/anima_hydra-<tag>_moe.safetensors \
  --dataset_dir post_image_dataset \
  --num_samples 32 \
  --out_json bench/hydralora/results/sigma_correlation_<tag>.json
```

Recommended smoke tests:

```bash
python -m pytest tests/test_network_registry.py tests/test_loss_registry.py tests/test_config.py -q
make test-hydra
```

For image comparison, use the same prompts and seeds across:

- non-Hydra LoRA/Ortho/T-LoRA baseline
- current Hydra baseline
- best adaptive-balance run
- best layer-scoped run
- hybrid-router run, if attempted

## Metrics To Track

Router metrics:

- median normalized entropy
- p10 normalized entropy
- total dead experts
- dead experts per module
- collapsed module count
- mean dominant top1 fraction
- mean per-sample normalized entropy
- mean/max JS across sigma buckets
- block-bin aggregates for entropy, dead experts, top1 dominance, and JS

Training metrics:

- train loss and validation loss by checkpoint
- validation loss grouped by sigma bucket
- router balance loss magnitude by module group
- local balance coefficient distribution, if Track A is implemented
- per-expert LoRA-up norm and gradient norm

Image metrics:

- fixed-seed visual grid
- prompt/style consistency
- identity preservation if the dataset has a clear subject
- artifact rate
- comparison to non-Hydra baseline

Optional embedding metrics:

- CLIP/image embedding distance to style references
- prompt-group separability by expert assignment
- mutual information between expert top1 and prompt/style bucket

## Decision Criteria

Treat Hydra as healthy only if router metrics and images agree.

Healthy enough to keep:

- median normalized entropy in roughly `0.55..0.9`
- total dead experts below about 10-20 percent, or clearly concentrated in
  intentionally single-expert layers
- no broad block-bin collapse with `H < 0.45`
- mean dominant top1 fraction roughly `0.2..0.8`
- sigma JS improves in the blocks where sigma routing is intended
- fixed-seed images are at least as good as the non-Hydra baseline

Reject or revise:

- `H ~= 1.0` everywhere with near-zero JS and weak visuals: inert averaging
- `H < 0.4` with many dead experts: collapsed routing
- better router metrics but worse images: balance is optimizing the wrong thing
- sigma-router checkpoint looks unchanged visually before inference sigma
  propagation is fixed: invalid comparison

## Suggested Next Steps

1. ~~Fix CLI/Spectrum sigma propagation and add a regression test.~~ done
2. Add a small report helper that aggregates
   [`analyze_router_sigma_correlation.py`](analyze_router_sigma_correlation.py)
   output by block bins.
3. Run the static balance sweep to identify the useful pressure range.
4. Implement Track A adaptive per-router balancing if the sweep confirms that
   different layers need different pressure.
5. Run Track B gate-floor training as the simplest anti-starvation intervention.
6. Run late-only or mid-late-only Hydra to check whether full-MLP Hydra is
   wasting capacity in early blocks.
7. Try hybrid conditioning only if local/sigma routing remains weak after the
   above fixes.

## Bottom Line

Use per-router adaptive balancing, but treat it as a control fix, not as the
whole solution. The deeper issue is that routers differ by layer: some are
inert, some collapse, and late blocks appear to carry more sigma signal than
early blocks. The practical path is to fix sigma inference first, measure router
health per block, then decide whether Hydra should be adaptive, layer-scoped, or
given a small global conditioning bias.
