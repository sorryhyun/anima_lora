# AdaLN LoRA — training → ComfyUI-compatible shipping

Status: extraction path shipped & verified; training path plumbed and DEFAULT-ON
(`train_adaln = true` in `configs/base.toml`, 2026-07-16) but still unbenched.
The flip aligns us with diffusion-pipe rather than resting on a Phase-0 result — no
bench script or invariant test covers the train-side default for the LoRA family, so
the CONTRIBUTING Tier 1.5 gate is outstanding, not satisfied. Treat the default as
provisional: if you are A/B-ing anything else, pin `train_adaln` explicitly on both
arms rather than letting it ride.

## What adaln is in this architecture

Anima inherits Cosmos-Predict2's per-block modulation: each `Block` has three
branch-modulation MLPs (`adaln_modulation_{self_attn,cross_attn,mlp}`) producing
shift/scale/gate from the timestep embedding only — no text, no spatial
conditioning. Every Anima checkpoint uses the AdaLN-LoRA bottleneck form
(`use_adaln_lora=True, adaln_lora_dim=256`):

```
SiLU → Linear 2048→256 (".1", down) → Linear 256→6144 (".2", up)
```

The name "AdaLN-LoRA" is NVIDIA's — it's a pretraining architecture choice
(low-rank bottleneck on the modulation MLPs), not an adapter. The param math is
why: full-rank adaln would be 2048×6144 ×3 branches ×28 blocks ≈ 1.06B params;
the 256-bottleneck is ≈176M. It is not optional per checkpoint — the weights
(`blocks.{b}.adaln_modulation_{br}.1/.2.weight`) exist in base and turbo alike.

### The `use_adaln_lora=False` "mystery" in diffusion-pipe

Not a decision by the Anima author. `models/cosmos_predict2_modeling.py` is
vendored NVIDIA reference code (SPDX NVIDIA header) and `False` is just the
upstream class-signature default preserving the vanilla-DiT form. diffusion-pipe
hardcodes `use_adaln_lora=True, adaln_lora_dim=256` for every checkpoint it
loads (`models/cosmos_predict2.py::get_dit_config`, ~line 125). ComfyUI's
detection does the same (`comfy/model_detection.py:801`).

## Who trains adaln today — we are the outlier

| Trainer | adaln in LoRA target set? | Evidence |
|---|---|---|
| diffusion-pipe (Anima author's trainer) | YES — targets every `nn.Linear` under `Block`/`TransformerBlock`, no exclusions and no config filter | `models/base.py::configure_adapter` (~line 219) |
| sd-scripts (official kohya, upstream Anima support) | NO by default — appends `.*(_modulation\|_norm\|_embedder\|final_layer).*`; opt-in via `include_patterns` (include beats exclude, same mechanism as ours) | `sd-scripts/networks/lora_anima.py:254` |
| this repo | YES since 2026-07-16 — `_DEFAULT_EXCLUDE` still blocks `adaln_up_`, but `train_adaln = true` in `configs/base.toml` rescues it via `include_patterns` on every LoRA-family method | `networks/lora_anima/config.py` (`_DEFAULT_EXCLUDE`, `from_kwargs`) |

We were the outlier; we no longer are. Both other trainers reach adaln by default
(diffusion-pipe unconditionally, sd-scripts via the same include-beats-exclude
opt-in we now enable), so a stock run here is closer to the reference behaviour.

Our exclusion is inherited verbatim from kohya's `lora_anima.py` (ours extends
the same regex with the runtime rename names) — a conservative
"attention+MLP only" convention bundled with norms/embedders, specific to the
Anima implementation (kohya's Flux LoRA has no such exclude machinery at all).
Notably kohya also ships `sd-scripts/networks/convert_anima_lora_to_comfy.py`, which
already handles adaln module names — so an sd-scripts user who opts in gets
a ComfyUI-shippable adaln LoRA today. That converter is prior art for our
missing trained-checkpoint export step.

diffusion-pipe also saves in "ComfyUI format" (`diffusion_model.`-prefixed PEFT
keys, `models/cosmos_predict2.py::save_adapter`), so diffusion-pipe-trained
Anima LoRAs all carry adaln and ComfyUI already applies it; sd-scripts-trained
ones default to no adaln, like ours. Two sharpenings on the diffusion-pipe side:

- Ecosystem LoRAs carry both `.1` and `.2` adaln LoRAs (both are Linears
  inside `Block`), plus the LLM adapter (`TransformerBlock` is in the target
  list) — strictly more surface than our target set at equal rank. Keep this in
  mind for any expressivity comparison against third-party LoRAs.
- Cross-tool render mismatch: loading an ecosystem LoRA in-repo via
  `--lora_weight` drops its adaln + LLM-adapter keys (different naming, excluded
  targets) — the same file renders differently here vs in ComfyUI. First thing
  to check when a "repo inference ≠ Comfy" report involves a third-party LoRA.

## ComfyUI support (verified 2026-07-15)

ComfyUI needs no special-casing: `comfy/lora.py::model_lora_keys_unet` (generic
branch, ~line 191) maps every weight in the model's state dict to a
`lora_unet_<path-with-underscores>` key. Base Anima and turbo are the same model
class, so the same keys apply to both. Verified end-to-end by feeding our
adaln-inclusive extraction through ComfyUI's own `comfy.lora.load_lora` against
a key map built from the real checkpoint: all keys consumed — 364 patch
targets = 280 attn/MLP + 84 adaln, aimed at
`diffusion_model.blocks.{b}.adaln_modulation_{br}.2.weight`.

## Key-naming contract (the one real gotcha)

The same Linear has two names, and ComfyUI only knows one of them:

| Surface | Module path | LoRA key |
|---|---|---|
| Checkpoint / ComfyUI | `blocks.{b}.adaln_modulation_{br}.2` | `lora_unet_blocks_{b}_adaln_modulation_{br}_2` |
| This repo, runtime (post `_dit_rename_hook`, `library/anima/weights.py`) | `blocks.{b}.adaln_up_{br}` | `lora_unet_blocks_{b}_adaln_up_{br}` |

Runtime-named adaln keys are **silently dropped** by ComfyUI. Anything shipped
must use the comfy layout; anything consumed in-repo (warm-start, `--lora_weight`)
wants the runtime layout.

## Path 1 — extraction (SHIPPED)

`scripts/toolkits/extract_delta_lora.py` extracts adaln up-projections from a full-model
delta with `--include_adaln`, and `--adaln_layout comfy` renames to the ComfyUI
layout at save time (default `runtime` keeps warm-start compat; layout recorded
in `ss_delta_extract` metadata).

```bash
python scripts/toolkits/extract_delta_lora.py \
    --tuned models/diffusion_models/anima_turboV10.net.safetensors \
    --rank 96 --act_scales models/extracted/act_scales_base_4step.safetensors \
    --include_adaln --adaln_layout comfy \
    --out models/extracted/anima_turboV10_delta_r96_asvd_adaln.safetensors
```

Shipped artifact: `models/extracted/anima_turboV10_delta_r96_asvd_adaln.safetensors`.

Delta facts (official turbo vs base) that motivated this:

- `.2` up-projs are the largest movers in the whole delta (rel-Δ 0.008–0.03);
  `.1` down-projs barely moved (0.0002–0.006) — extractor deliberately takes `.2` only.
- `.2` in-dim is 256, so r=96 captures ~99% energy (recon rel-err ≈ 0.10) —
  adaln is essentially lossless at this rank; residual infidelity lives in attn/MLP.
- Adding adaln lifted mean captured energy 0.803 → 0.860 vs the adaln-less
  r96 ASVD file (min stays 0.423, a non-adaln module).

## Path 2 — training (DEFAULT-ON, NOT BENCHED)

No new machinery needed to train adaln: `include_patterns` beats the default
exclude (`networks/lora_anima/network.py` ~line 245: `if excluded and not
included: continue`), and runtime `adaln_up_{br}` is a plain `nn.Linear`.

```toml
# method TOML — the raw primitive; `train_adaln = true` is the sugar for it
include_patterns = [".*adaln_up_.*"]
```

### The `train_adaln` surface (LoRA family, default-on)

`train_adaln` / `adaln_rank` / `adaln_alpha` are read by
`LoRANetworkCfg.from_kwargs` (`networks/lora_anima/config.py`) and desugar to
exactly the primitives above: `include_patterns += [".*adaln_up_.*"]`, plus
`network_reg_dims` / `network_reg_alphas` entries on the same pattern. `adaln_rank`
defaults to the network's rank; `adaln_alpha` is derived from the network's
own rank/alpha by the √r law below (`network_alpha·√(adaln_rank/network_dim)`)
whenever it is 0/unset — set it only to override. `adaln_rank` / `adaln_alpha`
without `train_adaln` raise.

They reach the network as top-level TOML keys, not `network_args` — the
allowlist is AST-derived from the `kwargs.get()` reads (`networks/__init__.py::
_derive_network_kwargs`) and `train.py::resolve_network_kwargs` forwards them. So
`configs/base.toml`'s `train_adaln = true` applies to every LoRA-family method
(lora / chimera / soup / byg) with no per-method opt-in. Verify a merge with
`make print-config METHOD=<m> PRESET=<p>`.

Frozen-DiT methods. `soft_tokens` takes `**kwargs` and ignores the adaln
keys, so the base default is silently a no-op there rather than an error.
`easycontrol` grew its own opt-in consumer 2026-07-18 — see next section.

### EasyControl (opt-in, cond-gated — 2026-07-18)

EasyControl reads the same three knobs but with method-TOML pins keeping it
off by default (`configs/easycontrol/easycontrol.toml` +
`configs/gui-methods/easycontrol.toml` pin `train_adaln = false` — required
because `resolve_network_kwargs` forwards base.toml's LoRA-family
`train_adaln = true` to every network module). Unlike the LoRA family, `adaln_rank`/
`adaln_alpha` are inert when `train_adaln` is off (never a raise — base
always supplies them). Enable with `--train_adaln true` on a `make`/CLI run.

Shape: per-block `_LoRAProj(256 → 3·2048)` deltas on the target stream's
three `adaln_up_{br}` up-projections, applied inside the two-stream inner and
the cached-cond-KV inference path only — the no-cond fallback runs the
original `Block.forward`, so the delta is cond-gated by construction and
"no reference = exact baseline DiT" survives (unconditional target-stream
drift was the objection to naive adaln-in-EC). The cond stream's own
modulation (`t_embedder(zeros)`) stays stock; zero-init up preserves step-0
equivalence; `multiplier` scales the delta (a strength knob at inference).
Defaults: `adaln_rank = 8` (global-grade hypothesis, ~87% of a full-rank
delta's energy), alpha via the √r law from the cond LoRA
(`32·√(8/32) = 16`). Motivating arm: colorize "how far to push" — a t-only
global chroma/tone-commitment prior (it cannot carry reference content; a
per-image decision is out of reach). Pre-check before training the arm:
measure residual mean-saturation bias of colorize outputs vs GT — if the
miner saturation floor already zeroed it, adaln has nothing to absorb.
Guards in `tests/test_easycontrol_adaln.py`. EC checkpoints are consumed by
the EC KSampler node, not ComfyUI's LoRA loader — the comfy relayout does not
apply; the node needs to learn the `adaln_lora_*` keys before an
adaln-trained EC checkpoint ships.

Turbo knobs (2026-07-15): `network.train_adaln` wires the include on student+fake;
`network.fake_adaln = false` builds an adaln-less critic (VRAM lever);
`network.adaln_rank = N` builds the adaln modules at their own rank via the
factory's `network_reg_dims` regex→dim override — measured on the r96 extract,
the adaln ΔW keeps 0.980/0.991/0.995 of its energy at r32/48/64 (in-dim is 256),
so sub-attn rank there is near-free. The reg_dims path keeps the NETWORK alpha
(hotter alpha/rank scale on adaln by rank/adaln_rank; warm start folds scale so
init is exact). The extractor mirrors it as `--adaln_rank`. Guards in
`tests/test_turbo_adaln.py`.

### Sizing `adaln_rank` / `adaln_alpha` (measured 2026-07-18)

Trained adaln deltas are strongly low-effective-rank — rank-matching adaln to
the network rank is overprovisioning. SVD spectrum of the learned `.2` ΔW
(mean over the 84 modules; energy fraction captured at rank r):

| Source | eff. rank | top-1 | r8 | r16 | r32 |
|---|:-:|:-:|:-:|:-:|:-:|
| Official turboV10, exact full-rank delta (full fine-tune, all 256 dims free) | 10.8 | 51% | 86% | 93% | 96% |
| r96 ASVD extract of it | 7.9 | 52% | 88% | 95% | 98% |
| superturbo_E student (alloc r16) | 8.6 | 43% | 87% | 96% | — |
| soup s1001 style LoRA (alloc r32) | 13.1 | 37% | 74% | 89% | 100% |
| same ckpt, attn/MLP for contrast (alloc r32) | 25.3 | 13% | 49% | 73% | 100% |

Half the energy sits in ONE direction regardless of objective (DP-DMD distill
vs style FM) or parametrization — a property of the pathway (global
tone/gate/σ-remap), not the task. attn/MLP nearly saturates whatever rank it is
given; adaln fills ~40–55% of it with a fast-decaying tail. Default
`adaln_rank = 16` (keeps ~93% of even the official full-rank delta); 8 is
defensible; matching a rank-64 network wastes ~30M params on low-energy tail
(4.3M at r8). The r32/48/64 extract-energy numbers above are the near-lossless
extraction bar, not the training-allocation bar.

`adaln_alpha` follows the √r law, not linear α/r. Optimal LoRA α scales
sublinearly as α\*(r) ≈ C·√r ([LoRA-α, arXiv 2606.12883](https://arxiv.org/abs/2606.12883)),
so cross-rank scale preservation means keeping α/√r constant:

```
adaln_alpha = network_alpha · sqrt(adaln_rank / network_rank)
```

This is the LoRA-family default as of 2026-07-20 — `from_kwargs` computes it
whenever `adaln_alpha` is 0/unset, so the branch stays scale-matched under any
`network_dim`/`network_alpha` (including a CLI `--network_alpha` override).
base.toml previously hard-coded `adaln_alpha = 90`, which is this formula
evaluated for one stack and was ~4× hot on every `gui-methods` variant (32/32).
The bespoke turbo distill loop (`scripts/distill_turbo/`) keeps its own
explicitly-pinned `adaln_alpha` and is unaffected.

e.g. superturbo: student 180@r64 → adaln 90@r16 (both α/√r = 22.5). The naive
linear rule (45 in that example) systematically under-scales the smaller-rank
module — the ×2 hotter α/r multiplier is the √r law's prescribed compensation,
not a hot arm. We borrow only the *relative* √r consistency; the paper's
absolute α_base ≈ 256√r calibration is for fresh-init LLM SFT and does not
transfer to warm-started adapters.

Compile interaction (fixed 2026-07-15): training adaln under
`compile_dynamic_seq` initially crashed at the first grad-bearing forward with a
`ConstraintViolationError` on the marked seq range. The adaln LoRA makes
shift/scale/gate require grad, so the backward gains a seq-axis reduction whose
inductor mix-order-reduction fusion records a 4096-boundary guard (`Ge(seq, 4096)`
or `seq <= 4095`, per the first-traced hint) — contradicting the strict
`mark_dynamic` bound. Fixed by disabling `triton.mix_order_reduction` whenever
dynamic-seq marks are active, via `pin_inductor_flag` (2026-07-17) — the
initial plain-assignment kill was context-local (inductor config overrides are
thread-local ContextVars) and reverted in the grad-enabled compile context,
which is what crashed v1.14.0 `make lora` runs under grad-ckpt presets. Full
mechanism in
[`../optimizations/for_compile.md`](../optimizations/for_compile.md) §2.6 and
[[project_inductor_mix_order_reduction_guard]]. Applies to ANY LoRA on a
broadcast-consumed Linear, not just adaln.

Shipping trained adaln: the save pipeline relays runtime-layout keys to the
comfy layout for you — `lora_save.py::_relayout_adaln_to_comfy` runs on the
standard write path (after the qkv defuse, before hashing), stamps
`ss_adaln_layout = "comfy"`, and is presence-gated so adaln-less checkpoints are
untouched. The in-repo loader renames back on load
(`create_network_from_weights` → `relayout_adaln_comfy_to_runtime`), so one file
round-trips both ecosystems. Every method on the standard path is covered,
turbo included; the MoE/chimera `_moe` siblings are not ComfyUI-loadable
regardless, and the turbo per-step-expert layout (`step_expert_K > 1`) writes
verbatim and stays runtime-layout.

### What adaln can and can't learn

t-conditioned global modulation only: per-σ tone/contrast/gating shifts.
No text or spatial pathway. Plausible fits:

- Turbo distillation — proven load-bearing: the σ→behavior remap is exactly
  what few-step distillation changes, and the official turbo moved adaln hardest
  ([[project_official_turbo_v10_eval]] already lists adaln targets as a lever).
- Style LoRAs — speculative: a home for untagged global style (grade,
  palette bias), given cross-attn only learns labeled tags
  ([[project_lora_crossattn_learns_labeled_only]]). Could equally be inert or a
  new coupling hazard. Bench decides.

### Interaction with modulation guidance (MOD=1)

The adaln modulation MLPs are a shared pipe with three tenants
(`../inference/mod-guidance.md`):

1. Stock: t-embedding → per-σ shift/scale/gate.
2. Mod-guidance: adds `schedule[ℓ]·δ` (pooled-text steering delta) to the
   t-embedding *upstream of each block's modulation MLPs* — the same MLPs
   transduce the delta into coefficient steering. It also installs
   `pooled_text_proj`, making the t-embedding itself pooled-text-aware.
3. AdaLN LoRA (this doc): changes the MLP weights themselves.

Consequences:

- An adaln LoRA reshapes mod-guidance's steering. Guided coefficients become
  `(W+ΔW)(t_emb + w·δ) = stock + ΔW·t_emb + w·ΔW·δ` — the `ΔW·δ` term re-rotates
  the calibrated steering direction. Mod-guidance schedules and the external
  Spectrum node's scalar defaults were tuned with stock `W`; "MOD=1 composes
  with any checkpoint" silently becomes "…any checkpoint that doesn't touch
  adaln." Any bench arm that ships adaln to users must include a MOD=1 A/B.
  (Turbo inference runs CFG 1.0 / 4-step without mod-guidance, so the turbo
  student arm is unaffected.)
- The "adaln is text-blind" claim is stock-model-only. On a mod-distilled
  model (`pooled_text_proj` installed and loaded — note the gotcha in
  [[project_sea_delta_generalizes_guidance]]: it loads in `load_dit_model`),
  the t-embedding carries max-pooled text, so an adaln LoRA *trained on such a
  model* would learn a weakly pooled-text-conditioned modulation response
  (global tags, not token-level). Tested 2026-07-18 — refuted at Phase 0:
  a soup ingredient retrained with the projection loaded frozen (train.py now
  has a `--pooled_text_proj` knob for this) learned zero extra text-coupling —
  ΔU text-coupling index ≈0.003, identical to the stock-trained arm. The FM
  loss gives modulation no pressure to carry text while cross-attn already
  does; shifting the operating point alone is insufficient. The only measurable
  effect is a cost: the checkpoint becomes proj-coupled (~1.7× higher
  injection dependence at render, i.e. off-distribution in a stock ComfyUI
  flow without the proj). Don't re-propose mod-aware LoRA training without an
  explicit text→modulation objective. [[project_modaware_adaln_phase0]]
- Independent evidence convergence: mod-guidance's effectiveness (ICLR'26
  result) and the official turbo's adaln movement both say the modulation
  pathway is the high-leverage lever for global behavior — and neither says
  it carries content. Consistent with keeping expectations modest for style
  LoRAs.

#### Compatibility design options (escalate only on evidence)

Magnitude bound first: the steering distortion is `‖ΔW‖/‖W‖` per block — ≤1–3%
for the turbo extract; trained style LoRAs at hot alpha reach 3.6% (soup
s1001, alpha 128 @ r32). Measured 2026-07-18 (weight-space probe): even at
3.6% the adaln LoRA transduces the steering delta untouched — norm ratio
1.000, cosine 0.9999, all σ. So MOD=1 looking similar to MOD=0 on a soup is
NOT adaln interference; the steering effect is just inherently small on that
LoRA (mod3−w0 LPIPS ≈0.008 at w=3), absorbed LoRA-generically if at all.

0. Do nothing + verify (default): measured-correct at current scales
   (≤~4% rel. magnitude) — stop here. Options below stay documented only for
   a future regime (much hotter adaln or evidence of real rendered drift).
1. Ship adaln as a separate file — worth doing regardless of the coupling:
   the extractor's module specs already make an adaln-only emit trivial, ComfyUI
   chains LoRA loaders natively, and it buys a free per-part strength knob,
   free A/Bs, and a zero-cost opt-out for MOD users. Downside is two-file UX.
2. Magnitude renorm (cheap patch): rescale the baked δ per block so the
   transduced steering norm matches the stock-`W` calibration. Fixes magnitude,
   not rotation — but rotation is the ≤3% part.
3. Frozen steering path (structural fix): compute the guidance transduction
   through stock modulation weights — bake per-block coefficient-space deltas
   `δc(ℓ,br) = M⁰(ē+δ) − M⁰(ē)` offline from the base checkpoint (σ-flat, same
   approximation the current bake already accepts) and add them post-MLP.
   Guidance becomes checkpoint-independent by construction; distributable like
   the CNS γ auto-download. Only worth building if option 0 shows real drift —
   note ComfyUI merges LoRA into weights at patch time, so stock `W` must come
   from an offline bake, not the live model.
4. Per-LoRA recalibration: retrain/re-bake mod-guidance with the LoRA
   merged. Correct but doesn't scale to an ecosystem; not recommended.

### Existing checkpoints

Every LoRA trained in this repo to date has zero adaln keys (verified on
atlas / colorize / soup checkpoints) — the pathway was never trained, nothing is
mis-shipped, and there is nothing to retrofit. Trained models learned to
compensate through attn/MLP; see caveat below.

## Bench plan (gate before any default change)

1. Turbo student arm (highest expected value): student targets `adaln_up_`
   via include_patterns, warm-started from a runtime-layout adaln-inclusive
   extraction; vs current student. Rank by rendered 4-step grid
   ([[project_turbo_lr_instability_threshold]] — never by fm_mse), CMMD
   within-run only ([[project_cmmd_val_signal]]).
2. Style-LoRA arm: same artist dataset ± adaln targeting, rendered A/B.
3. Zero-training stacking probe (cheapest first signal): in ComfyUI, chain
   our published turbo student + an adaln-only comfy-layout extract of the
   official delta. Caveat: the student trained *without* the adaln shift and
   compensated around it — stacking is off-distribution; a win here is a strong
   green light, a loss is not conclusive against arm 1.

## Checklist

- [x] Extractor: `--include_adaln` + `--adaln_layout comfy` (rename at save)
- [x] Shipped comfy-layout extraction artifact (r96 ASVD + adaln)
- [x] Verified against ComfyUI's own `load_lora` (all 364 targets matched)
- [x] Comfy-layout export step for trained checkpoints (save-time, all methods)
- [ ] Bench arm 3 (stacking probe — no training)
- [ ] Bench arm 1 (turbo student + adaln, warm-started)
- [ ] Bench arm 2 (style LoRA ± adaln)
