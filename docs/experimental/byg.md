# BYG — Bootstrap Your Generator (unpaired instruction editing)

Training-based instruction editing for Anima that needs no paired
`(source, edited)` data and no reward model — only unpaired images plus
`(src_caption, tgt_caption, instruction, reverse_instruction)` text tuples. The
trainable weights are an ordinary rank-64 Anima LoRA (zero inference
overhead, plain-LoRA checkpoint); everything BYG-specific lives in the training
step and a parameter-free conditioning patch.

- Paper: Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching
  (Tewel, Atzmon, Chechik, Wolf; arXiv 2606.03911), Alg. 1.
- Design rationale + Phase-0 validation gate: `bench/byg/README.md`
  (demoted from `docs/proposal/` 2026-07-02 — re-propose after the gate passes).
- Code: `networks/methods/byg.py` (`BYGMethodAdapter` + `BYGConditioning`),
  `library/training/forward/ste.py`, `library/datasets/base.py` (sidecar loader),
  `scripts/byg/build_edit_tuples.py` (data), `configs/methods/byg.toml`.

> Status (v1, phased). Training is functional end-to-end and verified
> (data build → multi-forward step with all four losses + staged identity
> backward + snapshot bootstrap → plain-LoRA save). Inference is not wired
> yet — `exp-test-byg` raises with a pointer; the trained checkpoint loads via
> the standard `--lora_weight` path but the source-concat conditioning patch
> still needs installing at generation time. See [Status & next phase](#status--next-phase).

## Why it's interesting

Instruction editing normally needs paired before/after data (expensive, scarce)
or RL against a reward model (unstable, reward-hackable). BYG sidesteps both: it
bootstraps its own pseudo-pairs from the frozen base T2I model and an
unpaired image corpus, then trains a small LoRA to follow instructions while a
directional prior keeps the edit on the base model's manifold and a cycle loss
keeps it reversible. The output is a plain LoRA — no architectural surgery, no
extra inference cost.

## Conditioning — parameter-free source-latent concat

BYG conditions the edit on the source image via Kontext-native single-concat
semantics (gate-free; the source shares the target's RoPE positions because
source and target are always same-resolution). On Anima a literal pre-PatchEmbed
sequence concat breaks the eager 5D bucketing (source + target form no
rectangular patch grid), so the concat is realized at the attention level:

- Each block's target tokens run extended self-attention over
  `[target_k ; source_k]` / `[target_v ; source_v]`.
- The source stream evolves through the same LoRA'd block linears (its
  `qkv_proj` already carries the standard LoRA delta — the monkey-patched
  `Linear.forward`), so the one trainable LoRA adapts the unified model.

This reuses EasyControl's extended-self-attention LSE helper
(`_extended_target_attention`) but strips the `b_cond` gate (fixed at 0) and
the cond-LoRA branch — the only trainable weights are the standard LoRA, so the
checkpoint is a plain LoRA loadable by the normal loader. The source stream's own
self-attn/MLP feed only the next block, so they are skipped on the last block
(dead output).

`BYGConditioning` holds no trainable parameters (deliberately not an
`nn.Module`, so it stays off `network.parameters()`). It patches every
`Block.forward` to run the stripped two-stream when a source latent is primed via
`set_source(...)` and falls through to the exact baseline DiT forward when no
source is set — which is how the prior forwards (§objective step 3) get a clean
frozen-base T2I pass.

```
target stream (frozen DiT + standard LoRA)     source stream (same DiT + same LoRA, t=0)
──────────────────────────────────────────     ──────────────────────────────────────────
AdaLN_self(t_emb = t)                           AdaLN_self(src_emb = t_embedder(0))
self_attn.compute_qkv(target_normed)            self_attn.compute_qkv(src_normed)   ← same LoRA'd qkv_proj
        │                                               │
        ▼  ◄── target attends to [target_k; src_k] ─────┘   (b_cond = 0, no gate)
target_out = LSE-extended attn
        │                                       src_out = own SDPA self-attn → feeds NEXT block only
   cross-attn (target only) + MLP                    (skipped on last block)
```

## Objective (paper Alg. 1)

Per step with `t ~ U(0,1)`, `ε ~ N(0,I)`, source latent `x` (the training image
*is* the source). `t` is discretized to the rollout grid so the mid-rollout
state `ỹ_t` is captured exactly. `c` = instruction, `c̄` = reverse instruction,
`p_src` / `p_tgt` = source / target captions.

1. Bootstrap (no_grad, snapshot weights, source=x, instr=c): an n-step
   Euler rollout `ỹ_1 = ε → ỹ_0`, capturing `ỹ_t` mid-rollout. Yields both
   the pseudo-noisy target `ỹ_t` and the clean multi-step estimate `ỹ_0`.
2. Forward (grad, source=x, instr=c): `v_fwd = G(ỹ_t, t, c, x)`;
   one-step clean-edit prediction `ŷ = ỹ_t − t·v_fwd`.
3. Prior (no_grad base, LoRA multiplier=0, no source → baseline T2I):
   `v_src = G(ỹ_t, t, p_src)`, `v_tgt = G(ỹ_t, t, p_tgt)`;
   `L_dir = 1 − cos(v_fwd − v_src, v_tgt − v_src)`,
   `L_MSE = ‖v_fwd − v_tgt‖²`, `L_prior^fwd = L_dir + α·L_MSE`. (DDS directional
   prior — keeps the edit aligned with the base model's `src→tgt` direction.)
   With `byg_prior_symmetric = true` (v2 default, paper Eq. 5) a reverse prior
   is added on the cycle state: anchor `v_rev` (step 4) to the base's
   `edited→source` direction — the same DDS term at `x_t` with `p_src`/`p_tgt`
   swapped — so both edit directions stay on-manifold. `L_prior = L_prior^fwd +
   L_prior^rev`. Logged split as `byg/L_prior_fwd` / `byg/L_prior_rev` (+ total
   `byg/L_prior`).
4. Cycle (grad, source=ŷ_hyb, instr=c̄): `x_t = (1−t)x + tε`,
   `ŷ_hyb = sg(ỹ_0) + (ŷ − sg(ŷ))` (STE blend, Eq. 4),
   `v_rev = G(x_t, t, c̄, ŷ_hyb)`, `L_cycle = ‖v_rev − (ε − x)‖²` — applying the
   reverse instruction to the edit must reconstruct the source.
5. Identity (grad, source=x, instr=c̄): `L_id = ‖G(x_t, t, c̄, x) − (ε − x)‖²`
   — staged on an independent graph (anti-collapse anchor + a VRAM win, see
   below).

Total (non-identity steps): `L = λ_cycle·L_cycle + λ_prior·L_prior`, with the
identity term backwarded separately.

### STE clean/one-step blend (Eq. 4)

`library/training/forward/ste.py::ste_clean_blend` — `ŷ_hyb = sg(ỹ_0) + (ŷ − sg(ŷ))`.
The forward value equals the clean multi-step estimate `ỹ_0` (so the cycle pass
conditions on inference-quality inputs), while the gradient flows through the
cheap one-step prediction `ŷ` (differentiable w.r.t. the trainable forward
velocity). It is a general flow-matching train/inference-mismatch fix, so it
lives as a standalone utility, not buried in the BYG module. It asserts the 5D
`(B,C,1,H,W)` dim-2 invariant rather than letting a silent ndim mismatch
broadcast (the exact class of dim-2 bug that bit FreeText repeatedly).

## Snapshot bootstrap (vs EMA)

The bootstrap rollout (step 1) must run against a stable copy of the LoRA, not
the live weights mid-optimization. `BYGMethodAdapter` keeps a `_shadow` dict of
the trainable LoRA tensors:

- v1 default — snapshot: hard-refresh the shadow every `byg_snapshot_every`
  steps (200). EMA-free, cheap.
- paper-faithful — EMA: set `byg_ema_decay > 0` for a per-step EMA shadow.

The shadow is populated lazily on the first `_update_shadow` (when the LoRA
params are guaranteed on-device) — cloning at build time would capture CPU
tensors before accelerate's device move and corrupt the rollout swap. During the
rollout, `_SwapShadow` swaps live LoRA `.data` ↔ shadow and restores on exit; on
the very first step (no snapshot yet) it rolls out against the live weights.

## Deviations from the paper

Locked in the proposal's "Decisions locked"; all are config toggles:

| Default | Paper | Toggle |
|---|---|---|
| Snapshot bootstrap (refresh every N) | per-step EMA | `byg_ema_decay > 0` |
| Symmetric prior, Eq. 5 (v2 default) | symmetric (fwd + reverse) prior, Eq. 5 | `byg_prior_symmetric = false` ⇒ v1 fwd-only |
| `t` discretized to rollout grid | continuous `t` | (exact `ỹ_t` capture; structural) |
| Tag-swap data (no VLM) | VLM-generated tuples | `--vlm` tail (App. D.1, TODO) |

The reverse prior adds two frozen-base forwards per non-identity step (~14 →
~16, both `no_grad` so no activation retention). Set `byg_prior_symmetric = false`
for the cheaper v1 fwd-only prior.

## Identity loss — independent staged graph

`L_id` is backwarded on its own graph before the coupled forward builds. Two
reasons: (a) it's an anti-collapse anchor (`G(·, c̄, source=x)` with the reverse
instruction on the source must be a no-op edit → predict the plain FM target),
and (b) staging its backward early frees its activations before the much larger
coupled (bootstrap+forward+cycle) graph allocates — the documented +1.2 GB vs
+2.2 GB win. On identity-only steps (warmup or random `byg_identity_prob`
draws) BYG returns `λ_id·L_id` and lets the normal training loop backward it.

## Data — tag-swap edit tuples

`scripts/byg/build_edit_tuples.py` (`make exp-byg-data`) makes one offline pass
over the captioned corpus, emitting `post_image_dataset/byg/<stem>_byg.safetensors`
per image — the encoded (post-LLM-adapter `crossattn_emb` + mask) for the four
roles `src_caption / tgt_caption / instruction / reverse_instruction`.

v1 = tag-swap (paper App. D shortcut, no VLM): find a color word in the
caption and swap it for another, which mechanically yields a self-contained
reverse instruction (`change <new> to <old>`) and a temporal-language-free target
caption. Free, but cannot express style edits — the `--vlm` tail (App. D.1,
Qwen3-VL) is a later phase and currently raises `NotImplementedError`. Images
without a color tag emit no tuple (counted as `no_color`).

```bash
make exp-byg-data                      # full corpus → post_image_dataset/byg/
make exp-byg-data ARGS="--limit 200"   # smoke subset
make exp-byg-data ARGS="--overwrite"   # rebuild existing sidecars
```

The image VAE/TE caches are the standard `preprocess` ones (the source image
is the training image — no separate cond cache). Stems are matched to those
caches by basename, so the sidecar dir is flat. The dataset
(`library/datasets/base.py`) loads each sidecar into `batch["byg_{role}_emb"]` /
`["byg_{role}_mask"]` via `byg_text_dir` (default `post_image_dataset/byg`).

## Training

```bash
make exp-byg-data        # build edit-tuple sidecars first
make exp-byg             # train (configs/methods/byg.toml)
```

Key config (`configs/methods/byg.toml`, read by `BYGMethodAdapter` via args):

| Key | Default | Meaning |
|---|---|---|
| `network_dim` / `network_alpha` | 64 / 64 | plain LoRA rank (paper App. B.1) |
| `learning_rate` | 3e-4 | AdamW, wd 1e-2 (App. B.1) |
| `byg_lambda_prior` / `_cycle` / `_id` | 1.0 / 1.0 / 0.2 | loss weights |
| `byg_alpha` | 0.1 | DDS magnitude-anchor weight (`L_MSE`) |
| `byg_rollout_steps` | 10 | bootstrap Euler steps (and `t` grid) |
| `byg_snapshot_every` | 200 | snapshot refresh cadence |
| `byg_ema_decay` | 0.0 | >0 ⇒ EMA shadow instead of snapshot |
| `byg_identity_warmup_steps` | 200 | identity-only warmup |
| `byg_identity_prob` | 0.15 | random identity-only step probability |
| `byg_prior_symmetric` | true | symmetric prior (paper Eq. 5); false ⇒ v1 fwd-only |
| `byg_text_dir` | `post_image_dataset/byg` | sidecar dir |

### Step-owning adapter

The standard `flow_match` training path doesn't fit BYG — there is no
`target = noise − latents` and a step runs ~14 DiT forwards with staged
backward. So `BYGMethodAdapter.owns_training_step()` returns true and `train.py`
delegates the entire step to `compute_loss` (the
`MethodAdapter.owns_training_step` / `compute_loss` hooks). This is the general
escape hatch for methods whose objective isn't a single FM regression.

### Invariants & gotchas

- `blocks_to_swap = 0` is enforced — BYG runs many DiT forwards per step and
  the block-swap offloader desyncs on a second forward
  (`project_blockswap_extra_forwards_gradcache`). The adapter raises on nonzero.
- **`unsloth_offload_checkpointing` MUST stay `false`** — *not* for the
  offloader-desync reason (that's a separate, block-swap-only issue); the unsloth
  path is a reentrant checkpointer that only builds an autograd node when an
  explicit input requires grad. BYG's Forward (`L_prior`) and Identity
  (`L_id`) passes feed detached latents + a frozen source, so the only
  grad-requiring tensors are the closed-over LoRA params — which the reentrant
  path silently drops, leaving `L_prior` pinned at chance and only `L_cycle`
  training. Use plain `gradient_checkpointing` (`use_reentrant=False`). Proof &
  regression guard: `bench/byg/grad_parity.py`.
- No caption dropout — the instruction/captions *are* the supervision signal
  (`caption_dropout_rate = 0.0`).
- 5D dim-2 boundary — `_dit_velocity` does `unsqueeze(2)` in / `squeeze(2)`
  out; `BYGConditioning.encode_source` unsqueezes a 4D source to 5D. Never bare
  `squeeze()`.
- Prior path = exact baseline — `_BaseMode` clears the source (baseline
  `Block.forward` fall-through) and sets LoRA multiplier to 0, so `v_src` /
  `v_tgt` are the frozen base T2I velocities.

## Status & next phase

- ✅ Data (`exp-byg-data`) — tag-swap sidecars; now with a tqdm progress bar.
- ✅ Training (`exp-byg`) — full multi-forward step, all four losses, staged
  identity backward, snapshot bootstrap; saves a plain LoRA.
- ⛔ Inference (`exp-test-byg`) — not wired. Raises with a pointer. The
  trained weights load via `--lora_weight`, but generation still needs the
  `BYGConditioning` source-concat patch installed and primed with the
  VAE-encoded reference (mirrors the EasyControl KV-prefill path) before edits
  work. Until then, collapse-watch validation can't run.
- 🔜 VLM tuples (`--vlm`) — App. D.1 Qwen3-VL data for style edits, beyond
  the tag-swap color edits.

See `bench/byg/README.md` for the full design and the P2
inference plan.
