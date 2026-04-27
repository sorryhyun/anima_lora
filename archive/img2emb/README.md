# img2emb — image → embedding resampler

Maps a vision-encoder-encoded reference image to a DiT-compatible
cross-attention context, replacing the text-encoder path entirely. A
Perceiver resampler with per-group classifier heads predicts class
prototypes for a handful of "anchor" slots (rating, people_count, …) and
fills the remaining content slots from vision features.

Anchor injection runs at two places in the resampler:
- **Pre-injection**: the predicted (or teacher-forced) anchor prototype is
  projected into the query space and written into the initial latent queries
  at each anchor's slot position *before* the backbone runs. This conditions
  the non-anchor slots on the classifier's decision. The input projection is
  identity-initialized when `d_out == d_model`.
- **Post-injection**: the exact prototype mix is spliced into the resampler
  output at the same slots (replace / residual modes), via
  `inject_spec_anchors`.

## Pipeline

Three stages, run top-to-bottom. Each reads the previous stage's outputs.

| Stage | Script | Role |
|---|---|---|
| 1. preprocess | `preprocess.py` | Cache TIPSv2 patch tokens + pooled features; build train/eval split; scan T5 active lengths. (Also runs `align_variants.py` defensively as a no-op for caches written by the current TE encoder, which already aligns by construction — see "T5 variant alignment" below.) |
| 2. pretrain | `pretrain.py` | Train `AnchoredResampler` on cached T5 `crossattn_emb` targets with CE/BCE classifier heads + prototype-anchor injection. |
| 3. finetune | `finetune.py` | Warm-start from the pretrain ckpt; supervise via flow-matching MSE through the frozen DiT. |

`train.py` is the top-level dispatcher — it imports each stage's `*(args)`
entrypoint and runs them in-process (no subprocesses). Inference uses
`infer.py`.

### Encoders

Three vision towers are wired in. Pick one with `--encoder`; **train and
infer must use the same encoder** because the cache filenames + `T_MAX_TOKENS`
both vary per encoder.

| `--encoder` | Model | Native res | Patch | Buckets ~tokens | D_enc / D_pool | Setup |
|---|---|---|---|---|---|---|
| `tipsv2` (default) | TIPSv2-L/14 | 448² | 14 | ~1024 (+CLS) | 1024 / 1024 | `make download-tipsv2`, `trust_remote_code` |
| `pe` | PE-Core-L14-336 (Meta) | 336² | 14 | ~576 (+CLS) | 1024 / 1024 | `make download-pe` (vision tower vendored at `library/models/pe.py`) |
| `pe-g` | PE-Core-G14-448 (Meta) | 448² | 14 | ~1024 (no CLS) | 1536 / 1280 | `make download-pe-g` (same vendored tower; bigger backbone — 50 layers, width 1536) |

Aspect-preserving bucketed preprocessing is always on — each image is
resized to the closest patch-grid bucket for its encoder (`scripts/img2emb/buckets.py`)
and tokens are zero-padded to that encoder's `T_MAX_TOKENS` so the cache
stays a flat `(N, T_MAX, D)` tensor.

PE is run via a vendored vision-only port of Meta's `perception_models` —
no clone, no `xformers`, no `core.*` package on `sys.path`. The vendored
module needs only `torch`, `einops`, and `timm.layers.DropPath` (already
in this project's deps).

### Running it

Preprocessing (features + anchor artifacts) is split from training so the
heavy one-time steps don't rerun every time you tune hyperparams.

```bash
# One-time preprocessing: vision-encoder features + anchor prototypes.
# Requires `make download-tipsv2` (default) or `make download-pe`.
make preprocess-img2emb
python tasks.py preprocess-img2emb

# Training (pretrain → finetune).
make img2emb
python tasks.py img2emb

# Run with PE-Core-L14-336 instead of TIPSv2 — the flag must match across
# every stage of one run, and is part of every cached/output filename.
python scripts/img2emb/train.py all --encoder pe
python scripts/img2emb/preprocess.py --encoder pe
python scripts/img2emb/pretrain.py   --encoder pe
python scripts/img2emb/finetune.py   --encoder pe
python scripts/img2emb/infer.py      --encoder pe --ref_image ref.png

# One stage at a time; trailing args forward to the underlying script.
python scripts/img2emb/train.py preprocess
python scripts/img2emb/train.py pretrain
python scripts/img2emb/train.py finetune --steps 20000
make img2emb-pretrain
make img2emb-finetune
make img2emb-calibrate       # step-0 loss magnitudes, no backward

# Or invoke a stage directly.
python scripts/img2emb/preprocess.py
python scripts/img2emb/pretrain.py --steps 5000
python scripts/img2emb/finetune.py --steps 20000

# Inference from a reference image. Without --image_size, the closest
# CONSTANT_TOKEN_BUCKETS aspect ratio to the ref is auto-picked.
make test-img2emb REF_IMAGE=post_image_dataset/foo.png
python scripts/img2emb/infer.py --ref_image ref.png
```

Outputs land under `output/img2embs/{features,anchors,pretrain,finetune}/`.
`infer.py` loads the finetune ckpt from `output/img2embs/finetune/` and
anchor prototypes from `output/img2embs/anchors/` by default.

## Anchor spec (`anchors.yaml`)

Each top-level key defines one **anchor group** — a classifier head over the
pooled encoder feature, a frozen prototype table loaded from inversionv2's
`tag_slot` outputs, and anchor slot(s) in the resampler output.

```yaml
rating:
  mutex: true
  proto_key_prefix: "rating="
  default_slot: 0
  classes: [explicit, sensitive, general]

people_count:
  mutex: true
  proto_key_prefix: "people="
  default_slot: 2
  # Composite, exhaustive, disjoint buckets. Prototypes are re-derived from
  # real slot-level T5 embeddings (not averages of per-tag prototypes).
  classes: ["1girl", "1girl, 1boy", "2girls", "2girls, 1boy", "1boy", "multi", "no_people"]
```

### Fields

| Field | Default | Meaning |
|---|---|---|
| `mutex` | `true` | `true` → softmax + CE, 1 anchor slot/image holds the softmax-weighted prototype mix. `false` → sigmoid + BCE (Option A multi-label); 1 anchor slot **per class**, populated only when that class is positive. |
| `classes` | *(required)* | List of class names, or `auto` to load every key from `proto_file`. Names are looked up as `f"{proto_key_prefix}{name}"`. |
| `proto_file` | `phase2_class_prototypes.safetensors` | Prototype safetensors under `--tag_slot_dir`. |
| `proto_key_prefix` | `""` | Prepended to each class name during lookup. |
| `default_slot` | `0` | Inference fallback (training uses per-image slots from `phase1_positions.json`). For multi-label groups, broadcast to every class unless `default_slots` is set. |
| `default_slots` | `None` | Per-class slot list (multi-label only). Length must equal `classes`. |

Missing prototypes are zero-filled with a warning. Each group also gets an
implicit `<unknown>` zero row at `index == n_classes`.

### Mutex vs multi-label

Groups that are internally exclusive (`people_count`'s exhaustive buckets,
`rating`'s explicit/sensitive/general) stay `mutex: true` — the head is
softmax over `n_classes + 1` rows (last row absorbs probability mass when
nothing matches). Training picks the earliest-slot class when a caption has
multiple active classes in the group (canonical booru ordering).

If you add a group whose classes can co-occur on the same image, flip
`mutex: false`. Each class then gets its own slot and is injected
independently based on the sigmoid decision (> 0.5).

Inference slot overrides (`infer.py`):

```bash
python scripts/img2emb/infer.py --ref_image ref.png \
    --slot_override rating=0,people_count=2
```

## Files

```
anchors.yaml          # Anchor-group spec (user-editable).
anchors.py            # AnchorSpec/AnchorGroup, AnchoredResampler, label building,
                      # inject_anchors, aux_cls_loss. Shared by all stages.
resampler.py          # PerceiverResampler backbone (no anchor / classifier code).
buckets.py            # Per-encoder patch-grid bucket specs + aspect-pick helpers.
encoders.py           # Vision-encoder registry (TIPSv2 + PE) — loaders, processors,
                      # model-id defaults, pooled/token dims.
data.py               # Shared dataset helpers, cache loader, resampler regression loss.
align_variants.py     # Hungarian-align T5 caption-shuffle variants in cached _anima_te.safetensors.
                      #   Runs implicitly as the first step of preprocess.py; idempotent (marker key).
preprocess.py         # Stage 1: variant alignment + vision-encoder feature extraction + split + active-length scan.
pretrain.py           # Stage 2: pretrain against cached T5 crossattn targets.
finetune.py           # Stage 3: flow-matching finetune through frozen DiT.
train.py              # Top-level in-process 3-stage dispatcher.
infer.py              # Reference-image → generated image inference.
rebuild_anchor_artifacts.py  # Refresh phase1_positions.json + class prototypes.
```

## T5 variant alignment

The text-encoding pipeline produces V=4 caption *shuffles* per image (booru
tag reordering) and caches each variant's T5 outputs under keys
`crossattn_emb_v{0..3}` in `{stem}_anima_te.safetensors`. T5 is positional,
so shuffling tags produces wildly different per-token outputs even though
the underlying content (set of tokens) is identical:

| Within-image inter-variant per-token cosine | |
|---|---|
| Raw, no alignment | ~0.33 |
| Hungarian-aligned to v0 (max-pool also ~1.0 confirms same content) | ~0.97 |

Without alignment, position-wise MSE / cosine against the variant_mean
target is pathological — the variant_mean averages over RNG-randomized
positions, smearing the positional structure to a noise centroid that no
resampler can fit. Pretrain plateaus at moderate cos (~0.5/0.63 mid/tail)
not because of architecture or encoder limits, but because it's chasing
malformed targets.

**Where alignment happens.** As of the current TE encoder
(`library/anima/strategy.py::AnimaTextEncoderOutputsCachingStrategy._align_crossattn_to_v0`),
alignment is folded into the cache writer itself: every newly produced
`{stem}_anima_te.safetensors` is written with v1..v{N-1}'s
`crossattn_emb_v*` already Hungarian-aligned to v0, and an
`aligned_to_v0` marker tensor set. The standalone `align_variants.py`
script (and `make img2emb-align`) exists to upgrade *legacy* caches
written before this change — idempotent via the same marker. After a
fresh re-encode, both are no-ops.

**What gets permuted.** *Only* `crossattn_emb_v*`. The cache also holds
`prompt_embeds_v*` (pre-adapter Qwen3 hidden states), `t5_input_ids_v*`,
and per-variant attention masks. These feed the LLM adapter at training
time when `cache_llm_adapter_outputs=false`, and the adapter's internal
cross-attn DOES apply K-side RoPE, so permuting them would change adapter
outputs and silently corrupt LoRA training in that path. We deliberately
leave them in their original (per-shuffle) order; the cache is internally
inconsistent (`prompt_embeds_v1` no longer round-trips through the adapter
to `crossattn_emb_v1`), but no code path reads both at once.

**Why it's a no-op for LoRA training.** DiT cross-attention has no K-side
RoPE (see `library/anima/models.py::compute_qkv` — RoPE is gated on
`is_selfattn`). The cross-attn output `Σⱼ softmax(QK)ⱼ Vⱼ` is invariant to
permutation of `(K, V)` rows. So row-permuting `crossattn_emb_v*` doesn't
change what the LoRA sees — only the supervision target seen by img2emb
pretrain (which uses positional MSE) actually benefits.

**Standalone re-run.** `make img2emb-align` re-aligns all TE files
(no-op for already-aligned ones).

## Conventions + notes

- All `--tag_slot_dir` defaults point at `output/img2embs/anchors/` (phase1
  positions + class prototypes). Regenerate with
  `python scripts/img2emb/rebuild_anchor_artifacts.py` (also wired in as
  `make img2emb-anchors`), then re-run pretrain — no code changes needed.
- Checkpoint state_dict keys use `heads.<group>.{weight,bias}` and
  `<group>_protos` buffers. Renaming a group in `anchors.yaml` breaks ckpt
  load; add a new group instead and retrain.
- Pretrain saves a JSON sidecar alongside each `.safetensors` containing
  `anchor_spec` (the resolved group/class metadata). This is the canonical
  record of what the model was trained against.
- The resampler's `forward` accepts `teacher_labels` + `tf_ratio` for
  per-sample teacher forcing during pretrain (anneal 1→0 so the model
  eventually trusts its own classifier). Without `teacher_labels`, every
  sample pre-fills with the predicted soft mix at each group's default slot.
- Training loads the cache without materializing the variant-mean targets
  (used to cost ~2 GB RAM). The mean is a poor training target anyway — it
  shrinks norms and sits off the T5 manifold — so pretrain / finetune sample
  one variant per step via `_ResamplerTrainDataset`. Diagnostics that still
  need the mean call `data.load_targets_mean` directly.
- See `bench/active/img2emb/proposal.md` and `bench/active/img2emb/phase2_proposal.md` for
  design history and ablation notes.
- **K-slot cap.** Pretrain / finetune / `infer` default `--n_slots 256`.
  The resampler predicts K=256 content slots — covers ~95% of active T5
  lengths L — and the output is zero-padded to 512 at every boundary that
  talks to the DiT (matching the cached `crossattn_emb` shape). The pad tail
  is exactly zero by construction, so no `--zero_pad_weight` / `--w_pad` is
  needed.
- **Multi-positive InfoNCE over shuffled caption variants.** Pretrain and
  finetune both accept `--w_infonce` (default 0.1 for pretrain, 0 for
  finetune) and `--infonce_tau` (default 0.07). The loss pools the
  resampler output over active slots (SupCon-style) and contrasts it
  against per-variant pooled T5 targets cached at
  `features/target_pooled.safetensors` (produced by `preprocess.py`;
  ~65 MB for N=1987, V=8, D=1024 fp32). Positives = all V variants of the
  same image; negatives = variants of other images in the batch. If the
  file is missing, InfoNCE is skipped with a warning.
