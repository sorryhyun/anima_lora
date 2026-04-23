# img2emb — image → embedding resampler

Maps a siglip2-encoded reference image to a DiT-compatible cross-attention
context, replacing the text-encoder path entirely. A Perceiver resampler with
per-group classifier heads predicts class prototypes for a handful of
"anchor" slots (rating, people_count, …) and fills the remaining content
slots from vision features.

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
| 1. features | `extract_features.py` | Cache siglip2 patch tokens + pooled features; build train/eval split; scan T5 active lengths. |
| 2. pretrain | `phase1_5_anchored.py` | Train `AnchoredResampler` on cached T5 `crossattn_emb` targets with CE/BCE classifier heads + prototype-anchor injection. |
| 3. finetune | `phase2_flow.py` | Warm-start from the phase-1.5 ckpt; supervise via flow-matching MSE through the frozen DiT. |

`train_img2emb.py` is the top-level dispatcher — it just `subprocess`'s
each stage script with shared paths. Inference uses `test_img2emb.py`.

### Running it

Preprocessing (features + anchor artifacts) is split from training so the
heavy one-time steps don't rerun every time you tune hyperparams.

```bash
# One-time preprocessing: siglip2/dinov3 features + anchor prototypes.
# Requires `make download-siglip2` once (vision tower lives at models/siglip2).
make preprocess-img2emb
python tasks.py preprocess-img2emb

# Alternative encoder: TIPSv2-L/14 at 448² (patch 14, 1025 tokens, D=1024).
# Fixed-res like siglip2 — no variable aspect. Needs `make download-tipsv2`
# once, then pass ENCODER=tipsv2 to both preprocessing and training:
ENCODER=tipsv2 make preprocess-img2emb
ENCODER=tipsv2 make img2emb
# And at inference time:
ENCODER=tipsv2 make test-img2emb REF_IMAGE=post_image_dataset/foo.png

# Aspect-preserving bucketing (TIPSv2 only). Each image is resized to the
# closest patch-14 bucket (~1024 tokens, aspects 1:2..2:1) instead of
# square-cropping to 448². Tokens are zero-padded to a single T_MAX so the
# cache stays (N, T_MAX, D) — no resampler/sampler changes needed. See
# scripts/img2emb/buckets.py for the spec. **Auto-enabled with ENCODER=tipsv2**;
# must be set consistently across preprocess AND test-img2emb (the resampler
# sees a different KV length with vs without).
ENCODER=tipsv2 make preprocess-img2emb                # buckets ON by default
ENCODER=tipsv2 make img2emb
ENCODER=tipsv2 make test-img2emb REF_IMAGE=post_image_dataset/foo.png
# Force-disable to match an older fixed-448² cache:
ENCODER=tipsv2 BUCKETS= make preprocess-img2emb

# Training (pretrain → finetune).
make img2emb
python tasks.py img2emb

# One stage at a time; trailing args forward to the underlying script
python scripts/img2emb/train_img2emb.py pretrain
python scripts/img2emb/train_img2emb.py finetune --steps 20000
make img2emb-pretrain
make img2emb-finetune
make phase2-calibrate       # step-0 loss magnitudes, no backward

# Inference from a reference image. Without --image_size, the closest
# CONSTANT_TOKEN_BUCKETS aspect ratio to the ref is auto-picked.
make test-img2emb REF_IMAGE=post_image_dataset/foo.png
python scripts/img2emb/test_img2emb.py --ref_image ref.png
```

Outputs land under `output/img2embs/{features,anchors,pretrain,finetune}/`.
`test_img2emb.py` loads the finetune ckpt from `output/img2embs/finetune/`
and anchor prototypes from `output/img2embs/anchors/` by default.

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

Inference slot overrides (`test_img2emb.py`):

```bash
python scripts/img2emb/test_img2emb.py --ref_image ref.png \
    --slot_override rating=0,people_count=2
```

## Files

```
anchors.yaml          # Anchor-group spec (user-editable).
anchors.py            # AnchorSpec/AnchorGroup, AnchoredResampler, label building,
                      # inject_anchors, aux_cls_loss. Shared by all stages.
resampler.py          # PerceiverResampler backbone (no anchor / classifier code).
data.py               # Shared dataset helpers, cache loader, resampler regression loss.
extract_features.py   # Stage 1: siglip2/dinov3 feature extraction + split generation.
phase1_5_anchored.py  # Stage 2: pretrain against cached T5 crossattn targets.
phase2_flow.py        # Stage 3: flow-matching finetune through frozen DiT.
train_img2emb.py      # Top-level 3-stage dispatcher.
test_img2emb.py       # Reference-image → generated image inference.
```

## Conventions + notes

- All `--tag_slot_dir` defaults point at `output/img2embs/anchors/` (phase1
  positions + class prototypes). Regenerate with
  `python scripts/img2emb/rebuild_anchor_artifacts.py` (also wired in as
  `make img2emb-anchors`), then re-run pretrain — no code changes needed.
- Checkpoint state_dict keys use `heads.<group>.{weight,bias}` and
  `<group>_protos` buffers. Renaming a group in `anchors.yaml` breaks ckpt
  load; add a new group instead and retrain.
- Phase 1.5 saves a JSON sidecar alongside each `.safetensors` containing
  `anchor_spec` (the resolved group/class metadata). This is the canonical
  record of what the model was trained against.
- The resampler's `forward` accepts `teacher_labels` + `tf_ratio` for
  per-sample teacher forcing during pretrain (anneal 1→0 so the model
  eventually trusts its own classifier). Without `teacher_labels`, every
  sample pre-fills with the predicted soft mix at each group's default slot.
- Training loads the cache without materializing the variant-mean targets
  (used to cost ~2 GB RAM). The mean is a poor training target anyway — it
  shrinks norms and sits off the T5 manifold — so phase 1 / 1.5 / 2 sample
  one variant per step via `_ResamplerTrainDataset`. Phase-0 diagnostics
  that still need the mean call `data.load_targets_mean` directly.
- See `bench/img2emb/proposal.md` and `bench/img2emb/phase2_proposal.md` for
  design history and ablation notes.
- **K-slot cap.** Phase 1.5 / phase 2 / `test_img2emb` default `--n_slots 256`.
  The resampler predicts K=256 content slots — covers ~95% of active T5
  lengths L — and the output is zero-padded to 512 at every boundary that
  talks to the DiT (matching the cached `crossattn_emb` shape). The pad tail
  is exactly zero by construction, so no `--zero_pad_weight` / `--w_pad` is
  needed.
- **Multi-positive InfoNCE over shuffled caption variants.** Phase 1.5 and
  phase 2 both accept `--w_infonce` (default 0.1) and `--infonce_tau`
  (default 0.07). The loss pools the resampler output over active slots
  (SupCon-style) and contrasts it against per-variant pooled T5 targets
  cached at `features/target_pooled.safetensors` (produced by
  `extract_features.py`; ~65 MB for N=1987, V=8, D=1024 fp32). Positives =
  all V variants of the same image; negatives = variants of other images in
  the batch. If the file is missing, InfoNCE is skipped with a warning.
