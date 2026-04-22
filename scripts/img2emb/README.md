# img2emb — image → embedding resampler

Maps a siglip2-encoded reference image to a DiT-compatible cross-attention
context, replacing the text-encoder path entirely. A Perceiver resampler with
per-group classifier heads predicts class prototypes for a handful of
"anchor" slots (rating, girl/boy count, …) and fills the remaining content
slots from vision features.

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

```bash
# End-to-end (features → pretrain → finetune, with production defaults)
python scripts/img2emb/train_img2emb.py all

# One stage at a time; trailing args forward to the underlying script
python scripts/img2emb/train_img2emb.py pretrain
python scripts/img2emb/train_img2emb.py finetune --steps 20000

# Makefile shortcuts (see Makefile for exact flags)
make img2emb                # alias for train_img2emb.py all
make img2emb-pretrain
make img2emb-finetune
make phase2-calibrate       # step-0 loss magnitudes, no backward

# Inference from a reference image
make test-img2emb REF_IMAGE=post_image_dataset/foo.png
python scripts/img2emb/test_img2emb.py --ref_image ref.png
```

Outputs land under `output/img2embs/{features,pretrain,finetune}/`. Phase
2a's ablation writes to `bench/img2emb/results/phase2a/` (legacy path, kept
so the bench comparisons still line up).

## Anchor spec (`anchors.yaml`)

Each top-level key defines one **anchor group** — a classifier head over the
pooled encoder feature, a frozen prototype table loaded from inversionv2's
`tag_slot` outputs, and anchor slot(s) in the resampler output.

```yaml
rating:
  mutex: true
  proto_key_prefix: "rating="
  default_slot: 0
  classes: [explicit, sensitive, general, absurdres]

girl_count:
  mutex: true
  default_slot: 2
  classes: [1girl, 2girls, 3girls]

boy_count:
  mutex: true
  default_slot: 5
  classes: [1boy, 2boys]
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

Groups that are internally exclusive (1girl vs 2girls vs 3girls) stay
`mutex: true` — the head is softmax over `n_classes + 1` rows (last row
absorbs probability mass when nothing matches). Training picks the
earliest-slot class when a caption has multiple active classes in the group
(canonical booru ordering).

If you add a group whose classes can co-occur on the same image, flip
`mutex: false`. Each class then gets its own slot and is injected
independently based on the sigmoid decision (> 0.5). This keeps 1girl and
1boy co-existence correct — put them in separate groups (as above) **or**
in one non-mutex group.

Inference slot overrides (`test_img2emb.py`):

```bash
python scripts/img2emb/test_img2emb.py --ref_image ref.png \
    --slot_override rating=0,girl_count=2,boy_count=10
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
- See `bench/img2emb/proposal.md` and `bench/img2emb/phase2_proposal.md` for
  the design history and ablation notes.
