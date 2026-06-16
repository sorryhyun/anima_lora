# near_twins — in-artist variant-pair miner

Mines near-duplicate **variant pairs** within a single artist where the two
members differ by a **specified attribute** — e.g. one panel has a speech bubble
(or burned-in text) and the other doesn't. It localizes *where* the two differ
for free, ranks the cleanest single-attribute pairs first, and materializes them
as a `_tags` / `_no_tags` pair tree wired straight into an EasyControl control
task (text/bubble removal — "sanitize").

The bottleneck in building an EasyControl removal/attribute-edit adapter is
finding real with-/without-attribute data. This tool finds those pairs in an
existing image corpus and stages them for training.

It lives under `easycontrol_adapters/` but is dataset-agnostic — point
`--image-dirs` at any `<dir>/<artist>/<id>.<ext>` tree (defaults to the crawl
pool `$CAPTION_CORPUS_DIR/{retrieved,selected}`, falling back to `~/gelcrawl/`).

> **Note:** this README is the live doc. The original design lived at a
> `near_twin_tag_gap_miner` proposal (removed once implemented). The one
> intentional divergence from that proposal: the **HTML contact sheet / TSV**
> "eyeball" artifacts were *not* built — iteration happens by editing the
> `[staging]` TOML and re-running (features are cached, so re-runs are seconds).

## Quick start (the EasyControl pipeline)

The tool is step 1 of a three-step EasyControl flow, all driven by one config
(`configs/easycontrol/near_twins.toml`) and selected with `EASYADAPTER`:

```bash
# 1. Mine the pair tree + (re)write the dataset blueprint
make easycontrol-staging    EASYADAPTER=near_twins
# 2. Resize → VAE/TE cache the pairs, then symlink each _tags latent in as the cond
make easycontrol-preprocess EASYADAPTER=near_twins
# 3. Train the EasyControl adapter on the mined pairs
make easycontrol            EASYADAPTER=near_twins
```

Everything routes under `post_image_dataset/easycontrol/<name>/{staging,resized,
cache,cond}` where `<name>` is the top-level `name` key in the config (ships as
`name = "sanitize"`). Change `name`, re-run preprocess → train, and the whole
pipeline reroutes with no path edits.

### Standalone (no make)

```bash
python -m easycontrol_adapters.tools.near_twins \
  --tag-any "speech bubble,thought bubble" \
  --image-dirs "$CAPTION_CORPUS_DIR/retrieved" \
  --sim-min 0.85 --match-frac-min 0.5 --max-extra-diff 6
```

CLI flags override the `[staging]` table in `--config`
(`configs/easycontrol/near_twins.toml` by default; `--config ''` disables it).

## How it works

Per artist (artists are small, so all-pairs is fine — `--id-window` is the escape
hatch for an oversized one). A **prefilter → rerank** pipeline keeps the
expensive dense match off the obvious non-pairs.

1. **Gather + size gate.** Discover `<artist>/<id>` members across `--image-dirs`;
   keep only members that *cohabit a size* with a candidate twin (region/signal
   modes need an exact `W×H` match; tag mode also needs a tagged↔untagged pivot).
   The GPU encoder is skipped entirely if nothing survives the gate.
2. **Embed (PE-Spatial-B16-512).** The spatial Perception Encoder shipped with
   the repo (`load_pe_encoder(name="pe_spatial")`, no extra HF gate) gives both
   halves from one forward: the **CLS** vector (Stage A) and the **32×32 patch
   grid** (Stage B). Features are cached per image under `~/.cache/near_twin/`.
3. **Stage A — global prefilter.** CLS-cosine over within-artist pairs; keep
   `≥ --sim-min`. One dot product per pair.
4. **Stage B — dense grid match (twin confirm + diff localizer).** On Stage-A
   survivors: pool each grid to `--grid`×`--grid` cells, mutual-NN + ratio test
   (`--ratio`), count inliers at cell-cosine `≥ --cell-match-min`. A pair is a
   near-twin when the inlier fraction `≥ --match-frac-min`. **The unmatched cells
   are the difference region** — the free attribute localizer. `--geom-check`
   adds a RANSAC translation consistency pass to reject "same character, different
   pose" twins.
5. **Discriminator** (mutually exclusive — pick one). Keeps pairs where the
   attribute is in **exactly one** member:
   - `--tag "speech bubble"` / `--tag-any "a,b,c"` — tag in exactly one tagset
     (matched space-insensitively: `speech bubble` ≡ `speech_bubble`).
   - `--region` — use Stage B's unmatched-cell region directly (no caption tag
     needed; recommended for bubbles / any untagged visual attribute). Shaped by
     `--region-min-frac` / `--region-max-frac` / `--region-scatter-max`.
   - `--signal mit_text` — a per-image scalar (MIT text-area fraction, the
     detector behind `post_image_dataset/masks/`) differs by `≥ --signal-delta`
     with the low side ≈ 0.
6. **Rank by edit-cleanliness.** Primary sort = number of *other* differences
   ascending (so pairs differing **only** by the target float to the top), capped
   by `--max-extra-diff`; secondary = CLS cosine descending. `--rest-jaccard-min`
   optionally enforces a "same scene" tag-overlap floor.

## Output

The single training-shaped artifact is the **materialized pair tree** under
`--export-dir` (default `…/easycontrol/<name>/staging/`):

```
…/<name>/staging/<artist>/
  <id_a>-<id_b>_tags.png      # holds the attribute (gap_holder)  → EasyControl cond
  <id_a>-<id_b>_tags.txt      #   caption sidecar
  <id_a>-<id_b>_no_tags.png   # clean counterpart                 → denoising target
  <id_a>-<id_b>_no_tags.txt
  <id_a>-<id_b>_mask.png      # Stage-B diff region (only with --emit-mask)
```

Images are **symlinked** from the source tree by default (`--copy` to copy).
Alongside it, `--config-out` (re)writes the EasyControl dataset blueprint into
the config's tail: the clean `_no_tags` member is the denoising **target**
(`cache_dir`); its `_tags` twin rides in as the **condition** (`cond_cache_dir`),
symlinked under the `_no_tags` stem by the preprocess step. The adapter thus
learns: *given a text-bearing panel as the reference, regenerate the clean one.*
The blueprint write preserves everything above its sentinel, so your edited
`[staging]` / `[preprocess]` tables survive re-runs.

## Key knobs

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/easycontrol/near_twins.toml` | `[staging]` table of run knobs (CLI overrides; `''` disables) |
| `--name` | config `name` / `near_twins` | output slug; reroutes the whole export/cache tree |
| `--tag` / `--tag-any` / `--region` / `--signal` | — | discriminator (exactly one) |
| `--image-dirs` | `$CAPTION_CORPUS_DIR/{retrieved,selected}` | source trees (`{VAR}`/`$VAR`/`~` expansion) |
| `--sim-min` | `0.85` | Stage-A CLS-cosine prefilter threshold |
| `--grid` | `7` | Stage-B pooled grid edge (G×G cells) |
| `--cell-match-min` | `0.9` | per-cell cosine for an inlier match |
| `--ratio` | `0.8` | ratio-test distinctiveness (lower = stricter) |
| `--match-frac-min` | `0.5` | inlier fraction to call a near-twin |
| `--geom-check` | off | RANSAC translation consistency (reject pose twins) |
| `--max-extra-diff` | `6` | cap on non-target differing tags (`-1` = off) |
| `--add-identity-pairs` | `0.0` | fraction of the final tree to fill with identity (cond==target) pairs — clean singles (no target tag) staged so the adapter learns the no-op (stops globally washing out clean inputs). Sized as `round(n_pairs·f/(1−f))` |
| `--identity-saturation-min` | `0.0` | only draw identity images with mean HSV saturation ≥ this (`0`=off); biases them vivid to counter the mined-twin desaturation bias |
| `--export-dir` | `…/easycontrol/near_twins/staging` | pair tree (`''` disables) |
| `--copy` / `--emit-mask` | off | copy instead of symlink / also write diff-region mask |

Identity pairs are written as `id_{stem}_{tags,no_tags}` from one source image, so
the preprocess cond-pairing symlinks an identical cond latent (cond==target). They
need ≥1 mined pair to size against; `--identity-seed` (default `0`) fixes the
random draw for idempotent re-runs.

Run `python -m easycontrol_adapters.tools.near_twins --help` for the full list
(`--signal-delta`, `--region-*`, `--id-window`, `--per-artist-topk`,
`--rest-jaccard-min`, `--batch-size`, `--num-workers`, `--device`).

## Layout

| Module | Role |
|---|---|
| `engine.py` | discovery → embed/cache → Stage-A/B match → discriminators → `run_artist` |
| `outputs.py` | `_tags`/`_no_tags` pair-tree export + EasyControl dataset blueprint |
| `__main__.py` | `[staging]` TOML layering + argparse CLI + orchestration |

## Notes

- **Captions use danbooru's spaced display form** on this corpus (`speech bubble`,
  not `speech_bubble`) — 1,212 of 15,905 caption files carry `speech bubble`
  (+`thought bubble` ×81). Tag mode works for bubbles today; matching is
  space-insensitive either way.
- **Output is not direct EasyControl training data in the naive sense** — real
  mined pairs differ in bubbles + expression + crop at once and aren't
  pixel-aligned, which is why this feeds the EasyControl extended-self-attn cond
  path (loose reference, not a pixel-aligned delta), eval sets, and unpaired
  editing — not a pixel-supervised removal loss.
