---
name: bucketing
description: Free-fit native-shape bucketing — the token bands per edge tier, tier choice at preprocess time, the compile_dynamic_seq coupling and per-tier graph budget, and why training never needs --target_res. Load before touching resize/bucket code, changing a tier or band, debugging graph recompiles or token counts, or reasoning about which images landed where.
---

# Free-fit bucketing

Free-fit is the **sole** resize mode. The discrete constant-token bucket pool
(`CONSTANT_TOKEN_BUCKETS` and the per-tier tables) was removed 2026-06-19; the migration
kept only each tier's numeric token band in `EDGE_TOKEN_BANDS`. There is no `freefit`
flag any more — it's implicit. The legacy pad-to-static path went 2026-05-24
(`static_token_count` / `static_pad` etc.).

Free-fit keeps each image's **native aspect ratio** and lands its patch-grid token count
*anywhere* inside its tier's band, driving crop loss to ~zero (sub-patch <16px residual).

**Ownership**: `freefit_bucket` / `freefit_band_for_edge` are **owned by
`anime_tools.buckets` since 2026-09-03**, re-exported by `library/datasets/buckets.py`
the way `library/models/pe.py` re-exports the PE tower. The resize pass itself is the
package's `anime_tools.stages.resize`, which `make preprocess-resize` runs as a
`ResizeRequest` (see the `anime-tools` skill). Design:
`_archive/proposals/free_aspect_token_band_resize.md`.

## Tiers

`EDGE_TOKEN_BANDS` defines per-tier bands for edges **512 768 896 1024 1280 1536**:

| Edge | Token families |
|---|---|
| 512 | 1008, 1024 |
| 768 | 2160 |
| 896 | 3000, 3024 |
| 1024 | 4032, 4200 |
| 1280 | 6300 |
| 1536 | 8640 |

Preprocess `--target_res <subset>` selects which tiers are active; each image goes to the
tier that **resizes it the least** — `choose_edge` is an area-based
`|log(nominal_tokens/native_tokens)|` minimum, scale-symmetric, so a 0.95MP image stays at
1024 rather than downscaling to 768.

The 1024 tier's band is **frozen at (4032, 4200)** (`FREEFIT_FROZEN_EDGES`) because the
frozen top-5 aspect set (`DCW_ASPECT_BUCKETS`, consumed by CNS calibration + mod-distill)
is drawn from it. All tiers stay within the rope cap (≤256 patches/axis).

## Compile coupling — the load-bearing part

Free-fit populates many distinct `(W,H)` inside a tier's band, which would explode the
static N-graph cascade, so it **requires `compile_dynamic_seq`** — auto-enabled by
`train.py` whenever `torch_compile` is on, and unconditionally forced in the bespoke
distill loops via `ensure_dynamic_seq_for_freefit`. `dynamic_seq` marks only the seq axis
dynamic and bounds it to the tier's `seq_range`, collapsing the whole band to **one graph
per tier**.

Each forward runs at its real token count; `compile_blocks()` sets `_native_flatten`,
which flattens each patch grid to a fake-5D `(B, 1, seq_len, 1, D)` shape so the block
graph keys on **token count alone** — bit-exact to the eager 5D path.

## Caches are the source of truth

`make_buckets()` uses the actual on-disk cached `(W,H)` as the bucket set, so nothing
AR-snaps at load.

**Training is self-describing and does NOT need `--target_res`** (a preprocess-only
knob): every cached latent exact-matches its true `(W,H)`, and the
`compile_blocks(n_token_families=…)` dynamo budget is derived from the buckets the
`path_pattern`-filtered images **actually populate** (`train.py::_derive_token_budget`)
**plus the sample-prompt resolutions when sampling is enabled**. A sample prompt outside
the training range added to the file *mid-run* is skipped with a warning at sample time.

**Snap-era caches still train fine** — a snap pool is just a free-fit pool that landed
only on the old discrete counts. Re-preprocess only to gain the reduced-crop benefit.

After a `target_res` tier change run `make preprocess-reconcile` (dry-run;
`ARGS="--delete"` to act) to drop the orphaned latent npz / stale resized PNG / PE sidecar
/ mask for every image whose bucket moved. TE caches are text-only and never touched.
