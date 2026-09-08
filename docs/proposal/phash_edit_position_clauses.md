# phash_edit position-clause instructions — can a spatially-addressed delta kill the edit-or-copy coin flip?

Status: PROPOSED. Phase 0 (diff localization, CPU-only) RUN 2026-08-21
(`bench/phash_edit/results/20260821-2225-diffloc/`) — verdict: viable, numbers
below. Phase 1 (re-mine + re-train + probe) not started.

## Problem

The phash_edit EasyControl adapter (`configs/easycontrol/phash_edit.toml`) is
random at inference: with near-identical prompts and settings, an edit lands
on one seed and copy-locks or fires an unrequested variant-axis edit
(clothing lift, watermark stamp, expression flip) on the next. The probe
evidence (`bench/phash_edit/report.md`, 2026-08-21):

- small deltas (Δ1–3) copy-lock at b0; the b_cond offset is a binary switch
  with no usable midpoint — the system is bistable, and tiny instructions
  sit on the separatrix, so the seed decides the basin;
- the only recipe that lands a small in-place edit is the masked-compose
  hybrid — i.e. the edit lands when something external supplies the
  "where" (a hand-painted mask);
- full captions are catastrophically off-distribution; the adapter's entire
  usable interface is the mined delta grammar.

Diagnosis: the flat delta bag says *what* but never *where* or *how much*.
An under-determined instruction leaves the copy/edit decision to step noise,
and the adapter fills the content vacuum from its training marginal (the
variant-axis prior of booru variant series).

## Proposal

Emit mined instructions as position clauses — the repo's existing caption
grammar (`anime_tools.captions.position_clauses`) — with the header derived
from the pair's own pixel diff at mining time:

```
On the top left, -english text.
speech bubble. On the bottom, -blush.      # multi-blob, two clauses
```

- Localization is free. The pairs are same-canvas near-twins, so the diff
  region is a subtraction — no SAM3, no tagger attribution (unlike
  `caption-position`, which needs the whole crop→tag pipeline). Quantize the
  dominant diff blob's centroid into the 3×3 header vocabulary.
- Text-space mask. This is the masked-compose recipe's mask moved into
  the prompt: coarse, but user-writable without mask painting, and it raises
  the information content of a Δ1 instruction without stretching it toward
  the (broken) full-caption regime.
- Purity gate for free. A pair whose tags claim an edit but whose pixels
  don't move (or vice versa) is mechanically detectable in the same pass.
- Composes with `compose_caption` — clauses are atomic under
  `caption_shuffle_variants`, so shuffle keeps working; `both_directions` is
  symmetric (same header both ways); the `-` removal prefix rides inside the
  clause unchanged.

The adapter must learn the header→region binding — do not assume the frozen
base DiT honors clauses on an instruction-format prompt. The vocabulary is ~9
headers against hundreds of clause records, learnable at dim 32.

## Phase 0 — is the pair distribution clause-addressable? (RUN)

`bench/phash_edit/run_diff_localize.py`: 64×64 grid diff over all 1,856
unique edit pairs (3,712 directed), adaptive noise floor, 8-connected blobs,
dominant-blob header. Results (`20260821-2225-diffloc`):

| class | frac | meaning |
|---|---|---|
| single (top blob ≥ 75% of diff mass) | 30.0% | one clause suffices |
| multi | 69.6% | dominant blob + satellites (see below) |
| near_zero | 0.4% | tags claim an edit, pixels don't |
| global | 0.1% | whole-canvas redraw |

- Header histogram is healthy, not degenerate: bottom 173 / center 150 /
  right 46 / top 43 / bottom-left 36 / bottom-right 36 / left 36 / top-left
  22 / top-right 14. "center" is only 27% of singles.
- The multi class is diluted by decode-noise satellites, not genuinely
  scattered edits: median 13 blobs but median top_share 0.44, and 481 multi
  pairs (26% of all) have top_share ≥ 0.5 — eyeballing the example panels
  shows one real edit region plus JPEG/screentone specks. A mass-floor merge
  before classification should push the clause-addressable fraction to
  ~50–60%. Conservative floor: 30%.
- Label noise is negligible: 6 silent-delta suspects, 2 unlabeled
  globals. This weakens the "dirty mined labels taught the variant-axis
  prior" hypothesis and strengthens under-determination as the live
  mechanism — the data is clean; the instruction channel is just too poor.

Verdict: **worth Phase 1.** Every mining-side precondition holds.

## Phase 1 — re-mine, re-train, probe

Miner changes (staging step only — pool latents and VAE caches are untouched;
only delta captions change, so the re-mine costs a TE re-encode, not a VAE
pass):

1. Compute the Phase-0 diff map per accepted pair (same code, promoted into
   the miner). Merge/drop blobs below a mass floor before classifying.
2. Single-blob pair → all delta tags into one clause with the derived header.
   Multi-blob → flat delta unchanged (attribution of which tag belongs to
   which blob needs crop tagging — out of scope; flat is the shipped
   behavior, so this arm only ever adds information).
3. Compose with `compose_caption`; never string-concat (clause atomicity
   under shuffle variants depends on it).
4. Drop the ~8 near_zero/global suspects.

Training: same budget as the shipped checkpoint (dim 32 / α128 / 4 epochs /
b_cond −4). One change per arm:

- Arm A: shipped checkpoint (exists — no rerun).
- Arm B: clause-annotated re-mine, config otherwise identical.
- Arm C (optional, only if B moves): B + `colorize_frac=0` +
  `identity_frac=0` + `cond_diff_loss=true` — with the colorize arm gone the
  diff map is meaningful again, and it is the same subtraction that derived
  the clause header, so text supervision and loss weighting agree about where
  the edit is.

Probe (`run_edit_probe.py`, extended):

- Stratify by tag-delta size × clause-annotated-or-not. Success gate: on
  the Δ1–3 stratum, clause-annotated pairs show ec_b0 mse_vs_cond separated
  from noop_b0 (copy-lock broken) while staying far below noec (identity
  kept). Arm A is the baseline for the same pairs.
- Seed-variance probe (the actual user-facing complaint): fixed
  cond+instruction, 4 seeds per case, variance of per-pixel diff-vs-cond
  inside vs outside the instructed region. Arm B should collapse the
  outside-region variance relative to Arm A.
- Held-out pairs (train-set success is an upper bound — twin_edit lesson).

## Risks / honest caveats

- Thin clause supervision: 556 single pairs (×2 directions) at the
  conservative criterion. The mass-floor merge matters — at ~1,100+ clause
  records this is comfortable; at 556 it may be marginal.
- Copy-lock may survive: clauses attack the localization vacuum, not the
  b_cond gating. If Δ1-with-clause still copies, the bistability is gated
  elsewhere and Arm C / strength-sweep is the next lever.
- Coarse addressing only — 3×3 headers resolve "which region", not
  surgical placement; masked compose remains the precision tool.
- Inference stack mismatch is a separate bug: the Spectrum node prepends
  `high quality` to every positive, which is off-distribution for a
  delta-grammar adapter regardless of clauses. Either bake the prefix into
  training captions or add a node toggle; do not let it confound the Arm A/B
  comparison (probe uses `inference.py`, which doesn't inject it).

## Pointers

- Phase 0 script: `bench/phash_edit/run_diff_localize.py`; run
  `bench/phash_edit/results/20260821-2225-diffloc/` (per_pair.csv + example
  panels per class).
- Probe + copy-lock evidence: `bench/phash_edit/report.md`.
- Grammar: `anime_tools.captions.position_clauses`; conventions in the
  `captions` skill.
- Miner config: `configs/easycontrol/phash_edit.toml` (the `[staging]`
  tables survive re-mines verbatim; the blueprint below the sentinel is
  regenerated).
