# From memorization to quality: the uncond-init ladder (2026-07-04 → 05)

One arc on a single question: **when a style LoRA overfits, what does an
unconditional inter-training phase (Self-Soupervision, arXiv:2602.02890) buy —
and what does it cost?** Everything below is one artist (`sincos`, 334 images),
one recipe, and paired designs throughout, so every delta attributes to the one
knob it varies.

## Operating point

All adapters share the calibrated overfit recipe (`bench_sincos_half_*`):
`--method lora --preset default`, `path_pattern sincos/*`, `sample_ratio 0.5`,
4 epochs (668 steps), dim 16 / α 90, `weight_svd` init, REPA on,
`caption_dropout_rate 0.1` unless stated. Two invariants make everything
comparable:

- The `sample_ratio` half is keyed to `validation_seed` (42), **not** the
  training seed (`library/datasets/dreambooth.py`) — so every run trains on the
  **identical 167 members**, and the other 167 images are a holdout no adapter
  ever saw.
- Fine-tune pairs share training seeds (identical data order + FM noise), so
  init-vs-init contrasts are exactly paired.

## Tools (all in `bench/memorization/`)

| tool | axis | statistic |
|---|---|---|
| `loss_gap.py` | weight-side memorization | calibrated loss-MIA: member-vs-holdout AUC on Δconfidence (LoRA − base) |
| `eyeball.py` | human check | contact sheet: MEMBER / HOLDOUT / OTHER-artist captions × seeds, source image as reference column |
| `generalize.py` | generalization / quality | paired holdout-caption renders → PE-Core CMMD vs real holdout & member pools (new today) |
| `probe.py` | generation-side memorization | PE self-vs-other contrast (not run today; next escalation if needed) |

## Act 1 — the baseline memorizes (and t1 barely helps)

| adapter | recipe delta | member-vs-holdout AUC | run |
|---|---|---|---|
| `bench_sincos_half_plain` | — | **0.82** (p=1e-4) | `20260704-1839-sincos_half_plain` |
| `bench_sincos_half_t1` | + T-LoRA t1 mask | 0.77 (p=1e-4) | `20260704-1825-sincos_half_t1` |

Sanity anchors from earlier runs: `plain_tenth` AUC 0.54 = clean;
signal lives at σ≤0.7. The plain sheet
(`20260704-1903-eyeball_sincos_half_plain`) shows the visible counterpart:
MEMBER-caption renders reproduce the source frame across seeds.

## Act 2 — uncond on the *same* frames backfires

Design: A = uncond run (`caption_dropout_rate 1.0`, seed 1000) on the same
members; B = captioned fine-tune initialized from A (`--network_weights`,
seed 1001). Same epochs, same everything else.

| adapter | uncond member exposure | AUC | run |
|---|---|---|---|
| A `…_uncond` | 4 ep × 100% share | 0.74 (p=1e-4) | `20260704-2000-uncond` |
| B `…_uncondft` | A-init + 4 captioned ep | **0.92** (p=1e-4) | `20260704-1946-uncondft` |

Two lessons:

1. **Captions are not the memorization vector.** A never saw a caption yet
   carries strong member signal — instance memorization binds through the
   unconditional branch (consistent with the finding that LoRA cross-attention
   learns labeled tags only; everything else goes to the uncond/style path).
2. **Same-frame uncond inter-train is just more epochs.** B stacked exposure
   (8 effective member epochs) and became the worst memorizer of the day; its
   member Δ went absolutely positive and the gap spread into high σ.

## Act 3 — the relaxed pool version: dose-response confirmed

Design: A′ = uncond on a 4-artist pool of *member halves*
(`sincos|suujiniku|hews|ama_mitsuki`, sincos ≈ 51% share), 2 epochs = 654 steps
(matched to A's 668 within 2%), seed 1000. B′ = the same fine-tune as B with
seed 1001 → **exactly data-order-paired with B; only the init differs.**
Probes at 24/24 (vs 48/48 above; treat ±0.05 as noise).

| adapter | uncond member exposure | AUC | run |
|---|---|---|---|
| A′ `…_uncondpool` | 2 ep × ~51% share | 0.56 (p=0.24) — **no member signal** | `20260704-2139-uncondpool` |
| B′ `…_uncondpoolft` | A′-init + 4 captioned ep | 0.83 ≈ plain's 0.82 | `20260704-2132-uncondpoolft` |

Full memorization ladder: `A(4ep, 100%) 0.74 → B 0.92` vs
`A′(2ep, 51%) 0.56 → B′ 0.83` vs `plain 0.82`. Uncond member exposure is a
**dose-dependent memorization adder**; diluted to ~harmless it is exactly
neutral — the uncond init buys **zero memorization reduction**. It is not a
memorization fix.

## Act 4 — but it is a big *quality* win

`generalize.py` (`20260704-2206-generalize_ladder`): the same 24 never-trained
sincos captions rendered under every model with identical seeds (20 steps,
CFG 1.0 — the trainer's CMMD convention), PE-Core-embedded, MMD²-scored
against 96 real holdout images (disjoint from the prompt items) and 96 real
member images. Real-vs-real noise floor: **0.155**.

| model | cmmd_holdout ↓ | cmmd_member |
|---|---|---|
| **B′ `…_uncondpoolft`** | **0.460** | 0.543 |
| base (no adapter) | 0.635 | 0.776 |
| B `…_uncondft` | 0.696 | 0.796 |
| plain | **1.003** | 1.143 |

Three results:

1. **The pool init generalizes far better than plain** — 0.54 CMMD closer to
   the artist's real unseen work, ≈3.5× the noise floor, fully paired.
   Matches the eyeball verdict on the B′ sheet
   (`20260704-2139-eyeball_uncondpoolft`): better member/holdout/other rows,
   less face-cloning on holdouts.
2. **Plain is worse than no adapter at all** on unseen captions. That is the
   sharpest statement of the overfit: the LoRA narrowed the render
   distribution onto member look-alikes and moved it *away* from the artist's
   real distribution — the style it added is outweighed by the diversity it
   destroyed.
3. **Memorization and generalization are partly independent axes.** B — the
   worst memorizer (0.92) — still generalizes better than plain (0.70 vs
   1.00). The uncond phase broadens the distribution even when it also burns
   member frames in. You need both probes; neither implies the other.

## The recipe that won

```
# Phase 1 — uncond inter-train on a small pool of adjacent artists' halves
python train.py --method lora --preset default \
  --sample_ratio 0.5 --caption_dropout_rate 1.0 --seed 1000 \
  --max_train_epochs 2 \
  --path_pattern 'sincos/*|suujiniku/*|hews/*|ama_mitsuki/*' \
  --output_name <name>_uncondpool

# Phase 2 — normal captioned fine-tunes from that init, one per seed
for SEED in 1001 1002 1003; do
  python train.py --method lora --preset default \
    --sample_ratio 0.5 --seed $SEED \
    --network_weights output/ckpt/<name>_uncondpool.safetensors \
    --output_name <name>_uncondpoolft_s$SEED
done

# Phase 3 — exact ΔW soup, then SVD-truncate back to the ingredient rank
python _archive/bench/uncond_soup/soup_svd.py --rank 16 \
  --ckpts output/ckpt/<name>_uncondpoolft_s100{1,2,3}.safetensors \
  --out output/ckpt/<name>_soup16.safetensors
cp output/ckpt/<name>_uncondpoolft_s1001.snapshot.toml \
   output/ckpt/<name>_soup16.snapshot.toml   # probes replay membership from this
```

Scorecard vs plain: **memorization neutral-to-slightly-worse (AUC 0.89 vs
0.82), generalization much better (CMMD 0.47 vs 1.00), and immune to the
training-seed lottery** (Act 5) — at the cost of three cheap extra runs
(~7 min each here). A single fine-tune (Phase 2 at one seed, no soup) keeps
most of the quality win but gambles on the seed.

Rules of thumb extracted:

- **Disjointness first.** Uncond data must not repeat the fine-tune frames.
  If you need a clean probe too, don't spend the complement half on training —
  use other artists (as here) or a 3-way split.
- **Keep the dose low.** Uncond epochs × member-share is the exposure budget;
  2 ep at ~51% share was neutral, 4 ep at 100% was destructive.
- **Shift is the point.** The paper's gains come from *shifted* unlabeled
  data; stylistically adjacent artists are the Anima analog, and truly
  uncaptioned pools are the real payoff case (uncond needs no caption work).

## What this means for PR #70 (uncond-soup bench)

- The headline UNCOND gate should not expect (or claim) memorization
  reduction; the live hypotheses are **holdout quality** and **soup variance
  reduction** across paired seeds.
- The pool phase must exclude the target's fine-tune images (this ladder shows
  it is both necessary — Act 2 — and sufficient — Act 3 — to avoid the
  failure mode).
- `generalize.py` is the natural quality gate alongside the bench's CMMD
  probe; `loss_gap.py` stays as the safety gate.
- The soup-variance hypothesis now has in-house evidence (Act 5): within the
  self family, souping absorbed a catastrophic seed draw and landed at
  best-ingredient level. What PR #70 still uniquely adds is the **base
  family** — the paired plain soup that separates souping-generic from
  uncond-init-specific gains.

## Act 5 — the soup (2026-07-05)

The soup phase of PR #70's design, run on the winning recipe: two more
fine-tunes from the *same* saved `uncondpool` init at seeds 1002/1003
(everything else identical to B′, so all three ingredients are exactly
data-order-paired and train on the identical 167 members), then the exact
ΔW-average soup by rank concatenation (`_archive/bench/uncond_soup/soup.py`, rank 48)
and a rank-16 SVD truncation of the same average (`soup_svd.py`, new —
Eckart–Young best rank-16 per module, computed via QR reduction so the full
ΔW is never materialized).

**Rank-16 truncation is essentially free.** Retained energy at rank 16:
mean **99.87%**, min 96.8% (worst modules are all early-block `mlp_layer2`).
The three ingredients share a weight_svd-pinned init subspace, so their ΔWs
overlap almost completely — the averaged delta is barely more than rank 16
to begin with.

**The training-seed lottery is violent** (`20260705-0002-generalize_soup3`,
same probe convention as Act 4; floor 0.155):

| model | cmmd_holdout ↓ | AUC (24/24) | run |
|---|---|---|---|
| B′ s1001 (Act 3) | 0.455 | 0.83 | — |
| s1002 | **1.021** | 0.88 | `20260704-2316-loss-gap` |
| s1003 | 0.486 | 0.82 | `20260704-2324-loss-gap` |
| soup48 (exact concat) | 0.530 | 0.88 | `20260704-2331-loss-gap` |
| **soup16 (SVD)** | **0.469** | 0.89 | `20260704-2338-loss-gap` |
| plain (Act 1) | 0.961 | 0.82 | — |
| base | 0.593 | — | — |

Three findings:

1. **s1002 is a catastrophic draw** — same recipe, same init, same data,
   only the training seed differs, and it lands at CMMD 1.02: as bad as
   plain, worse than no adapter. The gap to its siblings (0.46/0.49) is
   ~3.5× the noise floor. Nothing on the memorization axis flags it (its
   AUC 0.88 is unremarkable) — **a bad lottery draw is invisible without a
   quality probe.** This is the FID-lottery claim (arXiv:2606.20536)
   reproduced in-house at n=3.
2. **The soup rescues from the lottery.** Both soups land within the noise
   floor of the *best* ingredient (SOUP gate passes) and far below the
   ingredient mean (0.65) — despite one third of the soup being the bad
   draw. Expected value if you only train once: random single ≈ 0.65,
   soup ≈ 0.47–0.53. Souping is cheap insurance, and rank-16 SVD makes the
   artifact the same size as a single adapter (soup16 ≈ soup48 on both
   axes; the 0.06 CMMD difference is well under the floor).
3. **Souping does not reduce memorization.** Soup AUCs (0.88/0.89) sit at
   the *max* of the ingredients, not the mean — every ingredient's member
   confidence-gap is positive, so averaging preserves rather than cancels
   it. Same lesson as Act 3: the memorization axis needs its own mitigation;
   soup is a quality/variance tool.

Eyeball (`20260704-2338-eyeball_…uncondpoolsoup3` and
`20260705-0005-eyeball_…uncondpoolsoup3_r16`): both soups render clean — no
rank-concat or truncation artifacts, and the two sheets are visually
near-identical row for row (as 99.87% retained energy predicts).
Member/holdout rows track source composition exactly as all variants do
(captions pin composition; AUC stays the discriminator).

**Open control:** all of this is within the self family. The plain-soup
control (3 plain fine-tunes from stock base at the same seeds, souped) is
what separates "souping helps generically" from "the uncond init makes
better soup ingredients" — that's PR #70's base family, still unrun.

## Repro

Training runs live in `output/ckpt/bench_sincos_half_*.safetensors` (+
`.snapshot.toml` ground truth). Probes:

```
uv run python bench/memorization/loss_gap.py   --adapter output/ckpt/<a>.safetensors [--num_members 24 --num_holdout 24]
uv run python bench/memorization/eyeball.py    --adapter output/ckpt/<a>.safetensors
uv run python bench/memorization/generalize.py --adapters <a1> <a2> ... [--with_base]
```

Soups (`soup.py` is PR #70's exact ΔW rank-concat builder; `soup_svd.py` is
the rank-r SVD truncation on top — both in `_archive/bench/uncond_soup/`, currently
untracked copies of/next to the `worktree-uncond-soup-bench` branch):

```
uv run python _archive/bench/uncond_soup/soup.py     --ckpts <i1> <i2> <i3> --out soup48.safetensors
uv run python _archive/bench/uncond_soup/soup_svd.py --ckpts <i1> <i2> <i3> --rank 16 --out soup16.safetensors
cp <i1>.snapshot.toml soup{48,16}.snapshot.toml   # probes replay membership from the snapshot
```

**Repro trap:** `sample_ratio = 0.5` lives nowhere in tracked config —
`configs/base.toml` at HEAD says 1.0; the July-4 runs used an uncommitted
working-tree edit. Relaunching from the toml chain silently trains on the
full 334 (holdout included) and voids the MIA probe. Pin `--sample_ratio 0.5`
on the CLI; the `.snapshot.toml` files are the ground truth.
