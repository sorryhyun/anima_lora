# Soup — uncond-init ΔW LoRA soup

A quality-and-robustness recipe for a LoRA over any glob-selectable slice of the
dataset, packaged as one command. It is not a new adapter family: the output
is an ordinary, bakeable plain LoRA (`output/ckpt/anima_soup_<slug>.safetensors`)
that infers and merges like any other. Soup buys two things over a single
fine-tune — a generalization (holdout-quality) win, and immunity to the
training-seed lottery — at the cost of a few extra cheap runs.

Shipped as `make soup PATH_PATTERN="<glob>"` (promoted from the `bench/uncond_soup`
line). Recipe validated in `bench/memorization/report.md` (the uncond-init
ladder, 2026-07-04/05). GUI: Experimental tab → Method → soup.

## Quick start

```bash
make soup PATH_PATTERN="sincos/*"                # attach-by-default (foreground)
make soup TARGET=sincos                          # shorthand ⇒ PATH_PATTERN="sincos/*" NAME=sincos
make soup PATH_PATTERN="art_a/*|art_b/*" NAME=ab --queue   # multi-slice, detached
make soup TARGET=sincos ARGS="--network_dim 32 --max_train_epochs 8"
```

`PATH_PATTERN` is a fnmatch glob (`|` separates alternatives) matched against
each image's path relative to its subset `image_dir` — it selects the Phase 2
fine-tune images. It is not an artist directory; it can span any slice the
glob expresses. `TARGET=<dir>` is a convenience shorthand for the common
per-directory case: `TARGET=x` ⇒ `PATH_PATTERN="x/*" NAME=x`. The output slug
derives from the pattern (a plain `<dir>/*` → `<dir>`; anything richer → a
sanitized prefix + short hash) unless you pass `NAME`.

`ARGS` is forwarded verbatim to the Phase 2 fine-tunes. It runs as one daemon
command job; the pipeline drives its own `train.py` phases as direct subprocesses
(never nested daemon jobs — that would deadlock the serial queue).

Env knobs (all optional): `NAME` (output slug), `POOL_PATH_PATTERN` (Phase-1
uncond pool glob, default `*` = whole dataset — the fine-tune set is always
unioned in), `UNCOND_RATIO` (0.5), `UNCOND_EPOCHS` (2), `NUM_SOUP` (3 — number
of seeded fine-tunes to soup; seeds are `1001..1000+N`), `LR_POOL` /
`LR_INTERVAL` (per-ingredient LR diversity, see below), `RANK` (default =
method `network_dim`), `PRESET`.

## Ingredient diversity — seed by default, LR opt-in

By default the ingredients differ only in `--seed`; every one trains at the
`learning_rate` in the ingredient config. That is a deliberate narrowing of the
model-soup recipe (Wortsman et al.), where ingredients come from a random
hyperparameter search — LR, weight decay, augmentation, seed.

Learning-rate diversity is available opt-in, on either of two mutually exclusive
knobs (`[soup]` table defaults, `--lr_pool` / `--lr_interval` flags, or
`LR_POOL=` / `LR_INTERVAL=` env):

```bash
make soup TARGET=sincos LR_POOL="1e-5,2e-5,4e-5"   # explicit list, cycled to NUM_SOUP
make soup TARGET=sincos LR_INTERVAL="1e-5:4e-5"    # geometric spread over NUM_SOUP points
```

Phase 1 is unaffected — the uncond init is a shared, reusable artifact and stays
on the config LR. Passing `--learning_rate` in `ARGS` alongside either knob is
refused rather than silently resolved.

Read the caveat before turning this on. In the paper the diverse pool is
protected by greedy soup: an ingredient is accepted only if it improves a
held-out metric, which is what stops a bad-LR draw from dragging the average.
This pipeline is a uniform ΔW average with no such gate (greedy selection was
closed as unverifiable — we have no trustworthy held-out quality metric at these
sample sizes; see `bench/memorization/report.md` and the CMMD fragility guards).
So a too-hot ingredient is averaged in, not dropped. There is also a prior
against a large win: the ingredient ΔWs already share ~98% of one uncond
component (cos +0.96..0.98 pairwise), with diversity living in the residuals — a
wider LR mostly rescales that shared direction rather than decorrelating the
residuals. Keep the spread narrow, and treat any gain as unproven until it is
eyeballed on renders.

## What it is — three phases

`scripts/soup/pipeline.py` orchestrates:

1. Uncond inter-train (Self-Soupervision, arXiv:2602.02890). A short
   `caption_dropout_rate 1.0` run on a diluted pool selected by
   `--pool_path_pattern` (default `*` = the whole dataset) at
   `sample_ratio 0.5`. The fine-tune selection is always unioned into the pool
   glob so the target is present but a minority share. The checkpoint name is
   deterministic in (pool, ratio, epochs) (`uncond_name()` hashes the pool
   glob), so it is trained once and reused across every soup drawing the
   same pool — with the default `*` pool, one shared uncond init serves them all
   — `anima_uncond_<hash>_r<ratio>_e<epochs>.safetensors`.
2. Seeded fine-tunes. `--num_soup` (default 3) normal captioned runs on the
   `--path_pattern` images, each `--network_weights`-initialized from the uncond
   checkpoint, one per derived seed (`1001..1000+N`) →
   `anima_soup_<slug>_s<seed>.safetensors`. Seed is the only axis that varies
   unless `--lr_pool` / `--lr_interval` is set (see above).
3. ΔW soup, SVD-truncated to the ingredient rank (`scripts/soup/build.py`)
   → `anima_soup_<slug>.safetensors`. The first ingredient's
   `.snapshot.toml` is copied next to the soup so the memorization probes can
   replay membership from it.

## Why ΔW-level, not parameterwise

Averaging `A`'s and `B`'s independently is wrong twice over:
`avg(B) @ avg(A) ≠ avg(B @ A)`, and with `down_init="weight_svd"` the
randomized SVD's per-vector sign ambiguity means row-wise `A` averaging can
actively cancel rows. So `build.py` soups at the ΔW level, which is
invariant to any per-ingredient `(A, B)` reparameterization. A weighted average
of rank-r deltas is exactly representable as one rank-`sum(r_i)` LoRA by block
concatenation:

```
dW_soup = Σ_i w_i · scale_i · up_i @ down_i'
        = [w_1·scale_1·up_1 | … ] @ [down_1' ; … ]
```

where `down_i' = down_i · inv_scale_i` folds the persisted channel-scaling
buffer back in (mirrors `LoRAModule.get_weight`), and the soup's alpha is set to
its rank (scale = 1). `truncated_soup` then keeps the top-r singular directions
(Eckart–Young best rank-r Frobenius approximation); the full ΔW is never
materialized — QR on the concatenated factors reduces the SVD to an `(R × R)`
core where `R = Σ r_i`. Per-module retained energy is measured and printed,
not assumed.

Rank-r truncation is essentially free on shared-init ingredients: at rank
16, mean retained energy 99.87%, min 96.8% (worst = early-block
`mlp_layer2`). The ingredients share a `weight_svd`-pinned init subspace, so
their ΔWs overlap almost completely and the averaged delta is barely more than
rank 16 to begin with.

## Constraints

- Plain-LoRA checkpoints only. The ingredient runs use `--method soup`
  (`configs/soup/soup.toml`) — a dedicated, stable plain-LoRA stack (`weight_svd`
  down-init + REPA; T-LoRA is training-only, so its checkpoints are still plain).
  This is deliberately *not* `configs/methods/lora.toml`, whose comment-toggle-able
  ortho / Hydra blocks and drifting `path_pattern` / `output_name` would break
  the soup: Hydra / Chimera / stacked-experts / ortho key shapes are refused
  loudly by `build.py`. Override per run via `ARGS="--network_dim 32 …"`.
- The pool must dilute the target, not repeat it at full dose. The uncond
  phase exposes the target's own frames (the fine-tune set is unioned into the
  pool), so the pool must be broad enough that the target is a minority
  share — the default `*` (whole dataset) at `sample_ratio 0.5` makes it
  tiny. If you narrow `POOL_PATH_PATTERN`, keep it a strict superset of the
  fine-tune set spanning many other images. Act 2 of the report shows same-frame
  full-dose uncond is destructive; Act 3 shows the diluted pool is both
  necessary and sufficient to avoid it.
- Keep the uncond dose low. Exposure budget = `uncond_epochs × member-share`.
  2 epochs at ~51% share was neutral on memorization; 4 epochs at 100% share
  was destructive (just more epochs of the same frames).
- `sample_ratio 0.5` is not in tracked config. `configs/base.toml` at HEAD
  says 1.0. The pipeline pins `--sample_ratio` for the uncond phase from
  `--uncond_ratio`; for the fine-tunes pass it via `ARGS` if you need the
  half-split (e.g. to keep a clean MIA holdout).

## What the bench found

One artist (`sincos`, 334 images), paired designs throughout (see
`bench/memorization/report.md`). CMMD is PE-Core MMD² vs real holdout, 20 steps
/ CFG 1.0; real-vs-real noise floor 0.155.

- Generalization win (Act 4). The pool-init fine-tune reaches CMMD 0.46 to
  the artist's real *unseen* work vs plain LoRA's 1.00 — plain is worse than no
  adapter at all on unseen captions (it narrowed the render distribution onto
  member look-alikes). ≈3.5× the noise floor, fully paired.
- Seed-lottery insurance (Act 5). Same recipe / init / data, only the
  training seed differs: siblings landed at CMMD 0.46 / 1.02 / 0.49 — a
  catastrophic draw (s1002 = 1.02, as bad as plain) that nothing on the
  memorization axis flags (its AUC 0.88 is unremarkable). The rank-16 soup of
  all three lands at 0.469 — within the noise floor of the best
  ingredient and far below the ingredient mean (0.65), despite one third being
  the bad draw. Souping is cheap insurance; SVD keeps the artifact
  single-adapter-sized (soup16 ≈ soup48 on both axes).
- **NOT a memorization fix.** Uncond member exposure is a dose-dependent
  memorization adder; diluted to harmless it is exactly neutral (buys zero
  reduction). Soup AUCs sit at the *max* of the ingredients, not the mean —
  every ingredient's member confidence-gap is positive, so averaging preserves
  it. The memorization axis needs its own mitigation.

## Files

| Path | Role |
|---|---|
| `configs/soup/soup.toml` | ingredient method config (plain-LoRA stack) + `[soup]` pipeline defaults (pool / dose / num_soup / rank) |
| `scripts/soup/pipeline.py` | 3-phase orchestrator (`make soup` entry) |
| `scripts/soup/build.py` | ΔW rank-concat soup + rank-r SVD truncation (`python -m scripts.soup.build`) |
| `scripts/tasks/training.py::cmd_soup` | `make soup` / env-knob → pipeline argv |
| `gui/tabs/soup_tab.py` | GUI launcher (Experimental tab) |
| `bench/memorization/report.md` | the validating study (uncond-init ladder) |

## Standalone soup builder

`build.py` is usable on any set of plain-LoRA checkpoints, independent of the
pipeline:

```bash
# exact rank-concat soup (rank = Σ ingredient ranks)
python -m scripts.soup.build --ckpts a.safetensors b.safetensors --out soup.safetensors
# rank-16 SVD-truncated soup (same size as one ingredient; prints retained energy)
python -m scripts.soup.build --ckpts a.st b.st c.st --rank 16 --out soup16.safetensors
# lambda=0.25 interpolation of a pair (LMC probe point)
python -m scripts.soup.build --ckpts a.st b.st --weights 0.75 0.25 --out mid.safetensors
```
