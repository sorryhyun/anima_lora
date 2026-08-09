# σ-Demoted Training (`--sigma_lowres`) — resolution routing by noise level

A **training-time throughput feature**: on high-noise steps, train the DiT on a
lower-resolution latent of the *same* image instead of its native one. The
signal a denoiser can actually use at large σ is concentrated in low spatial
frequencies, so the tokens carrying fine detail are largely paying for noise —
demoting the grid there buys wall-clock at a footprint that stays inside the
run-to-run seed lottery. Nothing changes at inference: the output is an ordinary
LoRA checkpoint.

> **Background and derivation**: [arXiv:2608.04448](https://arxiv.org/abs/2608.04448).
> The per-route certification map, the RoPE/residual decomposition of the demote
> gap, and the trajectory-propagator argument behind the schedule knobs all come
> from there. The experiment records live in `project/sigma_lowres/` (E4 the
> harness/yardstick protocol, E14 the per-σ route map, E16 the routing and
> placement study this doc's shipped recipe is drawn from).

## Quick start

Two steps: emit the sibling latents, then flip the trainer flags.

```toml
# configs/preprocess.toml — one VAE pass emits both routes' sibling keys
sigma_demote = "1024:896,1024:768"
```

```bash
make preprocess          # or: make preprocess-vae
```

Then train with the shipped recipe (**combo** — the stacked router, each σ
routed to the deepest grid certified for it). `configs/base.toml` already
carries the whole recipe behind one boolean, so the usual way in is:

```toml
# configs/base.toml
sigma_lowres = true
```

or, equivalently, from the CLI (yarnsig rides along by default — it is part of
the recipe; pass `--sigma_lowres_yarnsig off` to drop it):

```bash
python train.py --method lora --preset default --sigma_lowres
```

Spelled out in full, that is:

```bash
python train.py --method lora --preset default \
  --sigma_lowres \
  --sigma_lowres_route2 1024:768 \
  --sigma_lowres_threshold2 0.65 \
  --sigma_lowres_threshold2_max 0.95 \
  --sigma_lowres_yarnsig
```

Per step, that reads: **768 if σ ∈ (0.65, 0.95); elif σ > 0.5 → 896 (+yarnsig);
else native.** Measured at E16 scale (480 steps, 2 corpora × 3 seeds):
**−18.3% wall, inside the seed lottery on both corpora** — the best measured
combination of the two.

Adding `--sigma_lowres_span late:0.75 --sigma_lowres_span2 late` gives
`combolate`, which trades ~4pp of that throughput for a much closer endpoint
weight-space footprint (ΔW 0.75 vs 0.37). It is **not** the default: see
[when to schedule](#when-to-schedule) — on per-step-certified routes the
schedule bought no render-level improvement and cost speed.

Validation always stays native, so val loss remains comparable across arms.

## Why it is safe (and where it isn't)

The unit of certification is the **per-step demote gap**: how much a
demoted-latent forward's prediction differs from the native one at a given σ,
against a re-encoding control. The measured map:

| route | certified region | note |
|---|---|---|
| 1024 → 896 | σ > 0.5 (half-line) | the original shipped gate; gap ≈ 0 below the noise floor above it |
| 1024 → 768 | σ ∈ ~(0.65, 0.95) | a **window**, not a half-line — below ~0.45 the gap is +0.25…+0.38 (badly off-map), and the σ=1 endpoint is elevated again (+0.130) |
| 1024 → 1280 etc. | σ* ∈ (0.625, 0.875) | route-dependent; not wired into a shipped recipe |

Two consequences the flags exist to express:

1. A half-line gate is the wrong shape for 768 — hence
   `--sigma_lowres_threshold_max`, turning `(threshold, ∞)` into a window.
2. The two routes' certified regions **do not nest**, so the best schedule is
   not "pick one route" but "route each step to the deepest grid certified for
   its σ" — hence the stacked router (`--sigma_lowres_route2`).

Gating is per-batch and strict: a step demotes only if **every** σ in the batch
is inside the region. The σ draw itself is never skipped, so the σ marginal is
identical to a native run.

### Placement matters for uncertified bias

E16 measured the training trajectory's response to demotion bias placed at
different points in training. With the route run *uncertified* (σ-gate off, so
every step demotes), endpoint ΔW cosine against a native twin:

| placement of the same demoted mass | cos(ΔW, native) |
|---|---|
| late half | **0.906** |
| every other step | 0.281 |
| early half | 0.193 |

The regime is **amplification**: bias placed while the from-zero LoRA is still
selecting its subspace redirects the whole trajectory (early↔late cos 0.176 —
the two placements build nearly unrelated adapters), while the same mass placed
late costs 0.094 of cosine. That is what `--sigma_lowres_span` is for: protect
the first epoch(s), demote late.

The honest scope: **on per-step-certified routes, scheduling is not required** —
`sigma896late ≈ sigma896` at render level. The amplification law governs
*off-map* bias. Spans buy weight-space closeness to a native run; they do not by
themselves buy render quality, and they cost throughput.

## Flags

| flag | default | meaning |
|---|---|---|
| `--sigma_lowres` | off | master switch. Demotes on train steps only; needs the sibling latents cached |
| `--sigma_lowres_route` | `1024:896` | primary route as `NATIVE:DEMOTE` |
| `--sigma_lowres_threshold` | `0.5` | primary rule's lower σ bound (strict `>`) |
| `--sigma_lowres_threshold_max` | none | optional upper σ bound → window semantics |
| `--sigma_lowres_span` | none | `early\|late\|spread[:FRAC]` step-span gate on the primary rule |
| `--sigma_lowres_route2` | none | secondary route, **priority over the primary** |
| `--sigma_lowres_threshold2` | none | secondary rule's lower σ bound — **required** with `route2` |
| `--sigma_lowres_threshold2_max` | none | secondary rule's upper σ bound — **required** with `route2` |
| `--sigma_lowres_span2` | none | secondary rule's step-span gate |
| `--sigma_lowres_yarnsig[=A,B,C,G]` | **on with `--sigma_lowres`**, at `1,4,0.35,2`; `off` to disable | σ-gated YaRN-banded RoPE on **primary-rule** demoted steps |

Router precedence, per step: **rule 2 if its gate *and* span pass → rule 1 if
its gate *and* span pass → native.** Because rule 2 wins wherever it fires, it
must be given an explicit window — `--sigma_lowres_route2` without both of its
σ bounds is a **setup-time error**, not a default; an unbounded rule 2 would
shadow the primary rule everywhere and silently disable yarnsig (primary-only).
Routes, windows and span specs are all validated before the model loads.

A rule whose sibling latent is missing from the npz falls through to the next
one (partial-emit degrades rather than crashing) and warns once per rule, from
both sides: the dataset warns per route on the missing `demoted_{H}x{W}` key,
and the trainer warns when a rule passed its gate with no sibling on the batch.
Treat either as "the deep route is dead" — otherwise a half-emitted cache costs
you it with no symptom but wall-clock. The trainer also logs the first demoted
step (`latent grid … → … at σ=…`) and a running `demoted N/T eligible steps`
every 500 steps; check both against the mass you expect, and the
`token_step_hist` below for the exact split.

`--sigma_lowres_span` modes:

- `early[:f]` — demote only in the first `f·T` train forwards (default `f=0.5`);
- `late[:f]` — only the last `f·T`;
- `spread[:p]` — per-step coin at probability `p`, seeded from `--seed` and the
  step index alone.

`T = max_train_steps × gradient_accumulation_steps`. `early:f` and `late:1-f`
partition the run exactly. The spread coin touches neither the global nor the
paired torch RNG streams, so common-random-number pairing across arms holds.

`--sigma_lowres_yarnsig` deliberately applies to the primary rule only: the deep
route's window was measured on plain demotion, so the trainer does not
extrapolate a rope treatment onto it.

## Measured arms

E16.1 protocol: 2 corpora (hews 8 ep / channel 32 ep) × 3 seeds × 480 steps,
`--deterministic --paired_step_rng`. "Render cos" is the paired cosine against a
native twin; the **yardstick** is the cross-seed native lottery — two native runs
at different seeds sit this far apart, so an arm at-or-inside it is not
distinguishable from having changed the seed. ΔW cos is the endpoint
weight-space read (deterministic twin control: 1.000).

| arm | wall Δ (hews/channel) | render cos (hews/channel) | ΔW cos | vs yardstick 0.9547/0.9541 |
|---|---|---|---|---|
| **`combo`** (shipped recipe) | **−18.2% / −18.4%** | 0.9576 / 0.9580 | 0.365 / 0.434 | inside both |
| `sigma896` (previous σ-gate) | −14.9% / −14.6% | 0.9538 / 0.9641 | 0.365 / 0.432 | boundary hews, inside channel |
| `combolate` (combo + late spans) | −14.6% / −13.1% | 0.9461 / 0.9664 | 0.753 / 0.771 | **below hews**, inside channel |
| `sigma896late` | −10.1% / −10.6% | 0.9535 / 0.9636 | 0.753 / 0.770 | ≈ `sigma896` |
| `win768late` (768 window + late, no stack) | −6.1% / −6.3% | **0.9678 / 0.9728** | 0.959 / 0.962 | comfortably inside both |
| `896only` (gate off) | −31.7% / −30.2% | 0.9494 / 0.9500 | 0.183 / 0.236 | **below both** |

Reads worth carrying:

- **The gating, not the resolution, is what keeps the footprint small.**
  Gate-free 896 buys −31% and lands outside the lottery on both corpora; the
  same route gated at σ > 0.5 lands inside at −15%.
- **In-window 768 is nearly free.** `combo` and `sigma896` sit at the same ΔW
  radius even though 126 of `combo`'s steps go deeper — consistent with the
  per-step certification of the 768 window.
- **ΔW closeness ≠ render closeness.** `combo` at ΔW 0.37–0.43 renders inside
  the lottery; `896only` at 0.18–0.24 renders below it. Do not read arm quality
  off the ΔW column.
- CMMD was recorded throughout and carries **no** quality verdict at this N.

### When to schedule

The span flags exist for the amplification result above, but E16's standing
verdict is that **scheduling is unnecessary on per-step-certified routes** —
and the arm table is what settles it:

- `sigma896late ≈ sigma896` at render level, for 4.5pp of throughput.
- On **hews** (the lenient corpus) `combolate` rendered at 0.9461 against a
  0.9547 yardstick — *below* the lottery, on all three seeds — where the
  unscheduled `combo` was inside on both corpora at a better −18.3%. On
  **channel** `combolate` was comfortably inside (0.9664).
- So the late schedule bought ΔW closeness and cost throughput **without**
  improving the render footprint. Since ΔW closeness ≠ render closeness (see
  the reads above), that is not a trade worth taking by default.

Reach for spans when the bias is *off-map* — an uncertified route, an
uncertified σ region, a probe. That is the regime the amplification law
governs. Concretely:

- `combolate` (`--sigma_lowres_span late:0.75 --sigma_lowres_span2 late`) —
  −13.9%, ΔW 0.75. Pick it if you specifically want the endpoint weights close
  to a native run and can spend the throughput.
- `win768late` (drop the primary rule, keep `--sigma_lowres_span2 late`) —
  −6%, the max-margin arm (0.9678 / 0.9728). Maximum conservatism.

## Cache mechanics

A sibling latent is **a key inside the image's existing native `.npz`**, not a
separate file: `demoted_{H}x{W}`, where `(W, H)` is the free-fit bucket the
image lands on in the demoted tier. Two routes never collide because their
buckets differ. Consequences:

- `make preprocess` / `make preprocess-vae` chain the emit once per route listed
  in `sigma_demote`, so the sibling cache tracks images as they are added.
- `make preprocess-demote` emits **every** route in `sigma_demote` (same source
  as the chain above), so the stacked router's two siblings both land from this
  target too. `ARGS="--sigma_demote 1024:768"` overrides with an explicit route
  (a comma list there is expanded into one pass each). Idempotent; requires
  `preprocess-vae` to have run first.
- The key lives outside the latents namespace, so the normal cached-latent
  loader never sees it — checkpoints and non-`--sigma_lowres` runs are
  untouched.
- Only images whose native bucket is actually in the route's source tier get a
  sibling; everything else trains native regardless of σ.

Both demote bands are folded into the `compile_blocks()` token-family budget, so
the demoted forwards land inside the compiled dynamic-seq range rather than
triggering a recompile cascade.

## Accounting

With `--sigma_lowres` on, the run-end progress event carries `token_step_hist` —
`{patch_tokens: examples}` counted at the grid each step *actually ran on*, after
the demote swap. That is the per-arm FLOPs read; use it to confirm the realized
demote mass matches the gate you configured (e.g. `combolate` at 480 steps:
59 deep + 121 shallow + 300 native).

Note when eyeballing the histogram that the 896 band spans 2925–3096 tokens —
count sub-4032, not sub-3000.

## Interactions and limits

- **Adapters must opt in.** `--sigma_lowres` **raises** if any attached adapter
  is not `sigma_demote_safe` — fixed-grid cond / extra-forward streams
  (EasyControl and friends) need their own operating-point probe first.
  Grid-agnostic adapters (repa) are allowed.
- **A timestep sampler without a flat-σ draw disables demotion.** The run warns
  once and trains native throughout, rather than failing.
- **Validation is never demoted**, by construction.
- **Route parametrization is for probes.** `--sigma_lowres_route` accepts
  anything, but only the routes in the map above are certified; an uncertified
  route is an experiment, not a recipe.
- The measurements above are at E16 scale (480 steps, LoRA, 2 corpora). They are
  a footprint argument, not a promise about a different recipe or a much longer
  run.
