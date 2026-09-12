# LoRA distill — dataset-free `teacher stack → one plain LoRA` on sampler-trajectory probes

Status: PROPOSAL, no GPU work done. Reduced from an external "teacher-free LoRA
generation as annealed Langevin sampling" ideation note (2026-09-11) to the one
piece that survives this repo's measured constraints. Reuse inventory verified
by code-reading (cited inline).

## TL;DR

One operation, `lora-distill`: given a **teacher** configuration
`T = base + adapter stack + (CFG, steps)` and a **student** configuration
`S = base + optional frozen stack`, fit a single plain LoRA `θ` so that
`S + θ` reproduces `T`'s velocity on probes `(x_t, t, c)` drawn from the base
sampler's own trajectories:

```
L(θ) = E_{c, t, x_t} ‖ v_{S+θ}(x_t, t, c) − v_T(x_t, t, c) ‖²
```

No images, no captions, no cache — probes come from generic prompts rolled
out by the sampler. The output is an ordinary LoRA (`--lora_weight`, `merge`,
ComfyUI, soup ingredient — everything a plain LoRA already does).

Three consumers, in order of how much they'd be used:

| consumer | teacher `T` | student `S` | replaces |
|---|---|---|---|
| **consolidate** | `base + Σ_i LoRA_i` under artist tag *i* (N artists) | `base` | `merge_loras.py` (exact concat, rank Σr_i) — when the rank budget is r, not Σr_i |
| **ensemble-soup** | mean over N seed-ingredients `base + LoRA_seed_k` (output-space average) | `base` | `scripts/soup/build.py` (weight-space ΔW mean + SVD) — no shared-init / LMC requirement |
| **turbo port** | `base + artist LoRA` at 28 steps, CFG 4 | `base + turbo student` (frozen), 4 steps, CFG 1 | linear stacking (`docs/methods/turbo.md` §Limitations & composition) |

Optional **guidance bake**: `v_T ← (1+w)·v_{T,c} − w·v_{T,∅}` so the LoRA
carries a CFG-sharpened teacher. This is exactly guidance distillation; it is
the only `w > 0` reading that has a density-ratio behind it.

## What this is NOT (so it isn't re-proposed as those)

- **Not a seed-lottery fix for ordinary dataset training.** Ordinary training
  has no teacher; there is nothing to swap the `ε` target for. The "remove
  label variance" pitch of the source note only exists once a teacher exists,
  and then the lottery was never the problem.
- **Not per-probe selection / gradient-agreement scoring.** See constraints
  below — per-probe gradient cosines are unmeasurable noise on this model.
- **Not a weight-space adapter dataset / hypernetwork bootstrap.** ΔW
  closeness ≠ render closeness (both directions, measured three times); a
  dataset of `θ`s would be dominated by co-adaptation + gauge variation.
- **Not an EasyControl → i2i port.** A plain LoRA has no image input; the
  only EC-derived product is a *per-reference* compile (one fixed reference
  → one LoRA), which is a different proposal with a different gate (needs an
  identity `ref==target` EC adapter, which does not exist).
- **Not Langevin / diffusion in weight space.** The loop is SGD on `L(θ)`.
  Injected noise is flat along the `A→AP, B→P⁻¹B` gauge orbit and just
  random-walks there.

## Method

**Probes.** Roll the base sampler on a fixed prompt pool (the existing
sample-prompt / E4 prompt-grid sets), record `x_t` at every knot
(`--traj_stats` already records per-step latents; extend or re-derive from
the sidecar). Fixed finite pool, seeded once — every probe is scored across
the run, not seen once. Forward-noised references are not used (off the
manifold the sampler visits).

**σ-weighting.** Population-level only: a σ-bin weight (Min-SNR-class or the
trainer's logit-normal marginal). Style-heavy consumers (consolidate) can tilt
toward low σ; the turbo port must keep the student's 4-knot grid inside the
probe support.

**Loop.** Two-view forward on one frozen DiT, exactly the turbo shape: teacher
view = teacher stack on, student LoRA off; student view = frozen student stack
+ trainable `θ`. `scripts/distill_turbo/setup.py::_forward(view, …)` +
`turbo.set_view` is the mechanism; the `_teacher_cfg_velocity` closure is the
CFG'd teacher. Plain `use_reentrant=False` checkpointing (the reentrant
unsloth variant drops grad on all-detached multi-forward objectives). Mirror
`--deterministic` for any paired read. Lives under `scripts/distill_lora/`
next to turbo, **not** in `train.py` (bespoke loops don't inherit train.py
infra; keep it explicit).

**Init.** For consolidate / ensemble-soup, warm-start `θ` from the exact
concat or the SVD soup (i.e. from the baseline it must beat), so the run
starts at the baseline's error and can only be read as "did distillation
find something the closed-form merge missed."

## Reuse inventory (live in-tree)

| piece | where | state |
|---|---|---|
| load → apply → compile ordering | `library/runtime/harness.py::build_anima` | shipped; use it, don't open-code |
| view-toggled LoRA stacks on one frozen DiT, CFG'd teacher closure | `scripts/distill_turbo/setup.py` (`_forward`, `_teacher_cfg_velocity`), `steps.py::teacher_anchor` | shipped for DMD; the two-view forward is the reusable part |
| exact N→1 concat merge (baseline + warm start) | `scripts/toolkits/merge_loras.py` | shipped; `--normalize global` is the strength baseline |
| weight-space SVD soup (baseline + warm start) | `scripts/soup/build.py` | shipped |
| ΔW → rank-r plain LoRA (defused standard layout) | `scripts/toolkits/extract_delta_lora.py` | shipped; the writer for a warm-start `θ` from any closed-form ΔW |
| per-step latent recorder (probe source) | `library/inference/traj_stats.py` (`--traj_stats`) | shipped; stores k=4 codes + `tok` today — needs the raw `x_t` sidecar for probes |
| typed generation for probe rollouts | `library/inference/request.py::GenerationRequest` | shipped |
| bit-exact paired runs | `train.py --deterministic` (`attention_dispatch.set_deterministic`) | shipped in train.py; mirror in the new loop |
| render-level yardstick | paired PE-cos vs. a reference render set (the turbo / soup benches' in-batch read) + full-res eyeball | shipped in bench land; CMMD at 20-step/CFG-1 n=24 is **not** the read |

## Constraints this must respect (measured on this model; sigma_lowres research record, unpublished)

1. **Per-probe gradient rankings have ≈ 0 split-half reliability** at any
   feasible draw count (heavy-tailed in σ; noise did not shrink K=16→32).
   Per-sample ‖Δg‖ ≈ 0.7–1.6·‖g‖ while the batch aggregate is 0.15–0.35.
   → Any weighting is σ-bin × population, never per probe.
2. **Per-step gradient equivalence does not imply endpoint equivalence.** A
   conditioning arm absorbed a probe substitution per step and still
   converged to a different ΔW. → Never certify from step-level loss alone;
   the ship read is at the render level.
3. **ΔW cosine ≠ render closeness, in both directions.** → No ΔW-based gate;
   `compare_ckpt_dw`-style reads are descriptive only.
4. **Nondeterministic paired runs carry a ΔW-cos chaos floor ≈ 0.41.** →
   `--deterministic` for any paired comparison, or read only at render level.
5. **Placement amplifies**: interventions early in a from-zero LoRA
   trajectory redirect the endpoint (late-vs-early ΔW cos 0.91 vs 0.19). →
   The warm start above is load-bearing, not a convenience.
6. **Teacher-matching MSE is the one loss that tracks quality here**
   (`docs/findings/agsm_reward_premise_holds.md`); data-FM-MSE does not. →
   `L(θ)` is a legitimate progress signal for *this* loop; it still is not
   the ship gate (constraint 2).

## Experiments (each has a kill; stop at the first kill)

**P0 — consolidate, 3 artists, rank 32 vs. exact concat rank 96.**
Teacher = three existing artist LoRAs (any three from `output/ckpt/`),
tags routed per probe prompt. Student `θ` rank 32 warm-started from
`merge_loras --normalize global` SVD-truncated to 32. ~300 steps, batch of
probes ≥ 8 per σ-bin. Read: `L(θ)` at start vs. end (did it move off the
warm start at all), then a paired PE-cos render grid (3 artists × 8 prompts
× 2 seeds) vs. (a) each single artist LoRA, (b) the rank-96 exact concat,
(c) the rank-32 SVD-truncated concat = the warm start. **Kill:** (c) ≈ end
of distillation within the paired noise — the closed-form merge already
had it; no toolkit script is written. ≈ 1–1.5 GPU-h.

**P1 — ensemble-soup on existing soup ingredients.** Take one finished
`make soup` run's N seed ingredients (already on disk; no new fine-tunes).
Teacher = output-space mean of the N; student warm-started from the shipped
SVD soup. Read as P0 vs. the SVD soup. **Kill:** same as P0. This is also the
only cheap probe of whether the output-space average differs from the
weight-space one on shared-init ingredients (the soup line's LMC premise);
if they coincide the ensemble reading is closed too. ≈ 1 GPU-h.

**P2 — turbo port, one artist.** Only if P0 passes (the loop exists then).
Teacher = `base + artist` at 28 steps CFG 4; student = frozen turbo stack +
`θ`, probes on the turbo 4-knot grid plus interior knots. Compare against
linear stacking `turbo ⊕ artist` at 4 steps CFG 1 on the turbo bench prompt
grid. **Kill:** no visible gain over linear stacking at full-res eyeball —
LCM-LoRA-style composition was already good enough. ≈ 2 GPU-h.

**Not scheduled:** guidance bake `w > 0` (only after a P0/P2 pass; it is a
one-flag extension of the same loop), the EC per-reference compile (own
proposal), any per-probe selection variant (constraint 1).

## Cost / placement

Total to a P0 verdict ≈ 1.5 GPU-h + the loop (~300 lines lifting the turbo
two-view forward). Tier 1.5 toolkit item if P0 passes (`make lora-distill`,
bench under `bench/lora_distill/` with the paired-render envelope). Does not
touch `train.py`, configs, caches, or the render line.

## Open questions

- Does the tag-routed teacher for consolidate need the probe prompts to be
  artist-tagged one at a time (N× forward cost) or can the batch mix tags?
  Mixed is cheaper and tests interference directly.
- Probe support for the turbo port: the student only ever sees its 4 knots
  at inference, but the artist teacher's effect lives across all 28 — which
  interior knots matter is the same question the DP-DMD dynamic schedule
  answered for the distiller (`dynamic_schedule`, `docs/methods/turbo.md`).
- Whether `--traj_stats` should grow a raw-`x_t` sidecar or the probe
  rollout should be its own tiny recorder (the sidecar was sized for codes,
  not latents; ~4096 tok × 16 ch × 28 knots × f16 ≈ 3.7 MB per trace — fine).
