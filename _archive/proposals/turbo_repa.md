# Turbo × REPA — relational alignment for the DP-DMD student

Status: **ARCHIVED — REFUTED (2026-06-17).** REPA on the DP-DMD student did not
work: the post-fix retrain (`repa2`) showed REPA *amplifying* visual drift 5–10×
rather than reducing it. Line closed.
Original status preserved below.

Status: **Phase 0 PASSED (DRIFT confirmed) — Phase 1 wired; first A/B arm
INVALID (view bug, fixed 2026-06-13); post-fix retrain (`repa2`) FAILED — REPA
*amplifies* visual drift 5–10× instead of reducing it, with a train-loss /
probe contradiction that points at the grad path, not the loss (see "Phase 1 —
first results" below). Line blocked pending a grad-path diagnostic.** The
`[repa]` section
is live in `configs/methods/turbo.toml` (`weight = 0.0` default ⇒
byte-identical; `--repa_weight 0.05` is the A/B arm); PE sidecars ride the
distill `CachedDataset` (`load_repa_pe`), `train/repa_align_loss` +
`train/repa_active` ride TB; the REPA forward is checkpointed
(`selective_block_grad_ckpt` — un-checkpointed it OOM'd a 16 GB card) with a
grad-requiring input guarding the unsloth closed-over-param grad drop. Wiring
tests: `tests/test_turbo_repa.py`.

**Post-mortem of the first arm** (`anima_turbo_N_repa`, 2026-06-12 overnight —
do not evaluate it as a REPA result): the term originally ran AFTER the GAN gen
forward and rode `loss_student.backward()`; `set_view` is global state read at
forward time and checkpoint recompute defers to backward, so the GAN teacher
checkpoint recomputed under the student view all run → silently corrupted GAN
generator grads (samples awful, caption-probe 0c cosine 0.27, align profile
scrambled +166%@σ0.75/−51%@σ1.0, while forward metrics and val/div stayed
healthy). Fixed: the REPA term now backwards immediately (div_loss split-bwd
pattern) BEFORE the GAN forward, restoring the invariant that the GAN gen
forward is the last view-switch before the main backward. Same-class guards
added at config time (repa+per_step_expert+--grad_ckpt = error;
--grad_ckpt+GAN = pre-existing broken combo, warned).
Probe: `bench/turbo/probe_alignment_drift.py` (run
`bench/turbo/results/20260612-2244-alignment-drift/`, N=200, paired
arms). Student align loss worse than base at every σ except exactly 1.0:
+21.8% @ σ=0.75 and +21.2% @ σ=0.50 (frac_worse 1.00 — all 200 images),
+12.7% @ σ=0.97, +8.9% @ σ=0.25; at σ=1.00 (pure-ε input, no image content)
the student is −3.0% (slightly *better*). Reading: the visual-representation
drift is real and peaks mid-σ on the student's operating band → primary
real-data arm confirmed (renoise-distribution τ sampling already covers the
peak); the σ=1.0 sign flip says the step-0 caption-ranking gap is a
text-conditioning problem (soft-rank's site), not PE-visible — consistent
with the three-axis split. The Gram/pooling math is factored into reusable
helpers (`library/training/repa.py::relational_align_loss`) shared
bit-identically with the training adapter.
Depends on REPA v2 Phase 0 (closed
2026-06-12, relational/Gram arm won — `docs/experimental/repa.md`) and the
turbo feature-tap API (landed — `forward_mini_train_dit(...,
return_block_features=, return_features_early=)`, bit-exact no-op when off).
Sibling proposals on the same loop: `turbo_caption_ranking.md` (Phase 1
wired, soft-rank at the step-0 diversity site) and `turbo_gan.md` (Idea 1
implemented, off by default).

## Why now

Three lines converge:

1. **The student's representation measurably degrades.** The caption-ranking
   probe (Phase 0 of `turbo_caption_ranking.md`) showed the DP-DMD student
   loses caption discriminability at the step-0 state: shuffled rank@1
   **0.750 vs base 0.958** at σ=1.0. That is *text-side* evidence that
   distillation pulls the student's mid-block representation off the base
   manifold. Soft-rank attacks the text axis directly; nothing currently
   anchors the **visual** axis.
2. **REPA is validated on this model.** Phase 0 + the quarter-run readouts
   established the mechanism end to end on the LoRA family: relational Gram
   vs PE-Spatial at block 8, weight 0.05 ≈ 4% of total loss, align curve
   plateaus by ~18% of a run, decisively better grids. The same machinery
   (grid match, spatial_norm, the Gram form's no-head property) transplants.
3. **The loop already has everything within reach.** The distill batch
   carries real cached latents + matched captions (used by the diversity
   anchor and mean-var calibration); PE-Spatial sidecars exist for the same
   cache; and the feature-tap API can capture student block-8 features with
   an early exit (~9/28 of a forward) — no hooks, no second graph.

Positioning vs the GAN head (`turbo_gan.md` Idea 1): the discriminator puts
*adversarial, distribution-level* pressure on teacher features of
**generated** samples. REPA is *direct, per-sample* alignment on **real**
data. Different data, different failure modes, compose freely. Together with
soft-rank the three span text / visual / distribution axes of the same
degradation.

## Mechanism

Every `every_n`-th student step, one extra **partial** student forward on
noisy real data:

```
x_τ   = (1 − τ)·latents_real + τ·ε            # the loop's renoise primitive
feats = _forward("student", x_τ, τ, c_matched,
                 return_block_features={repa.layer},   # default 8
                 return_features_early=True)            # exit after block 8
L_repa = GramMSE(pool(feats), spatial_norm(PE_sidecar))  # relational, fp32
loss_student += repa.weight * L_repa
```

- **Relational (Gram) form only** — no projection head on the student, no new
  params, dimensions never need to match (`library/training/repa.py`
  `extra_forwards` is the reference implementation of the pooling + Gram;
  factor the math into a reusable helper rather than copying).
- **Backward site**: the term joins the student loss at the DMD-phase
  accumulation (where the GAN generator loss lands, `distill.py:~1069`) —
  NOT inside the step-0 `split_bwd` diversity graph. The load-bearing
  detach-after-first structure stays untouched.
- **σ choice**: sample τ from the same renoise distribution the DMD path
  uses, so the alignment pressure lives on the student's actual operating
  inputs. Per-step-expert students (`step_expert_K > 1`) select head
  k = nearest student-grid step for the sampled τ.
- **Cost**: `every_n = 4` ⇒ ~0.08 extra partial forwards per step amortized
  (~9/28 depth, grad-bearing but early-exited) — small next to the existing
  2·k_anchor teacher + N student + 4 fake forwards per step.

**Bespoke-loop caveat (load-bearing)**: none of train.py's REPA
infrastructure reaches `scripts/distill_turbo/` — the dataset PE gate
(`dataset.load_repa_pe`), the adapter dispatch, and the step metrics all have
to be mirrored explicitly in the distill loop (the established
bespoke-loop-mirroring pattern). Concretely: set `load_repa_pe` on the
distill `CachedDataset`, extend the batch tuple, and log
`repa/align_loss` + `repa/active` to the loop's own metric stream.

## Phase 0 — drift probe (no training, the gate for everything else)

`bench/turbo/probe_alignment_drift.py`: measure the unweighted
relational align loss (block 8 vs PE-Spatial, spatial_norm on) for **base**
vs the **existing trained turbo student** checkpoint, over real renoised
latents at σ ∈ {1.0, 0.97, 0.75, 0.5, 0.25} (matching the caption-probe σ
grid), N ≥ 200 images, fixed seed. Envelope per `bench/_common.py`.

Pre-registered readout:

- **Student align loss clearly worse than base at σ ≥ 0.75** (where the
  ranking gap lives) → the premise holds, proceed to Phase 1.
- **Student ≈ base everywhere** → CLOSE the line without a training run: the
  representation drift isn't PE-visible, the caption degradation is a
  text-conditioning problem, and soft-rank alone owns the fix.

This mirrors how the heatmap diagnostic closed REPA lever 3 in one
measurement — buy the answer before buying the training runs.

## Phase 1 — wiring + A/B (gated on Phase 0)

New `[repa]` section in `configs/methods/turbo.toml`, following the `[gan]`
pattern (**`weight = 0.0` default ⇒ byte-identical training**):

| key | default | note |
|---|---|---|
| `weight` | `0.0` | off → byte-identical; first live value 0.05 (LoRA-validated scale) |
| `layer` | `8` | block tap (matches LoRA REPA) |
| `encoder` | `"pe_spatial"` | sidecar `{stem}_anima_pe_spatial.safetensors` next to the TE cache |
| `every_n` | `4` | amortization (soft-rank convention) |
| `spatial_norm` | `true` | iREPA target standardization (eyeball-validated on LoRA) |

A/B at fixed seed/data/iterations, `--infer_steps 4 --cfg 1.0`,
`weight = 0` vs `0.05` (one order-of-magnitude sweep only if directionally
right). Pre-registered gates, in order:

1. **Caption-ranking probe** (`bench/turbo/caption_ranking_probe.py`) —
   shuffled rank@1 at σ=0.97/1.0 must not regress; recovery toward base is
   the win condition (same metric contract as soft-rank Phase 1, so the two
   levers are directly comparable and their composition measurable).
2. **CMMD** non-regressing.
3. **Grids** — no diversity collapse (the DP in DP-DMD is the point of the
   method; an alignment term that collapses diversity fails regardless of
   probe gains), anatomy intact.

If both this and soft-rank individually pass, a third composed run (both on)
decides whether they stack.

## Phase 1 — first results (2026-06-13): FAILED, blocked on a grad-path bug

Post-fix retrain `anima_turbo_N_repa2` (`repa_weight=0.05`, `layer=8`,
`pe_spatial`, `every_n=4`, `spatial_norm=true`, `per_step_expert=false`,
`student_steps=4`; TB `output/logs/turbo/20260613-115205`). This is the
**retrain after the view-bug fix**, so it should be a clean read of the arm.
It is not — REPA made the very thing it optimizes worse.

**Alignment-drift probe** (`bench/turbo/probe_alignment_drift.py`, the
unweighted relational Gram align loss to PE-Spatial — REPA's *own* objective;
lower = better, "excess" = (student−base)/base):

| σ | base | no-REPA `turbo_N_1250` | `repa2_625` | `repa2_1250` |
|---|---|---|---|---|
| 0.25 | 0.391 | +8.9% | +16.0% | +26.6% |
| 0.50 | 0.207 | +21.2% | **+147.7%** | +161% |
| 0.75 | 0.131 | +21.8% | **+194.8%** | +214% |
| 0.97 | 0.137 | +12.7% | +53.8% | +65.1% |
| 1.00 | 0.242 | −3.0% | −43.9% | −40.4% |

REPA amplifies drift **5–10×** at every signal-bearing σ (0.25–0.97) vs the
no-REPA baseline, and only improves alignment at pure noise (σ=1.0). The
profile is **baked in by step 625 and flat through 1250** — same shape, same
magnitude — so it is not a late-run blow-up that an earlier stop would dodge.
(Runs: `bench/turbo/results/20260613-1501` (1250), `…-1530` (625);
baseline `…20260612-2244`.)

**The smoking gun — train loss ≠ checkpoint reality.** `train/repa_align_loss`
logged **0.080** at step 630 (below base — looks like it is working), but the
offline probe of that exact checkpoint measures **0.39–0.51** at mid-σ
(catastrophically *above* base). A healthy-looking forward loss sitting on a
checkpoint that didn't move is the signature of
[[project_turbo_view_ckpt_recompute_hazard]] (set_view/set_student_step are
global; a ckpt'd forward must backward while its view is live or recompute
silently corrupts grads — loss values stay healthy). The REPA forward
(`distill.py:~1101`) runs under `selective_block_grad_ckpt` right after a
`set_student_step()`, and the one consistent improvement — σ=1.0 (−44%) — fits:
only the noise-end head gets a clean gradient, everywhere with image content
drifts further off-manifold. **The 2026-06-13 view-bug "fix" did not resolve
this** — either it was incomplete or the cause is elsewhere on the grad path.

**Caption-ranking probe** (`bench/turbo/.../20260613-1517`, vs no-REPA
`…20260611-2020`) is a wash-to-worse: REPA nudged up the σ≈1 *shuffled* tail
(0.750→0.792 @1.0, the originally-flagged hotspot) but made the **hard**
(semantically-close contrastive) rank@1 worse at every σ (e.g. 0.75:
0.941→0.824; 0.97: 0.882→0.706). Both probe gates still fire DEGRADATION/DRIFT.

**Next step (blocking).** Before any further training or weight sweep, confirm
the REPA gradient actually lands on the deployed student: rerun a short REPA
stint with grad-ckpt off for the REPA forward (or assert view-liveness at the
REPA backward) and check whether the step-N checkpoint's *probed* align loss
tracks the *logged* `train/repa_align_loss`. If they reconcile, the
ckpt-recompute view corruption is confirmed and fixable; if they don't, the
drift is real and the arm is dead. Do not interpret any weight/anneal sweep
until this contradiction is closed.

## Secondary arm (fallback only)

If real-data PE loading proves awkward in the distill loop, a cheaper
variant exists: relational alignment of student block-8 Grams to **teacher**
block-8 Grams at the GAN tap (`x_renoised`, generated data — the teacher
feature capture at `distill.py:~1021` already exists). No PE cache, no
dataset change. It is, however, a proximal-to-base pressure on generated
states (teacher features at few-step states can be off-manifold, and it may
directly fight the distillation update), so it is the fallback, not the
primary. Don't run both arms blind — pick by Phase-0's σ profile.

## Non-goals

- **Absolute / MLP-head arm** — closed by REPA Phase 0 + iREPA.
- **REPA on the fake critic** — the critic must track the student's current
  distribution, not real data; anchoring it breaks the DMD score estimate.
- **Input-token masking** — same constant-token-bucketing constraint as the
  LoRA family.
- **Token-subset Gram** — lever 3 closed ~uniform on the LoRA family; if it
  is ever revisited here, the `repa_grad_heatmap` machinery is reusable.

## References

- `docs/experimental/repa.md` — validated mechanism + operating numbers;
  §"Annealing plan" carries the live lever status (anneal open, candidate
  0.25; spatial_norm on; lever 3 closed by heatmap). Historical plan archived
  at `_archive/proposals/repa_phase1_operating_point.md`.
- `docs/proposal/turbo_caption_ranking.md` — degradation evidence + the
  shared Phase-1 gate contract.
- `docs/proposal/turbo_gan.md` — feature-tap API + the adversarial sibling.
- `docs/experimental/dpdmd.md`, `docs/structure/dpdmd.md` — loop structure.
- `library/training/repa.py` — Gram/pooling reference implementation.
