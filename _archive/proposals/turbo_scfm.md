# SCFM — velocity-space self-distillation for turbo (DP-DMD alternative)

> **ARCHIVED / CLOSED 2026-06-30.** This line was built, trained across five runs,
> and falsified — Term B is inert on Anima in both `renoise` and `rollout` modes,
> so SCFM reduces to per-step FM distillation onto the teacher's instantaneous
> field (blurry at 4 steps). All SCFM code (loss loop, config knobs, EMA plumbing,
> `make scfm`, tests, `scfm.toml`) was removed from the live tree. The verdict +
> evidence live in `docs/findings/scfm_velocity_fidelity_closed.md`. This file is
> kept only for the §9 progress log. Do not re-propose a pure velocity-fidelity
> distiller for Anima.

> Proposal to add **SCFM** (*Shortcutting Pre-trained Flow Matching Diffusion
> Models is Almost Free Lunch*, Cai, Y. Wu, Chen, H. Wu, Xiang, Wen — NeurIPS
> 2025; project page `shortcutfm.github.io`; PDF `2431_Shortcutting_Pre_trained_.pdf`
> in repo root) as a **selectable turbo objective** (`base_loss = "scfm"`)
> alongside the incumbent `dpdmd` / `dmd`. SCFM is a *trajectory-fidelity*
> self-distillation: no critic, no GAN, no diversity anchor. Output is a plain
> velocity-field LoRA — merges and infers exactly like the shipped turbo student.
>
> For the incumbent see `docs/methods/turbo.md` (ops), `docs/structure/turbo.md`
> (math), and the migration record `_archive/proposals/dpdmd.md`. Prior review of
> this paper: [[project_scfm_paper_verdict]].

Status: **Phase 1 BUILT; lr↑ to 5e-5 is a clear win (2026-06-29) — sharpness
recovered toward Arm-3 *and* diversity/text kept; Term B still inert.** The full
progress log is §9 (read it first); the short version:

- **Phase 0 (probes, no training):** image gate read NO-GO — a naive 4-step
  teacher (Arm 2 = SCFM's predicted ceiling) renders washed-out, *below* the
  DP-DMD student (Arm 3). The on-manifold consistency-residual scan was ~0.044
  (field already ~95% straight ⇒ Term B has little to optimize). **But** a
  teacher-trajectory straightness probe falsified the *structural* reason for the
  NO-GO: Anima's cfg=4 transport is nearly straight (cos 0.96–0.997) and
  non-crossing in the low-σ band (commits by σ≈0.5) — the reflow-friendly regime
  where straightening *can* work. So the Arm-2-as-lower-bound was too pessimistic
  and the structural objection was withdrawn; no cheaper probe remained →
  settling it needed the training run.
- **Phase 1 (built, §3/§4):** selectable `base_loss="scfm"` implemented (files in
  §9). A 1.5k-step student (rank 64, ~25 min) trains clean.
- **First result (`student_lr=1e-5`):** coherent, real pose variety + legible-ish
  speech bubbles (the axes DP-DMD loses) but **soft / washed-out = Arm-2** —
  because `scfm_consistency_residual` stayed flat ~0.05 (**Term B inert**), so Term
  A pulls the student toward the teacher's instantaneous field whose 4-step rollout
  *is* Arm-2. `1500 > 750` ⇒ under-trained, not ceiling-bound.
- **lr↑ to 5e-5 (the win, §9.3):** same 1.5k, **sharp + saturated** renders (off the
  washout, toward Arm-3) while **keeping** diversity/text; `div_ac_sim` 0.34→0.30
  (more diverse too). Stable bar one transient at the step-1000 EMA restart that
  recovers. Term B still inert — the gain is Term A converging harder. **Most
  promising the line has looked.** Next = seed-matched montage vs `anima_turbo_R_4500`
  + teacher (the GO gate), then train longer.

**Run it:** `make turbo` with `base_loss="scfm"` (already set in the toml), or
`--base_loss scfm`. Infer at **`--cfg 1.0`** (CFG=4 is baked into Term A's teacher
target) with `--infer_steps = student_steps`, plain Euler.

> Original status: **PROPOSAL — Phase 0 not yet run.** This is written *because* the
> incumbent has empirically plateaued below the teacher (§0). It is **not** a
> pivot recommendation: SCFM ships as a selectable objective so DP-DMD stays the
> default and the two are A/B'd on the same seeds.

---

## 0. Why now (read this first)

The shipped DP-DMD+GAN student has **plateaued, and the plateau is a deficit, not
teacher-parity**.
On `anima_turbo_R` (6k iters, GAN@0.03, rank 96):

- training converged by ~2–4.5k (4.5k ≈ 6k by eye; `div_loss` 85%-done by 2k);
- side-by-side vs the 28-step cfg=4 teacher on fixed seeds, the gap is **large and
  structured**: (1) **pose/composition diversity collapse** — every student
  sample is the same frontal standing pose while the teacher gives genuinely
  varied shots; (2) **text/caption fidelity lost in distillation** — the teacher
  renders bubble text cleanly, the student garbles it (this is *not* a base
  ceiling — the teacher proves base capability; it is the step-0
  caption-discriminability loss of [[project_turbo_caption_ranking_phase0]] made
  visible); (3) background/detail richness.

The wired levers aimed at this gap are nearly exhausted:

- **REPA** — empirically net-negative on turbo; a
  representation-space pull fights the reverse-KL grad. Out.
- **GAN@0.03** — already on in the plateaued run; spent.
- **f-distill (KL reweighting)** — tried; no win. Spent.
- **div_weight↑ / softrank** — the only cheap wired knobs left; worth a sweep but
  they tune the existing DMD objective, they don't change its ceiling.

The common thread: **DP-DMD's distribution-matching objective, on Anima, mode-
collapses and sheds conditioning faster than its diversity anchor can repair.**
The "exceed the teacher" property that motivated DMD is *worthless to us right
now* — we are well *below* the teacher; we would be delighted to *match* it. That
is exactly what a fidelity objective optimizes for. SCFM is the cheapest
structural arm that targets the gap we measured rather than the gap DMD was built
for.

## 1. The SCFM mechanism (mapped to our notation)

SCFM trains a **single velocity field** `V_θ(x_t, t)` to be **self-consistent
across step sizes** so that few-step Euler sampling matches the teacher — with no
step-size embedding (unlike shortcut models). Linear flow path `x_t = (1−t)x₀ +
t·ε`, velocity `v = ε − x₀` — identical to Anima. Two terms (their Eq. 13), mixed
per-sample over a batch with ratio `k/N = 0.4`:

**Term A — teacher rectification (first `k` samples).** Coarse-step direction is
pinned to the teacher:

```
loss_A = ‖ V_θ(x_t, t) − V_teacher(x_t, t) ‖²        # V_teacher CFG-guided, no-grad
```

This is the *only* term that carries teacher information; it sets the quality
ceiling. `x_t` is a renoised **real cached latent** (SCFM is a real-data arm —
we already have the latents; cf. REPA's renoise path).

**Term B — velocity-space self-consistency (remaining `N−k` samples).** One
coarse step must equal two finer steps, evaluated on a **stop-grad EMA copy of
the student** `θ⁻` (their Eq. 12):

```
# adjacent finer sub-steps d_i, d_{i+1} on the shifted σ-grid, from x_{t_i}
v1            = V_θ⁻(x_{t_i},   t_i)                  # EMA fwd #1, no-grad
x_{t_{i+1}}   = euler_step(x_{t_i}, v1, d_i)
v2            = V_θ⁻(x_{t_{i+1}}, t_{i+1})            # EMA fwd #2, no-grad
v_target      = (d_i·v1 + d_{i+1}·v2) / (d_i + d_{i+1})
loss_B        = ‖ V_θ(x_{t_i}, t_i) − v_target ‖²    # student grad at the SAME x_{t_i}
```

Intuitively: Term A says "point where the teacher points at coarse scale", Term B
says "a big step = two small steps on your own field" → the trajectory
**straightens**, which is what makes 4-step Euler work. Diversity is preserved
*because the objective never collapses onto predicted clean samples* (the DMD
failure mode) — it only constrains the velocity field's geometry.

**EMA / stop-grad.** `θ⁻ ← μθ⁻ + (1−μ)θ` after each student step (Eq. 14), on the
LoRA params (Eq. 15 gives the LoRA-space form). Vanilla uses one EMA (`μ=0.999`)
with a cyclic restart `θ⁻ ← θ` every ~1000 steps; their accelerator is **dual-EMA**
(fast `μ=0.99` + slow `μ=0.999`, no manual restart) — Phase 2, not the minimal
port.

**No critic, no GAN, no anchor, no teacher-generated reference images.** Output is
a plain velocity LoRA; inference is unchanged (`--infer_steps N --cfg 1.0`).

Reference: paper Eq. 12–16, Alg. 1 (App. A); their Flux-Dev result distills a
32-step teacher → 3-step student in <1 A100-day, and a few-shot (10-image) variant.

## 2. Grounding on Anima (Phase 0 — cheap, inference-only)

SCFM's quality ceiling is the **teacher's own velocity field sampled at few
steps**, plus whatever the straightening term recovers on top. So the decisive
pre-build probe needs no training:

**Probe:** on the fixed `turbo2` seed set, render three arms at 1024²:
1. **28-step teacher** (cfg=4) — ceiling, already in `comfy/output/turbo2/teacher__*`.
2. **N-step naive teacher** — the *same* teacher velocity field rolled at the
   student grid (`student_steps=4`, cfg baked) with plain Euler. This is the
   **lower bound** on a perfect SCFM student (SCFM ≥ naive-Euler because Term B
   straightens the field).
3. **current DP-DMD student** (`anima_turbo_R_4500`, 4-step cfg=1.0) — the
   incumbent we must beat.

Plus a **consistency-residual scan** (the Eq. 11 residual of the *base* field
across σ): sample `x_t`, compare `V(x_t,t)` against its two-sub-step composition;
report the residual per σ-band. Large residual at few-step sizes = headroom for
the straightening term; near-zero = SCFM adds little over naive Euler.

**Read (the gate, §7):**
- If **naive N-step teacher already beats the DP-DMD student** on the structured
  gaps (pose variety + text), SCFM has a high ceiling and almost certainly wins —
  **GO**.
- If naive N-step teacher is *also* collapsed/blurry (i.e. the 28→4 gap is mostly
  irreducible discretization, not recoverable by field-fidelity), then no
  velocity-fidelity method beats it at 4 steps — you genuinely need
  distribution-matching, and SCFM is **NO-GO** (stay on DP-DMD, spend the cheap
  knobs). The consistency-residual scan disambiguates the marginal middle case.

Metric caveat (same as DP-DMD's): score **pose/structure**, not pooled cosine —
PE-Core pooled is blind to the axis we care about. Read the grids; use `pe_spatial` spatial
tokens for a number. Bench home: `bench/turbo/`, standard `result.json` envelope
(`bench/_common.py`).

## 3. What changes in our code

SCFM **reuses the entire turbo harness** — the one structural addition is the EMA
copy of the student, and even that piggybacks on existing infra: under
`base_loss="scfm"` the unused **fake stack becomes the EMA-student `θ⁻`**
(identical shape to the student; no optimizer, updated by parameter EMA).
`set_view("fake")` then evaluates `θ⁻` with zero new plumbing.

| Piece | DP-DMD | SCFM | Action |
|---|---|---|---|
| Objective selector | `base_loss ∈ {dpdmd, dmd}` | add `"scfm"` | **extend** (`config.py:307`, `:835`) |
| Student forward | N-step rollout, step-1 detached | per-sample single fwd at sampled `t` (Term A) / `t_i` (Term B) | **add branch** |
| Teacher use | K-step CFG anchor + DMD real score | Term-A CFG target only (1 renoise fwd) | **simplify** |
| Diversity anchor (`k_anchor`,`div_weight`) | load-bearing | **inert** (kept for A/B) | keep-inert |
| Fake critic stack | trainable score net | **repurposed as EMA `θ⁻`** (no grad, no opt) | **re-role** |
| Fake update loop (`distill.py:1198`) | 5× MSE steps + disc | EMA param update (Eq. 14) | **replace** |
| GAN (gen+disc) | optional, on | **off** under scfm | bypass |
| Consistency target | — | Eq. 12 (2 EMA fwds) | **add** |
| Output / inference | plain LoRA, N-step cfg=1 | identical | **keep** |

### 3.1 New config knobs (`scripts/distill_turbo/config.py`)

```
# base_loss selector gains "scfm"  (choices → ("dpdmd","dmd","scfm"))
# [scfm]
k_ratio: float        = 0.4     # k/N — fraction of batch on Term A (teacher), rest on Term B
ema_mu: float         = 0.999   # θ⁻ decay (vanilla single-EMA)
ema_restart: int      = 1000    # cyclic restart θ⁻←θ every N steps (0 = off)
dual_ema: bool        = false   # Phase 2: fast(0.99)+slow(0.999), disables ema_restart
n_consistency_grid: int = 8     # finer sub-step grid Term B samples adjacent pairs from
teacher_cfg reused from [dmd]   # Term-A target CFG (bake CFG=4, like the paper's Flux path)
flow_shift reused from [sampling]
student_steps reused from [dmd] # the inference grid SCFM is consistent down to
```

Validation: `0 < k_ratio < 1`; `fake_rank == student_rank` under `scfm` (the EMA
must match the student exactly — error otherwise); `n_consistency_grid ≥
student_steps`. Knobs inert under `scfm` (kept for A/B, documented like the
DP-DMD doc's inert-CA list): `k_anchor`, `div_weight`, `detach_after_first`,
`dmd_grad_step`, `dm_x0_norm`, all `gan.*` / `f_distill.*`.

**CFG handling — deliberately the Flux path, not the SD3.5 path.** The paper
CFG-*conditions* the velocity predictor for SD3.5 (samples `w∈[3.5,5]`, Eq. 16).
We instead **bake a single CFG=4** into the Term-A teacher target (the paper's
Flux-Dev route, since Flux is already guidance-distilled). This keeps the output a
plain LoRA at `--cfg 1.0` and avoids adding a `w` input that would break the bake
([[project_tlora_inference_full_rank]] logic — no per-call scalar at inference).

### 3.2 Loss assembly (new branch, replaces the anchor/DMD/GAN region for scfm)

```python
# real-data trajectory (cached latents → x_t); split batch by k_ratio
x0 = latents
is_termA = (arange(B) < int(round(k_ratio * B)))     # per-sample role

# Term A (teacher rectification) — first-k samples
t_a            = sample_t(B, ...)
x_t_a          = renoise(x0, t_a, randn_like(x0))
v_tea          = teacher_cfg_velocity(x_t_a, t_a, teacher_cfg)   # no-grad, CFG (2 fwd)

# Term B (self-consistency) — remaining samples, EMA θ⁻ view
i              = randint_adjacent(n_consistency_grid, B)         # sample t_i, d_i, d_{i+1}
x_ti           = renoise(x0, t_i, randn_like(x0))
with view("fake"):                                              # θ⁻ (EMA), no_grad
    v1  = student_ema(x_ti, t_i)
    x_ti1 = euler_step(x_ti, v1, d_i)
    v2  = student_ema(x_ti1, t_i1)
v_target_B     = (d_i*v1 + d_i1*v2) / (d_i + d_i1)

# single student grad forward over the whole batch (mixed t = t_a / t_i)
with view("student"):
    v_stu = student(where(is_termA, x_t_a, x_ti), where(is_termA, t_a, t_i))
target = where(is_termA, v_tea, v_target_B)
loss   = masked_mse(v_stu, target.detach())                    # use_masked_loss honoured
loss.backward(); clip; student_opt.step()

# EMA update (Eq. 14) — no backward, no optimizer
turbo.update_ema(ema_mu)                                        # θ⁻ ← μθ⁻ + (1−μ)θ
if ema_restart and step % ema_restart == 0: turbo.reset_ema()  # θ⁻ ← θ
```

`teacher_cfg_velocity` already exists (the DP-DMD DM real score uses it).
`update_ema` / `reset_ema` are ~10-line additions to `TurboDMDNetwork` over the
student/fake param lists (`student_params()`/`fake_params()`,
`turbo_dmd.py:453/457`). `save_student` (`:483`) is unchanged — `θ` is saved, `θ⁻`
and the (now-EMA) fake stack are discarded, so the artifact stays a plain LoRA.

### 3.3 Constraints that survive contact with Anima

- **Block-swap + multi-forward.** SCFM runs 1 student-grad + 2 EMA + (CFG) teacher
  forwards per step — multi-forward, same offloader hazard as DP-DMD. Keep `blocks_to_swap=0` for
  Phase 1 (default already); the cost analysis says we don't need swap (see §below).
- **Compile.** `compile_blocks()` keys on token count; all forwards reuse the same
  bucket. View flips (student↔fake) are flag flips on the frozen base, no
  adapter-multiplier change mid-graph → compile is clean (same property DP-DMD
  relies on). The EMA stack is a sibling LoRA, not a recompile trigger.
- **Memory.** Only the single student forward is grad-bearing (no BPTT — SCFM is
  per-step, not a rollout). Peak VRAM is **below** DP-DMD (no critic backward, no
  N-step graph). Grad-ckpt likely unnecessary.
- **Output bake.** Plain LoRA — `merge`/inference untouched.

### Cost vs the incumbent (per training iteration, in full-depth-forward units `f`)

| | DP-DMD + GAN (`anima_turbo_R`) | SCFM |
|---|---|---|
| teacher anchor | ~12f | — |
| student | ~4f (rollout) | ~1f (single fwd) |
| DMD real+fake | ~3f | — |
| Term-A teacher target | — | ~2f (CFG, k-subset) |
| Term-B EMA target | — | ~2f |
| fake update | ~5f | — (EMA = ~0f) |
| GAN gen+disc | ~5.5f | — |
| **≈ total** | **~30f** | **~5f** |

SCFM is **~6× cheaper per step**, and the paper reports **fewer steps** to
converge → plausibly an order of magnitude less training compute to a comparable
checkpoint. The saving is *not* dropping the GAN (~5.5f) — it is removing the
critic maintenance (~10f) and the anchor rollout (~12f).

## 4. Phasing

- **Phase 0 — naive-Euler-teacher probe + consistency-residual scan** (§2). No
  training. Gate in §7.
- **Phase 1 — minimal port.** `base_loss="scfm"`, single EMA + cyclic restart,
  Term A + Term B, fake-stack-as-EMA, GAN bypassed. `blocks_to_swap=0`, batch=1,
  1024² native buckets. Gate: a checkpoint matching `anima_turbo_R_4500` on
  CMMD/quality while **beating it on the structured gap** (pose variety + text
  legibility) at fixed seeds, grids confirmed.
- **Phase 2 — dual-EMA + k_ratio / EMA-μ sweep.** Replace cyclic restart with
  fast/slow dual-EMA (their convergence accelerator); sweep `k_ratio` (teacher↔
  consistency balance — the quality↔straightening dial) and `n_consistency_grid`.
- **Phase 3 — scale + few-shot check + block-swap audit.** Confirm the paper's
  few-shot claim on an Anima artist subset; audit the multi-forward offloader
  before enabling swap on larger runs.

## 5. Risks / open questions

1. **Quality ceiling = the teacher field at few steps.** SCFM cannot exceed the
   teacher (it has no distribution-matching term). Phase 0 exists to confirm that
   ceiling is high enough to beat the DP-DMD student. If the naive-Euler teacher
   is itself collapsed at 4 steps, SCFM is NO-GO — this is the load-bearing risk.
2. **Inert-EMA — less of a threat here than the archive feared.** The shelved
   consistency aux was flagged
   "EMA-LoRA teacher inert on frozen backbone". But SCFM's `θ⁻` is **not** a
   capability gap (teacher-vs-student); it is a stop-grad *stabilizer* of a
   **geometric** constraint (one big step = two small steps) evaluated on the same
   weights. The constraint has signal even when `θ⁻ ≈ θ`, because it compares
   *different step sizes*, not different models. So the objection that sank the
   output-space LCM aux does not transfer cleanly. (It is still worth logging the
   Term-B residual to confirm it doesn't trivially collapse to a straight line
   that ignores the teacher — Term A is what prevents that.)
3. **Trivial-straight-line solution.** Term B alone is minimized by *any*
   self-consistent field, including a degenerate straight line that loses the
   teacher's detail. Term A (`k_ratio`) is the anchor against this — too low and
   the student straightens away from the teacher; too high and it reduces to plain
   per-step flow-matching distillation (blur, the SPD failure
   [[project_spd_distill_blur_snr_gate]]). The `k_ratio` sweep (Phase 2) is not
   optional.
4. **Few-step grid coverage.** Term B samples adjacent pairs on
   `n_consistency_grid`; if the grid doesn't densely cover σ<0.45 (where Anima
   detail/text resolves, [[project_sigma_signal_resolves_by_045]]) the student
   won't straighten in the band that carries the text we're trying to recover.
   Bias the grid toward the low-σ tail.
5. **CMMD blind to the win.** Our live val is blind
   to pose and only partly sees text. Checkpoint selection must use the
   structure-sensitive diversity metric + grids, like DP-DMD.

## 6. Decision gate (Phase 0)

**GO to Phase 1 if** the naive N-step teacher (arm 2) visibly beats the DP-DMD
student (arm 3) on pose variety and/or text legibility on the `turbo2` seeds —
i.e. there is recoverable teacher-fidelity headroom at 4 steps that DMD is
leaving on the table. The consistency-residual scan should also show non-trivial
Eq.-11 residual at the student step sizes (room for the straightening term).

**NO-GO (stay on DP-DMD)** if the naive 4-step teacher is itself collapsed/blurry
— then the 28→4 gap is irreducible discretization, only distribution-matching can
help, and the move is the cheap DP-DMD knobs (`div_weight↑`, `softrank`) +
accepting `anima_turbo_R_4500`.

> **What actually happened (2026-06-29):** the gate read NO-GO on the image axis
> but the structural reason was falsified, so Phase 1 was built and run anyway.
> See the §9 progress log for the full record — this §6 is kept as the original
> pre-build decision logic.

## 7. Relationship to the incumbent

SCFM ships as a **selectable** `base_loss="scfm"`, never replacing `dpdmd` by
default — the two are A/B'd on identical seeds (the same dual-path discipline the
DP-DMD migration used, `_archive/proposals/dpdmd.md §8`). Decommissioning DP-DMD
is **out of scope** for this proposal and would require SCFM to win the Phase-1
gate *and* hold up across Phase 2/3; until then both objectives coexist. The two
are not exclusive in principle — a future arm could use SCFM's Term-B consistency
as an auxiliary on top of the DP-DMD anchor — but that composition is explicitly
**not** the build here (start with the clean, ~6×-cheaper standalone objective and
measure it against DP-DMD before mixing).

## 8. References

- SCFM — Cai, Y. Wu, Chen, H. Wu, Xiang, Wen, NeurIPS 2025. `shortcutfm.github.io`;
  PDF `2431_Shortcutting_Pre_trained_.pdf` (repo root). Prior review:
- `_archive/proposals/dpdmd.md` — the incumbent's migration proposal (this doc
  mirrors its structure + dual-path discipline).
- `docs/structure/turbo.md` / `docs/methods/turbo.md` — incumbent math + ops.
- Plateau / teacher-gap evidence: [[project_turbo_R_plateau]],
- Shelved-aux context (why the inert-EMA objection is weaker here):
- Phase-0 probes (live): `bench/turbo/probe_consistency_residual.py`,
  `bench/turbo/probe_teacher_straightness.py`; montage
  `bench/turbo/results/20260629-1440-scfm-consistency-residual/`.

## 9. Progress log

### 9.1 Phase 0 — probes, no training (2026-06-29)

Three seed-matched arms on the `turbo2` set (1 prompt × 8 seeds 0–7, 768×1344,
plain Euler, `flow_shift=3`), montage in
`bench/turbo/results/20260629-1440-scfm-consistency-residual/`:

| Arm | Field | Steps | CFG | Role |
|---|---|---|---|---|
| 1 | base DiT | 28 | 4.0 | clean teacher ceiling |
| 2 | base DiT | **4** | 4.0 | naive teacher = **SCFM's predicted ceiling** |
| 3 | `anima_turbo_R_4500` | 4 | 1.0 | DP-DMD incumbent |

**Result: Arm 2 is dramatically *worse* than Arm 3** — pale, washed-out,
sketch-like, faint bubbles; Arm 3 is sharp/saturated/complete (pose-collapsed +
some bubble garble, but another tier). Arm 1 confirms the base *can* render
cleanly, so the 28→4 gap is real discretization loss, not a base ceiling. This is
the §6 NO-GO branch.

Consistency-residual scan (`probe_consistency_residual.py`, relative Eq.-11 error
of a coarse Euler step vs its two-sub-step composition, base field, 24 latents × 2
noises × 4 intervals):

```
 σ_a    σ_b    σ_c    curvature  residual
1.000  0.950  0.900    0.115      0.057
0.900  0.825  0.750    0.069      0.034
0.750  0.625  0.500    0.063      0.032
0.500  0.250  0.000    0.104      0.052   ← low-σ text/detail band
overall residual_rel = 0.044
```

The field is **already ~95% straight** at the student step sizes → Term B's
straightening has little to optimize *on-manifold*.

**Straightness probe that re-opened it** (`probe_teacher_straightness.py`, 16
seeds, 28-step cfg=4 rollout): the teacher transport is **nearly straight**
(straight cos(v, x−x0) 0.962 high-σ → 0.997 low-σ; turn cos 0.97→0.999) and
**non-crossing in the low-σ band** (NN-preservation: σ0.9→0.00, σ0.75→0.19,
σ0.5→0.81, σ0.25→1.00 — image commits by σ≈0.5; high-σ NN=0 is "not decided near
pure noise", not a multivalued-blur obstruction). This is the reflow-friendly
regime where straightening *can* reach Arm 1.

**Reconciliation:** naive 4-step Euler washes out not because the transport is
curved/crossing but because it integrates the cfg=4 field in big steps at
**off-trajectory** points (the residual scan was measured *on-manifold*, so it
*understates* the off-manifold gap that drives Arm 2). That off-manifold big-step
error is exactly what a *coarse-grid* Term B would target — see 9.3. Net: §6
image gate = NO-GO but the structural objection withdrawn; settling Arm-1
reachability needed the training run.

### 9.2 Phase 1 — minimal port built + first run (2026-06-29)

Implemented as selectable `base_loss="scfm"` (branch `turbo-scfm-phase1`):

- `scripts/distill_turbo/scfm.py` — `run_scfm`: Term A teacher rectification +
  Term B velocity self-consistency, per-step Bernoulli(`k_ratio`) role at B=1,
  single EMA + cyclic restart, fake-stack-as-θ⁻, GAN/REPA/soft-rank/mean-var
  bypassed.
- `networks/methods/turbo_dmd.py` — `freeze_ema` / `update_ema` (lerp_
  μθ⁻+(1−μ)θ, Eq. 14) / `reset_ema`, shape-paired student↔fake.
- `scripts/distill_turbo/config.py` + `[scfm]` in `configs/methods/turbo.toml` —
  the knobs + the validation protecting the EMA-as-fake-stack contract
  (`fake_rank == student_rank`, `n_consistency_grid ≥ student_steps`, plain-LoRA
  only; dual_ema/per_step_expert/ortho refused; GAN/REPA/etc force-inert + warn).
- `tests/test_scfm.py` — config + EMA-math + masked-loss invariants (17 tests).

**First run** (`anima_turbo_scfm`, rank 64, 1.5k iters, ~25 min, k_ratio 0.4,
ema_mu 0.999, restart 1000, n_consistency_grid 8):

- **Renders** (turbo2 seeds, 4-step cfg=1): coherent, **good pose variety + legible-ish
  speech bubbles** (the axes DP-DMD loses) but **soft / hazy / low-contrast =
  Arm-2 territory**, below the DP-DMD student on sharpness. `1500 > 750` by eye →
  still under-trained, asymptote unknown.
- **Telemetry:** `scfm_consistency_residual` **flat ~0.05** the whole run → **Term
  B inert** (no straightening work, matching the on-manifold 9.1 scan); `loss_b`
  saturated ~0.002–0.004 from step 0; `loss_a` noisy, no clear downtrend;
  `val/div_ac_sim` 0.39→0.34 (mild diversity gain), `val/div_xpred_ac_sim`
  0.95→0.92 (still high cross-seed similarity).

**Read:** the predicted **trade** — SCFM recovers diversity + text but regresses to
the teacher-field softness at few steps. Term A pins the student to the teacher's
*instantaneous* field (whose 4-step rollout = Arm 2); Term B, as the paper
specifies (on-manifold renoised-real points, *finer* sub-steps), operates where
the field is already straight and so adds nothing. Neither term constrains the
**off-manifold points the 4-step rollout actually visits** — that is the gap.

### 9.3 lr↑ to 5e-5 — validated win (2026-06-29)

`anima_turbo_scfm_highlr` = same config, `student_lr` 1e-5 → **5e-5** (5×),
1.5k iters, rendered turbo2 seeds at 4-step cfg=1:

- **Renders are sharp + saturated** (blue skies, colored bikinis, clean speech
  bubbles) — a large step off the 1e-5 washout toward Arm-3, while **keeping** the
  pose variety + text legibility.
- **Diversity also improved:** `val/div_ac_sim` 0.34→**0.30** (vs 1e-5's 0.39→0.34;
  lower = more diverse), `div_xpred_ac_sim` 0.95→**0.90**. So 5e-5 won on *both*
  quality and diversity.
- **Stable, not broken:** one transient `loss_a` spike (~0.29 at step ~1000) that
  **fully recovers** by ~1490 (back to ~0.005) — unlike the DP-DMD GAN case which
  broke *permanently* within ~1k steps. Confirms the SCFM (non-adversarial)
  objective tolerates the high lr; the [[project_turbo_lr_instability_threshold]]
  GAN-oscillation threshold does **not** bind here.
- **Caveat:** that spike sits right on the **EMA restart at step 1000** (`θ⁻←θ`) —
  under high lr the restart discontinuity is the likely cause. Watch it; consider
  `ema_restart=0` or dual_ema if it recurs / lands a bad checkpoint.
- **Term B is *still* inert** (`consistency_residual` flat ~0.05 regardless of lr)
  — the gain came from **Term A converging harder**, not from straightening. The
  structural Term-B lever (9.4 item 3) remains untouched and available.

### 9.4 Open decision (next)

With lr 5e-5 as the new baseline:

1. **Seed-matched gate vs the incumbent.** Render the 5e-5 SCFM student, the
   DP-DMD student (`anima_turbo_R_4500`), and the 28-step teacher on the **same**
   turbo2 seeds → one montage (the §4 Phase-1 gate). The qualitative read is
   "competitive with DP-DMD on sharpness, ahead on diversity/text," but confirm it
   side-by-side before declaring GO.
2. **Train the 5e-5 config to 3–4k** (cheap): does it keep climbing or plateau?
3. **lr fine-tune:** 3e-5 as a steadier middle ground if the step-~1000 spike
   worries (nearly as fast, less edge-of-stability); or keep 5e-5 + set
   `ema_restart=0` to remove the restart discontinuity.
4. **If quality plateaus soft → the structural lever: redesign Term B to roll the
   EMA student on its own *coarse student grid* from noise** (DMD2-style
   train-on-your-own-trajectory), enforcing 1-big-step = 2-grid-steps *there* —
   the only thing that targets the off-manifold washout (9.1 reconciliation).
   Toggle to A/B against the on-manifold variant. **Bigger lever than dual_ema.**
5. **`dual_ema`: still deferred,** but it now has a concrete secondary motivation
   (smooth the high-lr restart transient by dropping the manual restart). Revisit
   only after the seed-matched gate.
6. **`k_ratio` is a near-dead dial while Term B is inert.**

### 9.5 Grad-accum + off-trajectory Term B built (2026-06-29)

Two builds on the clean `configs/methods/scfm.toml` (the `make scfm` sibling of
`make turbo`, both driven by the same `scripts/distill_turbo/` loop):

- **Gradient accumulation (`optim.gradient_accumulation_steps`, SCFM-only).** The
  paper's Eq. 13 splits a batch of N into k Term-A + (N−k) Term-B so every
  optimizer step sees both terms; at `batch_size=1` a micro-step is a pure-A-OR-B
  coin flip, throttling Term-A duty cycle to `k_ratio`. Accumulation mixes both
  roles within the window (each micro-loss scaled `1/accum`, so LR is
  accum-invariant). The dpdmd/dmd loops reject `accum>1` (they step every
  micro-step). **`anima_scfm_accum` run (accum 4, lr 5e-5): confirmed grad-accum
  is NOT the Term-B fix** — `consistency_residual` stayed flat 0.037–0.054
  (mean ~0.045), identical to the highlr run. Expected: accum addresses Term-A
  throughput, not Term-B inertness (which is the on-manifold straightness of
  §9.1). Settled on **accum 2** as the speed/mixing trade (accum 4 too slow;
  P(≥1 A) at k_ratio 0.4, accum 2 ≈ 0.64).

- **Off-trajectory Term B (item-4 redesign), `scfm.term_b_point`.** New toggle:
  `"renoise"` (default, paper-faithful — on-manifold renoised real, inert here)
  vs `"rollout"` (rolls θ⁻ from noise on the **coarse student grid** to the
  off-manifold states the few-step Euler rollout visits, then enforces the
  "1 coarse step == 2 half-steps via the midpoint" velocity consistency *there*).
  Grad flows only through the student forward at the visited state; the rollout +
  both EMA sub-steps are stop-grad θ⁻ (DMD2-style train-on-your-own-trajectory).
  Trivial-straight-line collapse (§6 risk 3) is guarded by Term A still pinning
  the on-manifold field to the teacher (`k_ratio>0`). Cost: +~N/2 no-grad θ⁻
  forwards per Term-B micro-step. **`scfm.toml` now ships `term_b_point="rollout"`
  + accum 2.** Code + smoke (rollout/renoise × accum) + config tests landed
  (`tests/test_scfm.py`, 22 pass). **The gate is `train/scfm_consistency_residual`
  climbing off the ~0.05 floor** — if it stays flat in rollout mode too, the
  washout is not a velocity-inconsistency the student can fix and the line is
  closed; if it climbs and the seed-matched montage sharpens vs `anima_turbo_R`,
  rollout becomes the SCFM default.
