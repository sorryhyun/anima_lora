# StelLA-Chimera — keep the experts orthogonal *while* they move

Status: **proposal, nothing implemented.** The first deliverable is a
*diagnosis* probe, not a method: figure out what the OrthoInit-chimera collapse
at ~4k steps actually is. StelLA is only justified if that collapse is loss of
the orthogonal structure during training, not router collapse. Frozen-Cayley
chimera is the decision benchmark; the `expert_basis_mult`/`expert_diag`
capacity levers are the *real* baseline to beat, not plain chimera.

## Executive claim

ChimeraHydra's experts are supposed to be mutually orthogonal: the content pool
reads a shared input subspace `Q_basis_c` disjoint from the freq pool's
`Q_basis_f`, and within each pool every B-expert owns a disjoint output
subspace (`networks/lora_modules/chimera.py`). Today that orthogonality is
bought one of two ways, and each pays for it:

- **Frozen-Cayley (default):** bases are frozen to the top-2r SVD slices; only
  the Cayley rotations and `λ` train. Orthogonality and disjointness are exact
  and *stay* exact, but `colspace(ΔW)` is capped inside the initial span — the
  "OrthoLoRA feels too weak" problem, inherited per pool.
- **`use_ortho_init=True`:** the bases become free `Parameters` (no Cayley, no
  frozen subspace) — full expressivity — but nothing keeps them orthonormal or
  the experts disjoint, and the arm **collapsed at ~4k steps**
  ([[project_chimera_expert_capacity_levers]]).

That is exactly the gap StelLA fills. [StelLA](https://github.com/SonyResearch/stella)
(NeurIPS 2025) keeps the adapter's input/output factors on the **Stiefel
manifold throughout training** via a Riemannian projection + polar retraction,
so the subspaces are *learned* (uncapped) yet *stay orthonormal* every step. The
claim worth testing:

> Chimera's `use_ortho_init` collapse is loss of the orthogonal/disjoint expert
> structure under free training. Constraining the per-pool bases to the Stiefel
> manifold — so experts can leave the SVD span without losing mutual
> orthogonality — prevents the collapse and beats the frozen-Cayley cap.

The claim is deliberately narrow and **front-loaded with a kill switch**: if the
~4k collapse is router collapse (the gate concentrating on ~2 live experts, not
the bases drifting together), Stiefel on the bases cannot rescue it and this
proposal is dead on arrival. Phase 0 is that diagnosis.

## Why this is a better fit than the plain-LoRA StelLA case

I was lukewarm on StelLA for plain LoRA: there is no *measured* deficiency on
Anima that maintaining orthonormality fixes, and StelLA's own initialization
ablation (Table 5) shows the SVD seed **washes out** — SVD-major ≈ SVD-minor ≈
random once the subspace is trainable. Two things make chimera different:

1. **There is a measured failure.** The OrthoInit-chimera collapse at ~4k is a
   concrete, reproducible defect. StelLA's constraint targets precisely the
   failure mode (uncontrolled basis drift) that a free trainable basis exhibits.
2. **The load-bearing part of StelLA here is the *constraint*, not the init.**
   Table 5's washout is about initialization; it does **not** undercut this use,
   because chimera's value from StelLA is maintaining orthogonality *during*
   training, which is the thing the collapse says is missing. So we should
   expect — and Phase 1 should confirm — that the win (if any) is robust to the
   seed and comes from the manifold, not from where we start.

## The chimera-specific extension: mutual, not just per-factor, orthogonality

Vanilla StelLA keeps a single `(U, V)` orthonormal. Chimera needs more: content
⊥ freq, and the K experts within a pool mutually disjoint. The clean
generalization is to put the *stacked* factors on one Stiefel manifold:

- Stack each pool's shared input basis and both pools together:
  `Q = [Q_c; Q_f] ∈ St(2r, d_in)` — columns orthonormal ⟹ content and freq
  input subspaces stay disjoint every step.
- Stack a pool's K per-expert output bases:
  `P_pool = [P_1 … P_K] ∈ St(K·r, d_out)` — columns orthonormal ⟹ the experts'
  output subspaces stay mutually disjoint every step.

The middle factor (chimera's per-expert `diag(λ)` / `S`) stays Euclidean and
carries the amplitude, exactly as in StelLA. This is the property frozen-Cayley
buys *statically* (disjoint SVD slices) — StelLA-Chimera buys it *dynamically*,
which is the whole point: the subspaces can rotate out of the SVD span toward
the task while the disjointness invariant is preserved by the retraction.

Keep ΔW = 0 at init the way chimera already does: `S = 0` (λ = 0) plus the
centered gate, both pools. StelLA's "Zero" init arm performed ~equal to its
"non-zero" default in Table 5, so the zero start costs nothing measurable and
preserves the clean-LoRA property the save/distill path depends on.

## Relation to what chimera already has

| Variant | Bases | Orthonormal during training | Disjointness | Expressivity | Status |
|---|---|---|---|---|---|
| Frozen-Cayley (default) | frozen SVD slices | yes (exact) | static, exact | capped to span | stable, the cap |
| `expert_basis_mult`/`expert_diag` | frozen, over-complete + per-expert diag | yes | static | deepened, still in-span | working capacity lever, distills to standard |
| `use_ortho_init` | free `Parameter` | **no** | **none** | full | **collapsed ~4k** |
| **StelLA-Chimera** | Stiefel (trainable) | **yes (retraction)** | **dynamic** | full | proposed |

StelLA-Chimera is the missing cell: the only one that is both *trainable to full
expressivity* and *orthogonality-preserving*. It is the principled repair of the
`use_ortho_init` collapse, and the head-to-head it must win is against the
frozen capacity levers (`expert_basis_mult`/`expert_diag`) — those already get
"deeper experts without freeing the basis" and distill cleanly, so beating plain
chimera proves nothing.

## Minimal implementation surface

This is **not** the cheap, checkpoint-free surface of SVD-Down. It is a
training-loop addition, justified only if Phase 0 indicts basis drift.

- Reuse the `use_ortho_init` branch of `ChimeraHydraLoRAModule` (bases already
  become fp32 `Parameter`s there) and add a `chimera_stiefel=True` mode that
  registers the stacked `Q`/`P_pool` factors as the trainable bases.
- A Riemannian optimizer wrapper (StelLA's recipe): pre-hook converts the
  Euclidean grad to the tangent space; the base AdamW step runs unchanged;
  post-hook projects the perturbed update back to the tangent space and applies
  a **polar retraction** (`uf(Y + Δ)`). Batch the retraction across layers
  (StelLA reports 15–20× from one stacked SVD) and run it in **fp32** — bf16 SVD
  is a known stability trap.
- It lives in the **optimizer step**, outside `compile_blocks` (which compiles
  `block._forward`), so there is no compile interaction — but the per-step SVD
  cost scales with (K_c + K_f) × adapted Linears and must be reported separately
  from per-step throughput.
- **Save/distill is free.** At save, materialize `B_k = P_k S_k`, `A = Q` — that
  is already the free-form per-pool layout `lora_save` distills to the standard
  `*_chimera.safetensors`. The inference module (`ChimeraHydraInferenceModule`)
  is untouched; on-disk/merge/inference paths are identical.
- New TOML kwarg `chimera_stiefel` must be registered in
  `networks/__init__.py::NETWORK_KWARGS` ([[project_network_kwarg_toml_allowlist]]).

Note: per-channel `channel_scaling` is provably **inert on frozen-basis ortho**
([[project_per_channel_scaling_audit]]). StelLA bases are trainable, so it may
become *live* again here — audit before assuming the calib transfers, or hold it
at α=0 for the A/B to keep the comparison clean.

## Evidence plan and decision gate

### Phase 0 — diagnose the collapse (this gates everything)

Re-run the `use_ortho_init` chimera arm that collapses and, over the run, log:

- **per-expert subspace principal angles** (content↔freq, and expert↔expert
  within each pool) — does mutual orthogonality decay before the collapse?
- **per-expert effective rank** of `B_k` — are experts losing rank / merging?
- **gate entropy / live-expert count** per pool — is the router concentrating
  on ~2 experts (the content half already shows K_c=6 → ~2 live,
  [[project_chimera_content_half_weak_overprovisioned]])?

Phase 0 **indicts StelLA** only if orthogonality/rank decay *leads* the collapse.
**Kill or redirect** if gate collapse leads — then the fix is router/`K` sizing
(drop `num_experts_content` to 2–3), not a manifold constraint. Either way the
probe is reusable and the K-sizing question gets answered.

### Phase 1 — StelLA-Chimera vs the real baselines

Only if Phase 0 indicts basis drift. Identical rank, K_c/K_f, LR, scheduler,
seed, dataset order. Arms:

1. Frozen-Cayley default (the cap to beat);
2. `expert_basis_mult`/`expert_diag` (the working capacity lever — the *real*
   baseline);
3. `use_ortho_init` (the collapsing arm — the thing we claim to repair);
4. **StelLA-Chimera.**

Measure: does arm 4 reach ≥4k steps **without** the orthogonality/rank decay
Phase 0 found in arm 3; loss-vs-step AUC and CMMD ([[project_cmmd_val_signal]] —
FM-MSE is uninformative here, [[project_fm_val_loss_uninformative]]); per-pool
expert specialization (content NMI, freq σ-band selectivity); fixed-prompt
fixed-seed grids at matched steps. Confirm any win across ≥3 seeds — init
changes the lottery, and a single non-collapsing run proves little.

**Ship** only if StelLA-Chimera beats frozen-Cayley *and* the capacity levers on
CMMD/rendered quality without the collapse. **Kill** if it collapses anyway
(then the cause was never the bases), if it merely ties the frozen levers (the
retraction cost is not worth it), or if the gain disappears once `K` is
right-sized per Phase 0.
a
## Risks and honest limits

- **The collapse may be router-side.** This is the dominant risk and the reason
  Phase 0 exists. If the gate collapses to ~2 live experts, no amount of Stiefel
  on the bases helps.
- **Retraction cost.** Per-step SVD × (K_c+K_f) × layers. Even batched, it is
  real and must be reported; chimera already carries two pools of experts.
- **A working frozen alternative exists.** `expert_basis_mult`/`expert_diag`
  already deepen experts collapse-safe and distill to standard. StelLA-Chimera
  must beat *that*, and it is strictly more machinery.
- **Measurement resolution.** Anima ortho-family deltas have been hard to A/B
  ([[project_per_channel_scaling_audit]], the ortho-family A/B history). A win
  this subtle may sit under CMMD noise — budget multiple seeds.
- **The seed should not matter.** If a Phase-1 win tracks the SVD seed rather
  than the manifold constraint, that contradicts StelLA's Table 5 and means
  something other than the proposed mechanism is acting — treat as a red flag,
  not a win.

## Recommendation

Build **Phase 0 only** for now — it is a diagnosis probe, not a method, and it
answers a question worth answering regardless (what *is* the `use_ortho_init`
collapse, and is K over-provisioned?). Green-light the StelLA-Chimera build only
if Phase 0 shows the experts lose their orthogonal/disjoint structure before
they collapse. If instead the router collapses first, the cheaper, higher-value
move is right-sizing `num_experts_content` and the FreqRouter — and StelLA stays
shelved with a clear, falsifiable reason on record.

## References

- Li et al., *StelLA: Subspace Learning in Low-rank Adaptation using Stiefel
  Manifold*, NeurIPS 2025 (Spotlight),
  [arXiv:2510.01938](https://arxiv.org/abs/2510.01938). Local copy:
  `NeurIPS-2025-stella-subspace-learning-in-low-rank-adaptation-using-stiefel-manifold-Paper-Conference.pdf`.
  Code: <https://github.com/SonyResearch/stella>.
- ChimeraHydra deep-dive: `docs/experimental/chimera-hydra.md`;
  module `networks/lora_modules/chimera.py`.
- Sibling proposal (the plain-LoRA case, Phase 0 passed):
  `_archive/proposals/svd_down_lora_init.md`.
- Memory: [[project_chimera_expert_capacity_levers]],
  [[project_chimera_content_half_weak_overprovisioned]],
  [[project_orthoinit_variant]], [[project_cmmd_val_signal]].
