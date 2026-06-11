# Programmatic variant stacking (examples) + `lora_modules/` dedup

Status: Part A **shipped** (2026-06-11); Part B **fully shipped** (B0 + B1 + B2 + B3,
2026-06-11)
Date: 2026-06-11

Two threads, independently shippable, ordered by risk:

- **Part A** ✅ **DONE** — an `examples/` script showing how to *stack* LoRA-family
  features from Python (`create_network(use_ortho_init=True, use_timestep_mask=True, …)`
  + the per-step conditioning hook), so the TOML toggle blocks are no longer the only
  documented way to compose variants. Zero-risk: pure addition. Shipped as
  `examples/07_stack_ortho_init_tlora.py` + README/doc links — see the §Part A
  status note below.
- **Part B** — deduplicate the ~40–50% boilerplate repeated across the seven
  `lora_modules/` variant classes (dtype policy, forward scaffold, router-state
  plumbing) into shared mixins/template methods, plus formalize the de facto
  network-level interface as a `typing.Protocol` with a contract test.
  Numerically inert by construction; verified bit-exact.

Explicit **non-goal**: a trait-composition framework where ortho / tlora / routing are
freely stackable plugins assembled by the factory. The combinations that don't exist
today are blocked by *math*, not wiring (ortho's frozen top-r basis vs MoE's disjoint
expert subspaces; channel scaling provably inert on frozen-basis variants; the chimera
mask deliberately scoped to the content pool). The explicit cascade in
`networks/__init__.py::resolve_network_spec` — every shipped combo is a named,
benched class — stays. Making unbenched combos *expressible* would be a regression,
not a feature (CONTRIBUTING Tier 2 requires a bench per new method anyway).

---

## Part A — example: stack OrthoInit + T-LoRA timestep masking from Python

> **Status: shipped 2026-06-11.** Delivered as `examples/07_stack_ortho_init_tlora.py`
> (+ `examples/README.md` row 07 + a cross-link in `docs/methods/timestep_mask.md`).
> One deviation from the sketch below: the forward/loss/backward runs under
> `torch.autocast(bf16)` — both what the real trainer uses (so the adapter's
> `org_forwarded.dtype` compute-dtype policy lands on its intended bf16 path) and
> what keeps the example portable to CPU. The rest matches the plan: kwargs →
> `resolve_network_spec`, the single `apply_router_conditioning` hook, a few real
> synthetic-latent DiT forwards, per-step live mask-rank print, `--compile` flag.

### Why

`examples/06_frozen_dit_training_build.py` shows the *plainest* programmatic build
(default LoRA, no variant kwargs). Everything beyond that is documented only as TOML
toggle blocks in `configs/methods/lora.toml`. An embedder (or a future bespoke
trainer in `scripts/`) who wants "OrthoInit bases + T-LoRA rank schedule" has to
reverse-engineer three facts that are nowhere demonstrated together:

1. Variant selection is **kwargs → `resolve_network_spec`** — the same keys the TOML
   carries pass straight through `create_network(**kwargs)`
   (`networks/lora_anima/factory.py:81`). No config file required.
2. T-LoRA is **not a class** — it's the `_timestep_mask` buffer every variant
   inherits from `BaseLoRAModule` (`networks/lora_modules/base.py:85`), rebound
   per-step by `LoRANetwork.set_timestep_mask` (`networks/lora_anima/network.py:979`).
   Stacking it on OrthoInit is therefore *free*: the mask gates the singular-value
   gate `lambda_layer` (`networks/lora_modules/ortho.py:323`).
3. The per-step driving in training is one call —
   `apply_router_conditioning(...)` in
   `library/training/forward/router_conditioning.py:18` — which `hasattr`-probes the
   network and fires `set_timestep_mask` / `set_sigma` / `set_fei` in a stable order.
   A bespoke loop should call that, not hand-roll the individual setters.

### Deliverable

`examples/07_stack_ortho_init_tlora.py`, following 06's structure (argparse,
`default_checkpoints()`, numbered build steps, stops after a few demonstration
training steps — no dataset, synthetic latents). Sketch of the load-bearing part:

```python
from networks.lora_anima.factory import create_network
from library.training.forward.router_conditioning import apply_router_conditioning

# 1. Variant stacking happens HERE — the same keys lora.toml carries, as kwargs.
#    use_ortho_init  → OrthoInitLoRAModule (trainable SVD-seeded bases, λ-gated,
#                      ΔW=0 at init; distills to plain LoRA at save)
#    use_timestep_mask → T-LoRA power-law rank schedule; gates λ per step.
#    (use_ortho_init=True + use_moe_style would raise in resolve_network_spec —
#     impossible combos fail loudly at build, they don't silently degrade.)
network = create_network(
    multiplier=1.0,
    network_dim=32,
    network_alpha=16.0,
    vae=None,
    text_encoders=[],
    unet=model,
    use_ortho_init=True,      # ortho (init-only) axis
    use_timestep_mask=True,   # T-LoRA axis — stacks: mask gates lambda_layer
    min_rank=1,               # T-LoRA floor (cfg default 1)
    alpha_rank_scale=1.0,     # T-LoRA schedule exponent (cfg default 1.0)
)
network.apply_to(text_encoders=[], unet=model,
                 apply_text_encoder=False, apply_unet=True)

# ... place / compile LAST / grad-ckpt / freeze base — identical to example 06 ...

# 2. Per-step conditioning — the ONE hook the real trainer fires
#    (timestep_mask → sigma → fei → balance, stable order for cudagraph).
#    For this stack only set_timestep_mask does work; the rest no-op.
warmup_step = 0
for step, (latents, timesteps) in enumerate(synthetic_batches()):
    warmup_step = apply_router_conditioning(
        network=network,
        noisy_model_input=noisy_latents,
        timesteps=timesteps,
        is_train=True,
        warmup_step=warmup_step,
        max_train_steps=args.steps,
    )
    # forward / loss / backward as usual — the mask is already live on every
    # adapted Linear (single shared GPU buffer, one write per step).
```

Things the example's docstring must state (each one is a documented bite point):

- **T-LoRA is training-only** — inference runs full rank at every `t`; never call
  `set_timestep_mask` in a sampling loop (`docs/methods/timestep_mask.md`).
- **The mask schedule barely moves learned effective rank** on Anima
  (`bench/timestep_mask/`) — the example demonstrates the *mechanism*; don't expect
  the schedule itself to be a big quality lever, and question `network_dim` before
  tuning `alpha_rank_scale`.
- Save path: OrthoInit **distills to a standard LoRA** (`sqrt-split λ`), so the
  output checkpoint loads anywhere a plain LoRA does — the stack is a *training-time*
  composition with no inference-side footprint.
- Which combos exist: point at the variant matrix in `networks/CLAUDE.md`
  (three-axis table) rather than re-listing it.

Also: add one paragraph + a link to `examples/README.md` ("07 — stacking variants
programmatically") and a cross-link from `docs/methods/timestep_mask.md`.

### Acceptance

- `python examples/07_stack_ortho_init_tlora.py --steps 3` runs on a single GPU,
  prints the trainable-param split (à la 06) plus the live mask rank at each step
  (e.g. `step 1: t=0.83 → effective rank 7/32`), exits 0.
- `--compile` flag exercises the compile-after-apply ordering with the stack on.

---

## Part B — `lora_modules/` dedup

### Current duplication (measured)

Across `lora.py` / `ortho.py` (3 classes) / `hydra.py` / `stacked_experts.py` /
`chimera.py` (~3.5K lines total), roughly 1.3–1.5K lines are near-verbatim repeats:

| Repeated scaffold | Where | ~Lines |
|---|---|---|
| Training-forward dtype dance: `work = org_forwarded.dtype`, cast-at-GEMM-boundary, `_rebalance(x.to(work))`, final `.to(org_forwarded.dtype)` | all 7 forwards | ~200 |
| Forward chain: skip-module → rebalance → down-GEMM → `* _timestep_mask` → dropout → rank-dropout → up-GEMM → scale | all 7 forwards | ~350 |
| Router buffer plumbing: per-class `set_sigma`/`set_fei`/`set_routing_weights` wrappers + gate-expand around the shared `router_state.py` free functions | hydra / ortho_hydra / stacked_experts / chimera | ~400 |
| `__init__` scaffolding: SVD init variants, buffer registration, channel-scale absorption | all 7 | ~400 |

This duplication is not just aesthetic — it is the repo's **demonstrated bug
multiplier**. The compute-dtype fix (key GEMMs off `org_forwarded.dtype`, not
`x.dtype`; commit `8c2005c`) had to be hand-applied per variant, **missed OrthoInit
on the first pass**, and EasyControl's cond-LoRA *still* carries the old `x.dtype`
policy today (tracked as a latent risk). One shared scaffold turns that bug class
into a single-point fix.

### B1 — forward scaffold (template method on `BaseLoRAModule`)

> **Status: shipped 2026-06-11** (with B0 in one commit). The scaffold landed on
> `BaseLoRAModule.forward` with `_down` / `_gate` / `_up` / `_eval_delta` hooks
> (default `_gate = lx * _timestep_mask`). Migrated the three modules that fit the
> standard two-GEMM, elementwise-`lx`-gate shape bit-exactly — **LoRAModule,
> OrthoInitLoRAModule, StepExpertLoRAModule** — i.e. exactly the `work =
> org_forwarded.dtype` family where the 8c2005c dtype fix was the load-bearing
> line (OrthoInit was the one it originally missed). The dtype-policy comment and
> the T-LoRA gate are now single-point. The frozen-basis Cayley modules
> (OrthoLoRA / OrthoHydra — one batched Cayley solve shared between down and up,
> `work` = basis dtype) and the router-gated MoE modules (Hydra / StackedExperts /
> Chimera — the "gate" is a routing tensor consumed inside the up-projection, not
> an `lx` multiply) keep bespoke forwards: forcing them into the `lx→lx` hook adds
> conditional plumbing for negative value (the proposal pre-authorized leaving
> chimera bespoke; the same reasoning extends to the other Cayley/router forwards,
> and B2's `RouterStateMixin` is the right home for the router-plumbing dedup).
> Verified bit-exact via B0's golden harness + `tests/test_lora_dtype_policy.py`.

Highest value, do first. Move the invariant chain into the base; variants override
only the rank computation:

```python
# base.py
def forward(self, x):
    if not self.enabled or getattr(self, "_fused", False):
        return self.org_forward(x)
    org_forwarded = self.org_forward(x)
    if not self.training:
        return org_forwarded + self._eval_delta(x, org_forwarded)
    if self._skip_module():
        return org_forwarded
    work = org_forwarded.dtype          # THE dtype policy, stated once
    x_lora = self._rebalance(x.to(work))
    lx = self._down(x_lora, work)        # ← variant hook (free / Cayley / einsum)
    lx = self._gate(lx, work)            # ← variant hook (mask / λ·mask / routing·mask)
    if self.dropout is not None:
        lx = torch.nn.functional.dropout(lx, p=self.dropout)
    lx, scale = self._apply_rank_dropout(lx)
    lx = self._up(lx.to(work), work)     # ← variant hook
    return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)
```

- Default `_gate` is `lx * self._timestep_mask` — T-LoRA composition becomes a
  property of the scaffold, inherited by every current and future variant.
- The long dtype-policy comment currently pasted into each forward
  (`lora.py:84-95`, `ortho.py:303-314`, …) lives **once**, above `work = …`.
- `LoRAModule`'s Conv2d branch and `_fused` short-circuit stay in its overrides
  (`_down`/`_up` dispatch on Linear-vs-Conv as today).
- **Chimera is last and optional**: its dual-pool forward (two A's, two gates,
  centered gating) genuinely differs; force-fitting it into the hooks adds
  conditional plumbing for negative value. If it doesn't drop out naturally,
  leave its forward bespoke and take the win on the other six.

Compile note: the hook indirection is plain Python method dispatch — Dynamo inlines
it; the always-a-Tensor / no-None-guard invariants (`base.py:82-89`,
`router_state.py` header) are preserved because the scaffold keeps the mask multiply
unconditional. Still verified explicitly (see Verification).

### B2 — `RouterStateMixin`

> **Status: shipped 2026-06-11.** `RouterStateMixin` landed in
> `router_state.py` carrying the six-method surface (`set_sigma` / `clear_sigma`
> / `set_fei` / `clear_fei` / `set_routing_weights` / `clear_routing_weights`)
> plus a `_register_router_io_buffers` helper for the σ/FEI/routing-weights
> placeholder trio. HydraLoRA / OrthoHydra / StackedExperts now inherit it
> (`class X(RouterStateMixin, BaseLoRAModule)`), deleting their pasted wrappers.
> One design refinement over the sketch below: each setter is
> **buffer-presence-guarded** (`hasattr(self, "_sigma"/"_fei"/"_routing_weights")`)
> rather than flag-guarded. That single guard (a) subsumes the old
> `getattr(self, "use_global_router", False)` check — the routing buffer is
> registered iff `use_global_router`, so `hasattr` is the exact condition — and
> (b) lets StackedExperts (which registers only `_routing_weights`) inherit the
> full surface with `set_sigma`/`set_fei` as safe no-ops, equivalent to never
> defining them: `network.py::_wire_shared_*` keys its `_*_aware_loras` lists on
> *buffer* presence, not method presence, so the mixin adds zero new wiring. The
> FeRA grad-path contract (slot-assign, no `.detach()`/`.copy_()`) stays in
> `router_state._set_routing_weights` (one place) and the mixin just gates it.
> Chimera keeps its own `_ChimeraRoutingMixin` (two routing buffers, different
> surface). Verified bit-exact via B0's golden harness + the full routing test
> set (`test_global_router`, `test_hydra_sigma_band`, `test_lora_dtype_policy`).

`router_state.py` already centralizes the *functions* (`_register_*` / `_set_*` /
`_clear_*`, pointer-stability + grad-carrying contracts in its docstrings). What's
still duplicated is the per-class *method surface* wrapping them. Promote to a mixin
providing `set_sigma` / `set_fei` / `set_routing_weights` / `clear_*` and the
registration calls in `__init__`, consumed by hydra / ortho_hydra / stacked_experts
(chimera again optional — it has two routing buffers). The FeRA grad-path contract
(direct slot assignment, **no** `.detach()`/`.copy_()` on `_routing_weights` —
`router_state.py:224-237`) moves into the mixin once instead of being a convention
each class must remember. A future router source becomes a one-place edit.

### B3 — network-level `Protocol` + contract test (cheap, optional but recommended)

> **Status: shipped 2026-06-11.** `networks/protocol.py` defines two
> `@runtime_checkable` non-data protocols: `AdapterNetwork` (the core
> trainer-facing surface — `apply_to` / `load_weights` /
> `prepare_optimizer_params_with_multiple_te_lrs` / `set_multiplier` /
> `is_mergeable` / `enable_gradient_checkpointing` / `prepare_grad_etc` /
> `on_epoch_start` / `get_trainable_params`) and `RouterConditionableNetwork`
> (the optional per-step setters `set_timestep_mask` / `set_sigma` / `set_fei`
> / `set_crossattn_routing`). `tests/test_adapter_protocol.py` asserts all three
> shipped networks (`LoRANetwork`, `EasyControlNetwork`, `SoftTokensNetwork`)
> satisfy the core protocol via `issubclass` (no instantiation — non-data
> protocols check method presence on the class, so no live DiT needed), that
> only the LoRA family satisfies the routing sub-protocol, and the import-
> boundary invariant (`library/inference/` + `anima_lora/` import nothing from
> `library/training/` or `train`) via an AST scan. The de facto interface
> chose `prepare_optimizer_params_with_multiple_te_lrs` (not the plain
> `prepare_optimizer_params`, which exists only on the method networks) as the
> always-present optimizer entry — that's the one `train.py` falls back to.

The trainer/inference-facing surface already exists informally:
`apply_to` / `load_weights` / `prepare_optimizer_params*` /
`set_timestep_mask` / `set_sigma` / `set_fei` / `set_crossattn_routing`, probed via
`hasattr` (`router_conditioning.py:39-50`, `network.py:1464`). Write it down as a
`typing.Protocol` (e.g. `networks/protocol.py::AdapterNetwork`, with the optional
per-step setters as a sub-protocol), and add two contract tests:

1. `LoRANetwork`, `EasyControlNetwork`, `SoftTokensNetwork` (via
   `methods/base.py::AdapterNetworkBase`) satisfy the core protocol —
   `isinstance` checks with `@runtime_checkable`, or plain attribute asserts.
2. Boundary test: `library/inference/` and `anima_lora/` import nothing from
   `library/training/` / `train.py` (today this is true by accident — zero direct
   imports either way; make it an invariant so the inference/training split stays
   extractable for free).

The `hasattr` probes in `apply_router_conditioning` stay as-is (they're the
duck-typing *consumers*); the Protocol documents what they're probing for.

### Out of scope for Part B

- `distill_save_state_dict` / `build_moe_state_dict` — per-variant by design
  (Cayley vs sqrt-split vs per-pool MoE layouts share little; ~200 dup lines not
  worth the abstraction).
- Unifying `methods/base.py::AdapterNetworkBase` with the `LoRANetwork` hierarchy —
  different injection tiers (Block-level vs Linear-level); B3's Protocol is the
  right amount of unification.
- Any behavioral change. If a step can't be shown bit-exact, it doesn't ship.

### Verification (gates every B-phase PR)

This is a pure refactor — numerically inert — so per CONTRIBUTING it needs the
invariant test but not a quality bench:

1. **Bit-exact forward equivalence**, per variant, train + eval mode: fixed-seed
   random Linear + input, instantiate the module at the pre-refactor commit and at
   HEAD, assert `torch.equal` on outputs — with and without: timestep mask bound,
   channel_scale, dropout off (dropout paths compared under fixed `torch.manual_seed`).
   Lives in `tests/test_lora_module_equivalence.py`; the pre-refactor reference is
   captured as golden tensors checked in under `tests/golden/` (small: r≤8, dim≤64).
2. **State-dict key equality** per variant (no save/load format drift) — keys +
   shapes vs golden lists. The save pipeline (`lora_save.py`) is untouched, this
   guards accidents.
3. Existing suites: `tests/test_lora_dtype_policy.py` (extended to assert the
   scaffold's `work` policy on every variant in one parametrized pass — this is the
   payoff test that would have caught the OrthoInit miss), config tests,
   `make test-unit`.
4. **Compile smoke**: `examples/07` (Part A) with `--compile` on the refactored
   modules — exercises compile-after-apply + the guard-free mask path under Dynamo.
5. Grep-level: no remaining per-variant copies of the dtype comment block.

### Sequencing

| Phase | Content | Size | Risk |
|---|---|---|---|
| A ✅ | `examples/07` + README/doc links | ~150 lines, additive | none (**shipped 2026-06-11**) |
| B0 ✅ | Golden-tensor equivalence harness (full variant matrix, captured against HEAD *before* touching modules) | `tests/test_lora_module_equivalence.py` + `tests/golden/*.pt` | none (**shipped 2026-06-11**) |
| B1 ✅ | Forward scaffold on `BaseLoRAModule`; migrated lora + ortho_init + step_expert (the `org_forwarded.dtype` two-GEMM family). Cayley/MoE forwards stay bespoke — see B1 status note | −~80 net in modules, +75 base scaffold | low — **shipped 2026-06-11**, golden-equivalent |
| B2 ✅ | `RouterStateMixin` (6-method surface + `_register_router_io_buffers`); migrated Hydra / OrthoHydra / StackedExperts, buffer-presence-guarded | −~50 net in modules, +~85 mixin | low — **shipped 2026-06-11**, golden-equivalent |
| B3 ✅ | `AdapterNetwork` + `RouterConditionableNetwork` protocols + the two contract tests | +~80 protocol, +~110 test | none (**shipped 2026-06-11**) |

Net effect: `lora_modules/` shrinks ~600–800 lines, the dtype policy and the
T-LoRA mask become single-point definitions, and the next "fix it in seven places"
bug class is closed. EasyControl's cond-LoRA dtype policy is *not* migrated here
(different package, different forward shape) but B1 makes the correct policy
importable, which is the prerequisite for fixing that latent risk in a follow-up.
