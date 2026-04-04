# Advice on `proposal.md`

## Executive Summary

`proposal.md` points at a real opportunity, but it currently mixes:

- a valid hardware thesis,
- an optimistic implementation plan,
- and a weakly-defended accuracy story.

My recommendation is:

1. Treat native SM120 MXF4 attention as a research track, not an immediate integration plan.
2. Prioritize lower-risk wins in the current SM120 FA4 path first.
3. If pursuing MXF4, start with a narrow prototype: standalone GEMM or QK-only attention.
4. Do not make full `P -> MXF4 -> V` quantization the phase-1 plan.

In short: theoretically valuable, technically feasible as an experiment, not yet justified as the best next engineering investment in its current form.

## What Is Strong in the Proposal

### 1. The hardware thesis is real

SM120 does have native block-scaled narrow-precision MMA support. The direction is not speculative in the "hardware may not support this" sense.

That means there is a legitimate reason to explore:

- `mma.sync.aligned.kind::mxf4.block_scale`
- `e2m1` inputs
- `ue8m0` scale factors
- FP32 accumulation

This is enough to justify a prototype.

### 2. The current SM120 FA4 path is clearly not the final architecture-aware solution

The existing `flash-attention-sm120` integration is still structurally close to the SM80 path:

- `FlashAttentionForwardSm120` subclasses `FlashAttentionForwardSm80`
- `FlashAttentionBackwardSm120` subclasses `FlashAttentionBackwardSm80`
- the interface hardcodes conservative SM120 choices
- the codebase already needed architecture-specific bug fixes

So the instinct that "SM120 still has unclaimed performance" is correct.

### 3. The proposal picks the right first domain

Attention is one of the few places where:

- the shapes are regular,
- the arithmetic intensity is meaningful,
- and the repo already has a custom kernel path.

If you are going to do custom SM120 work anywhere, attention is a reasonable target.

## What Is Weak in the Proposal

### 1. The speedup claim is too aggressive

The proposal infers fused attention speedup from raw GEMM / `_scaled_mm` microbenchmarks.

That is not enough.

Those numbers do not include:

- BF16 -> MXF4 packing
- scale-factor generation
- softmax bookkeeping
- shared memory traffic for auxiliary state
- epilogue costs
- the cost of quantizing `P`

So the current "~2-4x attention speedup" and "~1.5-2x end-to-end training speedup" should be treated as aspirational, not conservative.

A more defensible initial framing would be:

- best case: large forward speedup on self-attention
- likely case: moderate forward-only speedup
- uncertain case: small end-to-end win unless backward is also improved

### 2. The main accuracy risk is underweighted

The proposal correctly identifies the hardest part:

- softmax produces `P` in FP32
- then `P` is quantized to MXF4
- then `P·V` runs in MXF4

That is the most dangerous numerical step in the whole design.

Quantizing Q/K is one thing. Quantizing normalized attention weights is much harder to defend.

Why this matters:

- `P` can be very peaky
- small changes near the top of the distribution matter
- diffusion noise does not automatically make attention errors irrelevant
- "FP32 accumulator" does not undo input quantization damage

If you want a phase-1 kernel with a chance of surviving practical validation, avoid quantizing `P` initially.

### 3. The implementation plan is too short

The proposal estimates roughly:

- 1-2 days for proof-of-concept GEMM
- 3-5 days for fused forward
- 1-2 days for integration

That would only be plausible if:

- the conversion path were trivial,
- the CUTLASS DSL path were already proven for this exact use case,
- and the existing SM120 FA4 code were already stable.

That is not the situation here.

The current FA4 fork already exposed:

- TMA path mismatches
- SM120-specific backward config gaps
- API drift issues

So the realistic timeline is "research sprint with unknowns," not "small feature branch."

### 4. The proposal assumes an easy BF16 -> MXF4 path

This is underspecified.

The key unresolved question is:

How expensive is packing BF16 tiles plus generating `ue8m0` scales at runtime, at the exact attention tiling used by Anima?

If that step is not cheap, the theoretical MMA advantage shrinks fast.

This is the first thing that should be measured, before any fused attention work.

## Best Next Direction

### Direction A: Tune the current SM120 FA4 path first

This is the highest-confidence next step.

The current SM120 forward path is conservative:

- it uses the SM80-derived implementation
- it hardcodes `num_stages=1`
- it uses simple tile choices
- it excludes more advanced execution paths

Before building an entirely new numeric path, you should answer:

- Can the current BF16 FA4 kernel get noticeably better with better tile/stage choices?
- Is the current runtime limited by architecture mismatches rather than arithmetic precision?

Suggested work:

- autotune `tile_m`, `tile_n`, `num_stages`
- benchmark self-attn and cross-attn separately
- profile forward and backward separately
- verify whether backward is the actual dominant remaining cost

This is the most likely place to find a real win with the least risk.

### Direction B: Build a standalone MXF4 GEMM or QK-only prototype

If you want the CUTLASS / SM120 research path, this is the correct phase 0.

Target:

- BF16 inputs
- runtime quantization to MXF4
- `dot_block_scaled`
- FP32 accumulation

Success criteria:

- real wall-clock speedup after quantization overhead
- acceptable error versus BF16 reference
- stable build path on your installed toolchain

Do this for QK first, not full attention.

Why:

- QK is the cleaner half of attention
- it avoids the `P` quantization problem
- it lets you validate packing + scale-factor overhead
- it will tell you quickly whether MXF4 is worth integrating at all

### Direction C: If attention is attempted, use a hybrid design first

The most defensible first fused design is:

- MXF4 for Q·K^T
- FP32 online softmax
- BF16 or FP8 for P·V

This keeps the highest-risk numerical decision out of phase 1.

That kernel is still hard, but it is much easier to justify than full FP4 attention.

It also gives you better experimental signal:

- Did native MXF4 help where it should help most?
- Did the gain survive real attention control flow?

If yes, then you can decide whether full MXF4 `P·V` is worth exploring.

### Direction D: Only pursue full MXF4 attention after a go/no-go gate

Do not proceed straight from "the hardware supports it" to "implement full fused forward."

Set explicit gates:

1. Standalone MXF4 GEMM beats BF16 enough after packing overhead.
2. QK-only prototype shows meaningful speedup.
3. Error versus BF16 is acceptable on attention-shaped inputs.
4. End-to-end training profiling says forward attention is still worth attacking.

If any of these fail, stop there.

## Suggested Rewrite of `proposal.md`

If you want to keep the proposal, I would rewrite its posture as follows.

### Replace the headline claim

Instead of:

- "targets a 2-4x attention speedup"

Use:

- "explores whether SM120 native MXF4 can provide a meaningful forward-attention speedup over the current SM80-derived path"

That is more defensible and still strong.

### Split the proposal into two tracks

Track 1:

- current SM120 FA4 tuning and stabilization

Track 2:

- native MXF4 research prototype

This avoids presenting a high-risk kernel rewrite as the only path forward.

### Make the risk table harsher

Specifically change:

- quantization risk from low -> medium/high
- implementation risk from medium -> high
- schedule confidence from implicit high -> low

### Add explicit go/no-go milestones

For example:

1. GEMM prototype
2. QK-only attention prototype
3. hybrid attention prototype
4. full MXF4 attention only if earlier gates pass

## Suggested Concrete Plan

### Phase 0: Current-path profiling

- benchmark current SM120 FA4 forward and backward separately
- sweep tile sizes and stages
- identify whether self-attn or cross-attn matters most in wall clock

### Phase 1: MXF4 microkernel

- standalone BF16 -> MXF4 -> blockscaled GEMM
- measure real throughput and conversion overhead
- compare output error to BF16

### Phase 2: QK-only prototype

- replace only the score-generation matmul
- keep the rest numerically conservative

### Phase 3: Hybrid attention

- MXF4 QK
- FP32 softmax
- non-MXF4 PV path

### Phase 4: Re-evaluate

- if the hybrid kernel is not clearly valuable, stop
- only then consider full MXF4 `P·V`

## Final Recommendation

If the goal is maximum short-term performance improvement for Anima LoRA training on SM120:

- do not start with full fused MXF4 attention
- first improve the existing SM120 FA4 path
- then prototype MXF4 narrowly

If the goal is research value and kernel exploration:

- yes, MXF4 is worth investigating
- but frame it as an experimental kernel track with explicit kill criteria

That is the version of the idea that is both technically serious and worth doing.
