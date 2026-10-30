# Tasks (2026-10-30)

## 1. Self-attention with a bf16 forward and an int8 backward

### Question

Keep the self-attention forward in 16-bit and run only the backward GEMMs in int8. The
completed attention work ([reports.md](reports.md)) put the gradient error of int8 QKᵀ on
the forward. Does a 16-bit forward leave room for an int8 backward?

### Why it is open

Every int8-backward arm so far ran on the int8 forward
([reports/attn_bwd.md](reports/attn_bwd.md)). The only 16-bit-forward arms were FA2 and
`tri_fp16`, both with a 16-bit backward. The kernel already covers the missing cell:
`attn_bwd(..., saved=None)` recomputes P in bf16, and `dp` / `dv` / `dqk` each take
`int8` on their own. Only the `ARMS` table in `src/attn_bwd_bench.py` lacks the arms.

### Prediction

Each backward GEMM's own error, estimated by subtracting the int8 forward's share in
quadrature between arm pairs of [reports/attn_bwd.md](reports/attn_bwd.md). The
estimates are in FA2 units, as that report gives them; `tri_fp16` sits at 0.70–0.95x
FA2's error, so in its units they are 1.05–1.4x larger.

| int8 GEMM | arm pair | estimate (FA2 units) |
|---|---|---|
| dV = Pᵀ·dO | `int8qk_fp16` → `int8_dp_fp16` | 4–8x |
| dQ = dS·K, dK = dSᵀ·Q | `int8qk_fp16acc` → `int8_dp_fp16acc` | dQ 3.8–4.4x, dK 2.4–3.3x |
| dP = dO·Vᵀ | `int8_dp_fp16` → `int8_all` | dQ ~3x, dK ~2.3x |

Each one alone is expected over the 2x criterion. int8 dV fails on P's rounding itself:
per tile row, the small entries round to 0, and over 4k queries that tail carries real
mass.

Speed is not expected to be the obstacle: the all-int8 backward is 1.66x `tri_fp16`.
Within that, dV, dK, dQ in int8 saved nothing (4.236 → 4.229 ms); the gain comes from dP
and the fp16-accumulate modes.

### Steps

1. **Arms.** Add to `ARMS` in `src/attn_bwd_bench.py`, with `int8 forward = False`:
   - `bf16_dv8`: `("fp16", "int8", "fp16")`
   - `bf16_dqk8`: `("fp16", "fp16", "int8")`
   - `bf16_dp8`: `("int8", "fp16", "fp16")`
   - `bf16_dpacc`: `("fp16acc", "fp16acc", "fp16")`, the no-int8 speed reference
   - `bf16_all8`: `("int8", "int8", "int8")`
2. **Baseline in the summary.** The bench prints error and time as multiples of FA2.
   Make `tri_fp16` the unit for both; keep FA2's absolute numbers in the JSON.
3. **Run** on the capture (`output/int8_train/attn_capture.pt`), a few minutes:
   `make daemon-run ARGS="project/int8_train/src/attn_bwd_bench.py"`. Report to
   `reports/attn_bwd_bf16fwd.md`, raw JSON to `assets/`, one entry in `reports.md`.
4. **Register check.** Read `n_regs` / `n_spills` of `_bwd_dkdv_kernel` and
   `_bwd_dq_kernel` in fp16 and int8 modes. This tests the claim that the int32 tiles and
   the per-iteration dequant cancel the int8 gain for dV, dK, dQ. Compile only.

### Criterion

dQ, dK, dV each at most 2x `tri_fp16`'s distance from fp32, median per σ, and at least
1.3x `tri_fp16`'s backward time.
