# Row gradient on the bf16 stream against fp32 — no bias above the control (2026-11-06)

Question (user, 11-05): would the vocab rows train better on the fp32 stream? Every CJK
run so far trained on the bf16 stream (branches from before `e370a23c`, 10-31).

## Setup

`probes/row_grad_bias.py`, daemon, 19.5 min. `hira_cold_hp` at `s4050` (81 hiragana rows,
mid-run), the trainer's box-share FM loss, gradient on `rows.delta.raw`. Two batches of
the run's own data (scene 640×448, grid 416×624, 4 items each) × σ 0.3125 / 0.625 (exact
in bf16), 128 paired draws of ε per cell. Each arm is paired against `stream` on the same
draws, `bias² = ‖ē‖² − s²/D` (int8_train's `anima_grad_bias.py` estimator), % of
‖mean g_stream‖:

- `bf16`: the trainer of record (bf16 noisy and σ, bf16 autocast, compiled, FA2);
- `stream`: `stream_fp32_`, compiled;
- `stream_jit`: `stream` + 2^-9 relative noise on `x` at each block entry: **the control**,
  a rounding-sized perturbation with no bias;
- `stream_adapter`: `stream` with `llm_adapter` in fp32. This also keeps the pack rows
  and the ExtDelta delta in fp32: they round to bf16 at the embed output on every other arm.

Results: `results/20261106-1708-row_grad_bias/result.json`.

## Result

| σ (pooled) | ε spread, one draw | arm | bias | raw bias² (%²) | se | single |
|---|---|---|---|---|---|---|
| 0.625 | 223 | bf16 | 2.8 | +8.0 | 8.4 | 95 |
| | | stream_jit (control) | 2.9 | +8.3 | 8.1 | 92 |
| | | stream_adapter | 0 | −3.8 | 7.5 | 85 |
| 0.3125 | 322 | bf16 | 0 | −20.5 | 19.2 | 217 |
| | | stream_jit (control) | 0 | −61.2 | 19.5 | 220 |
| | | stream_adapter | 0 | −42.9 | 24.8 | 281 |

Per cell, the cosine of bf16's mean gradient with stream's equals the control's
(scene σ 0.625: 0.9988 against 0.9984; grid σ 0.625: 0.9936 against 0.9940).

## What it says

1. **bf16 reads as the control in every cell.** Its single-draw distance (95 / 217)
   and its bias are the control's. To the row gradient, the bf16 stream is one more
   rounding-sized perturbation, not a shift.
2. **The row gradient is chaotic.** A 2^-9 nudge at the block inputs moves one draw's
   row gradient by about as much as ε does (92 against 223 at σ 0.625). Training noise
   swamps any numeric difference per step. A bf16 run and an fp32-stream run should
   differ as two seeds do.
3. **Resolution:** at σ 0.625 a bias under about 10% cannot be told from zero (se 8). σ 0.3125 is
   not readable (se 19). LoRA's late-block 6–8% (`anima_stream_attrib.md`) would not
   show at σ 0.625. Reaching se ≈ 2% takes about 16× the draws, hour-scale.
4. **The fp32 adapter shows nothing either.** That includes the bf16 rounding of row
   values at the embed output.
5. The 16 ref draws (whole model fp32) do not rank the arms. Stream sits closer to
   fp32 on scene σ 0.625 (19 against bf16's 36) and farther on grid σ 0.625 (222 against
   100): the chaos again.

## The fp32 adapter arm, checked (`probes/adapter_dtype_check.py`)

Forward only, the scene batch at σ 0.625, on the stream:

- The pack table loads F32 (`load_ext_assets` → `materialize` keeps the stored dtype)
  and rounds only at the embed output, to that output's dtype (`vocab_pack.py:384`).
- Under the probe's wrapper, the embed output, all 61 adapter linears and the context
  reaching block 0's `cross_attn.kv_proj` are fp32. Autocast rounds the context at that
  GEMM. On bf16 they are all bf16. The arm is what it claims to be.
- On the valid tokens, the adapter's output on bf16 sits 1.4 % from fp32. Over all positions
  it is 75 %: the max-padded positions hold 96 % of the output norm before the DiT zeroes
  them.
- A nudge of 2^-10 relative on one trained row (ext 31), valid tokens; a rerun with no
  nudge moves nothing on either:

  | adapter | its token | every other token |
  |---|---|---|
  | fp32 | 8.5e-4 | 2.9e-5 |
  | bf16 | 1.3e-2 | 9.9e-3 |

  On fp32 the response is linear and local (≈ 2^-10 on the row's own token). On bf16 a change
  below the resolution re-rolls the rounding of the whole caption by about 1 %, the size
  of bf16's own error. In the forward, a row's small changes sit under that noise. The
  gradient, linearized through the same ops, shows no bias above (§ Result).

## The adapter at inference (`probes/adapter_v_error.py`)

`library.anima.stream.adapter_fp32_`: the adapter computed in fp32 on its kept bf16
weights (opt-in, no caller). Forward only, the same rows, 2 scene and 2 grid batches × 2 ε
seeds. The adapter runs in the forward from the Qwen states and T5 ids. The Anima LoRA
TE cache stores the adapter's bf16 *output* (`crossattn_emb_v*`, BF16 [512, 1024]), so
`_archive/bench/anima_stream` never ran it. % from the whole-fp32 ref: v over the latent | v in the
glyph boxes | adapter output on the valid tokens.

| | σ | bf16 | stream | stream + adapter fp32 |
|---|---|---|---|---|
| grid | 0.1 | 1.80 \| 2.29 \| 1.53 | 0.53 \| 0.82 \| 1.53 | 0.53 \| 0.82 \| 0.00 |
| | 0.5 | 0.82 \| 1.12 \| 1.53 | 0.20 \| 0.50 \| 1.53 | 0.20 \| 0.51 \| 0.00 |
| | 0.9 | 1.35 \| 1.25 \| 1.53 | 0.19 \| 0.25 \| 1.53 | 0.17 \| 0.21 \| 0.00 |
| scene | 0.1 | 1.97 \| 2.22 \| 1.37 | 0.60 \| 0.70 \| 1.37 | 0.61 \| 0.71 \| 0.00 |
| | 0.5 | 0.84 \| 0.90 \| 1.37 | 0.24 \| 0.34 \| 1.37 | 0.24 \| 0.32 \| 0.00 |
| | 0.9 | 1.09 \| 1.01 \| 1.37 | 0.22 \| 0.24 \| 1.37 | 0.18 \| 0.19 \| 0.00 |

- The adapter's 1.4–1.5 % context error reaches v at under 0.05 points. At σ 0.1 / 0.5
  it does not show. At σ 0.9 the fp32 adapter takes 0.22 → 0.18 (scene) and, in the
  boxes, 0.24 → 0.19 / 0.25 → 0.21. The context matters most at high σ, and the
  `kv_proj` GEMM rounds it to bf16 on every arm.
- With bf16 weights the adapter output equals the ref's (0.00): bf16 weights cast to
  fp32 exactly.
- Renders, `probes/adapter_render.py` (`results/20261106-1759-adapter_render/`):
  shipped `jp_v1`, no delta, the identity grid's 13 words + 5 kanji lines × p00 / p01 ×
  seeds 0 / 1, 512², compiled, 72 pairs. Mean |Δpx| 3.9 / 255 (median 3.0, max 13.9),
  about the 3.8 session noise of § 1 in `status.md`. Composition, layout and type style
  are the same in every pair. The drawn string differs in about 6 / 72, both ways:
  カメラ p00 s0 オフメラ → カフメラ and こうえん p01 s0 こううこえん → こうええん
  (fp32 nearer), たいせつ p01 s0 たないせつ → たな い?つ (fp32 farther), てつだう p00
  s0 てつつだEう → て っ?だモう (both wrong). The strings correct on one arm are correct
  on the other.

## Verdict

Retraining a pack on the fp32 stream is not expected to change the rows beyond seed
noise. A paired training A/B would need several seeds an arm to read anything. Not
tested: other runs or saves, warm dialogue rows (`sent_kanji_pres`), σ outside the two
points.

Note: on main, `gen_args` → `load_dit_model` now loads the fp32 stream (inference's
`--stream` default), so `run.py <run> train` on main trains on it, with bf16 noisy and σ.
