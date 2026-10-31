# anima_stream

Anima's fp32 stream on the generation path. Doc: `docs/anima/optimizations/fp32_stream.md`.

- `run_bench.py`: prediction error at fixed noised images, final-latent distance over full
  trajectories through `generate_body`, ms/step and peak, each against a true fp32 forward.
- `comfy_check.py`: one step through ComfyUI's `apply_model` against the library
  (`--mode numeric`), the same on the INT8 file (`--mode int8`), and `common_ksampler` off and on
  with the dtypes that reach the GEMMs and the head (`--mode compat --cases ...`).

```bash
make daemon-run ARGS="--stall-timeout 600 bench/anima_stream/run_bench.py"
make daemon-run ARGS="--stall-timeout 600 bench/anima_stream/comfy_check.py --cases lora"
```

Good: fp32 arms well under bf16 on prediction, ComfyUI fp32 within about 0.3 of the library's
no-RoPE arm. `comfy_check.py` passes `patcher.wrappers` itself: ComfyUI merges them into
`transformer_options` only when sampling, so a bare `apply_model` runs unpatched.

Observed: `results/20261031-2154-comfy` (numeric, `lora`), `results/20261031-2158-comfy-int8`
(int8, `int8_lora`). `results/` is gitignored. `run_bench.py` has no
full run: only a 2-image, 8-step smoke, whose numbers are in the doc.
