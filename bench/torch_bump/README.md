# bench/torch_bump — torch version A/B on a real compiled LoRA run

Kill-gate bench for `docs/proposal/torch214.md`: does bumping the pinned torch
move step time, first-step (load + compile) wall, or peak VRAM on the shipped
`train.py --method lora` path, on the training box (RTX 5070 Ti, sm_120)?

## Scripts

| Script | What |
|---|---|
| `run_bench.py submit` | Enqueues one short compiled LoRA run per arm through the daemon (`VER:MODE[:ENV]` arms; `VER` maps to a venv, `MODE` is `--compile_inductor_mode`). Cold private Inductor/Triton cache per arm; a repeated arm reuses it (warm-cache load) and gives the in-batch noise floor. Writes `manifest.json`. |
| `run_bench.py analyze <run_dir>` | Waits on the daemon, digests each arm's `progress.jsonl` + stdout into `result.json` / `summary.md`. |
| `vram_shim.py` | Launch shim: re-execs under the arm's interpreter (the daemon always launches with `.venv`) and samples `nvidia-smi` peak memory, since `train.py` logs no VRAM number. |
| `nvgemm_smoke.py` | Standalone probe: does Inductor offer / select the 2.14 NVGEMM (cutlass.operators) backend on this GPU for LoRA-shaped bf16 GEMMs? Run with `TORCH_LOGS=autotuning`. |

The scratch venv is built without touching `pyproject.toml` / `uv.lock`:
`uv export --frozen` → strip the torch lines → `uv pip install --no-deps` into
`.venv-t214` → `uv pip install torch==2.14.0+cu132 torchvision==0.29.0+cu132`
from the cu132 index (pulls cuDNN 9.24 / NCCL 2.30 / triton 3.8) → the
`torch2.14` flash-attn prebuild → `uv pip install --no-deps -e .`.

## Results

See `results/<ts>-<label>/summary.md`; the verdict is `docs/findings/torch214_no_win_on_sm120.md` (per-arm compile caches were deleted after the run).
