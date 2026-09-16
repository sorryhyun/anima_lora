#!/usr/bin/env bash
# Nsight Systems timeline pass for training-loop bottleneck inspection.
#
# What this answers (in order of usefulness):
#   1. Where does wall-clock time go per step? — forward vs backward vs
#      optimizer NVTX ranges are emitted by library/training/loop.py.
#   2. Is the GPU busy or starved? — look for gaps between kernels on the
#      CUDA HW row; gaps == CPU-bound (dataloader, Python, host syncs).
#   3. Which kernels/categories dominate? — `nsys stats` summary printed
#      after the run; full per-kernel ranking lives in the .nsys-rep.
#   4. Are graphs actually capturing? — cudaGraphLaunch rows should appear
#      under production compile (reduce-overhead). Their absence means
#      cudagraph_trees fell back to eager.
#
# For per-kernel comp-vs-mem (SOL %, memory workload) use ncu instead
# (memory `project_attention_compute_bound`: attention is 86-89% SM SOL on
# this box).
#
# CUDA Graphs stay on: nsys traces through cudaGraphLaunch, so the production
# reduce-overhead path is what gets profiled.

set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${NSYS_OUT:-output/nsys/profile}"
mkdir -p "$(dirname "$OUT")"

# Profile window (steps 3-7: past step-0 warmup). Widen it for tail-step
# events such as a saver firing on step N.
PROFILE_START="${PROFILE_START:-3}"
PROFILE_END="${PROFILE_END:-7}"

# Tracing categories. Default: cuda,nvtx,osrt.
#   cuda    — kernels, memcpy, cudaGraphLaunch, runtime API
#   nvtx    — forward/backward/optimizer ranges from loop.py
#   osrt    — pthread/syscall waits (Python GIL stalls, dataloader file I/O)
# Opt-in:
#   cudnn   — conv/norm dispatch boundaries
#   cublas  — GEMM dispatch boundaries (matches kernels back to their call)
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,}"

# CPU IP/backtrace sampling. Off by default; enable when Python is the suspect.
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"

# cuda-memory-usage events. Off by default (they fatten the .nsys-rep); enable
# for alloc/free patterns.
NSYS_CUDA_MEM="${NSYS_CUDA_MEM:-false}"

# Symbol resolution. Off by default: resolving at trace-end stalls for a long
# time fetching symbol files. Mangled kernel names suffice to rank kernels in
# `nsys stats`; enable for readable Python frames or libstdc++ syscall names.
# --cudabacktrace=none captures no backtraces, so there is nothing to resolve.
NSYS_RESOLVE_SYMBOLS="${NSYS_RESOLVE_SYMBOLS:-false}"
NSYS_CUDABACKTRACE="${NSYS_CUDABACKTRACE:-none}"

METHOD="${METHOD:-chimera}"
PRESET="${PRESET:-default}"

echo "[nsys] export -> ${OUT}.nsys-rep"
echo "[nsys] step ${PROFILE_START}-${PROFILE_END}, trace=${NSYS_TRACE}, sample=${NSYS_SAMPLE}, cuda-mem=${NSYS_CUDA_MEM}"
echo "[nsys] resolve-symbols=${NSYS_RESOLVE_SYMBOLS}, cudabacktrace=${NSYS_CUDABACKTRACE}"
echo "[nsys] method=${METHOD} preset=${PRESET}"

# --capture-range=cudaProfilerApi + --capture-range-end=stop pairs with
# loop.py's torch.cuda.profiler.start()/stop() so the .nsys-rep only
# contains the profile window — not the cold-start text-encoder caching,
# VAE caching, compile warmup, etc. The profiler.stop() at PROFILE_END
# also ends the capture (capture-range-end=stop), so no --duration is needed.
nsys profile \
    --output "$OUT" \
    --force-overwrite true \
    --trace "$NSYS_TRACE" \
    --sample "$NSYS_SAMPLE" \
    --cuda-memory-usage "$NSYS_CUDA_MEM" \
    --cudabacktrace "$NSYS_CUDABACKTRACE" \
    --resolve-symbols "$NSYS_RESOLVE_SYMBOLS" \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    python -m accelerate.commands.accelerate_cli launch \
        --num_cpu_threads_per_process 3 \
        --mixed_precision bf16 \
        train.py \
        --method "$METHOD" --preset "$PRESET" \
        --profile_steps "${PROFILE_START}-${PROFILE_END}" \
        --max_train_steps "$((PROFILE_END + 2))"

echo
echo "[nsys] === summary (nsys stats) ==="
echo "[nsys] open ${OUT}.nsys-rep in the Nsight Systems GUI for the full timeline."
echo

# Terminal rankings of the dominant kernels and NVTX ranges; per-call detail
# is in the GUI.
nsys stats \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report nvtx_sum \
    --format column \
    "${OUT}.nsys-rep" || echo "[nsys] stats failed (open the .nsys-rep in the GUI instead)"
