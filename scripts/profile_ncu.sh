#!/usr/bin/env bash
# Nsight Compute pass on the kernels that dominate plain-LoRA training:
#   - cutlass_80 ..._tn_align8   (forward GEMM)
#   - cutlass_80 ..._nn_align8   (frozen-base dgrad — the suspected bottleneck)
#   - cutlass_80 ..._nt_align8   (LoRA wgrad)
#   - flash_bwd_dq_dk_dv         (FA2 backward)
#
# Goal: answer "comp-bound or mem-bound?" via the SOL + Memory Workload sections.
# If achieved tensor-core throughput is high → no FP8 win. If DRAM SOL is high
# while tensor SOL is low → FP8 (smaller bytes) helps the dgrad path.
#
# Why we disable CUDA Graphs for this run: ncu's --replay-mode=kernel can't
# replay individual nodes inside a captured graph (would need
# --replay-mode=application, which re-runs the entire training per kernel —
# many minutes per kernel). The cutlass + flash kernels we care about are
# bit-identical with or without graphs; only launch overhead differs.

set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${NCU_OUT:-output/ncu/profile}"
mkdir -p "$(dirname "$OUT")"

# Kernel name regex. Override via NCU_KERNELS=... if you want a narrower set.
#
# Default matches the matmul kernels cuBLAS actually picks on this box (RTX
# 5090, SM_120) plus the FA2 backward kernel:
#   - nvjet_sm120_tst_mma   Blackwell-native TMA matmul (default cuBLAS pick
#                           in non-CUDAGraph mode; e.g. nvjet_sm120_tst_mma_
#                           64x64x128_3_32x32x128_tmaAB_bz_NTNN)
#   - s16816gemm            Ampere 16x8x16 mma (what cuBLAS picks at graph
#                           capture time under reduce-overhead — production)
#   - s1688gemm             Turing 16x8x8 mma (small auxiliary matmuls)
#   - flash_bwd             FA2 backward variants (dq_dk_dv, dot_do_o, etc.)
# KEEP THIS NARROW: each matching kernel costs ~1-2s of replay; matching
# elementwise prep kernels burns the profiler window before the heavy DiT
# forward fires.
KERNEL_REGEX="${NCU_KERNELS:-regex:nvjet_sm120|s16816gemm|s1688gemm|flash_bwd|sm80_xmma_gemm|ampere_.*gemm}"

# Step where cuProfilerStart fires (single step is plenty — each kernel is
# replayed many times by ncu to gather metrics).
PROFILE_START="${PROFILE_START:-3}"
PROFILE_END="${PROFILE_END:-3}"

# Skip the first N matching launches inside the profiled region to land on
# steady-state shapes (avoids whatever warmup happens at step boundary).
LAUNCH_SKIP="${LAUNCH_SKIP:-30}"
# Capture this many matching kernels total — picks a mix of self-attn (long K)
# and cross-attn (short K) flash bwd, plus a few cutlass GEMM shapes.
LAUNCH_COUNT="${LAUNCH_COUNT:-12}"

# 'detailed' = SOL + MemoryWorkloadAnalysis + Occupancy + LaunchStats +
# Scheduler + WarpStateStats. Right balance for the comp-vs-mem question;
# 'full' adds source attribution and is ~10x slower.
NCU_SET="${NCU_SET:-detailed}"

echo "[ncu] export -> ${OUT}.ncu-rep"
echo "[ncu] kernels: ${KERNEL_REGEX}"
echo "[ncu] step ${PROFILE_START}-${PROFILE_END}, skip ${LAUNCH_SKIP}, count ${LAUNCH_COUNT}, set ${NCU_SET}"

exec ncu \
    --target-processes all \
    --profile-from-start no \
    --replay-mode kernel \
    --kernel-name "$KERNEL_REGEX" \
    --launch-skip "$LAUNCH_SKIP" \
    --launch-count "$LAUNCH_COUNT" \
    --set "$NCU_SET" \
    --import-source no \
    --cache-control all \
    --export "$OUT" \
    --force-overwrite \
    python -m accelerate.commands.accelerate_cli launch \
        --num_cpu_threads_per_process 3 \
        --mixed_precision bf16 \
        train.py \
        --method lora --preset default \
        --profile_steps "${PROFILE_START}-${PROFILE_END}" \
        --max_train_steps "$((PROFILE_END + 2))" \
        --compile_inductor_mode "${INDUCTOR_MODE:-default}"
# Inductor mode default is 'default': matmuls fall through to cuBLAS and
# fire the same cutlass_80 s16816gemm kernels nsys captured under production
# (compile_mode=full + reduce-overhead). 'max-autotune-no-cudagraphs' was
# tried first but its autotuner picks Triton template GEMMs (triton_tem_*)
# that don't match the s16816gemm regex. 'reduce-overhead' itself is out
# because ncu's --replay-mode=kernel can't replay nodes inside a captured
# CUDAGraph; 'default' is the closest non-graph mode that keeps cuBLAS.
#
# Sampling/saving cadence is left at TOML defaults (save_every_n_epochs=12 in
# lora.toml; no save_every_n_steps set). With max_train_steps≈5 nothing fires.
# Don't pass --save_every_n_steps 0 — the saver's modulo check doesn't null it
# (cli_args only does that for sample_*), so 0 → ZeroDivisionError.
