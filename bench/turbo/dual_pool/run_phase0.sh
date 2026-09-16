#!/usr/bin/env bash
# Dual-pool gradient routing — Phase 0 A/B driver.
# docs/proposal/turbo_dual_pool_grad_routing.md
#
# Queues both arms on the daemon (serial, survives the terminal) and returns.
# This does NOT poll — check progress with `make run-status RUN=<output_name>`
# or `make daemon-jobs`. Renders/eval are a separate step (see README.md).
#
#   bash bench/turbo/dual_pool/run_phase0.sh
#
# One variable: parameter routing. Everything else (warm start, GAN,
# dynamic_schedule, div_weight, seed, data) is fixed across the two configs.
set -euo pipefail
cd "$(dirname "$0")/../../.."

DIR="bench/turbo/dual_pool"

echo "Queuing Arm A (dual 32/32) ..."
make turbo ARGS="--config ${DIR}/arm_A_dual_32_32.toml --queue"

echo "Queuing Arm B (single r=64 baseline) ..."
make turbo ARGS="--config ${DIR}/arm_B_single_r64.toml --queue"

echo
echo "Both arms queued. Watch:  make daemon-jobs"
echo "Per-run:                  make run-status RUN=anima_superturbo_dualpool_A_v1"
echo "                          make run-status RUN=anima_superturbo_dualpool_B_v1"
echo "Then render + eval per bench/turbo/dual_pool/README.md."
