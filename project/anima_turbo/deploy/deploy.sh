#!/usr/bin/env bash
# Release a merged14-teacher turbo student in the three forms of deploy/README.md.
#
#   project/anima_turbo/deploy/deploy.sh <turbo_ema.safetensors> <name> [lora|bake|gptq|all]
#
#   lora  one LoRA for plain anima-base-v1.0: soup ΔW14 ++ turbo ΔT, exact concat
#         -> output/ckpt/release/<name>.safetensors        (DTYPE=fp32 for the bit-exact file)
#   bake  base + ΔW14 + ΔT as a full bf16 DiT, merged on the CPU (BAKE_DEVICE=cuda to change)
#         -> models/diffusion_models/<name>.safetensors
#   gptq  ConvRot int8 GPTQ of the baked DiT, calibrated for 4 steps / cfg 1, queued on the daemon
#         -> models/diffusion_models/<name>_int8_gptq.safetensors (+ .report.json)
#   all   lora, bake, gptq in order (the default)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n 2,12p "$0"
    exit 1
fi
TURBO=$1
NAME=$2
STEP=${3:-all}

cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
SOUP=output/ckpt/anima_merged14.safetensors
TEACHER_DIT=models/diffusion_models/anima_merged14.safetensors
BAKED=models/diffusion_models/${NAME}.safetensors
ONE_LORA=output/ckpt/release/${NAME}.safetensors

[[ -f $TURBO ]] || { echo "no turbo LoRA at $TURBO" >&2; exit 1; }

step_lora() {
    mkdir -p "$(dirname "$ONE_LORA")"
    # --normalize off / --adaln sum keep the concat exact; the defaults (global / average)
    # put a relative error of 0.35 / 0.50 on the merged delta.
    $PY scripts/toolkits/merge_loras.py "$SOUP" "$TURBO" \
        --normalize off --adaln sum --dtype "${DTYPE:-bf16}" --out "$ONE_LORA"
}

step_bake() {
    $PY scripts/toolkits/merge_to_dit.py --dit "$TEACHER_DIT" --adapter "$TURBO" \
        --multiplier 1.0 --dtype bf16 --device "${BAKE_DEVICE:-cpu}" --out "$BAKED"
}

step_gptq() {
    [[ -f $BAKED ]] || { echo "no baked DiT at $BAKED; run the bake step first" >&2; exit 1; }
    # The 4-step grid at shift 3 (σ 1 / 0.9 / 0.75 / 0.5), caption pass only (cfg 1 has no
    # uncond). per_artist 4 keeps the sample count of the two-pass teacher calibration (368).
    make daemon-run ARGS="--queue --stall-timeout 1800 project/finished/int8_train/src/dit_gptq.py \
        --dit $BAKED --steps 4 --flow_shift 3.0 --no_negative --per_artist 4"
    echo "wait with: make daemon-wait JOB=<id above>"
}

case $STEP in
    lora) step_lora ;;
    bake) step_bake ;;
    gptq) step_gptq ;;
    all) step_lora; step_bake; step_gptq ;;
    *) echo "unknown step: $STEP" >&2; exit 1 ;;
esac
