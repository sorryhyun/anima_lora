#!/usr/bin/env bash
set -euo pipefail

# === Model paths ===
MODELS="/home/sorryhyun/comfy/ComfyUI/models"
DIT_PATH="${MODELS}/diffusion_models/anima-preview.safetensors"
VAE_PATH="${MODELS}/vae/qwen_image_vae.safetensors"
QWEN3_PATH="${MODELS}/text_encoders/qwen_3_06b_base.safetensors"

# === Training config ===
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_CONFIG="${DATASET_CONFIG:-${PROJECT_DIR}/configs/example_lora.toml}"
OUTPUT_DIR="${PROJECT_DIR}/output"
OUTPUT_NAME="anima_postfix"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

accelerate launch --mixed_precision bf16 \
  --num_cpu_threads_per_process 2 train.py \
  --pretrained_model_name_or_path="${DIT_PATH}" \
  --vae="${VAE_PATH}" \
  --qwen3="${QWEN3_PATH}" \
  --dataset_config="${DATASET_CONFIG}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_name="${OUTPUT_NAME}" \
  --save_model_as=safetensors \
  --network_module=networks.postfix_anima \
  --network_train_unet_only \
  --network_dim=16 \
  --learning_rate=2e-3 \
  --optimizer_type="AdamW8bit" \
  --timestep_sampling="sigmoid" \
  --discrete_flow_shift=1.0 \
  --max_train_epochs=8 \
  --save_every_n_epochs=4 \
  --mixed_precision="bf16" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0.1 \
  --cache_latents \
  --caption_shuffle_variants=8 \
  --cache_text_encoder_outputs \
  --blocks_to_swap=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --attn_mode=flash \
  --validation_split=0.1 \
  --validation_seed=42 \
  --validate_every_n_epochs=1
