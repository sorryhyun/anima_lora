LORA_DIR := ../comfy/ComfyUI/models/loras

.PHONY: lora dora tdora tlora sync step test mask mask-sam mask-mit mask-clean preprocess download-models download-sam3 download-mit

lora:
	accelerate launch --num_cpu_threads_per_process 3 --mixed_precision bf16 \
		train.py --config_file configs/example_lora.toml

dora:
	accelerate launch --num_cpu_threads_per_process 3 --mixed_precision bf16 \
		train.py --config_file configs/training_config_dora.toml \
		--network_args use_dora=true

tdora:
	accelerate launch --num_cpu_threads_per_process 3 --mixed_precision bf16 \
		train.py --config_file configs/training_config_doratimestep.toml

tlora:
	accelerate launch --num_cpu_threads_per_process 3 --mixed_precision bf16 \
		train.py --config_file configs/training_config.toml \
		--network_args use_ortho=true sig_type=last ortho_reg_weight=0.01 \
		use_timestep_mask=true min_rank=1 alpha_rank_scale=1.0

sync:
	cp output/*.safetensors $(LORA_DIR)/

test:
	python inference.py \
		--dit models/diffusion_models/anima-preview2.safetensors \
		--text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
		--vae models/vae/qwen_image_vae.safetensors \
		--vae_chunk_size 64 --vae_disable_cache \
		--attn_mode flash \
		--lora_weight $$(ls -t output/*.safetensors | head -1) \
		--lora_multiplier 1.0 \
		--prompt "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top and denim shorts is standing outdoors. She's holding a rectangular sign out in front of her that reads \"ANIMA\". She's looking at the viewer with a smile. The background features some trees and blue sky with clouds." \
		--negative_prompt "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia" \
		--image_size 1024 1024 \
		--infer_steps 30 \
		--flow_shift 1.0 \
		--sampler er_sde \
		--guidance_scale 4.0 \
		--seed 42 \
		--save_path test_output

step:
	python graft_step.py

preprocess:
	python scripts/post_images.py \
		--src image_dataset \
		--dst post_image_dataset

# --- Model downloads ---

download-sam3:
	@mkdir -p models/sam3
	huggingface-cli download facebook/sam3 --local-dir models/sam3

download-mit:
	@mkdir -p models/mit
	curl -L -o models/mit/comictextdetector.pt \
		https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt
	curl -L -o models/mit/comictextdetector.pt.onnx \
		https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx

download-models: download-sam3 download-mit

# --- Masking ---

mask-sam:
	python scripts/generate_masks.py \
		--config configs/sam_mask.yaml \
		--image-dir post_image_dataset \
		--mask-dir masks_sam \
		--checkpoint models/sam3/sam3.pt \
		--batch-size 2

mask-mit:
	python scripts/generate_masks_mit.py \
		--image-dir post_image_dataset \
		--mask-dir masks_mit \
		--model-path models/mit \
		--detect-size 1024

mask: mask-sam mask-mit
	python scripts/merge_masks.py \
		masks_sam masks_mit \
		--output-dir masks

mask-clean:
	rm -rf masks/ masks_sam/ masks_mit/
