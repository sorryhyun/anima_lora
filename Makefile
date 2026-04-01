LORA_DIR := ../comfy/ComfyUI/models/loras

.PHONY: lora dora tdora tlora sync step test mask mask-sam mask-mit mask-clean preprocess download-models download-anima download-sam3 download-mit

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
		--attn_mode flex \
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
		--dst post_image_dataset \
		--vae models/vae/qwen_image_vae.safetensors \
		--vae_batch_size 4 \
		--vae_chunk_size 64

# --- Model downloads ---

download-sam3:
	@mkdir -p models/sam3
	huggingface-cli download facebook/sam3 --local-dir models/sam3

download-mit:
	@mkdir -p models/mit
	huggingface-cli download a-b-c-x-y-z/Manga-Text-Segmentation-2025 \
		model.pth --local-dir models/mit

download-anima:
	@mkdir -p models/diffusion_models models/text_encoders models/vae
	huggingface-cli download circlestone-labs/Anima \
		split_files/diffusion_models/anima-preview2.safetensors \
		split_files/text_encoders/qwen_3_06b_base.safetensors \
		split_files/vae/qwen_image_vae.safetensors \
		--local-dir models --include "split_files/*"
	@# Move files from split_files/ subdirs into models/
	@mv models/split_files/diffusion_models/* models/diffusion_models/
	@mv models/split_files/text_encoders/* models/text_encoders/
	@mv models/split_files/vae/* models/vae/
	@rm -rf models/split_files

download-models: download-anima download-sam3 download-mit

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
		--model-path models/mit/model.pth

mask:
	@if [ ! -d masks_sam ]; then $(MAKE) mask-sam; fi
	@if [ ! -d masks_mit ]; then $(MAKE) mask-mit; fi
	python scripts/merge_masks.py \
		masks_sam masks_mit \
		--output-dir masks

mask-clean:
	rm -rf masks/ masks_sam/ masks_mit/
