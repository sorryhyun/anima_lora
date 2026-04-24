ACCELERATE := python -m accelerate.commands.accelerate_cli
LATEST_LORA = $(shell python -c "import glob,os; files=[f for f in glob.glob('output/ckpt/*.safetensors') if not f.endswith('_moe.safetensors')]; print(max(files,key=os.path.getmtime))")
LATEST_HYDRA = $(shell python -c "import glob,os; files=[f for f in glob.glob('output/ckpt/anima_hydra*_moe.safetensors') if '.bak.' not in f]; print(max(files,key=os.path.getmtime))")
LATEST_APEX = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/anima_apex*.safetensors'); print(max(files,key=os.path.getmtime))")
LATEST_PREFIX = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/anima_prefix*.safetensors'); print(max(files,key=os.path.getmtime))")
LATEST_REF = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/anima_ref*.safetensors'); print(max(files,key=os.path.getmtime) if files else '')")
LATEST_POSTFIX = $(shell python -c "import glob,os; files=[f for f in glob.glob('output/ckpt/anima_postfix*.safetensors') if '_exp' not in os.path.basename(f) and '_func' not in os.path.basename(f)]; print(max(files,key=os.path.getmtime))")
LATEST_POSTFIX_EXP = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/anima_postfix_exp*.safetensors'); print(max(files,key=os.path.getmtime))")
LATEST_POSTFIX_FUNC = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/anima_postfix_func*.safetensors'); print(max(files,key=os.path.getmtime))")
LATEST_MOD = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/pooled_text_proj*.safetensors'); print(max(files,key=os.path.getmtime))")
MODEL_DIR ?= output_temp
LATEST_MERGED = $(shell python -c "import glob,os; p='$(MODEL_DIR)'; files=[p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p,'*_merged.safetensors')),key=os.path.getmtime); print(files[-1] if files else '')")

.PHONY: lora lora-fast lora-low-vram lora-gui apex postfix step test test-mod test-apex test-hydra test-prefix test-postfix test-postfix-exp test-postfix-func test-spectrum test-merge test-ref invert invert-ref test-invert bench-inversion distill-mod img2emb img2emb-preprocess img2emb-anchors img2emb-pretrain img2emb-finetune preprocess-img2emb test-img2emb mask mask-sam mask-mit mask-clean preprocess preprocess-resize preprocess-vae preprocess-te download-models download-anima download-sam3 download-mit download-tipsv2 gui comfy-batch test-unit print-config merge

TEST_COMMON = python inference.py \
	--dit models/diffusion_models/anima-preview3-base.safetensors \
	--text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
	--vae models/vae/qwen_image_vae.safetensors \
	--vae_chunk_size 64 --vae_disable_cache \
	--attn_mode flash \
	--prompt "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top and denim shorts is standing outdoors. She's holding a rectangular sign out in front of her that reads \"ANIMA\". She's looking at the viewer with a smile. The background features some trees and blue sky with clouds." \
	--negative_prompt "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia" \
	--image_size 1024 1024 \
	--infer_steps 30 \
	--flow_shift 1.0 \
	--sampler er_sde \
	--guidance_scale 4.0 \
	--seed 42 \
	--save_path output/tests

gui:
	python -m gui

TRAIN = $(ACCELERATE) launch --num_cpu_threads_per_process 3 --mixed_precision bf16 train.py --method
PRESET ?= default

lora:
	$(TRAIN) lora --preset $(PRESET)

lora-half:
	$(TRAIN) lora --preset half

lora-fast:
	$(TRAIN) lora --preset fast_16gb

lora-low-vram:
	$(TRAIN) lora --preset low_vram

# Clean per-variant path for basic users: picks the chosen variant directly
# out of configs/gui-methods/<variant>.toml (no toggle-block hand-editing).
# Example: `make lora-gui GUI_PRESETS=tlora` — see configs/gui-methods/ for
# the full list (lora, ortholora, tlora, reft, tlora_ortho_reft, hydralora,
# hydralora_sigma, postfix, postfix_exp, postfix_func, postfix_sigma, prefix).
GUI_PRESETS ?= lora
lora-gui:
	$(TRAIN) $(GUI_PRESETS) --methods_subdir gui-methods --preset $(PRESET)

apex:
	$(TRAIN) apex --preset $(PRESET)

postfix:
	$(TRAIN) postfix --preset $(PRESET)

distill-mod:
	python scripts/distill_modulation.py \
		--data_dir post_image_dataset \
		--dit_path models/diffusion_models/anima-preview3-base.safetensors \
		--output_path output/ckpt/pooled_text_proj.safetensors \
		--iterations 1500 \
		--lr 1e-5 \
		--warmup 0.05 \
		--blocks_to_swap 0 \
		--attn_mode flash \
		--no_grad_ckpt \
		$(ARGS)

# img2emb — image→embedding resampler training (TIPSv2-L/14 encoder).
# Three stages: preprocess → pretrain → finetune. See scripts/img2emb/train.py.
# Images are assigned to the closest patch-14 bucket (~1024 tokens, aspects
# 1:2..2:1); tokens are zero-padded to a single T_MAX so the cache stays a
# flat (N, T_MAX, D) tensor. img2emb-anchors refreshes phase1_positions.json
# + phase2_class_prototypes.safetensors (people=* prototypes) from live
# captions + TE cache; required before pretrain. Requires
# `make download-tipsv2` and `trust_remote_code`.

img2emb-preprocess:
	python scripts/img2emb/preprocess.py $(ARGS)

img2emb-anchors:
	python scripts/img2emb/rebuild_anchor_artifacts.py $(ARGS)

img2emb-pretrain:
	python scripts/img2emb/pretrain.py $(ARGS)

img2emb-finetune:
	python scripts/img2emb/finetune.py $(ARGS)

preprocess-img2emb: img2emb-preprocess img2emb-anchors

img2emb: img2emb-pretrain img2emb-finetune

# Generate a single image conditioned on REF_IMAGE via the finetuned resampler.
# Example: make test-img2emb REF_IMAGE=post_image_dataset/foo.png
REF_IMAGE ?=
test-img2emb:
	@if [ -z "$(REF_IMAGE)" ]; then echo "Set REF_IMAGE=path/to/ref.png"; exit 1; fi
	python scripts/img2emb/infer.py --ref_image $(REF_IMAGE) $(ARGS)

test:
	$(TEST_COMMON) \
		--lora_weight $(LATEST_LORA) \
		--lora_multiplier 1.0

test-mod:
	$(TEST_COMMON) \
		--pooled_text_proj $(LATEST_MOD)

test-apex:
	$(TEST_COMMON) \
		--lora_weight $(LATEST_APEX) \
		--lora_multiplier 1.0 \
		--infer_steps 4 \
		--guidance_scale 1.0 \
		--sampler euler

test-hydra:
	$(TEST_COMMON) \
		--lora_weight $(LATEST_HYDRA) \
		--lora_multiplier 1.0

test-prefix:
	$(TEST_COMMON) \
		--prefix_weight $(LATEST_PREFIX)

test-postfix:
	$(TEST_COMMON) \
		--postfix_weight $(LATEST_POSTFIX)

test-postfix-exp:
	$(TEST_COMMON) \
		--postfix_weight $(LATEST_POSTFIX_EXP)

test-postfix-func:
	$(TEST_COMMON) \
		--postfix_weight $(LATEST_POSTFIX_FUNC)

# Inference with latest reference-inversion prefix (anima_ref_*.safetensors,
# produced by `make invert-ref`). Uses the existing prefix loader today, which
# prepends the K slot vectors at position 0 of crossattn_emb — matching exactly
# how the slots were assembled during inversion.
test-ref:
	$(TEST_COMMON) \
		--prefix_weight $(LATEST_REF)

# Inference with a baked (merged) DiT. MODEL_DIR accepts either a directory
# (picks the latest *_merged.safetensors inside) or a direct .safetensors path.
# No --lora_weight — the LoRA is already folded into the weights. The trailing
# --dit overrides the base one in TEST_COMMON.
# Example: make test-merge MODEL_DIR=output_temp
#          make test-merge MODEL_DIR=output_temp/my-model_merged.safetensors
test-merge:
	$(TEST_COMMON) \
		--dit $(LATEST_MERGED)

test-spectrum:
	$(TEST_COMMON) \
		--lora_weight $(LATEST_LORA) \
		--lora_multiplier 1.0 \
		--spectrum \
		--spectrum_window_size 2.0 \
		--spectrum_flex_window 0.25 \
		--spectrum_warmup 7 \
		--spectrum_w 0.3 \
		--spectrum_m 3 \
		--spectrum_lam 0.1 \
		--spectrum_stop_caching_step 29 \
		--spectrum_calibration 0.0

INVERT_N ?= 1
INVERT_SWAP ?= 0
INVERT_STEPS ?= 50
INVERT_LR ?= 1e-3
INVERT_AGG ?= 1
INVERT_OUT ?= output/inversions
INVERT_PROBE_BLOCKS ?= 8,12,16,20
invert:
	python scripts/inversion/invert_embedding.py \
		--dit models/diffusion_models/anima-preview3-base.safetensors \
		--attn_mode flash \
		--image_dir post_image_dataset \
		--num_images $(INVERT_N) --shuffle \
		--steps $(INVERT_STEPS) --lr $(INVERT_LR) \
		--aggregate_by $(INVERT_AGG) \
		--save_per_run \
		--probe_functional --probe_blocks $(INVERT_PROBE_BLOCKS) \
		--output_dir $(INVERT_OUT) \
		--blocks_to_swap $(INVERT_SWAP) \
		--log_block_grads \
		--init_zeros

# Reference inversion: learn K prefix-slot vectors that encode a single
# reference image's subject/style. Output plugs into `make test-ref` via the
# existing --prefix_weight loader.
#
# Image source:
#   - If REF_IMAGE is set, use that file.
#   - Otherwise pick a random file from REF_IMAGE_DIR (default post_image_dataset).
#     Re-running `make invert-ref` picks a new random image each time.
#
# Optional:  REF_TEMPLATE="a photo" REF_K=8 REF_STEPS=100 REF_LR=0.01
#            REF_NAME=latest  (output saved as output/ckpt/anima_ref_$(REF_NAME).safetensors)
#            REF_SWAP=0       (blocks_to_swap; >0 for low VRAM, <0 for grad checkpointing)
REF_IMAGE_DIR ?= post_image_dataset
# `?=` with `$(shell ...)` is RECURSIVE — every use of $(REF_IMAGE) re-runs the
# picker and would pick a different random file each expansion. Guard with
# ifndef + immediate `:=` so one random pick is frozen for the whole target.
ifndef REF_IMAGE
REF_IMAGE := $(shell python -c "import glob,os,random; d='$(REF_IMAGE_DIR)'; files=sum((glob.glob(os.path.join(d,'**',e),recursive=True) for e in ('*.png','*.jpg','*.jpeg','*.webp')),[]); print(random.choice(files) if files else '')")
endif
REF_TEMPLATE ?= a photo
REF_K ?= 8
REF_STEPS ?= 100
REF_LR ?= 0.01
REF_NAME ?= latest
REF_SAVE_PATH ?= output/ckpt/anima_ref_$(REF_NAME).safetensors
REF_SWAP ?= 0
invert-ref:
	@if [ -z "$(REF_IMAGE)" ]; then \
		echo "Error: no images found in REF_IMAGE_DIR=$(REF_IMAGE_DIR)/ and REF_IMAGE not set."; \
		echo "       Either pass REF_IMAGE=path/to/ref.png or point REF_IMAGE_DIR at a directory with .png/.jpg/.webp files."; \
		exit 1; \
	fi
	@echo "  > using reference image: $(REF_IMAGE)"
	python scripts/inversion/invert_reference.py \
		--image "$(REF_IMAGE)" \
		--dit models/diffusion_models/anima-preview3-base.safetensors \
		--vae models/vae/qwen_image_vae.safetensors \
		--text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
		--attn_mode flash \
		--template "$(REF_TEMPLATE)" \
		--num_tokens $(REF_K) \
		--steps $(REF_STEPS) \
		--lr $(REF_LR) \
		--save_path $(REF_SAVE_PATH) \
		--blocks_to_swap $(REF_SWAP) \
		--verify

BENCH_INVERSIONS ?= 5
bench-inversion:
	python bench/inversion/inversion_stability.py \
		--dit models/diffusion_models/anima-preview3-base.safetensors \
		--vae models/vae/qwen_image_vae.safetensors \
		--num_inversions $(BENCH_INVERSIONS) \
		--steps 100 --lr 0.01

INVERT_NAME ?= latest
test-invert:
	python scripts/inversion/interpret_inversion.py \
		--dit models/diffusion_models/anima-preview3-base.safetensors \
		--vae models/vae/qwen_image_vae.safetensors \
		--attn_mode flash \
		--name $(INVERT_NAME) \
		--verify --verify_steps 30

WORKFLOW ?= workflows/modhydra.json
comfy-batch:
	python scripts/comfy_batch.py $(WORKFLOW)

# img2emb finetune step-0 loss-weight calibration — no backward, just reports
# raw loss magnitudes so the loss.* weights in finetune.py can be tuned.
FINETUNE_WARM ?= output/img2embs/pretrain/tipsv2_resampler_4layer_anchored.safetensors
FINETUNE_BS ?= 1
# -1 = gradient checkpointing (required on 16 GB; no-swap backward OOMs, block-swap
# forward-only doesn't help backward activations). Override with FINETUNE_SWAP=N for
# >16 GB cards.
FINETUNE_SWAP ?= -1
img2emb-calibrate:
	python scripts/img2emb/finetune.py \
		--dit models/diffusion_models/anima-preview3-base.safetensors \
		--warm_start $(FINETUNE_WARM) \
		--calibrate_only \
		--batch_size $(FINETUNE_BS) \
		--blocks_to_swap $(FINETUNE_SWAP)

graft-step:
	python scripts/graft_step.py

preprocess: preprocess-resize preprocess-vae preprocess-te

preprocess-resize:
	python preprocess/resize_images.py \
		--src image_dataset \
		--dst post_image_dataset

preprocess-vae:
	python preprocess/cache_latents.py \
		--dir post_image_dataset \
		--vae models/vae/qwen_image_vae.safetensors \
		--batch_size 4 \
		--chunk_size 64

preprocess-te:
	python preprocess/cache_text_embeddings.py \
		--dir post_image_dataset \
		--qwen3 models/text_encoders/qwen_3_06b_base.safetensors \
		--dit models/diffusion_models/anima-preview3-base.safetensors \
		--caption_shuffle_variants 4

# --- Model downloads ---

download-sam3:
	python -c "import os; os.makedirs('models/sam3',exist_ok=True)"
	hf download facebook/sam3 --local-dir models/sam3

download-mit:
	python -c "import os; os.makedirs('models/mit',exist_ok=True)"
	hf download a-b-c-x-y-z/Manga-Text-Segmentation-2025 \
		model.pth --local-dir models/mit

download-anima:
	python -c "import os; [os.makedirs(d,exist_ok=True) for d in ['models/diffusion_models','models/text_encoders','models/vae']]"
	hf download circlestone-labs/Anima \
		split_files/diffusion_models/anima-preview3-base.safetensors \
		split_files/text_encoders/qwen_3_06b_base.safetensors \
		split_files/vae/qwen_image_vae.safetensors \
		--local-dir models --include "split_files/*"
	python -c "import shutil,os; [shutil.move(os.path.join('models/split_files',d,f),os.path.join('models',d,f)) for d in ['diffusion_models','text_encoders','vae'] for f in os.listdir(os.path.join('models/split_files',d))]; shutil.rmtree('models/split_files')"

# TIPSv2-L/14 vision encoder for img2emb (see scripts/img2emb/preprocess.py).
# Ships custom code (trust_remote_code); the whole repo must be present locally
# so `from_pretrained("models/tipsv2", trust_remote_code=True)` can resolve the
# python modules without network.
download-tipsv2:
	python -c "import os; os.makedirs('models/tipsv2',exist_ok=True)"
	hf download google/tipsv2-l14 --local-dir models/tipsv2

download-models: download-anima download-sam3 download-mit download-tipsv2

# --- Masking ---

mask-sam:
	python preprocess/generate_masks.py \
		--config configs/sam_mask.yaml \
		--image-dir post_image_dataset \
		--mask-dir masks/sam \
		--checkpoint models/sam3/sam3.pt \
		--batch-size 2

mask-mit:
	python preprocess/generate_masks_mit.py \
		--image-dir post_image_dataset \
		--mask-dir masks/mit \
		--model-path models/mit/model.pth

mask:
	python -c "import os,subprocess; [subprocess.check_call(['$(MAKE)',t]) for t,d in [('mask-sam','masks/sam'),('mask-mit','masks/mit')] if not os.path.isdir(d)]"
	python preprocess/merge_masks.py \
		masks/sam masks/mit \
		--output-dir masks/merged

mask-clean:
	python -c "import shutil; shutil.rmtree('masks',ignore_errors=True)"

test-unit:
	pytest -q tests/ $(ARGS)

METHOD ?= lora
print-config:
	python train.py --method $(METHOD) --preset $(PRESET) --print-config --no-config-snapshot

# Dump TensorBoard scalar logs to JSON (one file per run dir).
# Example: make export-logs RUN=output/logs/20260424161100
#          make export-logs RUN=output/logs ALL=1
#          make export-logs RUN=output/logs/20260424161100 JSONL=1
RUN ?= output/logs
export-logs:
	python scripts/export_logs_json.py $(RUN) $(if $(ALL),--all) $(if $(JSONL),--jsonl) $(ARGS)

# Bake a LoRA adapter into the base DiT (standalone merged checkpoint).
# Picks the latest bakeable .safetensors in ADAPTER_DIR (skips _moe/postfix/prefix/.bak).
# Example: make merge ADAPTER_DIR=output/ckpt MULTIPLIER=1.0
ADAPTER_DIR ?= output/ckpt
MULTIPLIER ?= 1.0
merge:
	python scripts/merge_to_dit.py \
		--adapter_dir $(ADAPTER_DIR) \
		--multiplier $(MULTIPLIER) \
		$(ARGS)
