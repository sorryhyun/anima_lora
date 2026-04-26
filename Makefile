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
LATEST_IP = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/anima_ip_adapter*.safetensors'); print(max(files,key=os.path.getmtime) if files else '')")
LATEST_EC = $(shell python -c "import glob,os; files=glob.glob('output/ckpt/anima_easycontrol*.safetensors'); print(max(files,key=os.path.getmtime) if files else '')")
MODEL_DIR ?= output_temp
LATEST_MERGED = $(shell python -c "import glob,os; p='$(MODEL_DIR)'; files=[p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p,'*_merged.safetensors')),key=os.path.getmtime); print(files[-1] if files else '')")

.PHONY: lora lora-fast lora-low-vram lora-gui apex postfix ip-adapter ip-adapter-cache easycontrol test-easycontrol step test test-mod test-apex test-hydra test-prefix test-postfix test-postfix-exp test-postfix-func test-ip test-spectrum test-merge test-ref invert invert-ref test-invert bench-inversion distill-mod img2emb img2emb-preprocess img2emb-anchors img2emb-align img2emb-pretrain img2emb-finetune preprocess-img2emb test-img2emb mask mask-sam mask-mit mask-clean preprocess preprocess-resize preprocess-vae preprocess-te download-models download-anima download-sam3 download-mit download-tipsv2 download-pe download-pe-g gui comfy-batch test-unit print-config merge

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

# ARTIST=<name> trains an artist-only LoRA: filters the dataset to images whose
# caption contains the `@<name>` tag and saves to output/ckpt-artist/<output_name>_<name>.safetensors.
# Pass with or without leading `@`. Example: `make lora ARTIST=sincos`.
ARTIST ?=
ARTIST_ARG := $(if $(ARTIST),--artist_filter $(ARTIST),)

lora:
	$(TRAIN) lora --preset $(PRESET) $(ARTIST_ARG)

lora-half:
	$(TRAIN) lora --preset half $(ARTIST_ARG)

lora-fast:
	$(TRAIN) lora --preset fast_16gb $(ARTIST_ARG)

lora-low-vram:
	$(TRAIN) lora --preset low_vram $(ARTIST_ARG)

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

# IP-Adapter — decoupled image cross-attention. Adapter-only (DiT frozen).
# Reference image and target both come from post_image_dataset/. PE-Core is
# loaded live during training (cache_latents method-forced false).
ip-adapter:
	$(TRAIN) ip_adapter --preset $(PRESET)

# Pre-cache PE-Core patch features for every image in post_image_dataset/.
# Writes {stem}_anima_pe.safetensors sidecars (bf16, [T_pe, d_enc]). Idempotent.
# IP_ENCODER overrides the registry name (default: pe). Useful when iterating
# IP-Adapter training: avoids re-running the vision encoder every step.
IP_ENCODER ?= pe
ip-adapter-cache:
	python preprocess/cache_pe_encoder.py \
		--dir post_image_dataset \
		--encoder $(IP_ENCODER)

# EasyControl — extended self-attention image conditioning. Adapter-only (DiT
# frozen). Reference and target both come from post_image_dataset/. Reuses the
# existing cache_latents output as the cond input — no separate cache step.
easycontrol:
	$(TRAIN) easycontrol --preset $(PRESET)

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

# img2emb — image→embedding resampler training.
# Three stages: preprocess → pretrain → finetune. See scripts/img2emb/train.py.
# Three encoders are wired in (`make img2emb-... ENCODER=tipsv2|pe|pe-g`):
#   tipsv2 (default) — TIPSv2-L/14, 32x32=1024 patch tokens at 448 px.
#                      Requires `make download-tipsv2` + `trust_remote_code`.
#   pe               — Meta PE-Core-L14-336, 24x24=576 patch tokens at 336 px.
#                      Requires `make download-pe`; vision tower vendored at
#                      library/models/pe.py (no perception_models clone).
#   pe-g             — Meta PE-Core-G14-448, 32x32=1024 patch tokens at 448 px,
#                      no CLS token, larger backbone (50 layers, width=1536).
#                      Requires `make download-pe-g`; same vendored tower.
# Images are assigned to the closest patch-14 bucket; tokens are zero-padded
# to a single T_MAX so the cache stays a flat (N, T_MAX, D) tensor.
# img2emb-anchors refreshes phase1_positions.json + phase2_class_prototypes
# (encoder-agnostic — only depends on cached T5 embeddings).

# Encoder selector — flows into every stage as --encoder, so the cache,
# pretrain ckpt, and finetune ckpt filenames stay coherent. Stage scripts
# accept --encoder explicitly, so command-line wins over ENCODER if both
# are set; this is by design (re-running just one stage with a different
# encoder is a footgun, but sometimes intentional).
ENCODER ?=
ENCODER_FLAG := $(if $(ENCODER),--encoder $(ENCODER),)

img2emb-preprocess:
	python scripts/img2emb/preprocess.py $(ENCODER_FLAG) $(ARGS)

img2emb-anchors:
	python scripts/img2emb/rebuild_anchor_artifacts.py $(ARGS)

# One-shot Hungarian alignment of T5 variants v1..vN to v0 in
# post_image_dataset/*_anima_te.safetensors. Runs implicitly inside
# img2emb-preprocess; this target is for re-running standalone (idempotent).
img2emb-align:
	python scripts/img2emb/align_variants.py $(ARGS)

img2emb-pretrain:
	python scripts/img2emb/pretrain.py $(ENCODER_FLAG) $(ARGS)

img2emb-finetune:
	python scripts/img2emb/finetune.py $(ENCODER_FLAG) $(ARGS)

preprocess-img2emb: img2emb-preprocess img2emb-anchors

img2emb: img2emb-pretrain img2emb-finetune

# Generate a single image conditioned on REF_IMAGE via the finetuned resampler.
# Example: make test-img2emb REF_IMAGE=post_image_dataset/foo.png ENCODER=pe
REF_IMAGE ?=
test-img2emb:
	@if [ -z "$(REF_IMAGE)" ]; then echo "Set REF_IMAGE=path/to/ref.png"; exit 1; fi
	python scripts/img2emb/infer.py --ref_image $(REF_IMAGE) $(ENCODER_FLAG) $(ARGS)

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

# IP-Adapter inference. REF_IMAGE is the reference image (style/identity);
# PROMPT describes the scene to render with that style. IP_SCALE overrides the
# saved adapter strength (default: use ss_ip_scale from the checkpoint).
# Examples:
#   make test-ip REF_IMAGE=post_image_dataset/foo.png \
#                PROMPT="a girl drinking coffee at a cafe"
#   make test-ip REF_IMAGE=foo.png PROMPT="..." NEG="bad anatomy" IP_SCALE=0.8
IP_SCALE ?=
IP_SCALE_ARG := $(if $(IP_SCALE),--ip_scale $(IP_SCALE),)
# Short default — IP-Adapter is meant to carry style/identity from REF_IMAGE,
# so the prompt is intentionally minimal. Override with PROMPT="..." for content.
PROMPT ?= double peace, v v,
PROMPT_ARG := --prompt "$(PROMPT)"
NEG ?=
NEG_ARG := $(if $(NEG),--negative_prompt "$(NEG)",)
test-ip:
	@if [ -z "$(REF_IMAGE)" ]; then echo "Set REF_IMAGE=path/to/ref.png"; exit 1; fi
	@mkdir -p output/tests/ip
	$(TEST_COMMON) \
		--save_path output/tests/ip \
		--ip_adapter_weight $(LATEST_IP) \
		--ip_image $(REF_IMAGE) \
		--ip_image_match_size \
		$(IP_SCALE_ARG) \
		$(PROMPT_ARG) \
		$(NEG_ARG)
	@gen=$$(ls -1t output/tests/ip/*.png 2>/dev/null | grep -v '_ref\.png$$' | head -1); \
	 if [ -n "$$gen" ]; then ref_dst="$${gen%.png}_ref.png"; cp "$(REF_IMAGE)" "$$ref_dst"; echo "Ref pasted: $$ref_dst"; fi

# EasyControl inference. REF_IMAGE is the reference image (style/identity).
# EC_SCALE overrides the saved adapter strength (default: ss_cond_scale).
EC_SCALE ?=
EC_SCALE_ARG := $(if $(EC_SCALE),--easycontrol_scale $(EC_SCALE),)
test-easycontrol:
	@if [ -z "$(REF_IMAGE)" ]; then echo "Set REF_IMAGE=path/to/ref.png"; exit 1; fi
	@mkdir -p output/tests/easycontrol
	$(TEST_COMMON) \
		--save_path output/tests/easycontrol \
		--easycontrol_weight $(LATEST_EC) \
		--easycontrol_image $(REF_IMAGE) \
		--easycontrol_image_match_size \
		$(EC_SCALE_ARG) \
		$(PROMPT_ARG) \
		$(NEG_ARG)
	@gen=$$(ls -1t output/tests/easycontrol/*.png 2>/dev/null | grep -v '_ref\.png$$' | head -1); \
	 if [ -n "$$gen" ]; then ref_dst="$${gen%.png}_ref.png"; cp "$(REF_IMAGE)" "$$ref_dst"; echo "Ref pasted: $$ref_dst"; fi

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
# Warm path defaults to the pretrain output for the selected ENCODER (tipsv2
# unless overridden); set FINETUNE_WARM=... to point elsewhere.
FINETUNE_ENCODER := $(if $(ENCODER),$(ENCODER),tipsv2)
FINETUNE_WARM ?= output/img2embs/pretrain/$(FINETUNE_ENCODER)_resampler_4layer_anchored.safetensors
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
		--blocks_to_swap $(FINETUNE_SWAP) \
		$(ENCODER_FLAG)

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

# Perception Encoder PE-Core-L14-336 vision tower for img2emb. Vision tower is
# vendored at library/models/pe.py — only the .pt checkpoint is needed.
download-pe:
	python -c "import os; os.makedirs('models/pe',exist_ok=True)"
	hf download facebook/PE-Core-L14-336 PE-Core-L14-336.pt --local-dir models/pe

# Larger PE sibling — 448px native, 1024 patch tokens, no CLS, 50-layer 1536-wide
# backbone. Same vendored vision tower in library/models/pe.py.
download-pe-g:
	python -c "import os; os.makedirs('models/pe',exist_ok=True)"
	hf download facebook/PE-Core-G14-448 PE-Core-G14-448.pt --local-dir models/pe

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
