"""Bilingual help text for config fields and LoRA variant descriptions."""

from __future__ import annotations

from gui.i18n import current_language

# ── Per-field tooltips ─────────────────────────────────────────
# Keys match config field names. Each maps to {lang: description}.

FIELD_HELP: dict[str, dict[str, str]] = {
    # Architecture
    "network_dim": {
        "en": "LoRA rank (dimension of low-rank matrices). Higher = more expressive but more VRAM. Typical: 8\u201364.",
        "ko": "LoRA 랭크 (저랭크 행렬의 차원). 높을수록 표현력이 좋지만 VRAM 사용량 증가. 일반적: 8\u201364.",
    },
    "network_alpha": {
        "en": "LoRA scaling factor. Effective scale = alpha / dim. When alpha == dim, scale is 1.0. Lower alpha = more conservative updates.",
        "ko": "LoRA 스케일링 계수. 실효 스케일 = alpha / dim. alpha == dim이면 1.0. 낮을수록 보수적 업데이트.",
    },
    "network_module": {
        "en": "Python module path for the LoRA network implementation.",
        "ko": "LoRA 네트워크 구현의 Python 모듈 경로.",
    },
    "use_dora": {
        "en": "Enable DoRA (Weight-Decomposed LoRA). Learns separate magnitude and direction components. Often more stable at higher ranks.",
        "ko": "DoRA (가중치 분해 LoRA) 활성화. 크기와 방향을 분리하여 학습. 높은 랭크에서 더 안정적.",
    },
    "use_timestep_mask": {
        "en": "Enable T-LoRA: effective rank varies with denoising timestep via power-law schedule. Full rank at high noise, reduced at low noise.",
        "ko": "T-LoRA 활성화: 디노이징 타임스텝에 따라 유효 랭크 변동. 높은 노이즈에서 전체 랭크, 낮은 노이즈에서 축소.",
    },
    "min_rank": {
        "en": "Minimum active rank when T-LoRA timestep masking is enabled. At the lowest-noise timesteps, rank drops to this value.",
        "ko": "T-LoRA 타임스텝 마스킹 사용 시 최소 활성 랭크. 가장 낮은 노이즈에서 이 값까지 감소.",
    },
    "alpha_rank_scale": {
        "en": "Scale alpha proportionally when T-LoRA reduces rank, keeping effective learning rate stable across timesteps.",
        "ko": "T-LoRA가 랭크를 줄일 때 alpha를 비례적으로 조정하여 타임스텝별 실효 학습률 유지.",
    },
    "network_train_unet_only": {
        "en": "Train only the DiT (U-Net). Text encoder weights are frozen. Recommended for most LoRA training.",
        "ko": "DiT(U-Net)만 학습. 텍스트 인코더 가중치는 동결. 대부분의 LoRA 학습에 권장.",
    },

    # Training
    "learning_rate": {
        "en": "Base learning rate for the optimizer. Typical: 1e-5 to 1e-4. DoRA often benefits from slightly higher LR than standard LoRA.",
        "ko": "옵티마이저 기본 학습률. 일반적: 1e-5 ~ 1e-4. DoRA는 표준 LoRA보다 약간 높은 LR이 효과적.",
    },
    "max_train_epochs": {
        "en": "Total training epochs. One epoch = one full pass through the dataset.",
        "ko": "총 학습 에폭 수. 1 에폭 = 데이터셋 전체를 1회 순회.",
    },
    "save_every_n_epochs": {
        "en": "Save a checkpoint every N epochs. Set equal to max_train_epochs to save only the final model.",
        "ko": "N 에폭마다 체크포인트 저장. max_train_epochs와 같게 설정하면 최종 모델만 저장.",
    },
    "checkpointing_epochs": {
        "en": "Save resumable training state every N epochs. State files are large; use a larger interval than save_every_n_epochs.",
        "ko": "N 에폭마다 학습 재개 상태 저장. 상태 파일이 크므로 save_every_n_epochs보다 큰 간격 권장.",
    },
    "gradient_accumulation_steps": {
        "en": "Accumulate gradients over N steps before updating. Effective batch size = batch_size \u00d7 accumulation_steps.",
        "ko": "N 스텝 동안 그레이디언트 누적 후 업데이트. 실효 배치 크기 = batch_size \u00d7 accumulation_steps.",
    },
    "caption_shuffle_variants": {
        "en": "Number of random caption tag-order variants per image. Higher = better caption robustness, slower caching.",
        "ko": "이미지당 캡션 태그 순서 무작위 변형 수. 높을수록 캡션 견고성 향상, 캐싱 속도 감소.",
    },
    "optimizer_type": {
        "en": "Optimizer algorithm. AdamW8bit: memory-efficient 8-bit Adam. Others: AdamW, Lion, Prodigy, etc.",
        "ko": "옵티마이저 알고리즘. AdamW8bit: 메모리 효율적 8비트 Adam. 기타: AdamW, Lion, Prodigy 등.",
    },
    "lr_scheduler": {
        "en": "Learning rate schedule. constant: fixed LR. Others: cosine, cosine_with_restarts, polynomial.",
        "ko": "학습률 스케줄. constant: 고정 LR. 기타: cosine, cosine_with_restarts, polynomial.",
    },
    "timestep_sampling": {
        "en": "How denoising timesteps are sampled during training. sigmoid: biased toward middle timesteps (recommended for flow matching).",
        "ko": "학습 중 디노이징 타임스텝 샘플링 방법. sigmoid: 중간 타임스텝 편향 (flow matching 권장).",
    },
    "discrete_flow_shift": {
        "en": "Flow-matching shift parameter controlling the noise schedule distribution. Default: 1.0.",
        "ko": "노이즈 스케줄 분포를 제어하는 flow-matching 시프트 매개변수. 기본값: 1.0.",
    },

    # Performance
    "attn_mode": {
        "en": "Attention backend. flash4: FlashAttention-4 (Linux, fastest). flash: FlashAttention-2. flex: PyTorch flex attention (cross-platform).",
        "ko": "어텐션 백엔드. flash4: FlashAttention-4 (Linux, 최속). flash: FlashAttention-2. flex: PyTorch flex attention (크로스 플랫폼).",
    },
    "gradient_checkpointing": {
        "en": "Recompute activations during backward pass instead of storing them. Trades compute for VRAM. Essential for low-VRAM setups.",
        "ko": "역전파 시 활성값을 저장 대신 재계산. 연산으로 VRAM 절약. 저사양 필수.",
    },
    "unsloth_offload_checkpointing": {
        "en": "Offload gradient checkpoints to CPU RAM. Further VRAM reduction at cost of speed. Requires gradient_checkpointing=true.",
        "ko": "그레이디언트 체크포인트를 CPU RAM으로 오프로드. 속도 감소 대신 VRAM 추가 절약. gradient_checkpointing=true 필요.",
    },
    "blocks_to_swap": {
        "en": "Number of DiT blocks to swap between GPU and CPU. 0: all on GPU. Higher values = more CPU offloading for low VRAM.",
        "ko": "GPU와 CPU 간 스왑할 DiT 블록 수. 0: 전부 GPU. 높을수록 더 많이 CPU로 오프로드.",
    },
    "torch_compile": {
        "en": "Enable torch.compile for the forward pass. Faster training after initial compilation. Best with static_token_count=true.",
        "ko": "torch.compile 활성화. 초기 컴파일 후 학습 속도 향상. static_token_count=true와 함께 사용 권장.",
    },
    "compile_mode": {
        "en": "'blocks': compile each DiT block individually (default). 'full': compile entire model as one graph for cross-block memory optimization. Full mode is incompatible with gradient checkpointing and block swap.",
        "ko": "'blocks': 각 DiT 블록을 개별 컴파일 (기본값). 'full': 전체 모델을 하나의 그래프로 컴파일하여 블록 간 메모리 최적화. full 모드는 gradient checkpointing 및 block swap과 호환 불가.",
    },
    "trim_crossattn_kv": {
        "en": "Remove zero-padding from cross-attention KV for efficiency. Flash4 applies LSE correction to maintain correct softmax.",
        "ko": "효율을 위해 크로스 어텐션 KV에서 제로 패딩 제거. Flash4는 정확한 softmax를 위해 LSE 보정 적용.",
    },
    "cache_llm_adapter_outputs": {
        "en": "Cache the LLM adapter layer outputs to disk. Avoids recomputing text encoder projections each epoch.",
        "ko": "LLM 어댑터 레이어 출력을 디스크에 캐싱. 매 에폭 텍스트 인코더 투영 재계산 회피.",
    },
    "masked_loss": {
        "en": "Apply loss only to non-masked regions (e.g., exclude text bubbles). Requires mask files in masks/ directory.",
        "ko": "마스크되지 않은 영역에만 손실 적용 (예: 말풍선 제외). masks/ 디렉토리에 마스크 파일 필요.",
    },
    "mixed_precision": {
        "en": "Mixed precision mode. bf16: recommended for modern GPUs. fp16: for older GPUs without bf16 support.",
        "ko": "혼합 정밀도 모드. bf16: 최신 GPU 권장. fp16: bf16 미지원 구형 GPU용.",
    },
    "lora_fp32_accumulation": {
        "en": "Compute LoRA forward in fp32 then cast back. Improves training precision at minimal speed cost.",
        "ko": "LoRA 순전파를 fp32로 계산 후 변환. 속도 손실 최소화하며 학습 정밀도 향상.",
    },
    "static_token_count": {
        "en": "Fixed 4096 token count for all batches. Gives torch.compile a single static shape \u2014 no recompilation across aspect ratios.",
        "ko": "모든 배치에 4096 토큰 고정. torch.compile에 단일 정적 셰이프 제공 \u2014 화면비별 재컴파일 없음.",
    },
    "vae_chunk_size": {
        "en": "VAE decoding chunk size. Larger = faster but more VRAM. 64 is a good balance.",
        "ko": "VAE 디코딩 청크 크기. 클수록 빠르지만 VRAM 더 사용. 64가 적절.",
    },
    "vae_disable_cache": {
        "en": "Disable VAE's internal KV cache. Reduces VRAM during VAE encoding/decoding.",
        "ko": "VAE 내부 KV 캐시 비활성화. VAE 인코딩/디코딩 시 VRAM 감소.",
    },
    "cache_latents": {
        "en": "Cache VAE-encoded latents in memory. Avoids re-encoding images every epoch.",
        "ko": "VAE 인코딩된 레이턴트를 메모리에 캐싱. 매 에폭 이미지 재인코딩 회피.",
    },
    "cache_latents_to_disk": {
        "en": "Save cached latents to disk instead of RAM. Frees system memory at cost of disk I/O.",
        "ko": "캐시된 레이턴트를 RAM 대신 디스크에 저장. 디스크 I/O 대신 시스템 메모리 절약.",
    },
    "cache_text_encoder_outputs": {
        "en": "Cache text encoder outputs. Essential for lazy loading: encode \u2192 cache \u2192 free encoder \u2192 load DiT.",
        "ko": "텍스트 인코더 출력 캐싱. 지연 로딩 필수: 인코딩 \u2192 캐시 \u2192 인코더 해제 \u2192 DiT 로드.",
    },
    "cache_text_encoder_outputs_to_disk": {
        "en": "Save cached text encoder outputs to disk. Required for the lazy loading sequence to free VRAM before loading DiT.",
        "ko": "캐시된 텍스트 인코더 출력을 디스크에 저장. DiT 로드 전 VRAM 해제를 위한 지연 로딩 필수.",
    },
    "skip_cache_check": {
        "en": "Skip validation of cached files on startup. Faster startup when caches are known to be valid.",
        "ko": "시작 시 캐시 파일 검증 건너뛰기. 캐시가 유효함을 알 때 빠른 시작.",
    },

    # Paths
    "pretrained_model_name_or_path": {
        "en": "Path to the base DiT model weights (.safetensors).",
        "ko": "기본 DiT 모델 가중치 경로 (.safetensors).",
    },
    "qwen3": {
        "en": "Path to the Qwen3 text encoder weights for text-to-image conditioning.",
        "ko": "텍스트-투-이미지 컨디셔닝용 Qwen3 텍스트 인코더 가중치 경로.",
    },
    "vae": {
        "en": "Path to the VAE model for image encoding/decoding.",
        "ko": "이미지 인코딩/디코딩용 VAE 모델 경로.",
    },
    "output_dir": {
        "en": "Directory for saving trained LoRA checkpoints.",
        "ko": "학습된 LoRA 체크포인트 저장 디렉토리.",
    },
    "output_name": {
        "en": "Base filename for saved checkpoints (epoch number is appended automatically).",
        "ko": "저장되는 체크포인트의 기본 파일명 (에폭 번호 자동 추가).",
    },
    "save_model_as": {
        "en": "Checkpoint format. safetensors: recommended (fast, safe).",
        "ko": "체크포인트 형식. safetensors: 권장 (빠르고 안전).",
    },
}


def field_help(key: str) -> str | None:
    """Return the help string for *key* in the current language, or None."""
    entry = FIELD_HELP.get(key)
    if entry is None:
        return None
    lang = current_language()
    return entry.get(lang) or entry.get("en")


# ── LoRA variant guide (rich HTML) ────────────────────────────

LORA_GUIDE: dict[str, str] = {
    "en": (
        "<p><b>LoRA</b> &mdash; Classic low-rank adaptation. Adds small trainable matrices "
        "(down &times; up) to existing weight layers.<br>"
        "<code>y = x + (x @ down @ up) &times; scale &times; multiplier</code><br>"
        "Simple, effective, and the default choice for most fine-tuning tasks.</p>"

        "<p><b>DoRA</b> &mdash; Weight-Decomposed LoRA. Separates each weight matrix into "
        "<i>magnitude</i> (per-output-channel scalar) and <i>direction</i> (unit-norm matrix). "
        "LoRA adapts direction while a learned <code>dora_scale</code> adjusts magnitude.<br>"
        "Often more stable than standard LoRA at higher ranks. Enable with <code>use_dora = true</code>.</p>"

        "<p><b>OrthoLoRA</b> &mdash; Orthogonal LoRA. Uses QR-decomposed orthonormal bases "
        "with learned singular values: <code>P @ diag(&lambda;) @ Q</code>. "
        "Includes orthogonality regularization to keep updates structured.<br>"
        "Linear layers only; incompatible with DoRA.</p>"

        "<p><b>T-LoRA</b> &mdash; Timestep-dependent rank masking. The effective LoRA rank changes "
        "with the denoising timestep via a power-law schedule:<br>"
        "&bull; High noise (early steps) &rarr; full rank (maximum expressiveness)<br>"
        "&bull; Low noise (late steps) &rarr; reduced rank (down to <code>min_rank</code>)<br>"
        "Compatible with both LoRA and DoRA. Enable with <code>use_timestep_mask = true</code>.</p>"
    ),
    "ko": (
        "<p><b>LoRA</b> &mdash; 클래식 저랭크 적응. 기존 가중치 레이어에 작은 학습 가능한 "
        "행렬(down &times; up)을 추가.<br>"
        "<code>y = x + (x @ down @ up) &times; scale &times; multiplier</code><br>"
        "간단하고 효과적이며, 대부분의 파인튜닝에 기본 선택.</p>"

        "<p><b>DoRA</b> &mdash; 가중치 분해 LoRA. 각 가중치 행렬을 "
        "<i>크기</i>(출력 채널별 스칼라)와 <i>방향</i>(단위 노름 행렬)으로 분리. "
        "LoRA가 방향을 적응하고, 학습된 <code>dora_scale</code>이 크기를 조정.<br>"
        "높은 랭크에서 표준 LoRA보다 안정적. <code>use_dora = true</code>로 활성화.</p>"

        "<p><b>OrthoLoRA</b> &mdash; 직교 LoRA. QR 분해된 정규 직교 기저와 "
        "학습된 특이값 사용: <code>P @ diag(&lambda;) @ Q</code>. "
        "업데이트 구조 유지를 위한 직교성 정규화 포함.<br>"
        "선형 레이어만 지원; DoRA와 비호환.</p>"

        "<p><b>T-LoRA</b> &mdash; 타임스텝 의존 랭크 마스킹. 디노이징 타임스텝에 따라 "
        "유효 LoRA 랭크가 거듭제곱 스케줄로 변동:<br>"
        "&bull; 높은 노이즈 (초기 스텝) &rarr; 전체 랭크 (최대 표현력)<br>"
        "&bull; 낮은 노이즈 (후기 스텝) &rarr; 축소된 랭크 (<code>min_rank</code>까지)<br>"
        "LoRA와 DoRA 모두와 호환. <code>use_timestep_mask = true</code>로 활성화.</p>"
    ),
}


def lora_guide() -> str:
    """Return the LoRA variant guide HTML for the current language."""
    lang = current_language()
    return LORA_GUIDE.get(lang) or LORA_GUIDE["en"]
