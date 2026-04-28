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
    "use_timestep_mask": {
        "en": "Enable T-LoRA: effective rank varies with denoising timestep via power-law schedule. Full rank at high noise, reduced at low noise.",
        "ko": "T-LoRA 활성화: 디노이징 타임스텝에 따라 유효 랭크 변동. 높은 노이즈에서 전체 랭크, 낮은 노이즈에서 축소.",
    },
    "use_ortho": {
        "en": "Enable OrthoLoRA: SVD-based orthogonal parameterization of the update matrix (linear layers only). Regularizes toward structured updates; saved as plain LoRA via thin SVD at checkpoint time.",
        "ko": "OrthoLoRA 활성화: 업데이트 행렬의 SVD 기반 직교 파라미터화 (선형 레이어 전용). 구조화된 업데이트로 정규화되며, 저장 시 thin SVD로 일반 LoRA로 변환.",
    },
    "use_hydra": {
        "en": "Enable HydraLoRA: MoE-style multi-head routing with shared lora_down and per-expert lora_up heads. Produces a *_moe.safetensors sibling for router-live inference. Requires cache_llm_adapter_outputs=true.",
        "ko": "HydraLoRA 활성화: 공유 lora_down + 전문가별 lora_up 헤드를 가진 MoE 스타일 멀티헤드 라우팅. 라우터-라이브 추론용 *_moe.safetensors 동반 파일 생성. cache_llm_adapter_outputs=true 필요.",
    },
    "num_experts": {
        "en": "HydraLoRA expert count. More experts = more capacity but more VRAM and slower training. Typical: 2–8.",
        "ko": "HydraLoRA 전문가 수. 많을수록 표현력 증가하지만 VRAM 사용량 증가 및 학습 속도 감소. 일반적: 2–8.",
    },
    "balance_loss_weight": {
        "en": "HydraLoRA load-balancing loss weight. Discourages router collapse onto a single expert. Typical: 0.01.",
        "ko": "HydraLoRA 부하 균형 손실 가중치. 라우터가 단일 전문가로 붕괴되는 것을 방지. 일반적: 0.01.",
    },
    "balance_loss_warmup_ratio": {
        "en": "Fraction of training steps to hold the balance loss at 0 before activating it. Lets the router specialize first, then switches the penalty on to stop further collapse of a diverged router. 0.0 disables the warmup. Typical: 0.3–0.5.",
        "ko": "밸런스 손실을 0으로 유지하는 학습 스텝 비율. 먼저 라우터가 전문화되도록 한 뒤 페널티를 활성화해 분화된 라우터의 추가 붕괴를 방지. 0.0 = 비활성화. 일반적: 0.3–0.5.",
    },
    "add_reft": {
        "en": "Enable ReFT: block-level residual-stream intervention (Wu et al. 2024). Adds R^T·(ΔW·h + b)·scale to each selected DiT block's output. Composes with any LoRA variant.",
        "ko": "ReFT 활성화: 블록 수준 잔차 스트림 개입 (Wu et al. 2024). 선택된 DiT 블록 출력에 R^T·(ΔW·h + b)·scale 추가. 모든 LoRA 변형과 함께 사용 가능.",
    },
    "reft_dim": {
        "en": "ReFT intervention rank — dimension of R and ΔW in each ReFTModule. Typical: 32–64.",
        "ko": "ReFT 개입 랭크 — 각 ReFTModule의 R 및 ΔW 차원. 일반적: 32–64.",
    },
    "reft_alpha": {
        "en": "ReFT scaling factor (effective scale = alpha / dim). Typical: same as reft_dim.",
        "ko": "ReFT 스케일링 계수 (실효 스케일 = alpha / dim). 일반적: reft_dim과 동일.",
    },
    "reft_layers": {
        "en": "Which DiT blocks receive ReFT modules. 'all', 'last_8', 'first_4', 'stride_2', or comma-separated indices like '3,7,11,15'.",
        "ko": "ReFT 모듈이 적용될 DiT 블록. 'all', 'last_8', 'first_4', 'stride_2', 또는 '3,7,11,15'와 같은 쉼표 구분 인덱스.",
    },
    "use_sigma_router": {
        "en": "Add a tiny sinusoidal(σ)→E bias MLP to each HydraLoRA router, letting expert routing vary with denoising timestep. Zero-init at final layer → step-0 identical to base HydraLoRA.",
        "ko": "각 HydraLoRA 라우터에 sinusoidal(σ)→E 바이어스 MLP 추가하여 타임스텝에 따라 전문가 라우팅 변동. 최종 레이어 zero-init → 초기에는 기본 HydraLoRA와 동일.",
    },
    "sigma_feature_dim": {
        "en": "Sinusoidal σ feature dimension fed into the σ-router bias MLP. Typical: 128.",
        "ko": "σ 라우터 바이어스 MLP에 입력되는 sinusoidal σ 특징 차원. 일반적: 128.",
    },
    "sigma_hidden_dim": {
        "en": "σ-router bias MLP hidden dimension. Typical: 128.",
        "ko": "σ 라우터 바이어스 MLP 히든 차원. 일반적: 128.",
    },
    "sigma_router_layers": {
        "en": "Regex over layer names — only matching layers get a σ-conditional router branch. Typical: limit to cross_attn.q_proj and self_attn.qkv_proj where σ-signal lives.",
        "ko": "레이어 이름에 대한 정규식 — 일치하는 레이어만 σ-조건부 라우터 분기 추가. 일반적: σ 신호가 있는 cross_attn.q_proj 및 self_attn.qkv_proj로 제한.",
    },
    "per_bucket_balance_weight": {
        "en": "Extra per-σ-bucket load-balance penalty, scaled by balance_loss_weight. Encourages routing diversity within each timestep bucket. Typical: 0.3.",
        "ko": "σ 버킷별 추가 부하 균형 페널티, balance_loss_weight로 스케일. 각 타임스텝 버킷 내 라우팅 다양성 유도. 일반적: 0.3.",
    },
    "num_sigma_buckets": {
        "en": "Number of timestep buckets used for per-bucket balance accounting. Typical: 3 (low / mid / high noise).",
        "ko": "버킷별 균형 계산에 사용되는 타임스텝 버킷 수. 일반적: 3 (저/중/고 노이즈).",
    },
    "network_args": {
        "en": "Extra kwargs passed to the network module. For postfix: list of 'key=value' strings (e.g., 'mode=cond-timestep', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Pick a Variant to auto-fill.",
        "ko": "네트워크 모듈에 전달되는 추가 kwargs. postfix의 경우 'key=value' 문자열 리스트 (예: 'mode=cond-timestep', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Variant 선택으로 자동 채우기 가능.",
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
        "en": "Base learning rate for the optimizer. Typical: 1e-5 to 1e-4.",
        "ko": "옵티마이저 기본 학습률. 일반적: 1e-5 ~ 1e-4.",
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
    "source_image_dir": {
        "en": (
            "Where raw images and .txt captions live. The Preprocess button feeds "
            "this to resize_images.py (writes resized PNGs) and "
            "cache_text_embeddings.py (caches captions). Override per preset/method "
            "if you keep multiple datasets side by side."
        ),
        "ko": (
            "원본 이미지와 .txt 캡션이 있는 디렉토리. 전처리 버튼이 이 경로를 "
            "resize_images.py(리사이즈된 PNG 저장)와 cache_text_embeddings.py"
            "(캡션 캐시)에 전달합니다. 여러 데이터셋을 병행할 때 프리셋/메소드별로 "
            "오버라이드하세요."
        ),
    },
    "resized_image_dir": {
        "en": (
            "Where preprocess writes VAE-aligned PNGs. Also resolved into the dataset "
            "subset's image_dir at training time (via {resized_image_dir} template "
            "in base.toml), so editing this propagates to both preprocess and training."
        ),
        "ko": (
            "전처리가 VAE에 맞춰 리사이즈한 PNG를 저장하는 디렉토리. 학습 시 "
            "데이터셋 서브셋의 image_dir로도 사용됩니다(base.toml의 "
            "{resized_image_dir} 템플릿 치환). 이 값을 바꾸면 전처리와 학습 양쪽에 "
            "반영됩니다."
        ),
    },
    "lora_cache_dir": {
        "en": (
            "Where preprocess writes VAE latent (.npz) and text-encoder "
            "(_anima_te.safetensors) caches. Also resolved into the dataset subset's "
            "cache_dir at training time."
        ),
        "ko": (
            "전처리가 VAE 잠재 변수(.npz)와 텍스트 인코더 출력"
            "(_anima_te.safetensors) 캐시를 저장하는 디렉토리. 학습 시 데이터셋 "
            "서브셋의 cache_dir로도 사용됩니다."
        ),
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

APPLY_NOTE_HTML: dict[str, str] = {
    "en": (
        "<div style='background:#1e2a33; padding:10px 14px; border-left:3px solid #6aa4d8; "
        "margin-bottom:14px; border-radius:3px;'>"
        "<p style='margin:0 0 6px 0;'><b>What does <span style='color:#e67e22;'>Apply</span> do?</b></p>"
        "<p style='margin:0 0 6px 0;'>It fills the form below with a known-good set of values "
        "for the picked variant — the right toggles (<code>use_ortho</code>, <code>use_hydra</code>, "
        "<code>add_reft</code>, …), rank defaults, <code>output_name</code>, σ-router knobs, etc. "
        "It also strips fields the previous variant owned but this one doesn't "
        "(e.g. <code>num_experts</code> disappears when you switch hydralora → plain lora).</p>"
        "<p style='margin:0; color:#f0c14b;'><b>Nothing is saved until you click "
        "<span style='color:#fff;'>Save</span>.</b> Review the filled-in values first, tweak "
        "whatever you want, then save to overwrite <code>configs/methods/&lt;method&gt;.toml</code>.</p>"
        "</div>"
    ),
    "ko": (
        "<div style='background:#1e2a33; padding:10px 14px; border-left:3px solid #6aa4d8; "
        "margin-bottom:14px; border-radius:3px;'>"
        "<p style='margin:0 0 6px 0;'><b><span style='color:#e67e22;'>Apply</span>는 정확히 무엇을 하나요?</b></p>"
        "<p style='margin:0 0 6px 0;'>선택한 variant의 검증된 값 세트로 아래 폼을 채웁니다 — "
        "토글(<code>use_ortho</code>, <code>use_hydra</code>, <code>add_reft</code> 등), "
        "랭크 기본값, <code>output_name</code>, σ-router 파라미터 등. "
        "이전 variant가 소유했지만 현재 variant에는 없는 필드는 자동으로 제거됩니다 "
        "(예: hydralora → plain lora로 바꾸면 <code>num_experts</code>가 사라짐).</p>"
        "<p style='margin:0; color:#f0c14b;'><b><span style='color:#fff;'>Save</span>를 "
        "누르기 전까지는 디스크에 저장되지 않습니다.</b> 채워진 값을 먼저 확인하고 필요하면 "
        "수정한 뒤 Save를 눌러 <code>configs/methods/&lt;method&gt;.toml</code>에 기록하세요.</p>"
        "</div>"
    ),
}


LORA_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>LoRA Variants</h2>"
        "<p><b>LoRA</b> &mdash; Classic low-rank adaptation. Adds small trainable matrices "
        "(down &times; up) to existing weight layers.<br>"
        "<code>y = x + (x @ down @ up) &times; scale &times; multiplier</code><br>"
        "Simple, effective, and the default choice for most fine-tuning tasks.</p>"
        "<p><b>OrthoLoRA</b> &mdash; Orthogonal LoRA. Uses QR-decomposed orthonormal bases "
        "with learned singular values: <code>P @ diag(&lambda;) @ Q</code>. "
        "Includes orthogonality regularization to keep updates structured.<br>"
        "Linear layers only. Enable with <code>use_ortho = true</code>.</p>"
        "<p><b>T-LoRA</b> &mdash; Timestep-dependent rank masking. The effective LoRA rank changes "
        "with the denoising timestep via a power-law schedule:<br>"
        "&bull; High noise (early steps) &rarr; full rank (maximum expressiveness)<br>"
        "&bull; Low noise (late steps) &rarr; reduced rank (down to <code>min_rank</code>)<br>"
        "Enable with <code>use_timestep_mask = true</code>.</p>"
        "<p><b>HydraLoRA</b> &mdash; MoE-style routing: shared <code>lora_down</code> plus "
        "<code>num_experts</code> per-expert <code>lora_up</code> heads, routed layer-locally "
        "from the adapted Linear's input. Produces a <code>*_moe.safetensors</code> sibling "
        "used by <code>make test-hydra</code>.<br>"
        "Enable with <code>use_hydra = true</code>. Requires <code>cache_llm_adapter_outputs = true</code>.</p>"
        "<p><b>σ-router</b> &mdash; Add-on to HydraLoRA: a tiny sinusoidal(σ)&rarr;E bias MLP "
        "in each router so expert choice can vary with denoising timestep. Zero-init at the "
        "final layer &rarr; step-0 identical to base HydraLoRA.<br>"
        "Enable with <code>use_sigma_router = true</code>.</p>"
        "<p><b>ReFT</b> &mdash; Block-level residual-stream intervention (Wu et al., NeurIPS 2024). "
        "One module per selected DiT block adds <code>R^T &middot; (&Delta;W&middot;h + b) &middot; scale</code> "
        "to the block's output &mdash; an additive side-channel that composes with any LoRA variant and "
        "lives in the same <code>.safetensors</code>.<br>"
        "Enable with <code>add_reft = true</code>; pick blocks via <code>reft_layers</code> "
        "(e.g. <code>last_8</code>).</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>LoRA 변형</h2>"
        "<p><b>LoRA</b> &mdash; 클래식 저랭크 적응. 기존 가중치 레이어에 작은 학습 가능한 "
        "행렬(down &times; up)을 추가.<br>"
        "<code>y = x + (x @ down @ up) &times; scale &times; multiplier</code><br>"
        "간단하고 효과적이며, 대부분의 파인튜닝에 기본 선택.</p>"
        "<p><b>OrthoLoRA</b> &mdash; 직교 LoRA. QR 분해된 정규 직교 기저와 "
        "학습된 특이값 사용: <code>P @ diag(&lambda;) @ Q</code>. "
        "업데이트 구조 유지를 위한 직교성 정규화 포함.<br>"
        "선형 레이어만 지원. <code>use_ortho = true</code>로 활성화.</p>"
        "<p><b>T-LoRA</b> &mdash; 타임스텝 의존 랭크 마스킹. 디노이징 타임스텝에 따라 "
        "유효 LoRA 랭크가 거듭제곱 스케줄로 변동:<br>"
        "&bull; 높은 노이즈 (초기 스텝) &rarr; 전체 랭크 (최대 표현력)<br>"
        "&bull; 낮은 노이즈 (후기 스텝) &rarr; 축소된 랭크 (<code>min_rank</code>까지)<br>"
        "<code>use_timestep_mask = true</code>로 활성화.</p>"
        "<p><b>HydraLoRA</b> &mdash; MoE 스타일 라우팅: 공유 <code>lora_down</code> + "
        "<code>num_experts</code>개의 전문가별 <code>lora_up</code> 헤드를 적응된 Linear의 "
        "입력으로부터 레이어 로컬하게 라우팅. <code>make test-hydra</code>에 사용되는 "
        "<code>*_moe.safetensors</code> 동반 파일 생성.<br>"
        "<code>use_hydra = true</code>로 활성화. <code>cache_llm_adapter_outputs = true</code> 필요.</p>"
        "<p><b>σ-router</b> &mdash; HydraLoRA 확장: 각 라우터에 sinusoidal(σ)&rarr;E 바이어스 MLP를 "
        "추가하여 전문가 선택이 디노이징 타임스텝에 따라 변동. 최종 레이어 zero-init &rarr; "
        "초기에는 기본 HydraLoRA와 동일.<br>"
        "<code>use_sigma_router = true</code>로 활성화.</p>"
        "<p><b>ReFT</b> &mdash; 블록 수준 잔차 스트림 개입 (Wu et al., NeurIPS 2024). "
        "선택된 각 DiT 블록에 하나의 모듈이 <code>R^T &middot; (&Delta;W&middot;h + b) &middot; scale</code>을 "
        "블록 출력에 추가 &mdash; 모든 LoRA 변형과 함께 사용 가능한 추가 사이드 채널이며, "
        "동일한 <code>.safetensors</code>에 저장됨.<br>"
        "<code>add_reft = true</code>로 활성화; <code>reft_layers</code>로 블록 선택 "
        "(예: <code>last_8</code>).</p>"
    ),
}


def lora_guide() -> str:
    """Return the LoRA variant guide HTML for the current language."""
    lang = current_language()
    return LORA_GUIDE.get(lang) or LORA_GUIDE["en"]


POSTFIX_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>Postfix / Prefix Variants</h2>"
        "<p><b>postfix</b> &mdash; Appends N learned vectors at the end of the "
        "adapter cross-attention sequence (<code>mode=postfix</code>). Simple "
        "continuous-token tuning — the vectors are pure learned parameters, independent "
        "of the caption.</p>"
        "<p><b>postfix_exp (cond)</b> &mdash; Caption-conditional MLP injection "
        "(<code>mode=cond</code>). A small MLP turns the pooled caption embedding into "
        "the postfix vectors, so the appended tokens vary with the text.</p>"
        "<p><b>postfix_func</b> &mdash; Same as <i>postfix_exp</i> plus a functional MSE "
        "loss against saved inversion runs. Needs <code>inversion_dir</code> populated "
        "and forces <code>attn_mode=flash</code> / <code>trim_crossattn_kv=false</code>.</p>"
        "<p><b>postfix_sigma (cond-timestep)</b> &mdash; Caption-conditional base plus "
        "a zero-init σ-conditional residual (<code>mode=cond-timestep</code>). Training "
        "starts identical to <i>postfix_exp</i>; σ-dependence only emerges if gradients "
        "push it. Not compatible with <code>--spectrum</code> or tiled inference.</p>"
        "<p><b>prefix</b> &mdash; Prepends N vectors to the cross-attention sequence "
        "(<code>mode=prefix</code>). No base LoRA required.</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>Postfix / Prefix 변형</h2>"
        "<p><b>postfix</b> &mdash; 어댑터 크로스 어텐션 시퀀스 끝에 N개의 학습된 벡터를 "
        "추가 (<code>mode=postfix</code>). 간단한 연속 토큰 튜닝 — 벡터는 캡션과 무관한 "
        "순수 학습 파라미터.</p>"
        "<p><b>postfix_exp (cond)</b> &mdash; 캡션 조건부 MLP 주입 "
        "(<code>mode=cond</code>). 작은 MLP가 풀링된 캡션 임베딩을 postfix 벡터로 변환하여, "
        "추가되는 토큰이 텍스트에 따라 달라짐.</p>"
        "<p><b>postfix_func</b> &mdash; <i>postfix_exp</i> + 저장된 inversion 결과에 대한 "
        "functional MSE 손실. <code>inversion_dir</code> 필요, "
        "<code>attn_mode=flash</code> / <code>trim_crossattn_kv=false</code> 강제.</p>"
        "<p><b>postfix_sigma (cond-timestep)</b> &mdash; 캡션 조건부 베이스 + "
        "zero-init σ-조건부 잔차 (<code>mode=cond-timestep</code>). 학습 시작 시 "
        "<i>postfix_exp</i>와 동일; σ-의존성은 그래디언트가 발생시킬 때만 발현. "
        "<code>--spectrum</code> 및 타일 추론과 비호환.</p>"
        "<p><b>prefix</b> &mdash; 크로스 어텐션 시퀀스 앞쪽에 N개의 벡터 추가 "
        "(<code>mode=prefix</code>). 베이스 LoRA 불필요.</p>"
    ),
}


def postfix_guide() -> str:
    """Return the postfix/prefix variant guide HTML for the current language."""
    lang = current_language()
    return POSTFIX_GUIDE.get(lang) or POSTFIX_GUIDE["en"]


def apply_note() -> str:
    """HTML block explaining Apply semantics — shown above variant guides."""
    lang = current_language()
    return APPLY_NOTE_HTML.get(lang) or APPLY_NOTE_HTML["en"]


def method_guide(method: str) -> str | None:
    """Right-panel default HTML for *method*, or None if no guide is registered."""
    if method == "lora":
        return apply_note() + lora_guide()
    if method == "postfix":
        return apply_note() + postfix_guide()
    return None
