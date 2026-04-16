# anima_lora

[English](README.md)

Anima 디퓨전 모델(DiT 기반, flow-matching)을 위한 LoRA 학습 및 추론 엔진. Standard LoRA, OrthoLoRA, T-LoRA(타임스텝 의존 랭크 마스킹)를 지원합니다.

## 하이라이트

**15.2 GB 피크 VRAM · 1.3 s/iter** — 단일 소비자용 GPU에서 달성. 데이터 파이프라인, 어텐션, 컴파일러 스택을 함께 설계하여 최적화:

| 최적화 | 설명 |
|---|---|
| **고정 토큰 버켓팅** | 모든 버킷 해상도를 `(H/16)×(W/16) ≈ 4096` 패치가 되도록 선택. 모든 배치 요소를 정확히 4096 토큰으로 제로 패딩하여 `torch.compile`에 단일 정적 shape를 제공 — 종횡비가 달라져도 재컴파일 없음. |
| **Max-padded 텍스트 인코더** | 텍스트 인코더 출력을 `max_length`(512)로 패딩 후 제로 필링. 사전학습된 DiT는 이 제로 키들을 cross-attention softmax의 학습된 **attention sink**로 활용하므로, 패딩을 제거하면 검은 이미지가 생성됨. 패딩 유지가 모델 동작을 보존하면서 컴파일러에 또 다른 고정 차원을 제공. |
| **Flash Attention 2** | 고정 길이 및 가변 길이 어텐션 모두에 `flash_attn` 2.x를 사용하며, SDPA로 자동 폴백. FA4는 평가 후 제거됨 — [docs/optimizations/fa4.md](docs/optimizations/fa4.md) 참고. |
| **블록별 `torch.compile`** | 각 DiT 블록을 Inductor 백엔드로 독립적으로 컴파일. 고정 토큰 수와 결합하여 Dynamo guard 재컴파일을 완전히 제거. |
| **디스크 캐싱 (latent & 텍스트 임베딩)** | VAE latent, 텍스트 인코더 출력, LLM 어댑터 출력을 사전 계산 후 디스크에 캐싱 — VAE와 텍스트 인코더가 학습 VRAM을 전혀 차지하지 않음. |
| **Unsloth 그래디언트 체크포인팅** | Forward pass 중 활성화를 non-blocking 전송으로 CPU에 오프로드하고 backward pass에서 다시 스트리밍 — PCIe 대역폭과 VRAM을 교환. |

## 벤치마크

RTX 5060 Ti 16GB에서 테스트. LoRA rank=32, lr=5e-5, batch_size=2, epochs=2 (182 steps), seed=42.
검증 로스는 고정 시드, 타임스텝 sigma = {0.05, 0.1, 0.2, 0.35}로 측정.
gradient_checkpointing=true, unsloth_offload_checkpointing=true, latent과 텍스트 임베딩은 디스크 캐싱.

| 구성 | 피크 VRAM | 총 시간 | 2번째 에포크 | Train Loss | Val Loss |
|---|---|---|---|---|---|
| FA2 (plain) | 7.0 GB | 14:51 | 7:26 | 0.092 | 0.212 |
| FA2 + compile (eager fallback) | 7.7 GB | 15:10 | 7:26 | 0.089 | 0.211 |
| FA2 + compile (고정 토큰) | 6.2 GB | 11:07 | 5:01 | 0.086 | 0.193 |
| FA2 + compile - grad ckpt | 15.2 GB | **7:07** | **3:30** | 0.088 | 0.206 |
| FA2 + compile - grad ckpt (fast, rank 32) | 15.6 GB | 6:20 | 2:59 | 0.09 | 0.212 |

## 설치

```bash
# 1. 의존성 설치 (Python 3.13)
uv sync

# 2. Hugging Face 인증 (모델 다운로드에 필요)
hf auth login

# 3. 모델 가중치 다운로드 (DiT, 텍스트 인코더, VAE)
make download-models

# 4. 학습 이미지를 image_dataset/에 배치 (.txt 캡션 사이드카 파일 함께)

# 5. 필요 시 GUI 실행
make gui

# 또는 CLI에서 전처리부터 시작 (VAE 호환 리사이즈 및 검증)
make preprocess
make lora
```

선택사항: Flash Attention 지원을 위해 `flash-attn` 설치.

### 모델 가중치

`make download-models`로 [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)에서 `models/`로 자동 다운로드:

| 파일 | 경로 |
|------|------|
| Anima DiT | `models/diffusion_models/anima-preview3-base.safetensors` |
| Qwen3 0.6B 텍스트 인코더 | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

## 학습

학습 설정은 세 단계 체인 구조: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI 인자`. 겹치는 값은 method가 우선. HF Accelerate로 실행:

```bash
accelerate launch --mixed_precision bf16 train.py --method lora --preset default
```

CLI에서 설정값 오버라이드 가능:

```bash
accelerate launch --mixed_precision bf16 train.py --method tlora --preset low_vram \
    --network_dim 32 --max_train_epochs 64 --learning_rate 2e-5
```

### Method 파일 (`configs/methods/`)

| Method | 설명 |
|--------|-------------|
| `lora` | 표준 LoRA (rank 16) |
| `tlora` | OrthoLoRA + 타임스텝 마스킹 |
| `hydralora` | HydraLoRA 멀티헤드 라우팅 |
| `postfix` / `postfix_exp` / `postfix_func` | Postfix tuning 변형 |
| `prefix` | Prefix tuning |
| `graft` | GRAFT 루프 기본 설정 |

### Preset 파일 (`configs/presets.toml`)

| Preset | 설명 |
|--------|-------------|
| `default` | 리눅스 기본 / Windows 16GB (`blocks_to_swap=8`) |
| `fast_16gb` | swap 없음 + `layer_start=2` (16GB 카드용) |
| `low_vram` | gradient checkpointing + unsloth offload (Windows 8GB 겸용) |
| `graft` | GRAFT 전용 swap 예산 |

### 주요 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|-----------|---------|-------------|
| `network_dim` | 16–32 | LoRA 랭크 |
| `network_alpha` | = dim | LoRA 스케일링 alpha |
| `learning_rate` | 2e-5 | 기본 학습률 |
| `optimizer_type` | AdamW8bit | AdamW8bit, Lion, DAdapt, Prodigy |
| `max_train_epochs` | 4–64 | 학습 에포크 수 |
| `mixed_precision` | bf16 | bf16, fp16, 또는 no |
| `attn_mode` | flash | flash (FA2), torch, xformers, flex, sageattn |
| `gradient_checkpointing` | true | 메모리 효율적 역전파 |
| `cache_latents_to_disk` | true | VAE latent 디스크 캐싱 |
| `cache_text_encoder_outputs` | true | 텍스트 인코더 출력 캐싱 |

## 추론

```bash
python inference.py \
    --dit ../models/diffusion_models/anima-preview3-base.safetensors \
    --text_encoder ../models/text_encoders/qwen_3_06b_base.safetensors \
    --vae ../models/vae/qwen_image_vae.safetensors \
    --lora_weight ../output/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 50 \
    --guidance_scale 3.5 \
    --save_path ../output/images
```

## 임베딩 인버전

타깃 이미지에 일치하도록 텍스트 임베딩을 최적화 — frozen DiT를 통해 역전파. 모델이 임베딩 공간에서 이미지를 어떻게 해석하는지 보여줍니다.

```bash
make invert                    # 전처리된 데이터셋에 대해 배치 인버전
make invert INVERT_SWAP=12     # 저-VRAM GPU용 블록 스왑 사용
```

초기화, VRAM 모드, 블록 그래디언트 분석에 대한 자세한 내용은 [docs/methods/invert.md](docs/methods/invert.md)를 참고.

## 상세 문서

| 문서 | 내용 |
|------|------|
| [docs/guidelines/training.md](docs/guidelines/training.md) | LoRA 변형 (OrthoLoRA, T-LoRA), 캡션 셔플, 마스크 로스, 데이터셋 설정 |
| [docs/optimizations/fa4.md](docs/optimizations/fa4.md) | FA4 / flash-attention-sm120과 cross-attention KV 트림이 제거된 이유 |
| [docs/methods/prefix-tuning.md](docs/methods/prefix-tuning.md) | Prefix 튜닝 — 12 GB VRAM, ~1 step/s, 작동 원리, 설정 레퍼런스 |
| [docs/guidelines/inference.md](docs/guidelines/inference.md) | 추론 플래그, P-GRAFT 추론, 프롬프트 파일 형식, LoRA 포맷 변환 |
| [docs/methods/invert.md](docs/methods/invert.md) | 임베딩 인버전 — 최적화 플래그, VRAM 모드, 블록 그래디언트 로깅 |
