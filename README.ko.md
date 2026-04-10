# anima_lora

Anima 디퓨전 모델(DiT 기반, flow-matching)을 위한 LoRA 학습 및 추론 엔진. Standard LoRA, DoRA, OrthoLoRA, T-LoRA(타임스텝 의존 랭크 마스킹)를 지원합니다.

## 하이라이트

**15.2 GB 피크 VRAM · 1.3 s/iter** — 단일 소비자용 GPU에서 달성. 데이터 파이프라인, 어텐션, 컴파일러 스택을 함께 설계하여 최적화:

| 최적화 | 설명 |
|---|---|
| **고정 토큰 버켓팅** | 모든 버킷 해상도를 `(H/16)×(W/16) ≈ 4096` 패치가 되도록 선택. 모든 배치 요소를 정확히 4096 토큰으로 제로 패딩하여 `torch.compile`에 단일 정적 shape를 제공 — 종횡비가 달라져도 재컴파일 없음. |
| **Max-padded 텍스트 인코더** | 텍스트 인코더 출력을 `max_length`(512)로 패딩 후 제로 필링. 사전학습된 DiT는 이 제로 키들을 cross-attention softmax의 학습된 **attention sink**로 활용하므로, 패딩을 제거하면 검은 이미지가 생성됨. 패딩 유지가 모델 동작을 보존하면서 컴파일러에 또 다른 고정 차원을 제공. |
| **Cross-attention KV 트림** | 일반적인 캡션은 512 토큰 중 30–80개만 사용 — ~85%가 제로 패딩. KV를 버켓 길이(64/128/256/512)로 트림하고 LSE 기반 시그모이드 보정으로 attention-sink 기여를 정확하게 복원. **~4배 적은 cross-attention 연산**, 품질 손실 없음, 컴파일 안전(4가지 shape만 존재). |
| **Flash Attention 4** | `flash_attn.cute` (Hopper GPU 최적화 FA4 커널)을 고정 길이 및 가변 길이 어텐션 모두에 사용. 공식 FA4는 소비자 Blackwell(SM120)을 지원하지 않음 — [sisgrad의 SM120 브랜치](https://github.com/sisgrad/flash-attention/tree/dz/sm120_tma_optimized)의 버그 수정 [포크](https://github.com/sorryhyun/flash-attention-sm120-fix) 사용. FA2/SDPA로 자동 폴백. |
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
| FA4 + compile (고정 토큰) | 6.3 GB | 11:01 | 5:17 | 0.089 | 0.204 |
| + fp32 누적 | 6.4 GB | 10:57 | 5:15 | 0.089 | 0.196 |
| + DoRA + fp32 누적 | 6.4 GB | 12:04 | 5:25 | 0.092 | 0.204 |
| + T-LoRA + fp32 누적 | 6.9 GB | 12:57 | 5:44 | 0.093 | 0.210 |

하위 3행은 FA4 + compile (고정 토큰)을 기준으로 측정.

## 설치

```bash
# 1. 의존성 설치 (Python 3.13)
uv sync

# 2. Hugging Face 인증 (모델 다운로드에 필요)
hf auth login

# 3. 모델 가중치 다운로드 (DiT, 텍스트 인코더, VAE)
make download-models

# 4. 학습 이미지를 image_dataset/에 배치 (.txt 캡션 사이드카 파일 함께)

# 5. 이미지 전처리 (VAE 호환 리사이즈 및 검증)
make preprocess
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

모든 학습은 TOML 설정 파일 기반. HF Accelerate로 실행:

```bash
accelerate launch --mixed_precision bf16 train.py --config_file configs/training_config.toml
```

CLI에서 설정값 오버라이드 가능:

```bash
accelerate launch --mixed_precision bf16 train.py --config_file configs/training_config.toml \
    --network_dim 32 --max_train_epochs 64 --learning_rate 2e-5
```

### 제공 설정 파일

| 설정 파일 | 설명 |
|--------|-------------|
| `configs/training_config.toml` | Standard LoRA (rank 32, 64 epochs) |
| `configs/training_config_dora.toml` | DoRA (rank 16) |
| `configs/training_config_doratimestep.toml` | DoRA + T-LoRA 타임스텝 마스킹 |
| `configs/dataset_config.toml` | 동적 버켓팅 데이터셋 레이아웃 |

### 주요 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|-----------|---------|-------------|
| `network_dim` | 16–32 | LoRA 랭크 |
| `network_alpha` | = dim | LoRA 스케일링 alpha |
| `learning_rate` | 2e-5 | 기본 학습률 |
| `optimizer_type` | AdamW8bit | AdamW8bit, Lion, DAdapt, Prodigy |
| `max_train_epochs` | 4–64 | 학습 에포크 수 |
| `mixed_precision` | bf16 | bf16, fp16, 또는 no |
| `attn_mode` | flash | flash, torch, xformers |
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

## 상세 문서

| 문서 | 내용 |
|------|------|
| [docs/training.md](docs/training.md) | LoRA 변형 (DoRA, OrthoLoRA, T-LoRA), KV 트림, 캡션 셔플, 마스크 로스, 데이터셋 설정 |
| [docs/inference.md](docs/inference.md) | 추론 플래그, P-GRAFT 추론, 프롬프트 파일 형식, LoRA 포맷 변환 |
