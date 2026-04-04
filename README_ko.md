# anima_lora

Anima 디퓨전 모델(DiT 기반, flow-matching)을 위한 LoRA 학습 및 추론 엔진. Standard LoRA, DoRA, OrthoLoRA, T-LoRA(타임스텝 의존 랭크 마스킹)를 지원합니다.

## 하이라이트

**batch = 2 기준 15.2 GB 피크 VRAM · 2.4 s/iter** — 단일 소비자용 GPU에서 달성. 데이터 파이프라인, 어텐션, 컴파일러 스택을 함께 설계하여 최적화:

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
| FA2 + compile - grad ckpt | 15.2 GB | 7:07 | 3:30 | 0.088 | 0.206 |
| FA4 + compile (고정 토큰) | 6.3 GB | 11:01 | 5:17 | 0.089 | 0.204 |
| + fp32 누적 | 6.4 GB | 10:57 | 5:15 | 0.089 | 0.196 |
| + DoRA + fp32 누적 | 6.4 GB | 12:04 | 5:25 | 0.092 | 0.204 |
| + T-LoRA + fp32 누적 | 6.9 GB | 12:57 | 5:44 | 0.093 | 0.210 |

하위 3행은 FA4 + compile (고정 토큰)을 기준으로 측정.

## 설치

```bash
# 1. 의존성 설치 (Python 3.11+)
uv sync

# 2. Hugging Face 인증 (모델 다운로드에 필요)
huggingface-cli login

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
| Anima DiT | `models/diffusion_models/anima-preview2.safetensors` |
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

## LoRA 변형

### DoRA (Weight-Decomposed Low-Rank Adaptation)

가중치 업데이트에서 크기와 방향을 분리하여 낮은 랭크에서도 향상된 학습 효율을 달성. ([arXiv 2402.09353](https://arxiv.org/abs/2402.09353))

```toml
# network_args
use_dora = true
```

추론/병합 시 크기 벡터가 `dora_scale`로 내보내져 ComfyUI와 호환.

### OrthoLoRA

SVD 기반 직교 가중치 파라미터화. 제로 초기화 보장 및 직교성 정규화 포함.

```toml
# network_args
use_ortho = true
sig_type = "last"           # "principal", "last", 또는 "middle"
ortho_reg_weight = 0.01     # 직교성 페널티 가중치
```

참고: OrthoLoRA는 DoRA와 호환되지 않음. Linear 레이어만 지원 (Conv2d 미지원).

### T-LoRA (타임스텝 의존 랭크 마스킹)

디노이징 타임스텝에 따라 유효 LoRA 랭크를 동적 조정. 초기(고노이즈) 스텝은 전체 랭크를 사용하고, 후기 스텝은 감소된 랭크를 사용. LoRA와 DoRA 모두와 호환.

```toml
# network_args
use_timestep_mask = true
min_rank = 1                # 유지할 최소 랭크
alpha_rank_scale = 1.0      # 거듭제곱 법칙 스케줄 지수
```

랭크 스케줄:

```
r = ((max_t - t) / max_t) ^ alpha_rank_scale * (max_rank - min_rank) + min_rank
```

## FP32 누적

LoRA forward pass(down → up 프로젝션)를 fp32로 계산하여 수치 정밀도를 향상시킨 후 bf16으로 캐스트. 오버헤드 무시 가능; 학습 안정성을 위해 권장.

```toml
lora_fp32_accumulation = true
```

## Cross-Attention KV 트림

제로 패딩 토큰에 대한 cross-attention 낭비 연산을 제거. 사전학습 모델은 텍스트 인코더 출력을 512 토큰으로 패딩하지만, 일반적인 캡션은 30–80개만 사용 — 나머지는 attention sink로 작동하는 제로(softmax 분모에 `exp(0) = 1`을 기여하고 분자에는 0을 기여).

**작동 방식:** KV를 버켓 길이(`KV_BUCKETS = [64, 128, 256, 512]`)로 트림한 후 프로젝션하고, FA4가 반환하는 LSE(log-sum-exp)를 사용하여 attention-sink 기여를 정확히 복원하는 사후 시그모이드 보정을 적용:

```
out_corrected = out_trimmed * sigmoid(lse - log(N_pad))
```

이것은 전체 패딩 어텐션과 수학적으로 동일 — 근사가 아님. 버켓 트림 길이(4가지 shape만 존재)로 워밍업 후 `torch.compile` 재컴파일 없이 안정적으로 유지.

Flash Attention 4(`attn_mode = "flash4"`) 필요. 다른 백엔드는 자동으로 전체 512 길이 KV로 폴백.

## 캡션 셔플 변형

이미지당 에포크당 여러 셔플된 캡션 순열을 생성하고, 별도의 텍스트 인코더 출력으로 캐싱. 디스크 오버헤드 없이 캡션 다양성을 증가.

```toml
caption_shuffle_variants = 8    # 이미지당 변형 수
shuffle_caption = true
cache_text_encoder_outputs = true
```

스마트 셔플 알고리즘은 `@artist` 태그와 섹션 구분자(`On the ...`, `In the ...`)를 보존하면서 각 섹션 내의 태그를 셔플. 학습 중 배치 아이템당 하나의 변형이 랜덤 선택.

## 마스크 로스 (SAM / MIT)

공간 마스크를 사용하여 특정 영역(예: 텍스트 말풍선)을 학습 로스에서 제외.

### 마스크 생성

**SAM3** (Segment Anything Model):

```bash
python scripts/generate_masks.py \
    --config configs/sam_mask.yaml \
    --image-dir ../image_dataset \
    --mask-dir ../image_dataset/masks \
    --device cuda
```

**MIT** (Manga-Image-Translator / ComicTextDetector):

```bash
python scripts/generate_masks_mit.py \
    --image-dir ../image_dataset \
    --mask-dir ../image_dataset/masks \
    --device cuda \
    --detect-size 1024 \
    --text-threshold 0.5 \
    --dilate 5
```

두 방법 모두 그레이스케일 PNG를 생성: 255 = 학습, 0 = 제외.

### 학습에서 마스크 사용

```toml
# 학습 설정
masked_loss = true

# 데이터셋 설정 — 마스크 디렉토리 지정
[[datasets.subsets]]
image_dir = '../image_dataset'
mask_dir = '../image_dataset/masks'
```

마스크는 latent 공간 차원에 맞게 보간되어 로스에 요소별로 적용.

## 추론

```bash
python inference.py \
    --dit ../models/diffusion_models/anima-preview2.safetensors \
    --text_encoder ../models/text_encoders/qwen_3_06b_base.safetensors \
    --vae ../models/vae/qwen_image_vae.safetensors \
    --lora_weight ../output/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 50 \
    --guidance_scale 3.5 \
    --save_path ../output/images
```

### 주요 추론 플래그

| 플래그 | 기본값 | 설명 |
|------|---------|-------------|
| `--lora_weight` | — | LoRA 가중치 경로 (여러 개는 공백으로 구분) |
| `--lora_multiplier` | 1.0 | LoRA 강도 배율 |
| `--infer_steps` | 50 | 디노이징 스텝 수 |
| `--guidance_scale` | 3.5 | CFG 스케일 |
| `--flow_shift` | 5.0 | Flow-matching 스케줄 시프트 |
| `--sampler` | euler | euler (결정적 ODE) 또는 er_sde (확률적) |
| `--from_file` | — | 텍스트 파일에서 배치 프롬프트 |
| `--interactive` | false | 대화형 프롬프트 모드 |
| `--fp8` | false | DiT FP8 양자화 |
| `--compile` | false | torch.compile 속도 향상 |

### P-GRAFT 추론

LoRA를 정적 병합 대신 동적 훅으로 로드하여 디노이징 도중 LoRA를 끌 수 있음:

```bash
python inference.py ... \
    --pgraft \
    --lora_cutoff_step 37    # LoRA 활성: 스텝 0–36, 비활성: 37+
```

### 프롬프트 파일 형식

```
a girl standing in a field --w 1024 --h 1024 --s 50 --g 3.5
another prompt --seed 42 --flow_shift 4.0
```

## LoRA 포맷 변환

anima와 ComfyUI 키 포맷 간 변환:

```bash
python scripts/convert_lora_to_comfy.py input.safetensors output.safetensors          # anima → ComfyUI
python scripts/convert_lora_to_comfy.py --reverse input.safetensors output.safetensors  # ComfyUI → anima
```

## 데이터셋 설정

```toml
[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 3              # 셔플에서 처음 N개 토큰 보존

[[datasets]]
resolution = 1024
batch_size = 4
enable_bucket = true         # 동적 종횡비 버켓팅
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64
validation_split = 0.05
validation_seed = 42

  [[datasets.subsets]]
  image_dir = '../image_dataset'
  num_repeats = 1
```

각 이미지에는 같은 디렉토리에 대응하는 `.txt` 캡션 사이드카 파일이 필요합니다.

## 프로젝트 구조

```
anima_lora/
├── train.py                    # AnimaTrainer — 메인 학습 루프
├── inference.py                # 독립 실행형 이미지 생성
├── configs/                    # TOML 학습/데이터셋 설정
├── networks/
│   ├── lora_anima.py           # 네트워크 생성, 모듈 타겟팅, T-LoRA 로직
│   ├── lora_modules.py         # LoRA, DoRA, OrthoLoRA 모듈 구현
│   └── postfix_anima.py        # LLM 어댑터 continuous postfix 튜닝
├── library/
│   ├── anima_models.py         # Anima DiT 아키텍처
│   ├── anima_utils.py          # 모델 로딩/저장
│   ├── anima_train_utils.py    # 캡션 셔플, 로스 가중치, EMA, 검증
│   ├── strategy_anima.py       # 토크나이제이션/인코딩 전략 (Qwen3 + T5)
│   ├── train_util.py           # 학습 유틸리티 재내보내기 파사드
│   ├── custom_train_functions.py  # 마스크 로스 적용
│   ├── inference_utils.py      # Flow-matching 샘플러 (Euler, ER-SDE)
│   ├── attention.py            # 어텐션 백엔드 (flash, xformers, torch)
│   ├── config_util.py          # TOML 파싱 (Voluptuous)
│   ├── qwen_image_autoencoder_kl.py  # QwenImageVAE (WanVAE)
│   ├── datasets/               # 데이터셋 클래스, 버켓팅, 이미지 유틸
│   └── training/               # 옵티마이저/스케줄러/체크포인트 유틸리티
└── scripts/
    ├── generate_masks.py       # SAM3 텍스트 말풍선 마스킹
    ├── generate_masks_mit.py   # MIT/ComicTextDetector 마스킹
    ├── merge_masks.py          # 여러 마스크 결합
    └── convert_lora_to_comfy.py  # LoRA 포맷 변환
```
