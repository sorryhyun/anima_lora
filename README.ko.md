# anima_lora

[English](README.md) · 📖 [**가이드북 (Windows 초보자용 한국어 종합 가이드)**](docs/guidelines/가이드북.md)

[Anima](https://huggingface.co/circlestone-labs/Anima) 디퓨전 모델(DiT 기반, flow-matching)을 위한 LoRA / T-LoRA 학습 및 추론 엔진.

> 처음 사용하시나요? [**가이드북**](docs/guidelines/가이드북.md)이 CUDA 설치 → 데이터셋 준비 → 학습 → ComfyUI 배포까지 전 과정을 Windows 초보자 관점에서 안내합니다.

이 저장소가 지향하는 세 가지:

1. **빠른 LoRA 학습** — 컴파일러 친화적인 데이터 파이프라인을 엔드-투-엔드로 튜닝하여 소비자용 GPU에서 동작.
2. **검증된 머지 호환 변형** — LoRA, OrthoLoRA, T-LoRA가 한 세트로 스택되고, 독립형 DiT 체크포인트로 그대로 구워넣을 수 있음.
3. **넓은 실험적 기능 표면** — HydraLoRA, ReFT, APEX distillation, postfix/prefix tuning, 임베딩 인버전, img2emb, Spectrum 추론, modulation guidance.

> **한눈에 보는 구조도** (DiT 내부, LoRA, OrthoLoRA, T-LoRA, HydraLoRA, ReFT, Spectrum, modulation, 컴파일 최적화)는 [`docs/structure_images_korean/`](docs/structure_images_korean/)에 있습니다. 글로 된 해설은 [`docs/structure/`](docs/structure/) 참고.

---

## 1. 빠른 학습

**12.8 GB 피크 VRAM · 1.3 s/iter** — 단일 RTX 5060 Ti 기준. 데이터 파이프라인 · 어텐션 · 컴파일러 스택을 함께 설계하여 Dynamo가 학습 전체에서 단일 정적 shape만 보게 만든 결과:

| 레버 | 요약 |
|---|---|
| 고정 토큰 버켓팅 | 모든 버킷을 `(H/16)×(W/16) ≈ 4096` 패치로 맞추고, 배치를 정확히 4096 토큰으로 제로 패딩. 단일 정적 shape → 재컴파일 없음. |
| Max-padded 텍스트 인코더 | 텍스트 출력을 512로 패딩 후 제로 필링. 사전학습된 DiT는 이 제로 키를 cross-attn sink로 사용하므로 패딩을 제거하면 동작이 깨짐. 컴파일러에 또 다른 고정 차원도 제공. |
| 블록별 `torch.compile` | 각 DiT 블록을 Inductor로 독립 컴파일. 고정 토큰 수와 결합하여 guard 재컴파일을 제거. |
| 컴파일 친화적 핫패스 | 모든 forward 경로에서 dynamo가 깔끔하게 추적하기 어려운 패턴을 제거 — `einops.rearrange`는 명시적 `.unflatten()/.permute()` 체인으로, `torch.autocast` 컨텍스트 매니저는 직접 `.to(dtype)` 캐스팅으로, dict `.items()` 루프는 컴파일 영역 밖으로 호이스트, FA4는 `@torch.compiler.disable`로 래핑하여 clean graph break 유도. |
| Flash Attention 2 | `flash_attn` 2.x, SDPA 자동 폴백. FA4는 평가 후 제거 — [fa4.md](docs/optimizations/fa4.md). |

**벤치마크** — RTX 5060 Ti 16GB, LoRA rank 32, bs 2, 182 steps, seed 42. Val loss는 σ ∈ {0.05, 0.1, 0.2, 0.35}에서 측정.

| 구성 | 피크 VRAM | 총 시간 | 2번째 에포크 | Train | Val |
|---|---|---|---|---|---|
| FA2 (plain) | 7.0 GB | 14:51 | 7:26 | 0.092 | 0.212 |
| FA2 + compile (eager fallback) | 7.7 GB | 15:10 | 7:26 | 0.089 | 0.211 |
| FA2 + compile (고정 토큰) | 6.2 GB | 11:07 | 5:01 | 0.086 | 0.193 |
| FA2 + compile − grad ckpt | 15.2 GB | **7:07** | **3:30** | 0.088 | 0.206 |
| **same, custom autograd** | **12.8 GB** | **6:40** | **3:15** | 0.090 | 0.212 |

> CUDA 13.2에서는 **1.2 s/iter**, 15.5 GB 피크. PyTorch 2.12 릴리스 시 지원 예정. 자세한 내용은 [cuda132.md](docs/optimizations/cuda132.md).

컴파일 파이프라인 상세는 [docs/optimizations/for_compile.md](docs/optimizations/for_compile.md).

---

## 2. 검증된 머지 호환 변형

기본 학습 설정은 **LoRA + OrthoLoRA + T-LoRA**를 함께 스택합니다. 세 변형 모두 저장 시점의 thin-SVD 내보내기를 통해 독립형 DiT 체크포인트로 무손실 병합되므로, 별도 어댑터 로더 없이 ComfyUI 호환 `*_merged.safetensors`를 그대로 배포할 수 있습니다.

| 변형 | 요약 | 상세 |
|---|---|---|
| **LoRA** | 고전 low-rank, rank 16–32. | — |
| **OrthoLoRA** | SVD 파라미터화 + 직교성 정규화. 저장 시 일반 LoRA로 내보냄. | [psoft-integrated-ortholora.md](docs/methods/psoft-integrated-ortholora.md) |
| **T-LoRA** | 타임스텝 의존 랭크 마스킹 — 고노이즈 구간은 저랭크, 저노이즈 구간은 풀 랭크. 마스크가 학습 전용이라 머지 결과는 비트 동일. | [timestep_mask.md](docs/methods/timestep_mask.md) |

**사이드 바이 사이드** — 동일 프롬프트, `er_sde` 30 스텝, `cfg=4.0`, 1024². 각 LoRA는 rank 16, 2 에포크, 20% 서브셋, 학습 seed 42로 학습했고 추론 seed는 `{41, 42, 43}`. 재현은 `python scripts/bench_methods.py`.

|  | **LoRA** | **OrthoLoRA + T-LoRA** |
|:---:|:---:|:---:|
| seed 41 | <img src="bench/side_by_side/lora/20260423-154854-014_41_.png" width="320"> | <img src="bench/side_by_side/ortho_tlora/20260423-155545-258_41_.png" width="320"> |
| seed 42 | <img src="bench/side_by_side/lora/20260423-154938-584_42_.png" width="320"> | <img src="bench/side_by_side/ortho_tlora/20260423-155631-762_42_.png" width="320"> |
| seed 43 | <img src="bench/side_by_side/lora/20260423-155024-080_43_.png" width="320"> | <img src="bench/side_by_side/ortho_tlora/20260423-155718-280_43_.png" width="320"> |

<details>
<summary>베이스 모델 및 개별 변형 (plain, OrthoLoRA, T-LoRA)</summary>

|  | **plain (베이스)** | **OrthoLoRA** | **T-LoRA** |
|:---:|:---:|:---:|:---:|
| seed 41 | <img src="bench/side_by_side/plain/20260423-160513-382_41_.png" width="240"> | <img src="bench/side_by_side/ortholora/20260423-155109-338_41_.png" width="240"> | <img src="bench/side_by_side/tlora/20260423-155327-834_41_.png" width="240"> |
| seed 42 | <img src="bench/side_by_side/plain/20260423-160556-697_42_.png" width="240"> | <img src="bench/side_by_side/ortholora/20260423-155155-526_42_.png" width="240"> | <img src="bench/side_by_side/tlora/20260423-155413-304_42_.png" width="240"> |
| seed 43 | <img src="bench/side_by_side/plain/20260423-160640-759_43_.png" width="240"> | <img src="bench/side_by_side/ortholora/20260423-155241-905_43_.png" width="240"> | <img src="bench/side_by_side/tlora/20260423-155458-996_43_.png" width="240"> |

</details>

**머지**:

```bash
make merge                                  # output/ckpt 내 최신 LoRA를 배율 1.0으로 구워넣음
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8
```

Linear 가중치 델타가 아닌 변형(ReFT / HydraLoRA `_moe` / postfix / prefix)은 기본적으로 머지 거부. `--allow-partial`로 넘기면 해당 파트를 drop하고 LoRA 부분만 구워냅니다.

---

## 3. 실험적 기능

각 항목마다 전용 문서가 있습니다 — 사용법, 플래그, 주의사항은 링크 참고.

| 기능 | 설명 | 문서 |
|---|---|---|
| **HydraLoRA** | MoE 스타일 멀티헤드 라우팅: 공유 `lora_down`, 전문가별 `lora_up_i`, 레이어 로컬 라우터. `AnimaAdapterLoader` ComfyUI 노드 필요. | [hydra-lora.md](docs/methods/hydra-lora.md) |
| **ReFT** | 블록 단위 residual-stream intervention (LoReFT, NeurIPS 2024). 어떤 LoRA 변형과도 조합 가능. | [reft.md](docs/methods/reft.md) |
| **APEX** | 학습된 condition shift를 활용한 self-adversarial 1–4 NFE distillation. 판별자 · 외부 teacher 불필요. | [apex.md](docs/methods/apex.md) |
| **Postfix / prefix tuning** | 어댑터 cross-attention에 연속 벡터를 뒤에(postfix) 또는 앞에(prefix) 붙임. postfix 변형 5종. | [postfix-sigma.md](docs/methods/postfix-sigma.md), [prefix-tuning.md](docs/methods/prefix-tuning.md) |
| **임베딩 인버전** | frozen DiT를 통과시켜 타깃 이미지에 맞도록 텍스트 임베딩을 최적화. | [invert.md](docs/methods/invert.md) |
| **img2emb 리샘플러** | TIPSv2-L/14 features + anchor injection을 이용한 참조 이미지 → 임베딩 매핑 학습. | [scripts/img2emb/README.md](scripts/img2emb/README.md) |
| **Spectrum 추론** | Chebyshev 특성 예측으로 학습 없이 약 3.75× 가속 (Han et al., CVPR 2026). 별도 안정판 ComfyUI 노드: [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler). | [spectrum.md](docs/methods/spectrum.md) |
| **Modulation guidance** | AdaLN 계수를 조향하는 `pooled_text_proj` MLP distillation (Starodubcev et al., ICLR 2026). | [mod-guidance.md](docs/methods/mod-guidance.md) |
| **GRAFT** | 리젝션 샘플링 파인튜닝 — 학습 → 생성 → survivor 큐레이션 → 재학습 루프. | [graft-guideline.md](docs/guidelines/graft-guideline.md) |

---

## 설치

```bash
uv sync                   # Python 3.13 with pre-built flash attention 2
hf auth login
make download-models      # DiT + Qwen3 텍스트 인코더 + QwenImage VAE를 models/로
# 학습 이미지를 image_dataset/에 배치 (.txt 캡션 사이드카 함께)
make gui                  # 추천 — 설정 에디터 + 데이터셋 브라우저 + 학습 모니터
```

CLI 경로:

```bash
make preprocess           # VAE 호환 리사이즈 및 검증
make lora                 # 또는: make lora-fast / lora-low-vram / make postfix / make apex
make test                 # 최신 학습된 LoRA로 샘플 생성
```

설정 체인: `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI 인자`. `PRESET=low_vram make lora` 또는 `--network_dim 32 --max_train_epochs 64` 형태로 오버라이드. 전체 플래그는 [docs/guidelines/training.md](docs/guidelines/training.md), [docs/guidelines/inference.md](docs/guidelines/inference.md)에.

---

## 문서

| 문서 | 내용 |
|------|------|
| [guidelines/training.md](docs/guidelines/training.md) | 학습 플래그, LoRA 변형, 캡션 셔플, 마스크 로스, 데이터셋 설정 |
| [guidelines/inference.md](docs/guidelines/inference.md) | 추론 플래그, P-GRAFT, 프롬프트 파일, LoRA 포맷 변환 |
| [guidelines/graft-guideline.md](docs/guidelines/graft-guideline.md) | GRAFT 큐레이션 워크플로우 |
| [optimizations/](docs/optimizations/) | 컴파일 파이프라인, FA4 회고, CUDA 13.2 |
| [methods/](docs/methods/) | 각 방법별 전용 문서 — APEX, HydraLoRA, ReFT, Spectrum, 인버전, mod guidance, postfix/prefix, T-LoRA, OrthoLoRA |

---

## 라이선스

툴킷 코드: [MIT](LICENSE).

Anima / CircleStone **베이스 모델 가중치**는 **CircleStone Labs Non-Commercial License v1.0**에 따라 배포되며, 본 저장소가 재라이선스하지 않습니다. 해당 가중치로부터 본 툴킷으로 학습한 모든 LoRA · 파인튜닝 · 머지 체크포인트는 파생물로 간주되어 비상업 조항을 승계합니다. 자세한 내용은 [NOTICE](NOTICE).
