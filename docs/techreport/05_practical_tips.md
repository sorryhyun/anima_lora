# 5. 실전 팁: 16 GB 카드에서 파이프라인 굴리기

> 초안 (2026-11-12). 측정 날짜가 오래된 것은 날짜를 함께 적었다. 확인하지 못한 부분은 `TODO(확인 필요)`로
> 남겼다.

**요약.** 이 섹션은 1–4장(LoRA → soup → bake/int8 → turbo)에 공통으로 걸리는 엔지니어링 팁을 모은다.
결과를 망가뜨리는 모델 불변식(TE padding, 5D latent), 기본값인 fp32 residual stream, torch.compile과
free-fit 버킷의 결합, 16 GB에서 VRAM을 맞추는 순서(activation budget → block swap → grad checkpointing),
추론 쪽 설정, 그리고 run noise를 넘는 평가를 내리기 위한 실험 위생을 다룬다. 각 팁은 무엇을 할지, 왜인지,
그걸 알려준 수치나 실패 순으로 적는다.

기준 환경은 RTX 5070 Ti 16 GB, RAM 64 GB(사용 가능 55–58 GB) + swap 8 GB다. 아래 수치는 별도 표기가
없으면 이 카드에서 잰 것이다.

---

## 5.1 깨도 에러가 안 나는 모델 불변식

Anima 라인의 불변식은 대부분 위반해도 예외가 나지 않는다. 결과만 틀어진다. MOSS / Qwen 라인의 불변식은
반대인 경우가 많다(예: MOSS는 TE를 절대 padding하지 않는다).

### TE 출력은 512 토큰까지 zero-padding한 채로 쓴다

- **할 일.** 학습과 추론 모두 TE 출력을 512 토큰까지 padding한다. padding 위치를 attention mask로 가리지
  않는다.
- **이유.** 사전학습된 Anima는 zero-padding 위치를 cross-attention의 attention sink("볼 곳이 없을 때 보는
  곳")로 쓰도록 학습됐다. caption 길이로 잘라내거나 mask로 지우면 softmax 분모가 무너지고
  cross-attention이 포화되어 **검은 이미지**가 나온다.
- **부수 효과.** KV 길이가 항상 512로 고정되므로 cross-attention 쪽은 compile shape가 공짜로 안정된다.
  실제 caption은 30–80 토큰만 채운다.

### Latent는 5D `(B, C, T=1, H, W)`, 바깥은 4D

DiT만 frame 축(T=1)이 있는 5D latent를 받고, VAE encode와 캐시된 latent, 학습 루프는 4D다.

### 블록 수는 checkpoint에서 읽는다

- 깊이는 checkpoint의 블록 수가 정한다. 40-block Anima-2.9B도 같은 방식으로 읽힌다.
- **LoRA는 깊이에 묶인다.** 40-block LoRA를 28-block base에 올리면 뒤쪽 블록이 경고 하나만 남기고
  버려진다.
- int8 GPTQ 파일도 원본 checkpoint에 묶이므로, 2.9B나 bake한 DiT는 자기 GPTQ 파일이 필요하다(3장 참조).
- 2.9B 학습은 16 GB에서 기본 설정으로 바로 OOM이고(bf16 가중치만 5.9 GB), block swap 16은 한 스텝 뒤
  OOM, gradient checkpointing에 swap 0이 동작한다(4.6 → 2.2 s/it). 추론은 832×1216, 32 step, CFG 4에서
  이미지당 약 31.5 s로 28-block base(약 24.5 s)보다 28% 느리다.

---

## 5.2 fp32 residual stream (기본값)

### 무엇이 fp32로 도는가

fp32 stream은 block GEMM 바깥을 fp32로 돌린다. block linear와 FA2는 bf16(또는 int8) 그대로이고,
파라미터 dtype은 바뀌지 않는다.

| fp32 | bf16 / int8 그대로 |
|---|---|
| residual stream `x`, modulation, q / k norm, RoPE, embedder, head | block linear(qkv, output_proj, cross q, mlp), FA2, LoRA 자체 가중치 |

구현은 embedder / head / modulation linear를 호출 시에만 weight를 fp32로 cast하는 linear(autocast 밖)로,
q / k norm을 fp32 RMSNorm으로 바꾸고 나머지 DiT forward를 bf16 autocast로 감싼다. 그래서 block linear
입력은 bf16으로 반올림되고 int8 linear는 fp32 kernel 입력을 받는다. 파라미터가 늘지 않으므로 가중치
VRAM은 그대로다.

### 왜 쓰는가: 수치

예측 오차 `‖v − v_fp32‖ / ‖v_fp32‖`(%, LoRA 없음, 6장 평균):

| σ | bf16 stream | fp32 stream | int8 GPTQ + bf16 kernel 입력(ComfyUI stock) | int8 GPTQ + fp32 stream |
|---|---|---|---|---|
| 0.1 | 1.76 | 0.52 | 2.29 | 1.32 |
| 0.5 | 0.85 | 0.22 | 0.99 | 0.48 |
| 0.9 | 1.15 | 0.23 | 1.07 | 0.50 |

- fp32 stream은 bf16 GEMM 주변 예측 오차를 1.5–2.1배, int8 base 주변 오차를 21–36% 줄인다. LoRA
  gradient는 bf16 GEMM 주변에서 블록 14–27이 개선되고, int8 주변에서는 섞여 있다.
- fp16 stream을 쓰지 않는 이유: stream은 블록 12–20에서 261k–285k까지 올라가 fp16 최대값(65504)의
  4.3배다. all-fp16 forward는 모든 이미지에서 블록 12부터 NaN이 된다. block GEMM 입출력은 ≤ 747로 fp16에
  충분히 들어간다. 즉 fp16 범위를 넘는 것은 stream뿐이다.

### 비용

- 추론(8-step, 768², smoke 1회): bf16 261 ms/step, 4.11 GiB → fp32 280 ms/step, 4.17 GiB.
- Turbo 학습(int8 stream mode, budget 0.99): bf16 stream 6.28 s/step, 11.87 GB → fp32 6.7–7.0 s/step,
  12.39 GB.
- fp32 stream의 saved set이 더 크므로, bf16 시절의 activation budget 표는 그대로 옮기지 않는다.

### 주의: 같은 seed가 다른 그림을 낸다

noise 자체는 같지만(fp32 draw가 bf16 draw로 반올림됨) 최종 latent가 768²에서 6–12 step 5회 실행 기준
12–24% 움직인다. bf16 시절 렌더와 seed 단위로 비교하지 않는다.

### "눈에 보이는 이득"에 대해

사용자가 렌더를 보고 "일관되게 낫다(글자 렌더링, 손가락 위치)"고 평가한 것은 **MOSS** 라인의 fp32
stream이다. Anima 쪽 근거는 위의 수치(예측 오차, gradient)다. 2026-10-31 리포트는 "Anima keeps its bf16
stream"으로 끝났지만, 그 뒤 기본값이 fp32로 바뀌었다.

> TODO(확인 필요): Anima에서 fp32 vs bf16 stream을 렌더로 A/B한 기록(작은 구조, 글자, 손가락 기준)을
> 찾지 못했다. 있으면 링크하고, 없으면 "Anima는 수치 근거로 기본값을 옮겼다"로 문장을 정리한다.

---

## 5.3 torch.compile

### free-fit 버킷과 dynamic seq compile은 한 묶음

- 이미지는 원래 비율을 유지하고, 토큰 수가 tier별 band 안 어디든 떨어진다.

  | edge | 토큰 family |
  |---|---|
  | 512 | 1008, 1024 |
  | 768 | 2160 |
  | 896 | 3000, 3024 |
  | 1024 | 4032, 4200 |
  | 1280 | 6300 |
  | 1536 | 8640 |

- band 안의 토큰 수마다 static graph를 만들면 폭발하므로, seq 축만 dynamic으로 표시하고 tier의 범위로
  묶어서 **tier당 graph 하나**가 된다.

### compile 설정은 기본값이 맞다

2026-09-09, 기본 LoRA 학습(70장, 210 step) 실측:

| 설정 | s/it (warm) | 첫 스텝 cold / warm 캐시 | peak |
|---|---|---|---|
| torch 2.12 default | 0.588 | 17.1 s / 9.6 s | 13.9 GB |
| torch 2.14 default | 0.584 (−0.7%) | 19.4 s / 10.0 s | 13.9 GB |
| 2.12 `max-autotune` | 평균 1.032, 중앙 0.556, p90 2.77 s | 97 s | 15.1 GB |
| 2.12 + `combo_kernels=True` | 0.597 (+1.5%) | 16.9 s | 13.9 GB |

- `max-autotune`은 run 중간에 다시 autotune해서 평균이 1.7배 나빠지고 VRAM이 1.2 GB 늘어난다.
- `combo_kernels`는 2.12 / 2.14 모두 기본 off이고, 켜면 1.5–2.4% 느리다.
- torch 2.14의 NVGEMM은 sm_120(RTX 50)에서 `cudaErrorNoKernelImageForDevice`로 전부 실패한다.
  `cutlass.operators` 0.2.0이 sm100 계열 커널만 가지고 있다. 릴리스 노트의 "Blackwell"은 B200을
  가리킨다.

---

## 5.4 VRAM과 속도

### 레버를 쓰는 순서

16 GB에서 OOM이 나면 아래 순서로 내린다. 앞의 것일수록 속도 손해가 적다.

1. **activation memory budget** (기본 0.99). 안 맞으면 더 내린다.
2. **int8 base**. 3장 참조.
3. **block swap**.
4. **gradient checkpointing**.

### 1. activation memory budget

- AOT autograd의 min-cut partitioner가 backward용으로 무엇을 저장할지 정한다. 1.0 미만이면 knapsack으로 재계산이 효율적인
  중간값을 골라 backward에서 다시 계산한다.
- 기본값은 0.99다(turbo도 동일). free-fit 워크로드에서 1.0은 OOM, 0.99는 약 13.1 GB로 학습된다
  (2026-07-04).
- int8 stream mode 기준 budget sweep:

  | budget | ms/step | peak |
  |---|---|---|
  | 0.99 | 427 | 10.87 GB |
  | 0.4 | 434–439 | 10.00 GB |
  | 0.3 | 472 | 6.87 GB |
  | 0.1 | 511 | 5.32 GB |

  "16 GB 아래에서는 budget을 먼저 내려라. block swap보다 peak를 더 많이 깎는다." 다만 0.99 미만으로 끝까지
  학습한 LoRA는 아직 없다.

**배운 경위.** 2026-06-10, 수치상 아무 영향이 없다고 확인된 LoRA down-projection `autograd.Function`을
지웠더니 다음 no-grad-ckpt 학습이 step 0에서 OOM이 났다. custom Function은 `save_for_backward`로 저장 집합을
고정하고 있었고, 지우자 partitioner가 블록마다 약 0.8 GB를 더 저장했다. config 수준의 추측 다섯 개가 모두
빗나갔고, 인접 commit 두 개에서 26-step 실제 job을 돌린 bisect가 몇 분 만에 원인을 찾았다. 교훈:
**수치적으로 무해하다고 메모리도 무해한 것은 아니다.** `autograd.Function` / checkpoint wrapper를 지울
때는 출력뿐 아니라 저장 텐서 크기를 비교한다.

### 2. int8 base (3장으로)

block GEMM을 int8로 돌리면 bf16 593 ms → int8 full 433 ms(1.37배), 최종 loss는 0.0737로 같다
(ama mitsuki, 848 step). stream mode는 `Wᵀ` 사본을 host에 두어 full보다 peak가 1.4 GB 낮다
(12.38 → 10.96 GB). mode 선택, GPTQ 파일, bake한 DiT의 전용 GPTQ는 3장이 다룬다.

### 3. Block swap

- block swap은 모든 int8 mode에서 동작하고, swap한 stack의 출력과 `dX`가 resident stack과 bit 단위로
  같다.
- LoRA, int8 stream mode, fp32 stream, budget 0.99, 106 step:

  | swap | median | peak |
  |---|---|---|
  | 0 | 451 ms | 11.47 GB |
  | 14 | 475 ms | 10.44 GB |
  | 24 | 524 ms | 9.70 GB |

- **turbo에서는 비싸다.** 매 forward가 swap을 되감기 때문에 한 스텝에 swap된 블록이 여러 번 이동한다.
  30 step, CDM on, GAN off: swap 0에서 5.8 s/step, 12.36 GB, swap 14에서 8.4 s/step, 11.32 GB. 약 1.5배
  시간으로 1 GB를 산다. 그래서 turbo는 swap을 끈다.
- swap을 켜면 같은 seed라도 다른 샘플을 뽑는다. swap / resident run의 loss를 step 단위로 비교하지 않는다
  (epoch 평균 loss 0.0749 vs 0.0733).

### 4. Gradient checkpointing

- 2026-06-10 워크로드에서 grad ckpt는 step time을 약 18% 늘렸다.

### 기본 학습과 turbo의 실측 peak

| 작업 | 설정 | 속도 | peak |
|---|---|---|---|
| LoRA (기본 설정) | bf16 base, torch 2.12, 2026-09 | 0.588 s/it | 13.9 GB |
| LoRA | int8 stream mode, fp32 stream, swap 0 | 451 ms | 11.47 GB |
| turbo | int8 stream mode, fp32 stream, warm start | 6.7–7.0 s/step | 12.39 GB |
| turbo 2-step smoke | int8 GPTQ, fp32 stream | – | 12.35 GB |
| turbo 4-step arm 0 | merged14 teacher, 250 it | 5.7–5.8 s/it, 약 27분 | – |

> TODO(확인 필요): 요청서에 있던 "turbo peak ~13.5 GB"의 출처를 찾지 못했다. 문서에 있는 turbo peak는
> 12.35–12.39 GB(swap 0)이고, GAN on / swap 14에서 11.35 GB다. GAN on, swap 0의 peak 실측이 있으면
> 여기에 추가한다.

### 옵티마이저: fused AdamW

- 기본 optimizer는 fused AdamW다.
- bitsandbytes `AdamW8bit`는 파라미터 블록마다 커널을 하나씩 띄워 optimizer 구간에서 GPU 사용률이 88%로
  떨어졌다. 한 스텝 1.17 s 중 308 ms가 optimizer였고, fused AdamW로 바꾸자 0.86 s가 됐다(RTX 5060 Ti,
  당시 static-pad 설정). 대가는 optimizer state +180 MB 정도(rank 32 LoRA, 약 30M 파라미터).

### RTX 50용 FlashAttention 빌드

- sm_120 전용 패치된 FA2 2.8.3. Anima 기준 fp16-accumulate backward가 step의 5.1%, fp16 input이 3.1%를
  줄인다. peak VRAM은 같다.
- head dim 64 / 128만 지원한다. Anima `llm_adapter`의 64는 stock backward로 간다.

### fp16은 쓰지 않는다 (아직)

같은 카드에서 fp32 누적 fp16 GEMM은 bf16과 속도가 같고(약 102 TFLOPS), fp16 누적만 1.6배다. 그래도 int8
`_int_mm`(약 330 TFLOPS)보다 느리고, backward는 loss scale 2^20 근처가 필요하다. 출하된 fp16 계획은 없다.

### RAM

int8 stream mode는 `Wᵀ` 사본을, block swap은 블록을 host에 두므로 RAM을 더 쓴다.

---

## 5.5 추론 쪽

- int8 base에서는 LoRA가 P-GRAFT hook으로 붙는다. 정적으로 merge하면 버려질 bf16 가중치에 접히기
  때문이다. bf16 base를 int8 base로 바꾸면 생성 픽셀이 약 4/255 움직인다.

### 학습 없는 추론 스택 (요약)

| 기능 | 한 줄 |
|---|---|
| Spectrum | Chebyshev feature 예측으로 캐시된 step에서 28 블록을 모두 건너뛴다 |
| CNS | SDE noise를 아직 해상되지 않은 주파수 대역으로 재분배. **er_sde 전용**, euler에서는 no-op |
| mod-guidance | 학습된 `pooled_text_proj`로 AdaLN을 조절. 전역 톤만 바꾸는 레버 |
| DAVE | 같은 프롬프트의 seed 간 다양성 회복 |

### Turbo LoRA 추론

- euler, 4 step, CFG 1.0. CFG는 증류 때 student에 구워졌으므로 1.0을 유지한다. 두 번 guide하지 않는다.
- 추론의 flow shift 3.0은 turbo 학습의 flow shift 3.0과 맞는다. 하나를 바꾸면 다른 것도 바꾼다.
- **Spectrum과 쌓지 않는다.** Spectrum의 Chebyshev 캐시는 ≥16 step을 가정한다. CNS도 er_sde 전용이라 turbo의
  euler 4-step에는 해당이 없다. mod-guidance는 turbo student에서 확인되지 않았다.
- checkpoint가 학습한 step 수보다 더 많은 step에서 더 좋아 보이면, 아직 진짜 N-step 맵에 도달하지 못했다는
  신호다.
- 공개된 4-step student는 `huggingface.co/sorryhyun/anima-turbo-4step`. 증류 자체는 4장.

---

## 5.6 실험 위생

### 평가 기준은 결과를 보기 전에 정한다

- gate는 근거가 통과시킨다. 모든 지표와 렌더가 한 arm을 가리키면 차이가 작아도 통과다. 기준은 결과를 읽기
  전에 정하고, 구조적으로 작을 수밖에 없는 값(낮은 σ에서 one-step x̂₀ 차이처럼 x_t가 이미 이미지를 담고
  있는 경우)에는 threshold를 걸지 않는다.
- 예: turbo A/B 프로토콜은 "arm의 EMA 렌더가 두 prompt 모두에서 control–twin 차이보다 크게 이기고, S–T4
  nW2²가 |control − twin|보다 크게 낮으면 통과"를 run 전에 적어 두었다. gradient norm 쪽 결정 규칙(pooled
  top/bottom ≤ 1.5 → adaptive clip alone)도 미리 고정됐고, arm 0 결과(1.43)는 그 규칙대로 읽혔다.

### null twin으로 noise floor를 먼저 잰다

- raw checkpoint 하나와 다른 하나를 비교하면 대부분 run noise를 읽는다. control을 seed만 바꿔 한 번 더 돌린
  twin이 이후 모든 arm의 noise floor가 된다.
- arm 0 실측: seed만 다른 두 run의 ΔW가 공유하는 비율은 cos 0.09–0.10이다. 목적함수나 LR이 다른 쌍도
  0.09–0.11이었다. 즉 **ΔW overlap으로는 arm을 구분할 수 없다.** 평가는 렌더와 `moment_gap`으로 한다.
- 같은 run 안에서도 |ΔW|가 √t로 자라고 125-step 이동끼리 직교한다(cos 0.018 / 0.028). 상수 LR에서
  random walk다. EMA(window 0.2)가 배포 대상인 이유다.
- **raw checkpoint는 발산 burst 안에 있을 수 있다.** control의 raw 250은 judge 12장 전부 평평한 noise
  texture였다. 마지막 4 step에서 student gradient norm이 231 / 244 / 106 / 72로 튀었고, 이 recipe에서
  burst는 약 250 iteration에 한 번 나온다. EMA 250은 정상 렌더였다.
- twin은 한 번만 돌린다. 통과한 arm은 새 control이 되고, step time이나 loss 항이 바뀌지 않는 한 새 twin을
  만들지 않는다.

### 렌더가 gate다

- 첫 checkpoint에서 grid를 렌더한다. APEX turbo arm은 loss, `x_pred` std, rollout std가 모두 평평하고
  건강했는데 4-step grid는 warm start보다 흐려지고 있었다. 흐린 샘플도 latent std는 정상이다.
- turbo의 `fm_mse`는 품질과 역상관이다. checkpoint 순위는 렌더한 4-step 샘플로 매긴다. pose 다양성은
  grid에서만 보이고, PE-pooled 지표는 pose를 못 본다.
- 평가용 prompt는 쉬우면 안 된다. ANIMA 표지판, nozomi prompt는 arm을 가르기에 너무 쉬워서 turbo는 어려운
  prompt(이름 있는 다섯 명 + position clause, 일본어 메뉴 + 영어 간판 카페, 아래에서 본 점프 구도)를
  768 × 1344에서 teacher 옆에 놓고 본다.

### 작은 것부터, 한 번만

- 메커니즘 질문은 full run 전에 약 25분짜리 micro arm으로 답한다. 이전 세션에서 끝난 측정은 다시 재지 않고
  그 리포트 위에 쌓는다. job을 큐에 넣기 전에 "이 결과가 어떤 결정을 바꾸는가"를 말할 수 있어야 한다.
  없으면 코드를 읽는다.
- 기존 checkpoint나 렌더가 control이다. code drift나 recipe 변경으로 불공정하면 결과 옆에 적고, 새 control은
  treatment 뒤에 follow-up으로 제안한다.
- 원인을 모를 때는 config 추측보다 실제 job의 bisect가 빠르다. 5.4절의 partitioner OOM은 추측 다섯 개가
  빗나가고, 인접 commit 두 개의 26-step run이 몇 분 만에 끝냈다.

### 논문은 PDF로 읽는다

웹 요약은 표, 식, 부록을 버린다. 예: "MFD는 1-step용"이라는 인상과 달리, PDF의 T2I 결과는 SANA 4-step,
3000 step이었다.
