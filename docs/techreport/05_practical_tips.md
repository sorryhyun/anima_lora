# 5. 실전 팁: 16 GB 카드에서 학습과 증류

> **요약.** 1–4장(LoRA → soup → bake / int8 → turbo)에 공통으로 걸리는 엔지니어링 팁을 모은다. bf16 GEMM 주변의
> residual stream을 fp32로 돌리면 예측 오차가 1.5–2.1배 줄고, step time은 7–11% 늘어난다. 원래 종횡비 그대로 학습할 때는
> 시퀀스 길이 축만 dynamic으로 compile하고 토큰 band로 묶으면 tier당 graph가 하나다. 16 GB에서 VRAM이 모자라면
> activation memory budget → int8 base → block swap → gradient checkpointing 순으로 내린다. 앞의 것일수록 속도
> 손해가 적다. 평가는 결과를 보기 전에 기준을 정하고, seed만 다른 twin으로 noise floor를 잰 뒤, 렌더로 한다.

수치는 별도 표기가 없으면 RTX 5070 Ti 16 GB에서 잰 것이다.

---

## 5.1 fp32 residual stream

### 무엇이 fp32로 도는가

block GEMM 바깥을 fp32로 돌린다. block linear와 attention은 bf16(또는 int8) 그대로이고, 파라미터 dtype은 바뀌지
않는다.

| fp32 | bf16 / int8 그대로 |
|---|---|
| residual stream `x`, modulation, q / k norm, RoPE, embedder, head | block linear(qkv, output proj, cross-attention q, MLP), attention, LoRA 가중치 |

embedder, head, modulation linear는 호출할 때만 weight를 fp32로 cast해서 autocast 밖에서 계산하고, q / k norm은
fp32 RMSNorm으로 바꾸고, 나머지 forward는 bf16 autocast로 감싼다. 그래서 block linear 입력은 bf16으로 반올림되고
int8 linear는 fp32 입력을 받는다. 파라미터가 늘지 않으므로 가중치 VRAM은 그대로다.

### 수치

예측 오차 `‖v − v_fp32‖ / ‖v_fp32‖`(%, LoRA 없음, 6장 평균):

| σ | bf16 stream | fp32 stream | int8 GPTQ + bf16 입력 | int8 GPTQ + fp32 stream |
|---|---|---|---|---|
| 0.1 | 1.76 | 0.52 | 2.29 | 1.32 |
| 0.5 | 0.85 | 0.22 | 0.99 | 0.48 |
| 0.9 | 1.15 | 0.23 | 1.07 | 0.50 |

- fp32 stream은 bf16 GEMM 주변의 예측 오차를 1.5–2.1배, int8 base 주변의 오차를 21–36% 줄인다. LoRA gradient는
  bf16 GEMM 주변에서 뒤쪽 절반 블록(14–27)이 개선되고, int8 주변에서는 블록마다 엇갈린다.
- 렌더 비교에서 글자와 손가락 위치가 일관되게 나아졌다(MOSS-Image). Anima 쪽 근거는 위의 예측 오차와 gradient
  수치다.
- stream을 fp16으로 돌릴 수는 없다. stream은 중간 블록(12–20)에서 261k–285k까지 올라가 fp16 최대값(65504)의
  4.3배가 된다. 전부 fp16으로 돌린 forward는 모든 이미지에서 블록 12부터 NaN이 된다. block GEMM 입출력은 747 이하라
  fp16 범위에 들어간다. fp16 범위를 넘는 것은 stream뿐이다.

### 비용

- 추론(8 step, 768², 1회): bf16 261 ms/step, 4.11 GiB → fp32 280 ms/step, 4.17 GiB.
- turbo 학습(int8 base, activation budget 0.99): bf16 stream 6.28 s/step, 11.87 GB → fp32 6.7–7.0 s/step, 12.39 GB.
- fp32 stream은 backward용으로 저장하는 텐서가 더 크다. bf16 stream에서 잰 activation budget 수치를 그대로 쓰지
  않는다.

### 같은 seed가 다른 그림을 낸다

noise 자체는 같지만(fp32로 뽑은 noise가 bf16으로 반올림된다) 최종 latent가 768²에서 6–12 step, 5회 실행 기준
12–24% 움직인다. bf16 stream 렌더와 seed 단위로 비교할 수 없다.

---

## 5.2 torch.compile

### dynamic 시퀀스 길이와 토큰 band

원래 종횡비 그대로 학습하면(free-fit, 1장 §3) 토큰 수가 tier별 band 안 어디든 떨어진다. 토큰 수마다 static graph를
만들면 graph 수가 폭발한다. 시퀀스 길이 축만 dynamic으로 표시하고 그 범위를 tier의 band로 묶으면 **tier당 graph가
하나**다. DiT 블록은 RoPE를 밖에서 받으므로 블록 안에서는 시퀀스 길이만 다르다.

### inductor 옵션 측정

LoRA 학습(70장, 210 step):

| 설정 | s/it (warm) | 첫 step cold / warm 캐시 | peak |
|---|---|---|---|
| torch 2.12 기본 | 0.588 | 17.1 s / 9.6 s | 13.9 GB |
| torch 2.14 기본 | 0.584 (−0.7%) | 19.4 s / 10.0 s | 13.9 GB |
| 2.12 `max-autotune` | 평균 1.032, 중앙값 0.556, p90 2.77 s | 97 s | 15.1 GB |
| 2.12 + `combo_kernels` | 0.597 (+1.5%) | 16.9 s | 13.9 GB |

- `max-autotune`은 run 중간에 다시 autotune해서 평균이 1.7배 나빠지고 VRAM이 1.2 GB 늘어난다.
- `combo_kernels`는 2.12 / 2.14 모두 기본값이 off이고, 켜면 1.5–2.4% 느리다.
- torch 2.14의 NVGEMM 백엔드는 sm_120(RTX 50)에서 전부 실패한다(`cudaErrorNoKernelImageForDevice`). 포함된 커널이
  sm100 계열뿐이다. 릴리스 노트의 "Blackwell"은 B200을 가리킨다.

---

## 5.3 VRAM과 속도

### 레버를 쓰는 순서

16 GB에서 OOM이 나면 아래 순서로 내린다. 앞의 것일수록 속도 손해가 적다.

1. activation memory budget
2. int8 base
3. block swap
4. gradient checkpointing

### activation memory budget

- torch.compile의 min-cut partitioner가 backward용으로 무엇을 저장할지 정한다. budget이 1.0 미만이면 knapsack으로
  다시 계산하기 효율적인 중간값을 골라 저장하지 않고 backward에서 다시 계산한다.
- free-fit 학습에서 1.0은 OOM, 0.99는 약 13.1 GB로 학습된다.
- int8 base에서 budget을 바꿔 가며 쟀다.

  | budget | ms/step | peak |
  |---|---|---|
  | 0.99 | 427 | 10.87 GB |
  | 0.4 | 434–439 | 10.00 GB |
  | 0.3 | 472 | 6.87 GB |
  | 0.1 | 511 | 5.32 GB |

  0.3까지 내려도 step은 11% 느려질 뿐이고 peak는 4 GB 줄어든다. 아래 block swap 표(swap 24에서 1.8 GB, 16%
  느려짐)보다 peak를 더 많이 깎는다.

**수치적으로 무해하다고 메모리도 무해한 것은 아니다.** 출력에 아무 영향이 없다고 확인된 LoRA down-projection의
custom `autograd.Function`을 지웠더니 다음 학습이 step 0에서 OOM이 났다. custom Function은 `save_for_backward`로
저장할 텐서를 고정하고 있었고, 지우자 partitioner가 블록마다 약 0.8 GB를 더 저장했다. `autograd.Function`이나
checkpoint wrapper를 지울 때는 출력과 함께 저장 텐서 크기도 비교한다.

### int8 base

block GEMM을 GPTQ로 양자화한 int8로 돌리면 bf16 593 ms → int8 433 ms(1.37배)
이고, 최종 loss는 0.0737로 같다(106장, 848 step). backward용 전치 가중치 사본을 host에 두고 필요할 때 올리면, card에
둘 때보다 peak가 1.4 GB 낮다(12.38 → 10.96 GB). 자세한 내용은 3장에서 다룬다.

### block swap

block swap은 블록 가중치를 host RAM에 두고 forward마다 card로 옮긴다. host RAM을 그만큼 더 쓴다. swap한 stack의
출력과 `dX`는 card에 모두 올린 stack과 bit 단위로 같다.

LoRA, int8 base, fp32 stream, budget 0.99, 106 step:

| swap 블록 수 | step 중앙값 | peak |
|---|---|---|
| 0 | 451 ms | 11.47 GB |
| 14 | 475 ms | 10.44 GB |
| 24 | 524 ms | 9.70 GB |

- **turbo에서는 비싸다.** turbo는 한 step에 forward를 여러 번 하고 forward마다 swap을 처음부터 다시 돌기 때문에,
  swap된 블록이 한 step에 여러 번 옮겨진다. 30 step 기준 swap 0은 5.8 s/step, 12.36 GB, swap 14는 8.4 s/step, 11.32 GB였다. 약
  1.5배의 시간으로 1 GB를 산다.
- swap을 켜면 같은 seed라도 다른 샘플을 뽑는다. swap run과 swap 없는 run의 loss는 step 단위로 비교할 수 없다(epoch
  평균 loss 0.0749 vs 0.0733).

### gradient checkpointing

gradient checkpointing은 step time을 약 18% 늘렸다. 그래서 마지막에 쓴다.

### 실측 peak

| 작업 | 설정 | 속도 | peak |
|---|---|---|---|
| LoRA | bf16 base | 0.588 s/it | 13.9 GB |
| LoRA | int8 base, fp32 stream, swap 0 | 451 ms | 11.47 GB |
| turbo | int8 base, fp32 stream | 6.7–7.0 s/step | 12.39 GB |
| turbo 4-step, 250 it | int8 base, fp32 stream | 5.7–5.8 s/it, 약 27분 | – |

### optimizer: fused AdamW

AdamW를 fused 커널로 쓴다. bitsandbytes `AdamW8bit`는 파라미터 블록마다
커널을 하나씩 띄워 optimizer 구간에서 GPU 사용률이 88%로 떨어졌다. 한 step 1.17 s 중 308 ms가 optimizer였고, fused
AdamW로 바꾸자 0.86 s가 됐다(RTX 5060 Ti). 대가는 optimizer state 약 180 MB다(rank 32 LoRA, 약 30M 파라미터).

### FlashAttention-2의 fp16 누적 backward

FlashAttention-2의 backward를 fp16 누적으로 바꾼 빌드에서, fp16 누적 backward는 step
time을 5.1%, fp16 입력은 3.1% 줄인다(RTX 50). peak VRAM은 같다.

### fp16 GEMM

같은 카드에서 fp32 누적 fp16 GEMM은 bf16과 속도가 같고(약 102 TFLOPS), fp16 누적만 1.6배 빠르다. 그래도 int8
GEMM(약 330 TFLOPS)보다 느리고, backward에는 loss scaling이 2^20 근처로
필요하다. 그래서 학습 GEMM은 bf16 또는 int8로 돌린다.

---

## 5.4 추론

- int8 base에서는 LoRA를 정적으로 merge하지 않고 forward에 hook으로 더한다. 정적으로 merge하면 LoRA가 int8로
  변환하며 버리는 bf16 가중치에 접히기 때문이다. bf16 base를 int8 base로 바꾸면 생성 픽셀이 약 4/255 움직인다.

### 학습 없는 추론 기법

| 기법 | 내용 |
|---|---|
| Spectrum | Chebyshev 다항식으로 feature를 예측해, 캐시된 step에서는 블록을 모두 건너뛴다 |
| CNS | SDE sampler가 넣는 noise를 아직 해상되지 않은 주파수 대역으로 재분배한다. ER-SDE 같은 SDE sampler에서만 작동하고, euler에서는 효과가 없다 |
| mod-guidance | 학습한 pooled text projection으로 AdaLN modulation을 조절한다. 전역 톤만 바꾸는 레버다 |
| DAVE | 같은 프롬프트에서 seed 간 다양성을 회복한다 |

### turbo LoRA 추론

- euler, 4 step, CFG 1.0. CFG는 증류 때 student에 구워졌으므로 1.0을 유지한다. guidance를 두 번 걸지 않는다.
- 추론의 flow shift(3.0)는 turbo 학습의 flow shift와 맞춘다. 하나를 바꾸면 다른 것도 바꾼다.
- Spectrum과 함께 쓰지 않는다. Spectrum의 Chebyshev 캐시는 16 step 이상을 가정한다. CNS도 SDE sampler 전용이라
  euler 4-step에는 해당하지 않는다.
- checkpoint가 학습한 step 수보다 더 많은 step에서 더 좋아 보이면, 아직 N-step 맵에 도달하지 못했다는 신호다.

---

## 5.5 실험 위생

### 평가 기준은 결과를 보기 전에 정한다

- 근거가 gate를 통과시킨다. 모든 지표와 렌더가 한 arm을 가리키면 차이가 작아도 통과다. 기준은 결과를 읽기 전에
  정하고, 구조적으로 작을 수밖에 없는 값(낮은 σ에서 one-step x̂₀ 차이처럼 x_t가 이미 이미지를 담고 있는 경우)에는
  threshold를 걸지 않는다.
- 예: turbo A/B는 "arm의 EMA 렌더가 두 prompt 모두에서 control–twin 차이보다 크게 이기고, 모멘트 갭(4장)이
  |control − twin|보다 크게 낮으면 통과"를 run 전에 적어 두었다. gradient norm 쪽 결정 기준(pooled top/bottom ≤ 1.5
  → adaptive clip만 쓴다)도 미리 정했고, 첫 arm의 결과(1.43)는 그 기준대로 읽었다.

### null twin으로 noise floor를 먼저 잰다

- raw checkpoint 둘을 비교하면 대부분 run noise를 읽는다. control을 seed만 바꿔 한 번 더 돌린 twin이 이후 모든
  arm의 noise floor가 된다.
- seed만 다른 두 run의 ΔW cosine은 0.09–0.10이었다. 목적함수나 LR이 다른 쌍도 0.09–0.11이었다. **ΔW overlap으로는
  arm을 구분할 수 없다.** 평가는 렌더와 모멘트 갭으로 한다.
- 같은 run 안에서도 |ΔW|가 √t로 자라고 125-step 이동끼리 직교한다(cos 0.018 / 0.028). 상수 LR에서는 random walk다.
  배포 대상을 EMA(window 0.2)로 하는 이유다.
- **raw checkpoint는 발산 burst 안에 있을 수 있다.** control의 raw 250은 judge 12장 전부 평평한 noise texture였다.
  마지막 4 step에서 student gradient norm이 231 / 244 / 106 / 72로 튀었고, 이 recipe에서 burst는 약 250 iteration에
  한 번 나온다. 같은 시점의 EMA는 정상 렌더였다.
- twin은 한 번만 돌린다. 통과한 arm은 새 control이 되고, step time이나 loss 항이 바뀌지 않으면 새 twin을 만들지
  않는다.

### 렌더가 gate다

- 첫 checkpoint에서 grid를 렌더한다. APEX turbo arm(4장)은 loss, x̂₀ 예측의 std, rollout std가 모두 평평하고
  건강했는데 4-step grid는 warm start보다 흐려지고 있었다. 흐린 샘플도 latent std는 정상이다.
- turbo의 flow-matching MSE는 품질과 반대로 움직인다. checkpoint 순위는 렌더한 4-step 샘플로 매긴다. pose 다양성은
  grid에서만 보이고, pooled feature 지표는 pose를 보지 못한다.
- 평가용 prompt는 어려워야 한다. 간판 글자 하나, 인물 한 명짜리 prompt는 arm을 가르기에 너무 쉬웠다. turbo는 어려운
  prompt(이름 있는 다섯 명 + 위치 절, 일본어 메뉴와 영어 간판이 있는 카페, 아래에서 올려다본 점프 구도)를 768 × 1344에서
  teacher 옆에 놓고 본다.

### 작은 것부터, 한 번만

- 메커니즘 질문은 full run 전에 약 25분짜리 micro arm으로 답한다. 끝난 측정은 다시 재지 않고 그 결과 위에 쌓는다.
  run을 돌리기 전에 "이 결과가 어떤 결정을 바꾸는가"를 말할 수 있어야 한다. 말할 수 없으면 코드를 읽는다.
- 기존 checkpoint나 렌더가 control이다. code drift나 recipe 변경으로 비교가 불공정하면 결과 옆에 적고, 새 control은
  treatment 뒤에 따로 돌린다.
- 원인을 모를 때는 config 추측보다 실제 run의 bisect가 빠르다. §5.3의 partitioner OOM은 config 추측 다섯 개가 모두
  빗나갔고, 인접 commit 두 개에서 26-step run을 돌린 bisect가 몇 분 만에 원인을 찾았다.
