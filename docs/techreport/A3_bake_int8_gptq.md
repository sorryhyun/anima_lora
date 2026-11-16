# 부록 3. bake, int8 base, GPTQ의 세부와 측정

`03_bake_int8_gptq.md` 본문의 세부 설정과 측정입니다.

## A3.1 GPTQ

base GPTQ는 17.6분, soup teacher GPTQ는 1045초(calibration 수집 18초, 블록당 약 36초) 걸렸고, 결과는
2.37 GiB입니다.

### calibration 설정

- 작가 폴더마다 캐시된 이미지를 최대 2장 고릅니다. 토큰 수가 3000 이상 4608 이하인 것만 고르고, seed 0으로
  섞습니다. 캐시된 VAE latent와 TE 출력을 그대로 씁니다.
- 이미지마다 σ를 하나씩 뽑습니다. σ는 추론 schedule(28 steps, flow shift 3.0)에서 고르고, 그 σ로 노이즈를
  섞습니다.
- 각 샘플을 두 번 돌립니다. 한 번은 자기 캡션으로, 한 번은 negative prompt로 돌립니다(CFG uncond 쪽).
- 블록 0에 들어가는 입력을 잡아 두고, 블록을 **순서대로** 양자화합니다. 양자화가 끝난 블록(W8A8 kernel)의 출력이
  다음 블록의 입력이 되므로, 앞쪽 블록의 양자화 오차가 뒤쪽 calibration에 반영됩니다.

| | base | soup teacher |
|---|---|---|
| 작가 / 이미지 / 샘플 | 91 / 182 / 364 | 92 / 184 / 368 |
| linear당 토큰 | 1,487,472 | 1,510,348 |
| σ schedule | 28 steps, shift 3.0 | 28 steps, shift 3.0 |

두 calibration은 같은 설정으로 돌았고, 고른 작가 수만 91에서 92로 다릅니다.

### 반올림 방식

linear 하나마다 회전한 입력 `X_r`로 `H_r = X_rᵀX_r`를 모읍니다. 같은 grid(256열 회전, 행 스케일 `amax/127`)에서
반올림을 세 가지로 해 봤습니다.

- RTN: round to nearest입니다.
- GPTQ(Frantar et al. 2022, arXiv 2210.17323): `H_r`에 대해 열 단위로 반올림하고 오차를 남은 열로 넘깁니다.
  block 128, damp 0.01, act-order를 켭니다.
- act-aware: activation 양자화까지 고려한 최소제곱 가중치 `W*`에 GPTQ를 적용합니다(ridge 1e-4).

세 방식 모두 W8A8 출력 오차 `‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²`로 채점하고, RTN과 GPTQ는 weight-only
오차(W8A16)로도 채점합니다. 저장하고 다음 블록의 입력을 만드는 데 쓰는 것은 GPTQ입니다.

## A3.2 int8 형식과 forward / backward copy

### 형식

- 가중치를 256열 그룹마다 정규화된 Hadamard 행렬로 회전한 뒤 int8로 저장합니다. 스케일은 출력 행마다 fp32
  하나이고, 값은 `amax / 127`입니다. 이 Hadamard는 대칭이고 자기 자신이 역행렬이라서, 같은 회전으로 입력을 가중치
  기저로 옮길 수 있습니다.
- activation은 토큰마다 int8로 양자화하고(W8A8), 출력은 bf16입니다.

### forward copy와 backward copy

forward는 항상 int8 copy `W_q`로 돕니다. 입력 gradient `dX`를 어떻게 계산하느냐에 따라 세 가지 배치가 있습니다.

- bf16 base 가중치를 카드에 남겨 두고 `dX`를 bf16로 계산합니다.
- `Wᵀ`의 두 번째 int8 copy(backward copy)를 카드에 두고 `dX`도 int8로 계산합니다. bf16 가중치는 버립니다.
- backward copy를 모든 블록에 대해 page-locked host 메모리에 두고, 카드에는 블록 크기 slot 두 개만 둡니다. 하나는
  backward가 지금 지나는 블록용이고, 하나는 worker thread가 미리 올리는 아래 블록용입니다. 터보 증류는 이 배치로
  돕니다.

생성에는 backward copy가 없습니다.

- **GPTQ가 다루는 것은 forward copy뿐입니다.** backward copy는 항상 round to nearest로 만들고, 만들려면 float
  가중치가 필요합니다. int8에서 다시 만든 가중치는 원래 가중치와 1.1% 다르고, 그 결과 backward copy가 1.6%
  움직입니다. 그래서 학습과 터보에서는 bf16 DiT를 함께 읽습니다.
- 생성에서는 LoRA를 정적으로 merge하지 않고 hook으로 붙입니다. 정적으로 merge하면 int8 변환 과정에서 버려지는 bf16
  가중치에 LoRA가 접히기 때문입니다. 즉 base는 `int8(x, W_q) + B·A·x`를 계산하고, 이는 학습한 함수와 같습니다.

### kernel input

bf16 입력을 받는 kernel은 activation 스케일 나눗셈 근처에서 값을 bf16로 반올림합니다. 입력을 fp32로 넘기면 bf16
input보다 예측 오차가 6–8% 줄고, fp32 residual stream은 int8 base의 오차를 21–36% 더 줄입니다(A3.3).

## A3.3 fp32 residual stream

### 구현

embedder, head, modulation linear는 호출할 때만 weight를 fp32로 cast해서 autocast 밖에서 계산하고, q / k norm은
fp32 RMSNorm으로 바꾸고, 나머지 forward는 bf16 autocast로 감쌉니다. 그래서 block linear 입력은 bf16으로
반올림되고, int8 linear는 fp32 입력을 받습니다. 파라미터가 늘지 않으므로 가중치 VRAM은 그대로입니다.

### 수치

예측 오차 `‖v − v_fp32‖ / ‖v_fp32‖`(%, LoRA 없음, 6장 평균):

| σ | bf16 stream | fp32 stream | int8 GPTQ + bf16 입력 | int8 GPTQ + fp32 stream |
|---|---|---|---|---|
| 0.1 | 1.76 | 0.52 | 2.29 | 1.32 |
| 0.5 | 0.85 | 0.22 | 0.99 | 0.48 |
| 0.9 | 1.15 | 0.23 | 1.07 | 0.50 |

- fp32 stream은 bf16 GEMM 주변의 예측 오차를 1.5–2.1배, int8 base 주변의 오차를 21–36% 줄입니다. LoRA gradient는
  bf16 GEMM 주변에서 뒤쪽 절반 블록(14–27)이 개선되고, int8 주변에서는 블록마다 엇갈립니다.
- 렌더 비교에서는 글자와 손가락 위치가 일관되게 나아졌습니다(MOSS-Image). Anima 쪽 근거는 위의 예측 오차와
  gradient 수치입니다.
- stream을 fp16으로 돌릴 수는 없습니다. stream은 중간 블록(12–20)에서 261k–285k까지 올라가 fp16 최대값(65504)의
  4.3배가 되고, 전부 fp16으로 돌린 forward는 모든 이미지에서 블록 12부터 NaN이 됩니다. block GEMM 입출력은 747
  이하라 fp16 범위에 들어가므로, fp16 범위를 넘는 것은 stream뿐입니다.

### 비용

- 추론(8 step, 768², 1회): bf16 261 ms/step, 4.11 GiB → fp32 280 ms/step, 4.17 GiB.
- turbo 학습(int8 base, activation budget 0.99): bf16 stream 6.28 s/step, 11.87 GB → fp32 6.7–7.0 s/step, 12.39 GB.
- fp32 stream은 backward용으로 저장하는 텐서가 더 큽니다. bf16 stream에서 잰 activation budget 수치를 그대로 쓰면
  안 됩니다.

### 같은 seed가 다른 그림을 낸다

noise 자체는 같지만(fp32로 뽑은 noise가 bf16으로 반올림됩니다) 최종 latent가 768²에서 6–12 step, 5회 실행 기준
12–24% 움직입니다. 즉 bf16 stream 렌더와 seed 단위로 비교할 수 없습니다.
