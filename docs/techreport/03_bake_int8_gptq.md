# 3. LoRA bake, int8 base, 체크포인트별 GPTQ

**요약**

- GPTQ로 만든 int8 forward copy는 그 자체가 가중치다. base DiT에 calibration한 GPTQ 파일을 LoRA를 구운 DiT에 쓰면
  int8 forward는 base를 계산하고, 구운 delta는 빠진다. 구운 DiT(여기서는 증류 teacher)에는 그 체크포인트 전용 GPTQ가
  필요하고, 만드는 데 약 17분이 걸린다.
- bake는 모듈마다 fp32로 `W += m · (up @ down) · scale`을 계산하고 bf16로 저장한다. 구울 수 있는 것은 가중치 delta로
  표현되는 어댑터뿐이다.
- frozen DiT의 큰 linear를 W8A8 int8 GEMM으로 돌리면 LoRA 학습 step이 1.37배 빨라지고 최종 loss는 같다.
- GPTQ는 weight 양자화 오차의 81%를 없애지만 W8A8 오차는 36%만 줄인다. 남는 오차는 대부분 activation 항이다.
  예측 오차로는 약 10%, CFG velocity 오차로는 2–7%가 줄어든다.

실험 모델은 28블록 DiT(Anima)이고, 시간은 모두 RTX 5070 Ti에서 쟀다.

---

## 3.1 base용 GPTQ는 구운 DiT에 맞지 않는다

증류의 teacher는 LoRA soup을 multiplier 1.0으로 base에 구운 DiT이고(→ [02_soup_merge.md](02_soup_merge.md)), 이 teacher를
int8로 돌린다.

1. **base 파일은 틀린 값이다.** int8 forward copy는 그 자체가 가중치다. base 파일을
   쓰면 teacher의 int8 forward는 soup 없는 base를 계산하고, 구운 LoRA delta는 int8 forward에서 빠진다.
2. **GPTQ는 가중치와 activation 통계 모두에 묶여 있다.** 반올림은 그 모델의 블록 입력으로 모은 Hessian `H_r`에 맞춰
   최적화되고, 블록 i의 calibration 입력은 양자화된 블록 0..i−1을 지나 만들어진다(§3.4). 그래서 다른 DiT의 파일을 다시
   쓸 방법이 없다.
3. **RTN과 비교하면** RTN으로도 구운 DiT를 int8로 돌릴 수 있다. 대신 per-linear W8A8 오차가 약 1.5배(6.53e-5 대
   4.24e-5)이고, base에서 RTN은 GPTQ보다 예측 오차가 약 10% 크다. 만드는 데 17분이면 되므로 전용 GPTQ를 만드는 쪽을
   택했다.

구운 LoRA가 바뀔 때마다 bake와 GPTQ를 다시 해야 한다.

soup을 구웠다고 int8이 더 어려워지지는 않았다. 같은 설정으로 calibration한 두 GPTQ의 per-linear 오차는 전체 평균에서
3% 안으로 같았다(§3.4 표).

---

## 3.2 Bake: LoRA를 DiT 가중치에 굽기

모듈마다 **fp32로** `W += multiplier · (up @ down) · scale`을 계산하고, 결과를 bf16으로 저장한다. 13초 걸린다. 구운
DiT는 텐서 수, dtype, 키 이름과 shape가 base와 같으므로 base를 받는 곳이면 어디서나 base 대신 쓸 수 있다.

가중치 delta가 아닌 어댑터는 구울 수 없다. 레이어 router가 입력마다 head를 고르는 어댑터(HydraLoRA 등)는 고정 가중치로
옮길 수 없고, cross-attn KV에 토큰을 끼우는 prefix / postfix나 self-attn 시퀀스에 붙는 register token도 가중치가 아니다.
plain LoRA와, timestep 마스크를 학습에서만 쓰는 T-LoRA는 bake해도 결과가 같다.

---

## 3.3 int8 base

### 무엇이 int8로 바뀌는가

| int8 (블록당 6개, 28블록에서 168개) | float으로 남는 것 |
|---|---|
| self-attn qkv (fused), self-attn 출력 | cross-attn k / v (M이 context 길이) |
| cross-attn q, cross-attn 출력 | adaLN (M이 batch) |
| MLP의 두 linear | attention, norm, residual stream, embedder, head |
| | LoRA 자체의 GEMM |

cross-attn k / v와 adaLN은 int8이 오히려 느리게 측정되어 bf16로 둔다.

### 형식

activation outlier 때문에 W8A8이 어렵다는 점은 SmoothQuant(Xiao et al. 2022, arXiv 2211.10438)가 짚었고, 회전으로
outlier를 퍼뜨린 뒤 양자화하는 방법이 QuaRot(Ashkboos et al. 2024, arXiv 2404.00456)이다. 여기서 쓰는 형식은 이를
DiT에 맞게 그룹 단위 Hadamard로 바꾼 ConvRot(Huang et al. 2025, arXiv 2512.03673)이다.

- 가중치를 256열 그룹마다 정규화된 Hadamard 행렬로 회전한 뒤 int8로 저장한다. 스케일은 출력 행마다 fp32 하나이고,
  값은 `amax / 127`이다. 이 Hadamard는 대칭이고 자기 자신이 역행렬이라서, 같은 회전으로 입력을 가중치 기저로
  옮길 수 있다.
- activation은 토큰마다 int8로 양자화하고(W8A8), 출력은 bf16이다.
- base 가중치의 gradient는 계산하지 않는다. LoRA 학습에 필요한 것은 입력 gradient `dX = dY·W`뿐이다.

### forward copy와 backward copy

forward는 항상 int8 copy `W_q`로 돈다. 입력 gradient `dX`를 어떻게 계산하느냐에 따라 세 가지 배치가 있다.

- bf16 base 가중치를 카드에 남겨 두고 `dX`를 bf16로 계산한다.
- `Wᵀ`의 두 번째 int8 copy(backward copy)를 카드에 두고 `dX`도 int8로 계산한다. bf16 가중치는 버린다.
- backward copy를 모든 블록에 대해 page-locked host 메모리에 두고, 카드에는 블록 크기 slot 두 개만 둔다. 하나는
  backward가 지금 지나는 블록용이고, 하나는 worker thread가 미리 올리는 아래 블록용이다. 터보 증류는 이 배치로 돈다.

생성에는 backward copy가 없다.

- **GPTQ가 다루는 것은 forward copy뿐이다.** backward copy는 항상 round to nearest로 만들고, 만들려면 float 가중치가
  필요하다. int8에서 다시 만든 가중치는 원래 가중치와 1.1% 다르고, 그 결과 backward copy가 1.6% 움직인다. 그래서
  학습과 터보에서는 bf16 DiT를 함께 읽는다.
- 생성에서는 LoRA를 정적으로 merge하지 않고 hook으로 붙인다. 정적으로 merge하면 int8 변환 과정에서 버려지는 bf16
  가중치에 LoRA가 접히기 때문이다. base는 `int8(x, W_q) + B·A·x`를 계산하는데, 이는 학습한 함수와 같다.

### 측정값

LoRA 학습 속도다. 106장 데이터셋, 848 steps, plain LoRA, backward copy를 카드에 둔 배치 기준이다.

| | bf16 | int8, bf16 kernel input | int8, fp32 kernel input |
|---|---|---|---|
| step | 593 ms | 433 ms (1.37x) | 424 ms (첫 compile 포함) |
| 최종 평균 loss | 0.0737 | 0.0737 | 0.0739 |

두 LoRA를 bf16 base에서 생성한 샘플의 품질은 같았다. base를 bf16에서 int8로 바꾸면 생성 픽셀이 약 4/255
움직인다.

backward copy를 카드에 둘 때와 host에 둘 때의 비교다. activation memory budget 0.99, 106 샘플 전체 기준이다.

| backward copy | median | peak |
|---|---|---|
| 카드 | 482 ms | 12.38 GB |
| host | 499 ms | 10.96 GB |

host 배치는 17 ms/step을 더 쓰고 peak를 1.42 GB 줄인다. loss는 같다. budget 1.0에서는 step 0에서 OOM이 난다.

block swap(106 steps, budget 0.99, GPTQ forward copy, fp32 stream)에서 host 배치는 swap 0 / 14 / 24일 때 451 / 475 / 524 ms,
11.47 / 10.44 / 9.70 GB였다. swap 14에서 backward copy를 카드에 둔 배치는 526 ms / 11.10 GB, bf16 가중치로 `dX`를
계산하는 배치는 667 ms / 11.92 GB였다.

activation budget을 내린 경우 host 배치는 0.99일 때 427 ms / 10.87 GB, 0.3일 때 472 ms / 6.87 GB, 0.1일 때 511 ms / 5.32 GB다.

터보 증류는 host 배치로 돌고, teacher도 int8로 돈다. warm start, budget 0.99 기준 수치는 다음과 같다.

| 터보 base | s/step | peak |
|---|---|---|
| bf16 base | 9.65 | – |
| int8, bf16 stream | 6.28 | 11.87 GB |
| int8, fp32 stream | 6.7–7.0 | 12.39 GB |

block swap을 켠 터보(30 steps, fp32 stream)는 swap 0에서 5.8 s/step / 12.36 GB, swap 14에서 8.4 s/step / 11.32 GB였다.
터보는 forward마다 swap을 되감기 때문에 swap 비용이 학습보다 크다.

생성(768², 8 steps, 셀마다 1회 측정)은 bf16 261 ms/step / 4.11 GiB, int8 164 ms/step / 2.64 GiB, int8 + fp32 stream
179 ms/step / 2.68 GiB였다.

### kernel input과 fp32 stream

bf16 입력을 받는 kernel은 activation 스케일 나눗셈 근처에서 값을 bf16로 반올림한다. 입력을 fp32로 넘기면 bf16 input보다
예측 오차가 6–8% 줄어든다. fp32 residual stream은 int8 base의 오차를 21–36% 더 줄인다. stream 자체는
[05_practical_tips.md](05_practical_tips.md)에서 다룬다.

---

## 3.4 GPTQ

base GPTQ는 17.6분, soup teacher GPTQ는 1045초(calibration 수집 18초, 블록당 약 36초) 걸렸고, 결과는 2.37 GiB다.

### calibration

- 작가 폴더마다 캐시된 이미지를 최대 2장 고른다. 토큰 수가 3000 이상 4608 이하인 것만 고르고, seed 0으로 섞는다.
  캐시된 VAE latent와 TE 출력을 그대로 쓴다.
- 이미지마다 σ를 하나씩 뽑는다. σ는 추론 schedule(28 steps, flow shift 3.0)에서 고르고, 그 σ로 노이즈를 섞는다.
- 각 샘플을 두 번 돌린다. 한 번은 자기 캡션으로, 한 번은 negative prompt로 돌린다(CFG uncond 쪽).
- 블록 0에 들어가는 입력을 잡아 두고, 블록을 **순서대로** 양자화한다. 양자화가 끝난 블록(W8A8 kernel)의 출력이 다음 블록의
  입력이 되므로, 앞쪽 블록의 양자화 오차가 뒤쪽 calibration에 반영된다.

| | base | soup teacher |
|---|---|---|
| 작가 / 이미지 / 샘플 | 91 / 182 / 364 | 92 / 184 / 368 |
| linear당 토큰 | 1,487,472 | 1,510,348 |
| σ schedule | 28 steps, shift 3.0 | 28 steps, shift 3.0 |

두 calibration은 같은 설정으로 돌았고, 고른 작가 수만 91에서 92로 다르다.

### 양자화 방법

linear 하나마다 회전한 입력 `X_r`로 `H_r = X_rᵀX_r`를 모은다. 같은 grid(256열 회전, 행 스케일 `amax/127`)에서 반올림을
세 가지로 해 본다.

- RTN: round to nearest다.
- GPTQ(Frantar et al. 2022, arXiv 2210.17323): `H_r`에 대해 열 단위로 반올림하고 오차를 남은 열로 넘긴다. block 128,
  damp 0.01, act-order를 켠다.
- act-aware: activation 양자화까지 고려한 최소제곱 가중치 `W*`에 GPTQ를 적용한다(ridge 1e-4).

세 방식 모두 W8A8 출력 오차 `‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²`로 채점한다. RTN과 GPTQ는 weight-only 오차(W8A16)로도
채점한다. 저장하고 다음 블록의 입력을 만드는 데 쓰는 것은 GPTQ다.

### base와 soup teacher 비교

168개 linear의 평균이고, 단위는 상대 출력 오차 에너지다.

| | base | soup teacher |
|---|---|---|
| W8A8 RTN → GPTQ | 6.40e-5 → 4.12e-5 (−35.5%) | 6.53e-5 → 4.24e-5 (−35.1%) |
| W8A8 act-aware | 4.10e-5 | 4.22e-5 |
| W8A16 RTN → GPTQ | 2.79e-5 → 0.52e-5 (−81.4%) | 2.84e-5 → 0.55e-5 (−80.6%) |
| 평균 ridge damp | 1.21e-4 | 1.11e-4 |
| W8A8 GPTQ 최악 linear | 블록 17 cross-attn 출력, 1.41e-4 | 블록 15 cross-attn 출력, 1.48e-4 |

linear 종류별 W8A8 GPTQ 평균:

| linear | base | soup teacher |
|---|---|---|
| self-attn qkv | 3.68e-5 | 3.72e-5 |
| self-attn 출력 | 6.74e-5 | 7.04e-5 |
| cross-attn q | 4.64e-5 | 4.41e-5 |
| cross-attn 출력 | 5.40e-5 | 5.57e-5 |
| MLP 첫 linear | 2.17e-5 | 2.20e-5 |
| MLP 둘째 linear | 2.12e-5 | 2.53e-5 |

전체 평균 차이는 3% 안이고, 가장 크게 움직인 것은 MLP 둘째 linear(+19%)다. act-aware가 plain GPTQ보다 나은 점이 없다는
결론도 두 DiT에서 같다. 다른 soup을 구운 DiT도 W8A8 RTN → GPTQ가 6.51e-5 → 4.22e-5로 거의 같았다.

### GPTQ가 실제로 줄이는 것 (base 측정)

- GPTQ는 weight 항의 81%를 없앤다. W8A8 오차는 36% 줄고, 남은 오차는 대부분 activation 항이다.
- end to end로 보면 cond / uncond 예측 오차가 약 10% 줄고(0.87–0.92x), CFG velocity 오차는 2–7% 줄어든다(0.93–0.99x).
  base 자체의 trajectory에서 재도 held-out 실제 이미지에서 재도 같아서, calibration 소스는 결과에 영향이 없다. σ 구간
  세 개 모두에서 0.87–0.92x로 고르게 나왔다.
- CFG velocity에서 이득이 작은 이유는 이렇다. weight 오차는 cond와 uncond가 공유하므로 `v_c − v_u`에서 대부분 상쇄되고,
  CFG가 증폭하는 것은 공유되지 않는 activation 반올림이다.
- LoRA 학습은 CFG 없이 cond 예측을 쓰므로, 학습 쪽 이득은 약 10% 쪽 수치에 해당한다.

### round to nearest

GPTQ 파일이 없으면 그 자리에서 RTN으로 양자화한다. GPTQ 대비 손해는 위에서 본 예측 오차 약 10%다. RTN은 calibration
없이 어떤 DiT에서도 동작하므로, GPTQ 파일이 없는 체크포인트에서 쓸 수 있는 유일한 선택지다.
