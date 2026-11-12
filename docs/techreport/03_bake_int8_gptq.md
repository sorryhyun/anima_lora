# 3. LoRA를 DiT에 bake하고, int8 base와 teacher 전용 GPTQ 만들기

**요약**

- soup LoRA를 multiplier 1.0으로 base DiT에 구워 bf16 teacher DiT를 만듭니다. 13초 걸리는 작업입니다.
- 터보는 이 teacher를 int8 base(`stream` mode) 위에서 돌립니다. 블록당 큰 linear 6개(28블록에서 168개)를 ConvRot
  int8 GEMM으로 바꾸고, 나머지는 float으로 둡니다.
- int8 forward copy는 GPTQ로 만듭니다. base용 GPTQ는 soup을 구운 DiT에 맞지 않으므로, teacher에는 **teacher 전용**
  GPTQ(2.37 GiB, 약 17분)를 따로 만듭니다.

---

## 3.1 파이프라인에서 이 단계의 위치

앞 단계([02_soup_merge.md](02_soup_merge.md))의 결과물은 **LoRA 파일**입니다. 8개 shard를 exact concat으로 합친 rank 256
LoRA이고, adaLN은 평균으로 합쳤습니다. 터보 증류([04_turbo_distillation.md](04_turbo_distillation.md))는
이 soup이 반영된 모델을 teacher로 씁니다.

배포는 bake된 DiT로 하지 않습니다. soup LoRA와 터보 LoRA를 exact concat해서 base에 올라가는 LoRA 하나로 만듭니다.

---

## 3.2 Bake: LoRA를 DiT 가중치에 굽기

모듈마다 **fp32로** `W += multiplier · (up @ down) · scale`을 계산합니다. channel scaling의 `inv_scale`이 있으면 `down`에
먼저 곱해서 흡수를 되돌립니다. 결과는 bf16으로 저장됩니다.

merged14 teacher는 텐서 수(685), dtype(BF16), 키 이름과 shape가 base와 같으므로, base를 받는 곳이라면 어디서나 base 대신
쓸 수 있습니다.

가중치 delta가 아닌 어댑터는 구울 수 없습니다. HydraLoRA(와 step별 head를 쓰는 step-expert 터보)는 레이어 router가 입력마다
head를 고르므로 고정 가중치로 옮길 수 없고, postfix / prefix는 cross-attn KV splice, register tokens는 self-attn 시퀀스에
붙는 토큰입니다. 지원 대상은 plain LoRA와 T-LoRA입니다. T-LoRA의 timestep 마스크는 학습에서만 쓰이므로 bake해도 결과가
같습니다.

---

## 3.3 int8 base

### 무엇이 int8로 바뀌는가

| int8 (블록당 6개, 28블록에서 168개) | float으로 남는 것 |
|---|---|
| `self_attn.qkv_proj` (fused), `self_attn.output_proj` | cross-attn `k` / `v` (M이 context 길이) |
| `cross_attn.q_proj`, `cross_attn.output_proj` | adaLN (M이 batch) |
| `mlp.layer1`, `mlp.layer2` | attention, norm, residual stream, embedder, head, `llm_adapter` |
| | LoRA 자체의 GEMM |

cross `k/v`와 adaLN은 phase 0에서 int8이 오히려 느리게 측정되어 bf16로 둡니다.

### ConvRot 형식

- 가중치를 256열 그룹마다 정규화된 Hadamard 행렬로 회전한 뒤 int8로 저장합니다. 스케일은 출력 행마다 fp32 하나이고,
  값은 `amax / 127`입니다. 이 Hadamard는 대칭이고 자기 자신이 역행렬이라서, 같은 회전으로 입력을 가중치 기저로
  옮길 수 있습니다.
- activation은 토큰마다 int8로 양자화하고(W8A8), 출력은 bf16입니다.
- base 가중치의 gradient는 계산하지 않습니다. LoRA 학습에 필요한 것은 입력 gradient `dX = dY·W`뿐입니다.

### mode

| mode | forward | 입력 gradient `dX` | bf16 base 가중치 | 쓰는 곳 |
|---|---|---|---|---|
| `off` | bf16 | bf16 | 유지 | 기본값 |
| `fwd` | int8 | bf16, 유지된 `W`로 계산 | 유지 | 학습 |
| `full` | int8 | int8, `Wᵀ`의 두 번째 int8 copy(`wq_t`) | 버림 | 학습 |
| `stream` | int8 | `full`과 같지만 `wq_t`는 pinned host 메모리에 두고, backward가 지나는 블록 것만 카드로 보냄 | 버림 | 학습, 터보 |
| `infer` | int8 | 없음 | 버림 | 생성 |

- **GPTQ가 다루는 것은 forward copy(`wq`)뿐입니다.** backward copy `wq_t`는 항상 round to nearest로 만들고, 만들려면
  float 가중치가 필요합니다. int8에서 다시 만든 가중치는 원래 가중치와 1.1% 다르고, 그 결과 `wq_t`가 1.6% 움직입니다.
  그래서 학습과 터보에서는 bf16 DiT를 함께 읽습니다.
- `stream`은 모든 블록의 `wq_t`를 page-locked host 버퍼에 두고, 카드에는 블록 크기 slot 두 개만 둡니다. 하나는 backward가
  지금 지나는 블록용이고, 하나는 worker thread가 미리 올리는 아래 블록용입니다.
- 생성(`infer`)에서는 LoRA를 정적으로 merge하지 않고 hook으로 붙입니다. 정적으로 merge하면 변환 과정에서 버려지는 bf16
  가중치에 LoRA가 접히기 때문입니다. base는 `int8(x, W_q) + B·A·x`를 계산하는데, 이는 trainer가 학습한 함수와 같습니다.

### 측정값

학습 속도는 `ama mitsuki`, 848 steps, method `lora`, 5070 Ti 기준입니다.

| | bf16 | int8 `full`, bf16 kernel input | int8 `full`, fp32 kernel input |
|---|---|---|---|
| step | 593 ms | 433 ms (1.37x) | 424 ms (첫 compile 포함) |
| 최종 `avr_loss` | 0.0737 | 0.0737 | 0.0739 |

두 LoRA를 bf16 base에서 생성한 샘플의 품질은 같았습니다. base를 bf16에서 int8로 바꾸면 생성 픽셀이 약 4/255
움직입니다.

`stream`과 `full`의 peak입니다. budget 0.99, 106 샘플 전체 기준입니다.

| arm | median | peak |
|---|---|---|
| `full` | 482 ms | 12.38 GB |
| `stream` | 499 ms | 10.96 GB |
| `stream`, budget 1.0 | step 0에서 OOM | – |

`stream`은 17 ms/step을 더 쓰고 peak를 1.42 GB 줄입니다. loss는 `full`과 같습니다.

block swap(`ama mitsuki`, 106 steps, budget 0.99, GPTQ forward copy, fp32 stream)에서 `stream`은 swap 0 / 14 / 24일 때
451 / 475 / 524 ms, 11.47 / 10.44 / 9.70 GB였습니다. swap 14에서 `full`은 526 ms / 11.10 GB, `fwd`는 667 ms / 11.92 GB였습니다.

activation budget을 내린 경우 `stream`은 0.99일 때 427 ms / 10.87 GB, 0.3일 때 472 ms / 6.87 GB, 0.1일 때 511 ms / 5.32 GB입니다.

터보는 `stream`으로 돌고, teacher도 int8로 돕니다. warm start, budget 0.99 기준 수치는 다음과 같습니다.

| 터보 base | s/step | peak |
|---|---|---|
| bf16 base | 9.65 | – |
| int8 `stream`, bf16 stream | 6.28 | 11.87 GB |
| int8 `stream`, fp32 stream | 6.7–7.0 | 12.39 GB |

block swap을 켠 터보(30 steps, CDM on, GAN off, fp32 stream)는 swap 0에서 5.8 s/step / 12.36 GB, swap 14에서
8.4 s/step / 11.32 GB였습니다. 터보는 forward마다 swap을 되감기 때문에 swap 비용이 학습보다 큽니다.

생성 쪽은 smoke 한 번만 있습니다. 768², 8 steps, 한 셀에 한 번씩 그렸고, 전체 run은 중단됐으니 결과로 볼 수는 없습니다.
bf16 261 ms/step / 4.11 GiB, int8 164 ms/step / 2.64 GiB, int8 + fp32 stream 179 ms/step / 2.68 GiB입니다.

### kernel input과 fp32 stream (짧게)

stock `comfy_kitchen`(0.2.40 / 0.2.41)은 bf16 입력을 받으면 activation 나눗셈 근처에서 값을 bf16로 반올림합니다
(comfy-kitchen#289). 그래서 이 경우 입력을 fp32로 바꿔 넘깁니다. 이 fp32 kernel input은 bf16 input보다 예측 오차를
6–8% 줄입니다. fp32 residual stream은 int8 base의 오차를 21–36% 더 줄입니다. stream 자체는
[05_practical_tips.md](05_practical_tips.md)에서 다룹니다.

---

## 3.4 GPTQ

base GPTQ는 17.6분, merged14 teacher GPTQ는 1045초(calibration 수집 18초, 블록당 약 36초, 5070 Ti) 걸렸고, 결과는 2.37 GiB입니다.

### calibration

- 작가 폴더마다 캐시된 이미지를 최대 2장 고릅니다. 토큰 수가 3000 이상 4608 이하인 것만 고르고, seed 0으로 섞습니다.
  캐시된 VAE latent와 TE 출력을 그대로 씁니다.
- 이미지마다 σ를 하나씩 뽑습니다. σ는 추론 schedule(28 steps, flow shift 3.0)에서 고르고, 그 σ로 노이즈를 섞습니다.
- 각 샘플을 두 번 돌립니다. 한 번은 자기 캡션으로, 한 번은 negative prompt로 돌립니다(CFG uncond 쪽). negative는
  `worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia`입니다.
- 블록 0에 들어가는 입력을 잡아 두고, 블록을 **순서대로** 양자화합니다. 양자화가 끝난 블록(W8A8 kernel)의 출력이 다음 블록의
  입력이 되므로, 앞쪽 블록의 양자화 오차가 뒤쪽 calibration에 반영됩니다.

| | base | merged14 |
|---|---|---|
| 작가 / 이미지 / 샘플 | 91 / 182 / 364 | 92 / 184 / 368 |
| linear당 토큰 | 1,487,472 | 1,510,348 |
| σ schedule | 28 steps, shift 3.0 | 28 steps, shift 3.0 |

두 job은 같은 설정으로 돌았고, 고른 작가 수만 91에서 92로 다릅니다.

### 양자화 방법

linear 하나마다 회전한 입력 `X_r`로 `H_r = X_rᵀX_r`를 모읍니다. 같은 grid(256열 회전, 행 스케일 `amax/127`)에서 반올림을
세 가지로 해 봅니다.

- `rtn`: round to nearest입니다.
- `gptq`: `H_r`에 대한 GPTQ입니다. 열 단위로 반올림하고 오차를 남은 열로 넘깁니다. block 128, damp 0.01, act-order를 켭니다.
- `act_aware`: activation 양자화까지 고려한 최소제곱 가중치 `W*`에 GPTQ를 적용합니다(ridge 1e-4).

세 방식 모두 W8A8 출력 오차 `‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²`로 채점합니다. `rtn`과 `gptq`는 weight-only 오차(W8A16)로도
채점합니다. 저장하고 다음 블록의 입력을 만드는 데 쓰는 것은 `gptq`입니다.

### base와 merged14 비교

168개 linear의 평균이고, 단위는 상대 출력 오차 에너지입니다.

| | base | merged14 |
|---|---|---|
| W8A8 RTN → GPTQ | 6.40e-5 → 4.12e-5 (−35.5%) | 6.53e-5 → 4.24e-5 (−35.1%) |
| W8A8 act-aware | 4.10e-5 | 4.22e-5 |
| W8A16 RTN → GPTQ | 2.79e-5 → 0.52e-5 (−81.4%) | 2.84e-5 → 0.55e-5 (−80.6%) |
| 평균 ridge damp | 1.21e-4 | 1.11e-4 |
| W8A8 GPTQ 최악 linear | `blocks.17.cross_attn.output_proj` 1.41e-4 | `blocks.15.cross_attn.output_proj` 1.48e-4 |

linear 종류별 W8A8 GPTQ 평균:

| linear | base | merged14 |
|---|---|---|
| `self_attn.qkv_proj` | 3.68e-5 | 3.72e-5 |
| `self_attn.output_proj` | 6.74e-5 | 7.04e-5 |
| `cross_attn.q_proj` | 4.64e-5 | 4.41e-5 |
| `cross_attn.output_proj` | 5.40e-5 | 5.57e-5 |
| `mlp.layer1` | 2.17e-5 | 2.20e-5 |
| `mlp.layer2` | 2.12e-5 | 2.53e-5 |

soup을 구웠다고 int8이 더 어려워지지는 않았습니다. 전체 평균 차이는 3% 안이고, 가장 크게 움직인 것은 `mlp.layer2`(+19%)입니다.
act-aware가 plain GPTQ보다 나은 점이 없다는 결론도 base에서와 같습니다. merged13 GPTQ도 W8A8 RTN → GPTQ가
6.51e-5 → 4.22e-5로 거의 같았습니다.

### GPTQ가 실제로 줄이는 것 (base 측정)

- GPTQ는 weight 항의 81%를 없앱니다. W8A8 오차는 36% 줄고, 남은 오차는 대부분 activation 항입니다.
- end to end로 보면 cond / uncond 예측 오차가 약 10% 줄고(0.87–0.92x), CFG velocity 오차는 2–7% 줄어듭니다(0.93–0.99x).
  base 자체의 trajectory에서 재도 held-out 실제 이미지에서 재도 같아서, calibration 소스는 결과에 영향이 없습니다.
- CFG velocity에서 이득이 작은 이유는 이렇습니다. weight 오차는 cond와 uncond가 공유하므로 `v_c − v_u`에서 대부분 상쇄되고,
  CFG가 증폭하는 것은 공유되지 않는 activation 반올림입니다.
- LoRA 학습은 CFG 없이 cond 예측을 쓰므로, 학습 쪽 이득은 약 10% 쪽 수치에 해당합니다.

merged14 GPTQ의 end-to-end 오차는 측정하지 않았습니다. 위 수치는 base 기준이고, merged14는 per-linear 오차만 있습니다.

### round to nearest

GPTQ 파일이 없으면 그 자리에서 RTN으로 양자화합니다. GPTQ 대비 손해는 위에서 본 예측 오차 약 10%입니다. RTN은 어떤 DiT에서도
동작하므로, 40블록 Anima-2.9B나 GPTQ를 아직 만들지 않은 merged DiT에서 쓸 수 있는 유일한 선택지입니다.

---

## 3.5 teacher에 전용 GPTQ가 필요한 이유

1. **base 파일은 틀린 값입니다.** forward copy는 그 자체가 가중치입니다. base 파일을 쓰면 teacher의 int8 forward는
   soup 없는 base를 계산하고, 구운 LoRA delta는 int8 forward에서 사라집니다. 그러면 teacher가 soup이 아니게 되고,
   soup을 teacher로 쓰려던 목적이 사라집니다.
2. **GPTQ는 가중치와 activation 통계 모두에 묶여 있습니다.** 반올림은 그 가중치의 `H_r`(그 모델의 블록 입력)에 맞춰 최적화되고,
   블록 i의 calibration 입력은 양자화된 블록 0..i−1을 지나 만들어집니다. 그래서 다른 DiT의 파일을 다시 쓸 방법이 없습니다.
3. **RTN으로 버티는 것과 비교하면** RTN으로도 merged DiT를 int8로 돌릴 수 있습니다. 대신 per-linear W8A8 오차가
   약 1.5배(merged14: 6.53e-5 대 4.24e-5)이고, base 측정 기준으로 예측 오차가 약 10% 큽니다. 만드는 데 17분이면 되므로
   전용 GPTQ를 만드는 쪽을 택했습니다.

soup이 바뀔 때마다(merged13 → merged14) bake와 GPTQ를 다시 해야 합니다.

int8 teacher(전용 GPTQ)와 bf16 teacher로 학습한 터보 student를 비교한 A/B는 없습니다. int8 base가 터보 품질에 미치는
영향은 위의 per-linear / per-prediction 오차로만 간접적으로 알 수 있습니다.

GPTQ calibration은 28-step, shift 3 추론 schedule에서 σ를 뽑습니다. 4-step student가 지나는 σ에서 따로 오차를 잰 적이
있는지는 확인하지 못했습니다. base 측정은 σ 구간 세 개 모두에서 0.87–0.92x로 고르게 나왔습니다.

---

## 3.6 다른 섹션

- LoRA 학습 레시피: [01_lora_recipe.md](01_lora_recipe.md)
- soup, merged14 LoRA를 만드는 과정: [02_soup_merge.md](02_soup_merge.md)
- DP-DMD 터보 루프, 레시피, 평가: [04_turbo_distillation.md](04_turbo_distillation.md)
- fp32 stream: [05_practical_tips.md](05_practical_tips.md)
