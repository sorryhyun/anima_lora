# 2. LoRA soup과 merge: 여러 LoRA를 하나로

> 초안 (2026-11-12). 이 문서를 쓰면서 CPU로 직접 잰 값은 **[측정]**으로 표시했다.

**요약**

- soup은 같은 설정으로 시드만 바꾼 LoRA 여러 개를 ΔW 수준에서 평균한 것이다. 한 번 학습한 LoRA의 ΔW는
  대부분 그 런에만 있는 성분이고, 평균하면 그 성분이 줄고 공통 성분이 남는다. 그래서 시드 운에 덜 흔들린다.
- merge는 서로 다른 LoRA(작가 shard 등)를 up/down 행렬을 이어 붙여(concat) 하나의 LoRA로 만든다.
  SVD로 자르지 않으므로 Σ w_i·ΔW_i가 정확히 보존되고, 대신 rank가 Σ r_i로 커진다.
- turbo teacher인 merged14는 91명 작가를 8개 shard로 나눠 shard마다 3-seed soup(rank 32)을 만들고, 8개를
  global normalize, adaln 평균으로 이어 붙인 rank 256 LoRA다.
- turbo를 배포할 때는 normalize 없이, adaln도 더해서 merged14와 turbo LoRA를 합친다. global normalize와 adaln
  평균으로 합치면 ΔW14 + ΔT에서 35–50 % 벗어난다 **[측정]**.

---

## 2.1 파이프라인에서의 위치

```
작가별 LoRA 레시피 (→ 01_lora_recipe.md)
  └─ shard마다 3 seed 학습 → ΔW soup (rank 32)            ← 이 문서
       └─ 8 shard soup을 concat_exact merge → merged14 (rank 256)   ← 이 문서
            └─ DiT에 bake + merged14 전용 GPTQ (→ 03_bake_int8_gptq.md)
                 └─ 4-step DP-DMD 증류 → turbo LoRA ΔT (→ 04_turbo_distillation.md)
                      └─ merged14 ⊕ ΔT 를 exact merge → base에 올리는 LoRA 하나   ← 이 문서
```

이 단계의 결과물인 merged14의 값은 파일 메타데이터와 텐서 형상에서 읽었다 **[측정]**.

| 항목 | 값 |
|---|---|
| 합친 방식 | concat_exact |
| normalize / global scale | global / 0.342427 |
| adaln 합치기 | 평균 |
| 입력 가중치 | 1.0 × 8 |
| 재료 | 8 shard soup |
| 모듈 수 | 364 (일반 280, adaln modulation 84) |
| rank | 일반 모듈 256 (= 8 × 32), adaln 128 (= 8 × 16), 모든 모듈에서 alpha = rank |

## 2.2 왜 soup인가

### 한 번의 학습은 시드 운을 탄다

같은 레시피, 같은 init, 같은 데이터(334장)에서 학습 시드만 바꾼 세 런을 비교했다(2026-07-05). 지표는 실제
미공개 작품과의 CMMD(PE-Core MMD², 낮을수록 좋음)이고, real-vs-real 노이즈 바닥은 0.155다.

| 런 | CMMD |
|---|---|
| s1001 / s1002 / s1003 | 0.46 / **1.02** / 0.49 |
| 세 런의 rank-16 SVD soup | **0.469** |

s1002는 plain LoRA만큼 나쁜데, membership AUC(0.88)로는 드러나지 않았다. soup은 나쁜 시드가 1/3 섞여
있어도 가장 좋은 재료와 노이즈 바닥 안에서 같았다. soup은 **암기(memorization) 대책이 아니다**. soup의 AUC는
재료들의 최댓값 근처에 있다.

### ΔW 대부분은 그 런에만 있다

같은 설정을 두 번 돌리면 ΔW 코사인 중앙값이 0.146(bs1, 5e-5)이고, GA2 1e-4에서는 0.115다.
merged14 재료로 다시 재 보면 더 낮다 **[측정]**:

| 재료 세트 | 레시피 | 에너지 비중 | 쌍별 full-sum cos |
|---|---|---|---|
| merged14 shard 1 (337장) | 5e-5 constant, GA2 | 36.7 / 32.5 / 30.8 % | 0.082–0.090 |
| merged14 shard 7 (639장) | 〃 | 35.5 / 34.4 / 30.2 % | 0.055–0.058 |
| merged13 shard 1 | 2e-5 cosine, bs1, REPA on | 34.0 / 33.5 / 32.4 % | 0.193–0.202 |
| merged13 shard 7 | 〃 | 32.2 / 33.7 / 34.1 % | 0.217–0.227 |

N개 재료를 평균하면

$$
\Big\|\tfrac1N\textstyle\sum_i \Delta W_i\Big\|_F^2=\tfrac1{N^2}\Big(\sum_i\|\Delta W_i\|_F^2+2\sum_{i<j}\langle\Delta W_i,\Delta W_j\rangle_F\Big)
$$

이므로, 재료의 노름이 비슷하고 쌍별 cos가 c이면 soup 에너지는 재료 하나의 (1 + (N−1)c)/N배다. 공통 성분은 그대로
남고, 런마다 다른 성분은 노름 기준 1/√N으로 줄어든다. merged14 shard 1에서 확인하면 soup의 ‖ΔW‖²는
9.839² + 7.694² ≈ 156.0이고 재료 평균은 406.1이므로 비율은 0.384다. c ≈ 0.085로 계산한 (1 + 2c)/3 ≈ 0.39와
맞는다 **[측정]**. soup은 rank 32로 잘린 파일이라 정확히 같지는 않다(아래 retained energy 참고).

주의할 점이 하나 있다. 채널 스케일링의 dead channel에서 down 행렬 컬럼이 무작위로 떠돌아, 676-update soup
shard에서는 저장된 ‖ΔW‖²의 33–43 %가 출력에 거의 기여하지 않는 컬럼에 실린다. merged14 재료는 이를 막는 floor
없이 학습됐다. 그래서 위 cos와 에너지 비중은 출력(함수 공간) 기준보다 낮게 또는 다르게 읽힐 수 있다.

> TODO(확인 필요): dead-column을 뺀 merged14 재료의 cos / 에너지 비중. 측정 기록을 찾지 못했다.

## 2.3 ΔW 수준에서 섞어야 하는 이유

A(down)와 B(up)를 따로 평균하면 두 가지가 틀어진다.

1. `avg(B) @ avg(A) ≠ avg(B @ A)`.
2. down을 W0의 SVD로 초기화하면 특이벡터의 부호가 런마다 달라질 수 있어서, A를 행 단위로 평균하면 행이
   서로 지워질 수 있다.

ΔW는 재료마다 (A, B)를 어떻게 다시 매개화하든 변하지 않는다. 그리고 rank-r ΔW들의 가중 평균은 블록을 이어
붙인 rank-Σr LoRA 하나로 **정확히** 쓸 수 있다.

$$
\Delta W_{soup}=\sum_i w_i\,s_i\,U_iD_i=\big[\,w_1s_1U_1\;\big|\;w_2s_2U_2\;\big|\cdots\big]\begin{bmatrix}D_1\\ D_2\\ \vdots\end{bmatrix},\qquad s_i=\alpha_i/r_i
$$

결과의 alpha를 rank로 둬서 scale을 1로 만든다. 출력은 두 가지다.

| 방식 | 출력 |
|---|---|
| 정확한 concat | rank = Σ r_i |
| truncated soup | 평균 ΔW를 상위 r개 특이방향으로 자름(Eckart–Young). 전체 ΔW는 만들지 않고, 이어 붙인 인자를 QR로 줄여 (R×R) 코어만 SVD한다 |

truncated soup은 모듈마다 r을 재료의 rank로 제한한다. 그래서 adaln(rank 16)은 r = 32에서도 16으로 남고, shard
soup의 adaln rank가 16인 이유가 이것이다. 모듈별로 남은 에너지 비율은 다음과 같다 **[측정, 메타데이터]**.

| soup | 재료 | rank | retained energy 평균 | 최솟값 |
|---|---|---|---|---|
| 3-seed (uncond init) | rank 16 | 16 | 99.87 % | 96.8 % |
| merged13의 8 shard (nouncond, svd_slice) | rank 32 × 3 | 32 | 99.70–99.82 % | 90.4–94.2 % |
| merged14의 8 shard (nouncond, svd_slice) | rank 32 × 3 | 32 | 98.59–99.18 % | 92.8–95.7 % |

재료끼리 덜 겹칠수록(merged14 cos 0.06–0.09, merged13 0.19–0.23) 평균 ΔW의 실제 rank가 커지고, rank 32로
자를 때 잃는 에너지도 늘어난다.

**재료 점검.** soup을 만들기 전에 재료별 에너지 비중과 쌍별 cos를 본다. 비중은 대략 고르게 나와야 한다. cos가
낮다는 사실만으로 재료를 빼지는 않는다.

## 2.4 uncond-init soup

| 단계 | 하는 일 |
|---|---|
| 1. uncond inter-train | caption dropout 1.0으로 희석된 pool(기본은 데이터 전체, fine-tune 대상은 항상 포함)을 짧게 학습. seed 1000 고정 |
| 2. seeded fine-tune | 대상 이미지로 N개(기본 3) 학습. seed는 1001…1000+N, 1단계 체크포인트에서 시작 |
| 3. soup | rank-r SVD soup |

1단계 pool은 대상 작가가 소수가 되도록 넓게 두고, 같은 pool을 쓰는 soup들은 uncond init 하나를 공유한다. 지금
있는 uncond init은 pool 전체, lr 2e-5, 4 epoch이고, 3,008장에 12,032 step이다.

svd_slice ≠ 0이면 uncond init을 쓸 수 없다. uncond init에서 시작하면 A가 slice 0 행으로 덮어써져 slice가
사라지기 때문이다.

## 2.5 서로 다른 LoRA를 하나로

### 계산

입력마다 `(α_i/r_i)·w_i`를 up에 접고 down과 up을 이어 붙인다.

```
down_merged = cat_i(down_i)              # (Σr_i, in)
up_merged   = cat_i((α_i/r_i)·w_i·up_i)  # (out, Σr_i)
alpha       = Σr_i                       # scale 1 → multiplier 1.0이 Σ w_i·ΔW_i
```

soup의 식과 같다. 다른 점은 다음과 같다.

| | soup | merge |
|---|---|---|
| 기본 가중치 | 1/N (평균) | 1.0 (합) |
| 모듈 집합 | 모든 재료가 같아야 함 | 합집합. 일부 입력에만 있는 모듈은 그 입력의 블록만 들어감 |
| SVD truncation | 선택 | 없음. 항상 concat |
| 크기 보정 | 없음 | normalize |
| adaln 처리 | 다른 모듈과 같음 | 평균 또는 합 |

| 설정 | 값 | 동작 |
|---|---|---|
| normalize | global | 일반 모듈 전체에 스칼라 하나를 곱해, 합친 ΔW의 전체 노름이 "입력 하나의 평균 노름"이 되게 한다 |
| | per_module | 모듈마다 따로 맞춘다. 층 사이 비율이 바뀐다 |
| | off | 그대로 둔다. 정확한 concat 합 |
| adaln | average | adaln modulation 모듈은 가중 평균(`w_i / Σw`)으로 합치고 normalize에서 뺀다 |
| | sum | 다른 모듈과 똑같이 합치고 normalize 대상에 넣는다 |

### global normalize가 하는 일

거의 직교하는 ΔW N개를 그냥 더하면 노름이 √N배로 커져 DiT를 과구동한다. global normalize는 모듈 m마다
`target_m` = 입력별 ‖(α_i/r_i)·up_i·down_i‖의 평균(가중치 w는 넣지 않은 값)을 구하고, 다음 스칼라를 일반 모듈
전체에 곱한다.

$$
g=\sqrt{\frac{\sum_m \text{target}_m^2}{\sum_m \|\Delta W^{merged}_m\|_F^2}}
$$

여기서 두 가지가 나온다.

- 입력이 서로 직교하고 노름이 같으면 g = 1/√N이다. 노름이 고르지 않으면 1/√N보다 작다. merged14의 g = 0.342427은
  1/√8 = 0.354보다 조금 작은데, 8 shard의 cos가 거의 0이고(아래 표) 노름이 9.8–19.7로 고르지 않은 것과 맞는다.
  merged14 안에서 shard 하나는 단독 세기의 0.342배로 들어간다. 평균(1/8 = 0.125)보다 2.7배 세다.
- target에 w가 들어가지 않으므로 global normalize에서 입력별 가중치는 **입력 사이의 비율만** 정한다. 가중치 전체를
  몇 배 하든 결과는 같다. 절대 세기를 정하려면 normalize를 끈다.

### adaln을 평균하는 이유

adaln modulation의 입력은 σ만으로 정해지는 벡터다. 따라서 adaln ΔW는 모든 이미지에 똑같이 걸리는 채널별
shift/scale/gate 편향이 된다. 이 입력은 σ ∈ [0, 1]에서 거의 일정하고(평균과의 cos ≥ 0.99) 에너지의 73 %가 W0의
첫 번째 우특이벡터에 있다. 그래서 svd_slice는 adaln을 자르지 않고, 모든 shard가 같은 adaln init을 공유한다.
실제로 merged14의 8 shard를 재 보면 일반 모듈은 서로 직교하지만 adaln은 겹친다 **[측정]**:

| 8 shard soup 쌍 (28쌍) | full-sum cos 최소 / 평균 / 최대 |
|---|---|
| 일반 모듈 (280) | −0.0074 / 0.0003 / 0.0105 |
| adaln (84) | 0.115 / 0.126 / 0.138 |

일반 모듈은 shard마다 다른 입력 부분공간에 있어서 더해도 서로 간섭하지 않는다. adaln은 같은 init에서 출발한 전역
편향 8개라 일부(cos ≈ 0.13)가 겹치고, 그대로 더하면 공통 성분이 shard 수만큼 쌓인다. adaln을 더하지 않고 평균하는
이유다.

### concat_exact와 SVD식 merge

| 방식 | 정확도 | rank / 크기 |
|---|---|---|
| concat_exact | normalize off, adaln 합일 때만 Σ w_i ΔW_i 그대로(bf16 출력은 α/r ≠ 1인 입력에서 ~1e-3 반올림). global normalize / adaln 평균은 delta를 다시 스케일하므로 Σ w_i ΔW_i가 아니다 | Σ r_i. merged14 871 MB(shard soup 하나는 109 MB) |
| ΔW 평균 + rank-r SVD | 상위 r 방향만. 잃는 크기는 위 retained energy 표 | r |
| 두 DiT의 차이를 SVD | ΔW = tuned − base를 rank r로 자름 | r |

shard처럼 서로 직교하는 입력을 SVD로 자르면 손실이 크다. svd_slice가 다른 재료들을 한 soup에서 rank r로 자르면
에너지의 약 (N−1)/N이 사라진다. 그래서 shard 안에서는 SVD soup을 쓰고, shard끼리는 concat한다.

## 2.6 merged14를 실제로 만든 방법

### 계보

| 파일 | 날짜 | 재료 | 비고 |
|---|---|---|---|
| merged11 | 2026-10-13 | 8 shard soup (nouncond, svd_slice k−1) | global 0.321908, adaln average |
| merged12 | 2026-11-02 | 8 shard soup, 재료 LR 1e-5 / 2e-5 / 5e-5 | global 0.320523. slice 배치가 달랐다 |
| merged12_calibrated | 2026-11-02 | 재료를 에너지로 재가중 | **렌더가 망가짐.** 2.8 참고 |
| merged12_shardbalanced | 2026-11-02 | shard 비중만 조정(가중치 0.663–1.465) | merged12와 거의 같은 렌더 |
| merged13 | 2026-11-04 | 8 shard × 3 seed, 2e-5 cosine | global 0.302609. teacher였으나 지금은 대체됨 |
| merged14 | 2026-11-07 | 8 shard × 3 seed, 5e-5 constant(01 레시피) | global 0.342427. 현재 teacher |

> TODO(확인 필요): merged13에서 merged14로 바꾼 결정의 렌더 근거. 두 soup을 렌더로 비교한 기록은 찾지 못했다.
> 01 레시피가 2026-11-06에 확정된 직후 재료를 새 레시피로 다시 학습한 것으로 보인다.

### 재료 레시피: merged13과 merged14

재료의 메타데이터에서 읽었다 **[측정]**.

| 항목 | merged13 | merged14 |
|---|---|---|
| LR / scheduler | 2e-5 / cosine | 5e-5 / constant with warmup (warmup 5 %) |
| batch | 1, GA 1 | 1, GA 2 |
| clip | 고정 grad norm 1.0 | median 기준, k = 2, micro-batch마다, σ band 0.2/0.4/0.6/0.8 |
| adapter EMA | 없음 | window 0.1 |
| REPA | on | off |
| caption dropout | 0.05 | 0.1 |
| rank / alpha | 32 / 128 | 32 / 128 |
| adaln | rank 16, 학습 | 같음 |
| down init | W0 SVD + svd_slice = k−1 | 같음 |
| epoch | 8 | 8 |
| init | uncond init 없음 (1단계 없음) | 같음 |
| seed | 1001 / 1002 / 1003 | 같음 |

merged14는 uncond-init soup 파이프라인으로 만들었지만 **uncond init은 쓰지 않았다**. shard마다 다른 svd_slice를
쓰려면 uncond init을 빼야 하기 때문이다(2.4).

### shard 구성

merged14는 2026-10-11 기준 91개 작가 폴더를 8개로 나눴다(shard 1–3은 12명, 4–8은 11명). 장 수와 step은 재료
메타데이터, 에너지 비중은 merged14에 들어간 shard soup의 일반 모듈 ‖ΔW‖² 비율이다 **[측정]**.

| shard | svd_slice | 장 수 | merged14 step | merged13 step | merged14 retained energy (평균 / 최소) | merged14 내 에너지 비중 |
|---|---|---|---|---|---|---|
| 1 | 0 | 337 | 1,352 | 2,696 | 99.01 / 95.51 % | 6.1 % |
| 2 | 1 | 428 | 1,712 | 3,424 | 98.83 / 93.31 % | 13.7 % |
| 3 | 2 | 410 | 1,640 | 3,280 | 98.85 / 94.20 % | 12.7 % |
| 4 | 3 | 471 | 1,888 | 3,768 | 98.81 / 93.78 % | 15.0 % |
| 5 | 4 | 383 | 1,536 | 3,064 | 98.94 / 95.05 % | 13.5 % |
| 6 | 5 | 313 | 1,256 | 2,504 | 99.10 / 94.76 % | 7.8 % |
| 7 | 6 | 639 | 2,560 | 5,112 | 98.59 / 92.77 % | 24.4 % |
| 8 | 7 | 261 | 1,048 | 2,088 | 99.18 / 95.66 % | 6.9 % |

합계 3,242장, 재료 런 24개다. 재료 한 런은 약 20분(shard 8, 261장)에서 49분(shard 7, 639장)이 걸렸다. GA 2이므로
optimizer update는 step 수의 절반이다(shard 1: 676 update).

shard의 에너지 비중은 6–24 %로 고르지 않고, 대체로 장 수(= step 수)를 따른다. merged12에서 shard 비중만 맞춘
shardbalanced는 merged12와 거의 같은 렌더였으므로(2.8), merged14에서는 shard 비중을 맞추지 않았다.

### 확인

merged14가 정확히 "0.342427 × 8 shard 합 + adaln 평균"인지 Gram trace로 쟀다 **[측정]**
(‖X − Y‖² = ‖X‖² + ‖Y‖² − 2⟨X, Y⟩, ⟨UD, U′D′⟩ = tr((UᵀU′)(D′Dᵀ)), ΔW를 직접 만들지 않음, CPU).

| 모듈 | 기대값 | 상대 Frobenius 오차 |
|---|---|---|
| 일반 280 | 0.342427 · Σ_k ΔW_k | 1.7e-3 (bf16 반올림) |
| adaln 84 | (1/8) · Σ_k ΔW_k | 1.4e-8 |

## 2.7 정확성이 왜 중요한가: turbo 배포 설정

turbo LoRA ΔT는 **merged14가 1.0으로 구워진 DiT** 위에서 증류됐다(→ 03_bake_int8_gptq.md, 04_turbo_distillation.md).
즉 student가 배운 것은 "W0 + ΔW14 위에 얹는 ΔT"다. ΔT만 base에 올리면 의미가 없고, base에 올리는 배포 파일은
정확히 ΔW14 + ΔT여야 한다.

- normalize off: merged14의 up에는 이미 0.342427이 접혀 있고 alpha = rank라 scale이 1이다. 그대로 이어 붙이면
  merged14 부분은 teacher의 정의 그대로 들어간다. merged14 내부의 global scale은 teacher를 정의하는 일부일 뿐이고,
  다시 normalize하면 teacher 자체가 바뀐다.
- adaln 합: merged14의 adaln(이미 8 shard의 평균)과 ΔT의 adaln은 더해야 한다. 평균은 둘을 반으로 나눈다.
- multiplier는 1.0으로만 쓴다. 이 파일의 multiplier는 ΔW14와 ΔT를 함께 늘리거나 줄인다.

turbo LoRA(r64/α180, adaln r16/α90)로 직접 확인했다 **[측정]**. 기대값 ΔW14 + ΔT와의 상대 오차를 위 Gram
trace로 쟀다.

| 설정 | 출력 rank (일반 / adaln) | 일반 모듈 오차 | adaln 오차 |
|---|---|---|---|
| normalize off, adaln 합 (bf16) | 320 / 144 | 8.1e-4 | 1.2e-3 |
| normalize off, adaln 합 (fp32) | 320 / 144 | 0 | 0 |
| global normalize, adaln 평균 | 320 / 144 | **0.350** | **0.500** |

- global normalize에서는 scale 0.6498이 merged14와 ΔT 모두에 곱해지고, adaln은 반이 된다. 그 결과 teacher
  부분까지 0.65배로 약해진다.
- bf16 출력의 오차 ~1e-3은 ΔT의 up에 α/r = 2.8125(adaln 5.625)를 곱한 뒤 bf16으로 반올림해서 생긴다. merged14 쪽은
  α/r = 1이라 반올림이 없다. fp32로 쓰면 비트 단위로 정확하지만 파일이 2,144,802,712 bytes로 bf16(1,072,469,152 bytes)의
  두 배다.

> TODO(확인 필요): bf16 반올림(상대 ~1e-3)이 4-step 렌더에 보이는지. 배포는 bf16으로 해 왔고, fp32 배포와 비교한 기록은 없다.

입력 사이 간섭을 재 보면 merged14와 ΔT는 서로 직교한다(energy ratio 1.006, cos +0.006, 부분공간 overlap 2.2× random) **[측정]**.

## 2.8 정착된 교훈

### LR이 섞인 재료를 나중에 에너지로 맞추지 않는다

merged12는 shard마다 재료 LR을 1e-5 / 2e-5 / 5e-5로 나눴고, 5e-5 재료가 shard soup 에너지의 81–88 %를
차지했다. 이걸 문제로 보고 재료를 ΔW 크기의 역수로 재가중한 merged12_calibrated를 만들었더니
(1e-5 재료가 원래 가중치의 3–4.6배) 선이 뭉개지고 손과 눈이 번지고 표지판 글자가 녹았다. 재료는 그대로 두고 shard
soup 8개의 크기만 맞춘 shardbalanced는 merged12와 거의 같았다.

- 에너지 비중은 재료가 무엇을 배웠는지 알려 주지 않는다. 5e-5의 90 % 비중이 그 자체로 결함은 아니었다.
- 크기는 shard 사이에서만 맞춘다. 맞췄으면 보정됐다고 말하기 전에 렌더한다.
- model soup 논문(Wortsman et al.)은 greedy selection으로 나쁜 재료를 걸렀지만, 이 파이프라인은 게이트 없는 균일
  평균이라 나쁜 LR 재료도 그대로 섞인다. merged13/14는 seed만 다른 재료로 돌아갔다.

### ΔW cos는 겹침의 크기를 읽는 도구다

Gram trace로 ΔW의 노름과 쌍별 cos를 CPU에서 잰다. 같은 목적과 같은 init에서 출발한 두 런의 cos는 둘이 공유하는
ΔW 비율로 읽는다.

| 비교 | cos |
|---|---|
| 01 레시피 LoRA, 같은 설정 두 번 | 0.146 (중앙값) |
| merged14 shard 재료끼리 | 0.055–0.090 (2.2) **[측정]** |
| merged14 shard soup끼리 (일반 모듈) | ≈ 0 (2.5) **[측정]** |
| turbo arm 0, seed만 다른 쌍 | 0.087–0.100 |

여기서 나오는 결론은 두 가지다.

- 시드 하나로 학습한 결과물의 대부분은 그 런에만 있는 성분이다. soup이 효과가 있는 이유가 이것이고, 같은 이유로
  "endpoint ΔW cos로 두 설정을 비교"하는 것은 신호가 없다.
- 공통 성분이 거의 없는 입력(shard)끼리는 평균하면 각자가 1/N로 약해지고, 더하면 노름이 √N배로 커진다. merged14는
  global normalize로 그 사이(shard당 0.342)를 골랐다.

turbo checkpoint를 ΔW로 평균해 random walk를 줄이는 시도도 있었다. 늦게 생긴 drift는 없앴지만 250 step 체크포인트를
넘지 못했다(→ 04_turbo_distillation.md).

### 그 밖에

- merged14 재료는 dead-column 문제(2.2)가 있는 채로 노름을 쟀다. global normalize와 에너지 비중을 해석할 때 이 점을
  감안한다. 채널 스케일링 floor 0.01이 dead-column 비중을 0으로 만든다.
