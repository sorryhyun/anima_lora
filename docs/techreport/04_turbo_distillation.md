# 4. 터보 증류: 공식 turbo에서 warm start하고 스텝 0을 맞추기

soup을 bake한 DiT(03편)를 teacher로 4스텝 student LoRA를 만듭니다. 처음에는 DP-DMD 루프 전체(DM, CDM, fake,
GAN)를 cold start로 돌렸는데, 그렇게 만든 student는 시드가 달라도 프롬프트마다 구도를 하나만 그렸습니다. 이 글은
그 뒤에 자리 잡은 방법을 적습니다. 공식 turbo 릴리스의 delta에서 출발해 스텝 1–3은 그대로 두고, 스텝 0만 teacher
궤적에 회귀시킵니다. 공동 루프의 구성과 측정은 부록 A4에 있습니다.

## 1. 배경

### 1.1 4스텝 student와 DP-DMD 앵커

teacher는 28스텝 × (cond + uncond)로 이미지당 DiT forward 56회입니다. student는 CFG를 구워 넣은 LoRA이고,
shift 3의 static 격자 σ 1 → 0.9 → 0.75 → 0.5 → 0을 cfg 1로 4번 forward합니다. student 격자는 teacher 격자의
부분집합이라 두 궤적은 같은 σ에서 만납니다.

DMD(Yin et al. 2023)는 student 샘플의 분포를 teacher 분포 쪽으로 미는데, 이 gradient는 reverse-KL이라 시드 간
구도를 한 mode로 모읍니다. DP-DMD(Wu et al. 2026)는 일을 나눕니다. 스텝 0은 teacher 궤적에 회귀시켜 구도를 정하게
하고, DM은 나머지 스텝에만 줍니다. 스텝 0의 목표는 같은 노이즈 ε에서 CFG teacher를 k스텝 굴린 상태 $z_{t_k}$까지의
평균 velocity입니다.

$$
v_{\text{target}} = \frac{\varepsilon - z_{t_k}}{1 - t_k}, \qquad
\mathcal{L}_{\text{anchor}} = \big\| G_\theta(\varepsilon, 1, c) - v_{\text{target}} \big\|^2
$$

이 회귀를 앵커라고 부릅니다. 목표가 (프롬프트, ε)마다 정해지므로 시드마다 다른 구도가 그대로 남습니다.

### 1.2 문제: 구도는 스텝 0에서 잃는다

같은 teacher에서 DP-DMD 공동 루프로 학습한 student는 merged14와 merged15 모두에서 프롬프트마다 구도를 하나만
그렸습니다. 어느 스텝이 구도를 잃는지는 handoff probe로 쟀습니다. teacher가 student 격자의 경계 b까지 내려가고
student가 나머지를 마치는 arm(`T>S@b`)을 student 혼자 그린 것(`S`)과 함께 그리고, 각 최종 latent와 teacher 이미지의
cos를 8× average pool한 뒤 채널 평균을 빼고 잽니다. 같은 노이즈로 teacher 이미지에 얼마나 가까운지를 보는 숫자이고,
품질 점수는 아닙니다. 다른 구도를 같은 품질로 그려도 낮게 나옵니다.

student에 스텝을 하나씩 더 넘길 때 떨어지는 cos입니다(merged14 student 두 개, 프롬프트 3개 × 시드 4개).

| student가 맡는 스텝 | σ 구간 | 손실 |
|---|---|---|
| 3 | 0.5 → 0 | 0.02–0.05 |
| 2 | 0.75 → 0.5 | 0.05–0.09 |
| 1 | 0.9 → 0.75 | 0.13–0.16 |
| 0 | 1 → 0.9 | 0.24–0.27 |

스텝 0에서 가장 많이 잃습니다.

## 2. 공식 turbo의 delta: 좋은 스텝 1–3, 맞지 않은 스텝 0

공식 turbo v1.1은 base DiT 위의 few-step 모델입니다. base에 대한 delta를 rank 96 ASVD로 뽑아(adaLN 포함,
부록 A4.6) LoRA로 씁니다. 이 저장소가 배포한 base teacher 터보(Hub v3)는 이 delta에서 warm start해 공동 루프로
10k iteration 학습한 것입니다.

두 LoRA를 학습 없이 merged15 위에서 handoff probe로 쟀습니다. `T>S@1`은 teacher가 σ 0.9까지 그리고 student가
스텝 1–3을 맡은 arm이고, 앵커 MSE는 스텝 0의 velocity와 앵커 목표의 거리입니다. sameness는 시드 간 유사도로,
낮을수록 시드마다 다르게 그립니다(teacher 0.315).

| 스텝 0 / 스텝 1–3 | sameness | teacher 이미지와 cos | `T>S@1` cos | 앵커 MSE |
|---|---|---|---|---|
| v1.1 / v1.1 | 0.396 | 0.677 | 0.932 | 0.138 |
| Hub v3 / Hub v3 | 0.296 | 0.635 | 0.805 | 0.062 |
| Hub v3 / v1.1 | 0.318 | 0.742 | | |

- v1.1의 스텝 1–3은 teacher의 스텝 0을 받으면 0.932로 마칩니다. merged teacher에서 학습한 student들은
  0.745–0.841이었습니다.
- v1.1의 스텝 0은 앵커 MSE 0.138로, cold start student와 같은 수준입니다.
- Hub v3의 10k iteration은 스텝 0을 맞췄고(0.138 → 0.062) 스텝 1–3을 잃었습니다(0.932 → 0.805).
- Hub v3의 스텝 0에 v1.1의 스텝 1–3을 붙이면 학습 없이 시드가 teacher만큼 갈리고(0.318) teacher 이미지에 가장
  가깝습니다.

그래서 목표를 v1.1의 스텝 1–3에 맞춘 스텝 0을 더한 것으로 잡았습니다. merged15는 base에 충분히 가까워서 delta가
그대로 옮겨 갑니다.

## 3. 스텝 0만 맞추기

### 3.1 한 iteration

1. 프롬프트 하나와 노이즈 ε 하나를 뽑습니다.
2. frozen teacher(merged15, CFG 4)를 ε에서 자기 추론 격자를 따라 σ 0.75까지 굴리고 $v_{\text{target}}$을 구합니다.
   gradient는 없습니다.
3. student를 σ 1에서 ε에 한 번 forward하고, 그 velocity와 $v_{\text{target}}$의 MSE를 loss로 씁니다.

스텝 1–3은 loss를 받지 않고, 스텝 0과 공유하는 가중치를 통해서만 움직입니다. DM, CDM, fake 네트워크가 모두
빠집니다.

### 3.2 앵커 목표: teacher 28스텝 궤적의 chord

앵커를 teacher의 12스텝 롤아웃에서 6스텝 굴린 상태(σ 0.75)로 잡으면, 끝 상태가 28스텝 teacher 이미지와 다른
그림이 됩니다. teacher는 28스텝으로 렌더하고 student도 그것에 대해 평가하므로, 앵커를 같은 28스텝 격자의 14스텝째
상태로 옮겼습니다. 앵커 목표 자체를 스텝 0으로 넣고 student의 스텝 1–3이 마치게 하면(`A>S@1`) 차이가 보입니다.

| 앵커 | 끝 σ | `A>S@1` cos | sameness (teacher 0.315) |
|---|---|---|---|
| 12스텝 중 6 | 0.75 | 0.818 | 0.346 |
| 28스텝 중 14 | 0.75 | 0.967 | 0.249 |
| 28스텝 중 7 (teacher 자신의 σ 0.9 상태) | 0.9 | 0.945 | 0.313 |

- 비용을 만드는 것은 12스텝 롤아웃입니다. 같은 σ 0.75까지 가도 28스텝 궤적의 chord는 teacher 이미지를 거의 그대로
  남깁니다.
- σ 0.75까지의 chord가 teacher 자신의 σ 0.9 상태보다 스텝 0으로 낫습니다.
- 앵커는 Euler 스텝마다 teacher forward가 2번이므로 14 / 28은 iteration당 29 forward입니다.

### 3.3 공동 루프와 비교

같은 init(v1.1 delta를 rank 64로 자른 것), 같은 길이(250 iteration)에서 공동 루프와 앵커만 쓴 런을 비교했습니다.
이때 앵커는 12스텝 중 6입니다.

| | sameness | `S` cos | `T>S@1` cos | 앵커 MSE |
|---|---|---|---|---|
| 시작(v1.1, 학습 없음) | 0.396 | 0.677 | 0.932 | 0.138 |
| 공동 루프 | 0.383 | 0.721 | 0.922 | 0.107 |
| 앵커만 | 0.350 | 0.766 | 0.946 | 0.082 |

앵커만 쓴 런이 네 숫자 모두에서 앞섰고, 스텝 1–3을 건드리지 않았습니다(`T>S@1` 0.946). 공동 루프는 선이 또렷하고
색이 진한 대신 구도를 일부 되돌렸고, 카페 간판 글자를 네 시드 모두에서 깨뜨렸습니다. 앵커만 쓴 런은 구도가 4스텝
student 중 teacher에 가장 가까웠지만 마감이 무르고 사물이 깨졌습니다(기타 몸통이 덩어리가 되고, 그룹 아래쪽
인물이 합쳐짐). teacher의 스텝 0을 넣으면 같은 student의 스텝 1–3이 깨끗하게 마치므로, 깨지는 원인은 스텝 0입니다.

### 3.4 앵커 캐시와 데이터 규모

앵커 목표는 (프롬프트, ε)와 frozen teacher로만 정해지므로 미리 계산해 둘 수 있습니다. 캐시를 쓰면 iteration은
student forward와 backward 한 번이라 0.4초로 줄어듭니다(라이브 앵커는 3.4초). 데이터셋 캡션 3242개 × ε 2개로
6484쌍을 만들고, 한 번씩만 봅니다(`grad_accum` 2, 3242 update, 43분).

held-out 앵커 MSE(평가 프롬프트 3개 × 시드 4개)는 쌍 수를 따라 내려가지만 기울기가 얕습니다.

| 캐시 쌍 | student update | 학습 앵커 MSE | held-out 앵커 MSE |
|---|---|---|---|
| 128 (각 약 4번) | 500 | 0.036 | 0.089 |
| 1024 (각 1번) | 512 | 0.065 | 0.078 |
| 6484 (각 1번) | 3242 | 0.058 | 0.070 |

- 같은 쌍을 다시 보면 그 쌍에서만 맞습니다. 128쌍 런은 학습 MSE가 가장 낮고 held-out이 가장 높습니다.
- 쌍을 6–8배 늘릴 때마다 held-out이 0.01쯤 내려갑니다.
- 6484쌍 런의 `S` cos는 0.760 → 0.801로 올랐고, `T>S@1` 0.944와 sameness 0.300은 유지했습니다. 기타리스트
  seed 1이 teacher의 자세로 돌아왔고, 그룹 아래쪽 인물은 여전히 합쳐졌습니다.

### 3.5 LR과 rank

- **LR.** 맞춤은 학습 끝의 LR이 정합니다. 상수 LR은 250에서 1000 iteration까지 제자리였고, cosine으로 끝을
  내리면 내려갔습니다. 레시피는 5e-5에서 시작해 cosine으로 5e-6까지 내립니다. student LR 2e-5는 6484쌍에서
  held-out 앵커 MSE를 0.0695 → 0.0674로 낮췄지만 `S` cos(0.801 → 0.792)와 sameness(0.300 → 0.284)가 함께
  내려갔습니다. 앵커에는 조금 더 가깝고, 이미지는 teacher에서 멀어지며 서로 비슷해집니다.
- **rank.** delta 전체를 쓰는 rank 96(alpha는 √rank 비례로 220)은 rank 64와 held-out 앵커 MSE가 0.001 안에서
  같았습니다. rank는 맞춤을 제한하지 않습니다.
- **grad 누적.** 쌍 4개를 한 update에 평균하는 것과 update 4번은 같은 곳에 도착했습니다. 맞춤은 본 쌍의 수를
  따릅니다.

## 4. 얼마나 깊게 맞춰야 하나

남은 깨짐이 스텝 0의 잔차에서 오는지, 얼마나 지워야 하는지를 학습 없이 쟀습니다. 스텝 0의 velocity를 앵커 목표
쪽으로 잔차의 일부만큼 옮기고(`M0.25`는 잔차의 25 %를 지움), 나머지 스텝은 student가 그립니다. 1024쌍 student
기준입니다.

| 스텝 0 | 남은 앵커 MSE | teacher 이미지와 cos | sameness (teacher 0.314) |
|---|---|---|---|
| student 자신의 것 | 0.078 | 0.760 | 0.296 |
| `M0.25` | 0.044 | 0.880 | 0.264 |
| `M0.5` | 0.019 | 0.940 | 0.255 |
| `M0.75` | 0.005 | 0.961 | 0.248 |
| 앵커 목표 자체 | 0 | 0.967 | 0.242 |
| teacher 자신의 스텝 0 | | 0.941 | 0.303 |

- 잔차의 4분의 1을 지우면 기타리스트의 거대한 손과 꼬인 몸이 사라지고, 절반이면 teacher 자신의 스텝 0과 같은
  수준이 됩니다. 그룹 아래쪽의 뭉개진 얼굴은 더 많이 지워야 사라집니다.
- 잔차를 8× 저주파 대역과 나머지로 나눠 한쪽만 지워도 각각 MSE의 절반쯤을 지우고 cos 대부분을 되찾습니다. 두
  대역 모두 깨짐을 실어 나릅니다.

목표는 held-out 앵커 MSE 0.02–0.04입니다. 6484쌍 런은 0.070이고, 데이터를 더 넣는 것만으로는 이 기울기로 닿지
않습니다.

## 5. teacher feature 위의 항

MSE는 velocity의 모든 원소를 같은 무게로 봅니다. 이미지를 깨는 것은 잔차 중 스텝 1–3이 읽는 부분이므로, 스텝 0의
상태를 teacher가 어떻게 보는지에 항을 걸었습니다. 같은 ε에서 student의 σ 0.75 상태 $x_s = \varepsilon - 0.25\,
v_{\text{student}}$와 teacher의 상태 $x_t = \varepsilon - 0.25\, v_{\text{target}}$(앵커 캐시에 있음)를 frozen
teacher에 σ 0.75와 프롬프트로 넣고 중간 블록 feature를 읽습니다. 두 상태가 ε로 짝지어져 있으므로 어떤 항도 한
시드를 다른 시드의 구도로 당기지 않습니다.

### 5.1 feature matching

$x_s$와 $x_t$의 feature 사이의 상대 L2를 앵커 MSE에 더합니다. tap은 student 잔차를 같은 norm의 랜덤 잔차와 비교해
골랐습니다. 마지막 블록(27)은 student 잔차를 랜덤 잔차보다 약 28배 크게 읽고, 중간 블록(14, 20)은 약 10배입니다.
가중치는 feature 항의 gradient가 MSE gradient의 약 6배가 되도록 15로 둡니다. probe 숫자는 FM 없는 런보다 조금 낮았지만(held-out 앵커
MSE 0.0772 대 0.0741) 렌더가 괜찮게 읽혀서 이 위에 GAN을 얹었습니다.

### 5.2 짝지은 GAN

discriminator는 같은 forward의 tap feature를 읽습니다. real은 $x_t$, fake는 $x_s$입니다. teacher 샘플을 real로
쓰는 배치는 LADD(Sauer et al. 2024)와 같고, disc가 자기 teacher forward를 따로 돌리지 않습니다. generator gradient는
매 스텝 FM gradient rms의 정해진 배수(`paired_ratio`)로 맞춥니다. 처음 200스텝은 generator 항을 끄고, 다음
200스텝에 걸쳐 올립니다.

- **token head(블록 27).** 비율 0.5에서 구도와 앵커 맞춤은 FM 런 그대로 두고 국소 구조를 바꿨습니다. 기타가
  네 시드 모두에서 온전하고, 그룹의 한 캐릭터가 두 번 그려지지 않습니다. 비율 1.0은 블라인드 비교에서 2 대 6으로
  졌습니다(물체의 원근이 모호하고 단조로워짐).
- **FastGen의 conv head.** NVlabs FastGen의 이미지 DiT head(4×4 stride 2 conv → GroupNorm → LeakyReLU → 1×1
  conv → 전역 평균)를 Cosmos Predict2.5-2B 레시피의 tap 13, 20, 27에 걸었습니다. 이 저장소의 16 GB 카드에는
  FastGen의 폭(hidden 1024, head 세 개에 1억 파라미터)이 들어가지 않아 hidden 256을 씁니다. disc LR 5e-5에서
  R1이 없으면 disc가 포화합니다.
  real과 fake가 같은 ε의 결정적 함수이고 student 잔차가 체계적이라 두 집합이 겹치지 않으므로, 자유로운 disc는
  margin을 끝없이 벌립니다.
- **R1.** real 상태 주변의 기울기를 근사 R1 항 $\lambda\,\mathbb{E}\big[(D(x_t) - D(x_t + \alpha\eta))^2\big]$로
  묶습니다(APT, Lin et al. 2025; FastGen의 구현). λ 100, α 0.1은 FastGen이 APT의 세기에 맞춘 값입니다. 무엇을
  구분하는지는 그대로 두고 disc의 기울기만 제한합니다.

| step 구간 | R1 없음: disc margin / loss | R1 100: disc margin / loss |
|---|---|---|
| 512–640 | 6.74 / 0.36 | 4.23 / 0.60 |
| 1024–1152 | 9.53 / 0.21 | 5.24 / 0.48 |
| 1920–2048 | 11.56 / 0.10 | 5.88 / 0.40 |

R1을 넣은 런은 1280스텝부터 margin이 5.4–5.9에서 멈췄습니다. probe 숫자는 token head 런과 비슷하고(held-out
앵커 MSE 0.0751 대 0.0748, `S` cos 0.770 대 0.779, sameness 0.277 대 0.279), 블라인드 비교에서 6 대 2로 이겼습니다.

## 6. 레시피

| 부분 | 설정 |
|---|---|
| teacher | soup을 bake한 merged15, int8 GPTQ base, fp32 stream, CFG 4, 28스텝 |
| student init | 공식 turbo v1.1 delta(rank 96 ASVD, adaLN 포함)를 rank 64로 자름, down은 Kaiming |
| LoRA | rank 64 / alpha 180, adaLN rank 16 / alpha 90 |
| 앵커 | 28스텝 중 14(σ 0.75까지의 chord), 6484쌍 캐시, 쌍마다 한 번 |
| feature matching | 블록 27, 가중 15 |
| GAN | conv head, tap 13 / 20 / 27, hidden 256, `paired_ratio` 1.0, disc LR 5e-5, R1 100 (α 0.1), 200스텝 지연 뒤 200스텝 ramp |
| 스텝 1–3 | loss 없음 |
| LR | student 5e-5, 2 % warmup 뒤 cosine으로 5e-6까지 |
| batch | 1, `grad_accum` 2 |
| clip | 1.0 |
| 배포 | EMA(window 0.2, LR 합 기준) |

**비용.** 캐시만 쓰면 iteration당 0.4초, 블록 27의 FM을 더하면 약 0.9초, GAN과 R1까지 더하면 약 1.1초입니다. 2048
iteration에 GAN 런은 약 40분이 걸리고, 피크는 13.4 GB로 16 GB 카드(5070 Ti)에서 돕니다.

## 7. 평가

1. **handoff probe**(arm당 약 3분). 평가 프롬프트 3개(위치 절로 자리를 정한 다섯 명의 그룹, 일본어 메뉴와 영어
   간판이 있는 카페, 아래에서 올려다본 점프하는 기타리스트) × 시드 1–4, 768 × 1344, CPU에서 뽑은 노이즈라 probe끼리
   시드 단위로 짝이 맞습니다. 숫자는 held-out 앵커 MSE, `S` cos, `T>S@1` cos(스텝 1–3이 유지되는지), sameness입니다.
2. **블라인드 시트.** student 열만 두 칸, 행마다 0 / 1을 섞고, 한 행을 한 파일로 probe 해상도 그대로 냅니다.
   한 장에 여러 행을 쌓으면 뷰어가 줄여서 보여 줍니다. 사람이 행마다 고른 0 / 1 문자열을 키와 맞춰 셉니다.
3. **렌더가 통과 여부를 정합니다.** probe 숫자는 렌더 옆에 두는 숫자입니다. 이 장의 여러 arm은 probe 숫자가
   비슷하거나 조금 낮았는데 렌더에서 이겼습니다.
4. arm마다 시드 twin이 없어서 노이즈 바닥을 모릅니다. LR만 다른 두 짧은 런의 차이가 `S` cos 0.014, sameness
   0.025였고, 이보다 작은 차이는 숫자만으로 가르지 않습니다.

## 8. 남은 것

- **2스텝 student.** 2스텝 static 격자는 σ 1 → 0.75 → 0입니다. 앵커가 σ 0.75까지의 chord이므로, 맞춘 스텝 0은
  2스텝 격자에서 teacher 자신의 σ 0.75 상태에 내립니다. 남는 것은 σ 0.75에서 0까지의 한 번의 점프이고, 아직 이
  점프를 학습한 적이 없습니다. 이전의 2스텝 시도는 cold start, 12스텝 앵커, merged14였습니다.
- **스텝 1–3.** teacher의 스텝 0이 주어지면 v1.1의 스텝 1–3은 0.94–0.97로 마칩니다. 남은 것은 student 자신의
  스텝 0이 넘기는 잔차를 견디는 것과, v1.1의 무른 마감(대비, 글자)입니다. teacher가 넘긴 입력에서 스텝 1–3에 DM을
  준 런은 구도를 지켰지만 마감이 양쪽으로 움직였습니다(수선과 새 깨짐이 비슷하게).
- **스텝 0의 깊이.** held-out 앵커 MSE 0.070에서 목표 0.02–0.04까지입니다.

## 참고

- DMD: Yin et al. 2023, "One-step Diffusion with Distribution Matching Distillation", arXiv:2311.18828
- DMD2: Yin et al. 2024, "Improved Distribution Matching Distillation for Fast Image Synthesis", arXiv:2405.14867
- DP-DMD: Wu et al. 2026, "Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis",
  arXiv:2602.03139
- LADD: Sauer et al. 2024, "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation",
  arXiv:2403.12015
- APT: Lin et al. 2025, "Diffusion Adversarial Post-Training for One-Step Video Generation", arXiv:2501.08316
- ASVD: Yuan et al. 2023, "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models",
  arXiv:2312.05821
- NVlabs FastGen, commit `898e757`: `fastgen/networks/discriminators.py`, `fastgen/configs/`
