# 4. DP-DMD 터보 증류: 4스텝 student LoRA 만들기

## 1. 루프

### 1.1 DMD와 다양성 앵커

teacher는 28스텝 × (cond + uncond) = 이미지당 DiT forward 56회입니다. CFG를 student에 구워 넣어 4스텝 × 1회로
줄이면 약 14배 빨라집니다.

DMD(Distribution Matching Distillation; Yin et al. 2023, arXiv:2311.18828)는 궤적 대신 분포를 맞춥니다. student가
만든 샘플을, student 분포를 따라가는 "fake" 점수 모델과 teacher 점수의 차이로 밀어 줍니다. 이 gradient는
reverse-KL이라 mode-seeking 성향이 있어서, 그대로 두면 시드가 달라도 포즈와 구도가 한 mode로 모입니다.
DP-DMD(Wu et al. 2026, arXiv:2602.03139)는 일을 나눕니다. 첫 스텝은 teacher의 다양한 궤적에 회귀시켜
mode(포즈, 레이아웃)를 정하게 하고, DMD gradient는 나머지 스텝에만 흐르게 합니다. 이 글의 루프는 DMD2(Yin et al.
2024, arXiv:2405.14867)의 구성, 즉 regression loss 없는 DM 신호, fake를 더 자주 업데이트하는 two time-scale
update, GAN 항에 DP-DMD 앵커를 얹은 것입니다.

### 1.2 세 역할, DiT 하나

LoRA 두 개(student, fake)를 같은 frozen DiT에 걸고, forward마다 어느 쪽을 켤지 바꿉니다.

| view | student LoRA | fake LoRA | 얻는 것 |
|---|---|---|---|
| teacher | off | off | frozen DiT의 velocity (CFG는 바깥에서 $v_u + \alpha(v_c - v_u)$) |
| student | on | off | $v_{\text{student}}$, 롤아웃 |
| fake | off | on | $v_{\text{fake}}$, student 분포의 점수 추적기 |

view 전환은 모듈마다 플래그만 뒤집으므로 DiT forward 한 번에 비하면 비용이 없습니다. optimizer는 LoRA마다 AdamW
하나씩 두 개이고, DiT는 학습하지 않습니다. 한 iteration 안의 순서와 forward 횟수는 부록 A4.1에 있습니다.

teacher의 uncond 입력은 빈 프롬프트("")를 인코딩한 임베딩입니다. 영벡터 cross-attention 입력은 분포 밖이어서, 그
$v_{\text{uncond}}$가 (α−1)=3배로 증폭되면 student가 하얗게 포화된 출력으로 밀려납니다.

### 1.3 DM 서로게이트

teacher는 flow matching 모델입니다. 경로는 $x_t = (1-t)\,x_0 + t\,\varepsilon$이고 velocity는 $v = \varepsilon -
x_0$입니다. student가 만든 깨끗한 추정 $x_\theta$에 새 노이즈 레벨 $\tau \sim U(0,1)$로 다시 노이즈를
입힙니다(§3과 부록에서는 이 τ를 σ_a로 적습니다).

$$
x_\tau = (1-\tau)\,x_\theta^{\text{detach}} + \tau\,\varepsilon_{dm}
$$

여기에 teacher(CFG를 적용한 real score)와 fake(cond만)의 velocity 차를 구합니다.

$$
\Delta_{dm} = \underbrace{v_u + \alpha\,(v_c - v_u)}_{\text{teacher, CFG }\alpha=4} - v_{\text{fake}}(x_\tau),
\qquad x_0^{\text{real}} - x_0^{\text{fake}} = -\tau\,\Delta_{dm}
$$

$x_\theta$를 $x_0^{\text{real}}$ 쪽으로 옮기려면 서로게이트의 $\partial/\partial x_\theta$가 $\tau\Delta_{dm}$이어야
합니다. 그래서 DMD2처럼 이 값을 gradient로 갖는 서로게이트 loss를 씁니다.

- **real score는 반드시 CFG를 적용합니다.** CFG가 들어가는 곳은 여기뿐입니다. 양쪽 다 cond만 쓰면 teacher와 fake
  velocity의 cos가 0.9999가 되어 신호가 노이즈뿐입니다.
- **부호를 맞춥니다.** 부호가 뒤집힌 루프의 증상은 "학습이 안 된 few-step blur"였습니다.
- 레시피는 신호의 크기를 샘플마다 정규화하고(floor 0.05), (τ, ε_dm)는 한 번만 뽑습니다(부록 A4.1).

### 1.4 DP-DMD 다양성 앵커

같은 시작 노이즈 $\varepsilon$에서 CFG teacher를 앵커용 teacher 격자 위로 K Euler 스텝만큼 굴려 $z_{t_k}$를
얻습니다. 스텝 0의 목표는 $[t_k, 1]$ 구간의 평균 velocity입니다.

$$
v_{\text{target}} = \frac{\varepsilon - z_{t_k}}{1 - t_k}, \qquad
\mathcal{L}_{div} = \big\| v_{\text{first}} - v_{\text{target}} \big\|^2,\quad v_{\text{first}} = G_\theta(\varepsilon, 1, c)
$$

레시피는 길이 12인 teacher 격자에서 K = 6 스텝을 굴려 앵커를 σ 0.5에 두고, 가중 λ는 0.1입니다. 다양성 항은 즉시
backward하고 첫 스텝 뒤에서 그래프를 끊어서, 뒤쪽 스텝의 DM gradient가 롤아웃을 거쳐 첫 스텝으로 흘러들지 않게
합니다. DMD의 reverse-KL gradient가 첫 스텝에 닿으면 선호도는 오르고 다양성은 줄어들기 때문입니다(DP-DMD 논문
Fig. 5).

다만 레시피는 첫 스텝 자신의 x0 예측에도 DM 신호를 줍니다(§1.5). 앵커 회귀만으로는 앵커 MSE를 어디까지 낮춰도
구도가 돌아오지 않았고, 렌더에서 구도를 움직인 변화는 첫 스텝의 DM 신호뿐이었습니다. 그 대가로 시드 간 다양성이
조금 줄어드는데, DP-DMD 논문이 보인 것과 같은 거래입니다. 이때도 앵커는 끄지 않습니다. 앵커 없이 첫 스텝에 DM만
주면 루프가 붕괴합니다(§3.7). 각 값의 의미는 부록 A4.1에 있습니다.

### 1.5 grad 스텝, 격자, CDM

N스텝을 전부 backprop하지 않습니다. 매 iteration에 $g \sim U\{0..N-1\}$을 뽑고, 0..g−1은 grad 없이 굴린 뒤, 스텝
g의 one-step x0 예측 $x_g - \sigma_g v_g$에만 grad를 줍니다. 그래서 메모리가 스텝 수와 무관하고 첫 스텝을 포함한
모든 σ가 감독을 받습니다. g = 0이면 첫 스텝의 다양성 항과 DM 신호가 같은 backward에 실립니다.

학습 롤아웃은 추론과 같은 static 격자(shift 3, σ 1 → 0.9 → 0.75 → 0.5 → 0)를 씁니다. CDM(Liu et al. 2026,
arXiv:2605.06376 §3.2)처럼 매 iteration마다 격자 길이와 내부 점을 새로 뽑는 동적 격자도 시험했는데, static 격자가
시드만 다른 동적 격자 두 런보다 시드 간 다양성과 teacher 유사도에서 모두 나았습니다(부록 A4.2). 대신 학습한 4스텝
밖의 스텝 수는 포기합니다.

CDM의 off-trajectory 항도 씁니다. grad 스텝의 $(x_g, v_g, \sigma_g)$를 detach한 뒤 Euler로 임의의 $t' \sim
U(0,1)$까지 외삽하고, 거기서 grad를 받는 student forward를 한 번 더 돌려 그 x0 추정 $x_{off} - t' v_{off}$에 같은
DM 신호를 줍니다. few-step Euler가 지나가지만 on-trajectory 롤아웃은 들르지 않는 영역을 감독하는 항입니다.
2스텝 100 iteration 비교에서 CDM을 켜면 하늘이 하늘로 남았고, 끄면 물결과 난간 같은 잡구조가 생겼습니다.

이 항의 real score에는 CFG를 걸지 않고 cond teacher만 씁니다(CDM 논문 Eq. 9). CFG는 on-trajectory DM 항으로만
들어갑니다. CFG를 건 teacher를 쓴 런과 비교하면 teacher 유사도, 앵커 MSE, 시드 간 다양성이 모두 조금씩 나았고,
렌더에서는 기타리스트를 teacher처럼 아래에서 올려다본 점프로 그렸습니다(부록 A4.2).

### 1.6 fake(critic)

fake는 student 출력 분포를 평범한 flow matching으로 학습합니다.

$$
\mathcal{L}_{fake} = \big\| \text{fake}\big((1-\tau)x_\theta^{\text{detach}} + \tau\varepsilon,\ \tau,\ c\big) - (\varepsilon - x_\theta^{\text{detach}}) \big\|^2
$$

- **업데이트 비율.** student 스텝당 fake 스텝은 4입니다. DMD2의 two time-scale update rule(그 논문은 5:1)처럼,
  움직이는 student를 fake가 앞서가게 하는 장치입니다.
- **fake warmup.** 본 루프 전에 fake만 200회 업데이트해서, fake가 이미 calibrate된 상태로 본 루프에 들어가게
  합니다. 그러지 않으면 초반 약 50스텝 동안 DM 신호가 튑니다.
- **rank.** fake rank ≥ student rank입니다. fake는 점수 추적기이므로 student가 만드는 섭동을 따라갈 용량이
  필요합니다.
- **transition-aware fake.** VA-TDM(Luo et al. 2026, arXiv:2611.06796)의 첫 아이디어입니다. fake 샘플이 어느
  student 스텝 σ_g에서 왔는지를 zero-init 임베딩으로 fake의 t-embedding에 더합니다. 저장할 때는 버립니다. A/B에서
  해가 없어서 레시피에 둡니다(부록 A4.1).

### 1.7 GAN: teacher feature 위의 token head

DMD2는 DMD에 GAN 항을 더했습니다. 이 루프의 discriminator는 LADD(Sauer et al. 2024, arXiv:2403.12015)처럼 frozen
teacher DiT의 중간 블록 activation을 읽는 약 2M 파라미터 head입니다. 구현은 NVlabs FastGen에서 가져왔습니다.
토큰마다 logit을 내는 token head라서 글자, 손, 질감 같은 국소 결함을 벌합니다. real 쪽은 데이터셋 latent이고, 루프
안에서 실제 데이터가 들어가는 곳은 여기뿐입니다. GAN 기울기는 DM 기울기와 원소 단위로 직교합니다. 즉 텍스처는
사 오고, 다양성은 사 오지 않습니다(부록 A4.4).

### 1.8 최적화

- **grad clip 1.0.** student, fake, disc 각각에 겁니다. 거의 모든 스텝에서 발동하므로 사실상 정규화기 역할을
  합니다(§3.6).
- **LR 스케줄.** 총 스텝의 2 %는 선형 warmup이고, 그 뒤는 cosine으로 5e-5에서 5e-6까지 내립니다. 첫 스텝이
  앵커를 얼마나 잘 맞추는지는 런 길이보다 학습 끝의 LR이 정했습니다. 상수 LR에서는 250에서 1000 iteration까지
  앵커 MSE가 제자리였고, 끝을 annealing하면 내려갔습니다(부록 A4.2).
- **EMA.** EMA window는 런 길이에 대한 비율이고, 기본 decay는 $1 - 1/(\text{window}\cdot\text{updates})$입니다.
  cosine에서는 마지막 20 %의 update가 가중치 이동의 약 5 %만 담으므로, update 수로 센 EMA는 raw 최종과 거의
  같아집니다. 그래서 update마다 decay를 $\text{decay}^{\,\text{lr}/\overline{\text{lr}}}$로 적용해 span을 LR 합에
  대한 비율로 셉니다. window 0.2면 상수 LR의 EMA 0.2와 같은 몫의 이동을 평균하고, update 수로는 마지막 약 45 %를
  덮습니다. EMA는 LR warmup이 끝나는 시점부터 돕니다.

### 1.9 출력과 추론

student만 plain LoRA 레이아웃으로 저장합니다. fake, disc, transition 임베딩은 버립니다. 추론은 4스텝, cfg
1.0입니다. CFG가 이미 구워져 있으므로 CFG를 다시 걸면 이중 guidance가 됩니다. 이 LoRA는 다른 개념 LoRA와 선형으로
합쳐집니다(LCM-LoRA(Luo et al. 2023, arXiv:2311.05556)를 스타일 LoRA와 합치는 것과 같은 방식). 추론 시 step 크기나
t 입력이 필요한 것(Shortcut, MeanFlow Δt, T-LoRA 마스크)은 실을 수 없습니다.

---

## 2. 레시피

### 2.1 teacher: soup을 구운 DiT

작가 91명을 그룹 8개로 나눠 그룹마다 LoRA를 하나씩 만들고, 이 LoRA 8개를 exact concat한 soup(rank 256, adaLN은
평균)을 base DiT에 1.0으로 bake한 것이 teacher입니다(soup과 exact concat은 02편, bake와 teacher 전용 int8 양자화는
03편). student는 bake된 DiT 위에서 학습하고, 배포할 때는 soup LoRA와 student LoRA를 normalize 없이, adaLN은 더해서
이어 붙입니다. 그러면 plain base에 올라가는 LoRA 하나가 정확히 $\Delta W_{soup} + \Delta T$가 됩니다. 기본
merge(전역 normalize, adaLN 평균)를 쓰면 정확성이 깨집니다. student LoRA만 base에 올리면 의미가 없고, 합친 LoRA는
multiplier 1.0으로만 씁니다. student는 warm start 없이 weight-SVD init으로 시작하므로 ΔW가 0에서 출발하고, 그래서
ΔW를 런 사이에서 바로 비교할 수 있습니다.

base DiT를 그대로 teacher로 쓰고 공식 turbo 릴리스에서 warm start한 설정도 돌렸습니다(부록 A4.6).

### 2.2 설정

| 부분 | 설정 |
|---|---|
| teacher | soup을 bake한 DiT(int8), CFG 4, 28스텝 |
| 다양성 앵커 | DP-DMD: λ 0.1, K 6 / teacher 격자 12, 첫 스텝 뒤 detach |
| DM 신호 | 점 하나의 DMD 차, τ ~ U(0,1), x0 정규화(floor 0.05), draw 1, 첫 스텝 포함(g ~ U{0..3}) |
| 격자 | 4스텝 static 격자(shift 3) |
| CDM | 가중 1.0, real score는 cond teacher(CFG 없음) |
| fake | transition-aware, warmup 200, 4:1 |
| GAN | token head, generator 가중 0.03, disc LR 1e-5 |
| LR | student, fake 모두 5e-5, warmup 2 % 뒤 cosine으로 5e-6까지 |
| 길이 | 1000 iteration |
| clip | 1.0 (student, fake, disc 각각) |
| walk 제거 | EMA window 0.2(LR 합 기준), **배포하는 것은 EMA** |
| batch | 1, gradient 누적 없음 |
| LoRA | student, fake 모두 rank 64 / alpha 180, adaLN rank 16 / alpha 90, weight-SVD init |

**비용.** iteration당 약 5.75초라 1000 iteration에 약 100분이 걸립니다(fake warmup 200회 별도). 피크는 약
13.7 GB allocated이고, block swap과 gradient checkpointing 없이 16 GB 카드(5070 Ti)에서 돌렸습니다.

---

## 3. 배운 것

측정 설정과 표는 부록 A4.3에 있습니다. 이 절의 측정은 §2.2 레시피를 정하기 전의 기준 설정(상수 LR 250
iteration, 동적 격자, 첫 스텝 DM off, CFG를 건 CDM)에서 했습니다. 2스텝 측정은 그래프를 이어 스텝 0도 최종
이미지의 gradient를 받게 한 설정이고, 4스텝 측정은 그 기준 설정으로 seed 42와 43을 한 번씩 돌린 control과
twin입니다.

### 3.1 batch 1의 student는 random walk한다

상수 LR에서 student의 |ΔW|는 √t로 자라고, 연속한 125스텝 이동끼리는 직교합니다(cos 0.03 이하). 되돌리는 힘도 한쪽으로
미는 힘도 없습니다. LR을 2.5배 낮추면 이동이 2.56배 줄어들 뿐 다른 것은 바뀌지 않습니다. Adam은 drift와 노이즈를
같은 비율로 줄이므로 노이즈가 학습을 앞지르는 시점이 그대로입니다. 2스텝과 4스텝 모두 같았고, GAN을 꺼도
마찬가지였습니다.

시드만 바꾼 두 4스텝 런은 ΔW의 9–10 %만 공유합니다. 즉 ΔW의 약 90 %는 그 런에만 있는 성분이고, 목적함수가 다른
arm도 reseed와 똑같이 읽히므로 **ΔW overlap으로는 arm을 가를 수 없습니다.** 런 한 쌍을 렌더로 비교한 A/B는 이런
추첨 두 번을 비교한 것입니다.

### 3.2 스텝당 gradient는 대부분 노이즈다

2스텝 런의 체크포인트 하나에서 student 스텝 방향의 에너지를 나눠 보면, 한 샘플 안의 DM draw 사이가 77 %, 샘플
사이가 20 %이고, 여러 스텝에 걸쳐 누적되는 전체 평균은 3 %입니다. DM gradient norm은 σ_a가 클수록 커지는데,
clip 아래에서는 이런 스텝도 다른 스텝과 같은 무게로 들어갑니다.

### 3.3 노이즈를 줄여도 수준은 오르지 않는다

2스텝에서 노이즈를 줄이는 세 방법(DM draw 4번, 체크포인트 평균, gradient 누적 4)을 시험했는데, 모두 기준 런의
250스텝 수준에서 멈췄습니다. 4 draw가 4배 효과를 내지 못하는 이유는 σ_a tail로, 큰 draw는 대부분 노이즈입니다.
즉 수준은 목적함수가 정하고, batch-1 walk는 거기서 빼기만 합니다.

### 3.4 EMA는 walk 제거 장치다

4스텝 control에서 EMA 0.2는 raw 방향의 96 %를 유지하면서 ΔW 길이를 15 % 줄입니다. 2스텝에서 window 0.2는
체크포인트 평균과 같은 수준까지 갔고, window 0.1은 walk를 걷어 내지 못했습니다. 2스텝에서 cosine LR은 walk를 멈춘
유일한 설정이었는데, LR은 walk가 시작되는 시점을 옮기지 않고 끝에서 walk를 줄일 뿐이라 EMA는 여전히 필요합니다.
그래서 레시피는 cosine LR 위에 EMA를 두고(§1.8), 배포하는 것은 EMA 체크포인트입니다.

### 3.5 발산 burst: raw 체크포인트가 그 안에 걸릴 수 있다

4스텝 control의 raw 250은 평가 렌더 모두에서 평평한 노이즈 텍스처를 냈습니다. 런이 마지막 몇 스텝에서
발산했는데(student grad norm 5–16 → 231), student와 fake가 함께 터졌고 σ_a와는 관계가 없었습니다. twin도 중간에
같은 종류의 burst를 겪고 회복했습니다. EMA 0.2는 burst를 희석해서, control의 EMA 250은 정상으로 렌더됩니다. 상수
LR에서는 250 iteration에 한 번꼴로 burst가 났고, raw 체크포인트 하나가 그 안에 걸릴 수 있으므로 raw를 배포하거나
raw끼리 비교하지 않습니다.

### 3.6 grad clip 1.0은 정규화기다

clip 이전 student norm은 2스텝에서 5스텝 구간 평균이 모두 1.0을 넘었고, 4스텝에서는 1.0 아래인 iteration이
하나도 없었습니다(중앙값 2.9–5.0). Adam은 gradient의 상수배를 무시하므로 clip을 올려도 스텝이 커지지 않고, 스텝
크기는 LR이 정합니다. clip이 정하는 것은 스텝끼리의 상대 무게입니다. 1.0이면 모든 스텝이 단위 norm으로 들어가고,
clip이 없으면 norm만큼의 무게로 들어가는데, 2스텝에서는 큰 norm이 곧 노이즈였습니다.

### 3.7 앵커 없는 루프는 붕괴한다

스텝 0에도 DM을 주고 다양성 가중을 0으로 두면(MFD 논문의 앵커 없는 루프) 250스텝 안에 student가 붕괴합니다.
스텝 0의 출력이 시드와 프롬프트에 상관없이 한 색이 되고, 4스텝 렌더는 인물 없는 노이즈 텍스처가 됩니다. 이 붕괴는
학습 스칼라로는 보이지 않고 validation probe에서만 보입니다(§5.2). 즉 앵커는 다양성 장치이면서, 이 루프에서는
student를 붙잡아 두는 힘이기도 합니다. 레시피가 첫 스텝에 DM 신호를 주면서도 앵커를 유지하는 이유입니다.

### 3.8 MFD를 DM 슬롯에 넣기

MFD(Zhao et al. 2026, arXiv:2606.11155)는 점 하나의 차 $\tau(v_{real} - v_{fake})$ 대신, σ_a에서 σ_b까지 K번의
Euler로 teacher와 fake를 각각 적분한 끝점의 차를 씁니다. 2스텝 100 iteration 비교에서 DM 신호의 스텝 간 sd는 K 1 →
3.4에서 약 35 % 줄었지만, init에 없던 잡내용이 K가 클수록 또렷한 사물이 되었습니다. 레시피는 MFD를 쓰지 않습니다.
MFD 논문의 T2I 결과는 4스텝 SANA 3000스텝이라 이 비교보다 훨씬 깁니다.

---

## 4. 효과가 없었던 시도

자세한 결과는 부록 A4.4에 있습니다.

- **2스텝 rollout grad.** 250 체크포인트가 가장 좋았는데, 이것은 walk의 모습이었습니다.
- **APEX**(Liu et al. 2026, arXiv:2604.12322). LoRA 하나가 생성기와 fake를 겸하게 해서 step은 3–5배 짧았지만, 모든
  체크포인트가 자기 warm start보다 흐리고 어두웠습니다.
- **DMAD**(Yu et al. 2026, arXiv:2610.02188). fake critic을 classifier로 바꿨는데, 신호가 2–18배 작고 step은 약
  1.8배 느렸으며 disc가 포화된 구간에서 렌더가 붕괴했다가 1k 스텝에서 회복했습니다.
- **주파수 대역 결손에 맞춘 CFG 재가중.** 고칠 blur 결손이 없었습니다.
- **GAN ↔ DM gradient surgery**(OPD²; Heo et al. 2026, arXiv:2607.15161). GAN과 DM gradient는 서로 직교해서 막을
  충돌이 없었습니다.
- **τ 대역별 critic.** 실제 학습 LR에서는 효과가 기준을 넘지 못했습니다.

---

## 5. 터보 런을 평가하는 법

### 5.1 A/B 프로토콜

1. **250 iteration**, 125스텝마다 저장, EMA on.
2. **250에서 EMA끼리 비교합니다.** raw 125와 250의 exact 평균은 교차 확인용입니다.
3. **null twin은 한 번만 만듭니다.** control에서 seed만 42 → 43으로 바꾼 런입니다. control과 twin의 차이가 이후
   모든 arm의 노이즈 바닥이 됩니다.
4. **평가는 렌더로 합니다.** 4스텝, cfg 1, 프롬프트 3개 × 시드 1–4, 768 × 1344(W × H)로 렌더하고, teacher(28스텝,
   cfg 4)를 옆에 둡니다. 프롬프트는 어려운 것으로 고릅니다. 위치 절로 자리를 정한 다섯 명의 슬랩스틱, 일본어
   메뉴와 영어 간판이 있는 두 소녀의 카페, 아래에서 더치 앵글로 본 관객 위로 점프하는 기타리스트입니다. 단어
   하나짜리 간판이나 캐릭터 하나를 그리는 프롬프트는 너무 쉬워서 arm을 가르지 못했습니다.
5. **렌더 옆에 분포 숫자를 둡니다.** IDM(Liu et al. 2026, arXiv:2610.39633)의 모멘트 갭으로, student(S)와 teacher
   CFG 4 렌더(T4) 사이의 nW2²와 T4 대비 pooled spread 비율입니다(예시는 부록 A4.5).

**통과 기준(미리 고정).** arm의 EMA 렌더가 두 프롬프트 모두에서 control보다 낫되 그 차이가 control–twin 차이보다
커야 하고, S–T4 nW2²가 control보다 |control − twin| 이상 낮아야 합니다. 통과한 arm은 새 control이 됩니다. step
시간이나 loss 항이 바뀌지 않으면 twin을 새로 만들지 않습니다.

**렌더를 읽는 순서.** 시드 간 구도 다양성을 먼저 보고, 그다음 선명도, 잡내용, 글자 순서로 봅니다. 448 px
다운스케일 그리드로 읽은 것과 글자 단위로 원본 크기에서 읽은 것은 구분해서 적습니다.

### 5.2 validation probe

validation 때마다 held-out 조건 하나로 8개 시드의 student를 static 격자로 굴리고, 시드 사이의 유사도를 잽니다.
블록 출력과 최종 latent에서 토큰 평균(DC)을 뺀 AC 잔차의 시드 간 cos를 보는데, 붕괴하면 최종 latent의 AC 유사도가
1로, 블록 AC 유사도가 0으로 **함께** 갑니다. flow-matching MSE는 발산 여부를 확인하는 숫자이고, 이 루프에서는 품질을
따라가지 않습니다. 지표별 정의와 건강한 범위는 부록 A4.5에 있습니다.

### 5.3 믿으면 안 되는 학습 스칼라

FM MSE는 품질과 반대로 움직이고, teacher–fake cos, DM 신호의 크기, grad norm은 앵커 없는 붕괴에서도 정상 범위에
있었습니다. 모든 velocity를 ε가 지배하기 때문입니다. x0 예측 std 같은 크기 지표는 폭주만 잡고 흐림은 잡지
못합니다. 그래서 런의 순위는 렌더로 매깁니다. 스칼라별 설명과 4스텝에서 잰 범위는 부록 A4.5에 있습니다.

### 5.4 학습한 스텝 수보다 많은 스텝에서 더 좋아 보이는 체크포인트

체크포인트가 학습한 스텝 수보다 더 많은 스텝에서 더 좋아 보인다면, 아직 진짜 N스텝 맵에 도달하지 못했다는
신호입니다. 2스텝은 σ 0.75 → 0 한 번의 점프로 σ≈0.5 아래의 디테일 형성 대역을 통째로 건너뜁니다.

---

## 참고 논문

- DMD: Yin et al. 2023, "One-step Diffusion with Distribution Matching Distillation", arXiv:2311.18828
- DMD2: Yin et al. 2024, "Improved Distribution Matching Distillation for Fast Image Synthesis", arXiv:2405.14867
- LADD: Sauer et al. 2024, "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation",
  arXiv:2403.12015
- LCM-LoRA: Luo et al. 2023, "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module", arXiv:2311.05556
- ASVD: Yuan et al. 2023, "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models",
  arXiv:2312.05821
- DP-DMD: Wu et al. 2026, "Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis",
  arXiv:2602.03139
- CDM: Liu et al. 2026, "Continuous-Time Distribution Matching for Few-Step Diffusion Distillation", arXiv:2605.06376
- MFD: Zhao et al. 2026, "Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models",
  arXiv:2606.11155
- VA-TDM: Luo et al. 2026, "VA-TDM: A Unified View for Improving Few-Step Diffusion Models by Velocity Adversarial
  Distillation", arXiv:2611.06796
- APEX: Liu et al. 2026, "Self-Adversarial One Step Generation via Condition Shifting", arXiv:2604.12322
- DMAD: Yu et al. 2026, "DMAD: Distribution Matching as Adversarial Distillation for Fast Visual Generation",
  arXiv:2610.02188
- IDM: Liu et al. 2026, "Intrinsic Distribution Matching", arXiv:2610.39633
- OPD²: Heo et al. 2026, "On-Policy Delta Distillation", arXiv:2607.15161
