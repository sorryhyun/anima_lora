# 4. DP-DMD 터보 증류: 4스텝 student LoRA 만들기

> **요약.** CFG를 쓰는 teacher(28스텝, 스텝마다 cond와 uncond로 forward 2회)를 CFG 1로 4스텝에 그리는 student
> LoRA로 증류한다. 손실은 DMD 계열이다. student 샘플을 teacher 점수와 fake 점수(student 분포를 따라가는 critic)의
> 차이로 밀고, 첫 스텝은 DP-DMD 앵커로 teacher 궤적에 회귀시켜 시드 간 구도 다양성을 지킨다. 여기에
> off-trajectory 항(CDM)과 teacher feature 위의 GAN 항을 더한다. teacher, student, fake는 frozen DiT 하나를 나눠
> 쓰고, 결과물은 추론 쪽 코드가 필요 없는 plain LoRA다. 가장 쓸모 있는 발견은 batch 1의 student가 random walk한다는
> 것이다. 125스텝 이동끼리 직교하고 |ΔW|가 √t로 자라며, 시드만 바꾼 두 런은 ΔW의 9–10 %만 공유한다. 노이즈를
> 줄이는 방법 세 가지(DM draw 여러 번, gradient 누적, 체크포인트 평균)는 모두 같은 수준에 머물렀고, walk는 EMA로
> 걷어 낸다. 레시피는 DP-DMD 앵커(λ 0.1), transition-aware fake 4:1, CDM 1.0, token GAN 0.03, constant LR 5e-5,
> EMA window 0.2이고, 배포하는 것은 EMA 체크포인트다. 런은 렌더로 평가한다. EMA끼리 비교하고, 시드만 바꾼 twin으로
> 노이즈 바닥을 재고, 어려운 프롬프트의 렌더를 teacher 옆에 놓는다. 학습 스칼라는 붕괴를 놓치거나 품질과 반대로
> 움직인다.

---

## 1. 루프

### 1.1 DMD와 다양성 앵커

teacher는 28스텝 × (cond + uncond) = 이미지당 DiT forward 56회다. CFG를 student에 구워 넣어 4스텝 × 1회로 줄이면
약 14배 빨라진다.

DMD(Distribution Matching Distillation; Yin et al. 2023, arXiv:2311.18828)는 궤적 대신 분포를 맞춘다. student가
만든 샘플을, student 분포를 따라가는 "fake" 점수 모델과 teacher 점수의 차이로 밀어 준다. 이 gradient는
reverse-KL이라 mode-seeking 성향이 있고, 그대로 두면 시드가 달라도 포즈와 구도가 한 mode로 모인다.
DP-DMD(Wu et al. 2026, arXiv:2602.03139)는 일을 나눈다. 첫 스텝은 teacher의 다양한 궤적에 회귀시켜 mode(포즈,
레이아웃)를 정하게 하고, DMD gradient는 나머지 스텝에만 흐르게 한다. 이 글의 루프는 DMD2(Yin et al. 2024,
arXiv:2405.14867)의 구성, 즉 regression loss 없는 DM 신호, fake를 더 자주 업데이트하는 two time-scale update,
GAN 항에 DP-DMD 앵커를 얹은 것이다.

### 1.2 세 역할, DiT 하나

LoRA 두 개(student, fake)를 같은 frozen DiT에 걸고, forward마다 어느 쪽을 켤지 바꾼다.

| view | student LoRA | fake LoRA | 얻는 것 |
|---|---|---|---|
| teacher | off | off | frozen DiT의 velocity (CFG는 바깥에서 $v_u + \alpha(v_c - v_u)$) |
| student | on | off | $v_{\text{student}}$, 롤아웃 |
| fake | off | on | $v_{\text{fake}}$, student 분포의 점수 추적기 |

view 전환은 모듈마다 플래그만 뒤집으므로 DiT forward 한 번에 비하면 비용이 없다. optimizer는 LoRA마다 AdamW
하나씩 두 개이고, DiT는 학습하지 않는다.

teacher의 uncond 입력은 빈 프롬프트("")를 인코딩한 임베딩이다. 영벡터 cross-attention 입력은 분포 밖이어서, 그
$v_{\text{uncond}}$가 (α−1)=3배로 증폭되면 student가 하얗게 포화된 출력으로 밀려난다.

### 1.3 한 iteration의 순서

| # | 단계 | forward | grad |
|---|---|---|---|
| 1 | teacher K-step CFG 앵커 → $v_{\text{target}}$ | teacher × 2K (K 6이면 12) | 없음 |
| 2 | student 롤아웃, 스텝 0 다양성 loss, grad 스텝 g의 x0 예측 | student × (g+1) | 스텝 0 (앵커), 스텝 g |
| 3 | DM 서로게이트 (DMD 또는 MFD) | teacher × 2, fake × 1 (MFD면 K배) | 없음 (신호만) |
| 4 | CDM 항, 따로 backward | student × 1 (grad), teacher × 2, fake × 1 | student |
| 5 | GAN generator 항 | teacher × 1 (블록 중간에서 early exit, grad 통과) | student |
| 6 | student loss 합산 → backward → clip → step → EMA | | |
| 7 | fake 업데이트 × (student 스텝당 fake 스텝 수), 매번 disc 1회 | fake × 1 (grad) + teacher × 2 (grad 없음, 절반 깊이) | fake, disc |

### 1.4 DM 서로게이트

teacher는 flow matching 모델이다. 경로는 $x_t = (1-t)\,x_0 + t\,\varepsilon$이고 velocity는 $v = \varepsilon - x_0$다.
student가 만든 깨끗한 추정 $x_\theta$에 새 노이즈 레벨 $\tau \sim U(0,1)$로 다시 노이즈를 입힌다(§3에서는 이 τ를
σ_a로 적는다).

$$
x_\tau = (1-\tau)\,x_\theta^{\text{detach}} + \tau\,\varepsilon_{dm}
$$

여기에 teacher(CFG를 적용한 real score)와 fake(cond만)의 velocity 차를 구한다.

$$
\Delta_{dm} = \underbrace{v_u + \alpha\,(v_c - v_u)}_{\text{teacher, CFG }\alpha=4} - v_{\text{fake}}(x_\tau),
\qquad x_0^{\text{real}} - x_0^{\text{fake}} = -\tau\,\Delta_{dm}
$$

$x_\theta$를 $x_0^{\text{real}}$ 쪽으로 옮기려면 서로게이트의 $\partial/\partial x_\theta$가 $\tau\Delta_{dm}$이어야
한다. DMD2처럼 이 값을 gradient로 갖는 서로게이트 loss를 쓴다.

- **real score는 반드시 CFG를 적용한다.** CFG가 들어가는 곳은 여기뿐이다. 양쪽 다 cond만 쓰면 teacher와 fake
  velocity의 cos가 0.9999가 되어 신호가 노이즈뿐이다.
- **부호를 맞춘다.** 부호가 뒤집힌 루프의 증상은 "학습이 안 된 few-step blur"였다.
- **x0 정규화.** 켜면 샘플마다 $\tau\Delta_{dm} / \max\big(\operatorname{mean}|\tau v_{\text{real}}|,\ \text{floor}\big)$로
  크기를 정규화하고, 이때 τ가 대부분 상쇄된다. 끄면 τ-damping만 남는다. 레시피는 정규화를 켜고 floor를 0.05로 둔다.
- **DM draw 수.** 같은 x0 예측에 (τ, ε_dm)를 여러 번 뽑아 신호를 평균할 수 있다. loss가 신호에 선형이라 backward는
  한 번이다. 2스텝에서 효과가 없어서 레시피는 1이다(§3.3).

### 1.5 DP-DMD 다양성 앵커

같은 시작 노이즈 $\varepsilon$에서 CFG teacher를 앵커용 teacher 격자 위로 K Euler 스텝만큼 굴려 $z_{t_k}$를 얻는다.
스텝 0의 목표는 $[t_k, 1]$ 구간의 평균 velocity다.

$$
v_{\text{target}} = \frac{\varepsilon - z_{t_k}}{1 - t_k}, \qquad
\mathcal{L}_{div} = \big\| v_{\text{first}} - v_{\text{target}} \big\|^2,\quad v_{\text{first}} = G_\theta(\varepsilon, 1, c)
$$

| 항목 | 레시피 값 | 의미 |
|---|---|---|
| 앵커 스텝 K / teacher 앵커 격자 길이 | 6 / 12 | 둘의 비가 앵커가 떨어지는 σ 지점을 정한다(6/12 = 0.5). 스텝 수는 teacher를 얼마나 촘촘히 적분하는지와 비용만 정한다 |
| 다양성 가중 λ | 0.1 | 스텝 0 MSE의 가중치다. 0이면 앵커 자체를 건너뛴다(teacher forward 2·K회 절약) |
| 첫 스텝 뒤 detach | on | **필수**다. 다양성 항을 즉시 backward하고 스텝 0 그래프를 끊는다. DMD reverse-KL이 첫 스텝으로 흘러들면 다양성이 다시 준다(DP-DMD 논문 Fig. 5) |
| 스텝 0에도 DM | off | 켜면 grad 스텝을 $U\{0..N-1\}$에서 뽑아 스텝 0도 DM 신호를 받는다. λ 0과 함께 쓰면 MFD 논문의 앵커 없는 루프가 되고, 이 조합은 250스텝 안에 붕괴했다(§3.7) |

### 1.6 grad 스텝, 동적 격자, CDM

N스텝을 전부 backprop하지 않는다. 매 iteration에 $g \sim U\{1..N-1\}$을 뽑고, 1..g−1은 grad 없이 굴린 뒤, 스텝 g의
one-step x0 예측 $x_g - \sigma_g v_g$에만 grad를 준다. 메모리가 스텝 수와 무관하고 모든 refinement σ가 감독을 받는다.

동적 격자를 켜면 CDM(Liu et al. 2026, arXiv:2605.06376 §3.2) 방식으로 매 iteration마다 롤아웃 격자를 새로 뽑는다.
길이는 $N \sim U\{2..\text{student 스텝 수}\}$이고, 내부 점은 (0,1) 균등분포에서 뽑아 정렬하며, $t_1 = 1$은 고정해서
앵커가 그대로 맞물리게 한다. validation 렌더와 추론은 항상 shift 3의 static 격자를 쓴다. 4스텝 static 격자의 σ는
1 → 0.9 → 0.75 → 0.5 → 0이다.

CDM의 off-trajectory 항도 쓴다. grad 스텝의 $(x_g, v_g, \sigma_g)$를 detach한 뒤 Euler로 임의의 $t' \sim U(0,1)$까지
외삽하고, 거기서 grad를 받는 student forward를 한 번 더 돌려 그 x0 추정 $x_{off} - t' v_{off}$에 같은 DM 신호를 준다.
few-step Euler가 지나가지만 on-trajectory 롤아웃은 들르지 않는 영역을 감독하는 항이다. 2스텝 100 iteration
비교에서 CDM을 켜면 하늘이 하늘로 남았고, 끄면 물결과 난간 같은 잡구조가 생겼다. 레시피의 CDM 가중은 1.0이다.

### 1.7 fake(critic)

fake는 student 출력 분포를 평범한 flow matching으로 학습한다.

$$
\mathcal{L}_{fake} = \big\| \text{fake}\big((1-\tau)x_\theta^{\text{detach}} + \tau\varepsilon,\ \tau,\ c\big) - (\varepsilon - x_\theta^{\text{detach}}) \big\|^2
$$

- **업데이트 비율.** student 스텝당 fake 스텝 4다. DMD2의 two time-scale update rule(그 논문은 5:1)처럼, 움직이는
  student를 fake가 앞서가게 하는 장치다. 매번 (τ, ε)를 새로 뽑고, GAN이 켜져 있으면 disc도 이 안쪽 루프에서 매번
  한 번 업데이트된다.
- **fake warmup.** 본 루프 전에 fake만 200회 업데이트한다. student는 건드리지 않는다. student LR warmup이 끝날 때
  fake가 아직 teacher와 같은 상태여서 생기는 초반 DM 신호 RMS 스파이크(~50스텝)를 없앤다. fake는 이미
  calibrate되고 full LR인 상태로 본 루프에 들어간다.
- **rank.** fake rank ≥ student rank다. fake는 점수 추적기이므로 student가 만드는 섭동을 따라갈 용량이 필요하다.
- **transition-aware fake.** VA-TDM(Luo et al. 2026, arXiv:2611.06796)의 첫 아이디어다. fake 샘플이 어느 student
  스텝 σ_g에서 왔는지를 zero-init 임베딩으로 fake의 t-embedding에 더한다. fake view에서만 동작하고 저장할 때는
  버린다. 시드, init, 데이터 순서가 같은 런 한 쌍의 500스텝 A/B에서 350–500스텝의 쌍별 차는 fake loss
  −0.002 ± 0.003, DM 차 RMS +0.002 ± 0.003이었고, teacher와 fake velocity의 cos는 둘 다 0.978이었다. 렌더는
  transition-aware 쪽이 더 선명하고 teacher에 가까웠는데, 이것은 런 한 쌍의 비교다(§3.1). 해가 없어서 레시피에 둔다.

### 1.8 GAN: teacher feature 위의 token head

DMD2는 DMD에 GAN 항을 더했다. 이 루프의 discriminator는 LADD(Sauer et al. 2024, arXiv:2403.12015)처럼 frozen
teacher DiT의 중간 블록 activation을 읽는 약 2M 파라미터 head다. 구현은 NVlabs FastGen에서 가져왔다.

- generator 항은 $\text{softplus}(-D(\text{feat}(x_\tau)))$이고, disc 항은
  $\text{softplus}(D_{fake}) + \text{softplus}(-D_{real})$이다. real 쪽은 데이터셋 latent다. 루프 안에서 실제
  데이터가 들어가는 곳은 여기뿐이다.
- token head(LADD 방식, 토큰마다 logit)와 pooled head(탭마다 logit 하나)는 파라미터 수가 같다. token head는 글자,
  손, 질감 같은 국소 결함을 벌한다.
- 레시피는 generator 가중 0.03, disc LR 1e-5, disc AdamW betas (0, 0.99), 중간 블록이다.
- GAN 기울기는 DM 기울기와 원소 단위로 직교한다(§4). 텍스처를 사 오고, 다양성은 사 오지 않는다.
- 2스텝에서 token disc가 폭주한 적이 있다(margin 0.22 → 6.94, spread → 0.99). 4스텝에서는 같은 설정이
  평탄했다. 폭주의 서명과 완화법은 §5.3에 있다.

### 1.9 최적화

- **grad clip 1.0.** student, fake, disc 각각에 건다. 실측으로는 거의 모든 스텝에서 발동하므로 사실상 정규화기
  역할을 한다(§3.6).
- **LR 스케줄.** 총 스텝의 2 %는 선형 warmup이다. 그 뒤 cosine(0.1·lr까지)이나 constant로 간다. EMA 없이는
  constant가 cosine보다 렌더가 나빴고, constant는 EMA가 정리를 맡는 런에 쓴다.
- **EMA.** EMA window는 런 길이에 대한 비율이고, decay는 $1 - 1/(\text{window}\cdot\text{updates})$다. 250 update에
  window 0.2면 약 50 update를 평균하는 decay 0.98이 된다. EMA는 LR warmup이 끝나는 시점부터 돈다.
- **gradient 누적.** iteration마다 clip한 student gradient의 평균으로 한 번 step하고, fake와 disc는 매 iteration
  업데이트한다. 2스텝에서 효과가 없어서 레시피는 1이다(§3.3).

### 1.10 출력과 추론

student만 plain LoRA 레이아웃으로 저장한다. fake, disc, transition 임베딩은 버린다. 추론은 4스텝, cfg 1.0이다.
CFG가 이미 구워져 있으므로 CFG를 다시 걸면 이중 guidance가 된다. 이 LoRA는 다른 개념 LoRA와 선형으로 합쳐진다
(LCM-LoRA(Luo et al. 2023, arXiv:2311.05556)를 스타일 LoRA와 합치는 것과 같은 방식). 추론 시 step 크기나 t 입력이
필요한 것(Shortcut, MeanFlow Δt, T-LoRA 마스크)은 실을 수 없다.

---

## 2. 레시피

### 2.1 teacher: base 모델 또는 soup

**base teacher.** base DiT를 그대로 teacher로 쓰고, student와 fake는 기존 few-step 모델(공식 turbo 릴리스)에서 base에
대한 delta를 rank 96으로 뽑아(adaLN 포함) warm start했다. activation-aware SVD(ASVD; Yuan et al. 2023,
arXiv:2312.05821) 추출은 같은 rank의 plain SVD보다 재구성이 좋아서, rank 96 ASVD가 plain rank 128과 비슷하다.
공식 turbo는 작동하는 few-step 맵이지만 mode가 붕괴해 있고, DP-DMD 앵커가 mode를 다시 펼친다. warm start된 student는
0스텝부터 이미 few-step 맵이므로 증류는 fine-tune에 가깝다. 이 설정에서 본 것:

- warm start에서 LR 5e-5는 안정적이지만 경계에 가깝다. ~1k 부근에서 한 번 벗어났다 돌아오고, 쓸 만한 체크포인트는
  2k 이후다. cold start는 student LR이 1e-5를 넘으면 불안정했다.
- 10k 스텝 런(5070 Ti 한 장에서 30시간)은 2k 이후로도 계속 좋아졌고, 최종 |ΔW|는 12.4다. warm start delta 자체의
  norm은 3.17이다.

**soup teacher.** 작가 91명을 그룹 8개로 나눠 그룹마다 LoRA를 하나씩 만들고, 이 LoRA 8개를 exact concat한 soup(rank
256, adaLN은 평균)을 base DiT에 1.0으로 bake한 것이 teacher다(soup과 exact concat은 02편, bake와 teacher 전용 int8
양자화는 03편). student는 bake된 DiT 위에서 학습하고, 배포할 때는 soup LoRA와 student LoRA를 normalize 없이, adaLN은
더해서 이어 붙인다. 그러면 plain base에 올라가는 LoRA 하나가 정확히 $\Delta W_{soup} + \Delta T$가 된다. 기본
merge(전역 normalize, adaLN 평균)를 쓰면 정확성이 깨진다. student LoRA만 base에 올리면 의미가 없고, 합친 LoRA는
multiplier 1.0으로만 쓴다. 이 경우 student는 warm start 없이 weight-SVD init으로 시작하므로 ΔW가 0에서 출발하고,
ΔW를 런 사이에서 바로 비교할 수 있다. 아래 레시피와 §3의 4스텝 측정은 soup teacher 설정이다.

### 2.2 설정과 근거

| 부분 | 설정 | 근거 |
|---|---|---|
| teacher | soup을 bake한 DiT(int8), CFG 4, 28스텝 | §2.1 |
| 다양성 앵커 | DP-DMD: λ 0.1, K 6 / teacher 격자 12, 첫 스텝 뒤 detach, 스텝 0 DM off | DP-DMD(Wu et al. 2026). 앵커 없는 루프는 250스텝 안에 붕괴(§3.7) |
| DM 신호 | 점 하나의 DMD 차, τ ~ U(0,1), x0 정규화(floor 0.05), draw 1 | DMD2(Yin et al. 2024). draw 4는 2스텝에서 수준을 바꾸지 못함(§3.3). MFD는 §3.8 |
| 격자 | 4스텝, 동적 격자 + CDM 1.0 | CDM(Liu et al. 2026). 2스텝 100 iteration 비교에서 CDM이 하늘을 하늘로 유지 |
| fake | transition-aware, warmup 200, 4:1 | two time-scale update(DMD2), VA-TDM(Luo et al. 2026). transition-aware는 500스텝 A/B에서 해가 없음 |
| GAN | token head, generator 가중 0.03, disc LR 1e-5 | DMD2, LADD(Sauer et al. 2024). 폭주는 2스텝의 실패이고 4스텝에서는 평탄 |
| LR | student, fake 모두 5e-5, constant, warmup 2 % | LR은 walk가 시작되는 시점을 옮기지 않음(§3.1) |
| clip | 1.0 (student, fake, disc 각각) | 매 스텝 발동하는 정규화기(§3.6) |
| walk 제거 | EMA window 0.2, **배포하는 것은 EMA** | 2스텝에서 EMA 0.2가 체크포인트 평균과 같은 수준(§3.4) |
| batch | 1, gradient 누적 없음 | 누적 4는 2스텝에서 수준을 바꾸지 못함(§3.3) |
| LoRA | student, fake 모두 rank 64 / alpha 180, adaLN rank 16 / alpha 90, weight-SVD init | ΔW가 0에서 시작해 런 사이에서 바로 비교됨 |

**비용.** iteration당 5.7–5.8초이고, fake warmup 200회 뒤 250 iteration에 약 27분이 걸렸다. 피크는 13.70 / 13.73 GB
allocated(시드만 다른 두 런)다. 둘 다 block swap과 gradient checkpointing 없이 16 GB 카드(5070 Ti)에서 돌렸다.

---

## 3. 배운 것

**측정 설정.** 2스텝 측정은 static 격자(σ 1 → 0.75 → 0)에서 했다. 이 격자에서 스텝 0은 앵커만, 스텝 1은 모든 DM을
받아서 두 스텝이 gradient를 나누지 않으므로, 그래프를 이어 스텝 0도 최종 이미지의 gradient를 받게 했다(rollout
grad). 2스텝 기준 런은 LR 2e-5 constant에 GAN을 켠 런이다. 4스텝의 기준 측정은 §2.2 레시피 그대로 250 iteration,
seed 42로 돌린 control과, seed만 43으로 바꾼 null twin이다. 둘 다 125스텝마다 저장하고 validation 시드 8개로 probe를 돌렸다.

### 3.1 batch 1의 student는 random walk한다

**2스텝.** 저장된 체크포인트의 ΔW = up·down을 364개 모듈 전체에 대해 Frobenius norm으로 쟀다.

| 런 | LR, 스케줄 | \|ΔW\| @125 / 250 / 375 / 500 | 연속 125스텝 이동 사이 cos |
|---|---|---|---|
| 기준 런 (GAN) | 2e-5, constant | 2.77 / 3.98 / 4.85 / 5.50 | +0.018, −0.001 |
| GAN | 5e-5, constant | 6.54 / 9.74 / 12.37 / 14.77 | +0.012, +0.012 |
| GAN off, EMA 0.1 | 5e-5, constant | 6.66 / 9.27 / 11.23 / 12.77 | – |
| GAN off, CDM off, transition-aware fake | 5e-5, **cosine** | 5.71 / 7.74 / 8.21 / 8.26 | – |

- 상수 LR에서는 |ΔW|가 √t를 따른다(기준 런: 2.77·√(t/125) = 3.92 / 4.80 / 5.54). 125스텝 이동은 서로 직교한다.
  되돌리는 힘도 한쪽으로 미는 힘도 없다.
- LR을 2.5배 낮추면 이동이 2.56배 줄어들 뿐 다른 것은 바뀌지 않는다. Adam은 drift와 노이즈를 같은 비율로 줄이므로
  노이즈가 학습을 앞지르는 시점이 그대로다.
- walk는 GAN이 없어도 일어난다. 멈춘 것은 cosine 런 하나(8.21 → 8.26)이고, 마지막 150스텝에 앵커 loss, DM 신호
  RMS, teacher와 fake velocity의 cos가 좋아진 것도 그 런뿐이다.

**4스텝.**

| | control | twin |
|---|---|---|
| \|ΔW\| raw 125 / 250 | 7.34 / 10.33 | 7.49 / 10.54 |
| √2 · \|ΔW\|₁₂₅ | 10.37 | 10.59 |
| \|이동\| 0–125 / 125–250 | 7.34 / 7.14 | 7.49 / 7.21 |
| cos(이동 0–125, 이동 125–250) | 0.018 | 0.028 |
| \|ΔW\| EMA 250 | 8.73 | 8.98 |
| cos(EMA 250, raw 250) | 0.961 | 0.965 |

4스텝도 같은 상태다. 125스텝부터 √t로 자라고, 길이가 같고 서로 직교하는 두 번의 이동으로 이루어진다.

**런 사이 공유분.** 두 런 ΔW 사이의 cos는 공유하는 에너지 비율로 읽을 수 있다.

| 쌍 | cos |
|---|---|
| 4스텝 control vs twin (시드만 다름), raw 125 / raw 250 / EMA 250 | 0.087 / 0.093 / 0.100 |
| 4스텝 500스텝 런 한 쌍 (시드·init·데이터 순서 같음, transition-aware fake만 다름) | 0.14 |
| 2스텝 기준 런 vs DM draw 4 런 (DM draw 수만 다름) @125 / 250 / 375 / 500 | 0.19 / 0.17 / 0.16 / 0.17 |
| 그 밖의 쌍 (LR이 다르거나 2스텝 vs 4스텝) @500 | 0.09–0.11 |

- 4스텝 student의 ΔW 중 약 85–90 %는 그 런에만 있는 성분이다. 공유분은 특정 위치에 몰려 있지 않다. 모듈 종류별로
  0.11–0.19, 블록 그룹별로 0.10–0.21이고 마지막 블록 그룹(21–27)이 가장 높다.
- 시드만 바꿔도 0.09–0.10이 남는다. 0.14였던 쌍은 데이터 순서까지 같았으므로, 그 0.14에는 데이터 순서의 몫이 들어
  있을 가능성이 크다.
- **ΔW overlap으로는 arm을 가를 수 없다.** 목적함수가 다른 arm도 reseed와 똑같이 읽힌다.
- 시드 평균: 공유분이 c ≈ 0.14일 때 k개 시드의 exact ΔW 평균은 공유분을 $kc/(1+(k-1)c)$로 올린다. k = 2면 0.25,
  k = 4면 0.39다. 2스텝에서 두 런의 500 체크포인트를 평균한 것은 250 수준을 넘지 못했으므로, 시드 평균은 품질을
  올리지 못하고 정리 작업에 그친다.
- 런 한 쌍을 렌더로 비교한 A/B(예: §1.7의 transition-aware fake)는 이런 추첨 두 번을 비교한 것이다.

### 3.2 스텝당 gradient는 대부분 노이즈다

2스텝 기준 런의 375스텝 체크포인트 하나에서 16샘플 × DM draw 3으로 gradient를 분해했다. 16샘플은 모두 가장 큰
토큰 버킷(4154–4200 토큰)에서 나왔다.

| student 스텝 방향 에너지 (clip 1.0이 남기는 단위 벡터 기준) | 비율 |
|---|---|
| 한 샘플 안의 (σ_a, ε_dm) draw 사이 | 77 % |
| 샘플 사이 | 20 % |
| 전체 평균 (누적되는 몫) | 3 % |

- 항별 제곱 norm 비율은 rollout 경로를 거친 DM 56 %, 스텝 1의 DM 13 %, 앵커 7 %, GAN 7 %다.
- DM gradient norm은 σ_a에 따라 커진다. 중앙값이 0.25 미만에서 1.24, 이후 2.03, 2.36이고 0.75 초과에서 4.93이다.
  σ_a 0.96에서 21.9, 0.99에서 30.5가 나왔다. clip 아래에서는 이런 스텝도 다른 스텝과 같은 무게로 들어간다.
- 평균 방향을 보면 DM과 앵커는 일치한다(cos +0.55). GAN의 평균은 DM과 약간 반대다(−0.19).

### 3.3 노이즈를 줄여도 수준은 오르지 않는다

2스텝에서 노이즈를 줄이는 세 경로를 시험했고, 모두 기준 런 250스텝 수준에서 멈췄다.

| 경로 | 런 | 결과 |
|---|---|---|
| DM draw 4 | 9.5 s/step (기준 런 6.5) | 실패. 250은 기준 런과 비슷하고 375는 자기 250보다 나쁘며 500은 회복. "좋아졌다 나빠졌다 다시 좋아지는" 것은 walk의 모습이다. \|ΔW\| 2.80 / 3.92 / 4.70 / 5.35, 평균 몫 2.9 % (기준 런 3.0 %) |
| 체크포인트 exact 평균 | 기준 런의 250 / 375 / 500, 기준 런과 DM draw 4 런의 500, 두 런의 250 / 375 / 500(6개) | 500에서 생긴 벤치, 차량, 갈색 머리를 걷어 낸다. 그러나 어느 것도 기준 런 250을 넘지 못함 |
| gradient 누적 4 | 1000 it = 250 update, 72분, 4.0 s/it, 피크 13.79 GB | 실패. raw 최종은 둘 다보다 나쁘고(뒤틀린 몸), EMA 최종은 체크포인트 6개 평균 수준. \|ΔW\| 2.78 / 3.84, 이동 cos 0.022 |

- 4 draw가 4배 효과를 내지 못하는 이유는 σ_a tail이다. 큰 draw는 대부분 노이즈이고, norm 가중으로 본 DM 평균 몫은
  1.6 %, 단위 가중으로는 4.3 %다. 4 draw 중 유효 draw 수는 2.23(iid 가정)이다.
- **해석: 수준은 목적함수가 정하고, batch-1 walk는 거기서 빼기만 한다.**

### 3.4 EMA는 walk 제거 장치다

- 2스텝에서 window 0.1(약 50스텝)의 EMA는 walk를 걷어 내지 못했다(기준 런 500: EMA |ΔW| 5.12, raw 5.50). window
  0.2(gradient 누적 런)는 체크포인트 평균과 같은 수준까지 갔다.
- 4스텝 control에서 EMA 0.2는 raw 방향의 96 %를 유지하고 ΔW 길이를 15 % 줄인다. 125스텝 이동 하나 안의 walk는 지우지
  못한다.
- 그래서 레시피는 상수 LR에 EMA를 두고, 배포하는 것은 EMA 체크포인트다. cosine LR은 walk를 멈춘 유일한 설정이었지만,
  레시피는 "LR이 walk 시작 시점을 옮기지 않으며 스케줄 모양은 레버가 아니다"라는 판단으로 constant를 골랐다.

### 3.5 발산 burst: raw 체크포인트가 그 안에 걸릴 수 있다

4스텝 control의 raw 250은 평가 렌더 12장(프롬프트 3개 × 시드 4개) 모두에서 평평한 노이즈 텍스처를 냈다. 파일은
정상이었고(같은 키, 유한값), 런이 마지막 몇 스텝에서 발산했다.

- student grad norm이 242–245스텝에서 5.5 / 14 / 6 / 16이다가 246–249에서 **231 / 244 / 106 / 72**가 되었다. fake
  grad norm도 같이 올라(25 → 111 / 69 / 145 / 85), student와 fake가 함께 터졌다.
- 그 구간의 σ_a는 0.03–0.96이므로 σ_a tail 탓이 아니다.
- x0 예측의 std가 0.74에서 1.98로 올랐고, 250 validation에서 FM MSE가 0.29 → 0.88, 최종 latent AC 유사도(§5.2)가
  0.948 → 0.999로 바뀌었다. §3.7의 붕괴 서명과 같다.
- twin도 153–162스텝에서 같은 종류의 burst(37 / 29 / 58 / **242** / 131)를 겪었지만 회복했다.
- EMA 0.2는 burst를 희석해서, control의 EMA 250은 정상으로 렌더된다. raw 125 + 250 평균에는 burst가 일부 남는다
  (기타리스트 프롬프트의 블록 같은 군중).

**이 레시피에서는 250 iteration에 한 번꼴로 burst가 났고, raw 체크포인트 하나가 그 안에 걸릴 수 있다. raw를
배포하거나 raw끼리 비교하지 않는다.**

### 3.6 grad clip 1.0은 정규화기다

clip 이전 student norm(5스텝 구간 평균)은 2스텝 런 네 개 모두에서 **모든 구간이 1.0을 넘었다**(중앙값 3.20–5.00,
p95 7.4–10.1). fake도 마찬가지다(중앙값 2.2–2.9).

- Adam은 gradient의 상수배를 무시하므로 clip을 올려도 스텝이 커지지 않는다. 스텝 크기는 LR이 정한다. clip이 정하는
  것은 스텝끼리의 상대 무게다. 1.0이면 모든 스텝이 단위 norm으로 들어가고, clip이 없으면 norm만큼의 무게로 들어간다.
- 2스텝에서는 큰 norm이 곧 노이즈였다. norm 가중을 하면 DM 평균 몫이 4.3 %에서 1.6 %로 떨어진다.

**4스텝 σ_a 프로필** (clip 이전 student grad norm의 σ_a 사분위별 중앙값):

| | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | top/bottom | 중앙값 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| control | 2.67 | 2.50 | 2.34 | 4.41 | 1.65 | 2.86 | 18.6 | 244 |
| twin | 2.80 | 2.54 | 3.05 | 3.57 | 1.27 | 2.91 | 12.9 | 242 |
| 합산 | 2.76 | 2.52 | 2.62 | 3.94 | **1.43** | 2.88 | 15.0 | 244 |

- 1.0 아래인 iteration이 하나도 없으므로 4스텝에서도 clip은 매 스텝 작동한다.
- σ_a < 0.75에서는 평탄하고 맨 위 사분위에서만 오른다. 상승폭은 2스텝(4.9 / 1.2)보다 완만한데, rollout 경로가
  없다는 점과 맞는다.

### 3.7 앵커 없는 루프는 붕괴한다

스텝 0에도 DM을 주고 다양성 가중을 0으로 두면(MFD 논문의 앵커 없는 루프) 250스텝 안에 student가 붕괴한다. 스텝 0의
출력이 시드와 프롬프트에 상관없이 한 색이 되고, 4스텝 렌더는 인물 없는 노이즈 텍스처가 된다. 공통 설정은 soup
teacher(int8), weight-SVD r64 / α180, transition-aware fake, 4스텝 + 동적 격자, DM 슬롯에 MFD(τ 간격 0.1, 최대
4스텝), GAN과 CDM off, lr 5e-5, 4:1, batch 1이다.

| 런 | fake warmup | 최종 latent AC 유사도 @250 | 블록 AC 유사도 @250 | FM MSE @250 |
|---|---|---|---|---|
| 앵커 on (참조) | 200 | 0.940 | 0.678 | 0.357 |
| 앵커 off | 200 | 0.9994 | 0.077 | 0.371 |
| 앵커 off (250에서 중단) | 0 | 0.9996 | 0.155 | 0.452 |

- **학습 스칼라로는 보이지 않는다.** 붕괴한 런의 DM 차의 상대 크기, teacher와 fake velocity의 cos, DM 차 RMS, grad
  norm은 참조 런 범위 안에 있다. 모든 velocity를 ε가 지배하기 때문이다. 서명은 validation probe에 나타난다(§5.2).
- 원인에서 제외된 것은 g = 0 경로, transition-aware fake, fake LR warmup이다. LoRA 자체도 원인이 아니다(MFD 논문의
  SANA 런도 student와 fake 모두 LoRA r64).
- 함정: 앵커를 없애도 DM 스칼라는 건강해 보인다. 앵커는 다양성 장치이면서, 이 루프에서는 student를 붙잡아 두는
  힘이기도 하다. 2스텝 측정에서 앵커와 DM의 평균 방향은 일치했다(§3.2).

### 3.8 MFD를 DM 슬롯에 넣기

MFD(Zhao et al. 2026, arXiv:2606.11155)는 점 하나의 차 $\tau(v_{real} - v_{fake})$ 대신, σ_a에서 σ_b까지 K번의
Euler로 teacher와 fake를 각각 적분한 끝점의 차를 쓴다. σ_b = 0, K = 1이면 DMD와 같다. 2스텝, base teacher와 공식
turbo delta init(§2.1), GAN off, 100 iteration 결과:

| arm | CDM | τ 간격 | 평균 K | s/step | DM 차 RMS 평균 ± sd |
|---|---|---|---|---|---|
| DMD | 1.0 | – | 1 | 5.33 | 0.194 ± 0.057 |
| MFD | 1.0 | 0.1 | 3.4 | 6.32 | 0.167 ± 0.041 |
| MFD | 0 | 0.03 | 17.2 | 11.30 | 0.185 ± 0.035 |

- DM 신호의 스텝 간 sd는 K 1 → 3.4에서 약 35 % 줄고, K 17에서는 더 줄지 않았다.
- 학습한 모든 arm에 init에는 없던 잡내용(하늘의 검은 형체, 두 번째 캔)이 생겼고, **K가 클수록 그 형체가 또렷한
  사물이 되었다**(K 17에서 악마 꼬리). 분산 감소로도, teacher ODE를 더 정확히 적분한 편향으로도 설명된다.
- CDM을 켜면 하늘이 하늘로 남았다. 500스텝 τ 간격 0.03, CDM off에서는 떠다니는 캔과 탑이 사라졌지만 꼬리와 얼룩은
  남았다.
- 레시피는 MFD를 쓰지 않는다. 이 비교는 2스텝 100 iteration이고, MFD 논문의 T2I 결과는 4스텝 SANA 3000스텝이다.

---

## 4. 효과가 없었던 시도

**2스텝 rollout grad.** §3의 2스텝 측정이 이 설정이다. LR 5e-5와 2e-5 모두 250 체크포인트가 가장 좋았고, DM draw
여러 번, gradient 누적, 체크포인트 평균도 이 수준을 넘지 못했다. LR 5e-5 런에서는 ~350스텝부터 disc가 폭주했다
(margin 0.9–1.6, x0 예측 std → 0.43). **함정:** "250이 제일 좋다"는 walk의 모습이었다. raw 체크포인트 하나를 다른
하나와 비교하면 walk를 읽게 된다.

**APEX: student, reference, fake를 LoRA 하나로** (Liu et al. 2026, arXiv:2604.12322). 조건 이동 $c_{fake} = -c$로 LoRA
하나가 생성기와 fake를 겸하게 했다. step 시간은 3–5배 짧았다(1.3–2.1 s/it 대 6.7–7.0). 처음 제안한 변형($F - d$)은
55스텝부터 x0 예측 std가 0.77 → 7.6으로 폭주했고, 논문 형태(1500스텝)는 안정적이었지만 모든 체크포인트가 자기 warm
start보다 흐리고 어두웠다. 감독 loss $\mathcal{L}_{sup}$이 student를 velocity field로 다시 끌어당기고, velocity
field를 4스텝 Euler로 풀면 조건부 평균이 나오기 때문으로 본다. **함정:** 논문 형태 런의 학습 스칼라(x0 예측 std
0.62, 롤아웃 std 포함)는 그리드가 망가지는 동안 내내 건강했다. 롤아웃 std는 폭주만 잡고 흐림은 못 잡는다. 첫
체크포인트에서 그리드를 렌더한다.

**DMAD: disc가 DM을 대신** (Yu et al. 2026, arXiv:2610.02188). fake critic을 classifier(teacher 대 student, real 대
student의 head 두 개)로 바꿨다. 같은 x0 예측에서 독립 DM 추정 두 개는 cos ≈ 0.12로만 일치한다. 신호는 DP-DMD보다
2–18배 작았고, step은 11.9 s 대 6.6 s로 논문은 4–5배 빠르다고 했지만 약 1.8배 느렸다. 결과는 "덜 움직였다"였다.
raw 신호에 CDM을 더한 런(λ 150, 1500스텝)에서는 ~300–980스텝 동안 disc가 포화되어 입력 gradient가 사라졌고, x0
예측 std가 0.6 → 5로 올라 500 / 750 렌더가 줄무늬 텍스처로 붕괴했다가 1k에서 스스로 회복했다. **함정:** raw 신호는
상한이 없어 과하게 튀고, 포화된 disc는 student를 붙잡는 힘이 없다. 둘 다 λ로는 막을 수 없다.

**주파수 대역 결손에 맞춘 CFG 재가중.** student가 teacher보다 뒤처진 주파수 대역에 맞춰 CFG uplift를 재가중하는
방법이다. 첫 probe는 student 롤아웃 상태(과도한 blur → 고주파 가중)를 쟀는데, loss는 실제 데이터를 다시 노이즈
입힌 점에서 재고 있었다(→ 저주파 가중). 부호가 반대였다. on-trajectory에서 σ를 맞춰 다시 재니 student가 teacher보다
저주파를 더 가진(과도하게 흐린) 샘플이 3 %였다. 고칠 blur 결손이 없었고, DP-DMD 앵커가 이미 흡수한 것으로 본다.
**함정:** go/no-go probe는 loss가 볼 바로 그 분포에서 같은 지표를 재야 한다.

**GAN ↔ DM gradient surgery** (OPD²; Heo et al. 2026, arXiv:2607.15161). GAN gradient 중 DM과 부호가 어긋나는 성분만
막는 게이트를 검토했다. agree-energy가 0.489 ± 0.032로 permutation null 0.488 ± 0.011과 구별되지 않았고, GAN
gradient와 DM 신호의 cos는 0.006이었다. GAN은 DM과 직교 방향으로 밀며, 원소당 크기는 DM의 약 6배(tail이 두꺼움)다.
**함정 두 개:** 원소 부호 일치율 0.5는 직교의 기준선이고 충돌을 뜻하지 않는다. 그리고 DM 서로게이트가 평균으로
줄여지므로 실제 적용되는 DM 힘은 DM 신호를 원소 수로 나눈 값이고, 원본 텐서끼리 비교하면 GAN을 ~10⁵배 과소평가한다.

**τ 대역별 critic.** fake를 τ 대역별로 나누면 대역 간 간섭이 줄어드는지 봤다. 학습 끝의 annealed LR(4.04e-6)에서는
신호가 17–19배로 통과 기준을 넘었지만, 실제 학습 peak LR(3e-5)에서는 기준을 넘지 못했다. 낮은 τ 대역만 학습한
clone이 높은 τ 대역도 개선했다. **함정:** probe fine-tune은 실제 학습 LR에서 돌려야 한다. 작은 LR은 init 근처의
선형 영역만 재고, 노이즈 바닥을 낮춰서 SNR을 부풀린다.

---

## 5. 터보 런을 평가하는 법

### 5.1 A/B 프로토콜

1. **250 iteration**, 125스텝마다 저장, EMA on.
2. **250에서 EMA끼리 비교한다.** raw 125와 250의 exact 평균은 교차 확인용이다.
3. **null twin은 한 번만 만든다.** control에서 seed만 42 → 43으로 바꾼 런이다. control과 twin의 차이가 이후 모든
   arm의 노이즈 바닥이 된다.
4. **평가는 렌더로 한다.** 4스텝, cfg 1, 프롬프트 3개 × 시드 1–4, 768 × 1344(W × H)로 렌더하고, teacher(28스텝,
   cfg 4)를 옆에 둔다. 프롬프트는 어려운 것으로 고른다. 위치 절로 자리를 정한 다섯 명의 슬랩스틱, 일본어 메뉴와
   영어 간판이 있는 두 소녀의 카페, 아래에서 더치 앵글로 본 관객 위로 점프하는 기타리스트다. 단어 하나짜리 간판이나
   캐릭터 하나를 그리는 프롬프트는 너무 쉬워서 arm을 가르지 못했다.
5. **렌더 옆에 분포 숫자를 둔다.** IDM(Liu et al. 2026, arXiv:2610.39633)의 모멘트 갭으로, student(S)와 teacher
   CFG 4 렌더(T4) 사이의 nW2²와 T4 대비 pooled spread 비율이다. T4 세트는 한 번 렌더해서 재사용한다.

**통과 기준(미리 고정).** arm의 EMA 렌더가 두 프롬프트 모두에서 control보다 낫되 그 차이가 control–twin 차이보다
커야 하고, S–T4 nW2²가 control보다 |control − twin| 이상 낮아야 한다. 통과한 arm은 새 control이 된다. step 시간이나
loss 항이 바뀌지 않으면 twin을 새로 만들지 않는다.

**렌더를 읽는 순서.** 시드 간 구도 다양성을 먼저 보고, 그다음 선명도, 잡내용, 글자 순서로 본다. 448 px 다운스케일
그리드로 읽은 것과 글자 단위로 원본 크기에서 읽은 것은 구분해서 적는다.

**모멘트 갭의 예.** mode가 붕괴한 공식 turbo delta(§2.1)를 base teacher와 비교했다. 128 item, teacher 마지막 블록의
pooled feature로 쟀다. teacher CFG 1(T1)과 T4 사이의 nW2²는 0.186 / 0.189(τ 0.8 / 0.5)로 null의 약 17배이고, S–T4는
0.032 / 0.060이다. S의 pooled spread는 T4 대비 0.885(τ 0.8), 0.858(τ 0.5), 0.798(latent 채널 평균)로 student의
분산이 부족하다. 데이터셋은 CFG 1보다 CFG 4에 더 가깝다.

### 5.2 validation probe와 그 서명

validation 때마다 held-out 조건 하나로 8개 시드의 student를 static 격자로 굴린다.

| 지표 | 정의 | 건강 | 붕괴 서명 |
|---|---|---|---|
| 블록 AC 유사도 | 블록 출력에서 토큰 평균(DC)을 뺀 AC 잔차의 시드 간 cos를 모든 스텝·블록에 대해 평균. 낮을수록 다양 | 참조 런 0.678 @250, 4스텝 control / twin 0.68 / 0.59 @125 | 0 근처 (0.077, 0.155): 최종 출력이 서로 무관한 노이즈 |
| 최종 latent AC 유사도 | N스텝 롤아웃이 끝난 최종 latent에 같은 DC/AC 분해를 적용한 시드 간 cos | 0.940–0.95 | 1 근처 (0.9994, 0.999) |
| FM MSE | held-out 샘플의 student 격자 σ에서 flow-matching MSE | 0.27–0.36 | 앵커 없는 붕괴 0.37–0.45, burst 0.88 |
| DC 유사도 | DC의 시드 간 cos. 높게 유지되어야 함 | — | 떨어지면 조건이 흔들림 |

붕괴 여부는 최종 latent AC 유사도 → 1과 블록 AC 유사도 → 0이 **함께** 나타나는지로 평가한다. FM MSE는 품질 점수가
아니다. 발산과 정상 여부를 확인하는 숫자이고, 이 루프에서는 품질을 따라가지 않는다.

### 5.3 믿으면 안 되는 학습 스칼라

| 스칼라 | 왜 오해하게 되나 |
|---|---|
| FM MSE (train / val) | 품질과 **반대로** 움직인다. 순위는 렌더로 매긴다 |
| fake loss 상승 | 선명해지는 student를 쫓는 정상 평형이다. fake에 손대는 것은 DM 차의 상대 크기가 오르거나 teacher–fake cos가 떨어질 때뿐이다 |
| teacher–fake cos, DM 차의 상대 크기와 RMS, grad norm | 앵커 없는 붕괴에서도 참조 런 범위에 있었다. ε가 모든 velocity를 지배하기 때문이다 |
| 앵커 loss | 앵커를 맞추는 정도이고 다양성을 재지 않는다. 다양성은 그리드에서만 보인다 |
| x0 예측 std, 롤아웃 std, student velocity RMS | 폭주(급등)만 잡는다. 흐린 샘플의 latent std는 정상이다 |
| GAN generator / disc loss | hinge 손실은 disc가 이기고 있어도 ≈0.69 / 1.39에 머문다. disc margin과 logit spread를 본다 |
| grad norm 구간 평균 | 구간 평균이라 스파이크 몇 개(최대 242)에 끌려간다. 중앙값은 같았다 |
| logit spread | logit이 하나뿐인 pooled head에서는 0이다. token head 런과 pooled head 런을 같은 기준으로 비교하지 않는다 |
| ΔW overlap | 시드만 바꿔도 0.09–0.10이다. 목적함수 차이를 가르지 못한다 |

**4스텝에서 잰 범위** (레시피 그대로, 0–125 / 125–220스텝 평균):

| 스칼라 | control | twin |
|---|---|---|
| teacher–fake cos | 0.953 / 0.966 | 0.960 / 0.961 |
| DM 차의 상대 크기 | 0.269 / 0.245 | 0.263 / 0.268 |
| x0 예측 std | 0.60 / 0.64 | 0.60 / 0.66 |
| disc margin | 0.11 / 0.14 | 0.10 / 0.17 |
| student grad norm 중앙값 / p95 | 2.86 / 18.6 | 2.91 / 12.9 |

**GAN 폭주 서명.** 4스텝에서는 token head가 2–3k까지 spread 0.27–0.30의 평형 고원을 가지므로 spread 상승만으로는
판단할 수 없다. 폭주는 bin 중앙값 spread > 0.40, disc margin > 0.5, disc loss < ~1.1(건강하면 1.33–1.36)이 **함께**
유지될 때다. 완화는 R1 가중 ≈ 100이나 disc LR 1e-5 → 3e-6으로 한다.

### 5.4 학습한 스텝 수보다 많은 스텝에서 더 좋아 보이는 체크포인트

체크포인트가 학습한 스텝 수보다 더 많은 스텝에서 더 좋아 보인다면, 아직 진짜 N스텝 맵에 도달하지 못했다는 신호다.
2스텝은 σ 0.75 → 0 한 번의 점프로 σ≈0.5 아래의 디테일 형성 대역을 통째로 건너뛴다.

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
