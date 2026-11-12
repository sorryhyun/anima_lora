# 4. DP-DMD 터보 증류: 4스텝 student LoRA 만들기

> 초안 (2026-11-12). Arm 0은 이 글을 쓰는 시점에 아직 읽는 중입니다.

**요약**

1. 터보는 CFG 4 teacher(28스텝, 스텝당 forward 2회)를 4스텝·CFG 1 student LoRA로 줄이는 DP-DMD 증류이고, 결과물은 추론 쪽 코드가 필요 없는 평범한 LoRA입니다.
2. 한 iteration은 teacher 앵커(첫 스텝의 다양성 목표) → student 롤아웃 → DM 서로게이트(teacher − fake 점수 차) → CDM·GAN 항 → fake(critic) 업데이트 순서로 돕니다. 세 역할이 frozen DiT 하나를 나눠 씁니다.
3. 지금 레시피는 merged14 soup을 teacher로 고정하고 DP-DMD 앵커, transition-aware fake, CDM 1.0, token GAN 0.03, 상수 LR 5e-5, EMA 0.2를 씁니다. EMA 체크포인트를 배포하고, 배포할 때는 soup LoRA와 exact concat해서 LoRA 하나로 냅니다.
4. 가장 쓸모 있는 발견: batch 1에서는 student 가중치가 random walk합니다(|ΔW| ∝ √t, 125스텝 이동끼리 직교, 시드만 바꾼 두 런의 ΔW 공유분은 9–10 %). 노이즈를 줄이는 방법 세 가지(DM draw 여러 번, gradient 누적, 체크포인트 평균)는 모두 같은 수준에 머물렀습니다. 그래서 raw 체크포인트 하나끼리 비교하면 대부분 노이즈를 읽게 됩니다.
5. 평가는 렌더로 합니다. EMA끼리 비교하고, null twin으로 노이즈 바닥을 재고, teacher 옆에 어려운 judge 프롬프트 렌더를 놓고, 모멘트 갭을 곁들입니다. 학습 스칼라(`fm_mse`, `fake_loss`, DM 스칼라)는 붕괴를 놓치거나 품질과 반대로 움직입니다.

다른 편과의 관계:

- LoRA 학습 자체는 [01_lora_recipe.md](01_lora_recipe.md)에서 다룹니다.
- soup과 exact concat merge, merged14가 어떻게 만들어졌는지는 [02_soup_merge.md](02_soup_merge.md)에 있습니다.
- teacher DiT로 bake하는 법과 teacher 전용 GPTQ int8 파일은 [03_bake_int8_gptq.md](03_bake_int8_gptq.md)에 있습니다.
- fp32 stream, VRAM 같은 실행 환경은 [05_practical_tips.md](05_practical_tips.md)에 있습니다.

---

## 1. 루프 구조

### 1.1 왜 DMD이고 왜 "diversity-preserved"인가

teacher는 28스텝 × (cond + uncond) = 이미지당 DiT forward 56회입니다. 4스텝 × 1회(CFG를 student에
구워 넣음)로 줄이면 약 14배 빨라집니다.

DMD(Distribution Matching Distillation)는 궤적 대신 분포를 맞춥니다. student가 만든 샘플을, student 분포를
따라가는 "fake" 점수 모델과 teacher 점수의 차이로 밀어 줍니다. 이 gradient는 reverse-KL이라 mode-seeking
성향이 있고, 그대로 두면 시드가 달라도 포즈와 구도가 한 mode로 모입니다. DP-DMD(Wu, Li, Zhang, Ma,
arXiv:2602.03139)는 일을 나눕니다. 첫 스텝은 teacher의 다양한 궤적에 회귀시켜 mode(포즈, 레이아웃)를
정하게 하고, DMD gradient는 나머지 스텝에만 흐르게 합니다.

### 1.2 세 역할, DiT 하나

LoRA 두 개(student, fake)를 같은 frozen DiT에 겁니다. 매 forward마다 어느 쪽을 켤지 바꿉니다.

| view | student LoRA | fake LoRA | 얻는 것 |
|---|---|---|---|
| teacher | off | off | base velocity (CFG는 바깥에서 `v_u + α(v_c − v_u)`) |
| student | on | off | `v_student`, 롤아웃 |
| fake | off | on | `v_fake`, student 분포의 점수 추적기 |

view 전환은 모듈마다 플래그만 뒤집으므로 DiT forward 한 번에 비하면 비용이 없습니다. optimizer는 두 개(LoRA마다
AdamW 하나)이고 DiT는 학습하지 않습니다.

teacher의 uncond 입력은 T5("") 임베딩입니다. 영벡터 crossattn은 분포 밖 입력이어서, 그 `v_uncond`가
(α−1)=3배로 증폭되면 student가 하얗게 포화된 출력으로 밀려납니다.

### 1.3 한 iteration의 순서

| # | 단계 | forward | grad |
|---|---|---|---|
| 1 | teacher K-step CFG 앵커 → `v_target` | teacher × 2K (6이면 12) | 없음 |
| 2 | student 롤아웃, 스텝 0 다양성 loss, grad 스텝 g의 x0 예측 | student × (g+1) | 스텝 0 (앵커), 스텝 g |
| 3 | DM 서로게이트 (DMD 또는 MFD) | teacher × 2, fake × 1 (MFD면 K배) | 없음 (신호만) |
| 4 | L_CDM, 따로 backward | student × 1 (grad), teacher × 2, fake × 1 | student |
| 5 | GAN generator 항 | teacher × 1 (블록 중간에서 early exit, grad 통과) | student |
| 6 | student loss 합산 → backward → clip → step → EMA | | |
| 7 | fake 업데이트 × (student 스텝당 fake 스텝 수), 매번 disc 1회 | fake × 1 (grad) + teacher × 2 (no-grad, half depth) | fake, disc |

### 1.4 DM 서로게이트

Anima는 flow matching 모델입니다. 경로는 $x_t = (1-t)\,x_0 + t\,\varepsilon$이고 velocity는
$v = \varepsilon - x_0$입니다. student가 만든 깨끗한 추정 $x_\theta$를 새 노이즈 레벨 $\tau \sim U(0,1)$로 다시
노이즈를 입힙니다.

$$
x_\tau = (1-\tau)\,x_\theta^{\text{detach}} + \tau\,\varepsilon_{dm}
$$

여기에 teacher(CFG를 적용한 real score)와 fake(cond만)의 velocity 차를 구합니다.

$$
\Delta_{dm} = \underbrace{v_u + \alpha\,(v_c - v_u)}_{\text{teacher, CFG }\alpha=4} - v_{\text{fake}}(x_\tau),
\qquad x_0^{\text{real}} - x_0^{\text{fake}} = -\tau\,\Delta_{dm}
$$

$x_\theta$를 $x_0^{\text{real}}$ 쪽으로 옮기려면 서로게이트의 $\partial/\partial x_\theta$가
$\tau\Delta_{dm}$이어야 합니다. DMD2의 grad trick을 씁니다.

- **real score는 반드시 CFG를 적용합니다.** CFG가 들어가는 곳은 여기뿐입니다. 양쪽 다 cond만 쓰면
  `v_real ≈ v_fake`(`dm_cos ≈ 0.9999`)가 되어 신호가 노이즈뿐입니다.
- **부호를 맞추는 것이 필수입니다.** 2026-05-27 수정 전에는 부호가 뒤집혀 있었고, 그때 증상은
  "학습이 안 된 few-step blur"였습니다.
- **x0 정규화.** 켜면 `grad = τΔ / clamp(mean|τ·v_real|, norm_floor)`로 샘플마다 크기를 정규화하고, 이때
  τ가 대부분 상쇄됩니다. 끄면 τ-damping만 남습니다. 둘 중 하나만 씁니다. 현재 레시피는
  정규화를 켭니다.
- **DM draw 여러 번.** 같은 `x_pred`에 (σ_a, ε_dm)를 여러 번 뽑아 신호를 평균합니다. loss가 신호에 선형이라
  backward는 한 번입니다. 2스텝에서 효과가 없어서 레시피는 1입니다(§4.3).
- **τ 분포.** uniform(기본)이나 logit-normal 중 하나입니다. logit-normal 런은 계획만 있고 실행하지 않았습니다.

### 1.5 DP-DMD 다양성 앵커

같은 시작 노이즈 $\varepsilon$에서 CFG teacher를 앵커용 teacher 격자 위로 K Euler 스텝만큼
굴려 $z_{t_k}$를 얻습니다. 스텝 0의 목표는 $[t_k, 1]$ 구간의 평균 velocity입니다.

$$
v_{\text{target}} = \frac{\varepsilon - z_{t_k}}{1 - t_k}, \qquad
\mathcal{L}_{div} = \big\| v_{\text{first}} - v_{\text{target}} \big\|^2,\quad v_{\text{first}} = G_\theta(\varepsilon, 1, c)
$$

| 노브 | 레시피 값 | 의미 |
|---|---|---|
| 앵커 스텝 K / teacher 앵커 격자 길이 | 6 / 12 | 둘의 비가 앵커가 떨어지는 σ 지점을 정합니다(6/12 = 0.5). 스텝 수는 teacher를 얼마나 촘촘히 적분하는지와 비용만 정합니다 |
| 다양성 가중 λ | 0.1 | 스텝 0 MSE의 가중치입니다. 0이면 앵커 자체를 건너뜁니다 (teacher forward 2·K회 절약) |
| 첫 스텝 뒤 detach | on | **필수**입니다. 다양성 항을 즉시 backward하고 스텝 0 그래프를 끊습니다. DMD reverse-KL이 첫 스텝으로 흘러들면 다양성이 다시 줄어듭니다 (논문 Fig 5) |
| 스텝 0에도 DM | off | grad 스텝을 `U{0..N−1}`에서 뽑아 스텝 0도 DM 신호를 받게 합니다. λ 0과 함께 쓰면 MFD 논문의 앵커 없는 루프가 되고, 이 조합은 250스텝 안에 붕괴했습니다 (§4.7) |
| rollout grad | off | 스텝 0에서 스텝 1까지 그래프를 유지합니다. 2스텝 static 격자 전용입니다 (superturbo, §4.10에서 종료) |

### 1.6 grad 스텝 g와 동적 격자

N스텝을 전부 backprop하지 않습니다. 매 iteration에 $g \sim U\{1..N-1\}$을 뽑고, 1..g−1은 no_grad로
흉내 낸 뒤, 스텝 g의 one-step x0 예측 $x_g - \sigma_g v_g$에만 grad를 줍니다. 메모리가 스텝 수와 무관하고 모든
refinement σ가 감독을 받습니다.

동적 격자를 켜면 CDM(arXiv:2605.06376 §3.2) 방식으로 매 iteration마다 롤아웃 격자를 새로
뽑습니다. 길이는 $N \sim U\{2..\text{student steps}\}$, 내부 점은 (0,1) 균등분포에서 뽑아 정렬하고,
$t_1 = 1$은 고정해서 앵커가 그대로 맞물리게 합니다. validation 렌더와 추론은
항상 shift 3의 static 격자를 씁니다. 4스텝 static 격자의 σ는 1 → 0.9 → 0.75 → 0.5 → 0입니다.

### 1.7 CDM off-trajectory loss

CDM 가중이 0보다 크면 grad 스텝의 (x_g, v_g, σ_g)를 detach한 뒤 Euler로 임의의 $t' \sim U(0,1)$까지
외삽합니다. 거기서 grad를 받는 student forward를 한 번 더 돌리고, 그 x0 추정 $x_{off} - t' v_{off}$에 같은
DM 신호를 줍니다. few-step Euler가 지나가지만 on-trajectory 롤아웃은 들르지 않는 영역을 감독하는 항입니다.
2스텝 100-step arm에서는 CDM이 "하늘을 하늘로" 유지했고, 끄면 물결과 난간 같은 잡구조가 생겼습니다. 레시피는 1.0입니다.

### 1.8 fake(critic): 업데이트, 비율, warmup, transition-aware fake

fake는 student 출력 분포를 평범한 flow matching으로 학습합니다.

$$
\mathcal{L}_{fake} = \big\| \text{fake}\big((1-\tau)x_\theta^{\text{detach}} + \tau\varepsilon,\ \tau,\ c\big) - (\varepsilon - x_\theta^{\text{detach}}) \big\|^2
$$

- **비율.** student 스텝당 fake 스텝 4입니다. 매번 (τ, ε)를 새로 뽑습니다. GAN이 켜져 있으면 disc도 이
  안쪽 루프에서 매번 한 번 업데이트됩니다. 움직이는 student를 fake가 앞서가게 하는 장치입니다.
- **fake warmup.** 본 루프 전에 fake만 200회 업데이트합니다. student는 건드리지 않습니다.
  student LR warmup이 끝날 때 fake가 아직 teacher와 같은 상태여서 생기는 초반 `grad_signal_rms` 스파이크
  (~50스텝)를 없앱니다. fake는 이미 calibrate되고 full LR인 상태로 본 루프에 들어갑니다.
- **rank.** fake rank ≥ student rank입니다. fake는 점수 추적기이므로 student가 만드는
  섭동을 따라갈 용량이 필요합니다.
- **transition-aware fake** (VA-TDM 아이디어 1). fake 샘플이 어느 student 스텝 σ_g에서 왔는지를 zero-init
  임베딩으로 fake의 t-embedding에 더합니다. fake view에서만 동작하고 저장할 때는 버립니다.
  500스텝 A/B에서 loss는 같았고 렌더는 더 선명했지만, 그 차이가 런 노이즈일 수 있습니다(§4.9).

### 1.9 GAN: teacher-feature token head

NVlabs FastGen에서 가져온 항입니다. frozen teacher DiT의 중간 블록 activation을 hook으로 읽는 약 2M 파라미터
discriminator를 둡니다.

- generator 항은 $\text{softplus}(-D(\text{feat}(x_\tau)))$이고, disc 항은
  $\text{softplus}(D_{fake}) + \text{softplus}(-D_{real})$입니다.
  real 쪽은 데이터셋 latent입니다. 루프 안에서 실제 데이터가 들어가는 곳은 여기뿐입니다.
- token head(LADD 방식, 토큰마다 logit)와 pooled head(탭마다 logit 하나)는 파라미터가 같습니다. token은
  글자, 손, 질감 같은 국소 결함을 벌합니다.
- 레시피는 generator 가중 0.03, disc LR 1e-5, disc AdamW betas (0, 0.99), 중간 블록입니다.
- GAN 기울기는 DM 기울기와 원소 단위로 직교합니다(§4.10). 텍스처를 사 오지 다양성은 사 오지 않습니다.
- 2스텝에서 token disc가 폭주한 기록이 있습니다(margin 0.22 → 6.94, spread → 0.99). 4스텝에서는
  같은 설정이 평탄했습니다. 폭주의 서명과 완화법은 §5.3에 있습니다.

### 1.10 최적화: clip, LR 스케줄, EMA

- **grad clip 1.0.** student, fake, disc 각각에 걸립니다. 실측으로는 거의 모든 스텝에서
  발동하므로 사실상 정규화기 역할을 합니다(§4.6).
- **LR 스케줄.** 총 스텝의 2 %는 선형 warmup입니다. 그 뒤 cosine(0.1·lr까지)이나 constant로
  갑니다. EMA 없이는 constant가 cosine보다 렌더가 나빴고, constant는 EMA가 정리를 맡는 런에 씁니다.
- **EMA.** EMA window는 런 길이에 대한 비율입니다. decay는 $1 - 1/(\text{window}\cdot\text{updates})$입니다.
  250 update에 window 0.2면 약 50 update를 평균하는 decay 0.98이 됩니다. EMA는 LR warmup이 끝나는 시점부터 돕니다.
- **gradient 누적.** iteration마다 clip한 student gradient의 평균으로 한 번 step합니다. fake와 disc는 매
  iteration 업데이트합니다. 2스텝에서 효과가 없어서 레시피는 1입니다.

### 1.11 출력과 추론

student만 standard plain-LoRA 레이아웃으로 저장합니다. fake, disc, transition 임베딩은 버립니다. 추론은 4스텝,
cfg 1.0입니다. CFG가 이미 구워져 있으므로 CFG를 다시 걸면 이중 guidance가 됩니다. 이 LoRA는 다른 개념 LoRA와
선형으로 합쳐집니다(LCM-LoRA + 스타일 LoRA와 같은 방식). 다만 Spectrum(≥16스텝 가정)과는 맞지 않고, 추론 시
step 크기나 t 입력이 필요한 것(Shortcut, MeanFlow Δt, T-LoRA 마스크)은 실을 수 없습니다.

체크포인트가 학습한 스텝 수보다 더 많은 스텝에서 더 좋아 보인다면, 아직 진짜 N스텝 맵에 도달하지 못했다는
신호입니다. 2스텝은 σ 0.75 → 0 한 번의 점프로 σ≈0.5 아래의 디테일 형성 대역을 통째로 건너뜁니다.

---

## 2. 배포된 base-teacher 터보 (Hub)

`huggingface.co/sorryhyun/anima-turbo-4step`은 **base Anima를 teacher로 쓴** 4스텝 student입니다. 카드에 적힌
내용은 "10k steps, 30h in single 5070ti"와 "CFG=1.0, steps=4, er_sde sampler"입니다.

그 런(`v3_longer`)은 이전 런 `U` 대비 다음이 바뀌었습니다.

| 항목 | `U` | `v3_longer` |
|---|---|---|
| iterations | 4000 | 10000 |
| alpha | 192 | 180 |
| warm start | turbo V10 delta r80 ASVD adaln | 공식 turbo v1.1 delta r96 ASVD adaln |
| CDM 가중 | 0 | 1.0 |
| 다양성 가중 λ | 0.05 | 0.1 |
| disc head | pooled | token |

당시 teacher는 bf16 base였습니다.

**warm start.** 이 줄의 student와 fake는 둘 다 공식 turbo v1.1 릴리스에서 base에 대한 delta를 rank 96으로
뽑아(adaln 포함) 시작합니다. activation-whitened(ASVD) 추출은 같은 rank의 plain SVD보다 재구성이 좋아서, rank 96
ASVD가 plain rank 128과 비슷합니다. 공식 turbo는 작동하는 few-step 맵이지만 mode가 붕괴해 있고, DP-DMD 앵커가
mode를 다시 펼칩니다. warm start된 student는 0스텝부터 이미 few-step 맵이므로 증류는 fine-tune에
가깝습니다.

이 레시피로 쌓인 경험치:

- warm start에서 LR 5e-5는 안정적이지만 경계에 가깝습니다. ~1k 부근에서 한 번 벗어났다 돌아오고, 쓸 만한
  체크포인트는 2k 이후입니다. cold start는 student LR이 1e-5를 넘으면 불안정합니다.
- v3_longer는 2k 이후로도 계속 좋아졌고, 최종 |ΔW|는 12.4입니다. v1.1 delta 자체의 norm은 3.17입니다.

학습 롤아웃과 이 글의 렌더는 Euler입니다. Hub 카드가 적은 `er_sde`는 확인하지 않았습니다.

---

## 3. 현재 레시피: soup teacher (merged14)

### 3.1 목표와 배포 규칙

목표는 soup merged14(8-shard concat_exact LoRA, rank 256, adaln 평균)를 teacher로 쓰고, 결과를 **plain
anima-base-v1.0에 올라가는 LoRA 하나**로 내는 것입니다.

- teacher DiT는 merged14를 1.0으로 bake한 DiT이고, 그 DiT 전용 int8 GPTQ 파일을 씁니다(03편).
- 배포는 normalize 없이, adaln을 더해서 exact $\Delta W_{14} + \Delta T$를 만듭니다. 기본 merge(global
  normalize, adaln average)를 쓰면 정확성이 깨집니다. 터보 LoRA만 base에 올리면 의미가 없고, 합친 LoRA는
  multiplier 1.0으로만 씁니다. exact concat 자체는 02편에서 설명합니다.

> TODO(확인 필요): 합친 LoRA를 base에서 렌더한 결과와 merged14 DiT + 터보 LoRA 렌더가 같은지 비교한 기록을
> 찾지 못했습니다. arm 0의 렌더는 merged14 DiT 위에 터보 LoRA만 얹었습니다.

### 3.2 레시피 표 (근거로 고정, 다시 A/B하지 않음)

| 부분 | 설정 | 근거 |
|---|---|---|
| teacher, base | merged14, int8 GPTQ, fp32 stream | soup turbo 기록, fp32 stream은 결론이 난 사항 |
| 다양성 앵커 | DP-DMD on: λ 0.1, K 6/12, 첫 스텝 뒤 detach, 스텝 0 DM off | 앵커 없는 MFD는 250스텝 안에 붕괴 |
| fake | transition-aware on, warmup 200, 4:1 | 500스텝에서 해가 없음 |
| CDM | 1.0 | 100-step arm에서 하늘을 하늘로 유지 |
| GAN | token head, generator 가중 0.03 | 폭주는 NFE 2의 실패이고 NFE 4에서는 평탄 |
| LR | student, fake 모두 5e-5, constant | LR은 walk가 시작되는 시점을 옮기지 않음. 스케줄 모양은 레버가 아님 |
| walk 제거 | EMA window 0.2, **배포하는 것은 EMA** | 2스텝에서 EMA 0.2가 체크포인트 평균과 같은 수준 |

4스텝에서 다시 A/B하지 않을 것: DM draw 수, gradient 누적, LR 스케일. 2스텝에서 셋 다 수준을 바꾸지 못했습니다.

warm start 없이 weight-SVD init을 쓰므로 ΔW가 0에서 시작합니다. 그래서 ΔW를 런 사이에서 바로 비교할 수 있습니다.

### 3.3 설정, 비용, 렌더

arm 0 control(`1112_0833`)은 250 iteration, batch 1, seed 42입니다. student와 fake 모두 rank 64 / alpha 180,
weight-SVD init, adaln rank 16 / alpha 90을 학습합니다. 4스텝, teacher CFG 4, 동적 격자, x0 정규화(floor 0.05),
DM draw 1, MFD off입니다. 나머지는 §3.2 표와 같고, grad clip 1.0, validation 시드 8개, 125스텝마다 저장과 validation을 합니다.
null twin `1112_0834`는 seed 43만 다릅니다.

- **비용(arm 0 실측).** iteration당 5.7–5.8 s이고, 200스텝 fake warmup 뒤 250 iteration에 약 27분이 걸렸습니다.
  피크는 13.70 / 13.73 GB allocated입니다. 둘 다 block swap과 gradient checkpointing 없이 16 GB 카드(5070 Ti)에서
  돌렸습니다.
- **렌더.** 4스텝, cfg 1, euler, shift 3, 768 × 1344(W × H), judge 프롬프트 세트를 씁니다. EMA 250, avg2, raw 250,
  EMA 125를 같이 렌더합니다.

---

## 4. 배운 것

### 4.1 batch 1의 student는 random walk한다

**2스텝.** 저장된 체크포인트의 ΔW = up·down을 364개 모듈 전체에 대해 Frobenius norm으로 쟀습니다.

| 런 | LR, 스케줄 | \|ΔW\| @125 / 250 / 375 / 500 | 연속 125스텝 이동 사이 cos |
|---|---|---|---|
| `1111_1115` (GAN) | 2e-5, constant | 2.77 / 3.98 / 4.85 / 5.50 | +0.018, −0.001 |
| `1111_1005` (GAN) | 5e-5, constant | 6.54 / 9.74 / 12.37 / 14.77 | +0.012, +0.012 |
| `…_k3_nogan_…_const_ema10` | 5e-5, constant | 6.66 / 9.27 / 11.23 / 12.77 | – |
| `…_k6_nogan_nocdm_500_tafake` | 5e-5, **cosine** | 5.71 / 7.74 / 8.21 / 8.26 | – |

- 상수 LR에서는 |ΔW|가 √t를 따릅니다(`1111_1115`: 2.77·√(t/125) = 3.92 / 4.80 / 5.54). 125스텝 이동은 서로
  직교합니다. 되돌리는 힘도 한쪽으로 미는 힘도 없습니다.
- LR을 2.5배 낮추면 이동이 2.56배 줄어들 뿐 다른 것은 바뀌지 않습니다. Adam은 drift와 노이즈를 같은 비율로
  줄이므로 노이즈가 학습을 앞지르는 시점이 그대로입니다.
- walk는 GAN이 없어도 일어납니다. 멈춘 것은 cosine 런 하나(8.21 → 8.26)이고, 마지막 150스텝에 `div_loss`,
  `grad_signal_rms`, `dm_cos`가 좋아진 것도 그 런뿐입니다.

**4스텝.**

| | control `1112_0833` | twin `1112_0834` |
|---|---|---|
| \|ΔW\| raw 125 / 250 | 7.34 / 10.33 | 7.49 / 10.54 |
| √2 · \|ΔW\|₁₂₅ | 10.37 | 10.59 |
| \|이동\| 0–125 / 125–250 | 7.34 / 7.14 | 7.49 / 7.21 |
| cos(이동 0–125, 이동 125–250) | 0.018 | 0.028 |
| \|ΔW\| EMA 250 | 8.73 | 8.98 |
| cos(EMA 250, raw 250) | 0.961 | 0.965 |

4스텝도 같은 상태입니다. 125스텝부터 √t로 자라고, 길이가 같고 서로 직교하는 두 번의 이동으로 이루어집니다.

**런 사이 공유분.** 두 런 ΔW 사이의 cos는 공유하는 에너지 비율로 읽을 수 있습니다.

| 쌍 | cos |
|---|---|
| arm 0 control vs twin (시드만 다름), raw 125 / raw 250 / EMA 250 | 0.087 / 0.093 / 0.100 |
| `m14_dpdmd_svd_500` vs `_tafake` (4스텝, 시드·init·데이터 순서 같음, transition-aware fake만 다름) | 0.14 |
| `1111_1115` vs `1111_1740` (2스텝, DM draw 수만 다름) @125 / 250 / 375 / 500 | 0.19 / 0.17 / 0.16 / 0.17 |
| 그 밖의 쌍 (LR이 다르거나 2스텝 vs 4스텝) @500 | 0.09–0.11 |

- 4스텝 student의 ΔW 중 약 85–90 %는 그 런에만 있는 성분입니다. 공유분은 특정 위치에 몰려 있지 않습니다.
  모듈 종류별로 0.11–0.19, 블록 그룹별로 0.10–0.21이고 블록 21–27이 가장 높습니다.
- 시드만 바꿔도 0.09–0.10이 남습니다. 0.14였던 쌍은 데이터 순서까지 같았으므로, 그 0.14에는 데이터 순서의
  몫이 들어 있을 가능성이 큽니다.
- **결론: ΔW overlap으로는 arm을 가를 수 없습니다.** 목적함수가 다른 arm도 reseed와 똑같이 읽힙니다.
- 시드 평균: 공유분이 c ≈ 0.14일 때 k개 시드의 exact ΔW 평균은 공유분을 $kc/(1+(k-1)c)$로 올립니다. k = 2면
  0.25, k = 4면 0.39입니다. 2스텝에서 두 런을 평균한 `avg2_500`은 250 수준을 넘지 못했으므로, 이것은 품질을
  올리지 못하고 정리 작업에 그칩니다.
- 이 측정 때문에 과거 해석 하나가 흔들립니다. 2026-11-10의 transition-aware fake A/B(tafake가 더 선명하고, plain은
  갈색 머리와 가운데 간판 하나로 흘렀다)는 이런 두 번의 추첨을 비교한 것이었고, 렌더 차이의 대부분은 런
  노이즈일 수 있습니다.

### 4.2 스텝당 gradient는 대부분 노이즈다

2스텝 `1111_1115`의 375스텝 체크포인트에서 16샘플 × 3 DM draw로 gradient를 분해했습니다.

| student 스텝 방향 에너지 (clip 1.0이 남기는 단위 벡터 기준) | 비율 |
|---|---|
| 한 샘플 안의 (σ_a, ε_dm) draw 사이 | 77 % |
| 샘플 사이 | 20 % |
| 전체 평균 (누적되는 몫) | 3 % |

- 항별 제곱 norm 비율은 rollout 경로를 거친 DM 56 %, 스텝 1의 DM 13 %, 앵커 7 %, GAN 7 %입니다.
- DM gradient norm은 σ_a에 따라 커집니다. 중앙값이 0.25 미만에서 1.24, 이후 2.03, 2.36이고 0.75 초과에서 4.93입니다.
  σ_a 0.96에서 21.9, 0.99에서 30.5가 나왔습니다. clip 아래에서는 이런 스텝도 다른 스텝과 같은 무게로 들어갑니다.
- 평균 방향을 보면 DM과 앵커는 일치합니다(cos +0.55). GAN의 평균은 DM과 약간 반대입니다(−0.19).
- 한계가 있습니다. 체크포인트 하나, draw 3개이고, 16샘플이 모두 가장 큰 토큰 버킷(4154–4200 토큰)에서 나왔습니다.
  4스텝에서는 이 측정을 하지 않았습니다.

### 4.3 노이즈를 줄여도 수준은 오르지 않는다

2스텝에서 노이즈를 줄이는 세 경로를 시험했고, 모두 `1111_1115` 250스텝 수준에서 멈췄습니다.

| 경로 | 런 | 결과 |
|---|---|---|
| DM draw 4 | `1111_1740`, 9.5 s/step (control 6.5) | 실패. 250은 control과 비슷하고 375는 자기 250보다 나쁘며 500은 회복. "좋아졌다 나빠졌다 다시 좋아지는" 것은 walk의 모습입니다. \|ΔW\| 2.80 / 3.92 / 4.70 / 5.35, 평균 몫 2.9 % (control 3.0 %) |
| 체크포인트 exact 평균 | `avg3_1115`, `avg2_500`, `avg6_both` | 500에서 생긴 벤치, 차량, 갈색 머리를 걷어 냅니다. 그러나 어느 것도 `1111_1115` 250을 넘지 못함 |
| gradient 누적 4 | `1111_2056`, 1000 it = 250 update, 72분, 4.0 s/it, 피크 13.79 GB | 실패. raw 최종은 둘 다보다 나쁘고(뒤틀린 몸), EMA 최종은 `avg6_both` 수준. \|ΔW\| 2.78 / 3.84, 이동 cos 0.022 |

- 4 draw가 4배 효과를 내지 못하는 이유는 σ_a tail입니다. 큰 draw는 대부분 노이즈이고, norm 가중으로 본 DM 평균
  몫은 1.6 %, 단위 가중으로는 4.3 %입니다. 4 draw 중 유효 draw 수는 2.23(iid 가정)입니다.
- **해석: 수준은 목적함수가 정하고, batch-1 walk는 거기서 빼기만 합니다.**

### 4.4 EMA는 walk 제거 장치다

- 2스텝에서 window 0.1(약 50스텝)의 EMA는 walk를 걷어 내지 못했습니다(`1111_1115` 500: EMA 5.12, raw 5.50).
  window 0.2(`1111_2056`)는 체크포인트 평균과 같은 수준까지 갔습니다.
- 4스텝 arm 0에서 EMA 0.2는 raw 방향의 96 %를 유지하고 ΔW 길이를 15 % 줄입니다. 125스텝 이동 하나 안의 walk는
  지우지 못합니다.
- 그래서 레시피는 상수 LR에 EMA를 두고, 배포하는 것은 EMA 체크포인트입니다. cosine LR은 walk를 멈춘 유일한
  설정이었지만, 레시피는 "LR이 walk 시작 시점을 옮기지 않으며 스케줄 모양은 레버가 아니다"라는 판단으로 constant를
  골랐습니다.

### 4.5 발산 burst: raw 체크포인트가 그 안에 걸릴 수 있다

arm 0 control의 raw 250은 judge 이미지 12장 모두에서 평평한 노이즈 텍스처를 냈습니다. 파일은 정상이었고(같은 키,
유한값), 런이 마지막 몇 스텝에서 발산했습니다.

- student grad norm이 242–245스텝에서 5.5 / 14 / 6 / 16이다가 246–249에서 **231 / 244 / 106 / 72**가 되었습니다.
  fake grad norm도 같이 올라(25 → 111 / 69 / 145 / 85), student와 fake가 함께 터졌습니다.
- 그 구간의 σ_a는 0.03–0.96이므로 σ_a tail 탓이 아닙니다.
- `x_pred_std`가 0.74에서 1.98로 올랐고, 250 validation에서 `fm_mse`가 0.29 → 0.88, `div_xpred_ac_sim`이 0.948 → 0.999로
  바뀌었습니다. §4.7의 붕괴 서명과 같습니다.
- twin도 153–162스텝에서 같은 종류의 burst(37 / 29 / 58 / **242** / 131)를 겪었지만 회복했습니다.
- EMA 0.2는 burst를 희석해서, control의 EMA 250은 정상으로 렌더됩니다. 125 + 250 평균에는 burst가 일부 남습니다
  (기타리스트 프롬프트의 블록 같은 군중).

**규칙: 이 레시피에서는 250 iteration에 한 번꼴로 burst가 나고, raw 체크포인트 하나가 그 안에 걸릴 수 있습니다.
raw를 배포하거나 raw끼리 비교하지 마세요.**

### 4.6 grad clip 1.0은 정규화기다

clip 이전 student norm(5스텝 구간 평균)은 2스텝 런 네 개 모두에서 **모든 구간이 1.0을 넘었습니다**(중앙값
3.20–5.00, p95 7.4–10.1). fake도 마찬가지입니다(중앙값 2.2–2.9).

- Adam은 gradient의 상수배를 무시하므로 clip을 올려도 스텝이 커지지 않습니다. 스텝 크기는 LR이 정합니다. clip이
  정하는 것은 스텝끼리의 상대 무게입니다. 1.0이면 모든 스텝이 단위 norm으로 들어가고, clip이 없으면 norm만큼의
  무게로 들어갑니다.
- 2스텝에서는 큰 norm이 곧 노이즈였습니다. norm 가중을 하면 DM 평균 몫이 4.3 %에서 1.6 %로 떨어집니다.

**4스텝 σ_a 프로필** (arm 0, clip 이전 student grad norm의 σ_a 사분위별 중앙값):

| | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | top/bottom | 중앙값 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| control | 2.67 | 2.50 | 2.34 | 4.41 | 1.65 | 2.86 | 18.6 | 244 |
| twin | 2.80 | 2.54 | 3.05 | 3.57 | 1.27 | 2.91 | 12.9 | 242 |
| 합산 | 2.76 | 2.52 | 2.62 | 3.94 | **1.43** | 2.88 | 15.0 | 244 |

- 1.0 아래인 iteration이 하나도 없으므로 4스텝에서도 clip은 매 스텝 작동합니다.
- σ_a < 0.75에서는 평탄하고 맨 위 사분위에서만 오릅니다. 상승폭은 2스텝(4.9 / 1.2)보다 완만한데, rollout 경로가
  없다는 점과 맞습니다.
- 계획에 미리 정해 둔 규칙(합산 top/bottom ≤ 1.5)에 따라 arm 3은 **adaptive clip 단독**입니다. 다만 런별 비율
  (1.65, 1.27)이 경계를 사이에 두고 갈리므로 아슬아슬한 평가입니다.

### 4.7 앵커 없는 루프는 붕괴한다 (MFD 논문 루프)

스텝 0에도 DM을 주고 다양성 가중을 0으로 두면(MFD 논문의 앵커 없는 루프) 250스텝 안에 student가 붕괴합니다.
스텝 0의 출력이 시드와 프롬프트에 상관없이 한 색이 되고, 4스텝 렌더는 인물 없는 노이즈 텍스처가 됩니다.
공통 설정은 merged14 int8 teacher, weight-SVD r64/α180, transition-aware fake, 4스텝 + 동적 격자, MFD dtau 0.1 /
max 4, GAN과 CDM off, lr 5e-5, 4:1, batch 1입니다.

| 런 | fake warmup | `div_xpred_ac_sim` @250 | `div_ac_sim` @250 | `fm_mse` @250 |
|---|---|---|---|---|
| `m14_dpdmd_svd_500_tafake` (참조: 앵커 on) | 200 | 0.940 | 0.678 | 0.357 |
| `…_mfd_noanchor_nogan_nocdm_500_tafake` | 200 | 0.9994 | 0.077 | 0.371 |
| `…_mfd_noanchor_nogan_nocdm_fwu0_500_tafake` (250에서 중단) | 0 | 0.9996 | 0.155 | 0.452 |

- **학습 스칼라로는 보이지 않습니다.** 붕괴한 런의 `rel_gap`, `cos`, `dm`, grad norm은 참조 런 범위 안에 있습니다.
  모든 velocity를 ε가 지배하기 때문입니다. 서명은 validation probe에 나타납니다(§5.2).
- 원인에서 제외된 것은 g = 0 경로, transition-aware fake, fake LR warmup입니다. LoRA 자체도 원인이 아닙니다(논문의
  SANA 런도 student와 fake 모두 LoRA r64).
- 아직 남은 차이는 fake:student 10:1(논문) 대 4:1, grad clip 0.5, logit-normal (s, t)와 adaptive ODE 대 uniform σ_a,
  1스텝 student 대 2–4스텝이 공유하는 LoRA 하나입니다. **원인은 아직 분리하지 못했습니다.**
- 함정: 앵커를 없애도 DM 스칼라는 건강해 보입니다. 앵커는 다양성 장치이면서, 이 루프에서는 student를 붙잡아 두는
  힘이기도 합니다. 2스텝 측정에서 앵커와 DM의 평균 방향은 일치했습니다(§4.2).

### 4.8 MFD를 DM 슬롯에 넣기 (2스텝 100-step arm)

MFD(arXiv:2606.11155)는 점 하나의 차 $\tau(v_{real} - v_{fake})$ 대신, σ_a에서 σ_b까지 K번의 Euler로 teacher와
fake를 각각 적분한 끝점의 차를 씁니다. σ_b = 0, K = 1이면 DMD와 같습니다. 2스텝, v1.1 delta init, GAN off,
100 iteration 결과:

| arm | CDM | dtau | 평균 K | s/step | `dm` 평균 ± sd |
|---|---|---|---|---|---|
| `dpdmd2_100` | 1.0 | – | 1 | 5.33 | 0.194 ± 0.057 |
| `mfd2_100` | 1.0 | 0.1 | 3.4 | 6.32 | 0.167 ± 0.041 |
| `mfd2_100_dt003_nocdm` | 0 | 0.03 | 17.2 | 11.30 | 0.185 ± 0.035 |

- DM 신호의 스텝 간 sd는 K 1 → 3.4에서 약 35 % 줄고, K 17에서는 더 줄지 않았습니다.
- 학습한 모든 arm에 init에는 없던 잡내용(하늘의 검은 형체, 두 번째 캔)이 생겼고, **K가 클수록 그 형체가 또렷한
  사물이 되었습니다**(K 17에서 악마 꼬리). 분산 감소로도, teacher ODE를 더 정확히 적분한 편향으로도 설명됩니다.
- CDM을 켜면 하늘이 하늘로 남았습니다. 500스텝 dt 0.03 no-CDM에서는 떠다니는 캔과 탑이 사라졌지만 꼬리와
  얼룩은 남았습니다.
- 현재 레시피는 MFD off이고, MFD는 arm 1로 4스텝에서 다시 시험합니다(§6). MFD 논문의 T2I 결과는 4스텝
  SANA, 3000스텝입니다.

### 4.9 transition-aware fake: 해는 없음, 평가는 아직

`m14_dpdmd_svd_500_tafake` 대 `…_500`(시드, init 같음, transition-aware fake만 다름, 500스텝).
350–500스텝의 쌍별 차이는 fake −0.002 ± 0.003, dm +0.002 ± 0.003이고 cos는 둘 다 0.978입니다. 렌더에서는 tafake가
더 선명하고 teacher에 가까웠고, plain은 갈색 머리와 가운데 간판 하나로 흘렀습니다.
§4.1의 측정 이후에는 이 렌더 차이도 런 노이즈일 수 있다고 봅니다. 레시피에는 "해가 없다"는 근거로 남아 있습니다.

### 4.10 접은 줄기들

각각 무엇을 했고, 왜 접었고, 어떤 함정이 있었는지 정리합니다.

**Superturbo: 2스텝 rollout grad (2026-11-10 ~ 11-12, 종료).**
2스텝 static 격자(σ 1 → 0.75 → 0)에서 스텝 0은 앵커만, 스텝 1은 모든 DM을 받습니다. 두 스텝이 gradient를 나누지
않으므로 그래프를 이어 스텝 0이 최종 이미지의 gradient를 받게 했습니다. `1111_1005`(LR 5e-5)와 `1111_1115`
(LR 2e-5) 모두 250이 최고였고, DM draw 여러 번, gradient 누적, 체크포인트 평균도 250 수준을 넘지 못했습니다.
`1111_1005`에서는 ~350스텝부터 disc가 폭주했습니다(margin 0.9–1.6, `x_pred_std` → 0.43).
**함정:** "250이 제일 좋다"는 것은 walk의 문제였고, raw 체크포인트 하나를 다른 하나와 비교하면 walk를
읽게 됩니다. 4스텝 계획은 이 줄의 측정을 그대로 가져갑니다.

**APEX: student, reference, fake를 LoRA 하나로 (2026-11-01, 종료).**
조건 이동 `c_fake = −c`로 LoRA 하나가 생성기와 fake를 겸하게 했습니다. step은 3–5배 저렴했습니다(1.3–2.1 s/it 대
6.7–7.0). 제안서 형태(arm A, `F − d`)는 55스텝부터 `x_pred` std가 0.77 → 7.6으로 폭주했고, 논문 형태(arm D, 1500스텝)는
안정적이었지만 모든 체크포인트가 자기 warm start보다 흐리고 어두웠습니다. 추론된 이유는 `L_sup`이 student를 velocity
field로 다시 끌어당기고, velocity field를 4스텝 Euler로 풀면 조건부 평균이 나온다는 것입니다.
**함정:** arm D의 스칼라(`mix_v`, `x_pred` std 0.62, `roll_std`)는 그리드가 망가지는 동안 내내 건강했습니다. `roll_std`는
폭주만 잡고 흐림은 못 잡습니다. 첫 체크포인트에서 그리드를 렌더하세요.

**DMAD: disc가 DM을 대신 (2026-10-10, 이득 없음).**
fake critic을 classifier(head T: teacher 대 student, head R: real 대 student)로 바꾸는 시도였습니다. 같은 `x_pred`에서
독립 DM 추정 두 개는 cos ≈ 0.12로만 일치합니다(Phase −1). Phase 0b에서 신호는 DP-DMD보다 2–18배 작았고, 11.9 s/step 대
6.6으로 논문이 말한 4–5배 빠른 것이 아니라 약 1.8배 느렸습니다. 결과는 "덜 움직였다"였습니다. Phase 0c(raw 신호 + CDM,
λ 150, 1500스텝)에서는 ~300–980스텝 동안 disc가 포화되어 입력 gradient가 사라졌고, `x_pred` std가 0.6 → 5로 올라
500 / 750 렌더가 줄무늬 텍스처로 붕괴했다가 1k에서 스스로 회복했습니다.
**함정:** raw 신호는 상한이 없어 과하게 튀고, 포화된 disc는 student를 붙잡는 힘이 없습니다. 둘 다 λ로는 막을 수
없습니다.

**Data-space iteration (실행 안 함).**
schedule 없이 x̂0를 반복 갱신하는 구조입니다. 이식하면 입력 projection이 하나 늘어서 출력이 plain LoRA가 아니게
되고, 논문도 class-conditional ImageNet 256만 다룹니다. 남은 저비용 아이디어는 DM 쿼리 τ ~ Beta(2.5, 1), student Adam
β1 = 0, real score의 guidance interval이고 모두 미실행입니다.

**FEI band-deficit 레버 (2026-06-20, 두 번 죽음).**
student가 teacher보다 뒤처진 주파수 대역에 맞춰 CFG uplift를 재가중하는 레버였습니다. Phase 0 probe는 student 롤아웃
상태(과도한 blur → `w_high`)를 쟀는데, loss는 실제 데이터를 다시 노이즈 입힌 점에서 재고 있었습니다(→ `w_low`).
부호가 반대였습니다. on-trajectory로 다시 쟀을 때 σ를 맞춘 Site A에서 `frac_over_low` 0.03이 나왔습니다. 고칠
blur 결손이 없고, DP-DMD 앵커가 이미 흡수했다고 봅니다.
**함정:** go/no-go probe는 loss가 볼 바로 그 분포에서 같은 양을 재야 합니다.

**GAN ↔ DM gradient surgery (2026-07-20, 종료).**
GAN gradient 중 DM과 부호가 어긋나는 성분만 막는 OPD² 방식 게이트를 검토했습니다. agree-energy가 0.489 ± 0.032로
permutation null 0.488 ± 0.011과 구별되지 않았고, cos(g_gan, grad_signal)은 0.006이었습니다. GAN은 DM과 싸우는 것이
아니라 직교 방향으로 밀며, 원소당 크기는 DM의 약 6배(tail이 두꺼움)입니다.
**함정 두 개:** 원소 부호 일치율 0.5는 직교의 기준선이지 충돌이 아닙니다. 그리고 DM 서로게이트가 평균으로 줄여지므로
실제 적용되는 DM 힘은 `grad_signal/numel`이고, 원본 텐서끼리 비교하면 GAN을 ~10⁵배 과소평가합니다.

**τ-split critic (2026-07-20, 종료).**
fake를 τ 대역별로 나누면 대역 간 간섭이 줄어드는지 봤습니다. annealed tail LR(4.04e-6)에서는 17–19배 신호로 "FIRE"였지만,
실제 학습 peak LR(3e-5)에서는 G1/G2 실패였습니다. lo 대역만 학습한 clone이 hi 대역도 개선했습니다.
**함정:** probe fine-tune은 실제 학습 LR에서 돌려야 합니다. 작은 LR은 init 근처의 선형 영역만 재고, 노이즈 바닥을
낮춰서 SNR을 부풀립니다.

**IDM `L_pop` 모멘트 갭 (2026-10-31, 측정만).**
128 item, teacher 블록 27의 pooled feature로 쟀습니다. T1–T4(CFG 갭)는 nW2² 0.186 / 0.189
(τ 0.8 / 0.5)로 null의 약 17배이고, S(v1.1 delta)–T4는 0.032 / 0.060입니다. S의 pooled spread는 T4 대비 0.885(τ 0.8),
0.858(τ 0.5), 0.798(latent 채널 평균)로 student의 분산이 부족합니다. 데이터셋은 CFG 1보다 CFG 4에 더 가깝습니다.
**함정:** S는 DP-DMD student가 아니라 mode가 붕괴한 v1.1 delta였습니다. DP-DMD student에서 다시 재기 전에는 `L_pop`에
arm을 쓰지 않습니다.

**VA-TDM (부분 채택).**
아이디어 1(transition-aware fake)은 레시피에 들어갔습니다. 아이디어 2(teacher로 정규화한 fake + 보정된
critic, λ_r 1)는 arm 4 후보이고 아직 만들지 않았습니다. 3(중간 상태 매칭), 4(B-form, flow-product)는 구조 변경이거나
fake를 통한 backward가 필요해서 뒤로 미뤘습니다. 논문 부록과 코드는 아직 읽지 않았습니다.

---

## 5. 터보 런을 평가하는 법

### 5.1 A/B 프로토콜

1. **250 iteration**, 125스텝마다 저장, EMA on.
2. **250에서 EMA끼리 비교합니다.** raw 125와 250의 exact 평균은 교차 확인용입니다.
3. **null twin은 한 번만 만듭니다.** control에서 seed만 42 → 43으로 바꾼 런입니다. control과 twin의 차이가 이후
   모든 arm의 노이즈 바닥이 됩니다.
4. **평가는 렌더로 합니다.** 4스텝, cfg 1, judge 프롬프트(프롬프트 3개 × 시드 1–4), 768 × 1344로 렌더하고,
   teacher(merged14 int8, 28스텝, cfg 4)를 옆에 둡니다. 프롬프트는 위치 절이 붙은
   블루 아카이브 학생 다섯 명의 슬랩스틱, 일본어 메뉴와 영어 간판이 있는 두 소녀의 카페, 아래에서 더치 앵글로 본
   관객 위로 점프하는 기타리스트입니다. ANIMA 간판과 nozomi 프롬프트는 너무 쉬워서 arm을 가르지 못합니다.
5. **렌더 옆에 분포 숫자를 둡니다.** 모멘트 갭의 S–T4 nW2²와 merged14 teacher(cfg 4) 대비 pooled
   spread 비율입니다. T4 세트는 한 번 렌더해서 재사용합니다.

**통과 기준(미리 고정):** arm의 EMA 렌더가 두 프롬프트 모두에서 control보다 낫되 그 차이가 control–twin 차이보다
커야 하고, S–T4 nW2²가 control보다 |control − twin| 이상 낮아야 합니다. 통과한 arm은 새 control이 됩니다. step 시간이나
loss 항이 바뀌지 않으면 twin을 새로 만들지 않습니다.

arm 0의 S–T4 수치는 아직 없습니다.

**렌더를 읽는 순서.** 시드 간 구도 다양성을 먼저 보고, 그다음 선명도, 잡내용, 글자 순서로 봅니다. 448 px 다운스케일
그리드로 읽은 것과 글자 단위로 원본 크기에서 읽은 것은 구분해서 적습니다.

### 5.2 validation probe와 그 서명

validation 때마다 held-out 조건 하나로 8개 시드의 student를 static 격자로 굴립니다.

| 지표 | 정의 | 건강 | 붕괴 서명 |
|---|---|---|---|
| `div_ac_sim` | 블록 출력에서 토큰 평균(DC)을 뺀 AC 잔차의 시드 간 cos를 모든 스텝·블록에 대해 평균. 낮을수록 다양 | 참조 런 0.678 @250, arm 0 0.68 / 0.59 @125 | 0 근처 (0.077, 0.155): 최종 출력이 서로 무관한 노이즈 |
| `div_xpred_ac_sim` | N스텝 롤아웃이 끝난 최종 latent에 같은 DC/AC 분해를 적용한 시드 간 cos | 0.940–0.95 | 1 근처 (0.9994, 0.999) |
| `fm_mse` | held-out 샘플의 student 격자 σ에서 flow-matching MSE | 0.27–0.36 | 앵커 없는 붕괴 0.37–0.45, burst 0.88 |
| `div_dc_sim` | DC의 시드 간 cos. 높게 유지되어야 함 | — | 떨어지면 조건이 흔들림 |

붕괴 여부는 `div_xpred_ac_sim → 1`과 `div_ac_sim → 0`이 **함께** 나타나는지로 평가합니다. `fm_mse`는 품질 점수가
아닙니다. 발산과 정상 여부를 확인하는 숫자이고, Anima에서는 품질을 따라가지 않습니다.

### 5.3 믿으면 안 되는 학습 스칼라

| 스칼라 | 왜 오해하게 되나 |
|---|---|
| `fm_mse` (train/val) | 품질과 **반대로** 움직입니다. 순위는 렌더로 매기세요 |
| `fake_loss` 상승 | 선명해지는 student를 쫓는 정상 평형입니다. fake 개입은 `dm_rel_gap` ↑ 또는 `dm_cos` ↓일 때만 |
| `dm_cos`, `dm_rel_gap`, `dm`, grad norm | 앵커 없는 붕괴에서도 참조 런 범위에 있었습니다. ε가 모든 velocity를 지배하기 때문 |
| `div_loss` | 앵커를 맞추는 정도이지 다양성이 아닙니다. 다양성은 그리드에서만 보입니다 |
| `x_pred_std`, `roll_std`, `v_student_rms` | 폭주(급등)만 잡습니다. 흐린 샘플의 latent std는 정상입니다 |
| `gan_gen_loss` / `gan_disc_loss` | hinge 손실은 disc가 이기고 있어도 ≈0.69 / 1.39에 머뭅니다. disc margin과 logit spread를 보세요 |
| grad norm 구간 평균 | 구간 평균이라 스파이크 몇 개(최대 242)에 끌려갑니다. 중앙값은 같았습니다 |
| logit spread, CDM grad rms | pooled head에서는 0으로 고정입니다. token 런과 pooled 런을 비교 기준으로 섞지 마세요 |
| ΔW overlap | 시드만 바꿔도 0.09–0.10입니다. 목적함수 차이를 가르지 못합니다 |

**GAN 폭주 서명.** NFE 4에서는 token head가 2–3k까지 spread 0.27–0.30의 평형 고원을 가지므로 spread 상승만으로는
판단할 수 없습니다. 폭주는 bin 중앙값 spread > 0.40, disc margin > 0.5, `gan_disc_loss` < ~1.1(건강하면 1.33–1.36)이
**함께** 유지될 때입니다. 완화는 R1 가중 ≈ 100이나 disc LR 1e-5 → 3e-6으로 합니다.

### 5.4 arm 0에서 잰 건강 범위 (4스텝, merged14, 레시피 그대로)

0–125 / 125–220 스텝 평균입니다. control, twin 순서.

| 스칼라 | control | twin |
|---|---|---|
| `dm_cos` | 0.953 / 0.966 | 0.960 / 0.961 |
| `dm_rel_gap` | 0.269 / 0.245 | 0.263 / 0.268 |
| `x_pred_std` | 0.60 / 0.64 | 0.60 / 0.66 |
| disc margin | 0.11 / 0.14 | 0.10 / 0.17 |
| student grad norm 중앙값 / p95 | 2.86 / 18.6 | 2.91 / 12.9 |

### 5.5 arm 0 control 렌더 (대략 읽음, 2026-11-12)

teacher, EMA 250, avg2, raw 250, EMA 125를 나란히 놓았습니다.

- **EMA 125가 가장 좋고, EMA 250은 더 획일적입니다.** 250에서는 카페가 네 시드 모두 같은 분홍 앞치마와 레이아웃이고,
  다섯 소녀 프롬프트는 회색 포장도로 위 고정된 배치에 하늘이 사라졌습니다. EMA 125는 앞치마 색이 바뀌고, 하늘과
  헤일로와 색종이가 남고, 군중 실루엣도 그립니다.
- **teacher의 구도에 도달한 student는 없습니다.** teacher는 아래에서 본 점프와 뒤엉킨 슬랩스틱 무리를 그리지만,
  student는 모든 시드에서 정면 선 자세를 그립니다. teacher는 `CAFE MILLENNIUM`을 쓰고, student는 뭉갭니다.
- 런 하나뿐이라 아직 노이즈 바닥 위의 평가가 아닙니다. "125가 250보다 낫다"가 레시피 탓인지 시드 탓인지는 twin의
  그리드가 정합니다.

---

## 6. 남은 질문과 다음 arm (미완, 2026-11-12 기준)

> 이 절은 진행 중입니다. Arm 0은 학습을 마쳤고 control 렌더를 대략 읽었으며, twin 렌더는 대기 중입니다. GAN-off
> probe `1112_0917`은 실행 중이고 125 체크포인트까지 나왔습니다.

**arm 순서** (각각 arm 0 대비 하나만 바꿈):

| arm | 바꾸는 것 | 상태 |
|---|---|---|
| 0 | control + null twin. 250 수준, 노이즈 바닥, 4스텝 \|ΔW\| 궤적, σ_a별 grad norm 프로필 | 학습 완료, 렌더 읽는 중 |
| GAN-off probe | GAN generator 가중 0, 측정만 | 실행 중. 연속 이동 cos ≥ 0.1이고 \|ΔW\|₂₅₀이 √2·\|ΔW\|₁₂₅보다 분명히 크면 GAN이 walk의 원인. 기준값은 arm 0의 cos 0.018 / 0.028, 비율 0.996 / 0.995 |
| 1 | MFD dtau 0.1 / max 4를 DM 슬롯에 | 미실행. 4스텝은 MFD 논문이 결과를 보인 설정 |
| 2 | static 격자 (동적 격자 off) | 미실행. 동적 격자는 이미 노이즈가 지배하는 스텝에 추첨 하나를 더 얹고, 추론은 static 격자를 씀 |
| 3 | gradient norm: adaptive clip | **미구현.** arm 0의 σ_a 프로필로 adaptive clip 단독이 정해짐. burst를 겨냥하므로 student와 fake를 모두 cap해야 함 |
| 4 | teacher로 정규화한 fake + 보정된 critic (VA-TDM 아이디어 2, λ_r 1) | 미구현. fake step마다 teacher forward가 하나씩 늘어남 |

**만들지 않은 노브:**

- adaptive clip: 임계값을 clip 이전 norm의 running median(log 공간 EMA) × k(≈ 2)로 net마다 둡니다.
  보통 스텝은 자기 norm 그대로 통과하고 스파이크만 잘립니다. 임계값은 런 동안의 drift(`1111_2056`에서 2.5 → 5)를 따라갑니다.
- σ_a별 DM 가중: DM loss를 σ_a bin(8개)의 running median norm으로 나눕니다. arm 0 프로필에서는 필요 없다고
  평가했습니다.

**남은 질문:**

- **250 iteration에서 4스텝 수준에 도달하는가.** base teacher의 v3_longer는 2k 이후에도 좋아졌습니다. arm 0의 125 / 250
  렌더와 |ΔW|로 판단합니다. 평평하지 않으면 arm 1 전에 A/B 길이를 늘립니다. control 렌더에서 EMA 125 > EMA 250으로 보인
  것은 반대 방향 신호이므로 twin을 보고 판단합니다.
- **앵커 없는 붕괴의 원인.** 다음 분리 실험은 스텝 0 DM off + 다양성 가중 0(스텝 0을 init에 고정, 앵커 없음)과
  TTUR 10:1입니다.
- **burst의 빈도와 원인.** 250 iteration에 약 한 번이고 σ_a tail과는 무관합니다. adaptive clip이 해결하는지는 arm 3에서
  봅니다.
- **student가 teacher의 구도를 따라가지 못하는 문제.** 아래에서 본 점프나 슬랩스틱 같은 구도입니다. 앵커 σ 지점(6/12 = 0.5)이나
  다양성 가중 쪽 레버는 이 레시피에서 다시 측정하지 않았습니다.
- **2스텝에서 제안만 되고 만들지 않은 것:** rollout-grad 스케일 λ, 4스텝 m14 체크포인트에서 warm start,
  저역 통과한 최종 이미지 앵커, 스텝 0에 σ-routed low-res 적용.

---

## 참고 논문

DP-DMD arXiv:2602.03139, CDM arXiv:2605.06376, MFD arXiv:2606.11155, VA-TDM arXiv:2611.06796, APEX arXiv:2604.12322,
DMAD arXiv:2610.02188, data-space iteration arXiv:2610.12102, IDM arXiv:2610.39633, OPD² arXiv:2607.15161
