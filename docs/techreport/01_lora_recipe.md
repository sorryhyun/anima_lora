# 1. Anima LoRA 학습 레시피

> **요약.** 터보 파이프라인의 첫 단계는 작가 하나에 plain LoRA 하나를 만드는 일이다. 이 LoRA들이 다음 단계의
> soup 재료가 된다. TE는 512 토큰까지 max-padding되고, 리사이즈는 free-fit 버킷으로 한다. 레시피는 블라인드
> A/B 14회(s29–s42)로 굳힌 것으로, rank 32 / alpha 128 / lr 5e-5 constant, GA2, σ 구간별 median clip, adapter
> EMA 0.1, shift 1이며 REPA와 T-LoRA mask는 끈다. rank, alpha, shift, REPA, down init은 모두 결과 차이가
> 없었다(null). 런을 평가할 때 loss 수치는 쓰지 않는다. 블라인드로 렌더를 비교해서 평가한다.

이 문서가 다루는 범위는 Anima 라인뿐이다. MOSS-Image와 Qwen-Image-2.1은 불변식이 다른 별개 라인이므로 여기서
다루지 않는다.

---

## 0. 파이프라인에서 이 단계의 위치

| 단계 | 산출물 | 문서 |
|---|---|---|
| **1. LoRA 학습** | 작가별 plain LoRA | 이 문서 |
| 2. soup / merge | 여러 LoRA를 하나로 (concat_exact, merged14) | → `02_soup_merge.md` |
| 3. bake / int8 / GPTQ | soup를 DiT에 굽고, 그 DiT 전용 GPTQ 파일 생성 | → `03_bake_int8_gptq.md` |
| 4. 터보 증류 | DP-DMD 4-step student LoRA | → `04_turbo_distillation.md` |
| 5. 실전 팁 | fp32 stream, daemon, compile, VRAM | → `05_practical_tips.md` |

이 단계에서 어댑터 종류를 고를 때 따져야 할 제약은 하나다. soup 단계는 **plain LoRA 키 구조만 받는다**.
Hydra는 거부되고, T-LoRA는 mask가 학습에만 쓰이기 때문에 저장된 체크포인트가 plain 구조라서 받아진다(§4.1).

---

## 1. 데이터와 캡션

### 1.1 캡션 문법

태그 순서는 `[meta] [character] [series] [artist] [general]`이고 작가 태그는 `@` 접두사를 붙인다.

```
absurdres, safe, 1girl, chitanda eru, hyouka, @channel (caststation), full body, serafuku, She is saying hi.
```

캡션에는 태그 묶음 뒤로 절이 더 붙을 수 있다.

- **위치 절**: `<tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.` 절과 절은 마침표로
  나누고, 한 절 안의 태그는 콤마로 나눈다.
- **OCR 텍스트 절**: `… Japanese text reads as "…"`. 그릴 텍스트는 영어 문장 안의 큰따옴표에 넣는다.

### 1.2 TE 출력과 캡션 변형

1. **max-padding.** Anima는 TE 출력을 최대 길이까지 패딩한 상태를 전제로 한다. 0으로 패딩된 위치는
   cross-attention에서 attention sink 역할을 한다. 텍스트 길이에 맞춰 잘라 내면 **검은 이미지**가 나온다.
   학습과 추론 모두 최대 길이까지 패딩하고, 패딩을 마스킹하지 않는다. 패딩 길이는 Qwen3와 T5 쪽 모두 512다.
2. **캡션 변형.** 캡션마다 셔플 변형 4개를 만들고(tag dropout 0.1), 학습은 이 변형을 쓴다. v0은 셔플하지 않은
   원본이고, 학습 때는 v0을 20%, v1..v3을 80% 확률로 뽑는다. 셔플은 `@artist` 태그와 절 표지(`On the ...`)를
   고정한 채 절 안의 태그만 돌린다. 이와 별개로 캡션 전체를 비우는 caption dropout 0.1이 있다.

---

## 2. Free-fit 버킷

Anima가 쓰는 리사이즈 방식은 free-fit뿐이다. 이미지마다 **원래 종횡비를 유지**한 채, 패치 그리드의 토큰
수가 자기 tier의 band 안 어디든 들어가도록 크기를 정한다. 그래서 크롭되는 부분은 16px 미만 수준으로 거의
없다.

| tier edge | 토큰 band (natural) |
|---|---|
| 512 | 1008–1024 |
| 768 | 2160 |
| 896 | 3000–3024 |
| 1024 | 4032–4200 (고정) |
| 1280 | 6300 |
| 1536 | 8640 |

1024가 아닌 tier는 ±2.5%만큼 band를 넓혀 종횡비를 맞출 여유를 둔다. 종횡비는 4.0에서 잘린다.

tier는 1024와 896을 쓰고, 이미지마다 리사이즈 폭이 가장 작은 tier로 들어간다. tier는
`|log(nominal_tokens / native_tokens)|`를 최소화하도록 고르므로, 0.95MP 이미지는 768로 줄지 않고 1024에 남는다.

리사이즈에는 최소 해상도 하한이 없다. 이미지는 원본 비율 그대로 free-fit band에 맞게 줄어들거나 늘어나고,
빠지는 것은 큐레이션에서 제외한 이미지뿐이다.

---

## 3. 확정 레시피와 그 근거

### 3.1 레시피

| 항목 | 값 | 근거 |
|---|---|---|
| rank / alpha | 32 / 128 | rank 16은 null(s34 28–20, s41 17–15), alpha 512 / lr 1.25e-5도 null(s35 25–22, s36 18–13, s42 23–22) |
| lr / 스케줄 | 5e-5, constant (warmup 5%) | 옛 기본값(2e-5 cosine bs1 clip 1.0)을 41–16으로 이김(s29). 단 ‖dW‖를 맞춘 재실험(s33)에서는 A가 52–42로 앞섬. 1e-4는 손이 엉킴 |
| batch | bs 1, GA 2 | per-micro clip 아래에서 스파이크 하나가 갖는 몫은 update의 `cap / accum` 이하 |
| clip | σ 구간별 median clip, k 2.0, micro-batch마다, 구간 경계 0.2 / 0.4 / 0.6 / 0.8, window 50 | 구간별 clip이 단일 window를 39–21로 이김(s30, p 0.047) |
| adapter EMA | 0.1 | 일정한 lr에서 raw 가중치에 남는 말풍선 같은 잔여물을 없앤다 |
| σ 샘플링 | logit-normal(0, 1), shift 1 | shift 2와 무승부(s32 22–23). REPA를 끄면 shift 1이 28–20(s39) |
| REPA | off | alpha 512에서는 off가 32–16(s37+s38, pair p 0.015). alpha 128에서는 23–22 무승부(s40) |
| T-LoRA mask | off | 논문대로 학습과 추론 모두 mask를 건 경우 plain과 11–13(s31) |
| down init | `weight_svd` | 렌더 차이 없음(§4.2). `weight_svd_nodc` + floor 0.01도 25–23 null(s43) |
| 기타 | AdamW, 8 epoch, caption dropout 0.1, fp32 stream | 이 A/B 라인 이전에 정해진 값 |

**무승부의 뜻.** 이 레시피에서 대부분의 손잡이보다 학습 seed가 결과를 더 크게 흔든다. (작가, seed) 묶음
하나만 봐도 어느 A/B에서든 6–2로 한쪽에 쏠린다. 무승부일 때는 더 단순하거나 효율적인 쪽을 고르고, 차이가 없으면
원래 출시된 쪽을 유지한다.

### 3.2 어떻게 평가했나 (프로토콜)

- arm마다 학습 seed를 여러 개(42–44, 때로 45) 쓰고, 렌더 seed는 2개나 4개, 프롬프트는 4개(교실/교복, 포옹,
  메이드/트레이/카페, 비키니/해변)다. 렌더 설정은 28 steps euler, cfg 4, shift 3, 1024²다. 채점자 한 명이
  블라인드 쌍으로 채점했다.
- **‖dW‖ 맞추기(s30부터).** treatment의 `lora_up`을 다시 스케일해서 모듈별 `‖(alpha/rank)·up@down‖`의 중앙값을
  같은 (작가, seed)의 control에 맞춘다. 이렇게 하면 방향만 비교하게 된다. s29는 이 보정 없이
  돌았고, B가 2.3–2.5배 멀리 움직인 효과와 섞였다.
- **평가 통계.** (작가, seed) 묶음 단위로 sign test를 하고, 쌍 단위 이항검정은 옆에 참고로 적는다. 학습 seed가
  2개면 묶음 단위 p의 하한이 0.25, 3개면 0.125다.
- 데이터셋: caststation(30장, GA2에서 약 120 update), sweetonedollar(31장, 약 124), ama mitsuki(106장, 424),
  sincos(354장, 약 1416).

### 3.3 clip: 그래디언트 tail은 σ 양 끝에 몰린다

ama mitsuki, bs1 / 5e-5, 848 update에서 측정한 결과다. epoch별 평균 pre-clip norm은 2.20에서 약 0.95로
떨어졌다(전체 중앙값 1.01). 스파이크는 12.8까지 갔고 loss와의 상관은 약했다(r = 0.27). 고정 cap 1.0은
중앙값 바로 위에 있어서, epoch 1에서는 update의 94%를, epoch 5–8에서는 22–29%를 잘랐다. 스파이크를 잡는다기보다
update 절반을 정규화한 셈이다.

tail은 σ가 만든다(caststation / ama mitsuki / sweetonedollar, seed 42).

| σ | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---:|---:|---:|---:|---:|
| 샘플 비율 (합산) | 6.7 % | 23.5 % | 34.3 % | 28.0 % | 7.4 % |
| pre-clip norm 중앙값 | 1.6–2.7 | 0.9–1.4 | 0.9–1.3 | 1.4–2.2 | 2.5–4.9 |
| 전역 median clip에서 잘린 비율 | 33 % | 2.6 % | 1.1 % | 9.8 % | 62 % |

σ 양 끝(σ < 0.2 또는 ≥ 0.8)은 샘플의 13–16%인데 잘린 micro-batch의 50–80%를 차지한다. 이미지 하나가 지배하는
경우는 없었다(8 epoch 동안 이미지당 많아야 2–4회 clip). σ 전체에서 median 하나를 쓰면 σ 끝의 정상 샘플을
스파이크로 오판한다. 그래서 σ 구간마다 window를 따로 둔다. 이 분포 모양은 T-LoRA mask 유무, rank 16, REPA off
조건에서도 그대로였다.

레시피에 반영한 결정:

- **median 기준 clip을 micro-batch마다 건다.** 스파이크를 `k ×` 평소 norm으로 줄이되 방향은 유지하므로 그
  이미지도 학습된다. 평균을 낸 뒤에 자르면 이미 늦다. norm 10짜리 micro-batch가 norm 1짜리 옆에서 방향을
  정해 버리기 때문이다. 자르지 않은 스파이크는 Adam의 1차 모멘트를 약 10 update 동안 끌고, 2차 모멘트를
  수백 update 동안 부풀린다.
- **GA2로 충분하다.** GA4로 가면 update 수가 반으로 준다. 30장 세트면 60 update인데, 여기서 EMA 0.1의 span은
  6이다.
- 처음 10개 norm이 쌓일 때까지는 고정 cap 1.0이 걸리므로, clip 비율을 셀 때 이 구간은 뺀다.

### 3.4 lr, 스케줄, EMA

- **EMA에는 일정한 lr이 필요하다.** cosine 감쇠에서는 마지막 약 10%의 update가 거의 움직이지 않아서 EMA 0.1이
  raw 가중치와 같아진다(ama mitsuki: ‖dW‖ 0.997배, cos 1.000). constant 5e-5에서는 0.91배 / cos 0.974이고,
  raw에 남는 말풍선이나 엉뚱한 "?" 같은 잔여물을 없앤다.
- EMA의 감쇠는 `1 - 1/(window × updates)`다. 848 update에서 0.988이다. EMA는 lr warmup이 끝난 시점부터 fp32
  shadow로 시작하고, 저장할 때는 live 가중치 대신 **EMA를 저장한다.**
- **lr은 변위를 정하고, 변위에는 상한이 있다.** GA2 / 1e-4(bs1 EMA ‖dW‖의 1.41배)에서는 포옹과 메이드
  프롬프트에서 팔과 손이 엉킨다. GA2 / 5e-5(0.68배)는 깨끗하다. 이 결과는 블라인드 A/B 없이 렌더를 보고
  읽었다.
- s29의 승리(41–16)는 T-LoRA mask를 켠 상태였고, ‖dW‖도 맞추지 않았다. mask를 끄고 ‖dW‖를 맞춘 재실험(s33)에서는
  옛 레시피 A가 52–42로 앞섰다(cluster p 0.17). 그러니 B의 이점은 update 방향이 아니다. "B를 자기 ‖dW‖ 그대로
  두고 A와 비교"하는 실험은 아직 남아 있다(§3.8).

### 3.5 rank와 alpha: 손대지 않는다

arXiv 2606.12883은 LoRA 한 step을 task 방향 신호 `s·(BΔA + ΔB·A)`(Θ(αη))와 bilinear drift `s·ΔB·ΔA`(Θ(αη²))로
나눈다. α·η가 같으면 1차 update는 같고 drift는 1/η에 비례해 줄어든다. 그래서 "alpha를 높이고 lr을 낮추면 더
깨끗하다"는 예측이 나온다. 실험은 α·η를 출시값에 고정하고 둘의 배분만 바꿨다.

- **α·η가 같으면 ‖dW‖도 같다.** a512는 a128의 0.96–0.99배였다(30장). sincos(약 1416 update)에서는 0.97배였다.
- **alpha/rank를 고정하면 Adam에서 ‖dW‖는 √r에 비례한다.** r16 / alpha 64는 r32의 0.72배만 움직였다(s34).
  alpha를 √r에 비례시키면(r16 / a362 vs r32 / a512) 학습 중 보폭이 맞았다(0.95배 / 1.09배, s41).
- **drift는 실제로 있지만 렌더에는 나타나지 않는다.** 모듈 중앙값 기준 `‖B·D‖ / ‖B·A0‖`는 a128에서 0.15–0.17,
  a512에서 0.06이었다. 블라인드 다섯 세트(s35, s36, s42; rank는 s34, s41)가 두 arm을 구별하지 못했다.
- rank는 세 작가에서 REPA 유무와 관계없이 null이었다. rank 16으로 얻는 것은 파일 크기가 절반이 된다는 점뿐이다.

결론: **rank 32 / alpha 128 / lr 5e-5를 유지한다.** rank를 바꿔야 한다면 alpha를 √r 비율로 맞춘다.
adaln 브랜치는 이미 alpha를 이렇게 계산한다(§4.3).

### 3.6 σ shift: 1을 유지한다

추론 중 σ 구간 `[LO, HI)` 밖에서 LoRA를 끄고, sweetonedollar, 네 프롬프트, 평균 절대 픽셀 차(0–255)로 측정했다.

| 조건 | vs LoRA 전 구간 on | vs base |
|---|---:|---:|
| static merge ×1.0 (hook 경로의 잡음 바닥) | 1.6–2.0 | — |
| σ < 0.4에서 off (28 step 중 5) | 2.4–2.6 | — |
| σ ≥ 0.8에서 off (28 step 중 12) | 48–58 | **8.5–8.9** |
| LoRA 전 구간 on | — | 50–59 |

σ ≥ 0.8에서 LoRA를 끄면 렌더가 base와 거의 같아진다. 반대로 σ < 0.4에서 끄면 잡음 바닥 이상의 변화가 없다.
그러니 LoRA가 일하는 곳은 높은 σ다. shift 1에서 이 구간은 학습 샘플의 7.4%(런당 약 18 micro-batch)만 받는다.
그래서 shift를 올려 보는 것이 자연스러운 시도였지만, shift 2는 무승부였다(s32 22–23). REPA를 끄자
caststation은 동률, sweetonedollar는 shift 1 쪽으로 남았다(s39 28–20). shift 1을 유지한다.

### 3.7 REPA와 T-LoRA mask: 끈다

- **REPA**(relational, weight 0.05, layer 8, PE-Spatial). 손해 여부가 step 크기에 따라 달랐다. a512 / lr 1.25e-5에서는
  16–32로 졌고(s37+s38), a128 / lr 5e-5에서는 23–22로 비겼다(s40). 그래서 출시 step에서 REPA를 켜고 학습한
  LoRA가 손해를 본 것은 측정되지 않는다. **loss로는 REPA를 걸러낼 수 없다.** REPA 항을 빼면 두 arm의 fm loss가
  같았고(caststation s42: 0.0731 vs 0.0723), grad norm과 ‖dW‖도 거의 그대로였다. REPA는 fm 적합도는 그대로 두고
  렌더만 바꾼다. EasyControl 설정에서는 계속 켜 둔다(cond ≠ target인 다른 regime이라 여기서 확인하지 않았다).
- **T-LoRA mask.** 2026-11-05까지 출시본은 학습에만 mask를 걸고 추론은 full rank로 했다. 학습한 모델과 렌더하는
  모델이 서로 달랐던 셈이다. 논문대로 추론에도 mask를 걸면 plain과 11–13으로 비겼다(s31, sweetonedollar 한 명).
  추론 쪽 비용만 늘고 얻은 것이 없어서 기본값에서 뺐다.

### 3.8 남은 문제

- B를 자기 ‖dW‖ 그대로 두고 A와 비교하기(둘 다 mask 없이). B가 이기면 지렛대는 clip이 아니라 보폭(lr / 스케줄)이다.
- 긴 런(sincos, 약 1416 update)에서의 REPA. 지금까지 확인한 세트는 모두 30–106장이었다.
- T-LoRA, Hydra, soup은 REPA off를 따로 확인하지 않고 물려받았다.
- σ 구간별 clip은 두 작가, p 0.047로 출시되었다.
- sweetonedollar는 ×1.0에서 기형적으로 큰 손을 그리고, ×0.8 / ×0.6에서는 손이 줄어든다. 손과 스타일을 무엇이
  가르는지(σ-gated LoRA, 렌더 종횡비, 데이터)는 확인하지 않았다.

---

## 4. 어댑터 선택

### 4.1 plain LoRA, T-LoRA, Hydra

| 변형 | soup(`02`) | merge / bake(`03`) | ComfyUI 기본 LoraLoader |
|---|---|---|---|
| **plain LoRA** | 가능 | 가능 | 가능 |
| T-LoRA | 가능 (mask는 학습 전용이라 키 구조가 plain) | 가능 | 가능 |
| HydraLoRA (shared A, σ router) | **거부** | router-live는 불가 | baked-down 파일만 가능. router 키는 버려짐 |

이 파이프라인의 재료로는 **plain LoRA**를 쓴다. T-LoRA는 §3.7에서 본 대로 이득이 측정되지 않았고, Hydra는 soup
단계를 통과하지 못한다.

### 4.2 `weight_svd` down init

`lora_down`을 Kaiming 대신 `W₀`의 상위 r개 right singular vector로 시작한다. 정확히는 `A₀ = V_rᵀ/√3`,
`B₀ = 0`이다. `1/√3`은 Kaiming의 기대 row-norm에 맞춘 값이라 바뀌는 것은 방향뿐이다. 초기 ΔW는
0이다. 저장되는 키, 병합, 추론 경로는 plain LoRA와 같다.

측정된 사실:

- 첫 step의 gradient 에너지 중 `weight_svd`가 담는 비율은 0.21, 작가 자신의 gradient basis는 0.74다. 그런데
  256 step(`aak`, r32 / α128, 8 epoch)을 학습하고 나면 **어떤 init도 렌더를 바꾸지 못한다.** weight_svd vs
  kaiming 12–9, vs grad_svd 11–9, vs basis_file 10–10이고 모두 seed 쌍 잡음 바닥(15–9) 안이다(블라인드 s22–s25).
- `weight_svd_nodc` + channel scaling floor 0.01 vs `weight_svd`는 25–23 null이었다(s43). 기본값은 바꾸지 않았다.

렌더에서 차이가 없는데도 `weight_svd`를 쓰는 실질적 이유는 soup 쪽에 있다. svd slice k는 `lora_down`을
`[k·r, (k+1)·r)` 번째 singular vector에서 시작시킨다. slice끼리는 서로 직교하므로, 작가마다 다른 k를 주면 병합할
때 입력 부분공간이 겹치지 않는다. r=32에서는 256행짜리 adaln `.1` Linear가 상한이 되어 k ≤ 7, 즉 작가 8명까지다.
adaln 모듈은 slice 설정과 관계없이 항상 top-r로 시작한다. slice 배치와 병합 방식은 → `02_soup_merge.md`.

### 4.3 adaln LoRA

Anima 블록마다 있는 세 modulation MLP(self-attn, cross-attn, mlp용, 2048→256→6144)는 timestep embedding만 입력으로
받는다. 그래서 여기에 붙는 LoRA는 σ에 따른 전역 톤, 대비, gate만 배울 수 있고 텍스트나 공간 정보는 배울 수 없다.

- 모든 LoRA 계열 학습에 rank 16 adaln LoRA가 함께 붙는다.
- adaln alpha는 √r 비율에 따라 `alpha · sqrt(adaln_rank / rank)`으로 계산된다. 출시 레시피에서는 128 · √(16/32) ≈ 90.5다.
- rank 16의 근거: 학습된 adaln ΔW는 유효 rank가 낮다. 공식 turboV10의 full-rank delta도 r16에서 에너지의 93%가
  담긴다. soup s1001 스타일 LoRA는 r16에서 89%다.
- soup에서는 shard별 adaln delta를 합치지 않고 평균 낸다. → `02_soup_merge.md`.

### 4.4 Channel scaling (α = 0.5)

SmoothQuant 방식의 재매개화다. 보정된 채널별 스케일 `s[c] = (mean|x[c]|)^α`를 `lora_down` 열에 흡수시키고
forward에서는 `x / s`를 쓴다. 초기 출력은 같고, 달라지는 것은 Adam의 채널별 보폭뿐이다. α = 0.5에서 채널
지배도(max/mean leverage)는 약 13–80배에서 3.8–9.3배로 내려간다.

- 저장할 때 `inv_scale`을 `lora_down`에 접어 넣으므로, 디스크의 체크포인트는 그냥 plain LoRA다. 추론에는 영향이
  없다.
- **일반 FM LoRA에서 α 0.5 vs 0을 비교한 블라인드 A/B는 없다.** 관련된 측정은 하나다. 거의 죽은 채널은
  `lora_down` 열이 random walk를 하는데, 676-update soup shard에서 이 열들이 저장된 ‖ΔW‖²의 33–43%를 차지했다.
  이것이 ‖ΔW‖를 읽는 도구를 왜곡한다. floor 0.01을 걸면 이 비중이 0이 되지만, 렌더 A/B(s43)에서는 null이었고
  기본값은 0(off)이다.
- 터보 증류에서는 α 0.5가 확실히 손해였고 기본값이 0.0이다. → `04_turbo_distillation.md`.

---

## 5. 런 평가

### 5.1 loss가 낮다고 좋은 런이 아니다

- REPA 실험이 그 예다. REPA를 켠 arm과 끈 arm은 fm loss가 같았는데(0.0731 vs 0.0723) 렌더는 블라인드에서
  16–32로 갈렸다(§3.7). loss는 렌더 차이를 보지 못한다.
- per-σ FM-MSE 검증 loss는 Anima에서 샘플 품질과 상관관계를 보이지 않았다. 검증을 켠다면 PE-Core 기반 paired
  CMMD²를 쓴다.
- 다만 CMMD도 만능은 아니다. down init E1 실험에서는 모든 arm의 CMMD가 real-vs-real 바닥(0.61) 아래였고, 그래서
  순위를 매길 수 없었다.
- 마지막 평가는 렌더로 한다. 레시피 A/B와 같은 방식이 기준이다. 학습 seed를 2–3개, 렌더 seed를 2개 이상 쓰고,
  28 step, cfg 4, shift 3, 1024²로 렌더해서 블라인드 쌍으로 비교하며, ‖dW‖를 맞춘다(§3.2).

### 5.2 평가할 때 빠지기 쉬운 함정

- **arm 간 끝점 ΔW cosine에는 신호가 없다.** 같은 설정을 두 번 돌려도 dW cosine 중앙값이 0.146(bs1 5e-5)이다.
  반면 변위(‖dW‖) 자체는 ±3% 안에서 재현된다.
- **런 하나의 렌더 하나는 그 설정의 성질이 아니다.** bs1 쌍둥이 런은 말풍선은 재현했지만 첫 런의 과포화 마젠타는
  재현하지 않았다. 학습 seed를 바꿔도 반복되는 결함만 그 설정의 것으로 본다.
- **seed가 같아도 쌍 비교가 아니다.** 두 arm이 같은 σ를 뽑더라도 clip이 한 번 다르게 걸리는 순간 갈라진다.
- **cross-seed MAD로 사전 선별하지 않는다.** s32–s35에서는 6건 중 5건이 맞았지만 s39–s42에서는 맞지 않았다.

### 5.3 학습 규모 감 잡기

update 수는 이미지 수 × epoch / GA다. 30장 × 8 epoch / 2 = 120으로, caststation 수치(약 120)와 같다. 레시피는
30–354장 범위에서 확인했다. 이보다 훨씬 큰 세트에서 REPA나 rank가 다르게 작동하는지는 아직 불확실하다(§3.8).

> TODO(확인 필요): 5070 Ti에서 30장 / 8 epoch 기준 런 하나의 소요 시간과 피크 VRAM.

---

## 6. 다음 단계

- LoRA는 depth에 묶여 있다. 40-block Anima-2.9B에서 학습한 LoRA를 28-block base에 올리면 뒤쪽 블록의 가중치가 버려진다.
- 다음 단계: 작가별 LoRA 여러 개를 soup나 merge로 묶는다. → `02_soup_merge.md`. soup는 별도 설정(lr 2e-5,
  4 epoch, uncond-init)을 쓰므로 이 문서의 레시피와 다르다. 차이는 02에서 다룬다.
