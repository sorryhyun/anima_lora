# 부록 1. Anima LoRA 학습 레시피의 근거

`01_lora_recipe.md` §3의 설정을 고른 근거와 측정입니다. 설정은 블라인드 A/B 14회(s29–s42)로 정했고, 런은 loss로
평가하지 않고 렌더를 블라인드로 비교해서 평가했습니다. 점수는 두 arm이 이긴 횟수입니다.

| 항목 | 값 | 근거 |
|---|---|---|
| batch | bs 1, GA 2 | free-fit은 batch 1이 기본. micro-batch마다 clip하므로 스파이크 하나가 update에 주는 영향은 cap / 2 이하 |
| clip | σ 구간별 median clip | 구간별 clip이 단일 window를 39–21로 이김(s30, p 0.047) |
| lr / 스케줄 | 5e-5, constant | 이전 설정(2e-5 cosine, 고정 clip 1.0)을 41–16으로 이김(s29). 단 ‖dW‖를 맞춘 재실험(s33)에서는 이전 설정이 52–42로 앞섬. 1e-4는 손이 엉킴 |
| adapter EMA | 0.1 | 일정한 lr에서 raw 가중치에 남는 말풍선 같은 잔여물을 없앰 |
| rank / alpha | 32 / 128 | rank 16은 null(s34 28–20, s41 17–15), alpha 512 / lr 1.25e-5도 null(s35 25–22, s36 18–13, s42 23–22) |
| σ 샘플링 | shift 1 | shift 2와 무승부(s32 22–23). REPA를 끄면 shift 1이 28–20(s39) |
| REPA | off | alpha 512에서는 off가 32–16(s37+s38, pair p 0.015). alpha 128에서는 23–22 무승부(s40) |
| T-LoRA mask | off | 학습과 추론 모두 mask를 건 경우 plain과 11–13(s31) |
| down init | W₀의 상위 singular vector | Kaiming과 렌더 차이 없음(A1.4) |

## A1.1 batch와 clip: gradient tail은 σ 양 끝에 몰린다

bs1 / 5e-5, 848 update에서 측정했습니다. epoch별 평균 pre-clip norm은 2.20에서 약 0.95로 떨어졌고(전체
중앙값 1.01), 스파이크는 12.8까지 갔지만 loss와의 상관은 약했습니다(r = 0.27). 고정 cap 1.0은 중앙값 바로 위에
있어서 epoch 1에서는 update의 94%를, epoch 5–8에서는 22–29%를 잘랐습니다. 스파이크를 잡는다기보다 update 절반을
정규화한 셈입니다.

tail은 σ가 만듭니다(caststation / ama mitsuki / sweetonedollar, seed 42).

| σ | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---:|---:|---:|---:|---:|
| 샘플 비율 (합산) | 6.7 % | 23.5 % | 34.3 % | 28.0 % | 7.4 % |
| pre-clip norm 중앙값 | 1.6–2.7 | 0.9–1.4 | 0.9–1.3 | 1.4–2.2 | 2.5–4.9 |
| 전역 median clip에서 잘린 비율 | 33 % | 2.6 % | 1.1 % | 9.8 % | 62 % |

σ 양 끝(σ < 0.2 또는 ≥ 0.8)은 샘플의 13–16%인데 잘린 micro-batch의 50–80%를 차지합니다. 이미지 하나가 지배하는
경우는 없었습니다(8 epoch 동안 이미지당 많아야 2–4회 clip). σ 전체에 median 하나를 쓰면 σ 끝의 정상 샘플을
스파이크로 잘못 보기 때문에, σ 구간마다 window를 따로 둡니다. 이 분포 모양은 T-LoRA mask 유무, rank 16, REPA off
조건에서도 그대로였습니다.

- **clip은 micro-batch마다 겁니다.** 스파이크를 평소 norm의 `k`배로 줄이되 방향은 유지하므로 그 이미지도
  학습됩니다. accumulate한 뒤에 자르면 늦는데, norm 10짜리 micro-batch가 norm 1짜리 옆에서 방향을 정해 버리기
  때문입니다. 자르지 않은 스파이크는 Adam의 1차 모멘트를 약 10 update 동안 끌고, 2차 모멘트를 수백 update 동안
  부풀립니다.
- **GA 2로 충분합니다.** GA 4로 가면 update 수가 반으로 줍니다. 30장 세트면 60 update이고, 여기서 EMA 0.1의 span은
  6입니다.
- 처음 10개 norm이 쌓일 때까지는 고정 cap 1.0을 씁니다. clip 비율을 셀 때 이 구간은 뺍니다.

## A1.2 lr, 스케줄, EMA

- **EMA에는 일정한 lr이 필요합니다.** cosine 감쇠에서는 마지막 약 10%의 update가 거의 움직이지 않아서 EMA 0.1이
  raw 가중치와 같아집니다(ama mitsuki: ‖dW‖ 0.997배, cos 1.000). constant 5e-5에서는 0.91배 / cos 0.974이고, raw에
  남는 말풍선이나 엉뚱한 "?" 같은 잔여물을 없애 줍니다.
- EMA 감쇠는 `1 - 1/(window × updates)`로, 848 update에서 0.988입니다. EMA는 lr warmup이 끝난 시점부터 fp32 사본으로
  시작하고, 저장할 때는 live 가중치 대신 EMA를 저장합니다.
- **lr은 변위를 정하고, 변위에는 상한이 있습니다.** GA2 / 1e-4(bs1 EMA ‖dW‖의 1.41배)에서는 포옹과 메이드
  프롬프트에서 팔과 손이 엉키고, GA2 / 5e-5(0.68배)는 깨끗합니다. 이 결과는 블라인드 A/B 없이 렌더를 보고
  읽었습니다.
- s29의 승리(41–16)는 T-LoRA mask를 켠 상태였고 ‖dW‖도 맞추지 않았습니다. mask를 끄고 ‖dW‖를 맞춘
  재실험(s33)에서는 이전 설정 A가 52–42로 앞섰습니다(cluster p 0.17). 그러니 B의 이점은 update 방향에 있지
  않습니다.

## A1.3 rank와 alpha: 손대지 않는다

LoRA는 update를 `(α/r)·BA`로 스케일합니다(Hu et al. 2021, arXiv 2106.09685). rsLoRA(Kalajdzievski 2023, arXiv
2312.03732)는 α/r을 고정하면 rank에 따라 학습 보폭이 달라진다는 점을 보이고 `α/√r` 스케일을 제안했는데, 이
레시피에서도 같았습니다. α/r을 고정하면 Adam에서 ‖dW‖가 √r에 비례해서 r16 / alpha 64는 r32의 0.72배만
움직였고(s34), alpha를 √r에 비례시키면(r16 / a362 vs r32 / a512) 보폭이 맞았습니다(0.95배 / 1.09배, s41).

alpha와 lr의 배분은 arXiv 2606.12883을 따라 정했습니다. 이 논문은 LoRA 한 step을 task 방향 신호(Θ(αη))와
bilinear drift `s·ΔB·ΔA`(Θ(αη²))로 나누므로, α·η를 고정하고 alpha를 높이고 lr을 낮추면 drift가 줄어듭니다. α·η를
출시값에 고정하고 배분만 바꾸면 ‖dW‖는 같았고(a512가 a128의 0.96–0.99배), drift 비율 `‖B·D‖ / ‖B·A0‖`(모듈
중앙값)는 a128의 0.15–0.17에서 a512의 0.06으로 줄었습니다. 그래도 블라인드 세 세트(s35, s36, s42)는 두 arm을
구별하지 못했습니다. rank 16도 세 작가에서 REPA 유무와 관계없이 null이었습니다(s34, s41).

결론적으로 rank 32 / alpha 128 / lr 5e-5를 유지합니다. rank를 바꿀 때는 rsLoRA대로 alpha를 √r 비율로 맞춥니다.

## A1.4 down init

down 행렬을 Kaiming 대신 W₀의 상위 r개 right singular vector로 시작합니다(`A₀ = V_rᵀ/√3`, `B₀ = 0`). `1/√3`은
Kaiming의 기대 row-norm에 맞춘 값이라 바뀌는 것은 방향뿐이고, 초기 ΔW는 0입니다. W₀의 주성분으로 adapter를
시작하는 것은 PiSSA(Meng et al. 2024, arXiv 2404.02948)와 같은 발상이지만, 여기서는 B₀ = 0이라 W₀를 바꾸지
않습니다.

첫 step의 gradient 에너지 중 이 init이 담는 비율은 0.21, 작가 자신의 gradient basis는 0.74입니다. 그런데 256
step(r32 / α128, 8 epoch)을 학습하고 나면 어떤 init도 렌더를 바꾸지 못합니다. SVD init vs Kaiming 12–9, vs gradient
SVD 11–9, vs 작가 basis 10–10이고, 모두 seed 쌍 잡음 바닥(15–9) 안입니다(블라인드 s22–s25). 렌더 차이가 없는데도
SVD init을 쓰는 이유는 soup에 있습니다. 그룹마다 서로 다른 singular vector 구간에서 시작시키면 병합할 때 입력
부분공간이 겹치지 않습니다(→ `02_soup_merge.md`).
