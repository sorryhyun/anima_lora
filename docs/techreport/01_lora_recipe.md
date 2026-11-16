# 1. Anima LoRA 학습 레시피

## 1. 캡션은 상세할수록 좋다

태그만 나열한 캡션은 어떤 속성이 어떤 대상에 붙는지 표현하지 못합니다. `3girls, blonde hair, aqua hair, red hair`
에서는 누가 금발인지 알 수 없고, 한 캐릭터의 의상 네 벌을 나란히 그린 `multiple views` 이미지에서는 어느
의상이 어느 자리인지 알 수 없습니다. 학습 데이터에도 이런 캡션이 적지 않았는데, 여러 명이 나오는 캡션이 258개,
`multiple views` 캡션이 약 350개였습니다.

그래서 캡션 뒤에 위치 절을 붙입니다.

```
<태그 묶음>. On the left, akita neru, yellow eyes. On the right, kasane teto.
```

학습하기 어렵거나 결함이 있어 보이는 이미지도 같은 방식으로 다룰 수 있습니다. MOSS-Image 팀에 따르면 워터마크,
흐림, 압축 손상, 다량의 텍스트가 있는 데이터는 제외하지 않고 캡션에 그대로 적는 것이 모델 성능에 도움이 된다고
합니다([MOSS-Image 기술 보고서, arXiv 2610.23374, App. B.2](https://arxiv.org/abs/2610.23374)). 다만 캡션이
이미지와 상이한 경우와 완전히 중복된 경우는 제외합니다.

캡션 변형: robustness를 위해 태그 순서를 섞거나 태그 일부를 드롭아웃(0.1)한 변형을 만들고, 학습 때 무작위로 골라
씁니다. 이와 별개로 캡션 전체를 비우는 caption dropout 0.1을 씁니다. CFG가 쓰는 unconditional 예측을 같은 모델
안에서 학습하는 표준 방법이고, 0.1–0.2가 가장 좋았다는 Ho & Salimans(2022, arXiv 2207.12598)의 값을 따랐습니다.

---

## 2. Free-fit: 원래 종횡비 그대로 학습하기

보통은 compile 효율을 위해 이미지를 정해진 크기 목록에 맞춰 자르는데, 자르면 생각보다 많은 정보를 잃습니다.
그래서 토큰 band로 compile 제약을 풀고, 이미지를 원래 종횡비 그대로 학습합니다.

### 2.1 compile 제약 풀기

DiT 블록이 입력에서 보는 것은 토큰 시퀀스의 길이뿐입니다. RoPE 위치는 블록 밖에서 실제 `(Hp, Wp)` 격자로 만들어
펼친 뒤 넣으므로, 블록 안에서는 4096 토큰이 64×64인지 32×128인지 구별하지 않습니다. 따라서 시퀀스 길이 축만
dynamic으로 표시하고 그 범위를 tier의 토큰 band로 묶으면, band 안의 모든 크기를 graph 하나로 처리할 수 있습니다.
각 forward는 실제 토큰 수로 돌고 padding이 없으며, 고정 크기로 compile한 경로와 같은 결과를 냅니다.

바뀐 부분만 줄이면 다음과 같습니다. 숫자는 1024 tier의 band입니다.

```diff
 # 블록 밖: RoPE를 실제 (Hp, Wp) 격자로 만든 뒤 토큰을 한 줄로 펼칩니다
 rope_cos, rope_sin = build_rope(Hp, Wp)       # 각각 (Hp·Wp, d)
 x = x.flatten(1, 2)                           # (B, Hp, Wp, D) → (B, Hp·Wp, D)

-# 크기마다 graph가 따로 생깁니다
-block = torch.compile(block, dynamic=False)
+compiled = torch.compile(block, dynamic=False)
+lo, hi = 4032, 4200                           # band 안의 토큰 수는 graph 하나로 처리합니다
+
+def block(x, rope_cos, rope_sin, *rest):
+    torch._dynamo.mark_dynamic(x, 1, min=lo, max=hi)
+    torch._dynamo.mark_dynamic(rope_cos, 0, min=lo, max=hi)
+    torch._dynamo.mark_dynamic(rope_sin, 0, min=lo, max=hi)
+    return compiled(x, rope_cos, rope_sin, *rest)
```

`mark_dynamic`은 compile 전에 한 번만 걸지 않고 forward마다 다시 겁니다. gradient checkpointing의 backward
recompute가 `x`를 새 텐서로 떼어 내면서 표시가 사라지는데, RoPE 쪽에는 표시가 남아 있어 둘이 어긋나면 compile이
에러를 냅니다.

### 2.2 크기 정하기

이미지마다 다음 순서로 크기를 정합니다.

1. **tier 고르기.** `|log(tier 토큰 수 / 원본 토큰 수)|`가 가장 작은 tier, 즉 크기를 가장 적게 바꾸는 tier를
   고릅니다. 이 값은 확대와 축소에 대칭이라 0.95MP 이미지는 768로 줄지 않고 1024에 남습니다.
2. **종횡비 제한.** 종횡비를 1:4–4:1로 제한합니다. 이보다 긴 이미지만 이 한도까지 잘립니다. 이전 목록에서 가장
   긴 크기(2016×512, 4:1에 가까움)와 같은 범위라 이전보다 더 자르는 이미지는 없습니다.
3. **격자 찾기.** 토큰 수 N에서 `Hp = round(sqrt(N/a))`, `Wp = round(Hp·a)`로 시작해, 주변 격자 중 토큰 수가
   band 안에 있고 RoPE 축 한도(256 patch) 안에 있으면서 종횡비 오차가 가장 작은 것을 고릅니다.

종횡비 오차는 patch 하나(16px) 미만이고, 늘어나는 비율은 1.6% 미만입니다. 최소 해상도 하한은 없습니다. 작은
이미지는 늘리고 큰 이미지는 줄일 뿐 버리지 않습니다.

1024가 아닌 tier는 band를 ±2.5% 넓혀 종횡비를 맞출 여유를 둡니다. 이 레시피는 1024와 896 tier를 씁니다.
이따금씩 이미지를 축소하여 학습하는 경우가 있는데, 개인 연구 결과 이미지를 줄여서 학습하는 것은 다수의 경우 총
계산량 대비 학습 효과를 유의미하게 늘리지 않습니다([arXiv 2608.04448](https://arxiv.org/abs/2608.04448)).

이때 제약은 샘플마다 토큰 길이가 달라 배치를 구성하기 어렵다는 것으로, 배치 1로도 최대한 GPU 부하가 가게끔
세팅하거나 gradient accumulation으로 update 크기를 키웁니다(부록 A1.1).

---

## 3. 학습 설정

| 항목 | 값 |
|---|---|
| batch | bs 1, GA 2 |
| clip | σ 구간별 median clip, k 2.0, micro-batch마다, 구간 경계 0.2 / 0.4 / 0.6 / 0.8, window 50 |
| lr / 스케줄 | 5e-5, constant (warmup 5%) |
| adapter EMA | 0.1 |
| rank / alpha | 32 / 128 |
| σ 샘플링 | logit-normal(0, 1), shift 1 |
| REPA | off |
| T-LoRA mask | off |
| down init | W₀의 상위 singular vector |
| 기타 | AdamW, 8 epoch, caption dropout 0.1 |

**down init.** LoRA의 down 행렬을 Kaiming 대신 W₀의 상위 r개 right singular vector로 시작하고, up은 0으로 둡니다.
크기는 Kaiming과 같게 맞추므로 바뀌는 것은 방향뿐이고, 초기 ΔW는 0입니다. 학습을 마친 뒤의 렌더는 Kaiming과
구별되지 않았습니다(부록 A1.4). 그래도 이 init을 쓰는 이유는 soup에 있습니다. 작가 그룹마다 서로 다른 singular
vector 구간에서 시작시키면 여러 LoRA를 병합할 때 입력 부분공간이 겹치지 않습니다(→ `02_soup_merge.md`).

**옵티마이저.** AdamW(Loshchilov & Hutter, arXiv 1711.05101)를 씁니다. 다른 옵티마이저는 AdamW에 비해 유의미한
효과를 내지 않았습니다.

나머지 설정을 고른 근거와 측정은 부록(`A1_lora_recipe.md`)에 있습니다.
