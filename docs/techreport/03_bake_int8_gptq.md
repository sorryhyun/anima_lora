# 3. LoRA bake, int8 base, 체크포인트별 GPTQ

실험 모델은 28블록 DiT(Anima)이고, 시간은 모두 RTX 5070 Ti에서 쟀습니다.

## 1. Bake: LoRA를 DiT 가중치에 굽기

모듈마다 **fp32로** `W += multiplier · (up @ down) · scale`을 계산하고, 결과를 bf16으로 저장합니다. 13초 걸립니다.
구운 DiT는 텐서 수, dtype, 키 이름과 shape가 base와 같으므로 base를 받는 곳이면 어디서나 base 대신 쓸 수 있습니다.

가중치 delta가 아닌 어댑터는 구울 수 없습니다. 레이어 router가 입력마다 head를 고르는 어댑터(HydraLoRA 등)는 고정
가중치로 옮길 수 없고, cross-attn KV에 토큰을 끼우는 prefix / postfix나 self-attn 시퀀스에 붙는 register token도
가중치가 아닙니다. plain LoRA와, timestep 마스크를 학습에서만 쓰는 T-LoRA는 bake해도 결과가 같습니다.

---

## 2. int8 base

frozen DiT의 큰 linear를 W8A8 int8 GEMM으로 돌리면 LoRA 학습 step이 1.37배 빨라지고 최종 loss는 같습니다.

| int8 (블록당 6개, 28블록에서 168개) | float으로 남는 것 |
|---|---|
| self-attn qkv (fused), self-attn 출력 | cross-attn k / v (M이 context 길이) |
| cross-attn q, cross-attn 출력 | adaLN (M이 batch) |
| MLP의 두 linear | attention, norm, residual stream, embedder, head |
| | LoRA 자체의 GEMM |

cross-attn k / v와 adaLN은 int8이 오히려 느리게 측정되어 bf16로 둡니다.

activation outlier 때문에 W8A8이 어렵다는 점은 SmoothQuant(Xiao et al. 2022, arXiv 2211.10438)가 짚었고, 회전으로
outlier를 퍼뜨린 뒤 양자화하는 방법이 QuaRot(Ashkboos et al. 2024, arXiv 2404.00456)입니다. 여기서 쓰는 형식은
이를 DiT에 맞게 그룹 단위 Hadamard로 바꾼 ConvRot(Huang et al. 2025, arXiv 2512.03673)입니다. 가중치를 256열
그룹마다 Hadamard 행렬로 회전한 뒤 출력 행마다 스케일 하나로 int8에 저장하고, activation은 토큰마다 int8로
양자화합니다.

forward는 int8 copy로 돌고, LoRA 학습에 필요한 입력 gradient `dX = dY·W`는 bf16 가중치나 `Wᵀ`의 두 번째 int8
copy(backward copy)로 계산합니다. base 가중치의 gradient는 계산하지 않습니다. 생성에서는 LoRA를 가중치에 merge하지
않고 hook으로 붙여 `int8(x, W_q) + B·A·x`를 계산하는데, 이는 학습한 함수와 같습니다(배치별 세부는 부록 A3.2).

---

## 3. 체크포인트별 GPTQ

GPTQ(Frantar et al. 2022, arXiv 2210.17323)로 만든 int8 forward copy는 그 자체가 가중치입니다. base DiT에
calibration한 GPTQ 파일을 LoRA를 구운 DiT에 쓰면 int8 forward는 base를 계산하고, 구운 delta는 빠집니다. 그래서 구운
DiT(여기서는 증류 teacher, → `04_turbo_distillation.md`)에는 그 체크포인트 전용 GPTQ가 필요하고, 만드는 데 약
17분이 걸립니다.

calibration은 작가마다 이미지 2장을 골라 추론 schedule에서 뽑은 σ로 노이즈를 섞고, 자기 캡션과 negative prompt로
한 번씩 돌려 모읍니다. 블록은 순서대로 양자화하므로, 앞쪽 블록의 양자화 오차가 뒤쪽 calibration에
반영됩니다(부록 A3.1).

GPTQ는 weight 양자화 오차의 81%를 없애지만 W8A8 오차는 36%만 줄입니다. 남는 오차는 대부분 activation 항이라,
예측 오차로는 약 10%, CFG velocity 오차로는 2–7%가 줄어듭니다.

---

## 4. fp32 residual stream

block GEMM 바깥을 fp32로 돌립니다. block linear와 attention은 bf16(또는 int8) 그대로이고, 파라미터 dtype은 바뀌지
않으므로 가중치 VRAM도 그대로입니다.

| fp32 | bf16 / int8 그대로 |
|---|---|
| residual stream `x`, modulation, q / k norm, RoPE, embedder, head | block linear(qkv, output proj, cross-attention q, MLP), attention, LoRA 가중치 |

- 예측 오차가 bf16 GEMM 주변에서 1.5–2.1배, int8 base 주변에서 21–36% 줄고, step time은 7–11% 늘어납니다.
- 렌더 비교에서는 글자와 손가락 위치가 일관되게 나아졌습니다(MOSS-Image).
- stream은 fp16으로 돌릴 수 없습니다. 중간 블록에서 stream 값이 fp16 최대값의 4배를 넘습니다.
- 같은 seed라도 bf16 stream 렌더와 다른 그림이 나오므로, 두 설정의 렌더는 seed 단위로 비교할 수 없습니다.

오차표, 비용, fp16 측정은 부록 A3.3에 있습니다.
