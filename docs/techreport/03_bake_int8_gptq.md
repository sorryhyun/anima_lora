# 3. 증류를 위한 경량화

## 1. LoRA를 DiT 가중치에 굽기

LoRA를 그대로 들고 있으면 생각보다 연산량이 많이 듭니다. teacher로 쓰는 rank 256 soup을 붙인 채로 돌리면 이미지
토큰당 block linear 연산이 약 20 % 늘고, 이 작은 GEMM들은 int8로 돌리지 않으므로 bf16에 남습니다. 그래서 이걸 증류
소스로 효율적으로 사용하려면 하나의 모델로 만드는 게 좋습니다. 이걸 bake라고 하고, 방법은 LoRA가 하는 계산 그대로
W + ΔW로 새 W를 만드는 것입니다.

모듈마다 **fp32로** `W += multiplier · (up @ down) · scale`을 계산하고, 결과를 bf16으로 저장합니다. fp32로 계산하는
이유는 반올림을 저장할 때 한 번으로 줄이기 위해서인데, 실제로 재 보면 bf16으로 계산한 경우와의 차이는 ΔW 노름의
0.1 % 안입니다. 오차는 대부분 마지막 bf16 저장에서 생깁니다. ΔW가 W 노름의 1–8 %밖에 되지 않아서, 저장 반올림이
구운 ΔW를 노름 기준 평균 7 %(블록과 모듈에 따라 2–16 %) 흔듭니다.

## 2. int8 base

frozen DiT의 큰 linear를 W8A8 int8 GEMM으로 돌리면 LoRA 학습 step이 1.37배 빨라지고 최종 loss는 같습니다.

| int8 양자화 대상 | 유지 |
|---|---|
| self-attn qkv (fused), self-attn 출력 | cross-attn k / v (M이 context 길이) |
| cross-attn q, cross-attn 출력 | adaLN (M이 batch) |
| MLP의 두 linear | attention, norm, residual stream, embedder, head |
| | LoRA 자체의 GEMM |

DiT에 최적화된 방식인 그룹 단위 Hadamard ConvRot(Huang et al. 2025, arXiv 2512.03673)를 사용하면 대부분의 경우
무난하게 양자화할 수 있습니다. 가중치를 256열 그룹마다 Hadamard 행렬로 회전한 뒤 출력 행마다 스케일 하나로 int8에
저장하고, activation은 토큰마다 int8로 양자화합니다.

## 3. 체크포인트별 GPTQ

상술한 양자화 방식은 어떤 입력에 대해 어떤 출력을 낼지 고려하지 않습니다. 따라서 실사나 캐릭터가 없는 그림처럼
지원 대상이 아닌 입력이 아니라, 목표로 하는 입력에서 오차를 특히 줄이는 GPTQ(Frantar et al. 2022, arXiv
2210.17323) 방식을 사용합니다.

calibration은 작가마다 이미지 2장을 골라 추론 schedule에서 뽑은 σ로 노이즈를 섞고, 자기 캡션과 negative prompt로
한 번씩 돌려 모읍니다. 블록은 순서대로 양자화하므로, 앞쪽 블록의 양자화 오차가 뒤쪽 calibration에
반영됩니다(부록 A3.1).
