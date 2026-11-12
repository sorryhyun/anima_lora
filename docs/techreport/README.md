# LoRA 학습에서 4스텝 터보 증류까지 (초안)

diffusion DiT 위에서 LoRA를 학습하고, 여러 LoRA를 하나로 합치고, 그 결과를 teacher로 4스텝 student를 증류하기까지
쓴 방법과 실험 결과를 정리한 기술 리포트입니다. 실험은 Anima(28블록 DiT, 16 GB 카드)에서 했지만, 각 장은
다른 DiT에도 옮길 수 있는 방법과 근거를 중심으로 적었습니다. 2026-11-12 기준 초안입니다.

| # | 문서 | 내용 |
|---|---|---|
| 1 | [01_lora_recipe.md](01_lora_recipe.md) | LoRA 학습 레시피: 상세한 캡션, crop의 손실과 free-fit, batch와 clip, A/B로 정한 설정 |
| 2 | [02_soup_merge.md](02_soup_merge.md) | LoRA soup과 merge: 시드 운, ΔW 수준의 exact concat, 크기 맞추기, 다시 합칠 때의 정확성 |
| 3 | [03_bake_int8_gptq.md](03_bake_int8_gptq.md) | DiT에 bake, int8 base, 구운 teacher에 GPTQ를 따로 맞춰야 하는 이유 |
| 4 | [04_turbo_distillation.md](04_turbo_distillation.md) | DP-DMD 터보 증류: 루프, 레시피, 배운 것, 평가 방법 |
| 5 | [05_practical_tips.md](05_practical_tips.md) | 실전 팁: fp32 residual stream, compile, VRAM, 추론, 실험 위생 |

파이프라인 한 줄 요약: 작가 그룹별 LoRA 학습(1) → soup으로 묶어 하나의 LoRA로 merge(2) → DiT에 bake하고 그
DiT 전용 GPTQ int8 가중치 생성(3) → 그 DiT를 teacher로 4스텝 student LoRA 증류(4) → soup LoRA와 exact
concat해서 base에 올라가는 LoRA 하나로 배포.
