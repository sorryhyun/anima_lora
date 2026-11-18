# LoRA 학습에서 4스텝 터보 증류까지 (초안)

diffusion DiT 위에서 LoRA를 학습하고, 여러 LoRA를 하나로 합치고, 그 결과를 teacher로 4스텝 student를 증류하기까지
쓴 방법과 실험 결과를 정리한 기술 리포트입니다. 실험은 Anima(28블록 DiT, 16 GB 카드)에서 했지만, 각 장은
다른 DiT에도 옮길 수 있는 방법과 근거를 중심으로 적었습니다. 2026-11-12 기준 초안이고, 4장은 2026-11-18에 다시 썼습니다.

| # | 문서 | 내용 | 부록 |
|---|---|---|---|
| 1 | [01_lora_recipe.md](01_lora_recipe.md) | LoRA 학습 레시피: 상세한 캡션, free-fit, 학습 설정 | [A1](A1_lora_recipe.md): 설정별 A/B, batch와 clip, lr과 EMA, rank와 alpha, down init |
| 2 | [02_soup_merge.md](02_soup_merge.md) | LoRA soup과 merge: 시드 운과 같은 시드로도 재현되지 않는 런, 시드 soup, ΔW 수준의 exact concat, 크기 맞추기 | [A2](A2_soup_merge.md): 결정성, 재료별 ΔW, truncation 손실, merge 정확도 |
| 3 | [03_bake_int8_gptq.md](03_bake_int8_gptq.md) | 증류를 위한 경량화: DiT에 bake, int8 base, 체크포인트별 GPTQ | [A3](A3_bake_int8_gptq.md): calibration, int8 형식과 copy 배치, fp32 stream 수치 |
| 4 | [04_turbo_distillation.md](04_turbo_distillation.md) | 터보 증류: 공식 turbo에서 warm start하고 스텝 0을 teacher 궤적에 맞추기, teacher feature 위의 FM과 짝지은 GAN, 평가 방법 | [A4](A4_turbo_distillation.md): 이전의 DP-DMD 공동 루프(세부, 측정, 실패한 시도, 평가 지표), 공식 turbo delta |

파이프라인 한 줄 요약: 작가 그룹별 LoRA 학습(1) → soup으로 묶어 하나의 LoRA로 merge(2) → DiT에 bake하고 그
DiT 전용 GPTQ int8 가중치 생성(3) → 그 DiT를 teacher로 4스텝 student LoRA 증류(4) → soup LoRA와 exact
concat해서 base에 올라가는 LoRA 하나로 배포.
