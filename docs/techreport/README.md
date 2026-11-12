# Anima 터보 증류 레시피 A to Z (초안)

Anima LoRA 학습부터 4스텝 터보 LoRA까지, 실제로 쓰는 파이프라인을 순서대로 정리한 기술 리포트입니다.
2026-11-12 기준 초안이며, 검토와 퇴고 전입니다.

| # | 문서 | 내용 |
|---|---|---|
| 1 | [01_lora_recipe.md](01_lora_recipe.md) | LoRA 학습 레시피: 데이터와 캡션, 버킷, 확인된 설정 |
| 2 | [02_soup_merge.md](02_soup_merge.md) | LoRA soup과 merge: uncond-init soup, merge, exact concat, merged14 |
| 3 | [03_bake_int8_gptq.md](03_bake_int8_gptq.md) | DiT에 bake, int8 base, teacher 전용 GPTQ |
| 4 | [04_turbo_distillation.md](04_turbo_distillation.md) | DP-DMD 터보 증류: 루프, 레시피, 배운 것, 평가 방법 |
| 5 | [05_practical_tips.md](05_practical_tips.md) | 실전 팁: fp32 stream, 모델 불변식, compile, VRAM, 실험 위생 |

파이프라인 한 줄 요약: 작가별 LoRA 학습(1) → soup으로 묶어 하나의 LoRA로 merge(2) → DiT에 bake하고 그
DiT 전용 GPTQ int8 가중치 생성(3) → 그 DiT를 teacher로 4스텝 student LoRA 증류(4) → soup LoRA와 exact
concat해서 base에 올라가는 LoRA 하나로 배포.
