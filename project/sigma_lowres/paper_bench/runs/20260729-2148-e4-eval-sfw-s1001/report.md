# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4964 | 0.5689 | +0.0725 |
| sigma896 | 0.5574 | 0.6596 | +0.1022 |
| unsafe768 | 0.5656 | 0.6593 | +0.0937 |
| 896only | 0.5790 | 0.6974 | +0.1184 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9616 / 0.9300 |
| native~unsafe768 | 0.9510 / 0.9118 |
| native~896only | 0.9666 / 0.9432 |
| sigma896~unsafe768 | 0.9512 / 0.9004 |
| sigma896~896only | 0.9769 / 0.9484 |
| unsafe768~896only | 0.9597 / 0.9134 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.2460 | 0.5723 | +0.3263 |
| sigma896 | 0.3283 | 0.6628 | +0.3345 |
| unsafe768 | 0.3294 | 0.7161 | +0.3867 |
| 896only | 0.3332 | 0.6933 | +0.3601 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9667 / 0.8647 |
| native~unsafe768 | 0.9471 / 0.7761 |
| native~896only | 0.9367 / 0.7638 |
| sigma896~unsafe768 | 0.9551 / 0.8478 |
| sigma896~896only | 0.9529 / 0.8345 |
| unsafe768~896only | 0.9422 / 0.7823 |
