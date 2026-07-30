# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4964 | 0.5689 | +0.0725 |
| sigma896 | 0.5574 | 0.6596 | +0.1022 |
| 896only | 0.5790 | 0.6974 | +0.1184 |
| sigma768 | 0.3823 | 0.5035 | +0.1212 |
| unsafe768 | 0.5656 | 0.6593 | +0.0937 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9616 / 0.9300 |
| native~896only | 0.9666 / 0.9432 |
| native~sigma768 | 0.9503 / 0.9027 |
| native~unsafe768 | 0.9510 / 0.9118 |
| sigma896~896only | 0.9769 / 0.9484 |
| sigma896~sigma768 | 0.9514 / 0.8941 |
| sigma896~unsafe768 | 0.9512 / 0.9004 |
| 896only~sigma768 | 0.9568 / 0.9316 |
| 896only~unsafe768 | 0.9597 / 0.9134 |
| sigma768~unsafe768 | 0.9540 / 0.8934 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.2460 | 0.5723 | +0.3263 |
| sigma896 | 0.3283 | 0.6628 | +0.3345 |
| 896only | 0.3332 | 0.6933 | +0.3601 |
| sigma768 | 0.3068 | 0.8116 | +0.5047 |
| unsafe768 | 0.3294 | 0.7161 | +0.3867 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9667 / 0.8647 |
| native~896only | 0.9367 / 0.7638 |
| native~sigma768 | 0.9459 / 0.8413 |
| native~unsafe768 | 0.9471 / 0.7761 |
| sigma896~896only | 0.9529 / 0.8345 |
| sigma896~sigma768 | 0.9500 / 0.8522 |
| sigma896~unsafe768 | 0.9551 / 0.8478 |
| 896only~sigma768 | 0.9341 / 0.7923 |
| 896only~unsafe768 | 0.9422 / 0.7823 |
| sigma768~unsafe768 | 0.9463 / 0.8914 |
