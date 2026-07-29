# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3279 | 0.4103 | +0.0824 |
| sigma896 | 0.4876 | 0.6096 | +0.1221 |
| unsafe768 | 0.4951 | 0.6026 | +0.1075 |
| 896only | 0.5541 | 0.6243 | +0.0702 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9506 / 0.9086 |
| native~unsafe768 | 0.9652 / 0.9443 |
| native~896only | 0.9533 / 0.9257 |
| sigma896~unsafe768 | 0.9573 / 0.9128 |
| sigma896~896only | 0.9570 / 0.8885 |
| unsafe768~896only | 0.9624 / 0.8971 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3607 | 1.0513 | +0.6906 |
| sigma896 | 0.3102 | 0.7540 | +0.4438 |
| unsafe768 | 0.2362 | 0.5926 | +0.3564 |
| 896only | 0.3533 | 0.6362 | +0.2829 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9622 / 0.8751 |
| native~unsafe768 | 0.9472 / 0.8508 |
| native~896only | 0.9483 / 0.8156 |
| sigma896~unsafe768 | 0.9550 / 0.7647 |
| sigma896~896only | 0.9619 / 0.8386 |
| unsafe768~896only | 0.9668 / 0.9265 |
