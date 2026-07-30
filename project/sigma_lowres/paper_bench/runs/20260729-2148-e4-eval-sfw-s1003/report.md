# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4427 | 0.4953 | +0.0526 |
| sigma896 | 0.4894 | 0.5685 | +0.0792 |
| 896only | 0.4327 | 0.5453 | +0.1125 |
| sigma768 | 0.4694 | 0.5360 | +0.0665 |
| unsafe768 | 0.4457 | 0.5493 | +0.1036 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9532 / 0.9032 |
| native~896only | 0.9312 / 0.7818 |
| native~sigma768 | 0.9516 / 0.9027 |
| native~unsafe768 | 0.9417 / 0.8635 |
| sigma896~896only | 0.9534 / 0.8775 |
| sigma896~sigma768 | 0.9503 / 0.8941 |
| sigma896~unsafe768 | 0.9395 / 0.8577 |
| 896only~sigma768 | 0.9507 / 0.9024 |
| 896only~unsafe768 | 0.9468 / 0.8695 |
| sigma768~unsafe768 | 0.9497 / 0.8367 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.1575 | 0.4002 | +0.2427 |
| sigma896 | 0.2186 | 0.4779 | +0.2593 |
| 896only | 0.2161 | 0.4377 | +0.2216 |
| sigma768 | 0.1777 | 0.4591 | +0.2813 |
| unsafe768 | 0.2160 | 0.4727 | +0.2567 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9634 / 0.9216 |
| native~896only | 0.9649 / 0.9390 |
| native~sigma768 | 0.9698 / 0.9205 |
| native~unsafe768 | 0.9692 / 0.9030 |
| sigma896~896only | 0.9687 / 0.9147 |
| sigma896~sigma768 | 0.9691 / 0.9190 |
| sigma896~unsafe768 | 0.9546 / 0.8752 |
| 896only~sigma768 | 0.9682 / 0.9182 |
| 896only~unsafe768 | 0.9606 / 0.9260 |
| sigma768~unsafe768 | 0.9626 / 0.9059 |
